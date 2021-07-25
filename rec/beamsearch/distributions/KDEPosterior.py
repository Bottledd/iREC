import torch
import torch.distributions as dist


class KDEPosterior:
    def __init__(self,
                 target,
                 coding_sampler,
                 beamwidth,
                 ):
        
        self.z_samples = target.component_distribution.mean
        self.var_kde = target.component_distribution.covariance_matrix[0][0][0]
        self.current_z_mixing_weights = torch.tile(target.mixture_distribution.probs, (beamwidth, 1))
        self.current_z_means = torch.tile(target.component_distribution.mean, (beamwidth, 1, 1))
        self.current_z_covariance = target.component_distribution.covariance_matrix[0]
        self.beamwidth = beamwidth
        self.coding_sampler = coding_sampler
        self.problem_dimension = target.mean.shape[0]

    def q_ak_given_history(self, aux_history, k, mask=None, beam_indices=None):
        # need to mask our mean and covariance if mask is not None
        if mask is not None:
            current_z_means = torch.masked_select(self.current_z_means, mask).reshape(self.beamwidth, -1, self.problem_dimension)
            current_z_covariance = torch.masked_select(self.current_z_covariance, torch.repeat_interleave(mask[:, None, :], self.problem_dimension, dim=1)).reshape(self.problem_dimension, self.problem_dimension)
        else:
            current_z_means = self.current_z_means
            current_z_covariance = self.current_z_covariance

        if beam_indices is not None:
            current_z_means = current_z_means[beam_indices]

        s_k_minus_one = torch.sum(self.coding_sampler.auxiliary_vars[k:])
        s_k = torch.sum(self.coding_sampler.auxiliary_vars[k + 1:])
        sigma_k = self.coding_sampler.auxiliary_vars[k]
        mean_scalar = sigma_k / s_k_minus_one
        variance_term_one = (sigma_k * s_k / s_k_minus_one) * torch.eye(self.problem_dimension)

        if k == 0:
            variance_term_two = current_z_covariance * (sigma_k / s_k_minus_one) ** 2
            covariance = variance_term_one + variance_term_two
            b_k_minus_one = torch.sum(aux_history[:k], dim=0)
            mean = (current_z_means[0] - b_k_minus_one) * mean_scalar
            mixing_weights = self.current_z_mixing_weights
        else:
            # need to tile the current z mean and var
            n_samples_per_aux = aux_history.shape[0] // self.beamwidth
            tiled_z_mean = torch.tile(current_z_means, (n_samples_per_aux, 1, 1))
            tiled_z_covariance = torch.tile(current_z_covariance, (n_samples_per_aux, self.z_samples.shape[0], 1, 1))
            variance_term_two = tiled_z_covariance * (sigma_k / s_k_minus_one) ** 2
            covariance = variance_term_one + variance_term_two
            b_k_minus_one = torch.sum(aux_history[:, :k], dim=1)
            repeated_b_k_minus_one = b_k_minus_one[:, None].repeat(1, self.z_samples.shape[0], 1)
            mean = (tiled_z_mean - repeated_b_k_minus_one) * mean_scalar
            mixing_weights = torch.tile(self.current_z_mixing_weights, (n_samples_per_aux, 1))

        components = dist.multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=covariance)
        mixture_dist = dist.Categorical(probs=mixing_weights)

        return dist.MixtureSameFamily(mixture_dist, components)

    def q_z_given_trajectory(self, aux_history, k):

        b_k = torch.sum(aux_history[:, :k+1], dim=1)
        s_k = torch.sum(self.coding_sampler.auxiliary_vars[k + 1:])

        C_d = torch.inverse(((1./self.var_kde) + (1./s_k)) * torch.eye(self.problem_dimension))

        c_d = ((b_k / s_k) + (self.z_samples[None] / self.var_kde))

        self.current_z_means = torch.einsum("ij, bnj -> bni", C_d, c_d)
        self.current_z_covariance = C_d

        lambdas = dist.MultivariateNormal(loc=self.z_samples,
                                          covariance_matrix=(s_k + self.var_kde) * torch.eye(self.problem_dimension))

        unnormed_mixing_weights = lambdas.log_prob(b_k)
        self.current_z_mixing_weights = torch.softmax(unnormed_mixing_weights, dim=0)
