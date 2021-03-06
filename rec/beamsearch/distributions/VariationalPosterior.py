import torch
import torch.distributions as dist


class VariationalPosterior:
    def __init__(self,
                 target,
                 coding_sampler,
                 beamwidth,
                 ):

        self.current_z_mean = torch.tile(target.mean, (beamwidth, 1))
        self.current_z_var = torch.tile(target.covariance_matrix, (beamwidth, 1, 1))
        self.coding_sampler = coding_sampler
        self.problem_dimension = target.mean.shape[0]

    def q_ak_given_history(self, aux_history, k, mask=None, beam_indices=None):
        # need to mask our mean and covariance if mask is not None
        if mask is not None:
            current_z_mean = torch.masked_select(self.current_z_mean, mask).reshape(-1, self.problem_dimension)
            current_z_var = torch.masked_select(self.current_z_var, torch.repeat_interleave(mask[:, None, :], self.problem_dimension, dim=1)).reshape(-1, self.problem_dimension, self.problem_dimension)
        else:
            current_z_mean = self.current_z_mean
            current_z_var = self.current_z_var

        if beam_indices is not None:
            current_z_mean = current_z_mean[beam_indices]
            current_z_var = current_z_var[beam_indices]

        s_k_minus_one = torch.sum(self.coding_sampler.auxiliary_vars[k:])
        s_k = torch.sum(self.coding_sampler.auxiliary_vars[k + 1:])
        sigma_k = self.coding_sampler.auxiliary_vars[k]
        mean_scalar = sigma_k / s_k_minus_one
        variance_term_one = (sigma_k * s_k / s_k_minus_one) * torch.eye(self.problem_dimension)

        if k == 0:
            variance_term_two = current_z_var[0] * (sigma_k / s_k_minus_one) ** 2
            covariance = variance_term_one + variance_term_two
            b_k_minus_one = torch.sum(aux_history[:k], dim=0)
            mean = (current_z_mean[0] - b_k_minus_one) * mean_scalar
        else:
            # need to tile the current z mean and var
            beamwidth = current_z_mean.shape[0]
            n_samples_per_aux = aux_history.shape[0] // beamwidth
            tiled_z_mean = torch.tile(current_z_mean, (n_samples_per_aux, 1))
            tiled_z_var = torch.tile(current_z_var, (n_samples_per_aux, 1, 1))
            variance_term_two = tiled_z_var * (sigma_k / s_k_minus_one) ** 2
            covariance = variance_term_one + variance_term_two
            b_k_minus_one = torch.sum(aux_history[:, :k], dim=1)
            mean = (tiled_z_mean - b_k_minus_one) * mean_scalar

        return dist.multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=covariance)

    def q_z_given_trajectory(self, aux_history, k, beam_indices=None):
        if beam_indices is not None:
            current_z_mean = self.current_z_mean[beam_indices]
            current_z_var = self.current_z_var[beam_indices]
        else:
            current_z_mean = self.current_z_mean
            current_z_var = self.current_z_var
        b_k_minus_one = torch.sum(aux_history[:, :k], dim=1)
        s_k_minus_one = torch.sum(self.coding_sampler.auxiliary_vars[k:])
        s_k = torch.sum(self.coding_sampler.auxiliary_vars[k + 1:])
        sigma_k = self.coding_sampler.auxiliary_vars[k]

        c = (sigma_k / (s_k_minus_one * s_k)) * b_k_minus_one + (1. / s_k) * aux_history[:, k] \
            + torch.einsum("bij, bj -> bi", torch.inverse(current_z_var), current_z_mean)

        C = torch.inverse((sigma_k / (s_k_minus_one * s_k)) * torch.eye(self.problem_dimension) + torch.inverse(current_z_var))

        mean = torch.einsum("bij, bj -> bi", C, c)

        self.current_z_var = C
        self.current_z_mean = mean
