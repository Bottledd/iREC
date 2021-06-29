import torch
import torch.distributions as dist


class VariationalPosterior:
    def __init__(self,
                 target,
                 coding_sampler,
                 ):

        self.current_z_mean = target.mean
        self.current_z_var = target.covariance_matrix
        self.coding_sampler = coding_sampler
        self.problem_dimension = self.current_z_mean.shape[0]

    def q_ak_given_history(self, aux_history, k):

        b_k_minus_one = torch.sum(aux_history[:k], dim=0)
        s_k_minus_one = torch.sum(self.coding_sampler.auxiliary_vars[k:])
        s_k = torch.sum(self.coding_sampler.auxiliary_vars[k+1:])

        sigma_k = self.coding_sampler.auxiliary_vars[k]
        mean_scalar = sigma_k / s_k_minus_one
        variance_scalar_term_one = (sigma_k * s_k / s_k_minus_one) * torch.eye(self.problem_dimension)
        variance_scalar_term_two = self.current_z_var * (sigma_k / s_k_minus_one) ** 2
        covariance = variance_scalar_term_one + variance_scalar_term_two
        mean = (self.current_z_mean - b_k_minus_one) * mean_scalar

        return dist.multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=covariance)

    def q_z_given_trajectory(self, aux_history, k):
        b_k_minus_one = torch.sum(aux_history[:k], dim=0)
        s_k_minus_one = torch.sum(self.coding_sampler.auxiliary_vars[k:])
        s_k = torch.sum(self.coding_sampler.auxiliary_vars[k + 1:])
        sigma_k = self.coding_sampler.auxiliary_vars[k]

        c = (sigma_k / (s_k_minus_one * s_k)) * b_k_minus_one + (1. / s_k) * aux_history[k] \
            + torch.inverse(self.current_z_var) @ self.current_z_mean

        C = torch.inverse((sigma_k / (s_k_minus_one * s_k)) * torch.eye(self.problem_dimension) + torch.inverse(self.current_z_var))

        mean = C @ c

        self.current_z_var = C
        self.current_z_mean = mean
