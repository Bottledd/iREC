import torch
import torch.distributions as dist


class VariationalPosterior:
    def __init__(self,
                 target,
                 coding_sampler,
                 ):

        self.current_z_mean = target.mean
        self.current_z_var_scalar = target.stddev[0]**2
        self.coding_sampler = coding_sampler
        self.problem_dimension = self.current_z_mean.shape[0]

    def q_ak_given_history(self, aux_history, k):

        b_k_minus_one = torch.sum(aux_history[:k], dim=0)
        s_k_minus_one = torch.sum(self.coding_sampler.auxiliary_vars[k:])
        s_k = torch.sum(self.coding_sampler.auxiliary_vars[k+1:])

        sigma_k = self.coding_sampler.auxiliary_vars[k]
        mean_scalar = sigma_k / s_k_minus_one
        variance_scalar_term_one = (sigma_k * s_k / s_k_minus_one)
        variance_scalar_term_two = self.current_z_var_scalar * (sigma_k / s_k_minus_one) ** 2
        variance_scalar = variance_scalar_term_one + variance_scalar_term_two
        mean = (self.current_z_mean - b_k_minus_one) * mean_scalar
        covariance = variance_scalar * torch.eye(self.problem_dimension)

        return dist.multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=covariance)

    def q_z_given_trajectory(self, aux_history, k):
        # TODO Think there must be a bug here as with high epsilon this method causes things to go past the target
        b_k_minus_one = torch.sum(aux_history[:k], dim=0)
        s_k_minus_one = torch.sum(self.coding_sampler.auxiliary_vars[k:])
        s_k = torch.sum(self.coding_sampler.auxiliary_vars[k + 1:])
        sigma_k = self.coding_sampler.auxiliary_vars[k]

        mean_numerator_term_one = aux_history[k] * self.current_z_var_scalar * s_k_minus_one
        mean_numerator_term_two = b_k_minus_one * sigma_k * self.current_z_var_scalar
        mean_numerator_term_three = self.current_z_mean * s_k * s_k_minus_one
        mean_denominator = sigma_k * self.current_z_var_scalar + s_k * s_k_minus_one

        mean = (mean_numerator_term_one + mean_numerator_term_two + mean_numerator_term_three) / mean_denominator

        covariance_numerator = self.current_z_var_scalar * s_k * s_k_minus_one
        covariance_denominator = self.current_z_var_scalar * sigma_k + s_k * s_k_minus_one

        covariance_scalar = (covariance_numerator / covariance_denominator)

        self.current_z_mean = mean
        self.current_z_var_scalar = covariance_scalar
