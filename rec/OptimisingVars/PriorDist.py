import math

import torch
import torch.distributions as dist
import torch.nn as nn
from models.SimpleBayesianLinRegressor import BayesLinRegressor
from rec.utils import kl_estimate_with_mc


class PriorDist(nn.Module):
    def __init__(self, optimise_index, n_auxiliary, current_optimised_sigmas=torch.tensor([]), prior_var=1, target_dist=None,
                 omega=5):
        super(PriorDist, self).__init__()
        self.dim = target_dist.mean.shape[0]
        self.omega = omega

        remaining_variance = prior_var - torch.sum(current_optimised_sigmas[:optimise_index])
        new_uniform_variance = remaining_variance / (n_auxiliary - optimise_index)
        self.register_buffer('already_optimised_aux_vars', current_optimised_sigmas[:optimise_index])
        self.register_buffer('aux_vars_left_to_optimise',
                             torch.ones((n_auxiliary - optimise_index - 1)) * new_uniform_variance)
        self.var_to_learn = nn.Parameter(torch.tensor((new_uniform_variance,)))

    def forward(self, samples):
        p_ak = self.p_ak
        log_prob = p_ak.log_prob(samples)
        return log_prob, p_ak

    @property
    def p_ak(self):
        mean = torch.zeros((self.dim,))
        covariance = self.var_to_learn * torch.eye(self.dim)
        return dist.MultivariateNormal(loc=mean,
                                       covariance_matrix=covariance)

    @property
    def auxiliary_vars(self):
        x = torch.cat([self.already_optimised_aux_vars, self.var_to_learn, self.aux_vars_left_to_optimise])
        return x


if __name__ == '__main__':
    initial_seed = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blr = BayesLinRegressor(prior_mean=torch.tensor([0.0, 0.0]),
                            prior_alpha=0.01,
                            signal_std=1,
                            num_targets=10,
                            seed=initial_seed)
    blr.sample_feature_inputs()
    blr.sample_regression_targets()
    blr.posterior_update()
    target = blr.weight_posterior
    prior = PriorDist(0, 25, target_dist=target)
    prior.auxiliary_vars

