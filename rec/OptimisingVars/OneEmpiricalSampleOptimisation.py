import math

import torch
import torch.distributions as dist
import torch.nn as nn

from models.BayesianLinRegressor import BayesLinRegressor
from rec.utils import kl_estimate_with_mc


class OneSampleOptimise(nn.Module):
    def __init__(self, n_trajectories, n_auxiliaries, dim, target, omega):
        super(OneSampleOptimise, self).__init__()
        self.n_auxiliaries = n_auxiliaries
        self.n_trajectories = n_trajectories
        self.dim = dim
        self.target = target
        self.omega = omega
        self.optimise_index = 0
        self.register_buffer("trajectories", torch.zeros((n_trajectories, n_auxiliaries, dim)))
        self.register_buffer("z_sample", target.sample((1,)))
        self.register_buffer('already_optimised_aux_vars', torch.zeros((n_auxiliaries,)))
        self.register_buffer('total_var', torch.ones((1,)))
        self.sigma_k = nn.Parameter(torch.ones((1,)))

    def get_aux_post_params(self, index):
        s_k = self.total_var - torch.sum(self.already_optimised_aux_vars[:index] + self.sigma_k)
        s_k_minus_1 = self.total_var - torch.sum(self.already_optimised_aux_vars[:index])
        unscaled_mean = self.z_sample - torch.sum(self.trajectories[:, :index], dim=1)
        mean_scalar = self.sigma_k / s_k_minus_1
        variance_scalar = self.sigma_k * s_k / s_k_minus_1

        return unscaled_mean, mean_scalar, variance_scalar

    def loss_function(self):
        return 1 / 2 * (dist.kl_divergence(self.aux_posterior, self.aux_prior) - self.omega) ** 2

    @property
    def aux_prior(self):
        mean = torch.zeros((self.dim,)).to(self.sigma_k.device)
        covariance = self.sigma_k * torch.eye(self.dim).to(self.sigma_k.device)
        return dist.MultivariateNormal(loc=mean, covariance_matrix=covariance)

    @property
    def aux_posterior(self):
        unscaled_mean, mean_scalar, variance_scalar = self.get_aux_post_params(self.optimise_index)
        mean = unscaled_mean * mean_scalar
        covariance = torch.eye(self.dim).to(self.sigma_k.device) * variance_scalar
        return dist.MultivariateNormal(loc=mean, covariance_matrix=covariance)

    def optimise_for_ak(self, epochs=50000):
        optimiser = torch.optim.Adam(self.parameters())
        i = 0
        loss = 10
        while loss > 1e-3 and i < epochs:
            optimiser.zero_grad()
            loss = torch.mean(self.loss_function())
            loss.backward()
            optimiser.step()
            if i % 1000 == 0:
                print(f"The current sigma for aux {self.optimise_index + 1} is {self.sigma_k.detach().cpu().numpy()[0]}\nThe current loss is {loss}.")
            i += 1

    def run_optimiser(self):
        for k in range(self.n_auxiliaries):
            # run the optimisation
            self.optimise_index = k
            self.get_aux_post_params(k)
            self.optimise_for_ak()
            print(f"The final optimised variance for aux {k + 1} is {self.sigma_k.detach().cpu().numpy()[0]}")
            self.already_optimised_aux_vars[self.optimise_index] = self.sigma_k.detach()
            self.trajectories[:, self.optimise_index] = self.aux_posterior.sample((self.n_trajectories,))[:, 0].detach()
        return self.already_optimised_aux_vars


if __name__ == '__main__':
    initial_seed = 100
    blr = BayesLinRegressor(prior_mean=torch.tensor([0.0, 0.0]),
                            prior_alpha=0.01,
                            signal_std=1,
                            num_targets=10,
                            seed=initial_seed)
    blr.sample_feature_inputs()
    blr.sample_regression_targets()
    blr.posterior_update()
    target = blr.weight_posterior

    dim = target.mean.shape[0]
    prior_var = 1
    omega = 5
    n_trajectories = 64
    # first try to compute KL between q(z) and p(z) with torch.distributions
    try:
        kl_q_p = dist.kl_divergence(target, dist.MultivariateNormal(loc=torch.zeros((dim,)),
                                                                    covariance_matrix=prior_var * torch.eye(
                                                                        dim)))
    except:
        kl_q_p = kl_estimate_with_mc(target=target, coder=dist.MultivariateNormal(loc=torch.zeros((dim,)),
                                                                                  covariance_matrix=prior_var * torch.eye(
                                                                                      dim)))

    n_auxiliaries = math.ceil(kl_q_p / omega)
    print(f"Num of Aux is: {n_auxiliaries}")

    optimising = OneSampleOptimise(n_trajectories, n_auxiliaries, dim, target, omega)
    best_vars = optimising.run_optimiser()
