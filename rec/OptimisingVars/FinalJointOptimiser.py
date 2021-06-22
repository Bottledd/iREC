import math
import torch
import torch.distributions as dist
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import trange
from models.SimpleBayesianLinRegressor import BayesLinRegressor
from rec.utils import kl_estimate_with_mc, plot_2d_distribution
import numpy as np

class FinalJointOptimiser(nn.Module):
    def __init__(self, z_sample, omega, n_auxiliaries, total_kl, n_trajectories=1000, total_var=1):
        super(FinalJointOptimiser, self).__init__()
        self.n_auxiliaries = n_auxiliaries
        self.n_trajectories = n_trajectories
        self.omega = omega
        self.register_buffer("z_sample", z_sample)
        self.total_var = total_var
        self.dim = z_sample.shape[0]
        self.register_buffer('trajectories', torch.zeros(n_trajectories, n_auxiliaries, self.dim))
        self.pre_softmax_aux_vars = nn.Parameter(torch.ones(n_auxiliaries))
        self.kl_history = []
        self.total_kl = float(total_kl)

    def get_aux_post_params(self, index):
        sigma_ks = nn.functional.softmax(self.pre_softmax_aux_vars, dim=0)
        sigma_k = sigma_ks[index]
        s_k_minus_1 = self.total_var - torch.sum(sigma_ks[:index])
        s_k = s_k_minus_1 - sigma_k
        unscaled_mean = self.z_sample - torch.sum(self.trajectories[:, :index], dim=1)
        mean_scalar = sigma_k / s_k_minus_1
        variance_scalar = sigma_k * s_k / s_k_minus_1

        return unscaled_mean, mean_scalar, variance_scalar

    def aux_prior(self, index):
        sigma_k = nn.functional.softmax(self.pre_softmax_aux_vars, dim=0)[index]
        mean = torch.zeros((self.dim,)).to(self.pre_softmax_aux_vars.device)
        covariance = sigma_k * torch.eye(self.dim).to(self.pre_softmax_aux_vars.device)
        return dist.multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=covariance)

    def aux_posterior(self, index):
        unscaled_mean, mean_scalar, variance_scalar = self.get_aux_post_params(index)
        mean = unscaled_mean * mean_scalar
        covariance = torch.eye(self.dim).to(self.pre_softmax_aux_vars.device) * variance_scalar
        return dist.multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=covariance)

    def loss_function(self, index, remaining_kl):
        # kl = kl_estimate_with_mc(self.aux_posterior, self.aux_prior, rsample=True, num_samples=100)
        aux_kl = dist.kl_divergence(self.aux_posterior(index - 1), self.aux_prior(index - 1))
        kl_loss = (aux_kl - self.omega) ** 2
        remaining_kl_loss = ((remaining_kl - aux_kl) - (self.n_auxiliaries - index - 1) * self.omega) ** 2
        loss = torch.mean(kl_loss + remaining_kl_loss)

        return loss, torch.mean(aux_kl)

    def run_optimiser(self, epochs=1500):
        optimiser = torch.optim.Adam(self.parameters(), lr=3e-2)
        pbar = trange(epochs)
        for i in pbar:
            losses = torch.zeros(self.n_auxiliaries - 1).to(self.pre_softmax_aux_vars.device)
            kls = torch.zeros(self.n_auxiliaries - 1).to(self.pre_softmax_aux_vars.device)
            remaining_kl = self.total_kl
            # reset optimiser
            optimiser.zero_grad()
            for k in range(1, self.n_auxiliaries):
                # compute loss for a_k
                loss, aux_kl = self.loss_function(index=k, remaining_kl=remaining_kl)
                losses[k - 1] = loss
                kls[k - 1] = aux_kl
                remaining_kl -= aux_kl

                # sample trajectory
                if k < self.n_trajectories - 1:
                    self.trajectories[:, k - 1] = self.aux_posterior(index=k - 1).sample()

            mean_loss = torch.mean(losses)
            mean_kl = torch.mean(kls)
            pbar.set_description(f"The mean loss is {mean_loss:.5f}. The mean KL is: {mean_kl:.5f}")
            mean_loss.backward()
            optimiser.step()
        return nn.functional.softmax(self.pre_softmax_aux_vars.detach(), dim=0)

    def compute_run_of_kls(self):
        kl_hist = []
        trajectories = torch.zeros(self.trajectories.shape).to(self.trajectories.device)
        for k in range(self.n_auxiliaries - 1):
            aux_post = self.aux_posterior(index=k)
            aux_prior = self.aux_prior(index=k)
            trajectories[:, k] = aux_post.sample()
            kl = dist.kl_divergence(aux_post, aux_prior)
            kl_hist.append(torch.mean(kl))
            print(f"KL of Aux {k+1} is {kl_hist[-1]}")
        return kl_hist


if __name__ == '__main__':
    initial_seed_target = 0
    blr = BayesLinRegressor(prior_mean=torch.zeros(20),
                            prior_alpha=1,
                            signal_std=1,
                            num_targets=10000,
                            seed=initial_seed_target)
    blr.sample_feature_inputs()
    blr.sample_regression_targets()
    blr.posterior_update()

    target = blr.weight_posterior
    z_sample = target.mean

    dim = z_sample.shape[0]
    prior_var = 1.
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

    optimising = FinalJointOptimiser(z_sample, omega, n_auxiliaries, kl_q_p, n_trajectories, prior_var)
    best_vars = optimising.run_optimiser()
    fig, axes = plt.subplots(1, 2, figsize=(9, 5))

    axes[0].plot(best_vars)
    axes[1].plot(best_vars / (1 - torch.cumsum(best_vars, dim=0)))

    fig.tight_layout()
    plt.show()
