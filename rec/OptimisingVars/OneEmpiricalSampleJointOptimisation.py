import math
import torch
import torch.distributions as dist
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.SimpleBayesianLinRegressor import BayesLinRegressor
from rec.utils import kl_estimate_with_mc, plot_2d_distribution
import numpy as np

class JointOptimisation(nn.Module):
    def __init__(self, n_auxiliaries, n_trajectories, dim, target, omega, total_kl):
        super(JointOptimisation, self).__init__()
        self.n_auxiliaries = n_auxiliaries
        self.n_trajectories = n_trajectories
        self.dim = dim
        self.target = target
        self.omega = omega
        self.optimise_index = 0
        self.register_buffer("trajectories", torch.zeros((n_trajectories, n_auxiliaries, dim)))
        # self.register_buffer("z_sample", target.sample((1,)))
        self.register_buffer("z_sample", target.mean)
        self.register_buffer('total_var', torch.ones((1,)))
        self.pre_softmax_aux_vars = nn.Parameter(dist.normal.Normal(loc=0., scale=1).sample((n_auxiliaries,)))
        self.pre_softmax_aux_vars = nn.Parameter(torch.ones(n_auxiliaries))
        self.kl_history = []
        self.remaining_kl = total_kl

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

    def loss_function(self, index):
        # kl = kl_estimate_with_mc(self.aux_posterior, self.aux_prior, rsample=True, num_samples=100)
        if index < self.n_auxiliaries - 1:
            kl = dist.kl_divergence(self.aux_posterior(index), self.aux_prior(index))
            kl_loss = 0.5 * (kl - self.omega) ** 2

        return kl_loss, kl

    # def final_aux_loss(self):
    #     final_var_prior = self.aux_prior(index=self.n_auxiliaries-1)
    #     samples = final_var_prior.rsample(self.n_auxiliaries)
    #     z = self.trajectories + samples
    #     log_probs = target.log_prob(z)
    #     return log_probs

    def run_optimiser(self, epochs=1000):
        #optimiser = torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)
        optimiser = torch.optim.Adam(self.parameters(), lr=3e-3)
        pbar = tqdm(range(epochs))
        for i in pbar:
            losses = torch.zeros((self.n_trajectories, self.n_auxiliaries)).to(self.pre_softmax_aux_vars.device)
            # reset optimiser
            optimiser.zero_grad()
            for k in range(self.n_auxiliaries-1):
                # compute loss for a_k
                current_loss, aux_kl = self.loss_function(index=k)
                losses[:, k] = current_loss


                # sample trajectory
                if k < self.n_trajectories - 1:
                    self.trajectories[:, k] = self.aux_posterior(index=k).sample()

            mean_loss = torch.mean(losses)
            pbar.set_description(f"The mean loss is {mean_loss}")
            #print(f"The mean loss is {mean_loss}")
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
    torch.set_default_tensor_type(torch.DoubleTensor)
    initial_seed_target = 100
    blr = BayesLinRegressor(prior_mean=torch.tensor([0.0, 0.0]),
                            prior_alpha=0.05,
                            signal_std=10,
                            num_targets=1,
                            seed=initial_seed_target)
    blr.sample_feature_inputs()
    blr.sample_regression_targets()
    blr.posterior_update()
    target = blr.weight_posterior

    # plot_2d_distribution(target)
    # plt.show()

    dim = 2
    prior_var = 1
    omega = 8
    n_trajectories = 256

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

    optimising = JointOptimisation(n_auxiliaries, n_trajectories, dim, target, omega, kl_q_p)
    best_vars = optimising.run_optimiser()
    plt.plot(best_vars, 'o')
    plt.show()
    optimising.compute_run_of_kls()
