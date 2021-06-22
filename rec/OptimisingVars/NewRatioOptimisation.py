import torch
import torch.distributions as dist
import torch.nn as nn
from tqdm import trange
from models.SimpleBayesianLinRegressor import BayesLinRegressor
from rec.utils import kl_estimate_with_mc
import math
import matplotlib.pyplot as plt


class OptimiseRatio(nn.Module):
    def __init__(self, z_sample, omega, n_auxiliaries, n_trajectories, total_kl, total_var):
        super(OptimiseRatio, self).__init__()
        self.total_kl = float(total_kl)
        self.total_var = float(total_var)
        self.dim = z_sample.shape[0]
        self.omega = omega
        self.n_auxiliaries = n_auxiliaries
        self.n_trajectories = n_trajectories
        self.register_buffer('z_sample', z_sample)
        self.register_buffer('optimised_ratios', torch.zeros(n_auxiliaries - 1))
        self.register_buffer('optimised_vars', torch.zeros(n_auxiliaries - 1))
        self.ratio_k = nn.Parameter(self.total_var / self.n_auxiliaries * torch.ones(1))

    def p_ak_given_traj_and_z(self, trajectory, ratio, remaining_var):
        b = torch.sum(trajectory, dim=1)
        s_k = (1. - ratio) * remaining_var

        mean = (self.z_sample - b) * ratio
        var = s_k * ratio * torch.eye(self.dim).to(self.ratio_k.device)

        return mean, var

    def sample_trajectory(self, index):
        remaining_var = self.total_var
        num_optimised_aux_ratios = self.optimised_ratios[:index].shape[0]
        aux_traj = torch.zeros((self.n_trajectories, num_optimised_aux_ratios, self.dim))
        for i, ratio in enumerate(self.optimised_ratios[:index]):
            mean, var = self.p_ak_given_traj_and_z(aux_traj[:, :i], ratio, remaining_var)

            # sample
            aux_traj[:, i] = dist.multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=var).sample()

            remaining_var = (1. - ratio) * remaining_var

        return aux_traj

    def optimise_kth_ratio(self, index, remaining_var, remaining_kl, num_steps=500):
        optimiser = torch.optim.Adam(self.parameters(), lr=3e-3)
        pbar = trange(num_steps)
        for i in pbar:
            optimiser.zero_grad()

            # Sample q(a_{1:k - 1})
            aux_traj = self.sample_trajectory(index - 1)

            # Calculate p(a_k | a_{1:k - 1}^(m), z)
            aux_post_mean, aux_post_var = self.p_ak_given_traj_and_z(aux_traj, self.ratio_k, remaining_var)

            aux_post = dist.multivariate_normal.MultivariateNormal(loc=aux_post_mean, covariance_matrix=aux_post_var)
            aux_prior = dist.multivariate_normal.MultivariateNormal(loc=torch.zeros(self.dim).to(self.ratio_k.device),
                                                                    covariance_matrix=torch.eye(self.dim) * self.ratio_k * remaining_var)

            aux_kl = dist.kl_divergence(aux_post, aux_prior)

            kl_loss = (aux_kl - self.omega) ** 2
            remaining_kl_loss = ((remaining_kl - aux_kl) - (self.n_auxiliaries - index - 1)*self.omega) ** 2

            loss = torch.mean(kl_loss + remaining_kl_loss)

            loss.backward()
            optimiser.step()

            pbar.set_description(f"Aux var ratio: {self.ratio_k.detach().numpy()[0]:.5f}, Loss: {loss:.5f}")
        return aux_kl

    def run_optimisation(self):
        remaining_kl = self.total_kl
        remaining_var = self.total_var

        for i in range(1, self.n_auxiliaries):
            print(f"Optimising aux {i}/{self.n_auxiliaries - 1}!")

            aux_kl = self.optimise_kth_ratio(i, remaining_var, remaining_kl)

            aux_var = self.ratio_k.detach() * remaining_var
            mean_kl = torch.mean(aux_kl).detach()
            remaining_var -= aux_var
            remaining_kl -= mean_kl

            self.optimised_ratios[i - 1] = self.ratio_k.detach()
            self.optimised_vars[i - 1] = aux_var.detach()

            print(f"Final KL for aux {i}: {mean_kl.numpy():.5f} nats.\n"
                  f"Aux var: {aux_var.numpy()[0]:.5f}\n"
                  f"Remaining KL: {remaining_kl:.5f}/{self.total_kl:.5f}\n\n")

        return self.optimised_vars


if __name__ == '__main__':
    initial_seed_target = 0
    blr = BayesLinRegressor(prior_mean=torch.zeros(50),
                            prior_alpha=1,
                            signal_std=1e-2,
                            num_targets=100,
                            seed=initial_seed_target)
    blr.sample_feature_inputs()
    blr.sample_regression_targets()
    blr.posterior_update()
    target = blr.weight_posterior
    z_sample = target.mean

    dim = z_sample.shape[0]
    prior_var = 1.
    omega = 8
    n_trajectories = 1000

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

    optimisation = OptimiseRatio(z_sample, omega, n_auxiliaries, n_trajectories, total_kl=kl_q_p, total_var=prior_var)
    best_vars = optimisation.run_optimisation()

    fig, axes = plt.subplots(1, 2, figsize=(9, 5))

    axes[0].plot(optimisation.optimised_vars)
    axes[1].plot(optimisation.optimised_ratios)

    fig.tight_layout()
    plt.show()
