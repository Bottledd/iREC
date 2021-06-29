import math
import torch
import torch.distributions as dist
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.SimpleBayesianLinRegressor import BayesLinRegressor
from rec.utils import kl_estimate_with_mc, plot_2d_distribution


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
        self.register_buffer("z_sample", target.mean)
        self.register_buffer('already_optimised_aux_vars', torch.zeros(n_auxiliaries))
        self.register_buffer('already_optimised_ratios', torch.zeros(n_auxiliaries))
        self.register_buffer('total_var', torch.ones(1))
        self.ratio_k = nn.Parameter(self.total_var / self.n_auxiliaries * torch.ones(1))
        self.kl_history = []

    def get_aux_post_params(self, index):
        unscaled_mean = self.z_sample - torch.sum(self.trajectories[:, :index], dim=1)
        mean_scalar = self.ratio_k
        sigma_k = self.ratio_k * (self.total_var - torch.sum(self.already_optimised_aux_vars[:self.optimise_index]))
        variance_scalar = sigma_k * (1 - self.ratio_k)

        return unscaled_mean, mean_scalar, variance_scalar

    def loss_function(self):
        #kl = kl_estimate_with_mc(self.aux_posterior, self.aux_prior, rsample=True, num_samples=100)
        kl = dist.kl_divergence(self.aux_posterior, self.aux_prior)
        return 0.5*(kl - self.omega)**2, torch.mean(kl)

    @property
    def aux_prior(self):
        mean = torch.zeros((self.dim,)).to(self.ratio_k.device)
        sigma_k = self.ratio_k * (self.total_var - torch.sum(self.already_optimised_aux_vars[:self.optimise_index]))
        covariance = sigma_k * torch.eye(self.dim).to(self.ratio_k.device)
        return dist.multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=covariance)

    @property
    def aux_posterior(self):
        unscaled_mean, mean_scalar, variance_scalar = self.get_aux_post_params(self.optimise_index)
        mean = unscaled_mean * mean_scalar
        covariance = torch.eye(self.dim).to(self.ratio_k.device) * variance_scalar
        return dist.multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=covariance)

    def optimise_for_ak(self, epochs=int(1e3)):
        optimiser = torch.optim.Adam(self.parameters(), lr=1e-3)
        for i in range(epochs):
            optimiser.zero_grad()
            big_loss, kl = self.loss_function()
            loss = torch.mean(big_loss)
            loss.backward()
            self.kl_history.append(kl)
            optimiser.step()
            if i % 100 == 0:
                print(f"The current KL is {kl}, loss is {loss}.")
            if loss < 1e-10:
                break
        return kl.detach()

    def run_optimiser(self):
        pbar = tqdm(range(self.n_auxiliaries - 1), position=0, leave=True)
        for k in pbar:
            # run the optimisation
            self.optimise_index = k
            self.get_aux_post_params(k)
            kl = self.optimise_for_ak()
            trained_ratio = self.ratio_k.detach()
            trained_sigma = trained_ratio * (self.total_var - torch.sum(self.already_optimised_aux_vars[:self.optimise_index].detach()))
            pbar.set_description(f"The final optimised variance for aux {k + 1} is {trained_sigma.cpu().numpy()[0]}\nFinal KL is {kl}")
            # print(f"The final optimised variance for aux {k + 1} is {trained_sigma.cpu().numpy()[0]}\nFinal KL is {kl}")
            self.already_optimised_aux_vars[self.optimise_index] = trained_sigma
            self.already_optimised_ratios[self.optimise_index] = trained_ratio
            self.trajectories[:, self.optimise_index] = self.aux_posterior.sample((self.n_trajectories,))[:, 0].detach()
            with torch.no_grad():
                remaining_var = self.total_var - torch.sum(self.already_optimised_aux_vars)
                uniform_remaining_var = remaining_var / (self.n_auxiliaries - (self.optimise_index + 1))
                self.ratio_k.copy_(uniform_remaining_var * torch.ones((1,)))
        return self.already_optimised_aux_vars

    def compute_run_of_kls(self):
        kl_hist = []
        self.optimise_index = 0
        for k in range(self.n_auxiliaries - 1):
            self.trajectories[:, self.optimise_index] = self.aux_posterior.sample((self.n_trajectories,))[:, 0]
            losses, kls = self.loss_function()
            print(f"KL of Aux {k+1} is {kls}")
            kl_hist.append(kls)
            self.optimise_index += 1
        return kl_hist


if __name__ == '__main__':
    # torch.set_default_tensor_type(torch.DoubleTensor)
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

    # plot_2d_distribution(target)
    # plt.show()

    dim = 2
    prior_var = 1
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

    optimising = OneSampleOptimise(n_trajectories, n_auxiliaries, dim, target, omega)
    best_vars = optimising.run_optimiser()
    plt.plot(best_vars, 'o')
    plt.plot(optimising.kl_history, 'x')
    plt.show()
    kls = optimising.compute_run_of_kls()


