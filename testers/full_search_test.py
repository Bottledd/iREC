import math

import matplotlib.pyplot as plt
import torch
import torch.distributions as dist
from tqdm import tqdm

from models.BayesianLinRegressor import BayesLinRegressor
from rec.distributions.CodingSampler import CodingSampler
from rec.distributions.EmpiricalMixturePosterior import EmpiricalMixturePosterior
from rec.utils import kl_estimate_with_mc, plot_running_sum, plot_2d_distribution


def get_parameters_for_search(target, omega, dim, epsilon=0.):
    coding_z_prior = CodingSampler(dim, n_auxiliary=1)
    try:
        kl_q_p = dist.kl_divergence(target, coding_z_prior)
        print(f"{kl_q_p}")
    except:
        # need to do MC estimate
        kl_q_p = kl_estimate_with_mc(target, coding_z_prior, dim=0)

    # compute parameters for auxiliary method
    n_auxiliary = math.ceil(kl_q_p / omega)
    n_samples_per_aux = math.ceil(torch.exp(torch.tensor(omega * (1 + epsilon))))

    return n_auxiliary, n_samples_per_aux


def run_search(target, omega):
    dim = target.mean.shape[0]
    n_auxiliary, n_samples_per_aux = get_parameters_for_search(target, omega, dim)
    # create sampling distribution
    sampler = CodingSampler(problem_dimension=dim,
                                                 n_auxiliary=n_auxiliary)

    # create tensor storing all paths
    trajectories = torch.zeros((n_auxiliary, n_samples_per_aux, dim))
    greedy_traj = torch.zeros((n_auxiliary, dim))
    greedy_sample = torch.zeros(dim)
    norm = 0
    for i in range(n_auxiliary):
        # sample aks and append to tensor
        samples = sampler.auxiliary_coding_dist(i).sample((n_samples_per_aux,))
        norm_of_samples = torch.norm(samples, dim=1, p=2)
        norm += torch.max(norm_of_samples)
        # trial_sample = greedy_sample + trajectories[i]
        # log_q = target.log_prob(trial_sample)
        # best_sample_idx = torch.argmax()
        # greedy_sample = trial_sample[best_sample_idx]
        # greedy_traj[i] = trajectories[i][best_sample_idx]

    return norm


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

    norm = run_search(target, 8)