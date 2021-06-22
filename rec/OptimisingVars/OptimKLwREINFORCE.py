import math
import matplotlib.pyplot as plt
import torch
import torch.distributions as dist
from tqdm import tqdm
from models.SimpleBayesianLinRegressor import BayesLinRegressor
from rec.OptimisingVars.CodingSampler import CodingSampler
from rec.OptimisingVars.EmpiricalMixturePosterior import EmpiricalMixturePosterior
from rec.utils import kl_estimate_with_mc
from rec.utils import plot_2d_distribution


def loss_function(log_q_cond_ak, log_p_ak, log_q_joint, aux_target, aux_prior, omega=5):
    with torch.no_grad():
        first_term_part_a = log_q_cond_ak - log_p_ak
    first_term_part_b = log_q_joint
    first_term = first_term_part_a * first_term_part_b

    second_term = log_q_cond_ak - log_p_ak
    with torch.no_grad():
        kl_minus_omega = kl_estimate_with_mc(aux_target, aux_prior) - omega
    return 2 * kl_minus_omega * (first_term + second_term)


def dummy_loss(aux_vars):
    return torch.sum(aux_vars)

#def debug():
# def optimisation_for_ak(aux_var_index, aux_var_history, q_joint_hist, aux_component_hist, post_object,
#                         num_mc_samples=50,
#                         epochs=100):
#     torch.autograd.set_detect_anomaly(True)
#     optim = torch.optim.Adam([post_object.coding_sampler.learnt_var], lr=1e-3)
#     for epoch in range(epochs):
#         optim.zero_grad()
#         pre_sum = post_object.coding_sampler.auxiliary_vars
#
#         #var = post_object.coding_sampler.learnt_var
#         #ones = torch.ones(3)
#         #ones[0] = var
#         #pre_sum = ones
#         loss = torch.sum(pre_sum)
#         print(f"{loss}")
#         loss.backward() #retain_graph=True)
#         optim.step()


def optimisation_for_ak_(aux_var_index, aux_var_history, q_joint_hist, aux_component_hist, post_object, n_trajectories,
                        num_mc_samples,
                        epochs=1000,
                         ):
    torch.autograd.set_detect_anomaly(True)
    optim = torch.optim.Adam([post_object.coding_sampler.learnt_var], lr=1e-4)
    for epoch in range(epochs):
        optim.zero_grad()
        full_losses = torch.zeros((n_trajectories,))
        current_kl = torch.zeros((n_trajectories,))
        for t in range(n_trajectories):
            # compute new conditional q
            aux_cond_post = post_object.q_ak_given_history(aux_var_history[:, t], aux_var_index, aux_component_hist[t], log_prob=True)
            sample_ak = aux_cond_post.sample((num_mc_samples,))
            log_q_cond_ak = aux_cond_post.log_prob(sample_ak)
            aux_prior = post_object.coding_sampler.auxiliary_coding_dist(aux_var_index)
            log_p_ak = aux_prior.log_prob(sample_ak)
            log_q_joint = q_joint_hist[t] + log_q_cond_ak
            losses = loss_function(log_q_cond_ak, log_p_ak, log_q_joint, aux_target=aux_cond_post, aux_prior=aux_prior)
            total_loss = torch.mean(losses, dim=0)
            full_losses[t] = total_loss
            current_kl[t] = kl_estimate_with_mc(aux_cond_post, aux_prior, num_samples=100)
        mean_loss = torch.mean(current_kl)
        if epoch % 10 == 0:
            print(f"Current KL estimate: {mean_loss}, Current sigma_k is :{post_object.coding_sampler.learnt_var}")
        total_loss.backward()
        optim.step()


def run_optimisation(mc_samples, target, n_auxiliary, problem_dim, n_trajectories):
    q_joint_history = torch.zeros((n_trajectories, mc_samples,)).detach()
    aux_component_hist = torch.zeros((n_trajectories, n_samples_from_target,)).detach()
    aux_var_history = torch.zeros((n_auxiliary, n_trajectories, problem_dim)).detach()
    best_aux_variances = torch.zeros((n_auxiliary,)).detach()
    for i in tqdm(range(n_auxiliary - 1)):
        coding_sampler = CodingSampler(
            problem_dimension=2,
            n_auxiliary=n_auxiliary,
            sigma_vector=best_aux_variances,
            optimise_index=i,
        )

        post_object = EmpiricalMixturePosterior(target=target,
                                                        n_samples_from_target=n_samples_from_target,
                                                        coding_sampler=coding_sampler)
        optimisation_for_ak_(i, aux_var_history, q_joint_history, aux_component_hist, post_object, num_mc_samples=mc_samples, n_trajectories=n_trajectories)
        # append new samples and update joint probs
        for t in range(n_trajectories):
            cond_post = post_object.q_ak_given_history(aux_var_history[:, t], i, aux_component_hist[t], log_prob=True)
            aux_var_history[i, t] = cond_post.sample((1,)).clone().detach()
            q_joint_history[t] += cond_post.log_prob(aux_var_history[i, t]).clone().detach()
            for z in range(n_samples_from_target):
                aux_component_hist[t, z] += cond_post.component_distribution.log_prob(aux_var_history[i, t])[z].clone().detach()

        best_aux_variances[i] = coding_sampler.auxiliary_vars[i].clone()

    return best_aux_variances


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
    plot_2d_distribution(target)
    plt.show()
    kl_q_p = dist.kl_divergence(target, dist.MultivariateNormal(loc=torch.zeros(2, ), covariance_matrix=torch.eye(2)))
    print(f"{kl_q_p}")
    omega = 5
    n_samples_from_target = 1
    n_auxiliary = math.ceil(kl_q_p / omega)
    print(f"{n_auxiliary} aux vars")
    n_samples_per_aux = math.ceil(torch.exp(torch.tensor(omega)))
    # coding_sampler = CodingSampler(
    #     problem_dimension=2,
    #     n_auxiliary=n_auxiliary,
    #     sigma_vector=None,
    #     optimise_index=0,
    #
    # )

    # auxiliary_posterior = EmpiricalMixturePosterior(target=target,
    #                                                 n_samples_from_target=n_samples_from_target,
    #                                                 coding_sampler=coding_sampler
    #                                                 )
    n_trajectories = 50
    mc_samples = 1000
    problem_dim = 2
    # optimisation_for_ak_(0, torch.zeros((n_trajectories, 1)), q_joint_hist=torch.zeros((n_trajectories,1)),
    #                     aux_component_hist=torch.zeros((n_trajectories, n_samples_from_target)), post_object=auxiliary_posterior)

    best_vars = run_optimisation(mc_samples, target, n_auxiliary, problem_dim, n_trajectories)
