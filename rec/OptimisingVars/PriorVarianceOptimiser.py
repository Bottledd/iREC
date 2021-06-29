import math
import torch
import torch.distributions as dist
import torch.nn as nn
from tqdm import tqdm
from models.SimpleBayesianLinRegressor import BayesLinRegressor
from rec.OptimisingVars.GMMPosteriorDist import GMMPosteriorDist
from rec.OptimisingVars.PriorDist import PriorDist
from rec.utils import kl_estimate_with_mc


class OptimisePriorVars(nn.Module):
    def __init__(self, target, omega, n_trajectories, n_auxiliary, n_samples_from_target, n_mc_samples):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.target = target
        self.dim = target.mean.shape[0]
        self.omega = omega
        self.n_trajectories = n_trajectories
        self.n_auxiliary = n_auxiliary
        self.n_mc_samples = n_mc_samples
        self.n_samples_from_target = n_samples_from_target
        self.q_joint_history = torch.zeros((n_trajectories,))
        self.aux_component_hist = torch.zeros((n_trajectories, n_samples_from_target))
        self.aux_trajectories = torch.zeros((n_trajectories, n_auxiliary, self.dim))
        self.best_aux_variances = torch.zeros((n_auxiliary,))

    def run_optimiser(self):
        for k in range(self.n_auxiliary - 1):
            # create new objects
            prior = PriorDist(optimise_index=k, current_optimised_sigmas=self.best_aux_variances,
                              target_dist=self.target, n_auxiliary=self.n_auxiliary)
            prior.to(self.device)
            posterior = GMMPosteriorDist(self.target, self.n_samples_from_target)
            posterior.to(self.device)
            self.optimise_sigma_k(k, prior, posterior)

            # now update values
            optimised_variance = prior.var_to_learn.clone().detach()
            sigmas = prior.auxiliary_vars.clone().detach()
            p_ak = prior.p_ak
            self.best_aux_variances[k] = optimised_variance
            samples, log_q, log_q_joint, q_ak = posterior(sigmas, self.aux_trajectories, k, self.aux_component_hist,
                                                          self.q_joint_history, n_mc_samples=1
                                                          )
            self.aux_trajectories[:, k] = samples.clone().detach()
            self.q_joint_history += log_q_joint.flatten().clone().detach()
            for z in range(self.n_samples_from_target):
                self.aux_component_hist[:, z] += q_ak.component_distribution.log_prob(self.aux_trajectories[:, k])[
                    z].clone().detach()

            final_kl = torch.mean(kl_estimate_with_mc(q_ak, p_ak, num_samples=1000))
            print(f"final kl for aux {k + 1} is {final_kl}")
        return self.best_aux_variances

    def optimise_sigma_k(self, index, prior, posterior, epochs=5000):
        optimiser = torch.optim.Adam(prior.parameters(), lr=1e-3)
        for epoch in tqdm(range(epochs)):
            optimiser.zero_grad()
            sigmas = prior.auxiliary_vars
            # sample a_k from q
            samples, log_q, log_q_joint, q_ak = posterior(sigmas, self.aux_trajectories, index, self.aux_component_hist,
                                                          self.q_joint_history, n_mc_samples=self.n_mc_samples)

            # compute log_prob under p
            log_p, p_ak = prior(samples)

            # calculate psuedo-loss
            loss = torch.mean(loss_function(log_q, log_p, log_q_joint, q_ak, p_ak, self.omega, self.n_mc_samples))
            loss.backward()
            optimiser.step()


def loss_function(log_q_cond_ak, log_p_ak, log_q_joint, aux_target, aux_prior, omega, n_mc_samples):
    """ Uses the REINFORCE gradient estimate. """
    with torch.no_grad():
        first_term_part_a = log_q_cond_ak - log_p_ak
    first_term_part_b = log_q_joint
    first_term = first_term_part_a * first_term_part_b
    second_term = log_q_cond_ak - log_p_ak
    with torch.no_grad():
        kl_minus_omega = kl_estimate_with_mc(aux_target, aux_prior, num_samples=n_mc_samples) - omega
    return torch.mean(2 * kl_minus_omega * (first_term + second_term), dim=0)


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
    n_samples_from_target = 1
    n_mc_samples = 10000
    # first try to compute KL between q(z) and p(z) with torch.distributions
    try:
        kl_q_p = dist.kl_divergence(target, dist.MultivariateNormal(loc=torch.zeros((dim,)),
                                                                    covariance_matrix=prior_var * torch.eye(
                                                                        dim)))
    except:
        kl_q_p = kl_estimate_with_mc(target=target, coder=dist.MultivariateNormal(loc=torch.zeros((dim,)),
                                                                                  covariance_matrix=prior_var * torch.eye(
                                                                                      dim)))

    n_auxiliary = math.ceil(kl_q_p / omega)
    print(f"Num of Aux is: {n_auxiliary}")
    optim = OptimisePriorVars(target, omega, n_trajectories, n_auxiliary, n_samples_from_target, n_mc_samples)
    optimised_vars = optim.run_optimiser()
