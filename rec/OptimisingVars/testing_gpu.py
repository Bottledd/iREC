import math
import torch
import torch.distributions as dist
import torch.nn as nn
from tqdm.notebook import tqdm
from models.BayesianLinRegressor import BayesLinRegressor
from rec.utils import kl_estimate_with_mc


class PriorDist(nn.Module):
    def __init__(self, optimise_index, n_auxiliary, initial_sigmas=None, current_optimised_sigmas=torch.tensor([]),
                 prior_var=1, target_dist=None,
                 omega=5):
        super(PriorDist, self).__init__()
        self.dim = target_dist.mean.shape[0]
        self.omega = omega

        if initial_sigmas is not None:
            self.register_buffer('already_optimised_aux_vars', current_optimised_sigmas[:optimise_index])
            self.register_buffer('aux_vars_left_to_optimise', initial_sigmas[optimise_index + 1:])
            self.var_to_learn = nn.Parameter(torch.tensor((initial_sigmas[optimise_index],)))
        else:
            remaining_variance = prior_var - torch.sum(current_optimised_sigmas[:optimise_index])
            new_uniform_variance = remaining_variance / (n_auxiliary - optimise_index)
            self.register_buffer('already_optimised_aux_vars', current_optimised_sigmas[:optimise_index])
            self.register_buffer('aux_vars_left_to_optimise',
                                 torch.ones((n_auxiliary - optimise_index - 1)).to(
                                     self.already_optimised_aux_vars.device) * new_uniform_variance)
            self.var_to_learn = nn.Parameter(torch.tensor((new_uniform_variance,)))

    def forward(self, samples):
        p_ak = self.p_ak
        log_prob = p_ak.log_prob(samples)
        return log_prob, p_ak

    @property
    def p_ak(self):
        mean = torch.zeros((self.dim,)).to(self.var_to_learn.device)
        covariance = self.var_to_learn * torch.eye(self.dim).to(self.var_to_learn.device)
        return dist.MultivariateNormal(loc=mean,
                                       covariance_matrix=covariance)

    @property
    def auxiliary_vars(self):
        x = torch.cat([self.already_optimised_aux_vars, self.var_to_learn, self.aux_vars_left_to_optimise])
        return x


class GMMPosteriorDist(nn.Module):
    def __init__(self, target, n_samples_from_target):
        super(GMMPosteriorDist, self).__init__()
        self.register_buffer('empirical_samples', target.sample((n_samples_from_target,)))
        self.n_samples_from_target = n_samples_from_target
        self.dim = self.empirical_samples.shape[-1]

    def p_ak_given_history_and_z(self, aux_history, k, z, sigmas):
        """
        :param aux_history: Previous auxiliary variables in shape (n_traj, n_aux, dim)
        :param k: Index of auxiliary variable in question
        :param z: Specific empirical sample
        :return: Gaussian distribution
        """

        b_k = torch.sum(aux_history, dim=1)
        s_k_minus_one = torch.sum(sigmas[k:])
        s_k = torch.sum(sigmas[k + 1:])

        mean_scalar = sigmas[k] / s_k_minus_one

        variance_scalar = sigmas[k] * s_k / s_k_minus_one

        mean = (z - b_k) * mean_scalar
        covariance = torch.eye(self.dim).to(self.empirical_samples.device) * variance_scalar

        return mean, covariance

    def q_z_given_aks_mixing_weights(self, previous_conditional_joints, log_prob=False):
        """
        Turn p(a_{1:k} | z_d) into mixing weights
        :param previous_conditional_joints: p(a_{1:k} | z_d) for each z_d, in shape (n_traj, n_empirical_samples)
        :return normalised mixing weights
        """
        if log_prob:
            return torch.softmax(previous_conditional_joints, dim=1)
        else:
            total_weight = torch.sum(previous_conditional_joints, dim=1)
            return previous_conditional_joints / total_weight

    def q_ak_given_history(self, aux_history, k, previous_conditional_coding_joints, sigmas, log_prob=True):
        # first need to get mixing weights
        mixing_weights = self.q_z_given_aks_mixing_weights(previous_conditional_coding_joints, log_prob)

        component_means = torch.zeros((aux_history.shape[0], self.n_samples_from_target, self.dim)).to(
            self.empirical_samples.device)
        component_variances = torch.zeros(
            (aux_history.shape[0], self.n_samples_from_target, self.dim, self.dim)).to(self.empirical_samples.device)

        for i, z_d in enumerate(self.empirical_samples):
            component_means[:, i], component_variances[:, i] = self.p_ak_given_history_and_z(aux_history, k, z_d,
                                                                                             sigmas)

        # make categorical from mixing weights
        mixing_categorical = dist.Categorical(probs=mixing_weights)

        # make component distributions
        component_gaussians = dist.MultivariateNormal(loc=component_means, covariance_matrix=component_variances)

        # make mixture distribution
        gaussian_mixture = dist.MixtureSameFamily(mixing_categorical, component_gaussians)

        return gaussian_mixture

    def forward(self, sigmas, aux_trajectories, index, aux_component_hist, q_joint_history, n_mc_samples=100):
        # create q_ak_given_hist
        q_ak_given_hist = self.q_ak_given_history(aux_trajectories, index, aux_component_hist, sigmas=sigmas)

        # sample potential aks, should be shape (n_traj, n_mc_samples, dim)
        samples = q_ak_given_hist.sample((n_mc_samples,)).to(self.empirical_samples.device)

        # compute log probs
        log_q = q_ak_given_hist.log_prob(samples)
        log_q_joint = log_q + q_joint_history

        return samples, log_q, log_q_joint, q_ak_given_hist


class OptimisePriorVars(nn.Module):
    def __init__(self, target, omega, n_trajectories, n_auxiliary, n_samples_from_target, n_mc_samples,
                 initial_sigmas=None):
        super(OptimisePriorVars, self).__init__()
        self.target = target
        self.dim = target.mean.shape[0]
        self.omega = omega
        self.n_trajectories = n_trajectories
        self.n_auxiliary = n_auxiliary
        self.n_mc_samples = n_mc_samples
        self.n_samples_from_target = n_samples_from_target
        if initial_sigmas is not None:
            self.register_buffer("initial_sigmas", initial_sigmas)
        else:
            self.initial_sigmas = None
        self.register_buffer("q_joint_history", torch.zeros((n_trajectories,)))
        self.register_buffer("aux_component_hist", torch.zeros((n_trajectories, n_samples_from_target)))
        self.register_buffer("aux_trajectories", torch.zeros((n_trajectories, n_auxiliary, self.dim)))
        self.register_buffer("best_aux_variances", torch.zeros((n_auxiliary,)))
        self.signal_to_noise = []
        # self.q_joint_history = torch.zeros((n_trajectories,))
        # self.aux_component_hist = torch.zeros((n_trajectories, n_samples_from_target))
        # self.aux_trajectories = torch.zeros((n_trajectories, n_auxiliary, self.dim))
        # self.best_aux_variances = torch.zeros((n_auxiliary,))

    def run_optimiser(self):
        for k in range(self.n_auxiliary - 1):
            # create new objects
            prior = PriorDist(optimise_index=k, current_optimised_sigmas=self.best_aux_variances,
                              initial_sigmas=self.initial_sigmas, target_dist=self.target, n_auxiliary=self.n_auxiliary)
            prior.to(self.q_joint_history.device)
            initial_sigmas = prior.auxiliary_vars.clone().detach()[k]
            print(f"Initial sigmas for aux {k} is {initial_sigmas}")
            posterior = GMMPosteriorDist(self.target, self.n_samples_from_target)
            posterior.to(self.q_joint_history.device)
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

            final_kl = torch.mean(kl_estimate_with_mc(q_ak, p_ak, num_samples=self.n_mc_samples))
            print(f"final kl for aux {k + 1} is {final_kl} and sigma is {optimised_variance}")
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
            all_losses = loss_function(log_q, log_p, log_q_joint, q_ak, p_ak, self.omega, self.n_mc_samples)
            loss = torch.mean(all_losses)
            loss_std = torch.std(all_losses)
            self.signal_to_noise.append(loss / loss_std)
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

    dim = target.mean.shape[0]
    prior_var = 1
    omega = 5
    n_trajectories = 64
    n_samples_from_target = 1
    n_mc_samples = 40000
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
    previous_best = torch.tensor([0.0419, 0.0428, 0.0413, 0.0421, 0.0403, 0.0405, 0.0414, 0.0401, 0.0403,
                                  0.0420, 0.0412, 0.0436, 0.0408, 0.0396, 0.0412, 0.0431, 0.0400, 0.0401,
                                  0.0334, 0.0405, 0.0368, 0.0355, 0.0171, 0.0094, 0.0851])
    # previous_best=None
    print(f"Num of Aux is: {n_auxiliary}")
    optim = OptimisePriorVars(target, omega, n_trajectories, n_auxiliary, n_samples_from_target, n_mc_samples,
                              initial_sigmas=previous_best)
    optim.to(device)
    optimised_vars = optim.run_optimiser()
