import torch
import torch.distributions as dist
import torch.nn as nn


class GMMPosteriorDist(nn.Module):
    def __init__(self, target, n_samples_from_target):
        self.empirical_samples = target.sample((n_samples_from_target,))
        self.n_samples_from_target = n_samples_from_target
        self.dim = self.empirical_samples.shape[-1]
        super(GMMPosteriorDist, self).__init__()

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
        covariance = torch.eye(self.dim) * variance_scalar

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

        component_means = torch.zeros((aux_history.shape[0], self.n_samples_from_target, self.dim))
        component_variances = torch.zeros(
            (aux_history.shape[0], self.n_samples_from_target, self.dim, self.dim))

        for i, z_d in enumerate(self.empirical_samples):
            component_means[:, i], component_variances[:, i] = self.p_ak_given_history_and_z(aux_history, k, z_d, sigmas)

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
        samples = q_ak_given_hist.sample((n_mc_samples,))

        # compute log probs
        log_q = q_ak_given_hist.log_prob(samples)
        log_q_joint = log_q + q_joint_history

        return samples, log_q, log_q_joint, q_ak_given_hist
