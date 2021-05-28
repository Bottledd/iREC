import torch
import torch.distributions as dist


class EmpiricalMixturePosterior:
    def __init__(self,
                 target,
                 n_samples_from_target,
                 coding_sampler,
                 ):

        self.empirical_samples = target.sample((n_samples_from_target,))
        self.n_samples_from_target = n_samples_from_target
        self.coding_sampler = coding_sampler
        self.problem_dimension = self.empirical_samples.shape[-1]

    def p_ak_given_history_and_z(self, k, aux_history, z):
        """
        :param aux_history: Previous auxiliary variables
        :param k: Index of auxiliary variable in question
        :param z: Specific empirical sample
        :return: Gaussian distribution
        """
        if k == 0:
            b_k = torch.sum(aux_history[:k], dim=0)
        else:
            b_k = torch.sum(aux_history[:, :k], dim=1)

        s_k_minus_one = torch.sum(self.coding_sampler.auxiliary_vars[k:])
        s_k = torch.sum(self.coding_sampler.auxiliary_vars[k+1:])

        mean_scalar = self.coding_sampler.auxiliary_vars[k] / s_k_minus_one
        variance_scalar = self.coding_sampler.auxiliary_vars[k] * s_k / s_k_minus_one

        mean = (z - b_k) * mean_scalar
        covariance = torch.eye(self.problem_dimension) * variance_scalar

        return mean, covariance

    def q_z_given_aks_mixing_weights(self, k, previous_conditional_joints, log_prob=False):
        """
        Turn p(a_{1:k} | z_d) into mixing weights
        :param previous_conditional_joints: p(a_{1:k} | z_d) for each z_d
        :return normalised mixing weights
        """
        if log_prob:
            if k == 0:
                return torch.softmax(previous_conditional_joints, dim=0)
            else:
                return torch.softmax(previous_conditional_joints, dim=1)
        else:
            if k == 0:
                total_weight = torch.sum(previous_conditional_joints, dim=0)
                return previous_conditional_joints / total_weight
            else:
                total_weight = torch.sum(previous_conditional_joints, dim=1)
                return previous_conditional_joints / total_weight

    def q_ak_given_history(self, aux_history, k, previous_conditional_coding_joints, log_prob=True):
        # first need to get mixing weights
        mixing_weights = self.q_z_given_aks_mixing_weights(k, previous_conditional_coding_joints, log_prob)

        if k == 0:
            component_means = torch.zeros((self.n_samples_from_target, self.problem_dimension))
            component_variances = torch.zeros(
                (self.n_samples_from_target, self.problem_dimension, self.problem_dimension))

            for i, z_d in enumerate(self.empirical_samples):
                component_means[i], component_variances[i] = self.p_ak_given_history_and_z(k, aux_history, z_d)

            # make categorical from mixing weights
            mixing_categorical = dist.categorical.Categorical(probs=mixing_weights)

            # make component distributions
            component_gaussians = dist.multivariate_normal.MultivariateNormal(loc=component_means, covariance_matrix=component_variances)

            # make mixture distribution
            gaussian_mixture = dist.mixture_same_family.MixtureSameFamily(mixing_categorical, component_gaussians)

            return gaussian_mixture

        else:

            # compute components p(ak | a_{1:k}, z_d)
            # shape for mean is (batch_size, n_samples_from_target, problem_dim)
            # shape for covariances is (batch_size, n_samples_from_target, problem_dim, problem_dim)
            component_means = torch.zeros((aux_history.shape[0], self.n_samples_from_target, self.problem_dimension))
            component_variances = torch.zeros((aux_history.shape[0], self.n_samples_from_target, self.problem_dimension, self.problem_dimension))

            # TODO instead of loop, can try flattening and then reshaping
            for i, z_d in enumerate(self.empirical_samples):
                component_means[:, i], component_variances[:, i] = self.p_ak_given_history_and_z(k, aux_history, z_d)

            # make categorical from mixing weights
            mixing_categorical = dist.categorical.Categorical(probs=mixing_weights)

            # make component distributions
            component_gaussians = dist.multivariate_normal.MultivariateNormal(loc=component_means, covariance_matrix=component_variances)

            # make mixture distribution
            gaussian_mixture = dist.MixtureSameFamily(mixing_categorical, component_gaussians)

            return gaussian_mixture
