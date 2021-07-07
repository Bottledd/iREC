import torch
import torch.distributions as dist


class EmpiricalMixturePosterior:
    def __init__(self,
                 empirical_samples,
                 coding_sampler,
                 ):

        self.empirical_samples = empirical_samples
        self.n_samples_from_target = empirical_samples.shape[0]
        self.coding_sampler = coding_sampler
        self.problem_dimension = self.empirical_samples.shape[-1]

    def p_ak_given_history_and_z(self, aux_history, k, z_samples):
        """
        :param aux_history: Previous auxiliary variables
        :param k: Index of auxiliary variable in question
        :param z: Specific empirical sample
        :return: Gaussian distribution
        """

        b_k = torch.sum(aux_history[:k], dim=0)
        s_k_minus_one = torch.sum(self.coding_sampler.auxiliary_vars[k:])
        s_k = torch.sum(self.coding_sampler.auxiliary_vars[k+1:])

        mean_scalar = self.coding_sampler.auxiliary_vars[k] / s_k_minus_one
        variance_scalar = self.coding_sampler.auxiliary_vars[k] * s_k / s_k_minus_one

        mean = (z_samples - b_k) * mean_scalar
        #mean = z_samples * 0 + torch.tensor([30, 0])
        covariance = torch.eye(self.problem_dimension) * variance_scalar

        return mean, covariance

    def q_z_given_aks_mixing_weights(self, previous_conditional_joints, log_prob=False):
        """
        Turn p(a_{1:k} | z_d) into mixing weights
        :param previous_conditional_joints: p(a_{1:k} | z_d) for each z_d
        :return normalised mixing weights
        """
        if log_prob:
            return torch.softmax(previous_conditional_joints, dim=0)
            # below is using uniform mixing weights - seems to help
            #return torch.softmax(0 * previous_conditional_joints, dim=0)
        else:
            total_weight = torch.sum(previous_conditional_joints, dim=0)
            return previous_conditional_joints / total_weight

    def q_ak_given_history(self, aux_history, k, previous_conditional_coding_joints, log_prob=True):
        # first need to get mixing weights
        mixing_weights = self.q_z_given_aks_mixing_weights(previous_conditional_coding_joints, log_prob)

        component_means, component_variances = self.p_ak_given_history_and_z(aux_history, k, self.empirical_samples)

        # make categorical from mixing weights
        mixing_categorical = dist.categorical.Categorical(probs=mixing_weights)

        # make component distributions
        component_gaussians = dist.multivariate_normal.MultivariateNormal(loc=component_means, covariance_matrix=component_variances)

        # make mixture distribution
        gaussian_mixture = dist.mixture_same_family.MixtureSameFamily(mixing_categorical, component_gaussians)

        return gaussian_mixture
