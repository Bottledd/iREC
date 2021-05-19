import math
import torch
import torch.distributions as dist
import matplotlib.pyplot as plt
from tqdm import tqdm
from rec.distributions.CodingSampler import CodingSampler
from rec.distributions.EmpiricalMixturePosterior import EmpiricalMixturePosterior
from models.BayesianLinRegressor import BayesLinRegressor
from rec.utils import kl_estimate_with_mc, plot_running_sum, plot_2d_distribution
from rec.samplers.ImportanceSampling import ImportanceSampler
from rec.samplers.GreedySampling import GreedySampler


class Encoder:
    def __init__(self,
                 target,
                 initial_seed,
                 coding_sampler,
                 selection_sampler,
                 auxiliary_posterior,
                 omega,
                 n_samples_from_target,
                 epsilon=0,
                 ):
        # deduce dimension of problem from target mean
        self.problem_dimension = target.mean.shape[0]

        # to decide how many auxiliary vars need to compute KL(q||p)
        # first try with torch distributions

        # create dummy coding object to compute kl with target
        coding_z_prior = coding_sampler(problem_dimension=self.problem_dimension, n_auxiliary=1)
        try:
            kl_q_p = dist.kl_divergence(target, coding_z_prior)
        except:
            # need to do MC estimate
            kl_q_p = kl_estimate_with_mc(target, coding_z_prior)

        # compute parameters for auxiliary method
        self.target = target
        self.n_auxiliary = math.ceil(kl_q_p / omega)
        self.n_samples_per_aux = math.ceil(torch.exp(torch.tensor(omega * (1 + epsilon))))

        # instantiate the coding sampler and auxiliary posterior
        instance_coding_sampler = coding_sampler(problem_dimension=self.problem_dimension,
                                                 n_auxiliary=self.n_auxiliary)

        self.auxiliary_posterior = auxiliary_posterior(target,
                                                       n_samples_from_target,
                                                       instance_coding_sampler)

        # auxiliary samples that are selected
        # keep track of samples themselves and indices separately so we can check to ensure they agree
        self.selected_samples = torch.zeros((self.n_auxiliary, self.problem_dimension))
        self.selected_samples_indices = torch.zeros((self.n_auxiliary,))

        # need to keep track of joint probabilities
        self.selected_samples_joint_coding_log_prob = torch.zeros((1,))
        self.selected_samples_joint_posterior_log_prob = torch.zeros((1,))

        # need to keep track of mixing weights
        self.selected_samples_mixing_weights = torch.zeros((n_samples_from_target,))

        # store selection sampler object
        self.selection_sampler = selection_sampler

        # set the initial seed
        self.initial_seed = initial_seed

    def update_stored_samples(self, index, indices, samples, coding_dist=None, auxiliary_posterior_dist=None, final_sample=False):
        # update selected samples
        self.selected_samples[index] = samples
        self.selected_samples_indices[index] = indices

        if not final_sample:
            coding_log_probs = coding_dist.log_prob(samples)
            auxiliary_posterior_log_prob = auxiliary_posterior_dist.log_prob(samples)

            # add new log probs to compute new joint
            self.selected_samples_joint_coding_log_prob += coding_log_probs
            self.selected_samples_joint_posterior_log_prob += auxiliary_posterior_log_prob

            # update mixture_weights
            self.selected_samples_mixing_weights = auxiliary_posterior_dist.component_distribution.log_prob(samples)

    def run_encoder(self):
        for i in tqdm(range(self.n_auxiliary)):
            # set the seed
            seed = i + self.initial_seed
            # create new auxiliary prior distribution, p(a_k)
            auxiliary_prior = self.auxiliary_posterior.coding_sampler.auxiliary_coding_dist(i)

            if i < self.n_auxiliary - 1:
                # compute conditional posterior distribution, q(a_k | a_{1:k-1})
                auxiliary_conditional_posterior = self.auxiliary_posterior.q_ak_given_history(self.selected_samples,
                                                                                              i,
                                                                                              self.selected_samples_mixing_weights,
                                                                                              log_prob=True)

                # create new selection sampler object
                selection_sampler = self.selection_sampler(coding=auxiliary_prior,
                                                           target=auxiliary_conditional_posterior,
                                                           seed=seed,
                                                           num_samples=self.n_samples_per_aux,
                                                           coding_joint_history=self.selected_samples_joint_coding_log_prob,
                                                           target_joint_history=self.selected_samples_joint_posterior_log_prob)

                trial_aks = selection_sampler.get_samples_from_coder()

                # here would add stuff like repeating etc for beamsearch

                indices, samples = selection_sampler.choose_samples_to_transmit(samples=trial_aks)

                # update stored samples, indices and log probs
                self.update_stored_samples(i, indices, samples, auxiliary_prior, auxiliary_conditional_posterior)
            else:
                # cannot compute a conditional posterior for final sample
                selection_sampler = self.selection_sampler(coding=auxiliary_prior,
                                                           target=self.target,
                                                           seed=seed,
                                                           num_samples=self.n_samples_per_aux,
                                                           coding_joint_history=self.selected_samples_joint_coding_log_prob,
                                                           target_joint_history=self.selected_samples_joint_posterior_log_prob,
                                                           is_final_sample=True)
                trial_aks = selection_sampler.get_samples_from_coder()
                indices, samples = selection_sampler.choose_samples_to_transmit(samples=trial_aks,
                                                                                previous_samples=self.selected_samples,
                                                                                topk=1)

                # update stored samples, indices and log probs
                self.update_stored_samples(i, indices, samples, final_sample=True)

        return torch.sum(self.selected_samples, dim=0), self.selected_samples_indices


if __name__ == '__main__':
    initial_seed = 0
    blr = BayesLinRegressor(prior_mean=torch.tensor([0.0, 0.0]),
                            prior_alpha=0.001,
                            signal_std=5,
                            num_targets=15,
                            seed=initial_seed)
    blr.sample_feature_inputs()
    blr.sample_regression_targets()
    blr.posterior_update()
    target = blr.weight_posterior

    coding_sampler = CodingSampler
    auxiliary_posterior = EmpiricalMixturePosterior
    selection_sampler = GreedySampler
    omega = 3
    n_samples_from_target = 2

    encoder = Encoder(target,
                     initial_seed,
                     coding_sampler,
                     selection_sampler,
                     auxiliary_posterior,
                     omega,
                     n_samples_from_target,
                     epsilon=0.0,
                     )

    z, indices = encoder.run_encoder()

    plot_2d_distribution(target)
    plot_running_sum(encoder.selected_samples, plot_index_labels=False)
    plt.show()

