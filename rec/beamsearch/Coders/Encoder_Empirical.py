import math

import matplotlib.pyplot as plt
import torch
import torch.distributions as dist
from tqdm import tqdm

from models.SimpleBayesianLinRegressor import BayesLinRegressor
from rec.beamsearch.distributions.CodingSampler import CodingSampler
from rec.beamsearch.distributions.EmpiricalMixturePosterior import EmpiricalMixturePosterior
from rec.beamsearch.samplers.GreedySampling_BNNs import GreedySampler
from rec.beamsearch.utils import convert_flattened_indices
from rec.utils import kl_estimate_with_mc, plot_running_sum_1d, plot_running_sum_2d, plot_2d_distribution, plot_1d_distribution, plot_samples_in_2d, plot_pairs_of_samples
from rec.OptimisingVars.FinalJointOptimiser import FinalJointOptimiser
from models.BNNs.BNN_for_HMC import BNN_for_HMC


class Encoder:
    def __init__(self,
                 model: BNN_for_HMC,
                 x_data,
                 y_data,
                 empirical_samples,
                 initial_seed,
                 coding_sampler,
                 selection_sampler,
                 auxiliary_posterior,
                 omega,
                 beamwidth,
                 epsilon=0.,
                 prior_var=1.,
                 total_kl=None):
        # store the base NN
        self.model = model
        self.x_data = x_data
        self.y_data = y_data

        # instantiate the beamwidth
        self.beamwidth = beamwidth

        # deduce dimension of problem from target mean
        self.problem_dimension = empirical_samples[0].shape[0]

        if total_kl is None:
            # create dummy coding object to compute kl with target
            coding_z_prior = coding_sampler(problem_dimension=self.problem_dimension, n_auxiliary=1, var=prior_var)

            # for each empirical sample, compute MC KL estimate
            # first need to compute estimate of KL with joint
            likelihoods = torch.empty([0])
            for sample in empirical_samples:
                # make a model
                self.model.make_weights_from_sample(sample)

                likelihood = self.model.data_likelihood(self.x_data, self.y_data)
                likelihoods = torch.cat([likelihoods, likelihood[None]])

            kl_term_1 = likelihoods.mean()

            # to work out second KL term p(D) need to sample from prior and measure likelihood
            prior_samples = coding_z_prior.sample((500,))
            log_marginal_data = torch.zeros([0])
            for sample in prior_samples:
                # make a model
                self.model.make_weights_from_sample(sample)

                log_marginal = self.model.data_likelihood(self.x_data, self.y_data)
                log_marginal_data = torch.cat([log_marginal_data, log_marginal[None]])

            kl_term_2 = - torch.log(torch.tensor(500)) + torch.logsumexp(log_marginal_data, dim=0)


            self.total_kl = kl_term_1 - kl_term_2
            print(self.total_kl)

        else:
            self.total_kl = total_kl

        # compute parameters for auxiliary method
        self.n_auxiliary = math.ceil(self.total_kl / omega)
        self.n_samples_per_aux = math.ceil(torch.exp(torch.tensor(omega * (1 + epsilon))))
        self.n_samples_from_target = empirical_samples.shape[0]

        # instantiate the coding sampler and auxiliary posterior
        instance_coding_sampler = coding_sampler(problem_dimension=self.problem_dimension,
                                                 n_auxiliary=self.n_auxiliary,
                                                 var=1)

        self.auxiliary_posterior = auxiliary_posterior(empirical_samples,
                                                       instance_coding_sampler)

        # auxiliary samples that are selected
        # keep track of samples themselves and indices separately so we can check to ensure they agree
        self.selected_samples = torch.ones((self.beamwidth, self.n_auxiliary, self.problem_dimension)) * float('nan')
        self.selected_samples_indices = torch.zeros((self.beamwidth, self.n_auxiliary,))

        # need to keep track of joint probabilities
        self.selected_samples_joint_coding_log_prob = torch.zeros((self.beamwidth,))
        self.selected_samples_joint_posterior_log_prob = torch.zeros((self.beamwidth,))

        # need to keep track of mixing weights
        self.selected_samples_mixing_weights = torch.ones((self.beamwidth, self.n_samples_from_target,)) * float('nan')

        # store selection sampler object
        self.selection_sampler = selection_sampler

        # set the initial seed
        self.initial_seed = initial_seed

        # store kls between aux variables
        self.aux_var_kl = torch.zeros((self.n_auxiliary,))

    def update_stored_samples(self, index, samples, beam_indices=None, sample_indices=None, coding_dist=None,
                              auxiliary_posterior_dist=None, final_sample=False):
        if index == 0:
            # number of samples to add to beam
            n_samples_to_add = samples.shape[0]
            # update selected samples
            self.selected_samples[:n_samples_to_add, index] = samples
            self.selected_samples_indices[:n_samples_to_add, index] = sample_indices

            if not final_sample:
                coding_log_probs = coding_dist.log_prob(samples)
                auxiliary_posterior_log_prob = auxiliary_posterior_dist.log_prob(samples)

                # add new log probs to compute new joint
                self.selected_samples_joint_coding_log_prob[:n_samples_to_add] += coding_log_probs
                self.selected_samples_joint_posterior_log_prob[:n_samples_to_add] += auxiliary_posterior_log_prob

                # update mixture_weights
                for i, (sample) in enumerate(samples):
                    self.selected_samples_mixing_weights[i] = auxiliary_posterior_dist.component_distribution.log_prob(
                        sample)
        else:
            # number of samples to add to beam
            n_samples_to_add = samples.shape[0]

            # update selected samples
            self.selected_samples[:n_samples_to_add, index] = samples
            self.selected_samples_indices[:n_samples_to_add, index] = sample_indices

            # update beams
            self.selected_samples[:n_samples_to_add, :index] = self.selected_samples[:, :index][beam_indices]
            self.selected_samples_indices[:n_samples_to_add, :index] = self.selected_samples_indices[:, :index][
                beam_indices]

            if not final_sample:
                coding_log_probs = coding_dist.log_prob(samples)
                auxiliary_posterior_log_prob = auxiliary_posterior_dist.log_prob(samples)

                # add new log probs to compute new joint
                self.selected_samples_joint_coding_log_prob[:n_samples_to_add] += coding_log_probs
                self.selected_samples_joint_posterior_log_prob[:n_samples_to_add] += auxiliary_posterior_log_prob

                # update mixture_weights
                new_mixing_weights = torch.zeros((n_samples_to_add, self.n_samples_from_target))
                for i, (sample, beam) in enumerate(zip(samples, beam_indices)):
                    new_mixing_weights[i] = self.selected_samples_mixing_weights[beam] + \
                                            auxiliary_posterior_dist.component_distribution.log_prob(sample)[beam]
                self.selected_samples_mixing_weights[:n_samples_to_add] = new_mixing_weights

    def run_encoder(self):
        #for i in tqdm(range(self.n_auxiliary)):
        for i in range(self.n_auxiliary):
            # set the seed
            seed = i + self.initial_seed
            # create new auxiliary prior distribution, p(a_k)
            auxiliary_prior = self.auxiliary_posterior.coding_sampler.auxiliary_coding_dist(i)

            # need to do something different for first auxiliary variable
            if i == 0 and self.n_auxiliary > 1:
                # if beamwidth > n_samples_per_aux select at most n_samples_per_aux on first round
                if self.beamwidth > self.n_samples_per_aux:
                    n_selections = self.n_samples_per_aux
                else:
                    n_selections = self.beamwidth

                # compute conditional posterior distribution, q(a_k | a_{1:k-1})
                # note since all starting values are zero we can select the 0th index for many
                auxiliary_conditional_posterior = self.auxiliary_posterior.q_ak_given_history(self.selected_samples[0],
                                                                                              i,
                                                                                              torch.zeros_like(self.selected_samples_mixing_weights[0]),
                                                                                              log_prob=True)

                # create new selection sampler object
                selection_sampler = self.selection_sampler(self.model,
                                                           self.x_data,
                                                           self.y_data,
                                                           coding=auxiliary_prior,
                                                           target=auxiliary_conditional_posterior,
                                                           seed=seed,
                                                           num_samples=self.n_samples_per_aux,
                                                           coding_joint_history=
                                                           self.selected_samples_joint_coding_log_prob[0],
                                                           target_joint_history=
                                                           self.selected_samples_joint_posterior_log_prob[0],
                                                           topk=n_selections,
                                                           is_first_index=True)

                trial_aks = selection_sampler.get_samples_from_coder()

                # here would add stuff like repeating etc for beamsearch

                indices, samples = selection_sampler.choose_samples_to_transmit(samples=trial_aks)

                # update stored samples, indices and log probs

                self.update_stored_samples(i, samples=samples, sample_indices=indices, coding_dist=auxiliary_prior,
                                           auxiliary_posterior_dist=auxiliary_conditional_posterior)

            elif self.n_auxiliary - 1 > i > 0:
                # first need to ignore any unfilled beams from previous run, i.e. mask 0 values
                mask = self.selected_samples[:, i - 1].ne(float('nan'))
                mixing_weights_mask = self.selected_samples_mixing_weights.ne(float('nan'))
                expanded_mask = torch.repeat_interleave(mask[:, None, :], self.n_auxiliary, dim=1)
                single_dim_mask = mask[:, 0]
                pruned_beams = torch.masked_select(self.selected_samples, expanded_mask).reshape(-1, self.n_auxiliary,
                                                                                                 self.problem_dimension)
                pruned_mixing_weights = torch.masked_select(self.selected_samples_mixing_weights,
                                                            mixing_weights_mask).reshape(-1, self.n_samples_from_target)
                pruned_coding_log_prob = torch.masked_select(self.selected_samples_joint_coding_log_prob,
                                                             single_dim_mask)
                pruned_posterior_log_prob = torch.masked_select(self.selected_samples_joint_posterior_log_prob,
                                                                single_dim_mask)
                # need to expand the beams to have B*M elements
                tiled_beams = torch.tile(pruned_beams, (self.n_samples_per_aux, 1, 1))
                tiled_mixing_weights = torch.tile(pruned_mixing_weights, (self.n_samples_per_aux, 1))
                # compute conditional posterior distribution, q(a_k | a_{1:k-1})
                auxiliary_conditional_posterior = self.auxiliary_posterior.q_ak_given_history(tiled_beams,
                                                                                              i,
                                                                                              tiled_mixing_weights,
                                                                                              log_prob=True)
                # need to see if masked B_hat * M > B
                if self.beamwidth > tiled_beams.shape[0]:
                    n_new_beams = tiled_beams.shape[0]
                else:
                    n_new_beams = self.beamwidth
                # create new selection sampler object
                selection_sampler = self.selection_sampler(self.model,
                                                           self.x_data,
                                                           self.y_data,
                                                           coding=auxiliary_prior,
                                                           target=auxiliary_conditional_posterior,
                                                           seed=seed,
                                                           num_samples=self.n_samples_per_aux,
                                                           coding_joint_history=pruned_coding_log_prob,
                                                           target_joint_history=pruned_posterior_log_prob,
                                                           topk=n_new_beams)

                trial_aks = selection_sampler.get_samples_from_coder()

                # here would add stuff like repeating etc for beamsearch

                repeated_aks = torch.repeat_interleave(trial_aks, pruned_beams.shape[0], dim=0)

                indices, samples = selection_sampler.choose_samples_to_transmit(samples=repeated_aks,
                                                                                n_samples_per_aux=self.n_samples_per_aux)

                # now need to convert indices from flattened to refer to specific beam/ak sample
                beam_indices, sample_indices = convert_flattened_indices(indices, beamwidth=pruned_beams.shape[0])

                # make new auxiliary distribution using current beams
                auxiliary_conditional_posterior_untiled = self.auxiliary_posterior.q_ak_given_history(
                    self.selected_samples[beam_indices],
                    i,
                    self.selected_samples_mixing_weights[beam_indices],
                    log_prob=True)
                # update stored samples, indices and log probs

                self.update_stored_samples(i, beam_indices=beam_indices, sample_indices=sample_indices, samples=samples,
                                           coding_dist=auxiliary_prior,
                                           auxiliary_posterior_dist=auxiliary_conditional_posterior_untiled)
            else:
                # cannot compute a conditional posterior for final sample
                selection_sampler = self.selection_sampler(self.model,
                                                           self.x_data,
                                                           self.y_data,
                                                           coding=auxiliary_prior,
                                                           target=None,
                                                           seed=seed,
                                                           num_samples=self.n_samples_per_aux,
                                                           coding_joint_history=self.selected_samples_joint_coding_log_prob,
                                                           target_joint_history=self.selected_samples_joint_posterior_log_prob,
                                                           is_final_sample=True,
                                                           topk=self.beamwidth)
                trial_aks = selection_sampler.get_samples_from_coder()
                repeated_aks = torch.repeat_interleave(trial_aks, self.beamwidth, dim=0)
                indices, samples = selection_sampler.choose_samples_to_transmit(samples=repeated_aks,
                                                                                previous_samples=self.selected_samples[:, :i], )

                # now need to convert indices from flattened to refer to specific beam/ak sample
                beam_indices, sample_indices = convert_flattened_indices(indices, beamwidth=self.beamwidth)

                # update stored samples, indices and log probs
                self.update_stored_samples(i, beam_indices=beam_indices, sample_indices=sample_indices, samples=samples,
                                           final_sample=True)

        return torch.sum(self.selected_samples, dim=1), self.selected_samples_indices