import math

import matplotlib.pyplot as plt
import torch
import torch.distributions as dist
from tqdm import tqdm

from models.BayesianLinRegressor import BayesLinRegressor
from rec.beamsearch.distributions.CodingSampler import CodingSampler
from rec.beamsearch.distributions.EmpiricalMixturePosterior import EmpiricalMixturePosterior
from rec.beamsearch.samplers.GreedySampling import GreedySampler
from rec.beamsearch.utils import convert_flattened_indices
from rec.utils import kl_estimate_with_mc, plot_running_sum, plot_2d_distribution


class Encoder:
    def __init__(self,
                 target,
                 initial_seed,
                 coding_sampler,
                 selection_sampler,
                 auxiliary_posterior,
                 omega,
                 n_samples_from_target,
                 beamwidth,
                 epsilon=0,
                 ):
        # instantiate the beamwidth
        self.beamwidth = beamwidth

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
        self.selected_samples = torch.zeros((self.beamwidth, self.n_auxiliary, self.problem_dimension))
        self.selected_samples_indices = torch.zeros((self.beamwidth, self.n_auxiliary,))

        # need to keep track of joint probabilities
        self.selected_samples_joint_coding_log_prob = torch.zeros((self.beamwidth,))
        self.selected_samples_joint_posterior_log_prob = torch.zeros((self.beamwidth,))

        # need to keep track of mixing weights
        self.selected_samples_mixing_weights = torch.zeros((self.beamwidth, n_samples_from_target,))

        # store selection sampler object
        self.selection_sampler = selection_sampler

        # set the initial seed
        self.initial_seed = initial_seed

    def update_stored_samples(self, index, samples, beam_indices=None, sample_indices=None, coding_dist=None,
                              auxiliary_posterior_dist=None, final_sample=False):
        if index == 0:
            # update selected samples
            self.selected_samples[:, index] = samples
            self.selected_samples_indices[:, index] = sample_indices

            if not final_sample:
                coding_log_probs = coding_dist.log_prob(samples)
                auxiliary_posterior_log_prob = auxiliary_posterior_dist.log_prob(samples)

                # add new log probs to compute new joint
                self.selected_samples_joint_coding_log_prob += coding_log_probs
                self.selected_samples_joint_posterior_log_prob += auxiliary_posterior_log_prob

                # update mixture_weights
                for i, (sample) in enumerate(samples):
                    self.selected_samples_mixing_weights[i] = auxiliary_posterior_dist.component_distribution.log_prob(
                        sample)
        else:
            # update selected samples
            self.selected_samples[:, index] = samples
            self.selected_samples_indices[:, index] = sample_indices

            # update beams
            self.selected_samples[:, :index] = self.selected_samples[:, :index][beam_indices]
            self.selected_samples_indices[:, :index] = self.selected_samples_indices[:, :index][beam_indices]

            if not final_sample:
                coding_log_probs = coding_dist.log_prob(samples)
                auxiliary_posterior_log_prob = auxiliary_posterior_dist.log_prob(samples)

                # add new log probs to compute new joint
                self.selected_samples_joint_coding_log_prob += coding_log_probs
                self.selected_samples_joint_posterior_log_prob += auxiliary_posterior_log_prob

                # update mixture_weights
                for i, (sample, beam) in enumerate(zip(samples, beam_indices)):
                    self.selected_samples_mixing_weights[i] += \
                    auxiliary_posterior_dist.component_distribution.log_prob(sample)[beam]

    def run_encoder(self):
        for i in tqdm(range(self.n_auxiliary)):
            # set the seed
            seed = i + self.initial_seed
            # create new auxiliary prior distribution, p(a_k)
            auxiliary_prior = self.auxiliary_posterior.coding_sampler.auxiliary_coding_dist(i)

            # need to do something different for first auxiliary variable
            if i == 0:
                # compute conditional posterior distribution, q(a_k | a_{1:k-1})
                # note since all starting values are zero we can select the 0th index for many
                auxiliary_conditional_posterior = self.auxiliary_posterior.q_ak_given_history(self.selected_samples[0],
                                                                                              i,
                                                                                              self.selected_samples_mixing_weights[
                                                                                                  0],
                                                                                              log_prob=True)

                # create new selection sampler object
                selection_sampler = self.selection_sampler(coding=auxiliary_prior,
                                                           target=auxiliary_conditional_posterior,
                                                           seed=seed,
                                                           num_samples=self.n_samples_per_aux,
                                                           coding_joint_history=
                                                           self.selected_samples_joint_coding_log_prob[0],
                                                           target_joint_history=
                                                           self.selected_samples_joint_posterior_log_prob[0],
                                                           topk=self.beamwidth,
                                                           is_first_index=True)

                trial_aks = selection_sampler.get_samples_from_coder()

                # here would add stuff like repeating etc for beamsearch

                indices, samples = selection_sampler.choose_samples_to_transmit(samples=trial_aks)

                # update stored samples, indices and log probs

                self.update_stored_samples(i, samples=samples, sample_indices=indices, coding_dist=auxiliary_prior,
                                           auxiliary_posterior_dist=auxiliary_conditional_posterior)

            elif self.n_auxiliary - 1 > i > 0:
                # need to expand the beams to have B*M elements
                tiled_beams = torch.tile(self.selected_samples, (self.n_samples_per_aux, 1, 1))
                tiled_mixing_weights = torch.tile(self.selected_samples_mixing_weights, (self.n_samples_per_aux, 1))

                # compute conditional posterior distribution, q(a_k | a_{1:k-1})
                auxiliary_conditional_posterior = self.auxiliary_posterior.q_ak_given_history(tiled_beams,
                                                                                              i,
                                                                                              tiled_mixing_weights,
                                                                                              log_prob=True)

                # create new selection sampler object
                selection_sampler = self.selection_sampler(coding=auxiliary_prior,
                                                           target=auxiliary_conditional_posterior,
                                                           seed=seed,
                                                           num_samples=self.n_samples_per_aux,
                                                           coding_joint_history=self.selected_samples_joint_coding_log_prob,
                                                           target_joint_history=self.selected_samples_joint_posterior_log_prob,
                                                           topk=self.beamwidth)

                trial_aks = selection_sampler.get_samples_from_coder()

                # here would add stuff like repeating etc for beamsearch

                repeated_aks = torch.repeat_interleave(trial_aks, self.beamwidth, dim=0)

                indices, samples = selection_sampler.choose_samples_to_transmit(samples=repeated_aks)

                # now need to convert indices from flattened to refer to specific beam/ak sample
                beam_indices, sample_indices = convert_flattened_indices(indices, beamwidth=self.beamwidth)

                # make new auxiliary distribution using current beams
                auxiliary_conditional_posterior_untiled = self.auxiliary_posterior.q_ak_given_history(
                    self.selected_samples,
                    i,
                    self.selected_samples_mixing_weights,
                    log_prob=True)
                # update stored samples, indices and log probs

                self.update_stored_samples(i, beam_indices=beam_indices, sample_indices=sample_indices, samples=samples,
                                           coding_dist=auxiliary_prior,
                                           auxiliary_posterior_dist=auxiliary_conditional_posterior_untiled)
            else:
                # cannot compute a conditional posterior for final sample
                selection_sampler = self.selection_sampler(coding=auxiliary_prior,
                                                           target=self.target,
                                                           seed=seed,
                                                           num_samples=self.n_samples_per_aux,
                                                           coding_joint_history=self.selected_samples_joint_coding_log_prob,
                                                           target_joint_history=self.selected_samples_joint_posterior_log_prob,
                                                           is_final_sample=True,
                                                           topk=self.beamwidth)
                trial_aks = selection_sampler.get_samples_from_coder()
                repeated_aks = torch.repeat_interleave(trial_aks, self.beamwidth, dim=0)
                indices, samples = selection_sampler.choose_samples_to_transmit(samples=repeated_aks,
                                                                                previous_samples=self.selected_samples,)

                # now need to convert indices from flattened to refer to specific beam/ak sample
                beam_indices, sample_indices = convert_flattened_indices(indices, beamwidth=self.beamwidth)

                # update stored samples, indices and log probs
                self.update_stored_samples(i, beam_indices=beam_indices, sample_indices=sample_indices, samples=samples,
                                           final_sample=True)

        return torch.sum(self.selected_samples, dim=1), self.selected_samples_indices


if __name__ == '__main__':
    initial_seed = 2
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
    omega = 9
    n_samples_from_target = 2
    beamwidth = 1
    encoder = Encoder(target,
                      initial_seed,
                      coding_sampler,
                      selection_sampler,
                      auxiliary_posterior,
                      omega,
                      n_samples_from_target,
                      epsilon=0.0,
                      beamwidth=beamwidth)

    z, indices = encoder.run_encoder()

    best_sample = torch.argmax(target.log_prob(z))
    plot_2d_distribution(target)
    plot_running_sum(encoder.selected_samples[0], plot_index_labels=False)
    plt.show()
