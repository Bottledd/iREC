import math

import matplotlib.pyplot as plt
import torch
import torch.distributions as dist
from tqdm import tqdm

from models.BayesianLinRegressor import BayesLinRegressor
from rec.beamsearch.distributions.CodingSampler import CodingSampler
from rec.beamsearch.distributions.VariationalPosterior import VariationalPosterior
from rec.beamsearch.samplers.GreedySampling import GreedySampler
from rec.beamsearch.utils import convert_flattened_indices
from rec.utils import kl_estimate_with_mc, compute_variational_posterior, plot_samples_in_2d, plot_running_sum_2d
from rec.OptimisingVars.FinalJointOptimiser import FinalJointOptimiser

class Encoder:
    def __init__(self,
                 target,
                 initial_seed,
                 coding_sampler,
                 selection_sampler,
                 auxiliary_posterior,
                 omega,
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
        coding_z_prior = coding_sampler(problem_dimension=self.problem_dimension, n_auxiliary=1, var=1)

        try:
            kl_q_p = dist.kl_divergence(target, coding_z_prior)
        except:
            # need to do MC estimate
            kl_q_p = kl_estimate_with_mc(target, coding_z_prior)

        self.total_kl = kl_q_p

        # compute parameters for auxiliary method
        self.target = target
        self.n_auxiliary = math.ceil(kl_q_p / omega)
        self.n_samples_per_aux = math.ceil(torch.exp(torch.tensor(omega * (1 + epsilon))))

        # instantiate the coding sampler and auxiliary posterior
        instance_coding_sampler = coding_sampler(problem_dimension=self.problem_dimension,
                                                 n_auxiliary=self.n_auxiliary,
                                                 var=1)

        self.auxiliary_posterior = auxiliary_posterior(target,
                                                       instance_coding_sampler,
                                                       beamwidth=self.beamwidth)

        # auxiliary samples that are selected
        # keep track of samples themselves and indices separately so we can check to ensure they agree
        self.selected_samples = torch.zeros((self.beamwidth, self.n_auxiliary, self.problem_dimension))
        self.selected_samples_indices = torch.zeros((self.beamwidth, self.n_auxiliary,))

        # need to keep track of joint probabilities
        self.selected_samples_joint_coding_log_prob = torch.zeros((self.beamwidth,))
        self.selected_samples_joint_posterior_log_prob = torch.zeros((self.beamwidth,))

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

    def run_encoder(self):
        for i in tqdm(range(self.n_auxiliary)):
            # set the seed
            seed = i + self.initial_seed
            # create new auxiliary prior distribution, p(a_k)
            auxiliary_prior = self.auxiliary_posterior.coding_sampler.auxiliary_coding_dist(i)

            # need to do something different for first auxiliary variable
            if i == 0:
                # if beamwidth > n_samples_per_aux select at most n_samples_per_aux on first round
                if self.beamwidth > self.n_samples_per_aux:
                    n_selections = self.n_samples_per_aux
                else:
                    n_selections = self.beamwidth

                # compute conditional posterior distribution, q(a_k | a_{1:k-1})
                # note since all starting values are zero we can select the 0th index for many
                auxiliary_conditional_posterior = self.auxiliary_posterior.q_ak_given_history(self.selected_samples[0],
                                                                                              i)

                # create new selection sampler object
                selection_sampler = self.selection_sampler(coding=auxiliary_prior,
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

                self.auxiliary_posterior.q_z_given_trajectory(self.selected_samples, i)

            elif self.n_auxiliary - 1 > i > 0:
                # first need to ignore any unfilled beams from previous run, i.e. mask 0 values
                mask = self.selected_samples[:, i - 1].ne(0)
                expanded_mask = torch.repeat_interleave(mask[:, None, :], self.n_auxiliary, dim=1)
                single_dim_mask = mask[:, 0]
                pruned_beams = torch.masked_select(self.selected_samples, expanded_mask).reshape(-1, self.n_auxiliary,
                                                                                                 self.problem_dimension)
                pruned_coding_log_prob = torch.masked_select(self.selected_samples_joint_coding_log_prob,
                                                             single_dim_mask)
                pruned_posterior_log_prob = torch.masked_select(self.selected_samples_joint_posterior_log_prob,
                                                                single_dim_mask)
                # need to expand the beams to have B*M elements
                tiled_beams = torch.tile(pruned_beams, (self.n_samples_per_aux, 1, 1))
                # compute conditional posterior distribution, q(a_k | a_{1:k-1})
                auxiliary_conditional_posterior = self.auxiliary_posterior.q_ak_given_history(tiled_beams,
                                                                                              i)
                # need to see if masked B_hat * M > B
                if self.beamwidth > tiled_beams.shape[0]:
                    n_new_beams = tiled_beams.shape[0]
                else:
                    n_new_beams = beamwidth
                # create new selection sampler object
                selection_sampler = self.selection_sampler(coding=auxiliary_prior,
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
                    i)
                # update stored samples, indices and log probs

                self.update_stored_samples(i, beam_indices=beam_indices, sample_indices=sample_indices, samples=samples,
                                           coding_dist=auxiliary_prior,
                                           auxiliary_posterior_dist=auxiliary_conditional_posterior_untiled)

                self.auxiliary_posterior.q_z_given_trajectory(self.selected_samples, i)
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
                                                                                previous_samples=self.selected_samples, )

                # now need to convert indices from flattened to refer to specific beam/ak sample
                beam_indices, sample_indices = convert_flattened_indices(indices, beamwidth=self.beamwidth)

                # update stored samples, indices and log probs
                self.update_stored_samples(i, beam_indices=beam_indices, sample_indices=sample_indices, samples=samples,
                                           final_sample=True)

        return torch.sum(self.selected_samples, dim=1), self.selected_samples_indices


if __name__ == '__main__':
    torch.set_default_tensor_type(torch.DoubleTensor)
    initial_seed_target = 0
    blr = BayesLinRegressor(prior_mean=torch.zeros(10),
                            prior_alpha=1,
                            signal_std=1,
                            num_targets=100,
                            seed=initial_seed_target)
    blr.sample_feature_inputs()
    blr.sample_regression_targets()
    blr.posterior_update()
    true_target = blr.weight_posterior

    coding_sampler = CodingSampler
    auxiliary_posterior = VariationalPosterior
    selection_sampler = GreedySampler
    omega = 8

    var_target = compute_variational_posterior(true_target)
    initial_seed = 0

    beamwidth = 2

    encoder = Encoder(var_target,
                      initial_seed,
                      coding_sampler,
                      selection_sampler,
                      auxiliary_posterior,
                      omega,
                      epsilon=0.,
                      beamwidth=beamwidth)

    n_auxiliaries = encoder.n_auxiliary
    kl_q_p = encoder.total_kl

    option = 3

    if option == 1:
        encoder.auxiliary_posterior.coding_sampler.auxiliary_vars = torch.tensor(
            [0.3094, 0.2134, 0.1457, 0.1002, 0.0698, 0.0491, 0.0351, 0.0255, 0.0194,
             0.0156, 0.0085, 0.0040, 0.0025, 0.0019])

    elif option == 2:
        print(f"First Optimise Prior Variances!")
        optimising = FinalJointOptimiser(z_sample, omega, n_auxiliaries, kl_q_p, n_trajectories=1000, total_var=1)
        encoder.auxiliary_posterior.coding_sampler.auxiliary_vars = optimising.run_optimiser(
            epochs=5000 // n_auxiliaries)
    else:
        pass

    z, indices = encoder.run_encoder()
    best_sample_idx = torch.argmax(true_target.log_prob(z))
    best_sample = z[best_sample_idx]
    mahalobonis_dist = torch.sqrt((true_target.mean - best_sample).T@true_target.covariance_matrix @(true_target.mean - best_sample))
    print(f"The MSE is: {blr.measure_performance(best_sample, type='MSE')}\n"
          f"The MAE is: {blr.measure_performance(best_sample, type='MAE')}\n"
          f"The Mahalobonis distance is: {mahalobonis_dist}\n"
          f"The MSE of the mean is: {blr.measure_performance(true_target.mean, type='MSE')}\n"
          f"The MAE of the mean is: {blr.measure_performance(true_target.mean, type='MAE')}\n"
          f"log q(z)/p(z) is: {true_target.log_prob(best_sample) - encoder.auxiliary_posterior.coding_sampler.log_prob(best_sample)}")
    plot_samples_in_2d(target=true_target)
    plot_samples_in_2d(coded_sample=best_sample)
    # plot_2d_distribution(target)
    plot_running_sum_2d(encoder.selected_samples[best_sample_idx], plot_index_labels=True)
    # plt.plot(encoder.auxiliary_posterior.empirical_samples[:, 0], encoder.auxiliary_posterior.empirical_samples[:, 1],
    #          'x')
    # plt.plot(best_sample[0], best_sample[1], 'o')
    plt.show()