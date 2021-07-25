import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as dist
from tqdm import tqdm
from models.SimpleBayesianLinRegressor import BayesLinRegressor
from rec.beamsearch.distributions.CodingSampler import CodingSampler
from rec.beamsearch.distributions.KDEPosterior import KDEPosterior
from rec.beamsearch.samplers.GreedySampling import GreedySampler
from rec.beamsearch.utils import convert_flattened_indices
from rec.utils import kl_estimate_with_mc, compute_variational_posterior, plot_samples_in_2d, plot_running_sum_2d, plot_pairs_of_samples


class EncoderKDE:
    def __init__(self,
                 target,
                 initial_seed,
                 coding_sampler,
                 selection_sampler,
                 auxiliary_posterior,
                 omega,
                 beamwidth,
                 epsilon=0.,
                 prior_var=1.,
                 total_kl=None
                 ):
        # instantiate the beamwidth
        self.beamwidth = beamwidth

        # deduce dimension of problem from target mean
        self.problem_dimension = target.mean.shape[0]

        # to decide how many auxiliary vars need to compute KL(q||p)
        # first try with torch distributions

        # create dummy coding object to compute kl with target
        coding_z_prior = coding_sampler(problem_dimension=self.problem_dimension, n_auxiliary=1, var=prior_var)
        
        if total_kl is not None:
            kl_q_p = total_kl
        else:
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
                                                 var=prior_var)

        self.auxiliary_posterior = auxiliary_posterior(target,
                                                       instance_coding_sampler,
                                                       beamwidth=self.beamwidth)

        # auxiliary samples that are selected
        # keep track of samples themselves and indices separately so we can check to ensure they agree

        self.selected_samples = torch.ones((self.beamwidth, self.n_auxiliary, self.problem_dimension)) * float('nan')
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
                mask = self.selected_samples[:, i - 1].ne(float('nan'))
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
                                                                                              i,
                                                                                              mask)

                # need to see if masked B_hat * M > B
                if self.beamwidth > tiled_beams.shape[0]:
                    n_new_beams = tiled_beams.shape[0]
                else:
                    n_new_beams = self.beamwidth

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
                    i,
                    beam_indices=beam_indices)
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
                                                           topk=self.beamwidth,
                                                           z_prior=self.auxiliary_posterior.coding_sampler)
                trial_aks = selection_sampler.get_samples_from_coder()
                repeated_aks = torch.repeat_interleave(trial_aks, self.beamwidth, dim=0)
                indices, samples = selection_sampler.choose_samples_to_transmit(samples=repeated_aks,
                                                                                previous_samples=self.selected_samples[:, :i],)

                # now need to convert indices from flattened to refer to specific beam/ak sample
                beam_indices, sample_indices = convert_flattened_indices(indices, beamwidth=self.beamwidth)

                # update stored samples, indices and log probs
                self.update_stored_samples(i, beam_indices=beam_indices, sample_indices=sample_indices, samples=samples,
                                           final_sample=True)

        return torch.sum(self.selected_samples, dim=1), self.selected_samples_indices


if __name__ == '__main__':
    torch.set_default_tensor_type(torch.DoubleTensor)
    # blr = BayesLinRegressor(prior_mean=torch.zeros(50),
    #                         prior_alpha=1,
    #                         signal_std=1,
    #                         num_targets=100,
    #                         seed=1)
    # blr.sample_feature_inputs()
    # blr.sample_regression_targets()
    # blr.posterior_update()
    # target = blr.weight_posterior
    # target = compute_variational_posterior(target)
    # plt.imshow(target.covariance_matrix)
    # plt.show()
    #
    var_mean = torch.tensor([-1.2828, 1.6508, 1.4314, -0.7785, -0.1488, 0.2930, -0.1225, 2.6420,
                             0.4913, -0.6382, -0.5173, -1.5689, -1.2808, 1.4096, 1.3054, -0.5755,
                             -0.1463, 0.1514, 0.1904, 0.4206, -0.4569, 0.5137, 0.4990, -0.4522,
                             1.6059, -0.0308, 0.7416, 0.1244, 0.4371, 1.4866, -0.0216, 0.0246,
                             1.6946])
    var_std = torch.tensor([0.3433, 0.1766, 0.3178, 0.0081, 0.6837, 0.4910, 0.5820, 0.0304, 0.0328,
                            0.0484, 0.0475, 0.0928, 0.3547, 0.3492, 0.3406, 0.5405, 1.2291, 0.9722,
                            0.5047, 0.5389, 0.6762, 0.6502, 0.8707, 0.8854, 0.0551, 0.3796, 0.7789,
                            0.8914, 0.0145, 0.0085, 0.0081, 0.0118, 0.0069])

    target = dist.MultivariateNormal(loc=var_mean, covariance_matrix=torch.diag(var_std ** 2))
    import pickle as pkl
    # emp_samples = pkl.load(open("../../../PickledStuff/emp_samples.pkl", "rb"))
    emp_samples = target.sample((10,))
    KDE_var = 0.002 ** 2
    #n_samples = 100
    KDE_weights = dist.Categorical(torch.ones(emp_samples.shape[0]))
    initial_seed = 0
    torch.manual_seed(initial_seed)
    KDE_components = dist.MultivariateNormal(loc=emp_samples,
                                             covariance_matrix=KDE_var * torch.eye(emp_samples.shape[-1]))

    KDE_target = dist.MixtureSameFamily(KDE_weights, KDE_components)

    coding_sampler = CodingSampler
    auxiliary_posterior = KDEPosterior
    selection_sampler = GreedySampler
    omega = 5

    beamwidth = 1
    epsilon = 0.
    encoder = EncoderKDE(KDE_target,
                          initial_seed,
                          coding_sampler,
                          selection_sampler,
                          auxiliary_posterior,
                          omega,
                          epsilon=epsilon,
                          beamwidth=beamwidth)

    n_auxiliaries = encoder.n_auxiliary
    kl_q_p = encoder.total_kl

    option = -1

    if option == 1:
        encoder.auxiliary_posterior.coding_sampler.auxiliary_vars = torch.tensor(
            [0.3094, 0.2134, 0.1457, 0.1002, 0.0698, 0.0491, 0.0351, 0.0255, 0.0194,
             0.0156, 0.0085, 0.0040, 0.0025, 0.0019])

    elif option == 2:
        print(f"First Optimise Prior Variances!")
        optimising = VariationalOptimiser(var_target, omega, n_auxiliaries, kl_q_p, n_trajectories=50, total_var=1)
        encoder.auxiliary_posterior.coding_sampler.auxiliary_vars = optimising.run_optimiser(
            epochs=5000 // n_auxiliaries)
    else:
        pass

    z, indices = encoder.run_encoder()
    best_sample_idx = torch.argmax(KDE_target.log_prob(z))
    best_sample = z[best_sample_idx]
    #plot_pairs_of_samples(target, encoder.selected_samples[best_sample_idx])
    #plt.show()
    print(KDE_target.log_prob(best_sample))
    print(indices[0])
    # mahalanobis_dist = torch.sqrt(
    #     (true_target.mean - best_sample).T @ true_target.covariance_matrix @ (true_target.mean - best_sample))
    #
    # import sys
    #
    # parent_root = "../../../"
    # sys.stdout = open(parent_root + f"Logs/variational_beam{beamwidth}_epsilon{epsilon}", 'w')
    #
    # print(f"The MSE is: {blr.measure_performance(best_sample, kind='MSE')}\n"
    #       f"The MAE is: {blr.measure_performance(best_sample, kind='MAE')}\n"
    #       f"The Mahalanobis distance is: {mahalanobis_dist}\n"
    #       f"The MSE of the mean is: {blr.measure_performance(true_target.mean, kind='MSE')}\n"
    #       f"The MAE of the mean is: {blr.measure_performance(true_target.mean, kind='MAE')}\n"
    #       f"The % drop-off to MAP MSE is: {(blr.measure_performance(best_sample, kind='MSE') - blr.measure_performance(true_target.mean, kind='MSE')) / blr.measure_performance(true_target.mean, kind='MSE') * 100}\n"
    #       f"The % drop-off to MAP MAE is: {(blr.measure_performance(best_sample, kind='MAE') - blr.measure_performance(true_target.mean, kind='MAE')) / blr.measure_performance(true_target.mean, kind='MAE') * 100}\n"
    #       f"log q(z)/p(z) is: {true_target.log_prob(best_sample) - encoder.auxiliary_posterior.coding_sampler.log_prob(best_sample)}")
    # # plot_samples_in_2d(target=true_target)
    # # plot_samples_in_2d(coded_sample=best_sample)
    # # plot_2d_distribution(target)
    # # plot_running_sum_2d(encoder.selected_samples[best_sample_idx], plot_index_labels=True)
    # # plt.plot(encoder.auxiliary_posterior.empirical_samples[:, 0], encoder.auxiliary_posterior.empirical_samples[:, 1],
    # #          'x')
    # # plt.plot(best_sample[0], best_sample[1], 'o')
    # plt.savefig(f"../../../Figures/variational_beam{beamwidth}_epsilon{epsilon}.png", bbox_inches='tight')
    # plt.show()
    # sys.stdout.close()