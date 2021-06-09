import math

import matplotlib.pyplot as plt
import torch
import torch.distributions as dist
from tqdm import tqdm

from models.BayesianLinRegressor import BayesLinRegressor
from rec.distributions.CodingSampler import CodingSampler
from rec.distributions.VariationalPosterior import VariationalPosterior
from rec.samplers.GreedySampling import GreedySampler
from rec.samplers.ImportanceSampling import ImportanceSampler
from rec.utils import kl_estimate_with_mc, compute_variational_posterior, plot_samples_in_2d, plot_running_sum_2d


class Encoder:
    def __init__(self,
                 target,
                 initial_seed,
                 coding_sampler,
                 selection_sampler,
                 auxiliary_posterior,
                 omega,
                 epsilon=0.,
                 ):
        # deduce dimension of problem from target mean
        self.problem_dimension = target.mean.shape[0]

        # to decide how many auxiliary vars need to compute KL(q||p)
        # first try with torch distributions

        # create dummy coding object to compute kl with target
        coding_z_prior = coding_sampler(problem_dimension=self.problem_dimension, n_auxiliary=1, var=1)
        try:
            kl_q_p = dist.kl_divergence(target, coding_z_prior)
            print(f"{kl_q_p}")
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
                                                       instance_coding_sampler)

        # auxiliary samples that are selected
        # keep track of samples themselves and indices separately so we can check to ensure they agree
        self.selected_samples = torch.zeros((self.n_auxiliary, self.problem_dimension))
        self.selected_samples_indices = torch.zeros((self.n_auxiliary,))

        # need to keep track of joint probabilities
        self.selected_samples_joint_coding_log_prob = torch.zeros((1,))
        self.selected_samples_joint_posterior_log_prob = torch.zeros((1,))

        # store selection sampler object
        self.selection_sampler = selection_sampler

        # set the initial seed
        self.initial_seed = initial_seed

        # store kls between aux variables
        self.aux_var_kl = torch.zeros((self.n_auxiliary,))

    def update_stored_samples(self, index, indices, samples, coding_dist=None, auxiliary_posterior_dist=None,
                              final_sample=False):
        # update selected samples
        self.selected_samples[index] = samples
        self.selected_samples_indices[index] = indices

        if not final_sample:
            coding_log_probs = coding_dist.log_prob(samples)
            auxiliary_posterior_log_prob = auxiliary_posterior_dist.log_prob(samples)

            # add new log probs to compute new joint
            self.selected_samples_joint_coding_log_prob += coding_log_probs
            self.selected_samples_joint_posterior_log_prob += auxiliary_posterior_log_prob

    def run_encoder(self):
        pbar = tqdm(range(self.n_auxiliary), position=0, leave=True)
        for i in pbar:
            # set the seed
            # seed = (i * 10e4) + self.selected_samples_indices[i-1]
            seed = i + self.initial_seed
            # create new auxiliary prior distribution, p(a_k)
            auxiliary_prior = self.auxiliary_posterior.coding_sampler.auxiliary_coding_dist(i)

            if i < self.n_auxiliary - 1:
                # compute conditional posterior distribution, q(a_k | a_{1:k-1})
                auxiliary_conditional_posterior = self.auxiliary_posterior.q_ak_given_history(self.selected_samples,
                                                                                              i)

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

                # do kl estimate on q(a_k | a_{1:k-1})
                kl_estimate = kl_estimate_with_mc(target=auxiliary_conditional_posterior, coder=auxiliary_prior,
                                                  num_samples=1000)
                self.aux_var_kl[i] = kl_estimate
                # print(f"KL of aux {i+1} is {kl_estimate}")
                pbar.set_description(f"KL of aux {i + 1} is {kl_estimate}")
                # update stored samples, indices and log probs
                self.update_stored_samples(i, indices, samples, auxiliary_prior, auxiliary_conditional_posterior)

                self.auxiliary_posterior.q_z_given_trajectory(self.selected_samples, i)

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

    encoder = Encoder(var_target,
                      initial_seed,
                      coding_sampler,
                      selection_sampler,
                      auxiliary_posterior,
                      omega,
                      epsilon=0.,
                      )
    # encoder.auxiliary_posterior.coding_sampler.auxiliary_vars = torch.tensor([0.3094, 0.2134, 0.1457, 0.1002, 0.0698, 0.0491, 0.0351, 0.0255, 0.0194,
    #     0.0156, 0.0085, 0.0040, 0.0025, 0.0019])

    # encoder.auxiliary_posterior.coding_sampler.auxiliary_vars = torch.tensor([0.3189, 0.2091, 0.1406, 0.0963, 0.0668, 0.0465, 0.0323, 0.0228, 0.0174,
    #         0.0136, 0.0102, 0.0074, 0.0053, 0.0037, 0.0025, 0.0017, 0.0012, 0.0009,
    #         0.0007, 0.0006, 0.0005, 0.0010])
    # encoder.auxiliary_posterior.coding_sampler.auxiliary_vars = torch.tensor(
    #     [0.1561, 0.1479, 0.1372, 0.1241, 0.1091, 0.0932, 0.0784, 0.0660, 0.0438,
    #      0.0192, 0.0086, 0.0045, 0.0026, 0.0015, 0.0011, 0.0011, 0.0010, 0.0010,
    #      0.0009, 0.0009, 0.0008, 0.0010])

    z, indices = encoder.run_encoder()
    plot_samples_in_2d(target=true_target)
    plot_samples_in_2d(coded_sample=z)
    mahalobonis_dist = torch.sqrt((true_target.mean - z).T @ true_target.covariance_matrix @ (true_target.mean - z))
    print(f"The standardised / normal MSE is: {blr.measure_performance(z, type='MSE')}\n"
          f"The MAE is: {blr.measure_performance(z, type='MAE')}\n"
          f"The Mahalobonis distance is: {mahalobonis_dist}\n"
          f"The MSE of the mean is: {blr.measure_performance(true_target.mean, type='MSE')}\n"
          f"The MAE of the mean is: {blr.measure_performance(true_target.mean, type='MAE')}")
    # print(sum(encoder.aux_var_kl))
    # plot_2d_distribution(target)
    plot_running_sum_2d(encoder.selected_samples, plot_index_labels=True)
    # plt.plot(z[0], z[1], 'o')
    plt.show()
