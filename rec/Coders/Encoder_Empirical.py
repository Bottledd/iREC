import math
import matplotlib.pyplot as plt
import torch
import torch.distributions as dist
from tqdm import tqdm
from models.SimpleBayesianLinRegressor import BayesLinRegressor
from rec.distributions.CodingSampler import CodingSampler
from rec.distributions.EmpiricalMixturePosterior import EmpiricalMixturePosterior
from rec.samplers.GreedySampling import GreedySampler
from rec.samplers.ImportanceSampling import ImportanceSampler
from rec.utils import kl_estimate_with_mc, plot_running_sum_1d, plot_running_sum_2d, plot_2d_distribution, plot_1d_distribution, plot_samples_in_2d
from rec.OptimisingVars.FinalJointOptimiser import FinalJointOptimiser


class Encoder:
    def __init__(self,
                 target,
                 initial_seed,
                 coding_sampler,
                 selection_sampler,
                 auxiliary_posterior,
                 omega,
                 n_samples_from_target,
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
            kl_q_p = kl_estimate_with_mc(target, coding_z_prior, dim=0, num_samples=10000)

        self.total_kl = kl_q_p

        # compute parameters for auxiliary method
        self.target = target
        self.n_auxiliary = math.ceil(kl_q_p / omega)
        self.n_samples_per_aux = math.ceil(torch.exp(torch.tensor(omega * (1 + epsilon))))

        # instantiate the coding sampler and auxiliary posterior
        instance_coding_sampler = coding_sampler(problem_dimension=self.problem_dimension,
                                                 n_auxiliary=self.n_auxiliary, var=1)

        torch.manual_seed(initial_seed)
        self.empirical_samples = target.sample((n_samples_from_target,))

        self.auxiliary_posterior = auxiliary_posterior(self.empirical_samples,
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

            # update mixture_weights - since these are the joint p(a_{1:k}|z) simply add on p(a_k | a_{1:k-1}, z_d)
            self.selected_samples_mixing_weights += auxiliary_posterior_dist.component_distribution.log_prob(samples)

    def run_encoder(self):
        pbar = tqdm(range(self.n_auxiliary), position=0, leave=True)
        for i in pbar:
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

                # do kl estimate on q(a_k | a_{1:k-1})
                kl_estimate = kl_estimate_with_mc(target=auxiliary_conditional_posterior, coder=auxiliary_prior,
                                                  num_samples=1000)
                self.aux_var_kl[i] = kl_estimate
                # print(f"KL of aux {i+1} is {kl_estimate}")
                pbar.set_description(f"KL of aux {i + 1} is {kl_estimate}")
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
    torch.set_default_tensor_type(torch.DoubleTensor)
    # initial_seed_target = 10
    # blr = BayesLinRegressor(prior_mean=torch.zeros(2),
    #                         prior_alpha=1,
    #                         signal_std=1,
    #                         num_targets=10000,
    #                         seed=initial_seed_target)
    # blr.sample_feature_inputs()
    # blr.sample_regression_targets()
    # blr.posterior_update()
    # target = blr.weight_posterior

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

    coding_sampler = CodingSampler
    auxiliary_posterior = EmpiricalMixturePosterior
    selection_sampler = GreedySampler
    n_samples_from_target = 100
    omega = 5
    initial_seed = 0

    epsilon = 0.
    encoder = Encoder(target,
                      initial_seed,
                      coding_sampler,
                      selection_sampler,
                      auxiliary_posterior,
                      omega,
                      n_samples_from_target,
                      epsilon=0.,
                      )

    z_sample = target.mean
    n_auxiliaries = encoder.n_auxiliary
    kl_q_p = encoder.total_kl

    option = -1

    if option == 1:
        encoder.auxiliary_posterior.coding_sampler.auxiliary_vars = torch.tensor([0.3094, 0.2134, 0.1457, 0.1002, 0.0698, 0.0491, 0.0351, 0.0255, 0.0194,
        0.0156, 0.0085, 0.0040, 0.0025, 0.0019])

    elif option == 2:
        print(f"First Optimise Prior Variances!")
        optimising = FinalJointOptimiser(z_sample, omega, n_auxiliaries, kl_q_p, n_trajectories=1000, total_var=1)
        encoder.auxiliary_posterior.coding_sampler.auxiliary_vars = optimising.run_optimiser(epochs=5000 // n_auxiliaries)
    else:
        pass

    print(f"Run Encoder!")
    z, indices = encoder.run_encoder()
    print(f"{target.log_prob(z)}")
    # mahalobonis_dist = torch.sqrt((target.mean - z).T @ target.covariance_matrix @ (target.mean - z))
    # print(f"The MSE is: {blr.measure_performance(z, kind='MSE')}\n"
    #       f"The MAE is: {blr.measure_performance(z, kind='MAE')}\n\n"
    #       f"The MSE of MAP is: {blr.measure_performance(target.mean, kind='MSE')}\n"
    #       f"The MAE of MAP is: {blr.measure_performance(target.mean, kind='MAE')}\n"
    #       f"The Mahalobonis distance is: {mahalobonis_dist}\n"
    #       f"The MSE using true samples is: {blr.measure_true_performance(kind='MSE')}\n"
    #       f"The MAE using true samples is: {blr.measure_true_performance(kind='MAE')}\n"
    #       f"The % drop-off to MAP MSE is: {(blr.measure_performance(z, kind='MSE') - blr.measure_performance(target.mean, kind='MSE')) / blr.measure_performance(target.mean, kind='MSE') * 100}\n"
    #       f"The % drop-off to MAP MAE is: {(blr.measure_performance(z, kind='MAE') - blr.measure_performance(target.mean, kind='MAE')) / blr.measure_performance(target.mean, kind='MAE') * 100}\n"
    #       f"log q(z)/p(z) is: {target.log_prob(z) - encoder.auxiliary_posterior.coding_sampler.log_prob(z)}")
    # plot_samples_in_2d(target=target)
    # plot_samples_in_2d(empirical_samples=encoder.auxiliary_posterior.empirical_samples)
    # plot_samples_in_2d(coded_sample=z)

    # plot_2d_distribution(target)
    # plot_running_sum_2d(encoder.selected_samples, plot_index_labels=True)
    # plt.plot(encoder.auxiliary_posterior.empirical_samples[:, 0], encoder.auxiliary_posterior.empirical_samples[:, 1],
    #          'x')
    # plt.plot(z[0], z[1], 'o')
    # # plt.plot(z, torch.exp(target.log_prob(z)), 'o')
    plt.show()
