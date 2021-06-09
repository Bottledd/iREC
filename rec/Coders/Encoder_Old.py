import math

import matplotlib.pyplot as plt
import torch
import torch.distributions as dist
from tqdm import tqdm

from models.BayesianLinRegressor import BayesLinRegressor
from rec.distributions.CodingSampler import CodingSampler
from rec.distributions.EmpiricalMixturePosterior_Old import EmpiricalMixturePosterior
from rec.samplers.GreedySampling import GreedySampler
from rec.samplers.ImportanceSampling import ImportanceSampler
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
                 epsilon=0.,
                 ):
        # deduce dimension of problem from target mean
        self.problem_dimension = target.mean.shape[0]

        # to decide how many auxiliary vars need to compute KL(q||p)
        # first try with torch distributions

        # create dummy coding object to compute kl with target
        coding_z_prior = coding_sampler(problem_dimension=self.problem_dimension, n_auxiliary=1)
        try:
            kl_q_p = dist.kl_divergence(target, coding_z_prior)
            print(f"{kl_q_p}")
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
            # seed = (i * 10e4) + self.selected_samples_indices[i-1]
            seed = i
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
                pbar.set_description(f"KL of aux {i+1} is {kl_estimate}")
                # print(f"KL of aux {i+1} is {kl_estimate}")
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
    #torch.set_default_tensor_type(torch.DoubleTensor)
    initial_seed = 100
    blr = BayesLinRegressor(prior_mean=torch.tensor([0.0, 0.0]),
                            prior_alpha=0.0001,
                            signal_std=10,
                            num_targets=5,
                            seed=initial_seed)
    blr.sample_feature_inputs()
    blr.sample_regression_targets()
    blr.posterior_update()
    target = blr.weight_posterior

    coding_sampler = CodingSampler
    auxiliary_posterior = EmpiricalMixturePosterior
    selection_sampler = GreedySampler
    omega = 8
    n_samples_from_target = 1000

    encoder = Encoder(target,
                      initial_seed,
                      coding_sampler,
                      selection_sampler,
                      auxiliary_posterior,
                      omega,
                      n_samples_from_target,
                      epsilon=0.,
                      )
    # encoder.auxiliary_posterior.coding_sampler.auxiliary_vars = torch.tensor([0.0136, 0.0136, 0.0136, 0.0136, 0.0136, 0.0136, 0.0136, 0.0136, 0.0136,
    #         0.0136, 0.0136, 0.0136, 0.0136, 0.0136, 0.0136, 0.0136, 0.0137, 0.0137,
    #         0.0137, 0.0137, 0.0137, 0.0137, 0.0137, 0.0138, 0.0138, 0.0138, 0.0139,
    #         0.0139, 0.0139, 0.0139, 0.0140, 0.0140, 0.0140, 0.0141, 0.0141, 0.0141,
    #         0.0142, 0.0142, 0.0142, 0.0143, 0.0143, 0.0143, 0.0144, 0.0144, 0.0145,
    #         0.0145, 0.0146, 0.0147, 0.0148, 0.0149, 0.0150, 0.0151, 0.0153, 0.0155,
    #         0.0157, 0.0159, 0.0162, 0.0165, 0.0169, 0.0172, 0.0174, 0.0175, 0.0173,
    #         0.0165, 0.0150, 0.0125, 0.0091, 0.0063, 0.0043, 0.0031, 0.0024, 0.0074,
    #         0.0074, 0.0074])

    z, indices = encoder.run_encoder()
    print(target.log_prob(z))
    print(sum(encoder.aux_var_kl))
    plot_2d_distribution(target)
    plot_running_sum(encoder.selected_samples, plot_index_labels=False)
    plt.plot(encoder.auxiliary_posterior.empirical_samples[:, 0], encoder.auxiliary_posterior.empirical_samples[:, 1],
             'x')
    plt.show()
