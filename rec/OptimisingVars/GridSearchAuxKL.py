import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from models.SimpleBayesianLinRegressor import BayesLinRegressor
from rec.Coders.Encoder_Old import Encoder
from rec.distributions.CodingSampler import CodingSampler
from rec.distributions.EmpiricalMixturePosterior_Old import EmpiricalMixturePosterior
from rec.samplers.GreedySampling import GreedySampler
from rec.utils import plot_running_sum, plot_2d_distribution


def compute_new_sigmas(total_var, sigmas, k):
    """
    From previously optimised sigmas, compute remaining sigmas uniformly to sum to variance on z
    :param total_var: Total var on z
    :param sigmas: Tensor storing var for p(a_k)
    :param k: index of current aux var
    :return: Tensor with aux vars that sum to correct variance
    """
    current_sum = torch.sum(sigmas[:k + 1])
    remaining_var = total_var - current_sum
    remaining_aux_vars = remaining_var / sigmas[k + 1:].shape[0]
    sigmas[k + 1:] = remaining_aux_vars
    return sigmas


if __name__ == '__main__':
    initial_seed = 10
    blr = BayesLinRegressor(prior_mean=torch.tensor([0.0, 0.0]),
                            prior_alpha=0.01,
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
    omega = 5
    n_samples_from_target = 1

    encoder = Encoder(target,
                      initial_seed,
                      coding_sampler,
                      selection_sampler,
                      auxiliary_posterior,
                      omega,
                      n_samples_from_target,
                      epsilon=0.0,
                      )
    n_aux = encoder.n_auxiliary
    # gridsearch on aux_var_1
    num_points = 50
    optimised_vars = torch.zeros((n_aux - 1,))
    optimised_kls = torch.zeros((n_aux - 1,))
    optimised_log_probs = torch.zeros((1,))
    for i in tqdm(range(n_aux - 1)):
        previous_sum_of_vars = torch.sum(optimised_vars[:i])
        kls_for_aux = torch.zeros((num_points,))
        # grid = torch.linspace(0.0001, 0.99999 - previous_sum_of_vars, num_points)
        grid = torch.linspace(0.005, 0.05, num_points)

        for j, var in enumerate(grid):
            encoder = Encoder(target,
                              initial_seed,
                              coding_sampler,
                              selection_sampler,
                              auxiliary_posterior,
                              omega,
                              n_samples_from_target,
                              epsilon=0.0,
                              )
            encoder.auxiliary_posterior.coding_sampler.auxiliary_vars[:i] = optimised_vars[:i]
            encoder.auxiliary_posterior.coding_sampler.auxiliary_vars[i] = var
            encoder.auxiliary_posterior.coding_sampler.auxiliary_vars = compute_new_sigmas(total_var=1,
                                                                                           sigmas=encoder.auxiliary_posterior.coding_sampler.auxiliary_vars,
                                                                                           k=i)
            z, indices = encoder.run_encoder()
            kls_for_aux[j] = encoder.aux_var_kl[i]

        MSE = torch.argmin((kls_for_aux - omega-0.2) ** 2)
        print(f"MSE of kl to omega is: {((kls_for_aux - omega) ** 2)[MSE]}")
        print(f"KL of aux {i+1} is: {kls_for_aux[MSE]}")
        aux_var = grid[MSE]
        optimised_vars[i] = aux_var
        optimised_kls[i] = ((kls_for_aux - omega) ** 2)[MSE]

    new_sigmas = compute_new_sigmas(1, torch.hstack((optimised_vars, torch.tensor([0]))), n_aux - 2)
    encoder.auxiliary_posterior.coding_sampler.auxiliary_vars = new_sigmas
    z, indices = encoder.run_encoder()
    print(target.log_prob(z))
    plot_2d_distribution(target)
    plot_running_sum(encoder.selected_samples, plot_index_labels=False)
    plt.plot(encoder.auxiliary_posterior.empirical_samples[:, 0], encoder.auxiliary_posterior.empirical_samples[:, 1],
             'x')
    plt.show()

