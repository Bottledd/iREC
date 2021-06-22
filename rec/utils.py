import torch
import matplotlib.pyplot as plt


def kl_estimate_with_mc(target, coder, num_samples=1000, dim=0, rsample=False):
    if rsample:
        samples = target.rsample((num_samples,))
    else:
        samples = target.sample((num_samples,))

    target_log_prob = target.log_prob(samples)
    coding_log_prob = coder.log_prob(samples)

    return torch.mean(target_log_prob - coding_log_prob, dim=dim)


def plot_2d_distribution(distribution):
    m1, m2 = distribution.mean
    variance = torch.max(distribution.variance)
    x_grid, y_grid = torch.meshgrid(torch.linspace(m1 - 2*variance, m1 + 2*variance, 1000),
                                    torch.linspace(m2 - 2*variance, m2 + 2*variance, 1000))
    cat_grid = torch.cat((x_grid.reshape(1000, 1000, 1), y_grid.reshape(1000, 1000, 1)), dim=-1)
    log_probs = distribution.log_prob(cat_grid)
    plt.contourf(x_grid, y_grid, torch.exp(log_probs), levels=50)
    plt.xlabel(r"$w_1$")
    plt.ylabel(r"$w_0$")
    plt.gca().set_aspect('equal', adjustable='box')


def plot_1d_distribution(distribution):
    mean = distribution.mean[0]
    variance = torch.max(distribution.variance)
    grid = torch.linspace(mean - 10 , mean + 10 , 1000)
    probs = torch.exp(distribution.log_prob(grid.reshape(-1,1)))
    plt.plot(grid, probs)


def plot_running_sum_2d(samples, indices=(0,1), plot_index_labels=False):
    n_auxiliary = samples.shape[0]
    running_sum = torch.cumsum(samples, dim=0)
    plt.plot(running_sum[:, indices[0]], running_sum[:, indices[1]], '-')

    if plot_index_labels:
        for i, txt in enumerate(range(n_auxiliary)):
            plt.annotate(txt, (running_sum[i, indices[0]], running_sum[i, indices[1]]), color='black')


def plot_running_sum_1d(target, samples, plot_index_labels=False):
    n_auxiliary = samples.shape[0]
    running_sum = torch.cumsum(samples, dim=0)
    plt.plot(running_sum, torch.exp(target.log_prob(running_sum)), '-')

    if plot_index_labels:
        for i, txt in enumerate(range(n_auxiliary)):
            plt.annotate(txt, (running_sum[i, 0], running_sum[i, 1]), color='black')


def compute_variational_posterior(target):
    mean = target.mean

    # minimum KL when variation approx variance is diag(rho) where rho = diag(V^-1)^-1
    V_inverse = torch.inverse(target.covariance_matrix)
    diag_V_inverse = torch.diag(V_inverse)
    variational_covariance = torch.inverse(torch.diag(diag_V_inverse))

    return torch.distributions.multivariate_normal.MultivariateNormal(loc=mean,
                                                                      covariance_matrix=variational_covariance)


def plot_samples_in_2d(target=None, dimensions=(0, 1), num_samples=10000, coded_sample=None, empirical_samples=None):
    torch.manual_seed(0)
    if coded_sample is not None:
        plt.plot(coded_sample[dimensions[0]], coded_sample[dimensions[1]], 'ko')
    elif empirical_samples is not None:
        plt.plot(empirical_samples[:, dimensions[0]], empirical_samples[:, dimensions[1]], 'xr')
    else:
        # sample from target
        samples = target.sample((num_samples,))
        plt.plot(samples[:, dimensions[0]], samples[:, dimensions[1]], 'x')


def plot_pairs_of_samples(target, coded_sample, num_samples=10000, empirical_samples=None):
    dim = target.mean.shape[0]
    samples = target.sample((num_samples,))
    assert dim % 2 == 0, "Need dim divisible by 2 to plot nicely!"

    columns = 1
    rows = dim // 2
    running_sum = torch.cumsum(coded_sample, dim=0)
    final_sample = running_sum[-1]
    fig, ax = plt.subplots(columns, rows, figsize=(50, 5))

    for i in range(rows):
        ax[i].plot(samples[:, 2 * i], samples[:, 2 * i + 1], 'x')
        if empirical_samples is not None:
            ax[i].plot(empirical_samples[:, 2 * i], empirical_samples[:, 2 * i + 1], 'x')
        ax[i].plot(running_sum[:, 2 * i], running_sum[:, 2 * i + 1], '-')
        ax[i].plot(final_sample[2 * i], final_sample[2 * i + 1], 'ko')
        ax[i].set_aspect('equal', adjustable='box')

    fig.tight_layout()
