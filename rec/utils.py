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
    x_grid, y_grid = torch.meshgrid(torch.linspace(m1 - 10 * variance, m1 + 10 * variance, 1000),
                                    torch.linspace(m2 - 10 * variance, m2 + 10 * variance, 1000))
    cat_grid = torch.cat((x_grid.reshape(1000, 1000, 1), y_grid.reshape(1000, 1000, 1)), dim=-1)
    log_probs = distribution.log_prob(cat_grid)
    plt.contourf(x_grid, y_grid, torch.exp(log_probs), levels=50)
    plt.xlabel(r"$w_1$")
    plt.ylabel(r"$w_0$")
    plt.gca().set_aspect('equal', adjustable='box')

def plot_1d_distribution(distribution):
    mean = distribution.mean[0]
    variance = torch.max(distribution.variance)
    grid = torch.linspace(mean - 10 * variance, mean + 10 * variance, 1000)
    probs = torch.exp(distribution.log_prob(grid.reshape(-1,1)))
    plt.plot(grid, probs)



def plot_running_sum_2d(samples, plot_index_labels=False):
    n_auxiliary = samples.shape[0]
    running_sum = torch.cumsum(samples, dim=0)
    plt.plot(running_sum[:, 0], running_sum[:, 1], '-')

    if plot_index_labels:
        for i, txt in enumerate(range(n_auxiliary)):
            plt.annotate(txt, (running_sum[i, 0], running_sum[i, 1]), color='white')


def plot_running_sum_1d(target, samples, plot_index_labels=False):
    n_auxiliary = samples.shape[0]
    running_sum = torch.cumsum(samples, dim=0)
    plt.plot(running_sum, torch.exp(target.log_prob(running_sum)), '-')

    if plot_index_labels:
        for i, txt in enumerate(range(n_auxiliary)):
            plt.annotate(txt, (running_sum[i, 0], running_sum[i, 1]), color='white')


def compute_variational_posterior(target):
    mean = target.mean

    # minimum KL when variation approx variance is diag(rho) where rho = diag(V^-1)^-1
    V_inverse = torch.inverse(target.covariance_matrix)
    diag_V_inverse = torch.diag(V_inverse)
    variational_covariance = torch.inverse(torch.diag(diag_V_inverse))

    return torch.distributions.multivariate_normal.MultivariateNormal(loc=mean,
                                                                      covariance_matrix=variational_covariance)