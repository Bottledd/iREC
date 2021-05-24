import torch
import matplotlib.pyplot as plt


def kl_estimate_with_mc(target, coder, num_samples=1000, dim=0):
    samples = target.sample((num_samples,))

    target_log_prob = target.log_prob(samples)
    coding_log_prob = coder.log_prob(samples)

    return torch.mean(target_log_prob - coding_log_prob, dim=dim)


def plot_2d_distribution(distribution):
    m1, m2 = distribution.mean
    x_grid, y_grid = torch.meshgrid(torch.linspace(m1 - 10, m1 + 10, 1000),
                                    torch.linspace(m2 - 10, m2 + 10, 1000))
    cat_grid = torch.cat((x_grid.reshape(1000, 1000, 1), y_grid.reshape(1000, 1000, 1)), dim=-1)
    log_probs = distribution.log_prob(cat_grid)
    plt.contourf(x_grid, y_grid, torch.exp(log_probs), levels=50)
    plt.xlabel(r"$w_1$")
    plt.ylabel(r"$w_0$")
    plt.gca().set_aspect('equal', adjustable='box')


def plot_running_sum(samples, plot_index_labels=False):
    n_auxiliary = samples.shape[0]
    running_sum = torch.cumsum(samples, dim=0)
    plt.plot(running_sum[:, 0], running_sum[:, 1], '-')

    if plot_index_labels:
        for i, txt in enumerate(range(n_auxiliary)):
            plt.annotate(txt, (running_sum[i, 0], running_sum[i, 1]), color='white')