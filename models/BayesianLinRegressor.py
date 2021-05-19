import torch
import torch.distributions as dst
import matplotlib.pyplot as plt


class BayesLinRegressor:
    def __init__(self,
                 prior_mean: torch.tensor,
                 prior_alpha: torch.float,
                 signal_std: torch.float,
                 num_targets: torch.float,
                 seed: torch.float
                 ):
        torch.manual_seed(seed)
        self.prior_mean = prior_mean
        self.dim = prior_mean.shape[0] - 1  # since we include a bias in the weights
        self.signal_std = signal_std
        self.feature_targets = None
        self.regression_targets = None
        self.weight_prior = dst.MultivariateNormal(loc=prior_mean,
                                                   precision_matrix=prior_alpha * torch.eye(prior_mean.shape[0]))
        self.weight_posterior = None
        self.num_targets = num_targets

    def sample_feature_inputs(self):
        standard_normal = dst.Normal(loc=0.0, scale=1.0)
        standard_normal = dst.Uniform(-5, 5)
        self.feature_targets = standard_normal.sample((self.num_targets, self.dim))

    def sample_regression_targets(self):
        # sample weights
        weight_sample = self.weight_prior.sample()
        # broadcast to correct shapes
        weight_matrix = torch.tile(weight_sample, (self.feature_targets.shape[0], 1))

        # make feature matrix
        feature_matrix = torch.hstack((self.feature_targets, torch.ones(self.feature_targets.shape[0], 1)))
        # multiply weights to features
        feature_times_weights = torch.sum(weight_matrix * feature_matrix, dim=1)

        # add signal noise
        epsilon = dst.Normal(loc=0.0, scale=self.signal_std)
        additive_noise = epsilon.sample((feature_times_weights.shape))
        self.regression_targets = feature_times_weights + additive_noise

    def posterior_update(self):
        # make feature matrix
        feature_matrix = torch.hstack((self.feature_targets, torch.ones(self.feature_targets.shape[0], 1)))

        # compute posterior covariance
        post_precision = (1 / self.signal_std ** 2) * (
                feature_matrix.T @ feature_matrix) + self.weight_prior.precision_matrix

        post_mean = (1 / self.signal_std ** 2) * (
                torch.inverse(post_precision) @ feature_matrix.T @ self.regression_targets)

        self.weight_posterior = dst.MultivariateNormal(loc=post_mean, precision_matrix=post_precision)


    def plot_regression(self, num_lines=10):
        # plot 1d example
        plt.plot(self.feature_targets, self.regression_targets, 'o', label="observations")

        # plot samples from posterior
        x_axis = torch.linspace(-10, 10, 1000)
        x_matrix = torch.hstack((x_axis.reshape(-1, 1), torch.ones(x_axis.shape[0], 1)))
        post_samples = self.weight_posterior.sample((num_lines,))
        for sample in post_samples:
            weight_matrix = torch.tile(sample, (x_axis.shape[0], 1))
            preds = torch.sum(x_matrix * weight_matrix, dim=1)
            plt.plot(x_axis, preds, '-', color='red')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def plot_sampled_regressor(self, sample):
        # plot 1d example
        plt.plot(self.feature_targets, self.regression_targets, 'o', label="observations")

        # plot samples from posterior
        x_axis = torch.linspace(-10, 10, 1000)
        x_matrix = torch.hstack((x_axis.reshape(-1, 1), torch.ones(x_axis.shape[0], 1)))
        weight_matrix = torch.tile(sample, (x_axis.shape[0], 1))
        preds = torch.sum(x_matrix * weight_matrix, dim=1)
        plt.plot(x_axis, preds, '-', color='red')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
