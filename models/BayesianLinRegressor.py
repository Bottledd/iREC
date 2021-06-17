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
        self.feature_targets_train = None
        self.feature_targets_test = None
        self.regression_targets = None
        self.regression_targets_train = None
        self.regression_targets_test = None
        self.weight_prior = dst.MultivariateNormal(loc=prior_mean,
                                                   precision_matrix=prior_alpha * torch.eye(prior_mean.shape[0]))
        self.weight_posterior = None
        self.num_targets = num_targets
        self.true_sample = None

    def sample_feature_inputs(self):
        standard_normal = dst.Normal(loc=0.0, scale=1.0)
        # standard_normal = dst.Uniform(-10, 10)
        self.feature_targets = standard_normal.sample((self.num_targets, self.dim))

        # train test split with 75% train
        num_train_samples = int(self.num_targets * 0.75)
        self.feature_targets_train = self.feature_targets[:num_train_samples]
        self.feature_targets_test = self.feature_targets[num_train_samples:]

    def sample_regression_targets(self):
        # sample weights
        weight_sample = self.weight_prior.sample()
        self.true_sample = weight_sample
        # broadcast to correct shapes
        weight_matrix = torch.tile(weight_sample, (self.feature_targets.shape[0], 1))

        # make feature matrix
        feature_matrix = torch.hstack((self.feature_targets, torch.ones(self.feature_targets.shape[0], 1)))
        # multiply weights to features
        feature_times_weights = torch.sum(weight_matrix * feature_matrix, dim=1)

        # add signal noise
        epsilon = dst.Normal(loc=0.0, scale=self.signal_std)
        additive_noise = epsilon.sample(feature_times_weights.shape)
        self.regression_targets = feature_times_weights + additive_noise

        num_train_samples = int(self.num_targets * 0.75)
        self.regression_targets_train = self.regression_targets[:num_train_samples]
        self.regression_targets_test = self.regression_targets[num_train_samples:]

    def posterior_update(self):
        # make feature matrix
        feature_matrix = torch.hstack((self.feature_targets_train, torch.ones(self.feature_targets_train.shape[0], 1)))

        # compute posterior covariance
        post_precision = (1 / self.signal_std ** 2) * (
                feature_matrix.T @ feature_matrix) + self.weight_prior.precision_matrix

        post_mean = (1 / self.signal_std ** 2) * (
                torch.inverse(post_precision) @ feature_matrix.T @ self.regression_targets_train)

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

    def plot_regression_with_uncertainty(self, plot_samples=False):
        if plot_samples:
            plt.plot(self.feature_targets, self.regression_targets, 'o', label="observations")
        x_axis = torch.linspace(-10, 10, 1000)
        mean, error = self.predictive_distribution(x_axis.reshape(-1, 1))
        plt.plot(x_axis, mean, '-', color='red')
        plt.fill_between(x_axis, mean - 1.96 * error ** 0.5, mean + 1.96 * error ** 0.5,
                         color='gray', alpha=0.2)
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

    def predictive_distribution(self, inputs):
        design_matrix = torch.hstack((inputs, torch.ones(inputs.shape[0], 1))) # shape [num_targets, dim]

        mean = self.weight_posterior.mean
        covariance = self.weight_posterior.covariance_matrix
        mean_matrix = torch.tile(mean, (design_matrix.shape[0], 1))
        covar_matrix = torch.tile(covariance, (design_matrix.shape[0], 1, 1))
        predictive_mean = torch.einsum("ij, ij -> i", mean_matrix, design_matrix)
        predictive_variance = torch.einsum("bi, bik, bk -> b", design_matrix, covar_matrix, design_matrix) + self.signal_std ** 2

        return predictive_mean, predictive_variance

    def empirical_prediction(self, weights):
        # broadcast to correct shapes
        weight_matrix = torch.tile(weights, (self.feature_targets_test.shape[0], 1))

        # make feature matrix
        feature_matrix = torch.hstack((self.feature_targets_test, torch.ones(self.feature_targets_test.shape[0], 1)))

        # multiply weights to features
        feature_times_weights = torch.sum(weight_matrix * feature_matrix, dim=1)

        return feature_times_weights

    def measure_performance(self, weights, kind='MSE'):
        # broadcast to correct shapes
        weight_matrix = torch.tile(weights, (self.feature_targets_test.shape[0], 1))

        # make feature matrix
        feature_matrix = torch.hstack((self.feature_targets_test, torch.ones(self.feature_targets_test.shape[0], 1)))

        # multiply weights to features
        feature_times_weights = torch.sum(weight_matrix * feature_matrix, dim=1)
        if kind == 'MSE':
            sample_mse = (self.regression_targets_test - feature_times_weights) ** 2
            return torch.mean(sample_mse)

        elif kind == 'MAE':
            mae = torch.abs(self.regression_targets_test - feature_times_weights)
            return torch.mean(mae)

    def measure_true_performance(self, kind='MSE', samples=1000):
        weights = self.weight_posterior.sample((samples,))
        # broadcast to correct shapes
        weight_matrix = torch.tile(weights[None], (self.feature_targets_test.shape[0], 1, 1))

        # make feature matrix
        feature_matrix = torch.hstack((self.feature_targets_test, torch.ones(self.feature_targets_test.shape[0], 1)))

        # multiply weights to features
        feature_times_weights = torch.einsum("ibj, ij -> bi", weight_matrix, feature_matrix)
        if kind == 'MSE':
            sample_mse = (self.regression_targets - feature_times_weights) ** 2
            return torch.mean(sample_mse)

        elif kind == 'MAE':
            mae = torch.abs(self.regression_targets - feature_times_weights)
            return torch.mean(mae)

    def log_likelihood_test(self, weights):
        preds = self.empirical_prediction(weights)

        likelihood_gaussian = dst.normal.Normal(loc=preds, scale=self.signal_std)

        # compute likelihood of truth
        log_likelihood = likelihood_gaussian.log_prob(self.regression_targets_test)

        return torch.mean(log_likelihood)


if __name__ == '__main__':
    blr = BayesLinRegressor(prior_mean=torch.zeros(2),
                        prior_alpha=1,
                        signal_std=1,
                        num_targets=10000,
                        seed=10)
    blr.sample_feature_inputs()
    blr.sample_regression_targets()
    blr.posterior_update()
    target = blr.weight_posterior
    blr.plot_regression_with_uncertainty(plot_samples=True)
