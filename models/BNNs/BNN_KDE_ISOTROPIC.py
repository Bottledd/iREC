import torch
import torch.distributions as D
from torch import nn
from torch.nn import functional as F

class BNN_KDE(nn.Module):
    def __init__(self, emp_samples, input_size=1, num_nodes=2, output_size=1, alpha=1., beta=5., kl_beta=1.):
        super(BNN_KDE, self).__init__()
        self.register_buffer('emp_samples', emp_samples)
        self.log_kde_rhos = nn.Parameter(torch.zeros(emp_samples.shape[0]))
        self.prior_alpha = alpha
        self.likelihood_beta = beta
        self.weight_prior = D.Normal(loc=0., scale=1. / alpha ** 0.5)
        self.activation = nn.Tanh()
        self.input_size = input_size
        self.num_nodes = num_nodes
        self.output_size = output_size
        self.kl_beta = kl_beta

    @property
    def kde(self):
        batch_dim, problem_dim = self.emp_samples.shape
        mixture_weights = D.Categorical(probs=torch.ones(batch_dim))
        kde_var = F.softplus(self.log_kde_rhos, beta=1.) ** 2
        covariances = torch.eye(problem_dim)[None].repeat(self.log_kde_rhos.shape[0], 1, 1)
        gaussian_components = D.MultivariateNormal(loc=self.emp_samples,
                                                   covariance_matrix=kde_var * covariances)

        return D.MixtureSameFamily(mixture_weights, gaussian_components)

    def sample_from_kde(self, n_samples):
        kde_std = F.softplus(self.log_kde_rhos, beta=1.)
        # first need to randomly sample indices for which component to choose
        rand_idxs = torch.randint(low=0, high=self.emp_samples.shape[0], size=(n_samples,))

        # create batch of eps ~ N(0, I)
        eps = torch.randn(size=(n_samples, self.emp_samples.shape[-1]))

        # for each chosen idx, sample from that mixture component
        chosen_emp_samples = self.emp_samples[rand_idxs]
        chosen_stds = kde_std[rand_idxs]

        samples = chosen_emp_samples + eps * chosen_stds

        return samples

    def batch_regression(self, weight_samples, x_data, n_samples):
        # get shapes
        batch_x_dim = x_data.shape[0]

        # create batches of layers from posterior sample
        idx = 0
        layer_1_w = weight_samples[:, idx:idx + self.num_nodes]
        idx += self.num_nodes
        layer_1_b = weight_samples[:, idx:idx + self.num_nodes]
        idx += self.num_nodes
        layer_2_w = weight_samples[:, idx:idx + self.num_nodes * self.num_nodes].reshape(n_samples, self.num_nodes,
                                                                                         self.num_nodes)
        idx += self.num_nodes * self.num_nodes
        layer_2_b = weight_samples[:, idx:idx + self.num_nodes]
        idx += self.num_nodes
        layer_3_w = weight_samples[:, idx:idx + self.num_nodes]
        idx += self.num_nodes
        layer_3_b = weight_samples[:, idx]

        # compute activations for layer one
        x_data = torch.einsum("bi, kj -> bki", layer_1_w, x_data) + layer_1_b[:, None].repeat(1, batch_x_dim, 1)
        x_data = self.activation(x_data)

        # compute activations for layer 2
        x_data = torch.einsum("bij, bkj -> bki", layer_2_w, x_data) + layer_2_b[:, None].repeat(1, batch_x_dim, 1)
        x_data = self.activation(x_data)

        # compute final output
        x_data = torch.einsum("bi, bki -> bk", layer_3_w, x_data) + layer_3_b[:, None].repeat(1, batch_x_dim)

        return x_data

    def data_likelihood(self, y_preds, y_data):
        likelihood_lp = D.Normal(loc=y_preds, scale=1. / self.likelihood_beta ** 0.5).log_prob(y_data.flatten()).sum(
            -1)
        return likelihood_lp.mean()

    def weight_prior_lp(self, weight_samples):
        return self.weight_prior.log_prob(weight_samples).sum(1)

    def joint_log_prob(self, x, y, n_samples):
        y_preds, weight_samples = self.batch_regression(x, n_samples)

        weight_prior_lp = self.weight_prior.log_prob(weight_samples).sum(1)

        likelihood_lp = self.data_likelihood(y_preds, y)

        return likelihood_lp.mean() + weight_prior_lp.mean(), weight_samples

    def elbo(self, x, y, n_samples):
        # first sample weights from KDE
        weight_samples = self.sample_from_kde(n_samples)
        y_preds = self.batch_regression(weight_samples, x, n_samples)
        weight_prior_lp = self.weight_prior_lp(weight_samples)
        data_likelihood_lp = self.data_likelihood(y_preds, y)
        q_lp = self.kde.log_prob(weight_samples)

        kl_term = (q_lp - weight_prior_lp).mean()

        return data_likelihood_lp - self.kl_beta * kl_term

    def log_q_w(self, weight_samples):
        return self.kde.log_prob(weight_samples).mean()