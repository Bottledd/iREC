import torch
import torch.distributions as D
from torch import nn
from torch.nn import functional as F


class BNN_KDE(nn.Module):
    def __init__(self, emp_samples, input_size=4, num_nodes=2, output_size=3, alpha=1., kl_beta=1.,
                 initial_rho=-3.0):
        super(BNN_KDE, self).__init__()
        self.register_buffer('emp_samples', emp_samples)
        self.log_kde_rho = nn.Parameter(torch.tensor([initial_rho]))
        self.prior_alpha = alpha
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
        kde_var = F.softplus(self.log_kde_rho, beta=1.) ** 2
        covariances = torch.eye(problem_dim)
        gaussian_components = D.MultivariateNormal(loc=self.emp_samples,
                                                   covariance_matrix=kde_var * covariances)

        return D.MixtureSameFamily(mixture_weights, gaussian_components)

    def sample_from_kde(self, n_samples):
        kde_std = F.softplus(self.log_kde_rho, beta=1.)
        # first need to randomly sample indices for which component to choose
        rand_idxs = torch.randint(low=0, high=self.emp_samples.shape[0], size=(n_samples,))

        # create batch of eps ~ N(0, I)
        eps = torch.randn(size=(n_samples, self.emp_samples.shape[-1]))

        # for each chosen idx, sample from that mixture component
        chosen_emp_samples = self.emp_samples[rand_idxs]

        samples = chosen_emp_samples + eps * kde_std

        return samples
    
    def batch_predict(self, weight_samples, x_data):
        # get shapes
        batch_x_dim = x_data.shape[0]
        n_samples = weight_samples.shape[0]

        # create batches of layers from posterior sample
        idx = 0
        layer_1_w = weight_samples[:, idx:idx + self.num_nodes * self.input_size].reshape(n_samples, self.num_nodes, self.input_size)
        idx += self.num_nodes * self.input_size
        layer_1_b = weight_samples[:, idx:idx + self.num_nodes]
        idx += self.num_nodes
        layer_2_w = weight_samples[:, idx:idx + self.num_nodes * self.num_nodes].reshape(n_samples, self.num_nodes,
                                                                                         self.num_nodes)
        idx += self.num_nodes * self.num_nodes
        layer_2_b = weight_samples[:, idx:idx + self.num_nodes]
        idx += self.num_nodes
        layer_3_w = weight_samples[:, idx:idx + self.num_nodes * self.output_size].reshape(n_samples, self.output_size,
                                                                             self.num_nodes)
        idx += self.num_nodes * self.output_size
        layer_3_b = weight_samples[:, idx:idx + self.output_size]

        # compute activations for layer one
        x_data = torch.einsum("bij, kj -> bki", layer_1_w, x_data) + layer_1_b[:, None].repeat(1, batch_x_dim, 1)
        x_data = self.activation(x_data)


        # compute activations for layer 2
        x_data = torch.einsum("bij, bkj -> bki", layer_2_w, x_data) + layer_2_b[:, None].repeat(1, batch_x_dim, 1)
        x_data = self.activation(x_data)

        # compute final output
        x_data = torch.einsum("bij, bkj -> bki", layer_3_w, x_data) + layer_3_b[:, None].repeat(1, batch_x_dim, 1)
        
        return x_data

    def data_likelihood(self, y_preds, y_data):
        batch_y_dim = y_preds.shape[1]
        n_samples = y_preds.shape[0]
        assert batch_y_dim == y_data.shape[0]
        categorical = D.Categorical(logits=y_preds)
        return categorical.log_prob(y_data).sum(-1)

    
    def weight_prior_lp(self, weight_samples):
        return self.weight_prior.log_prob(weight_samples).sum(1)
    
    def joint_log_prob(self, x, y, n_samples):
        # first sample weights from KDE
        weight_samples = self.sample_from_kde(n_samples)
        y_preds = self.batch_predict(weight_samples, x)
        weight_prior_lp = self.weight_prior_lp(weight_samples)
        data_likelihood_lp = self.data_likelihood(y_preds, y)
        
        return (data_likelihood_lp + weight_prior_lp).mean()

    def elbo(self, x, y, n_samples):
        # first sample weights from KDE
        weight_samples = self.sample_from_kde(n_samples)
        y_preds = self.batch_predict(weight_samples, x)
        weight_prior_lp = self.weight_prior_lp(weight_samples)
        data_likelihood_lp = self.data_likelihood(y_preds, y)
        q_lp = self.kde.log_prob(weight_samples)

        kl_term = q_lp - weight_prior_lp

        return (data_likelihood_lp - self.kl_beta * kl_term).mean()

    def log_q_w(self, weight_samples):
        return self.kde.log_prob(weight_samples).mean()
    
    def reinforce_loss(self, x, y, n_samples):
        # first sample weights from KDE
        weight_samples = self.kde.sample((n_samples,))
        y_preds = self.batch_predict(weight_samples, x)
        weight_prior_lp = self.weight_prior_lp(weight_samples)
        data_likelihood_lp = self.data_likelihood(y_preds, y)
        q_lp = self.kde.log_prob(weight_samples)
        
        with torch.no_grad():
            no_grad_elbo = data_likelihood_lp + self.kl_beta * (weight_prior_lp - q_lp)
        reinforce_term_1 = q_lp * no_grad_elbo
        reinforce_term_2 = data_likelihood_lp + self.kl_beta * (weight_prior_lp - q_lp)

        return (reinforce_term_1 + reinforce_term_2).mean()


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(0)
    device = torch.device('cpu')
    data = load_iris()
    x_ = data['data']
    y_ = data['target']
    N_tr = 10  # 50
    N_val = 140
    a = np.arange(x_.shape[0])
    train_index = np.random.choice(a, size=N_tr, replace=False)
    val_index = np.delete(a, train_index, axis=0)
    x_train = x_[train_index]
    y_train = y_[train_index]
    x_val = x_[val_index][:]
    y_val = y_[val_index][:]
    x_m = x_train.mean(0)
    x_s = x_train.std(0)
    x_train = (x_train - x_m) / x_s
    x_val = (x_val - x_m) / x_s
    D_in = x_train.shape[1]
    x_train = torch.FloatTensor(x_train)
    y_train = torch.FloatTensor(y_train)
    x_val = torch.FloatTensor(x_val)
    y_val = torch.FloatTensor(y_val)
    plt.scatter(x_train.numpy()[:, 0], y_train.numpy())

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)
    num_nodes = 3
    hmc_weights = torch.zeros((1000, 39))
    test = BNN_KDE(hmc_weights, num_nodes=num_nodes)
    weight_sample = test.sample_from_kde(50)
    test_pred_logits = test.batch_predict(weight_sample, x_train)
    test.elbo(x_train, y_train, 149)
