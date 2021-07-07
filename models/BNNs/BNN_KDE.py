import torch
import torch.distributions as D
from torch import nn


class BNN_KDE(nn.Module):
    def __init__(self, emp_samples, input_size=1, num_nodes=2, output_size=1, alpha=1., beta=5.):
        super(BNN_KDE, self).__init__()
        self.register_buffer('emp_samples', emp_samples)
        self.log_kde_std = nn.Parameter(torch.tensor([-1.]))
        self.prior_alpha = alpha
        self.likelihood_beta = beta
        self.weight_prior = D.Normal(loc=0., scale=1./alpha ** 0.5)
        self.activation = nn.Tanh()
        self.input_size = input_size
        self.num_nodes = num_nodes
        self.output_size = output_size

    @property
    def kde(self):
        batch_dim, problem_dim = self.emp_samples.shape
        mixture_weights = D.Categorical(probs=torch.ones(batch_dim))
        kde_std = torch.exp(self.log_kde_std)
        gaussian_components = D.MultivariateNormal(loc=self.emp_samples,
                                                   covariance_matrix=kde_std ** 2 * torch.eye(problem_dim))

        return D.MixtureSameFamily(mixture_weights, gaussian_components)

    def sample_from_kde(self, n_samples):
        kde_std = torch.exp(self.log_kde_std)
        # first need to randomly sample indices for which component to choose
        rand_idxs = torch.randint(low=0, high=self.emp_samples.shape[0], size=(n_samples,))

        # create batch of eps ~ N(0, I)
        eps = torch.randn(size=(n_samples, self.emp_samples.shape[-1]))

        # for each chosen idx, sample from that mixture component
        chosen_emp_samples = self.emp_samples[rand_idxs]

        samples = chosen_emp_samples + eps * kde_std

        return samples

    def batch_regression(self, x_data, n_samples):
        # get shapes
        batch_x_dim = x_data.shape[0]

        # first sample weights from KDE
        weight_samples = self.sample_from_kde(n_samples)

        # create batches of layers from posterior sample
        idx = 0
        layer_1_w = weight_samples[:, idx:idx + self.num_nodes]
        idx += self.num_nodes
        layer_1_b = weight_samples[:, idx:idx + self.num_nodes]
        idx += self.num_nodes
        layer_2_w = weight_samples[:, idx:idx + self.num_nodes * self.num_nodes].reshape(n_samples, self.num_nodes, self.num_nodes)
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

        return x_data, weight_samples

    def data_likelihood(self, y_preds, y_data):
        likelihood_lp = D.Normal(loc=y_preds, scale=1. / self.likelihood_beta ** 0.5).log_prob(y_data.flatten()).sum(
            -1)
        return likelihood_lp

    def joint_log_prob(self, x, y, n_samples):
        y_preds, weight_samples = self.batch_regression(x, n_samples)

        weight_prior_lp = self.weight_prior.log_prob(weight_samples).sum(1)

        likelihood_lp = self.data_likelihood(y_preds, y)

        return likelihood_lp.mean() + weight_prior_lp.mean(), weight_samples

    def elbo(self, x, y, n_samples):
        p_joint_lp, weight_samples = self.joint_log_prob(x, y, n_samples)
        q_lp = self.kde.log_prob(weight_samples).mean()

        return p_joint_lp - q_lp


if __name__ == '__main__':
    import pickle as pkl
    torch.set_default_tensor_type(torch.DoubleTensor)
    emp_samples = pkl.load(open("../../PickledStuff/emp_samples.pkl", "rb"))
    x_data = pkl.load(open("../../PickledStuff/x_data.pkl", "rb"))
    y_data = pkl.load(open("../../PickledStuff/y_data.pkl", "rb"))
    xs = pkl.load(open("../../PickledStuff/xs.pkl", "rb"))
    ys = pkl.load(open("../../PickledStuff/ys.pkl", "rb"))

    test = BNN_KDE(emp_samples[torch.unique(torch.randint(low=0, high=emp_samples.shape[0], size=(100,)))])
    test_r, w = test.batch_regression(x_data, 10000)

    import matplotlib.pyplot as plt

    plt.plot(x_data, test_r.detach().numpy().mean(0))
    plt.scatter(x_data, y_data)
    plt.show()

    num_epochs = 1000
    num_parallel_samples = 100
    optimiser = torch.optim.Adamax(test.parameters(), lr=1e-1)
    losses = []
    stds = []
    for i in range(num_epochs):
        if i == 500:
            print('break')
        optimiser.zero_grad()
        loss = -test.elbo(x_data, y_data, num_parallel_samples)
        losses.append(loss.item())
        stds.append(torch.exp(test.log_kde_std)[0])
        if i % 100 == 0:
            print(f'The loss is: {loss.item():.5f}, val of std is: {torch.exp(test.log_kde_std)[0]:.5f}')
        loss.backward()
        optimiser.step()