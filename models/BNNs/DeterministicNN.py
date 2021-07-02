import matplotlib.pyplot as plt
import torch
import torch.distributions as D
import torch.nn as nn
from torch.nn import functional as F

from models.BNNs.Layers.DeterministicLayer import DeterministicLayer


class Deterministic_NN(nn.Module):
    def __init__(self, input_size=1, num_nodes=10, output_size=1, alpha=1., beta=5.):
        super(Deterministic_NN, self).__init__()
        self.input_size = input_size
        self.input_layer = DeterministicLayer(input_size, num_nodes)
        self.hidden_layer = DeterministicLayer(num_nodes, num_nodes)
        self.final_layer = DeterministicLayer(num_nodes, output_size)
        self.prior_alpha = alpha
        self.likelihood_beta = beta

        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.input_layer(x.view(-1, 1)))
        x = self.activation(self.hidden_layer(x))
        x = self.final_layer(x)
        return x

    def loss_function(self, y, y_preds):
        targets = y.view(-1, 1)
        weights = torch.zeros([0])
        for w in self.parameters():
            weights = torch.cat([weights, w.flatten()])

        prior_term = self.prior_alpha / 2. * torch.einsum("i, i -> ", weights, weights)
        likelihood_term = (self.likelihood_beta / 2 * (targets - y_preds) ** 2).sum()

        return prior_term + likelihood_term

    def sample_weights_from_prior(self):
        weights = torch.zeros([0])
        for w in self.parameters():
            weights = torch.cat([weights, w.flatten()])
        prior_dist = D.MultivariateNormal(loc=torch.zeros_like(weights),
                                          covariance_matrix=(1. / self.prior_alpha) * torch.eye(weights.shape[0]))

        weight_sample = prior_dist.sample()

        return weight_sample

    def make_weights_from_sample(self, weight_sample):
        current_idx = 0
        input_layer_shape = self.input_layer.weight.shape
        input_layer_len = input_layer_shape[1]
        input_layer_sample_weights = weight_sample[current_idx: current_idx + input_layer_len]
        current_idx = current_idx + input_layer_len
        input_layer_sample_bias = weight_sample[current_idx: current_idx + input_layer_len]
        current_idx = current_idx + input_layer_len
        hidden_layer_shape = self.hidden_layer.weight.shape
        hidden_layer_len = hidden_layer_shape[0] * hidden_layer_shape[1]
        hidden_layer_sample_weights = weight_sample[current_idx: current_idx + hidden_layer_len]
        current_idx = current_idx + hidden_layer_len
        hidden_layer_sample_bias = weight_sample[current_idx: current_idx + hidden_layer_shape[1]]
        current_idx = current_idx + hidden_layer_shape[1]
        final_layer_shape = self.final_layer.weight.shape
        final_layer_len = final_layer_shape[0]
        final_layer_sample_weights = weight_sample[current_idx: current_idx + final_layer_len]
        current_idx = current_idx + final_layer_len
        final_layer_sample_bias = weight_sample[current_idx]

        self.input_layer.weight = nn.Parameter(input_layer_sample_weights.reshape(input_layer_shape))
        self.input_layer.bias = nn.Parameter(input_layer_sample_bias)
        self.hidden_layer.weight = nn.Parameter(hidden_layer_sample_weights.reshape(hidden_layer_shape))
        self.hidden_layer.bias = nn.Parameter(hidden_layer_sample_bias)
        self.final_layer.weight = nn.Parameter(final_layer_sample_weights.reshape(final_layer_shape))
        self.final_layer.bias = nn.Parameter(final_layer_sample_bias)


def train_model(model, x, y):
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    for epoch in range(10000):
        opt.zero_grad()
        y_preds = model(x)
        loss = model.loss_function(y, y_preds)
        loss.backward()
        if epoch % 1000 == 0:
            print(f"The loss is : {loss.item()}")
        opt.step()


if __name__ == '__main__':
    # ----------------------------------------------------------------------------------------------------------------#
    # ------------------------------------------------MAP EXPERIMENTS-------------------------------------------------#
    # create toy dataset

    torch.manual_seed(0)
    x = torch.Tensor(20).uniform_(-4, 4)

    # generate some data
    alpha, beta = 1., 1.

    # generate some data
    data_generator_model = Deterministic_NN(alpha=alpha, beta=beta)
    sampled_weights = data_generator_model.sample_weights_from_prior()[0]
    data_generator_model.make_weights_from_sample(sampled_weights)
    y = data_generator_model(x).detach() + (1 / data_generator_model.likelihood_beta) * torch.randn_like(
        data_generator_model(x).detach())
    x_train, y_train = x[:70], y[:70]
    x_test, y_test = x[70:], y[70:]
    # plot data
    f, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, label='Targets w/ noise')
    xs = torch.linspace(-10, 10, 100)
    ax.plot(xs, data_generator_model(xs).detach(), 'k-', label='Targets wo/ noise')
    ax.legend()
    f.tight_layout()

    f.show()

    # Model the Data

    bnn = Deterministic_NN()
    bnn.train()
    train_model(bnn, x_train, y_train)

    # plot trained model
    f, ax = plt.subplots(figsize=(8, 6))
    xs = torch.linspace(-10, 10, 100)
    bnn.eval()
    ax.scatter(x, y, label='Truth')
    ax.plot(xs, bnn(xs).detach(), 'k-', label='Model')
    ax.legend()
    f.tight_layout()
    f.show()
