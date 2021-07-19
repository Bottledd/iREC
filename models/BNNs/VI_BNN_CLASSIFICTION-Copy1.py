import matplotlib.pyplot as plt
import torch
import torch.distributions as D
import torch.nn as nn
from torch.nn import functional as F
from tqdm import trange

from models.BNNs.DeterministicNN import Deterministic_NN
from models.BNNs.Layers.BBPLinear import BBPLinear


class VI_BNN(nn.Module):
    def __init__(self, input_size=13, num_nodes=10, output_size=1, alpha=1., kl_beta=1.):
        super(VI_BNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        self.kl_beta = kl_beta
        self.priors = {'prior_mu': 0.,
                       'prior_sigma': 1. / alpha ** 0.5}

        self.input_layer = BBPLinear(self.input_size, num_nodes, self.priors)
        self.hidden_layer = BBPLinear(num_nodes, num_nodes, self.priors)
        self.output_layer = BBPLinear(num_nodes, self.output_size, self.priors)

        self.activation = torch.tanh

    def forward(self, x):

        x, kld_input_layer = self.input_layer(x)

        x = self.activation(x)

        x, kld_hidden_layer = self.hidden_layer(x)

        x = self.activation(x)

        y, kld_output_layer = self.output_layer(x)

        return y, kld_input_layer + kld_hidden_layer + kld_output_layer

    def sample_predictions(self, x, num_samples):
        y_preds = x.data.new(num_samples, x.shape[0], self.output_size)

        for i in range(num_samples):
            y, _ = self.forward(x.view(-1, 1))
            y_preds[i] = y

        return y_preds

    def loss_function(self, y, y_preds, kld):
        likelihood_dist = D.Normal(loc=y_preds, scale=(1. / self.likelihood_beta) ** 0.5)
        likelihood_term = -likelihood_dist.log_prob(y).sum()

        return likelihood_term + kld

    def get_mvn_params(self):
        means = torch.empty([0])
        stds = torch.empty([0])

        weight_params = list(self.parameters())

        for mu in weight_params[::2]:
            means = torch.cat([means, mu.detach().flatten()])

        for rho in weight_params[1::2]:
            stds = torch.cat([stds, F.softplus(rho.detach(), beta=1.).flatten()])

        return means, stds