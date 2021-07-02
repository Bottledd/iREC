import matplotlib.pyplot as plt
import torch
import torch.distributions as D
import torch.nn as nn
from torch.nn import functional as F

from models.BNNs.Layers.DeterministicLayer import DeterministicLayer


class BNN_for_HMC(nn.Module):
    def __init__(self, input_size=1, num_nodes=10, output_size=1, alpha=1., beta=5.):
        super(BNN_for_HMC, self).__init__()
        self.input_size = input_size
        self.input_layer = DeterministicLayer(input_size, num_nodes)
        self.hidden_layer = DeterministicLayer(num_nodes, num_nodes)
        self.final_layer = DeterministicLayer(num_nodes, output_size)
        self.prior_alpha = alpha
        self.likelihood_beta = beta
        self.weight_prior = D.Normal(loc=0., scale=1./alpha ** 0.5)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.input_layer(x.view(-1, 1)))
        x = self.activation(self.hidden_layer(x))
        x = self.final_layer(x)
        return x

    def weight_prior_lp(self):
        prior_lp = 0
        for w in self.parameters():
            prior_lp += self.weight_prior.log_prob(w).sum()
        return prior_lp

    def joint_log_prob(self, x, y):
        likelihood = D.Normal(loc=self.forward(x),
                                scale=1./self.likelihood_beta ** 0.5)

        return likelihood.log_prob(y).sum() + self.weight_prior_lp()

    def data_likelihood(self, x, y):
        likelihood = D.Normal(loc=self.forward(x),
                              scale=1. / self.likelihood_beta ** 0.5)
        return likelihood.log_prob(y).sum()

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
