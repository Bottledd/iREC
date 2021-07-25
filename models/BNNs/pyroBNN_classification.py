import torch
import pyro
from torch import nn
from pyro.nn import PyroModule, PyroSample
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDiagonalNormal
from models.BNNs.DeterministicNN import Deterministic_NN
from pyro.contrib.bnn.hidden_layer import HiddenLayer


class BayesianNeuralNetwork(PyroModule):
    def __init__(self, in_features=4, hidden_nodes=10, out_features=3, prior_var=1., kl_beta=1.):
        super(BayesianNeuralNetwork, self).__init__()
        self.input_layer = PyroModule[nn.Linear](in_features, hidden_nodes)
        self.input_layer.weight = PyroSample(dist.Normal(0., prior_var ** 0.5).expand([hidden_nodes, in_features]).to_event(2))
        self.input_layer.bias = PyroSample(dist.Normal(0., prior_var ** 0.5).expand([hidden_nodes]).to_event(1))
        self.hidden_layer = PyroModule[nn.Linear](hidden_nodes, hidden_nodes)
        self.hidden_layer.weight = PyroSample(
            dist.Normal(0., prior_var ** 0.5).expand([hidden_nodes, hidden_nodes]).to_event(2))
        self.hidden_layer.bias = PyroSample(dist.Normal(0., prior_var ** 0.5).expand([hidden_nodes]).to_event(1))
        self.output_layer = PyroModule[nn.Linear](hidden_nodes, out_features)
        self.output_layer.weight = PyroSample(
            dist.Normal(0., prior_var ** 0.5).expand([out_features, hidden_nodes]).to_event(2))
        self.output_layer.bias = PyroSample(dist.Normal(0., prior_var ** 0.5).expand([out_features]).to_event(1))
        self.kl_beta = kl_beta
        self.prior_var = prior_var
        self.activation = torch.tanh

    def forward(self, x, y=None):
        x = self.activation(self.input_layer(x))
        x = self.activation(self.hidden_layer(x))
        logits = self.output_layer(x)
        
        with pyro.poutine.scale(scale=1. / self.kl_beta):
            with pyro.plate('data'):
                pyro.sample('obs', dist.Categorical(logits=logits), obs=y)
        
        return logits
