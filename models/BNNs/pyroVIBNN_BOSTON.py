import torch
import pyro
from torch import nn
from pyro.nn import PyroModule, PyroSample
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDiagonalNormal
from models.BNNs.DeterministicNN import Deterministic_NN
from pyro.contrib.bnn.hidden_layer import HiddenLayer


class BayesianNeuralNetwork(PyroModule):
    def __init__(self, in_features=1, hidden_nodes=10, out_features=1, prior_var=1., likelihood_var=1./25.):
        super(BayesianNeuralNetwork, self).__init__()
        print(prior_var)
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

        self.prior_var = prior_var
        self.likelihood_var = likelihood_var
        self.activation = torch.tanh

    def forward(self, x, y=None):
        x = self.activation(self.input_layer(x))
        x = self.activation(self.hidden_layer(x))
        mean = self.output_layer(x).squeeze()

        with pyro.plate("data", x.shape[0]):
            if y is not None:
                y = y.squeeze()
            obs = pyro.sample("obs", dist.Normal(mean, self.likelihood_var ** 0.5), obs=y)
        return mean
   