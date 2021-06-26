import torch
import torch.distributions as D
import torch.nn as nn
from torch.nn import functional as F


class DeterministicLayer(nn.Module):
    """
    Linear layer following style of Bayes-by-backprop with local re-parametrisation
    """
    def __init__(self, in_features, out_features):
        super(DeterministicLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # learnable weight and bias
        self.weight = nn.Parameter(torch.Tensor(self.in_features, self.out_features).uniform_(-0.1, 0.1))
        self.bias = nn.Parameter(torch.Tensor(self.out_features).uniform_(-0.1, 0.1))

    def forward(self, x):

        # act mu is given by AW where A is  inputs and W_ij = mu_ij
        activation = torch.einsum("bi, ij -> bj", x, self.weight) + self.bias[None].repeat(x.shape[0], 1)

        return activation