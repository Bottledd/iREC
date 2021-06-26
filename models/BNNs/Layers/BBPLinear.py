import torch
import torch.distributions as D
import torch.nn as nn
from torch.nn import functional as F


class BBPLinear(nn.Module):
    """
    Linear layer following style of Bayes-by-backprop with local re-parametrisation
    """
    def __init__(self, in_features, out_features, priors=None):
        super(BBPLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if priors is None:
            priors = {
                'prior_mu' : 0.,
                'prior_sigma' : 1.,
            }

        # weight priors
        self.w_prior_mu = priors['prior_mu']
        self.w_prior_sigma = priors['prior_sigma']

        # bias priors
        self.b_prior_mu = priors['prior_mu']
        self.b_prior_sigma = priors['prior_sigma']

        # learnable posterior parameters
        self.w_post_mu = nn.Parameter(torch.Tensor(self.in_features, self.out_features).uniform_(-0.1, 0.1))
        self.w_post_rho = nn.Parameter(torch.Tensor(self.in_features, self.out_features).uniform_(-3., -2.))

        self.b_post_mu = nn.Parameter(torch.Tensor(self.out_features).uniform_(-0.1, 0.1))
        self.b_post_rho = nn.Parameter(torch.Tensor(self.out_features).uniform_(-3., -2.))

    def forward(self, x, sample=True):
        if not sample:
            # act mu is given by AW where A is  inputs and W_ij = mu_ij
            activation_mu = torch.einsum("bi, ij -> bj", x, self.w_post_mu) + self.b_post_mu[None].repeat(x.shape[0], 1)

            return activation_mu

        else:
            # compute stddvs
            w_post_sigma = F.softplus(self.w_post_rho, beta=1.)
            b_post_sigma = F.softplus(self.b_post_rho, beta=1.)

            # act mu is given by AW where A is  inputs and W_ij = mu_ij
            activation_mu = torch.einsum("bi, ij -> bj", x, self.w_post_mu) + self.b_post_mu[None].repeat(x.shape[0], 1)

            # act sigma is given by A*W* where A*_ij = a_ij^2 and W* = sigma_ij^2
            activation_var = torch.einsum("bi, ij -> bj", x**2, w_post_sigma ** 2) + b_post_sigma.pow(2)[None].repeat(x.shape[0], 1)
            activation_sigma = torch.sqrt(activation_var)

            # sample epsilon ~ N(0, 1)
            eps = torch.empty_like(activation_mu).normal_(mean=0, std=1)

            # local reparametrisation
            output = activation_mu + activation_sigma * eps

            # compute KL divergence of layer
            kld = analytical_kl(self.w_prior_mu, self.w_prior_sigma, self.w_post_mu, w_post_sigma) + \
                  analytical_kl(self.b_prior_mu, self.b_prior_sigma, self.b_post_mu, b_post_sigma)

            return output, kld

def analytical_kl(prior_mu, prior_sigma, post_mu, post_sigma):
    neg_KLD = 0.5 * (1 + 2 * torch.log(post_sigma / prior_sigma) - ((post_mu - prior_mu) / prior_sigma).pow(2)
                     - (post_sigma / prior_sigma).pow(2)).sum()

    return - neg_KLD