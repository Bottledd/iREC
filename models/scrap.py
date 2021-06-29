import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
from models.BNNs.DeterministicNN import Deterministic_NN
import torch
import pyro
from pyro.infer import Predictive
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS


# the non-linearity we use in our neural network
def nonlin(x):
    return torch.tanh(x)


# a two-layer bayesian neural network with computational flow
# given by D_X => D_H => D_H => D_Y where D_H is the number of
# hidden units. (note we indicate tensor dimensions in the comments)
def model(X, Y, D_H):

    D_X, D_Y = X.shape[1], 1

    # sample first layer (we put unit normal priors on all weights)
    w1 = pyro.sample(
        "w1", dist.Normal(torch.zeros((D_X, D_H)), torch.ones((D_X, D_H)))
    )  # D_X D_H
    z1 = nonlin(torch.matmul(X, w1))  # N D_H  <= first layer of activations

    # sample second layer
    w2 = pyro.sample(
        "w2", dist.Normal(torch.zeros((D_H, D_H)), torch.ones((D_H, D_H)))
    )  # D_H D_H
    z2 = nonlin(torch.matmul(z1, w2))  # N D_H  <= second layer of activations

    # sample final layer of weights and neural network output
    w3 = pyro.sample(
        "w3", dist.Normal(torch.zeros((D_H, D_Y)), torch.ones((D_H, D_Y)))
    )  # D_H D_Y
    z3 = torch.matmul(z2, w3)  # N D_Y  <= output of the neural network

    # we put a prior on the observation noise
    prec_obs = pyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
    sigma_obs = 1.0 / torch.sqrt(prec_obs)

    # observe data
    pyro.sample("Y", dist.Normal(z3, sigma_obs), obs=Y)


def run_inference(model, X, Y, D_H):
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        warmup_steps=100,
        num_samples=500,
        num_chains=1,
    )

    mcmc.run(X, Y, D_H)
    return mcmc.get_samples()


def summary(samples):
    site_stats = {}
    for k, v in samples.items():
        site_stats[k] = {
            "mean": torch.mean(v, 0),
            "std": torch.std(v, 0),
            "5%": v.kthvalue(int(len(v) * 0.05), dim=0)[0],
            "95%": v.kthvalue(int(len(v) * 0.95), dim=0)[0],
        }
    return site_stats


if __name__ == '__main__':
    # create toy dataset
    torch.manual_seed(20)
    x = torch.cat([torch.Tensor(50).uniform_(-5, -3).sort()[0].reshape(-1,1), torch.Tensor(50).uniform_(3, 5).sort()[0].reshape(-1,1)])
    i = 50
    x_data = torch.cat([x[0:i - 15], x[i + 14:]])

    # generate some data
    alpha, beta = 1., 10.

    # generate some data
    data_generator_model = Deterministic_NN(alpha=alpha, beta=beta, num_nodes=5)
    sampled_weights = data_generator_model.sample_weights_from_prior()
    data_generator_model.make_weights_from_sample(sampled_weights)
    y_data = data_generator_model(x_data).detach() + (1 / data_generator_model.likelihood_beta ** 0.5) * torch.randn_like(
        data_generator_model(x_data).detach())

    xs = torch.linspace(-10, 10, 100)
    ys = data_generator_model(xs).detach().flatten()

    samples = run_inference(model, x_data, y_data, 10)
    pred = Predictive(model, samples)(xs, None)
    sum = summary(pred)
    hmc_y = sum["obs"]

    import pandas as pd
    hmc_predictions = pd.DataFrame({
        "inputs": xs,
        "y_mean": hmc_y["mean"],
        "y_perc_5": hmc_y["5%"],
        "y_perc_95": hmc_y["95%"],
        "true": ys,
    })
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6), sharey=True)
    ax.scatter(x_data, y_data)
    ax.plot(hmc_predictions['inputs'], hmc_predictions['true'], 'k-', label='Targets wo/ noise')
    ax.plot(hmc_predictions['inputs'], hmc_predictions['y_mean'], label='Mean Prediction')
    ax.fill_between(hmc_predictions['inputs'], hmc_predictions['y_perc_5'], hmc_predictions['y_perc_95'], alpha=0.5)
    ax.set(xlabel=r'$x$', ylabel=r'$y$', title='HMC Inference')
    fig.tight_layout()
    fig.show()