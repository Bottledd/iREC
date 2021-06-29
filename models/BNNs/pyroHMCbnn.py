import matplotlib.pyplot as plt
import torch
import pyro
from pyro.distributions import Normal, Delta
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc.nuts import NUTS
from pyro.infer.mcmc.util import predictive


class BNN(object):
    def __init__(self, input_size, hidden_size, output_size, w_sigma, y_sigma):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.w_sigma = w_sigma
        self.y_sigma = y_sigma

    def model(self, x_data, y_data):
        with pyro.plate("w1_plate_dim2", self.hidden_size):
            with pyro.plate("w1_plate_dim1", self.input_size):
                w1 = pyro.sample("w1", Normal(0, self.w_sigma))
        with pyro.plate("w2_plate_dim2", self.output_size):
            with pyro.plate("w2_plate_dim1", self.hidden_size):
                w2 = pyro.sample("w2", Normal(0, self.w_sigma))

        f = lambda x: torch.mm(torch.tanh(torch.mm(x, w1)), w2)
        with pyro.plate("map", len(x_data)):
            prediction_mean = f(x_data).squeeze()
            pyro.sample("obs", Normal(prediction_mean, self.y_sigma), obs=y_data)
            return prediction_mean

    def nuts_sampling(self, x_data, y_data, num_samples, warmup_steps):
        nuts_kernel = NUTS(self.model, target_accept_prob=0.65)
        mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps)
        mcmc.run(x_data, y_data)
        self.posterior_samples = mcmc.get_samples()

    def predict(self, x_pred):
        def wrapped_model(x_data, y_data):
            pyro.sample("prediction", Delta(self.model(x_data, y_data)))

        samples = predictive(wrapped_model, self.posterior_samples, x_pred, None)
        return samples["prediction"], samples["obs"]
