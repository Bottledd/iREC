import matplotlib.pyplot as plt
import torch
import pyro
from pyro.distributions import Normal, Delta
from pyro.infer.autoguide.guides import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.infer.predictive import Predictive


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

    def VI(self, x_data, y_data, num_samples=1000, num_iterations=30000):
        self.guide = AutoDiagonalNormal(self.model)
        optim = Adam({"lr": 1e-3})
        loss = Trace_ELBO()
        svi = SVI(self.model, self.guide, optim=optim, loss=loss)

        # train
        pyro.clear_param_store()
        for j in range(num_iterations):
            loss = svi.step(x_data, y_data)
            if j % (num_iterations // 10) == 0:
                print("[iteration %05d] loss: %.4f" % (j + 1, loss / len(x_data)))


        dict = {}
        for i in range(num_samples):
            sample = self.guide()  # sampling
            for name, value in sample.items():
                if not dict.keys().__contains__(name):
                    dict[name] = value.unsqueeze(0)
                else:
                    dict[name] = torch.cat([dict[name], value.unsqueeze(0)], dim=0)
        self.posterior_samples = dict

    def predict(self, x_pred):
        def wrapped_model(x_data, y_data):
            pyro.sample("prediction", Delta(self.model(x_data, y_data)))

        predictive = Predictive(wrapped_model, self.posterior_samples)
        samples = predictive.forward(x_pred, None)
        return samples["prediction"], samples["obs"]