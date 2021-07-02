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
        x = self.activation(self.input_layer(x.view(-1, 1)))
        x = self.activation(self.hidden_layer(x))
        mean = self.output_layer(x).squeeze()

        with pyro.plate("data", x.shape[0]):
            if y is not None:
                y = y.squeeze()
            obs = pyro.sample("obs", dist.Normal(mean, self.likelihood_var ** 0.5), obs=y)
        return mean


if __name__ == '__main__':
    # make data
    torch.set_default_tensor_type(torch.DoubleTensor)
    # create toy dataset
    torch.manual_seed(10)
    x = torch.Tensor(40).uniform_(-5, 5).sort()[0]
    i = 20
    x_data = torch.cat([x[0:i - 10], x[i + 9:]])

    # generate some data
    alpha, beta = 1., 25.

    # generate some data
    data_generator_model = Deterministic_NN(alpha=alpha, beta=beta)
    sampled_weights = data_generator_model.sample_weights_from_prior()
    data_generator_model.make_weights_from_sample(sampled_weights)
    y_data = data_generator_model(x_data).detach() + (1 / data_generator_model.likelihood_beta ** 0.5) * torch.randn_like(
        data_generator_model(x_data).detach())

    # create models
    model = BayesianNeuralNetwork()
    guide = AutoDiagonalNormal(model)

    from pyro.infer import SVI, Trace_ELBO

    adam = pyro.optim.Adam({"lr": 0.005})
    svi = SVI(model, guide, adam, loss=Trace_ELBO())

    num_iterations = 5000
    pyro.clear_param_store()
    for j in range(num_iterations):
        # calculate the loss and take a gradient step
        loss = svi.step(x_data, y_data)
        if j % 100 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(x_data)))

    guide.requires_grad_(False)

    for name, value in pyro.get_param_store().items():
        print(name, pyro.param(name))

    from pyro.infer import Predictive


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


    predictive = Predictive(model, guide=guide, num_samples=800,
                            return_sites=("obs", "_RETURN"))


    xs = torch.linspace(-10, 10, 100)
    ys = data_generator_model(xs).detach().flatten()
    samples = predictive(xs)
    pred_summary = summary(samples)

    mu = pred_summary["_RETURN"]
    y = pred_summary["obs"]

    import pandas as pd
    predictions = pd.DataFrame({
        "inputs": xs,
        "mu_mean": mu["mean"],
        "mu_perc_5": mu["5%"],
        "mu_perc_95": mu["95%"],
        "y_mean": y["mean"],
        "y_perc_5": y["5%"],
        "y_perc_95": y["95%"],
        "true": ys,
    })

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6), sharey=True)
    ax.scatter(x_data, y_data)
    ax.plot(predictions['inputs'], predictions['true'], 'k-', label='Targets wo/ noise')
    ax.plot(predictions['inputs'], predictions['mu_mean'], label='Mean Prediction')
    ax.fill_between(predictions['inputs'], predictions['mu_perc_5'], predictions['mu_perc_95'], alpha=0.5)
    ax.set(xlabel=r'$x$', ylabel=r'$y$', title='Mean-Field Variational Inference')
    fig.tight_layout()
    fig.show()