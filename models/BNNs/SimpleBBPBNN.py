import torch
import torch.distributions as D
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

from tqdm import trange
from models.BNNs.Layers.BBPLinear import BBPLinear
from models.BNNs.DeterministicNN import Deterministic_NN


class SimpleBBPBNN(nn.Module):
    def __init__(self, input_size=1, num_nodes=5, output_size=1, alpha=1., beta=5.):
        super(SimpleBBPBNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.priors = {'prior_mu' : 0.,
                       'prior_sigma' : 1. / alpha}

        self.input_layer = BBPLinear(self.input_size, num_nodes, self.priors)
        self.output_layer = BBPLinear(num_nodes, self.output_size, self.priors)

        self.activation = nn.ReLU(inplace=True)
        self.likelihood_beta = beta

    def forward(self, x):

        x, kld_input_layer = self.input_layer(x.view(-1, 1))

        x = self.activation(x)

        y, kld_output_layer = self.output_layer(x)

        return y, kld_input_layer + kld_output_layer

    def forward_using_mean(self, x):
        x = self.input_layer(x.view(-1, 1), sample=False)

        x = self.activation(x)

        y = self.output_layer(x, sample=False)

        return y

    def sample_predictions(self, x, num_samples):
        y_preds = x.data.new(num_samples, x.shape[0], self.output_size)

        for i in range(num_samples):
            y, _ = self.forward(x.view(-1, 1))
            y_preds[i] = y

        return y_preds

    def loss_function(self, y, y_preds, kld):
        likelihood_term = (self.likelihood_beta / 2 * (y - y_preds) ** 2).sum()

        return likelihood_term + kld

    def get_mvn_params(self):
        means = torch.empty([0])
        stds = torch.empty([0])

        weight_params = list(self.parameters())

        for mu in weight_params[::2]:
            means = torch.cat([means, mu.detach().flatten()])

        for std in weight_params[1::2]:
            stds = torch.cat([stds, F.softplus(std.detach(), beta=1.).flatten()])

        return means, stds


def train_bnn(model, x, y, epochs=1000, num_bnn_samples_per_epoch=16):
    opt = torch.optim.Adam(model.parameters(), lr=5e-2)
    pbar = trange(epochs)
    for epoch in pbar:
        opt.zero_grad()
        loss = 0
        for i in range(num_bnn_samples_per_epoch):
            y_preds, kld = model(x)
            loss = loss + model.loss_function(y, y_preds, kld)
        loss = loss / num_bnn_samples_per_epoch
        loss.backward()
        pbar.set_description(f"The loss is : {loss.item()}")
        opt.step()


if __name__ == '__main__':
    # ----------------------------------------------------------------------------------------------------------------#
    # ------------------------------------------------BBP EXPERIMENTS-------------------------------------------------#
    # ----------------------------------------------------------------------------------------------------------------#
    # create toy dataset
    torch.manual_seed(0)
    x = torch.Tensor(20).uniform_(-4, 4)

    # generate some data
    alpha, beta = 1., 100.

    # generate some data
    data_generator_model = Deterministic_NN(alpha=alpha, beta=beta)
    sampled_weights = data_generator_model.sample_weights_from_prior()[0]
    data_generator_model.make_weights_from_sample(sampled_weights)
    y = data_generator_model(x).detach() + (1 / data_generator_model.likelihood_beta) * torch.randn_like(
        data_generator_model(x).detach())
    x_train, y_train = x[:70], y[:70]
    x_test, y_test = x[70:], y[70:]

    # plot data
    f, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, label='Targets w/ noise')
    xs = torch.linspace(-10, 10, 100)
    ax.plot(xs, data_generator_model(xs).detach(), 'k-', label='Targets wo/ noise')
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    f.tight_layout()
    f.show()

    # Model the Data
    bnn = SimpleBBPBNN(alpha=alpha, beta=beta)
    bnn.train()
    train_bnn(bnn, x, y)

    # plot trained model
    f, ax = plt.subplots(figsize=(8, 6))
    xs = torch.linspace(-10, 10, 100)
    # sample some predictions
    bnn.eval()
    y_pred_samples = bnn.sample_predictions(xs, 500).detach()
    ax.scatter(x, y, label='Truth')
    for pred in y_pred_samples:
        ax.plot(xs, pred, 'r-', alpha=0.2)
    ax.plot(xs, data_generator_model(xs).detach(), 'k-', label='Targets wo/ noise')
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    f.tight_layout()
    f.show()

    # get uncertainty bounds
    f, ax = plt.subplots(figsize=(8, 6))
    y_pred_mean, y_pred_std = y_pred_samples.mean(0).flatten(), y_pred_samples.std(0).flatten()
    ax.scatter(x, y, label='Truth')
    ax.plot(xs, y_pred_mean, 'r-', label='Predictive Mean')
    ax.fill_between(xs, y_pred_mean - 1.96 * y_pred_std ** 0.5, y_pred_mean + 1.96 * y_pred_std ** 0.5,
                     color='gray', alpha=0.2, label='95% Error Bars')
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    f.tight_layout()
    f.show()

    # compute MVN used to communicate weights
    means, stds = bnn.get_mvn_params()
    variational_posterior = D.MultivariateNormal(loc=means, covariance_matrix=torch.diag(stds ** 2))

    # plot true samples
    num_compressed_samples = 1000
    true_ensemble_preds = torch.zeros([num_compressed_samples, xs.shape[0], 1])
    ensemble_weights = variational_posterior.sample((num_compressed_samples,))
    for i, compressed_sample in enumerate(ensemble_weights):
        ensemble_model = Deterministic_NN(alpha=alpha, beta=beta)
        ensemble_model.make_weights_from_sample(compressed_sample)
        true_ensemble_preds[i] = ensemble_model(xs).detach()

    true_ensemble_preds_mean = true_ensemble_preds.mean(0).flatten()
    true_ensemble_preds_stds = true_ensemble_preds.std(0).flatten()

    # get uncertainty bounds
    f, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, label='Truth')
    ax.plot(xs, true_ensemble_preds_mean, 'r-', label='Predictive Mean')
    ax.fill_between(xs, true_ensemble_preds_mean - 1.96 * true_ensemble_preds_stds ** 0.5, true_ensemble_preds_mean + 1.96 * true_ensemble_preds_stds ** 0.5,
                    color='gray', alpha=0.2, label='95% Error Bars')
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    f.tight_layout()
    f.show()

    #### sample weights with compression algorithm
    from rec.beamsearch.Coders.Encoder_Variational import Encoder
    from rec.beamsearch.distributions.CodingSampler import CodingSampler
    from rec.beamsearch.distributions.VariationalPosterior import VariationalPosterior
    from rec.beamsearch.samplers.GreedySampling import GreedySampler

    coding_sampler = CodingSampler
    auxiliary_posterior = VariationalPosterior
    selection_sampler = GreedySampler
    omega = 5

    initial_seed = 0
    beamwidth = 1
    epsilon = 0.

    compressed_weights = torch.zeros([0])
    num_compressed_samples = 1000
    for i in trange(num_compressed_samples):
        initial_seed = initial_seed + torch.randint(low=10, high=500, size=(1,))
        encoder = Encoder(variational_posterior,
                          initial_seed,
                          coding_sampler,
                          selection_sampler,
                          auxiliary_posterior,
                          omega,
                          epsilon=epsilon,
                          beamwidth=beamwidth)

        w, idx = encoder.run_encoder()

        compressed_weights = torch.cat([compressed_weights, w[0][None]])

    # plot compressed samples
    ensemble_preds = torch.zeros([num_compressed_samples, xs.shape[0], 1])
    for i, compressed_sample in enumerate(compressed_weights):
        compressed_model = Deterministic_NN(alpha=alpha, beta=beta)
        compressed_model.make_weights_from_sample(compressed_sample)
        ensemble_preds[i] = compressed_model(xs).detach()

    ensemble_preds_mean = ensemble_preds.mean(0).flatten()
    ensemble_preds_stds = ensemble_preds.std(0).flatten()

    # get uncertainty bounds
    f, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, label='Truth')
    ax.plot(xs, ensemble_preds_mean, 'r-', label='Predictive Mean')
    ax.fill_between(xs, ensemble_preds_mean - 1.96 * ensemble_preds_stds ** 0.5, ensemble_preds_mean + 1.96 * ensemble_preds_stds ** 0.5,
                    color='gray', alpha=0.2, label='95% Error Bars')
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    f.tight_layout()
    f.show()
