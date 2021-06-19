import torch
import torch.distributions as dist
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import trange

from rec.beamsearch.Coders.Encoder_Variational import Encoder


class VAE(nn.Module):
    def __init__(self, input_size, enc_num_hidden_layers, dec_num_hidden_layers, enc_latent_size, dec_latent_size,
                 latent_dim):
        super(VAE, self).__init__()
        self.input_size = input_size
        self.enc_input_layer = nn.Linear(input_size, enc_latent_size)
        self.enc_hidden_layers = nn.ModuleList(
            [nn.Linear(enc_latent_size, enc_latent_size) for i in range(enc_num_hidden_layers)])
        self.enc_final_layer_mean = nn.Linear(enc_latent_size, latent_dim)
        self.enc_final_layer_logvar = nn.Linear(enc_latent_size, latent_dim)
        self.dec_input_layer = nn.Linear(latent_dim, dec_latent_size)
        self.dec_hidden_layers = nn.ModuleList(
            [nn.Linear(dec_latent_size, dec_latent_size) for i in range(dec_num_hidden_layers)])
        self.dec_final_layer = nn.Linear(dec_latent_size, input_size)
        self.log_scale = nn.Parameter(torch.tensor([0.0]))

    def encode(self, x):
        x = F.relu(self.enc_input_layer(x))

        for f in self.enc_hidden_layers:
            x = F.relu(f(x))

        return self.enc_final_layer_mean(x), self.enc_final_layer_logvar(x)

    def sample_latent(self, mu, logvar):
        std = torch.exp(logvar / 2)

        qzx = dist.normal.Normal(loc=mu, scale=std)

        z = qzx.rsample()
        return z

    def decode(self, z):
        z = F.relu(self.dec_input_layer(z))

        for f in self.dec_hidden_layers:
            z = F.relu(f(z))

        # pass through sigmoid so pixel intensities between [0,1]
        return torch.sigmoid(self.dec_final_layer(z))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_size))
        z = self.sample_latent(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z

    def loss_function(self, x, x_recon, mu, logvar, z):
        qzx = dist.normal.Normal(loc=mu, scale=torch.exp(logvar / 2))
        pz = dist.normal.Normal(loc=torch.zeros_like(mu), scale=torch.ones_like(logvar))

        pxz = dist.normal.Normal(loc=x_recon, scale=torch.exp(self.log_scale))

        log_p_x_z = pxz.log_prob(x)

        log_p_z = pz.log_prob(z)

        log_q_z_x = qzx.log_prob(z)

        # need to sum across all pixels to get joint probability
        log_p_x_z_per_batch = torch.sum(log_p_x_z, dim=1)

        # need to average
        log_p_z_per_batch = torch.sum(log_p_z, dim=1)
        log_q_z_x_per_batch = torch.sum(log_q_z_x, dim=1)
        ELBO = log_p_x_z_per_batch + 0.5 * (log_p_z_per_batch - log_q_z_x_per_batch)

        return -torch.mean(ELBO)


def train(epoch, model, device, optimiser, pbar, print_interval=100):
    # set model in train mode
    model.train()
    epoch_loss = 0
    i = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        i += 1
        # move data to device of model
        data = data.to(device)
        optimiser.zero_grad()
        reconstructions, mu, logvar, z_samples = model(data)
        loss = model.loss_function(data.view(-1, reconstructions.shape[-1]), reconstructions, mu, logvar, z_samples)
        loss.backward()
        epoch_loss += loss.item()
        optimiser.step()
        if i % print_interval == 0:
            pbar.set_description(f"Epoch {epoch}: The current loss for batch {i} is: {loss.item():.5f}")

    return epoch_loss / i


def test(epoch, model, device):
    # set model to evaluation mode
    model.eval()
    test_loss = 0
    i = 0
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader):
            i += 1
            data = data.to(device)
            reconstructions, mu, logvar, z_samples = model(data)
            test_loss += model.loss_function(data.view(-1, reconstructions.shape[-1]), reconstructions, mu, logvar, z_samples).item()
            if batch_idx == 0:
                n_examples = min(data.size(0), 4)
                comparison = torch.cat([data[:n_examples], reconstructions.view(data.size(0), 1, 28, 28)[:n_examples]])
                save_image(comparison.cpu(), f"../results/vae/2d/reconstruction_{epoch}.png", nrow=n_examples)

    return test_loss / i


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from pathlib import Path
    import os
    seed = 100
    batch_size = 128
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True, **kwargs)

    model_kwargs = {'input_size': 784, 'enc_num_hidden_layers': 1, 'dec_num_hidden_layers': 1,
                    'enc_latent_size': 500, 'dec_latent_size': 500, 'latent_dim': 50}
    model = VAE(**model_kwargs)
    model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    # lets load the model
    target_path = os.path.join(Path.cwd().parent, "saved_models/vae/50dlatent")
    # model.load_state_dict(torch.load(target_path))

    # lets train our model
    total_epochs = 10
    pbar = trange(1, total_epochs + 1)
    train_losses = []
    test_losses = []
    for epoch in pbar:
        tr_loss = train(epoch, model, device, optimiser, pbar)
        train_losses.append(tr_loss)
        te_loss = test(epoch, model, device)
        test_losses.append(te_loss)

    # lets save the model
    torch.save(model.state_dict(),target_path)
    plt.plot(range(1, total_epochs + 1), train_losses, label='train loss')
    plt.plot(range(1, total_epochs + 1), test_losses, label='test loss')
    plt.legend()
    plt.show()

    # lets save one latent example
    example_batch, _ = next(iter(test_loader))
    example_image = example_batch[0]
    example_reconstructed, example_mu, example_logvar, z_samples = model(example_image)

    # form variational posterior object
    target_dist = dist.multivariate_normal.MultivariateNormal(loc=example_mu[0].detach(),
                                                              covariance_matrix=torch.eye(model_kwargs['latent_dim']) *
                                                                                torch.exp(example_logvar.detach())[0])

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
    encoder = Encoder(target_dist,
                      initial_seed,
                      coding_sampler,
                      selection_sampler,
                      auxiliary_posterior,
                      omega,
                      epsilon=epsilon,
                      beamwidth=beamwidth)

    z, idx = encoder.run_encoder()

    compressed_reconstruction = model.decode(z[0])
    comparison = torch.cat(
        [example_image[None], model.decode(target_dist.mean).view(1, 1, 28, 28),
         example_reconstructed.view(1, 1, 28, 28),
         model.decode(target_dist.sample((4,))).view(4, 1, 28, 28), compressed_reconstruction.view(1, 1, 28, 28)])
    save_image(comparison.cpu(), f"../results/vae/2d/compressed_reconstruction.png")

