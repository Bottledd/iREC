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

    def encode(self, x):
        x = F.relu(self.enc_input_layer(x))

        for f in self.enc_hidden_layers:
            x = F.relu(f(x))

        return self.enc_final_layer_mean(x), self.enc_final_layer_logvar(x)

    def reparam(self, mu, logvar):
        std = torch.exp(logvar * 0.5)
        eps = torch.rand_like(std)

        return mu + std * eps

    def decode(self, z):
        z = F.relu(self.dec_input_layer(z))

        for f in self.dec_hidden_layers:
            z = F.relu(f(z))

        # pass through sigmoid so pixel intensities between [0,1]
        return torch.sigmoid(self.dec_final_layer(z))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_size))
        z = self.reparam(mu, logvar)

        return self.decode(z), mu, logvar


def loss_function(x_reconstructed, x, mu, logvar):
    KL_TERM = 0.5 * torch.sum(1 + logvar - torch.pow(mu, 2) - torch.exp(logvar))

    reconstruction_error = F.binary_cross_entropy(x_reconstructed, x.view(-1, x_reconstructed.shape[-1]),
                                                  reduction='sum')

    return - KL_TERM + reconstruction_error

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
        reconstructions, mu, logvar = model(data)
        loss = loss_function(reconstructions, data, mu, logvar)
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
            reconstructions, mu, logvar = model(data)
            test_loss += loss_function(reconstructions, data, mu, logvar).item()
            if batch_idx == 0:
                n_examples = min(data.size(0), 4)
                comparison = torch.cat([data[:n_examples], reconstructions.view(data.size(0), 1, 28, 28)[:n_examples]])
                save_image(comparison.cpu(), f"../results/vae/2d/reconstruction_{epoch}.png", nrow=n_examples)

    return test_loss / i


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    seed = 100
    batch_size = 64
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

    model_kwargs = {'input_size': 784, 'enc_num_hidden_layers': 3, 'dec_num_hidden_layers': 3,
                    'enc_latent_size': 500, 'dec_latent_size': 500, 'latent_dim': 2}
    model = VAE(**model_kwargs)
    model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=3e-3)

    # # lets load the model
    # model.load_state_dict(torch.load("../saved_models/vae/2dimlatent"))

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
    torch.save(model.state_dict(), "../saved_models/vae/2dlatent")
    plt.plot(range(1, total_epochs + 1), train_losses, label='train loss')
    plt.plot(range(1, total_epochs + 1), test_losses, label='test loss')
    plt.legend()
    plt.show()

    # lets save one latent example
    example_batch, _ = next(iter(test_loader))
    example_image = example_batch[0]
    example_reconstructed, example_mu, example_logvar = model(example_image)

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
