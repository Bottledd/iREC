import matplotlib.pyplot as plt
import torch
import torch.distributions as D
from tqdm import trange

from models.BNNs.DeterministicNN import Deterministic_NN
from models.BNNs.BNN_for_HMC import BNN_for_HMC


def hmc(initialising_distribution, x, y, model, step_size, num_inner_steps, num_outer_steps=5000):
    trajectory = torch.zeros([0])
    pbar = trange(num_outer_steps)
    k = 0
    for i in pbar:
        if k > 5:
            k = 0
            if torch.mean(p_accepts[-5:]) > 0.8:
                step_size = step_size * 1.1
            else:
                step_size = step_size * 0.9
        k = k + 1
        if i == 0:
            current_z = initialising_distribution.sample()
            p_accepts = torch.zeros([0])

        z = current_z
        current_m = torch.rand_like(z)
        m = current_m

        # half step for momentum
        m = m - step_size * grad_U(x, y, z, model) / 2

        # loop through for full steps of momentum and position
        for j in range(num_inner_steps):
            # make a step in z space
            z = z + step_size * m
            # if not last step do full gibbs step
            if j != num_inner_steps - 1:
                m = m - step_size * grad_U(x, y, z, model)

        # make half step for momentum at end
        m = m - step_size * grad_U(x, y, z, model) / 2

        # negate momentum for symmetric proposal
        m = -m

        # evaluate energies
        with torch.no_grad():
            current_U = U(x, y, current_z, model)
            current_K = torch.sum(current_m ** 2, dim=-1) / 2
            proposed_U = U(x, y, z, model)
            proposed_K = torch.sum(m ** 2, dim=-1) / 2

        # metropolis acceptance criterion
        p_accept = torch.exp(current_U - proposed_U + current_K - proposed_K)
        pbar.set_description(f"The 5 rolling mean p_accept is {p_accepts[-5:].mean()}. Step size is: {step_size.data}")
        if p_accept > torch.rand(size=(1,)):
            current_z = z

        if p_accept != float('inf'):
            p_accept = torch.clamp_max(p_accept, 1)
            p_accepts = torch.cat([p_accepts, p_accept[None]])
        trajectory = torch.cat([trajectory, current_z[None]])

    return current_z, trajectory, p_accepts


def U(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, model: torch.nn.Module):
    with torch.no_grad():
        model.make_weights_from_sample(w)
    return - model.joint_log_prob(x, y)


def grad_U(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, model: torch.nn.Module):
    U_x = U(x, y, w, model)
    grad = torch.autograd.grad(U_x, model.parameters(), grad_outputs=torch.ones_like(U_x))
    losses = torch.zeros([0])
    for loss in grad:
        losses = torch.cat([losses, loss.flatten()])

    return losses


if __name__ == '__main__':
    # create toy dataset
    torch.manual_seed(20)
    x = torch.cat([torch.Tensor(75).uniform_(-5, -2).sort()[0].reshape(-1, 1),
                   torch.Tensor(50).uniform_(2, 5).sort()[0].reshape(-1, 1)])
    i = 30
    x_data = torch.cat([x[0:i - 15], x[i + 14:]])

    # generate some data
    alpha, beta, num_nodes = .01, 25., 5

    # generate some data
    data_generator_model = Deterministic_NN(alpha=alpha, beta=beta, num_nodes=num_nodes)
    sampled_weights = data_generator_model.sample_weights_from_prior()
    data_generator_model.make_weights_from_sample(sampled_weights)
    y_data = data_generator_model(x_data).detach() + (
                1 / data_generator_model.likelihood_beta ** 0.5) * torch.randn_like(
        data_generator_model(x_data).detach())

    prior = D.MultivariateNormal(loc=torch.zeros_like(sampled_weights),
                                 covariance_matrix=(1. / alpha) * torch.eye(sampled_weights.shape[-1]))
    model = BNN_for_HMC(alpha=alpha, beta=beta, num_nodes=num_nodes)
    empirical_samples = torch.zeros([0])
    torch.manual_seed(79)
    accepted_samples, traj, p_accepts = hmc(prior, x_data, y_data, model, step_size=torch.tensor([1e-2]),
                                            num_inner_steps=20, num_outer_steps=5000)

    print(f"The average acceptance prob is: {p_accepts[100:].mean()}")

    # plot accepted samples
    f, ax = plt.subplots(figsize=(8, 6))
    xs = torch.linspace(-20, 20, 100)
    true_ensemble_preds = torch.zeros([traj[100:].shape[0], xs.shape[0], 1])
    for i, compressed_sample in enumerate(traj[100:]):
        ensemble_model = Deterministic_NN(alpha=alpha, beta=beta, num_nodes=num_nodes)
        ensemble_model.make_weights_from_sample(compressed_sample)
        true_ensemble_preds[i] = ensemble_model(xs).detach()
        ax.plot(xs, true_ensemble_preds[i], c='gray', alpha=0.1)
    ax.plot(xs, data_generator_model(xs).detach(), 'k-', label='Targets wo/ noise')
    ax.scatter(x_data, y_data, label='Truth')
    f.tight_layout()
    f.show()
    true_ensemble_preds_mean = true_ensemble_preds.mean(0).flatten()
    true_ensemble_preds_stds = true_ensemble_preds.std(0).flatten()

    # get uncertainty bounds
    f, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x_data, y_data, label='Truth')
    ax.plot(xs, true_ensemble_preds_mean, 'r-', label='Predictive Mean')
    ax.plot(xs, data_generator_model(xs).detach(), 'k-', label='Targets wo/ noise')
    ax.fill_between(xs, true_ensemble_preds_mean - 1.96 * true_ensemble_preds_stds ** 0.5,
                    true_ensemble_preds_mean + 1.96 * true_ensemble_preds_stds ** 0.5,
                    color='gray', alpha=0.2, label='95% Error Bars')
    ax.legend()
    # ax.set_aspect('equal', adjustable='box')
    f.tight_layout()
    f.show()

    # for i, point in enumerate(traj.unique(dim=0)[-100:]):
    #     plt.scatter(point[0], point[-1])
    #     plt.annotate(i, (point[0], point[-1]), color='black')
    #
    # plt.show()

    weight_indices_to_track = [
        1,
        5,
        10,
        20,
        -5
        -10,
        -1
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharex=True)
    axes = axes.ravel()

    for ax, index in zip(axes, weight_indices_to_track):
        samp_path = traj[:, index]

        ax.plot(samp_path)
        ax.set_title(f"Weight {index}")

    fig.tight_layout()
    fig.show()