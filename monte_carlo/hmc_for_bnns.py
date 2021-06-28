import matplotlib.pyplot as plt
import torch
import torch.distributions as D
from tqdm import trange

from models.BNNs.DeterministicNN import Deterministic_NN


def hmc(initialising_distribution, x, y, model, step_size, num_inner_steps, num_outer_steps=5000):
    trajectory = torch.zeros([0])
    pbar = trange(num_outer_steps)
    k = 0
    for i in pbar:
        if k > 25:
            k = 0
            if torch.mean(p_accepts[-25:]) > 0.2:
                step_size = step_size * 1.1
            else:
                step_size = step_size * 0.9
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
            p_accepts = torch.cat([p_accepts, p_accept[None]])
        trajectory = torch.cat([trajectory, current_z[None]])
        k = k + 1
    return current_z, trajectory, p_accepts


def U(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, model: torch.nn.Module):
    model.make_weights_from_sample(w)
    prior_term = 0
    for weight in model.parameters():
        prior = D.Normal(loc=torch.zeros_like(weight.flatten()), scale=(1. / model.prior_alpha ** 0.5))
        prior_term = prior_term + prior.log_prob(weight.flatten()).sum()
    y_preds = model(x)
    likelihood = D.Normal(loc=y_preds, scale=(1. / model.likelihood_beta ** 0.5))
    likelihood_term = likelihood.log_prob(y).sum()

    return -(likelihood_term + prior_term)


def grad_U(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, model: torch.nn.Module):
    U_x = U(x, y, w, model)
    grad = torch.autograd.grad(U_x, model.parameters(), grad_outputs=torch.ones_like(U_x))
    losses = torch.zeros([0])
    for loss in grad:
        losses = torch.cat([losses, loss.flatten()])

    return losses


if __name__ == '__main__':
    torch.set_default_tensor_type(torch.DoubleTensor)
    # create toy dataset
    torch.manual_seed(10)
    x = torch.Tensor(40).uniform_(-10, 10).sort()[0]
    i = 20
    x = torch.cat([x[0:i - 10], x[i + 9:]])

    # standardise inputs
    x_star = (x - x.mean()) / x.std()
    # generate some data
    alpha, beta = 1., 25.

    # generate some data
    data_generator_model = Deterministic_NN(alpha=alpha, beta=beta)
    sampled_weights = data_generator_model.sample_weights_from_prior()[0]
    data_generator_model.make_weights_from_sample(sampled_weights)
    y = data_generator_model(x_star).detach() + (1 / data_generator_model.likelihood_beta ** 0.5) * torch.randn_like(
        data_generator_model(x_star).detach())

    xs = torch.linspace(-10, 10, 100)

    # standardise new inputs
    xs_star = (xs - x.mean()) / x.std()

    # standardise outputs
    y_star = (y - y.mean()) / y.std()

    prior = D.MultivariateNormal(loc=torch.zeros_like(sampled_weights),
                                 covariance_matrix=(1. / alpha ** 2) * torch.eye(sampled_weights.shape[-1]))
    model = Deterministic_NN(alpha=alpha, beta=beta)
    empirical_samples = torch.zeros([0])
    torch.manual_seed(0)
    accepted_samples, traj, p_accepts = hmc(prior, x_star, y_star, model, step_size=torch.tensor([5e-2]),
                                            num_inner_steps=10, num_outer_steps=2500)

    print(f"The average acceptance prob is: {p_accepts[100:].mean()}")

    # plot accepted samples
    xs = torch.linspace(-10, 10, 100)
    xs_star = (xs - x.mean()) / x.std()
    true_ensemble_preds = torch.zeros([traj[50:].shape[0], xs.shape[0], 1])
    for i, compressed_sample in enumerate(traj[50:]):
        ensemble_model = Deterministic_NN(alpha=alpha, beta=beta)
        ensemble_model.make_weights_from_sample(compressed_sample)
        true_ensemble_preds[i] = ensemble_model(xs_star).detach()

    # unstardardise samples
    true_ensemble_preds = (true_ensemble_preds * y.std()) + y.mean()

    true_ensemble_preds_mean = true_ensemble_preds.mean(0).flatten()
    true_ensemble_preds_stds = true_ensemble_preds.std(0).flatten()

    # get uncertainty bounds
    f, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, label='Truth')
    ax.plot(xs, true_ensemble_preds_mean, 'r-', label='Predictive Mean')
    ax.plot(xs, data_generator_model(xs_star).detach(), 'k-', label='Targets wo/ noise')
    ax.fill_between(xs, true_ensemble_preds_mean - 1.96 * true_ensemble_preds_stds ** 0.5,
                    true_ensemble_preds_mean + 1.96 * true_ensemble_preds_stds ** 0.5,
                    color='gray', alpha=0.2, label='95% Error Bars')
    ax.legend()
    # ax.set_aspect('equal', adjustable='box')
    f.tight_layout()
    f.show()

    for i, point in enumerate(traj.unique(dim=0)[-50:]):
        plt.scatter(point[0], point[-1])
        plt.annotate(i, (point[0], point[-1]), color='black')

    plt.show()