import torch
import torch.distributions as D


def hmc(initialising_distribution, prior_z, likelihood_z, step_size, num_inner_steps, num_outer_steps=5000,
        num_parallel_runs=50):
    trajectory = torch.zeros([0])
    for i in range(num_outer_steps):
        current_z = initialising_distribution.sample((num_parallel_runs,))
        z = current_z
        current_m = torch.rand_like(z)
        m = current_m

        # half step for momentum
        m = m - step_size * grad_U(z, prior_z, likelihood_z) / 2

        # loop through for full steps of momentum and position
        for j in range(num_inner_steps):
            # make a step in z space
            z = z + step_size * m
            # if not last step do full gibbs step
            if j != num_inner_steps - 1:
                m = m - step_size * grad_U(z, prior_z, likelihood_z)

        # make half step for momentum at end
        m = m - step_size * grad_U(z, prior_z, likelihood_z) / 2

        # negate momentum for symmetric proposal
        m = -m

        # evaluate energies
        current_U = U(current_z, prior_z, likelihood_z)
        current_K = torch.sum(current_m ** 2, dim=-1) / 2
        proposed_U = U(z, prior_z, likelihood_z)
        proposed_K = torch.sum(m ** 2, dim=-1) / 2

        # metropolis acceptance criterion
        p_accept = torch.exp(current_U-proposed_U+current_K-proposed_K)[:, None].repeat(1, current_z.shape[-1])
        accept_mask = (p_accept > torch.rand(p_accept.shape)).int()
        # need to update our current_z with the accepted samples and keep rejected ones the same
        current_z = current_z * (1 - accept_mask) + z * accept_mask
        trajectory = torch.cat([trajectory, current_z[None]])

    return current_z, trajectory


def U(x: torch.Tensor, prior_z: D.distribution, likelihood_z: D.distribution):
    return -(likelihood_z.log_prob(x) + prior_z.log_prob(x))



def grad_U(x: torch.Tensor, prior_z: D.distribution, likelihood_z: D.distribution):
    x_for_grad = x.clone().requires_grad_(True)
    U_x = U(x_for_grad, prior_z, likelihood_z)
    return torch.autograd.grad(U_x, x_for_grad, grad_outputs=torch.ones_like(U_x))[0]


if __name__ == '__main__':
    target = D.MultivariateNormal(loc=torch.tensor([50.]), covariance_matrix=torch.eye(1))
    prior = D.MultivariateNormal(loc=torch.tensor([0.]), covariance_matrix=torch.eye(1))
    accepted_samples, traj = hmc(prior, prior, target, step_size=torch.tensor([1]),
                                 num_inner_steps=3, num_outer_steps=1,
                                 num_parallel_runs=100)