import torch
import torch.distributions as dist


class CodingSampler(dist.MultivariateNormal):
    def __init__(self,
                 problem_dimension,
                 n_auxiliary,
                 var=1,
                 sigma_vector=None,
                 optimise_index=None):

        self.problem_dimension = problem_dimension
        self.n_auxiliary = n_auxiliary

        # create auxiliary variable variances
        self.optimise_index = torch.tensor(optimise_index, requires_grad=False)
        self.fixed_vars = torch.ones((n_auxiliary,), requires_grad=False) / n_auxiliary
        if sigma_vector is not None:
            self.fixed_vars[:optimise_index] = sigma_vector[:optimise_index]
            remaining_var = var - torch.sum(sigma_vector[:optimise_index])
            N = n_auxiliary - sigma_vector[:optimise_index].detach().shape[0]
        else:
            remaining_var = var
            N = n_auxiliary
        self.fixed_vars[optimise_index:] = remaining_var * (torch.ones((N,), requires_grad=False) / N)
        self.learnt_var = torch.tensor(remaining_var / N, requires_grad=True)
        # self.auxiliary_vars = torch.flip(sigmas, dims=(0,))
        coding_mean = torch.zeros((problem_dimension,))
        coding_covar = torch.eye(problem_dimension) * var
        super(CodingSampler, self).__init__(loc=coding_mean,
                                            covariance_matrix=coding_covar)

    def auxiliary_coding_dist(self, index):
        auxiliary_coding_mean = torch.zeros((self.problem_dimension,))
        auxiliary_coding_covar = self.auxiliary_vars[index] * torch.eye(self.problem_dimension)
        return dist.MultivariateNormal(loc=auxiliary_coding_mean,
                                       covariance_matrix=auxiliary_coding_covar)

    @property
    def auxiliary_vars(self):
        x = self.fixed_vars.detach()
        x.index_put_((self.optimise_index.detach(),), self.learnt_var)
        return x


if __name__ == '__main__':
    coder = CodingSampler(var=1,
                          problem_dimension=2,
                          n_auxiliary=10)

    test_dist = coder.auxiliary_coding_dist(2, )
