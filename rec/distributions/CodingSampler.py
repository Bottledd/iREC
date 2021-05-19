import torch
import torch.distributions as dist

class CodingSampler(dist.MultivariateNormal):
    def __init__(self,
                 problem_dimension,
                 n_auxiliary,
                 var=1,
                 use_power_rule=True,
                 power_rule_exponent=0.79):

        self.problem_dimension = problem_dimension
        self.n_auxiliary = n_auxiliary

        # create auxiliary variable variances
        # if using power rule
        if use_power_rule:
            sigmas = torch.zeros((n_auxiliary,))
            sigma_idxs = torch.arange(1, n_auxiliary + 1)
            for i, idx in enumerate(sigma_idxs):
                sigmas[i] = (var - torch.sum(sigmas[:i])) * (n_auxiliary + 1 - idx) ** (-1 * power_rule_exponent)
        else:
            raise Exception("Not Implemented yet")

        self.auxiliary_vars = sigmas
        coding_mean = torch.zeros((problem_dimension,))
        coding_covar = torch.eye(problem_dimension) * var
        super(CodingSampler, self).__init__(loc=coding_mean,
                                            covariance_matrix=coding_covar)

    def auxiliary_coding_dist(self, index):
        auxiliary_coding_mean = torch.zeros((self.problem_dimension,))
        auxiliary_coding_covar = self.auxiliary_vars[index] * torch.eye(self.problem_dimension)
        return dist.MultivariateNormal(loc=auxiliary_coding_mean,
                                       covariance_matrix=auxiliary_coding_covar)


if __name__ == '__main__':
    coder = CodingSampler(var=1,
                          problem_dimension=2,
                          n_auxiliary=10)

    test_dist = coder.auxiliary_coding_dist(2, )
