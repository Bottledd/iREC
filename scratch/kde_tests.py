import torch
import torch.distributions as dist
from sklearn.neighbors import KernelDensity
import numpy as np


if __name__ == '__main__':
    # var_mean = torch.tensor([-1.2828, 1.6508, 1.4314, -0.7785, -0.1488, 0.2930, -0.1225, 2.6420,
    #                          0.4913, -0.6382, -0.5173, -1.5689, -1.2808, 1.4096, 1.3054, -0.5755,
    #                          -0.1463, 0.1514, 0.1904, 0.4206, -0.4569, 0.5137, 0.4990, -0.4522,
    #                          1.6059, -0.0308, 0.7416, 0.1244, 0.4371, 1.4866, -0.0216, 0.0246,
    #                          1.6946])
    # var_std = torch.tensor([0.3433, 0.1766, 0.3178, 0.0081, 0.6837, 0.4910, 0.5820, 0.0304, 0.0328,
    #                         0.0484, 0.0475, 0.0928, 0.3547, 0.3492, 0.3406, 0.5405, 1.2291, 0.9722,
    #                         0.5047, 0.5389, 0.6762, 0.6502, 0.8707, 0.8854, 0.0551, 0.3796, 0.7789,
    #                         0.8914, 0.0145, 0.0085, 0.0081, 0.0118, 0.0069])
    #
    # target = dist.MultivariateNormal(loc=var_mean, covariance_matrix=torch.diag(var_std ** 2))

    #
    #
    # emp_samples = target.sample((1000,))
    from models.BNNs.BNN_KDE import BNN_KDE

    import pickle as pkl
    x_data = pkl.load(open("../PickledStuff/x_data.pkl", "rb"))
    y_data = pkl.load(open("../PickledStuff/y_data.pkl", "rb"))
    xs = pkl.load(open("../PickledStuff/xs.pkl", "rb"))
    ys = pkl.load(open("../PickledStuff/ys.pkl", "rb"))
    emp_samples = pkl.load(open("../PickledStuff/emp_samples.pkl", "rb"))
    # instantiate and fit the KDE model
    bnn = BNN_KDE(emp_samples=emp_samples)
    fits = []
    bandwidths = np.logspace(np.log10(1e-10), np.log10(0.5), 1000)
    for b in bandwidths:
        bandwidth = b
        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        kde.fit(emp_samples.numpy())

        # fit a gmm model with same bandwidth
        KDE_var = bandwidth ** 2
        KDE_weights = dist.Categorical(torch.ones(emp_samples.shape[0]))
        KDE_components = dist.MultivariateNormal(loc=emp_samples,
                                                 covariance_matrix=KDE_var * torch.eye(emp_samples.shape[-1]))

        KDE_target = dist.MixtureSameFamily(KDE_weights, KDE_components)

        # print(f'KDE fit for bandwidth {b} is: {target.log_prob(torch.tensor(kde.sample((1000,)), dtype=torch.float)).mean()}')
        # print(f'GMM fit for bandwidth {b} is: {target.log_prob(KDE_target.sample((1000,))).mean()}')
        #fits.append(target.log_prob(KDE_target.sample((50,))).mean().numpy())
        bnn.log_kde_std = torch.nn.Parameter(torch.log(torch.tensor(KDE_var * 0.5)))
        fits.append(bnn.joint_log_prob(xs.view(-1, 1), ys.view(-1, 1), 100)[0].item())

    import matplotlib.pyplot as plt

    plt.plot(bandwidths, fits)
    plt.show()