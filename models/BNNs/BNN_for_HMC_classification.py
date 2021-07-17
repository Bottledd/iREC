import torch
import torch.distributions as D
from torch import nn
from torch.nn import functional as F


class BNN_for_HMC(nn.Module):
    def __init__(self, input_size=4, num_nodes=2, output_size=3, alpha=1.):
        super(BNN_for_HMC, self).__init__()
        self.prior_alpha = alpha
        self.weight_prior = D.Normal(loc=0., scale=1. / alpha ** 0.5)
        self.activation = nn.Tanh()
        self.input_size = input_size
        self.num_nodes = num_nodes
        self.output_size = output_size
    
    def make_weights_from_sample(self, weight_samples):
        self.weight_sample = weight_samples
        
        idx = 0
        self.layer_1_w = weight_samples[idx:idx + self.num_nodes * self.input_size].reshape(self.num_nodes, self.input_size)
        idx += self.num_nodes * self.input_size
        self.layer_1_b = weight_samples[idx:idx + self.num_nodes]
        idx += self.num_nodes
        self.layer_2_w = weight_samples[idx:idx + self.num_nodes * self.num_nodes].reshape(self.num_nodes,
                                                                                         self.num_nodes)
        idx += self.num_nodes * self.num_nodes
        self.layer_2_b = weight_samples[idx:idx + self.num_nodes]
        idx += self.num_nodes
        self.layer_3_w = weight_samples[idx:idx + self.num_nodes * self.output_size].reshape(self.output_size,
                                                                             self.num_nodes)
        idx += self.num_nodes * self.output_size
        self.layer_3_b = weight_samples[idx:idx + self.output_size]
        
    def predict(self, x_data):
        # get shapes
        batch_x_dim = x_data.shape[0]

        # compute activations for layer one
        x_data = torch.einsum("ij, kj -> ki", self.layer_1_w, x_data) + self.layer_1_b[None].repeat(batch_x_dim, 1)
        x_data = self.activation(x_data)

        # compute activations for layer 2
        x_data = torch.einsum("ij, kj -> ki", self.layer_2_w, x_data) + self.layer_2_b[None].repeat(batch_x_dim, 1)
        x_data = self.activation(x_data)
        
        # compute final output
        x_data = torch.einsum("ij, ki -> ki", self.layer_3_w, x_data) + self.layer_3_b[None].repeat(batch_x_dim, 1)

        return x_data

    def data_likelihood(self, y_preds, y_data):
        categorical = D.Categorical(logits=y_preds)
        return categorical.log_prob(y_data).sum()
    
    def weight_prior_lp(self, weight_samples):
        return self.weight_prior.log_prob(weight_samples).sum()
    
    def joint_log_prob(self, x, y):
        return self.data_likelihood(self.predict(x), y) + self.weight_prior_lp(self.weight_sample)