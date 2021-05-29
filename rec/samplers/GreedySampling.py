import torch
import torch.distributions as dist
import numpy as np

class GreedySampler:
    def __init__(self,
                 coding: dist.Distribution,
                 target: dist.Distribution,
                 seed: torch.float64,
                 num_samples: torch.float64,
                 coding_joint_history,
                 target_joint_history,
                 use_ratio=True,
                 is_final_sample=False):

        self.seed = seed
        self.coding = coding
        self.target = target
        self.num_samples = num_samples
        self.use_ratio = use_ratio
        self.coding_joint_history = coding_joint_history
        self.target_joint_history = target_joint_history
        self.is_final_sample = is_final_sample

    def get_samples_from_coder(self):
        torch.manual_seed(self.seed)
        print(torch.random.get_rng_state())
        # sample from coding distribution
        samples = self.coding.sample((self.num_samples,))

        # variance = self.coding.covariance_matrix.numpy()
        # samples_np = np.random.multivariate_normal(mean=np.zeros(2,), cov=variance, size=(self.num_samples,))
        # samples = torch.from_numpy(samples_np)
        return samples

    def final_sample(self, samples, previous_samples):
        # add samples together
        z_samples = torch.sum(previous_samples, dim=0) + samples

        log_probs = self.target.log_prob(z_samples)

        return log_probs

    def choose_samples_to_transmit(self, samples, previous_samples=None, topk=1):
        if self.is_final_sample:
            log_probs = self.final_sample(samples, previous_samples)

            # compute best sample
            _, top_indices = torch.topk(log_probs, k=topk)
            samples_to_transmit = samples[top_indices]
        else:
            # compute new joint log probs
            target_joint_log_prob = self.target_joint_history + self.target.log_prob(samples)
            coding_joint_log_prob = self.coding_joint_history + self.coding.log_prob(samples)

            # compute ratio of joint probs
            if self.use_ratio:
                log_ratios = target_joint_log_prob - coding_joint_log_prob
                # take top index
                top_ratios, top_indices = torch.topk(log_ratios, k=topk)

            else:
                top_target_log_probs, top_indices = torch.topk(target_joint_log_prob, k=topk)

            # sample corresponding to index
            samples_to_transmit = samples[top_indices]

        return top_indices, samples_to_transmit
