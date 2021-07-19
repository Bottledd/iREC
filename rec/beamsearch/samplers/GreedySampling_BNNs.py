import torch
import torch.distributions as dist
from models.BNNs.BNN_for_HMC import BNN_for_HMC


class GreedySampler:
    def __init__(self,
                 model: BNN_for_HMC,
                 x_data: torch.Tensor,
                 y_data: torch.Tensor,
                 coding: dist.Distribution,
                 target: dist.Distribution,
                 seed: torch.float64,
                 num_samples: torch.float64,
                 coding_joint_history,
                 target_joint_history,
                 use_ratio=True,
                 is_final_sample=False,
                 is_first_index=False,
                 topk=None):

        self.model = model
        self.model.eval()
        self.x_data = x_data
        self.y_data = y_data
        self.seed = seed
        self.coding = coding
        self.target = target
        self.num_samples = num_samples
        self.use_ratio = use_ratio
        self.coding_joint_history = coding_joint_history
        self.target_joint_history = target_joint_history
        self.is_final_sample = is_final_sample
        self.is_first_index = is_first_index
        self.topk = topk

        assert (self.is_first_index != True) or (self.is_final_sample != True), "Can't be both the first and final index!"

    def get_samples_from_coder(self):
        torch.manual_seed(self.seed)

        # sample from coding distribution
        samples = self.coding.sample((self.num_samples,))

        return samples

    def final_sample(self, samples, previous_samples):
        # tile beams to match n_auxiliary samples
        n_auxiliary = samples.shape[0] // self.topk
        tiled_previous_samples = torch.tile(previous_samples, (n_auxiliary, 1, 1))

        # add samples together
        z_samples = torch.sum(tiled_previous_samples, dim=1) + samples

        losses = torch.empty([0])
        for z_sample in z_samples:
            # make a model
            self.model.make_weights_from_sample(z_sample)

            loss = self.model.joint_log_prob(self.x_data, self.y_data)
            losses = torch.cat([losses, loss[None]])
        return losses

    def choose_samples_to_transmit(self, samples, n_samples_per_aux=None, previous_samples=None):
        if self.is_final_sample:
            log_probs = self.final_sample(samples, previous_samples)

            # compute best sample
            _, top_indices = torch.topk(log_probs, k=self.topk)
            samples_to_transmit = samples[top_indices]

        elif self.is_first_index:
            # compute new joint log probs
            target_joint_log_prob = self.target_joint_history + self.target.log_prob(samples)
            coding_joint_log_prob = self.coding_joint_history + self.coding.log_prob(samples)

            # need to use a mask if beamwidth is bigger than number of samples

            # compute ratio of joint probs
            if self.use_ratio:
                log_ratios = target_joint_log_prob - coding_joint_log_prob
                # take top index
                top_ratios, top_indices = torch.topk(log_ratios, k=self.topk)
            else:
                top_target_log_probs, top_indices = torch.topk(target_joint_log_prob, k=self.topk)

            # sample corresponding to index
            samples_to_transmit = samples[top_indices]

        else:
            # tile the joint histories
            tiled_coding_joint_history = torch.tile(self.coding_joint_history, (n_samples_per_aux,))
            tiled_target_joint_history = torch.tile(self.target_joint_history, (n_samples_per_aux,))

            # compute new joint log probs
            target_joint_log_prob = tiled_target_joint_history + self.target.log_prob(samples)
            coding_joint_log_prob = tiled_coding_joint_history + self.coding.log_prob(samples)

            # mask the 0's since beamwidth may be larger than
            # compute ratio of joint probs
            if self.use_ratio:
                log_ratios = target_joint_log_prob - coding_joint_log_prob
                # take top index
                top_ratios, top_indices = torch.topk(log_ratios, k=self.topk)

            else:
                top_target_log_probs, top_indices = torch.topk(target_joint_log_prob, k=self.topk)

            # sample corresponding to index
            samples_to_transmit = samples[top_indices]

        return top_indices, samples_to_transmit
