import torch
import torch.distributions as dist


class GreedySampler:
    def __init__(self,
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

        log_probs = self.target.log_prob(z_samples)

        return log_probs

    def choose_samples_to_transmit(self, samples, previous_samples=None):
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
            mask = torch.ones
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
            # number of aux samples
            n_auxiliary = samples.shape[0] // self.topk
            # tile the joint histories
            tiled_coding_joint_history = torch.tile(self.coding_joint_history, (n_auxiliary,))
            tiled_target_joint_history = torch.tile(self.target_joint_history, (n_auxiliary,))

            # compute new joint log probs
            target_joint_log_prob = tiled_target_joint_history + self.target.log_prob(samples)
            coding_joint_log_prob = tiled_coding_joint_history + self.coding.log_prob(samples)

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
