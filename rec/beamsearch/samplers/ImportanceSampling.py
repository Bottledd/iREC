import torch
import torch.distributions as dist


class ImportanceSampler:
    def __init__(self,
                 coding: dist.Distribution,
                 target: dist.Distribution,
                 seed: torch.float64,
                 num_samples: torch.float64,
                 is_final_sample=False,
                 coding_joint_history=None,
                 target_joint_history=None,
                 topk=None
                 ):
        self.seed = seed
        self.coding = coding
        self.target = target
        self.num_samples = num_samples
        self.is_final_sample = is_final_sample
        self.topk = topk

    def get_samples_from_coder(self):
        torch.manual_seed(self.seed)

        # sample from coding distribution
        samples = self.coding.sample((self.num_samples,))

        return samples

    def get_weightings(self, samples):
        # compute ratio of probs
        log_ratios = self.target.log_prob(samples) - self.coding.log_prob(samples)

        # put ratios through a softmax

        return torch.softmax(log_ratios, dim=0)

    def final_sample(self, samples, previous_samples):
        # add samples together
        z_samples = torch.sum(previous_samples, dim=0) + samples

        log_probs = self.target.log_prob(z_samples)

        return log_probs

    def choose_samples_to_transmit(self, samples, previous_samples=None):
        if self.is_final_sample:
            log_probs = self.final_sample(samples, previous_samples)

            # compute best sample
            _, idx_to_transmit = torch.topk(log_probs, k=self.topk)
            samples_to_transmit = samples[idx_to_transmit]
        else:
            normalised_weightings = self.get_weightings(samples)
            # TODO add way to sample without replacement so we can do beamsearch like this
            # create categorical distribution
            iw_categorical = dist.Categorical(probs=normalised_weightings)

            # sample from categorical
            idx_to_transmit = iw_categorical.sample()

            # sample corresponding to index
            samples_to_transmit = samples[idx_to_transmit]

        return idx_to_transmit, samples_to_transmit
