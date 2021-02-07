from gpytorch.distributions import base_distributions
import torch
from time import process_time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class SamplerBasic:
    def __init__(self, dist, likelihood, target, maxn=10):
        self.dist = dist
        self.likelihood = likelihood
        self.target = target
        self.maxn = maxn

        self.sample_std = None
        self.sample_dist = None

    def logprob_i(self, i, f):
        return self.likelihood.forward(f).log_prob(self.target[i])

    def per_sample(self):
        mean, var = self.dist.mean, self.dist.variance
        sample_dist = base_distributions.Normal(loc=mean, scale=self.sample_std)
        candidates = sample_dist.sample(torch.Size([2]))
        l_prob = self.likelihood.forward(candidates).log_prob(self.target)
        dist = base_distributions.Normal(loc=mean, scale=torch.sqrt(var))
        valid = (self.global_logmax_likelihood + torch.log(torch.rand_like(candidates)) + sample_dist.log_prob(
            candidates) <= l_prob + dist.log_prob(candidates))
        sample = torch.zeros_like(mean)
        for i in range(mean.size(0)):
            if valid[0][i]:
                sample[i] = candidates[0][i]
            elif valid[1][i]:
                sample[i] = candidates[1][i]
            else:
                dist_i = base_distributions.Normal(loc=mean[i], scale=torch.sqrt(var[i]))
                succ = False
                while not succ:
                    sample_i = self.sample_dist[i].sample(torch.Size([1]))
                    l_prob = self.likelihood.forward(sample_i).log_prob(self.target[i])
                    if self.sample_dist[i].log_prob(
                            sample_i) + self.global_logmax_likelihood[i] < l_prob + dist_i.log_prob(sample_i):
                        print("Not passing sanity check, ", self.sample_dist[i].log_prob(sample_i),
                              self.global_logmax_likelihood[i], l_prob, dist_i.log_prob(sample_i))
                    if torch.log(torch.rand(1)) + self.sample_dist[i].log_prob(sample_i) + \
                            self.global_logmax_likelihood[i] <= l_prob + dist_i.log_prob(sample_i):
                        # print("success")
                        sample[i] = sample_i
                        succ = True
        return sample

    def sample(self, size):
        tensors = []
        start = process_time()
        for i in range(size):
            # print("sample successful, ", i)
            tensors.append(self.per_sample())
        end = process_time()
        print("Sampling {} samples takes {:.2f}s".format(size, end - start))
        return torch.stack(tensors, dim=0)
