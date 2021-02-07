from gpytorch.distributions import base_distributions
import torch
import math
from time import process_time
from sampler_basic import SamplerBasic


class SamplerBinary(SamplerBasic):
    def __init__(self, dist, likelihood, target, maxn=10):
        super(SamplerBinary, self).__init__(dist, likelihood, target, maxn)

        mean, std = self.dist.mean, torch.sqrt(self.dist.variance)
        self.sample_dist = []
        self.sample_std = torch.zeros_like(std)
        self.global_logmax_likelihood = torch.zeros_like(std)
        start = process_time()

        log_max_likelihood = torch.max(self.likelihood.forward(mean + std).log_prob(self.target),
                                       self.likelihood.forward(mean - std).log_prob(self.target))
        for i in range(mean.size(0)):
            if log_max_likelihood[i] > math.log(0.5):
                self.sample_dist.append(base_distributions.Normal(loc=mean[i], scale=std[i]))
                self.sample_std[i] = std[i]
            else:
                # determine n
                bestn = 1
                for n in reversed(range(2, maxn + 1)):
                    region = math.sqrt(math.log(n) / (1 - 1 / n)) * std[i]
                    logm1 = max(self.logprob_i(i, mean[i] + region), self.logprob_i(i, mean[i] - region))
                    logm2 = -0.5 * torch.log(torch.tensor(float(n)))
                    if logm1 <= logm2:
                        bestn = n
                        break
                self.sample_std[i] = math.sqrt(bestn) * std[i]
                self.sample_dist.append(base_distributions.Normal(loc=mean[i], scale=math.sqrt(bestn) * std[i]))
        end = process_time()
        # print("Determine n takes {:.2f}s".format(end - start))
