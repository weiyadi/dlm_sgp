from gpytorch.likelihoods.likelihood import _OneDimensionalLikelihood
from gpytorch.distributions import base_distributions
import torch


class PoissonExp(_OneDimensionalLikelihood):
    def forward(self, function_samples, **kwargs):
        rates = torch.exp(function_samples)
        return base_distributions.Poisson(rate=rates)

    def log_marginal(self, observations, function_dist, *args, **kwargs):
        # prob_lambda = lambda function_samples: torch.exp(
        #     base_distributions.Poisson(rate=torch.exp(function_samples)).log_prob(observations))
        # prob = self.quadrature(prob_lambda, function_dist)
        num_samples = kwargs.pop('num_samples', 100)
        samples = function_dist.get_base_samples(
            torch.Size([num_samples]))
        samples.mul_(torch.sqrt(function_dist.variance)) \
            .add_(function_dist.mean)
        y_distribution = base_distributions.Poisson(rate=torch.exp(samples))

        return (torch.logsumexp(y_distribution.log_prob(observations), dim=0).sum()
                - observations.numel() \
                * torch.log(torch.tensor(float(self.sample_size))))

    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        mean, var = function_dist.mean, function_dist.variance
        return observations * mean - torch.exp(mean + 0.5 * var) - torch.lgamma(observations)

    def predictive_mean(self, function_dist):
        mean, var = function_dist.mean, function_dist.variance
        conditional_mean = torch.exp(mean + 0.5 * var)
        return conditional_mean


class PoissonLog1p(_OneDimensionalLikelihood):
    def forward(self, function_samples, *args, **kwargs):
        rates = torch.log1p(torch.exp(function_samples))
        return base_distributions.Poisson(rate=rates)

    def log_marginal(self, observations, function_dist, *args, **kwargs):
        prob_lambda = lambda function_samples: torch.exp(
            base_distributions.Poisson(rate=torch.log1p(torch.exp(function_samples))).log_prob(observations))
        prob = self.quadrature(prob_lambda, function_dist)
        return torch.log(prob)

    def predictive_mean(self, function_dist):
        # compute conditional mean
        expect_lambda = lambda function_samples: torch.log1p(torch.exp(function_samples))
        conditional_mean = self.quadrature(expect_lambda, function_dist)
        return conditional_mean
