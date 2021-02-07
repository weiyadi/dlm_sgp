import torch
from gpytorch.mlls._approximate_mll import _ApproximateMarginalLogLikelihood


class MonteCarloELBO(_ApproximateMarginalLogLikelihood):
    '''
    Monte carlo version for variational lower bound.
    '''

    def __init__(self, *args, **kwargs):
        sample_size = kwargs.pop('sample_size', 10)
        super().__init__(*args, **kwargs)
        self.sample_size = sample_size

    def _log_likelihood_term(self, approximate_dist_f, target, **kwargs):
        samples = approximate_dist_f.get_base_samples(
            torch.Size([self.sample_size]))
        samples.mul_(torch.sqrt(approximate_dist_f.variance)) \
            .add_(approximate_dist_f.mean)
        y_distribution = self.likelihood.forward(samples)

        return y_distribution.log_prob(target).mean(dim=0).sum()

    def forward(self, variational_dist_f, target, **kwargs):
        return super().forward(variational_dist_f, target, **kwargs)


class MonteCarloDLM(_ApproximateMarginalLogLikelihood):
    '''
    Monte carlo version for dlm.
    '''

    def __init__(self, *args, **kwargs):
        sample_size = kwargs.pop('sample_size', 10)
        super().__init__(*args, **kwargs)
        self.sample_size = sample_size

    def _log_likelihood_term(self, approximate_dist_f, target, **kwargs):
        samples = approximate_dist_f.get_base_samples(
            torch.Size([self.sample_size]))
        samples.mul_(torch.sqrt(approximate_dist_f.variance)) \
            .add_(approximate_dist_f.mean)
        y_distribution = self.likelihood.forward(samples)

        return (torch.logsumexp(y_distribution.log_prob(target), dim=0).sum()
                - target.numel() \
                * torch.log(torch.tensor(float(self.sample_size))))

    def forward(self, variational_dist_f, target, **kwargs):
        return super().forward(variational_dist_f, target, **kwargs)


class MonteCarloDLMJitter(_ApproximateMarginalLogLikelihood):
    '''
    Monte carlo version for dlm.
    '''

    def __init__(self, *args, **kwargs):
        sample_size = kwargs.pop('sample_size', 10)
        jitter = kwargs.pop('jitter', 1e-4)
        super().__init__(*args, **kwargs)
        self.sample_size = sample_size
        self.jitter = jitter

    def _log_likelihood_term(self, approximate_dist_f, target, **kwargs):
        samples = approximate_dist_f.get_base_samples(
            torch.Size([self.sample_size]))
        samples.mul_(torch.sqrt(approximate_dist_f.variance)) \
            .add_(approximate_dist_f.mean)
        y_distribution = self.likelihood.forward(samples)

        prob = torch.sum(torch.exp(y_distribution.log_prob(target)), dim=0) / self.sample_size
        return torch.log(prob + self.jitter).sum()

    def forward(self, variational_dist_f, target, **kwargs):
        return super().forward(variational_dist_f, target, **kwargs)
