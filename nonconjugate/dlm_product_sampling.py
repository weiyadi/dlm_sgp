from gpytorch.mlls._approximate_mll import _ApproximateMarginalLogLikelihood
from sampler_binary import SamplerBinary
from sampler_poisson import SamplerPoisson
from torch.autograd import Function
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import BernoulliLikelihood
import torch


class PsLoss(Function):
    @staticmethod
    def forward(ctx, mean, var, target, sample=1, maxn=None, likelihood=None):
        dist = MultivariateNormal(mean, torch.diag(var))
        if type(sample) == int:
            size = sample
            if isinstance(likelihood, BernoulliLikelihood):
                sampler = SamplerBinary(dist, likelihood, target, maxn=maxn)
                samples = sampler.sample(size)
            else:
                sampler = SamplerPoisson(dist, likelihood, target, maxn=maxn)
                samples = sampler.sample(size)
        else:
            samples = sample
        
        dlogq_dmu = (samples - mean) / var
        dlogq_dvar = 0.5 * ((samples - mean) ** 2 / var ** 2 - 1. / var)
        dl_dmu = dlogq_dmu.sum(dim=0) / samples.size(0)
        dl_dvar = dlogq_dvar.sum(dim=0) / samples.size(0)
        ctx.save_for_backward(dl_dmu, dl_dvar)
        return likelihood.log_marginal(target, dist).sum()

    @staticmethod
    def backward(ctx, grad_output):
        grad_mean = grad_var = None
        dl_dmu, dl_dvar = ctx.saved_tensors

        if ctx.needs_input_grad[0]:
            grad_mean = grad_output * dl_dmu
        if ctx.needs_input_grad[1]:
            grad_var = grad_output * dl_dvar
        return grad_mean, grad_var, None, None, None, None, None


class DLM_ProductSampling(_ApproximateMarginalLogLikelihood):
    '''
    Monte carlo version for dlm.
    '''

    def __init__(self, *args, **kwargs):
        sample_size = kwargs.pop('sample_size', 10)
        self.maxn = kwargs.pop('maxn', 10)
        super().__init__(*args, **kwargs)
        self.sample_size = sample_size

    def _log_likelihood_term(self, approximate_dist_f, target, **kwargs):
        if 'samples' in kwargs:
            return PsLoss.apply(approximate_dist_f.mean, approximate_dist_f.variance, target, kwargs.pop('samples'),
                                self.maxn, self.likelihood)
        else:
            return PsLoss.apply(approximate_dist_f.mean, approximate_dist_f.variance, target, self.sample_size,
                                self.maxn, self.likelihood)

    def forward(self, variational_dist_f, target, **kwargs):
        return super().forward(variational_dist_f, target, **kwargs)
