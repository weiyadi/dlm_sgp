"""
Direct loss minimization, but use Monte Carlo samples to estimate log E term.
"""
from vsgpr import VSGPR
from linalg import stable_sqrt, LOG_2PI, stable_log
import torch.nn.functional as F
import numpy as np
import torch
import math


class VSGPR_DLM_SAMPLE(VSGPR):
    def __init__(self, Xm, kernel, variance=None, beta=1.0, batch_size=None, sample_size=10):
        """
        :param Xm: Inducing inputs.
        :type Xm: torch.tensor [m, d]
        """
        super(VSGPR_DLM_SAMPLE, self).__init__(Xm, kernel, variance)
        self.beta = beta
        self.batch_size = batch_size
        self.sample_size = sample_size

    def loss(self, Xn, yn):
        if self.batch_size is not None and Xn.shape[0] > self.batch_size:
            # subsample data for minibatch.
            indices = np.random.choice(list(range(Xn.shape[0])), self.batch_size, replace=False)
            X_sample = Xn[indices]
            y_sample = yn[indices]
            mean, var = self.forward(X_sample, diag=True)  # mean and variance of q(f)
            eps = torch.randn(self.sample_size, self.batch_size, device=Xn.device)
            fs = mean.squeeze() + stable_sqrt(var.squeeze()) * eps
            noise_var = F.softplus(self.free_variance)
            # average = -log_gaussian(y_sample, fs, noise_var).sum(dim=0) / self.sample_size
            log_probs = -0.5 * (LOG_2PI + stable_log(noise_var) + (y_sample.squeeze() - fs) ** 2 / noise_var)
            result = torch.logsumexp(log_probs, dim=0) - math.log(self.sample_size)
            return -result.sum() * Xn.shape[0] / self.batch_size + self.beta * self.kullback_leibler()
        else:
            mean, var = self.forward(Xn, diag=True)
            eps = torch.randn(self.sample_size, Xn.shape[0], device=Xn.device)
            fs = mean.squeeze() + stable_sqrt(var.squeeze()) * eps
            noise_var = F.softplus(self.free_variance)
            log_probs = -0.5 * (LOG_2PI + stable_log(noise_var) + (yn.squeeze() - fs) ** 2 / noise_var)
            result = torch.logsumexp(log_probs, dim=0) - math.log(self.sample_size)
            return -result.sum() + self.beta * self.kullback_leibler()
