from vsgpr import VSGPR
from linalg import log_gaussian
import torch.nn.functional as F
import numpy as np
import math


class VSGPR_SVGP(VSGPR):
    """
    Add KL in loss function
    """

    def __init__(self, Xm, kernel, variance=None, beta=1.0, batch_size=None):
        """
        :param Xm: Inducing inputs.
        :type Xm: torch.tensor [m, d]
        """
        super(VSGPR_SVGP, self).__init__(Xm, kernel, variance)
        self.beta = beta
        self.batch_size = batch_size

    def loss(self, Xn, yn):
        if self.batch_size is not None and Xn.shape[0] > self.batch_size:
            # subsample data for minibatch.
            indices = np.random.choice(list(range(Xn.shape[0])), self.batch_size, replace=False)
            X_sample = Xn[indices]
            y_sample = yn[indices]
            mean, var = self.forward(X_sample, diag=True)
            noise_var = F.softplus(self.free_variance)
            res = ((y_sample - mean) ** 2 + var) / noise_var + noise_var.log() + math.log(2 * math.pi)
            res = res.mul(-0.5)
            return -res.sum() * Xn.shape[0] / self.batch_size + self.beta * self.kullback_leibler()
        else:
            mean, var = self.forward(Xn, diag=True)
            noise_var = F.softplus(self.free_variance)
            var += noise_var
            res = ((yn - mean) ** 2 + var) / noise_var + noise_var.log() + math.log(2 * math.pi)
            res = res.mul(-0.5)
            return -res.sum() + self.beta * self.kullback_leibler()
