"""
Base class for all Gaussian processes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from linalg import inverse_softplus


class Model(nn.Module):
    """
    Inherited class should implement forward and loss methods.
    Parameters are attached to a model through nn.Parameter.
    We use softplus transformation to guarantee positivity.
    """

    def __init__(self, kernel, variance=None, no_variance=False):
        super(Model, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not no_variance:
            if variance is None:
                self.free_variance = nn.Parameter(
                    inverse_softplus(torch.tensor(1.0, device=self.device)))
            else:
                self.free_variance = nn.Parameter(
                    inverse_softplus(torch.tensor(variance, device=self.device)))
        else:
            self.free_variance = None
        self.kernel = kernel

    def forward(self, Xt):
        """
        forward = prediction
        """
        raise NotImplementedError

    def loss(self):
        """
        Negative log marginal likelihood or negative evidence lower bound.
        """
        raise NotImplementedError

    def variance(self):
        if self.free_variance is None:
            return None
        with torch.no_grad():
            var = F.softplus(self.free_variance)
        return var.item()
