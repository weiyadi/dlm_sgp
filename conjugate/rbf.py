"""
Radial Basis Function (RBF) covariance module.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from linalg import inverse_softplus, stable_exp, stable_divide, pairwise_dots


class RBF(nn.Module):
    """
    Radial Basis Function (RBF) covariance/kernel.
    """

    def __init__(self, variance=None, lengthscale=None):
        super(RBF, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if variance is None:
            self.free_variance = nn.Parameter(
                inverse_softplus(torch.tensor(1.0)))
        else:
            self.free_variance = nn.Parameter(
                inverse_softplus(torch.tensor(variance)))
        if lengthscale is None:
            self.free_lengthscale = nn.Parameter(
                inverse_softplus(torch.tensor(1.0, device=self.device)))
        else:
            self.free_lengthscale = nn.Parameter(
                inverse_softplus(torch.tensor(lengthscale, device=self.device)))

    def forward(self, X1, X2=None, diag=False):
        """
        Compute covariance matrix.
        """
        variance = F.softplus(self.free_variance)
        if diag:
            return torch.ones(X1.size(0), 1, device=X1.device) * variance
        else:
            lengthscale = F.softplus(self.free_lengthscale)
            return variance * stable_exp(-0.5 * stable_divide(
                pairwise_dots(X1, X2), lengthscale ** 2))

    def variance(self):
        with torch.no_grad():
            var = F.softplus(self.free_variance)
        return var.item()

    def lengthscale(self):
        with torch.no_grad():
            ls = F.softplus(self.free_lengthscale)
        return ls.item()
