"""
Sparse Gaussian process regression.
"""
import torch.nn as nn

from model import Model


class SGPR(Model):
    """
    Sparse Gaussian process regression.
    """

    def __init__(self, inducing, kernel, variance=None):
        """
        :param inducing: Inducing inputs.
        :type inducing: torch.tensor [m, d]
        """
        super(SGPR, self).__init__(kernel, variance)
        self.inducing = nn.Parameter(inducing)

    def forward(self, Xt):
        raise NotImplementedError

    def loss(self):
        raise NotImplementedError
