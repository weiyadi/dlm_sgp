"""
Variational sparse Gaussian process regression.
"""
import torch
import torch.nn as nn

from model import Model
from linalg import logdet, stable_cholesky, triangular_solve, EPSILON


class VSGPR(Model):
    """
    Variational sparse Gaussian process regression(Whitened).
    """

    def __init__(self, Xm, kernel, variance=None, no_variance=False):
        """
        :param Xm: Inducing inputs.
        :type Xm: torch.tensor [m, d]
        """
        super(VSGPR, self).__init__(kernel, variance, no_variance)
        self.inducing = nn.Parameter(Xm)
        self.M = Xm.size(0)
        # Variational distribution q(fm | m, S = Ls @ Ls.T)
        self.m = nn.Parameter(torch.zeros(self.M, 1, device=self.device))
        self.L = nn.Parameter(torch.eye(self.M, device=self.device))

    def forward(self, Xt, diag=False):
        lower_mask = torch.ones(self.M, self.M, device=self.device).tril(0)
        L = self.L * lower_mask

        Kmm = self.kernel(self.inducing)
        Kmt = self.kernel(self.inducing, Xt)

        Lk = stable_cholesky(Kmm)
        invLk_Kmt = triangular_solve(Kmt, Lk)

        mean = invLk_Kmt.t() @ self.m

        if diag:
            Ktt_diag = self.kernel(Xt, diag=True)
            var = Ktt_diag + torch.sum((L.t() @ invLk_Kmt) ** 2, 0).view(-1, 1) \
                  - torch.sum(invLk_Kmt ** 2, 0).view(-1, 1)
            return mean, var.clamp(min=EPSILON ** 2)
        else:
            Ktt = self.kernel(Xt)
            cov = Ktt + invLk_Kmt.t() @ (L @ L.t() - torch.eye(self.M, device=self.device)) \
                  @ invLk_Kmt
            return mean, cov

    def loss(self, Xn, yn):
        raise NotImplementedError

    def kullback_leibler(self):
        lower_mask = torch.ones(self.M, self.M, device=self.device).tril(0)
        L = self.L * lower_mask
        return 0.5 * (torch.sum(L ** 2) + torch.sum(self.m ** 2) - logdet(L)
                      - self.M)
