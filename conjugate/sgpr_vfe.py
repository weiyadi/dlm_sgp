"""
Variational free energy sparse Gaussian process. See Titsias, 2009.
"""

import torch
from sgpr import SGPR
from linalg import stable_cholesky, stable_sqrt, triangular_solve, logdet, LOG_2PI, \
    stable_log
from torch.functional import F


class SGPR_VFE(SGPR):
    def __init__(self, Xm, Xn, yn, kernel, variance=None):
        """
        :param Xm: Inducing inputs.
        :type Xm: torch.tensor [m, d]
        """
        super(SGPR_VFE, self).__init__(Xm, kernel, variance)
        self.Xn = Xn
        self.yn = yn

    def forward(self, Xt, diag=False):
        Kmm = self.kernel(self.inducing)
        Kmn = self.kernel(self.inducing, self.Xn)
        variance = F.softplus(self.free_variance)
        sq_var_i = stable_sqrt(1. / variance)

        L = stable_cholesky(Kmm)
        A = sq_var_i * triangular_solve(Kmn, L)
        B = torch.eye(self.inducing.size(0), device=self.inducing.device) + A @ A.t()
        Lb = stable_cholesky(B)
        c = sq_var_i * triangular_solve(A @ self.yn, Lb)

        Ktm = self.kernel(Xt, self.inducing)
        Li_Kmt = triangular_solve(Ktm.t(), L)
        Lbi_Li_Kmt = triangular_solve(Li_Kmt, Lb)
        mean = Lbi_Li_Kmt.t() @ c

        if diag:
            # do sth
            Ktt_diag = self.kernel(Xt, diag=True)
            var = Ktt_diag - torch.sum(Li_Kmt ** 2, 0).unsqueeze(-1) + torch.sum(Lbi_Li_Kmt ** 2, 0).unsqueeze(-1)
            return mean, var
        else:
            Ktt = self.kernel(Xt, Xt)
            cov = Ktt - Li_Kmt.t() @ Li_Kmt + Lbi_Li_Kmt.t() @ Lbi_Li_Kmt
            return mean, cov

    def loss(self):
        Kmm = self.kernel(self.inducing)
        Kmn = self.kernel(self.inducing, self.Xn)
        Knn = self.kernel(self.Xn, diag=True)
        variance = F.softplus(self.free_variance)
        sq_var_i = torch.sqrt(1. / variance)

        L = stable_cholesky(Kmm)
        A = sq_var_i * triangular_solve(Kmn, L)
        B = torch.eye(self.inducing.size(0), device=self.inducing.device) + A @ A.t()
        Lb = stable_cholesky(B)
        c = sq_var_i * triangular_solve(A @ self.yn, Lb)
        double_loss = self.Xn.size(0) * LOG_2PI + logdet(Lb) + self.Xn.size(0) * stable_log(variance) + torch.sum(
            (sq_var_i * self.yn) ** 2) - torch.sum(c ** 2) + 1. / variance * torch.sum(Knn) - torch.sum(A ** 2)
        return 0.5 * double_loss
