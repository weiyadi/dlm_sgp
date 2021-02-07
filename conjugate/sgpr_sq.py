"""
Square loss DLM for sparse Gaussian process.
"""
import torch
from model import Model
from linalg import stable_cholesky, triangular_solve
import math


class SGPR_DLMKL(Model):
    def __init__(self, Xm, Xn, yn, kernel, beta=1.0):
        """
        :param Xm: Inducing inputs.
        :type Xm: torch.tensor [m, d]
        """
        super(SGPR_DLMKL, self).__init__(kernel, no_variance=True)
        self.inducing = torch.nn.Parameter(Xm)
        self.Xn = Xn
        self.yn = yn
        self.beta = beta

    def forward(self, Xt, diag=False):
        factor = math.sqrt(2. / self.beta)
        Kmm = self.kernel(self.inducing)
        Kmn = self.kernel(self.inducing, self.Xn)
        Ktm = self.kernel(Xt, self.inducing)
        L = stable_cholesky(Kmm)
        A = factor * triangular_solve(Kmn, L)
        Ay = A @ self.yn

        Lmid = stable_cholesky(torch.eye(self.inducing.size(0), device=self.device) + A @ A.t())
        B = triangular_solve(Ay, Lmid)
        Li_Ktm = factor * triangular_solve(Ktm.t(), L)
        Bt = triangular_solve(Li_Ktm, Lmid)
        mean = Bt.t() @ B

        if diag:
            return mean, torch.ones_like(mean, device=self.device)
        else:
            return mean, torch.eye(mean.size(0), device=self.device)

    def loss(self):
        factor = math.sqrt(2. / self.beta)
        Kmm = self.kernel(self.inducing)
        Kmn = self.kernel(self.inducing, self.Xn)
        L = stable_cholesky(Kmm)
        A = factor * triangular_solve(Kmn, L)
        Ay = A @ self.yn

        Lmid = stable_cholesky(torch.eye(self.inducing.size(0), device=self.device) + A @ A.t())
        B = triangular_solve(Ay, Lmid)

        return torch.sum(self.yn ** 2) - torch.sum(B ** 2)
