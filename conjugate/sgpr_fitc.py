"""
Fully Independent Training Conditional(FITC) sparse Gaussian process. See Snelson and Ghahraman, 2006.
"""
import torch
from sgpr import SGPR
from linalg import stable_cholesky, triangular_solve, stable_divide, LOG_2PI, stable_sqrt
from torch.functional import F


class SGPR_FITC(SGPR):
    def __init__(self, Xm, Xn, yn, kernel, variance=None):
        """
        :param Xm: Inducing inputs.
        :type Xm: torch.tensor [m, d]
        """
        super(SGPR_FITC, self).__init__(Xm, kernel, variance)
        self.Xn = Xn
        self.yn = yn

    def forward(self, Xt, diag=False):
        Kmm = self.kernel(self.inducing)
        Kmn = self.kernel(self.inducing, self.Xn)
        Ktm = self.kernel(Xt, self.inducing)
        Knn = self.kernel(self.Xn, diag=True)
        variance = F.softplus(self.free_variance)
        Lm = stable_cholesky(Kmm)
        LiUT = triangular_solve(Kmn, Lm)
        sigma_star = Knn.squeeze() + variance - torch.sum(LiUT ** 2, 0)
        sigma_star_sqrt_inv = stable_sqrt(stable_divide(1., sigma_star))

        Lmi_Kmn = LiUT
        sigma_Knm_Lmi = sigma_star_sqrt_inv.reshape(-1, 1) * Lmi_Kmn.t()
        woodbury_chol = stable_cholesky(
            torch.eye(self.inducing.size(0), device=Xt.device) + sigma_Knm_Lmi.t() @ sigma_Knm_Lmi)

        Lmi_Kmt = triangular_solve(Ktm.t(), Lm)
        left = triangular_solve(Lmi_Kmt, woodbury_chol)
        tmp = sigma_Knm_Lmi.t() @ (sigma_star_sqrt_inv.unsqueeze(-1) * self.yn)
        right = triangular_solve(tmp, woodbury_chol)
        mean = left.t() @ right

        if diag:
            # do sth
            Ktt_diag = self.kernel(Xt, diag=True)
            tmp = triangular_solve(Lmi_Kmt, woodbury_chol)
            var = Ktt_diag - torch.sum(Lmi_Kmt ** 2, dim=0).unsqueeze(-1) + torch.sum(tmp ** 2, dim=0).unsqueeze(-1)
            return mean, var
        else:
            Ktt = self.kernel(Xt, Xt)
            tmp = triangular_solve(Lmi_Kmt, woodbury_chol)
            cov = Ktt - Lmi_Kmt.t() @ Lmi_Kmt + tmp.t() @ tmp
            return mean, cov

    def loss(self):
        # directly copy GPy implementation
        num_inducing = self.inducing.size(0)
        num_data = self.yn.size(0)

        variance = F.softplus(self.free_variance)
        Kmm = self.kernel(self.inducing)
        Knn = self.kernel(self.Xn, diag=True)
        Knm = self.kernel(self.Xn, self.inducing)
        U = Knm

        Lm = stable_cholesky(Kmm)
        LiUT = triangular_solve(U.t(), Lm)

        sigma_star = Knn.squeeze() + variance - torch.sum(LiUT ** 2, 0)
        beta = stable_divide(1., sigma_star)

        tmp = LiUT * torch.sqrt(beta)
        A = tmp @ tmp.t() + torch.eye(num_inducing, device=self.device)
        LA = stable_cholesky(A)

        URiy = (U.t() * beta) @ self.yn
        tmp = triangular_solve(URiy, Lm)
        b = triangular_solve(tmp, LA)

        loss = 0.5 * num_data * LOG_2PI + torch.sum(torch.log(torch.diag(LA))) - 0.5 * torch.sum(
            torch.log(beta)) + 0.5 * torch.sum((self.yn.t() * torch.sqrt(beta)) ** 2) - 0.5 * torch.sum(b ** 2)
        return loss
