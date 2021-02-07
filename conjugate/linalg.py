"""
Linear algebra
"""
import math
import torch

EPSILON = 1e-4
LOG_2PI = math.log(2 * math.pi)


def diag_add(matrix, value):
    """Add value to the diagonal of matrix."""
    matrix.view(-1)[::matrix.size(0) + 1] += value
    return matrix


def stable_cholesky(matrix, max_tries=3):
    """Add small positive values (jitter) along the diagonal and then perform
    cholesky decomposition.
    :matrix: Matrix to be decomposed. (contiguous, square)
    :returns: Lower cholesky factor
    """
    num_tries = 0
    jitter = EPSILON
    while num_tries < max_tries:
        try:
            matrix = diag_add(matrix, jitter)
            return matrix.cholesky()
        except:
            jitter *= 10
            num_tries += 1
            print("Cholesky decomposition is not stable! Now jitter={}".format(
                jitter))
    raise RuntimeError("cholesky fails")


def stable_sqrt(tensor):
    """ Add small positive values before taking sqrt.  """
    return torch.sqrt(tensor + EPSILON ** 2)


def stable_log(tensor):
    """ Add small positive values before taking log. """
    return torch.log(tensor + EPSILON ** 2)


def stable_exp(tensor):
    """ Add small positive values before taking exp. """
    return torch.exp(tensor + EPSILON ** 2)


def stable_divide(numerator, denominator):
    """ Add small positive values to denominator before division """
    return numerator / (denominator + EPSILON ** 2)


def inverse_cholesky(L):
    """Compute inverse matrix of a lower Cholesky factor.
    :L: Lower Cholesky matrix.
    :returns: Inverse of L.
    """
    return torch.triangular_solve(torch.eye(L.size(0)), L, upper=False)[0]


def cholesky_solve(b, L):
    """Solve a symmetric positive-definite linear equation system.
    """
    return torch.cholesky_solve(b, L, upper=False)


def triangular_solve(b, L):
    """Solve a symmetric positive-definite linear equation system.
    Lx = b -> x = L⁻¹b
    """
    return torch.triangular_solve(b, L, upper=False)[0]


def logdet(L):
    """Compute log determinant of a symmetric positive-definite matrix
    M = L L.T given its lower Cholesky factor L.
    :L: Lower Cholesky factor
    :return: log|M|
    """
    return (L.diag() ** 2).log().sum()


def kl_gaussian(Lq, mq, Lp, mp=None):
    """Compute KL[q(fm) || p(fm)],
    q(fm) = N(fm | mq, S = Lq Lq.T)
    p(fm) = N(fm | mp, K = Lp Lp.T)
    If mp is not given, we assume zero mean.
    2KL[q(fm) || p(fm)] = log|K| - log|S| + tr(K⁻¹S) + (mp-mq).T K⁻¹ (mp-mq)
    (mp-mq).T K⁻¹ (mp-mq) = mq.T K⁻¹ mq - 2 mq.T K⁻¹ mp + mp.T K⁻¹ mp
                                        |_________mp related_________|
    """
    invLp = inverse_cholesky(Lp)
    invLp_Lq = invLp @ Lq
    trace_invK_S = (invLp_Lq ** 2).sum()
    invLp_mq = invLp @ mq
    mahalanobis = (invLp_mq ** 2).sum()
    double_kl = logdet(Lp) - logdet(Lq) + trace_invK_S + mahalanobis
    if mp is not None:
        invLp_mp = invLp @ mp
        mq_invK_mp = invLp_mq.t() @ invLp_mp
        mp_invK_mp = (invLp_mp ** 2).sum()
        double_kl += -2.0 * mq_invK_mp + mp_invK_mp
    return double_kl / 2.0


def inverse_softplus(positive_values):
    return torch.log(torch.exp(positive_values) - 1.0)


def log_gaussian(x, mean, covar):
    """
    Compute the log pdf of Gaussian distribution N(mean, covar) given variable x,
    :param x: value
    :param mean: the mean of Gaussian
    :param covar: the covariance matrix
    :return: log p(x|mean, covar)
    """
    n = x.size(0)
    if len(covar.size()) >= 2 and covar.size(1) == covar.size(0):
        # full covariance matrix
        L_cov = stable_cholesky(covar)
        invL_x_minus_mean = triangular_solve(x - mean, L_cov)
        return -0.5 * (n * LOG_2PI + logdet(L_cov) + (invL_x_minus_mean ** 2).sum())
    else:
        # only diagonal matrix
        return (-0.5 * LOG_2PI - 0.5 * stable_log(covar) - 0.5 * (x - mean) ** 2 / covar).sum()


def pairwise_dots(X1, X2=None):
    '''
    Compute pairwise inner products between matrix x1 and matrix y.
    inner_products[i, j] = ||X1[i, :] - X2[j, :]||^2.
    :param X1: Input matrix. Each row is a sample.
    :type X1: torch.Tensor torch.Size[(n, d)]
    :param X2: Input matrix. Each row is a sample.
    :type X2: torch.Tensor torch.Size[(m, d)]
    :return: Pairwise inner product matrix.
    :rtype: torch.Tensor torch.Size[(n, m)]
    '''
    if X2 is None:
        X2 = X1
    X1_square = (X1 ** 2).sum(1).view(-1, 1)
    X2_square = (X2 ** 2).sum(1).view(1, -1)
    inner_products = X1_square - 2.0 * X1 @ X2.t() + X2_square

    # Ensure diagonal is zero if x1 = x2
    if X2 is None:
        inner_products -= torch.diag(inner_products.diag())
    # Ensure inner product is positive and not infinite
    return torch.clamp(inner_products, 0.0, float('inf'))
