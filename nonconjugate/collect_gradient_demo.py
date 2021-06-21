import argparse
import gpytorch
from poisson_likelihood import PoissonExp, PoissonLog1p
from sgp_class import SGPClassModel
from monte_carlo import MonteCarloDLM, MonteCarloDLMJitter
import torch
from dlm_product_sampling import DLM_ProductSampling
from script import toy_data, get_base_model
import matplotlib.pyplot as plt
import numpy as np


def plot_condc(stats, exact, title, name, legend=None):
    assert len(stats) == 3
    plt.figure()
    plt.rcParams.update({'font.size': 11})
    plt.title(title)
    plt.xlabel(r'$\hat{\nabla f}^\top \nabla f / || \nabla f ||_2$')
    plt.ylabel('Count (100 total)')
    for i in range(3):
        values = [np.sum(grad * exact) / np.sum(exact * exact) for grad in stats[i]]
        plt.hist(values, alpha=0.7)
    if legend is not None:
        plt.legend(legend)
    plt.tight_layout()
    plt.savefig(name)


def plot_condd(stats, exact, title, name, legend=None):
    assert len(stats) == 3
    plt.figure()
    plt.rcParams.update({'font.size': 11})
    plt.title(title)
    if exact is not None:
        plt.vlines(np.sum(exact ** 2), ymin=0, ymax=25)
    plt.xlabel(r'$|| \hat{\nabla f} ||_2^2$')
    plt.ylabel('Count (100 total)')
    for i in range(3):
        values = [np.sum(grad ** 2) for grad in stats[i]]
        plt.hist(values, alpha=0.7)
    plt.legend(['True'] + legend)
    plt.tight_layout()
    plt.savefig(name)


def plot_err(stats, exact, title, name, legend=None, axis_rotate=False):
    assert len(stats) == 3
    if exact is None:
        return
    plt.figure()
    plt.rcParams.update({'font.size': 11})
    plt.title(title)
    plt.xlabel(r'$|| \hat{\nabla f} - \nabla f ||_2^2$')
    plt.ylabel('Cound (100 total)')
    for i in range(3):
        values = [np.sum((grad - exact) ** 2) for grad in stats[i]]
        plt.hist(values, alpha=0.7)
    # plt.legend(['mc1', 'mc10', 'mc100'])
    if axis_rotate == True:
        plt.xticks(rotation=30)
    plt.legend(legend)
    plt.tight_layout()
    plt.savefig(name)


def get_gradients(model, x_train, y_train, likelihood, sample_size=0, beta=0.1, ps=False, jitter=0.):
    if sample_size == 0:
        # Exact gradient calculation
        if isinstance(likelihood, gpytorch.likelihoods.BernoulliLikelihood):
            mll = gpytorch.mlls.PredictiveLogLikelihood(likelihood, model, y_train.numel(), beta=beta)
        else:
            # Poisson likelihood
            mll = MonteCarloDLM(likelihood, model, y_train.numel(), beta=beta, sample_size=10000)
    else:
        if ps == True:
            mll = DLM_ProductSampling(likelihood, model, y_train.numel(), beta=beta, sample_size=sample_size)
        else:
            if jitter == 0.:
                mll = MonteCarloDLM(likelihood, model, y_train.numel(), beta=beta, sample_size=sample_size)
            else:
                mll = MonteCarloDLMJitter(likelihood, model, y_train.numel(), beta=beta, sample_size=sample_size,
                                          jitter=jitter)
    mll.zero_grad()
    output = model(x_train)
    loss = -mll(output, y_train)
    loss.backward()
    grads = [p.grad.clone().detach() for p in mll.parameters() if p.requires_grad == True]
    assert len(grads) == 2
    return grads[0].cpu().numpy(), grads[1].cpu().numpy()


def plot_all_gradients(grads, true_grad, mll_type):
    dexact_dm, dexact_dchol = true_grad
    dl_dm1s, dl_dchol1s, dl_dm10s, dl_dchol10s, dl_dm100s, dl_dchol100s = zip(*grads)
    if mll_type == 'smooth-bMC':
        legend = ['smooth-bMC-1', 'smooth-bMC-10', 'smooth-bMC-100']
        mll = 'mc'
    elif mll_type == 'bMC':
        legend = ['bMC-1', 'bMC-10', 'bMC-100']
        mll = 'mc'
    else:
        # uPS
        legend = ['uPS-1', 'uPS-10', 'uPS-100']
        mll = 'ps'
    plot_condc([dl_dchol1s, dl_dchol10s, dl_dchol100s], dexact_dchol, 'chol ' + mll,
               'condc-{}-chol.pdf'.format(mll_type), legend=legend)
    plot_condc([dl_dm1s, dl_dm10s, dl_dm100s], dexact_dm, 'mean ' + mll_type,
               'condc-{}-m.pdf'.format(mll_type), legend=legend)
    plot_condd([dl_dchol1s, dl_dchol10s, dl_dchol100s], dexact_dchol, 'chol ' + mll,
               'condd-{}-chol.pdf'.format(mll_type), legend=legend)
    plot_condd([dl_dm1s, dl_dm10s, dl_dm100s], dexact_dm, 'mean ' + mll,
               'condd-{}-m.pdf'.format(mll_type), legend=legend)
    plot_err([dl_dchol1s, dl_dchol10s, dl_dchol100s], dexact_dchol, 'chol ' + mll,
             'err-{}-chol.pdf'.format(mll_type), legend=legend)
    plot_err([dl_dm1s, dl_dm10s, dl_dm100s], dexact_dm, 'mean ' + mll,
             'err-{}-m.pdf'.format(mll_type), legend=legend)


# Example to show how to collect gradients
NUM_INDUCING = 20
beta = 0.1
x_train, y_train, _, _ = toy_data(n_train=1000, n_test=200)
y_train = torch.where(y_train == -1, torch.tensor(0.), y_train).squeeze()
Z = x_train[:NUM_INDUCING]
kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
mean = gpytorch.means.ConstantMean()
likelihood = gpytorch.likelihoods.BernoulliLikelihood()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_train, y_train, Z = x_train.to(device), y_train.to(device), Z.to(device)

base_model = get_base_model(Z, mean, kernel, likelihood, x_train, y_train)

model = SGPClassModel(base_model.variational_strategy.inducing_points, mean=base_model.mean_module,
                      kern=base_model.covar_module, learning_inducing=False, fix_hyper=True).to(device)

true_grad = get_gradients(model, x_train, y_train, likelihood, beta=beta)

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, choices=['bMC', 'smooth-bMC', 'uPS'])
args = parser.parse_args()

if args.method == 'bMC':
    # collect bMC gradients
    grads_mc = []
    for i in range(100):
        dl_dm1, dl_dchol1 = get_gradients(model, x_train, y_train, likelihood, sample_size=1, beta=beta)
        dl_dm10, dl_dchol10 = get_gradients(model, x_train, y_train, likelihood, sample_size=10, beta=beta)
        dl_dm100, dl_dchol100 = get_gradients(model, x_train, y_train, likelihood, sample_size=100, beta=beta)
        grads_mc.append((dl_dm1, dl_dchol1, dl_dm10, dl_dchol10, dl_dm100, dl_dchol100))
    plot_all_gradients(grads_mc, true_grad, 'bMC')
elif args.method == 'smooth-bMC':
    # collect smooth-bMC gradients
    jitter = 1e-4
    grads_mc_jitter = []
    for i in range(100):
        dl_dm1, dl_dchol1 = get_gradients(model, x_train, y_train, likelihood, sample_size=1, beta=beta, jitter=jitter)
        dl_dm10, dl_dchol10 = get_gradients(model, x_train, y_train, likelihood, sample_size=10, beta=beta, jitter=jitter)
        dl_dm100, dl_dchol100 = get_gradients(model, x_train, y_train, likelihood, sample_size=100, beta=beta,
                                              jitter=jitter)
        grads_mc_jitter.append((dl_dm1, dl_dchol1, dl_dm10, dl_dchol10, dl_dm100, dl_dchol100))
    plot_all_gradients(grads_mc_jitter, true_grad, 'smooth-bMC')
else:
    # collect uPS gradients
    grads_ps = []
    for i in range(100):
        dl_dm1, dl_dchol1 = get_gradients(model, x_train, y_train, likelihood, sample_size=1, beta=beta, ps=True)
        dl_dm10, dl_dchol10 = get_gradients(model, x_train, y_train, likelihood, sample_size=10, beta=beta, ps=True)
        dl_dm100, dl_dchol100 = get_gradients(model, x_train, y_train, likelihood, sample_size=100, beta=beta, ps=True)
        grads_ps.append((dl_dm1, dl_dchol1, dl_dm10, dl_dchol10, dl_dm100, dl_dchol100))
    plot_all_gradients(grads_ps, true_grad, 'uPS')
