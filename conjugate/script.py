"""
This is a simple demo of all methods involved in the paper. The data is generated from a sine function.
"""
from rbf import RBF
from utils import train, sin_data, plot_1d_results
from linalg import log_gaussian
import torch
from sgpr_vfe import SGPR_VFE
from sgpr_fitc import SGPR_FITC
from vsgpr_dlm import VSGPR_DLM
from vsgpr_svgp import VSGPR_SVGP
from sgpr_sq import SGPR_DLMKL
from vsgpr_dlm_sample import VSGPR_DLM_SAMPLE
import argparse


def get_base_model():
    base_model = SGPR_VFE(Z.clone().detach(), x_train, y_train, RBF(variance=1.0, lengthscale=1.0).to(device),
                          variance=0.1)
    base_optimizer = torch.optim.Adam(base_model.parameters(), lr=0.1)
    train(base_model, base_optimizer, n_iter=num_opt, verbose=False, tol=tol)
    return base_model


if __name__ == '__main__':
    # running parameters
    num_opt = 1000
    tol = 1e-4
    batch_size = 6000

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--inducing', type=int, default=20, help='Number of inducing inputs.')
    parser.add_argument('--method', type=str, default='',
                        choices=['svgp', 'fitc', 'fixed-log-dlm', 'joint-log-dlm', 'fixed-sq-dlm', 'joint-sq-dlm'])
    parser.add_argument('--reg', type=float, default=1., help='Regularization parameter, default 1.0')
    parser.add_argument('--num_samples', type=int, default=0,
                        help='Number of Monte Carlo samples, only applicable for fixed- or joint-log-dlm.')

    args = parser.parse_args()
    NUM_INDUCING = args.inducing
    method = args.method
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train, y_train, x_test, y_test = sin_data(n_train=1000, n_test=200, noise_std=0.5)
    x_train, y_train, x_test, y_test = x_train.to(device), y_train.to(device), x_test.to(device), y_test.to(device)

    Z = x_train[-NUM_INDUCING:]

    if args.method == 'svgp':
        model = VSGPR_SVGP(Z.clone().detach(), RBF(variance=1.0, lengthscale=1.0).to(device), beta=args.reg,
                           batch_size=batch_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        train(model, optimizer, n_iter=num_opt, verbose=True, tol=tol, Xn=x_train, yn=y_train)
    elif args.method == 'fitc':
        base_model = get_base_model()
        # Since FITC has analytical solution, no need to train it.
        model = SGPR_FITC(base_model.inducing, x_train, y_train, base_model.kernel, base_model.variance())
    elif args.method == 'fixed-log-dlm':
        base_model = get_base_model()
        if args.num_samples == 0:
            model = VSGPR_DLM(base_model.inducing, base_model.kernel, base_model.variance(), beta=args.reg,
                              batch_size=batch_size)
        else:
            method += '-' + str(args.num_samples)
            model = VSGPR_DLM_SAMPLE(base_model.inducing, base_model.kernel, base_model.variance(), beta=args.reg,
                                     batch_size=batch_size, sample_size=args.num_samples)
        # fix kernel and variance
        [p.requires_grad_(False) for p in model.kernel.parameters()]
        model.free_variance.requires_grad = False
        model.inducing.requires_grad = False
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        train(model, optimizer, n_iter=num_opt, verbose=True, tol=tol, Xn=x_train, yn=y_train)
    elif args.method == 'joint-log-dlm':
        if args.num_samples == 0:
            model = VSGPR_DLM(Z.clone().detach(), RBF(variance=1.0, lengthscale=1.0).to(device), beta=args.reg,
                              batch_size=batch_size)
        else:
            method += '-' + str(args.num_samples)
            model = VSGPR_DLM_SAMPLE(Z.clone().detach(), RBF(variance=1.0, lengthscale=1.0).to(device), beta=args.reg,
                                     batch_size=batch_size, sample_size=args.num_samples)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        train(model, optimizer, n_iter=num_opt, verbose=True, tol=tol, Xn=x_train, yn=y_train)
    elif args.method == 'fixed-sq-dlm':
        base_model = get_base_model()
        # Since fixed-sq-dlm has analytical solution, no need to train.
        model = SGPR_DLMKL(base_model.inducing, x_train, y_train, base_model.kernel, beta=args.reg)
    elif args.method == 'joint-sq-dlm':
        model = SGPR_DLMKL(Z.clone().detach(), x_train, y_train, RBF(variance=1.0, lengthscale=1.0).to(device),
                           beta=args.reg)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        train(model, optimizer, n_iter=num_opt, verbose=True, tol=tol)
    else:
        print("No model called " + args.method)
        raise NotImplementedError

    print("Evaluate")
    model.eval()
    with torch.no_grad():
        mean, var = model(x_test, diag=True)
        if hasattr(model, 'free_variance') and model.free_variance is not None:
            var += model.variance()
        n_test = y_test.size(0)
        diff = y_test - mean
        logloss = -(log_gaussian(y_test, mean, var) / n_test).item()
        sqloss = (torch.sum(diff ** 2) / n_test).item()
        print("Log loss", logloss)
        print("Sq loss", sqloss)

        plot_1d_results((x_train, y_train, x_test, y_test), mean, torch.sqrt(var), name='result', title=method)
        print("An illustration is saved as 'result.pdf'.")
