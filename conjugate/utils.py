"""
Utility functions
"""
import torch
import matplotlib as mpl
import numpy as np
import math

mpl.use('Agg')
from matplotlib import pyplot as plt


def sin_data(n_train, n_test, noise_std, sort=False):
    """Create 1D sine function regression dataset
    :n_train: Number of training samples.
    :n_test: Number of testing samples.
    :noise_srd: Standard deviation of observation noise.
    :returns: x_train, y_train, x_test, y_test
    """

    def ground_truth(x):
        return torch.sin(math.pi * x)

    xn = torch.rand(n_train, 1) * 2 - 1  # Uniformly random in [-1, 1]
    yn = ground_truth(xn) + noise_std * torch.randn(n_train, 1)
    if sort:
        indices = torch.argsort(xn, axis=0)
        xn = xn[indices.squeeze()]
        yn = yn[indices.squeeze()]
    xt = torch.linspace(-1.1, 1.1, n_test).view(-1, 1)
    yt = ground_truth(xt) + noise_std * torch.randn(n_test, 1)
    return xn, yn, xt, yt


def plot_lengthscale(xt, lengthscale, uncertainty, name=None):
    """
    Visualize lengthscale function and its corresponding uncertainty.

    :lengthscale: Lengthscale mean.
    :uncertainty: Standard deviation of lengthscale prediction.
    """
    lengthscale = lengthscale.numpy().ravel()
    uncertainty = uncertainty.numpy().ravel()
    lower = lengthscale - 2.0 * uncertainty
    upper = lengthscale + 2.0 * uncertainty
    xt = xt.numpy().ravel()

    fig, ax = plt.subplots()
    ax.plot(xt, lengthscale, 'b', lw=2, alpha=0.8, label='Lengthscale')
    ax.fill_between(xt, lower, upper,
                    facecolor='b', alpha=0.3, label='95% CI')
    ax.set_xlim([xt.min(), xt.max()])
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2), ncol=3,
              borderaxespad=0, frameon=False)
    if name is not None:
        fig.savefig('../results/prediction/' + name + '.svg')


def plot_pytorch(dataset, preds, name=None):
    dataset = [tensor.numpy().ravel() for tensor in dataset]
    xn, yn, xt, ft = dataset

    mean = preds.mean.cpu().numpy().ravel()
    lower, upper = preds.confidence_region()
    lower = lower.cpu().numpy().ravel()
    upper = upper.cpu().numpy().ravel()

    fig, ax = plt.subplots()
    ax.plot(xn, yn, 'k.', label='Training data')
    ax.plot(xt, ft, 'r--', lw=2, alpha=0.8, label='Function')
    ax.plot(xt, mean, 'b', lw=2, alpha=0.8, label='Prediction')
    ax.fill_between(xt, lower, upper,
                    facecolor='b', alpha=0.3, label='95% CI')
    ax.set_xlim([xt.min(), xt.max()])
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2), ncol=3,
              borderaxespad=0, frameon=False)
    if name is not None:
        fig.savefig('../results/prediction/' + name + '.svg')


def plot_1d_results(dataset, mean, std, title=None, name=None):
    """
    Visualize training data, ground-truth function, and prediction.

    :dataset: A tuple containing (Xn, Yn, Xt, Ft)
    :mean: Mean of predictive Gaussian distribution.
    :std: Standard deviation of predictive Gaussian distribution.
    """
    dataset = [tensor.numpy().ravel() for tensor in dataset]
    xn, yn, xt, ft = dataset

    mean, std = mean.numpy().ravel(), std.numpy().ravel()
    lower = mean - 2.0 * std
    upper = mean + 2.0 * std

    fig, ax = plt.subplots()
    ax.plot(xn, yn, 'k.', label='Training data')
    ax.plot(xt, ft, 'r.', lw=2, alpha=0.8, label='Test data')
    ax.plot(xt, mean, 'b', lw=2, alpha=0.8, label='Prediction')
    ax.fill_between(xt, lower, upper,
                    facecolor='b', alpha=0.3, label='95% CI')
    ax.set_xlim([xt.min(), xt.max()])
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2), ncol=3,
              borderaxespad=0, frameon=False)
    if title is not None:
        ax.set_title(title, loc='center')
    if name is not None:
        fig.savefig(name + '.pdf')


def train(model, optimizer, n_iter, verbose=True, name=None, Xn=None, yn=None, tol=None):
    """
    Training helper function.
    """
    n_train = Xn.size(0) if Xn is not None else model.Xn.size(0)
    losses = []
    for i in range(n_iter):
        optimizer.zero_grad()
        if Xn is None and yn is None:
            loss = model.loss()
        else:
            loss = model.loss(Xn, yn)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if tol is not None:
            # if the result is stable for over 50 iteration, then we consider it converges.
            n = 50
            if len(losses) > n:
                last = losses[-n:]
                if max(last) - min(last) <= tol:
                    if verbose:
                        print("Converges at iteration: ", i)
                    break
        if verbose:
            print('Iteration: {0:04d} Loss: {1: .6f}'.format(i, loss.item() / n_train))

    if name is not None:
        plt.figure()
        plt.plot(losses, lw=2)
        plt.ylabel('Loss')
        plt.xlabel('Number of iteration')
        plt.savefig(name + '.svg')
