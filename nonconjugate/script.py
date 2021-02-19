import argparse
import gpytorch
from poisson_likelihood import PoissonExp, PoissonLog1p
from sgp_class import SGPClassModel
from monte_carlo import MonteCarloDLM, MonteCarloDLMJitter
import torch
import math
from dlm_product_sampling import DLM_ProductSampling


def binary_loss(likelihood, model, xt, yt):
    with torch.no_grad():
        f_pred = model(xt)
        logloss = -torch.mean(likelihood.log_marginal(yt, f_pred)).item()
        marginal_prob = likelihood.marginal(f_pred).probs
        prediction = (marginal_prob > 0.5).float()
        misclassification = torch.sum((prediction != yt).int()).item() / xt.size(0)
        return logloss, misclassification


def poisson_loss(likelihood, model, xt, yt):
    with torch.no_grad():
        pred = model(xt)
        logloss = -torch.mean(likelihood.log_marginal(yt, pred))

        y_hat = likelihood.predictive_mean(pred)
        mean_rel_err = torch.mean(torch.abs(yt - y_hat) / torch.where(yt != 0., yt, torch.tensor(1., device=yt.device)))
        return logloss.item(), mean_rel_err.item()


def train_model(model, mll, x_train, y_train, num_opt=1000, verbose=False, tol=None):
    '''
    A wrapper to train a model.
    '''
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
    converge_size = 20
    losses = []
    for i in range(num_opt):
        optimizer.zero_grad()
        output = model(x_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()
        if verbose:
            print('Iteration: {0:04d} Loss: {1: .6f} Kernel: {2: .6f} {3: .6f}'.format(i, loss.item(),
                                                                                       model.covar_module.outputscale.item(),
                                                                                       model.covar_module.base_kernel.lengthscale.item()))
        losses.append(loss.item())
        if len(losses) > converge_size:
            last_numbers = losses[-converge_size:]
            if tol is not None and max(last_numbers) - min(last_numbers) < tol:
                if verbose:
                    print("converged at iteration: ", i)
                break


def toy_data(n_train, n_test, sort=False):
    """Create 1D classification dataset
    :n_train: Number of training samples.
    :n_test: Number of testing samples.
    :returns: x_train, y_train, x_test, y_test
    """
    xn = torch.rand(n_train, 1) * 2 - 1  # Uniformly random in [-1, 1]
    yn = torch.sign(torch.cos(xn * math.pi)).add(1).div(2)
    if sort:
        indices = torch.argsort(xn, axis=0)
        xn = xn[indices.squeeze()]
        yn = yn[indices.squeeze()]
    xt = torch.linspace(-1.1, 1.1, n_test).view(-1, 1)
    yt = torch.sign(torch.cos(xt * math.pi)).add(1).div(2)
    return xn, yn, xt, yt


def get_base_model(inducing, mean, kernel, likelihood, x_train, y_train, num_opt=1000, tol=1e-4):
    base_model = SGPClassModel(inducing, mean=mean, kern=kernel, learning_inducing=True)
    base_mll = gpytorch.mlls.VariationalELBO(likelihood, base_model, y_train.numel())
    base_model.train()
    likelihood.train()
    train_model(base_model, base_mll, x_train, y_train, num_opt=num_opt, tol=tol, verbose=False)
    return base_model


def load_dataset(args):
    raise NotImplementedError


def generate_regs(size):
    result = []
    num = size / 2
    while num > 0.01:
        result.append(num)
        if num > 500:
            num /= 4
        else:
            num /= 2
    return result


def get_mean(mean_type):
    if mean_type == 'constant':
        mean = gpytorch.means.ConstantMean()
    elif mean_type == 'zero':
        mean = gpytorch.means.ZeroMean()
    else:
        print('Unrecognized mean function type')
        raise NotImplementedError

    return mean


def get_kernel(kern):
    if kern == 'rbf':
        kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    elif kern == 'matern':
        kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())
    else:
        print('Unrecognized kernel type')
        raise NotImplementedError

    return kernel


if __name__ == '__main__':
    # running parameters
    num_opt = 1000
    tol = 1e-4

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='toy')
    parser.add_argument('--n_train', type=int, default=1000)
    parser.add_argument('--n_test', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--inducing', type=int, default=20)
    parser.add_argument('--likelihood', type=str, default='')
    parser.add_argument('--mean_type', type=str, default='constant')
    parser.add_argument('--kern', type=str, default='rbf')
    parser.add_argument('--method', type=str, default='')
    parser.add_argument('--reg', type=float, default=1.)
    parser.add_argument('--num_samples', type=int, default=0)
    parser.add_argument('--jitter', type=float, default=0)
    parser.add_argument('--auto_select_reg', dest='auto_select_reg', action='store_true',
                        help='If sets auto_select_reg, we will pick regularization parameter automatically.')
    parser.set_defaults(auto_select_reg=False)

    args = parser.parse_args()
    index = args.seed
    NUM_INDUCING = args.inducing

    if args.likelihood == 'binary':
        likelihood = gpytorch.likelihoods.BernoulliLikelihood()
    elif args.likelihood == 'poisson_exp':
        likelihood = PoissonExp()
    elif args.likelihood == 'poisson_log1p':
        likelihood = PoissonLog1p()
    else:
        print("Invalid likelihood specification")
        raise NotImplementedError

    torch.manual_seed(index)
    if args.dataset == 'toy':
        x_train, y_train, x_test, y_test = toy_data(n_train=args.n_train, n_test=args.n_test)
    else:
        x_train, y_train, x_test, y_test = load_dataset(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # turning y values from {-1, 1} to {0, 1} for classification task.
    # y values remain unchanged for count regression task.
    y_train, y_test = torch.where(y_train == -1, torch.tensor(0., device=device), y_train).squeeze(), torch.where(
        y_test == -1, torch.tensor(0., device=device), y_test).squeeze()
    x_train, y_train, x_test, y_test = x_train.to(device), y_train.to(device), x_test.to(device), y_test.to(device)

    Z = x_train[:NUM_INDUCING]

    if not args.auto_select_reg:
        if args.method == 'svgp':
            model = SGPClassModel(Z, mean=get_mean(args.mean_type), kern=get_kernel(args.kern), learning_inducing=True)
            mll = gpytorch.mlls.VariationalELBO(likelihood, model, y_train.numel(), beta=args.reg)
        elif args.method == 'joint-dlm':
            model = SGPClassModel(Z, mean=get_mean(args.mean_type), kern=get_kernel(args.kern), learning_inducing=True)
            if args.num_samples == 0:
                mll = gpytorch.mlls.PredictiveLogLikelihood(likelihood, model, y_train.numel(), beta=args.reg)
            else:
                if args.jitter == 0:
                    mll = MonteCarloDLM(likelihood, model, y_train.numel(), beta=args.reg, sample_size=args.num_samples)
                else:
                    # smooth-bMC
                    mll = MonteCarloDLMJitter(likelihood, model, y_train.numel(), beta=args.reg,
                                              sample_size=args.num_samples, jitter=args.jitter)
        elif args.method == 'fixed-dlm':
            base_model = get_base_model(Z, get_mean(args.mean_type), get_kernel(args.kern), likelihood, x_train,
                                        y_train)

            model = SGPClassModel(base_model.variational_strategy.inducing_points, mean=base_model.mean_module,
                                  kern=base_model.covar_module, learning_inducing=False, fix_hyper=True)
            if args.num_samples == 0:
                mll = gpytorch.mlls.PredictiveLogLikelihood(likelihood, model, y_train.numel(), beta=args.reg)
            else:
                if args.jitter == 0:
                    mll = MonteCarloDLM(likelihood, model, y_train.numel(), beta=args.reg, sample_size=args.num_samples)
                else:
                    # smooth-bMC
                    mll = MonteCarloDLMJitter(likelihood, model, y_train.numel(), beta=args.reg,
                                              sample_size=args.num_samples, jitter=args.jitter)
        elif args.method == 'joint-dlm-ps':
            model = SGPClassModel(Z, mean=get_mean(args.mean_type), kern=get_kernel(args.kern), learning_inducing=True)
            mll = DLM_ProductSampling(likelihood, model, y_train.numel(), beta=args.reg, sample_size=args.num_samples)
        elif args.method == 'fixed-dlm-ps':
            base_model = get_base_model(Z, get_mean(args.mean_type), get_kernel(args.kern), likelihood, x_train,
                                        y_train)

            model = SGPClassModel(base_model.variational_strategy.inducing_points, mean=base_model.mean_module,
                                  kern=base_model.covar_module, learning_inducing=False, fix_hyper=True)
            mll = DLM_ProductSampling(likelihood, model, y_train.numel(), beta=args.reg, sample_size=args.num_samples)
        else:
            print('No method called ' + args.method)
            raise NotImplementedError

        model.train()
        likelihood.train()
        train_model(model, mll, x_train, y_train, num_opt=num_opt, tol=tol, verbose=True)

    else:
        regs = generate_regs(x_train.size(0))

        # split train/val
        n_train = int(x_train.size(0) * 0.8)
        x_val, y_val = x_train[n_train:], y_train[n_train:]
        x_train, y_train = x_train[:n_train], y_train[:n_train]

        if args.method == 'svgp':
            models = [
                SGPClassModel(Z, mean=get_mean(args.mean_type), kern=get_kernel(args.kern), learning_inducing=True)
                for _ in regs]
            mlls = [gpytorch.mlls.VariationalELBO(likelihood, model, y_train.numel(), beta=reg) for (model, reg) in
                    zip(models, regs)]
        elif args.method == 'joint-dlm':
            models = [
                SGPClassModel(Z, mean=get_mean(args.mean_type), kern=get_kernel(args.kern), learning_inducing=True)
                for _ in regs]
            if args.num_samples == 0:
                mlls = [gpytorch.mlls.PredictiveLogLikelihood(likelihood, model, y_train.numel(), beta=reg) for
                        (model, reg) in zip(models, regs)]
            else:
                if args.jitter == 0:
                    mlls = [MonteCarloDLM(likelihood, model, y_train.numel(), beta=reg, sample_size=args.num_samples)
                            for (model, reg) in zip(models, regs)]
                else:
                    # smooth-bMC
                    mlls = [MonteCarloDLMJitter(likelihood, model, y_train.numel(), beta=reg,
                                                sample_size=args.num_samples, jitter=args.jitter) for (model, reg) in
                            zip(models, regs)]
        elif args.method == 'fixed-dlm':
            base_model = get_base_model(Z, get_mean(args.mean_type), get_kernel(args.kern), likelihood, x_train,
                                        y_train)

            models = [SGPClassModel(base_model.variational_strategy.inducing_points, mean=base_model.mean_module,
                                    kern=base_model.covar_module, learning_inducing=False, fix_hyper=True) for _ in
                      regs]
            if args.num_samples == 0:
                mlls = [gpytorch.mlls.PredictiveLogLikelihood(likelihood, model, y_train.numel(), beta=reg) for
                        (model, reg) in zip(models, regs)]
            else:
                if args.jitter == 0:
                    mlls = [
                        MonteCarloDLM(likelihood, model, y_train.numel(), beta=reg, sample_size=args.num_samples)
                        for (model, reg) in zip(models, regs)]
                else:
                    # smooth-bMC
                    mlls = [MonteCarloDLMJitter(likelihood, model, y_train.numel(), beta=reg,
                                                sample_size=args.num_samples, jitter=args.jitter) for (model, reg) in
                            zip(models, regs)]
        elif args.method == 'joint-dlm-ps':
            models = [
                SGPClassModel(Z, mean=get_mean(args.mean_type), kern=get_kernel(args.kern), learning_inducing=True)
                for _ in regs]
            mlls = [DLM_ProductSampling(likelihood, model, y_train.numel(), beta=reg, sample_size=args.num_samples)
                    for (model, reg) in zip(models, regs)]
        elif args.method == 'fixed-dlm-ps':
            base_model = get_base_model(Z, get_mean(args.mean_type), get_kernel(args.kern), likelihood, x_train,
                                        y_train)

            models = [SGPClassModel(base_model.variational_strategy.inducing_points, mean=base_model.mean_module,
                                    kern=base_model.covar_module, learning_inducing=False, fix_hyper=True) for _ in
                      regs]
            mlls = [DLM_ProductSampling(likelihood, model, y_train.numel(), beta=reg, sample_size=args.num_samples)
                    for (model, reg) in zip(models, regs)]
        else:
            print('No method called ' + args.method)
            raise NotImplementedError

        for (model, mll) in zip(models, mlls):
            model.train()
            likelihood.train()
            train_model(model, mll, x_train, y_train, num_opt=num_opt, tol=tol, verbose=False)

        if args.likelihood == 'binary':
            val_losses = [binary_loss(likelihood, model, x_val, y_val)[0] for model in models]
        else:
            val_losses = [poisson_loss(likelihood, model, x_val, y_val)[0] for model in models]
        print('Validating log losses: ', val_losses)
        best_idx = val_losses.index(min(val_losses))
        print('Best reg is: ', regs[best_idx])
        model = models[best_idx]

    # Evaluate the result
    if args.likelihood == 'binary':
        logloss, err = binary_loss(likelihood, model, x_test, y_test)
    else:
        logloss, err = poisson_loss(likelihood, model, x_test, y_test)
    print("Log loss on test data", logloss)
    print("Err on test data", err)
