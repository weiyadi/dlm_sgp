from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import UnwhitenedVariationalStrategy
import gpytorch


class SGPClassModel(ApproximateGP):
    def __init__(self, inducing_inputs, kern, mean=None, learning_inducing=True, fix_hyper=False):
        variational_distribution = CholeskyVariationalDistribution(inducing_inputs.size(0))
        variational_strategy = UnwhitenedVariationalStrategy(self, inducing_inputs, variational_distribution,
                                                             learn_inducing_locations=learning_inducing)
        super(SGPClassModel, self).__init__(variational_strategy)
        if mean is None:
            self.mean_module = gpytorch.means.ConstantMean()
        else:
            self.mean_module = mean
        self.covar_module = kern
        if fix_hyper:
            [p.requires_grad_(False) for p in self.covar_module.parameters()]
            [p.requires_grad_(False) for p in self.mean_module.parameters()]

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred
