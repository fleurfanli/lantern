import attr
import gpytorch
import torch

from lantern import Module
from lantern.model.surface import Surface
from lantern.model.basis import Basis
from lantern.loss import ELBO_GP


@attr.s(cmp=False)
class Model(Module):
    """The base model interface for *lantern*, learning a surface along a low-dimensional basis of mutational data.
    """

    basis: Basis = attr.ib()
    surface: Surface = attr.ib()
    likelihood: gpytorch.likelihoods.Likelihood = attr.ib()

    @surface.validator
    def _surface_validator(self, attribute, value):
        if value.Kbasis != self.basis.K:
            raise ValueError(
                f"Basis ({self.basis.K}) and surface ({value.Kbasis}) do not have the same dimensionality."
            )

    def forward(self, X):

        Z = self.basis(X)
        f = self.surface(Z)

        return f

    # def loss(self, *args, **kwargs):
    #     return self.basis.loss(*args, **kwargs) + ELBO_GP.fromModel(
    #         self, *args, **kwargs
    #     )

    def loss(self, output, target, *args, **kwargs):
        basis_loss = self.basis.loss(*args, **kwargs)
        elbo_loss = ELBO_GP.fromModel(self, *args, **kwargs)

        # compute the bias penalty
        predicted_mean = output.mean
        bias_penalty = torch.mean((predicted_mean - target) ** 2)

        total_loss = basis_loss + elbo_loss + 0.5 * bias_penalty
        return total_loss
