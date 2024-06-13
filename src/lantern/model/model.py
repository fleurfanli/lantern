import attr
import gpytorch

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
        
    def hello(self):
        message = "Hii, using NEW model w/ language model embeddings in GP!"
        print("Hello!!")
        return message

    def forward(self, X):

        # pass in the embdedding matrix directly
        f = self.surface(X)

        return f

    def loss(self, *args, **kwargs):
        # only compute loss from the GP part
        return ELBO_GP.fromModel(self, *args, **kwargs)
