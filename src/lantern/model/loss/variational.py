import attr

from lantern.model.loss import Term
from lantern.model import Variational


@attr.s
class KL(Term):

    """ A variational KL loss term.
    """

    name: str = attr.ib()
    component: Variational = attr.ib(repr=False)

    def loss(self, *args, **kwargs):

        return {self.name: self.component._kl}
