import attr
import torch
from torch import nn
from torch.distributions.kl import kl_divergence

from lantern.model import Variational
from lantern.model.basis import Basis


@attr.s(cmp=False)
class LanguageBasis(Basis, Variational):
    """A language model basis for encoding mutational data.
    """
    # class implementation
    pass
