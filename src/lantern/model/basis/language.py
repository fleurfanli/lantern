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

    # W_mu: nn.Parameter = attr.ib()
    # W_log_sigma: nn.Parameter = attr.ib()
    # log_alpha: nn.Parameter = attr.ib()
    # log_beta: nn.Parameter = attr.ib()
    # alpha_prior: Gamma = attr.ib()

    @classmethod
    def hello(cls):
        print("Hii, using language model basis!")

    # class implementation
    # @classmethod
    # def fromDataset(cls, ds, K, alpha_0=0.001, beta_0=0.001, meanEffectsInit=False):
    #     p = ds.p
    #     Wmu = torch.randn(p, K)

    #     # initialize first dimensions to mean effects
    #     if meanEffectsInit:
    #         mu = ds.meanEffects()
    #         Wmu[:, : mu.shape[1]] = mu

    #     Wmu = nn.Parameter(Wmu)

    #     # try to prevent some instabilities on gradients
    #     log_alpha = nn.Parameter(torch.randn(K))
    #     log_alpha.register_hook(lambda grad: torch.clamp(grad, -10.0, 10.0))
    #     log_beta = nn.Parameter(torch.randn(K))
    #     log_beta.register_hook(lambda grad: torch.clamp(grad, -10.0, 10.0))

    #     return cls(
    #         Wmu,
    #         nn.Parameter(torch.randn(p, K) - 3),
    #         log_alpha,
    #         log_beta,
    #         Gamma(alpha_0, beta_0),
    #     )

    # @classmethod
    # def build(cls, p, K, alpha_0=0.001, beta_0=0.001):
    #     Wmu = torch.randn(p, K)

    #     Wmu = nn.Parameter(Wmu)

    #     # try to prevent some instabilities on gradients
    #     log_alpha = nn.Parameter(torch.randn(K))
    #     log_alpha.register_hook(lambda grad: torch.clamp(grad, -10.0, 10.0))
    #     log_beta = nn.Parameter(torch.randn(K))
    #     log_beta.register_hook(lambda grad: torch.clamp(grad, -10.0, 10.0))

    #     return cls(
    #         Wmu,
    #         nn.Parameter(torch.randn(p, K) - 3),
    #         log_alpha,
    #         log_beta,
    #         Gamma(alpha_0, beta_0),
    #     )

