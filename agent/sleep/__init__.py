"""SLEEP Continual Learning package (Google Research arXiv:2606.03979)."""

from .cms import CMSChain, MoEBlock
from .policy import FoundationVLAWithCMS, gaussian_kl, tokenize
from .sleep import Learner, SleepConfig

__all__ = [
    "CMSChain",
    "MoEBlock",
    "FoundationVLAWithCMS",
    "gaussian_kl",
    "tokenize",
    "Learner",
    "SleepConfig",
]
