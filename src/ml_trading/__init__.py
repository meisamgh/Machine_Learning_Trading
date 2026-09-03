"""Leakage-safe primitives for offline machine-learning trading research."""

from .factors import alpha_012, alpha_101
from .signals import SignalPolicy, make_signals
from .targets import continuous_residual_target
from .validation import PurgedDateSplit

__all__ = [
    "PurgedDateSplit",
    "SignalPolicy",
    "alpha_012",
    "alpha_101",
    "continuous_residual_target",
    "make_signals",
]
