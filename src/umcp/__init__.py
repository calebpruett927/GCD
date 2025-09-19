"""
UMCP package exposing core functions and CLI.

This file re‑exports the most commonly used functions for convenience.
"""

from .invariants import compute_invariants  # noqa: F401
from .regimes import assign_regimes  # noqa: F401
from .audit import audit_dataframe, audit_csv  # noqa: F401
from .weld import assert_kappa_continuity, assert_unit_invariance  # noqa: F401

__all__ = [
    'compute_invariants',
    'assign_regimes',
    'audit_dataframe',
    'audit_csv',
    'assert_kappa_continuity',
    'assert_unit_invariance',
]
