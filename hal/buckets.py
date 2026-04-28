"""Compatibility exports for the legacy bucketed action model.

New code should import from ``hal.action_model``.
"""

from .action_model import (
    LEAP_BUCKET,
    STANDARD_BUCKETS,
    bucket_pair_payoff,
    build_payoff_matrix,
    get_buckets,
    get_legal_buckets,
    resolve_bucket,
)

__all__ = [
    "LEAP_BUCKET",
    "STANDARD_BUCKETS",
    "bucket_pair_payoff",
    "build_payoff_matrix",
    "get_buckets",
    "get_legal_buckets",
    "resolve_bucket",
]

