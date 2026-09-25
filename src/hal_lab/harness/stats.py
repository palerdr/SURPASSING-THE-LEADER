"""Identity-clustered bootstrap intervals and the hard best response."""

from __future__ import annotations

import numpy as np


def cluster_interval(values, protocol):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"mean": None, "interval_95": None, "identities": 0}
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    samples = rng.choice(values, (protocol["bootstrap_replicates"], len(values)), replace=True).mean(axis=1)
    return {"mean": float(values.mean()), "interval_95": np.quantile(samples, [0.025, 0.975]).tolist() if len(values) > 1 else None, "identities": len(values)}


def hard_response(values):
    winners = values >= np.max(values) - 1e-12
    return winners / winners.sum()
