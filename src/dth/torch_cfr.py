"""Differentiable, bounded CFR+ for approximate training-time matrix solves.

The exact LP solver in :mod:`dth.solver` remains the authority for targets and
certification.  This module deliberately implements only a fixed unrolled
iteration budget so gradients can flow from induced policies back into learned
matrix entries.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class TorchCFRPlusSolution:
    value: Tensor
    drop_policy: Tensor
    check_policy: Tensor
    lower_bound: Tensor
    upper_bound: Tensor
    saddle_gap: Tensor
    iterations: int


def _regret_matching_plus(regrets: Tensor) -> Tensor:
    positive = torch.clamp(regrets, min=0.0)
    totals = positive.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(positive, 1.0 / positive.shape[-1])
    normalized = positive / totals.clamp_min(torch.finfo(positive.dtype).tiny)
    return torch.where(totals > 0.0, normalized, uniform)


def solve_matrix_cfr_plus_torch(
    matrix: Tensor,
    *,
    iterations: int,
    averaging_delay: int = 0,
) -> TorchCFRPlusSolution:
    """Approximately solve one matrix or a batch of row-max/column-min games.

    The final two dimensions are actions; any leading dimensions are treated as
    independent games.  Updates mirror :func:`dth.cfr.solve_matrix_cfr_plus`
    without early stopping or graph-breaking scalar conversions.
    """

    if matrix.ndim < 2 or matrix.shape[-2] == 0 or matrix.shape[-1] == 0:
        raise ValueError("matrix must have at least two non-empty dimensions")
    if not matrix.is_floating_point():
        raise ValueError("matrix must use a floating-point dtype")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if averaging_delay < 0:
        raise ValueError("averaging delay must be nonnegative")

    batch_shape = matrix.shape[:-2]
    rows, columns = matrix.shape[-2:]
    drop_regrets = matrix.new_zeros((*batch_shape, rows))
    check_regrets = matrix.new_zeros((*batch_shape, columns))
    drop_sum = matrix.new_zeros((*batch_shape, rows))
    check_sum = matrix.new_zeros((*batch_shape, columns))

    for iteration in range(1, iterations + 1):
        drop_policy = _regret_matching_plus(drop_regrets)
        check_policy = _regret_matching_plus(check_regrets)
        row_values = torch.matmul(matrix, check_policy.unsqueeze(-1)).squeeze(-1)
        column_values = torch.matmul(
            drop_policy.unsqueeze(-2), matrix
        ).squeeze(-2)
        expected = (drop_policy * row_values).sum(dim=-1)

        drop_regrets = torch.clamp(
            drop_regrets + row_values - expected.unsqueeze(-1), min=0.0
        )
        check_regrets = torch.clamp(
            check_regrets + expected.unsqueeze(-1) - column_values, min=0.0
        )

        weight = max(0, iteration - averaging_delay)
        if weight:
            drop_sum = drop_sum + float(weight) * drop_policy
            check_sum = check_sum + float(weight) * check_policy

    if iterations <= averaging_delay:
        average_drop = _regret_matching_plus(drop_regrets)
        average_check = _regret_matching_plus(check_regrets)
    else:
        average_drop = drop_sum / drop_sum.sum(dim=-1, keepdim=True)
        average_check = check_sum / check_sum.sum(dim=-1, keepdim=True)

    lower = torch.matmul(
        matrix.transpose(-2, -1), average_drop.unsqueeze(-1)
    ).squeeze(-1).amin(dim=-1)
    upper = torch.matmul(matrix, average_check.unsqueeze(-1)).squeeze(-1).amax(
        dim=-1
    )
    return TorchCFRPlusSolution(
        value=0.5 * (lower + upper),
        drop_policy=average_drop,
        check_policy=average_check,
        lower_bound=lower,
        upper_bound=upper,
        saddle_gap=torch.clamp(upper - lower, min=0.0),
        iterations=iterations,
    )
