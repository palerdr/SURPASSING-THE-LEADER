import numpy as np
import pytest
import torch

from dth.cfr import solve_matrix_cfr_plus
from dth.torch_cfr import solve_matrix_cfr_plus_torch


def test_torch_cfr_plus_matches_numpy_updates_and_supports_batches():
    matrices = np.asarray(
        [
            [[1.0, -1.0], [-1.0, 1.0]],
            [[0.0, 2.0], [1.0, -1.0]],
        ],
        dtype=np.float64,
    )
    torch_solution = solve_matrix_cfr_plus_torch(
        torch.as_tensor(matrices, dtype=torch.float64), iterations=200
    )

    for index, matrix in enumerate(matrices):
        numpy_solution = solve_matrix_cfr_plus(matrix, iterations=200)
        assert torch_solution.drop_policy[index].numpy() == pytest.approx(
            numpy_solution.drop_policy
        )
        assert torch_solution.check_policy[index].numpy() == pytest.approx(
            numpy_solution.check_policy
        )
        assert torch_solution.value[index].item() == pytest.approx(
            numpy_solution.value
        )


def test_torch_cfr_plus_induced_exact_gap_backpropagates_to_matrix():
    approximate = torch.tensor(
        [[0.2, -0.4], [0.7, 0.1]], dtype=torch.float64, requires_grad=True
    )
    exact = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float64)
    induced = solve_matrix_cfr_plus_torch(approximate, iterations=64)
    lower = torch.min(exact.T @ induced.drop_policy)
    upper = torch.max(exact @ induced.check_policy)
    loss = upper - lower

    loss.backward()

    assert loss.item() > 0.0
    assert approximate.grad is not None
    assert torch.isfinite(approximate.grad).all()
    assert torch.count_nonzero(approximate.grad).item() > 0


def test_torch_cfr_plus_rejects_invalid_iteration_controls():
    matrix = torch.zeros((2, 2))
    with pytest.raises(ValueError, match="iterations"):
        solve_matrix_cfr_plus_torch(matrix, iterations=0)
    with pytest.raises(ValueError, match="averaging delay"):
        solve_matrix_cfr_plus_torch(matrix, iterations=1, averaging_delay=-1)
