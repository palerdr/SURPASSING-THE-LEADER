import numpy as np
import torch

from pure.train import grouped_state_split, soft_cross_entropy


def test_grouped_split_prevents_physical_state_leakage():
    states = np.asarray(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [2, 0, 0, 0],
            [3, 0, 0, 0],
        ],
        dtype=np.int16,
    )
    train, validation = grouped_state_split(
        states, validation_fraction=0.25, seed=4
    )
    train_states = {tuple(row) for row in states[train]}
    validation_states = {tuple(row) for row in states[validation]}
    assert train_states.isdisjoint(validation_states)
    assert sorted(np.concatenate((train, validation)).tolist()) == list(range(5))


def test_soft_cross_entropy_prefers_matching_logits():
    target = torch.tensor([[1.0, 0.0]])
    matching = soft_cross_entropy(torch.tensor([[5.0, -5.0]]), target)
    reversed_logits = soft_cross_entropy(torch.tensor([[-5.0, 5.0]]), target)
    assert matching.item() < reversed_logits.item()
