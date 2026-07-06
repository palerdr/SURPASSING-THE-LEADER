import torch
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import MSELoss

from stl.learning.model import ValueNet, FEATURE_DIM, value_output
from stl.learning.self_play import Experience


def train_value_net(
        net: ValueNet,
        dataset: list[Experience],
        epochs: int = 20,
        batch_size: int = 256,
        lr: float = 1e-3,
) -> list[float]:
    N = len(dataset)

    features = torch.tensor(np.stack([exp.features for exp in dataset]))
    outcomes = torch.tensor([[exp.outcome] for exp in dataset], dtype=torch.float32)

    optimizer = Adam(net.parameters(), lr=lr)
    total_steps = epochs * ((N + batch_size - 1) // batch_size)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.01)
    loss_fn = MSELoss()
    history = []

    for epoch in range(epochs):
        indices = torch.randperm(N)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, N, batch_size):
            batch_idx = indices[start : start + batch_size]
            batch_features = features[batch_idx]
            batch_outcomes = outcomes[batch_idx]

            predictions = value_output(net(batch_features))
            loss = loss_fn(predictions, batch_outcomes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        history.append(avg_loss)

    return history


def save_checkpoint(net: ValueNet, path: str) -> None:                         
    return torch.save(net.state_dict(), path)                                          
                                                                                 
                                                                                  
def load_checkpoint(path: str, n_features: int = FEATURE_DIM) -> ValueNet:
    """Load a ValueNet checkpoint, inferring hidden width from the file.

    Fails loudly on any unrecognized or partially-loadable checkpoint —
    the previous implementation fell through to ``load_state_dict({},
    strict=False)`` for modern ``trunk.*``-keyed checkpoints and silently
    returned a random-weight net.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)

    if "trunk.0.weight" in state:
        hidden_dim = int(state["trunk.0.weight"].shape[0])
        net = ValueNet(n_features, hidden_dim=hidden_dim)
        net.load_state_dict(state)  # strict: any mismatch raises
    elif any(key.startswith("layers.") for key in state):
        # Legacy pre-policy-head checkpoints: layers.{0,2,4} -> trunk/value_head.
        migrated = {}
        for key, value in state.items():
            if key.startswith("layers.0."):
                migrated[key.replace("layers.0.", "trunk.0.")] = value
            elif key.startswith("layers.2."):
                migrated[key.replace("layers.2.", "trunk.2.")] = value
            elif key.startswith("layers.4."):
                migrated[key.replace("layers.4.", "value_head.0.")] = value
        if "trunk.0.weight" not in migrated:
            raise RuntimeError(
                f"Legacy checkpoint at {path} has no layers.0 weights; "
                f"keys: {sorted(state)[:6]}"
            )
        hidden_dim = int(migrated["trunk.0.weight"].shape[0])
        net = ValueNet(n_features, hidden_dim=hidden_dim)
        missing, unexpected = net.load_state_dict(migrated, strict=False)
        # The ONLY tolerated gap: legacy checkpoints predate the policy head.
        bad_missing = [k for k in missing if not k.startswith("policy_head.")]
        if bad_missing or unexpected:
            raise RuntimeError(
                f"Checkpoint at {path} did not migrate cleanly: "
                f"missing={bad_missing}, unexpected={list(unexpected)}"
            )
    else:
        raise RuntimeError(
            f"Unrecognized checkpoint format at {path}; keys: {sorted(state)[:6]}"
        )
    net.eval()
    return net
