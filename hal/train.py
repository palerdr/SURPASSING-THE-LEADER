import torch
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import MSELoss

from .value_net import ValueNet, FEATURE_DIM
from .self_play import Experience


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

            predictions = net(batch_features)
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
      net = ValueNet(n_features)
      net.load_state_dict(torch.load(path))
      net.eval()
      return net
