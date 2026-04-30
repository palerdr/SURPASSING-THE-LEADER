"""Phase 3: AlphaZero-style bootstrap orchestration.

One pass of the loop:

    1. Load gen-N checkpoint (or train gen-0 from exact targets).
    2. Run MCTS using the loaded net as the leaf evaluator across a
       corpus of states. Record root values as gen-(N+1) labels.
    3. Train gen-(N+1) on the new labels (optionally combined with
       previous-gen labels).
    4. Evaluate gen-(N+1) calibration; reject if held-out MSE
       regressed vs gen-N.

This module is a thin orchestrator over ``training.value_targets``,
``training.train_value_net``, and ``training.calibration``. The
heavy lifting (training, MCTS, calibration) lives in those modules.

Usage (programmatic):

    bootstrap_one_generation(
        gen_in_checkpoint="checkpoints/gen0/best.pt",
        gen_out_targets="targets/gen1.npz",
        gen_out_dir="checkpoints/gen1",
        iterations_per_state=2000,
    )

The CLI wraps the same call and is documented at ``__main__``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from training.calibration import CalibrationReport, evaluate_value_net
from training.train_value_net import TrainConfig, TrainResult, load_checkpoint, make_predict_fn, train
from training.value_targets import (
    SOURCE_MCTS_BOOTSTRAP,
    ValueTarget,
    generate_mcts_bootstrap_targets,
    save_targets,
)


@dataclass(frozen=True)
class BootstrapConfig:
    """Hyperparameters for a single bootstrap generation."""

    iterations_per_state: int = 2000
    exploration_c: float = 1.0
    seed: int = 0
    train_config: TrainConfig = TrainConfig()


@dataclass
class GenerationResult:
    """Outcome of one bootstrap generation."""

    targets_count: int
    bootstrap_count: int
    targets_path: str
    train_result: TrainResult


def bootstrap_one_generation(
    *,
    gen_in_checkpoint: str | Path,
    gen_out_targets: str | Path,
    gen_out_dir: str | Path,
    config: BootstrapConfig | None = None,
    grids: dict | None = None,
) -> GenerationResult:
    """Run one full bootstrap generation: relabel + train.

    Args:
        gen_in_checkpoint: Path to the previous generation's saved
            ``state_dict``. Used as the leaf evaluator during MCTS.
        gen_out_targets: Path where the new generation's ``.npz``
            target file is written.
        gen_out_dir: Output directory for the trained-on-new-targets
            checkpoint and log (passed straight to ``train``).
        config: Bootstrap hyperparameters.
        grids: Optional dict of axis grid overrides for
            ``generate_mcts_bootstrap_targets``. None means default
            corpus grids.

    Returns:
        ``GenerationResult`` with target counts, the targets file
        path, and the inner ``TrainResult``.
    """
    config = config or BootstrapConfig()
    Path(gen_out_dir).mkdir(parents=True, exist_ok=True)

    model = load_checkpoint(gen_in_checkpoint, device=config.train_config.device)
    predict_fn = make_predict_fn(model, device=config.train_config.device)

    grid_kwargs = grids or {}
    targets: list[ValueTarget] = generate_mcts_bootstrap_targets(
        predict_fn,
        iterations_per_state=config.iterations_per_state,
        exploration_c=config.exploration_c,
        seed=config.seed,
        **grid_kwargs,
    )

    save_targets(targets, gen_out_targets)
    bootstrap_count = sum(1 for t in targets if t.source == SOURCE_MCTS_BOOTSTRAP)

    train_result = train(gen_out_targets, gen_out_dir, config.train_config)

    return GenerationResult(
        targets_count=len(targets),
        bootstrap_count=bootstrap_count,
        targets_path=str(gen_out_targets),
        train_result=train_result,
    )


def calibration_check(
    *,
    checkpoint_path: str | Path,
    held_out_targets: list[ValueTarget],
    device: str = "cpu",
) -> CalibrationReport:
    """Score a generation's checkpoint against a held-out target set.

    Wrapper that loads the checkpoint, builds a ``predict_fn`` over the
    raw feature vector (not over a Game object — held-out targets carry
    pre-extracted features), and feeds it to
    ``calibration.evaluate_value_net``.
    """
    import torch

    model = load_checkpoint(checkpoint_path, device=device)

    def feature_predict(features) -> float:
        x = torch.from_numpy(features).unsqueeze(0)
        with torch.no_grad():
            y = model(x).squeeze().item()
        return float(y)

    return evaluate_value_net(feature_predict, held_out_targets)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-checkpoint", required=True)
    parser.add_argument("--out-targets", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = BootstrapConfig(
        iterations_per_state=args.iterations,
        seed=args.seed,
        train_config=TrainConfig(
            epochs=args.epochs, seed=args.seed, device=args.device
        ),
    )
    result = bootstrap_one_generation(
        gen_in_checkpoint=args.in_checkpoint,
        gen_out_targets=args.out_targets,
        gen_out_dir=args.out_dir,
        config=cfg,
    )
    print(
        f"Generated {result.targets_count} targets "
        f"({result.bootstrap_count} via MCTS bootstrap). "
        f"Trained: best val MSE {result.train_result.best_val_mse:.5f} "
        f"at epoch {result.train_result.best_epoch}."
    )
