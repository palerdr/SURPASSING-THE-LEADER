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

from environment.cfr.evaluator import ValueNetEvaluator
from environment.cfr.mcts import MCTSConfig
from environment.cfr.tablebase import materialize_all, pinned_scenarios
from training.audit_pack import audit_gate, run_audit_pack
from training.calibration import CalibrationReport, evaluate_value_net
from training.train_value_net import TrainConfig, TrainResult, load_checkpoint, make_predict_fn, train
from training.value_targets import (
    SOURCE_EXACT_HORIZON_2,
    SOURCE_EXACT_HORIZON_3,
    SOURCE_MCTS_BOOTSTRAP,
    SOURCE_TABLEBASE,
    SOURCE_TERMINAL,
    ValueTarget,
    generate_mcts_bootstrap_targets,
    save_targets,
)


class CalibrationGateError(RuntimeError):
    """Raised when a generation fails the hard calibration gate."""


@dataclass(frozen=True)
class BootstrapConfig:
    """Hyperparameters for a single bootstrap generation."""

    iterations_per_state: int = 2000
    exploration_c: float = 1.0
    seed: int = 0
    train_config: TrainConfig = TrainConfig()
    required_sources: tuple[str, ...] = (
        SOURCE_TERMINAL,
        SOURCE_TABLEBASE,
        SOURCE_EXACT_HORIZON_2,
        SOURCE_EXACT_HORIZON_3,
    )
    per_source_mse_thresholds: dict[str, float] | None = None
    tablebase_mse_threshold: float = 0.01
    max_unresolved_per_source: float = 0.05
    run_audit_pack: bool = True
    audit_include_holdout: bool = True
    audit_seeds: tuple[int, ...] = (0,)
    audit_max_drift: float = 0.05


@dataclass
class GenerationResult:
    """Outcome of one bootstrap generation.

    ``accepted`` is False only when a ``held_out_targets`` set was supplied
    AND the new gen's held-out MSE strictly exceeds ``prev_gen_holdout_mse``.
    Without a held-out set or a prior-gen baseline, the gate cannot fire and
    ``accepted`` defaults to True.

    Following Silver et al. (2018) AlphaZero §S2: each new generation must
    score at least as well as the previous on a held-out reference set,
    otherwise it is rejected and the previous-gen checkpoint is retained.
    """

    targets_count: int
    bootstrap_count: int
    targets_path: str
    train_result: TrainResult
    accepted: bool = True
    holdout_calibration: CalibrationReport | None = None
    prev_gen_holdout_mse: float | None = None
    audit_path: str | None = None


def enforce_calibration_gate(report: CalibrationReport, config: BootstrapConfig) -> None:
    missing = [source for source in config.required_sources if source not in report.mse_per_source]
    if missing:
        raise CalibrationGateError(f"missing required held-out source classes: {missing}")

    tablebase_mse = report.mse_per_source.get(SOURCE_TABLEBASE)
    if tablebase_mse is not None and tablebase_mse > config.tablebase_mse_threshold:
        raise CalibrationGateError(
            f"tablebase MSE {tablebase_mse:.6f} exceeds threshold {config.tablebase_mse_threshold:.6f}"
        )

    thresholds = config.per_source_mse_thresholds or {}
    for source, threshold in thresholds.items():
        mse = report.mse_per_source.get(source)
        if mse is None:
            continue
        if mse > threshold:
            raise CalibrationGateError(
                f"{source} MSE {mse:.6f} exceeds threshold {threshold:.6f}"
            )

    for source, unresolved in report.mean_unresolved_probability_per_source.items():
        if unresolved > config.max_unresolved_per_source:
            raise CalibrationGateError(
                f"{source} mean unresolved_probability {unresolved:.6f} exceeds "
                f"threshold {config.max_unresolved_per_source:.6f}"
            )


def bootstrap_one_generation(
    *,
    gen_in_checkpoint: str | Path,
    gen_out_targets: str | Path,
    gen_out_dir: str | Path,
    config: BootstrapConfig | None = None,
    grids: dict | None = None,
    held_out_targets: list[ValueTarget] | None = None,
    prev_gen_holdout_mse: float | None = None,
    anchor_targets: list[ValueTarget] | None = None,
) -> GenerationResult:
    """Run one full bootstrap generation: relabel + train + (optional) gate.

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
        held_out_targets: Optional reference set scored against the new
            gen's checkpoint via ``calibration_check``. Required to enable
            the regression-rejection gate.
        prev_gen_holdout_mse: Optional prior-gen MSE on the same
            ``held_out_targets``. If supplied alongside ``held_out_targets``,
            and the new gen's MSE strictly exceeds it, the new generation is
            marked ``accepted=False`` so an outer driver can retain the
            previous checkpoint.

    Returns:
        ``GenerationResult`` with target counts, the targets file
        path, the inner ``TrainResult``, the held-out calibration report
        (if scored), and the rejection flag.
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

    # Anchor with prior-generation exact LP labels. Without this, the
    # gen-N training corpus consists entirely of approximate MCTS labels
    # generated by gen-(N-1)'s evaluator — a self-referential loop that
    # amplifies search noise and drifts away from LP truth. Merging exact
    # anchors pulls training back toward equilibrium values where they
    # are available.
    if anchor_targets:
        targets = targets + anchor_targets

    save_targets(targets, gen_out_targets)
    bootstrap_count = sum(1 for t in targets if t.source == SOURCE_MCTS_BOOTSTRAP)

    train_result = train(gen_out_targets, gen_out_dir, config.train_config)

    holdout_calibration: CalibrationReport | None = None
    accepted = True
    if held_out_targets is not None:
        holdout_calibration = calibration_check(
            checkpoint_path=train_result.checkpoint_path,
            held_out_targets=held_out_targets,
            device=config.train_config.device,
        )
        enforce_calibration_gate(holdout_calibration, config)
        if (
            prev_gen_holdout_mse is not None
            and holdout_calibration.overall_mse > prev_gen_holdout_mse
        ):
            accepted = False

    audit_path: str | None = None
    if config.run_audit_pack:
        trained_model = load_checkpoint(
            train_result.checkpoint_path,
            device=config.train_config.device,
        )
        audit_predict = make_predict_fn(trained_model, device=config.train_config.device)
        audit_evaluator = ValueNetEvaluator(model_fn=audit_predict)
        audit_path = str(Path(gen_out_dir) / "audit_pack.jsonl")
        audit_scenarios = (
            materialize_all() if config.audit_include_holdout else pinned_scenarios()
        )
        audit_records = run_audit_pack(
            audit_scenarios,
            MCTSConfig(
                iterations=config.iterations_per_state,
                exploration_c=config.exploration_c,
                evaluator=None,
                use_tablebase=False,
            ),
            audit_evaluator,
            config.audit_seeds,
            output_path=audit_path,
        )
        audit_gate(audit_records, max_drift=config.audit_max_drift)

    return GenerationResult(
        targets_count=len(targets),
        bootstrap_count=bootstrap_count,
        targets_path=str(gen_out_targets),
        train_result=train_result,
        accepted=accepted,
        holdout_calibration=holdout_calibration,
        prev_gen_holdout_mse=prev_gen_holdout_mse,
        audit_path=audit_path,
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
    torch_device = torch.device(device)

    def feature_predict(features) -> float:
        x = torch.from_numpy(features).unsqueeze(0).to(torch_device)
        with torch.no_grad():
            out = model(x)
            y = (out[0] if isinstance(out, tuple) else out).squeeze().item()
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
    parser.add_argument(
        "--held-out-targets",
        type=str,
        default=None,
        help="Path to .npz file of held-out ValueTarget records. When supplied, "
        "the calibration gate evaluates against these and raises CalibrationGateError "
        "on threshold violations.",
    )
    parser.add_argument(
        "--prev-gen-holdout-mse",
        type=float,
        default=None,
        help="Prior generation's overall_mse on the same held-out set. Enables the "
        "regression-rejection check: new gen with strictly higher MSE marks accepted=False.",
    )
    parser.add_argument(
        "--tablebase-mse-threshold",
        type=float,
        default=0.01,
        help="Hard tablebase MSE ceiling. Default 0.01 (production).",
    )
    parser.add_argument(
        "--max-unresolved-per-source",
        type=float,
        default=0.05,
        help="Hard ceiling on per-source mean unresolved_probability. Default 0.05.",
    )
    parser.add_argument(
        "--audit-max-drift",
        type=float,
        default=0.05,
        help="Audit-pack drift ceiling vs pinned exact/tablebase values.",
    )
    parser.add_argument(
        "--require-source",
        action="append",
        default=None,
        help="Required source class for calibration gate. Repeat for multiple.",
    )
    parser.add_argument(
        "--anchor-targets",
        type=str,
        default=None,
        help="Path to .npz file of exact-LP anchor ValueTarget records (typically "
        "the prior generation's gen0_targets.npz). Merged into the new "
        "generation's training corpus to prevent MCTS-only training drift.",
    )
    args = parser.parse_args()

    held_out_targets = None
    if args.held_out_targets is not None:
        from training.value_targets import load_targets_as_records
        held_out_targets = load_targets_as_records(args.held_out_targets)

    anchor_targets = None
    if args.anchor_targets is not None:
        from training.value_targets import load_targets_as_records
        anchor_targets = load_targets_as_records(args.anchor_targets)

    required_sources = (
        tuple(args.require_source)
        if args.require_source
        else (SOURCE_TERMINAL, SOURCE_TABLEBASE, SOURCE_EXACT_HORIZON_2, SOURCE_EXACT_HORIZON_3)
    )

    cfg = BootstrapConfig(
        iterations_per_state=args.iterations,
        seed=args.seed,
        train_config=TrainConfig(
            epochs=args.epochs, seed=args.seed, device=args.device
        ),
        required_sources=required_sources,
        tablebase_mse_threshold=args.tablebase_mse_threshold,
        max_unresolved_per_source=args.max_unresolved_per_source,
        audit_max_drift=args.audit_max_drift,
    )
    result = bootstrap_one_generation(
        gen_in_checkpoint=args.in_checkpoint,
        gen_out_targets=args.out_targets,
        gen_out_dir=args.out_dir,
        config=cfg,
        held_out_targets=held_out_targets,
        prev_gen_holdout_mse=args.prev_gen_holdout_mse,
        anchor_targets=anchor_targets,
    )
    print(
        f"Generated {result.targets_count} targets "
        f"({result.bootstrap_count} via MCTS bootstrap). "
        f"Trained: best val MSE {result.train_result.best_val_mse:.5f} "
        f"at epoch {result.train_result.best_epoch}. "
        f"accepted={result.accepted}."
    )
    if result.audit_path:
        print(f"Audit pack: {result.audit_path}")
