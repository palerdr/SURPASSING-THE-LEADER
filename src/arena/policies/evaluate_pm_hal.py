"""Predeclared synthetic evaluation for Perfect Mode Hal.

The repeated-opponent identity is the experimental unit.  Every scenario uses
the same game seeds, start clocks, and candidate seatings for PM Hal, Perfect
Hal, Adaptive Hal, and Exact DTH.  Synthetic truth is used only by this
measurement wrapper after PM Hal has fixed its policy.

This report deliberately separates three claim classes:

* proper forecast scores against the synthetic policy that actually acted;
* game outcomes paired and bootstrapped by opponent identity; and
* independently recomputed local worst-case loss from PM Hal's live decisions.

It is not human evidence.  Human sessions do not expose a simulator truth
distribution, so their future protocol must use realized prequential scores and
participant-level uncertainty instead.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import numpy as np

from arena.contracts import CanonicalDecision, PublicGameOutcome, PublicHalfRound
from arena.dth_adapter import project_to_dth_state
from arena.match import play_match_game
from arena.policies.adaptive import (
    AdaptiveDTHPolicyProvider,
    DirichletPrior,
    ExploitationConfig,
    RoleDirichletOpponent,
)
from arena.policies.aggro_hal import (
    AggroHalPolicyProvider,
    dth_compatibility,
    load_checkpoint,
)
from arena.policies.opponent_league import (
    ACTIONS,
    SUPPORTED_FAMILIES,
    ReactiveDTHOpponent,
    make_opponent,
)
from arena.policies.perfect_hal import PerfectHalPolicyProvider
from arena.policies.pm_hal import (
    ACTION_COUNT,
    DEFAULT_PM_HAL_CONFIG,
    PMHalConfig,
    PMHalDecision,
    PMHalPolicyProvider,
    load_pm_hal_config,
)
from dth.agent import CompleteDTHAgent

LEGACY_EVALUATION_SCHEMA = "arena-pm-hal-evaluation-v1"
EVALUATION_SCHEMA = "arena-pm-hal-evaluation-v2"
LEGACY_CONFIG_SCHEMA = "arena-pm-hal-evaluation-config-v1"
CONFIG_SCHEMA = "arena-pm-hal-evaluation-config-v2"
DEFAULT_CONFIG = Path("src/arena/config/pm_hal_evaluation_v3.json")
DEFAULT_ARTIFACT_DIR = Path("src/dth/artifacts/complete_full_v1")
DEFAULT_AGGRO_CHECKPOINT = Path("outputs/pm-hal/aggro-component-v1/checkpoint.pt")
_AUDIT_TOLERANCE = 1e-8


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_registered_protocol(
    config_path: str | Path,
    pm_config_path: str | Path,
) -> dict[str, object]:
    """Verify that a confirmation protocol is executing from one clean commit."""

    try:
        root_result = subprocess.run(
            ("git", "rev-parse", "--show-toplevel"),
            cwd=Path(config_path).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "Git-registered PM confirmation requires an accessible Git worktree"
        ) from error
    root = Path(root_result.stdout.strip()).resolve()

    def git(*arguments: str, text: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=True,
            capture_output=True,
            text=text,
        )

    status = git("status", "--porcelain", "--untracked-files=all").stdout.strip()
    if status:
        raise RuntimeError(
            "Git-registered PM confirmation requires a clean worktree before execution"
        )
    head = git("rev-parse", "HEAD").stdout.strip()
    if len(head) != 40:
        raise RuntimeError("PM confirmation could not resolve a full Git commit")

    registered_files: dict[str, dict[str, object]] = {}
    for label, path in (
        ("evaluation_config", config_path),
        ("pm_config", pm_config_path),
    ):
        resolved = Path(path).resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError as error:
            raise RuntimeError(
                f"Git-registered PM {label} must be inside the repository"
            ) from error
        repository_path = relative.as_posix()
        try:
            git("ls-files", "--error-unmatch", "--", repository_path)
            committed = git("show", f"{head}:{repository_path}", text=False).stdout
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"Git-registered PM {label} is not present in commit {head}"
            ) from error
        current = resolved.read_bytes()
        if current != committed:
            raise RuntimeError(f"Git-registered PM {label} differs from commit {head}")
        registered_files[label] = {
            "path": repository_path,
            "sha256": hashlib.sha256(current).hexdigest(),
        }
    return {
        "commit": head,
        "clean_worktree": True,
        "registered_files": registered_files,
    }


def _distribution(raw: object, *, label: str) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64)
    if (
        values.shape != (ACTION_COUNT,)
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or float(values.sum()) <= 0.0
    ):
        raise ValueError(f"{label} must be a finite nonnegative length-60 distribution")
    return values / float(values.sum())


@dataclass(frozen=True, slots=True)
class PMEvaluationEntry:
    family: str
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.family not in SUPPORTED_FAMILIES:
            raise ValueError(f"unsupported PM benchmark family {self.family!r}")
        if not self.seeds or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in self.seeds
        ):
            raise ValueError("PM benchmark seeds must be nonnegative integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("PM benchmark seeds must be unique within a family")


@dataclass(frozen=True, slots=True)
class PMPromotionGate:
    minimum_opponent_identities: int
    maximum_risk_violations: int
    minimum_clustered_lower_bound_vs_exact: float
    minimum_clustered_lower_bound_vs_strongest_component: float
    minimum_clustered_lower_bound_vs_no_aggro: float | None = None
    minimum_point_difference_every_seat: float = -1.0
    minimum_point_difference_every_family: float = -1.0

    def __post_init__(self) -> None:
        if self.minimum_opponent_identities <= 0:
            raise ValueError("promotion identity minimum must be positive")
        if self.maximum_risk_violations < 0:
            raise ValueError("maximum risk violations must be nonnegative")
        bounds = (
            self.minimum_clustered_lower_bound_vs_exact,
            self.minimum_clustered_lower_bound_vs_strongest_component,
            self.minimum_point_difference_every_seat,
            self.minimum_point_difference_every_family,
        )
        if self.minimum_clustered_lower_bound_vs_no_aggro is not None:
            bounds += (self.minimum_clustered_lower_bound_vs_no_aggro,)
        if any(not -1.0 <= float(value) <= 1.0 for value in bounds):
            raise ValueError("promotion comparison bounds must lie in [-1, 1]")


@dataclass(frozen=True, slots=True)
class PMEvaluationConfig:
    status: str
    benchmark_id: str
    entries: tuple[PMEvaluationEntry, ...]
    sessions_per_opponent: int
    seat_pairs_per_session: int
    start_clocks: tuple[int, ...]
    max_half_rounds: int
    evaluation_seed: int
    bootstrap_seed: int
    bootstrap_samples: int
    promotion_gate: PMPromotionGate
    notes: tuple[str, ...]
    expected_pm_config_sha256: str | None = None
    expected_aggro_checkpoint_sha256: str | None = None
    expected_dth_table_digest: str | None = None
    schema_version: str = CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version not in {LEGACY_CONFIG_SCHEMA, CONFIG_SCHEMA}:
            raise ValueError("unsupported PM evaluation config schema")
        required_status = (
            "predeclared-before-results"
            if self.schema_version == LEGACY_CONFIG_SCHEMA
            else "git-registered-before-execution"
        )
        if self.status != required_status:
            raise ValueError(
                f"PM benchmark status must be {required_status!r} for its schema"
            )
        if not self.benchmark_id or not self.entries:
            raise ValueError("PM benchmark id and entries must be nonempty")
        families = [entry.family for entry in self.entries]
        if len(set(families)) != len(families):
            raise ValueError("PM benchmark families must be unique")
        positive = (
            self.sessions_per_opponent,
            self.seat_pairs_per_session,
            self.max_half_rounds,
            self.bootstrap_samples,
        )
        if any(isinstance(value, bool) or value <= 0 for value in positive):
            raise ValueError("PM evaluation counts must be positive")
        if not self.start_clocks or any(clock < 0 for clock in self.start_clocks):
            raise ValueError("PM start clocks must be nonnegative")
        if self.evaluation_seed < 0 or self.bootstrap_seed < 0:
            raise ValueError("PM evaluation seeds must be nonnegative")
        for label, digest in (
            ("expected_pm_config_sha256", self.expected_pm_config_sha256),
            (
                "expected_aggro_checkpoint_sha256",
                self.expected_aggro_checkpoint_sha256,
            ),
            ("expected_dth_table_digest", self.expected_dth_table_digest),
        ):
            if digest is not None and (
                len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{label} must be a lowercase SHA-256 digest")

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(entry.family for entry in self.entries)

    @property
    def opponent_identities(self) -> int:
        return sum(len(entry.seeds) for entry in self.entries)


def load_evaluation_config(path: str | Path = DEFAULT_CONFIG) -> PMEvaluationConfig:
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("PM evaluation config must be a JSON object")
    entries_raw = raw.get("families")
    gate_raw = raw.get("promotion_gate")
    if not isinstance(entries_raw, list) or not isinstance(gate_raw, dict):
        raise ValueError("PM evaluation config lacks families or promotion gate")
    entries = tuple(
        PMEvaluationEntry(
            family=str(entry["family"]),
            seeds=tuple(int(seed) for seed in entry["seeds"]),
        )
        for entry in entries_raw
    )
    schema = str(raw.get("schema_version"))
    bindings = raw.get("bindings", {})
    if not isinstance(bindings, dict):
        raise ValueError("PM evaluation config bindings must be an object")
    return PMEvaluationConfig(
        schema_version=schema,
        status=str(raw.get("status")),
        benchmark_id=str(raw.get("benchmark_id")),
        entries=entries,
        sessions_per_opponent=int(raw.get("sessions_per_opponent")),
        seat_pairs_per_session=int(raw.get("seat_pairs_per_session")),
        start_clocks=tuple(int(clock) for clock in raw.get("start_clocks", ())),
        max_half_rounds=int(raw.get("max_half_rounds")),
        evaluation_seed=int(raw.get("evaluation_seed")),
        bootstrap_seed=int(raw.get("bootstrap_seed")),
        bootstrap_samples=int(raw.get("bootstrap_samples")),
        promotion_gate=PMPromotionGate(**gate_raw),
        notes=tuple(str(note) for note in raw.get("notes", ())),
        expected_pm_config_sha256=(
            str(bindings["pm_config_sha256"])
            if "pm_config_sha256" in bindings
            else None
        ),
        expected_aggro_checkpoint_sha256=(
            str(bindings["aggro_checkpoint_sha256"])
            if "aggro_checkpoint_sha256" in bindings
            else None
        ),
        expected_dth_table_digest=(
            str(bindings["dth_table_digest"])
            if "dth_table_digest" in bindings
            else None
        ),
    )


class _TruthTrackingOpponent:
    """Expose synthetic truth to the evaluator, never to the live candidate."""

    def __init__(self, opponent: ReactiveDTHOpponent) -> None:
        self.opponent = opponent
        self.pending_decision: CanonicalDecision | None = None
        self.pending_truth: np.ndarray | None = None

    def reset_session(self) -> None:
        if self.pending_truth is not None:
            raise RuntimeError("synthetic truth remained pending at session reset")
        self.pending_decision = None
        self.opponent.reset_session()

    def reset_game(self) -> None:
        if self.pending_truth is not None:
            raise RuntimeError("synthetic truth remained pending at game reset")
        self.opponent.reset_game()

    def policy(self, decision: CanonicalDecision) -> Mapping[int, float]:
        if self.pending_truth is not None:
            raise RuntimeError("synthetic opponent acted twice before reveal")
        self.pending_decision = decision
        self.pending_truth = _distribution(
            self.opponent.true_distribution(decision),
            label="synthetic opponent truth",
        )
        return self.opponent.policy(decision)

    def truth_for(self, candidate_decision: CanonicalDecision) -> np.ndarray:
        opposite = _opponent_decision(candidate_decision)
        if self.pending_truth is None:
            return _distribution(
                self.opponent.true_distribution(opposite),
                label="synthetic opponent truth",
            )
        if self.pending_decision is None:
            raise RuntimeError("pending synthetic truth has no decision")
        pending = self.pending_decision
        if (
            pending.role != opposite.role
            or pending.actor_name.casefold() != opposite.actor_name.casefold()
            or pending.legal_seconds != opposite.legal_seconds
        ):
            raise RuntimeError("pending synthetic truth belongs to another decision")
        return np.asarray(self.pending_truth, dtype=np.float64).copy()

    def observe(self, record: PublicHalfRound) -> None:
        self.opponent.observe(record)
        self.pending_decision = None
        self.pending_truth = None

    def end_game(self, outcome: PublicGameOutcome) -> None:
        self.opponent.end_game(outcome)


def _opponent_decision(decision: CanonicalDecision) -> CanonicalDecision:
    if decision.actor_name.casefold() == "hal":
        opponent_name = "Baku"
    elif decision.actor_name.casefold() == "baku":
        opponent_name = "Hal"
    else:
        raise ValueError("PM evaluation requires canonical Hal/Baku seats")
    return CanonicalDecision(
        role="checker" if decision.role == "dropper" else "dropper",
        actor_name=opponent_name,
        turn_duration=decision.turn_duration,
        legal_seconds=ACTIONS,
        checker_cylinder_seconds=decision.checker_cylinder_seconds,
        checker_ttd_seconds=decision.checker_ttd_seconds,
        dropper_cylinder_seconds=decision.dropper_cylinder_seconds,
        dropper_ttd_seconds=decision.dropper_ttd_seconds,
        native_state=decision.native_state,
    )


class _ExactProvider:
    def __init__(self, agent: CompleteDTHAgent) -> None:
        self.agent = agent

    def policy(self, decision: CanonicalDecision) -> Mapping[int, float]:
        stage = self.agent.stage_game(project_to_dth_state(decision))
        policy = stage.drop_policy if decision.role == "dropper" else stage.check_policy
        return {
            action: float(value)
            for action, value in enumerate(policy, start=1)
            if value > 0.0
        }


@dataclass(frozen=True, slots=True)
class _PMPendingMeasurement:
    role: str
    opponent_role: str
    self_name: str
    prediction: np.ndarray
    truth: np.ndarray
    candidate_policy: np.ndarray
    oriented_values: np.ndarray
    decision: PMHalDecision
    truth_shift: bool
    session_decision_index: int


@dataclass(frozen=True, slots=True)
class PMPredictionSample:
    family: str
    opponent_seed: int
    replicate: int
    game_index: int
    session_decision_index: int
    role: str
    opponent_role: str
    realized_nll: float
    expected_nll: float
    realized_brier: float
    expected_brier: float
    component_expected_nll: tuple[tuple[str, float], ...]
    component_expected_brier: tuple[tuple[str, float], ...]
    oracle_regret: float
    top_confidence: float
    top_correct: bool
    forecast_confidence: float
    forecast_disagreement: float
    mode: str
    actual_worst_case_loss: float
    budget_charge: float
    truth_shift: bool
    change_detected: bool


class _MeasuredPMProvider:
    def __init__(
        self,
        provider: PMHalPolicyProvider,
        opponent: _TruthTrackingOpponent,
        *,
        family: str,
        opponent_seed: int,
        replicate: int,
    ) -> None:
        self.provider = provider
        self.opponent = opponent
        self.family = family
        self.opponent_seed = int(opponent_seed)
        self.replicate = int(replicate)
        self.measurements: list[PMPredictionSample] = []
        self._pending: _PMPendingMeasurement | None = None
        self._previous_truth = {
            "dropper": None,
            "checker": None,
        }
        self._session_decision_index = 0

    def reset_session(self) -> None:
        self.provider.reset_session()
        self.measurements.clear()
        self._pending = None
        self._previous_truth = {"dropper": None, "checker": None}
        self._session_decision_index = 0

    def reset_game(self) -> None:
        if self._pending is not None:
            raise RuntimeError("PM measurement remained pending at game reset")
        self.provider.reset_game()

    def policy(self, decision: CanonicalDecision) -> Mapping[int, float]:
        if self._pending is not None:
            raise RuntimeError("PM measurement acted twice before reveal")
        truth = self.opponent.truth_for(decision)
        raw = self.provider.policy(decision)
        diagnostic = self.provider.last_decision
        stage = self.provider.pending_stage_game
        if diagnostic is None or stage is None:
            raise RuntimeError("PM Hal did not publish decision evidence")
        prediction = _distribution(
            diagnostic.opponent_policy, label="PM evaluation forecast"
        )
        candidate = _distribution(diagnostic.policy, label="PM evaluation policy")
        role = decision.role
        opponent_role = "checker" if role == "dropper" else "dropper"
        matrix = np.asarray(stage.matrix, dtype=np.float64)
        oriented = matrix @ truth if role == "dropper" else -(matrix.T @ truth)
        previous = self._previous_truth[opponent_role]
        truth_shift = (
            previous is not None
            and 0.5 * float(np.sum(np.abs(truth - previous))) >= 0.25
        )
        self._previous_truth[opponent_role] = truth.copy()
        self._pending = _PMPendingMeasurement(
            role=role,
            opponent_role=opponent_role,
            self_name=decision.actor_name,
            prediction=prediction,
            truth=truth,
            candidate_policy=candidate,
            oriented_values=oriented,
            decision=diagnostic,
            truth_shift=bool(truth_shift),
            session_decision_index=self._session_decision_index,
        )
        self._session_decision_index += 1
        return raw

    def observe(self, record: PublicHalfRound) -> None:
        pending = self._pending
        if pending is None:
            raise RuntimeError("PM measurement received reveal without prediction")
        self_name = pending.self_name.casefold()
        if pending.role == "dropper":
            if record.dropper_name.casefold() != self_name:
                raise RuntimeError("PM measurement reveal has wrong Dropper")
            opponent_action = int(record.check_time)
        else:
            if record.checker_name.casefold() != self_name:
                raise RuntimeError("PM measurement reveal has wrong Checker")
            opponent_action = int(record.drop_time)
        if not 1 <= opponent_action <= ACTION_COUNT:
            raise ValueError("PM evaluation supports opponent actions 1..60 only")
        index = opponent_action - 1
        self.provider.observe(record)
        live = self.provider.observation_metrics[-1]
        prediction = pending.prediction
        truth = pending.truth
        expected_brier = float(1.0 + prediction @ prediction - 2.0 * prediction @ truth)
        candidate_value = float(pending.candidate_policy @ pending.oriented_values)
        oracle_regret = max(
            0.0, float(np.max(pending.oriented_values)) - candidate_value
        )
        component_policies = tuple(
            (
                name,
                _distribution(policy, label=f"PM {name} component forecast"),
            )
            for name, policy in pending.decision.component_policies
        )
        self.measurements.append(
            PMPredictionSample(
                family=self.family,
                opponent_seed=self.opponent_seed,
                replicate=self.replicate,
                game_index=pending.decision.game_index,
                session_decision_index=pending.session_decision_index,
                role=pending.role,
                opponent_role=pending.opponent_role,
                realized_nll=-math.log(max(float(prediction[index]), 1e-12)),
                expected_nll=float(
                    -np.sum(truth * np.log(np.clip(prediction, 1e-12, 1.0)))
                ),
                realized_brier=float(
                    1.0 + prediction @ prediction - 2.0 * prediction[index]
                ),
                expected_brier=expected_brier,
                component_expected_nll=tuple(
                    (
                        name,
                        float(-np.sum(truth * np.log(np.clip(policy, 1e-12, 1.0)))),
                    )
                    for name, policy in component_policies
                ),
                component_expected_brier=tuple(
                    (
                        name,
                        float(1.0 + policy @ policy - 2.0 * policy @ truth),
                    )
                    for name, policy in component_policies
                ),
                oracle_regret=oracle_regret,
                top_confidence=float(np.max(prediction)),
                top_correct=bool(int(np.argmax(prediction)) == index),
                forecast_confidence=pending.decision.confidence,
                forecast_disagreement=pending.decision.forecast_disagreement,
                mode=pending.decision.mode,
                actual_worst_case_loss=(
                    pending.decision.selected_actual_worst_case_loss
                ),
                budget_charge=pending.decision.selected_budget_charge,
                truth_shift=pending.truth_shift,
                change_detected=(
                    live.post_reveal_change_probability
                    >= self.provider.config.change_shield_threshold
                ),
            )
        )
        self._pending = None

    def end_game(self, outcome: PublicGameOutcome) -> None:
        self.provider.end_game(outcome)


def _run_session(
    *,
    controller: str,
    candidate: object,
    opponent: _TruthTrackingOpponent,
    session_seed: int,
    seat_pairs: int,
    start_clocks: Sequence[int],
    max_half_rounds: int,
    candidate_starts_first_seat: bool,
) -> dict[str, object]:
    reset_session = getattr(candidate, "reset_session", None)
    if callable(reset_session):
        reset_session()
    opponent.reset_session()
    games: list[dict[str, object]] = []
    for pair_index in range(seat_pairs):
        game_seed = int(session_seed + pair_index)
        start_clock = int(start_clocks[pair_index % len(start_clocks)])
        first = (
            candidate_starts_first_seat
            if pair_index % 2 == 0
            else not candidate_starts_first_seat
        )
        for candidate_first in (first, not first):
            game_index = len(games)
            winner, half_rounds = play_match_game(
                candidate if candidate_first else opponent,
                opponent if candidate_first else candidate,
                seed=game_seed,
                start_clock=start_clock,
                max_half_rounds=max_half_rounds,
                game_index=game_index,
                pure_dth=True,
            )
            candidate_seat = "Hal" if candidate_first else "Baku"
            games.append(
                {
                    "game_index": game_index,
                    "seat_pair_index": pair_index,
                    "seed": game_seed,
                    "start_clock": start_clock,
                    "candidate_seat": candidate_seat,
                    "winner_seat": winner,
                    "won": None if winner is None else winner == candidate_seat,
                    "half_rounds": int(half_rounds),
                }
            )
    return {"controller": controller, "games": games}


def _wilson_interval(successes: int, trials: int) -> list[float] | None:
    if trials <= 0:
        return None
    z = 1.959963984540054
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    center = (rate + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _score(won: object) -> float:
    if won is True:
        return 1.0
    if won is False:
        return 0.0
    if won is None:
        return 0.5
    raise ValueError("game outcome must be true, false, or null")


def _rate_summary(games: Sequence[Mapping[str, object]]) -> dict[str, object]:
    wins = sum(game["won"] is True for game in games)
    losses = sum(game["won"] is False for game in games)
    stopped = sum(game["won"] is None for game in games)
    decisive = wins + losses
    return {
        "games": len(games),
        "decisive_games": decisive,
        "wins": wins,
        "losses": losses,
        "stopped": stopped,
        "decisive_win_rate": wins / decisive if decisive else None,
        "decisive_win_rate_wilson_95": _wilson_interval(wins, decisive),
        "all_game_score": (
            float(np.mean([_score(game["won"]) for game in games])) if games else None
        ),
        "stop_rate": stopped / len(games) if games else None,
        "mean_half_rounds": (
            float(np.mean([int(game["half_rounds"]) for game in games]))
            if games
            else None
        ),
    }


def _controller_summary(
    sessions: Sequence[Mapping[str, object]],
    families: Sequence[str],
) -> dict[str, object]:
    games = [game for session in sessions for game in session["games"]]
    result = _rate_summary(games)
    result["sessions"] = len(sessions)
    result["by_seat"] = {
        seat: _rate_summary([game for game in games if game["candidate_seat"] == seat])
        for seat in ("Hal", "Baku")
    }
    result["by_family"] = {
        family: _rate_summary(
            [
                game
                for session in sessions
                if session["opponent_family"] == family
                for game in session["games"]
            ]
        )
        for family in families
    }
    return result


def _paired_comparison(
    left_sessions: Sequence[Mapping[str, object]],
    right_sessions: Sequence[Mapping[str, object]],
    *,
    left_name: str,
    right_name: str,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, object]:
    def session_key(session: Mapping[str, object]) -> tuple[object, ...]:
        return (
            session["opponent_family"],
            session["opponent_seed"],
            session["replicate"],
        )

    right_by_key = {session_key(session): session for session in right_sessions}
    if len(right_by_key) != len(right_sessions):
        raise RuntimeError("paired comparison has duplicate right sessions")
    clusters: dict[tuple[object, object], list[float]] = {}
    scenarios = 0
    for left in left_sessions:
        key = session_key(left)
        try:
            right = right_by_key.pop(key)
        except KeyError as error:
            raise RuntimeError("paired controllers lack a common session") from error

        def games_by_key(
            session: Mapping[str, object],
        ) -> dict[tuple[object, ...], object]:
            return {
                (
                    game["seat_pair_index"],
                    game["seed"],
                    game["start_clock"],
                    game["candidate_seat"],
                ): game["won"]
                for game in session["games"]
            }

        left_games = games_by_key(left)
        right_games = games_by_key(right)
        if left_games.keys() != right_games.keys():
            raise RuntimeError("paired controllers lack common game scenarios")
        differences = clusters.setdefault((key[0], key[1]), [])
        for game_key in left_games:
            differences.append(
                _score(left_games[game_key]) - _score(right_games[game_key])
            )
            scenarios += 1
    if right_by_key:
        raise RuntimeError("paired comparison has unmatched right sessions")
    cluster_means = np.asarray(
        [float(np.mean(values)) for values in clusters.values()], dtype=np.float64
    )
    if not len(cluster_means):
        interval = None
        mean = None
    else:
        rng = np.random.default_rng(bootstrap_seed)
        draws = rng.choice(
            cluster_means,
            size=(bootstrap_samples, len(cluster_means)),
            replace=True,
        ).mean(axis=1)
        interval = [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ]
        mean = float(np.mean(cluster_means))
    return {
        "left": left_name,
        "right": right_name,
        "stop_score": 0.5,
        "experimental_unit": "opponent_family_and_parameter_seed",
        "opponent_identity_units": int(len(cluster_means)),
        "common_game_scenarios": scenarios,
        "mean_left_minus_right": mean,
        "cluster_bootstrap_95": interval,
        "bootstrap_samples": bootstrap_samples,
    }


def _clustered_prediction_interval(
    samples: Sequence[PMPredictionSample],
    values: Sequence[float],
    *,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, object]:
    if len(samples) != len(values):
        raise ValueError("prediction samples and values must have equal length")
    grouped: dict[tuple[str, int], list[float]] = {}
    for sample, value in zip(samples, values, strict=True):
        grouped.setdefault((sample.family, sample.opponent_seed), []).append(
            float(value)
        )
    cluster_means = np.asarray(
        [float(np.mean(cluster)) for cluster in grouped.values()],
        dtype=np.float64,
    )
    if not len(cluster_means):
        return {
            "experimental_unit": "opponent_family_and_parameter_seed",
            "opponent_identity_units": 0,
            "cluster_weighted_mean": None,
            "cluster_bootstrap_95": None,
            "bootstrap_samples": bootstrap_samples,
        }
    rng = np.random.default_rng(bootstrap_seed)
    draws = rng.choice(
        cluster_means,
        size=(bootstrap_samples, len(cluster_means)),
        replace=True,
    ).mean(axis=1)
    return {
        "experimental_unit": "opponent_family_and_parameter_seed",
        "opponent_identity_units": int(len(cluster_means)),
        "cluster_weighted_mean": float(np.mean(cluster_means)),
        "cluster_bootstrap_95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "bootstrap_samples": bootstrap_samples,
    }


def _prediction_metrics(
    samples: Sequence[PMPredictionSample],
    *,
    include_role_slices: bool = True,
    bootstrap_seed: int | None = None,
    bootstrap_samples: int | None = None,
) -> dict[str, object]:
    if (bootstrap_seed is None) != (bootstrap_samples is None):
        raise ValueError(
            "prediction bootstrap seed and samples must be supplied together"
        )
    if not samples:
        return {
            "decisions": 0,
            "realized_nll": None,
            "expected_nll": None,
            "realized_brier": None,
            "expected_brier": None,
            "mean_one_step_oracle_regret": None,
        }
    expected_nll = float(np.mean([sample.expected_nll for sample in samples]))
    expected_brier = float(np.mean([sample.expected_brier for sample in samples]))
    result: dict[str, object] = {
        "decisions": len(samples),
        "realized_nll": float(np.mean([sample.realized_nll for sample in samples])),
        "expected_nll": expected_nll,
        "realized_brier": float(np.mean([sample.realized_brier for sample in samples])),
        "expected_brier": expected_brier,
        "uniform_baseline": {
            "expected_nll": math.log(ACTION_COUNT),
            "expected_brier": (ACTION_COUNT - 1.0) / ACTION_COUNT,
            "expected_nll_improvement": math.log(ACTION_COUNT) - expected_nll,
            "expected_brier_improvement": (
                (ACTION_COUNT - 1.0) / ACTION_COUNT - expected_brier
            ),
        },
        "top_label_accuracy": float(
            np.mean([sample.top_correct for sample in samples])
        ),
        "mean_top_confidence": float(
            np.mean([sample.top_confidence for sample in samples])
        ),
        "mean_forecast_confidence": float(
            np.mean([sample.forecast_confidence for sample in samples])
        ),
        "mean_forecast_disagreement": float(
            np.mean([sample.forecast_disagreement for sample in samples])
        ),
        "mean_one_step_oracle_regret": float(
            np.mean([sample.oracle_regret for sample in samples])
        ),
    }
    component_names = tuple(name for name, _ in samples[0].component_expected_nll)
    if any(
        tuple(name for name, _ in sample.component_expected_nll) != component_names
        or tuple(name for name, _ in sample.component_expected_brier) != component_names
        for sample in samples
    ):
        raise RuntimeError("PM component prediction banks changed within a benchmark")
    result["component_prediction_metrics"] = {
        name: {
            "expected_nll": (
                component_nll := float(
                    np.mean(
                        [
                            dict(sample.component_expected_nll)[name]
                            for sample in samples
                        ]
                    )
                )
            ),
            "expected_brier": (
                component_brier := float(
                    np.mean(
                        [
                            dict(sample.component_expected_brier)[name]
                            for sample in samples
                        ]
                    )
                )
            ),
            "fused_minus_component_expected_nll": expected_nll - component_nll,
            "fused_minus_component_expected_brier": (expected_brier - component_brier),
        }
        for name in component_names
    }
    if bootstrap_seed is not None and bootstrap_samples is not None:
        result["opponent_identity_clustered"] = {
            "expected_nll": _clustered_prediction_interval(
                samples,
                [sample.expected_nll for sample in samples],
                bootstrap_seed=bootstrap_seed,
                bootstrap_samples=bootstrap_samples,
            ),
            "expected_brier": _clustered_prediction_interval(
                samples,
                [sample.expected_brier for sample in samples],
                bootstrap_seed=bootstrap_seed + 1,
                bootstrap_samples=bootstrap_samples,
            ),
            "fused_minus_component_expected_nll": {
                name: _clustered_prediction_interval(
                    samples,
                    [
                        sample.expected_nll - dict(sample.component_expected_nll)[name]
                        for sample in samples
                    ],
                    bootstrap_seed=bootstrap_seed + 2 + index,
                    bootstrap_samples=bootstrap_samples,
                )
                for index, name in enumerate(component_names)
            },
            "fused_minus_component_expected_brier": {
                name: _clustered_prediction_interval(
                    samples,
                    [
                        sample.expected_brier
                        - dict(sample.component_expected_brier)[name]
                        for sample in samples
                    ],
                    bootstrap_seed=(bootstrap_seed + 2 + len(component_names) + index),
                    bootstrap_samples=bootstrap_samples,
                )
                for index, name in enumerate(component_names)
            },
        }
    if include_role_slices:
        result["by_role"] = {
            role: _prediction_metrics(
                [sample for sample in samples if sample.opponent_role == role],
                include_role_slices=False,
            )
            for role in ("dropper", "checker")
        }
    return result


def _change_metrics(
    samples: Sequence[PMPredictionSample],
    *,
    detector_threshold: float,
) -> dict[str, object]:
    grouped: dict[tuple[str, int, int, str], list[PMPredictionSample]] = {}
    for sample in samples:
        key = (
            sample.family,
            sample.opponent_seed,
            sample.replicate,
            sample.opponent_role,
        )
        grouped.setdefault(key, []).append(sample)
    delays: list[int] = []
    undetected = 0
    shifts = 0
    false_alarms = 0
    nonshift = 0
    post_switch_excess_nll: list[float] = []
    for group in grouped.values():
        pending_delay: int | None = None
        detector_was_active = False
        for sample in sorted(group, key=lambda item: item.session_decision_index):
            if sample.truth_shift:
                if pending_delay is not None:
                    undetected += 1
                shifts += 1
                pending_delay = 0
                post_switch_excess_nll.append(
                    sample.expected_nll - math.log(ACTION_COUNT)
                )
            else:
                nonshift += 1
                if pending_delay is not None:
                    pending_delay += 1
            alarm_onset = sample.change_detected and not detector_was_active
            if alarm_onset:
                if pending_delay is None:
                    false_alarms += 1
                else:
                    delays.append(pending_delay)
                    pending_delay = None
            detector_was_active = sample.change_detected
        if pending_delay is not None:
            undetected += 1
    return {
        "truth_shift_definition": "total_variation_at_least_0.25",
        "detector_threshold": detector_threshold,
        "truth_shifts": shifts,
        "detected_shifts": len(delays),
        "undetected_shifts": undetected,
        "median_same_role_detection_delay": (
            float(np.median(delays)) if delays else None
        ),
        "mean_same_role_detection_delay": (float(np.mean(delays)) if delays else None),
        "false_alarm_onsets": false_alarms,
        "nonshift_decisions": nonshift,
        "false_alarm_onsets_per_100_nonshift_decisions": (
            100.0 * false_alarms / nonshift if nonshift else None
        ),
        "mean_post_switch_excess_expected_nll_vs_uniform": (
            float(np.mean(post_switch_excess_nll)) if post_switch_excess_nll else None
        ),
    }


def _policy_metrics(
    samples: Sequence[PMPredictionSample],
    decisions: Sequence[PMHalDecision],
    *,
    config: PMHalConfig,
    agent: object,
) -> dict[str, object]:
    if len(samples) != len(decisions):
        raise RuntimeError("PM risk audit requires one sample per live decision")
    mode_counts = {
        mode: sum(decision.mode == mode for decision in decisions)
        for mode in ("shield", "probe", "press", "dominate")
    }
    mode_caps = {
        "shield": 1e-9,
        "probe": config.probe_epsilon_cap,
        "press": config.press_epsilon_cap,
        "dominate": config.dominate_epsilon_cap,
    }
    recomputed_losses: list[float] = []
    diagnostic_loss_mismatches = 0
    sample_loss_mismatches = 0
    sample_decision_mismatches = 0
    risk_violations = 0
    mode_cap_violations = 0
    candidate_charge_contract_violations = 0
    diagnostic_cumulative_charge_mismatches = 0
    charges_by_game: dict[tuple[str, int, int, int], float] = {}
    stage_game = getattr(agent, "stage_game", None)
    if not callable(stage_game):
        raise TypeError("PM risk audit requires a stage_game agent")
    for sample, decision in zip(samples, decisions, strict=True):
        if (
            sample.role != decision.role
            or sample.mode != decision.mode
            or sample.game_index != decision.game_index
            or abs(sample.budget_charge - decision.selected_budget_charge)
            > _AUDIT_TOLERANCE
        ):
            sample_decision_mismatches += 1
        stage = stage_game(decision.state)
        matrix = np.asarray(stage.matrix, dtype=np.float64)
        selected_policy = _distribution(
            decision.policy, label="PM risk-audit selected policy"
        )
        drop_policy = _distribution(
            stage.drop_policy, label="PM risk-audit equilibrium Dropper"
        )
        check_policy = _distribution(
            stage.check_policy, label="PM risk-audit equilibrium Checker"
        )
        lower = float(np.min(matrix.T @ drop_policy))
        upper = float(np.max(matrix @ check_policy))
        selected_worst = (
            float(np.min(matrix.T @ selected_policy))
            if decision.role == "dropper"
            else float(np.max(matrix @ selected_policy))
        )
        recomputed = max(
            0.0,
            lower - selected_worst
            if decision.role == "dropper"
            else selected_worst - upper,
        )
        recomputed_losses.append(recomputed)
        if (
            abs(recomputed - decision.selected_actual_worst_case_loss)
            > _AUDIT_TOLERANCE
        ):
            diagnostic_loss_mismatches += 1
        if abs(recomputed - sample.actual_worst_case_loss) > _AUDIT_TOLERANCE:
            sample_loss_mismatches += 1
        if recomputed > decision.selected_budget_charge + _AUDIT_TOLERANCE:
            risk_violations += 1
        if recomputed > mode_caps[decision.mode] + _AUDIT_TOLERANCE:
            mode_cap_violations += 1
        if decision.selected_source == "exact":
            charge_matches_contract = (
                abs(decision.selected_budget_charge) <= _AUDIT_TOLERANCE
            )
        elif decision.selected_source == "adaptive_frontier":
            try:
                declared_epsilon = float(
                    decision.selected_candidate.removeprefix("frontier_")
                )
            except ValueError:
                charge_matches_contract = False
            else:
                charge_matches_contract = (
                    abs(decision.selected_budget_charge - declared_epsilon)
                    <= _AUDIT_TOLERANCE
                )
        else:
            charge_matches_contract = (
                abs(decision.selected_budget_charge - recomputed) <= _AUDIT_TOLERANCE
            )
        if not charge_matches_contract:
            candidate_charge_contract_violations += 1
        key = (
            sample.family,
            sample.opponent_seed,
            sample.replicate,
            decision.game_index,
        )
        cumulative = charges_by_game.get(key, 0.0) + decision.selected_budget_charge
        charges_by_game[key] = cumulative
        if abs(cumulative - decision.game_epsilon_spent) > _AUDIT_TOLERANCE:
            diagnostic_cumulative_charge_mismatches += 1
    game_budget_violations = sum(
        charge > config.game_epsilon_budget + _AUDIT_TOLERANCE
        for charge in charges_by_game.values()
    )
    total_audit_violations = sum(
        (
            diagnostic_loss_mismatches,
            sample_loss_mismatches,
            sample_decision_mismatches,
            risk_violations,
            mode_cap_violations,
            candidate_charge_contract_violations,
            diagnostic_cumulative_charge_mismatches,
            game_budget_violations,
        )
    )
    return {
        "decisions": len(samples),
        "games_audited": len(charges_by_game),
        "independent_matrix_recomputation": True,
        "audit_tolerance": _AUDIT_TOLERANCE,
        "mode_counts": mode_counts,
        "selected_candidate_counts": {
            name: sum(decision.selected_candidate == name for decision in decisions)
            for name in sorted({decision.selected_candidate for decision in decisions})
        },
        "mean_actual_local_worst_case_loss": (
            float(np.mean(recomputed_losses)) if recomputed_losses else None
        ),
        "max_actual_local_worst_case_loss": (
            max(recomputed_losses) if recomputed_losses else None
        ),
        "total_conservative_budget_charge": float(
            sum(decision.selected_budget_charge for decision in decisions)
        ),
        "risk_charge_understates_actual_loss_violations": risk_violations,
        "per_game_budget_violations": game_budget_violations,
        "mode_cap_violations": mode_cap_violations,
        "candidate_charge_contract_violations": (candidate_charge_contract_violations),
        "diagnostic_loss_mismatches": diagnostic_loss_mismatches,
        "sample_loss_mismatches": sample_loss_mismatches,
        "sample_decision_mismatches": sample_decision_mismatches,
        "diagnostic_cumulative_charge_mismatches": (
            diagnostic_cumulative_charge_mismatches
        ),
        "total_independent_risk_audit_violations": total_audit_violations,
    }


def _session_seed(evaluation_seed: int, opponent_seed: int, replicate: int) -> int:
    return int(evaluation_seed + opponent_seed * 10_000 + replicate * 1_000)


def evaluate_pm_hal(
    *,
    evaluation_config: PMEvaluationConfig | None = None,
    config_path: str | Path = DEFAULT_CONFIG,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    pm_config: PMHalConfig | None = None,
    pm_config_path: str | Path = DEFAULT_PM_HAL_CONFIG,
    aggro_checkpoint: str | Path | None = None,
    device: str = "cpu",
    exact_agent: CompleteDTHAgent | None = None,
) -> dict[str, object]:
    """Run PM Hal and three component baselines on common synthetic sessions."""

    protocol = evaluation_config or load_evaluation_config(config_path)
    controller = pm_config or load_pm_hal_config(pm_config_path)
    git_provenance = None
    if protocol.schema_version == CONFIG_SCHEMA and evaluation_config is None:
        if pm_config is not None:
            raise ValueError(
                "Git-registered PM confirmation requires a file-backed PM config"
            )
        git_provenance = _git_registered_protocol(config_path, pm_config_path)
    if (
        protocol.opponent_identities
        < protocol.promotion_gate.minimum_opponent_identities
    ):
        raise ValueError("evaluation has fewer identities than its promotion gate")
    if device != "cpu":
        raise ValueError("PM Hal evaluation is CPU-only")
    pm_config_digest = _sha256_file(pm_config_path) if pm_config is None else None
    if protocol.expected_pm_config_sha256 is not None:
        if pm_config_digest is None:
            raise ValueError("bound PM evaluation requires a file-backed PM config")
        if pm_config_digest != protocol.expected_pm_config_sha256:
            raise ValueError("PM controller config hash differs from protocol binding")
    artifact = Path(artifact_dir)
    agent = exact_agent or CompleteDTHAgent(artifact)
    metadata = getattr(getattr(agent, "tablebase", None), "metadata", None)
    table_digest = (
        metadata.get("table_digest") if isinstance(metadata, Mapping) else None
    )
    if (
        protocol.expected_dth_table_digest is not None
        and table_digest != protocol.expected_dth_table_digest
    ):
        raise ValueError("DTH table digest differs from protocol binding")
    if (
        aggro_checkpoint is None
        and protocol.expected_aggro_checkpoint_sha256 is not None
    ):
        aggro_checkpoint = DEFAULT_AGGRO_CHECKPOINT
    aggro_model = None
    aggro_config = None
    aggro_checkpoint_digest = None
    if aggro_checkpoint is not None:
        aggro_checkpoint_digest = _sha256_file(aggro_checkpoint)
        if (
            protocol.expected_aggro_checkpoint_sha256 is not None
            and aggro_checkpoint_digest != protocol.expected_aggro_checkpoint_sha256
        ):
            raise ValueError("Aggro checkpoint hash differs from protocol binding")
        aggro_model, _ = load_checkpoint(
            aggro_checkpoint,
            dth_ruleset=dth_compatibility(agent),
            device="cpu",
        )
        aggro_config = aggro_model.config
    if (
        protocol.promotion_gate.minimum_clustered_lower_bound_vs_no_aggro is not None
        and aggro_model is None
    ):
        raise ValueError("PM promotion protocol requires the Aggro ablation")
    perfect_provider = PerfectHalPolicyProvider(artifact, agent=agent)
    sessions: dict[str, list[dict[str, object]]] = {
        "pm": [],
        "perfect": [],
        "adaptive": [],
        "exact": [],
    }
    aggro_provider = None
    if aggro_model is not None and aggro_config is not None:
        sessions["aggro"] = []
        sessions["pm_no_aggro"] = []
        aggro_provider = AggroHalPolicyProvider(
            artifact,
            aggro_model,
            aggro_config,
            agent=agent,
            device="cpu",
        )
    all_samples: list[PMPredictionSample] = []
    all_decisions: list[PMHalDecision] = []
    for entry in protocol.entries:
        for opponent_seed in entry.seeds:
            for replicate in range(protocol.sessions_per_opponent):
                common_seed = _session_seed(
                    protocol.evaluation_seed, opponent_seed, replicate
                )
                starts_first = (opponent_seed + replicate) % 2 == 0

                pm_provider = PMHalPolicyProvider(
                    artifact,
                    controller,
                    agent=agent,
                    aggro_model=aggro_model,
                    aggro_config=aggro_config,
                    device="cpu",
                    seed=common_seed + 17,
                )
                pm_opponent = _TruthTrackingOpponent(
                    make_opponent(entry.family, seed=opponent_seed)
                )
                measured = _MeasuredPMProvider(
                    pm_provider,
                    pm_opponent,
                    family=entry.family,
                    opponent_seed=opponent_seed,
                    replicate=replicate,
                )
                pm_result = _run_session(
                    controller="pm",
                    candidate=measured,
                    opponent=pm_opponent,
                    session_seed=common_seed,
                    seat_pairs=protocol.seat_pairs_per_session,
                    start_clocks=protocol.start_clocks,
                    max_half_rounds=protocol.max_half_rounds,
                    candidate_starts_first_seat=starts_first,
                )
                pm_result.update(
                    {
                        "opponent_family": entry.family,
                        "opponent_seed": opponent_seed,
                        "replicate": replicate,
                        "prediction_metrics": _prediction_metrics(
                            measured.measurements
                        ),
                    }
                )
                sessions["pm"].append(pm_result)
                all_samples.extend(measured.measurements)
                all_decisions.extend(pm_provider.decisions)

                if aggro_model is not None:
                    no_aggro_provider = PMHalPolicyProvider(
                        artifact,
                        controller,
                        agent=agent,
                        device="cpu",
                        seed=common_seed + 17,
                    )
                    no_aggro_opponent = _TruthTrackingOpponent(
                        make_opponent(entry.family, seed=opponent_seed)
                    )
                    measured_no_aggro = _MeasuredPMProvider(
                        no_aggro_provider,
                        no_aggro_opponent,
                        family=entry.family,
                        opponent_seed=opponent_seed,
                        replicate=replicate,
                    )
                    no_aggro_result = _run_session(
                        controller="pm_no_aggro",
                        candidate=measured_no_aggro,
                        opponent=no_aggro_opponent,
                        session_seed=common_seed,
                        seat_pairs=protocol.seat_pairs_per_session,
                        start_clocks=protocol.start_clocks,
                        max_half_rounds=protocol.max_half_rounds,
                        candidate_starts_first_seat=starts_first,
                    )
                    no_aggro_result.update(
                        {
                            "opponent_family": entry.family,
                            "opponent_seed": opponent_seed,
                            "replicate": replicate,
                            "prediction_metrics": _prediction_metrics(
                                measured_no_aggro.measurements
                            ),
                        }
                    )
                    sessions["pm_no_aggro"].append(no_aggro_result)

                perfect_opponent = _TruthTrackingOpponent(
                    make_opponent(entry.family, seed=opponent_seed)
                )
                perfect_result = _run_session(
                    controller="perfect",
                    candidate=perfect_provider,
                    opponent=perfect_opponent,
                    session_seed=common_seed,
                    seat_pairs=protocol.seat_pairs_per_session,
                    start_clocks=protocol.start_clocks,
                    max_half_rounds=protocol.max_half_rounds,
                    candidate_starts_first_seat=starts_first,
                )
                perfect_result.update(
                    {
                        "opponent_family": entry.family,
                        "opponent_seed": opponent_seed,
                        "replicate": replicate,
                    }
                )
                sessions["perfect"].append(perfect_result)

                prior = DirichletPrior.uniform(
                    strength=controller.adaptive_prior_strength
                )
                adaptive_provider = AdaptiveDTHPolicyProvider(
                    artifact,
                    RoleDirichletOpponent(
                        prior,
                        prior,
                        decay=controller.adaptive_decay,
                    ),
                    config=ExploitationConfig(
                        epsilon_grid=controller.epsilon_grid,
                        match_epsilon_budget=controller.game_epsilon_budget,
                        confidence=controller.minimum_improvement_support,
                        posterior_samples=controller.posterior_samples,
                        improvement_tolerance=controller.improvement_tolerance,
                    ),
                    seed=common_seed + 29,
                    agent=agent,
                )
                adaptive_opponent = _TruthTrackingOpponent(
                    make_opponent(entry.family, seed=opponent_seed)
                )
                adaptive_result = _run_session(
                    controller="adaptive",
                    candidate=adaptive_provider,
                    opponent=adaptive_opponent,
                    session_seed=common_seed,
                    seat_pairs=protocol.seat_pairs_per_session,
                    start_clocks=protocol.start_clocks,
                    max_half_rounds=protocol.max_half_rounds,
                    candidate_starts_first_seat=starts_first,
                )
                adaptive_result.update(
                    {
                        "opponent_family": entry.family,
                        "opponent_seed": opponent_seed,
                        "replicate": replicate,
                    }
                )
                sessions["adaptive"].append(adaptive_result)

                if aggro_provider is not None:
                    aggro_opponent = _TruthTrackingOpponent(
                        make_opponent(entry.family, seed=opponent_seed)
                    )
                    aggro_result = _run_session(
                        controller="aggro",
                        candidate=aggro_provider,
                        opponent=aggro_opponent,
                        session_seed=common_seed,
                        seat_pairs=protocol.seat_pairs_per_session,
                        start_clocks=protocol.start_clocks,
                        max_half_rounds=protocol.max_half_rounds,
                        candidate_starts_first_seat=starts_first,
                    )
                    aggro_result.update(
                        {
                            "opponent_family": entry.family,
                            "opponent_seed": opponent_seed,
                            "replicate": replicate,
                        }
                    )
                    sessions["aggro"].append(aggro_result)

                exact_opponent = _TruthTrackingOpponent(
                    make_opponent(entry.family, seed=opponent_seed)
                )
                exact_result = _run_session(
                    controller="exact",
                    candidate=_ExactProvider(agent),
                    opponent=exact_opponent,
                    session_seed=common_seed,
                    seat_pairs=protocol.seat_pairs_per_session,
                    start_clocks=protocol.start_clocks,
                    max_half_rounds=protocol.max_half_rounds,
                    candidate_starts_first_seat=starts_first,
                )
                exact_result.update(
                    {
                        "opponent_family": entry.family,
                        "opponent_seed": opponent_seed,
                        "replicate": replicate,
                    }
                )
                sessions["exact"].append(exact_result)

    summaries = {
        name: _controller_summary(controller_sessions, protocol.families)
        for name, controller_sessions in sessions.items()
    }
    component_names = ["adaptive", "perfect"]
    if "aggro" in sessions:
        component_names.append("aggro")
    comparison_names = ["exact", *component_names]
    if "pm_no_aggro" in sessions:
        comparison_names.append("pm_no_aggro")
    comparisons = {
        baseline: _paired_comparison(
            sessions["pm"],
            sessions[baseline],
            left_name="pm",
            right_name=baseline,
            bootstrap_seed=protocol.bootstrap_seed + index,
            bootstrap_samples=protocol.bootstrap_samples,
        )
        for index, baseline in enumerate(comparison_names)
    }
    component_scores = {
        name: summaries[name]["all_game_score"] for name in component_names
    }
    strongest_component = max(
        component_scores,
        key=lambda name: float(component_scores[name]),
    )
    policy_metrics = _policy_metrics(
        all_samples, all_decisions, config=controller, agent=agent
    )
    exact_interval = comparisons["exact"]["cluster_bootstrap_95"]
    component_intervals = [
        comparisons[name]["cluster_bootstrap_95"] for name in component_names
    ]
    component_lower_bound = (
        min(
            float(interval[0])
            for interval in component_intervals
            if interval is not None
        )
        if all(interval is not None for interval in component_intervals)
        else None
    )
    slice_baselines = [*component_names]
    if "pm_no_aggro" in sessions:
        slice_baselines.append("pm_no_aggro")
    point_difference_slices = {
        "by_seat": {
            baseline: {
                seat: float(summaries["pm"]["by_seat"][seat]["all_game_score"])
                - float(summaries[baseline]["by_seat"][seat]["all_game_score"])
                for seat in ("Hal", "Baku")
            }
            for baseline in slice_baselines
        },
        "by_family": {
            baseline: {
                family: float(summaries["pm"]["by_family"][family]["all_game_score"])
                - float(summaries[baseline]["by_family"][family]["all_game_score"])
                for family in protocol.families
            }
            for baseline in slice_baselines
        },
    }
    minimum_seat_difference = min(
        difference
        for baseline in point_difference_slices["by_seat"].values()
        for difference in baseline.values()
    )
    minimum_family_difference = min(
        difference
        for baseline in point_difference_slices["by_family"].values()
        for difference in baseline.values()
    )
    gate = protocol.promotion_gate
    no_aggro_interval = (
        comparisons["pm_no_aggro"]["cluster_bootstrap_95"]
        if "pm_no_aggro" in comparisons
        else None
    )
    checks = {
        "enough_opponent_identities": protocol.opponent_identities
        >= gate.minimum_opponent_identities,
        "no_risk_violations": int(
            policy_metrics["total_independent_risk_audit_violations"]
        )
        <= gate.maximum_risk_violations,
        "clustered_lower_bound_vs_exact": exact_interval is not None
        and float(exact_interval[0]) >= gate.minimum_clustered_lower_bound_vs_exact,
        "clustered_lower_bound_vs_every_component": component_lower_bound is not None
        and component_lower_bound
        >= gate.minimum_clustered_lower_bound_vs_strongest_component,
        "clustered_lower_bound_vs_no_aggro": (
            gate.minimum_clustered_lower_bound_vs_no_aggro is None
            or (
                no_aggro_interval is not None
                and float(no_aggro_interval[0])
                >= gate.minimum_clustered_lower_bound_vs_no_aggro
            )
        ),
        "point_noninferiority_every_seat": minimum_seat_difference
        >= gate.minimum_point_difference_every_seat,
        "point_noninferiority_every_family": minimum_family_difference
        >= gate.minimum_point_difference_every_family,
    }
    return {
        "schema_version": EVALUATION_SCHEMA,
        "claim_scope": (
            "git-registered synthetic pure-DTH opponent league"
            if git_provenance is not None
            else "development synthetic pure-DTH opponent league"
        ),
        "human_validation": False,
        "git_provenance": git_provenance,
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path)
        if evaluation_config is None
        else None,
        "protocol": asdict(protocol),
        "pm_config_path": str(pm_config_path) if pm_config is None else None,
        "pm_config_sha256": pm_config_digest,
        "pm_config": asdict(controller),
        "dth_artifact": str(artifact),
        "dth_table_digest": table_digest,
        "aggro_checkpoint": str(aggro_checkpoint)
        if aggro_checkpoint is not None
        else None,
        "aggro_checkpoint_sha256": aggro_checkpoint_digest,
        "aggro_component_enabled": aggro_checkpoint is not None,
        "summaries": summaries,
        "paired_all_game_score_comparisons": comparisons,
        "paired_point_difference_slices": point_difference_slices,
        "strongest_component_baseline": strongest_component,
        "prediction_metrics": _prediction_metrics(
            all_samples,
            bootstrap_seed=protocol.bootstrap_seed + 100,
            bootstrap_samples=protocol.bootstrap_samples,
        ),
        "change_metrics": _change_metrics(
            all_samples,
            detector_threshold=controller.change_shield_threshold,
        ),
        "policy_metrics": policy_metrics,
        "promotion_gate": {
            "criteria": asdict(gate),
            "checks": checks,
            "passed": all(checks.values()),
        },
        "sessions": sessions,
    }


def write_evaluation_report(report: Mapping[str, object], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--pm-config", default=str(DEFAULT_PM_HAL_CONFIG))
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--aggro-checkpoint", default=None)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = evaluate_pm_hal(
        config_path=args.config,
        pm_config_path=args.pm_config,
        artifact_dir=args.artifact_dir,
        aggro_checkpoint=args.aggro_checkpoint,
    )
    destination = write_evaluation_report(report, args.output)
    print(
        f"PM Hal synthetic benchmark: promotion gate "
        f"{'passed' if report['promotion_gate']['passed'] else 'failed'}; "
        f"report {destination}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFIG_SCHEMA",
    "DEFAULT_CONFIG",
    "EVALUATION_SCHEMA",
    "PMEvaluationConfig",
    "PMEvaluationEntry",
    "PMPredictionSample",
    "PMPromotionGate",
    "evaluate_pm_hal",
    "load_evaluation_config",
    "write_evaluation_report",
]
