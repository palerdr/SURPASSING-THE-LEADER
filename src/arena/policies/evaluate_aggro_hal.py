"""CPU-only repeated-session evaluation for direct recurrent Aggro Hal.

The experimental unit is a repeated-opponent session.  Each game seed is used
twice with Aggro Hal in opposite seats, while one recurrent state is retained
through the whole session.  Aggro Hal and the exact DTH baseline receive fresh,
identically seeded opponent instances so that neither can leak history into the
other controller's run.

Opponent prediction diagnostics are evaluated after the simultaneous actions
are revealed.  Oracle regret is evaluation-only: it uses the synthetic league
opponent's true current distribution and the certified continuation-adjusted
DTH matrix, neither of which is exposed to the live policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from arena.contracts import CanonicalDecision, PublicGameOutcome, PublicHalfRound
from arena.dth_adapter import project_to_dth_state
from arena.match import play_match_game
from arena.policies.aggro_hal import (
    ACTION_COUNT,
    AggroHalConfig,
    AggroHalNetwork,
    AggroHalPolicyProvider,
)
from arena.policies.opponent_league import (
    ACTIONS,
    FAMILY_MANIFESTS,
    OpponentFamilyManifest,
    ReactiveDTHOpponent,
    make_opponent,
)
from dth.agent import CompleteDTHAgent

EVALUATION_SCHEMA = "arena-aggro-hal-evaluation-v2"
DEFAULT_ARTIFACT_DIR = Path("src/dth/artifacts/complete_full_v1")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_digest(manifest: OpponentFamilyManifest) -> str:
    payload = json.dumps(
        asdict(manifest), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class SuccessCriteria:
    """Predeclared gate for exploitation of deliberately predictable policies."""

    min_decisive_games: int = 20
    min_opponent_seed_units: int = 8
    min_decisive_win_rate: float = 0.75
    min_each_seat_decisive_win_rate: float = 0.65
    min_each_family_decisive_win_rate: float = 0.60
    max_stop_rate: float = 0.10
    min_decisive_win_rate_uplift_vs_exact: float = 0.10

    def __post_init__(self) -> None:
        if self.min_decisive_games <= 0 or self.min_opponent_seed_units <= 0:
            raise ValueError("minimum game and opponent-seed counts must be positive")
        probabilities = (
            self.min_decisive_win_rate,
            self.min_each_seat_decisive_win_rate,
            self.min_each_family_decisive_win_rate,
            self.max_stop_rate,
        )
        if any(not 0.0 <= float(value) <= 1.0 for value in probabilities):
            raise ValueError("success rates must lie in [0, 1]")
        if not -1.0 <= float(self.min_decisive_win_rate_uplift_vs_exact) <= 1.0:
            raise ValueError("exact-baseline uplift must lie in [-1, 1]")


@dataclass(frozen=True, slots=True)
class _PredictionSample:
    role: str
    realized_nll: float
    expected_nll: float
    brier_score: float
    top_confidence: float
    top_correct: bool
    oracle_regret: float
    direct_weight: float


@dataclass(frozen=True, slots=True)
class _PendingPrediction:
    role: str
    self_name: str
    prediction: np.ndarray
    truth: np.ndarray
    candidate_policy: np.ndarray
    oriented_action_values: np.ndarray
    direct_weight: float


class _ExactProvider:
    """Share one read-only complete agent without rebuilding its memory maps."""

    def __init__(self, agent: CompleteDTHAgent) -> None:
        self.agent = agent

    def policy(self, decision: CanonicalDecision) -> Mapping[int, float]:
        move = self.agent.decide(project_to_dth_state(decision))
        policy = move.drop_policy if decision.role == "dropper" else move.check_policy
        return {
            action: float(probability)
            for action, probability in enumerate(policy, start=1)
            if probability > 0.0
        }


class _TruthTrackingOpponent:
    """Expose the policy the synthetic opponent actually used this half-round."""

    def __init__(self, opponent: ReactiveDTHOpponent) -> None:
        self.opponent = opponent
        self.pending_decision: CanonicalDecision | None = None
        self.pending_truth: np.ndarray | None = None

    def reset_game(self) -> None:
        if self.pending_truth is not None:
            raise RuntimeError("opponent truth remained pending at a game boundary")
        self.opponent.reset_game()

    def reset_session(self) -> None:
        if self.pending_truth is not None:
            raise RuntimeError("opponent truth remained pending at a session boundary")
        self.pending_decision = None
        self.opponent.reset_session()

    def policy(self, decision: CanonicalDecision) -> Mapping[int, float]:
        if self.pending_truth is not None:
            raise RuntimeError("opponent acted twice before its public reveal")
        truth = self.opponent.true_distribution(decision)
        raw = self.opponent.policy(decision)
        self.pending_decision = decision
        self.pending_truth = _distribution(truth, "opponent truth")
        return raw

    def truth_for(self, candidate_decision: CanonicalDecision) -> np.ndarray:
        """Return truth before or after the opponent's simultaneous policy call."""

        opposite = _opponent_decision(candidate_decision)
        if self.pending_truth is None:
            return _distribution(
                self.opponent.true_distribution(opposite), "opponent truth"
            )
        if self.pending_decision is None:
            raise RuntimeError("tracked opponent truth has no decision")
        pending = self.pending_decision
        if (
            pending.role != opposite.role
            or pending.actor_name.casefold() != opposite.actor_name.casefold()
            or pending.legal_seconds != opposite.legal_seconds
        ):
            raise RuntimeError("tracked opponent truth belongs to another decision")
        return self.pending_truth.copy()

    def observe(self, record: PublicHalfRound) -> None:
        self.opponent.observe(record)
        self.pending_decision = None
        self.pending_truth = None

    def end_game(self, outcome: PublicGameOutcome) -> None:
        self.opponent.end_game(outcome)


class _MeasuredAggroProvider:
    """Measure one live Aggro provider without changing its chosen policy."""

    def __init__(
        self,
        provider: AggroHalPolicyProvider,
        opponent: _TruthTrackingOpponent,
        *,
        reset_recurrent_each_game: bool = False,
        erase_history_each_decision: bool = False,
    ) -> None:
        self.provider = provider
        self.opponent = opponent
        self.samples: list[_PredictionSample] = []
        self._pending: _PendingPrediction | None = None
        self.reset_recurrent_each_game = bool(reset_recurrent_each_game)
        self.erase_history_each_decision = bool(erase_history_each_decision)

    def reset_session(self) -> None:
        self.provider.reset_session()
        self.samples.clear()
        self._pending = None

    def reset_game(self) -> None:
        if self._pending is not None:
            raise RuntimeError("Aggro prediction remained pending at a game boundary")
        if self.reset_recurrent_each_game:
            self.provider.reset_session()
        self.provider.reset_game()

    def policy(self, decision: CanonicalDecision) -> Mapping[int, float]:
        if self._pending is not None:
            raise RuntimeError("Aggro Hal acted twice before its public reveal")
        if self.erase_history_each_decision:
            self.provider.reset_session()
        truth = self.opponent.truth_for(decision)
        stage = self.provider.agent.stage_game(project_to_dth_state(decision))
        raw = self.provider.policy(decision)
        diagnostic = self.provider.last_decision
        if diagnostic is None:
            raise RuntimeError("Aggro Hal did not publish decision diagnostics")
        prediction = _distribution(diagnostic.opponent_policy, "opponent prediction")
        candidate = _distribution(diagnostic.policy, "Aggro policy")
        if decision.role == "dropper":
            oriented = np.asarray(stage.matrix, dtype=np.float64) @ truth
        else:
            oriented = -np.asarray(stage.matrix, dtype=np.float64).T @ truth
        legal = np.asarray(
            [action in decision.legal_seconds for action in ACTIONS], dtype=np.bool_
        )
        if not np.any(legal):
            raise ValueError("Aggro evaluation requires a legal pure-DTH action")
        oriented = np.where(legal, oriented, -np.inf)
        self._pending = _PendingPrediction(
            role=decision.role,
            self_name=decision.actor_name,
            prediction=prediction,
            truth=truth,
            candidate_policy=candidate,
            oriented_action_values=oriented,
            direct_weight=float(diagnostic.direct_weight),
        )
        return raw

    def observe(self, record: PublicHalfRound) -> None:
        pending = self._pending
        if pending is None:
            raise RuntimeError("Aggro Hal received a reveal without a prediction")
        if pending.role == "dropper":
            if record.dropper_name.casefold() != pending.self_name.casefold():
                raise RuntimeError("Aggro reveal has the wrong Dropper identity")
            opponent_action = int(record.check_time)
        else:
            if record.checker_name.casefold() != pending.self_name.casefold():
                raise RuntimeError("Aggro reveal has the wrong Checker identity")
            opponent_action = int(record.drop_time)
        if not 1 <= opponent_action <= ACTION_COUNT:
            raise ValueError("Aggro evaluation supports revealed actions 1..60 only")

        index = opponent_action - 1
        prediction = pending.prediction
        truth = pending.truth
        target = np.zeros(ACTION_COUNT, dtype=np.float64)
        target[index] = 1.0
        legal_values = pending.oriented_action_values
        oracle_value = float(np.max(legal_values))
        candidate_value = float(
            np.sum(
                pending.candidate_policy
                * np.where(np.isfinite(legal_values), legal_values, 0.0)
            )
        )
        regret = max(0.0, oracle_value - candidate_value)
        self.samples.append(
            _PredictionSample(
                role=pending.role,
                realized_nll=-math.log(max(float(prediction[index]), 1e-12)),
                expected_nll=float(
                    -np.sum(truth * np.log(np.clip(prediction, 1e-12, 1.0)))
                ),
                brier_score=float(np.sum((prediction - target) ** 2)),
                top_confidence=float(np.max(prediction)),
                top_correct=bool(int(np.argmax(prediction)) == index),
                oracle_regret=regret,
                direct_weight=pending.direct_weight,
            )
        )
        self._pending = None
        self.provider.observe(record)

    def end_game(self, outcome: PublicGameOutcome) -> None:
        self.provider.end_game(outcome)


def _distribution(values: Sequence[float] | np.ndarray, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if (
        result.shape != (ACTION_COUNT,)
        or not np.all(np.isfinite(result))
        or np.any(result < 0.0)
        or float(result.sum()) <= 0.0
    ):
        raise ValueError(f"{label} must be a finite nonnegative length-60 distribution")
    return result / float(result.sum())


def _opponent_decision(decision: CanonicalDecision) -> CanonicalDecision:
    if decision.actor_name.casefold() == "hal":
        opponent_name = "Baku"
    elif decision.actor_name.casefold() == "baku":
        opponent_name = "Hal"
    else:
        raise ValueError("Arena evaluation requires the canonical Hal/Baku seats")
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


def _ece(samples: Sequence[_PredictionSample], bins: int = 10) -> float | None:
    if not samples:
        return None
    confidences = np.asarray([sample.top_confidence for sample in samples])
    correct = np.asarray([sample.top_correct for sample in samples], dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(samples)
    result = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (confidences >= edges[index]) & (confidences <= edges[index + 1])
        else:
            selected = (confidences >= edges[index]) & (confidences < edges[index + 1])
        count = int(np.sum(selected))
        if count:
            result += (
                count
                / total
                * abs(
                    float(np.mean(correct[selected]))
                    - float(np.mean(confidences[selected]))
                )
            )
    return result


def summarize_prediction_metrics(
    samples: Sequence[_PredictionSample],
) -> dict[str, object]:
    """Aggregate proper prediction scores, calibration, oracle regret, and gate use."""

    result = _summarize_prediction_metrics(samples)
    result["by_role"] = {
        role: _summarize_prediction_metrics(
            [sample for sample in samples if sample.role == role]
        )
        for role in ("dropper", "checker")
    }
    return result


def _summarize_prediction_metrics(
    samples: Sequence[_PredictionSample],
) -> dict[str, object]:
    if not samples:
        return {
            "decisions": 0,
            "opponent_realized_nll": None,
            "opponent_expected_nll": None,
            "opponent_brier_score": None,
            "opponent_top_label_ece_10_bin": None,
            "mean_one_step_oracle_regret": None,
            "direct_gate": {
                "mean": None,
                "median": None,
                "p90": None,
                "fraction_at_least_half": None,
            },
        }
    direct = np.asarray([sample.direct_weight for sample in samples], dtype=np.float64)
    uniform_nll = math.log(ACTION_COUNT)
    uniform_brier = (ACTION_COUNT - 1.0) / ACTION_COUNT
    expected_nll = float(np.mean([sample.expected_nll for sample in samples]))
    brier = float(np.mean([sample.brier_score for sample in samples]))
    return {
        "decisions": len(samples),
        "opponent_realized_nll": float(
            np.mean([sample.realized_nll for sample in samples])
        ),
        "opponent_expected_nll": expected_nll,
        "opponent_brier_score": brier,
        "uniform_baseline": {
            "expected_nll": uniform_nll,
            "brier_score": uniform_brier,
            "expected_nll_improvement": uniform_nll - expected_nll,
            "brier_improvement": uniform_brier - brier,
            "beats_uniform_on_both": (
                expected_nll < uniform_nll and brier < uniform_brier
            ),
        },
        "opponent_top_label_ece_10_bin": _ece(samples),
        "mean_one_step_oracle_regret": float(
            np.mean([sample.oracle_regret for sample in samples])
        ),
        "direct_gate": {
            "mean": float(np.mean(direct)),
            "median": float(np.median(direct)),
            "p90": float(np.quantile(direct, 0.9)),
            "fraction_at_least_half": float(np.mean(direct >= 0.5)),
        },
    }


def _session_seed(
    evaluation_seed: int,
    opponent_seed: int,
    replicate: int,
) -> int:
    return int(evaluation_seed + opponent_seed * 10_000 + replicate * 1_000)


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
        game_seed = session_seed + pair_index
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
    samples = getattr(candidate, "samples", ())
    return {
        "controller": controller,
        "games": games,
        "prediction_metrics": summarize_prediction_metrics(samples),
    }


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
        "all_game_win_rate": wins / len(games) if games else None,
        "all_game_win_rate_wilson_95": _wilson_interval(wins, len(games)),
        "stop_rate": stopped / len(games) if games else None,
        "mean_half_rounds": (
            float(np.mean([int(game["half_rounds"]) for game in games]))
            if games
            else None
        ),
    }


def _wilson_interval(successes: int, trials: int) -> list[float] | None:
    """Return the two-sided 95% Wilson interval for a pooled binary rate."""

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


def _outcome_score(won: object) -> float:
    """Score a win/loss/stop as 1/0/0.5 for the predeclared paired statistic."""

    if won is True:
        return 1.0
    if won is False:
        return 0.0
    if won is None:
        return 0.5
    raise ValueError("game outcome must be True, False, or None")


def _opponent_seed_comparison(
    aggro_sessions: Sequence[Mapping[str, object]],
    exact_sessions: Sequence[Mapping[str, object]],
    *,
    bootstrap_seed: int,
    bootstrap_samples: int = 5_000,
) -> dict[str, object]:
    """Compare controllers on common scenarios, clustered by opponent identity."""

    def session_key(session: Mapping[str, object]) -> tuple[object, object, object]:
        return (
            session["opponent_family"],
            session["opponent_seed"],
            session["replicate"],
        )

    exact_by_session = {session_key(session): session for session in exact_sessions}
    if len(exact_by_session) != len(exact_sessions):
        raise RuntimeError("Exact evaluation contains duplicate session keys")
    clustered: dict[tuple[object, object], list[float]] = {}
    scenario_count = 0
    for aggro_session in aggro_sessions:
        key = session_key(aggro_session)
        try:
            exact_session = exact_by_session.pop(key)
        except KeyError as error:
            raise RuntimeError(
                "Aggro and Exact sessions are not scenario-matched"
            ) from error

        def games_by_key(
            session: Mapping[str, object],
        ) -> dict[tuple[object, ...], object]:
            games = session["games"]
            if not isinstance(games, list):
                raise RuntimeError("evaluation session games are malformed")
            return {
                (
                    game["seat_pair_index"],
                    game["seed"],
                    game["start_clock"],
                    game["candidate_seat"],
                ): game["won"]
                for game in games
            }

        aggro_games = games_by_key(aggro_session)
        exact_games = games_by_key(exact_session)
        if aggro_games.keys() != exact_games.keys():
            raise RuntimeError("Aggro and Exact games are not scenario-matched")
        cluster = (key[0], key[1])
        differences = clustered.setdefault(cluster, [])
        for game_key in aggro_games:
            differences.append(
                _outcome_score(aggro_games[game_key])
                - _outcome_score(exact_games[game_key])
            )
            scenario_count += 1
    if exact_by_session:
        raise RuntimeError("Exact evaluation has unmatched sessions")

    cluster_means = np.asarray(
        [float(np.mean(values)) for values in clustered.values()], dtype=np.float64
    )
    if not len(cluster_means):
        return {
            "stop_score": 0.5,
            "opponent_seed_units": 0,
            "common_scenarios": 0,
            "mean_aggro_minus_exact": None,
            "opponent_seed_bootstrap_95": None,
            "bootstrap_samples": bootstrap_samples,
        }
    rng = np.random.default_rng(bootstrap_seed)
    draws = rng.choice(
        cluster_means,
        size=(bootstrap_samples, len(cluster_means)),
        replace=True,
    ).mean(axis=1)
    return {
        "stop_score": 0.5,
        "experimental_unit": "opponent_family_and_parameter_seed",
        "opponent_seed_units": int(len(cluster_means)),
        "common_scenarios": scenario_count,
        "mean_aggro_minus_exact": float(np.mean(cluster_means)),
        "opponent_seed_bootstrap_95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "bootstrap_samples": bootstrap_samples,
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


def _success_gate(
    aggro: Mapping[str, object],
    exact: Mapping[str, object],
    comparison: Mapping[str, object],
    criteria: SuccessCriteria,
) -> dict[str, object]:
    aggro_rate = aggro["decisive_win_rate"]
    exact_rate = exact["decisive_win_rate"]
    uplift = (
        None
        if aggro_rate is None or exact_rate is None
        else float(aggro_rate) - float(exact_rate)
    )
    seat_rates = [
        aggro["by_seat"][seat]["decisive_win_rate"] for seat in ("Hal", "Baku")
    ]
    family_rates = [
        summary["decisive_win_rate"] for summary in aggro["by_family"].values()
    ]
    checks = {
        "enough_decisive_games": int(aggro["decisive_games"])
        >= criteria.min_decisive_games,
        "enough_independent_opponent_seeds": int(comparison["opponent_seed_units"])
        >= criteria.min_opponent_seed_units,
        "high_decisive_win_rate": aggro_rate is not None
        and float(aggro_rate) >= criteria.min_decisive_win_rate,
        "strong_in_both_seats": all(
            rate is not None and float(rate) >= criteria.min_each_seat_decisive_win_rate
            for rate in seat_rates
        ),
        "strong_across_every_family": all(
            rate is not None
            and float(rate) >= criteria.min_each_family_decisive_win_rate
            for rate in family_rates
        ),
        "bounded_stop_rate": aggro["stop_rate"] is not None
        and float(aggro["stop_rate"]) <= criteria.max_stop_rate,
        "confidence_supported_uplift_over_exact": (
            comparison["opponent_seed_bootstrap_95"] is not None
            and float(comparison["opponent_seed_bootstrap_95"][0])
            >= criteria.min_decisive_win_rate_uplift_vs_exact
        ),
    }
    return {
        "scope": "selected immutable predictable-opponent league split",
        "criteria": asdict(criteria),
        "observed_decisive_win_rate_uplift_vs_exact": uplift,
        "primary_common_scenario_comparison": dict(comparison),
        "checks": checks,
        "passed": all(checks.values()),
    }


def evaluate_aggro_hal(
    *,
    split: str = "validation",
    artifact_dir: str | Path | None = None,
    network: AggroHalNetwork | None = None,
    provider: AggroHalPolicyProvider | None = None,
    checkpoint: str | Path | None = None,
    config: AggroHalConfig | None = None,
    manifest: OpponentFamilyManifest | None = None,
    sessions_per_opponent: int = 1,
    seat_pairs_per_session: int = 2,
    start_clocks: Sequence[int] = (720,),
    max_half_rounds: int = 24,
    evaluation_seed: int = 940_000,
    success_criteria: SuccessCriteria = SuccessCriteria(),
    reset_recurrent_each_game: bool = False,
    erase_history_each_decision: bool = False,
    fast_adaptation: bool = False,
    device: str | torch.device = "cpu",
    exact_agent: CompleteDTHAgent | None = None,
) -> dict[str, object]:
    """Evaluate one supplied Aggro model/provider/checkpoint and Exact DTH.

    This harness is intentionally CPU-only.  It never probes for CUDA and
    rejects a non-CPU model/provider instead of moving it through GPU memory.
    """

    resolved_device = torch.device(device)
    if resolved_device.type != "cpu":
        raise ValueError("Aggro Hal evaluation is CPU-only; device must be 'cpu'")
    sources = sum(value is not None for value in (network, provider, checkpoint))
    if sources != 1:
        raise ValueError("supply exactly one of network, provider, or checkpoint")
    if sessions_per_opponent <= 0 or seat_pairs_per_session <= 0:
        raise ValueError("session and seat-pair counts must be positive")
    if max_half_rounds <= 0:
        raise ValueError("max_half_rounds must be positive")
    if not start_clocks or any(int(clock) <= 0 for clock in start_clocks):
        raise ValueError("start_clocks must contain positive literal seconds")
    selected_manifest = FAMILY_MANIFESTS.get(split) if manifest is None else manifest
    if selected_manifest is None or selected_manifest.split != split:
        raise ValueError("split must select a matching immutable opponent manifest")

    if provider is not None:
        if provider.device.type != "cpu" or any(
            parameter.device.type != "cpu" for parameter in provider.model.parameters()
        ):
            raise ValueError("supplied Aggro Hal provider must already be on CPU")
        aggro_provider = provider
        resolved_config = provider.config
        resolved_fast_adaptation = provider.fast_adaptation
        resolved_artifact = provider.artifact_dir
        resolved_agent = exact_agent or provider.agent
    else:
        if network is not None:
            if any(
                parameter.device.type != "cpu" for parameter in network.parameters()
            ):
                raise ValueError("supplied Aggro Hal network must already be on CPU")
            resolved_config = network.config if config is None else config
            if network.config != resolved_config:
                raise ValueError("network and evaluation configurations differ")
        else:
            resolved_config = config
        resolved_artifact = Path(artifact_dir or DEFAULT_ARTIFACT_DIR)
        resolved_agent = exact_agent or CompleteDTHAgent(resolved_artifact)
        if network is not None:
            aggro_provider = AggroHalPolicyProvider(
                resolved_artifact,
                network,
                resolved_config,
                agent=resolved_agent,
                device="cpu",
                fast_adaptation=fast_adaptation,
            )
        else:
            aggro_provider = AggroHalPolicyProvider.from_checkpoint(
                artifact_dir=resolved_artifact,
                checkpoint=Path(checkpoint),
                config=resolved_config,
                device="cpu",
                fast_adaptation=fast_adaptation,
            )
            resolved_config = aggro_provider.config
            if exact_agent is None:
                resolved_agent = aggro_provider.agent
        resolved_fast_adaptation = bool(fast_adaptation)

    sessions: dict[str, list[dict[str, object]]] = {"aggro": [], "exact": []}
    all_prediction_samples: list[_PredictionSample] = []
    for entry in selected_manifest.entries:
        for opponent_seed in entry.seeds:
            for replicate in range(sessions_per_opponent):
                common_seed = _session_seed(evaluation_seed, opponent_seed, replicate)

                aggro_opponent = _TruthTrackingOpponent(
                    make_opponent(entry.family, seed=opponent_seed)
                )
                measured = _MeasuredAggroProvider(
                    aggro_provider,
                    aggro_opponent,
                    reset_recurrent_each_game=reset_recurrent_each_game,
                    erase_history_each_decision=erase_history_each_decision,
                )
                aggro_result = _run_session(
                    controller="aggro",
                    candidate=measured,
                    opponent=aggro_opponent,
                    session_seed=common_seed,
                    seat_pairs=seat_pairs_per_session,
                    start_clocks=start_clocks,
                    max_half_rounds=max_half_rounds,
                    candidate_starts_first_seat=(opponent_seed + replicate) % 2 == 0,
                )
                aggro_result.update(
                    {
                        "opponent_family": entry.family,
                        "opponent_seed": opponent_seed,
                        "replicate": replicate,
                    }
                )
                sessions["aggro"].append(aggro_result)
                all_prediction_samples.extend(measured.samples)

                exact_opponent = _TruthTrackingOpponent(
                    make_opponent(entry.family, seed=opponent_seed)
                )
                exact_result = _run_session(
                    controller="exact",
                    candidate=_ExactProvider(resolved_agent),
                    opponent=exact_opponent,
                    session_seed=common_seed,
                    seat_pairs=seat_pairs_per_session,
                    start_clocks=start_clocks,
                    max_half_rounds=max_half_rounds,
                    candidate_starts_first_seat=(opponent_seed + replicate) % 2 == 0,
                )
                exact_result.update(
                    {
                        "opponent_family": entry.family,
                        "opponent_seed": opponent_seed,
                        "replicate": replicate,
                    }
                )
                sessions["exact"].append(exact_result)

    families = selected_manifest.families
    summaries = {
        name: _controller_summary(results, families)
        for name, results in sessions.items()
    }
    prediction_metrics = summarize_prediction_metrics(all_prediction_samples)
    family_comparison = {
        family: {
            "aggro_decisive_win_rate": summaries["aggro"]["by_family"][family][
                "decisive_win_rate"
            ],
            "exact_decisive_win_rate": summaries["exact"]["by_family"][family][
                "decisive_win_rate"
            ],
            "aggro_minus_exact": _optional_delta(
                summaries["aggro"]["by_family"][family]["decisive_win_rate"],
                summaries["exact"]["by_family"][family]["decisive_win_rate"],
            ),
        }
        for family in families
    }
    primary_comparison = _opponent_seed_comparison(
        sessions["aggro"],
        sessions["exact"],
        bootstrap_seed=evaluation_seed + 7919,
    )

    return {
        "schema_version": EVALUATION_SCHEMA,
        "device": "cpu",
        "checkpoint_sha256": _sha256_file(checkpoint)
        if checkpoint is not None
        else None,
        "split": split,
        "manifest_schema": selected_manifest.schema_version,
        "manifest_sha256": _manifest_digest(selected_manifest),
        "families": list(families),
        "sessions_per_opponent": sessions_per_opponent,
        "seat_pairs_per_session": seat_pairs_per_session,
        "start_clocks": [int(clock) for clock in start_clocks],
        "max_half_rounds": max_half_rounds,
        "evaluation_seed": evaluation_seed,
        "reset_recurrent_each_game": bool(reset_recurrent_each_game),
        "erase_history_each_decision": bool(erase_history_each_decision),
        "fast_adaptation": resolved_fast_adaptation,
        "summaries": summaries,
        "prediction_metrics": prediction_metrics,
        "exact_comparison": {
            "pooled_aggro_minus_exact_decisive_win_rate": _optional_delta(
                summaries["aggro"]["decisive_win_rate"],
                summaries["exact"]["decisive_win_rate"],
            ),
            "by_family": family_comparison,
        },
        "pooled_decisive_win_rate_difference": _optional_delta(
            summaries["aggro"]["decisive_win_rate"],
            summaries["exact"]["decisive_win_rate"],
        ),
        "common_scenario_all_game_score": primary_comparison,
        "success_gate": _success_gate(
            summaries["aggro"],
            summaries["exact"],
            primary_comparison,
            success_criteria,
        ),
        "sessions": sessions,
    }


def _optional_delta(left: object, right: object) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def write_evaluation_report(report: Mapping[str, object], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument(
        "--split", choices=tuple(FAMILY_MANIFESTS), default="validation"
    )
    parser.add_argument("--sessions-per-opponent", type=int, default=1)
    parser.add_argument("--seat-pairs-per-session", type=int, default=2)
    parser.add_argument("--max-half-rounds", type=int, default=24)
    parser.add_argument("--start-clock", type=int, default=720)
    parser.add_argument("--seed", type=int, default=940_000)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument(
        "--reset-recurrent-each-game",
        action="store_true",
        help="ablation: erase Aggro Hal's hidden state at every game boundary",
    )
    parser.add_argument(
        "--erase-history-each-decision",
        action="store_true",
        help="ablation: erase hidden state and queued public reveal before every action",
    )
    parser.add_argument(
        "--fast-adaptation",
        action="store_true",
        help="blend the learned forecast with concentrated online action evidence",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = evaluate_aggro_hal(
        split=args.split,
        artifact_dir=args.artifact_dir,
        checkpoint=args.checkpoint,
        sessions_per_opponent=args.sessions_per_opponent,
        seat_pairs_per_session=args.seat_pairs_per_session,
        start_clocks=(args.start_clock,),
        max_half_rounds=args.max_half_rounds,
        evaluation_seed=args.seed,
        reset_recurrent_each_game=args.reset_recurrent_each_game,
        erase_history_each_decision=args.erase_history_each_decision,
        fast_adaptation=args.fast_adaptation,
        device=args.device,
    )
    if args.output is not None:
        write_evaluation_report(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
