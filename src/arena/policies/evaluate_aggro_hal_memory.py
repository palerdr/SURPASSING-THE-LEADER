"""Causal hidden-state probe for Aggro Hal's learned opponent memory.

The ordinary reset ablations change more than recurrent state: they also clear
the queued public reveal and alter the new-game marker.  This evaluator instead
constructs two legal, equal-length public prefixes, captures their GRU states,
and feeds one bitwise-identical target token with the correct, swapped, and zero
hidden states.  Any resulting difference is therefore caused by recurrent
history rather than by a different current observation.

The probe is intentionally CPU-only and never queries CUDA availability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from arena.contracts import CanonicalDecision, PublicGameOutcome, PublicHalfRound
from arena.policies.aggro_env import AggroDecision, AggroSessionEnv
from arena.policies.aggro_hal import (
    ACTION_COUNT,
    OBSERVATION_FEATURES,
    AggroHalNetwork,
    dth_compatibility,
    load_checkpoint,
)
from dth.agent import CompleteDTHAgent

PROBE_SCHEMA = "arena-aggro-hal-latent-twin-probe-v2"
PROTOCOL_SCHEMA = "arena-aggro-hal-latent-twin-protocol-v2"
DEFAULT_ARTIFACT = Path("src/dth/artifacts/complete_full_v1")
DEFAULT_COVER_GAMES = (1, 8)
DEFAULT_TWIN_SEEDS = 32
DEFAULT_BOOTSTRAP_REPLICATES = 5_000
DEFAULT_BOOTSTRAP_SEED = 20_260_808
DEFAULT_START_CLOCK = 720
DEFAULT_COVER_ACTION = 30
_ROLES = ("dropper", "checker")
_MODES = ("a", "b")
_METRICS = (
    "payoff_correct_minus_swapped",
    "payoff_correct_minus_zero",
    "nll_swapped_minus_correct",
    "nll_zero_minus_correct",
    "mode_a_nll_swapped_minus_correct",
    "mode_b_nll_swapped_minus_correct",
    "policy_total_variation",
    "forecast_total_variation",
    "pre_target_hidden_l2",
    "post_target_hidden_l2",
    "normalized_payoff_crossover",
    "mode_a_normalized_payoff_crossover",
    "mode_b_normalized_payoff_crossover",
    "mode_a_payoff_correct_minus_swapped",
    "mode_b_payoff_correct_minus_swapped",
)


def _target_distributions() -> dict[str, np.ndarray]:
    mode_a = np.zeros(ACTION_COUNT, dtype=np.float64)
    mode_b = np.zeros(ACTION_COUNT, dtype=np.float64)
    mode_a[7], mode_a[59] = 0.20, 0.80
    mode_b[20], mode_b[58] = 0.90, 0.10
    return {"a": mode_a, "b": mode_b}


def _cue_multisets() -> dict[str, tuple[int, ...]]:
    return {
        "a": (8, 8, 60, 60, 60, 60, 60, 60, 60, 60),
        "b": (21, 21, 21, 21, 21, 21, 21, 21, 21, 59),
    }


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latent_twin_protocol(
    *,
    twin_seeds: Sequence[int] = tuple(range(DEFAULT_TWIN_SEEDS)),
    cover_games: Sequence[int] = DEFAULT_COVER_GAMES,
    start_clock: int = DEFAULT_START_CLOCK,
    cover_action: int = DEFAULT_COVER_ACTION,
) -> dict[str, object]:
    """Return the checkpoint-independent protocol commitment."""

    seeds = tuple(int(seed) for seed in twin_seeds)
    covers = tuple(int(value) for value in cover_games)
    if not seeds or any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("twin seeds must be nonempty, unique, and nonnegative")
    if (
        not covers
        or any(value <= 0 for value in covers)
        or len(set(covers)) != len(covers)
    ):
        raise ValueError("cover games must be nonempty, unique, and positive")
    if start_clock < 0:
        raise ValueError("start clock must be nonnegative")
    if not 1 <= cover_action <= ACTION_COUNT:
        raise ValueError("cover action must lie in 1..60")
    protocol: dict[str, object] = {
        "schema_version": PROTOCOL_SCHEMA,
        "game": "pure-dth",
        "device": "cpu",
        "target_state": [0, 0, 0, 0],
        "target_selection": (
            "DTH-only two-spike search; independent of checkpoint outputs; "
            "requires distinct unique best responses in both roles"
        ),
        "mode_targets": {
            "a": {"8": 0.20, "60": 0.80},
            "b": {"21": 0.90, "59": 0.10},
        },
        "cue_multisets": {
            mode: list(actions) for mode, actions in _cue_multisets().items()
        },
        "cue_order": "one common seed-index permutation applied to both multisets",
        "scripted_learner_policy": {
            "dropper": 1,
            "checker": 60,
            "latent_mode_independent": True,
        },
        "cover_action": int(cover_action),
        "cover_games": list(covers),
        "primary_cover_games": max(covers),
        "twin_seeds": list(seeds),
        "start_clock": int(start_clock),
        "max_half_rounds_per_prefix_game": 1,
        "target_intervention": "correct-hidden / swapped-hidden / zero-hidden",
        "cover_timing": {
            "policy_and_forecast": (
                "cover_games=N means N identical cover reveals are processed before "
                "the measured output; the Nth reveal is in the frozen target token"
            ),
            "pre_target_hidden_l2": "after N-1 common cover tokens",
            "post_target_hidden_l2": "after all N common cover tokens",
        },
        "fast_adaptation": False,
        "primary_metric": "symmetric exact expected-payoff correct-minus-swapped",
        "bootstrap_unit": "twin_seed",
        "practical_effect_thresholds": {
            "normalized_payoff_crossover": 0.02,
            "opponent_nll_nats": 0.01,
            "policy_total_variation": 0.01,
        },
        "inference_rules": {
            "directional_action": (
                "pooled payoff CI lower bound > 0, both pooled mode means > 0, "
                "and every role-by-mode payoff CI lower bound > 0"
            ),
            "practical_action": (
                "directional_action and pooled normalized-payoff CI lower bound > 0.02"
            ),
            "directional_forecast": (
                "pooled NLL-gain CI lower bound > 0 and every role-by-mode "
                "NLL-gain CI lower bound > 0"
            ),
            "practical_forecast": (
                "directional_forecast and pooled NLL-gain CI lower bound > 0.01 nat"
            ),
            "practical_no_adaptation_equivalence": (
                "pooled and every role-by-mode normalized-payoff CI within +/-0.02; "
                "pooled and every role-by-mode NLL-gain CI within +/-0.01 nat; "
                "pooled and each role policy-TV CI upper bound <= 0.01"
            ),
            "long_horizon": (
                "primary cover delay >= 8 and both practical action and practical "
                "forecast gates pass"
            ),
        },
    }
    protocol["protocol_sha256"] = hashlib.sha256(_canonical_json(protocol)).hexdigest()
    return protocol


@dataclass(frozen=True, slots=True)
class ProbeToken:
    """The exact tensors consumed for one recurrent network decision."""

    features: np.ndarray
    stage_matrix: np.ndarray
    exact_policy: np.ndarray
    role_is_dropper: bool
    legal_mask: np.ndarray

    @classmethod
    def from_decision(cls, decision: AggroDecision) -> "ProbeToken":
        legal = np.asarray(
            [
                action in decision.canonical_decision.legal_seconds
                for action in range(1, ACTION_COUNT + 1)
            ],
            dtype=np.bool_,
        )
        return cls(
            features=np.asarray(decision.observation, dtype=np.float32).copy(),
            stage_matrix=np.asarray(decision.stage_matrix, dtype=np.float32).copy(),
            exact_policy=np.asarray(decision.exact_policy, dtype=np.float32).copy(),
            role_is_dropper=decision.role == "dropper",
            legal_mask=legal,
        )

    def bitwise_equal(self, other: "ProbeToken") -> bool:
        def same_array(left: np.ndarray, right: np.ndarray) -> bool:
            return (
                left.dtype == right.dtype
                and left.shape == right.shape
                and np.ascontiguousarray(left).tobytes()
                == np.ascontiguousarray(right).tobytes()
            )

        return (
            self.role_is_dropper == other.role_is_dropper
            and same_array(self.features, other.features)
            and same_array(self.stage_matrix, other.stage_matrix)
            and same_array(self.exact_policy, other.exact_policy)
            and same_array(self.legal_mask, other.legal_mask)
        )

    def sha256(self) -> str:
        digest = hashlib.sha256()
        for label, array in (
            ("features", self.features),
            ("stage_matrix", self.stage_matrix),
            ("exact_policy", self.exact_policy),
            ("legal_mask", self.legal_mask),
        ):
            contiguous = np.ascontiguousarray(array)
            digest.update(label.encode("ascii"))
            digest.update(str(contiguous.dtype).encode("ascii"))
            digest.update(_canonical_json(list(contiguous.shape)))
            digest.update(contiguous.tobytes())
        digest.update(bytes((int(self.role_is_dropper),)))
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class LatentTwinCase:
    """Two legal histories and one frozen target decision."""

    twin_seed: int
    role: str
    cover_games: int
    cue_actions_a: tuple[int, ...]
    cue_actions_b: tuple[int, ...]
    prefix_a: tuple[ProbeToken, ...]
    prefix_b: tuple[ProbeToken, ...]
    target: ProbeToken
    truth_a: np.ndarray
    truth_b: np.ndarray
    target_sha256: str
    target_state: tuple[int, int, int, int]


class _LatentTwinOpponent:
    """Scheduled synthetic opponent used only to build legal public prefixes."""

    def __init__(
        self,
        *,
        cue_actions: Sequence[int],
        cover_games: int,
        cover_action: int,
        target_truth: np.ndarray,
    ) -> None:
        self.cue_actions = tuple(int(action) for action in cue_actions)
        self.cover_games = int(cover_games)
        self.cover_action = int(cover_action)
        self.target_truth = np.asarray(target_truth, dtype=np.float64).copy()
        if not self.cue_actions or any(
            not 1 <= action <= ACTION_COUNT for action in self.cue_actions
        ):
            raise ValueError("cue actions must be nonempty and lie in 1..60")
        if self.cover_games <= 0 or not 1 <= self.cover_action <= ACTION_COUNT:
            raise ValueError("cover schedule is invalid")
        if (
            self.target_truth.shape != (ACTION_COUNT,)
            or np.any(self.target_truth < 0.0)
            or not np.isclose(float(self.target_truth.sum()), 1.0)
        ):
            raise ValueError("target truth must be a length-60 distribution")
        self._game_index = -1

    @property
    def target_game_index(self) -> int:
        return len(self.cue_actions) + self.cover_games

    def reset_session(self) -> None:
        self._game_index = -1

    def reset_game(self) -> None:
        self._game_index += 1

    def _distribution(self) -> np.ndarray:
        if self._game_index < 0:
            raise RuntimeError("latent twin acted before game reset")
        if self._game_index < len(self.cue_actions):
            action = self.cue_actions[self._game_index]
            result = np.zeros(ACTION_COUNT, dtype=np.float64)
            result[action - 1] = 1.0
            return result
        if self._game_index < self.target_game_index:
            result = np.zeros(ACTION_COUNT, dtype=np.float64)
            result[self.cover_action - 1] = 1.0
            return result
        if self._game_index == self.target_game_index:
            return self.target_truth.copy()
        raise RuntimeError("latent twin schedule advanced beyond its target")

    def true_distribution(self, decision: CanonicalDecision) -> np.ndarray:
        del decision
        return self._distribution()

    def policy(self, decision: CanonicalDecision) -> Mapping[int, float]:
        del decision
        distribution = self._distribution()
        return {
            action: float(distribution[action - 1])
            for action in range(1, ACTION_COUNT + 1)
            if distribution[action - 1] > 0.0
        }

    def observe(self, record: PublicHalfRound) -> None:
        del record

    def end_game(self, outcome: PublicGameOutcome) -> None:
        del outcome


def _scripted_learner_action(decision: AggroDecision) -> int:
    # This makes every cue action produce the same win/loss class.  The twin
    # histories consequently differ in opponent action one-hots, not outcomes.
    return 1 if decision.role == "dropper" else ACTION_COUNT


def _cue_sequences(twin_seed: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    bases = _cue_multisets()
    permutation = np.random.default_rng(int(twin_seed) + 104_729).permutation(
        len(bases["a"])
    )
    return tuple(np.asarray(bases["a"])[permutation]), tuple(
        np.asarray(bases["b"])[permutation]
    )


def _collect_prefix(
    exact_agent: CompleteDTHAgent,
    *,
    role: str,
    cue_actions: Sequence[int],
    cover_games: int,
    cover_action: int,
    target_truth: np.ndarray,
    session_seed: int,
    start_clock: int,
) -> tuple[tuple[ProbeToken, ...], ProbeToken, tuple[int, int, int, int]]:
    target_index = len(cue_actions) + cover_games
    learner_is_hal = role == "dropper"
    starts_hal = learner_is_hal if target_index % 2 == 0 else not learner_is_hal
    opponent = _LatentTwinOpponent(
        cue_actions=cue_actions,
        cover_games=cover_games,
        cover_action=cover_action,
        target_truth=target_truth,
    )
    env = AggroSessionEnv(
        opponent,
        exact_agent,
        games_per_session=target_index + 1,
        seed=session_seed,
        start_clocks=(start_clock,),
        max_half_rounds=1,
        learner_starts_in_hal_seat=starts_hal,
    )
    decision = env.reset(seed=session_seed)
    prefix: list[ProbeToken] = []
    try:
        while decision.game_index < target_index:
            prefix.append(ProbeToken.from_decision(decision))
            step = env.step(_scripted_learner_action(decision))
            if step.next_decision is None:
                raise RuntimeError("latent twin prefix ended before its target")
            decision = step.next_decision
        if decision.game_index != target_index or decision.role != role:
            raise RuntimeError("latent twin target role/index is malformed")
        target = ProbeToken.from_decision(decision)
        state = tuple(int(value) for value in decision.dth_state)
        if len(state) != 4:
            raise RuntimeError("latent twin target is not a four-coordinate DTH state")
        return tuple(prefix), target, state  # type: ignore[return-value]
    finally:
        env.close()


def _assert_prefix_integrity(
    prefix_a: Sequence[ProbeToken],
    prefix_b: Sequence[ProbeToken],
    *,
    cue_games: int,
) -> None:
    if len(prefix_a) != len(prefix_b) or len(prefix_a) <= cue_games:
        raise RuntimeError("latent twin prefixes are not equal nontrivial sequences")
    allowed = np.asarray(
        [
            name.startswith("previous_drop_action_")
            or name.startswith("previous_check_action_")
            for name in OBSERVATION_FEATURES
        ],
        dtype=np.bool_,
    )
    saw_cue_difference = False
    for index, (left, right) in enumerate(zip(prefix_a, prefix_b, strict=True)):
        if (
            left.role_is_dropper != right.role_is_dropper
            or not np.array_equal(left.stage_matrix, right.stage_matrix)
            or not np.array_equal(left.exact_policy, right.exact_policy)
            or not np.array_equal(left.legal_mask, right.legal_mask)
        ):
            raise RuntimeError("latent twin mechanical prefix tensors differ")
        difference = left.features != right.features
        if np.any(difference & ~allowed):
            raise RuntimeError("latent twins differ outside revealed action cue slots")
        if np.any(difference):
            if not 1 <= index <= cue_games:
                raise RuntimeError(
                    "latent twin cue leaked into the common cover suffix"
                )
            saw_cue_difference = True
    if not saw_cue_difference:
        raise RuntimeError("latent twin prefixes contain no distinct cue")


def build_latent_twin_case(
    exact_agent: CompleteDTHAgent,
    *,
    role: str,
    cover_games: int,
    twin_seed: int,
    start_clock: int = DEFAULT_START_CLOCK,
    cover_action: int = DEFAULT_COVER_ACTION,
) -> LatentTwinCase:
    """Construct one matched legal-prefix intervention case."""

    if role not in _ROLES:
        raise ValueError("role must be dropper or checker")
    if cover_games <= 0:
        raise ValueError("cover_games must be positive so the target token is common")
    if twin_seed < 0:
        raise ValueError("twin_seed must be nonnegative")
    truths = _target_distributions()
    cue_a, cue_b = _cue_sequences(twin_seed)
    session_seed = 500_000_000 + int(twin_seed) * 100 + int(cover_games)
    prefix_a, target_a, state_a = _collect_prefix(
        exact_agent,
        role=role,
        cue_actions=cue_a,
        cover_games=cover_games,
        cover_action=cover_action,
        target_truth=truths["a"],
        session_seed=session_seed,
        start_clock=start_clock,
    )
    prefix_b, target_b, state_b = _collect_prefix(
        exact_agent,
        role=role,
        cue_actions=cue_b,
        cover_games=cover_games,
        cover_action=cover_action,
        target_truth=truths["b"],
        session_seed=session_seed,
        start_clock=start_clock,
    )
    _assert_prefix_integrity(prefix_a, prefix_b, cue_games=len(cue_a))
    if state_a != state_b or not target_a.bitwise_equal(target_b):
        raise RuntimeError("latent twin target tensors are not bitwise identical")
    return LatentTwinCase(
        twin_seed=int(twin_seed),
        role=role,
        cover_games=int(cover_games),
        cue_actions_a=cue_a,
        cue_actions_b=cue_b,
        prefix_a=prefix_a,
        prefix_b=prefix_b,
        target=target_a,
        truth_a=truths["a"],
        truth_b=truths["b"],
        target_sha256=target_a.sha256(),
        target_state=state_a,
    )


def _model_device(model: AggroHalNetwork) -> torch.device:
    device = next(model.parameters()).device
    if device.type != "cpu":
        raise ValueError("latent-twin evaluation is CPU-only")
    return device


def _forward_tokens(
    model: AggroHalNetwork,
    tokens: Sequence[ProbeToken],
    hidden_state: Tensor | None = None,
):
    if not tokens:
        raise ValueError("at least one probe token is required")
    device = _model_device(model)
    features = torch.as_tensor(
        np.stack([token.features for token in tokens]),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    matrices = torch.as_tensor(
        np.stack([token.stage_matrix for token in tokens]),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    exact = torch.as_tensor(
        np.stack([token.exact_policy for token in tokens]),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    roles = torch.as_tensor(
        [[token.role_is_dropper for token in tokens]],
        dtype=torch.bool,
        device=device,
    )
    legal = torch.as_tensor(
        np.stack([token.legal_mask for token in tokens]),
        dtype=torch.bool,
        device=device,
    ).unsqueeze(0)
    return model(features, matrices, exact, roles, legal, hidden_state)


@dataclass(frozen=True, slots=True)
class _ArmOutput:
    policy: np.ndarray
    opponent_policy: np.ndarray
    direct_policy: np.ndarray
    direct_weight: float


def _arm(output) -> _ArmOutput:
    return _ArmOutput(
        policy=output.policy[0, -1].detach().cpu().numpy().astype(np.float64),
        opponent_policy=(
            output.opponent_policy[0, -1].detach().cpu().numpy().astype(np.float64)
        ),
        direct_policy=(
            output.direct_policy[0, -1].detach().cpu().numpy().astype(np.float64)
        ),
        direct_weight=float(output.direct_weight[0, -1].detach().cpu()),
    )


def _score_arm(
    arm: _ArmOutput,
    *,
    truth: np.ndarray,
    action_values: np.ndarray,
) -> dict[str, float | int]:
    tiny = 1e-12
    expected_payoff = float(np.dot(arm.policy, action_values))
    oracle = float(np.max(action_values))
    best_action = int(np.argmax(action_values)) + 1
    return {
        "expected_payoff": expected_payoff,
        "oracle_payoff": oracle,
        "oracle_regret": max(0.0, oracle - expected_payoff),
        "opponent_expected_nll": float(
            -np.sum(truth * np.log(np.clip(arm.opponent_policy, tiny, 1.0)))
        ),
        "best_response_action": best_action,
        "best_response_mass": float(arm.policy[best_action - 1]),
        "top_action": int(np.argmax(arm.policy)) + 1,
        "top_action_probability": float(np.max(arm.policy)),
        "direct_top_action": int(np.argmax(arm.direct_policy)) + 1,
        "direct_weight": arm.direct_weight,
    }


def symmetric_crossover_metrics(
    *,
    payoff_a_correct: float,
    payoff_a_swapped: float,
    payoff_a_zero: float,
    payoff_b_correct: float,
    payoff_b_swapped: float,
    payoff_b_zero: float,
    nll_a_correct: float,
    nll_a_swapped: float,
    nll_a_zero: float,
    nll_b_correct: float,
    nll_b_swapped: float,
    nll_b_zero: float,
) -> dict[str, float]:
    """Compute the symmetric causal contrasts before aggregation."""

    mode_a_payoff = payoff_a_correct - payoff_a_swapped
    mode_b_payoff = payoff_b_correct - payoff_b_swapped
    mode_a_nll = nll_a_swapped - nll_a_correct
    mode_b_nll = nll_b_swapped - nll_b_correct
    return {
        "payoff_correct_minus_swapped": 0.5 * (mode_a_payoff + mode_b_payoff),
        "payoff_correct_minus_zero": 0.5
        * ((payoff_a_correct - payoff_a_zero) + (payoff_b_correct - payoff_b_zero)),
        "nll_swapped_minus_correct": 0.5 * (mode_a_nll + mode_b_nll),
        "nll_zero_minus_correct": 0.5
        * ((nll_a_zero - nll_a_correct) + (nll_b_zero - nll_b_correct)),
        "mode_a_nll_swapped_minus_correct": mode_a_nll,
        "mode_b_nll_swapped_minus_correct": mode_b_nll,
        "mode_a_payoff_correct_minus_swapped": mode_a_payoff,
        "mode_b_payoff_correct_minus_swapped": mode_b_payoff,
    }


def evaluate_latent_twin_case(
    model: AggroHalNetwork,
    case: LatentTwinCase,
) -> dict[str, object]:
    """Evaluate one frozen target under correct, swapped, and erased history."""

    model.eval()
    with torch.inference_mode():
        prefix_a = _forward_tokens(model, case.prefix_a)
        prefix_b = _forward_tokens(model, case.prefix_b)
        hidden_a = prefix_a.hidden_state.detach().clone()
        hidden_b = prefix_b.hidden_state.detach().clone()
        zero = model.initial_hidden(1, device=_model_device(model))
        raw_output_a = _forward_tokens(model, (case.target,), hidden_a)
        raw_output_b = _forward_tokens(model, (case.target,), hidden_b)
        raw_output_zero = _forward_tokens(model, (case.target,), zero)
        output_a = _arm(raw_output_a)
        output_b = _arm(raw_output_b)
        output_zero = _arm(raw_output_zero)

    matrix = np.asarray(case.target.stage_matrix, dtype=np.float64)
    oriented = matrix if case.role == "dropper" else -matrix.T
    values_a = oriented @ case.truth_a
    values_b = oriented @ case.truth_b
    scores = {
        "a": {
            "correct": _score_arm(output_a, truth=case.truth_a, action_values=values_a),
            "swapped": _score_arm(output_b, truth=case.truth_a, action_values=values_a),
            "zero": _score_arm(output_zero, truth=case.truth_a, action_values=values_a),
        },
        "b": {
            "correct": _score_arm(output_b, truth=case.truth_b, action_values=values_b),
            "swapped": _score_arm(output_a, truth=case.truth_b, action_values=values_b),
            "zero": _score_arm(output_zero, truth=case.truth_b, action_values=values_b),
        },
    }
    a, b = scores["a"], scores["b"]
    contrasts = symmetric_crossover_metrics(
        payoff_a_correct=float(a["correct"]["expected_payoff"]),
        payoff_a_swapped=float(a["swapped"]["expected_payoff"]),
        payoff_a_zero=float(a["zero"]["expected_payoff"]),
        payoff_b_correct=float(b["correct"]["expected_payoff"]),
        payoff_b_swapped=float(b["swapped"]["expected_payoff"]),
        payoff_b_zero=float(b["zero"]["expected_payoff"]),
        nll_a_correct=float(a["correct"]["opponent_expected_nll"]),
        nll_a_swapped=float(a["swapped"]["opponent_expected_nll"]),
        nll_a_zero=float(a["zero"]["opponent_expected_nll"]),
        nll_b_correct=float(b["correct"]["opponent_expected_nll"]),
        nll_b_swapped=float(b["swapped"]["opponent_expected_nll"]),
        nll_b_zero=float(b["zero"]["opponent_expected_nll"]),
    )
    reference_a = float(np.max(values_a)) - float(values_a[int(np.argmax(values_b))])
    reference_b = float(np.max(values_b)) - float(values_b[int(np.argmax(values_a))])
    reference = 0.5 * (reference_a + reference_b)
    if min(reference_a, reference_b) <= 0.0:
        raise RuntimeError(
            "latent target policies do not require conflicting responses"
        )
    contrasts.update(
        {
            "policy_total_variation": 0.5
            * float(np.sum(np.abs(output_a.policy - output_b.policy))),
            "forecast_total_variation": 0.5
            * float(
                np.sum(np.abs(output_a.opponent_policy - output_b.opponent_policy))
            ),
            # The target contains the final common cover reveal.  Report both
            # sides of that update so cover_games=N is unambiguously an N-token
            # delay at the policy/forecast output.
            "pre_target_hidden_l2": float(
                torch.linalg.vector_norm(hidden_a - hidden_b).cpu()
            ),
            "post_target_hidden_l2": float(
                torch.linalg.vector_norm(
                    raw_output_a.hidden_state - raw_output_b.hidden_state
                ).cpu()
            ),
            "normalized_payoff_crossover": (
                contrasts["payoff_correct_minus_swapped"] / reference
            ),
            "mode_a_normalized_payoff_crossover": (
                contrasts["mode_a_payoff_correct_minus_swapped"] / reference_a
            ),
            "mode_b_normalized_payoff_crossover": (
                contrasts["mode_b_payoff_correct_minus_swapped"] / reference_b
            ),
        }
    )
    return {
        "twin_seed": case.twin_seed,
        "role": case.role,
        "cover_games": case.cover_games,
        "prefix_tokens": len(case.prefix_a),
        "target_state": list(case.target_state),
        "target_sha256": case.target_sha256,
        "target_identity_passed": True,
        "reference_wrong_mode_penalty": reference,
        "reference_wrong_mode_penalties": {"a": reference_a, "b": reference_b},
        "contrasts": contrasts,
        "modes": scores,
    }


def cluster_bootstrap_interval(
    values_by_seed: Mapping[int, float],
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap independent twin-seed units deterministically."""

    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    ordered = np.asarray(
        [float(values_by_seed[key]) for key in sorted(values_by_seed)],
        dtype=np.float64,
    )
    if ordered.size == 0 or not np.all(np.isfinite(ordered)):
        raise ValueError("bootstrap values must be finite and nonempty")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, ordered.size, size=(replicates, ordered.size))
    means = ordered[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def _aggregate_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    if not rows:
        raise ValueError("cannot aggregate an empty latent-twin slice")
    result: dict[str, object] = {
        "twin_seed_units": len({row["twin_seed"] for row in rows})
    }
    metrics: dict[str, object] = {}
    for metric_index, metric in enumerate(_METRICS):
        by_seed: dict[int, list[float]] = {}
        for row in rows:
            seed = int(row["twin_seed"])
            contrasts = row["contrasts"]
            if not isinstance(contrasts, Mapping):
                raise TypeError("latent-twin contrasts are malformed")
            by_seed.setdefault(seed, []).append(float(contrasts[metric]))
        clustered = {seed: float(np.mean(values)) for seed, values in by_seed.items()}
        metrics[metric] = {
            "mean": float(np.mean(list(clustered.values()))),
            "bootstrap_95": list(
                cluster_bootstrap_interval(
                    clustered,
                    replicates=bootstrap_replicates,
                    seed=bootstrap_seed + metric_index * 1009,
                )
            ),
        }
    result["metrics"] = metrics
    return result


def _target_audit(agent: CompleteDTHAgent) -> dict[str, object]:
    stage = agent.stage_game((0, 0, 0, 0))
    matrix = np.asarray(stage.matrix, dtype=np.float64)
    truths = _target_distributions()
    result: dict[str, object] = {
        "state": [0, 0, 0, 0],
        "stage_value": float(stage.value),
        "saddle_gap": float(stage.saddle_gap),
        "roles": {},
    }
    for role in _ROLES:
        oriented = matrix if role == "dropper" else -matrix.T
        values = {mode: oriented @ truth for mode, truth in truths.items()}
        best = {mode: int(np.argmax(vector)) for mode, vector in values.items()}
        if best["a"] == best["b"]:
            raise RuntimeError(f"latent targets do not cross over for {role}")
        role_result: dict[str, object] = {}
        for mode, other in (("a", "b"), ("b", "a")):
            vector = values[mode]
            sorted_values = np.sort(vector)
            unique_margin = float(sorted_values[-1] - sorted_values[-2])
            wrong_mode_penalty = float(vector[best[mode]] - vector[best[other]])
            if unique_margin < 0.004 or wrong_mode_penalty <= 0.0:
                raise RuntimeError(
                    f"latent target certificate failed for {role}/{mode}: "
                    f"margin={unique_margin}, penalty={wrong_mode_penalty}"
                )
            role_result[mode] = {
                "best_response_action": best[mode] + 1,
                "oracle_payoff": float(vector[best[mode]]),
                "unique_best_margin": unique_margin,
                "wrong_mode_penalty": wrong_mode_penalty,
            }
        result["roles"][role] = role_result  # type: ignore[index]
    return result


def _metric(summary: Mapping[str, object], name: str) -> Mapping[str, object]:
    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping) or not isinstance(metrics.get(name), Mapping):
        raise TypeError(f"latent-twin summary is missing metric {name!r}")
    return metrics[name]  # type: ignore[return-value]


def _interval(summary: Mapping[str, object], name: str) -> tuple[float, float]:
    interval = _metric(summary, name).get("bootstrap_95")
    if not isinstance(interval, Sequence) or len(interval) != 2:
        raise TypeError(f"latent-twin metric {name!r} has no two-sided interval")
    return float(interval[0]), float(interval[1])


def _mean(summary: Mapping[str, object], name: str) -> float:
    return float(_metric(summary, name)["mean"])


def adaptation_conclusion(
    slices: Mapping[str, object],
    *,
    primary_cover: int,
    thresholds: Mapping[str, object],
) -> dict[str, object]:
    """Apply role- and mode-stratified directional and equivalence gates."""

    by_cover = slices.get("by_cover")
    by_cover_and_role = slices.get("by_cover_and_role")
    if not isinstance(by_cover, Mapping) or not isinstance(by_cover_and_role, Mapping):
        raise TypeError("latent-twin slices are malformed")
    primary = by_cover.get(str(primary_cover))
    roles = by_cover_and_role.get(str(primary_cover))
    if not isinstance(primary, Mapping) or not isinstance(roles, Mapping):
        raise TypeError("latent-twin primary slice is malformed")
    role_summaries: dict[str, Mapping[str, object]] = {}
    for role in _ROLES:
        summary = roles.get(role)
        if not isinstance(summary, Mapping):
            raise TypeError(f"latent-twin primary slice is missing role {role!r}")
        role_summaries[role] = summary

    action_rope = float(thresholds["normalized_payoff_crossover"])
    nll_rope = float(thresholds["opponent_nll_nats"])
    policy_rope = float(thresholds["policy_total_variation"])
    if min(action_rope, nll_rope, policy_rope) <= 0.0:
        raise ValueError("latent-twin practical thresholds must be positive")

    pooled_action_ci = _interval(primary, "payoff_correct_minus_swapped")
    pooled_forecast_ci = _interval(primary, "nll_swapped_minus_correct")
    pooled_normalized_ci = _interval(primary, "normalized_payoff_crossover")
    pooled_policy_ci = _interval(primary, "policy_total_variation")
    pooled_action_direction = bool(
        pooled_action_ci[0] > 0.0
        and _mean(primary, "mode_a_payoff_correct_minus_swapped") > 0.0
        and _mean(primary, "mode_b_payoff_correct_minus_swapped") > 0.0
    )
    pooled_forecast_direction = bool(pooled_forecast_ci[0] > 0.0)

    role_mode_action_direction = all(
        _interval(summary, f"mode_{mode}_payoff_correct_minus_swapped")[0] > 0.0
        for summary in role_summaries.values()
        for mode in _MODES
    )
    role_mode_forecast_direction = all(
        _interval(summary, f"mode_{mode}_nll_swapped_minus_correct")[0] > 0.0
        for summary in role_summaries.values()
        for mode in _MODES
    )
    directional_action = bool(pooled_action_direction and role_mode_action_direction)
    directional_forecast = bool(
        pooled_forecast_direction and role_mode_forecast_direction
    )
    practical_action = bool(
        directional_action and pooled_normalized_ci[0] > action_rope
    )
    practical_forecast = bool(directional_forecast and pooled_forecast_ci[0] > nll_rope)

    def inside(interval: tuple[float, float], radius: float) -> bool:
        return interval[0] >= -radius and interval[1] <= radius

    pooled_equivalence = bool(
        inside(pooled_normalized_ci, action_rope)
        and inside(pooled_forecast_ci, nll_rope)
        and pooled_policy_ci[1] <= policy_rope
    )
    stratified_equivalence = all(
        inside(
            _interval(summary, f"mode_{mode}_normalized_payoff_crossover"),
            action_rope,
        )
        and inside(
            _interval(summary, f"mode_{mode}_nll_swapped_minus_correct"),
            nll_rope,
        )
        for summary in role_summaries.values()
        for mode in _MODES
    ) and all(
        _interval(summary, "policy_total_variation")[1] <= policy_rope
        for summary in role_summaries.values()
    )

    return {
        "causal_recurrent_policy_sensitivity_detected": (pooled_policy_ci[0] > 0.0),
        "causal_recurrent_forecast_sensitivity_detected": (
            _interval(primary, "forecast_total_variation")[0] > 0.0
        ),
        "pooled_directional_action_effect_detected": pooled_action_direction,
        "pooled_directional_forecast_effect_detected": pooled_forecast_direction,
        "directional_action_effect_detected": directional_action,
        "directional_forecast_effect_detected": directional_forecast,
        "causal_recurrent_action_use_supported": practical_action,
        "causal_recurrent_forecast_use_supported": practical_forecast,
        "forecast_to_control_adaptation_supported": (
            practical_action and practical_forecast
        ),
        "long_horizon_adaptation_supported": (
            primary_cover >= 8 and practical_action and practical_forecast
        ),
        "practical_no_adaptation_equivalence_supported": (
            pooled_equivalence and stratified_equivalence
        ),
        "scope": (
            "This tests transfer to a synthetic latent-twin task absent from the "
            "training league; failure is not an architectural impossibility result."
        ),
    }


def evaluate_latent_twin_probe(
    *,
    checkpoint: str | Path,
    artifact_dir: str | Path = DEFAULT_ARTIFACT,
    twin_seeds: Sequence[int] = tuple(range(DEFAULT_TWIN_SEEDS)),
    cover_games: Sequence[int] = DEFAULT_COVER_GAMES,
    start_clock: int = DEFAULT_START_CLOCK,
    cover_action: int = DEFAULT_COVER_ACTION,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    cpu_threads: int = 6,
) -> dict[str, object]:
    """Run the complete frozen-checkpoint causal memory probe."""

    if cpu_threads <= 0:
        raise ValueError("cpu_threads must be positive")
    protocol = latent_twin_protocol(
        twin_seeds=twin_seeds,
        cover_games=cover_games,
        start_clock=start_clock,
        cover_action=cover_action,
    )
    torch.set_num_threads(int(cpu_threads))
    agent = CompleteDTHAgent(Path(artifact_dir))
    model, _ = load_checkpoint(
        checkpoint,
        dth_ruleset=dth_compatibility(agent),
        device="cpu",
    )
    rows: list[dict[str, object]] = []
    for cover in tuple(int(value) for value in cover_games):
        for role in _ROLES:
            for twin_seed in tuple(int(seed) for seed in twin_seeds):
                case = build_latent_twin_case(
                    agent,
                    role=role,
                    cover_games=cover,
                    twin_seed=twin_seed,
                    start_clock=start_clock,
                    cover_action=cover_action,
                )
                rows.append(evaluate_latent_twin_case(model, case))

    slices: dict[str, object] = {"by_cover_and_role": {}, "by_cover": {}}
    for cover_index, cover in enumerate(
        sorted({int(row["cover_games"]) for row in rows})
    ):
        cover_rows = [row for row in rows if int(row["cover_games"]) == cover]
        slices["by_cover"][str(cover)] = _aggregate_rows(  # type: ignore[index]
            cover_rows,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed + cover_index * 10_000,
        )
        slices["by_cover_and_role"][str(cover)] = {  # type: ignore[index]
            role: _aggregate_rows(
                [row for row in cover_rows if row["role"] == role],
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed
                + cover_index * 10_000
                + role_index * 1_000,
            )
            for role_index, role in enumerate(_ROLES)
        }
    overall = _aggregate_rows(
        rows,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed + 90_000,
    )
    primary_cover = int(protocol["primary_cover_games"])
    thresholds = protocol["practical_effect_thresholds"]
    if not isinstance(thresholds, Mapping):
        raise TypeError("latent-twin practical thresholds are malformed")
    conclusion = adaptation_conclusion(
        slices,
        primary_cover=primary_cover,
        thresholds=thresholds,
    )
    return {
        "schema_version": PROBE_SCHEMA,
        "device": "cpu",
        "fast_adaptation": False,
        "checkpoint": str(Path(checkpoint)),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "artifact_dir": str(Path(artifact_dir)),
        "protocol": protocol,
        "target_audit": _target_audit(agent),
        "target_identity": {
            "all_passed": all(bool(row["target_identity_passed"]) for row in rows),
            "case_hashes": len({str(row["target_sha256"]) for row in rows}),
            "cases": len(rows),
        },
        "slices": slices,
        "overall": overall,
        "primary_cover_games": primary_cover,
        "conclusion": conclusion,
        "cases": rows,
    }


def _write_json(path: str | Path, payload: object) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-output", type=Path)
    parser.add_argument("--twin-seeds", type=int, default=DEFAULT_TWIN_SEEDS)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--cover-games",
        type=int,
        nargs="+",
        default=list(DEFAULT_COVER_GAMES),
    )
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--cpu-threads", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.twin_seeds <= 0 or args.seed_start < 0:
        raise ValueError("twin seed count must be positive and seed start nonnegative")
    seeds = tuple(range(args.seed_start, args.seed_start + args.twin_seeds))
    protocol = latent_twin_protocol(
        twin_seeds=seeds,
        cover_games=args.cover_games,
    )
    protocol_path = args.protocol_output or args.output.with_name(
        args.output.stem + "-protocol.json"
    )
    # Commit the checkpoint-independent protocol before loading model weights.
    _write_json(protocol_path, protocol)
    report = evaluate_latent_twin_probe(
        checkpoint=args.checkpoint,
        artifact_dir=args.artifact_dir,
        twin_seeds=seeds,
        cover_games=args.cover_games,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        cpu_threads=args.cpu_threads,
    )
    _write_json(args.output, report)
    summary = {
        "output": str(args.output),
        "protocol_output": str(protocol_path),
        "checkpoint_sha256": report["checkpoint_sha256"],
        "primary_cover_games": report["primary_cover_games"],
        "conclusion": report["conclusion"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
