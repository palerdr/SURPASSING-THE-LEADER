"""Record one match game and telescope the candidate's exact DTH value.

At half-round t the candidate plays the mixed policy pi_t, the opponent plays
mu_t, and H_t is the candidate-view stage matrix of the complete tablebase.
The product pi_t @ H_t @ mu_t is the expected equilibrium value of the next
state, and W_H(s_t) is the equilibrium value of the current state. The gift
g_t = pi_t @ H_t @ mu_t - W_H(s_t) measures what the pair of policies moves
away from equilibrium value in one half-round.

For a finished pure-DTH game the realized outcome U (+1 win, -1 loss) equals
W_H(s_0) + sum_t g_t plus a sum of zero-mean terms
V(s_t+1) - E[V(s_t+1) | s_t, pi_t, mu_t].
The telescoped value W_H(s_0) + sum_t g_t has the expectation of U. A game
that reaches the half-round cap scores 0.5 raw, while its telescoped value
estimates the equilibrium value of the cap state, so the identity covers
finished games. Leap turns change the transition, so canonical games report
the raw result alone.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from arena.agent import normalize_legal_policy
from arena.contracts import CanonicalDecision, observe_provider
from arena.dth_adapter import project_to_dth_state
from arena.match import play_match_game
from arena.policies.exploit_continuation import hal_view

ACTION_COUNT = 60
TELESCOPED_FIELDS = (
    "initial_value",
    "gift_sum",
    "telescoped_value",
    "score_hat",
    "realized_gift_sum",
    "one_step_regret",
    "max_abs_gift",
)


class RecordingProvider:
    """Record each policy and each reveal of one provider without changing them.

    ``decisions`` holds ``(decision, mapping)`` pairs in call order, and
    ``records`` holds the revealed half-rounds. Every other attribute reads
    through to the wrapped provider, so the lifecycle hooks of
    ``arena.contracts`` reach it unchanged.
    """

    def __init__(self, provider) -> None:
        self.provider = provider
        self.decisions: list[tuple[CanonicalDecision, dict[int, float]]] = []
        self.records: list = []

    def __getattr__(self, name):
        if name in {"provider", "decisions", "records"}:
            raise AttributeError(name)
        return getattr(self.provider, name)

    def policy(self, decision: CanonicalDecision) -> Mapping[int, float]:
        mapping = self.provider.policy(decision)
        self.decisions.append((decision, dict(mapping)))
        return mapping

    def observe(self, record) -> None:
        self.records.append(record)
        observe_provider(self.provider, record)


def policy_vector(
    mapping: Mapping[int, float],
    decision: CanonicalDecision,
) -> np.ndarray:
    """Return the length-60 distribution that ``PolicyDrivenAgent`` samples."""

    actions, probabilities = normalize_legal_policy(mapping, decision.legal_seconds)
    if np.any(actions < 1) or np.any(actions > ACTION_COUNT):
        raise ValueError("a pure-DTH policy must lie on literal seconds 1..60")
    vector = np.zeros(ACTION_COUNT, dtype=np.float64)
    vector[actions - 1] = probabilities
    return vector


def telescope(agent, candidate: RecordingProvider, opponent: RecordingProvider) -> dict:
    """Pair the t-th decisions and reveal, then return per-half-round terms.

    The result holds arrays of the candidate-view state values ``values``, the
    expected gifts ``gifts``, the realized gifts ``realized_gifts``, and the
    one-step regrets ``regrets``.
    """

    count = len(candidate.decisions)
    if not (
        count
        == len(opponent.decisions)
        == len(candidate.records)
        == len(opponent.records)
    ):
        raise RuntimeError("recorded decisions and reveals do not pair by half-round")
    values = np.empty(count, dtype=np.float64)
    gifts = np.empty(count, dtype=np.float64)
    realized = np.empty(count, dtype=np.float64)
    regrets = np.empty(count, dtype=np.float64)
    rows = zip(candidate.decisions, opponent.decisions, candidate.records, strict=True)
    for t, ((decision, mapping), (other, other_mapping), record) in enumerate(rows):
        state = project_to_dth_state(decision)
        if project_to_dth_state(other) != state or other.role == decision.role:
            raise RuntimeError("paired decisions disagree on the state or the roles")
        if decision.role == "dropper":
            own_name, opponent_action = record.dropper_name, record.check_time
        else:
            own_name, opponent_action = record.checker_name, record.drop_time
        if own_name.casefold() != decision.actor_name.casefold():
            raise RuntimeError("a paired reveal does not match the candidate's role")
        stage = agent.stage_game(state)
        matrix, value = hal_view(stage.matrix, stage.value, decision.role)
        matrix = np.asarray(matrix, dtype=np.float64)
        pi = policy_vector(mapping, decision)
        mu = policy_vector(other_mapping, other)
        response = matrix @ mu
        expected = float(pi @ response)
        values[t] = float(value)
        gifts[t] = expected - values[t]
        realized[t] = float(pi @ matrix[:, int(opponent_action) - 1]) - values[t]
        regrets[t] = float(np.max(response)) - expected
    return {
        "values": values,
        "gifts": gifts,
        "realized_gifts": realized,
        "regrets": regrets,
    }


def play_recorded_game(
    candidate,
    opponent,
    *,
    agent,
    seed: int,
    start_clock: int,
    max_half_rounds: int,
    game_index: int,
    candidate_first: bool,
    pure_dth: bool = True,
) -> dict:
    """Play one ``arena.match`` game and return raw and telescoped results.

    The candidate holds the Hal seat when ``candidate_first`` is true. The raw
    score is 1 for a win, 0 for a loss, and 0.5 for a game that reaches
    ``max_half_rounds``. ``score_hat`` is ``(1 + telescoped_value) / 2``. For
    ``pure_dth=False`` every telescoped field is None.
    """

    recorded_candidate = RecordingProvider(candidate)
    recorded_opponent = RecordingProvider(opponent)
    if candidate_first:
        first, second = recorded_candidate, recorded_opponent
    else:
        first, second = recorded_opponent, recorded_candidate
    winner, half_rounds = play_match_game(
        first,
        second,
        seed=seed,
        start_clock=start_clock,
        max_half_rounds=max_half_rounds,
        game_index=game_index,
        pure_dth=pure_dth,
    )
    seat = "Hal" if candidate_first else "Baku"
    stopped = winner is None
    won = None if stopped else winner == seat
    result = {
        "winner": winner,
        "seat": seat,
        "won": won,
        "half_rounds": int(half_rounds),
        "stopped": stopped,
        "score": 0.5 if stopped else float(won),
        **dict.fromkeys(TELESCOPED_FIELDS),
    }
    if not pure_dth or not recorded_candidate.decisions:
        return result
    terms = telescope(agent, recorded_candidate, recorded_opponent)
    initial = float(terms["values"][0])
    gift_sum = float(terms["gifts"].sum())
    telescoped = initial + gift_sum
    result.update(
        initial_value=initial,
        gift_sum=gift_sum,
        telescoped_value=telescoped,
        score_hat=(1.0 + telescoped) / 2.0,
        realized_gift_sum=float(terms["realized_gifts"].sum()),
        one_step_regret=float(terms["regrets"].sum()),
        max_abs_gift=float(np.max(np.abs(terms["gifts"]))),
    )
    return result


__all__ = [
    "RecordingProvider",
    "TELESCOPED_FIELDS",
    "play_recorded_game",
    "policy_vector",
    "telescope",
]
