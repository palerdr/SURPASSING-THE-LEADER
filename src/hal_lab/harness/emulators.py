"""Human emulators fit to recorded games beyond ``FittedHumanOpponent``."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from arena.contracts import CanonicalDecision, PublicHalfRound
from arena.policies.bayesian_hal import ROLES
from hal_lab.harness.human_data import human_observation

ACTION_COUNT = 60


def _rate(repeats: int, pairs: int) -> float:
    return (repeats + 1) / (pairs + 2)


class SelfRepeatHumanOpponent:
    """Repeat the human's last second in a role, or draw from the human's counts.

    The fit reads each recorded ordinary turn through ``human_observation``, as
    ``FittedHumanOpponent`` does. For each role it keeps a smoothed categorical
    over the human's own seconds and two repeat probabilities of the form
    ``(repeats + 1) / (pairs + 2)``:

    - ``repeat_rates`` counts pairs of consecutive own actions in the same role
      within one game. A leap turn breaks the sequence, as it does in
      ``FittedHumanOpponent``.
    - ``cross_game_rates`` counts pairs across a game boundary: the last own
      ordinary action in the role in one game and the first own ordinary
      action in that role in the next game of the list. The recorded humans
      repeat less often across a boundary than inside a game.

    In play the policy is ``r * delta(previous) + (1 - r) * categorical`` when
    the emulator has an own action in the role earlier in the session, and the
    categorical otherwise. ``r`` is the cross-game rate for the first decision
    in a role after ``reset_game`` and the within-game rate after that.
    """

    def __init__(self, games, *, pseudocount: float = 0.5) -> None:
        counts = {role: np.full(ACTION_COUNT, float(pseudocount)) for role in ROLES}
        self.pairs = {role: 0 for role in ROLES}
        self.repeats = {role: 0 for role in ROLES}
        self.cross_pairs = {role: 0 for role in ROLES}
        self.cross_repeats = {role: 0 for role in ROLES}
        last = {role: None for role in ROLES}
        for game in games:
            previous = {role: None for role in ROLES}
            first = {role: True for role in ROLES}
            for move in game["public_history"]:
                observation = human_observation(move)
                if observation is None:
                    previous = {role: None for role in ROLES}
                    continue
                _, role, action, _ = observation
                counts[role][action - 1] += 1
                if previous[role] is not None:
                    self.pairs[role] += 1
                    self.repeats[role] += int(action == previous[role])
                elif first[role] and last[role] is not None:
                    self.cross_pairs[role] += 1
                    self.cross_repeats[role] += int(action == last[role])
                first[role] = False
                previous[role] = action
                last[role] = action
        self.categorical = {role: values / values.sum() for role, values in counts.items()}
        self.repeat_rates = {
            role: _rate(self.repeats[role], self.pairs[role]) for role in ROLES
        }
        self.cross_game_rates = {
            role: _rate(self.cross_repeats[role], self.cross_pairs[role]) for role in ROLES
        }
        self.name: str | None = None
        self.previous: dict[str, int | None] = {role: None for role in ROLES}
        self.first_in_game: dict[str, bool] = {role: True for role in ROLES}

    def policy(self, decision: CanonicalDecision) -> Mapping[int, float]:
        self.name = decision.actor_name
        role = decision.role
        policy = self.categorical[role]
        previous = self.previous[role]
        if previous is not None:
            rates = self.cross_game_rates if self.first_in_game[role] else self.repeat_rates
            rate = rates[role]
            policy = (1.0 - rate) * policy
            policy[previous - 1] += rate
        return {second: float(p) for second, p in enumerate(policy, start=1)}

    def observe(self, record: PublicHalfRound) -> None:
        if self.name is None:
            raise RuntimeError("the emulator received a reveal before acting")
        if record.dropper_name.casefold() == self.name.casefold():
            role, action = "dropper", int(record.drop_time)
        elif record.checker_name.casefold() == self.name.casefold():
            role, action = "checker", int(record.check_time)
        else:
            raise ValueError("public reveal does not include the emulator's seat")
        if not 1 <= action <= ACTION_COUNT:
            raise ValueError("the emulator's own action must lie in 1..60")
        self.previous[role] = action
        self.first_in_game[role] = False

    def reset_game(self) -> None:
        self.first_in_game = {role: True for role in ROLES}

    def reset_session(self) -> None:
        self.name = None
        self.previous = {role: None for role in ROLES}
        self.first_in_game = {role: True for role in ROLES}


__all__ = ["SelfRepeatHumanOpponent"]
