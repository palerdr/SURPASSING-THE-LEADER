"""Serve the frozen translated Hal with a canonical leap fallback."""
from __future__ import annotations

import json
from pathlib import Path

from arena.dth_adapter import project_to_dth_state
from arena.policies.perfect_hal import PerfectHalPolicyProvider, _validate_stage
from arena.policies.translated_hal import TranslatedHalConfig, TranslatedHalOpponentModel
from stl.engine.actions import legal_seconds, validate_action

SELECTION = Path(__file__).with_name("config") / "translated_hal_v1_selection.json"


def selected_config():
    return TranslatedHalConfig(**json.loads(SELECTION.read_text())["config"])


class TranslatedHalPolicyProvider:
    """Exploit ordinary turns; use DTH equilibrium during the leap window."""

    def __init__(self, artifact_dir, *, agent=None):
        config = selected_config()
        self.inner = PerfectHalPolicyProvider(artifact_dir, config, agent=agent,
            opponent_model=TranslatedHalOpponentModel(config))
        self._leap_pending = None
        self.leap_fallbacks = 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def policy(self, decision):
        if self._leap_pending is not None:
            raise RuntimeError("Hal was asked to act twice before a leap reveal")
        if decision.turn_duration == 60:
            return self.inner.policy(decision)
        if (decision.turn_duration != 61 or decision.role not in ("dropper", "checker")
            or tuple(decision.legal_seconds) != legal_seconds(decision.actor_name, decision.role, 61)):
            raise ValueError("invalid canonical leap decision")
        if self.inner._pending is not None:
            raise RuntimeError("Hal has an unrevealed ordinary action")
        stage = self.inner.agent.stage_game(project_to_dth_state(decision))
        _validate_stage(stage)
        policy = stage.drop_policy if decision.role == "dropper" else stage.check_policy
        self._leap_pending = decision
        self.leap_fallbacks += 1
        return {i + 1: float(p) for i, p in enumerate(policy) if p > 0}

    def observe(self, record):
        pending = self._leap_pending
        if pending is None:
            return self.inner.observe(record)
        name = record.dropper_name if pending.role == "dropper" else record.checker_name
        if name != pending.actor_name or record.pre_decision_state.turn_duration != 61:
            raise ValueError("leap reveal disagrees with the pending decision")
        if record.game_index < 0 or record.half_round_index < 0:
            raise ValueError("invalid leap reveal indices")
        validate_action(record.drop_time, actor=record.dropper_name, role="dropper", turn_duration=61)
        validate_action(record.check_time, actor=record.checker_name, role="checker", turn_duration=61)
        self.inner.opponent_model.break_sequence()
        self._leap_pending = None

    def reset_game(self):
        if self._leap_pending is not None:
            raise RuntimeError("game reset with an unrevealed leap action")
        self.inner.reset_game()
        self.inner.decisions.clear()
        self.inner._seen_reveals.clear()
        self.leap_fallbacks = 0

    def reset_session(self):
        if self._leap_pending is not None:
            raise RuntimeError("session reset with an unrevealed leap action")
        self.inner.reset_session()
        self.leap_fallbacks = 0

    def end_game(self, outcome):
        if self._leap_pending is not None:
            raise RuntimeError("game ended with an unrevealed leap action")
        self.inner.end_game(outcome)

    def experiment_diagnostics(self):
        return {**self.inner.experiment_diagnostics(), "pure_dth_only": False,
            "leap_fallbacks": self.leap_fallbacks,
            "leap_policy": "DTH equilibrium projection; no action-61 optimization claim"}
