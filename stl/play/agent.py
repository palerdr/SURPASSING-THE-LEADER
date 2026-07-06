"""Solver-backed playable agent: trained value net + matrix-game MCTS.

This is the strategic core demanded by the plan's Track A: public game
state in, sampled equilibrium action out. Design rules it enforces:

- **Loud loading.** A missing or incompatible checkpoint raises; there is
  no silent fallback to a handcrafted eval (bug B3's failure mode).
- **Mixed play.** ``choose_action`` SAMPLES from the root equilibrium
  mixture. Argmaxing a mixed equilibrium in this game is maximally
  exploitable (a fixed check second c invites drop at c+1 every
  half-round; a fixed drop d invites check at exactly d for ST=1).
- **Markov policy.** The search RNG is derived deterministically from the
  public state, so the mixture pi(s) is a pure function of s. The
  best-response exploitability probe (Track C) requires this; action
  *sampling* uses a separate stream so repeated visits stay stochastic.
- **No hidden information.** Inputs are the public ``Game`` state, the
  legality rules, and engine-derived chance — nothing else. Any future
  personality layer runs strictly after ``choose_action`` returns.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from stl.solver.search import (
    LeafEvaluator,
    TablebaseEvaluator,
    TerminalOnlyEvaluator,
    ValueNetEvaluator,
)
from stl.solver.exact import CFRPlusConfig, ExactSearchConfig, exact_public_state
from stl.solver.search import MCTSConfig, MCTSResult, mcts_search
from stl.play.opponents.base import Opponent
from stl.engine.game import Game

DEFAULT_CHECKPOINT = os.path.join(
    str(Path(__file__).resolve().parents[2]),
    "checkpoints",
    "gen_tier_a_aux_50k_w001_ft_lr5e-6_tw001_pw0_iw100_e5",
    "best.pt",
)


def _state_seed(base_seed: int, game: Game) -> int:
    """Stable per-state search seed: pi(s) becomes a pure function of s."""
    digest = hashlib.sha256(
        f"{base_seed}|{exact_public_state(game)!r}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "little")


def _load_evaluator(checkpoint_path: str | Path) -> LeafEvaluator:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(
            f"SolverAgent requires a trained checkpoint; none found at {path}. "
            f"Pull/produce one (see README §4) — there is no silent fallback."
        )
    # Import here so constructing a TerminalOnly agent never needs torch.
    from stl.learning.train import load_checkpoint, make_predict_fn

    net = load_checkpoint(str(path))
    # Pinned tablebase values short-circuit the net where exact values are
    # known — the syzygy-probe-inside-search pattern in miniature.
    return TablebaseEvaluator(fallback=ValueNetEvaluator(make_predict_fn(net)))


class SolverAgent(Opponent):
    """MCTS + value-net agent for either seat (Hal by default)."""

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
        *,
        player_name: str = "Hal",
        iterations: int = 200,
        exploration_c: float = 1.0,
        seed: int = 0,
        resolve_at_critical: bool = True,
        resolve_horizon: int = 1,
        resolve_cfr_iters: int = 2000,
        evaluator: LeafEvaluator | None = None,
        use_tier_a: bool = False,
        tier_a_width: float = 0.0,
    ) -> None:
        self.player_name = player_name
        self.iterations = int(iterations)
        self.exploration_c = float(exploration_c)
        self.resolve_at_critical = bool(resolve_at_critical)
        self.resolve_horizon = int(resolve_horizon)
        self.resolve_cfr_iters = int(resolve_cfr_iters)
        self._base_seed = int(seed)
        self._action_rng = np.random.default_rng(seed)
        self._exact_config = ExactSearchConfig()
        self._policy_cache: dict = {}
        self._search_cache: dict = {}
        base_evaluator = evaluator if evaluator is not None else _load_evaluator(checkpoint_path)
        if use_tier_a:
            from stl.solver.tablebase import TierAEvaluator

            self.evaluator = TierAEvaluator(base_evaluator, max_width=tier_a_width)
        else:
            self.evaluator = base_evaluator

    # ── Strategic core ────────────────────────────────────────────────

    def search(self, game: Game) -> MCTSResult:
        """One fresh, state-seeded search from the current public state.

        Cached per public state — the state-derived seed makes the result
        a pure function of the state, so caching changes nothing but cost.
        """
        key = exact_public_state(game)
        if key in self._search_cache:
            return self._search_cache[key]

        rng = np.random.default_rng(_state_seed(self._base_seed, game))
        config = MCTSConfig(
            iterations=self.iterations,
            exploration_c=self.exploration_c,
            evaluator=None,
            use_tablebase=False,
            include_playable_grid=True,
        )
        result = mcts_search(
            game,
            config,
            self.evaluator,
            rng,
            self._exact_config,
            subgame_resolve_at_critical=self.resolve_at_critical,
            subgame_resolve_horizon=self.resolve_horizon,
            subgame_resolve_cfr_plus_config=CFRPlusConfig(iterations=self.resolve_cfr_iters),
        )
        self._search_cache[key] = result
        return result

    def policy(self, game: Game, role: str) -> tuple[tuple[int, ...], np.ndarray]:
        """The agent's mixed strategy for ``role`` at this state.

        Returns (seconds, probabilities). Deterministic per state (cached,
        state-seeded search) — the object the best-response probe queries.
        """
        if role not in ("dropper", "checker"):
            raise ValueError(f"role must be 'dropper' or 'checker', got {role!r}")

        key = (exact_public_state(game), role)
        if key in self._policy_cache:
            return self._policy_cache[key]

        result = self.search(game)
        # Play the AVERAGE of the per-iteration root selection strategies —
        # the provably convergent SM-MCTS object. The final LP over mean-Q
        # picks an arbitrary vertex when the matrix is near-flat (small
        # budgets), which collapses to a pure, pattern-readable strategy.
        if role == "dropper":
            seconds = result.root_drop_seconds
            probs = np.asarray(result.root_strategy_dropper_avg, dtype=np.float64)
        else:
            seconds = result.root_check_seconds
            probs = np.asarray(result.root_strategy_checker_avg, dtype=np.float64)

        if len(seconds) == 0 or probs.size == 0:
            raise RuntimeError(f"search produced an empty {role} strategy at {key[0]!r}")
        probs = np.maximum(probs, 0.0)
        probs = probs / probs.sum()

        self._policy_cache[key] = (seconds, probs)
        return seconds, probs

    def choose_action(self, game: Game, role: str, turn_duration: int) -> int:
        dropper, checker = game.get_roles_for_half(game.current_half)
        actor = dropper if role == "dropper" else checker
        if actor.name.lower() != self.player_name.lower():
            raise ValueError(
                f"SolverAgent({self.player_name}) asked to act as {role}, "
                f"but the engine says {actor.name} holds that role"
            )

        seconds, probs = self.policy(game, role)
        idx = int(self._action_rng.choice(len(seconds), p=probs))
        return int(seconds[idx])

    def reset(self) -> None:
        self._policy_cache.clear()
        self._search_cache.clear()


class HalSolverAgent(SolverAgent):
    """Factory-friendly zero-arg construction: Hal seat, default checkpoint."""

    def __init__(self) -> None:
        super().__init__(DEFAULT_CHECKPOINT, player_name="Hal")


class BakuSolverAgent(SolverAgent):
    """The same machinery on the Baku seat (head-to-head / exploiter duty)."""

    def __init__(self) -> None:
        super().__init__(DEFAULT_CHECKPOINT, player_name="Baku")


def make_choose_action(agent: SolverAgent):
    """Adapt an agent to training.tournament's ChooseAction callable."""

    def choose(game: Game, role: str, turn_duration: int) -> int:
        return agent.choose_action(game, role, turn_duration)

    return choose
