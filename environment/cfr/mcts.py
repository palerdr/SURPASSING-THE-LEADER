"""MCTS over exact-second matrix-game-at-node search for STL.

Each node stores: candidate seconds, a Q-matrix and visit counts, an
ExactGameSnapshot for restoring the engine, and a prior. Selection
solves the matrix game at the node on an exploration-augmented Q;
expansion samples chance branches via the engine; backup propagates
leaf values without sign flips (Hal-perspective everywhere).
"""

from dataclasses import dataclass

import numpy as np

from src.Game import Game

from .candidates import generate_candidates
from .evaluator import LeafEvaluator
from .exact_transition import ExactGameSnapshot, ExactJointAction, ExactSearchConfig
from .minimax import solve_minimax
from .utility import terminal_value


@dataclass
class MCTSNode:
    drop_seconds: tuple[int, ...]
    check_seconds: tuple[int, ...]
    game_snapshot: ExactGameSnapshot
    Q: np.ndarray
    N_cell: np.ndarray
    N_node: int
    is_expanded: bool
    prior: np.ndarray
    terminal_value: float | None
    children: dict[tuple[int, int, bool | None], "MCTSNode"]
    hal_is_dropper: bool


@dataclass
class MCTSConfig:
    iterations: int
    exploration_c: float
    evaluator: None
    use_tablebase: bool


@dataclass
class MCTSResult:
    root_strategy_dropper: np.ndarray
    root_strategy_checker: np.ndarray
    root_value_for_hal: float
    root_visits: int
    principal_line: list[ExactJointAction]
    cells_used: int


def make_node(game: Game, config: ExactSearchConfig | None = None) -> MCTSNode:
    """Build an MCTSNode at the current game state.

    Snapshots the engine state, generates candidates, allocates zero
    Q/N matrices. Sets terminal_value if the game is already over.
    """
    config = config or ExactSearchConfig()
    tval = terminal_value(game=game, perspective_name=config.perspective_name)

    if game.game_over:
        return MCTSNode(
            drop_seconds=(),
            check_seconds=(),
            game_snapshot=ExactGameSnapshot(game=game),
            Q=np.zeros((0, 0), dtype=np.float64),
            N_cell=np.zeros((0, 0), dtype=np.int64),
            N_node=0,
            is_expanded=False,
            prior=np.zeros((0, 0), dtype=np.float64),
            terminal_value=tval,
            children={},
            hal_is_dropper=False,
        )

    dropper, _ = game.get_roles_for_half(game.current_half)
    hdrop = dropper.name.lower() == config.perspective_name.lower()
    cands = generate_candidates(game, config)
    D = len(cands.drop_seconds)
    C = len(cands.check_seconds)

    if D > 0 and C > 0:
        prior = np.full((D, C), 1.0 / (D * C), dtype=np.float64)
    else:
        prior = np.zeros((0, 0), dtype=np.float64)

    return MCTSNode(
        drop_seconds=cands.drop_seconds,
        check_seconds=cands.check_seconds,
        game_snapshot=ExactGameSnapshot(game=game),
        Q=np.zeros((D, C), dtype=np.float64),
        N_cell=np.zeros((D, C), dtype=np.int64),
        N_node=0,
        is_expanded=False,
        prior=prior,
        terminal_value=tval,
        children={},
        hal_is_dropper=hdrop,
    )


def _exploration_augmented_matrix(node: MCTSNode, exploration_c: float) -> np.ndarray:
    """Return Q + c * P * sqrt(N_node) / (1 + N_cell). Shape (D, C)."""
    U = exploration_c * node.prior * np.sqrt(node.N_node) / (1 + node.N_cell)
    return node.Q + U


def _select_joint_action(
    node: MCTSNode,
    exploration_c: float,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Sample (d_idx, c_idx) from the matrix-game equilibrium of the
    exploration-augmented Q at this node."""
    D = len(node.drop_seconds)
    C = len(node.check_seconds)
    Q_explore = _exploration_augmented_matrix(node, exploration_c)

    if node.hal_is_dropper:
        dropper_strategy, _ = solve_minimax(Q_explore)
        checker_strategy, _ = solve_minimax((-Q_explore).T)
    else:
        dropper_strategy, _ = solve_minimax(-Q_explore)
        checker_strategy, _ = solve_minimax(Q_explore.T)

    d_idx = rng.choice(D, p=dropper_strategy)
    c_idx = rng.choice(C, p=checker_strategy)
    return int(d_idx), int(c_idx)


def _step_into_child(
    node: MCTSNode,
    game: Game,
    d_idx: int,
    c_idx: int,
    rng: np.random.Generator,
    config: ExactSearchConfig,
) -> tuple["MCTSNode", bool | None]:
    """Apply the joint action at (d_idx, c_idx) to the engine, sample the
    chance outcome if a death is possible, and return the child node along
    with the survival flag (True/False/None). Caches the child in
    node.children so subsequent iterations reuse it.
    """
    d_time = node.drop_seconds[d_idx]
    c_time = node.check_seconds[c_idx]

    snap = ExactGameSnapshot(game)
    probe = game.resolve_half_round(d_time, c_time, survived_outcome=None)
    death_possible = probe.survived is not None
    survival_probability = probe.survival_probability
    snap.restore(game)

    survived_outcome: bool | None = None
    if death_possible:
        survived_outcome = bool(rng.random() < survival_probability)

    key = (d_time, c_time, survived_outcome)
    game.resolve_half_round(d_time, c_time, survived_outcome)
    if key not in node.children:
        node.children[key] = make_node(game, config)
    return node.children[key], survived_outcome


def _backup(path: list[tuple[MCTSNode, int, int]], value: float) -> None:
    """Apply value to every (node, d_idx, c_idx) in path.

    Increments N_node, N_cell[d_idx, c_idx], and updates Q[d_idx, c_idx]
    with a running-mean step. Mutates the nodes in place.
    """
    for (node, d_idx, c_idx) in path:
        node.N_node += 1
        node.N_cell[d_idx, c_idx] += 1
        node.Q[d_idx, c_idx] += (value - node.Q[d_idx, c_idx]) / node.N_cell[d_idx, c_idx]


def mcts_search(
    game: Game,
    config: MCTSConfig,
    evaluator: LeafEvaluator,
    rng: np.random.Generator,
    exact_config: ExactSearchConfig | None = None,
) -> MCTSResult:
    """Run MCTS for config.iterations iterations from the current game state.

    Returns the root's equilibrium strategies, the Hal-perspective value,
    and visit-count diagnostics.
    """
    exact_config = exact_config or ExactSearchConfig()
    root = make_node(game, exact_config)
    c = config.exploration_c

    for _ in range(config.iterations):
        root.game_snapshot.restore(game=game)
        node = root
        path: list[tuple[MCTSNode, int, int]] = []
        while True:
            if node.terminal_value is not None:
                leaf_value = node.terminal_value
                break
            if not node.is_expanded:
                leaf_value = evaluator(game)
                node.is_expanded = True
                break
            d_idx, c_idx = _select_joint_action(node, c, rng)
            path.append((node, d_idx, c_idx))
            node, _ = _step_into_child(node, game, d_idx, c_idx, rng, exact_config)
        _backup(path, leaf_value)

    Q = root.Q
    if root.hal_is_dropper:
        dropper_strat, _ = solve_minimax(Q)
        checker_strat, _ = solve_minimax((-Q).T)
    else:
        dropper_strat, _ = solve_minimax(-Q)
        checker_strat, _ = solve_minimax(Q.T)

    value_for_hal = float(dropper_strat @ Q @ checker_strat)

    return MCTSResult(
        root_strategy_dropper=dropper_strat,
        root_strategy_checker=checker_strat,
        root_value_for_hal=value_for_hal,
        root_visits=root.N_node,
        principal_line=[],
        cells_used=int(root.N_cell.sum()),
    )


def _principal_line(root: MCTSNode) -> list[ExactJointAction]:
    """Stub: most-visited path through the tree. Implemented in Slice 4c."""
    ...
