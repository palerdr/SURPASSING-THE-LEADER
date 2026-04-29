from dataclasses import dataclass
from .exact_transition import ExactGameSnapshot, ExactJointAction, ExactSearchConfig
from .candidates import CandidateActions, generate_candidates
from .utility import terminal_value
from .minimax import solve_minimax
from src.Game import Game
import numpy as np

@dataclass
class MCTSNode:
    drop_seconds: tuple[int, ...]
    check_seconds: tuple[int, ...]
    game_snapshot: ExactGameSnapshot
    Q : np.ndarray #matrix of |drop seconds|, |check seconds|, Q(i,j) gives the current estime of the EQ value of the subgame that follows from the joint action at (drop[i], check[j])
    N_cell : np.ndarray
    N_node : int #visits
    is_expanded: bool
    prior : np.ndarray
    terminal_value : float | None
    children : dict[tuple[int, int, bool | None], "MCTSNode"]
    hal_is_dropper : bool

@dataclass
class MCTSConfig:
    iterations : int
    exploration_c : float
    evaluator: None
    use_tablebase : bool

@dataclass
class MCTSResult:
    root_strategy_dropper: np.ndarray
    root_strategy_checker : np.ndarray
    root_value_for_hal : float
    root_visits : int
    principal_line: list[ExactJointAction]
    cells_used : int


#helpers
    
def make_node(game: Game, config: ExactSearchConfig | None = None) -> MCTSNode:
    """Build an MCTSNode at the current game state.
    Snapshots the engine state, generates candidates, allocates zero
    Q/N matrices. Sets terminal_value if the game is already over.
    """
    config = config or ExactSearchConfig()
    tval = terminal_value(game= game, perspective_name=config.perspective_name)

    if game.game_over:
        return MCTSNode(
            drop_seconds= (),
            check_seconds= (),
            game_snapshot= ExactGameSnapshot(game=game),
            Q = np.zeros((0,0), dtype = np.float64),
            N_cell = np.zeros((0,0), dtype = np.int64),
            N_node= 0,
            is_expanded= False,
            prior= np.full((0,0), 0, dtype = np.float64),
            terminal_value= tval,
            children = {},
            hal_is_dropper = False
        )
    dropper,_ = game.get_roles_for_half(game.current_half)
    hdrop = (dropper.name.lower() == config.perspective_name.lower())
    cands = generate_candidates(game, config)
    D = len(cands.drop_seconds)
    C = len(cands.check_seconds)

    #prior is uniform on creation

    if D > 0 and C > 0:
        prior = np.full((D, C), 1.0/(D*C), dtype=np.float64)
    else:
        prior = np.zeros((0,0), dtype=np.float64)


    return MCTSNode(
        drop_seconds= cands.drop_seconds,
        check_seconds= cands.check_seconds,
        game_snapshot= ExactGameSnapshot(game=game),
        Q = np.zeros((D,C), dtype= np.float64),
        N_cell = np.zeros((D,C), dtype= np.int64),
        N_node = 0,
        is_expanded = False,
        prior= prior,
        terminal_value= tval,
        children={},
        hal_is_dropper=hdrop,
    )
    

def _exploration_augmented_matrix(node: MCTSNode, exploration_c: float) -> np.ndarray:
    """Return Q + c * P * sqrt(N_node) / (1 + N_cell). 
    Has shape (D,C)
    """
    Q = node.Q
    U = exploration_c * node.prior * np.sqrt(node.N_node) / (1 + node.N_cell)
    return Q + U



def _select_joint_action(
        node:MCTSNode,
        exploration_c:float,
        rng:np.random.Generator,
) -> tuple[int, int]:
    """Return cell indexes (d_idx, c_idx) sampled from the matrix-game
    equilibrium of the exploration-augmented Q at this node
    """
    
    hal_is_dopper = node.hal_is_dropper
    D,C = len(node.drop_seconds), len(node.check_seconds)

    Q_explore = _exploration_augmented_matrix(node, exploration_c=exploration_c)

    if hal_is_dopper:
        dropper_strategy, _ = solve_minimax(Q_explore)
        checker_strategy, _ = solve_minimax((-Q_explore).T)
    else:
        dropper_strategy, _ = solve_minimax(-Q_explore)
        checker_strategy, _ = solve_minimax((Q_explore).T)
    
    d_idx = rng.choice(D, p=dropper_strategy)
    c_idx = rng.choice(C, p=checker_strategy)
    
    return int(d_idx), int(c_idx)

def _expand_node(node, game, evaluator) -> None:
    ...
def _backup(path, value) -> None:
    ...
def _principal_line(root) -> list[ExactJointAction]:
    ...