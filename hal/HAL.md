# HAL — Search-Based Canonical Opponent

This document is the authoritative specification for building Hal, a game-solving AI opponent for Drop The Handkerchief. It replaces all prior designs (scripted planner, pure RL, full-game CFR, heuristic trees).

## Design Philosophy

Hal should play like Stockfish plays chess: **search forward, evaluate positions, choose the best move.** The manga's documented game is not a script to follow — it is a benchmark to verify against. If the solver independently discovers the same moves the manga documents, the evaluation function and search are correct. If it deviates, something needs fixing.

### What This Is

- A **multi-turn game solver** with simultaneous-move backward induction
- A **reduced action space** (7-8 buckets per player) enabling deep search
- A **handcrafted evaluation function** (Phase 1) upgradeable to an RL-trained value network (Phase 2)
- An **exploitation layer** (belief tracker) that punishes predictable human play
- A **memory state machine** (Perfect Mode) that faithfully reproduces the manga's leap-second amnesia mechanic

### What This Is Not

- Not a scripted bot that follows the manga's action sequence
- Not a pure RL policy (which failed after 12+ training sprints)
- Not a full-game CFR solver (which lost precision through coarse abstraction)
- Not a heuristic decision tree (which grows unwieldy and is trivially exploitable)

### Why Search Works For This Game

| Property | Value | Implication |
|---|---|---|
| Actions per player per turn | 60-61 | Reducible to 7-8 buckets |
| Game length | 6-8 half-rounds | 3-4 half-round lookahead covers most of the remaining game |
| Branching (bucketed) | 7×7 = 49 per half-round | 49⁴ ≈ 5.7M nodes at depth 4 — trivially fast |
| Information structure | Near-perfect (only leap-second knowledge hidden) | No need for information-set abstraction |
| Randomness | Single Bernoulli per death | Expected value computation over 2 branches per death |
| State after half-round | Fully determined by actions + survival roll | Clean forward simulation |

---

## Architecture

```
Game State + History
       │
       ├──► Belief Tracker ──► exploitation_mode: bool
       │     (Baku's patterns from action history)
       │
       ├──► Memory Model ──► effective_turn_duration
       │     (Perfect Mode amnesia FSM)
       │
       └──► Search Engine
              │
              ├── Action Candidate Generator
              │     (state-dependent bucket selection)
              │
              ├── Forward Simulation
              │     (clone game state, play out bucket-pair outcomes)
              │
              ├── Multi-Turn Backward Induction
              │     (at each node: LP Nash or best-response)
              │     (at leaves: evaluate position)
              │
              └── Bucket Resolution
                    (sample actual second uniformly from chosen bucket)
```

---

## Component 1: Action Buckets

### Bucket Definitions

Each bucket is a contiguous range of seconds. The solver picks a bucket; the actual second played is sampled uniformly from that range. This gives the human player a sense of variance/luck while keeping the strategic computation tractable.

```python
@dataclass(frozen=True)
class Bucket:
    lo: int       # inclusive lower bound (1-indexed second)
    hi: int       # inclusive upper bound
    label: str    # human-readable name

STANDARD_BUCKETS: tuple[Bucket, ...] = (
    Bucket(1,   1,  "instant"),     # 0: drop/check at second 1
    Bucket(2,  10,  "early"),       # 1: seconds 2-10
    Bucket(11, 25,  "mid_early"),   # 2: seconds 11-25
    Bucket(26, 40,  "mid"),         # 3: seconds 26-40
    Bucket(41, 52,  "mid_late"),    # 4: seconds 41-52
    Bucket(53, 58,  "late"),        # 5: seconds 53-58
    Bucket(59, 60,  "safe"),        # 6: seconds 59-60
)

LEAP_BUCKET = Bucket(61, 61, "leap")  # 7: only during leap-second turn
```

7 standard buckets + 1 leap bucket = 8 maximum. During non-leap turns: 7×7 = 49 outcomes. During leap turns: 8×7 or 8×8 = 56-64 outcomes.

### Expected Payoff Per Bucket Pair

For each bucket pair (dropper bucket D, checker bucket C), compute the expected payoff by averaging over all (d, c) second pairs:

```python
def bucket_pair_payoff(D: Bucket, C: Bucket, checker_cylinder: float) -> float:
    """Expected checker payoff for dropper choosing from D, checker from C."""
    total = 0.0
    count = 0
    for d in range(D.lo, D.hi + 1):
        for c in range(C.lo, C.hi + 1):
            if c >= d:
                st = max(1, c - d)
                if checker_cylinder + st >= CYLINDER_MAX:
                    total += -CYLINDER_MAX
                else:
                    total += -st
            else:
                injection = min(checker_cylinder + FAILED_CHECK_PENALTY, CYLINDER_MAX)
                total += -injection
            count += 1
    return total / count
```

This builds a small `n_d × n_c` matrix (7×7 or 8×8) instead of the full 60×60. The sign convention matches `cfr/half_round.py:compute_payoff_matrix()`: negative = bad for checker.

### State-Dependent Bucket Selection

Not all buckets are relevant in every state. The action candidate generator filters:

- **Dropper during leap turn**: Include `LEAP_BUCKET`
- **Checker always**: Exclude `LEAP_BUCKET` (unless memory is RECOVERED and aware)
- **Checker in AMNESIA mode**: Restrict to standard buckets only (doesn't know about 61)
- **When cylinder is near overflow**: The exact overflow threshold second matters. If `checker_cylinder + 59 >= 300`, the "safe" bucket could cause overflow — the solver needs to see this in the payoff matrix (it already does, since `bucket_pair_payoff` handles overflow).

### Bucket Resolution

After the solver selects a bucket, sample the actual second:

```python
def resolve_bucket(bucket: Bucket, rng: Random) -> int:
    return rng.randint(bucket.lo, bucket.hi)
```

This is the source of "luck" — two games with the same search result will play out differently at the second level.

---

## Component 2: Multi-Turn Search

### Overview

At each turn, Hal builds a game tree from the current state looking 3-4 half-rounds ahead. At each internal node, both players choose buckets simultaneously — a matrix game. At leaf nodes, a position evaluation function scores the state. Backward induction with LP Nash (or best-response) at each node propagates values to the root.

### Game State Cloning

The engine uses mutable state (`Game`, `Player`, `Referee` are all mutable dataclasses). For search, we need to branch the game tree without corrupting the real game.

```python
import copy

def clone_game(game: Game) -> Game:
    """Deep copy of game state for search exploration."""
    return copy.deepcopy(game)
```

For performance: if `deepcopy` is too slow (profiling will tell), implement a lightweight `snapshot → restore` system that copies only the ~10 mutable fields (`cylinder`, `ttd`, `deaths`, `alive`, `game_clock`, `round_num`, `current_half`, `game_over`, `cprs_performed`, `history` length).

### Forward Simulation

Given a game state and a (dropper_bucket, checker_bucket) pair, simulate the outcome:

```python
def simulate_bucket_pair(game: Game, d_bucket: Bucket, c_bucket: Bucket) -> list[tuple[Game, float]]:
    """Returns list of (resulting_game_state, probability) pairs.
    
    For non-death outcomes: single result with probability 1.0.
    For death outcomes: two results (survived, died) weighted by survival probability.
    """
    # Use bucket midpoints for the simulation (representative seconds)
    drop_time = (d_bucket.lo + d_bucket.hi) // 2
    check_time = (c_bucket.lo + c_bucket.hi) // 2
    
    # Determine if this causes death (without actually rolling)
    success = check_time >= drop_time
    
    if success:
        st = max(1, check_time - drop_time)
        causes_death = (checker.cylinder + st >= CYLINDER_MAX)
    else:
        causes_death = True  # Failed check always causes death
    
    if not causes_death:
        # Deterministic outcome: clone, play, return
        cloned = clone_game(game)
        cloned.play_half_round(drop_time, check_time)  # with fixed RNG that always survives
        return [(cloned, 1.0)]
    else:
        # Stochastic: compute survival probability, branch
        surv_prob = compute_survival_probability(...)
        
        # Branch 1: survived
        survived_game = clone_game(game)
        # ... apply death + revival
        
        # Branch 2: died (terminal)
        died_game = clone_game(game)
        # ... apply death + permanent death
        
        return [(survived_game, surv_prob), (died_game, 1.0 - surv_prob)]
```

**Implementation note:** Rather than calling `play_half_round` (which rolls the RNG), implement a deterministic `simulate_half_round` that takes the survival outcome as a parameter. This keeps the search tree clean. The actual `play_half_round` with real RNG is only used when Hal plays his chosen action in the real game.

### Backward Induction

The core algorithm. Process the search tree from leaves to root:

```python
def search(game: Game, depth: int, belief: BeliefState, memory: MemoryMode) -> tuple[np.ndarray, float]:
    """Returns (mixed_strategy_over_buckets, game_value_for_hal).
    
    depth: remaining half-rounds to search.
    """
    if depth == 0 or game.game_over:
        return None, evaluate(game)  # leaf node
    
    # Determine who is Hal and who is Baku this half-round
    dropper, checker = game.get_roles_for_half(game.current_half)
    hal_is_dropper = dropper.name.lower() == "hal"
    
    # Get available buckets
    turn_duration = game.get_turn_duration()
    effective_td = 60 if (memory == AMNESIA and turn_duration == 61) else turn_duration
    hal_buckets = get_buckets(effective_td, is_dropper=hal_is_dropper)
    baku_buckets = get_buckets(turn_duration, is_dropper=not hal_is_dropper)
    
    n_hal = len(hal_buckets)
    n_baku = len(baku_buckets)
    
    # Build payoff matrix (from Hal's perspective)
    payoff = np.zeros((n_hal, n_baku))
    for i, h_bucket in enumerate(hal_buckets):
        for j, b_bucket in enumerate(baku_buckets):
            # Determine dropper/checker buckets
            d_bucket = h_bucket if hal_is_dropper else b_bucket
            c_bucket = b_bucket if hal_is_dropper else h_bucket
            
            # Simulate outcomes (possibly branching on death)
            outcomes = simulate_bucket_pair(game, d_bucket, c_bucket)
            
            # Expected value over outcomes, recursing for continuation
            ev = 0.0
            for result_game, prob in outcomes:
                if result_game.game_over:
                    # Terminal: +1 if Hal won, -1 if Hal lost
                    ev += prob * (1.0 if result_game.winner.name.lower() == "hal" else -1.0)
                else:
                    # Recurse
                    _, cont_val = search(result_game, depth - 1, belief, 
                                         update_memory(memory, ...))
                    ev += prob * cont_val
            
            payoff[i][j] = ev
    
    # Solve the matrix game
    if belief.exploitation_mode:
        # Best-response against predicted Baku strategy
        baku_strategy = predict_baku_strategy(belief, baku_buckets)
        ev_per_hal_action = payoff @ baku_strategy
        best_idx = np.argmax(ev_per_hal_action)
        strategy = np.zeros(n_hal)
        strategy[best_idx] = 1.0
        value = ev_per_hal_action[best_idx]
    else:
        # Nash equilibrium (safe play)
        strategy, value = solve_minimax(payoff)
    
    return strategy, value
```

### Search Depth Selection

| Rounds remaining until game end | Recommended search depth |
|---|---|
| ≥ 6 half-rounds | 3 half-rounds (49³ ≈ 117K nodes) |
| 4-5 half-rounds | 4 half-rounds (49⁴ ≈ 5.7M nodes) |
| ≤ 3 half-rounds | Full remaining (exact solve) |

Near the leap window, always search to at least the leap turn so the solver can see the leap-second payoff.

### LP Nash Solver

Same formulation as the previous plan. For an `m × n` matrix game (Hal has m actions, Baku has n):

```
maximize v
subject to:
    Σ_i x_i * A[i][j] ≥ v    ∀ j ∈ [0, n)
    Σ_i x_i = 1
    x_i ≥ 0
```

Python: `scipy.optimize.linprog`. OCaml: hand-rolled simplex. For 8×8 matrices this is microseconds.

---

## Component 3: Position Evaluation

### Phase 1: Handcrafted Evaluation

A linear combination of features, each normalized to roughly [-1, 1]:

```python
def evaluate(game: Game) -> float:
    """Evaluate position from Hal's perspective. Returns [-1, +1]."""
    if game.game_over:
        return 1.0 if game.winner.name.lower() == "hal" else -1.0
    
    hal, baku = get_named_players(game)
    ref = game.referee
    
    features = {}
    
    # 1. Cylinder pressure advantage
    #    Higher Baku cylinder relative to Hal = good for Hal
    features["cylinder_diff"] = (baku.cylinder - hal.cylinder) / CYLINDER_MAX
    
    # 2. Safe strategy budget advantage
    #    More safe checks remaining for Hal = good
    hal_budget = max(0, int((CYLINDER_MAX - 1 - hal.cylinder) // 60))
    baku_budget = max(0, int((CYLINDER_MAX - 1 - baku.cylinder) // 60))
    features["safe_budget_diff"] = (hal_budget - baku_budget) / 5.0
    
    # 3. Survival probability advantage
    #    Compare what happens if each player fails their next check
    hal_death_dur = min(hal.cylinder + FAILED_CHECK_PENALTY, CYLINDER_MAX)
    baku_death_dur = min(baku.cylinder + FAILED_CHECK_PENALTY, CYLINDER_MAX)
    hal_surv = survival_probability(hal_death_dur, hal.ttd, ref.cprs_performed, hal.physicality)
    baku_surv = survival_probability(baku_death_dur, baku.ttd, ref.cprs_performed, baku.physicality)
    features["survival_diff"] = hal_surv - baku_surv
    
    # 4. Death count / cardiac degradation advantage
    features["death_diff"] = (baku.deaths - hal.deaths) / 4.0
    
    # 5. TTD advantage (weighted by physicality asymmetry)
    features["ttd_diff"] = (baku.ttd - hal.ttd) / CYLINDER_MAX
    
    # 6. LSR variation favorability
    #    V2 (active LSR) when Hal is dropper = very favorable (can drop at 61)
    var = lsr_variation_from_clock(game.game_clock)
    dropper, _ = game.get_roles_for_half(game.current_half)
    hal_is_dropper = dropper.name.lower() == "hal"
    if var == 2 and hal_is_dropper:
        features["lsr_favor"] = 0.8   # Active LSR, Hal dropping = strong
    elif var == 2 and not hal_is_dropper:
        features["lsr_favor"] = -0.5  # Active LSR, Baku dropping = dangerous
    else:
        features["lsr_favor"] = 0.0
    
    # 7. Proximity to leap window
    #    Being close to leap with favorable position amplifies advantages
    dist_to_leap = max(0, LS_WINDOW_START - game.game_clock)
    proximity = 1.0 - (dist_to_leap / (LS_WINDOW_START - OPENING_START_CLOCK))
    features["leap_proximity"] = proximity * 0.3  # mild bonus for being closer
    
    # 8. Referee fatigue advantage
    #    More total CPRs = both players' survival drops, but Baku's drops faster
    #    due to physicality asymmetry (0.88 vs 1.0)
    if ref.cprs_performed > 0:
        features["referee_fatigue"] = ref.cprs_performed * 0.02 * (1.0 - PHYSICALITY_BAKU)
    else:
        features["referee_fatigue"] = 0.0
    
    # Weighted sum
    weights = {
        "cylinder_diff":   3.0,
        "safe_budget_diff": 5.0,
        "survival_diff":   8.0,
        "death_diff":      2.0,
        "ttd_diff":        2.0,
        "lsr_favor":       4.0,
        "leap_proximity":  1.0,
        "referee_fatigue":  1.0,
    }
    
    raw = sum(weights[k] * features[k] for k in features)
    return max(-1.0, min(1.0, math.tanh(raw)))  # squash to [-1, 1]
```

**Weight tuning:** Start with the values above. Tune by playing CanonicalHal vs scripted BakuTeachers and checking whether Hal's chosen actions match the manga's canonical play. Adjust weights until the solver independently discovers the documented opening (death trade in R1, early checks in R3-R5, pressure drops in R6-R7, bridge setup in R8, leap drop at 61).

### Phase 2: RL-Trained Value Network (AlphaZero-style)

Replace the handcrafted `evaluate()` with a neural network `V(s) → [-1, 1]`:

**Input features** (24-dimensional, normalized to [0, 1]):
```
hal_cylinder / 300          baku_cylinder / 300
hal_ttd / 300               baku_ttd / 300
hal_deaths / 4              baku_deaths / 4
hal_safe_budget / 5         baku_safe_budget / 5
hal_survival_prob            baku_survival_prob
hal_is_dropper (0/1)        game_clock / 3600
round_num / 10              current_half (0/1)
lsr_variation (one-hot 4)   is_leap_turn (0/1)
rounds_until_leap / 10      cprs_performed / 6
active_lsr (0/1)            proximity_to_leap
```

**Network architecture:**
```
Input(24) → Linear(64) → ReLU → Linear(64) → ReLU → Linear(1) → Tanh
```

Small network deliberately — the search does the heavy lifting. The network just needs to be a better position evaluator than the handcrafted function.

**Training loop (self-play):**
1. Play N games using search + current value network as leaf evaluator
2. At each visited state, record `(features, actual_game_outcome)` where outcome ∈ {-1, +1}
3. Train network to minimize MSE: `L = (V(features) - outcome)²`
4. Repeat

**OCaml compatibility:** `ocaml-torch` provides PyTorch bindings. The network is 2 hidden layers of 64 — trivially portable. Inference is ~1μs per evaluation, dominated by the LP solver cost.

---

## Component 4: Belief Tracker (Exploitation Layer)

### Purpose

Against a theoretically optimal opponent, Hal plays Nash equilibrium (safe, unexploitable). Against a human who has predictable patterns, Hal switches to **best-response** (maximum punishment).

### State

```python
@dataclass(frozen=True)
class BeliefState:
    baku_check_history: tuple[int, ...]   # last N check seconds observed
    baku_drop_history: tuple[int, ...]    # last N drop seconds observed
    exploitation_mode: bool = False       # True = play best-response
    baku_predicted_bucket_probs: tuple[float, ...] | None = None  # predicted Baku strategy
```

### Update Logic

After each half-round, record Baku's action and update:

```python
def update_belief(belief: BeliefState, last_record: HalfRoundRecord) -> BeliefState:
    # Append to history (keep last 10)
    new_checks = belief.baku_check_history
    new_drops = belief.baku_drop_history
    if last_record.checker == "baku":
        new_checks = (new_checks + (last_record.check_time,))[-10:]
    if last_record.dropper == "baku":
        new_drops = (new_drops + (last_record.drop_time,))[-10:]
    
    # Compute entropy of Baku's bucket distribution
    # Low entropy = predictable = exploit
    bucket_counts = count_buckets(new_checks + new_drops)
    entropy = compute_entropy(bucket_counts)
    
    exploit = entropy < EXPLOITATION_THRESHOLD and len(new_checks) + len(new_drops) >= 4
    
    # If exploiting, predict Baku's next action as the empirical distribution
    predicted = None
    if exploit:
        predicted = empirical_bucket_distribution(new_checks + new_drops)
    
    return BeliefState(
        baku_check_history=new_checks,
        baku_drop_history=new_drops,
        exploitation_mode=exploit,
        baku_predicted_bucket_probs=predicted,
    )
```

### Effect on Search

When `exploitation_mode = True`, the search uses **best-response** instead of Nash at each node:

```python
# Instead of: strategy, value = solve_minimax(payoff)
# Do:
baku_strategy = belief.baku_predicted_bucket_probs
ev_per_hal_action = payoff @ np.array(baku_strategy)
best_idx = np.argmax(ev_per_hal_action)
```

This means Hal will find the action that maximally punishes Baku's observed tendency. A human who "always checks late" gets instant drops forcing maximum ST. A human who "always drops early" gets early checks catching the drop.

The switch from Nash to exploitation is **automatic** — no scripting needed. The search finds the punishment naturally.

---

## Component 5: Memory Model (Perfect Mode)

### Purpose

Faithfully reproduce the manga's Perfect Mode mechanic: Hal deliberately forgets the leap second exists before the critical turn, genuinely limiting his own action space. This creates the dramatic moment where Baku can exploit the 61st second.

### State Machine

```
NORMAL ────────────┐
                   │ When: game_clock >= 3300 (approaching leap window)
                   │       AND hal has deduced leap second (after first death)
                   ▼
PRE_AMNESIA ───────┐
                   │ When: is_leap_turn == True
                   ▼
AMNESIA ───────────┐
                   │ When: Hal dies and is revived (re-evidence)
                   ▼
RECOVERED (terminal)
```

### Effect on Search

When `memory == AMNESIA`:
- Hal's bucket list **excludes** `LEAP_BUCKET`
- Hal's search uses `effective_turn_duration = 60` even if actual is 61
- The search genuinely cannot consider dropping or checking at 61
- If Baku drops at 61 and Hal's best bucket is "safe" (59-60), Hal fails the check

When `memory == PRE_AMNESIA`:
- The search still sees the full action space this turn
- But knows that next turn (the leap turn) the action space will shrink
- This lets the solver prepare for the amnesia (e.g., accumulate a margin of safety)

### Why This Matters For Gameplay

Without Perfect Mode, Hal would always check at 61 during the leap turn (or drop at 61 as dropper) — making the leap second irrelevant. The amnesia creates a window where the human player's leap-second knowledge actually matters. It's the core asymmetry of the game.

---

## Canon Accuracy

### How The Manga's Game Emerges From Optimal Play

The documented canonical game is not coded in — it should **emerge from the solver** when the evaluation function is correct. Here's why each canonical action is optimal:

| Round | Canonical Hal Action | Why It's Optimal |
|---|---|---|
| R1 H1: Drop at ~35 | Gives Baku 25s ST, establishing cylinder pressure without risking overflow | Evaluation: cylinder_diff advantage |
| R1 H2: Check at ~5 (death probe) | Intentional fail: opens LSR by shifting clock 7 minutes, and Hal survives 1m24s death easily | Search sees: failing now + surviving → favorable LSR variation later |
| R3 H2: Check at ~34 | Early check: reads Baku's timing (whether drop was before or after 34) | Search sees: the information value of early checking during probe phase |
| R5 H2: Check at ~8 | Even earlier check: narrower timing window | Same — search discovers progressively earlier checks are optimal for information |
| R7 H1: Instant drop | During active LSR, instant drop forces maximum 59s ST on Baku | Evaluation: massive cylinder_diff and safe_budget_diff gain |
| R7 H2: Instant check | Intentional death trade to position for leap window | Search sees: dying now → correct LSR variation at leap turn |
| Leap: Drop at 61 | The kill shot — Baku checks at 60, fails, gets full injection | Only available because Hal recovered from amnesia |

**Verification test:** Run CanonicalHal against a canonical-Baku opponent (one that plays the manga's Baku actions). Check that Hal's chosen buckets match the expected canonical play at each step. This is the primary regression test.

### What Can't Be Captured (Supernatural Caveats)

The manga attributes several abilities to Hal that are narrative, not mechanical:
- **Echolocation** (hearing the handkerchief drop) — can't observe opponent's action before choosing
- **Heartbeat reading** (detecting stress) — no stress signal in the engine
- **Volume manipulation** (lowering the speaking clock) — no environmental state
- **Microexpression reading** — no behavioral signal

These are replaced by the **belief tracker**: Hal infers Baku's tendencies from action history, achieving the same narrative effect (Hal "reads" you) through statistical inference rather than supernatural perception.

---

## File Structure

```
STL/environment/opponents/hal/
  __init__.py          # Exports CanonicalHal
  types.py             # Bucket, BeliefState, MemoryMode, HalState
  buckets.py           # Bucket definitions, bucket_pair_payoff, resolve_bucket
  search.py            # Multi-turn backward induction, forward simulation
  solver.py            # LP Nash solver (solve_minimax), best-response
  evaluate.py          # Handcrafted evaluation function (Phase 1)
  belief.py            # Belief tracker (exploitation detection)
  memory.py            # Perfect Mode state machine
  hal_opponent.py      # Main entry point — composes all components
  
  # Phase 2 additions:
  value_net.py         # NN value network definition + inference
  self_play.py         # Self-play data generation
  train.py             # Training loop for value network
```

---

## Files To Reuse

| File | What To Use |
|---|---|
| `src/Game.py` | `play_half_round()`, `get_roles_for_half()`, `get_turn_duration()`, `is_leap_second_turn()` |
| `src/Player.py` | Player state (cylinder, ttd, deaths, alive, physicality) |
| `src/Referee.py` | `compute_survival_probability()`, `attempt_revival()` |
| `src/Constants.py` | All constants (CYLINDER_MAX, FAILED_CHECK_PENALTY, etc.) |
| `environment/route_math.py` | `lsr_variation_from_clock()`, `safe_strategy_budget()`, `get_named_players()` |
| `environment/strategy_features.py` | `build_strategy_snapshot()` (for belief tracker context) |
| `environment/opponents/base.py` | `Opponent` ABC (interface to implement) |
| `environment/opponents/teacher_helpers.py` | `clamp_second()` |
| `cfr/half_round.py` | `survival_probability()`, `compute_payoff_matrix()` (reference for payoff logic) |

---

## Implementation Phases

### Phase 1: Core Solver (No ML)

Goal: A search-based Hal that plays at or above the level of the existing scripted teachers, using only handcrafted evaluation. Verifiable against the manga's canonical game.

**1.1 — Types and buckets** (`types.py`, `buckets.py`)
- Define `Bucket`, `BeliefState`, `MemoryMode`, `HalState` as frozen dataclasses/enums
- Implement `STANDARD_BUCKETS`, `LEAP_BUCKET`
- Implement `bucket_pair_payoff()` and `resolve_bucket()`
- Test: verify bucket payoffs match full 60×60 payoff matrix averages

**1.2 — LP solver** (`solver.py`)
- Implement `solve_minimax()` using `scipy.optimize.linprog`
- Implement `best_response()` for exploitation mode
- Test: verify against known 2×2 and 3×3 Nash solutions; cross-validate with `cfr/half_round.py:solve_half_round()`

**1.3 — Game state cloning and forward simulation** (`search.py`, part 1)
- Implement `clone_game()` (deepcopy or manual field copy)
- Implement `simulate_bucket_pair()` — deterministic forward simulation with survival probability branching
- Test: clone a game, simulate a half-round on the clone, verify original is unchanged

**1.4 — Search engine** (`search.py`, part 2)
- Implement `search()` — multi-turn backward induction with LP Nash at each node
- Configurable search depth (3-4 half-rounds)
- Handle death probability branching (expected value over survived/died)
- Test: search from a simple late-game state with known optimal play; verify solver finds it

**1.5 — Evaluation function** (`evaluate.py`)
- Implement the 8-feature handcrafted evaluation
- Initial weights from the specification above
- Test: verify evaluation correctly ranks obviously-good vs obviously-bad positions

**1.6 — Memory model** (`memory.py`)
- Implement Perfect Mode FSM (NORMAL → PRE_AMNESIA → AMNESIA → RECOVERED)
- Test: verify transitions at correct game states; verify AMNESIA restricts bucket selection

**1.7 — Belief tracker** (`belief.py`)
- Implement action history tracking and entropy-based exploitation detection
- Implement `predict_baku_strategy()` — empirical bucket distribution from history
- Test: verify exploitation triggers after 4+ predictable actions; verify Nash is default

**1.8 — Main opponent class** (`hal_opponent.py`)
- Implement `CanonicalHal(Opponent)` composing all components
- Wire into `factory.py` as `"hal_canonical"`
- Test: verify `choose_action()` returns valid seconds; verify `reset()` clears state

**1.9 — Canon regression tests** (`tests/test_hal_canonical.py`)
- Replay manga timeline: CanonicalHal vs canonical-Baku
- Verify Hal's chosen buckets match expected canonical play at each round
- This is the primary correctness signal — if it matches, the evaluation function is right

**1.10 — Evaluation weight tuning**
- Run CanonicalHal vs all BakuTeacher variants (500+ games each)
- Adjust weights until: winrate > 75% vs teachers, canonical regression passes
- Document final weights

### Phase 2: RL Value Network

Goal: Replace the handcrafted evaluation with a learned one. The search stays the same — only the leaf evaluator changes. This should increase strength significantly, especially in unusual positions the handcrafted function misjudges.

**2.1 — Value network definition** (`value_net.py`)
- 24-input, 2×64 hidden, 1-output (tanh) network
- PyTorch implementation
- Inference function that takes game state → value

**2.2 — Self-play data generation** (`self_play.py`)
- Play games: CanonicalHal (with search + current network) vs itself
- Record (state_features, game_outcome) at each visited node
- Generate 10K-50K games per iteration

**2.3 — Training loop** (`train.py`)
- MSE loss: `(V(s) - outcome)²`
- Adam optimizer, lr=1e-3, batch_size=256
- Train for 10-20 epochs per data batch
- Checkpoint after each iteration

**2.4 — Integration with search** (`evaluate.py` updated)
- `evaluate()` dispatches to handcrafted or NN based on config
- NN evaluation should be batched for efficiency (collect all leaf states, evaluate in one forward pass)

**2.5 — Evaluation vs Phase 1**
- Compare RL-Hal vs Handcrafted-Hal head-to-head (500+ games)
- Compare both vs BakuTeachers
- RL-Hal should be strictly stronger (higher winrate, faster wins)

### Phase 3: OCaml Port

**3.1 — Port types and buckets** (pure OCaml, no dependencies)
**3.2 — Port LP solver** (hand-rolled simplex, ~200 lines)
**3.3 — Port search engine** (immutable state threading, functional recursion)
**3.4 — Port handcrafted evaluation** (pure arithmetic)
**3.5 — Port value network inference** (`ocaml-torch` bindings, load PyTorch checkpoint)
**3.6 — Verify Python/OCaml parity** (same seed → same bucket selections)

---

## Source Map

This design draws from:

- [SURPASSING THE LEADER- HAL DOC.pdf](/home/jcena/STL/docs/canon/SURPASSING%20THE%20LEADER-%20HAL%20DOC.pdf) — Hal's deduction process, death trades, Perfect Mode
- [Leader's Deviation Strategy.pdf](/home/jcena/STL/docs/canon/Leader%E2%80%99s%20Deviation%20Strategy.pdf) — LSR variations, route control, 2-second margin
- [STL Rules.pdf](/home/jcena/STL/docs/canon/STL%20Rules.pdf) — Game mechanics
- Existing `cfr/tree.py` — backward induction structure (adapted from per-state CFR to per-turn search)
- Existing `hal_policy_helpers.py` — canonical action targets (used for regression testing, not as decision logic)
