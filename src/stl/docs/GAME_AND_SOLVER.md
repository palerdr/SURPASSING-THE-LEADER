# Full STL game and solver formulation

This document freezes the target formulation for the full Surpassing the
Leader (STL) match. It begins at the source opening, preserves the complete
public action history, and separates game rules from a model of Hal's memory.
It is the contract to implement before another STL solver or learning pipeline
is added.

## What we are building, in plain terms

We are building two games in order, not one undifferentiated learned agent:

1. **The public leap game (L2).** This is ordinary DTH plus the exact wall
   clock and Baku's possible Dropper action 61. It has no hidden memory. It is
   still a finite perfect-information stochastic game and is the robust
   baseline we should solve first.
2. **The Hal-exploitation game (L3).** This keeps the same exact mechanics but
   adds Hal's hidden awareness bit and a model that uses the complete revealed
   history to predict Hal. This layer is approximate because the source does
   not specify Hal's behavior on every counterfactual history.

The solved DTH tablebase is reused as the exact tail of L2 after the single
leap second has passed. It is not asked to understand the clock, the route,
Hal's memory, or public history. L3 then uses the solved L2 policy as its safe
baseline and departs from it only when the opponent model supplies enough
evidence to justify an exploit.

The shared mechanical rules remain owned by
[`ACTION_TIMING.md`](../../../docs/ACTION_TIMING.md),
[`CANONICAL_EXTENSIVE_FORM.md`](../../../docs/CANONICAL_EXTENSIVE_FORM.md), and
[`REVIVAL_MODEL.md`](../../../docs/REVIVAL_MODEL.md). The supported-claim
boundary remains owned by
[`FORMULATION_LADDER.md`](../../../docs/FORMULATION_LADDER.md). If this document
and one of those authorities disagree, the repository-wide authority wins.

## Evidence and modeling boundary

Three kinds of statement must not be conflated:

- **Documentary rule:** directly supported by the game or the numeric match
  ledger. These rules are invariant across opponent models.
- **Selected interpretation:** the best-supported reading of cognition or
  strategy in the two source analyses. It labels the canonical trace but does
  not create an invented off-path law.
- **Solver model:** a versioned approximation used to predict an opponent. It
  can be trained or replaced without changing the game.

The exact opening and four-route arithmetic are recorded by
[E-STL-OPENING](../../../docs/game-sources/EVIDENCE.md#e-stl-opening) and
[E-LSR-VARIANTS](../../../docs/game-sources/EVIDENCE.md#e-lsr-variants). The
memory chronology is recorded by
[E-HAL-MEMORY-SEQUENCE](../../../docs/game-sources/EVIDENCE.md#e-hal-memory-sequence).
The route derivation comes specifically from PDF pages 2--9 of
[`Leader-Deviation-Strategy.pdf`](../../../docs/game-sources/Leader-Deviation-Strategy.pdf).

The PDFs in `docs/game-sources/` are interpretive analyses containing primary
panels, not primary editions themselves. Their arithmetic and chronology are
strong evidence; their claims about subconscious motivation are hypotheses.

## Players, utility, and chance

The fixed identities are Hal and Baku. Hal is the first Dropper in every round;
Baku is the first Checker. Roles swap for the second half-round.

```text
u_Hal  = +1 if Hal wins, -1 if Baku wins
u_Baku = -u_Hal
```

There is no canonical draw. A horizon value or draw used by an approximate
solver is a computational boundary, not a game outcome.

Every injection has one chance event: survival under the single frozen
`P_rev(s,t)` in [`REVIVAL_MODEL.md`](../../../docs/REVIVAL_MODEL.md). Referee revival count
may be retained in public history, and player physicality may be retained as
descriptive metadata, but neither is a separate probability input. Adding a
second referee-success gate would double-count the referee degradation already
folded into `P_rev` and would define a different repository-wide game.

## Exact initial node

The canonical game starts at the following node, not at a configurable generic
clock:

```text
wall clock       8:12:00 AM, before the leap second at 8:59:60 AM
round / half     Round 1 / half 1
Dropper          Hal
Checker          Baku
Hal              vial = 0, TTD = 0
Baku             vial = 0, TTD = 0
route diagnostic V1
Hal leap memory false
public history   empty
```

Both players know from initialization that a leap second will occur and know
the ordinary game rules. That common rule knowledge is distinct from conscious
access to Baku's concrete leap-second plan.

## Live state, history, and terminal result

The minimal live transition state is:

```python
@dataclass(frozen=True, slots=True)
class WorldState:
    baku_load: int
    baku_ttd: int
    hal_load: int
    hal_ttd: int
    half: int
    clock: int
    hal_leap_memory: bool
```

`clock` is elapsed physical time from 8:00:00 AM with the inserted leap second;
the formatted wall clock, route variant, and leap-window legality are derived
from it. `half` is 1 or 2. Hal is always the first-half Dropper, so the current
Dropper is derived from `half` rather than stored independently.

The live state deliberately has no `alive`, death-count, or referee-attempt
fields. Permanent death immediately leaves the live domain and produces a
separate terminal result containing the winner. Loads and TTD contain every
prior death's physical consequence. Referee attempts do not affect any rule.

`round` is replay metadata, not physical state. It can be derived from the
ordered public history. A UI may cache derived labels, but they do not
participate in state identity or artifact hashes.

The public history is a separate immutable ordered sequence, not a member of
`WorldState`. Its minimal half-round record is:

```python
@dataclass(frozen=True, slots=True)
class PublicHalfRound:
    drop_second: int
    check_second: int
    survived: bool | None
```

A complete canonical node uses only immutable value types; it does not retain
the mutable compatibility-engine `HalfRoundRecord` or `Player` objects:

```python
PlayerIdentity = Literal["Hal", "Baku"]

@dataclass(frozen=True, slots=True)
class GameState:
    world: WorldState
    public_history: tuple[PublicHalfRound, ...]
    winner: PlayerIdentity | None
```

`winner` is `None` for a nonterminal node and the fixed identity `"Hal"` or
`"Baku"` after permanent death. It is an identity value, not mutable player
state.

`survived` is `None` when no injection occurred. Starting clock, round, half,
roles, success/failure, ST, dose, survival probability, and successor state are
all reproducible by replaying this sequence from the fixed opening and need not
be duplicated. Referee attempt count or announcements may be display events,
but they are not solver state.

Drop and check choices are simultaneous private commitments. Neither action is
public before both have committed. Once resolved, both actions and the outcome
enter public history. A client or policy never receives the opponent's
unrevealed current action.

The exact engine advances `WorldState`. Hal may observe his own memory bit, but
Baku and an exploitative agent must not. Their input is the public projection
of `WorldState` with `hal_leap_memory` removed, plus `public_history`; they carry
a belief over the hidden bit. Observation functions, not destructive edits to
history, represent forgetting. This preserves one exact replay without leaking
private state.

Put another way, one complete nonterminal game node is

```text
(physical state, hidden Hal memory, public history)

physical state = (
    baku_load, baku_ttd,
    hal_load,  hal_ttd,
    half, clock
)
hidden Hal memory = hal_leap_memory
public history = ordered revealed half-round records
```

These parts have different jobs:

- The physical state alone determines legal actions, doses, survival odds,
  clock advancement, and the next physical state.
- The hidden bit affects Hal's information and modeled policy, never physical
  legality.
- Public history is replay evidence and opponent-model input. It is not needed
  in an L2 Bellman cache because it does not alter public mechanics.
- Baku's solver state additionally contains a belief over Hal's hidden bit and
  behavioral type. That belief belongs to the agent, not to game physics.

`half` is needed because identities matter: Hal drops in half 1 and Baku drops
in half 2, and only Baku as Dropper can use action 61. `clock` is needed because
it decides whether that action exists. Route, round number, current Dropper,
death counts, alive flags, and referee attempts are derived or irrelevant and
are therefore not stored in the live state.

## Actions and physical transition

Normal action seconds are literal integers `1..60`; action 0 and passing are
illegal. A check succeeds exactly when `check >= drop`, and successful
Squandered Time is inclusive:

```text
ST = check - drop + 1
```

In the leap window only Baku as Dropper may choose 61. Checker remains capped
at 60, so Baku's drop at 61 necessarily defeats the check. Full legality is
owned by [`ACTION_TIMING.md`](../../../docs/ACTION_TIMING.md).

On a failed check, the current Checker's injected dose is `q = vial + 60`.
Capacity, cumulative TTD, revival eligibility, role exchange, and terminal
death follow
[`CANONICAL_EXTENSIVE_FORM.md`](../../../docs/CANONICAL_EXTENSIVE_FORM.md). On a
successful revival, the vial resets, `q` is added to TTD, and play continues.
If revival fails, the current Dropper wins.

The exact clock advances by the half-round duration and by every documented
procedural and near-death interval. The route is recomputed from the resulting
clock. It must never be advanced by a hard-coded route label.

The pure callable surface is intentionally small and lives in
`stl.solver.canonical`: `is_leap_window`, `turn_duration`, `lsr_variation`,
`is_active_lsr`, and `leap_drop_available`.

## Leap Second Route

Let `m` be the integer minute after 8:00 at the start of a round. The four LSR
variants are congruence classes:

```text
V(m) = 1 + ((m - 12) mod 4)

V1: 12, 16, 20, ..., 56
V2: 13, 17, 21, ..., 57    active
V3: 14, 18, 22, ..., 58
V4: 15, 19, 23, ..., 59
```

V2 is active because a round beginning at 8:57 places Baku's second-half
Dropper turn at 8:59, where second 61 exists. A no-death round advances four
minutes and preserves its congruence class. A death changes elapsed time by its
actual dose and procedure duration, so its successor class must be calculated,
not described as a fixed shift.

On the canonical trace the relevant progression is `V1 -> V4 -> V3 -> V2`,
ending with Round 9 at 8:57. This is a replay assertion, not a shortcut for
counterfactual play. “Opening LSR” means making an active V2 successor
reachable through legal deaths and survivals; “activating LSR” means actually
reaching it.

## Hal's binary awareness and imperfect recall

The private cognitive interface contains a binary variable

```text
A_H in {aware, unaware}
```

It answers one narrow question: does Hal currently have conscious access to
the concrete leap-second realization and its strategic consequence? It does
not change action legality, the physical clock, the route, or Hal's common
knowledge that leap seconds exist.

The canonical trace is labeled as follows:

| Event | Hal near-death count after event | `A_H` | Status |
|---|---:|---|---|
| Opening at 8:12 | 0 | unaware | selected interpretation |
| Baku's Round 1 near-death exposes the plan | 0 | aware | selected interpretation |
| Hal's Round 2 survived near-death | 1 | unaware | selective suppression/forgetting |
| Hal re-derives the plan before his Round 8 death | 1 | aware | selected interpretation |
| Hal's Round 8 survived near-death | 2 | unaware | scheduled broad memory loss |
| Round 9 leap injection | 3 | too late to affect the prior decision | documentary chronology |

The second Hal near-death, not the third, precedes and triggers the scheduled
memory reset before the leap turn. The third is the consequence of falling for
the leap second.

The analyses disagree about how much erased information continues to influence
Hal subconsciously. Therefore no environment transition says that `unaware`
forces a particular action. Subconscious carryover belongs in a versioned Hal
opponent model.

The source labels only the realized trace. It does not identify a complete
counterfactual transition kernel for awareness. A solver must therefore expose
its off-path model explicitly as

```text
K_A(A_H' | A_H, public_history, Hal_observation)
```

and report sensitivity to plausible alternatives. A learned `K_A` is an
opponent model, never documentary game physics. It receives public history, so
it can distinguish Hal's first and second survived deaths without duplicating
a death counter in `WorldState`. Hal's information set aliases histories
according to `A_H` and the chosen, versioned memory model.

## Exactly how the DTH tablebase is used

The DTH tablebase accepts the role-relative state

```text
(checker_load, checker_ttd, dropper_load, dropper_ttd)
```

and returns an exact value from the current Dropper's perspective plus
certified equilibrium policies over actions `1..60`. An STL physical state is
projected without approximation:

```text
half 1: (baku_load, baku_ttd, hal_load,  hal_ttd)   # Hal drops
half 2: (hal_load,  hal_ttd,  baku_load, baku_ttd) # Baku drops
```

For the repository-wide Hal utility, the DTH value keeps its sign in half 1
and is negated in half 2.

The projection does **not** make every ordinary-looking pre-leap state equal
to DTH. Before the leap second, future play may still reach action 61, so the
future incentives can differ even when the current actions are only `1..60`.
The strict exact boundary is:

```text
clock > 3600 (after 8:59:60 AM) -> use the DTH tablebase exactly
clock <= 3600                     -> solve the leap-aware STL continuation
```

A branch that jumps over the leap window because of procedure time may use DTH
as soon as its child clock is greater than 3600. Inside the leap window, the
STL stage matrix may be `61 x 60`; DTH has only the ordinary `60 x 60` game and
cannot supply the missing action's equilibrium consequence by itself.

Until L2 is solved, the DTH policy can be used before the leap as a conservative
engineering fallback, but it must be labeled a baseline rather than an exact
STL policy. Once L2 is solved, L2 replaces it as the pre-leap robust baseline;
DTH remains the exact post-leap tail.

The project boundary is deliberate. `src/dth/` owns and verifies the frozen
tablebase. `src/stl/` owns STL states and transitions and must not import DTH.
The neutral `src/arena/dth_adapter.py` is where the two public interfaces are
wired together.

## The actual solving approach

### Step 1: finish one pure canonical transition

`src/stl/solver/canonical.py` is currently only a state-and-clock skeleton. It
does not yet implement the canonical game or a solver. The next implementation
work is to add, in that file, pure functions for:

```text
roles(state)
legal_actions(state, actor)
transition(state, drop, check)
public_record(state, drop, check, chance_outcome)
replay(history)
```

`transition` must return terminal outcomes or explicit probability branches;
it must never roll random numbers. It owns load/TTD updates, the frozen revival
probability, half exchange, and exact clock advancement. The retained mutable
`stl.engine.Game` compatibility surface should then delegate to these rules
rather than becoming a second implementation.

The first gate is not training. It is a golden replay test from 8:12 that
reproduces the canonical eighteen half-rounds, five revival events, route
progression, memory labels, and final two-second TTD slack.

### Step 2: solve the public leap game exactly

Build a memoized Bellman solver keyed only by the L2 public Markov state:

```text
(baku_load, baku_ttd, hal_load, hal_ttd, half, clock)
```

At each state it constructs the legal simultaneous-action matrix, evaluates
terminal and revival branches, negates or reorients child values consistently,
and solves the zero-sum matrix. Every live transition raises physical damage,
so this remains a finite DAG. When a child has `clock > 3600`, the neutral
adapter supplies its exact DTH tablebase value instead of expanding it.

The initial L2 artifact only needs states reachable from the exact 8:12
opening. Store values and certified policies with one schema, one builder, and
one canonical document; do not create `v2` modules or status documents. Hydra
selects build and audit commands and their parameters.

### Step 3: add history-based exploitation without changing the game

After L2 passes its replay and Bellman audits, add a separate opponent model
that consumes

```text
(public physical state, complete public history)
```

and outputs a posterior over Hal's awareness/type and a predicted Hal action
distribution. It does not output authoritative game values and cannot change
legal actions or transitions. Start with an inspectable tabular or Bayesian
model around the canonical memory events; only use a neural sequence model if
the simpler model demonstrably cannot represent the data or hypotheses.

At play time, solve a bounded public-belief subgame against that posterior.
Use the solved L2 policy/value at the subgame boundary and as the fallback
policy. A candidate exploit is accepted only if evaluation against an
adversarial policy set keeps loss within an explicit safety budget. MCCFR or a
restricted safe-best-response calculation is the first appropriate tool. PPO
may later optimize a candidate response, but it is neither the value authority
nor the safety proof.

## Separation of concerns

| Surface | Owns | Must not own |
|---|---|---|
| `docs/` and this document | Frozen rules, state meaning, claim boundary | Experiment results or duplicate versioned plans |
| `src/stl/solver/canonical.py` | Pure STL state, legality, transition, replay | DTH imports, learning, artifact orchestration |
| STL public Bellman solver, added only after the transition is frozen | L2 matrix construction, recursion, certificates | Hal opponent assumptions |
| Hal opponent model, added only after L2 | History-to-belief and history-to-policy prediction | Physical rules or canonical values |
| `src/dth/` | Exact pure-DTH artifact and certified lookup | Clock, leap, history, or memory |
| `src/arena/dth_adapter.py` | Role/sign projection and DTH tail wiring | New game rules |
| Hydra configs | Parameters and experiment entry points | Alternative implementations of the game |

This order gives every approximate result an exact mechanical floor: canonical
transition first, exact L2 baseline second, opponent-specific L3 exploit last.

## Solver claim contract

The full formulation is a zero-sum stochastic game with hidden state and a
large public history. A single smooth state-value network is not the authority:
the exact capacity boundary, action 61, survival chance, memory events, and
opponent-specific history effects create real discontinuities.

A policy claim is incomplete unless it reports:

- performance against the modeled Hal distribution;
- worst-case loss against the exact baseline or an adversarial policy set;
- sensitivity to the off-path awareness kernel;
- exact legality and replay conformance; and
- the exploitability or safety budget used by the response.

## Claim boundary

This document freezes the target abstraction and the canonical-trace labels. It
does not claim that L3 has been implemented or solved. The retained STL engine
is a public-game compatibility surface; it is not evidence that the private
memory formulation exists in code. Any implementation must first reproduce the
exact opening, clock route, eighteen canonical half-rounds, all five revival
events, the two-second final TTD slack, and the memory chronology above.
