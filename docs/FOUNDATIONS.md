# Solver Foundations

This file owns shared solver mathematics. It does not redefine game rules;
those are frozen in [`ACTION_TIMING.md`](ACTION_TIMING.md),
[`CANONICAL_EXTENSIVE_FORM.md`](CANONICAL_EXTENSIVE_FORM.md), and
[`REVIVAL_MODEL.md`](REVIVAL_MODEL.md). Which games are claimed at all is frozen
in [`FORMULATION_LADDER.md`](FORMULATION_LADDER.md).

## Simultaneous zero-sum decision

At each public state the Dropper and Checker choose simultaneously. Expanding
all legal joint actions produces a matrix whose cells are expected continuation
values after deterministic resolution and engine-derived chance branches.
Alternating pure-action minimax is a different game because it reveals one
player's current action before the other acts.

The canonical matrix value is the zero-sum minimax equilibrium. Linear
programming is the truth oracle; CFR+ may approximate a matrix equilibrium when
its exploitability or primal-dual residual is certified against that oracle.

## Finite-horizon exactness

For horizon zero, terminal states have fixed utility and nonterminal states use
the declared cutoff contract. For horizon `h>0`, enumerate every legal joint
action, expand all chance successors from the owning engine, recursively solve
at `h-1`, and solve the resulting matrix. Induction on `h` proves exactness for
the declared finite-horizon public game.

## Search and learning boundary

Matrix-game MCTS may select and sample joint-action cells, but its canonical
improved policy is a pair of legal role marginals. Learned values are leaf
approximators around exact/tablebase anchors; they do not change engine rules or
certify themselves. Promotion requires calibration, deterministic comparison,
exploitability evidence, and any declared policy-drift/readability gates.

## Claim boundary

The current STL solver is a fully observed public-state baseline at ladder rung
L2. It does not claim to solve a private-information game with hidden leap
knowledge, memory loss, signaling, or unobserved within-turn information — those
are rungs L3 and L4, and neither is implemented. DTH (L1) and abstract (L0) make
still smaller explicit abstractions, documented locally.

Every rung shares one revival surface. A solver may not compensate for a rung it
cannot represent by altering revival constants.
