# Pure DTH exactness boundary

## Damage DAG

For every live transition in `dth.solver.transition`, raw damage

\[
R(x)=s_c+t_c+s_d+t_d
\]

strictly increases. A successful live transition increases it by the inclusive
lag \(c-d+1\). A revived failed check increases it by 60. Therefore the
complete-game dependency graph is acyclic, and descending-rank dynamic
programming is valid.

`complete_game_dependencies` enumerates the unique live children represented
by the 61 Bellman transition classes. Census expansion never solves an LP.
Solving begins only after the reachable frontier is exhausted, and a state is
eligible only after every greater-rank child has an exact committed value.

## Failure-dead quotient

If both ST coordinates are at least 240, every future failed check is fatal.
TTD can never affect another transition. Define remaining capacities

\[
(a,b)=(300-s_c,300-s_d),\qquad 1\le a,b\le60.
\]

All states with the same \((a,b)\) are behaviorally equivalent. A successful
lag \(\ell\) is live exactly when \(1\le\ell<a\), with quotient child

\[
(a,b)\to(b,a-\ell).
\]

Failures and lags \(\ell\ge a\) are terminal. This proves the quotient without
using a value approximation.

The persistent key gives quotient classes a disjoint negative ID range. Since
entering the quotient erases dead TTD, its scheduling ranks occupy a disjoint
band above every raw-state rank; successful lags strictly increase rank inside
that band. A checker-turn bitset stores reachable \(b\) values for each \(a\).
For root `(240,0,240,0)`, the bitset recurrence yields exactly

\[
1+59+59^2=3541
\]

reachable equivalence classes.

## Per-player TTD-dead quotient

The failure-dead quotient above requires *both* STs at 240. The per-player
generalization quotients each player independently: the complete game reads a
player's TTD in exactly one place, the revival probability
`revival_model(s_c, t_c)` of a failed check, and

1. `survive_injection(s, t)` is exactly \(s\le239\wedge s+t\le240\) and a
   failing profile has `revival_model == 0`;
2. failing survival is absorbing: a successful check raises that player's ST
   and leaves their TTD alone, a failed one ends the game for them unless
   they survive, so ST growth is the only motion available to a dead profile
   and it never revives it (`tests/test_dead_ttd_quotient.py` locks all
   three claims exhaustively).

A dead profile's TTD is therefore never read again: on the only branch that
reads TTD the probability is identically zero, so the transition distribution
— and hence the value — factors through the map that replaces a dead TTD
with a per-ST sentinel. That collapses the 72,600 per-player \((s,t)\)
profiles over the reachable TTD domain \(\{0\}\cup[60,300]\) to
\(16{,}711+300=17{,}011\) classes and the two-player space to
\(17{,}011^2=289{,}374{,}121\).

The domain \(\{0\}\cup[60,300]\) is transition closed: revival requires
\(s_c+t_c\le240\), so a failure child's TTD \(t_c+s_c+60\) never exceeds 300.
An **alive** profile with TTD in 1..59 is a valid live state outside this
closure; the packed codec (`dth.packed`) fails closed on it rather than
approximate. A dead profile is accepted with any TTD because the quotient
discards dead TTDs exactly.

## Backup sweep potential

The packed backup tablebase schedules the quotient space by the potential

\[
\Phi(x)=s_c+s_d+\rho(s_c,t_c)+\rho(s_d,t_d),\qquad
\rho(s,t)=\begin{cases}t&\text{alive}\\301&\text{dead,}\end{cases}
\]

with maximum \(299+299+301+301=1200\). Every live transition strictly
increases it, by cases on the mover's profile:

- successful lag \(\ell\), profile stays alive or stays dead:
  \(\Delta\Phi=\ell\);
- successful lag \(\ell\), alive profile dies: \(\Delta\Phi=\ell+301-t\ge
  \ell+61\) since alive implies \(t\le240\) (dead never revives, so the
  reverse move does not exist);
- failed check, revived profile alive: \(\Delta\Phi=60\) (the ST resets to
  zero and the new TTD is \(t_c+s_c+60\));
- failed check, revived profile dead: \(\Delta\Phi=301-(s_c+t_c)\ge61\).

`tests/test_backup_potential.py` checks all 1,035,541 live profile
transitions exhaustively. Since no edge stays inside a layer, solving whole
layers in descending \(\Phi\) sees every child before its parents, which is
the correctness basis of `dth.backup_tablebase`; its per-class certificates
are re-derivable on demand from stored child values (`recertify_class`), the
same Bellman-recertification standard the SQLite pipeline applies to queried
roots. Cross-backend behavior is governed by `docs/DTH_BACKUP_PARITY.md`.

## Certified value intervals

An unknown live child begins at the mathematical bound \([-1,1]\). If a child
has interval \([L,U]\), role reversal contributes \([-U,-L]\). Expected branch
bounds produce elementwise lower and upper matrices \(M^-\le M^+\).
Monotonicity of zero-sum game value gives

\[
\operatorname{val}(M^-)\le V(x)\le\operatorname{val}(M^+).
\]

Interval refinement may narrow these bounds but may never widen them. An
interval midpoint is not an authoritative child value. Exact rank solving
requires every live child to have an exact committed float64 value.

## Persistence and replay

One fail-closed SQLite schema owns roots, census rows, rank layers, value
intervals, exact scalar certificates, optional queried-root policy caches, and
queue status. Exact value insertion and queue completion occur in one
transaction. Internal states do not store policies.

A queried root policy is reconstructed from certified child values, checked
against the stored root value, and optionally cached. Reopening verifies table
shape, metadata, rule/schema hashes, interval digests, cached policy digests,
and the \(10^{-6}\) saddle-gap bound. Legacy schemas are rejected; there is no
silent migration.

No complete-game solution claim is valid until the requested root has an exact
committed value and its reconstructed policy passes Bellman recertification.
