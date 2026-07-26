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
