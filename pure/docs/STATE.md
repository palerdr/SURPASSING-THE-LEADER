# Pure STL state and exact finite-horizon solver

`pure/solver.py` is the executable authority for this abstraction. This
document describes that module exactly; it does not describe the canonical
engine or add rules that the module does not implement.

## Scope

The pure game keeps:

- fully observed simultaneous Dropper and Checker actions;
- one-second action slots over a normal 60-second half-turn;
- accumulated Squandered Time (ST) and prior cumulative Time to Death (TTD)
  for both players;
- stochastic revival determined by the current failed-check dose and prior
  TTD;
- role reversal after every nonterminal half-turn;
- terminal zero-sum utility and a finite half-turn horizon.

It omits the leap second, absolute clock, named player identities, physicality,
referee/CPR fatigue, death counters, and psychological or private information.

## Actions and elapsed-time convention

Both roles have the same legal action labels:

$$
\mathcal A_D=\mathcal A_C=\{1,2,\ldots,60\}.
$$

There is no action `0`, pass, or no-check action. Action `1` is the immediate
first slot. Equivalently, the elapsed drop timestamp represented by label $d$
is $d-1$, while check label $c$ resolves at elapsed time $c$.

A check succeeds exactly when

$$
c\ge d.
$$

Its inclusive ST increment is

$$
\ell=c-d+1.
$$

Consequently every success adds between 1 and 60 seconds of ST. In particular,
`d=1,c=1` succeeds for one ST, `d=1,c=60` succeeds for 60 ST, and
`d=60,c=59` fails.

## Live and terminal states

A live state is the immutable four-tuple

$$
x=(s_C,t_C,s_D,t_D),
$$

stored in Python as

```python
NTState = tuple[int, int, int, int]
```

where:

- $s_C$ and $t_C$ belong to the current Checker;
- $s_D$ and $t_D$ belong to the current Dropper;
- $s$ is current accumulated ST;
- $t$ is prior cumulative TTD.

The coordinates are role-based, not identity-based. Whenever play continues,
the old Dropper becomes the next Checker and the old Checker becomes the next
Dropper. No role bit or round parity is needed.

Live ST values are the exact integers `0..299`:

```python
ST_SUPPORT = [0, 1, ..., 299]
```

They must not be compressed at 240. Although any failed check from ST 240 or
higher is certainly fatal, values `240..299` remain strategically distinct:
a successful check adds between 1 and 60 ST, and the cylinder dumps immediately
when the resulting total reaches 300.

Reachable live TTD values are exact integers `0..300`. Exactly 300 is a live
boundary value; a later positive-duration death necessarily crosses above 300
and is fatal. See [`docs/CANONICAL_EXTENSIVE_FORM.md`](../docs/CANONICAL_EXTENSIVE_FORM.md)
for the shared source-backed distinction between cylinder capacity and TTD.

Terminal states are:

```python
TState.W = 1
TState.L = 0
```

Utility is measured from the current Dropper's perspective:

$$
u(W)=+1,\qquad u(L)=-1.
$$

The local transition function emits `W` when the current Checker dies
permanently, because that is a win for the current Dropper. `L` completes the
terminal payoff type but is not emitted by the current one-step transition.

## Revival model

On a failed check, the current Checker's dose is

$$
q=s_C+60.
$$

Revival is impossible when the current-dose threshold is reached or resulting
cumulative TTD strictly exceeds five minutes:

$$
q\ge300\quad\text{or}\quad t_C+q>300.
$$

Otherwise the revival probability is

$$
\boxed{
p(s_C,t_C)=
\left[1-\left(\frac{q}{300}\right)^3\right]
2^{-t_C/240}
},\qquad q=s_C+60.
$$

The dose term makes larger current deaths more dangerous. The exponential term
degrades survival as prior cumulative TTD increases. The formula uses prior TTD;
the current dose is added only in a surviving continuation. Thus
`t_C + q == 300` still has the displayed nonzero probability whenever `q < 300`.

## Transition distribution

`transition(x, d, c)` returns a tuple of `(probability, child)` branches.
Branch probabilities are explicit; the exact solver never samples revival.

### Successful check

If $c\ge d$, let $\ell=c-d+1$ and $s_C'=s_C+\ell$.

If $s_C'\ge300$, the cylinder reaches its physical capacity and dumps. The
injected dose is capped at 300, which is inclusively fatal under this model, so
the current Dropper wins:

$$
(s_C,t_C,s_D,t_D)\longrightarrow W.
$$

If $s_C'<300$, TTD is unchanged and roles reverse deterministically:

$$
(s_C,t_C,s_D,t_D)
\longrightarrow
(s_D,t_D,s_C',t_C).
$$

### Failed check

If $c<d$, calculate $p=p(s_C,t_C)$.

If $p=0$, the current Checker dies permanently:

$$
1\longrightarrow W.
$$

Otherwise there are two branches:

$$
\begin{aligned}
p &: (s_D,t_D,0,t_C+s_C+60),\\
1-p &: W.
\end{aligned}
$$

On revival, the failed Checker's ST resets to zero, their TTD increases by the
current dose, and roles reverse.

## Finite-horizon value

Let $V_h(x)$ be the equilibrium value for the current Dropper with $h$
half-turns remaining. A live state at the cutoff is a draw:

$$
V_0(x)=0.
$$

For one action pair, terminal branches use their terminal reward. Every live
child has reversed roles, so its continuation value is negated:

$$
Q_h(x,d,c)=
\sum_{(P,y)\in T(x,d,c)}
P\begin{cases}
u(y),&y\text{ terminal},\\
-V_{h-1}(y),&y\text{ live}.
\end{cases}
$$

The sign change is the complete role-orientation rule. Values and terminal
outcomes therefore lie in $[-1,1]$.

## Stage matrix and minimax LP

The payoff matrix is

$$
M_h(x)\in\mathbb R^{60\times60},
$$

with Dropper actions on rows, Checker actions on columns, and

$$
M_h(x)_{d,c}=Q_h(x,d,c).
$$

The Dropper maximizes and the Checker minimizes. The Dropper LP is

$$
\begin{aligned}
\max_{p,v}\quad &v\\
\text{subject to}\quad&M^\mathsf T p\ge v\mathbf1,\\
&\mathbf1^\mathsf T p=1,\quad p\ge0.
\end{aligned}
$$

The Checker dual is

$$
\begin{aligned}
\min_{q,w}\quad&w\\
\text{subject to}\quad&Mq\le w\mathbf1,\\
&\mathbf1^\mathsf T q=1,\quad q\ge0.
\end{aligned}
$$

`scipy.optimize.linprog(method="highs")` solves both programs. Returned
policies are clipped at zero and normalized. The numerical certificate is

$$
\epsilon=max_i(Mq)_i-\min_j(M^\mathsf T p)_j.
$$

`solve_matrix` rejects a result when this saddle gap exceeds $10^{-6}$.
`solve(x,h)` memoizes the complete value and both role policies.

## Reachable horizon-1--3 target artifact

Dataset generation is deliberately separate from the mathematical solver:

```powershell
uv run python -m pure.generate_dataset
```

Hydra loads `pure/config/dataset.yaml`; `horizon`, `output`, `root_states`, and
`progress_every` are command-line overridable. `pure/solver.py` performs no
filesystem writes when imported or executed.

Starting from the configured opening state `(0,0,0,0)`, the generator's
`live_successors` enumerates every distinct live one-step successor using 60
success-lag representatives and one failed-check representative. This is exact
because all successful cells with the same inclusive lag have the same
transition, and every failed cell has the same chance branches.

For the default cap $H=3$, the exporter stores precisely the state/horizon pairs
encountered in a three-half-turn finite game:

- depth 0 opening state with remaining horizon 3;
- depth 1 reachable states with remaining horizon 2;
- depth 2 reachable states with remaining horizon 1.

Terminal branches are represented in the backed-up values and do not become
dataset rows. States inside each depth are sorted lexicographically, and rows
are written from remaining horizon 1 through 3.

The output is:

```text
pure/artifacts/exact_h3.npz
```

with arrays:

| Array | Shape | Dtype | Meaning |
|---|---:|---|---|
| `states` | `(N,4)` | `int16` | `(sC,tC,sD,tD)` |
| `horizons` | `(N,)` | `uint8` | remaining half-turns |
| `values` | `(N,)` | `float32` | current-Dropper value |
| `drop_policies` | `(N,60)` | `float32` | equilibrium row marginals |
| `check_policies` | `(N,60)` | `float32` | equilibrium column marginals |
| `saddle_gaps` | `(N,)` | `float32` | LP certificate gap |
| `drop_actions` | `(60,)` | `int16` | action labels `1..60` |
| `check_actions` | `(60,)` | `int16` | action labels `1..60` |
| `schema_version` | scalar | string | `pure-v3-ttd-strict-overflow` |

The dynamic program and LP solves are numerical exact-target generation; the
NPZ intentionally stores compact `float32` training arrays.

### Consequence at horizon 3

The corrected game is not indefinitely safe: repeatedly choosing `check=60`
accumulates ST and eventually reaches the fatal 300-second cylinder cap.
However, horizon 3 is too short for that consequence to be reachable from the
empty opening state. A player acts as Checker at most twice in three alternating
half-turns and can accumulate at most 120 ST.

For the generated $H=3$ artifact:

- 3,721 depth-2 states have remaining horizon 1;
- 61 depth-1 states have remaining horizon 2;
- the opening state has remaining horizon 3;
- all 3,783 stored values are therefore still zero;
- HiGHS returns `drop=1` and `check=60` for every stored row.

These are shallow-horizon policies, not infinite-horizon conclusions. The
returned `drop=1` policy is one LP-selected member of a non-unique set of
optimal Dropper policies at this cutoff, so it must not be interpreted as a
uniquely identified behavioral target. This artifact is valid for
transition/LP/training-pipeline tests, but it contains no value variation.

## Learning contract

Each row is a supervised target

$$
(s_C,t_C,s_D,t_D,h)
\mapsto
(V_h,\pi_D^\star,\pi_C^\star).
$$

A small network therefore needs one scalar current-Dropper value output and two
independently normalized 60-action policy outputs. Any later MCTS implementation
must negate live-child values after role reversal and use a leaf value trained
for the same remaining horizon.

For self-play under this role-canonical value convention, terminal labels
alternate sign between consecutive surviving positions. Truncated positions
receive value zero.

## Required invariants

1. Both action sets are exactly `1..60`; zero and pass are illegal by contract.
2. Success is exactly `c >= d` and successful ST is `c - d + 1`.
3. Every transition branch probability lies in `[0,1]` and branches sum to one.
4. Successful accumulation to ST 300 immediately dumps the cylinder and is
   terminal; live successful continuations preserve exact ST below 300.
5. Successful and revived continuations swap the role-coordinate pairs.
6. Revival resets only the failed Checker's ST and adds the dose to their TTD.
7. Dose 300 is fatal; resulting total TTD 300 remains revival-eligible, while
   resulting total TTD above 300 is fatal.
8. Horizon zero has value zero and no policy targets.
9. Matrix rows are Dropper actions; columns are Checker actions.
10. Dropper maximizes, Checker minimizes, and every stored policy sums to one.
11. Every solved matrix has saddle gap at most `1e-6` before float32 storage.
