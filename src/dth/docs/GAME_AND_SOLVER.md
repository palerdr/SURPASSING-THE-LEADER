# Pure DTH rules and Bellman formulation

## Scope

Pure Drop the Handkerchief has no leap-second, route, observation, or
information-state mechanics. A live role-canonical state is

\[
x=(s_c,t_c,s_d,t_d),
\]

where `c` is the current Checker, `d` is the current Dropper, ST coordinates
are in `0..299`, and TTD coordinates are in `0..300`. Values are always from
the current Dropper's perspective.

## Actions and inclusive elapsed time

Both players choose literal integer seconds in `1..60`. Action 0 is illegal.
For Dropper action \(d\) and Checker action \(c\), a check succeeds exactly
when \(d \le c\). Its inclusive elapsed time is

\[
\operatorname{ST}(c,d)=c-d+1.
\]

Thus equal actions add one second, not zero.

## Transitions

On success, set

\[
s'_c=s_c+\operatorname{ST}(c,d).
\]

If \(s'_c \ge 300\), the current Dropper wins. Otherwise roles swap:

\[
(s_c,t_c,s_d,t_d)\to(s_d,t_d,s'_c,t_c).
\]

On failure, the injection dose is \(q=s_c+60\). Revival is impossible if
\(q\ge300\) or \(t_c+q>300\); equivalently, revival is possible exactly when
\(s_c\le239\) and \(s_c+t_c\le240\). Otherwise the revival probability is the
repository-wide frozen surface from
[`docs/REVIVAL_MODEL.md`](../../../docs/REVIVAL_MODEL.md):

\[
p_{\mathrm{revive}}(s_c,t_c)
=
0.95\left(1-\frac{s_c}{240}\right)\cdot 0.75^{\,t_c/60}.
\]

The dose factor reaches zero exactly at the documentary lethal dose \(q=300\)
and nowhere else; the largest survivable injection is \(s_c=239\), where the
probability is \(0.95/240=0.003958\). The TTD factor has a 144.3-second
half-life. STL's referee floor is omitted because it cannot bind while
eligibility forces \(t_c\le240\).

DTH does not own these constants and must not diverge from them. The surface is
bucket-invariant, so `abstract`'s 10-second and 5-second rulesets and the
leap-aware STL engine evaluate identically at the same physical \((s,t)\).

Death gives the current Dropper payoff \(+1\). Revival swaps roles and yields

\[
(s_d,t_d,0,t_c+q).
\]

All live children have strictly greater raw damage
\(s_c+t_c+s_d+t_d\).

`dth.solver.transition` is the executable rule authority for ladder rung L1
(see [`docs/FORMULATION_LADDER.md`](../../../docs/FORMULATION_LADDER.md)).
Persistent artifacts bind to a source-derived schema hash and fail closed after
any rule change, including changes to `revival_model` or the failure-dead
quotient threshold.

## Bellman matrix

Let \(V(x)\) be the complete-game value from the current Dropper's view. For
each joint action,

\[
M_x[d,c]
=
\mathbb E\left[
\begin{cases}
+1,&\text{current Dropper wins},\\
-V(x'),&\text{the game continues at }x'.
\end{cases}
\right].
\]

The role swap causes the minus sign. The state value is the zero-sum matrix
value

\[
V(x)=\max_p\min_q p^\top M_xq.
\]

There are only 61 transition classes: 60 successful lags \(c-d+1\) and one
failed-check class. `continuation_class_values` evaluates those classes and
`reconstruct_transition_class_matrix` expands them to the literal 60 by 60
matrix. Tests compare that reconstruction with the independent joint-action
builder to absolute tolerance \(10^{-12}\).

The preferred matrix path solves the full-support structured equilibrium
equations, then validates nonnegative normalized policies and the saddle gap.
Singular, boundary-support, or insufficiently accurate candidates fall back to
the HiGHS primal and dual LPs. Every accepted certificate must satisfy

\[
\max_d(Mq)_d-\min_c(M^\top p)_c\le10^{-6}.
\]

Finite-horizon values use the same matrix formulation with live cutoff value
zero at horizon zero and remain useful only for research comparisons. The
production play value is read from the complete quotient tablebase.
