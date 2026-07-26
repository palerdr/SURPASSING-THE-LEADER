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
\(q\ge300\) or \(t_c+q>300\). Otherwise

\[
p_{\mathrm{revive}}
=
\left(1-\left(\frac{q}{300}\right)^3\right)2^{-t_c/240}.
\]

Death gives the current Dropper payoff \(+1\). Revival swaps roles and yields

\[
(s_d,t_d,0,t_c+q).
\]

All live children have strictly greater raw damage
\(s_c+t_c+s_d+t_d\).

`dth.solver.transition` is the executable rule authority. Persistent artifacts
bind to its schema hash and fail closed after any rule change.

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
zero at horizon zero.
