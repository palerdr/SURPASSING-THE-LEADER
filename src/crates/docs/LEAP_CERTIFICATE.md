# Certificate from nonnegative weights

We use this bound in `kernel.c` to avoid a second triangular matrix product.
We retain the full matrix check after clipping a negative weight.

We treat the gathered binary64 payoffs as the exact entries of the stage
matrix. You must account for child-value error through the accumulated bound
in `paper/dth.tex`.

We require 60 actions, round-to-nearest binary64 arithmetic with gradual
underflow, and compilation without fast-math. We check the rounding mode and
underflow behavior at entry to the kernel. We use the shortcut for finite,
nonnegative computed weights with a finite positive sum, payoffs in
`[-2, 2]`, and `abs(D0) >= 1e-6`.

We store the center `v` of an interval with radius `CERT_RADIUS = 1e-10`.
We prove that this interval encloses both saddle bounds. Its width is
`2e-10`, within the required `1e-6` gate.

## Recurrence residual

We index actions from zero. Let

\[
d=S_0-F,\quad \Delta_m=S_m-S_{m-1},\quad
\widehat d=\operatorname{fl}(d),\quad
\widehat b_m=\operatorname{fl}
 \left(\operatorname{fl}(-\Delta_m)/\widehat d\right).
\]

We compute `r[0] = 1` and

\[
r_k=\operatorname{fl}\left(\sum_{j=0}^{k-1}
                  \widehat b_{k-j}r_j\right).
\]

We use the standard floating-point model with unit roundoff
\(u=2^{-53}\) and \(\gamma_n=nu/(1-nu)\). We bound dot-product error
by \(\gamma_{2k}\sum|a_jb_j|\); this bound permits separate multiply
and add operations or fused multiply-add operations. You can find the
summation analysis in [Higham, The Accuracy of Floating Point Summation](https://nhigham.com/wp-content/uploads/2023/10/high93s.pdf)
and dot-product analysis in [Castaldo, Whaley, and Chronopoulos, Reducing Floating Point Error in Dot Product Using the Superblock Family of Algorithms](https://www.cs.utsa.edu/~atc/pub/J42.pdf).

We apply the dot-product bound to the computed weights as fixed inputs.
We do not need a forward-error bound on the whole triangular solve.
Define the exact sum of those weights as \(W=\sum_{j=0}^{59}r_j\).
We require \(r_j\ge0\), which gives \(W\ge1\) because \(r_0=1\).

For each row difference, we expand its residual as

\[
\begin{aligned}
d r_k+\sum_{j<k}\Delta_{k-j}r_j
={}&\widehat d\left(r_k-\sum_{j<k}\widehat b_{k-j}r_j\right)\\
 &+(d-\widehat d)r_k
   +\sum_{j<k}(\widehat d\widehat b_{k-j}+\Delta_{k-j})r_j.
\end{aligned}
\]

We divide by \(W\). Since \(|d|,|\Delta_m|\le4\), and any subset
of the nonnegative weights sums to at most \(W\), we obtain

\[
\frac{\left|d r_k+\sum_{j<k}\Delta_{k-j}r_j\right|}{W}
\le4\gamma_{118}(1+\gamma_2)+4u+4\gamma_2+500\eta
<6\times10^{-14},
\]

where \(\eta=2^{-1075}\) bounds the absolute rounding error of an
underflowing operation. We include the absolute underflow terms from
coefficient formation and the dot product. We use \(W\ge1\) to bound
those terms after division. We exclude intermediate overflow under
round-to-nearest by checking the recurrence results: we cannot obtain a
finite result from an infinity through these multiply-add operations.

## Saddle bounds and stored value

We choose the exact strategies
\(p_j=r_j/W\) and \(q_j=r_{59-j}/W\).
We identify the residual above with an adjacent-row difference of \(Mq\).
Across 59 differences, we get

\[
\max_i(Mq)_i-\min_i(Mq)_i < 59(6\times10^{-14})=3.54\times10^{-12}.
\]

We use the reversal symmetry of the Toeplitz stage matrix to obtain
\((p^TM)_j=(Mq)_{59-j}\). Thus we can use the minimum and maximum of
\(Mq\) as the lower and upper saddle bounds.

We calculate the last row's exact payoff as

\[
(Mq)_{59}=F+(S_0-F)/W,
\]

because \(q_{59}=r_0/W=1/W\). In the kernel, we evaluate

\[
\widehat W=\operatorname{fl}\left(\sum_{j=59}^{0}r_j\right),\qquad
v=\operatorname{fl}\left(F+\operatorname{fl}(\widehat d/\widehat W)\right).
\]

We bound the positive sum's relative error by \(\gamma_{60}\), plus
its absolute underflow terms. With \(|F|\le2\), \(|d|\le4\), and
\(W\ge1\), the subtraction, division, and final addition give
\(|v-(Mq)_{59}|<4\times10^{-14}\).
We enclose both saddle bounds within \(4\times10^{-12}\)
of `v`. We use the larger radius `1e-10` in the code.

We retain the existing pure-saddle check. We use the full matrix certificate
for clipped weights or inputs outside the shortcut's bounds. We reject
nonfinite weights and overflowed sums before either equalizer path can pass.

We test the shortcut against 90-digit Decimal solutions, including signed
coefficients and a subnormal result. We test clipped mixtures against the
full matrix so that clipping cannot trigger the shortcut.

## Public leap row

The STL kernel uses this proof with separate row-local success and failure
child tables. Python supplies the frozen revival probabilities. We use
`mul_add` for the recurrence and require round-to-nearest with gradual
underflow on the calling thread and on each Rayon worker.

In a window stage, row 61 has payoff `f` in each of the 60 columns. A row
strategy with mass `a` on this row has minimum payoff

\[
 a f + (1-a)\min_c (p' M)_c.
\]

The row player maximizes this expression by choosing an endpoint in `a`.
Thus the window value is `max(f, v60)`. Given a certified enclosure `[MN, MX]`
for the square stage, we obtain `[max(MN, f), max(MX, f)]` for the window.
The function `max(., f)` is 1-Lipschitz, so it preserves the enclosure radius.
We store `max(v60, f)` and use kind 2 if `f >= v60`.

We reserve kind 255 and NaN for classes that fail both kernel rungs. Python
constructs the explicit 60-column matrix and uses HiGHS, then checks both
saddle bounds against the same `1e-6` gate. Python aborts if that check fails.
The artifact contains values. We do not claim a complete L2 solve from the
clock-3420 calibration segment.

## Reduced support candidate (rung 2b)

We use the recurrence as an inverse for adjacent row-payoff differences.
Write `d = s[0] - f`. For a Checker mixture `q`, define

\[
g_i=q_i-\sum_{j>i}b_{j-i}q_j,
\qquad q_j=\sum_{t\ge j}r_{t-j}g_t.
\]

For `i < 59`, we obtain `(Mq)[i] - (Mq)[i+1] = d*g[i]`.
Given guessed row and column supports `P` and `Q`, we set `g[i] = 0`
between adjacent supported rows. We retain unknowns at
`G = {59} union {t: t not in P or t+1 not in P}`. We impose `q[j] = 0`
outside `Q`, `sum(g[a:c]) = 0` between consecutive supported rows `a,c`
with a hole, and `sum(q) = 1`.

We solve the same equations with supports `59-Q` and `59-P`, then reverse
the result to obtain the Dropper mixture `p`. We reject incompatible equation
counts, singular systems, nonfinite weights, and negative weights. We normalize
both candidate mixtures and check all 60 row and column payoffs:

\[
L=\min_j(p^T M)_j,\qquad U=\max_i(Mq)_i.
\]

For feasible mixtures, the minimax inequalities give `L <= v60 <= U`.
We accept the midpoint only when the computed gap is at most `1e-6`, using
the same full-matrix gate as the clipped equalizer and HiGHS paths. We use
the public leap-row bounds above for window stages. We do not use the
nonnegative-recurrence shortcut certificate for reduced supports.

We guess a support from the preceding accepted class in the Checker row.
After a miss, we try at most 64 paired boundary moves. The guesses and the
linear solve propose mixtures; the full payoff check controls acceptance.
Python sends remaining misses to HiGHS and refreshes the row's support.
Python batches rows with reduced dimension above 16 through HiGHS by default.
This size limit controls work and does not change the certificate.

You enable this path by supplying support masks to `sweep_key_rs`, or by using
`SupportFallback` in place of `Fallback`. The full-sweep command leaves
support reuse disabled: measured support-search and refresh costs exceeded the saved LP
work on the sampled H2 batches. We retain kinds 4 and 5 for direct and moved
support certificates, and kind 254 for deferred classes after a row miss.

## Packing fallback

For `d = f-s[0] > 0`, define the upper-triangular Toeplitz matrix
`A[i,j] = (f-s[j-i])/d` when `j >= i`, and zero otherwise. Then
`M = f*11^T-d*A`. We solve the dual pair

\[
\max_{y\ge0}\{\mathbf1^T y:A^T y\le\mathbf1\}
=\min_{x\ge0}\{\mathbf1^T x:Ax\ge\mathbf1\}=z.
\]

The unit diagonal permits a finite feasible covering vector by back
substitution with nonnegative variables. Zero is feasible for the packing
problem. At an optimum, `p=y/z` and `q=x/z` give value `f-d/z`.

The native fallback uses a fresh slack basis and at most 512 simplex pivots.
It rejects `d <= 1e-12` and sends unsupported or uncertified stages to HiGHS.
It clips negative roundoff in candidate weights, normalizes both mixtures,
and checks all original matrix row and column payoffs at the same `1e-6`
gap gate. The formula proposes the candidate; the full matrix certificate
controls acceptance. We retain the constant-row proof for window games.
The Python binding checks finite inputs and the floating-point environment
on the calling thread and on Rayon workers. It records native certificates
and HiGHS solves as separate counts.
