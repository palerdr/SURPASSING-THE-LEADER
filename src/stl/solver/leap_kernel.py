"""We solve batches of simultaneous half-round games from the Dropper's view.

The Dropper maximizes the payoff; the Checker minimizes it. With zero-based
action indices `i` and `j`, we represent the 60-by-60 payoff matrix as
`M[i, j] = s[j - i]` for `j >= i`, and `f` for `j < i`. Here `s[k]` is the
payoff after a successful check with elapsed time `k + 1`; `f` includes the
revival chance after a failed check. We exploit the constant diagonals of
this Toeplitz matrix to avoid constructing it for each state class.

We try pure-action bounds before certifying a candidate mixed strategy from a
recurrence. In the leap window, Baku can add row 61, whose entries are all
`f`; we obtain the extended game's value as `max(v60, f)`. In native mode each
worker certifies the residue itself: a moved kink seed first, then the packing
LP. We certify every candidate against all rows and columns before acceptance.
`src/stl/docs/LEAP_CERTIFICATE.md` holds the enclosure proofs, and
`leap_oracle.py` and `leap_support.py` hold the scalar authorities.

The certificate needs round-to-nearest binary64 arithmetic with gradual
underflow. We check both on the calling thread and in each worker. We write
each fused multiply-add as `llvm.fma` and each minimum and maximum as
`llvm.minnum` and `llvm.maxnum`, and we compile without fast-math, so the
compiler can neither contract nor reorder an operation. Python owns the clock
schedule, the child tables, and the HiGHS fallback.

numba compiles these functions on first use, which takes tens of seconds, and
caches the machine code in `__pycache__/` beside this file. Set
`NUMBA_CACHE_DIR` to move the cache. A change to this file or to the numba
version invalidates it.
"""
import ctypes
import math
import sys

import numpy as np
from llvmlite import ir
from numba import get_num_threads, njit, prange, types
from numba.core import cgutils
from numba.extending import intrinsic

A = 60
N = 60
W = 61
L = 16
GAP = 1e-6
EPS = 1e-10
CHUNK = 1024
PACK_CHUNK = 2048
EDGE_LIMIT = 64
NEIGHBORS = 192
SUPPORT_BITS = (1 << 60)-1
# An attempt limit at or above this bound never binds, as usize::MAX did.
LIMIT_CAP = 1 << 62
NAN = math.nan
INF = math.inf
ERRORS = {1: 'worker floating-point environment is unsupported',
          2: 'nonfinite success child', 3: 'nonfinite failure child',
          4: 'nonfinite failure payoff', 5: 'stage value out of range',
          6: 'unsupported worker floating-point environment',
          7: 'packing value outside utility range'}
# MIN_POSITIVE, 0.5, 1, 1.5*2^-53, and 1+2^-52. We read the operands of the
# rounding and underflow probe from memory, so the compiler cannot fold them.
PROBE = np.array([2.2250738585072014e-308, 0.5, 1.0, 1.5*2.0**-53, 1.0+2.0**-52])


@intrinsic
def _fma(typingctx, a, b, c):
    """We return a*b+c with one rounding."""
    if not (a == types.float64 and b == types.float64 and c == types.float64):
        return None

    def codegen(context, builder, signature, args):
        return builder.fma(*args)
    return types.float64(types.float64, types.float64, types.float64), codegen


def _binary(name):
    def typer(typingctx, a, b):
        if not (a == types.float64 and b == types.float64):
            return None

        def codegen(context, builder, signature, args):
            double = ir.DoubleType()
            fnty = ir.FunctionType(double, [double, double])
            function = builder.module.declare_intrinsic(name, [double], fnty=fnty)
            return builder.call(function, args)
        return types.float64(types.float64, types.float64), codegen
    return intrinsic(typer)


# IEEE minNum and maxNum: a NaN operand yields the other operand.
_fmin = _binary('llvm.minnum')
_fmax = _binary('llvm.maxnum')


@intrinsic
def _bits(typingctx, x):
    if x != types.float64:
        return None

    def codegen(context, builder, signature, args):
        return builder.bitcast(args[0], ir.IntType(64))
    return types.int64(types.float64), codegen


@intrinsic
def _popcount(typingctx, x):
    if x != types.int64:
        return None

    def codegen(context, builder, signature, args):
        return builder.ctpop(args[0])
    return types.int64(types.int64), codegen


@intrinsic
def _bitreverse(typingctx, x):
    if x != types.int64:
        return None

    def codegen(context, builder, signature, args):
        return builder.bitreverse(args[0])
    return types.int64(types.int64), codegen


@intrinsic
def _claim(typingctx, counter):
    """We increment counter[0] atomically and return its previous value."""
    if not (isinstance(counter, types.Array) and counter.dtype == types.int64):
        return None

    def codegen(context, builder, signature, args):
        array = context.make_array(signature.args[0])(context, builder, args[0])
        pointer = cgutils.get_item_pointer(context, builder, signature.args[0], array,
                                           [context.get_constant(types.intp, 0)])
        return builder.atomic_rmw('add', pointer, context.get_constant(types.int64, 1), 'monotonic')
    return types.int64(counter), codegen


@njit(cache=True)
def _key(x):
    """We return an integer whose order is the IEEE total order of x."""
    b = _bits(x)
    return b ^ ((b >> 63) & 0x7FFFFFFFFFFFFFFF)


@njit(cache=True)
def _finite(x):
    return abs(x) < INF


@njit(cache=True)
def _env_ok(probe):
    tiny = probe[0]*probe[1]
    up = probe[2]+probe[3]
    down = -probe[2]-probe[3]
    return tiny != 0.0 and up == probe[4] and down == -probe[4]


# We solve the packing LP and certify the original stage matrix. A solve
# starts from the slack basis, from the last certified basis, or from the
# recurrence basis in which every packing variable is basic. The basis matrix
# `A^T` of that last start is lower triangular Toeplitz, so we write its
# inverse from one series recurrence without a factorization.

@njit(cache=True)
def _packer():
    """We return the tableau, the basic and nonbasic variables, the state, the
    work rows, and the warm-start factorization.

    The state holds [ready, Dropper mask, Checker mask] of the last certified
    basis: its basic Dropper actions and its nonbasic Checker slacks, two
    masks with equal bit counts. The work rows hold the pivot row, a, c, the
    prefix sums of c, p, q, the row payoffs, and the column payoffs.
    """
    return (np.zeros((W, W)), np.zeros(N, np.int64), np.zeros(N, np.int64),
            np.zeros(3, np.int64), np.zeros((8, W)), np.zeros((N+1, 2*N+1)))


@njit(cache=True)
def _coefficients(stage, a):
    f = stage[N]
    d = f-stage[0]
    if d <= 1e-12:
        return False
    for i in range(W):
        if not _finite(stage[i]):
            return False
    for i in range(N):
        a[i] = (f-stage[i])/d
    return True


@njit(cache=True)
def _zero(t):
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            t[i, j] = 0.0


@njit(cache=True)
def _cold(t, basic, nonbasic, a):
    _zero(t)
    for i in range(N):
        basic[i] = N+i
        nonbasic[i] = i
        t[i, N] = 1.0
        t[N, i] = 1.0
        for j in range(i+1):
            t[i, j] = a[i-j]


@njit(cache=True)
def _coefficient(a, row, variable):
    if variable >= N:
        return 1.0 if variable-N == row else 0.0
    if row >= variable:
        return a[row-variable]
    return 0.0


@njit(cache=True)
def _refactor(t, basic, nonbasic, a, factor):
    """We retain the variable identities, then rebuild the basis for new payoffs."""
    work = factor[:N]
    source = factor[N]
    for i in range(N):
        for j in range(N):
            work[i, j] = _coefficient(a, i, basic[j])
            work[i, N+j] = _coefficient(a, i, nonbasic[j])
        work[i, 2*N] = 1.0
    for col in range(N):
        # We take the last row of largest magnitude.
        pivot = col
        best = _key(abs(work[col, col]))
        for i in range(col+1, N):
            k = _key(abs(work[i, col]))
            if k >= best:
                pivot = i
                best = k
        if not _finite(work[pivot, col]) or abs(work[pivot, col]) < 1e-12:
            return False
        if pivot != col:
            for j in range(2*N+1):
                x = work[col, j]
                work[col, j] = work[pivot, j]
                work[pivot, j] = x
        scale = work[col, col]
        for j in range(col, 2*N+1):
            work[col, j] /= scale
        for j in range(2*N+1):
            source[j] = work[col, j]
        for i in range(N):
            if i == col:
                continue
            factor_i = work[i, col]
            for j in range(col, 2*N+1):
                work[i, j] = _fma(-factor_i, source[j], work[i, j])
    _zero(t)
    for i in range(N):
        for j in range(W):
            t[i, j] = work[i, N+j]
    _objective(t, basic, nonbasic)
    return _all_finite(t)


@njit(cache=True)
def _objective(t, basic, nonbasic):
    """We rebuild the objective row for the true costs: one per packing
    variable and zero per slack."""
    for j in range(W):
        t[N, j] = 0.0
    for j in range(N):
        t[N, j] = 1.0 if nonbasic[j] < N else 0.0
    for i in range(N):
        if basic[i] < N:
            for j in range(W):
                t[N, j] -= t[i, j]


@njit(cache=True)
def _all_finite(t):
    for i in range(W):
        for j in range(W):
            if not _finite(t[i, j]):
                return False
    return True


@njit(cache=True)
def _crash(t, basic, nonbasic, a, c, prefix):
    """We make every packing variable basic. The basis matrix is `A^T`, lower
    triangular Toeplitz with first column `a`. Its inverse is lower triangular
    Toeplitz with first column `c`, where `c[0] = 1` and
    `c[k] = -sum_{m=1..k} a[m] c[k-m]`. Basic value `i` is
    `c[0] + ... + c[i]`, and the reduced cost of slack `j` is
    `-(c[0] + ... + c[59-j])`, the negated sum of column `j` of the inverse.
    This start costs one recurrence instead of a 60-by-60 factorization.
    """
    c[0] = 1.0
    for k in range(1, N):
        x = 0.0
        for m in range(1, k+1):
            x = _fma(-a[m], c[k-m], x)
        c[k] = x
    total = 0.0
    for k in range(N):
        total += c[k]
        prefix[k] = total
    _zero(t)
    for i in range(N):
        basic[i] = i
        nonbasic[i] = N+i
        for j in range(i+1):
            t[i, j] = c[i-j]
        t[i, N] = prefix[i]
    _objective(t, basic, nonbasic)
    return _all_finite(t)


@njit(cache=True)
def _pivot(t, basic, nonbasic, row, col, source):
    scale = t[row, col]
    for j in range(W):
        t[row, j] /= scale
    t[row, col] = 1.0/scale
    for j in range(W):
        source[j] = t[row, j]
    for i in range(W):
        if i == row:
            continue
        factor = t[i, col]
        negative = -factor
        # We overwrite column col below, so its update here has no effect.
        for j in range(W):
            t[i, j] = _fma(negative, source[j], t[i, j])
        t[i, col] = -factor/scale
    b = basic[row]
    basic[row] = nonbasic[col]
    nonbasic[col] = b


@njit(cache=True)
def _optimize(t, basic, nonbasic, source):
    """We return the pivot count, or -1 when the simplex stops without an optimum."""
    for iteration in range(512):
        primal = True
        for i in range(N):
            if not (t[i, N] >= -EPS):
                primal = False
                break
        dual = True
        for j in range(N):
            if not (t[N, j] <= EPS):
                dual = False
                break
        if primal and dual:
            return iteration
        row = -1
        col = -1
        if primal:
            # We enter the last largest reduced cost.
            best = 0
            for j in range(N):
                x = t[N, j]
                if x > EPS:
                    k = _key(x)
                    if col < 0 or k >= best:
                        col = j
                        best = k
            if col < 0:
                return -1
            # We leave on the first smallest ratio; the basic index breaks ties.
            best = 0
            for i in range(N):
                y = t[i, col]
                if y > EPS:
                    k = _key(_fmax(t[i, N], 0.0)/y)
                    if row < 0 or k < best or (k == best and basic[i] < basic[row]):
                        row = i
                        best = k
            if row < 0:
                return -1
        elif dual:
            best = 0
            for i in range(N):
                x = t[i, N]
                if x < -EPS:
                    k = _key(x)
                    if row < 0 or k < best:
                        row = i
                        best = k
            if row < 0:
                return -1
            best = 0
            for j in range(N):
                y = t[row, j]
                if y < -EPS:
                    k = _key(t[N, j]/y)
                    if col < 0 or k < best or (k == best and nonbasic[j] < nonbasic[col]):
                        col = j
                        best = k
            if col < 0:
                return -1
        else:
            return -1
        _pivot(t, basic, nonbasic, row, col, source)
    return -1


@njit(cache=True)
def _certificate(t, basic, nonbasic, s, f, work, res):
    """We normalize the basis mixtures and check every row and column payoff."""
    p = work[4]
    q = work[5]
    rows = work[6]
    cols = work[7]
    for i in range(N):
        p[i] = 0.0
        q[i] = 0.0
    for i in range(N):
        if basic[i] < N:
            p[basic[i]] = _fmax(t[i, N], 0.0)
    for j in range(N):
        if nonbasic[j] >= N:
            q[nonbasic[j]-N] = _fmax(-t[N, j], 0.0)
    ps = -0.0
    qs = -0.0
    for i in range(N):
        ps += p[i]
    for i in range(N):
        qs += q[i]
    if not _finite(ps) or not _finite(qs) or ps <= 0.0 or qs <= 0.0:
        return False
    for i in range(N):
        p[i] /= ps
        q[i] /= qs
    prefix = 0.0
    suffix = 0.0
    for i in range(N):
        rows[i] = f*prefix
        prefix += q[i]
        cols[N-1-i] = f*suffix
        suffix += p[N-1-i]
    for i in range(N):
        for j in range(i, N):
            rows[i] = _fma(s[j-i], q[j], rows[i])
            cols[j] = _fma(s[j-i], p[i], cols[j])
    upper = -INF
    for i in range(N):
        upper = _fmax(upper, rows[i])
    lower = INF
    for i in range(N):
        lower = _fmin(lower, cols[i])
    gap = upper-lower
    if not _finite(gap) or not (-1e-12 <= gap and gap <= 1e-6):
        return False
    res[0] = (upper+lower)*0.5
    res[1] = gap
    return True


@njit(cache=True)
def _finish(basic, nonbasic, state, ok, pivots, start, restart, res):
    """We write [value, gap, pivots, start, restart] and record a certified basis."""
    state[0] = 1 if ok else 0
    if ok:
        p = 0
        q = 0
        for i in range(N):
            if basic[i] < N:
                p |= 1 << basic[i]
            if nonbasic[i] >= N:
                q |= 1 << (nonbasic[i]-N)
        state[1] = p
        state[2] = q
        res[2] = float(pivots)
    else:
        res[0] = NAN
        res[1] = NAN
        res[2] = 512.0
    res[3] = start
    res[4] = restart


@njit(cache=True)
def _rejected(state, res):
    state[0] = 0
    res[0] = NAN
    res[1] = NAN
    res[2] = 0.0
    res[3] = 0.0
    res[4] = 0.0


@njit(cache=True)
def _solve_crash(packer, stage, res):
    """We start from the recurrence basis. Its basic values and reduced costs
    can both have the wrong sign. We therefore zero each positive reduced
    cost, run the dual simplex to primal feasibility, restore the true costs,
    and finish with the primal simplex. If this path fails, we restart from
    the slack basis. The full-matrix certificate controls acceptance on both
    paths. Slot 3 of the result is 1 for this start; slot 4 is 1 after a
    slack restart.
    """
    t, basic, nonbasic, state, work, factor = packer
    a = work[1]
    if not _coefficients(stage, a):
        _rejected(state, res)
        return
    f = stage[N]
    pivots = -1
    ok = False
    if _crash(t, basic, nonbasic, a, work[2], work[3]):
        for j in range(N):
            if t[N, j] > 0.0:
                t[N, j] = 0.0
        first = _optimize(t, basic, nonbasic, work[0])
        if first >= 0:
            _objective(t, basic, nonbasic)
            second = _optimize(t, basic, nonbasic, work[0])
            if second >= 0:
                pivots = first+second
                ok = _certificate(t, basic, nonbasic, stage, f, work, res)
    restart = not ok
    if restart:
        _cold(t, basic, nonbasic, a)
        pivots = _optimize(t, basic, nonbasic, work[0])
        ok = pivots >= 0 and _certificate(t, basic, nonbasic, stage, f, work, res)
    _finish(basic, nonbasic, state, ok, pivots, 1.0, 1.0 if restart else 0.0, res)


@njit(cache=True)
def _solve(packer, stage, warm, res):
    """We start from the slack basis, or with warm from the last certified basis.
    We restart a warm start that fails from the slack basis."""
    t, basic, nonbasic, state, work, factor = packer
    a = work[1]
    if not _coefficients(stage, a):
        _rejected(state, res)
        return
    f = stage[N]
    reused = warm and state[0] == 1 and _refactor(t, basic, nonbasic, a, factor)
    if not reused:
        _cold(t, basic, nonbasic, a)
    pivots = _optimize(t, basic, nonbasic, work[0])
    ok = pivots >= 0 and _certificate(t, basic, nonbasic, stage, f, work, res)
    restart = False
    if reused and not ok:
        restart = True
        _cold(t, basic, nonbasic, a)
        pivots = _optimize(t, basic, nonbasic, work[0])
        ok = pivots >= 0 and _certificate(t, basic, nonbasic, stage, f, work, res)
    _finish(basic, nonbasic, state, ok, pivots, 1.0 if reused else 0.0, 1.0 if restart else 0.0, res)


# We guess residue supports and certify the reduced solve.

@njit(cache=True)
def _support_scratch():
    """We return the linear system, eight vectors, the positions, the neighbor
    lists, the tried pairs, and the accepted masks."""
    return (np.zeros((A, A+1)), np.zeros((8, A)), np.zeros(A, np.int64),
            np.zeros((2, NEIGHBORS), np.int64), np.zeros((2, EDGE_LIMIT), np.int64),
            np.zeros(2, np.int64))


@njit(cache=True)
def _reverse_support(mask):
    return (_bitreverse(mask) >> 4) & SUPPORT_BITS


@njit(cache=True)
def _reduced_weights(r, p_mask, q_mask, a, positions, g, cumulative, out):
    """We solve the hole equations in g and reconstruct a feasible column mix.

    We have (Mq)[i] - (Mq)[i+1] = (s[0]-f)*g[i]. Adjacent supported rows
    therefore require g[i]=0; across a hole we require a zero sum. The
    inverse recurrence gives q[j] = sum_{t>=j} r[t-j]*g[t].
    """
    if p_mask == 0 or q_mask == 0 or _popcount(p_mask) != _popcount(q_mask):
        return False
    n = 0
    for t in range(A-1):
        if ((p_mask >> t) & 3) != 3:
            positions[n] = t
            n += 1
    positions[n] = A-1
    n += 1
    for i in range(A):
        for j in range(A+1):
            a[i, j] = 0.0
    equation = 0
    for j in range(A):
        if (q_mask & (1 << j)) == 0:
            for k in range(n):
                t = positions[k]
                if t >= j:
                    a[equation, k] = r[t-j]
            equation += 1
    previous = -1
    for c in range(A):
        if (p_mask & (1 << c)) != 0:
            if previous >= 0 and c > previous+1:
                if equation >= n:
                    return False
                for k in range(n):
                    a[equation, k] = 1.0 if (positions[k] >= previous and positions[k] < c) else 0.0
                equation += 1
            previous = c
    if equation+1 != n:
        return False
    total = 0.0
    for j in range(A):
        total += r[j]
        cumulative[j] = total
    for k in range(n):
        a[equation, k] = cumulative[positions[k]]
    a[equation, n] = 1.0
    # We scale rows and pivot on magnitude. The full game certificate remains
    # the acceptance test if a system has poor conditioning.
    for row in range(n):
        scale = 0.0
        for k in range(n):
            scale = _fmax(scale, abs(a[row, k]))
        if not _finite(scale) or scale == 0.0:
            return False
        for k in range(n+1):
            a[row, k] /= scale
    for col in range(n):
        pivot = col
        for row in range(col+1, n):
            if abs(a[row, col]) > abs(a[pivot, col]):
                pivot = row
        if not _finite(a[pivot, col]) or abs(a[pivot, col]) < 1e-14:
            return False
        if pivot != col:
            for k in range(A+1):
                x = a[col, k]
                a[col, k] = a[pivot, k]
                a[pivot, k] = x
        for row in range(col+1, n):
            factor = a[row, col]/a[col, col]
            for k in range(col+1, n+1):
                a[row, k] = _fma(-factor, a[col, k], a[row, k])
    for j in range(n-1, -1, -1):
        x = a[j, n]
        for k in range(j+1, n):
            x = _fma(-a[j, k], g[k], x)
        g[j] = x/a[j, j]
    for j in range(A):
        out[j] = 0.0
    total = 0.0
    for j in range(A):
        if (q_mask & (1 << j)) == 0:
            continue
        x = 0.0
        for k in range(n):
            if positions[k] >= j:
                x = _fma(r[positions[k]-j], g[k], x)
        if not _finite(x) or x < 0.0:
            return False
        out[j] = x
        total += x
    if not _finite(total) or total <= 0.0:
        return False
    for j in range(A):
        out[j] /= total
    return True


@njit(cache=True)
def _support_attempt(s, f, r, p_mask, q_mask, window, scratch, res):
    a, vectors, positions = scratch[0], scratch[1], scratch[2]
    g, cumulative, q, reverse_p, p, rows, cols = (vectors[0], vectors[1], vectors[2], vectors[3],
                                                  vectors[4], vectors[5], vectors[6])
    if not _reduced_weights(r, p_mask, q_mask, a, positions, g, cumulative, q):
        return False
    if not _reduced_weights(r, _reverse_support(q_mask), _reverse_support(p_mask),
                            a, positions, g, cumulative, reverse_p):
        return False
    for j in range(A):
        p[j] = reverse_p[A-1-j]
    prefix_q = 0.0
    suffix_p = 0.0
    for j in range(A):
        rows[j] = f*prefix_q
        cols[A-1-j] = f*suffix_p
        prefix_q += q[j]
        suffix_p += p[A-1-j]
    if abs(prefix_q-1.0) > 1e-12 or abs(suffix_p-1.0) > 1e-12:
        return False
    # We check every row and column of the original matrix. We use its
    # constant failure triangle to avoid constructing 3,600 matrix cells.
    for i in range(A):
        for j in range(i, A):
            rows[i] = _fma(s[j-i], q[j], rows[i])
            cols[j] = _fma(s[j-i], p[i], cols[j])
    for i in range(A):
        if not _finite(rows[i]) or not _finite(cols[i]):
            return False
    lower = INF
    for i in range(A):
        lower = _fmin(lower, cols[i])
    upper = -INF
    for i in range(A):
        upper = _fmax(upper, rows[i])
    value = (lower+upper)*0.5
    if window:
        lower = _fmax(lower, f)
        upper = _fmax(upper, f)
    if upper-lower < -1e-12 or upper-lower > GAP:
        return False
    res[0] = value
    return True


@njit(cache=True)
def _shifted_isolated_holes(mask, mode):
    result = mask
    for i in range(A):
        if (mask & (1 << i)) != 0:
            continue
        if (i > 0 and (mask & (1 << (i-1))) == 0) or (i+1 < A and (mask & (1 << (i+1))) == 0):
            continue
        if mode == 0:
            direction = -1 if i < A//2 else 1
        elif mode == 1:
            direction = 1 if i < A//2 else -1
        elif mode == 2:
            direction = -1
        else:
            direction = 1
        target = i+direction
        if 0 <= target < A and (mask & (1 << target)) != 0:
            result = (result | (1 << i)) & ~(1 << target)
    return result


@njit(cache=True)
def _support_neighbors(mask, out):
    count = 0
    out[count] = mask
    count += 1
    i = 0
    while i < A:
        if (mask & (1 << i)) != 0:
            i += 1
            continue
        start = i
        while i+1 < A and (mask & (1 << (i+1))) == 0:
            i += 1
        end = i
        if start > 0:
            out[count] = mask ^ (1 << (start-1)) ^ (1 << end)
            count += 1
        if end+1 < A:
            out[count] = mask ^ (1 << start) ^ (1 << (end+1))
            count += 1
        i += 1
    for i in range(A):
        bit = (mask >> i) & 1
        boundary = (i > 0 and ((mask >> (i-1)) & 1) != bit) or (i+1 < A and ((mask >> (i+1)) & 1) != bit)
        if boundary:
            out[count] = mask ^ (1 << i)
            count += 1
        if i+1 < A and ((mask >> (i+1)) & 1) != bit:
            out[count] = mask ^ (3 << i)
            count += 1
    return count


@njit(cache=True)
def _tried(tried, count, p, q):
    for k in range(count):
        if tried[0, k] == p and tried[1, k] == q:
            return True
    return False


@njit(cache=True)
def _solve_support(s, f, r, seed_p, seed_q, window, limit, scratch, res):
    """We try the seed support, then at most 64 paired boundary moves.

    We stop after `limit` support attempts, counting the seed as the first.
    We return kind 4 for the seed and kind 5 for a move, with the value in
    res[0] and the accepted masks in scratch[5], or 0 for no certificate.
    """
    if limit == 0 or seed_p == 0 or seed_q == 0:
        return 0
    for k in range(A):
        if not _finite(r[k]):
            return 0
    masks = scratch[5]
    if _support_attempt(s, f, r, seed_p, seed_q, window, scratch, res):
        masks[0] = seed_p
        masks[1] = seed_q
        return 4
    tried = scratch[4]
    count = 0
    for mode in range(4):
        pp = _shifted_isolated_holes(seed_p, mode)
        qq = _shifted_isolated_holes(seed_q, mode)
        if not (pp == seed_p and qq == seed_q) and not _tried(tried, count, pp, qq):
            if count+1 >= limit:
                return 0
            tried[0, count] = pp
            tried[1, count] = qq
            count += 1
            if _support_attempt(s, f, r, pp, qq, window, scratch, res):
                masks[0] = pp
                masks[1] = qq
                return 5
    for i in range(A-1):
        if (((seed_p >> i) ^ (seed_p >> (i+1))) & 1) != 0 and \
                (((seed_q >> (A-2-i)) ^ (seed_q >> (A-1-i))) & 1) != 0:
            pp = seed_p ^ (3 << i)
            qq = seed_q ^ (3 << (A-2-i))
            if _tried(tried, count, pp, qq):
                continue
            if count+1 >= limit:
                return 0
            tried[0, count] = pp
            tried[1, count] = qq
            count += 1
            if _support_attempt(s, f, r, pp, qq, window, scratch, res):
                masks[0] = pp
                masks[1] = qq
                return 5
    neighbors = scratch[3]
    p_count = _support_neighbors(seed_p, neighbors[0])
    q_count = _support_neighbors(seed_q, neighbors[1])
    for a in range(p_count):
        pp = neighbors[0, a]
        for b in range(q_count):
            qq = neighbors[1, b]
            if (pp == seed_p and qq == seed_q) or _popcount(pp) != _popcount(qq) \
                    or _tried(tried, count, pp, qq):
                continue
            if count == EDGE_LIMIT or count+1 >= limit:
                return 0
            tried[0, count] = pp
            tried[1, count] = qq
            count += 1
            if _support_attempt(s, f, r, pp, qq, window, scratch, res):
                masks[0] = pp
                masks[1] = qq
                return 5
    return 0


@njit(cache=True)
def _kink_index(s):
    """We return the index k of the most negative step s[k] - s[k-1], the
    first on a tie, or -1 when no step is negative.

    A nondecreasing s with f > s[0] gives nonnegative recurrence weights, so
    a residue stage needs at least one negative step.
    """
    best = 0.0
    index = -1
    for k in range(1, A):
        step = s[k]-s[k-1]
        if step < best:
            best = step
            index = k
    return index


@njit(cache=True)
def _move_holes(mask, near, old, delta):
    """We move each hole with the nearer of the anchors near and 59-old.

    A hole at the first anchor moves by delta and a hole at the second by
    -delta. A hole that starts at action 0 keeps its place. We clip moved
    holes to actions 0..59.
    """
    far = A-1-old
    out = SUPPORT_BITS
    i = 0
    while i < A:
        if ((mask >> i) & 1) == 1:
            i += 1
            continue
        start = i
        while i+1 < A and ((mask >> (i+1)) & 1) == 0:
            i += 1
        end = i
        if start == 0:
            shift = 0
        elif abs(end-near) <= abs(end-far):
            shift = delta
        else:
            shift = -delta
        low = max(start+shift, 0)
        high = min(end+shift, A-1)
        for x in range(low, high+1):
            out &= ~(1 << x)
        i += 1
    return out


@njit(cache=True)
def _kink_seed(p, q, old, new):
    """We move a certified support from kink old to kink new; -1 marks no kink."""
    if old >= 0 and new >= 0 and old != new:
        delta = new-old
        return _move_holes(p, old, old, delta), _move_holes(q, old-1, old, delta)
    return p, q


@njit(cache=True)
def _solve_residue(stage, r, window, crash, kink_attempts, seed, packer, scratch, res, packing):
    """We solve one residue stage in the worker.

    We first try the moved seed support and at most `kink_attempts` supports.
    We then run the packing LP. We return kind 3, 4, or 5 with the square-stage
    value in res[0], or 0 for HiGHS. Each accepted support or LP basis becomes
    the next seed: seed holds [present, Dropper mask, Checker mask, kink].
    """
    f = stage[A]
    kink = _kink_index(stage)
    if seed[0] != 0:
        guess_p, guess_q = _kink_seed(seed[1], seed[2], seed[3], kink)
        kind = _solve_support(stage, f, r, guess_p, guess_q, window, kink_attempts, scratch, res)
        if kind != 0:
            seed[1] = scratch[5][0]
            seed[2] = scratch[5][1]
            seed[3] = kink
            return kind
    if crash:
        _solve_crash(packer, stage, packing)
    else:
        _solve(packer, stage, False, packing)
    if not _finite(packing[0]):
        return 0
    state = packer[3]
    seed[0] = 1
    seed[1] = state[1]
    seed[2] = state[2]
    seed[3] = kink
    res[0] = packing[0]
    return 3


@njit(cache=True)
def _sweep_chunk(chunk, pcs, pds, st, sw, ft, fw, sc, fc, rev, window, native, crash,
                 kink_attempts, has_seeds, seeds, support_out, stop_miss, out, kind, probe):
    """We solve one 1024-class chunk and return (failed, error code)."""
    failed = 0
    # We check each worker because it has its own floating-point state.
    if not _env_ok(probe):
        return failed, 1
    n = pcs.shape[0]
    first = chunk*CHUNK
    size = min(CHUNK, n-first)
    # We index scratch as [action, lane] to keep the lane loop contiguous.
    # We reuse q for recurrence coefficients and Checker weights.
    s = np.zeros((A, L))
    r = np.zeros((A, L))
    q = np.zeros((A, L))
    fl = np.zeros(L)
    d = np.zeros(L)
    mn = np.zeros(L)
    mx = np.zeros(L)
    total = np.zeros(L)
    acc = np.zeros(L)
    bounded = np.zeros(L, np.bool_)
    nonnegative = np.zeros(L, np.bool_)
    stage = np.zeros(A+1)
    rl = np.zeros(A)
    res = np.zeros(5)
    packing = np.zeros(5)
    # In native mode each chunk owns one packing tableau and the certified
    # support of its preceding residue class, with that class's kink index.
    # Seeds therefore restart at each chunk.
    seed = np.zeros(4, np.int64)
    packer = _packer()
    scratch = _support_scratch()
    previous_pc = -1
    blocked_pc = -1
    previous_p = 0
    previous_q = 0
    for base in range(0, size, L):
        m = min(L, size-base)
        for lane in range(L):
            # We fill spare lanes with the first class in this batch and write
            # results for the m active lanes after certification.
            i = first+base+(lane if lane < m else 0)
            pc = np.int64(pcs[i])
            pd = np.int64(pds[i])
            lo = INF
            hi = -INF
            for k in range(A):
                x = -st[pd*sw+np.int64(sc[pc*A+k])]
                if not _finite(x):
                    return failed, 2
                s[k, lane] = x
                lo = _fmin(lo, x)
                hi = _fmax(hi, x)
            child = ft[pd*fw+np.int64(fc[pc])]
            if not _finite(child):
                return failed, 3
            # We negate the revived child's value after the role swap. The
            # current Dropper receives +1 if the Checker dies.
            fv = rev[pc]*-child+(1.0-rev[pc])
            if not _finite(fv):
                return failed, 4
            fl[lane] = fv
            d[lane] = s[0, lane]-fv
            # The Dropper can guarantee mn, and the Checker can enforce mx.
            mn[lane] = _fmax(lo, _fmin(fv, s[0, lane]))
            mx[lane] = _fmin(hi, _fmax(fv, s[0, lane]))
            bounded[lane] = lo >= -2.0 and hi <= 2.0 and abs(fv) <= 2.0
        # We derive b[k] from adjacent success-payoff differences. We
        # substitute 1 for a zero denominator while filling scratch; the pure
        # test or the d-size guard controls acceptance.
        for k in range(1, A):
            for lane in range(L):
                dd = d[lane]
                q[k, lane] = (s[k-1, lane]-s[k, lane])/(dd if dd != 0.0 else 1.0)
        # We solve r[k] = sum_{j < k} b[k-j] * r[j], with r[0] = 1. Setting
        # adjacent row payoffs equal gives this relation; normalization
        # removes the scale that r[0] fixes.
        for lane in range(L):
            r[0, lane] = 1.0
        for k in range(1, A):
            for lane in range(L):
                acc[lane] = 0.0
            for j in range(k):
                for lane in range(L):
                    acc[lane] = _fma(q[k-j, lane], r[j, lane], acc[lane])
            for lane in range(L):
                r[k, lane] = acc[lane]
        # We reverse and clip the weights to form a Checker candidate. We
        # record any negative weight because clipping requires the full payoff
        # check; nonfinite weights must invalidate the sum.
        for lane in range(L):
            total[lane] = 0.0
            nonnegative[lane] = True
        for k in range(A):
            for lane in range(L):
                x = r[A-1-k, lane]
                nonnegative[lane] = nonnegative[lane] and x >= 0.0
                q[k, lane] = _fmax(x, 0.0) if _finite(x) else NAN
                total[lane] += q[k, lane]
        for lane in range(m):
            gi = first+base+lane
            # We retain these sentinels if no certificate passes.
            out[gi] = NAN
            kind[gi] = 255
            current_pc = np.int64(pcs[gi])
            if stop_miss and current_pc == blocked_pc:
                kind[gi] = 254
                failed += 1
                continue
            fv = fl[lane]
            lower = mn[lane]
            upper = mx[lane]
            if window:
                # We lift both bounds through max(., f). This preserves the
                # enclosure and can certify the constant leap row.
                lower = _fmax(lower, fv)
                upper = _fmax(upper, fv)
            v = 0.0
            rung = 0
            residue = False
            if upper-lower <= GAP:
                v = 0.5*(mn[lane]+mx[lane])
                rung = 0
            else:
                rung = 1
                sm = total[lane]
                if abs(d[lane]) < 1e-12 or not _finite(sm) or sm <= 0.0:
                    if not native:
                        blocked_pc = current_pc
                        failed += 1
                        continue
                    residue = True
                elif nonnegative[lane] and bounded[lane] and abs(d[lane]) >= GAP:
                    # Under the certificate's hypotheses, we enclose both saddle
                    # bounds within 1e-10 of this weight-sum value.
                    v = fv+d[lane]/sm
                else:
                    # We normalize the clipped candidate and evaluate Mq. With
                    # the Dropper's mix as reverse(q), (p^T M)[j] = (Mq)[59-j],
                    # so min(Mq) and max(Mq) give both saddle bounds.
                    for k in range(A):
                        q[k, lane] /= sm
                    cum = 0.0
                    lower = INF
                    upper = -INF
                    ok = True
                    for k in range(A):
                        # We sum the success suffix of row k and add f times
                        # the Checker mass before its diagonal.
                        x = 0.0
                        for j in range(A-k):
                            x = _fma(s[j, lane], q[k+j, lane], x)
                        x = _fma(fv, cum, x)
                        ok = ok and _finite(x)
                        lower = _fmin(lower, x)
                        upper = _fmax(upper, x)
                        cum += q[k, lane]
                    v = 0.5*(lower+upper)
                    if window:
                        lower = _fmax(lower, fv)
                        upper = _fmax(upper, fv)
                    if not ok or abs(cum-1.0) > 1e-12 or upper-lower > GAP:
                        reduced = 0
                        if has_seeds:
                            if current_pc != previous_pc:
                                previous_pc = current_pc
                                previous_p = seeds[current_pc, 0]
                                previous_q = seeds[current_pc, 1]
                            for k in range(A):
                                stage[k] = s[k, lane]
                                rl[k] = r[k, lane]
                            stage[A] = fv
                            reduced = _solve_support(stage, fv, rl, previous_p, previous_q, window,
                                                     LIMIT_CAP, scratch, res)
                        if reduced != 0:
                            v = res[0]
                            rung = reduced
                            previous_p = scratch[5][0]
                            previous_q = scratch[5][1]
                            support_out[gi, 0] = previous_p
                            support_out[gi, 1] = previous_q
                        elif native:
                            residue = True
                        else:
                            blocked_pc = current_pc
                            failed += 1
                            continue
            if residue:
                for k in range(A):
                    stage[k] = s[k, lane]
                    rl[k] = r[k, lane]
                stage[A] = fv
                accepted = _solve_residue(stage, rl, window, crash, kink_attempts, seed,
                                          packer, scratch, res, packing)
                if accepted == 0:
                    failed += 1
                    continue
                v = res[0]
                rung = accepted
            if window:
                # We choose between the square-stage value and the constant
                # leap row; ties belong to kind 2. Native mode keeps its
                # residue kinds for counting.
                if fv >= v and not (native and rung >= 3):
                    rung = 2
                v = _fmax(v, fv)
            if not _finite(v) or abs(v) > 1.0+1e-9:
                return failed, 5
            out[gi] = v
            kind[gi] = rung
    return failed, 0


@njit(parallel=True, cache=True)
def _sweep_parallel(pcs, pds, st, sw, ft, fw, sc, fc, rev, window, native, crash, kink_attempts,
                    has_seeds, seeds, support_out, stop_miss, out, kind, probe, failed, errors,
                    counter, workers):
    # Each worker claims the next chunk, so a slow chunk does not hold back
    # the others; prange alone splits chunks into fixed contiguous blocks.
    chunks = failed.shape[0]
    for _ in prange(workers):
        chunk = _claim(counter)
        while chunk < chunks:
            count, error = _sweep_chunk(chunk, pcs, pds, st, sw, ft, fw, sc, fc, rev, window,
                                        native, crash, kink_attempts, has_seeds, seeds,
                                        support_out, stop_miss, out, kind, probe)
            failed[chunk] = count
            errors[chunk] = error
            chunk = _claim(counter)


@njit(cache=True)
def _packing_chunk(chunk, s, f, window, crash, out, kind, support_out, probe):
    if not _env_ok(probe):
        return 0, 6
    n = f.shape[0]
    first = chunk*PACK_CHUNK
    size = min(PACK_CHUNK, n-first)
    packer = _packer()
    stage = np.zeros(A+1)
    res = np.zeros(5)
    failed = 0
    for i in range(size):
        j = first+i
        for k in range(A):
            stage[k] = s[j, k]
        stage[A] = f[j]
        if crash:
            _solve_crash(packer, stage, res)
        else:
            _solve(packer, stage, False, res)
        out[j] = NAN
        kind[j] = 255
        if _finite(res[0]):
            value = _fmax(res[0], f[j]) if window else res[0]
            if abs(value) > 1.0+1e-9:
                return failed, 7
            out[j] = value
            kind[j] = 3
            support_out[j, 0] = packer[3][1]
            support_out[j, 1] = packer[3][2]
        else:
            failed += 1
    return failed, 0


@njit(parallel=True, cache=True)
def _packing_parallel(s, f, window, crash, out, kind, support_out, probe, failed, errors,
                      counter, workers):
    chunks = failed.shape[0]
    for _ in prange(workers):
        chunk = _claim(counter)
        while chunk < chunks:
            count, error = _packing_chunk(chunk, s, f, window, crash, out, kind, support_out, probe)
            failed[chunk] = count
            errors[chunk] = error
            chunk = _claim(counter)


@njit(cache=True)
def _runs_chunk(chunk, s, f, groups, warm, out):
    """We solve one 2048-stage chunk; a new group value clears the warm basis."""
    n = f.shape[0]
    first = chunk*PACK_CHUNK
    size = min(PACK_CHUNK, n-first)
    packer = _packer()
    stage = np.zeros(A+1)
    res = np.zeros(5)
    group = NAN
    for i in range(size):
        j = first+i
        for k in range(A):
            stage[k] = s[j, k]
        stage[A] = f[j]
        if groups[j] != group:
            packer[3][0] = 0
            group = groups[j]
        _solve(packer, stage, warm, res)
        for k in range(5):
            out[j, k] = res[k]


@njit(parallel=True, cache=True)
def _runs_parallel(s, f, groups, warm, out, chunks, counter, workers):
    for _ in prange(workers):
        chunk = _claim(counter)
        while chunk < chunks:
            _runs_chunk(chunk, s, f, groups, warm, out)
            chunk = _claim(counter)


def _fegetround():
    try:
        library = ctypes.cdll.ucrtbase if sys.platform == 'win32' else ctypes.CDLL(None)
        function = library.fegetround
    except (AttributeError, OSError, TypeError):
        return None
    function.restype = ctypes.c_int
    function.argtypes = []
    return function


_FEGETROUND = _fegetround()


def environment_ok():
    """We check round-to-nearest and gradual underflow on the calling thread."""
    return (_FEGETROUND is None or _FEGETROUND() == 0) and bool(_env_ok(PROBE))


def _array(x, name, dtype, ndim, writable=False):
    if not isinstance(x, np.ndarray) or x.dtype != np.dtype(dtype) or x.ndim != ndim:
        raise TypeError(f"argument '{name}': expected a {ndim}-d {np.dtype(dtype)} array")
    if writable and not x.flags.writeable:
        raise TypeError(f"argument '{name}': array is not writeable")
    return x


def _flat(x):
    if not x.flags.c_contiguous:
        raise TypeError('The given array is not contiguous')
    return x.reshape(-1)


def _input(x):
    # numba compiles one version of a function for each array writability.
    # We pass every input read-only, so writable arrays and the read-only
    # memmaps of the child tables share one compiled kernel.
    x = x.view()
    x.flags.writeable = False
    return x


def _count(x, name):
    x = int(x)
    if x < 0:
        raise OverflowError(f"argument '{name}': can't convert negative int to unsigned")
    return x


def _raise(errors):
    bad = np.flatnonzero(errors)
    if len(bad):
        raise ValueError(ERRORS[int(errors[bad[0]])])


def sweep_batch(pcs, pds, succ_table, fail_table, succ_col, fail_col, rev, window,
                saddle_tolerance, out, kind, supports=None, support_out=None,
                stop_on_support_miss=False, native=False, crash=False, kink_attempts=0):
    """We certify one value per input class and return the count that needs HiGHS.

    You supply checker ids in `pcs` and dropper ids in `pds`, in matching
    order. After the role swap, the old Dropper is the child Checker, so we
    read row `pds[i]` from both child tables and negate its values. Python
    groups classes that share these tables before calling this function.

    You map each checker profile to 60 success columns through `succ_col` and
    one failure column through `fail_col`. The failure table can use
    compressed reset-profile columns; we use the supplied column map without
    decoding it. You reserve the last column of each table for WIN, fill it
    with -1.0, and map terminal children to that column. We reject nonfinite
    child values that we read; you can leave NaN in unreachable cells.

    You set `window` when Baku can drop at 61. A turn duration of 61 alone
    does not grant the extra row: Hal still has 60 actions in H1[59]. You must
    supply `saddle_tolerance = 1e-6`; we do not permit a weaker gate.

    We write `out` and `kind` in input order. We use kind 0 for the pure test,
    kind 1 for an equalizer certificate, and kind 2 when the leap row wins or
    ties the square-stage value. We leave NaN and kind 255 for uncertified
    classes. Python must send those classes to HiGHS before storing the key.
    With support seeds, kind 4 denotes a reduced support certificate and kind
    5 a certificate after a paired edge move.

    You supply `supports` as `[profile, 2]` masks over zero-based actions 0..59
    and `support_out` as `[class, 2]`. Sort residue classes by Checker and then
    Dropper to reuse each row's preceding accepted support. Each chunk starts
    from the supplied seed. Set `stop_on_support_miss` to defer later classes
    in that row with kind 254. Python then solves the first miss with HiGHS,
    updates its seed, and retries. The returned count includes failed and
    deferred classes. Support masks name the square game's actions; the
    constant-row proof lifts a window certificate.

    Set `native` to solve the residue inside each worker. We then run the
    packing LP on each class that fails the two rungs and record kind 3. Set
    `crash` to start that LP from the recurrence basis. Set `kink_attempts`
    above zero to try a support guess before the LP: we move the certified
    support of the preceding residue class in the same 1024-class chunk with
    the success-payoff kink, then try at most `kink_attempts` supports. The
    moved guess records kind 4 and a later edge move kind 5. In this mode we
    keep kinds 3, 4, and 5 in window stages, so Python can count the residue;
    kind 255 then marks only the classes that need HiGHS. The native mode
    excludes `supports`, `support_out`, and `stop_on_support_miss`.

    We raise ValueError for incompatible shapes or layouts, invalid indices or
    probabilities, unsupported floating-point environments, nonfinite gathered
    payoffs, and accepted values outside the utility range. After an error,
    discard both output buffers: workers may have written some entries.
    """
    pcs = _array(pcs, 'pcs', np.int32, 1)
    pds = _array(pds, 'pds', np.int32, 1)
    succ_table = _array(succ_table, 'succ_table', np.float64, 2)
    fail_table = _array(fail_table, 'fail_table', np.float64, 2)
    succ_col = _array(succ_col, 'succ_col', np.int32, 2)
    fail_col = _array(fail_col, 'fail_col', np.int32, 1)
    rev = _array(rev, 'rev', np.float64, 1)
    out = _array(out, 'out', np.float64, 1, True)
    kind = _array(kind, 'kind', np.uint8, 1, True)
    if supports is not None:
        supports = _array(supports, 'supports', np.uint64, 2)
    if support_out is not None:
        support_out = _array(support_out, 'support_out', np.uint64, 2, True)
    window = bool(window); native = bool(native); crash = bool(crash)
    stop_on_support_miss = bool(stop_on_support_miss)
    kink_attempts = _count(kink_attempts, 'kink_attempts')
    if float(saddle_tolerance) != GAP:
        raise ValueError('saddle tolerance must be 1e-6')
    if (crash or kink_attempts > 0) and not native:
        raise ValueError('crash and kink seeds require the native residue mode')
    if native and (supports is not None or support_out is not None or stop_on_support_miss):
        raise ValueError('native residue mode excludes support seeds')
    if not environment_ok():
        raise ValueError('unsupported floating-point environment')
    if not (succ_table.flags.c_contiguous and fail_table.flags.c_contiguous and succ_col.flags.c_contiguous):
        raise ValueError('tables must be C contiguous')
    sr, sw = succ_table.shape
    fr, fw = fail_table.shape
    profiles = succ_col.shape[0]
    pcs, pds, sc, fc, rev, st, ft = (_input(_flat(x)) for x in
                                     (pcs, pds, succ_col, fail_col, rev, succ_table, fail_table))
    out, kind = _flat(out), _flat(kind)
    if (succ_col.shape[1] != A or len(fc) != profiles or len(rev) != profiles
            or len(pds) != len(pcs) or len(out) != len(pcs) or len(kind) != len(pcs)):
        raise ValueError('incompatible input shapes')
    # We validate the column maps at entry. With the profile and row checks,
    # this bounds every child gather in the workers.
    if ((sc < 0).any() or (sc >= sw).any() or (fc < 0).any() or (fc >= fw).any()
            or (pcs < 0).any() or (pcs >= profiles).any() or (pds < 0).any()
            or (pds >= sr).any() or (pds >= fr).any()
            or not (np.isfinite(rev) & (rev >= 0.0) & (rev <= 1.0)).all()):
        raise ValueError('profile, column, or revival probability out of range')
    if supports is not None:
        if supports.shape != (profiles, 2):
            raise ValueError('support seed shape mismatch')
        seeds = _flat(supports).reshape(profiles, 2)
        if (seeds & np.uint64(~SUPPORT_BITS & (2**64-1))).any():
            raise ValueError('support bit out of range')
        seeds = _input(seeds.view(np.int64))
    else:
        seeds = _input(np.zeros((0, 2), np.int64))
    if support_out is not None:
        if support_out.shape != (len(pcs), 2):
            raise ValueError('support output shape mismatch')
        masks = _flat(support_out).reshape(len(pcs), 2)
        masks.fill(0)
        masks = masks.view(np.int64)
    else:
        masks = np.zeros((0, 2), np.int64)
    if (supports is None) != (support_out is None):
        raise ValueError('support seeds and output must be supplied together')
    if stop_on_support_miss and (supports is None or (np.diff(pcs) < 0).any()):
        raise ValueError('support deferral requires seeds and Checker-row order')
    chunks = (len(pcs)+CHUNK-1)//CHUNK
    failed = np.zeros(chunks, np.int64); errors = np.zeros(chunks, np.int64)
    _sweep_parallel(pcs, pds, st, np.int64(sw), ft, np.int64(fw), sc, fc, rev, window, native,
                    crash, np.int64(min(kink_attempts, LIMIT_CAP)), supports is not None, seeds,
                    masks, stop_on_support_miss, out, kind, PROBE, failed, errors,
                    np.zeros(1, np.int64), np.int64(get_num_threads()))
    if errors.any():
        masks.fill(0)
        _raise(errors)
    return int(failed.sum())


def solve_packing(s, f, window, out, kind, crash=False, support_out=None):
    """We certify the residue through a packing LP, with a fresh basis per stage.

    We return the count of failed stages, and we leave NaN and kind 255 for
    unsupported or uncertified stages. Python sends those stages to HiGHS.
    Kind 3 records a full-matrix certificate. Set `crash` to start each stage
    from the recurrence basis; we restart from the slack basis when that path
    fails. You can pass `support_out` with shape `[stage, 2]` to receive each
    certified basis as Dropper and Checker masks, the seed that native kernel
    mode keeps; we write zeros for failed stages.
    """
    s = _array(s, 's', np.float64, 2)
    f = _array(f, 'f', np.float64, 1)
    out = _array(out, 'out', np.float64, 1, True)
    kind = _array(kind, 'kind', np.uint8, 1, True)
    if support_out is not None:
        support_out = _array(support_out, 'support_out', np.uint64, 2, True)
    if s.shape != (len(f), A) or len(out) != len(f) or len(kind) != len(f):
        raise ValueError('packing shape mismatch')
    if support_out is not None:
        if support_out.shape != (len(f), 2):
            raise ValueError('packing support shape mismatch')
        masks = _flat(support_out).reshape(len(f), 2)
        masks.fill(0)
    s = _input(_flat(s).reshape(len(f), A))
    f, out, kind = _input(_flat(f)), _flat(out), _flat(kind)
    if not (np.isfinite(s).all() and np.isfinite(f).all()):
        raise ValueError('packing needs finite payoffs')
    if not environment_ok():
        raise ValueError('unsupported floating-point environment')
    chunks = (len(f)+PACK_CHUNK-1)//PACK_CHUNK
    failed = np.zeros(chunks, np.int64); errors = np.zeros(chunks, np.int64)
    bases = np.zeros((len(f), 2), np.int64)
    _packing_parallel(s, f, bool(window), bool(crash), out, kind, bases, PROBE, failed, errors,
                      np.zeros(1, np.int64), np.int64(get_num_threads()))
    _raise(errors)
    if support_out is not None:
        masks[:] = bases.view(np.uint64)
    return int(failed.sum())


def solve_packing_runs(s, f, groups, warm):
    """We solve the packing LP of each stage for the leap benchmarks.

    With `warm`, a stage starts from the last certified basis of its
    2048-stage chunk while `groups` keeps its value, and a failed warm start
    restarts from the slack basis. We return one row per stage: the value, the
    saddle gap, the pivot count, 1 for a warm start, and 1 for a restart. A
    rejected or uncertified stage has a NaN value.
    """
    s = np.ascontiguousarray(s, np.float64); f = np.ascontiguousarray(f, np.float64)
    groups = np.ascontiguousarray(groups, np.float64)
    if s.shape != (len(f), A) or groups.shape != f.shape:
        raise ValueError('packing shape mismatch')
    if not environment_ok():
        raise ValueError('unsupported floating-point environment')
    out = np.empty((len(f), 5))
    _runs_parallel(s, f, groups, bool(warm), out, np.int64((len(f)+PACK_CHUNK-1)//PACK_CHUNK),
                   np.zeros(1, np.int64), np.int64(get_num_threads()))
    return out


def kink_index(s):
    """We return the kernel's kink index for parity tests against `leap_support.py`."""
    s = np.asarray(s)
    if s.dtype != np.float64 or s.ndim != 1:
        raise TypeError("argument 's': expected a 1-d float64 array")
    if len(s) != A:
        raise ValueError('kink index needs 60 payoffs')
    if not np.isfinite(s).all():
        raise ValueError('kink index needs finite payoffs')
    k = int(_kink_index(np.ascontiguousarray(s)))
    return None if k < 0 else k


def kink_seed(p, q, old, new):
    """We return the kernel's kink seed move for parity tests against `leap_support.py`."""
    p = _count(p, 'p'); q = _count(q, 'q')
    if (p | q) & ~SUPPORT_BITS or (old is not None and not 0 < old < A) \
            or (new is not None and not 0 < new < A):
        raise ValueError('kink seed needs actions 0..59 and kinks 1..59')
    moved_p, moved_q = _kink_seed(np.int64(p), np.int64(q), np.int64(-1 if old is None else old),
                                  np.int64(-1 if new is None else new))
    return int(moved_p), int(moved_q)
