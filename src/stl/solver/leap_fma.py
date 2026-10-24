"""We compute a*b+c with one rounding, as math.fma does, on Python 3.12.

math.fma arrived in Python 3.13. The scalar authorities `leap_oracle.py` and
`leap_support.py` call `fma` from this module instead. It emits `llvm.fma`, the
instruction that the kernel's `_fma` emits, so it returns the bits that
math.fma returns. It also raises where math.fma raises: ValueError when the
result is NaN and no operand is, and OverflowError when finite operands give
an infinite result. The authorities import nothing from `leap_kernel.py`, the
code that they check, so this module keeps its own copy of the intrinsic.
"""
import math

from numba import njit, types
from numba.extending import intrinsic


@intrinsic
def _fma(typingctx, a, b, c):
    """We return a*b+c with one rounding."""
    if not (a == types.float64 and b == types.float64 and c == types.float64):
        return None

    def codegen(context, builder, signature, args):
        return builder.fma(*args)
    return types.float64(types.float64, types.float64, types.float64), codegen


@njit(cache=True)
def _fused(a, b, c):
    return _fma(a, b, c)


def fma(a, b, c):
    """We return a*b+c with one rounding and raise the errors of math.fma."""
    x, y, z = float(a), float(b), float(c)
    r = _fused(x, y, z)
    if math.isnan(r):
        if not (math.isnan(x) or math.isnan(y) or math.isnan(z)):
            raise ValueError('invalid operation in fma')
    elif math.isinf(r) and math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
        raise OverflowError('overflow in fma')
    return r
