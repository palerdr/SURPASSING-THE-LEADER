"""C recurrence accelerator and Python policy reference for pure DTH."""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import tempfile

import numpy as np

from dth.solver import reconstruct_transition_class_matrix, SADDLE_GAP_TOLERANCE


def recurrence_policy(success: np.ndarray, failed: float):
    """Return a full-matrix certificate, or None so the caller can use its LP."""
    if (
        success.shape != (60,)
        or not np.isfinite(success).all()
        or not np.isfinite(failed)
    ):
        return None
    delta = float(success[0] - failed)
    if abs(delta) < 1e-12:
        return None
    r = np.zeros(60)
    r[0] = 1.0
    with np.errstate(all="ignore"):
        b = (success[:-1] - success[1:]) / delta
        for k in range(1, 60):
            r[k] = np.dot(b[:k], r[k - 1 :: -1])
        if not np.isfinite(r).all():
            return None
        r = np.maximum(r, 0.0)
        total = r.sum()
        if not np.isfinite(total) or total <= 0:
            return None
        drop = r / total
    check = drop[::-1].copy()
    matrix = reconstruct_transition_class_matrix(success, failed)
    lower, upper = float(np.min(drop @ matrix)), float(np.max(matrix @ check))
    if not np.isfinite([lower, upper]).all() or upper - lower > SADDLE_GAP_TOLERANCE:
        return None
    return (lower + upper) / 2, drop, check, max(0.0, upper - lower)


def load_kernel():
    """Compile a content-addressed C library under this project's artifacts."""
    source = Path(__file__).with_suffix(".c")
    compiler = os.environ.get("CC", "cc")
    flags = [
        "-O3",
        "-std=c11",
        "-shared",
        "-fPIC",
        "-ffp-contract=off",
        "-fno-fast-math",
    ]
    identity = subprocess.check_output([compiler, "--version"])
    key = hashlib.sha256(
        source.read_bytes()
        + identity
        + repr(flags).encode()
        + platform.machine().encode()
    ).hexdigest()
    cache = source.parent / "artifacts" / "kernel-cache"
    cache.mkdir(parents=True, exist_ok=True)
    library = cache / f"{key}.so"
    if not library.exists():
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            target = Path(temporary) / "kernel.so"
            subprocess.run(
                [compiler, *flags, str(source), "-o", str(target), "-lm"], check=True
            )
            os.replace(target, library)
    kernel = ctypes.CDLL(str(library))
    array = lambda dtype: np.ctypeslib.ndpointer(
        dtype=dtype, flags=("C_CONTIGUOUS", "ALIGNED")
    )
    kernel.solve_chunk.restype = ctypes.c_int64
    kernel.solve_chunk.argtypes = [
        ctypes.c_int64,
        array(np.int32),
        array(np.int32),
        array(np.float64),
        array(np.uint8),
        ctypes.c_int64,
        array(np.int32),
        array(np.int32),
        array(np.float64),
    ]
    return kernel
