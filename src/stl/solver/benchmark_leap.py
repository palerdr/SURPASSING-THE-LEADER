"""We benchmark leap candidates without changing sweep tables or source identity."""
from dataclasses import dataclass
from pathlib import Path
import json
import subprocess
import time

import highspy
import numpy as np

from stl.solver.leap_oracle import stage_matrix
from stl.solver.leap_profiles import profiles
from stl.solver.leap_build import child_key, is_window

ROOT = Path(__file__).resolve().parents[3]


@dataclass
class PackingResult:
    value: float
    gap: float
    p: np.ndarray
    q: np.ndarray
    pivots: int


class PackingLP:
    """We maximize sum(y) under A.T y <= 1, then certify the original game."""
    def __init__(self, warm=True):
        self.warm = warm
        self.basis = None
        self.highs = highspy.Highs()
        for name, value in [('output_flag', False), ('threads', 1), ('solver', 'simplex'),
                            ('presolve', 'off'), ('primal_feasibility_tolerance', 1e-9),
                            ('dual_feasibility_tolerance', 1e-9)]:
            if self.highs.setOptionValue(name, value) != highspy.HighsStatus.kOk:
                raise RuntimeError(f'HiGHS rejected {name}')
        lp = self.lp = highspy.HighsLp()
        lp.num_col_ = lp.num_row_ = 60
        lp.col_cost_ = -np.ones(60)
        lp.col_lower_ = np.zeros(60); lp.col_upper_ = np.full(60, np.inf)
        lp.row_lower_ = np.full(60, -np.inf); lp.row_upper_ = np.ones(60)
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = np.r_[0, np.cumsum(np.arange(60, 0, -1))].astype(np.int32)
        lp.a_matrix_.index_ = np.concatenate([np.arange(j, 60, dtype=np.int32) for j in range(60)])

    def solve(self, s, f):
        s = np.asarray(s, dtype=float)
        if s.shape != (60,) or not np.isfinite(s).all() or not np.isfinite(f):
            raise ValueError('stage needs finite payoffs')
        d = f-s[0]
        if d <= 1e-12:
            raise ValueError('packing requires a positive diagonal')
        a = (f-s)/d
        self.lp.a_matrix_.value_ = np.concatenate([a[:60-j] for j in range(60)])
        if self.highs.passModel(self.lp) == highspy.HighsStatus.kError:
            raise RuntimeError('HiGHS rejected packing model')
        if self.basis is not None and self.warm:
            self.highs.setBasis(self.basis)
        self.highs.run()
        if self.highs.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            self.basis = None
            raise RuntimeError('packing LP status failed')
        solution = self.highs.getSolution()
        p = np.maximum(solution.col_value, 0.)
        q = np.maximum(-np.asarray(solution.row_dual), 0.)
        if min(p.sum(), q.sum()) <= 0:
            raise RuntimeError('packing LP has zero mass')
        p /= p.sum(); q /= q.sum()
        matrix = stage_matrix(s, f, False)
        lower = float((p @ matrix).min()); upper = float((matrix @ q).max())
        if not np.isfinite(upper-lower) or not -1e-12 <= upper-lower <= 1e-6:
            self.basis = None
            raise RuntimeError('packing failed the original saddle gate')
        self.basis = self.highs.getBasis() if self.warm else None
        return PackingResult((lower+upper)/2, upper-lower, p, q,
                             self.highs.getInfo().simplex_iteration_count)


def native_batch(s, f, groups, directory, *, warm, threads=16):
    """We include file transport and process startup in the returned wall time."""
    binary = ROOT/'target/release/examples/leap_bench'
    source = ROOT/'src/crates/stl_solver/examples/leap_bench.rs'
    if not binary.exists() or binary.stat().st_mtime < source.stat().st_mtime:
        subprocess.run(['cargo', 'build', '--release', '-p', 'stl_solver_rs', '--example', 'leap_bench'],
                       cwd=ROOT, check=True, capture_output=True)
    tick = time.perf_counter()
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    src = directory/'native-input.bin'; dst = directory/'native-output.bin'
    np.ascontiguousarray(np.c_[s, f, groups], dtype='<f8').tofile(src)
    response = subprocess.run([str(binary), str(src), str(dst), str(int(warm)), str(threads)],
                              check=True, capture_output=True, text=True)
    result = np.fromfile(dst, dtype='<f8').reshape(-1, 5)
    timing = json.loads(response.stdout)
    timing['wall_seconds'] = time.perf_counter()-tick
    src.unlink(); dst.unlink()
    return result, timing


def rev_signature(clock, pc):
    p = profiles(); key = ('REV', int(clock))
    fail = child_key(key, int(p.st[pc])+60) if p.rev[pc] else None
    return child_key(key), fail, is_window(key)
