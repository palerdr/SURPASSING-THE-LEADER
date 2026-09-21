"""We reuse HiGHS bases and certified revival values during the full sweep."""
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import multiprocessing
import os

import highspy
import numpy as np

from stl.solver.leap_oracle import StageResult, policy_masks, solve_lp, stage_matrix
from stl.solver.leap_profiles import N, profiles


class WarmLP:
    """We solve each explicit matrix with HiGHS, then check both saddle bounds."""
    def __init__(self, window):
        self.window = window; self.basis = None
        self.highs = highspy.Highs()
        for name, value in [('output_flag', False), ('threads', 1), ('solver', 'simplex'),
                            ('presolve', 'off'), ('primal_feasibility_tolerance', 1e-9),
                            ('dual_feasibility_tolerance', 1e-9)]:
            if self.highs.setOptionValue(name, value) != highspy.HighsStatus.kOk:
                raise RuntimeError(f'HiGHS rejected option {name}')
        rows = 60+int(window)
        self.lp = highspy.HighsLp()
        self.lp.num_col_ = rows+1; self.lp.num_row_ = 61
        self.lp.col_cost_ = np.r_[np.zeros(rows), -1.]
        self.lp.col_lower_ = np.r_[np.zeros(rows), -np.inf]
        self.lp.col_upper_ = np.full(rows+1, np.inf)
        self.lp.row_lower_ = np.r_[np.full(60, -np.inf), 1.]
        self.lp.row_upper_ = np.r_[np.zeros(60), 1.]
        self.lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        self.lp.a_matrix_.start_ = np.arange(rows+2, dtype=np.int32)*61
        self.lp.a_matrix_.index_ = np.tile(np.arange(61, dtype=np.int32), rows+1)

    def solve(self, s, f, *, support=False):
        s = np.asarray(s, np.float64)
        if s.shape != (60,) or not np.isfinite(s).all() or not np.isfinite(f):
            raise ValueError('stage needs finite payoffs')
        matrix = stage_matrix(s, f, self.window)
        coefficients = np.empty((len(matrix)+1, 61))
        coefficients[:-1, :60] = -matrix; coefficients[:-1, 60] = 1.
        coefficients[-1, :60] = 1.; coefficients[-1, 60] = 0.
        self.lp.a_matrix_.value_ = coefficients.ravel()
        status = self.highs.passModel(self.lp)
        if status == highspy.HighsStatus.kError:
            raise RuntimeError('HiGHS rejected the explicit stage matrix')
        if self.basis is not None:
            if self.highs.setBasis(self.basis) == highspy.HighsStatus.kError:
                raise RuntimeError('HiGHS rejected its saved basis')
        self.highs.run()
        if self.highs.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            self.basis = None
            return solve_lp(s, f, self.window, support=support)
        solution = self.highs.getSolution()
        p = np.maximum(solution.col_value[:-1], 0.)
        q = np.maximum(-np.asarray(solution.row_dual[:60]), 0.)
        p /= p.sum(); q /= q.sum()
        lower = float(np.min(p @ matrix)); upper = float(np.max(matrix @ q))
        if not np.isfinite(upper-lower) or upper-lower > 1e-8:
            # We discard an inaccurate warm basis and use the scalar LP authority.
            self.basis = None
            return solve_lp(s, f, self.window, support=support)
        self.basis = self.highs.getBasis()
        return StageResult((lower+upper)/2, 3, lower, upper, policy_masks(p, q) if support else None)


class RevivalCache:
    """We reuse a certified value only for identical child tables and profiles."""
    def __init__(self):
        self.groups = OrderedDict(); self.tokens = {}

    def _token(self, fail):
        return self.tokens.setdefault(fail, len(self.tokens))

    def get(self, success, fail, window, pcs, pds):
        output = np.full(len(pcs), np.nan)
        key = (success, window)
        if key not in self.groups:
            return output
        self.groups.move_to_end(key)
        values, tokens = self.groups[key]; cols = profiles().idx0[pds]
        eligible = (cols >= 0) & (cols < 183)
        indices = np.flatnonzero(eligible)
        indices = indices[tokens[pcs[indices], cols[indices]] == self._token(fail)]
        output[indices] = values[pcs[indices], cols[indices]]
        return output

    def put(self, success, fail, window, pcs, pds, output):
        cols = profiles().idx0[pds]; eligible = (cols >= 0) & (cols < 183)
        if not eligible.any():
            return
        if not np.isfinite(output).all():
            raise ValueError('cache requires finite certified values')
        key = (success, window)
        if key not in self.groups:
            if len(self.groups) == 2:
                self.groups.popitem(last=False)
            self.groups[key] = (np.full((N, 183), np.nan), np.full((N, 183), -1, np.int32))
        values, tokens = self.groups[key]
        values[pcs[eligible], cols[eligible]] = output[eligible]
        tokens[pcs[eligible], cols[eligible]] = self._token(fail)


@lru_cache(maxsize=2)
def _table(path):
    return np.load(path, mmap_mode='r')


@lru_cache(maxsize=2)
def _solver(window):
    return WarmLP(window)


def _solve_chunk(job):
    succ_path, fail_path, pcs, pds, window, compressed, support = job
    succ = _table(succ_path); fail = _table(fail_path)
    p = profiles(); solver = _solver(False if support else window); result = np.empty(len(pcs))
    masks = np.zeros((len(pcs), 2), np.uint64) if support else None
    for i, (pc, pd) in enumerate(zip(pcs, pds)):
        col = p.idx0[p.fail[pc]] if compressed else p.fail[pc]
        s = -succ[pd, p.succ[pc]]
        f = p.rev[pc]*-fail[pd, col]+1-p.rev[pc]
        solved = solver.solve(s, f, support=support)
        result[i] = max(solved.value, f) if support and window else solved.value
        if support:
            masks[i] = solved.support
    return (result, masks) if support else result


class FallbackPool:
    """We send independent LP batches to processes with one HiGHS thread each."""
    def __init__(self, workers=None):
        self.workers = workers or min(16, os.cpu_count() or 1)
        self.executor = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)

    def solve(self, succ_path, fail_path, pcs, pds, window, compressed, *, support=False):
        if self.executor is None:
            self.executor = ProcessPoolExecutor(self.workers, mp_context=multiprocessing.get_context('spawn'))
        chunk = max(1 if support else 32, min(2048, len(pcs)//self.workers))
        jobs = [(str(succ_path), str(fail_path), pcs[i:i+chunk], pds[i:i+chunk], window, compressed, support)
                for i in range(0, len(pcs), chunk)]
        output = np.empty(len(pcs)); offset = 0
        masks = np.zeros((len(pcs), 2), np.uint64) if support else None
        for response in self.executor.map(_solve_chunk, jobs):
            values, policies = response if support else (response, None)
            output[offset:offset+len(values)] = values
            if support:
                masks[offset:offset+len(values)] = policies
            offset += len(values)
        return (output, masks) if support else output


class Fallback:
    def __init__(self, workers=None):
        self.pool = FallbackPool(workers); self.cache = RevivalCache()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.pool.__exit__(*args)

    def solve(self, success, failure, succ_table, fail_table, pcs, pds, window):
        output = self.cache.get(success, failure, window, pcs, pds)
        missing = np.flatnonzero(np.isnan(output))
        if len(missing):
            output[missing] = self.pool.solve(succ_table.filename, fail_table.filename,
                    pcs[missing], pds[missing], window, failure is not None and failure[0] == 'REV')
            self.cache.put(success, failure, window, pcs[missing], pds[missing], output[missing])
        return output, len(missing), len(pcs)-len(missing)


class NativeFallback(Fallback):
    """We try certified native pivots, then send the remaining residue to HiGHS."""
    def __init__(self, workers=None):
        super().__init__(workers)
        import stl_solver_rs
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]/'crates/stl_solver/src'
        if stl_solver_rs.LEAP_PACKING_SOURCE != (root/'leap_packing.rs').read_text():
            raise ValueError('stale packing extension; rebuild with maturin')
        self.native_solves = 0
        self.native_seconds = self.highs_seconds = 0.

    def solve(self, success, failure, succ_table, fail_table, pcs, pds, window):
        import time
        from stl_solver_rs import solve_packing_rs
        output = self.cache.get(success, failure, window, pcs, pds)
        missing = np.flatnonzero(np.isnan(output)); cached = len(pcs)-len(missing)
        if not len(missing):
            return output, 0, cached
        tick = time.perf_counter(); p = profiles()
        cs = pcs[missing]; ds = pds[missing]
        compressed = failure is not None and failure[0] == 'REV'
        columns = p.idx0[p.fail[cs]] if compressed else p.fail[cs]
        s = np.ascontiguousarray(-succ_table[ds[:,None], p.succ[cs]])
        f = np.ascontiguousarray(p.rev[cs]*-fail_table[ds,columns]+1-p.rev[cs])
        values = np.empty(len(missing)); kind = np.empty(len(missing),np.uint8)
        failed = solve_packing_rs(s,f,window,values,kind)
        self.native_seconds += time.perf_counter()-tick
        self.native_solves += len(missing)-failed
        residual = np.flatnonzero(kind == 255)
        if len(residual):
            tick = time.perf_counter()
            values[residual] = self.pool.solve(succ_table.filename,fail_table.filename,
                    cs[residual],ds[residual],window,compressed)
            self.highs_seconds += time.perf_counter()-tick
        if not np.isfinite(values).all():
            raise RuntimeError('uncertified native/HiGHS residue')
        output[missing] = values
        self.cache.put(success,failure,window,cs,ds,values)
        return output, len(residual), cached


class SupportFallback(Fallback):
    """We refresh a missed support with HiGHS before continuing its row."""
    def __init__(self, workers=None, max_dimension=16):
        if not 1 <= max_dimension <= 60:
            raise ValueError('max_dimension must be in 1..60')
        super().__init__(workers)
        self.max_dimension = max_dimension
        self.large_support_lp = 0
        self.support_groups = OrderedDict()
        self.support_hits = self.edge_hits = 0
        self.support_seconds = self.lp_seconds = 0.

    def solve(self, success, failure, succ_table, fail_table, pcs, pds, window):
        import time
        from stl_solver_rs import sweep_key_rs
        output = self.cache.get(success, failure, window, pcs, pds)
        pending = np.flatnonzero(np.isnan(output))
        cached = len(output)-len(pending); solved = 0
        if not len(pending):
            return output, solved, cached
        group = (success, failure, window)
        if group not in self.support_groups:
            if len(self.support_groups) == 16:
                self.support_groups.popitem(last=False)
            self.support_groups[group] = np.zeros((N, 2), np.uint64)
        self.support_groups.move_to_end(group)
        seeds = self.support_groups[group]
        p = profiles(); compressed = failure is not None and failure[0] == 'REV'
        fail_col = p.idx0[p.fail] if compressed else p.fail
        pending = pending[np.lexsort((pds[pending], pcs[pending]))]

        def large_rows():
            from .leap_support import reverse_support
            def dimension(mask):
                return 1 + ((~(mask & (mask >> 1))) & ((1 << 59)-1)).bit_count()
            rows = np.unique(pcs[pending])
            return rows[np.array([max(dimension(int(seeds[row, 0])),
                    dimension(reverse_support(int(seeds[row, 1])))) > self.max_dimension
                    for row in rows], dtype=bool)]

        def lp(indices):
            nonlocal solved
            tick = time.perf_counter()
            values, masks = self.pool.solve(succ_table.filename, fail_table.filename,
                    pcs[indices], pds[indices], window, compressed, support=True)
            self.lp_seconds += time.perf_counter()-tick
            output[indices] = values; seeds[pcs[indices]] = masks
            solved += len(indices)

        # We need one certified LP support for each row's first failed class.
        unseeded = pending[(seeds[pcs[pending], 0] == 0) | np.isin(pcs[pending], large_rows())]
        if len(unseeded):
            _, first = np.unique(pcs[unseeded], return_index=True)
            lp(unseeded[first])
            pending = pending[np.isnan(output[pending])]
        while len(pending):
            # We batch large systems through HiGHS to bound support-search cost.
            bulk = np.isin(pcs[pending], large_rows())
            if bulk.any():
                indices = pending[bulk]
                tick = time.perf_counter()
                output[indices] = self.pool.solve(succ_table.filename, fail_table.filename,
                        pcs[indices], pds[indices], window, compressed)
                self.lp_seconds += time.perf_counter()-tick
                solved += len(indices); self.large_support_lp += len(indices)
                pending = pending[~bulk]
                if not len(pending):
                    break
            values = np.empty(len(pending)); kinds = np.empty(len(pending), np.uint8)
            masks = np.zeros((len(pending), 2), np.uint64)
            tick = time.perf_counter()
            sweep_key_rs(np.ascontiguousarray(pcs[pending]), np.ascontiguousarray(pds[pending]),
                    succ_table, fail_table, p.succ, fail_col, p.rev, window, 1e-6,
                    values, kinds, seeds, masks, stop_on_support_miss=True)
            self.support_seconds += time.perf_counter()-tick
            accepted = kinds < 254
            if not np.isfinite(values[accepted]).all():
                raise RuntimeError('reduced support path returned an uncertified value')
            output[pending[accepted]] = values[accepted]
            self.support_hits += int(((kinds == 4) | (kinds == 5) | ((kinds == 2) & (masks[:, 0] != 0))).sum())
            self.edge_hits += int((kinds == 5).sum())
            # We retain the last accepted support when the next batch starts.
            with_policy = np.flatnonzero(accepted & (masks[:, 0] != 0))
            if len(with_policy):
                reverse = with_policy[::-1]
                _, last = np.unique(pcs[pending[reverse]], return_index=True)
                chosen = reverse[last]
                seeds[pcs[pending[chosen]]] = masks[chosen]
            pending = pending[~accepted]
            if len(pending):
                # We solve one miss per row. Later deferred classes then see
                # that LP's support, rather than an obsolete first-row guess.
                _, first = np.unique(pcs[pending], return_index=True)
                lp(pending[first])
                pending = pending[np.isnan(output[pending])]
        self.cache.put(success, failure, window, pcs, pds, output)
        return output, solved, cached
