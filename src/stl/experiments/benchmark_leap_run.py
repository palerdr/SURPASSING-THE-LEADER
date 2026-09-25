"""You can reproduce four leap hypotheses with retained, read-only children.

The children and the stopped H2_42 batch came from the superseded partial runs
in PARTIALS. We removed those runs on 2026-09-25, and git never held them.
Uncommitted builders made leap-full, leap-full-ipm and leap-full-centered
(aea64aaf..., fcbe4f9c..., 19bd5373...). src/stl/outputs/leap-full-native/
provenance/ keeps their sources. A new build from any commit finishes H2_42
under another builder, so it cannot recreate the stopped batch, and this
command cannot run again.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import statistics
import time

import numpy as np

from stl.solver import leap_build as build
from stl.experiments.benchmark_leap import PackingLP, native_batch, rev_signature, ROOT
from stl.solver.leap_lp import WarmLP
from stl.solver.leap_oracle import solve_lp
from stl.solver.leap_profiles import N, profiles

OUTPUTS = ROOT/'src/stl/outputs'
PARTIALS = ('leap-full-centered', 'leap-full-ipm', 'leap-full', 'leap-full-initial', 'leap')
PREFIX = OUTPUTS/PARTIALS[0]


def table(key):
    if key is None:
        return np.load(ROOT/'src/dth_compact/artifacts/V.npy', mmap_mode='r')
    for folder in PARTIALS:
        path = OUTPUTS/folder/'tables'/f'{build.key_name(key)}.npy'
        if path.exists():
            return np.load(path, mmap_mode='r')
    raise FileNotFoundError(f'no retained dense child: {key}')


def gather(key, pc, pd):
    p = profiles(); success = table(build.child_key(key))
    s = -success[pd[:, None], p.succ[pc]]; f = np.ones(len(pc))
    groups = {}
    for c in np.unique(pc):
        if p.rev[c]:
            groups.setdefault(build.child_key(key, int(p.st[c])+60), []).append(c)
    for dest, checkers in groups.items():
        ix = np.flatnonzero(np.isin(pc, checkers)); child = table(dest)
        col = p.idx0[p.fail[pc[ix]]] if dest is not None and dest[0] == 'REV' else p.fail[pc[ix]]
        f[ix] = p.rev[pc[ix]] * -child[pd[ix], col]+1-p.rev[pc[ix]]
    if not np.isfinite(s).all() or not np.isfinite(f).all():
        raise RuntimeError('sample gathered an unreachable child')
    return s, f


def current_batch():
    key = ('H2', 42); p = profiles(); kernel = build.require_kernel()
    bitmap = np.load(PREFIX/'reachability/H2_42.npy', mmap_mode='r')
    partial = np.load(PREFIX/'tables/H2_42.npy', mmap_mode='r')
    success = table(build.child_key(key)); groups = {}
    for checker in np.flatnonzero(np.any(bitmap, axis=1)):
        dest = build.child_key(key, int(p.st[checker])+60) if p.rev[checker] else None
        groups.setdefault(dest, []).append(int(checker))
    for dest, checkers in groups.items():
        checkers = np.array(checkers, np.int32)
        reached = np.unpackbits(bitmap[checkers], axis=1, count=N, bitorder='little').T
        fail = table(dest); width = max(64, min(N, 1048576//len(checkers)))
        for start in range(0, N, width):
            pd, local = np.nonzero(reached[start:start+width]); pd += start
            pc = checkers[local]; keep = ~np.isfinite(partial[pc, pd])
            pc = np.ascontiguousarray(pc[keep]); pd = np.ascontiguousarray(pd[keep], dtype=np.int32)
            if not len(pc):
                continue
            out = np.empty(len(pc)); kind = np.empty(len(pc), np.uint8)
            kernel(pc, pd, success, fail, p.succ, p.fail, p.rev, False, 1e-6, out, kind)
            pc = pc[kind == 255]; pd = pd[kind == 255]
            if len(pc):
                s, f = gather(key, pc, pd)
                return dict(s=s, f=f, pc=pc, pd=pd)
    raise RuntimeError('no stopped batch residue')


def random_residue(key, seed=832):
    rng = np.random.default_rng(seed)
    bitmap = np.load(PREFIX/'reachability'/f'{build.key_name(key)}.npy', mmap_mode='r')
    pc = rng.integers(N, size=50000, dtype=np.int32)
    pd = rng.integers(N, size=50000, dtype=np.int32)
    reached = (bitmap[pc, pd//8] & (1 << (pd % 8))) != 0
    pc = pc[reached]; pd = pd[reached]; s, f = gather(key, pc, pd)
    out = np.empty(len(pc)); kind = np.empty(len(pc), np.uint8)
    build.require_kernel()(np.zeros(len(pc), np.int32), np.arange(len(pc), dtype=np.int32),
            np.ascontiguousarray(-s), np.ascontiguousarray(-f[:, None]),
            np.arange(60, dtype=np.int32)[None, :], np.zeros(1, np.int32), np.ones(1),
            False, 1e-6, out, kind)
    bad = kind == 255
    if not bad.any():
        raise RuntimeError('holdout has no residue')
    return dict(s=s[bad], f=f[bad], pc=pc[bad], pd=pd[bad])


def worker(job):
    path, indices, method = job
    data = np.load(path); s = data['s'][indices]; f = data['f'][indices]
    solver = PackingLP() if method == 'packing' else WarmLP(False)
    output = np.empty(len(s)); gap = 0.; failures = pivots = 0
    for i, (success, failure) in enumerate(zip(s, f)):
        try:
            result = solver.solve(success, failure)
            output[i] = result.value; gap = max(gap, result.gap)
            pivots += solver.highs.getInfo().simplex_iteration_count
        except (ValueError, RuntimeError):
            result = solve_lp(success, failure, False)
            output[i] = result.value; gap = max(gap, result.gap); failures += 1
    return output, gap, failures, pivots


def parallel_highs(path, order, method, workers):
    tick = time.perf_counter()
    chunk = max(32, min(2048, len(order)//workers))
    jobs = [(str(path), order[i:i+chunk], method) for i in range(0, len(order), chunk)]
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        answers = list(pool.map(worker, jobs))
    values = np.concatenate([x[0] for x in answers])
    return values, {'seconds': time.perf_counter()-tick, 'max_gap': max(x[1] for x in answers),
                    'fallbacks': sum(x[2] for x in answers), 'pivots': sum(x[3] for x in answers)}


def solve_bench(path, directory, repeats, workers):
    sample = np.load(path); pc = sample['pc']; pd = sample['pd']; n = len(pc)
    orders = {'dropper_major': np.lexsort((pc, pd)), 'checker_major': np.lexsort((pd, pc))}
    experiments = [('game_highs', 'dropper_major'), ('packing', 'dropper_major'),
                   ('native_cold', 'dropper_major'), ('native_warm', 'dropper_major'),
                   ('game_highs', 'checker_major'), ('native_warm', 'checker_major')]
    records = {f'{method}/{order}': [] for method, order in experiments}
    reference = None
    rng = np.random.default_rng(791)
    for repeat in range(repeats):
        schedule = list(experiments)
        if repeat:
            rng.shuffle(schedule)
        for method, order_name in schedule:
            order = orders[order_name]
            if method.startswith('native'):
                result, timing = native_batch(sample['s'][order], sample['f'][order], np.zeros(n),
                                              directory, warm=method == 'native_warm', threads=workers)
                tick = time.perf_counter(); bad = ~np.isfinite(result[:, 0])
                gap = float(np.max(result[~bad, 1])) if (~bad).any() else 0.
                if bad.any():
                    values, fallback = parallel_highs(path, order[bad], 'game_highs', workers)
                    result[bad, 0] = values; gap = max(gap, fallback['max_gap'])
                stats = {'seconds': timing['wall_seconds']+time.perf_counter()-tick,
                         'kernel_seconds': timing['solve_seconds'], 'fallbacks': int(bad.sum()),
                         'max_gap': gap, 'accepted_path_pivots': float(result[:, 2].sum()),
                         'warm_bases': int(result[:, 3].sum()), 'cold_restarts': int(result[:, 4].sum())}
                values = result[:, 0]
            else:
                values, stats = parallel_highs(path, order, method, workers)
            restored = np.empty(n); restored[order] = values
            if reference is None:
                reference = restored
            stats['max_difference'] = float(np.max(np.abs(restored-reference)))
            if stats['max_difference'] > 1e-6 or stats['max_gap'] > 1e-6:
                raise RuntimeError('candidate failed benchmark certification')
            records[f'{method}/{order_name}'].append(stats)
            print(path.stem, repeat, method, order_name, json.dumps(stats), flush=True)
    baseline = statistics.median(x['seconds'] for x in records['game_highs/dropper_major'])
    return {'classes': n, 'checker_rows': len(np.unique(pc)), 'dropper_rows': len(np.unique(pd)),
            'source_sha256': build.file_hash(path), 'experiments': {name: {
                'trials': trials, 'median_seconds': statistics.median(x['seconds'] for x in trials),
                'speedup': baseline/statistics.median(x['seconds'] for x in trials)}
                for name, trials in records.items()}}


def rev_bench(directory, repeats):
    """We compare real writes and kernel work on sampled rows across 60 clocks."""
    p = profiles(); kernel = build.require_kernel(); rng = np.random.default_rng(921)
    clocks = list(range(3059, 2999, -1)); pcs = np.sort(rng.choice(N, 128, replace=False)).astype(np.int32)
    sampled = []; union = {}; full_states = 0; cache_tables = {}
    for clock in clocks:
        key = ('REV', clock); bitmap = np.load(PREFIX/'reachability'/f'REV_{clock}.npy', mmap_mode='r')
        full_states += int(build.POP[bitmap].sum())
        groups = {}
        for pc in range(N):
            groups.setdefault(rev_signature(clock, pc), []).append(pc)
        for signature, rows in groups.items():
            bits = union.setdefault(signature, np.zeros_like(bitmap))
            bits[rows] |= bitmap[rows]
        reached = np.unpackbits(bitmap[pcs], axis=1, count=183, bitorder='little').astype(bool)
        i, j = np.nonzero(reached); buckets = {}
        for pos, row in enumerate(i):
            signature = rev_signature(clock, int(pcs[row]))
            buckets.setdefault(signature, []).append(pos)
        sampled.append((i, j, [(signature, np.array(indices)) for signature, indices in buckets.items()]))
        for signature in buckets:
            for child in signature[:2]:
                if child not in cache_tables:
                    cache_tables[child] = table(child)
    distinct = sum(int(build.POP[bits].sum()) for bits in union.values())
    reference = None; reports = {'original': [], 'reuse': []}
    for repeat in range(repeats):
        for reuse in ((False, True) if repeat % 2 == 0 else (True, False)):
            path = directory/f'rev-{int(reuse)}.npy'
            output = np.lib.format.open_memmap(path, mode='w+', dtype=float, shape=(60, len(pcs), 184))
            tick = time.perf_counter(); output[:] = np.nan; output[:, :, -1] = -1.
            cache = {}; solved = 0; total = 0
            for k, (i, j, buckets) in enumerate(sampled):
                for signature, positions in buckets:
                    ii = i[positions]; jj = j[positions]; total += len(ii)
                    if reuse:
                        saved = cache.setdefault(signature, np.full((len(pcs), 183), np.nan))
                        values = saved[ii, jj].copy(); missing = np.isnan(values)
                    else:
                        values = np.empty(len(ii)); missing = np.ones(len(ii), bool)
                    if missing.any():
                        cs = np.ascontiguousarray(pcs[ii[missing]]); ds = np.ascontiguousarray(p.s0[jj[missing]])
                        out = np.empty(len(cs)); kind = np.empty(len(cs), np.uint8)
                        success, fail, window = signature
                        columns = p.idx0[p.fail] if fail is not None and fail[0] == 'REV' else p.fail
                        rejected = kernel(cs, ds, cache_tables[success], cache_tables[fail], p.succ,
                                          columns, p.rev, window, 1e-6, out, kind)
                        if rejected:
                            raise RuntimeError('selected REV band needs LP timing as well')
                        values[missing] = out; solved += len(cs)
                    if reuse:
                        saved[ii, jj] = values
                    output[k, ii, jj] = values
            output.flush(); seconds = time.perf_counter()-tick
            if reference is None:
                reference = np.array(output)
            difference = float(np.max(np.abs(output[np.isfinite(reference)]-reference[np.isfinite(reference)])))
            assert np.array_equal(np.isnan(output), np.isnan(reference)) and difference <= 1e-9
            reports['reuse' if reuse else 'original'].append({'seconds': seconds, 'states': total,
                'kernel_classes': solved, 'bytes_written': int(output.nbytes), 'max_difference': difference})
            del output; path.unlink()
    return {'clock_band': [3000, 3059], 'full_reachable_states': full_states,
            'distinct_continuations': distinct, 'existing_lp_failures_in_band': 0,
            'sampled_checker_rows': len(pcs), 'trials': reports,
            'speedup': statistics.median(x['seconds'] for x in reports['original'])/
                       statistics.median(x['seconds'] for x in reports['reuse'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUTS/'leap-hypotheses')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    missing = [str((OUTPUTS/name).relative_to(ROOT)) for name in PARTIALS if not (OUTPUTS/name).is_dir()]
    if missing:
        parser.exit(2, f'missing inputs: {", ".join(missing)}. We removed these superseded partial runs on '
                       '2026-09-25, and git never held them. src/stl/outputs/leap-full-native/provenance/ '
                       'keeps the sources of the uncommitted builders that made three of them. A new build '
                       'cannot recreate the stopped H2_42 batch, so this benchmark cannot run again.\n')
    args.output.mkdir(parents=True, exist_ok=True)
    row_sample = np.load(OUTPUTS/'leap-rung2b-row-samples.npz')
    first = args.output/'H2_43_rows.npz'; second = args.output/'H2_42_batch.npz'
    third = args.output/'H1_44_holdout.npz'
    if not first.exists():
        np.savez(first, **{k: row_sample[k] for k in ('s', 'f', 'pc', 'pd')})
    if not second.exists():
        np.savez(second, **current_batch())
    if not third.exists():
        np.savez(third, **random_residue(('H1', 44)))
    report = {'builder_sha256': build.builder_hash(), 'workers': args.workers, 'repeats': args.repeats,
              'timing_scope': 'gathered real residue; process startup, certification, and fallback included; no sweep writes',
              'benchmarks': {}}
    for path in (first, second, third):
        report['benchmarks'][path.stem] = solve_bench(path, args.output, args.repeats, args.workers)
        build.write_json(args.output/'report.json', report)
    report['rev'] = rev_bench(args.output, args.repeats)
    report['code_hashes'] = {str(path.relative_to(ROOT)): build.file_hash(path) for path in
                           [Path(__file__), ROOT/'src/stl/experiments/benchmark_leap.py',
                            ROOT/'src/crates/stl_solver/examples/leap_bench.rs']}
    build.write_json(args.output/'report.json', report)
    print(json.dumps(report['rev']), flush=True)


if __name__ == '__main__':
    main()
