"""We export labelled leap stages for the support predictor of docs/PREDICTION.md.

For each key we restore its certified children from the artifact, run the
production `sweep_key` with a fallback that keeps the packing basis of every
residue class, and compare every value with the artifact. We then sample
residue classes and easy classes with the protocol seed, gather the 60 success
payoffs and the failure payoff of each sampled stage, certify a packing basis
for the easy sample, and write one `.npz` per key. We never write to the
artifact, and we edit nothing under `src/stl/solver/`.
"""
import argparse
import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np

from stl.experiments.benchmark_leap_replay import ARTIFACT, ROOT, reference, stage_children
from stl.solver import leap_build as build
from stl.solver.leap_lp import NativeFallback
from stl.solver.leap_profiles import N, profiles

SEED = 20261016


def gather(succ_table, fail_table, pcs, pds, compressed):
    """We read each stage's 60 success payoffs and failure payoff, as the fallback does."""
    p = profiles()
    columns = p.idx0[p.fail[pcs]] if compressed else p.fail[pcs]
    s = np.ascontiguousarray(-succ_table[pds[:, None], p.succ[pcs]])
    f = np.ascontiguousarray(p.rev[pcs]*-fail_table[pds, columns]+1-p.rev[pcs])
    return s, f


class RecordingFallback(NativeFallback):
    """We solve the residue as NativeFallback does and keep each class's certified basis.

    We skip the revival cache so that every residue class reaches the packing
    LP and its basis is recorded. A class that the packing LP rejects goes to
    HiGHS and keeps a zero basis, so the sample leaves it out.
    """
    def __init__(self, workers):
        super().__init__(workers)
        self.rows = []

    def solve(self, success, failure, succ_table, fail_table, pcs, pds, window):
        from stl.solver.leap_kernel import solve_packing
        compressed = failure is not None and failure[0] == 'REV'
        s, f = gather(succ_table, fail_table, pcs, pds, compressed)
        values = np.empty(len(pcs)); kind = np.empty(len(pcs), np.uint8)
        masks = np.zeros((len(pcs), 2), np.uint64)
        solve_packing(s, f, window, values, kind, crash=False, support_out=masks)
        residual = np.flatnonzero(kind == 255)
        if len(residual):
            values[residual] = self.pool.solve(succ_table.filename, fail_table.filename,
                                               pcs[residual], pds[residual], window, compressed)
        if not np.isfinite(values).all():
            raise RuntimeError('uncertified native/HiGHS residue')
        self.rows.append((pcs.copy(), pds.copy(), values.copy(), masks))
        return values, len(residual), 0


def compare(store, key):
    """We compare the swept dense table with the artifact and return the largest difference."""
    table = np.load(store.path(key), mmap_mode='r'); cols = build.key_columns(key)
    mask, values = reference(key); difference = 0.; offset = 0
    for lo in range(0, N, 1024):
        rows = table[lo:lo+1024, :cols]; reached = mask[lo:lo+1024]
        block = rows[reached]; expected = values[offset:offset+len(block)]; offset += len(block)
        if not np.isfinite(block).all() or np.isfinite(rows[~reached]).any():
            raise RuntimeError('sweep coverage differs from the artifact')
        difference = max(difference, float(np.max(np.abs(block-expected), initial=0.)))
    if offset != len(values) or difference > 1e-6:
        raise RuntimeError(f'sweep differs from the artifact by {difference:.3e}')
    return difference


def stage_rows(store, key, dth, pcs, pds, window):
    """We gather payoffs, kernel kinds, and packing bases for sampled classes of one key.

    We group the classes by failure child as `sweep_key` does, so each group
    reads one success table and one failure table.
    """
    from stl.solver.leap_kernel import solve_packing, sweep_batch
    p = profiles(); success_key = build.child_key(key)
    success_table = dth if success_key is None else store.load(success_key)
    s = np.empty((len(pcs), 60)); f = np.empty(len(pcs)); value = np.empty(len(pcs))
    kind = np.empty(len(pcs), np.uint8); masks = np.zeros((len(pcs), 2), np.uint64)
    dest = [None if not p.rev[c] else build.child_key(key, int(p.st[c])+60) for c in pcs]
    for group in sorted({d for d in dest}, key=lambda d: (d is not None, d or ('', 0))):
        rows = np.flatnonzero([d == group for d in dest])
        fail_table = dth if group is None else store.load(group)
        compressed = group is not None and group[0] == 'REV'
        fail_col = p.idx0[p.fail] if compressed else p.fail
        cs = np.ascontiguousarray(pcs[rows]); ds = np.ascontiguousarray(pds[rows])
        s[rows], f[rows] = gather(success_table, fail_table, cs, ds, compressed)
        out = np.empty(len(rows)); kinds = np.empty(len(rows), np.uint8)
        sweep_batch(cs, ds, success_table, fail_table, p.succ, fail_col, p.rev, window, 1e-6, out, kinds)
        kind[rows] = kinds
        values = np.empty(len(rows)); packed = np.empty(len(rows), np.uint8); bases = np.zeros((len(rows), 2), np.uint64)
        solve_packing(np.ascontiguousarray(s[rows]), f[rows], window, values, packed, crash=False, support_out=bases)
        value[rows] = np.where(kinds == 255, np.nan, out); masks[rows] = bases
        certified = packed == 3
        if np.any(np.abs(values[certified]-out[certified]) > 1e-6):
            raise RuntimeError('packing basis disagrees with the kernel value')
    return s, f, value, kind, masks


def export(key, rows, seed, output, scratch, workers, dth, manifest):
    """We sweep one key, check it, sample its classes, and write its `.npz`."""
    name = build.key_name(key); window = build.is_window(key)
    store = stage_children(key, scratch, manifest)
    bitmap = np.load(ARTIFACT/'reachability'/f'{name}.npy', mmap_mode='r')
    tick = time.perf_counter()
    with RecordingFallback(workers) as fallback:
        record = build.sweep_key(key, bitmap, store, dth, min_clock=720, full=True, fallback=fallback)
        residue = [np.concatenate(part) for part in zip(*fallback.rows)]
    record['sweep_seconds'] = time.perf_counter()-tick
    record['max_artifact_difference'] = compare(store, key)
    if not residue:
        raise ValueError(f'{name} has no residue class; the study samples residue keys only')
    pcs, pds, values, masks = residue
    certified = np.flatnonzero(masks.any(axis=1))
    rng = np.random.default_rng([seed, key[1], 1 if key[0] == 'H1' else 2])
    pick = rng.choice(certified, min(rows, len(certified)), replace=False)
    cols = build.key_columns(key)
    reached = np.unpackbits(bitmap, axis=1, count=cols, bitorder='little').astype(bool)
    reached[pcs, pds] = False
    easy = rng.choice(np.flatnonzero(reached), min(rows, int(reached.sum())), replace=False)
    del reached
    pcs = np.concatenate([pcs[pick], easy//cols]).astype(np.int32)
    pds = np.concatenate([pds[pick], easy % cols]).astype(np.int32)
    is_residue = np.arange(len(pcs)) < len(pick)
    s, f, value, kind, bases = stage_rows(store, key, dth, pcs, pds, window)
    if np.any(kind[is_residue] != 255) or np.any(kind[~is_residue] == 255):
        raise RuntimeError('sampled kinds disagree with the sweep residue')
    value[is_residue] = values[pick]
    if np.any(masks[pick] != bases[is_residue]):
        raise RuntimeError('residue bases differ between the sweep and the resample')
    keep = bases.any(axis=1)
    shutil.rmtree(scratch)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output/f'{name}.npz', checker=pcs[keep], dropper=pds[keep], success=s[keep], failure=f[keep],
             value=value[keep], kind=kind[keep], p_mask=bases[keep, 0], q_mask=bases[keep, 1],
             residue=is_residue[keep], window=np.bool_(window))
    summary = {'key': list(key), 'window': window, 'seed': seed, 'rows_requested': rows,
               'residue_classes': int(len(certified)+int((~masks.any(axis=1)).sum())),
               'residue_highs_classes': int((~masks.any(axis=1)).sum()),
               'residue_rows': int(keep[is_residue].sum()), 'easy_rows': int(keep[~is_residue].sum()),
               'easy_rows_without_basis': int((~keep[~is_residue]).sum()), 'sweep': record,
               'builder_sha256': build.builder_hash(),
               'artifact_manifest_sha256': build.file_hash(ARTIFACT/'manifest.json'),
               'platform': platform.platform()}
    build.write_json(output/f'{name}.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'sweep'}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--keys', nargs='+', required=True, help='keys such as H2_39')
    parser.add_argument('--rows', type=int, default=1_000_000, help='residue rows and easy rows per key')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--output', type=Path, default=ROOT/'src/stl/outputs/leap-residue')
    args = parser.parse_args()
    manifest = json.loads((ARTIFACT/'manifest.json').read_text())
    if not manifest.get('complete'):
        raise ValueError('export needs the complete certified artifact')
    dth = np.load(ROOT/'src/dth_compact/artifacts/V.npy', mmap_mode='r')
    for name in args.keys:
        kind, index = name.split('_'); key = (kind, int(index))
        export(key, args.rows, args.seed, args.output, args.output/'scratch'/name, args.workers, dth, manifest)


if __name__ == '__main__':
    main()
