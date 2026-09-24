"""You can replay whole leap keys against the certified artifact to time residue paths.

We copy each key's packed children from the artifact, restore them with
`TableStore.load`, and run the production `sweep_key` once per residue
configuration. We compare every reachable value with the artifact's value for
the replayed key. Both values carry certificates of width at most 1e-6, so a
difference above 1e-6 aborts the benchmark. We never write to the artifact.
"""
import argparse
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np

from stl.solver import leap_build as build
from stl.solver.leap_lp import NativeFallback
from stl.solver.leap_profiles import N, profiles

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT/'src/stl/outputs/leap-full-native'
BASELINE = {'crash': False, 'kernel_native': False, 'kink_attempts': 0}
CONFIGS = {'baseline': BASELINE,
           'python-crash': {'crash': True, 'kernel_native': False, 'kink_attempts': 0},
           'kernel-crash': {'crash': True, 'kernel_native': True, 'kink_attempts': 0},
           'kernel-crash-kink1': {'crash': True, 'kernel_native': True, 'kink_attempts': 1},
           'kernel-crash-kink2': {'crash': True, 'kernel_native': True, 'kink_attempts': 2},
           'kernel-crash-kink4': {'crash': True, 'kernel_native': True, 'kink_attempts': 4}}


def children(key):
    """We list the success child and every failure child that the sweep reads."""
    p = profiles(); keys = {build.child_key(key)}
    bitmap = np.load(ARTIFACT/'reachability'/f'{build.key_name(key)}.npy', mmap_mode='r')
    for checker in np.flatnonzero(np.any(bitmap, axis=1)):
        if p.rev[checker]:
            keys.add(build.child_key(key, int(p.st[checker])+60))
    return sorted((k for k in keys if k is not None), key=build.key_clock)


def stage_children(key, scratch, manifest):
    """We copy verified packed children into an empty scratch and restore dense tables."""
    if scratch.exists():
        shutil.rmtree(scratch)
    store = build.TableStore(scratch)
    for child in children(key):
        for suffix in ('.bits.npy', '.values.npy'):
            name = f'tables/{build.key_name(child)}{suffix}'
            if build.file_hash(ARTIFACT/name) != manifest['files'][name]:
                raise ValueError(f'artifact hash mismatch: {name}')
            shutil.copyfile(ARTIFACT/name, scratch/f'{build.key_name(child)}{suffix}')
        store.load(child)
    return store


def warm(store, key):
    """We read every restored child once so that each timed sweep starts from the page cache."""
    for child in children(key):
        table = np.load(store.path(child), mmap_mode='r')
        for lo in range(0, len(table), 1024):
            np.asarray(table[lo:lo+1024]).max()


def reference(key):
    """We read the artifact's reachable mask and values for the replayed key."""
    prefix = ARTIFACT/'tables'/build.key_name(key); cols = build.key_columns(key)
    bits = np.load(prefix.with_suffix('.bits.npy'), mmap_mode='r')
    values = np.load(prefix.with_suffix('.values.npy'), mmap_mode='r')
    mask = np.unpackbits(bits, count=N*cols, bitorder='little').reshape(N, cols).astype(bool)
    return mask, values


def replay(key, config, store, dth, workers):
    """We time one production sweep of a key and compare it with the artifact."""
    if store.path(key).exists():
        store.path(key).unlink()
    bitmap = np.load(ARTIFACT/'reachability'/f'{build.key_name(key)}.npy', mmap_mode='r')
    with NativeFallback(workers, **config) as fallback:
        tick = time.perf_counter()
        record = build.sweep_key(key, bitmap, store, dth, min_clock=720, full=True, fallback=fallback)
        record['sweep_seconds'] = time.perf_counter()-tick
    table = np.load(store.path(key), mmap_mode='r'); cols = build.key_columns(key)
    mask, values = reference(key); difference = 0.; offset = 0
    # Packed values follow the set bits in row-major order.
    for lo in range(0, N, 1024):
        rows = table[lo:lo+1024, :cols]; reached = mask[lo:lo+1024]
        block = rows[reached]; expected = values[offset:offset+len(block)]; offset += len(block)
        if not np.isfinite(block).all() or np.isfinite(rows[~reached]).any():
            raise RuntimeError('replay coverage differs from the artifact')
        difference = max(difference, float(np.max(np.abs(block-expected), initial=0.)))
    if offset != len(values):
        raise RuntimeError('replay coverage differs from the artifact')
    if difference > 1e-6:
        raise RuntimeError(f'replay differs from the artifact by {difference:.3e}')
    del table; store.path(key).unlink()
    record['max_artifact_difference'] = difference
    return record


def summarize(report, records, exclude=()):
    """We project full-sweep time from replayed keys and the artifact's per-key records.

    A replayed key's baseline gives its non-residue time: sweep seconds minus
    fallback seconds, which leaves the two-rung kernel and table writes. Each
    configuration then costs its sweep seconds minus that time, per uncached
    residue class. For each artifact key we remove the recorded residue time
    (Python fallback, HiGHS, and recovered partial work) from its sweep
    seconds, and add its uncached residue times the measured cost for its type
    (H1 or H2). REV keys use the H baseline cost in every configuration,
    because their replay lacks the sweep's revival cache.
    """
    runs = [r for r in report['runs'] if r['key'][0] != 'REV' and build.key_name(r['key']) not in exclude]
    by_key = {}
    for run in runs:
        by_key.setdefault(tuple(run['key']), {}).setdefault(run['config'], []).append(run)
    costs = {}
    for key, configs in by_key.items():
        if 'baseline' not in configs:
            continue
        base = min(configs['baseline'], key=lambda r: r['sweep_seconds'])
        residue = base['failures']-base['cached_failures']
        fixed = base['sweep_seconds']-base['fallback_seconds']
        for name, rows in configs.items():
            cost = (min(r['sweep_seconds'] for r in rows)-fixed)/residue
            costs.setdefault(key[0], {}).setdefault(name, []).append((cost, residue))
    names = sorted({name for kind in costs.values() for name in kind})
    pooled = {name: [c for kind in costs.values() for c in kind.get(name, [])] for name in names}
    projection = {}
    for bound in ('best', 'central', 'worst'):
        def rate(kind, name):
            values = costs.get(kind, {}).get(name) or pooled[name]
            if bound == 'best':
                return min(v[0] for v in values)
            if bound == 'worst':
                return max(v[0] for v in values)
            return float(np.average([v[0] for v in values], weights=[v[1] for v in values]))
        totals = {name: 0. for name in names}
        for record in records:
            kind = record['key'][0]
            residue = record['failures']-record['cached_failures']
            spent = record.get('fallback_seconds', record['lp_seconds'])+record.get('recovered_work_wall_seconds', 0.)
            fixed = record['sweep_seconds']-spent
            for name in names:
                totals[name] += fixed+residue*rate('H2' if kind == 'REV' else kind, 'baseline' if kind == 'REV' else name)
        projection[bound] = totals
    replayed = {build.key_name(k): {name: {'sweep_seconds': min(r['sweep_seconds'] for r in rows),
                                           'kernel_native_solves': rows[0]['kernel_native_solves'],
                                           'support_solves': rows[0]['support_solves'],
                                           'lp_solves': rows[0]['lp_solves'],
                                           'max_artifact_difference': max(r['max_artifact_difference'] for r in rows)}
                                    for name, rows in configs.items()}
                for k, configs in by_key.items()}
    return {'replayed': replayed, 'projected_sweep_seconds': projection, 'excluded_keys': sorted(exclude),
            'recorded_sweep_seconds': sum(r['sweep_seconds'] for r in records),
            'residue_us_per_class': {kind: {name: [round(c*1e6, 3) for c, _ in values] for name, values in kinds.items()}
                                     for kind, kinds in costs.items()},
            'note': 'best and worst use the smallest and largest per-class residue cost among replayed keys of each type'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--keys', nargs='+', default=[], help='keys such as H2_39 or REV_2941')
    parser.add_argument('--summarize', action='store_true', help='project full-sweep time from the report')
    parser.add_argument('--exclude', nargs='*', default=[], help='replayed keys to leave out of the projection')
    parser.add_argument('--configs', nargs='+', default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--output', type=Path, default=ROOT/'src/stl/outputs/leap-replay')
    args = parser.parse_args()
    if args.summarize:
        report = json.loads((args.output/'report.json').read_text())
        records = json.loads((ARTIFACT/'checkpoint.json').read_text())['records']
        summary = summarize(report, records, set(args.exclude))
        build.write_json(args.output/'summary.json', summary)
        print(json.dumps(summary, indent=2), flush=True)
        return
    manifest = json.loads((ARTIFACT/'manifest.json').read_text())
    if not manifest.get('complete'):
        raise ValueError('replay needs the complete certified artifact')
    dth = np.load(ROOT/'src/dth_compact/artifacts/V.npy', mmap_mode='r')
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output/'report.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {'runs': []}
    report['timing_scope'] = ('production sweep_key per key: kernel, residue path, HiGHS fallback, and dense '
                              'table writes; children restored and read before timing. A replayed REV key starts '
                              'with an empty revival cache, unlike the sweep, and HiGHS pool startup counts when a '
                              'key needs HiGHS; each run lists the production record for comparison.')
    production = {tuple(r['key']): r for r in json.loads((ARTIFACT/'checkpoint.json').read_text())['records']}
    identity = {'builder_sha256': build.builder_hash(), 'artifact_manifest_sha256': build.file_hash(ARTIFACT/'manifest.json'),
                'platform': platform.platform(), 'workers': args.workers}
    for name in args.keys:
        kind, index = name.split('_'); key = (kind, int(index))
        scratch = args.output/'scratch'/name
        try:
            store = stage_children(key, scratch, manifest)
            for repeat in range(args.repeats):
                order = args.configs if repeat % 2 == 0 else args.configs[::-1]
                for config in order:
                    warm(store, key)
                    record = replay(key, CONFIGS[config], store, dth, args.workers)
                    record.update({'config': config, 'residue': CONFIGS[config], 'repeat': repeat, **identity,
                                   'production': {k: production[key].get(k) for k in
                                                  ('failures', 'cached_failures', 'lp_solves', 'sweep_seconds')}})
                    report['runs'].append(record)
                    build.write_json(report_path, report)
                    print(json.dumps({k: record[k] for k in ('key', 'config', 'repeat', 'sweep_seconds', 'kernel_seconds',
                                      'fallback_seconds', 'failures', 'kernel_native_solves', 'support_solves',
                                      'native_solves', 'lp_solves', 'max_artifact_difference')}), flush=True)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == '__main__':
    main()
