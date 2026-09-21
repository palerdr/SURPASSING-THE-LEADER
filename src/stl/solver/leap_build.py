"""Reachability, value storage, and bounded calibration of the public leap game."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

from stl.solver.leap_profiles import N, N_ALIVE, WIN, child_clock, profiles
from stl.solver.leap_oracle import solve_lp, solve_stage

POP = np.array([i.bit_count() for i in range(256)], np.uint8)
TOTALS = {'H1': 4180634990, 'H2': 4188677815, 'REV': 1084020312}


def key_clock(key):
    kind, index = key
    return index if kind == 'REV' else 60*index + (120 if kind == 'H2' else 0)


def key_name(key):
    return f'{key[0]}_{key[1]}'


def key_columns(key):
    return 183 if key[0] == 'REV' else N


def all_keys():
    return sorted([('H1', m) for m in range(12, 60)] +
                  [('H2', m) for m in range(12, 59)] +
                  [('REV', c) for c in range(1020, 3601)], key=key_clock)


def child_key(key, q=0):
    half = 1 if key[0] == 'H1' else 2
    clock = child_clock(half, key_clock(key), q)
    if clock > 3600:
        return None
    if half == 1:
        return ('REV', clock) if q else ('H2', key[1])
    return ('H1', clock//60)


def is_window(key):
    return key[0] != 'H1' and 3540 <= key_clock(key) <= 3600


def _transpose(bits, columns):
    return np.packbits(np.unpackbits(bits, axis=1, count=columns, bitorder='little').T,
                       axis=1, bitorder='little')


def _success_bits(bits):
    p = profiles()
    target = np.zeros_like(bits)
    for start, stop in (*p.blocks, (N_ALIVE, N)):
        length = stop-start
        expanded = np.zeros((min(300, length+60), bits.shape[1]), np.uint8)
        expanded[1:min(length+1, 300)] = bits[start:start+min(length, 299)]
        for shift in (1, 2, 4, 8, 16, 28):
            expanded[shift:] |= expanded[:-shift]
        if start == N_ALIVE:
            target[N_ALIVE:] |= expanded
        else:
            target[start:stop] |= expanded[:length]
            target[N_ALIVE+length:N_ALIVE+len(expanded)] |= expanded[length:]
    return target


@dataclass
class Reachability:
    counts: dict
    directory: Path | None

    @property
    def total(self):
        return sum(self.counts.values())

    @property
    def by_type(self):
        return {kind: sum(v for k, v in self.counts.items() if k[0] == kind) for kind in TOTALS}

    @property
    def window_count(self):
        return sum(v for k, v in self.counts.items() if is_window(k))

    def load(self, key):
        return np.load(self.directory / f'{key_name(key)}.npy', mmap_mode='r')


def forward_reachability(limit=3600, directory=None):
    """Union successor sets before counting each quotient state once."""
    if limit > 3600:
        raise ValueError('reachability ends at 3600')
    directory = Path(directory) if directory is not None else None
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    p = profiles(); pending = {}; counts = {}

    def target(key):
        if key not in pending:
            pending[key] = np.zeros((N, (key_columns(key)+7)//8), np.uint8)
        return pending[key]

    target(('H1', 12))[0, 0] = 1
    for key in all_keys():
        if key_clock(key) > limit:
            break
        if key not in pending:
            continue
        bits = pending.pop(key)
        count = int(POP[bits].sum())
        if not count:
            continue
        counts[key] = count
        if directory is not None:
            np.save(directory/f'{key_name(key)}.npy', bits)
        ncols = key_columns(key)
        rows = p.s0 if key[0] == 'REV' else slice(None)
        success = child_key(key)
        if success is not None and key_clock(success) <= limit:
            target(success)[rows] |= _transpose(_success_bits(bits), ncols)
        # We group by the destination clock before merging revival paths.
        groups = {}
        for pc in np.flatnonzero(np.any(bits[:N_ALIVE], axis=1)):
            dest = child_key(key, int(p.st[pc])+60)
            if dest is not None and key_clock(dest) <= limit:
                groups.setdefault(dest, []).append(int(pc))
        for dest, pcs in groups.items():
            merged = np.zeros((183, bits.shape[1]), np.uint8)
            np.bitwise_or.at(merged, p.idx0[p.fail[pcs]], bits[pcs])
            transposed = _transpose(merged, ncols)
            if dest[0] == 'REV':
                target(dest)[rows] |= transposed
            else:
                # We place the 183 reset-profile columns in their full profile positions.
                dense = np.unpackbits(transposed, axis=1, count=183, bitorder='little')
                dest_bits = target(dest)
                for j, col in enumerate(p.s0):
                    dest_bits[rows, col//8] |= dense[:, j] << int(col % 8)
        if directory is not None and key[0] == 'H1':
            print(f'reach {key_name(key)}: {count:,}', flush=True)
    result = Reachability(counts, directory)
    if directory is not None:
        (directory/'counts.json').write_text(json.dumps({key_name(k): v for k, v in counts.items()}))
        files = {path.name: file_hash(path) for path in directory.glob('*') if path.name != 'manifest.json'}
        write_json(directory/'manifest.json', {'builder_sha256': builder_hash(), 'files': files, 'limit': limit})
    return result


def evaluate_states(key, pairs, dth):
    """Evaluate bounded test subtrees without sweeping a key."""
    p = profiles()
    @lru_cache(None)
    def value(key, pc, pd):
        if key is None:
            return float(dth[pc, pd])
        success = child_key(key)
        s = np.array([1. if ch == WIN else -value(success, pd, int(ch)) for ch in p.succ[pc]])
        f = 1.
        if p.rev[pc]:
            f = p.rev[pc] * -value(child_key(key, int(p.st[pc])+60), pd, int(p.fail[pc])) + 1-p.rev[pc]
        return solve_stage(s, f, is_window(key)).value
    return np.array([value(key, int(pc), int(pd)) for pc, pd in pairs])


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def builder_hash():
    root = Path(__file__).resolve().parents[3]
    files = [*Path(__file__).parent.glob('leap_*.py'),
             root/'src/stl/engine/game.py', root/'src/stl/solver/canonical.py',
             root/'src/crates/stl_solver/Cargo.toml', root/'Cargo.lock', root/'uv.lock', root/'pyproject.toml',
             * (root/'src/crates/stl_solver/src').glob('*.rs')]
    h = hashlib.sha256()
    for path in sorted(files):
        name = str(path.relative_to(root)).encode(); data = path.read_bytes()
        h.update(len(name).to_bytes(8, 'big')); h.update(name)
        h.update(len(data).to_bytes(8, 'big')); h.update(data)
    return h.hexdigest()


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True)+'\n')
    temporary.replace(path)


class TableStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.write_bytes = 0; self.write_seconds = 0.
        self.pack_bytes = 0; self.pack_seconds = 0.

    def path(self, key):
        return self.directory/f'{key_name(key)}.npy'

    def create(self, key):
        start = time.perf_counter()
        table = np.lib.format.open_memmap(self.path(key), mode='w+', dtype=np.float64,
                                          shape=(N, key_columns(key)+1))
        table[:] = np.nan; table[:, -1] = -1.; table.flush()
        self.write_seconds += time.perf_counter()-start
        self.write_bytes += table.nbytes
        return table

    def pack(self, key, *, retain_dense=False):
        start = time.perf_counter(); table = np.load(self.path(key), mmap_mode='r')
        cols = key_columns(key)
        count = sum(int(np.isfinite(table[lo:lo+64, :cols]).sum()) for lo in range(0, N, 64))
        prefix = self.directory/key_name(key)
        bitmap_path = prefix.with_suffix('.bits.npy'); values_path = prefix.with_suffix('.values.npy')
        bitmap = np.lib.format.open_memmap(bitmap_path, mode='w+', dtype=np.uint8,
                                           shape=((N*cols+7)//8,))
        values = np.lib.format.open_memmap(values_path, mode='w+', dtype=np.float64, shape=(count,))
        offset = 0
        for lo in range(0, N, 64):
            block = table[lo:lo+64, :cols]; mask = np.isfinite(block)
            packed = np.packbits(mask.reshape(-1), bitorder='little')
            bitmap[lo*cols//8:lo*cols//8+len(packed)] = packed
            data = block[mask]; values[offset:offset+len(data)] = data; offset += len(data)
        bitmap.flush(); values.flush()
        self.pack_bytes += bitmap.nbytes+values.nbytes
        self.pack_seconds += time.perf_counter()-start
        del bitmap, values, table
        if not retain_dense:
            self.path(key).unlink()

    def pack_cold(self, clock):
        packed = []
        for path in self.directory.glob('*.npy'):
            if '.' in path.stem:
                continue
            kind, index = path.stem.split('_'); key = (kind, int(index))
            if key_clock(key) > clock+540:
                self.pack(key); packed.append(key)
        return packed

    def load(self, key):
        if self.path(key).exists():
            return np.load(self.path(key), mmap_mode='r')
        prefix = self.directory/key_name(key)
        bitmap = np.load(prefix.with_suffix('.bits.npy'), mmap_mode='r')
        values = np.load(prefix.with_suffix('.values.npy'), mmap_mode='r')
        table = self.create(key); cols = key_columns(key); offset = 0
        for lo in range(0, N, 64):
            rows = min(64, N-lo); length = rows*cols
            mask = np.unpackbits(bitmap[lo*cols//8:(lo*cols+length+7)//8],
                                 count=length, bitorder='little').reshape(rows, cols).astype(bool)
            count = int(mask.sum()); table[lo:lo+rows, :cols][mask] = values[offset:offset+count]
            offset += count
        if offset != len(values):
            raise ValueError('packed value count mismatch')
        table.flush()
        return table


def sweep_key(key, bitmap, store, dth, *, min_clock=3420, full=False, fallback=None):
    floor = 720 if full else 3420
    if min_clock < floor or key_clock(key) < min_clock:
        raise ValueError(f'key sweeps must stop at clock {floor}')
    sweep_key_rs = require_kernel()
    p = profiles(); table = store.create(key); cols = key_columns(key)
    success_key = child_key(key)
    success_table = dth if success_key is None else store.load(success_key)
    full_pds = p.s0 if key[0] == 'REV' else np.arange(N, dtype=np.int32)
    states = failures = lp_solves = cached_failures = 0; kernel_seconds = lp_seconds = 0.
    native_before = getattr(fallback, 'native_solves', 0)
    native_time_before = getattr(fallback, 'native_seconds', 0.)
    highs_before = getattr(fallback, 'highs_seconds', None)
    groups = {}
    for checker in np.flatnonzero(np.any(bitmap, axis=1)):
        dest = None if not p.rev[checker] else child_key(key, int(p.st[checker])+60)
        groups.setdefault(dest, []).append(int(checker))
    # We group by child table before batching to avoid repeated map validation
    # for small fragments of the same revival group. We retain Dropper row order.
    for dest, checkers in groups.items():
        checkers = np.array(checkers, np.int32)
        reached = np.unpackbits(bitmap[checkers], axis=1, count=cols, bitorder='little').T
        fail_table = dth if dest is None else store.load(dest)
        fail_col = p.idx0[p.fail] if dest is not None and dest[0] == 'REV' else p.fail
        rows_per_batch = max(64, min(cols, 1_048_576//len(checkers)))
        for start in range(0, cols, rows_per_batch):
            js, local_pc = np.nonzero(reached[start:start+rows_per_batch]); js += start
            if not len(js):
                continue
            pcs = np.ascontiguousarray(checkers[local_pc])
            pds = np.ascontiguousarray(full_pds[js], dtype=np.int32)
            out = np.empty(len(pcs)); kind = np.empty(len(pcs), np.uint8)
            tick = time.perf_counter()
            failed = sweep_key_rs(pcs, pds, success_table, fail_table, p.succ,
                                  fail_col, p.rev, is_window(key), 1e-6, out, kind)
            kernel_seconds += time.perf_counter()-tick
            failures += failed
            if failed:
                tick = time.perf_counter()
                indices = np.flatnonzero(kind == 255)
                if fallback is None:
                    for i in indices:
                        checker, dropper = int(pcs[i]), int(pds[i])
                        successes = -success_table[dropper, p.succ[checker]]
                        failure = p.rev[checker] * -fail_table[dropper, fail_col[checker]] + 1-p.rev[checker]
                        out[i] = solve_lp(successes, failure, is_window(key)).value
                    lp_solves += failed
                else:
                    out[indices], solved, cached = fallback.solve(success_key, dest, success_table,
                            fail_table, pcs[indices], pds[indices], is_window(key))
                    lp_solves += solved; cached_failures += cached
                    if failed > 1000:
                        native = failed-solved-cached
                        print(f'fallback {key_name(key)}: {native:,} native, {solved:,} HiGHS, {cached:,} cached', flush=True)
                lp_seconds += time.perf_counter()-tick
            if not np.isfinite(out).all():
                raise RuntimeError('uncertified class aborts the build')
            tick = time.perf_counter()
            table[pcs, js] = out
            store.write_seconds += time.perf_counter()-tick
            states += len(pcs)
    tick = time.perf_counter(); table.flush()
    store.write_seconds += time.perf_counter()-tick
    if states != int(POP[bitmap].sum()):
        raise RuntimeError('sweep missed reachable classes')
    return {'key': list(key), 'clock': key_clock(key), 'states': states,
            'kernel_seconds': kernel_seconds, 'classes_per_second': states/kernel_seconds,
            'failures': failures, 'fallback_seconds': lp_seconds,
            'lp_seconds': lp_seconds if highs_before is None else fallback.highs_seconds-highs_before,
            'native_solves': getattr(fallback, 'native_solves', 0)-native_before,
            'native_seconds': getattr(fallback, 'native_seconds', 0.)-native_time_before,
            'lp_solves': lp_solves, 'cached_failures': cached_failures}


def _load_reach(directory):
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest['builder_sha256'] != builder_hash():
        raise ValueError('reachability source mismatch')
    validate_files(directory, manifest['files'])
    data = json.loads((directory/'counts.json').read_text())
    return Reachability({(name.split('_')[0], int(name.split('_')[1])): n
                         for name, n in data.items()}, directory)


def calibration_report(records, reach, store, wall_seconds):
    rows = {}
    for kind in TOTALS:
        selected = [r for r in records if r['key'][0] == kind
                    and not (kind == 'H2' and is_window(r['key']))]
        seconds = sum(r['kernel_seconds'] for r in selected)
        states = sum(r['states'] for r in selected)
        rates = [r['classes_per_second'] for r in selected]
        rows[kind] = {'states': states, 'classes_per_second': states/seconds,
                      'rate_low': min(rates), 'rate_high': max(rates),
                      'failures': sum(r['failures'] for r in selected)}
    write_rate = store.write_bytes/store.write_seconds
    pack_rate = store.pack_bytes/store.pack_seconds
    done = {tuple(r['key']) for r in records}
    remaining = {k: v for k, v in reach.counts.items() if k not in done}
    remaining_states = {kind: sum(v for k, v in remaining.items() if k[0] == kind) for kind in TOTALS}
    window_records = [r for r in records if r['key'][0] == 'H2' and is_window(r['key'])]
    window_states = sum(r['states'] for r in window_records)
    window_seconds = sum(r['kernel_seconds'] for r in window_records)
    dense_bytes = sum(N*(key_columns(k)+1)*8 for k in remaining)
    packed_bytes = sum(v*8+(N*key_columns(k)+7)//8 for k, v in remaining.items())
    failures = sum(r['failures'] for r in records)
    lp_seconds = sum(r['lp_seconds'] for r in records)
    lp_cost = lp_seconds/failures if failures else 0.
    projected_residue = sum(remaining_states[k]*rows[k]['failures']/rows[k]['states'] for k in rows)
    write_rates = [r['write_bytes']/r['write_seconds'] for r in records]
    pack_rates = [r['pack_bytes']/r['pack_seconds'] for r in records]
    io_seconds = {'low': dense_bytes/max(write_rates)+packed_bytes/max(pack_rates),
                  'central': dense_bytes/write_rate+packed_bytes/pack_rate,
                  'high': dense_bytes/min(write_rates)+packed_bytes/min(pack_rates)}
    estimates = {bound: sum(remaining_states[k]/rows[k][rate] for k in rows)
                 + io_seconds[bound]+projected_residue*lp_cost
                 for bound, rate in [('low', 'rate_high'), ('central', 'classes_per_second'), ('high', 'rate_low')]}
    return {'rows': rows,
            'window_H2': {'states': window_states, 'classes_per_second': window_states/window_seconds if window_seconds else None,
                          'failures': sum(r['failures'] for r in window_records)},
            'memmap_write_MB_s': write_rate/1e6, 'pack_MB_s': pack_rate/1e6,
            'memmap_write_rate_range_MB_s': [min(write_rates)/1e6, max(write_rates)/1e6],
            'pack_rate_range_MB_s': [min(pack_rates)/1e6, max(pack_rates)/1e6],
            'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
            'wall_seconds': wall_seconds, 'remaining_dense_bytes': dense_bytes,
            'remaining_packed_bytes': packed_bytes, 'projected_residue': projected_residue,
            'seconds_per_lp': lp_cost if failures else None,
            'projection_seconds': estimates,
            'projection_note': 'Per-key kernel and I/O spread. Zero observed residue assumes zero future residue.'}


def calibrate(dth_path, directory, min_clock=3420):
    if min_clock != 3420:
        raise ValueError('calibration must stop at clock 3420')
    require_kernel()
    start = time.perf_counter(); directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    dth_path = Path(dth_path); dth = np.load(dth_path, mmap_mode='r')
    if dth.shape != (N, N+1) or dth.dtype != np.float64 or not np.all(dth[:, -1] == -1):
        raise ValueError('DTH table schema mismatch')
    identity = {'schema': 'stl-leap-values-v1', 'builder_sha256': builder_hash(),
                'dth_sha256': file_hash(dth_path), 'saddle_tolerance': 1e-6,
                'minimum_clock': min_clock, 'python': sys.version, 'numpy': np.__version__}
    checkpoint_path = directory/'checkpoint.json'
    records = []
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text())
        if checkpoint['identity'] != identity:
            raise ValueError('checkpoint source or DTH digest mismatch')
        records = checkpoint['records']
        manifest_path = directory/'manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get('stopped_at_clock') == min_clock and manifest.get('builder_sha256') == identity['builder_sha256']:
                validate_files(directory, manifest['files'])
                report = json.loads((directory/'report.json').read_text())
                print(json.dumps(report, indent=2), flush=True)
                return report
        validate_files(directory, checkpoint['files'])
    reach_dir = directory/'reachability'
    reach = _load_reach(reach_dir) if (reach_dir/'counts.json').exists() else forward_reachability(directory=reach_dir)
    if reach.by_type != TOTALS or reach.window_count != 470336555:
        raise RuntimeError('full reachability count mismatch')
    store = TableStore(directory/'tables')
    if checkpoint_path.exists():
        for name, value in checkpoint['io'].items():
            setattr(store, name, value)
    done = {tuple(r['key']) for r in records}
    files = checkpoint['files'] if checkpoint_path.exists() else {
        str(path.relative_to(directory)): file_hash(path) for path in reach_dir.glob('*')}
    for key in sorted(reach.counts, key=key_clock, reverse=True):
        if key_clock(key) < min_clock:
            break
        if key in done:
            continue
        before = {name: getattr(store, name) for name in ('write_bytes', 'write_seconds', 'pack_bytes', 'pack_seconds')}
        record = sweep_key(key, reach.load(key), store, dth)
        # We retain the dense hot set and pack a copy to measure final output I/O.
        store.pack(key, retain_dense=True)
        for path in store.directory.glob(f'{key_name(key)}.*npy'):
            files[str(path.relative_to(directory))] = file_hash(path)
        store.pack_cold(key_clock(key))
        record.update({name: getattr(store, name)-value for name, value in before.items()})
        records.append(record)
        io = {name: getattr(store, name) for name in ('write_bytes', 'write_seconds', 'pack_bytes', 'pack_seconds')}
        write_json(checkpoint_path, {'identity': identity, 'records': records, 'files': files, 'io': io})
        print(json.dumps(record), flush=True)
    for path in reach_dir.glob('*'):
        files[str(path.relative_to(directory))] = file_hash(path)
    report = calibration_report(records, reach, store, time.perf_counter()-start)
    write_json(directory/'report.json', report)
    files['report.json'] = file_hash(directory/'report.json')
    files['checkpoint.json'] = file_hash(checkpoint_path)
    write_json(directory/'manifest.json', {**identity, 'files': files, 'complete': False,
                                          'stopped_at_clock': 3420, 'states': sum(r['states'] for r in records)})
    print(json.dumps(report, indent=2), flush=True)
    return report


def _rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)


def commit_full_checkpoint(directory, identity, records, files, store, clock, elapsed, *, finalize=False):
    directory = Path(directory); cold = []
    for path in store.directory.glob('*.npy'):
        if '.' in path.stem:
            continue
        kind, index = path.stem.split('_'); key = (kind, int(index))
        if finalize or key_clock(key) > clock+540:
            # We retain the dense source until the checkpoint owns both packed files.
            store.pack(key, retain_dense=True)
            for suffix in ('.bits.npy', '.values.npy'):
                packed = store.directory/f'{key_name(key)}{suffix}'
                files[str(packed.relative_to(directory))] = file_hash(packed)
            files.pop(str(path.relative_to(directory)), None)
            cold.append(path)
    io = {name: getattr(store, name) for name in ('write_bytes', 'write_seconds', 'pack_bytes', 'pack_seconds')}
    write_json(directory/'checkpoint.json', {'identity': identity, 'records': records,
               'files': files, 'io': io, 'elapsed_seconds': elapsed,
               'peak_rss_bytes': _rss_bytes(), 'sweep_complete': finalize})
    for path in cold:
        path.unlink()


def run_full(dth_path, directory):
    """We sweep the public game to its clock-720 opening and audit the result."""
    from stl.solver.leap_audit import audit_full
    from stl.solver.leap_lp import NativeFallback
    import highspy
    require_kernel()
    started = time.perf_counter(); directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    dth_path = Path(dth_path); dth = np.load(dth_path, mmap_mode='r')
    if dth.shape != (N, N+1) or dth.dtype != np.float64 or not np.all(dth[:, -1] == -1):
        raise ValueError('DTH table schema mismatch')
    identity = {'schema': 'stl-leap-values-v1', 'builder_sha256': builder_hash(),
                'dth_sha256': file_hash(dth_path), 'saddle_tolerance': 1e-6,
                'minimum_clock': 720, 'mode': 'full', 'python': sys.version, 'numpy': np.__version__,
                'highs': highspy.Highs().version()}
    checkpoint_path = directory/'checkpoint.json'; store = TableStore(directory/'tables')
    records = []; files = {}; elapsed_base = 0.; prior_peak = 0
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text())
        if checkpoint['identity'] != identity:
            raise ValueError('checkpoint source or DTH digest mismatch')
        manifest_path = directory/'manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get('complete') and manifest.get('builder_sha256') == identity['builder_sha256']:
                validate_files(directory, manifest['files'])
                report = json.loads((directory/'report.json').read_text())
                print(json.dumps(report, indent=2), flush=True)
                return report
        files = checkpoint['files']; validate_files(directory, files)
        records = checkpoint['records']; elapsed_base = checkpoint.get('elapsed_seconds', 0.)
        prior_peak = checkpoint.get('peak_rss_bytes', 0)
        for name, value in checkpoint['io'].items():
            setattr(store, name, value)
        # We remove a dense copy left by an interruption after checkpoint commit.
        for path in store.directory.glob('*.npy'):
            if '.' not in path.stem and str(path.relative_to(directory)) not in files:
                packed = str(path.with_suffix('.values.npy').relative_to(directory))
                if packed in files:
                    path.unlink()
    reach_dir = directory/'reachability'
    reach = _load_reach(reach_dir) if (reach_dir/'counts.json').exists() else forward_reachability(directory=reach_dir)
    if reach.by_type != TOTALS or reach.window_count != 470336555:
        raise RuntimeError('full reachability count mismatch')
    if not files:
        files = {str(path.relative_to(directory)): file_hash(path) for path in reach_dir.glob('*')}
    done = {tuple(r['key']) for r in records}
    with NativeFallback() as fallback:
        for key in sorted(reach.counts, key=key_clock, reverse=True):
            if key in done:
                continue
            tick = time.perf_counter()
            record = sweep_key(key, reach.load(key), store, dth, min_clock=720, full=True, fallback=fallback)
            record['sweep_seconds'] = time.perf_counter()-tick
            path = store.path(key); files[str(path.relative_to(directory))] = file_hash(path)
            records.append(record)
            commit_full_checkpoint(directory, identity, records, files, store, key_clock(key),
                                   elapsed_base+time.perf_counter()-started)
            print(json.dumps(record), flush=True)
    commit_full_checkpoint(directory, identity, records, files, store, 720,
                           elapsed_base+time.perf_counter()-started, finalize=True)
    if sum(r['states'] for r in records) != sum(TOTALS.values()) or len(records) != len(reach.counts):
        raise RuntimeError('full sweep coverage mismatch')
    print('Sweep reached the opening; starting packed coverage and Bellman audit.', flush=True)
    audit_started = time.perf_counter()
    audit = audit_full(directory, reach, records, dth)
    write_json(directory/'audit.json', audit)
    rows = {}
    for kind in TOTALS:
        selected = [r for r in records if r['key'][0] == kind]
        seconds = sum(r['kernel_seconds'] for r in selected)
        rows[kind] = {'states': sum(r['states'] for r in selected),
                      'kernel_seconds': seconds,
                      'classes_per_second': sum(r['states'] for r in selected)/seconds,
                      'failures': sum(r['failures'] for r in selected)}
    report = {'root': audit['root'], 'rows': rows, 'states': sum(TOTALS.values()),
              'keys': len(records), 'failures': sum(r['failures'] for r in records),
              'lp_solves': sum(r['lp_solves'] for r in records),
              'native_solves': sum(r.get('native_solves', 0) for r in records),
              'native_seconds': sum(r.get('native_seconds', 0.) for r in records),
              'fallback_seconds': sum(r.get('fallback_seconds', r['lp_seconds']) for r in records),
              'cached_failures': sum(r['cached_failures'] for r in records),
              'kernel_seconds': sum(r['kernel_seconds'] for r in records),
              'lp_seconds': sum(r['lp_seconds'] for r in records),
              'wall_seconds': elapsed_base+time.perf_counter()-started,
              'audit_seconds': time.perf_counter()-audit_started,
              'peak_rss_bytes': max(prior_peak, _rss_bytes()),
              'memmap_write_MB_s': store.write_bytes/store.write_seconds/1e6,
              'pack_MB_s': store.pack_bytes/store.pack_seconds/1e6,
              'lp_audit_samples': audit['lp_samples'],
              'max_bellman_residual': audit['max_bellman_residual'],
              'changed_states': audit['changed_states'],
              'mean_baku_gain_over_state_classes': audit['mean_baku_gain_over_state_classes'],
              'max_baku_gain': audit['max_baku_gain'], 'min_baku_gain': audit['min_baku_gain']}
    write_json(directory/'report.json', report)
    for name in ('checkpoint.json', 'audit.json', 'report.json'):
        files[name] = file_hash(directory/name)
    write_json(directory/'manifest.json', {**identity, 'files': files, 'complete': True,
               'stopped_at_clock': 720, 'states': sum(TOTALS.values()),
               'audit': {'packed_coverage': 'all reachable states', 'lp_samples': audit['lp_samples'],
                         'max_bellman_residual': audit['max_bellman_residual']}})
    print(json.dumps(report, indent=2), flush=True)
    return report


def require_kernel():
    import stl_solver_rs
    source = Path(__file__).resolve().parents[2]/'crates/stl_solver/src/leap.rs'
    if getattr(stl_solver_rs, 'LEAP_SOURCE', None) != source.read_text():
        raise ValueError('stale leap extension; rebuild with maturin develop --release')
    if getattr(stl_solver_rs, 'LEAP_PACKING_SOURCE', None) != source.with_name('leap_packing.rs').read_text():
        raise ValueError('stale packing extension; rebuild with maturin develop --release')
    return stl_solver_rs.sweep_key_rs


def validate_files(directory, files):
    for name, digest in files.items():
        path = Path(directory)/name
        if not path.is_file() or file_hash(path) != digest:
            raise ValueError(f'checkpoint file hash mismatch: {name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dth-path', type=Path, default=Path('src/dth_compact/artifacts/V.npy'))
    parser.add_argument('--output', type=Path, default=Path('src/stl/outputs/leap'))
    parser.add_argument('--min-clock', type=int, default=3420)
    parser.add_argument('--full', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.full:
        if args.min_clock != 720:
            parser.error('--full requires --min-clock 720')
        run_full(args.dth_path, args.output)
    else:
        calibrate(args.dth_path, args.output, args.min_clock)


if __name__ == '__main__':
    main()
