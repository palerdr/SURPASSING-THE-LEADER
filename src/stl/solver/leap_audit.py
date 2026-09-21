"""We audit packed leap values and recover certified opening strategies."""
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from stl.solver.leap_build import POP, TOTALS, child_key, is_window, key_columns, key_name
from stl.solver.leap_profiles import N, WIN, profiles


class PackedReader:
    """We gather values by bitmap rank without restoring dense tables."""
    def __init__(self, directory):
        self.directory = Path(directory)

    @lru_cache(maxsize=32)
    def _open(self, key):
        prefix = self.directory/key_name(key)
        bits = np.load(prefix.with_suffix('.bits.npy'), mmap_mode='r')
        values = np.load(prefix.with_suffix('.values.npy'), mmap_mode='r')
        if bits.dtype != np.uint8 or bits.shape != ((N*key_columns(key)+7)//8,):
            raise ValueError('packed bitmap schema mismatch')
        if values.dtype != np.float64 or values.ndim != 1:
            raise ValueError('packed value schema mismatch')
        totals = np.add.reduceat(POP[bits].astype(np.int64), np.arange(0, len(bits), 4096))
        ranks = np.r_[np.int64(0), np.cumsum(totals)]
        if ranks[-1] != len(values):
            raise ValueError('packed bitmap and value count mismatch')
        return bits, values, ranks

    def get(self, key, rows, cols):
        rows, cols = np.broadcast_arrays(np.asarray(rows, np.int64), np.asarray(cols, np.int64))
        width = key_columns(key)
        if np.any((rows < 0) | (rows >= N) | (cols < 0) | (cols > width)):
            raise ValueError('packed index out of range')
        output = np.full(rows.shape, -1., dtype=np.float64)
        live = cols != width
        if not live.any():
            return output
        flat = rows[live]*width+cols[live]
        byte, bit = flat//8, flat % 8
        bits, values, ranks = self._open(tuple(key))
        if np.any((bits[byte] & (1 << bit)) == 0):
            raise ValueError('unreachable packed state')
        blocks = byte//4096
        offsets = np.empty(len(byte), np.int64)
        for block in np.unique(blocks):
            selected = blocks == block; start = int(block)*4096
            prefix = np.r_[np.int64(0), np.cumsum(POP[bits[start:start+4096]], dtype=np.int64)]
            offsets[selected] = ranks[block]+prefix[byte[selected]-start]
        offsets += POP[(bits[byte] & ((1 << bit)-1)).astype(np.uint8)]
        output[live] = values[offsets]
        if not np.isfinite(output).all():
            raise ValueError('packed values must be finite')
        return output

    def sample(self, key, count, rng):
        bits, values, ranks = self._open(tuple(key))
        targets = rng.choice(len(values), min(count, len(values)), replace=False)
        flat = []
        for target in targets:
            block = int(np.searchsorted(ranks, target, side='right')-1)
            start = block*4096; within = int(target-ranks[block])
            cumulative = np.cumsum(POP[bits[start:start+4096]], dtype=np.int64)
            byte = int(np.searchsorted(cumulative, within, side='right'))
            before = int(cumulative[byte-1]) if byte else 0
            set_bits = np.flatnonzero(np.unpackbits(np.array([bits[start+byte]]), bitorder='little'))
            flat.append((start+byte)*8+int(set_bits[within-before]))
        return np.divmod(np.array(flat, np.int64), key_columns(key))


def audit_packed_key(directory, key, reached, expected, dth=None):
    prefix = Path(directory)/key_name(key); cols = key_columns(key)
    bits = np.load(prefix.with_suffix('.bits.npy'), mmap_mode='r')
    values = np.load(prefix.with_suffix('.values.npy'), mmap_mode='r')
    if bits.dtype != np.uint8 or bits.shape != ((N*cols+7)//8,):
        raise ValueError('packed bitmap schema mismatch')
    if values.dtype != np.float64 or values.shape != (expected,):
        raise ValueError('packed reachability/value count mismatch')
    offset = 0; gain_sum = 0.; changed = 0; min_gain = np.inf; max_gain = -np.inf; largest = None
    for lo in range(0, N, 64):
        mask = np.unpackbits(reached[lo:lo+64], axis=1, count=cols, bitorder='little').astype(bool)
        packed = np.packbits(mask.reshape(-1), bitorder='little')
        if not np.array_equal(bits[lo*cols//8:lo*cols//8+len(packed)], packed):
            raise ValueError(f'packed reachability mismatch: {key}')
        count = int(mask.sum()); data = values[offset:offset+count]; offset += count
        if not np.isfinite(data).all():
            raise ValueError(f'packed values must be finite: {key}')
        if np.any(np.abs(data) > 1+1e-9):
            raise ValueError(f'packed value outside utility range: {key}')
        if dth is not None and count:
            baseline = dth[lo:lo+len(mask)]
            baseline = baseline[:, profiles().s0] if key[0] == 'REV' else baseline[:, :N]
            gain = baseline[mask]-data if key[0] == 'H1' else data-baseline[mask]
            gain_sum += float(gain.sum()); changed += int((gain > 1e-8).sum())
            min_gain = min(min_gain, float(gain.min()))
            if float(gain.max()) > max_gain:
                max_gain = float(gain.max())
                position = int(np.flatnonzero(mask)[int(gain.argmax())])
                row, col = divmod(position, cols)
                largest = [lo+row, int(profiles().s0[col]) if key[0] == 'REV' else col]
    if offset != expected:
        raise ValueError(f'packed reachability count mismatch: {key}')
    # We allow the accumulated prefix error here; each stage still uses 1e-6.
    if dth is not None and min_gain < -1.3e-5:
        raise ValueError(f'Baku action-set monotonicity failed: {key}: {min_gain}')
    return {'states': offset, 'baku_gain_sum': gain_sum, 'changed_states': changed,
            'min_baku_gain': min_gain if dth is not None else None,
            'max_baku_gain': max_gain if dth is not None else None, 'largest_gain_profiles': largest}


def certify_matrix(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError('matrix must contain finite payoffs')
    rows, cols = matrix.shape
    failure = 'opening/audit LP failed the 1e-6 gate'
    attempts = [('highs', None), ('highs-ipm', None),
                ('highs-ipm', float(matrix[-1, 0])), ('highs-ipm', float(matrix[0, 0]))]
    for method, offset in attempts:
        lp_matrix = matrix if offset is None else matrix-offset
        result = linprog(np.r_[np.zeros(rows), -1.],
                         A_ub=np.c_[-lp_matrix.T, np.ones(cols)], b_ub=np.zeros(cols),
                         A_eq=[np.r_[np.ones(rows), 0.]], b_eq=[1.],
                         bounds=[(0, None)]*rows+[(None, None)], method=method,
                         options={'primal_feasibility_tolerance': 1e-9,
                                  'dual_feasibility_tolerance': 1e-9})
        if not result.success:
            failure = f'opening/audit HiGHS failure: {result.message}'
            continue
        drop = np.maximum(result.x[:-1], 0.); check = np.maximum(-result.ineqlin.marginals, 0.)
        drop /= drop.sum(); check /= check.sum()
        lower = float(np.min(drop @ matrix)); upper = float(np.max(matrix @ check))
        if not np.isfinite(upper-lower) or upper-lower > 1e-6:
            failure = 'opening/audit LP failed the 1e-6 gate'
            continue
        return {'value': (lower+upper)/2, 'lower': lower, 'upper': upper, 'gap': upper-lower,
                'drop': drop.tolist(), 'check': check.tolist()}
    raise RuntimeError(failure)


def explicit_matrix(s, f, window):
    # We assemble cells from the action comparison for an independent LP audit.
    drop, check = np.indices((60+int(window), 60))
    matrix = np.full(drop.shape, f, dtype=np.float64)
    success = check >= drop
    matrix[success] = s[(check-drop)[success]]
    return matrix


def stage_inputs(reader, key, pc, pd, dth):
    p = profiles()
    def child_values(dest, columns):
        if dest is None:
            return dth[pd, columns]
        mapped = p.idx0[columns] if dest[0] == 'REV' else columns
        return reader.get(dest, pd, mapped)
    s = -child_values(child_key(key), p.succ[pc])
    f = 1.
    if p.rev[pc]:
        f = p.rev[pc] * -float(child_values(child_key(key, int(p.st[pc])+60), p.fail[pc])) + (1-p.rev[pc])
    return s, f


def audit_full(directory, reach, records, dth, samples_per_key=2):
    directory = Path(directory); reader = PackedReader(directory/'tables')
    rng = np.random.default_rng(93482)
    checked = samples = 0; worst_residual = worst_gap = 0.; results = {}
    for key in reach.counts:
        info = audit_packed_key(reader.directory, key, reach.load(key), reach.counts[key], dth)
        checked += info['states']; results[key_name(key)] = info
        rows, cols = reader.sample(key, 200 if key == ('H2', 57) else samples_per_key, rng)
        for pc, col in zip(rows, cols):
            pd = int(profiles().s0[col]) if key[0] == 'REV' else int(col)
            s, f = stage_inputs(reader, key, int(pc), pd, dth)
            certificate = certify_matrix(explicit_matrix(s, f, is_window(key)))
            stored = float(reader.get(key, pc, col))
            residual = abs(stored-certificate['value'])
            if residual > 1e-6:
                raise RuntimeError(f'Bellman audit failed: {key} {pc} {pd}: {residual}')
            worst_residual = max(worst_residual, residual)
            worst_gap = max(worst_gap, certificate['gap']); samples += 1
        if key[0] == 'H1':
            print(f'audit {key_name(key)}: {info["states"]:,} states', flush=True)
    if checked != sum(TOTALS.values()) or checked != sum(r['states'] for r in records):
        raise RuntimeError('full audit state count mismatch')
    p = profiles(); s, f = stage_inputs(reader, ('H1', 12), 0, 0, dth)
    root = certify_matrix(explicit_matrix(s, f, False))
    root['stored_value'] = float(reader.get(('H1', 12), 0, 0))
    root['hal_win_probability'] = (1+root['stored_value'])/2
    root['baku_win_probability'] = (1-root['stored_value'])/2
    root['dth_value'] = float(dth[0, 0])
    root['baku_gain_probability_points'] = 50*(root['dth_value']-root['stored_value'])
    root['success_payoffs'] = s.tolist(); root['failure_payoff'] = f
    root['dth_policies'] = certify_matrix(explicit_matrix(-dth[0, p.succ[0]],
                            p.rev[0]*-dth[0, p.fail[0]]+(1-p.rev[0]), False))
    return {'states_checked': checked, 'lp_samples': samples,
            'max_bellman_residual': worst_residual, 'max_lp_gap': worst_gap,
            'root': root, 'keys': results,
            'changed_states': sum(r['changed_states'] for r in results.values()),
            'mean_baku_gain_over_state_classes': sum(r['baku_gain_sum'] for r in results.values())/checked,
            'min_baku_gain': min(r['min_baku_gain'] for r in results.values()),
            'max_baku_gain': max(r['max_baku_gain'] for r in results.values())}
