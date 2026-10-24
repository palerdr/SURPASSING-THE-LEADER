"""We capture the retired Rust leap kernel's outputs as the numba kernel's golden fixture.

Run `uv run python -m stl.tests.make_leap_golden` with `stl_solver_rs` built
from `src/crates/stl_solver`. The script draws seeded stage families, runs
`sweep_key_rs` with the rungs alone and with the production residue options
of `leap_build.RESIDUE`, on square and window stages, and runs
`solve_packing_rs` and the `leap_bench` example on the same stages. It writes
the inputs and the Rust outputs to `data/leap_kernel_golden.npz`.
`test_leap_golden.py` checks the numba kernel against that file. We store the
inputs themselves, so the check does not depend on NumPy's random streams.
"""
from pathlib import Path
import hashlib
import json
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CRATE = ROOT/'src/crates/stl_solver'
OUTPUT = Path(__file__).with_name('data')/'leap_kernel_golden.npz'
SEED = 20261023


def kinked_run(rng, n, drop, f):
    """We slide a 60-second window over one kinked sequence, as consecutive Checker ST does."""
    sigma = -.35+np.cumsum(np.linspace(rng.uniform(.002, .005), rng.uniform(.006, .01), 65+n))
    sigma[int(rng.integers(35, 55)):] -= drop
    start = int(rng.integers(0, 5))
    return np.array([sigma[j:j+60] for j in range(start, start+n)]), np.full(n, f)


def stages(rng):
    """We return seeded stage families that reach every kernel kind."""
    families = []

    def add(name, s, f):
        families.append((name, np.clip(np.asarray(s, float), -1., 1.), np.clip(np.asarray(f, float), -1., 1.)))

    u = np.sort(rng.uniform(-1, 1, (16, 60)), axis=1)
    add('monotone', u, rng.uniform(-1, 1, 16))
    add('monotone-f-above', u*.5, .5+.5*rng.random(16))
    add('iid', rng.uniform(-1, 1, (24, 60)), rng.uniform(-1, 1, 24))
    for _ in range(4):
        add('kinked-run', *kinked_run(rng, 16, rng.uniform(.02, .06), rng.uniform(.5, .7)))
    base = np.cumsum(rng.uniform(.002, .01, (24, 60)), axis=1)-rng.uniform(.2, .5, (24, 1))
    at = rng.integers(1, 60, 24)
    base -= (np.arange(60)[None, :] >= at[:, None])*rng.uniform(.005, .08, (24, 1))
    add('kinked', base, np.minimum(1., base[:, 0]+rng.uniform(.2, 1., 24)))
    levels = np.array([-.5, -.25, 0., .25, .5])
    ties = np.sort(rng.choice(levels, (16, 60)), axis=1)
    ties[:4] = rng.choice(levels, (4, 1))
    add('ties', ties, np.where(rng.random(16) < .5, ties[:, 0], rng.choice(levels, 16)))
    c = rng.uniform(-.5, .5, (16, 1))
    delta = rng.choice([.5e-6, 1e-6*(1-1e-9), 1e-6, 1e-6*(1+1e-9), 1.5e-6, 2e-6], (16, 1))
    near = c+delta*np.sort(rng.random((16, 60)), axis=1)
    near[:, 0] = c[:, 0]; near[:, -1] = (c+delta)[:, 0]
    add('near-gate', near, (c+delta*rng.choice([0., .5, 1., 1.0000001, 2.], (16, 1)))[:, 0])
    flat = np.sort(rng.uniform(-.5, .5, (16, 60)), axis=1)
    tiny = rng.choice([1e-12, 1e-12*(1-1e-6), 1e-12*(1+1e-6), 1e-6, 1e-6*(1-1e-9), 1e-6*(1+1e-9),
                       0., 1e-300, 5e-324], 16)*rng.choice([-1., 1.], 16)
    add('tiny-d', flat, flat[:, 0]-tiny)
    c = rng.uniform(-.3, .3, 16)
    wild = c[:, None]+rng.choice([-.5, .5], (16, 60))*rng.random((16, 60))
    wild[:, 0] = c
    add('overflow', wild, c+rng.choice([1.5e-6, 1e-5, 1e-3, -1.5e-6], 16))
    big = np.sort(rng.uniform(.9, 1., (8, 60)), axis=1)
    add('near-utility-bound', np.where(rng.random((8, 1)) < .5, big, -big[:, ::-1]),
        rng.choice([-1., 1., .999999], 8))
    s = np.concatenate([x[1] for x in families]); f = np.concatenate([x[2] for x in families])
    groups = np.concatenate([[i]*len(x[2]) for i, x in enumerate(families)]).astype(np.float64)
    names = [x[0] for x in families]
    return np.ascontiguousarray(s), np.ascontiguousarray(f), groups, names


def tables(rng, rows=32, width=121, profiles=24, classes=1100):
    """We draw child tables with revival probabilities and nontrivial column maps."""
    succ = rng.uniform(-1, 1, (rows, width)); succ[:, -1] = -1.
    fail = rng.uniform(-1, 1, (rows, 61)); fail[:, -1] = -1.
    cols = np.sort(rng.integers(0, width, (profiles, 60)), axis=1)
    flip = rng.random(profiles) < .3
    cols[flip] = cols[flip, ::-1]
    rev = rng.choice([0., 1., .5, .95*.75], profiles)*np.where(rng.random(profiles) < .5, 1., rng.random(profiles))
    return {'succ_table': np.ascontiguousarray(succ), 'fail_table': np.ascontiguousarray(fail),
            'succ_col': np.ascontiguousarray(cols, np.int32),
            'fail_col': rng.integers(0, 61, profiles).astype(np.int32), 'rev': np.ascontiguousarray(rev),
            'pcs': np.sort(rng.integers(0, profiles, classes)).astype(np.int32),
            'pds': rng.integers(0, rows, classes).astype(np.int32)}


def sweep(fn, inputs, window, options):
    n = len(inputs['pcs']); out = np.full(n, -7.); kind = np.full(n, 77, np.uint8)
    failed = fn(inputs['pcs'], inputs['pds'], inputs['succ_table'], inputs['fail_table'],
                inputs['succ_col'], inputs['fail_col'], inputs['rev'], window, 1e-6, out, kind, **options)
    return out, kind, int(failed)


def runs(s, f, groups, warm, directory):
    """We run the leap_bench example, which the retired benchmark_leap.native_batch called."""
    binary = ROOT/'target/release/examples/leap_bench'
    subprocess.run(['cargo', 'build', '--release', '-p', 'stl_solver_rs', '--example', 'leap_bench'],
                   cwd=ROOT, check=True, capture_output=True)
    src = directory/'runs-input.bin'; dst = directory/'runs-output.bin'
    np.ascontiguousarray(np.c_[s, f, groups], dtype='<f8').tofile(src)
    response = subprocess.run([str(binary), str(src), str(dst), str(int(warm)), '1'],
                              check=True, capture_output=True, text=True)
    result = np.fromfile(dst, dtype='<f8').reshape(-1, 5)
    src.unlink(); dst.unlink()
    return result, int(json.loads(response.stdout)['failed'])


def main():
    import stl_solver_rs
    from stl.solver.leap_build import RESIDUE
    sweep_key_rs = stl_solver_rs.sweep_key_rs
    rng = np.random.default_rng(SEED)
    s, f, groups, names = stages(rng)
    n = len(f)
    # Two full 1024-class chunks and a partial chunk, so seeds restart and spare lanes occur.
    order = np.concatenate([np.tile(np.arange(n), 4), *(rng.permutation(n) for _ in range(4)),
                            np.arange(n)[:n-4]]).astype(np.int32)
    identity = {'succ_table': np.ascontiguousarray(-s), 'fail_table': np.ascontiguousarray(-f[:, None]),
                'succ_col': np.arange(60, dtype=np.int32)[None, :], 'fail_col': np.zeros(1, np.int32),
                'rev': np.ones(1), 'pcs': np.zeros(len(order), np.int32), 'pds': order}
    table = tables(rng)
    production = {'native': True, 'crash': RESIDUE['crash'], 'kink_attempts': RESIDUE['kink_attempts']}
    data = {'stage_s': s, 'stage_f': f, 'stage_groups': groups, 'stage_order': order,
            **{f'table_{k}': v for k, v in table.items()}}
    coverage = {}
    OUTPUT.parent.mkdir(exist_ok=True)
    for label, inputs in (('stages', identity), ('tables', table)):
        for window in (False, True):
            for mode, options in (('rungs', {}), ('production', production)):
                name = f"sweep_{label}_{'window' if window else 'square'}_{mode}"
                out, kind, failed = sweep(sweep_key_rs, inputs, window, options)
                data[f'{name}_value'] = out; data[f'{name}_kind'] = kind; data[f'{name}_failed'] = np.int64(failed)
                coverage[name] = {int(k): int(c) for k, c in zip(*np.unique(kind, return_counts=True))}
    for crash in (False, True):
        for window in (False, True):
            name = f"packing_{'crash' if crash else 'slack'}_{'window' if window else 'square'}"
            out = np.full(n, -7.); kind = np.full(n, 77, np.uint8); support = np.full((n, 2), 7, np.uint64)
            failed = stl_solver_rs.solve_packing_rs(s, f, window, out, kind, crash=crash, support_out=support)
            data.update({f'{name}_value': out, f'{name}_kind': kind, f'{name}_support': support,
                         f'{name}_failed': np.int64(failed)})
            coverage[name] = {int(k): int(c) for k, c in zip(*np.unique(kind, return_counts=True))}
    for warm in (False, True):
        name = f"runs_{'warm' if warm else 'cold'}"
        result, failed = runs(s, f, groups, warm, OUTPUT.parent)
        data[f'{name}_result'] = result; data[f'{name}_failed'] = np.int64(failed)
        coverage[name] = {'failed': failed, 'reused': int(result[:, 3].sum()), 'restarts': int(result[:, 4].sum())}
    sources = {name: hashlib.sha256((CRATE/'src'/name).read_bytes()).hexdigest()
               for name in ('leap.rs', 'leap_packing.rs')}
    sources['examples/leap_bench.rs'] = hashlib.sha256((CRATE/'examples/leap_bench.rs').read_bytes()).hexdigest()
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    rustc = subprocess.run(['rustc', '--version'], capture_output=True, text=True).stdout.strip()
    meta = {'engine': 'stl_solver_rs', 'commit': commit, 'sha256': sources, 'rustc': rustc,
            'seed': SEED, 'numpy': np.__version__, 'families': names, 'production': production,
            'coverage': coverage}
    data['meta'] = np.array(json.dumps(meta, sort_keys=True))
    np.savez_compressed(OUTPUT, **data)
    print(json.dumps(meta, indent=1, sort_keys=True))
    print(f'{OUTPUT}: {OUTPUT.stat().st_size} bytes')


if __name__ == '__main__':
    main()
