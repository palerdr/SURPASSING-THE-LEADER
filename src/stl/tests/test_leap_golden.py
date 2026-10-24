"""We hold the numba leap kernel to the outputs of the retired Rust kernel.

`data/leap_kernel_golden.npz` stores seeded stage families and child tables,
with the values, kinds, failure counts, and basis masks that `stl_solver_rs`
returned for them. The commit that added the fixture holds
`make_leap_golden.py`, which captured it. The fixture covers square and window
stages under the rungs alone and under the production residue options that
`leap_build.RESIDUE` held at the capture, the packing LP from the slack and
crash bases, and the warm and cold packing runs of the leap benchmarks. We run
the production cases with the captured options, so a later change of
`RESIDUE` keeps this parity check. A build reads its child tables as read-only
memmaps, and we hold those to the fixture too. We require equal kinds,
equal failure counts, equal basis masks, and equal values, with NaN in the
same places. Equal means equal as numbers, so a signed zero matches either
sign: `llvm.minnum` and `llvm.maxnum` may return either zero of a tie, the
fixture came from an arm64 Mac, and CI runs on x86-64. A gate on the certified
values would hide a change of rung or basis.
"""
import json
from pathlib import Path

import numpy as np
import pytest

FIXTURE = Path(__file__).with_name('data')/'leap_kernel_golden.npz'


@pytest.fixture(scope='module')
def golden():
    with np.load(FIXTURE) as data:
        return {name: data[name] for name in data.files}


def production(golden):
    return json.loads(str(golden['meta']))['production']


def inputs(golden, label):
    if label == 'tables':
        return {name: golden[f'table_{name}'] for name in
                ('pcs', 'pds', 'succ_table', 'fail_table', 'succ_col', 'fail_col', 'rev')}
    order = golden['stage_order']
    return {'pcs': np.zeros(len(order), np.int32), 'pds': order,
            'succ_table': np.ascontiguousarray(-golden['stage_s']),
            'fail_table': np.ascontiguousarray(-golden['stage_f'][:, None]),
            'succ_col': np.arange(60, dtype=np.int32)[None, :], 'fail_col': np.zeros(1, np.int32),
            'rev': np.ones(1)}


def test_fixture_covers_every_kind(golden):
    meta = json.loads(str(golden['meta']))
    assert meta['engine'] == 'stl_solver_rs'
    kinds = set()
    for name, counts in meta['coverage'].items():
        if name.startswith('sweep_'):
            kinds |= {int(k) for k in counts}
    assert kinds == {0, 1, 2, 3, 4, 255}
    assert FIXTURE.stat().st_size < 500_000


def test_fixture_holds_the_live_production_options(golden):
    from stl.solver.leap_build import RESIDUE
    from stl.solver.leap_lp import NativeFallback
    if NativeFallback(**RESIDUE).kernel_options(('H1', 44)) != production(golden):
        pytest.skip('production options changed since the Rust capture; '
                    'the fixture still checks the kernel under the captured options')


@pytest.mark.parametrize('mode', ['rungs', 'production'])
@pytest.mark.parametrize('window', [False, True])
@pytest.mark.parametrize('label', ['stages', 'tables'])
def test_sweep_matches_golden(golden, label, window, mode):
    from stl.solver.leap_kernel import sweep_batch
    name = f"sweep_{label}_{'window' if window else 'square'}_{mode}"
    args = inputs(golden, label)
    n = len(args['pcs']); out = np.full(n, -7.); kind = np.full(n, 77, np.uint8)
    failed = sweep_batch(args['pcs'], args['pds'], args['succ_table'], args['fail_table'],
                         args['succ_col'], args['fail_col'], args['rev'], window, 1e-6, out, kind,
                         **({} if mode == 'rungs' else production(golden)))
    assert failed == int(golden[f'{name}_failed'])
    assert np.array_equal(kind, golden[f'{name}_kind'])
    assert np.array_equal(out, golden[f'{name}_value'], equal_nan=True)


@pytest.mark.parametrize('mapped', [(True, True), (True, False), (False, True)])
def test_memmapped_tables_share_the_compiled_kernel(golden, tmp_path, mapped):
    from stl.solver import leap_kernel
    name = 'sweep_tables_square_production'
    args = inputs(golden, 'tables')

    def sweep(succ_table, fail_table):
        n = len(args['pcs']); out = np.full(n, -7.); kind = np.full(n, 77, np.uint8)
        failed = leap_kernel.sweep_batch(args['pcs'], args['pds'], succ_table, fail_table,
                                         args['succ_col'], args['fail_col'], args['rev'], False,
                                         1e-6, out, kind, **production(golden))
        return failed, out, kind

    sweep(args['succ_table'], args['fail_table'])
    compiled = set(leap_kernel._sweep_parallel.signatures)
    tables = []
    for table, memmap in zip(('succ_table', 'fail_table'), mapped):
        if memmap:
            np.save(tmp_path/f'{table}.npy', args[table])
            tables.append(np.load(tmp_path/f'{table}.npy', mmap_mode='r'))
        else:
            tables.append(args[table])
    failed, out, kind = sweep(*tables)
    assert set(leap_kernel._sweep_parallel.signatures) == compiled
    assert failed == int(golden[f'{name}_failed'])
    assert np.array_equal(kind, golden[f'{name}_kind'])
    assert np.array_equal(out, golden[f'{name}_value'], equal_nan=True)


@pytest.mark.parametrize('crash', [False, True])
@pytest.mark.parametrize('window', [False, True])
def test_packing_matches_golden(golden, crash, window):
    from stl.solver.leap_kernel import solve_packing
    name = f"packing_{'crash' if crash else 'slack'}_{'window' if window else 'square'}"
    s = golden['stage_s']; f = golden['stage_f']; n = len(f)
    out = np.full(n, -7.); kind = np.full(n, 77, np.uint8); support = np.full((n, 2), 7, np.uint64)
    failed = solve_packing(s, f, window, out, kind, crash=crash, support_out=support)
    assert failed == int(golden[f'{name}_failed'])
    assert np.array_equal(kind, golden[f'{name}_kind'])
    assert np.array_equal(support, golden[f'{name}_support'])
    assert np.array_equal(out, golden[f'{name}_value'], equal_nan=True)


@pytest.mark.parametrize('warm', [False, True])
def test_packing_runs_match_golden(golden, warm):
    from stl.solver.leap_kernel import solve_packing_runs
    name = f"runs_{'warm' if warm else 'cold'}"
    result = solve_packing_runs(golden['stage_s'], golden['stage_f'], golden['stage_groups'], warm)
    assert np.count_nonzero(~np.isfinite(result[:, 0])) == int(golden[f'{name}_failed'])
    assert np.array_equal(result, golden[f'{name}_result'], equal_nan=True)
