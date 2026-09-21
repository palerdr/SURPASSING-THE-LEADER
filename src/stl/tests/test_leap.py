"""Protect the leap kernel before its first calibrated key sweep."""
from functools import lru_cache
from pathlib import Path
import random
import time

import numpy as np
import pytest
from scipy.optimize import linprog

DTH_PATH = Path('src/dth_compact/artifacts/V.npy')


def lp(s, f, window):
    n = 60 + int(window)
    a = np.full((n, 60), f)
    for d in range(60):
        a[d, d:] = s[:60-d]
    r = linprog(np.r_[np.zeros(n), -1.], A_ub=np.c_[-a.T, np.ones(60)],
                b_ub=np.zeros(60), A_eq=[np.r_[np.ones(n), 0.]], b_eq=[1.],
                bounds=[(0, None)]*n+[(None, None)], method='highs',
                options={'dual_feasibility_tolerance': 1e-9,
                         'primal_feasibility_tolerance': 1e-9})
    assert r.success
    p = np.maximum(r.x[:-1], 0); p /= p.sum()
    q = np.maximum(-r.ineqlin.marginals, 0); q /= q.sum()
    lo, hi = (p @ a).min(), (a @ q).max()
    assert hi - lo <= 1e-6
    return (lo + hi)/2


def test_clock_matches_engine():
    from stl.solver.leap_profiles import child_clock, snap
    from stl.engine.game import Game, Player, Referee
    rng = random.Random(381)
    for clock in [720, 3420, 3479, 3480, 3539, 3540, 3599, 3600, 3601]:
        for half in (1, 2):
            for fail in (False, True):
                g = Game(Player('Hal'), Player('Baku'), Referee())
                g.seed(381); g.game_clock = clock; g.current_half = half
                _, checker = g.get_roles_for_half(half)
                checker.cylinder = rng.randrange(180)
                q = int(checker.cylinder + 60) if fail else 0
                g.resolve_half_round(60 if fail else 1, 1, survived_outcome=True)
                assert child_clock(half, clock, q) == g.game_clock
        g.game_clock = clock; g.snap_clock_to_next_minute()
        assert snap(clock) == g.game_clock


@pytest.mark.parametrize('limit,expected', [(1500, 570612), (1800, 12971150)])
def test_forward_counts(limit, expected):
    from stl.solver.leap_build import forward_reachability
    assert forward_reachability(limit=limit).total == expected


@pytest.mark.slow
def test_forward_counts_full(tmp_path):
    from stl.solver.leap_build import forward_reachability
    r = forward_reachability(directory=tmp_path)
    assert r.total == 9453333117
    assert r.by_type == {'H1': 4180634990, 'H2': 4188677815, 'REV': 1084020312}
    assert r.window_count == 470336555


@pytest.mark.parametrize('window', [False, True])
def test_oracle_vs_lp(window):
    from stl.solver.leap_oracle import solve_stage
    rng = np.random.default_rng(593)
    stages = [(rng.uniform(-1, 1, 60), rng.uniform(-1, 1)) for _ in range(30)]
    stages += [(np.linspace(.8, -.7, 60), -.9), (np.ones(60)*.3, .2)]
    for s, f in stages:
        result = solve_stage(s, f, window)
        assert result.gap <= 1e-6
        assert abs(result.value - lp(s, f, window)) <= 1e-8


def synthetic(n):
    rng = np.random.default_rng(1981)
    pcs = np.arange(n, dtype=np.int32) % 17
    pds = np.arange(n, dtype=np.int32) % 23
    # Monotone payoffs exercise the nonnegative equalizer certificate.
    children = np.sort(rng.uniform(-.7, .7, (23, 61)), axis=1)
    fail = np.full((23, 2), .8)
    cols = np.tile(np.arange(60, dtype=np.int32), (17, 1))
    fcols = np.zeros(17, dtype=np.int32)
    rev = np.linspace(.91, .99, 17)
    return pcs, pds, children, fail, cols, fcols, rev


def run_kernel(args, window=False, tolerance=1e-6):
    from stl_solver_rs import sweep_key_rs
    out = np.empty(len(args[0])); kind = np.empty(len(out), dtype=np.uint8)
    failed = sweep_key_rs(*args, window, tolerance, out, kind)
    return out, kind, failed


@pytest.mark.parametrize('n', [1, 15, 16, 17, 1000])
@pytest.mark.parametrize('window', [False, True])
def test_kernel_vs_oracle(n, window):
    from stl.solver.leap_oracle import solve_stage
    args = synthetic(n)
    out, kind, failed = run_kernel(args, window)
    assert failed == 0
    pc, pd, st, ft, sc, fc, rev = args
    for i in range(n):
        result = solve_stage(-st[pd[i], sc[pc[i]]],
                            rev[pc[i]] * -ft[pd[i], fc[pc[i]]] + 1-rev[pc[i]], window)
        assert abs(out[i]-result.value) <= 1e-9
        assert kind[i] == result.kind


@pytest.mark.parametrize('bad', ['nan', 'tolerance', 'column', 'rev', 'shape', 'row', 'fortran'])
def test_kernel_rejects_bad_input(bad):
    args = list(synthetic(17)); tol = 1e-6
    if bad == 'nan': args[2][0, 0] = np.nan
    if bad == 'tolerance': tol = 1e-5
    if bad == 'column': args[4][0, 0] = 61
    if bad == 'rev': args[6][0] = 1.01
    if bad == 'shape': args[1] = args[1][:-1]
    if bad == 'row': args[1][0] = 23
    if bad == 'fortran': args[2] = np.asfortranarray(args[2])
    with pytest.raises((ValueError, RuntimeError)):
        run_kernel(args, tolerance=tol)


def test_kernel_reports_failures():
    from stl.solver.leap_oracle import solve_stage
    rng = np.random.default_rng(492)
    for _ in range(100):
        s = rng.uniform(-1, 1, 60); f = float(rng.uniform(-1, 1))
        result = solve_stage(s, f, False, allow_lp=False)
        if result.kind == 255: break
    else: pytest.fail('seed must exercise the LP rung')
    args = (np.zeros(1, np.int32), np.zeros(1, np.int32), -s[None, :],
            np.array([[-f]]), np.arange(60, dtype=np.int32)[None, :],
            np.zeros(1, np.int32), np.ones(1))
    out, kind, failed = run_kernel(args)
    assert failed == 1 and kind[0] == 255 and np.isnan(out[0])
    result = solve_stage(s, f, False)
    assert result.kind == 3 and result.gap <= 1e-6
    assert abs(result.value-lp(s, f, False)) <= 1e-9


@pytest.fixture
def dth():
    if not DTH_PATH.exists(): pytest.skip('DTH artifact unavailable')
    return np.load(DTH_PATH, mmap_mode='r')


@pytest.mark.slow
def test_dth_full_parity(dth):
    from stl.solver.leap_profiles import profiles, N
    p = profiles(); start = time.perf_counter(); worst = 0.
    for lo in range(0, N, 64):
        pd = np.repeat(np.arange(lo, min(N, lo+64), dtype=np.int32), N)
        pc = np.tile(np.arange(N, dtype=np.int32), min(64, N-lo))
        out, kind, failed = run_kernel((pc, pd, dth, dth, p.succ, p.fail, p.rev))
        assert failed == 0
        worst = max(worst, float(np.max(np.abs(out-dth[pc, pd]))))
    elapsed = time.perf_counter()-start
    print(f'DTH: {N*N/elapsed:.0f} classes/s, max error {worst:.3g}', flush=True)
    assert worst <= 1e-9


@pytest.mark.parametrize('key', [('H1', 59), ('H2', 56), ('H1', 56)])
def test_exact_dth_keys(key, dth):
    from stl.solver.leap_build import evaluate_states
    from stl.solver.leap_profiles import N
    rng = np.random.default_rng(826)
    pairs = rng.integers(0, N, (200, 2), dtype=np.int32)
    values = evaluate_states(key, pairs, dth)
    assert np.max(np.abs(values-dth[pairs[:, 0], pairs[:, 1]])) <= 1e-9


def test_window_states_vs_lp(dth):
    from stl.solver.leap_build import evaluate_states
    from stl.solver.leap_profiles import profiles, N
    p = profiles(); rng = np.random.default_rng(296)
    pairs = rng.integers(0, N, (200, 2), dtype=np.int32)
    out = evaluate_states(('H2', 57), pairs, dth)
    for value, (pc, pd) in zip(out, pairs):
        s = -dth[pd, p.succ[pc]]
        f = p.rev[pc] * -dth[pd, p.fail[pc]] + 1-p.rev[pc]
        assert abs(value-lp(s, f, True)) <= 1e-8


def test_subtree_vs_naive(dth):
    from stl.solver.leap_build import evaluate_states
    from stl.solver.leap_profiles import profiles, child_clock, N_ALIVE, WIN
    p = profiles()
    @lru_cache(None)
    def naive(half, clock, pc, pd):
        if clock > 3600: return float(dth[pc, pd])
        next_clock = child_clock(half, clock, 0)
        s = np.array([1. if ch == WIN else -naive(3-half, next_clock, pd, int(ch))
                      for ch in p.succ[pc]])
        f = 1.
        if p.rev[pc]:
            f -= p.rev[pc] * (1+naive(3-half, child_clock(half, clock, int(p.st[pc])+60),
                                     pd, int(p.fail[pc])))
        return lp(s, f, half == 2 and 3540 <= clock <= 3600)
    pair = np.array([[N_ALIVE+294, N_ALIVE+294]], dtype=np.int32)
    expected = naive(1, 3180, *map(int, pair[0]))
    assert naive.cache_info().currsize < 20000
    assert abs(evaluate_states(('H1', 53), pair, dth)[0]-expected) <= 1e-8


def test_store_pack_checkpoint_and_floor(tmp_path):
    from stl.solver.leap_build import TableStore, calibrate
    from stl.solver.leap_profiles import N
    store = TableStore(tmp_path)
    a = store.create(('REV', 3420))
    a[3, 4] = .25; a[8, 9] = -.4; a.flush()
    store.pack(('REV', 3420))
    restored = store.load(('REV', 3420))
    assert restored.shape == (N, 184)
    assert restored[3, 4] == .25 and restored[8, 9] == -.4
    assert np.isnan(restored[0, 0]) and np.all(restored[:, -1] == -1.)
    with pytest.raises(ValueError, match='3420'):
        calibrate(DTH_PATH, tmp_path, min_clock=3419)


def test_kernel_rounding_mode():
    import ctypes
    import platform
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        pytest.skip('this test uses Darwin ARM64 fenv constants')
    libc = ctypes.CDLL(None)
    original = libc.fegetround()
    try:
        assert libc.fesetround(0x400000) == 0
        with pytest.raises(ValueError, match='floating-point'):
            run_kernel(synthetic(1))
    finally:
        assert libc.fesetround(original) == 0


def test_kernel_source_guard(monkeypatch):
    import stl_solver_rs
    from stl.solver.leap_build import require_kernel
    require_kernel()
    monkeypatch.setattr(stl_solver_rs, 'LEAP_SOURCE', 'stale source')
    with pytest.raises(ValueError, match='stale'):
        require_kernel()


def test_reachability_cache_hashes(tmp_path):
    from stl.solver.leap_build import validate_files, file_hash
    path = tmp_path/'reach.npy'; path.write_bytes(b'reach bitmap')
    hashes = {'reach.npy': file_hash(path)}
    validate_files(tmp_path, hashes)
    path.write_bytes(b'changed bitmap')
    with pytest.raises(ValueError, match='hash mismatch'):
        validate_files(tmp_path, hashes)


def test_hot_horizon(tmp_path):
    from stl.solver.leap_build import TableStore
    store = TableStore(tmp_path)
    a = store.create(('REV', 3600)); a[0, 0] = .75; a.flush(); del a
    assert store.pack_cold(3060) == []
    assert store.path(('REV', 3600)).exists()
    assert store.pack_cold(3059) == [('REV', 3600)]
    assert not store.path(('REV', 3600)).exists()
    assert store.load(('REV', 3600))[0, 0] == .75


def test_projection_uses_measured_spread(tmp_path):
    from stl.solver.leap_build import calibration_report, Reachability, TableStore
    store = TableStore(tmp_path)
    store.write_bytes = store.pack_bytes = 600
    store.write_seconds = store.pack_seconds = 3.
    records = []
    for kind in ('H1', 'H2', 'REV'):
        for index, rate in [(1, 100), (2, 200)]:
            records.append({'key': [kind, index], 'states': 100, 'kernel_seconds': 100/rate,
                            'classes_per_second': rate, 'failures': 0, 'lp_seconds': 0,
                            'write_bytes': 100, 'write_seconds': 100/rate,
                            'pack_bytes': 100, 'pack_seconds': 100/rate})
    reach = Reachability({('H1', 12): 500}, None)
    report = calibration_report(records, reach, store, 4.)
    assert report['memmap_write_rate_range_MB_s'] == [0.0001, 0.0002]
    assert report['pack_rate_range_MB_s'] == [0.0001, 0.0002]
    assert report['projection_seconds']['low'] < report['projection_seconds']['high']
    assert report['seconds_per_lp'] is None


def test_projection_separates_window_keys(tmp_path):
    from stl.solver.leap_build import calibration_report, Reachability, TableStore
    store = TableStore(tmp_path)
    store.write_bytes = store.pack_bytes = 400
    store.write_seconds = store.pack_seconds = 4.
    records = []
    for key, rate in [(('H1', 59), 100), (('H2', 56), 20),
                      (('H2', 57), 100), (('REV', 3540), 100)]:
        records.append({'key': list(key), 'states': 100, 'kernel_seconds': 100/rate,
                        'classes_per_second': rate, 'failures': 0, 'lp_seconds': 0,
                        'write_bytes': 100, 'write_seconds': 1,
                        'pack_bytes': 100, 'pack_seconds': 1})
    report = calibration_report(records, Reachability({('H2', 12): 500}, None), store, 4.)
    assert report['rows']['H2']['classes_per_second'] == 20
    assert report['rows']['H2']['states'] == 100
    assert report['window_H2']['states'] == 100


def test_reachability_source_guard(tmp_path):
    import json
    from stl.solver.leap_build import forward_reachability, _load_reach
    forward_reachability(limit=720, directory=tmp_path)
    assert _load_reach(tmp_path).total == 1
    path = tmp_path/'manifest.json'
    manifest = json.loads(path.read_text()); manifest['builder_sha256'] = 'stale'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='source mismatch'):
        _load_reach(tmp_path)


def test_oracle_recurrence_overflow_uses_lp():
    from stl.solver.leap_oracle import solve_stage
    s = np.full(60, -.9); s[0] = 2e-6
    failed = solve_stage(s, 0., False, allow_lp=False)
    assert failed.kind == 255 and np.isnan(failed.value)
    result = solve_stage(s, 0., False)
    assert result.kind == 3 and result.gap <= 1e-6
    assert abs(result.value-lp(s, 0., False)) <= 1e-9
    args = (np.zeros(1, np.int32), np.zeros(1, np.int32), -s[None, :],
            np.zeros((1, 1)), np.arange(60, dtype=np.int32)[None, :],
            np.zeros(1, np.int32), np.ones(1))
    out, kind, count = run_kernel(args)
    assert count == 1 and kind[0] == 255 and np.isnan(out[0])


def test_scheduler_sends_kernel_failures_to_highs(tmp_path, monkeypatch, dth):
    from stl.solver import leap_build as build
    from stl.solver.leap_profiles import N
    from stl.solver.leap_oracle import solve_lp
    calls = []
    def fail_kernel(*args):
        out, kind = args[-2:]
        out[:] = np.nan; kind[:] = 255
        return len(out)
    def highs(*args):
        calls.append(args[2])
        return solve_lp(*args)
    def forbid_retry(*args, **kwargs):
        raise AssertionError('Rust failures must enter HiGHS')
    monkeypatch.setattr(build, 'require_kernel', lambda: fail_kernel)
    monkeypatch.setattr(build, 'solve_stage', forbid_retry)
    monkeypatch.setattr(build, 'solve_lp', highs)
    bits = np.zeros((N, 23), np.uint8); bits[0, 0] = 1
    record = build.sweep_key(('REV', 3600), bits, build.TableStore(tmp_path), dth)
    assert calls == [True]
    assert record['states'] == record['failures'] == 1
    assert record['lp_seconds'] > 0
