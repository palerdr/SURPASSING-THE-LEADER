"""We certify the recurrence crash basis, kink seeds, and in-kernel residue path."""
import numpy as np
import pytest

from stl.solver.leap_oracle import solve_lp, solve_stage
from stl.tests.test_leap_support import S, F, VALUE


def shifted_stages(drop=.04, f=.55, n=30):
    """We slide a 60-second window over one kinked sequence, as consecutive Checker ST does."""
    sigma = -.35+np.cumsum(np.linspace(.004, .008, 100)); sigma[70:] -= drop
    return np.array([sigma[j:j+60] for j in range(11, 11+n)]), np.full(n, f)


def kinked_stages(n, seed):
    rng = np.random.default_rng(seed); s = []; f = []
    for _ in range(n):
        base = np.cumsum(rng.uniform(.002, .01, 60))-rng.uniform(.2, .5)
        base[int(rng.integers(1, 60)):] -= rng.uniform(.005, .08)
        s.append(base); f.append(min(1., base[0]+rng.uniform(.2, 1.)))
    return np.array(s), np.array(f)


def kernel(s, f, window, **options):
    from stl_solver_rs import sweep_key_rs
    n = len(f); out = np.empty(n); kind = np.empty(n, np.uint8)
    failed = sweep_key_rs(np.zeros(n, np.int32), np.arange(n, dtype=np.int32),
                          np.ascontiguousarray(-s), np.ascontiguousarray(-f[:, None]),
                          np.arange(60, dtype=np.int32)[None, :], np.zeros(1, np.int32), np.ones(1),
                          window, 1e-6, out, kind, **options)
    return out, kind, failed


@pytest.mark.parametrize('window', [False, True])
def test_crash_packing_matches_cold_and_lp(window):
    from stl_solver_rs import solve_packing_rs
    s, f = kinked_stages(300, 17)
    s = np.vstack([S, *shifted_stages()[0], s]); f = np.r_[F, shifted_stages()[1], f]
    results = {}
    for crash in (False, True):
        out = np.empty(len(f)); kind = np.empty(len(f), np.uint8)
        assert solve_packing_rs(np.ascontiguousarray(s), f, window, out, kind, crash=crash) == 0
        assert np.all(kind == 3); results[crash] = out
    assert np.max(np.abs(results[True]-results[False])) <= 1e-6
    assert abs(results[True][0]-(max(VALUE, F) if window else VALUE)) < 1e-9
    for i in range(0, len(f), 7):
        assert abs(results[True][i]-solve_lp(s[i], f[i], window).value) <= 1e-8


def test_crash_packing_keeps_unsupported_and_nonfinite_contract():
    from stl_solver_rs import solve_packing_rs
    out = np.empty(2); kind = np.empty(2, np.uint8)
    s = np.ones((2, 60)); f = np.array([0., 1.+1e-13])
    assert solve_packing_rs(s, f, False, out, kind, crash=True) == 2
    assert np.all(kind == 255) and np.isnan(out).all()
    with pytest.raises(ValueError, match='finite'):
        solve_packing_rs(np.full((1, 60), np.nan), np.ones(1), False, out[:1], kind[:1], crash=True)


def test_kink_index_matches_python_authority():
    from stl_solver_rs import kink_index_rs
    from stl.solver.leap_support import kink_index
    rng = np.random.default_rng(5)
    cases = [np.linspace(-.5, .5, 60), np.zeros(60), S, *shifted_stages()[0]]
    tie = np.linspace(0, .3, 60); tie[10:] -= .02; tie[40:] -= .02; cases.append(tie)
    cases += [rng.normal(size=60).cumsum()*.01 for _ in range(300)]
    for s in cases:
        assert kink_index_rs(np.ascontiguousarray(s)) == kink_index(s)
    assert kink_index(tie) == 10 and kink_index(np.zeros(60)) is None
    with pytest.raises(ValueError, match='60'):
        kink_index_rs(np.zeros(59))


def random_mask(rng):
    mask = (1 << 60)-1
    for _ in range(int(rng.integers(0, 4))):
        start = int(rng.integers(0, 60)); end = min(59, start+int(rng.integers(0, 15)))
        for x in range(start, end+1):
            mask &= ~(1 << x)
    return mask


def test_kink_seed_matches_python_authority():
    from stl_solver_rs import kink_seed_rs
    from stl.solver.leap_support import kink_seed
    rng = np.random.default_rng(8)
    for _ in range(3000):
        p, q = random_mask(rng), random_mask(rng)
        old, new = (None if rng.random() < .1 else int(rng.integers(1, 60)) for _ in range(2))
        assert kink_seed_rs(p, q, old, new) == kink_seed(p, q, old, new)
    # A Dropper hole at k*=45 moves right; the mirrored hole at 59-k* moves left.
    p = ((1 << 60)-1) & ~(0b111 << 12) & ~(0b111 << 43)
    q = ((1 << 60)-1) & ~(0b111 << 12) & ~(0b111 << 42)
    moved_p, moved_q = kink_seed(p, q, 45, 46)
    assert moved_p == ((1 << 60)-1) & ~(0b111 << 11) & ~(0b111 << 44)
    assert moved_q == ((1 << 60)-1) & ~(0b111 << 11) & ~(0b111 << 43)
    assert kink_seed(p, q, 45, None) == (p, q) and kink_seed(p, q, 45, 45) == (p, q)
    with pytest.raises(ValueError, match='kink'):
        kink_seed_rs(p, q, 0, 5)
    with pytest.raises(ValueError, match='kink'):
        kink_seed_rs(1 << 60, q, 5, 6)


def unsupported_residue(count, seed):
    """We find residue stages with f < s[0], which the packing LP does not accept."""
    rng = np.random.default_rng(seed); s = []; f = []
    while len(f) < count:
        success = rng.uniform(-1, 1, 60); failure = float(rng.uniform(-1, 1))
        if failure < success[0] and solve_stage(success, failure, False, allow_lp=False).kind == 255:
            s.append(success); f.append(failure)
    return np.array(s), np.array(f)


@pytest.mark.parametrize('crash', [False, True])
@pytest.mark.parametrize('attempts', [0, 1, 4])
@pytest.mark.parametrize('window', [False, True])
def test_kernel_native_residue_matches_lp(crash, attempts, window):
    s, f = kinked_stages(200, 23)
    sequence = shifted_stages(); hard = unsupported_residue(3, 29)
    s = np.vstack([S, *sequence[0], s, hard[0]]); f = np.r_[F, sequence[1], f, hard[1]]
    plain, plain_kind, plain_failed = kernel(s, f, window)
    out, kind, failed = kernel(s, f, window, native=True, crash=crash, kink_attempts=attempts)
    residue = plain_kind == 255; unsupported = f < s[:, 0]
    assert plain_failed == np.count_nonzero(residue)
    # With f > s[0], v60 <= f, so the lifted pure bounds settle every window stage.
    assert np.count_nonzero(residue & ~unsupported) == (0 if window else plain_failed-3)
    assert failed == np.count_nonzero(residue & unsupported)
    assert np.all(kind[residue & unsupported] == 255) and np.isnan(out[residue & unsupported]).all()
    solved = residue & ~unsupported
    assert set(np.unique(kind[solved])) <= {3, 4, 5}
    assert np.array_equal(kind[~residue], plain_kind[~residue])
    assert np.array_equal(out[~residue], plain[~residue])
    if attempts == 0:
        assert not np.isin(kind, [4, 5]).any()
    elif not window:
        assert np.count_nonzero(kind == 4) >= 10
    for i in np.flatnonzero(solved):
        assert abs(out[i]-solve_lp(s[i], f[i], window).value) <= 1e-8


def test_kernel_native_seeds_follow_the_shifted_kink():
    s, f = shifted_stages()
    out, kind, failed = kernel(s, f, False, native=True, crash=True, kink_attempts=1)
    assert failed == 0 and kind[0] == 3 and np.count_nonzero(kind == 4) >= 20
    for i in range(len(f)):
        assert abs(out[i]-solve_stage(s[i], f[i], False).value) <= 1e-8


def test_kernel_native_seeds_restart_in_each_worker_chunk():
    s, f = shifted_stages()
    s = np.tile(s, (80, 1)); f = np.tile(f, 80)
    out, kind, failed = kernel(s, f, False, native=True, crash=True, kink_attempts=1)
    assert failed == 0 and np.all(np.isin(kind, [3, 4]))
    # Each 1024-class chunk starts without a seed, so its first residue class uses the LP.
    assert np.all(kind[[0, 1024, 2048]] == 3)
    assert np.max(np.abs(out-np.tile(out[:30], 80))) <= 1e-6


def test_packing_reports_certified_basis_masks():
    from stl_solver_rs import solve_packing_rs
    s = np.vstack([S, np.ones(60)]); f = np.array([F, 0.])
    out = np.empty(2); kind = np.empty(2, np.uint8); basis = np.full((2, 2), 7, np.uint64)
    assert solve_packing_rs(s, f, False, out, kind, crash=True, support_out=basis) == 1
    p, q = map(int, basis[0])
    assert p and p.bit_count() == q.bit_count() and np.all(basis[1] == 0)
    with pytest.raises(ValueError, match='support shape'):
        solve_packing_rs(s, f, False, out, kind, support_out=np.zeros((1, 2), np.uint64))


def test_kernel_edge_move_limit_matches_python_authority():
    from stl_solver_rs import solve_packing_rs
    from stl.solver.leap_support import kink_index, kink_seed, solve_supported
    rng = np.random.default_rng(250)
    t = S+rng.normal(0, 3e-3, 60); ft = F+rng.normal(0, 3e-3)
    out = np.empty(1); kind = np.empty(1, np.uint8); basis = np.zeros((1, 2), np.uint64)
    assert solve_packing_rs(np.ascontiguousarray(S[None]), np.array([F]), False, out, kind,
                            crash=True, support_out=basis) == 0
    # The kernel seeds the second stage with the first stage's crash basis.
    guess = kink_seed(int(basis[0, 0]), int(basis[0, 1]), kink_index(S), kink_index(t))
    reference = solve_supported(t, ft, False, *guess)
    assert reference.stage.kind == 5 and reference.attempts > 1
    stages = np.vstack([S, t]); fs = np.array([F, ft])
    out, kind, failed = kernel(stages, fs, False, native=True, crash=True, kink_attempts=reference.attempts)
    assert failed == 0 and kind.tolist() == [3, 5]
    assert abs(out[1]-reference.stage.value) < 1e-12
    out, kind, failed = kernel(stages, fs, False, native=True, crash=True, kink_attempts=reference.attempts-1)
    assert failed == 0 and kind.tolist() == [3, 3]
    assert abs(out[1]-solve_lp(t, ft, False).value) <= 1e-8


def test_kernel_native_options_are_validated():
    s, f = shifted_stages(n=2)
    with pytest.raises(ValueError, match='native residue mode'):
        kernel(s, f, False, crash=True)
    with pytest.raises(ValueError, match='native residue mode'):
        kernel(s, f, False, kink_attempts=1)
    seeds = np.zeros((1, 2), np.uint64); masks = np.zeros((2, 2), np.uint64)
    with pytest.raises(ValueError, match='excludes support seeds'):
        kernel(s, f, False, native=True, supports=seeds, support_out=masks)


def test_support_oracle_caps_attempts():
    from stl.solver.leap_support import solve_supported
    from stl.tests.test_leap_support import P, Q
    for i in range(59):
        if ((P >> i) ^ (P >> (i+1))) & 1 and ((Q >> (58-i)) ^ (Q >> (59-i))) & 1:
            wrong = (P ^ (3 << i), Q ^ (3 << (58-i)))
            if solve_supported(S, F, False, *wrong, edges=False) is None:
                break
    result = solve_supported(S, F, False, *wrong)
    assert result is not None and result.attempts > 1
    assert solve_supported(S, F, False, *wrong, max_attempts=result.attempts-1) is None
    assert solve_supported(S, F, False, *wrong, max_attempts=result.attempts).stage.kind == 5
    assert solve_supported(S, F, False, P, Q, max_attempts=0) is None


def test_native_fallback_routes_only_h_keys_to_the_kernel():
    from stl.solver.leap_lp import NativeFallback
    fallback = NativeFallback(1, crash=True, kernel_native=True, kink_attempts=1)
    assert fallback.kernel_options(('H2', 40)) == {'native': True, 'crash': True, 'kink_attempts': 1}
    assert fallback.kernel_options(('REV', 2461)) == {}
    assert NativeFallback(1, crash=True).kernel_options(('H1', 40)) == {}
    with pytest.raises(ValueError, match='kernel residue mode'):
        NativeFallback(1, kink_attempts=1)


def test_sweep_counts_kernel_residue_kinds(tmp_path, monkeypatch):
    from stl.solver import leap_build as b
    monkeypatch.setattr(b, 'N', 8)
    bits = np.zeros((8, 2), np.uint8); bits[0, 0] = 0b1111
    options = {'native': True, 'crash': True, 'kink_attempts': 1}
    def kernel_stub(*args, **kwargs):
        assert kwargs == options
        out, kind = args[-2:]
        out[:] = [.1, .2, .3, np.nan]; kind[:] = [3, 4, 5, 255]
        return 1
    monkeypatch.setattr(b, 'require_kernel', lambda: kernel_stub)
    class Native:
        native_solves = 0; native_seconds = 0.; highs_seconds = 0.
        def kernel_options(self, key):
            return options
        def solve(self, *args):
            n = len(args[-3]); self.native_solves += n
            return np.full(n, .4), 0, 0
    store = b.TableStore(tmp_path)
    monkeypatch.setattr(store, 'load', lambda key: np.zeros((8, 17012)))
    record = b.sweep_key(('H1', 40), bits, store, np.zeros((8, 17012)),
                         min_clock=720, full=True, fallback=Native())
    assert record['failures'] == 4 and record['states'] == 4
    assert record['kernel_native_solves'] == 1 and record['support_solves'] == 2
    assert record['native_solves'] == 1 and record['lp_solves'] == 0


def test_sweep_kernel_residue_matches_python_fallback(tmp_path, monkeypatch):
    from stl.solver import leap_build as b
    from stl.solver.leap_lp import NativeFallback
    from stl.solver.leap_profiles import N, profiles
    p = profiles(); key = ('H1', 40); m = 40
    dest = b.child_key(key, int(p.st[0])+60)
    succ = np.zeros((N, N+1)); succ[:, -1] = -1.
    fail = np.zeros((N, 184)); fail[:, -1] = -1.
    s, f = kinked_stages(m, 41)
    succ[np.arange(m)[:, None], p.succ[0][None, :]] = -s
    fail[np.arange(m), p.idx0[p.fail[0]]] = (1-p.rev[0]-f)/p.rev[0]
    bits = np.zeros((N, (N+7)//8), np.uint8)
    for pd in range(m):
        bits[0, pd//8] |= 1 << (pd % 8)
    store = b.TableStore(tmp_path); children = {b.child_key(key): succ, dest: fail}
    monkeypatch.setattr(store, 'load', lambda child: children[child])
    tables = {}
    def create(k):
        tables[len(tables)] = np.lib.format.open_memmap(tmp_path/f'out{len(tables)}.npy', mode='w+', shape=(N, N+1))
        tables[len(tables)-1][:] = np.nan
        return tables[len(tables)-1]
    monkeypatch.setattr(store, 'create', create)
    records = []
    for options in ({}, {'crash': True, 'kernel_native': True, 'kink_attempts': 1}):
        with NativeFallback(1, **options) as fallback:
            records.append(b.sweep_key(key, bits, store, np.zeros((N, N+1)), min_clock=720, full=True, fallback=fallback))
    first, second = (tables[i][0, :m] for i in range(2))
    assert np.isfinite(first).all() and np.max(np.abs(first-second)) <= 1e-6
    assert records[0]['failures'] == records[1]['failures'] > 0
    assert records[1]['kernel_native_solves']+records[1]['support_solves'] == records[1]['failures']
    for pd in range(0, m, 5):
        assert abs(second[pd]-solve_lp(s[pd], f[pd], False).value) <= 1e-8
