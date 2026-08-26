"""The scalar ladder and the fused numba ladder against each other and against LP-certified values."""
import numpy as np
import pytest

import main as m


def random_batch(rng, B):
    return rng.uniform(-1, 1, (B, 60)), rng.uniform(-1, 1, B)


def structured_batch(rng, B):
    return np.sort(rng.uniform(-1, 1, (B, 60)), axis=1), rng.uniform(-1, 1, B)


def ladder(s, f):
    r = np.empty(60)
    q = np.empty(60)
    return m._ladder(np.ascontiguousarray(s, dtype=np.float64), float(f), r, q)


def test_solve_result_post_init():
    r = m.SolveResult(maximin=0.25, minimax=0.25 + 5e-7, kind=m.SUPPORT)
    assert r.certified and r.saddle_gap == pytest.approx(5e-7) and r.value == pytest.approx(0.25 + 2.5e-7)
    assert r.kind == m.SUPPORT
    r = m.SolveResult(maximin=0.0, minimax=1e-3, kind=m.LP)
    assert not r.certified and r.value is None
    r = m.SolveResult(maximin=np.nan, minimax=np.nan, kind=m.LP)
    assert not r.certified and r.value is None
    r = m.SolveResult(maximin=0.3, minimax=0.3 - 1e-15, kind=m.PURE)
    assert r.certified


def test_M_dot_matches_full_matrix():
    rng = np.random.default_rng(3)
    for _ in range(50):
        s, f = rng.uniform(-1, 1, 60), rng.uniform(-1, 1)
        q = rng.uniform(0, 1, 60)
        q /= q.sum()
        assert np.allclose(m._M_dot(s, f, q), m.full_matrix(s, f) @ q, atol=1e-12, rtol=0)


def test_ladder_matches_scalar_rungs():
    rng = np.random.default_rng(5)
    edge_cases = 0
    for maker in (random_batch, structured_batch):
        S, F = maker(rng, 300)
        for i in range(300):
            value, kind = ladder(S[i], F[i])
            r1 = m.try_rung1(S[i], F[i])
            r2 = m.try_rung2(S[i], F[i])
            if kind == m.PURE:
                assert r1.certified and value == pytest.approx(r1.value, abs=1e-12)
            elif kind == m.SUPPORT:
                assert not r1.certified
                if r2 is None or not r2.certified:
                    edge_cases += 1
                else:
                    assert value == pytest.approx(r2.value, abs=1e-9)
            else:
                assert not r1.certified
                if r2 is not None and r2.certified:
                    edge_cases += 1
    assert edge_cases <= 3


def test_scalar_bounds_bracket_lp_value():
    rng = np.random.default_rng(6)
    S, F = structured_batch(rng, 60)
    for i in range(60):
        res = m.try_rung3(S[i], F[i])
        assert res.certified and res.kind == m.LP
        v = res.value
        r1 = m.try_rung1(S[i], F[i])
        assert r1.maximin <= v + 2e-6 and r1.minimax >= v - 2e-6
        r2 = m.try_rung2(S[i], F[i])
        if r2 is not None:
            assert r2.maximin <= v + 2e-6 and r2.minimax >= v - 2e-6


def test_ladder_matches_solve_class():
    rng = np.random.default_rng(7)
    S, F = structured_batch(rng, 200)
    S[0] = 0.5; F[0] = 0.5
    S[1] = 1.0; F[1] = 1.0
    S[2] = np.linspace(-1, 1, 60); F[2] = -1.0
    kinds = []
    for i in range(200):
        value, kind = ladder(S[i], F[i])
        kinds.append(kind)
        expected = m.solve_class(S[i], F[i])
        if kind != m.LP:
            assert abs(value - expected.value) <= m.MAX_SADDLE_GAP + 1e-12
    assert kinds[0] == m.PURE and ladder(S[0], F[0])[0] == 0.5
    assert kinds[1] == m.PURE and ladder(S[1], F[1])[0] == 1.0
    assert kinds[2] == m.PURE and ladder(S[2], F[2])[0] == -1.0
    print(f"\nladder routing on the structured batch (pure, support, lp): {np.bincount(kinds, minlength=3)}")
