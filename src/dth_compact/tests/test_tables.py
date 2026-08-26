"""Step 1 acceptance checks for the profile tables (reference.md §5.1-5.2, §6)."""
import numpy as np
import pytest

import main as m

N = m.N
ALIVE = m.N_ALIVE


@pytest.fixture(scope="module")
def T():
    return m.build_table()


def test_census(T):
    assert T.alive.sum() == 16711
    assert T.st.size == N == 17011
    assert T.alive_id.shape == (300, 301)
    assert T.rev.dtype == np.float64
    assert T.success_children.shape == (N, 60)
    per_ttd = {t: int((T.ttd[: ALIVE] == t).sum()) for t in [0] + list(range(60, 301))}
    assert per_ttd[0] == 240
    for t in range(60, 241):
        assert per_ttd[t] == 241 - t
    for t in range(241, 301):
        assert per_ttd[t] == 0
    # normative order: TTD ascending, ST ascending inside each TTD
    order = np.lexsort((T.st[:ALIVE], T.ttd[:ALIVE]))
    assert np.array_equal(order, np.arange(ALIVE))


def test_spot_ids(T):
    assert T.alive_id[0, 0] == 0
    assert T.alive_id[239, 0] == 239
    assert T.alive_id[0, 60] == 240
    assert T.alive_id[180, 60] == 420
    assert m.profile(0, 0, T) == 0
    assert m.profile(240, 0, T) == 16951
    assert m.profile(299, 0, T) == 17010
    assert m.profile(240, 300, T) == 16951      # dead: TTD discarded
    with pytest.raises(IndexError):
        m.profile(5, 30, T)                      # alive TTD in 1..59 is off-domain


def test_phi_and_buckets(T):
    assert T.phi[T.alive].min() == 0 and T.phi[T.alive].max() == 240
    assert T.phi[~T.alive].min() == 301 and T.phi[~T.alive].max() == 600
    assert np.array_equal(T.phi[~T.alive], T.st[~T.alive] + 301)
    sizes = np.array([len(b) for b in T.bucket])
    assert sizes.size == 601
    assert sizes.sum() == N
    assert (sizes[241:301] == 0).all()
    assert (sizes[301:] == 1).all()
    for v, b in enumerate(T.bucket):
        assert b.dtype == np.int32
        assert (T.phi[b] == v).all()
        assert np.array_equal(b, np.sort(b))


def test_succ_runs(T):
    succ = T.success_children
    assert np.array_equal(succ[0], np.arange(1, 61))              # (0,0): every lag survives
    assert np.array_equal(succ[239], np.arange(16951, 17011))     # (239,0): every lag lands dead
    assert (succ[17010] == m.WIN).all()                           # (299, dead): every lag hits capacity
    # every row: [consecutive alive ids][consecutive dead ids][WIN...]
    kind = np.where(succ < ALIVE, 0, np.where(succ < N, 1, 2))
    assert (np.diff(kind, axis=1) >= 0).all()
    same = (kind[:, 1:] == kind[:, :-1]) & (kind[:, 1:] != 2)
    assert (np.diff(succ, axis=1)[same] == 1).all()
    assert (kind[~T.alive] >= 1).all()                            # dead profiles never have alive children
    # a child is the parent grown by lag, with the same TTD when alive
    lag = np.arange(1, 61)
    valid = succ != m.WIN
    assert (T.st[succ[valid]] == (T.st[:, None] + lag)[valid]).all()
    alive_child = succ < ALIVE
    assert (T.ttd[succ[alive_child]] == np.broadcast_to(T.ttd[:, None], succ.shape)[alive_child]).all()
    # WIN exactly when grown >= 300
    assert np.array_equal(succ == m.WIN, (T.st[:, None] + lag) >= 300)


def test_fail_and_rev(T):
    p = m.profile(0, 0, T)
    assert T.fail_child[p] == 240 and T.rev[p] == pytest.approx(0.95)
    p = m.profile(180, 60, T)
    assert T.fail_child[p] == ALIVE                              # t' = 300 cannot survive: dead sentinel at ST 0
    assert T.rev[p] == pytest.approx(0.178125)
    assert (T.fail_child[~T.alive] == m.WIN).all()
    assert (T.rev[~T.alive] == 0.0).all()
    assert (T.rev[T.alive] > 0.0).all() and (T.rev[T.alive] <= 0.95).all()
    # alive fail-children are revived at ST 0 with TTD = t + s + 60
    fc = T.fail_child[T.alive]
    alive_fc = fc < ALIVE
    assert (T.st[fc[alive_fc]] == 0).all()
    assert (T.ttd[fc[alive_fc]] == (T.ttd[:ALIVE] + T.st[:ALIVE] + 60)[alive_fc]).all()


def test_potential_strictly_increases(T):
    succ, phi = T.success_children, T.phi
    ok = succ != m.WIN
    assert (phi[succ[ok]] > np.broadcast_to(phi[:, None], succ.shape)[ok]).all()
    ok = T.fail_child != m.WIN
    assert (phi[T.fail_child[ok]] > phi[ok]).all()


def test_layer_census(T):
    sizes = np.array([len(b) for b in T.bucket], dtype=np.int64)
    count = np.convolve(sizes, sizes)                  # count[P] = sum_a sizes[a] * sizes[P - a]
    assert count.size == 1201
    assert count.sum() == 289_374_121
    assert count[:601].sum() == 283_297_201
    assert count[601:841].sum() == 6_015_600
    assert count[841:].sum() == 61_320
    assert count.argmax() == 374 and count[374] == 1_678_715


def test_encode(T):
    S = m.State
    assert m.encode_state(S(dropper_st=0, dropper_ttd=0, checker_st=0, checker_ttd=0), T) == 0
    assert m.encode_state(S(dropper_st=240, dropper_ttd=0, checker_st=240, checker_ttd=0), T) == 16951 * N + 16951
    assert m.encode_state(S(dropper_st=200, dropper_ttd=0, checker_st=10, checker_ttd=60), T) == m.profile(10, 60, T) * N + m.profile(200, 0, T)
