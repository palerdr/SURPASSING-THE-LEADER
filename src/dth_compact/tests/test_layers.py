"""Value storage, layer enumeration, and the numba kernel against the scalar oracle on one layer."""
import numpy as np
import numba as nb
import pytest

import main as m

N = m.N


@pytest.fixture(scope="module")
def T():
    return m.build_table()


@pytest.fixture(scope="module")
def counts(T):
    sizes = np.array([len(b) for b in T.bucket], dtype=np.int64)
    return np.convolve(sizes, sizes)


def test_storage_layout():
    V = m.values_array()
    assert V.shape == (N, N + 1) and V.dtype == np.float64 and V.flags.c_contiguous
    assert np.isnan(V[:, :N]).all() and (V[:, m.WIN] == -1.0).all()
    K = m.kinds_array()
    assert K.shape == (N, N) and K.dtype == np.uint8 and (K == m.UNSOLVED).all()


def test_layer_pairs(T, counts):
    total = 0
    for P in range(m.MAX_LAYER + 1):
        pcs, pds = m.layer_pairs(P, T)
        assert pcs.dtype == np.int32 and pds.dtype == np.int32
        assert pcs.size == counts[P]
        if pcs.size:
            assert (T.phi[pcs] + T.phi[pds] == P).all()
            keys = pcs.astype(np.int64) * N + pds
            assert np.unique(keys).size == keys.size
        total += pcs.size
    assert total == m.CLASSES
    pcs, pds = m.layer_pairs(374, T)
    a0 = max(0, 374 - m.MAX_PHI)
    while T.bucket[a0].size == 0 or T.bucket[374 - a0].size == 0:
        a0 += 1
    k = T.bucket[a0].size
    assert (pds[:k] == pds[0]).all() and np.array_equal(pcs[:k], T.bucket[a0])


def test_kernel_matches_oracle_on_one_layer(T):
    # smooth synthetic values so rung 2 certifies often; children live in other layers, so kernel writes
    # inside layer 374 never feed the oracle's reads
    rng = np.random.default_rng(0)
    V = 0.9 * np.sin(np.linspace(0, 6, N + 1)[None, :] + np.linspace(0, 3, N)[:, None])
    V += rng.normal(0, 0.005, size=V.shape)
    V[:, m.WIN] = -1.0
    K = m.kinds_array()
    pcs, pds = m.layer_pairs(374, T)
    pick = rng.choice(pcs.size, 3000, replace=False)
    pcs, pds = np.ascontiguousarray(pcs[pick]), np.ascontiguousarray(pds[pick])
    before = V[pcs, pds].copy()
    scratch = np.empty((nb.get_num_threads(), 3, m.LAGS))
    need_lp = np.empty(pcs.size, dtype=np.bool_)
    m._solve_layer_kernel(pcs, pds, V, K, T.success_children, T.fail_child, T.rev, scratch, need_lp)
    kinds = K[pcs, pds]
    assert np.array_equal(kinds == m.UNSOLVED, need_lp)
    routed = np.bincount(kinds[~need_lp], minlength=3)
    assert routed[m.SUPPORT] > 0                       # the smooth data exercises rung 2
    lp_disagreements = 0
    for i in range(pcs.size):
        pc, pd = int(pcs[i]), int(pds[i])
        V_read = V.copy() if False else V              # children untouched by this layer's writes
        s, f = m.class_values(pc, pd, V_read, T)
        if need_lp[i]:
            r1 = m.try_rung1(s, f)
            r2 = m.try_rung2(s, f)
            if r1.certified or (r2 is not None and r2.certified):
                lp_disagreements += 1                  # rounding at the certificate edge; must be rare
            continue
        expected = m.solve_class(s, f)
        assert abs(V[pc, pd] - expected.value) <= m.MAX_SADDLE_GAP + 1e-12
    assert lp_disagreements <= 3
    V[pcs, pds] = before
    print(f"\nkernel routing on 3000 synthetic classes (pure, support, lp): {routed[0]} {routed[1]} {need_lp.sum()}")
