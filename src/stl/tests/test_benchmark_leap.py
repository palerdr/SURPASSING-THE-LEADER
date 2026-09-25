"""We require benchmark candidates to preserve the full stage certificate."""
import numpy as np
import pytest

from stl.solver.leap_oracle import solve_lp, stage_matrix
from stl.tests.test_leap_support import S, F, VALUE


def test_packing_formulation_matches_original_game():
    from stl.experiments.benchmark_leap import PackingLP
    solver = PackingLP()
    rng = np.random.default_rng(914)
    stages = [(S, F), (np.linspace(-.4, .7, 60), .5)]
    stages += [(rng.uniform(-.8, .8, 60), .9) for _ in range(12)]
    for s, f in stages:
        result = solver.solve(s, f)
        matrix = stage_matrix(s, f, False)
        assert min(result.p.min(), result.q.min()) >= 0
        assert abs(result.p.sum()-1) < 1e-12 and abs(result.q.sum()-1) < 1e-12
        assert (matrix @ result.q).max()-(result.p @ matrix).min() <= 1e-6
        assert abs(result.value-solve_lp(s, f, False).value) < 1e-7


def test_packing_rejects_invalid_sign_and_nonfinite_stage():
    from stl.experiments.benchmark_leap import PackingLP
    solver = PackingLP()
    with pytest.raises(ValueError, match='diagonal'):
        solver.solve(np.ones(60), .5)
    with pytest.raises(ValueError, match='finite'):
        solver.solve(np.full(60, np.nan), .5)


def test_native_pivots_certify_asymmetric_and_random_stages(tmp_path):
    from stl.experiments.benchmark_leap import native_batch
    rng = np.random.default_rng(231)
    successes = np.vstack([S, S+.001, *[rng.uniform(-.9, .9, 60) for _ in range(24)]])
    failures = np.r_[F, F+.001, np.full(24, .95)]
    for warm in (False, True):
        result, timing = native_batch(successes, failures, np.zeros(len(failures)), tmp_path, warm=warm, threads=1)
        assert timing['failed'] == 0
        assert np.isfinite(result).all() and np.max(result[:, 1]) <= 1e-6
        assert abs(result[0, 0]-VALUE) < 1e-8
        for s, f, row in zip(successes, failures, result):
            assert abs(row[0]-solve_lp(s, f, False).value) < 1e-7


def test_rev_signatures_include_both_children_and_window():
    from stl.experiments.benchmark_leap import rev_signature
    from stl.solver.leap_profiles import profiles
    p = profiles()
    for pc in (0, 29, 59, 1000):
        assert len({rev_signature(c, pc) for c in range(2640, 2700)}) <= 2
    assert rev_signature(3540, 0) == rev_signature(3600, 0)
    assert rev_signature(3539, 0) != rev_signature(3540, 0)
    assert np.array_equal(p.succ[0, 1:], p.succ[1, :-1])


def test_native_rejects_invalid_and_small_diagonals(tmp_path):
    from stl.experiments.benchmark_leap import native_batch
    s = np.zeros((3, 60)); s[0, 3] = np.nan
    result, timing = native_batch(s, np.array([1., -1., 1e-13]), np.zeros(3),
                                  tmp_path, warm=True, threads=1)
    assert timing['failed'] == 3 and np.isnan(result[:, 0]).all()


def test_native_certifies_plateau_after_support_change(tmp_path):
    from stl.experiments.benchmark_leap import native_batch
    plateau = np.r_[.30501604147760514, .3054351558364657, .30585071224070093,
                    .3066373930873052, .3070028766823562, .3076328036693285,
                    .3081612091176772, .30861711515298623, .3090184553403741,
                    .3094632378905324, .3098809265627512, .3102710519555468,
                    .3106331298093312, .29784711454978263, np.full(46, .31127112835744686)]
    s = np.vstack([S, plateau, plateau+.0001, S])
    f = np.array([F, 1., 1.0001, F])
    result, timing = native_batch(s, f, np.zeros(4), tmp_path, warm=True, threads=1)
    assert timing['failed'] == 0 and np.max(result[:, 1]) <= 1e-6
    for row, success, failure in zip(result, s, f):
        assert abs(row[0]-solve_lp(success, failure, False).value) < 1e-8
