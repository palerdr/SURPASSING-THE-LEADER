"""We certify the production packing fallback before resuming the sweep."""
import numpy as np
import pytest
from stl.tests.test_leap_support import S, F, VALUE
from stl.solver.leap_oracle import solve_lp


@pytest.mark.parametrize('n', [1, 17, 2049])
@pytest.mark.parametrize('window', [False, True])
def test_native_binding_matches_lp(n, window):
    from stl_solver_rs import solve_packing_rs
    scales = np.linspace(.7, 1., n)
    s = np.ascontiguousarray(scales[:, None]*S); f = scales*F
    out = np.empty(n); kind = np.empty(n, np.uint8)
    assert solve_packing_rs(s, f, window, out, kind) == 0
    assert np.all(kind == 3)
    expected = scales*(F if window else VALUE)
    assert np.max(np.abs(out-expected)) < 1e-9


def test_native_binding_rejects_bad_inputs_and_reports_unsupported():
    from stl_solver_rs import solve_packing_rs
    out = np.empty(1); kind = np.empty(1, np.uint8)
    with pytest.raises(ValueError, match='finite'):
        solve_packing_rs(np.full((1,60), np.nan), np.ones(1), False, out, kind)
    with pytest.raises(ValueError, match='shape'):
        solve_packing_rs(np.zeros((1,59)), np.ones(1), False, out, kind)
    assert solve_packing_rs(np.ones((1,60)), np.zeros(1), False, out, kind) == 1
    assert kind[0] == 255 and np.isnan(out[0])


def test_native_fallback_sends_only_misses_to_highs(tmp_path):
    from stl.solver.leap_lp import NativeFallback
    from stl.solver.leap_profiles import N, profiles
    p = profiles(); n = 2
    succ = np.lib.format.open_memmap(tmp_path/'succ.npy', mode='w+', dtype=float, shape=(n,N+1))
    fail = np.lib.format.open_memmap(tmp_path/'fail.npy', mode='w+', dtype=float, shape=(n,N+1))
    succ[:] = fail[:] = 0.; succ[:, -1] = fail[:, -1] = -1.
    succ[0, p.succ[0]] = -S; fail[0, p.fail[0]] = (1-p.rev[0]-F)/p.rev[0]
    succ[1, p.succ[0]] = -.2; fail[1, p.fail[0]] = (1-p.rev[0]+.1)/p.rev[0]
    succ.flush(); fail.flush()
    with NativeFallback(workers=1) as fallback:
        out, lps, cached = fallback.solve(('H1',51), ('H1',54), succ, fail,
                                         np.zeros(n,np.int32), np.arange(n,dtype=np.int32), False)
        assert lps == 1 and cached == 0 and fallback.native_solves == 1
        assert abs(out[0]-VALUE) < 1e-9
        assert abs(out[1]-solve_lp(np.full(60,.2),-.1,False).value) < 1e-9


def test_native_fallback_reuses_certified_rev_values(tmp_path):
    from stl.solver.leap_lp import NativeFallback
    from stl.solver.leap_profiles import N, profiles
    p = profiles()
    succ = np.lib.format.open_memmap(tmp_path/'succ.npy',mode='w+',dtype=float,shape=(1,N+1))
    fail = np.lib.format.open_memmap(tmp_path/'fail.npy',mode='w+',dtype=float,shape=(1,184))
    succ[:] = fail[:] = 0.; succ[:,-1] = fail[:,-1] = -1.
    succ[0,p.succ[0]] = -S; fail[0,p.idx0[p.fail[0]]] = (1-p.rev[0]-F)/p.rev[0]
    succ.flush(); fail.flush()
    with NativeFallback(workers=1) as fallback:
        args = (('H1',51),('REV',3400),succ,fail,np.zeros(1,np.int32),np.zeros(1,np.int32),False)
        first, lps, cached = fallback.solve(*args)
        second, lps, cached = fallback.solve(*args)
        assert cached == 1 and lps == 0 and fallback.native_solves == 1
        assert np.array_equal(first,second)


def test_sweep_records_native_and_highs_work_apart(tmp_path, monkeypatch):
    from stl.solver import leap_build as b
    monkeypatch.setattr(b, 'N', 8)
    bits = np.zeros((8,23),np.uint8); bits[0,0] = 7
    def reject(*args):
        args[-2][:] = np.nan; args[-1][:] = 255
        return len(args[0])
    monkeypatch.setattr(b,'require_kernel',lambda:reject)
    class Native:
        native_solves = 0; native_seconds = 0.; highs_seconds = 0.
        def solve(self, *args):
            n = len(args[-3]); self.native_solves += n; self.native_seconds += .1
            return np.zeros(n), 0, 0
    record = b.sweep_key(('REV',3600),bits,b.TableStore(tmp_path),np.zeros((8,17012)),
                         min_clock=720,full=True,fallback=Native())
    assert record['native_solves'] == record['failures'] == 3
    assert record['lp_solves'] == 0 and record['lp_seconds'] == 0.
    assert record['native_seconds'] == .1
