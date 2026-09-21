"""We check warm HiGHS solves and exact fallback reuse against scalar LPs."""
import numpy as np
import pytest


@pytest.mark.parametrize('window', [False, True])
def test_warm_highs_matches_scalar_lp(window):
    from stl.solver.leap_lp import WarmLP
    from stl.solver.leap_oracle import solve_lp
    solver = WarmLP(window)
    rng = np.random.default_rng(7941)
    for _ in range(40):
        s = rng.uniform(-1, 1, 60); f = rng.uniform(-1, 1)
        result = solver.solve(s, f)
        expected = solve_lp(s, f, window)
        assert result.gap <= 1e-6
        assert abs(result.value-expected.value) < 1e-8
    with pytest.raises(ValueError, match='finite'):
        solver.solve(np.full(60, np.nan), .5)


def test_fallback_cache_requires_same_children_and_roles():
    from stl.solver.leap_lp import RevivalCache
    cache = RevivalCache()
    pcs = np.array([0, 1], np.int32); pds = np.array([0, 0], np.int32)
    assert np.isnan(cache.get(('H1', 51), ('H1', 54), False, pcs, pds)).all()
    cache.put(('H1', 51), ('H1', 54), False, pcs, pds, np.array([.1, .2]))
    assert np.array_equal(cache.get(('H1', 51), ('H1', 54), False, pcs, pds), [.1, .2])
    assert np.isnan(cache.get(('H1', 51), ('H1', 55), False, pcs, pds)).all()
    assert np.isnan(cache.get(('H1', 51), ('H1', 54), True, pcs, pds)).all()
    assert np.isnan(cache.get(('H1', 52), ('H1', 54), False, pcs, pds)).all()


def test_parallel_fallback_matches_oracle(tmp_path):
    from stl.solver.leap_lp import FallbackPool
    from stl.solver.leap_profiles import profiles, N
    from stl.solver.leap_oracle import solve_lp
    rng = np.random.default_rng(452); p = profiles()
    succ = rng.uniform(-1, 1, (2, N+1)); fail = rng.uniform(-1, 1, (2, N+1))
    succ[:, -1] = fail[:, -1] = -1
    np.save(tmp_path/'succ.npy', succ); np.save(tmp_path/'fail.npy', fail)
    pcs = np.arange(100, dtype=np.int32); pds = np.arange(100, dtype=np.int32) % 2
    with FallbackPool(workers=2) as pool:
        result = pool.solve(tmp_path/'succ.npy', tmp_path/'fail.npy', pcs, pds, False, False)
    expected = [solve_lp(-succ[pd, p.succ[pc]], p.rev[pc]*-fail[pd, p.fail[pc]]+1-p.rev[pc], False).value
                for pc, pd in zip(pcs, pds)]
    assert np.max(np.abs(result-expected)) < 1e-8


def test_fallback_reuses_only_certified_identical_stages(tmp_path):
    from stl.solver.leap_lp import Fallback
    from stl.solver.leap_profiles import N
    for name in ('succ', 'fail'):
        table = np.lib.format.open_memmap(tmp_path/f'{name}.npy', mode='w+', dtype=float, shape=(1, N+1))
        table[:] = .2; table[:, -1] = -1.; table.flush()
    succ = np.load(tmp_path/'succ.npy', mmap_mode='r'); fail = np.load(tmp_path/'fail.npy', mmap_mode='r')
    pcs = pds = np.array([0], np.int32)
    with Fallback(workers=1) as fallback:
        first, solved, cached = fallback.solve(('H1', 51), ('H1', 55), succ, fail, pcs, pds, False)
        assert solved == 1 and cached == 0
        second, solved, cached = fallback.solve(('H1', 51), ('H1', 55), succ, fail, pcs, pds, False)
        assert solved == 0 and cached == 1
        assert np.array_equal(first, second)


def test_scheduler_routes_failed_classes_to_parallel_highs(tmp_path, monkeypatch):
    from stl.solver import leap_build as b
    from stl.solver.leap_lp import Fallback
    rng = np.random.default_rng(87)
    child = np.lib.format.open_memmap(tmp_path/'child.npy', mode='w+', shape=(1, b.N+1), dtype=float)
    child[:] = rng.uniform(-1, 1, child.shape); child[:, -1] = -1.; child.flush()
    output = np.lib.format.open_memmap(tmp_path/'output.npy', mode='w+', shape=(1, 1), dtype=float)
    store = b.TableStore(tmp_path/'tables')
    monkeypatch.setattr(store, 'create', lambda key: output)
    monkeypatch.setattr(store, 'load', lambda key: child)
    def reject(*args):
        args[-2][:] = np.nan; args[-1][:] = 255
        return len(args[0])
    monkeypatch.setattr(b, 'require_kernel', lambda: reject)
    bitmap = np.zeros((b.N, (b.N+7)//8), np.uint8); bitmap[0, 0] = 1
    with Fallback(workers=1) as fallback:
        record = b.sweep_key(('H1', 56), bitmap, store, child, min_clock=720, full=True, fallback=fallback)
    p = b.profiles()
    expected = b.solve_lp(-child[0, p.succ[0]], p.rev[0]*-child[0, p.fail[0]]+1-p.rev[0], False)
    assert abs(output[0, 0]-expected.value) < 1e-8
    assert record['failures'] == record['lp_solves'] == 1
    assert record['cached_failures'] == 0


def test_lp_retries_unknown_simplex_status_with_ipm():
    from stl.solver.leap_oracle import solve_lp
    # We retain the REV[2933] matrix that triggered a HiGHS Unknown status.
    s = np.array([0.5607424418891126, 0.5655518491845215, 0.5703557003984117, 0.5751538917730719, 0.5799463026905874, 0.584732834250341, 0.6483151058350379, 0.7086023579242888, 0.47224604612927623, 0.4767158090516911, 0.48118325337218537, 0.4856483277407043, 0.4901109783322807, 0.49457114897107246, 0.4990287813388775, 0.5034838152362253, 0.5079361889304526, 0.512385839581746, 0.5168327037531836, 0.5212767180054091, 0.5257178195730805, 0.5301559879950309, 0.5345910960381096, 0.5390231125559077, 0.5434519856808143, 0.5478776693322658, 0.5523001279516182, 0.5567193234040602, 0.5611352345647862, 0.565547849433565, 0.5699571654486212, 0.5743631934830495, 0.5787659574049986, 0.5831654953098447, 0.5875618603403369, 0.5919551211579275, 0.596417074977373, 0.6008109522654486, 0.6052003990242172, 0.6095857644792779, 0.6139671369133468, 0.6183446727965163, 0.6227185920631692, 0.6270891738258304, 0.631456746960816, 0.6358216779805593, 0.6401957524718873, 0.6445576496959142, 0.648917109413709, 0.6532747511938127, 0.6576311560582244, 0.661988792656643, 0.6663442467358592, 0.6706995720939722, 0.6750555527746332, 0.679411789005464, 0.6837688775376115, 0.6881268553096092, 0.692485841342741, 0.6968458766397327])
    result = solve_lp(s, 0.5693978099931765, False)
    assert result.gap <= 1e-8
    assert abs(result.value-0.5693978096030541) < 1e-8


def test_audit_retries_unknown_status_with_ipm(monkeypatch):
    from types import SimpleNamespace
    from stl.solver import leap_audit as audit
    original = audit.linprog; methods = []
    def first_unknown(*args, **kwargs):
        methods.append(kwargs['method'])
        if len(methods) == 1:
            return SimpleNamespace(success=False, message='Unknown')
        return original(*args, **kwargs)
    monkeypatch.setattr(audit, 'linprog', first_unknown)
    result = audit.certify_matrix(np.array([[1., -1.], [-1., 1.]]))
    assert methods == ['highs', 'highs-ipm']
    assert result['gap'] < 1e-8 and abs(result['value']) < 1e-8


def test_lp_and_audit_retry_centered_matrix():
    from stl.solver.leap_oracle import solve_lp, stage_matrix
    from stl.solver.leap_audit import certify_matrix
    # We retain the H2[46] matrix that defeated both unshifted HiGHS methods.
    s = np.array([-0.6599336293341775, -0.6598371595049548, -0.6596519227840301, -0.6595688060954965, -0.6594876102392648, -0.6593841293334477, -0.6593010610142643, -0.6592300217172435, -0.6591642708311296, -0.6591013652732551, -0.6590401720591859, -0.6589796617506387, -0.6576852759956346, -0.6576660175796247, -0.657647521262862, -0.6576198116525441, -0.6576045536907242, -0.657587172316132, -0.6575733314790461, -0.6575611009335036, -0.6575499418399386, -0.6575391689487855, -0.6574303903079805, -0.65742772235835, -0.6574255733427329, -0.6574237748632162, -0.6574825326479494, -0.6574284951528107, -0.6574252466275179, -0.6574224647918139, -0.6574135443965261, -0.657412881064743, -0.6574113827295205, -0.657411064120967, -0.6574110056117766, -0.6574110112173694, -0.6574110407250207, -0.6574109953379883, -0.6574109935884253, -0.657410993581152, -0.657410993581152, -0.6574109935811518, -0.6574109935811518, -0.6574109935811518, -0.6574109935811518, -0.6574109935811518, -0.6574109935811518, -0.6574109935811518, -0.6574109935811518, -0.6574109935811518, -0.6574109935811518, -0.6574109935811518, -0.6574109935811518, -0.6574109935811518, -0.6574109935811518, -0.6574109935811518, -0.6574109935811518, -0.6574109935811518, -0.6574109935811518, -0.6574109935811518])
    f = 0.6619631342243604
    result = solve_lp(s, f, False)
    audited = certify_matrix(stage_matrix(s, f, False))
    assert result.gap <= 1e-8 and audited["gap"] <= 1e-8
    assert abs(result.value-(-0.657410993831167)) < 1e-8
    assert abs(result.value-audited["value"]) < 1e-8


def test_lp_retries_first_success_offset_and_returns_support():
    from stl.solver.leap_oracle import solve_lp
    from stl.solver.leap_audit import certify_matrix
    from stl.solver.leap_oracle import stage_matrix
    s = np.array([0.30501604147760514, 0.3054351558364657, 0.30585071224070093, 0.3066373930873052, 0.3070028766823562, 0.3076328036693285, 0.3081612091176772, 0.30861711515298623, 0.3090184553403741, 0.3094632378905324, 0.3098809265627512, 0.3102710519555468, 0.3106331298093312, 0.29784711454978263, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686, 0.31127112835744686])
    f = 0.9999999999999999
    result = solve_lp(s, f, False, support=True)
    audit = certify_matrix(stage_matrix(s, f, False))
    assert result.gap <= 1e-8 and audit["gap"] <= 1e-8
    assert abs(result.value-0.31127112833305104) < 1e-9
    assert result.support is not None and all(0 < x < (1<<60) for x in result.support)


def test_warm_highs_returns_support_on_request():
    from stl.solver.leap_lp import WarmLP
    from stl.solver.leap_support import solve_supported
    s=np.linspace(.7,-.4,60); f=-.8
    result=WarmLP(False).solve(s,f,support=True)
    assert result.support is not None
    recovered=solve_supported(s,f,False,*result.support,edges=False)
    assert recovered is not None and abs(recovered.stage.value-result.value)<1e-9


def test_parallel_pool_returns_seed_supports(tmp_path):
    from stl.solver.leap_lp import FallbackPool
    from stl.solver.leap_profiles import N,profiles
    from stl.solver.leap_oracle import solve_lp
    p=profiles();rng=np.random.default_rng(907)
    a=rng.uniform(-.5,.5,(2,N+1));a[:,-1]=-1
    np.save(tmp_path/'a.npy',a)
    pc=np.arange(12,dtype=np.int32);pd=pc%2
    with FallbackPool(workers=2)as pool:
        values,supports=pool.solve(tmp_path/'a.npy',tmp_path/'a.npy',pc,pd,False,False,support=True)
    assert supports.shape==(len(pc),2) and supports.dtype==np.uint64
    assert np.all(supports>0) and np.all(supports<(1<<60))
    expected=[solve_lp(-a[d,p.succ[c]],p.rev[c]*-a[d,p.fail[c]]+1-p.rev[c],False).value for c,d in zip(pc,pd)]
    assert np.max(np.abs(values-expected))<1e-8


def test_support_refresh_batches_use_available_workers(tmp_path):
    from stl.solver.leap_lp import FallbackPool
    jobs=[]
    class Executor:
        def map(self, function, submitted):
            jobs.extend(submitted)
            return [(np.zeros(len(job[2])),np.ones((len(job[2]),2),np.uint64))for job in jobs]
    pool=FallbackPool(workers=4);pool.executor=Executor()
    ids=np.arange(4,dtype=np.int32)
    pool.solve(tmp_path/'succ.npy',tmp_path/'fail.npy',ids,ids,False,False,support=True)
    assert len(jobs)==4
