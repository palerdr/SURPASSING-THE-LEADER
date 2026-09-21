"""We certify reduced support solves against the complete stage matrix."""
import numpy as np
import pytest

S = np.array([-0.1284722791674977, -0.12311238213031911, -0.11783938104329245, -0.11229663346467983, -0.10668796350604809, -0.10083315349960814, -0.09537470669625181, -0.08977130778795045, -0.08339761432886156, -0.07769516443214197, -0.07192164010554408, -0.06616436419483152, -0.06006093717963651, -0.053227655288462405, -0.047598391274268514, -0.041854762653576905, -0.03587453314612943, -0.03003252289938163, -0.023603057751273637, -0.017586484540291714, -0.011219842703011795, -0.005275671293213546, 0.0008117282032785167, 0.007153154358869429, 0.013428233497556619, 0.01974489537252866, 0.026399180249675525, 0.03279969710950498, -0.00047749039889410927, 0.005131084056217761, 0.010850843327395387, 0.01664725521396604, 0.022484377545262724, 0.02847221115882731, 0.03448311001780113, 0.04064462357322546, 0.04664102425924954, 0.05273846839689167, 0.05896383364466534, 0.06518391929773504, 0.07147635977875716, 0.07779174310726783, 0.08423778804274584, 0.09077531440470277, 0.09702645256663872, 0.1033539948162649, 0.1097535240115865, 0.11616685105592794, 0.12271761517131707, 0.1292247029577216, 0.13584770092279674, 0.14236099302853944, 0.14894853172567968, 0.15561553076972012, 0.1622855862936653, 0.17099723315775567, 0.1901691314113193, 0.19683910407608857, 0.20375096839599455, 0.2099309116592808])
F = 0.813835962781682
P = 1152921500328656895
Q = 1152921500588703743
VALUE = 0.1504047730058836


def test_support_oracle_certifies_known_asymmetric_support():
    from stl.solver.leap_support import solve_supported
    from stl.solver.leap_oracle import stage_matrix
    for window in (False, True):
        result = solve_supported(S, F, window, P, Q, edges=False)
        assert result is not None
        matrix = stage_matrix(S, F, window)
        assert np.min(result.p) >= 0 and np.min(result.q) >= 0
        assert abs(result.p.sum()-1) < 1e-12 and abs(result.q.sum()-1) < 1e-12
        assert abs(result.stage.lower-np.min(result.p @ matrix)) < 1e-12
        assert abs(result.stage.upper-np.max(matrix @ result.q)) < 1e-12
        assert result.stage.gap <= 1e-6
        assert abs(result.stage.value-(max(VALUE,F) if window else VALUE)) < 1e-9


def test_support_oracle_rejects_invalid_candidates():
    from stl.solver.leap_support import solve_supported
    assert solve_supported(S, F, False, (1<<60)-1, (1<<60)-1, edges=False) is None
    assert solve_supported(S, F, False, 0, Q) is None
    assert solve_supported(np.ones(60), 1., False, P, Q) is None
    with pytest.raises(ValueError, match='support'):
        solve_supported(S, F, False, 1<<60, Q)
    with pytest.raises(ValueError, match='finite'):
        solve_supported(np.full(60,np.nan), F, False, P, Q)


def test_support_oracle_recovers_paired_edge_move():
    from stl.solver.leap_support import solve_supported
    for i in range(59):
        if ((P>>i) ^ (P>>(i+1)))&1 and ((Q>>(58-i)) ^ (Q>>(59-i)))&1:
            wrong_p=P^(3<<i); wrong_q=Q^(3<<(58-i))
            if solve_supported(S,F,False,wrong_p,wrong_q,edges=False) is None:
                break
    else:
        pytest.fail('fixture must contain a support boundary')
    result=solve_supported(S,F,False,wrong_p,wrong_q,edges=True)
    assert result is not None and result.stage.kind == 5
    assert abs(result.stage.value-VALUE) < 1e-9


@pytest.mark.parametrize('n',[1,15,16,17,1000])
@pytest.mark.parametrize('window',[False,True])
def test_support_kernel_parity_and_spare_lanes(n,window):
    from stl_solver_rs import sweep_key_rs
    from stl.solver.leap_support import solve_supported
    scale=np.linspace(.7,.9,n);offset=.03*np.sin(np.arange(n))
    successes=scale[:,None]*S+offset[:,None];failures=scale*F+offset
    pc=np.arange(n,dtype=np.int32)%17;pd=np.arange(n,dtype=np.int32)
    supports=np.tile(np.array([P,Q],np.uint64),(17,1)); support_out=np.zeros((n,2),np.uint64)
    out=np.empty(n);kind=np.empty(n,np.uint8)
    failed=sweep_key_rs(pc,pd,np.ascontiguousarray(-successes),np.ascontiguousarray(-failures[:,None]),
            np.tile(np.arange(60,dtype=np.int32),(17,1)),np.zeros(17,np.int32),np.ones(17),
            window,1e-6,out,kind,supports,support_out)
    assert failed == 0
    for i in range(n):
        reference=solve_supported(successes[i],failures[i],window,P,Q,edges=False)
        assert abs(out[i]-reference.stage.value) < 1e-9
    if not window:
        assert np.all(kind==4) and np.all(support_out==supports[pc])
    else:
        assert np.all(kind==2)


def test_kernel_defers_row_after_support_miss():
    from stl_solver_rs import sweep_key_rs
    n=3;pc=np.zeros(n,np.int32);pd=np.arange(n,dtype=np.int32)
    out=np.empty(n);kind=np.empty(n,np.uint8);policies=np.zeros((n,2),np.uint64)
    seeds=np.full((1,2),(1<<60)-1,np.uint64)
    args=(pc,pd,np.tile(-S,(n,1)),np.full((n,1),-F),np.arange(60,dtype=np.int32)[None,:],
          np.zeros(1,np.int32),np.ones(1),False,1e-6,out,kind,seeds,policies)
    assert sweep_key_rs(*args,stop_on_support_miss=True)==3
    assert np.array_equal(kind,[255,254,254]) and np.isnan(out).all()
    seeds[0]=[P,Q]
    assert sweep_key_rs(*args,stop_on_support_miss=True)==0
    assert np.all(kind==4) and np.max(np.abs(out-VALUE))<1e-9


def test_support_fallback_seeds_then_reuses_row_policy(tmp_path):
    from stl.solver.leap_lp import SupportFallback
    from stl.solver.leap_profiles import N,profiles
    p=profiles();n=4
    succ=np.lib.format.open_memmap(tmp_path/'succ.npy',mode='w+',dtype=float,shape=(n,N+1))
    fail=np.lib.format.open_memmap(tmp_path/'fail.npy',mode='w+',dtype=float,shape=(n,N+1))
    succ[:]=0.;fail[:]=0.;succ[:,-1]=fail[:,-1]=-1.
    for i in range(n):
        succ[i,p.succ[0]]=-S
        fail[i,p.fail[0]]=(1-p.rev[0]-F)/p.rev[0]
    succ.flush();fail.flush()
    with SupportFallback(workers=1)as fallback:
        out,solved,cached=fallback.solve(('H1',51),('H1',54),succ,fail,np.zeros(n,np.int32),np.arange(n,dtype=np.int32),False)
        assert solved==1 and cached==0
        assert fallback.support_hits==n-1
    assert np.max(np.abs(out-VALUE))<1e-9


def test_paired_isolated_edges_move_in_opposite_directions():
    from stl.solver.leap_support import solve_supported
    from stl_solver_rs import sweep_key_rs
    s=np.array([-0.4026671677760904, -0.3979960627586634, -0.3927380866464558, -0.38796416833988606, -0.3831366054706515, -0.3783119299518909, -0.3734572377052099, -0.3676171318207585, -0.36278131968743343, -0.35790261663936157, -0.35281713229130485, -0.34788195053701454, -0.34844287017916925, -0.3433751252503391, -0.33789479973162684, -0.33277089815970234, -0.327552768099293, -0.3220826666813272, -0.31666186017030373, -0.3111993579252109, -0.3057042620654687, -0.30042788954730754, -0.3375764096216484, -0.3315649066967039, -0.32638188330358975, -0.32113306497675836, -0.31589901523638064, -0.3105279449808905, -0.3051827889678185, -0.29961118942194703, -0.2942310934543996, -0.2887891692158323, -0.2832073538085286, -0.2776241853120468, -0.2719893517885459, -0.26633305271138497, -0.26064268128379425, -0.2543819204709524, -0.24869933364195948, -0.24296433930829275, -0.23713741785742787, -0.23132694097086132, -0.2252380110841241, -0.21932540932331485, -0.213130166910518, -0.20715583260261583, -0.20109987242097038, -0.19488659433670147, -0.18868081201901848, -0.18243767804049366, -0.17615768757219252, -0.16985780388874677, -0.16216707795660473, -0.1560280363296488, -0.14983094055555018, -0.14350194546940354, -0.13723990970217614, -0.13079020559018828, -0.12434683661148044, -0.11768983102715774])
    f=0.25825181643253137
    seeds=np.array([[1152850863123914751, 1152850863128080383]],np.uint64)
    expected=-0.17866045816637277
    result=solve_supported(s,f,False,*map(int,seeds[0]))
    assert result is not None and result.stage.kind==5
    assert abs(result.stage.value-expected)<1e-8
    out=np.empty(1);kind=np.empty(1,np.uint8);masks=np.zeros((1,2),np.uint64)
    failed=sweep_key_rs(np.zeros(1,np.int32),np.zeros(1,np.int32),-s[None,:],np.array([[-f]]),np.arange(60,dtype=np.int32)[None,:],np.zeros(1,np.int32),np.ones(1),False,1e-6,out,kind,seeds,masks)
    assert failed==0 and kind[0]==5 and abs(out[0]-expected)<1e-8


def test_paired_hole_intervals_shift_one_action():
    from stl.solver.leap_support import solve_supported
    from stl_solver_rs import sweep_key_rs
    s=np.array([-0.35443151091336567, -0.3495385377182094, -0.3442039416613912, -0.3392275677743438, -0.3342054563604621, -0.3291819864852916, -0.324135476554298, -0.3183628554448803, -0.31332746229769, -0.30825510278973683, -0.30302401622376585, -0.29790255109574065, -0.2923764167924844, -0.28716480407531875, -0.281665357795775, -0.2764051542602377, -0.27106979567536726, -0.26555236877164307, -0.2600605216604688, -0.2545310551934664, -0.24897076241624222, -0.2437416370635117, -0.29111796631885567, -0.28476778931926416, -0.27939274991776764, -0.2739645103492518, -0.26853903384671485, -0.2630130455284063, -0.2574956837286908, -0.2518201556055816, -0.2462657833943129, -0.24066236214512923, -0.23495692484711755, -0.2292441961435014, -0.2234897568239272, -0.21771385191218395, -0.21190706236032458, -0.20573527484002602, -0.19992565342696395, -0.19407538955691817, -0.18815541118559814, -0.18223907444594561, -0.17613931212467626, -0.1701440524031323, -0.16396826979953272, -0.15792172265497217, -0.15181444586963172, -0.1456064018849137, -0.13939574469760113, -0.1331565188540435, -0.12688717004088534, -0.12060026481410879, -0.11340305136257517, -0.10721037322073801, -0.10098621886329184, -0.09467044350016385, -0.08838935206979687, -0.08198603501497365, -0.07557998799546839, -0.06904133912400101])
    f=0.2693233637785153
    seeds=np.array([[1152921230785970175, 1152921230794326015]],np.uint64)
    expected=-0.1321459733379427
    result=solve_supported(s,f,False,*map(int,seeds[0]))
    assert result is not None and result.stage.kind==5
    assert abs(result.stage.value-expected)<1e-8
    out=np.empty(1);kind=np.empty(1,np.uint8);masks=np.zeros((1,2),np.uint64)
    failed=sweep_key_rs(np.zeros(1,np.int32),np.zeros(1,np.int32),-s[None,:],np.array([[-f]]),np.arange(60,dtype=np.int32)[None,:],np.zeros(1,np.int32),np.ones(1),False,1e-6,out,kind,seeds,masks)
    assert failed==0 and kind[0]==5 and abs(out[0]-expected)<1e-8


def test_support_fallback_bulk_solves_large_systems(tmp_path):
    from stl.solver.leap_lp import SupportFallback
    from stl.solver.leap_profiles import N,profiles
    p=profiles();n=4
    succ=np.lib.format.open_memmap(tmp_path/'succ.npy',mode='w+',dtype=float,shape=(n,N+1))
    fail=np.lib.format.open_memmap(tmp_path/'fail.npy',mode='w+',dtype=float,shape=(n,N+1))
    succ[:]=0.;fail[:]=0.;succ[:,-1]=fail[:,-1]=-1.
    for i in range(n):
        succ[i,p.succ[0]]=-S
        fail[i,p.fail[0]]=(1-p.rev[0]-F)/p.rev[0]
    succ.flush();fail.flush()
    with SupportFallback(workers=1,max_dimension=1)as fallback:
        values,solved,cached=fallback.solve(('H1',51),('H1',54),succ,fail,np.zeros(n,np.int32),np.arange(n,dtype=np.int32),False)
        assert solved==n and cached==0 and fallback.support_hits==0
        assert fallback.large_support_lp==n-1
    assert np.max(np.abs(values-VALUE))<1e-9


def test_support_kernel_keeps_checker_seeds_separate():
    from stl_solver_rs import sweep_key_rs
    n=1000;pc=np.arange(n,dtype=np.int32)%2;pd=np.arange(n,dtype=np.int32)
    seeds=np.array([[P,Q],[(1<<60)-1,(1<<60)-1]],np.uint64)
    out=np.empty(n);kind=np.empty(n,np.uint8);masks=np.zeros((n,2),np.uint64)
    failures=sweep_key_rs(pc,pd,np.tile(-S,(n,1)),np.full((n,1),-F),
            np.tile(np.arange(60,dtype=np.int32),(2,1)),np.zeros(2,np.int32),np.ones(2),
            False,1e-6,out,kind,seeds,masks)
    assert failures==n//2
    assert np.all(kind[::2]==4) and np.max(np.abs(out[::2]-VALUE))<1e-9
    assert np.all(kind[1::2]==255) and np.isnan(out[1::2]).all()


def test_support_kernel_rejects_invalid_masks_and_deferral_order():
    from stl_solver_rs import sweep_key_rs
    pc=np.array([1,0],np.int32);pd=np.array([0,1],np.int32)
    seeds=np.array([[P,Q],[P,Q]],np.uint64);masks=np.zeros((2,2),np.uint64)
    args=(pc,pd,np.tile(-S,(2,1)),np.full((2,1),-F),
          np.tile(np.arange(60,dtype=np.int32),(2,1)),np.zeros(2,np.int32),np.ones(2),
          False,1e-6,np.empty(2),np.empty(2,np.uint8))
    with pytest.raises(ValueError,match='supplied together'):
        sweep_key_rs(*args,supports=seeds)
    with pytest.raises(ValueError,match='shape'):
        sweep_key_rs(*args,supports=seeds[:1],support_out=masks)
    bad=seeds.copy();bad[0,0]=1<<60
    with pytest.raises(ValueError,match='bit out of range'):
        sweep_key_rs(*args,supports=bad,support_out=masks)
    with pytest.raises(ValueError,match='Checker-row order'):
        sweep_key_rs(*args,supports=seeds,support_out=masks,stop_on_support_miss=True)
