"""Run read-only rung 2b benchmarks against retained sweep children."""
from pathlib import Path
import contextlib,io,json,platform,statistics,time
import numpy as np
from stl.experiments.rung2b import batch_benchmark,throughput_benchmark
from stl.solver.leap_build import builder_hash,require_kernel,file_hash
from stl.solver.leap_support import solve_supported

OUTPUTS=Path(__file__).resolve().parents[4]/'src/stl/outputs'


def main():
    base=OUTPUTS
    sample=np.load(base/'leap-rung2b-samples.npz')
    kernel=require_kernel();n=len(sample['s'])
    differences=[];gaps=[]
    for s,f,masks,v in zip(sample['s'],sample['f'],sample['supports'],sample['values']):
        result=solve_supported(s,f,False,*map(int,masks),edges=False)
        if result is None:raise RuntimeError('known support failed')
        differences.append(abs(result.stage.value-v));gaps.append(result.stage.gap)
    size=200000;pc=np.arange(size,dtype=np.int32)%n;pd=pc.copy()
    out=np.empty(size);kind=np.empty(size,np.uint8);masks=np.zeros((size,2),np.uint64)
    args=(pc,pd,np.ascontiguousarray(-sample['s']),np.ascontiguousarray(-sample['f'][:,None]),
          np.tile(np.arange(60,dtype=np.int32),(n,1)),np.zeros(n,np.int32),np.ones(n),False,1e-6,out,kind,sample['supports'],masks)
    seconds=[]
    for _ in range(6):
        tick=time.perf_counter();failed=kernel(*args);seconds.append(time.perf_counter()-tick)
        if failed:raise RuntimeError('known-support Rust solve failed')
    report={'source_hash':builder_hash(),'platform':platform.platform(),'workers':16,
        'samples_sha256':file_hash(base/'leap-rung2b-samples.npz'),
        'known_support':{'distinct_stages':n,'replicated_classes':size,'seconds':seconds[1:],
            'median_us_per_class':statistics.median(seconds[1:])*1e6/size,
            'max_lp_difference':float(max(differences)), 'max_gap':float(max(gaps)),
            'max_rust_lp_difference':float(np.max(np.abs(out-sample['values'][pc]))),
            'median_dimension':float(np.median(sample['sizes'])),'max_dimension':int(max(sample['sizes']))}}
    for name,module in (('throughput',throughput_benchmark),('batch',batch_benchmark)):
        trials=[]
        for _ in range(3):
            buf=io.StringIO()
            with contextlib.redirect_stdout(buf):module.main()
            trials.append(json.loads(buf.getvalue()))
        baseline=statistics.median(x['highs']['seconds']for x in trials)
        reduced=statistics.median(x['rung2b']['seconds']for x in trials)
        report[name]={'trials':trials,'median_highs_seconds':baseline,'median_rung2b_seconds':reduced,'speedup':baseline/reduced}
    (base/'leap-rung2b-benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
