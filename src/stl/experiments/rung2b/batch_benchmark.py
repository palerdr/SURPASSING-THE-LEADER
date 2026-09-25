"""Time HiGHS against rung 2b on the stopped H2_42 batch of the removed leap-full-centered run."""
from pathlib import Path
import json,time
import numpy as np
from stl.solver import leap_build as b
from stl.solver.leap_lp import Fallback,SupportFallback

ROOT=Path(__file__).resolve().parents[4]
OUTPUTS=ROOT/'src/stl/outputs'


def main():
    base=OUTPUTS/'leap-full-centered';key=('H2',42);p=b.profiles();kernel=b.require_kernel()
    bitmap=np.load(base/'reachability/H2_42.npy',mmap_mode='r');table=np.load(base/'tables/H2_42.npy',mmap_mode='r')
    def load(dest):return np.load(ROOT/'src/dth_compact/artifacts/V.npy' if dest is None else base/'tables'/f'{b.key_name(dest)}.npy',mmap_mode='r')
    success_key=b.child_key(key);success=load(success_key);groups={}
    for checker in np.flatnonzero(np.any(bitmap,axis=1)):
        dest=None if not p.rev[checker] else b.child_key(key,int(p.st[checker])+60)
        groups.setdefault(dest,[]).append(int(checker))
    for dest,checkers in groups.items():
        checkers=np.array(checkers,np.int32);reached=np.unpackbits(bitmap[checkers],axis=1,count=b.N,bitorder='little').T;fail=load(dest)
        width=max(64,min(b.N,1048576//len(checkers)))
        for start in range(0,b.N,width):
            pd,local=np.nonzero(reached[start:start+width]);pd+=start
            if not len(pd):continue
            pc=np.ascontiguousarray(checkers[local]);pd=np.ascontiguousarray(pd,dtype=np.int32);missing=~np.isfinite(table[pc,pd])
            if not missing.any():continue
            pc=pc[missing];pd=pd[missing];out=np.empty(len(pc));kind=np.empty(len(pc),np.uint8)
            kernel(pc,pd,success,fail,p.succ,p.fail,p.rev,False,1e-6,out,kind)
            bad=kind==255;pc=pc[bad];pd=pd[bad]
            if not len(pc):continue
            reports={'classes':len(pc),'checker_rows':len(np.unique(pc)),'key':list(key)};outputs={}
            for label,cls in [('highs',Fallback),('rung2b',SupportFallback)]:
                with cls(workers=16)as fallback:
                    tick=time.perf_counter();values,solved,cached=fallback.solve(success_key,dest,success,fail,pc,pd,False);seconds=time.perf_counter()-tick
                    reports[label]={'seconds':seconds,'lp_solves':solved,'cached':cached,'support_hits':getattr(fallback,'support_hits',0),'edge_hits':getattr(fallback,'edge_hits',0),'support_seconds':getattr(fallback,'support_seconds',0),'lp_seconds':getattr(fallback,'lp_seconds',0)};outputs[label]=values
            reports['speedup']=reports['highs']['seconds']/reports['rung2b']['seconds'];reports['max_difference']=float(np.max(np.abs(outputs['highs']-outputs['rung2b'])))
            (base.parent/'leap-rung2b-batch-benchmark.json').write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(reports,indent=2),flush=True);return

if __name__=='__main__':main()
