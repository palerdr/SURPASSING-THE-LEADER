"""Time HiGHS against rung 2b on the H2_43 row samples with children of the removed partial runs."""
from pathlib import Path
import time,json
import numpy as np
from stl.solver import leap_build as b
from stl.solver.leap_lp import Fallback,SupportFallback

ROOT=Path(__file__).resolve().parents[4]
OUTPUTS=ROOT/'src/stl/outputs'


def main():
    base=OUTPUTS/'leap-full-centered';sample=np.load(base.parent/'leap-rung2b-row-samples.npz');p=b.profiles();key=('H2',43)
    pc=sample['pc'];pd=sample['pd'];groups={}
    for c in np.unique(pc):
        dest=None if not p.rev[c] else b.child_key(key,int(p.st[c])+60)
        groups.setdefault(dest,[]).append(c)
    def table(dest):
        if dest is None:return np.load(ROOT/'src/dth_compact/artifacts/V.npy',mmap_mode='r')
        for folder in ['leap-full-centered','leap-full-ipm','leap-full','leap-full-initial']:
            path=base.parent/folder/'tables'/f'{b.key_name(dest)}.npy'
            if path.exists():return np.load(path,mmap_mode='r')
        raise RuntimeError(str(dest))
    success=b.child_key(key);succ=table(success);reports={};outputs={}
    for label,cls in [('highs',Fallback),('rung2b',SupportFallback)]:
        output=np.empty(len(pc));solved=cached=0;start=time.perf_counter()
        with cls(workers=16)as fallback:
            for dest,checkers in groups.items():
                ix=np.flatnonzero(np.isin(pc,checkers))
                values,n,hits=fallback.solve(success,dest,succ,table(dest),pc[ix],pd[ix],False)
                output[ix]=values;solved+=n;cached+=hits
            reports[label]={'seconds':time.perf_counter()-start,'lp_solves':solved,'cached':cached,
                    'support_hits':getattr(fallback,'support_hits',0),'edge_hits':getattr(fallback,'edge_hits',0),
                    'support_seconds':getattr(fallback,'support_seconds',0),'lp_seconds':getattr(fallback,'lp_seconds',0)}
        outputs[label]=output
    reports['stages']=len(pc);reports['max_difference']=float(np.max(np.abs(outputs['highs']-outputs['rung2b'])))
    reports['speedup']=reports['highs']['seconds']/reports['rung2b']['seconds']
    (base.parent/'leap-rung2b-throughput.json').write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(reports,indent=2),flush=True)

if __name__=='__main__':main()
