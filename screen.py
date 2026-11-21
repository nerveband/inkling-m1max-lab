"""Cheap source of equivalent greedy IDs before promoting any setting to a full leg."""
import json,os,pathlib,sys
from direct import request,panel
from pi_task import run
if __name__=='__main__':
    d=pathlib.Path(sys.argv[1]); d.mkdir(parents=True,exist_ok=True)
    for i,kind in enumerate(('fresh','copy','fresh')):
        r=request(panel(2048,kind,0),64 if i==0 else 256)
        (d/f'{i}-{kind}.json').write_text(json.dumps(r))
        print(kind,r['timings'],flush=True)
    run(d/'pi')
    if os.getenv('BENCH_HELD_SCREEN')=='1':run(d/'pi-held',held=True)
