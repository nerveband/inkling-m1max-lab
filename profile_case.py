"""Short native profiler input on the same two 2K visible-code prompts."""
import json
import pathlib
import sys
from direct import panel,request
from divergence import trace

if __name__=='__main__':
    dest=pathlib.Path(sys.argv[1])
    for kind in ('copy','fresh'):
        r=request(panel(2048,kind,0),256)
        (dest/('profile-'+kind+'.json')).write_text(json.dumps(r))
        print(kind,r['timings'],flush=True)
    if len(sys.argv)==4:
        import os
        os.environ.pop('BENCH_VISIBLE',None)
        trace(pathlib.Path(sys.argv[2]),pathlib.Path(sys.argv[3]),dest/'divergence.json')
