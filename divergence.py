"""Replay the common prefix before a greedy split, asking for target log probabilities."""
import json,pathlib,sys
from direct import panel,post
def trace(control,candidate,out):
    rows=[]
    for name,kind in [('1-copy.json','copy'),('2-fresh.json','fresh')]:
        a=json.loads((control/name).read_text())['ids'];b=json.loads((candidate/name).read_text())['ids']
        i=next((i for i,(u,v) in enumerate(zip(a,b)) if u!=v),None)
        if i is None: continue
        req={'prompt':panel(2048,kind,0)+a[:i],'n_predict':1,'temperature':0,'seed':17,'return_tokens':True,'cache_prompt':False,'n_probs':5,'post_sampling_probs':False}
        resp=post('/completion',req)
        rows.append(dict(panel=kind,at=i,plain=a[i],candidate=b[i],replay=resp.get('tokens'),probabilities=resp.get('completion_probabilities'),timings=resp.get('timings')))
    out.write_text(json.dumps(rows,indent=2))
if __name__=='__main__':trace(*(pathlib.Path(v) for v in sys.argv[1:]))
