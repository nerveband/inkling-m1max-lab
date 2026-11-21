"""Join native TensorFold single-stream clocks to matching synthetic API usage."""
import json
import pathlib
import re
import sys
from summary import summarize

def reconcile(root):
    matches=[]
    pattern=r'prompt=(\d+) cached=(\d+).*?tokens=(\d+).*?rounds=(\d+) accepted=(\d+)/(\d+) ttft=([\d.]+) decode=([\d.]+) elapsed=([\d.]+)'
    for line in (root/'server.log').read_text().splitlines():
        m=re.search(pattern,line)
        if m:
            n,cache,emit,rounds,accepted,proposed=map(int,m.groups()[:6])
            first,decode,elapsed=map(float,m.groups()[6:])
            matches.append(dict(prompt_n=n,cache_n=cache,predicted_n=emit,rounds=rounds,accepted=accepted,proposed=proposed,first=first,decode=decode,elapsed=elapsed))
    dst=pathlib.Path(__file__).resolve().parent/'raw'/root.name
    dst.mkdir(exist_ok=True,parents=True)
    (dst/'tensorfold-native.json').write_text(json.dumps(matches,indent=2)+'\n')
    # The two direct probes precede Pi. Then the primary and held-out tasks
    # run sequentially. Retain an unmatched stream as unknown.
    remaining=iter(matches[2:])
    scores=summarize(root)
    for s in scores:
        tt=[]
        for turn in s['turns']:
            u=turn['usage'];row=next(remaining,None)
            if not row or (u.get('prompt_tokens'),u.get('completion_tokens'))!=(row['prompt_n'],row['predicted_n']):
                return
            tt.append(row)
        if tt and sum(r['decode'] for r in tt)>0:
            s['generated_ids']=sum(r['predicted_n'] for r in tt)
            s['decode_seconds']=sum(r['decode'] for r in tt)
            s['decode_tps']=sum(r['predicted_n']-1 for r in tt)/s['decode_seconds']
            s['native_server_clocks']=tt
    if scores:(dst/'pi-scores.json').write_text(json.dumps(scores,indent=2)+'\n')
if __name__=='__main__':reconcile(pathlib.Path(sys.argv[1]))
