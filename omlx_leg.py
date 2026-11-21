"""Repeat one cache-enabled oMLX arm with independent, identically warmed processes."""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from summary import summarize

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=pathlib.Path,required=True)
    p.add_argument('--cache-root',type=pathlib.Path,required=True)
    p.add_argument('--depth',type=int,required=True)
    p.add_argument('--context',type=int,default=32768)
    p.add_argument('results',type=pathlib.Path)
    a=p.parse_args()
    root=pathlib.Path(__file__).resolve().parent
    target=a.results.resolve();target.mkdir(parents=True)
    cache_root=a.cache_root.resolve();cache_root.mkdir(exist_ok=True,parents=True)
    if not os.getenv('OMLX_APP'):p.error('Set OMLX_APP to the pinned oMLX 0.8.2 app bundle.')
    scores=[]
    for i in range(7):
        trial=target/f'trial-{i}'
        with tempfile.TemporaryDirectory(prefix='inkling-kv-',dir=cache_root) as scratch:
            command=[sys.executable,'-u',str(root/'omlx_run.py'),'--weights',str(a.weights.resolve()),'--state',str(target/f'state-{i}'),'--cache',str(pathlib.Path(scratch)/'blocks'),'--context',str(a.context),'--depth',str(a.depth),'--job','single',str(trial)]
            subprocess.run(command,check=True)
        row=summarize(trial)[0]
        if row['turns'][0]['usage']['prompt_tokens_details']['cached_tokens']!=0:
            raise RuntimeError('Initial task prefix unexpectedly hit a previous KV cache.')
        row['task']=f'pi-{i}';scores.append(row)
        (target/'leg-summary.json').write_text(json.dumps(scores,indent=2)+'\n')
