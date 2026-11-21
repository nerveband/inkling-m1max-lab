"""Join the per-response engine counters with the measured Pi task for each process."""
import json
from pathlib import Path
import sys
from omlx_native_clock import clocks

def collect(root):
    root=Path(root)
    trials=json.loads((root/'leg-summary.json').read_text())
    result=[]
    for i,trial in enumerate(trials):
        folder=root/f'trial-{i}'
        windows=clocks(folder)
        usage=[t['usage']['completion_tokens'] for t in trial['turns']]
        warm=json.loads((folder/'warm.json').read_text())['usage']['completion_tokens']
        endpoint=[warm,*usage]
        if len(windows)!=len(endpoint):
            raise ValueError('Native window count differs from the request count')
        for j,(w,n) in enumerate(zip(windows,endpoint)):
            if w['generated']-n not in (0,1):
                raise ValueError('Native and API emission counts disagree')
            result.append(dict(trial=i,excluded_first_trial=i==0,request=j,unrelated_warmup=j==0,api_generated=n,**w))
    out=Path(__file__).resolve().parent/'raw'/root.name/'native-cycles.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':
    for path in sys.argv[1:]:collect(path)
