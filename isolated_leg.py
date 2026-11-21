"""Publish clocks for process-isolated Pi trials with a 32-ID warmup and fresh KV."""
import json
import pathlib
import sys

def collect(source):
    src=pathlib.Path(source)
    data=json.loads((src/'leg-summary.json').read_text())
    result=[]
    evidence={}
    for i,row in enumerate(data):
        trial=src/f'trial-{i}'
        warm=json.loads((trial/'warm.json').read_text())
        ready=json.loads((trial/'ready.json').read_text())
        mon=[json.loads(line) for line in (trial/'monitor.jsonl').read_text().splitlines()]
        score=json.loads((trial/'pi/score.json').read_text())
        result.append(dict(**row,warmup=(i==0),server_ready_s=ready['ready_seconds'],unrelated_warmup_usage=warm['usage'],rss_peak_gib=max(x['rss_kib'] for x in mon)/2**20,gpu_system_peak_gib=max(x['gpu_system'] for x in mon)/2**30,swap_delta_mib=max(x['swap_mib'] for x in mon)-ready['initial_swap_mib'],pressure_levels=sorted({x['pressure'] for x in mon})))
        evidence[row['task']]=dict(passed=score['passed'],checks=score.get('checks'),tools=score['tools'],expected_final='PATCH VERIFIED',final_exact=score['answer']=='PATCH VERIFIED')
    root=pathlib.Path(__file__).resolve().parent/'raw'/src.name;root.mkdir(parents=True,exist_ok=True)
    (root/'pi-scores.json').write_text(json.dumps(result,indent=2)+'\n')
    (root/'fixture-checks.json').write_text(json.dumps(evidence,indent=2)+'\n')
if __name__=='__main__':
    for arg in sys.argv[1:]:collect(arg)
