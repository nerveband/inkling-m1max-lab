"""Summarize scored Pi turns and the instrumented direct endpoint."""
import json
import pathlib
import statistics
import sys

def stats(values):
    data=[x for x in values if x is not None]
    return [round(z,3) for z in (statistics.median(data),min(data),max(data))] if data else None

def report(folder):
    from summary import summarize
    root=pathlib.Path(folder)
    rows=[r for r in summarize(root) if r['task']!='pi-0']
    output={'n':len(rows),'passes':sum(r['correct'] for r in rows)}
    for key in ['wall','decode_tps','first_ttft','generated_ids']:
        output[key]=stats([r[key] for r in rows])
    follow=[t for row in rows for t in row['turns'][1:]]
    output['followup_ttft']=stats([t['ttft'] for t in follow])
    output['first_cached_ids']=stats([(r['turns'][0]['usage'].get('prompt_tokens_details') or {}).get('cached_tokens') for r in rows if r['turns']])
    output['prefill_followup_cached_ids']=stats([(t['usage'].get('prompt_tokens_details') or {}).get('cached_tokens') for t in follow])
    output['turn_generated_rate']=[]
    for r in rows:
        rates=[]
        for t in r['turns']:
            n=t.get('timings') or {};u=t.get('usage') or {}
            dt=n.get('predicted_ms',0)/1000 or u.get('generation_duration',0)
            count=n.get('predicted_n') or u.get('completion_tokens')
            rates.append(round((count-1)/dt,3) if count and dt else None)
        output['turn_generated_rate'].append(rates)
    for label in ('2k','8k'):
        p=root/(label+'.jsonl')
        if not p.exists():continue
        data=[r for l in p.read_text().splitlines() if not (r:=json.loads(l))['warmup']]
        output[label]={}
        for kind in ('copy','fresh'):
            group=[r for r in data if r['kind']==kind]
            output[label][kind]={k:stats([r['timings'][k] for r in group]) for k in ('prompt_per_second','predicted_per_second')}
            output[label][kind]['ttft']=stats([r['ttft_s'] for r in group])
    p=root/'monitor.jsonl'
    if p.is_file():
        mem=[json.loads(l) for l in p.read_text().splitlines()];ready=json.loads((root/'ready.json').read_text())
        if mem:output['resource']={'ready_s':ready['ready_seconds'],'rss_peak_gib':round(max(r['rss_kib'] for r in mem)/2**20,3),'gpu_system_peak_gib':round(max(r['gpu_system'] for r in mem)/2**30,3),'swap_delta_mib':max(r['swap_mib'] for r in mem)-ready['initial_swap_mib']}
    return output
if __name__=='__main__':
    for folder in sys.argv[1:]:print(pathlib.Path(folder).name,json.dumps(report(folder)))
