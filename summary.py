"""Summarize Pi turns with generated IDs counted by the endpoint."""
import json,pathlib,statistics,sys
def summarize(root):
    samples=[]
    for p in sorted(pathlib.Path(root).glob('pi*/score.json')):
        row=json.loads(p.read_text());turns=row['turns']
        timed=[]
        for t in turns:
            native=t.get('timings') or {}
            u=t.get('usage') or {}
            if native.get('predicted_ms',0)>0: timed.append(native)
            elif u.get('generation_duration',0)>0:
                timed.append(dict(predicted_n=u['completion_tokens'],predicted_ms=1000*u['generation_duration']))
        count=sum(x['predicted_n'] for x in timed)
        duration=sum(x['predicted_ms'] for x in timed)/1000
        samples.append(dict(task=p.parent.name,correct=row['passed'],wall=row['wall_s'],generated_ids=count,decode_seconds=duration,decode_tps=sum(max(0,x['predicted_n']-1) for x in timed)/duration if duration else None,first_ttft=turns[0].get('ttft') if turns else None,followup_ttft=[t['ttft'] for t in turns[1:]],turns=turns))
    return samples
if __name__=='__main__':
    for arg in sys.argv[1:]: print(json.dumps(summarize(arg),indent=2))
