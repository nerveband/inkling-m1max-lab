"""Summarize Pi turns with generated IDs counted by the endpoint."""
import json,pathlib,statistics,sys
def summarize(root):
    samples=[]
    for p in sorted(pathlib.Path(root).glob('pi*/score.json')):
        row=json.loads(p.read_text());turns=row['turns']
        timed=[t['timings'] for t in turns if t.get('timings',{}).get('predicted_ms',0)>0]
        count=sum(x['predicted_n'] for x in timed)
        duration=sum(x['predicted_ms'] for x in timed)/1000
        samples.append(dict(task=p.parent.name,correct=row['passed'],wall=row['wall_s'],generated_ids=count,decode_seconds=duration,decode_tps=count/duration if duration else None,first_ttft=turns[0].get('ttft') if turns else None,followup_ttft=[t['ttft'] for t in turns[1:]],turns=turns))
    return samples
if __name__=='__main__':
    for arg in sys.argv[1:]: print(json.dumps(summarize(arg),indent=2))
