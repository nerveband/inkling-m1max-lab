"""Extract benchmark counters and synthetic output, omitting session logs and host paths."""
import json,pathlib,sys
from summary import summarize
def extract(source):
    root=pathlib.Path(source)
    dst=pathlib.Path(__file__).resolve().parent/'raw'/root.name
    dst.mkdir(exist_ok=True,parents=True)
    for f in root.iterdir():
        if f.name in ('ready.json','monitor.jsonl') or f.suffix=='.jsonl' and f.name in ('2k.jsonl','8k.jsonl','small.jsonl','needles.jsonl') or f.name.endswith(('-copy.json','-fresh.json')):
            (dst/f.name).write_bytes(f.read_bytes())
    scores=summarize(root)
    if scores: (dst/'pi-scores.json').write_text(json.dumps(scores,indent=2)+'\n')
    return scores
if __name__=='__main__':
    for d in sys.argv[1:]:
        scores=extract(d)
        print(pathlib.Path(d).name,[(r['correct'],round(r['wall'],2),round(r['decode_tps'] or 0,2)) for r in scores])
