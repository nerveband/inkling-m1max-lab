import json,random,statistics,sys
from pathlib import Path
def intervals(a,b):
    rng=random.Random(17)
    ratios=sorted(statistics.median(rng.choices(b,k=len(b)))/statistics.median(rng.choices(a,k=len(a))) for _ in range(5000))
    return (round(ratios[125],4),round(ratios[4874],4))
def summarize(legs):
    sets=[[s for s in json.loads((Path(d)/'leg-summary.json').read_text()) if s['task']!='pi-0'] for d in legs]
    for d,x in zip(legs,sets):
        print(d,'passes',sum(s['correct'] for s in x),'of',len(x))
        for k in ('decode_tps','wall','first_ttft'):
            series=[s[k] for s in x]
            print(k,'median/min/max',statistics.median(series),min(series),max(series))
    for k in ('decode_tps','wall','first_ttft'):
        for i in (0,2):
            print(k,'candidate/control',i,intervals([s[k] for s in sets[i]],[s[k] for s in sets[1]]))
if __name__=='__main__':summarize(sys.argv[1:])
