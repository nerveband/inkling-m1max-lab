"""Compare exactly matched direct GGUF request panels and count output-ID agreement."""
import argparse
import json
from pathlib import Path
from statistics import median

def rows(path):
    return [row for line in path.read_text().splitlines() if not (row := json.loads(line))['warmup']]

def spread(values):
    return dict(median=median(values),low=min(values),high=max(values))

def compare(paths,filename):
    legs=[rows(p/filename) for p in paths]
    if len({len(leg) for leg in legs})!=1:
        raise ValueError('Unequal panel lengths')
    for matched in zip(*legs):
        for key in ('input_n','input_ids_sha256','kind','rep','requested'):
            if len({r[key] for r in matched})!=1:
                raise ValueError(f'Unmatched {key}')
    report={}
    for kind in sorted({r['kind'] for r in legs[0]}):
        groups=[[r for r in leg if r['kind']==kind] for leg in legs]
        metrics=[]
        for group in groups:
            metrics.append(dict(
                count=len(group),
                decode_tps=spread([r['timings']['predicted_per_second'] for r in group]),
                prefill_tps=spread([r['timings']['prompt_per_second'] for r in group]),
                ttft_s=spread([r['ttft_s'] for r in group]),
                wall_s=spread([r['wall_s'] for r in group]),
                generated_ids=[len(r['ids']) for r in group],
                accepted=sum(r['timings'].get('draft_n_accepted',0) for r in group),
                proposed=sum(r['timings'].get('draft_n',0) for r in group)))
        report[kind]=dict(legs=metrics,complete_output_ids_match=[a['ids']==b['ids']==c['ids'] for a,b,c in zip(*groups)])
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('filename')
    parser.add_argument('legs',nargs=3,type=Path)
    args=parser.parse_args()
    print(json.dumps(compare(args.legs,args.filename),indent=2))
