"""Measure the exact common output prefix against the supplied code, in characters."""
import itertools,json,pathlib,re,sys
REFERENCE=''.join(f'def compute_{i}(value):\n    return (value * {i+1} + {i%11}) % 257\n\n' for i in range(135))
def assess(text):
    start=text.find('def compute_0(')
    segment=text[start:] if start>=0 else ''
    count=sum(1 for _ in itertools.takewhile(lambda pair:pair[0]==pair[1],zip(segment,REFERENCE)))
    return {'characters_before_code':start,'exact_copy_characters':count,'complete_functions':REFERENCE[:count].count('% 257\n'),'only_reference_prefix':bool(segment) and len(segment)==count}
if __name__=='__main__':
    for name in sys.argv[1:]:
        f=pathlib.Path(name)
        rows=[json.loads(l) for l in f.read_text().splitlines()] if f.suffix=='.jsonl' else [json.loads(f.read_text())]
        print(f.parent.name,f.name,[assess(r['text']) for r in rows if r.get('kind','copy')=='copy' and not r.get('warmup')])
