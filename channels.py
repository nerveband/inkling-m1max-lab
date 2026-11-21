"""Count exposed Pi text with the pinned tokenizer; mark retokenized counts explicitly."""
import json,os,pathlib,sys
from tokenizers import Tokenizer
tok=Tokenizer.from_file(os.environ['TOKENIZER_JSON'])
def count(text):return len(tok.encode(text,add_special_tokens=False).ids)
for base in sys.argv[1:]:
    root=pathlib.Path(base); result={}
    for p in sorted(root.glob('pi*/pi.jsonl')):
        outputs=[]
        for line in p.read_text().splitlines():
            e=json.loads(line);m=e.get('message',{})
            if e.get('type')!='message_end' or m.get('role')!='assistant':continue
            parts=m['content']
            outputs.append(dict(reasoning_retokenized=sum(count(q['thinking']) for q in parts if q['type']=='thinking'),visible_retokenized=sum(count(q['text']) for q in parts if q['type']=='text'),tools=[q['name'] for q in parts if q['type']=='toolCall']))
        result[p.parent.name]=outputs
    print(root.name,json.dumps(result))
    dest=pathlib.Path(__file__).resolve().parent/'raw'/root.name
    if dest.is_dir():(dest/'channels.json').write_text(json.dumps(result,indent=2)+'\n')
