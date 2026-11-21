"""Each invocation occupies a single already-warm server slot."""
import json,pathlib,sys,time
from direct import run as direct,post,tokens
from pi_task import run as pi
def chat(message,**extra):
    return post('/v1/chat/completions',dict(model='inkling',messages=[dict(role='user',content=message)],temperature=0,max_tokens=128,**extra))
def capture(directory):
    records=[]
    for path in sorted(directory.glob('request-*.json')):
        p=json.loads(path.read_text())
        rendered=post('/apply-template',p)['prompt']
        ids=tokens(rendered)
        records.append(dict(messages=[m['role'] for m in p['messages']],tools=[t['function']['name'] for t in p.get('tools',[])],tokens=len(ids),initial_ids=ids[:8],final_ids=ids[-8:]))
        (path.parent/(path.stem+'-ids.txt')).write_text(' '.join(map(str,ids)))
    (directory/'shape.json').write_text(json.dumps(records,indent=2))
def main(out,mode):
    out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=True)
    if mode=='smoke':
        r=chat('Reply with just the word READY.')
        (out/'chat.json').write_text(json.dumps(r));print('chat',str(r['choices'])[:280],flush=True)
        pi(out/'pi');capture(out/'pi')
        direct(out/'small.jsonl',2048,64,1)
    elif mode=='leg':
        for i in range(7):
            pi(out/f'pi-{i}')
        direct(out/'2k.jsonl',2048,256,6)
    elif mode=='8k':
        direct(out/'8k.jsonl',8192,256,6)
if __name__=='__main__':main(*sys.argv[1:])
