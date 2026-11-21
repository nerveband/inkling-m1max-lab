"""Run the final qualification against an already running loopback GGUF endpoint."""
import json,pathlib,sys,time
from direct import post
from pi_task import run as pi
from qualification import needle
def qualify(path,context):
    path=pathlib.Path(path);path.mkdir(exist_ok=True,parents=True)
    msgs=[dict(role='user',content='What is 17 times 23? Return just the number.')]
    r=post('/v1/chat/completions',dict(model='inkling',messages=msgs,temperature=0,max_tokens=512))
    msg=r['choices'][0]['message']
    reasoning_ok=bool((msg.get('reasoning_content') or msg.get('reasoning') or '').strip()) and (msg.get('content') or '').strip()=='391'
    (path/'reasoning.json').write_text(json.dumps(dict(passed=reasoning_ok,answer=msg.get('content'),finish=r['choices'][0]['finish_reason'],usage=r['usage'])))
    held=pi(path/'pi-held',held=True)
    with (path/'needles.jsonl').open('x') as f:
        for length in (context//2,context):
            result=needle(length);print(json.dumps(result),file=f,flush=True);print('needle',result['input'],result['passed'],flush=True)
    messages=[dict(role='user',content='Remember the project label IRIS7. Reply with just IRIS7.')]
    rows=[]
    for i in range(12):
        if i: messages.append(dict(role='user',content=f'Checkpoint {i}. Output only the original project label.'))
        start=time.monotonic()
        obj=post('/v1/chat/completions',dict(model='inkling',messages=messages,temperature=0,max_tokens=256))
        answer=obj['choices'][0]['message'];messages.append(dict(role='assistant',content=answer.get('content','')))
        rows.append(dict(turn=i,passed=answer.get('content','').strip()=='IRIS7',wall=time.monotonic()-start,timings=obj.get('timings'),usage=obj.get('usage')))
        (path/'soak.json').write_text(json.dumps(rows,indent=2))
    print('reasoning',reasoning_ok,'held',held['passed'],'soak',sum(x['passed'] for x in rows),'/12',flush=True)
if __name__=='__main__':qualify(sys.argv[1],int(sys.argv[2]))
