"""Screen the same synthetic Pi fixture, then the backend's direct stream."""
import json,os,pathlib,sys,time,urllib.request
from pi_task import run
def probe():
    body={'model':os.environ.get('BENCH_MODEL_ID','default_model'),'messages':[{'role':'user','content':'Implement a Python ring buffer class with iteration, resizing and tests. Return complete code.'}],'temperature':0,'max_tokens':256,'stream':True,'stream_options':{'include_usage':True}}
    request=urllib.request.Request('http://127.0.0.1:18089/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    first=None;usage={};pieces=[];reasoning=[];t=time.monotonic()
    with urllib.request.urlopen(request,timeout=900) as response:
        for line in response:
            if not line.startswith(b'data:') or line[5:].strip()==b'[DONE]':continue
            event=json.loads(line[5:]); usage=event.get('usage') or usage
            delta=((event.get('choices') or [{}])[0]).get('delta') or {}
            if delta.get('content') or delta.get('reasoning_content') or delta.get('reasoning'):
                first=first or time.monotonic();pieces.append(delta.get('content') or '');reasoning.append(delta.get('reasoning_content') or delta.get('reasoning') or '')
    end=time.monotonic()
    return dict(ttft=first-t if first else None,wall=end-t,usage=usage,decode_tps=(usage['completion_tokens']-1)/(end-first) if first and usage.get('completion_tokens',0)>1 else None,text=''.join(pieces),reasoning=''.join(reasoning))
if __name__=='__main__':
    dest=pathlib.Path(sys.argv[1]);dest.mkdir(parents=True,exist_ok=True)
    for i in range(2):
        r=probe();(dest/f'direct-{i}.json').write_text(json.dumps(r));print('direct',r['decode_tps'],r['usage'],flush=True)
    run(dest/'pi')
