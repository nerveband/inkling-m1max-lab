"""Exact-length native oMLX completions on the shared copy/fresh panels."""
import hashlib
import json
import os
import pathlib
import sys
import time
import urllib.request
from direct import panel,post

def encode(text):
    return post('/tokenize',{'model':'inkling','prompt':text,'add_special_tokens':False})['tokens']

def suffix(messages,marker):
    ids=post('/tokenize',{'model':'inkling','messages':messages,'add_generation_prompt':True})['tokens']
    return ids+encode(marker)

def measure(length,kind,rep,limit):
    intended=panel(length,kind,rep,encode,suffix)
    prompt=post('/detokenize',{'model':'inkling','tokens':intended})['prompt']
    seen=post('/tokenize',{'model':'inkling','prompt':prompt})['tokens']
    if len(seen)!=length:
        diff=length-len(seen)
        if diff<0:
            intended=intended[:-64+diff]+intended[-64:]
            prompt=post('/detokenize',{'model':'inkling','tokens':intended})['prompt']
        else:
            prompt='|z'*((diff+1)//2)+prompt
        seen=post('/tokenize',{'model':'inkling','prompt':prompt})['tokens']
    if len(seen)!=length:raise RuntimeError(f'input mismatch {len(seen)} vs {length}')
    payload=dict(model='inkling',prompt=prompt,max_tokens=limit,temperature=0,stream=True,stream_options={'include_usage':True})
    req=urllib.request.Request('http://127.0.0.1:18089/v1/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    start=time.monotonic();first=None;parts=[];usage={};finish=None
    with urllib.request.urlopen(req,timeout=1200) as resp:
        for line in resp:
            if not line.startswith(b'data:') or line[5:].strip()==b'[DONE]':continue
            e=json.loads(line[5:]);c=(e.get('choices') or [{}])[0];piece=c.get('text') or ''
            if piece:
                first=first or time.monotonic()
                parts.append(piece)
            usage=e.get('usage') or usage
            finish=c.get('finish_reason') or finish
    wall=time.monotonic()-start
    if usage.get('prompt_tokens')!=length: raise RuntimeError(f'endpoint counted {usage.get("prompt_tokens")}, expected {length}')
    return dict(warmup=rep<0,kind=kind,rep=rep,input_n=length,input_ids_sha256=hashlib.sha256(json.dumps(seen,separators=(',',':')).encode()).hexdigest(),usage=usage,requested=limit,ttft_s=first-start if first else None,wall_s=wall,finish=finish,text=''.join(parts),decode_tps=(usage['completion_tokens']-1)/usage['generation_duration'] if usage.get('generation_duration') else None)

if __name__=='__main__':
    root=pathlib.Path(sys.argv[1]);os.environ['BENCH_VISIBLE']='1'
    for n,filename in ((2048,'native-2k.jsonl'),(8192,'native-8k.jsonl')):
        with (root/filename).open('x') as target:
            for i in range(-1,6):
                r=measure(n,'copy' if i%2==0 else 'fresh',i,256)
                print(json.dumps(r),file=target,flush=True);print(n,i,r['decode_tps'],flush=True)
