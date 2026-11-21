"""Native endpoint, exact token-ID prompts. Discard warmups, keep all raw counters."""
import hashlib, json, os, pathlib, sys, time, urllib.request
URL='http://127.0.0.1:18089'
def post(path,body,timeout=900):
    req=urllib.request.Request(URL+path,json.dumps(body).encode(),{'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as r: return json.load(r)
def tokens(text): return post('/tokenize',{'content':text,'add_special':False,'parse_special':True})['tokens']
def request(ids,n):
    body=dict(prompt=ids,n_predict=n,temperature=0,seed=17,stream=True,cache_prompt=False,return_tokens=True)
    req=urllib.request.Request(URL+'/completion',json.dumps(body).encode(),{'Content-Type':'application/json'})
    t=time.monotonic(); first=None; output=[]; emitted=[]; last={}
    with urllib.request.urlopen(req,timeout=1200) as resp:
        for line in resp:
            if not line.startswith(b'data:') or line[5:].strip()==b'[DONE]': continue
            ev=json.loads(line[5:])
            if ev.get('content') or ev.get('tokens'):
                if first is None: first=time.monotonic()
                output.append(ev.get('content',''));emitted.extend(ev.get('tokens',[]))
            if ev.get('stop'): last=ev
    end=time.monotonic()
    return dict(input_n=len(ids),input_ids_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),requested=n,ttft_s=first-t if first else None,wall_s=end-t,timings=last.get('timings',{}),evaluated=last.get('tokens_evaluated'),cached=last.get('tokens_cached'),stop_type=last.get('stop_type'),ids=emitted,text=''.join(output))
def panel(length,kind,rep,encode=tokens,chat_suffix=None):
    code=''.join(f'def compute_{i}(value):\n    return (value * {i+1} + {i%11}) % 257\n\n' for i in range(135))
    prefix=encode('Synthetic work item '+str(rep)+'.\n')
    text=code if kind=='copy' else '\n'.join(f'Case {i}: input is {i*17} and label is record_{i}.' for i in range(400))
    end='Repeat the following code exactly, starting from compute_0. Write 40 functions.\n'+code[:750] if kind=='copy' else 'Implement a Python ring buffer class with capacity validation, iteration, resizing and unit tests. Write the complete implementation.\n'
    messages=[{'role':'user','content':end}]
    marker='<|message_model|><|content_text|>' if os.environ.get('BENCH_VISIBLE')=='1' else ''
    if chat_suffix:
        tail=chat_suffix(messages,marker)
    else:
        rendered=post('/apply-template',{'messages':messages})['prompt']
        tail=encode(rendered+marker)
    pad=encode(text); k=length-len(prefix)-len(tail)
    assert k>0
    return prefix+(pad*((k+len(pad)-1)//len(pad)))[:k]+tail
def run(out,length,n,count):
    path=pathlib.Path(out);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        for rep in range(-1,count):
            kind='copy' if rep%2==0 else 'fresh'
            res=request(panel(length,kind,rep),n)
            res.update(warmup=rep==-1,kind=kind,rep=rep,visible_prefix=os.environ.get('BENCH_VISIBLE')=='1')
            print(json.dumps(res),file=f,flush=True)
            print(kind,rep,res['timings'],flush=True)
if __name__=='__main__': run(sys.argv[1],*map(int,sys.argv[2:]))
