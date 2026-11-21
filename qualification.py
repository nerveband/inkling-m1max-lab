import hashlib, json, os, pathlib, sys, time
from direct import post,tokens
def needle(length):
    codes=[hashlib.sha256(f'{length}:{i}:inkling'.encode()).hexdigest()[:10].upper() for i in range(3)]
    def prompt(n):
        lines=[f'Inventory {i}: sealed blue container.\n' for i in range(n)]
        for i,depth in reversed(list(enumerate((.12,.53,.94)))):
            lines.insert(int(depth*n),f'Verified audit key for section {i+1} is {codes[i]}.\n')
        return 'Read this inventory.\n'+''.join(lines)+'Return exactly the three verified audit keys in section order, separated by spaces.'
    def shape(s):
        if os.getenv('BENCH_TOKENIZER')=='omlx':
            return len(post('/tokenize',{'model':'inkling','messages':[{'role':'user','content':s}]})['tokens'])
        q=post('/apply-template',{'messages':[{'role':'user','content':s}]})['prompt']
        return len(tokens(q))
    lo,hi=0,length//3
    while lo<hi:
        mid=(lo+hi+1)//2
        if shape(prompt(mid))<=length-300: lo=mid
        else: hi=mid-1
    s=prompt(lo); observed=shape(s); t=time.monotonic()
    r=post('/v1/chat/completions',dict(model='inkling',messages=[dict(role='user',content=s)],temperature=0,max_tokens=256,seed=17),timeout=3600)
    answer=r['choices'][0]['message'].get('content','').strip()
    return dict(target=length,input=observed,usage=r.get('usage'),timings=r.get('timings'),answer=answer,expected=' '.join(codes),passed=answer==' '.join(codes),wall_s=time.monotonic()-t)
if __name__=='__main__':
    with open(sys.argv[1],'x') as f:
        for x in sys.argv[2:]:
            r=needle(int(x));print(json.dumps(r),file=f,flush=True); print(r['input'],r['passed'],flush=True)
