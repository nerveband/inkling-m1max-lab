"""Read GGUF tensor descriptors without loading tensor data."""
import collections, json, os, struct, sys
def tensors(path):
    with open(path,'rb') as f:
        def one(fmt): return struct.unpack('<'+fmt,f.read(struct.calcsize('<'+fmt)))[0]
        def string(): return f.read(one('Q')).decode()
        assert f.read(4)==b'GGUF' and one('I')==3
        n,meta=one('Q'),one('Q')
        formats=['B','b','H','h','I','i','f','?','','','Q','q','d']
        def value(kind):
            if kind==8: return string()
            if kind==9:
                t,count=one('I'),one('Q')
                if t not in (8,9): f.seek(count*struct.calcsize('<'+formats[t]),1)
                else:
                    for _ in range(count): value(t)
            else: return one(formats[kind])
        metadata={}
        for _ in range(meta):
            name=string();v=value(one('I'))
            if name.startswith('inkling2.') or name=='general.alignment': metadata[name]=v
        records=[]
        for _ in range(n):
            name=string();shape=[one('Q') for _ in range(one('I'))]; typ=one('I'); offset=one('Q')
            records.append((offset,name,shape,typ))
        align=metadata.get('general.alignment',32)
        start=(f.tell()+align-1)//align*align
        records.sort()
        size=int(os.getenv('GGUF_EXPECTED_BYTES',os.stat(path).st_size))
        ends=[r[0] for r in records[1:]]+[size-start]
        return [(name,shape,typ,end-off) for (off,name,shape,typ),end in zip(records,ends)],metadata
if __name__=='__main__':
    rec,meta=tensors(sys.argv[1]); print(meta)
    print([(n,s,t,b) for n,s,t,b in rec if n.startswith('blk.0.') or not n.startswith('blk.')][:60])

def estimate(records, length, bandwidth, tflops):
    import math
    cold_bytes=unique_bytes=operations=0
    for name,shape,kind,size in records:
        layer=int(name.split('.')[1]) if name.startswith('blk.') else -1
        visits=3 if 6<=layer<18 else 1
        fraction=8/128 if '_exps.weight' in name else 1
        if name=='token_embd.weight': fraction=1/shape[1]
        # The last four physical blocks are stored after the recurrent region.
        cold_bytes+=size*fraction*visits
        unique_bytes+=size*fraction
        if len(shape)>=2 and name!='token_embd.weight':
            operations+=2*math.prod(shape)*fraction*visits
    # Each sixth logical layer is global. FP16 K and V, eight heads by 128.
    positions=8*length+40*min(length,1024)
    kv=positions*2*2*8*128
    # Average causal span during prefill. No approximation to the dense score matrix allocation.
    avg_global=(length+1)/2
    avg_local=avg_global if length<=1024 else 1024-1024*1023/(2*length)
    attn=4*24*128*(8*avg_global+40*avg_local)
    return dict(context=length,weight_gb=cold_bytes/1e9,ideal_reuse_gb=unique_bytes/1e9,kv_gb=kv/1e9,
                decode_tps_range=[bandwidth*1e9/(cold_bytes+kv),bandwidth*1e9/(unique_bytes+kv)],
                prefill_tps_half_gemm=0.5*tflops*1e12/(operations+attn))
