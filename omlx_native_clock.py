"""Join oMLX native fixed-depth cycle timestamps to buffered raw completion streams."""
import datetime
import json
import pathlib
import re
import sys

def clocks(root):
    start={}
    windows=[]
    for line in (root/'server.log').read_text().splitlines():
        begin=re.search(r'MTP path activated for uid=(\d+)',line)
        finish=re.search(r'MTP\[(\d+)\] finish=\S+ tokens=(\d+) cycles=(\d+).*?accept=(\d+)/(\d+)',line)
        if not begin and not finish:continue
        t=datetime.datetime.strptime(line[:23],'%Y-%m-%d %H:%M:%S,%f')
        if begin:start[begin[1]]=t
        elif finish[1] in start:
            ms=re.search(r'timing\[([^\]]+)\]',line)
            components={k:float(v) for k,v in re.findall(r'([a-z_]+)=([\d.]+)ms',ms[1])} if ms else {}
            windows.append(dict(generated=int(finish[2]),seconds=(t-start[finish[1]]).total_seconds(),cycles=int(finish[3]),accepted=int(finish[4]),proposed=int(finish[5]),components_ms=components))
    return windows

def analyse(root):
    dst=pathlib.Path(__file__).resolve().parent/'raw'/root.name
    dst.mkdir(exist_ok=True,parents=True)
    rows=[]
    for name in ('native-2k.jsonl','native-8k.jsonl'):
        p=root/name
        if p.exists():
            (dst/name).write_bytes(p.read_bytes())
            rows.extend(json.loads(l) for l in p.read_text().splitlines())
    periods=clocks(root)
    if len(rows)!=len(periods):raise ValueError('raw sample / native clock count mismatch')
    results=[]
    for r,w in zip(rows,periods):
        count=r['usage']['completion_tokens']
        eos_gap=w['generated']-count
        if eos_gap!=0 and not (r['finish']=='stop' and eos_gap==1):raise ValueError('token count mismatch')
        results.append(dict(input_n=r['input_n'],kind=r['kind'],rep=r['rep'],input_ids_sha256=r['input_ids_sha256'],first_visible_s=r['ttft_s'],wall_s=r['wall_s'],warmup=r['warmup'],endpoint_count=count,stop_count_difference=eos_gap,decode_window_tps=(count-1)/w['seconds'],**w))
    (dst/'native-clock.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':analyse(pathlib.Path(sys.argv[1]))
