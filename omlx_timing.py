"""Copy native MTP counters from the synthetic server run without host paths."""
import json
import pathlib
import re
import sys

for arg in sys.argv[1:]:
    root=pathlib.Path(arg)
    rows=[]
    for line in (root/'server.log').read_text(errors='replace').splitlines():
        if ' INFO - MTP[' not in line or ' timing[' not in line:
            continue
        details=line.split(' INFO - ',1)[1]
        if '/' in details.split('MTP[',1)[0]:
            continue
        totals=re.search(r'tokens=(\d+) cycles=(\d+).*?accept=(\d+)/(\d+)',details)
        if totals:
            accounted=re.search(r'timing\[([^\]]+)\]',details)[1]
            milliseconds={k:float(v) for k,v in re.findall(r'([a-z_]+)=([\d.]+)ms',accounted)}
            tokens,cycles,accepted,proposed=map(int,totals.groups())
            rows.append(dict(tokens=tokens,cycles=cycles,accepted=accepted,proposed=proposed,ms=milliseconds,detail=details))
    dest=pathlib.Path(__file__).resolve().parent/'raw'/root.name
    dest.mkdir(parents=True,exist_ok=True)
    (dest/'mtp.json').write_text(json.dumps(rows,indent=2)+'\n')
