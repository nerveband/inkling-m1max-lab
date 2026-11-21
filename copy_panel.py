"""Run a warmed, six-request exact-copy panel against a local GGUF endpoint."""
import argparse
import json
from pathlib import Path
from direct import panel, request

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path)
    args=parser.parse_args()
    with (args.results/'copy-panel.jsonl').open('x') as output:
        for rep in range(-1,6):
            measured=request(panel(2048,'copy',rep),64 if rep<0 else 256)
            measured.update(kind='copy',rep=rep,warmup=rep<0)
            print(json.dumps(measured),file=output,flush=True)
