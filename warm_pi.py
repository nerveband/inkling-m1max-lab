"""Warm one isolated server with unrelated text, then run one Pi task on a fresh prefix."""
import json
import pathlib
import sys
from mlx_screen import probe
from pi_task import run

if __name__=='__main__':
    destination=pathlib.Path(sys.argv[1])
    warm=probe(32)
    (destination/'warm.json').write_text(json.dumps(warm))
    run(destination/'pi')
