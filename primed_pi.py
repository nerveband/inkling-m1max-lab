"""Measure one synthetic Pi edit with a solution-free, reusable project prefix."""
import itertools
import json
import os
from pathlib import Path
import sys
import time

from direct import post
from mlx_screen import probe
from pi_task import run

root = Path(__file__).resolve().parent
destination = Path(sys.argv[1])
destination.mkdir(parents=True, exist_ok=False)
shape = json.loads((root / "fixtures/pi-prefix.json").read_text())
shape["system"]["content"] = shape["system"]["content"].replace("${PI_PACKAGE}", os.environ["PI_PACKAGE"])
os.environ["BENCH_MODEL_ID"] = "inkling"
probe(32)


def tokenize(user):
    return post("/tokenize", {
        "model": "inkling", "tools": shape["tools"],
        "messages": [shape["system"], {"role": "user", "content": user}],
    })["tokens"]


start = time.monotonic()
first, second = tokenize("Alpine."), tokenize("Zebra.")
common = sum(1 for _ in itertools.takewhile(lambda pair: pair[0] == pair[1], zip(first, second)))
ids = first[:common + 1]
prompt = post("/detokenize", {"model": "inkling", "tokens": ids})["prompt"]
if post("/tokenize", {"model": "inkling", "prompt": prompt})["tokens"] != ids:
    raise ValueError("Prepared prefix changed in the tokenizer round trip.")
reply = post("/v1/completions", {"model": "inkling", "prompt": prompt, "max_tokens": 1, "temperature": 0})
record = {"prime_s": time.monotonic() - start, "common_ids": common, "primed_ids": len(ids), "prime_usage": reply["usage"]}
(destination / "prefix-prime.json").write_text(json.dumps(record) + "\n")
run(destination / "pi")
