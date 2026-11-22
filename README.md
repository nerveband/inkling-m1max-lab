# Inkling 2 Mini on Apple M1 Max

This repository holds benchmarks, Metal kernels, and setup scripts for Inkling 2 Mini on a 24-core Apple M1 Max with 64 GiB of unified memory.

The main goal is fast and correct coding agent turns in Pi. The promoted M1 Max v2 stack reaches 64.21 tokens per second in controlled Pi A/B/A tests, up from 49.09 to 49.45 tokens per second on v1 and 37.87 tokens per second on stock oMLX 0.8.2. On direct HTTP coding runs without thinking tokens, it reaches 61.36 tokens per second on code generation and 64.98 tokens per second on code edits. Both humans and coding agents can repeat every experiment here.

Start with the [interactive HTML dashboard](report.html), the [Inkling benchmark findings](FINDINGS.md), the [Ornith-1.5-35B-A3B findings](ornith/FINDINGS.md), the [steps to reproduce](REPRODUCE.md), and the [candidate list](CANDIDATES.md). The full experiment log is in [PLAN.md](PLAN.md). Hardware math is in [THEORY.md](THEORY.md). Boundary rules are in [PROTOCOL.md](PROTOCOL.md).

## What sits in this repository

- [serve_omlx.py](serve_omlx.py) starts oMLX 0.8.2 with the promoted M1 Max kernel overlay in [m1_overlay.py](m1_overlay.py), an 8 GB hot prefix cache in RAM, and default temperature 0.0 for greedy coding runs.
- [patches](patches) holds custom dual-bucket Metal MoE kernels in `m1_qmv_moe.py`, fast decode graph patches, and modular PR backports (`verify_flush`, `predraft_fix`, `dspark_2bit`, `bounded_cache`, `bf16_loop_ctrl`, `prefill_fast`, and `ttft_fixes`).
- [configs](configs) holds Pi provider files and Pi extensions for deterministic generation and prefix warmup.
- [screen_m1.py](screen_m1.py), [omlx_run.py](omlx_run.py), and [omlx_leg.py](omlx_leg.py) run primed single screens and multi-trial A/B/A comparisons.
- [raw](raw) and [ledger.jsonl](ledger.jsonl) hold raw JSON counters and pass or fail records from every run.

## Quick start for benchmarks

Install Python 3.12 and Pi 1.6.0 on macOS. Download the pinned oMLX 0.8.2 disk image and weights by following [REPRODUCE.md](REPRODUCE.md). Set `OMLX_DMG` to the disk image path and `MODEL_DIR` to your model folder.

Before running a foreground server, mount the disk image and start oMLX with the promoted M1 Max flags:

```sh
python3.12 serve_omlx.py --dmg "$OMLX_DMG" \
  --weights "$MODEL_DIR/omlx-oq4" --state "$PWD/inkling-m1-oq4-state" \
  --cache "$PWD/inkling-m1-oq4-kv" --depth 6 --context 65536 --slots 1 \
  --m1-fast --affine4 --hot-cache-max-size 8GB --default-temp 0.0
```

To run the baseline proxy benchmark against an existing endpoint on port 18089, run one command from this directory:

```sh
python3.12 run.py plain measurements/baseline python3.12 pi_task.py measurements/baseline/pi
```

The measurement harness points Pi at a local timing proxy on port 18090. That proxy forwards traffic to the model server on `127.0.0.1:18089`. Private logs and personal paths stay out of this repository. Always pick a fresh output folder for each measurement run.
