# How to Reproduce Every Benchmark and Serving Setup

Every test below runs on a 24-core Apple M1 Max with 64 GiB of unified memory on macOS 27.0.1. Use Python 3.12 and Pi 1.6.0.

All paths below use environment variables so anyone can run these steps on their own machine:

- Set `MODEL_DIR` to the folder holding your model directories.
- Set `OMLX_DMG` to your downloaded `oMLX-0.8.2-macos26-27.dmg` disk image.
- Run commands from the root of this `inkling-m1max-lab` checkout.

## Download the runtime and weights

1. Download the macOS 26/27 disk image from the [oMLX 0.8.2 release](https://github.com/jundot/omlx/releases/tag/v0.8.2). Its SHA-256 checksum must match `6316ad32cde3248bcf23abece8b9222847f7ae8e594cea1b378a34049252df78`.
2. Download the pinned 4-bit `oQ4e-MTP` weights (`3eea042751cd296b7266628a8342c1749242e1fe`):

```sh
hf download mlx-community/Inkling-2-Mini-oQ4e-mtp \
  --revision 3eea042751cd296b7266628a8342c1749242e1fe \
  --local-dir "$MODEL_DIR/omlx-oq4"
```

3. For 3-bit comparison tests, download the pinned `oQ3e-MTP` weights (`5552e32ec7a50fa8e9b0bfe2002aab72f8b78975`):

```sh
hf download mlx-community/Inkling-2-Mini-oQ3e-mtp \
  --revision 5552e32ec7a50fa8e9b0bfe2002aab72f8b78975 \
  --local-dir "$MODEL_DIR/omlx-oq3"
```

## Start the promoted 4-bit M1 Max server

Affine 4-bit is a 4-bit number format with scale and offset weights. The `--m1-fast` flag turns on our Metal kernels from [m1_overlay.py](m1_overlay.py).

To start the promoted server in a terminal on port `18089`, run:

```sh
python3.12 serve_omlx.py --dmg "$OMLX_DMG" \
  --weights "$MODEL_DIR/omlx-oq4" --state "$PWD/inkling-m1-oq4-state" \
  --cache "$PWD/inkling-m1-oq4-kv" --depth 6 --context 65536 --slots 1 \
  --m1-fast --affine4 --hot-cache-max-size 8GB --default-temp 0.0
```

Leave that terminal open while testing. Press `Ctrl-C` to stop the server and detach the disk image.

To test the 3-bit `oQ3e` speed setup (`49.38 tok/s` direct Pi), point `--weights` at `omlx-oq3`, set `--context 32768`, and drop `--affine4`:

```sh
python3.12 serve_omlx.py --dmg "$OMLX_DMG" \
  --weights "$MODEL_DIR/omlx-oq3" --state "$PWD/inkling-oq3-state" \
  --cache "$PWD/inkling-oq3-kv" --depth 6 --context 32768 --slots 1 \
  --m1-fast
```

## Connect Pi to the local server

For interactive Pi use with working thinking levels (`/thinking` or `Shift+Tab`), copy [configs/pi-interactive.json](configs/pi-interactive.json) into `~/.pi/agent/models.json` (or merge its `providers` block):

- `inkling-local` (`http://127.0.0.1:18089/v1`) sets `"reasoning": true`, `"thinkingFormat": "openai"`, and `"thinkingLevelMap"` mapping `off` through `max` to Inkling's native `reasoning_effort` values (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`).
- `omlx-ornith` (`http://127.0.0.1:8000/v1`) sets `"reasoning": true`, `"thinkingFormat": "qwen-chat-template"`, and hides middle levels so `off` and `high` toggle binary thinking cleanly on `Ornith-1.5-35B-A3B-Abliterated-MLX-4bit`.

Keep [configs/pi-direct-64k.json](configs/pi-direct-64k.json) (`"reasoning": false`, `"maxTokens": 512`) only for deterministic `512`-token benchmarks. Open a second terminal in this checkout and use a fresh temporary agent directory to test that fixed benchmark setup:

```sh
export PI_CODING_AGENT_DIR="$(mktemp -d)"
cp configs/pi-direct-64k.json "$PI_CODING_AGENT_DIR/models.json"
pi --offline --provider inkling-local --model inkling --thinking off \
  -e "$PWD/configs/pi-greedy.ts"
```

The `pi-greedy.ts` extension sets temperature `0`, seed `17`, and a `512` token output limit. Drop `-e "$PWD/configs/pi-greedy.ts"` when running normal everyday work.

Prefix cache is saved KV cache that lets the server skip re-reading shared system instructions. To warm the shared Pi prompt in your project directory, set `LAB` to this repository path and run both lines in order:

```sh
pi --offline --provider inkling-local --model inkling --thinking off \
  -e "$LAB/configs/pi-greedy.ts" -e "$LAB/configs/pi-warm-prefix.ts" \
  -p "Reply READY. Do not use tools."
pi --offline --provider inkling-local --model inkling --thinking off \
  -e "$LAB/configs/pi-greedy.ts"
```

Changing tools, project files, or system prompts creates a different prefix cache key.

## Repeat the M1 speed screen and A/B/A benchmark

Close heavy GPU apps so the M1 Max GPU starts idle. Mount `OMLX_DMG` at `../mount-omlx-082` when running `screen_m1.py`, or set `OMLX_APP` to the mounted `oMLX.app` path when running `omlx_leg.py` or `omlx_run.py`.

To repeat one primed M1 screen with all promoted flags on `oQ4e`, run:

```sh
python3.12 screen_m1.py speed-m1-test oq4 \
  sort32,caps,m1_attn,verify_compile,split_in_proj,skip_lm_head,fast_decode,fast_greedy,prefill_pad64,affine4 \
  6
```

To run one 6-task leg of the fresh-prefix A/B/A test (7 trials, first warmups dropped or scored per `omlx_leg.py`), run:

```sh
export OMLX_APP="/path/to/oMLX.app"
python3.12 omlx_leg.py --weights "$MODEL_DIR/omlx-oq4" \
  --cache-root "$PWD/scratch-kv" --depth 6 --context 32768 \
  --m1-fast --affine4 results/candidate-b
```

Run a control leg without `--m1-fast --affine4` before and after Candidate B so you can measure thermal or background drift.

## Re-run the 64K retrieval and qualification runs

The 64K qualification uses `--context 65536 --slots 1` on `omlx-oq4 --m1-fast --affine4` (or `--slots 2` for registry runs that require 131,072 total KV tokens). Exact ordered retrieval tests at 15,569, 31,954, and 64,717 input tokens reserve `768` output tokens. Full raw results sit in [raw/qual-m1-final](raw/qual-m1-final).

## Run stock GGUF baselines

Download `Inkling-2-Mini-UD-Q4_K_M.gguf` and `dspark-Inkling-2-Mini-Q8_0.gguf` from [the Unsloth revision](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/ae9327c5fbc7333116ba32eee78bb5d2bbefae4f). Download the matching [llama.cpp Apple Silicon runtime](https://github.com/unslothai/llama.cpp/releases/tag/b12588-mix-c0c3de8) and set `LLAMA_SERVER` to the server binary.

To run the plain GGUF baseline on port `18089`, run:

```sh
"$LLAMA_SERVER" -m "$MODEL_DIR/Inkling-2-Mini-UD-Q4_K_M.gguf" \
  --host 127.0.0.1 --port 18089 -a inkling -c 32768 -np 1 -ngl all \
  --jinja --metrics --slots --reasoning off --temp 0 --seed 17
```

To add DSpark depth 6 speculative drafting, append these flags:

```sh
--spec-type draft-dspark --spec-draft-n-max 6 \
  --spec-draft-model "$MODEL_DIR/dspark-Inkling-2-Mini-Q8_0.gguf"
```

Use [configs/pi-direct.json](configs/pi-direct.json) when connecting Pi to this 32K GGUF launch.
