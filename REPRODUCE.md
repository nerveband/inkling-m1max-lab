# Reproduce the local serving setup

The hardware was a 24-core GPU M1 Max with 64 GiB of unified memory on macOS 27.0.1. Use Python 3.12 and Pi 1.6.0. All benchmark work fits a single local endpoint. The synthetic agent configuration and normal Pi selection use the same model alias.

## oMLX release and weights

Get the macOS 26/27 disk image from the [oMLX 0.8.2 release](https://github.com/jundot/omlx/releases/tag/v0.8.2). Its SHA-256 is 6316ad32cde3248bcf23abece8b9222847f7ae8e594cea1b378a34049252df78. The launcher can mount it read-only and detach on exit. Supply the full native target directory, including mtp.safetensors, from [mlx-community/Inkling-2-Mini-oQ4e-mtp at 3eea042](https://huggingface.co/mlx-community/Inkling-2-Mini-oQ4e-mtp/tree/3eea042751cd296b7266628a8342c1749242e1fe). To download the pinned snapshot with the Hugging Face CLI:

```sh
hf download mlx-community/Inkling-2-Mini-oQ4e-mtp \
  --revision 3eea042751cd296b7266628a8342c1749242e1fe \
  --local-dir "$MODEL_DIR/omlx-oq4"
```

Set MODEL_DIR to the external models directory and OMLX_DMG to the release image. From the experiment checkout:

```sh
python3.12 serve_omlx.py --dmg "$OMLX_DMG" \
  --weights "$MODEL_DIR/omlx-oq4" --state "$PWD/inkling-omlx-state" \
  --cache "$MODEL_DIR/inkling-agent-cache" --depth 6 --context 32768
```

Leave this process running in its terminal. Ctrl-C stops its server and releases its mount. The first request loads the weights. On the published prefill/qualification runs, the hot-cache cap was 1 GB and the SSD-cache cap 2 GB. One request ran at a time.

## Start Pi in another terminal

From the same checkout, create a fresh agent directory. The normal provider file connects directly to the model server. The independent measurement harness uses its own proxy to capture timings.

```sh
export PI_CODING_AGENT_DIR="$(mktemp -d)"
cp configs/pi-direct.json "$PI_CODING_AGENT_DIR/models.json"
pi --offline --provider inkling-local --model inkling --thinking off \
  -e "$PWD/configs/pi-greedy.ts"
```

The optional greedy extension sets temperature 0, seed 17 and a 512-ID output budget for this provider. It leaves Pi’s normal tools and system prompt intact. Omit the extension to use the ordinary provider defaults.

The 65,536-token qualification used two configured request slots and one active request. To reproduce it, use `--context 65536 --slots 2` in the foreground command and copy `configs/pi-direct-64k.json` as the disposable agent's `models.json`. At fixed depths 3 and 6, the exact ordered retrieval test submitted 31,954 and 64,717 input IDs and reserved 768 output IDs. Pi still used 512 IDs per response. The benchmark and the registry tested one active stream.

## GGUF controls

Download both top-level GGUF files from [the Unsloth pinned revision](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/ae9327c5fbc7333116ba32eee78bb5d2bbefae4f). Use the pinned [Apple Silicon runtime](https://github.com/unslothai/llama.cpp/releases/tag/b12588-mix-c0c3de8). Target and draft checksums are in PLAN.md. Set LLAMA_SERVER to the extracted binary. The plain and depth-6 launches differ only by the draft flags.

```sh
"$LLAMA_SERVER" -m "$MODEL_DIR/Inkling-2-Mini-UD-Q4_K_M.gguf" \
  --host 127.0.0.1 --port 18089 -a inkling -c 32768 -np 1 -ngl all \
  --jinja --metrics --slots --reasoning off --temp 0 --seed 17
```

For fixed depth 6 append:

```sh
--spec-type draft-dspark --spec-draft-n-max 6 \
--spec-draft-model "$MODEL_DIR/dspark-Inkling-2-Mini-Q8_0.gguf"
```

Use the same Pi selection command. The three-pass production loop stays fixed throughout.
