# Ornith-1.5-35B-A3B M1 Max Optimization Plan

## Objective

Measure and optimize single-stream and warm-prefix throughput, speculative acceptance, and long-context stability for `PocketAiHub/Ornith-1.5-35B-A3B-Abliterated-MLX-4bit` on Apple M1 Max (24-core `applegpu_g13s`, 64 GiB RAM) across:

1. **Stock oMLX 0.8.2** (draft off / DFlash2 `jzinno` & `felkhayer` / Q4 vs FP16 drafter / `activation_fp16_enabled`).
2. **oMLX 0.8.2 + `ornith_m1_overlay.py`** independent A/B levers:
   - `m1_caps`: Enable `_SPEC_BUFFER_CAPS = (200, 512)` on `applegpu_g13*`.
   - `add_rms_2048`: Unlock fused `add_rms_norm` for `D = 2048` with `n_threads = 512`.
   - `min_n_512`: Lower `_MIN_N = 512` in `m1_verify_qmv` and `m1_verify_mma`.
   - `sorted_moe_verify`: Optional fallthrough (`OMLX_SORTED_MOE_VERIFY_MIN_ROWS=4`) from `routed_verify_window` at `M in 4..8` to expert-sorted `moe_verify_gather.py`.
   - `fp16_fast_router`: Add `omlx_router_topk<half, NE>` packed bit-key selection for `float16` activations.
3. **Splash `1.7.0-m1`** (staged side-by-side in `work/ornith-lab/splash-1.7.0-m1/` with `felkhayer/Ornith-1.5-35B-A3B-DFlash2-Splash`, `--language-only`).

## Benchmark Protocol (`bench_ornith.py`)

- Target endpoint: `--url http://127.0.0.1:18090/v1` (OpenAI-compatible `/chat/completions`, SSE streaming with `stream_options.include_usage=true`).
- Warmup request before scored runs; deterministic sampling (`--temperature 0.0` default).
- Five deterministic workloads:
  1. `code_gen`: Fresh Python parser/algorithm implementation (`max_tokens=512`), checks structural validity and non-degeneracy.
  2. `code_edit`: Multi-function ~2.5K-token Python file refactor (`max_tokens=512`), exercises suffix/copy drafting and DFlash2.
  3. `open_prose`: Analytical technical essay (`max_tokens=350`), checks non-repetitive prose generation.
  4. `warm_pi`: 8K-token shared coding prefix (1 cold turn to populate prefix cache + 2 warm follow-up turns appending small tasks) measuring cold vs warm TTFT and decode tok/s.
  5. `needle_32k`: ~32K-token non-trivial padded context with two unique passkey needles at ~20% and ~75% depth (`max_tokens=256`), verifying exact recall of both needles, zero repetition collapse, `memory_pressure`, and `sysctl vm.swapusage`.
- Supports `--mode single` and `--mode aba` (or `--repeats N`), writing raw JSON artifacts to `work/ornith-lab/raw/<tag>/` and appending rows to `work/ornith-lab/ledger.jsonl`.

## Promotion Gates

1. Zero repetition collapse and 100% pass on `code_gen`, `code_edit`, and `needle_32k` dual-needle recall.
2. Idle GPU baseline check (`AGXAccelerator` idle before starting) and zero swap growth / `memory_pressure` normal.
3. Statistical A/B/A gain over adjacent stock controls (> 3% margin on decode tok/s and/or end-to-end task wall time).
