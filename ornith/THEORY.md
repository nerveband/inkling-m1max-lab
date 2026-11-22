# Ornith-1.5-35B-A3B on Apple M1 Max: Hardware Ceilings and oMLX 0.8.2 Audit

## 1. Verified Hardware Baseline (`applegpu_g13s`)

Measured on this 24-core Apple M1 Max (64 GiB LPDDR5 unified memory, macOS 27.0.1) via Metal microbenchmarks (`work/lab/device.m`, median of 7 warmed command buffers):

| Metric | Value | Notes |
|---|---:|---|
| GPU architecture | `applegpu_g13s` | 24 cores, no hardware `bf16` ALUs, no M5 NAX |
| Unified RAM | 64 GiB | Shared CPU/GPU/compositor pool |
| Sustained DRAM bandwidth | `309.46 GB/s` | 128 MiB Metal blit read+write median |
| Peak FP16 compute | `5.462 TFLOP/s` | 4096x4096 MPS FP16 GEMM median |
| Threadgroup cap | 832 to 1024 | Register-dependent per Metal pipeline |

## 2. Model Byte Inventory: `PocketAiHub/Ornith-1.5-35B-A3B-Abliterated-MLX-4bit`

Architecture (`qwen3_5_moe`): 40 decoder layers (30 GDN `linear_attention` + 10 `full_attention` at interval 4), `hidden_size = 2048`, `vocab_size = 248320`, 256 routed experts (`top_k = 8`, `moe_intermediate_size = 512`), 1 shared expert (`intermediate_size = 512`), 4-bit affine `group_size = 64` (8-bit affine `mlp.gate` and `mlp.shared_expert_gate`).

Exact safetensors payload (`20,401,929,952` bytes total tensor data; `19.00 GiB`):

| Component | Total Bytes | Active `c1` Bytes/tok | Share of `c1` |
|---|---:|---:|---:|
| `switch_mlp` (256 experts x 40 layers, 4-bit) | 18,119,393,280 | 566,231,040 (`8/256`) | 31.49% |
| `linear_attn` (30 GDN layers, 4-bit + BF16 conv/norm) | 570,420,480 | 570,420,480 | 31.72% |
| `self_attn` (10 full-attn layers, 4-bit + BF16 norm) | 153,364,480 | 153,364,480 | 8.53% |
| `shared_expert` + `shared_expert_gate` (40 layers) | 70,865,920 | 70,865,920 | 3.94% |
| `mlp.gate` router (40 layers, 8-bit) | 22,282,240 | 22,282,240 | 1.24% |
| `lm_head` (4-bit, `[248320, 2048]`) | 286,064,640 | 286,064,640 | 15.91% |
| `embed_tokens` (1 row of 4-bit `[248320, 2048]`) | 286,064,640 | 1,152 | 0.00% |
| Layer & final RMSNorms | 331,776 | 331,776 | 0.02% |
| `vision_tower` / `visual` (skipped language-only) | 893,142,496 | 0 | 0.00% |
| **Active weight subtotal (`c1`)** | - | **1,669,561,728 (1,669.56 MB)** | **92.84%** |
| GDN recurrent state R+W (`30 * 32 * 128 * 128 * 4B * 2`) | 62,914,560 | 125,829,120 | 7.00% |
| GDN conv1d state R+W (`30 * 8192 * 3 * 2B * 2`) | 1,474,560 | 2,949,120 | 0.16% |
| **Total `c1` traffic at `ctx = 0`** | - | **1,798,339,968 (1,798.34 MB)** | **100.00%** |

At `309.46 GB/s` sustained DRAM bandwidth:
- **Serial `c1` decode ceiling (`ctx = 0`)**: `309.46e9 / 1,798,339,968` = **`172.08 tok/s`** (`5.81 ms/tok`).

## 3. Context KV + GDN State Scaling (4K / 16K / 32K / 131K)

Only the 10 full-attention layers allocate growing KV cache (`num_key_value_heads = 2`, `head_dim = 256`):
- FP16/BF16 KV: `10 * 2 * 256 * 2 (K+V) * 2 B` = **`20,480 B/tok`** (`20 KiB/tok`).
- INT8 KV (Splash `--kv-format int8` with per-group/head scales): **`~10,880 B/tok`** (~0.53x FP16).
- GDN state per slot (constant across context): **`64,389,120 B`** (`64.39 MB`, `61.41 MiB`; R+W = `128.78 MB/step`).

| Context (`L`) | FP16 KV Resident | GDN State | Total Slot State (FP16 KV) | `c1` DRAM Traffic (FP16 KV) | `c1` Ceiling (FP16 KV) | INT8 KV Resident (~) | `c1` Ceiling (INT8 KV) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,096 (4K) | 83.89 MB (80 MiB) | 64.39 MB | 148.28 MB | 1,882.23 MB/tok | 164.41 tok/s | 44.56 MB | 167.92 tok/s |
| 16,384 (16K) | 335.54 MB (320 MiB) | 64.39 MB | 399.93 MB | 2,133.88 MB/tok | 145.02 tok/s | 178.26 MB | 156.56 tok/s |
| 32,768 (32K) | 671.09 MB (640 MiB) | 64.39 MB | 735.48 MB | 2,469.43 MB/tok | 125.32 tok/s | 356.52 MB | 143.61 tok/s |
| 131,072 (131K) | 2,684.35 MB (2.50 GiB) | 64.39 MB | 2,748.74 MB | 4,482.69 MB/tok | 69.03 tok/s | 1,426.06 MB | 95.97 tok/s |

## 4. Speculative Verify (`M = 4 / 5 / 8`) + DFlash2 (`Q4` / `FP16`) Cycle Floors

With DFlash2 (`felkhayer/Ornith-1.5-35B-A3B-DFlash2-Splash`, `block_size = 8`, `sliding_window = 2048`; or `jzinno/Ornith-1.5-35B-A3B-DFlash2`, `block_size = 16`, `sliding_window = 4096`):
- Draft weight payload (`model.safetensors`): `1,052,503,040 B` (`797,966,336 B` quantizable BF16 linears across 6 layers + `fc` + conv projections + `254,536,704 B` selector codebooks/norms/convs).
- Quantized Q4 (`gs=64`) active draft linears: `224.43 MB` (vs `797.97 MB` FP16); plus borrowed target `lm_head` (`286.06 MB`) and draft ring/conv/norms (`~20 MB` in Q4 cache path) = **`~530.5 MB` (Q4 draft step)** vs **`~1,084.3 MB` (FP16 draft step)**.
- Expected unique routed experts per layer over `M` verify tokens under uniform `top_k = 8 / 256` sampling (`E[uniq] = 256 * (1 - (1 - 8/256)^M)`), plus `1,232.11 MB` dense target + GDN R+W:

| Verify Width `M` | `E[uniq]` Experts | Routed MoE Read | Target Verify Total | Target Only (`309.46 GB/s`) | + Q4 DFlash2 Floor | + FP16 DFlash2 Floor |
|---:|---:|---:|---:|---:|---:|---:|
| `M = 4` | 30.53 / 256 | 2,160.87 MB | 3,392.98 MB | 10.96 ms | **12.68 ms/cycle** | 14.47 ms/cycle |
| `M = 5` | 37.58 / 256 | 2,659.65 MB | 3,891.76 MB | 12.58 ms | 14.29 ms/cycle | 16.08 ms/cycle |
| `M = 8` | 57.42 / 256 | 4,064.18 MB | 5,296.29 MB | 17.11 ms | 18.83 ms/cycle | **20.62 ms/cycle** |

Range across `M = 4..8` and Q4..FP16 DFlash2: **`12.68 - 20.62 ms/cycle`** DRAM floor.

## 5. oMLX 0.8.2 M1 Gate Audit

Static inspection of `/Applications/oMLX.app/Contents/Resources/omlx/` identified five M1 Max (`applegpu_g13s`) bottlenecks on Ornith:

1. **`_raises_spec_buffer_caps()` (`omlx/patches/mlx_lm_mtp/batch_generator.py:463`)**:
   Only returns `True` for `is_nax_available()` (M5) or `architecture.startswith("applegpu_g15")` (M3). M1 (`applegpu_g13s`) is excluded, leaving MLX at default 50 MB / 10 in-flight command-buffer caps instead of `_SPEC_BUFFER_CAPS = (200, 512)` during speculative cycles.
2. **`_add_rms_eligible()` (`omlx/patches/qwen35_verify_qmm.py:1443`)**:
   Requires `4096 < a.shape[-1] <= 16384`, excluding Ornith's `hidden_size = 2048` across all 40 layers (80 residual-add + RMSNorm pairs per verify pass).
   - **Kernel safety at `D = 2048`**: `_ADD_RMS_SOURCE` computes `STEPS = (D + 4095) / 4096 = 1`, guards reads/writes with `idx < D` and `base < D`, and zeros all 32 entries of `local_sums` in `sg == 0`. Sizing `n_threads = 512 if D <= 2048 else 1024` in `add_rms_norm()` runs 16 full simdgroups (zero wasted threads, exact same IEEE-754 sum, and safely below M1 Max threadgroup caps).
3. **`routed_verify_window()` (`omlx/patches/qwen35_moe_routed_decode.py:911`)**:
   Intercepts `Qwen3_5BatchInvariantForward._feed_forward` for `1..16` rows and dispatches ungrouped `(slot, row)` windows, bypassing `qwen35_moe_router.verify` -> `qwen35_moe_gate_up._fused_verify_switch` -> `moe_verify_gather.py` (which expert-sorts `(row, expert)` pairs via `pair_order()` for `2..8` rows so consecutive threadgroups reuse the same expert weights in L2).
4. **`_MIN_N = 1024` (`omlx/patches/m1_verify_qmv.py:52`, `omlx/patches/m1_verify_mma.py:108`)**:
   Excludes Ornith's `512 x 2048` `self_attn.k_proj`/`v_proj` (20 calls/verify) and `shared_expert` `512 x 2048` / `2048 x 512` projections. Both `m1_verify_qmv` (`K % 512 == 0, N % 16 == 0`) and `m1_verify_mma` (`K % 64 == 0, N % 8 == 0`) divide `512` cleanly.
5. **`activation_fp16_enabled` (`omlx/model_settings.py:489`, `omlx/utils/model_loading.py:1702`)**:
   Defaults to `False`. Keeping `bfloat16` activations on `applegpu_g13s` runs non-quantized ops and DFlash2 in software-emulated `bf16` (~0.5x FP16 ALU throughput). Enabling `activation_fp16_enabled=True` casts BF16 parameters to `float16` (`match_target_dtype` then casts DFlash2 to `float16`). In `qwen35_moe_router.py`, `omlx_router_topk` only has a packed fast path for `bfloat16_t`; adding `omlx_router_topk<half, NE>` restores fast packed top-k selection for `float16` logits.
