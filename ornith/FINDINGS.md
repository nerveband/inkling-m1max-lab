# Ornith-1.5-35B-A3B Optimization Findings on M1 Max

## Hardware and Target Model

- Hardware: Apple M1 Max with 24 GPU cores and 64 GiB unified memory
- Model: `Ornith-1.5-35B-A3B-Abliterated-MLX-4bit` (~19.1 GiB text weights, 40 layers, 256 routed experts with top-8 routing)
- Port tested: `http://127.0.0.1:18090/v1` via [run_ornith_exp.py](run_ornith_exp.py) and [bench_ornith.py](bench_ornith.py)

## Root Cause of Earlier Slow Runs

Prior live runs capped Splash at ~20 GB (`--max-memory 20018135982`).
With `--expert-memory auto`, Splash kept only 8544 of 10240 routed experts in RAM (83 percent) and streamed the rest from SSD.
Raising the Splash memory budget to `38G` and passing `--expert-memory off` keeps 100 percent of routed experts in RAM.
Keeping `--max-cache-disk 0` prevents SSD prefix cache writes.

## Engine Exploration Summary

### 1. Splash 1.4.0-m1 vs Splash 1.7.0-m1

- `b0a-splash14-20g` (`splash-m1 1.4.0-m1`, 20G, `qwen36` draft, 83 percent resident): `63.52 / 82.71 / 47.96 / 67.08 / 37.01 tok/s`, warm TTFT `721.0 ms`
- `b0b-splash14-38g-res` (`splash-m1 1.4.0-m1`, 38G, `--expert-memory off`, `qwen36` draft): `92.02 / 114.75 / 63.87 / 90.67 / 54.59 tok/s`, warm TTFT `571.0 ms`
- `b1a-splash17-q36` (`splash-1.7.0-m1`, 38G, `--expert-memory off`, `qwen36` draft): `101.59 / 194.40 / 76.25 / 98.26 / 75.36 tok/s`, geomean `102.20 tok/s`
- `b1b-splash17-fel` (`splash-1.7.0-m1`, 38G, `--expert-memory off`, `felkhayer` draft): `95.33 / 195.26 / 70.01 / 91.85 / 72.77 tok/s`, geomean `97.28 tok/s`
- `b1b2-splash17-jzinno`: rejected at startup because `jzinno` sets `sliding_window=4096` while Splash Qwen3.6 runtime expects `2048`
- `b1c-splash17-lang` (`+ --language-only`): `104.69 / 196.43 / 78.77 / 101.75 / 73.16 tok/s`, geomean `103.81 tok/s`, warm TTFT `220.04 ms`. Kept `--language-only`.
- `b1d-splash17-t06` (`--temp 0.6`): `101.71 / 185.57 / 66.39 / 84.69 / 67.39 tok/s`, 5/5 pass
- `b1e-splash17-bf16s` (`--state-format bf16`): `111.28 / 190.61 / 73.13 / 96.65 / 74.10 tok/s`, geomean `102.13 tok/s`. Dropped in favor of default `fp32` recurrent state (`103.81 tok/s`).
- `b1f` 2-repeat confirmation across `code_gen, code_edit, open_prose, warm_pi`:
  - `qwen36` medians: `106.56 / 198.50 / 70.59 / 101.24 tok/s` (4/4 wins)
  - `felkhayer` medians: `100.63 / 178.04 / 64.63 / 96.70 tok/s`

### 2. Native oMLX 0.8.2 + Overlay Exploration

Screening runs across `code_gen, code_edit, open_prose, warm_pi`:

| Config | `code_gen` | `code_edit` | `open_prose` | `warm_pi` | Geomean |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `b2a-omlx-bf16` (no draft) | 89.87 | 85.78 | 91.87 | 79.24 | 86.55 |
| `b2b-omlx-fp16` (`--fp16`, no draft) | 97.64 | 93.89 | 98.04 | 85.89 | 93.74 |
| `b2c-omlx-dflash-q4` (`jzinno` Q4 adaptive) | 45.17 | 137.07 | 58.05 | 69.00 | 70.57 |
| `b2d-omlx-dflash-fp16` (`jzinno` FP16 adaptive) | 72.98 | 136.27 | 43.47 | 71.19 | 74.48 |
| `b2e-omlx-dflash-b8` (`dflash` block 8) | 81.99 | 146.45 | 43.35 | 87.58 | 82.17 |
| `b2f-ov-caps` (`+ m1_caps`) | 84.20 | 150.53 | 44.76 | 88.90 | 84.27 |
| `b2g-ov-fast` (`+ add_rms_2048, min_n_512, fp16_fast_router`) | 83.32 | 151.64 | 49.76 | 86.14 | 85.78 |
| `b2h-ov-sorted` (`+ sorted_moe_verify`) | 72.41 | 152.62 | 51.95 | 74.81 | 80.95 |
| `b2i-ov-qmvmoe` (`+ m1_qmv_moe`, `MIN_ROWS=2`) | 88.03 | 160.66 | 49.73 | 79.72 | 86.53 |
| `b2i2-ov-qmvmoe-r1` (`+ m1_qmv_moe`, `MIN_ROWS=1`) | 86.30 | 156.86 | 50.73 | 74.14 | 84.47 |
| `b2j-ov-suffix` (`+ suffix_policy`) | 81.84 | 190.11 | 47.38 | 72.01 | 85.36 |

Key finding for oMLX:

- Switching target activations to FP16 (`b2b-omlx-fp16`) beats BF16 cleanly across all 4 workloads (`93.74` vs `86.55 tok/s`).
- With DFlash2 on oMLX, [ornith_m1_overlay.py](ornith_m1_overlay.py) (`m1_qmv_moe` in `b2i`) raised 4-workload geomean from `70.57` (`b2c`) to `86.53 tok/s` and hit `160.66 tok/s` on `code_edit` (`190.11 tok/s` with `suffix_policy` in `b2j`).
- Pure FP16 without drafting (`b2b-omlx-fp16`) still achieved higher overall geomean (`93.74 tok/s`, `open_prose` `98.04 tok/s`) because Ornith already decodes fast unspeculated at 400 GB/s memory bandwidth, so drafting lowers speed on low-acceptance prose.
- Full 5-workload run `b2-best-all` (`b2b-omlx-fp16`): `97.08 / 94.00 / 98.36 / 85.71 / 67.39 tok/s`, geomean `87.69 tok/s`, warm TTFT `552.26 ms`.

## Interleaved 6-Round A/B/A Proof

Six alternating fresh-server rounds compared the B1 winner (`splash17`: `splash-1.7.0-m1`, `qwen36` draft, `38G`, `--expert-memory off`, `--kv-format int8`, `--language-only`) against the B2 winner (`omlx`: oMLX 0.8.2, `--fp16`, no draft). Raw aggregate JSON: [summary.json](raw/aba-final/summary.json).

| Workload | `b0a` 20G Baseline | `omlx` FP16 Median (6 rounds) | `splash17` Median (6 rounds) | `splash17 / omlx` Ratio (95% CI) | Speedup vs `b0a` |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `code_gen` | 63.52 tok/s | 97.58 tok/s | 103.69 tok/s | 1.0626x (1.0596 to 1.0669) | 1.63x |
| `code_edit` | 82.71 tok/s | 93.97 tok/s | 196.41 tok/s | 2.0903x (2.0830 to 2.1004) | 2.37x |
| `open_prose` | 47.96 tok/s | 98.28 tok/s | 74.04 tok/s | 0.7534x (0.7352 to 0.7722) | 1.54x |
| `warm_pi` decode | 67.08 tok/s | 86.30 tok/s | 97.50 tok/s | 1.1298x (1.1266 to 1.1643) | 1.45x |
| `warm_pi` warm TTFT | 721.00 ms | 648.81 ms | 232.34 ms | 2.79x faster TTFT | 3.10x faster TTFT |
| `needle_32k` | 37.01 tok/s | 67.20 tok/s | 75.26 tok/s | 1.1200x (1.0969 to 1.1324) | 2.03x |
| 5-workload Geomean | 57.58 tok/s | 87.82 tok/s | 102.13 tok/s | 1.1629x (1.1573 to 1.1693) | 1.77x |

Health and safety telemetry across all 6 rounds:

- Quality pass rate: `6/6` on every workload for both `splash17` and `omlx` (30/30 per arm, zero repetition collapse, 100 percent dual-needle 32K recall)
- Swap usage: `3293.62 MiB` min and `3293.62 MiB` max (`0.0 MiB` delta)
- macOS memory pressure: `normal` across every round

## Deployed Winner Configuration

- Winner deployed: `splash-1.7.0-m1` staged under `~/.local/share/splash-1.7.0-m1/` with wrapper at `~/.local/bin/splash-1.7.0-m1` (`Splash 1.7.0-m1`), leaving `/opt/homebrew/bin/splash-m1` untouched
- Wrapper `~/.local/bin/splash-no-disk`: runs `~/.local/bin/splash-1.7.0-m1`, keeps `--max-cache-disk 0` on `serve`, and adds `--expert-memory off` plus `--language-only` only when `serve` arguments match `Ornith-1.5-35B-A3B`
- Global settings `~/.omlx/settings.json`: `integrations.splash_path` = `~/.local/bin/splash-no-disk`, `integrations.splash_max_memory_gb` = `38`
- Per-model settings `~/.omlx/model_settings.json`: both `Ornith-1.5-35B-A3B-Abliterated-MLX-4bit` and `ornith-1.5-35b-a3b-mlx-4bit` keep `external_engine` = `"splash"` and set `dflash_draft_model` to `~/.omlx/models/incoai/Qwen3.6-35B-A3B-DFlash2`
