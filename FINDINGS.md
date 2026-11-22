# Inkling 2 Mini Findings on Apple M1 Max

Every test in this document ran on a 24-core Apple M1 Max with 64 GiB of unified memory on macOS 27.0.1.

The fastest setup that passes every quality test uses oMLX 0.8.2 with the promoted v2 M1 Max Metal kernel stack (`--m1-fast`), affine 4-bit experts (`--affine4`), an 8 GB RAM hot prefix cache (`--hot-cache-max-size 8GB`), and default greedy coding temperature `0.0` at draft depth 6 on `omlx-oq4`. Affine 4-bit is a 4-bit number format that uses a scale and offset for each group of 64 weights. Speculative drafting is a method where a small draft head guesses future tokens and the main model tests them in one pass.

On an idle GPU in a controlled 6-task A/B/A test ([v2 Pi compare](raw/m1-v2-compare/pi-compare.txt)), v2 reaches a median of **64.21 tokens per second** in 9.76 s, beating the prior v1 M1 Max stack at 49.45 and 49.39 tokens per second in 12.92 to 12.96 s (+30.0%) and stock `oQ4e` at 37.87 tokens per second (+69.6%). On 6-sample direct HTTP runs ([v2 mix summary](raw/m1-v2-compare/mix-aba-summary.json)), default mode reaches 42.50 tok/s on code generation, 57.45 tok/s on code edits (0.212 s warm TTFT versus 6.651 s on v1), and 30.21 tok/s on open prose. With `enable_thinking=false` and `generation_opener=content`, coding reaches **61.36 tok/s** on code generation (0.077 s warm TTFT) and **64.98 tok/s** on code edits (0.130 s warm TTFT).

Running v1 `--m1-fast` on the lighter `omlx-oq3` 3-bit weights previously hit 49.38 tokens per second in 7.28 s in direct normal Pi. We promote `omlx-oq4` instead of `omlx-oq3` because `omlx-oq4` passed all 6 reopened no-tool Pi turns, while `omlx-oq3` missed 2 of 6 turns.

Evidence links: [REPRODUCE.md](REPRODUCE.md) · [v2 Pi A/B/A](raw/m1-v2-compare/pi-compare.txt) · [v2 Mix A/B/A](raw/m1-v2-compare/mix-aba-summary.json) · [v2 Qualification](raw/qual-m1-v2) · [v1 A/B/A](raw/m1-final-compare/compare.txt) · [v1 Qualification](raw/qual-m1-final)

## Main benchmark results

| Setup | Decode tok/s | Task time | Test scope |
|---|---:|---:|---|
| Stock GGUF `UD-Q4_K_M`, draft off | 18.74 | 22.32 s | 6 Pi tasks |
| Stock GGUF `UD-Q4_K_M`, DSpark depth 6 | 27.94 | 18.66 s | 6 Pi tasks |
| Stock oMLX `oQ4e`, draft off | 23.83 | 18.27 s | 6 Pi tasks |
| Stock oMLX `oQ4e`, draft depth 3 | 33.73 | 15.77 s | 6 Pi tasks |
| Stock oMLX `oQ4e`, draft depth 6 | 38.17 | 15.09 s | 6 Pi tasks (`37.87` and `38.00` in v1 A/C controls) |
| oMLX 0.8.2 `oQ4e --m1-fast --affine4` (v1) | 49.45 | 12.96 s | 6 Pi tasks (`49.09` v1 promo, `49.45/49.39` v2 A/C) |
| **oMLX 0.8.2 `oQ4e --m1-fast --affine4` (v2)** | **64.21** | **9.76 s** | **6 Pi tasks (`63.97` to `64.68`, +30.0% vs v1)** |
| **v2 Direct HTTP, default thinking (n=6)** | **42.50 / 57.45 / 30.21** | **0.21 s warm TTFT** | **code_gen / code_edit / prose** |
| **v2 Direct HTTP, `--no-think` opener (n=6)** | **61.36 / 64.98 / 27.77** | **0.08 to 0.13 s TTFT** | **code_gen / code_edit / prose** |
| oMLX `oQ3e` `--m1-fast` d=6 (direct Pi) | 49.38 | 7.28 s | Direct prewarmed Pi (`178` tokens, 4 turns) |
| `LookAtMyQuant` 8.9 GB + v2 draft (`sg=16` shim) | 16.01 to 26.06 | cli-only | Code `26.06`, edit `22.92`, prose `16.01` |

## Why the promoted v2 setup is faster

KV cache is memory that stores earlier prompt and reply states so the model skips re-reading them. Prefix cache is saved KV cache reused when prompts start with the same text. Greedy decoding is picking the single top next token on every step.

The promoted `M1_FAST_FEATURES` overlay ([m1_overlay.py](m1_overlay.py)) layers 19 verified kernels and runtime fixes onto oMLX 0.8.2:

1. `m1_qmv_moe` ([m1_qmv_moe.py](patches/m1_qmv_moe.py)) replaces MLX `gather_sort`, `gather_qmm`, and `scatter_unsort` on `M=1..16` with a register-diet 5-stage Metal MoE pipeline (3 base pointers, strict `C==2` token pair splitting with sequential activation reuse, `0x64006400u` `half4` unpack, and a 3-kernel `DIRECT_M1` single-row path). On a 48-visit BF16 chain, it cuts `M=1` MoE time from `11.87 ms` to `7.96 ms` (`1.49x`) and `M=7` from `58.76 ms` to `43.61 ms` (`1.35x`).
2. `verify_flush`, `bounded_cache`, `bf16_loop_ctrl`, and `prefill_fast` remove per-layer stalls inside the main transformer, cut M1 loop controller cost by about `10 ms` per verify cycle across every workload, and cap KV state buffers cleanly.
3. `predraft_fix`, `dspark_2bit`, `draft4`, and `fuse_draft_qkv` fix the recurrent Markov starting row and compress speculative head screening while keeping exact draft acceptance.
4. `ttft_fixes` (`early_first_emit`, `enable_thinking_fix`, `prompt_tail_cache`) keeps prefix cache hits intact across prompt tails and supports `generation_opener=content` with `enable_thinking=false`, cutting warm 4K code-edit TTFT from `6.651 s` on v1 to `0.130 s` to `0.212 s`.
5. `--hot-cache-max-size 8GB` keeps multi-gigabyte coding prefixes in RAM instead of spilling to SSD, and `--default-temp 0.0` ensures `fast_greedy` activates when clients omit temperature.
6. All 9 v1 kernels remain active: `--affine4`, `sort32`, `m1_attn` (`verify_banded_sink_attention` with `nsplits <= 64`, `cos=0.99999964` at 32K and `cos=0.99999958` at 64K), `fast_decode`, `verify_compile`, `split_in_proj`, `skip_lm_head`, `fast_greedy`, `prefill_pad64`, and `caps`.

In the promoted v2 A/B/A test ([pi-compare.txt](raw/m1-v2-compare/pi-compare.txt)), Candidate B passed 6 of 6 tasks on an idle GPU with median `64.21 tok/s` (`63.97` to `64.68`), wall time `9.76 s`, and TTFT `4.793 s`. Control A scored `49.45 tok/s` (`12.96 s`) and Control C scored `49.39 tok/s` (`12.92 s`), with 95% bootstrap decode intervals of `1.2952` to `1.3057` and `1.2970` to `1.3073` over v1. Always test on an idle GPU because earlier v1 tests during 54% to 62% desktop GPU load slowed Control C down to `29.92 tok/s`.

## Quality and long context tests (v2)

Memory pressure is a macOS signal that shows how full RAM is. Swap is disk space used when RAM fills up.

With `--context 32768 --slots 1` ([raw/qual-m1-v2](raw/qual-m1-v2)), the promoted v2 `oQ4e` stack passed all 6 quality gates at pressure level 1 with zero swap growth (3365.62 MiB flat) and 22.75 GB peak GPU memory (down from 32.06 GB on v1):

- Reasoning separation passed (`391` tokens).
- Primed Pi task passed in `9.47 s` (down from `12.74 s` on v1).
- Held-out Pi task passed in `10.46 s` (down from `11.23 s` on v1).
- Ordered key retrieval passed at 15,569 tokens (`326.81` prefill, `43.56` decode tok/s, `51.95 s`) and at 31,954 tokens (`306.65` prefill, `43.33` decode tok/s versus `30.02` on v1, `109.96 s`).
- A 12-turn HTTP conversation passed 12 of 12 turns.
- A reopened 6-turn no-tool Pi session passed all 6 of 6 turns.

In an isolated `--context 65536 --slots 1` run ([needle-64k.json](raw/qual-m1-v2/needle-64k.json)), key retrieval at 64,717 input tokens passed all 3 keys at pressure level 1 with zero swap growth (3341.62 MiB flat) and 24.39 GB peak GPU memory (down from 41.28 GB on v1). It reached `270.61 tok/s` prefill, **37.15 tok/s decode** (versus `20.19 tok/s` on v1 and `15.87 tok/s` stock), and `246.81 s` total wall time.

## Ideas tested and rejected

We tested many ideas from hardware theory and public posts and rejected the ones that hurt speed or quality on M1 Max:

- `fp8_kv_full` (FP8 E4M3 KV cache on v2): Raised `code_gen` to `48.93` and `open_prose` to `32.95 tok/s` and passed 16K and 32K retrieval, yet drifted on turn 1 of the reopened no-tool Pi session (`5/6` with an empty content turn) and slowed Pi warm and held checks to `12.09 s` and `13.38 s`. Dropped from the promoted stack to protect strict `6/6` parity.
- `suffix_policy` (`OMLX_MTP_DRAFT_POLICY=1`): Raised Pi to `64.95` and `open_prose` to `33.71 tok/s`, yet dropped `code_gen` from `48.93` to `36.34 tok/s` and `code_edit` from `55.57` to `51.63 tok/s` due to plain `d0=44` fallback cycles. Dropped from default Inkling settings.
- Dropping `bf16_loop_ctrl` (`fast_decode` controller only): Added about `10 ms` per verify cycle across every workload (`code_gen` dropped `50.67` to `42.12`, `code_edit` `56.12` to `40.74`, Pi `59.76` to `53.07`). Kept `bf16_loop_ctrl` enabled.
- Draft depth sweep on M1 fast v1 stack: depth 3 hit `32.89 tok/s` (`9.05 s`), depth 4 hit `35.95 tok/s` (`8.61 s`), depth 5 hit `36.71 tok/s` (`8.49 s`), depth 6 reached `48.74` to `49.43 tok/s` (`7.14` to `7.59 s`), and depth 7 fell back to `37.10 tok/s` (`8.62 s`).
- `dspark_fast` (`draft4`, fused draft QKV, and `markov_topk=256` vocab shortlist): acceptance fell from `142/156` to `128/165` and speed dropped to `34.52` to `34.66 tok/s`.
- `subwarp_moe` (`47.93 tok/s`), `ctx_copy` on `affine4` (`48.74 tok/s`), `fuse_draft_qkv` alone (`48.60 tok/s`), `caps200` (`49.35 tok/s`), `fuse_moe_conv_res` (`46.28 tok/s`), `loom_draft_head` (`43.83 tok/s`), `attn4` (`43.72 tok/s`), `margin30` (`48.86 tok/s`), `top6` and `top6+margin30` (`correct=false`), `oQ2.5e` (`25.54 tok/s`), and FP16 expert casting (`correct=false`): all failed speed or correctness checks.

## Community X and Hugging Face leads

We tested six public posts and repositories on this M1 Max. Full revisions and raw numbers sit in [CANDIDATES.md](CANDIDATES.md) and [lamq-m1max-bench.json](raw/lamq-m1max-bench.json):

- `ShinMegamiBoson/lookatmyquant` (`@shinboson`): Stock draft testing fails on M1 Max because `matmul_rotated_sg8_f16` uses threadgroup size 1024, above the M1 Max limit of 832. Adding an `lm_head` `sg=16, ntg=160` shim let the 8.9 GB checkpoint and distilled v2 drafter run at `26.06 tok/s` on code (`adaptive`), `22.92` on edit (`linear_b6`), and `16.01` on prose. It has no OpenAI server or tool parser, and runs around half the speed of oMLX on M1 Max. Its later `7639fd0d` M5 `_has_mpp()` path returns `False` on M1 (`applegpu_g13s`).
- `exo-explore/exo` (`@exolabs`): Public code and open pull requests contain only `Inkling-Small` model cards and zero Inkling 2 Mini runtime or kernel code.
- `realugbun/llama.cpp` `inkling2-speed` (`@JacksonAtkinsX`): Targets CUDA graphs with no Metal path.
- `sybil-solutions/inference-optimisation-skill` (`@cheatyyyy` / `@0xSero`): Tested all ideas on M1 Max. Kept depth 6, Split-K draft-test attention, and fused decode. Rejected depth 3 and `markov_topk=256`.
- `loom-fun/unsloth-Inkling-2-Mini-loom` (`@bebuu_ai`): Built for AMD ROCm, RDNA, CDNA, XDNA, and Vulkan with zero Metal or ARM64 backend. Ported 4-bit draft head and 4-bit attention tricks both lost speed on M1 Max.

## Previous registry baseline

In earlier runs without `--m1-fast --affine4`, stock `oQ4e` at depth 3 passed all six gates in `local-ai-registry` PR 465 at `20.4 tok/s` on the short story test. Stock `oQ4e` depth 6 hit `14.8 tok/s` against the `15.0 tok/s` floor there. With `--m1-fast --affine4`, single-stream Pi decode rises from `37.87` to `49.09 tok/s` (+29.4%).
