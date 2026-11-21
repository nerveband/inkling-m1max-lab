# Inkling 2 Mini Findings on Apple M1 Max

Every test in this document ran on a 24-core Apple M1 Max with 64 GiB of unified memory on macOS 27.0.1.

The fastest setup that passes every quality test uses oMLX 0.8.2 with custom Metal kernels (`--m1-fast`) and affine 4-bit experts (`--affine4`) at draft depth 6 on `omlx-oq4`. Affine 4-bit is a 4-bit number format that uses a scale and offset for each group of 64 weights. Speculative drafting is a method where a small draft head guesses future tokens and the main model tests them in one pass.

On an idle GPU, this setup reaches 49.43 tokens per second in 7.14 s on a primed Pi screen. On a fresh 6-task controlled A/B/A test, it hits a median of 49.09 tokens per second in 13.17 s. Stock `oQ4e` controls reached 37.87 and 38.00 tokens per second on the same test. In direct normal Pi use, it hits 50.08 tokens per second in the native model window across 5 turns in 9.39 s.

Running `--m1-fast` on the lighter `omlx-oq3` 3-bit weights hits 49.38 tokens per second in 7.28 s in direct normal Pi. That beats the earlier 39.67 tokens per second and 8.48 s baseline. We promote `omlx-oq4` instead of `omlx-oq3` because `omlx-oq4` passed all 6 reopened no-tool Pi turns, while `omlx-oq3` missed 2 of 6 turns.

Evidence links: [REPRODUCE.md](REPRODUCE.md) · [Promoted screen](raw/speed-m1-final-affine4/pi-scores.json) · [Controlled A/B/A](raw/m1-final-compare/compare.txt) · [Qualification](raw/qual-m1-final) · [Direct 4-bit Pi](raw/public-m1-final-pi/direct-pi.json) · [Direct 3-bit Pi](raw/public-m1-oq3-pi/direct-pi.json)

## Main benchmark results

| Setup | Decode tok/s | Task time | Test scope |
|---|---:|---:|---|
| Stock GGUF `UD-Q4_K_M`, draft off | 18.74 | 22.32 s | 6 Pi tasks |
| Stock GGUF `UD-Q4_K_M`, DSpark depth 6 | 27.94 | 18.66 s | 6 Pi tasks |
| Stock oMLX `oQ4e`, draft off | 23.83 | 18.27 s | 6 Pi tasks |
| Stock oMLX `oQ4e`, draft depth 3 | 33.73 | 15.77 s | 6 Pi tasks |
| Stock oMLX `oQ4e`, draft depth 6 | 38.17 | 15.09 s | 6 Pi tasks (`37.87` and `38.00` in repeat A/C controls) |
| oMLX `oQ3e` with `sort32`, primed | 40.94 | 8.16 s | 1 primed screen (`39.67 tok/s`, `8.48 s` direct Pi) |
| oMLX `oQ4e` `--m1-fast` only, depth 6, primed | 42.26 | 9.62 s | 1 primed screen (keeps NVFP4 experts) |
| oMLX `oQ4e` `--m1-fast --affine4` d=6 (A/B/A) | 49.09 | 13.17 s | 6 fresh Pi tasks (`47.76` to `49.39`, +29.4%) |
| oMLX `oQ4e` `--m1-fast --affine4` d=6 (primed) | 49.43 | 7.14 s | Primed screen (`178` tokens, `3,323 ms` backbone) |
| oMLX `oQ4e` `--m1-fast --affine4` d=6 (direct Pi) | 50.08 | 9.39 s | Direct prewarmed Pi (`213` tokens, 5 turns) |
| oMLX `oQ3e` `--m1-fast` d=6 (direct Pi) | 49.38 | 7.28 s | Direct prewarmed Pi (`178` tokens, 4 turns) |
| `LookAtMyQuant` 8.9 GB + v2 draft (`sg=16` shim) | 16.01 to 26.06 | cli-only | Code `26.06`, edit `22.92`, prose `16.01` |
| GGUF n-gram lookup 256 + DSpark 6 (copy prompt) | 114.39 | n/a | 6 direct 2K copy prompts only |

## Why the promoted setup is faster

KV cache is memory that stores earlier prompt and reply states so the model skips re-reading them. Prefix cache is saved KV cache reused when prompts start with the same text. Greedy decoding is picking the single top next token on every step.

Nine code changes work together inside [m1_overlay.py](m1_overlay.py) and [patches](patches):

1. `--affine4` converts NVFP4 experts into affine 4-bit (`group_size=64`) experts during model load. On M1 Max, MVFP4 dequantize code runs slower than M1 affine 4-bit MLX kernels. Keeping NVFP4 with `--m1-fast` hit 42.26 tok/s, while adding `--affine4` jumped to 49.43 tok/s.
2. `sort32` sorts small MoE batches at 32 indices instead of 64.
3. `m1_attn` replaces draft-test attention with a custom Split-K Metal kernel (`verify_banded_sink_attention`). It caps `nsplits <= 64` for prompts longer than 8,192 tokens. Against the reference implementation, it measured `cos=0.99999964` (`max_err=0.000031`) at 31,954 tokens and `cos=0.99999958` (`max_err=0.000015`) at 64,717 tokens.
4. `fast_decode` and `verify_compile` fuse decode and draft-test steps and remove extra tensor copies inside [patches/inkling2_fast_decode.py](patches/inkling2_fast_decode.py).
5. `split_in_proj` splits the combined attention input projection into separate linear layers for faster decode and draft-test shapes.
6. `skip_lm_head` skips computing vocabulary logits on non-last prefill tokens. Logits are raw token scores before picking a word.
7. `fast_greedy` moves greedy token selection into a lightweight Metal kernel.
8. `prefill_pad64` pads prefill length up to multiples of 64 to keep Metal matrix kernels aligned.
9. `caps` adjusts Metal and MLX allocator caps in oMLX.

In the promoted A/B/A test ([compare.txt](raw/m1-final-compare/compare.txt)), Candidate B passed 6 of 6 tasks on an idle GPU. It scored median 49.09 tok/s (`47.76` to `49.39`), wall time 13.17 s, and TTFT (time to first token) 4.986 s. Control A scored 37.87 tok/s, 15.31 s wall time, and 5.448 s TTFT. Control C scored 38.00 tok/s, 15.39 s wall time, and 5.427 s TTFT.

Bootstrap 95% intervals for Candidate B over Controls A and C were `1.2723` to `1.3014` and `1.2700` to `1.3037` for decode speed. Earlier tests taken during 54% to 62% desktop GPU load slowed Leg B down to 47.19 tok/s median and Control C down to 29.92 tok/s. Test on an idle GPU when comparing speeds.

## Quality and long context tests

Memory pressure is a macOS signal that shows how full RAM is. Swap is disk space used when RAM fills up.

With `--context 32768 --slots 1` ([raw/qual-m1-final](raw/qual-m1-final)), the promoted `oQ4e` stack passed all 6 quality gates at pressure level 1 with zero swap growth and 32.06 GB peak GPU memory:

- Reasoning separation passed (`391` tokens).
- Primed Pi task passed in 12.74 s.
- Held-out Pi task passed in 11.23 s at 51.74 tok/s.
- Ordered key retrieval passed at 15,569 tokens (310.35 prefill, 33.86 decode tok/s, 58.52 s). It also passed at 31,954 tokens (289.39 prefill, 30.02 decode tok/s, 118.84 s).
- A 12-turn HTTP conversation passed 12 of 12 turns.
- A reopened 6-turn no-tool Pi session passed all 6 of 6 turns.

In an isolated `--context 65536 --slots 1` run ([needle-64k.json](raw/qual-m1-final/needle-64k.json)), key retrieval at 64,717 input tokens passed at pressure level 1. Peak GPU memory was 41.28 GB with zero swap growth. It reached 249.81 tok/s prefill, 20.19 tok/s decode, and 268.01 s total wall time. Stock `oQ4e` previously ran the same prompt at 15.87 tok/s decode in 307.58 s.

An earlier `--context 65536 --slots 2` test run before adding the `nsplits <= 64` cap hit macOS pressure level 2 at 41.64 GB GPU memory during 64K draft testing and stopped. Setting `--slots 1` and keeping `nsplits <= 64` avoids that spike.

## Ideas tested and rejected

We tested many ideas from hardware theory and public posts. Almost all of them hurt speed or broke tool calls on M1 Max:

- Draft depth sweep on M1 fast stack: depth 3 hit 32.89 tok/s (`9.05 s`), and depth 4 hit 35.95 tok/s (`8.61 s`). Depth 5 hit 36.71 tok/s (`8.49 s`), and depth 6 reached 48.74 to 49.43 tok/s (`7.14` to `7.59 s`). Depth 7 fell back to 37.10 tok/s (`8.62 s`), so depth 6 is the clear winner.
- `dspark_fast` (`draft4`, fused draft QKV, and `markov_topk=256` vocab shortlist): acceptance fell from `142/156` to `128/165` and speed dropped to `34.52` to `34.66 tok/s` (`8.91` to `8.95 s`).
- `subwarp_moe`: slower at `47.93 tok/s` (`7.58 s`).
- `ctx_copy` on `affine4`: slower at `48.74 tok/s` (`8.86 s`, generated 218 tokens).
- `fuse_draft_qkv` alone: slower at `48.60 tok/s` (`7.56 s`).
- `caps200`: slightly higher wall time at `49.35 tok/s` (`7.71 s`).
- `fuse_moe_conv_res` (`moe_combine_conv_res`): mathematically exact (`max_err=0.0`), but slower at `46.28 tok/s` (`7.55 s`).
- `loom_draft_head` (4-bit draft head + Markov matrix): regressed to `43.83 tok/s` (`8.08 s`).
- `attn4` (4-bit attention weights): regressed to `43.72 tok/s` (`8.20 s`).
- `margin30` (`max_logit_gap=0.30`): slowed wall time to `8.28 s` (`48.86 tok/s`).
- `top6` static routing and `top6+margin30`: both failed strict Pi task correctness (`correct=false`), even though `top6+margin30` printed at `51.43 tok/s` (`13.32 s`).
- `oQ2.5e` (2.5-bit experts): slower than `oQ3e` at `25.54 tok/s` fresh and `25.36 tok/s` primed.
- FP16 expert casting: printed repeated punctuation at `49.61 tok/s` and called zero tools (`correct=false`).
- N-gram lookup 256 + DSpark 6 on GGUF: hit `114.39 tok/s` on direct 2K copy tasks, yet dropped to `15.44 tok/s` on fresh code and `24.69 tok/s` (`19.74 s`) on Pi tasks.

## Community X and Hugging Face leads

We tested six public posts and repositories on this M1 Max. Full revisions and raw numbers sit in [CANDIDATES.md](CANDIDATES.md) and [lamq-m1max-bench.json](raw/lamq-m1max-bench.json):

- `ShinMegamiBoson/lookatmyquant` (`@shinboson`): Stock draft testing fails on M1 Max because `matmul_rotated_sg8_f16` uses threadgroup size 1024, above the M1 Max limit of 832. Adding an `lm_head` `sg=16, ntg=160` shim let the 8.9 GB checkpoint and distilled v2 drafter run at `26.06 tok/s` on code (`adaptive`), `22.92` on edit (`linear_b6`), and `16.01` on prose. It has no OpenAI server or tool parser, and runs around half the speed of oMLX on M1 Max. Its later `7639fd0d` M5 `_has_mpp()` path returns `False` on M1 (`applegpu_g13s`).
- `exo-explore/exo` (`@exolabs`): Public code and open pull requests contain only `Inkling-Small` model cards and zero Inkling 2 Mini runtime or kernel code.
- `realugbun/llama.cpp` `inkling2-speed` (`@JacksonAtkinsX`): Targets CUDA graphs with no Metal path.
- `sybil-solutions/inference-optimisation-skill` (`@cheatyyyy` / `@0xSero`): Tested all ideas on M1 Max. Kept depth 6, Split-K draft-test attention, and fused decode. Rejected depth 3 and `markov_topk=256`.
- `loom-fun/unsloth-Inkling-2-Mini-loom` (`@bebuu_ai`): Built for AMD ROCm, RDNA, CDNA, XDNA, and Vulkan with zero Metal or ARM64 backend. Ported 4-bit draft head and 4-bit attention tricks both lost speed on M1 Max.

## Previous registry baseline

In earlier runs without `--m1-fast --affine4`, stock `oQ4e` at depth 3 passed all six gates in `local-ai-registry` PR 465 at `20.4 tok/s` on the short story test. Stock `oQ4e` depth 6 hit `14.8 tok/s` against the `15.0 tok/s` floor there. With `--m1-fast --affine4`, single-stream Pi decode rises from `37.87` to `49.09 tok/s` (+29.4%).
