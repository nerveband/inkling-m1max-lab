# Candidate Paths Tested on M1 Max, 2026-11-21

Every row targets the full three-pass Inkling 2 Mini model on a 64 GiB M1 Max. Speed numbers from other Macs or GPUs did not rank candidates here.

## Ranked candidate table

| Order | Target and proposal | M1 Max path or blocker | Source |
|---:|---|---|---|
| 1 | Unsloth `UD-Q4_K_M` + `Q8_0` DSpark | Runs via `llama.cpp`. Separate 402,278,560-byte draft file. Maximum draft depth is 7. Draft taps target layers 5, 23, 41, and 48. | [Companion](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/blob/ae9327c5fbc7333116ba32eee78bb5d2bbefae4f/dspark/README.md), [loop fix](https://github.com/unslothai/llama.cpp/pull/389), [tap fix](https://github.com/unslothai/llama.cpp/pull/390) |
| 2 | Target n-gram suffix lookup | Runs `ngram-simple`, `map-k`, `map-k4v`, and `mod`. Fast only on exact text copies (`114.39 tok/s`). Slow on fresh code (`15.44 tok/s`). | [Runtime](https://github.com/unslothai/llama.cpp/releases/tag/b12588-mix-c0c3de8) |
| 3 | MLX-LM PR 2086 + `mlx-community/Inkling-2-Mini-4bit` | Code `0c4c9919`, weights `9856dc53` omit draft head. Stock parser lacked an Inkling branch. oMLX rejects MLX-LM tensor names. | [Loader](https://github.com/ml-explore/mlx-lm/pull/2086), [quant](https://huggingface.co/mlx-community/Inkling-2-Mini-4bit) |
| 4 | oMLX 0.8.2 + native `oQ4e-MTP` (`3eea0427`) | Promoted winner. Bundled `mtp.safetensors`. With `--m1-fast --affine4` depth 6, hits `49.09 tok/s` A/B/A and passes all 6 Pi gates + 64K retrieval. | [Loop and rollback](https://github.com/jundot/omlx/pull/5468), [native quant](https://huggingface.co/mlx-community/Inkling-2-Mini-oQ4e-mtp) |
| 5 | Upstream `b12600` + `IQ3_XXS` GGUF | Binary `29b52d39`, IQ3 `279557cf`. Changes model weights and runs slower than MLX 3-bit and 4-bit on M1 Max. | [Upstream](https://github.com/ggml-org/llama.cpp/releases/tag/b12600), [IQ3](https://huggingface.co/InterpreticAI/Inkling-2-Mini-IQ3_XXS-GGUF) |
| 6 | NetProjects MLX-VLM + DSpark | Fork `6208d249` (`0.8.0+inkling2mini.2`) + affine 4-bit `d90008cb`. Hit `25.22 tok/s` with prefix cache and drafting, half as fast as oMLX `--m1-fast`. | [Implementation](https://github.com/internet-projects/inkling2-mlx) |
| 7 | TensorFold native Metal (`PR 4337`) | Revision `f3a6a992` built locally with Zig 0.17. Measured `8.03 tok/s` with draft on or off and leaked tool markers in visible output. | [PR and device checks](https://github.com/ashhart/TensorFold/pull/4337) |
| 8 | EXO single Mac engine | Advertises M5 Max/Ultra results. Public `exo` `a60b9569` and PRs only carry `Inkling-Small` cards and no Mini runtime code. | [EXO post](https://x.com/exolabs/status/2123953913342935486), [source tree](https://github.com/exo-explore/exo/tree/a60b95698ca81737d367443674dd5563cbc9559b/resources/inference_model_cards/inkling) |
| 9 | DSpine, DFlash, DFlash 2, Splash, EAGLE heads | Tencent `DSpine` documents `Qwen3-8B` and `Qwen3.8-27B` heads. Hub search found only DSpark and bundled `oQ` draft heads for Mini. | [Tencent](https://github.com/Tencent/DSpine), [Mini model](https://huggingface.co/thinkingmachines/Inkling-2-Mini), [formats](https://huggingface.co/blog/inkling2mini) |
| 10 | Small-model draft, layer skip, GPU/ANE split | No matching small Mini tokenizer/target pair or working Apple Neural Engine draft artifact exists for Mini. | [Released model](https://huggingface.co/thinkingmachines/Inkling-2-Mini), [EXO implementation](https://github.com/exo-explore/exo) |

## Detailed notes on tested engines

Stock oMLX 0.8.0rc1 lacked DSpark loop support. We isolated oMLX 0.8.2 (`91a7b055` wheel SHA-256) with MLX `0.32.4`, MLX-LM `0.32.3.dev8`, and Transformers `5.22.2`. Adaptive drafting logged depth 6, while fixed drafting used depths 3 through 7.

Native `oQ3e` (`5552e32e`, 12.76 GiB) uses affine 3-bit experts. It hit `49.38 tok/s` in direct Pi with `--m1-fast`, yet missed 2 of 6 reopened no-tool Pi turns. Native `oQ2.5e` (`d4a4f3f5`, 10.77 GiB) scored `25.54 tok/s` fresh and `25.36 tok/s` primed.

GGUF `Q8_0` draft metadata sets `dflash.has_confidence_head=false`. The staged `llama.cpp` server turns off confidence cutoffs when that field is false.

## Community X and Hugging Face leads tested on M1 Max

| Source | Claim and code | M1 Max path, blocker, or measured speed | Outcome |
|---|---|---|---|
| [@shinboson](https://x.com/shinboson/status/2124147300343943444) | `ShinMegamiBoson/lookatmyquant` (`runtime_mlx/inkling/`): 8.9 GB (96.8%) and 5.4 GB (91.2%) trellis checkpoints + 4-bit DSpark trees (`101` to `125 tok/s` on M5 Max) | Stock verify fails (`Thread group size 1024 > max 832`). With an `lm_head` `sg=16, ntg=160` shim (`192` tokens, v2 drafter): `26.06 tok/s` code (`adaptive`), `22.92` edit (`linear_b6`), `16.01` prose. Trees scored lower (`17.09` to `23.59 tok/s`). No HTTP server or tool parser. | Evaluated in [lamq-m1max-bench.json](raw/lamq-m1max-bench.json). Runs at half oMLX speed on M1 Max. |
| [@shinboson](https://x.com/shinboson/status/2124179977906364606) | `lookatmyquant` commits `7639fd0d` / `8080bd77`: M5 MPP prefill, FA2 SDPA, `moe_wsum_sconv4_res`, `top6`, `max_logit_gap=0.30` | `_has_mpp()` is `False` on M1 (`applegpu_g13s`). Ported lossless `moe_combine_conv_res` (`max_err=0.0`, `46.28 tok/s`, `7.55 s`), `margin30` (`48.86 tok/s`, `8.28 s`), static `top6` (`47.69 tok/s`, `correct=false`), and `top6+margin30` (`51.43 tok/s`, `correct=false`). | Tested all M1 paths. Kept exact full top-8 greedy routing. |
| [@exolabs](https://x.com/exolabs/status/2123953913342935486) | Custom Apple Silicon engine (`125 tok/s` M5 Max, `210 tok/s` M5 Ultra) | Public `exo-explore/exo` (`a60b9569`) and open PRs (`#2621`) hold only `Inkling-Small` cards and zero Mini engine code. | Unreleased engine. |
| [@JacksonAtkinsX](https://x.com/JacksonAtkinsX/status/2124165894188511501) | `realugbun/llama.cpp` `inkling2-speed` branch | CUDA-only graph fusions and depth-state RMSNorm fix (`7ec7714`). No Metal path exists. | Not usable on M1 Max. |
| [@cheatyyyy](https://x.com/cheatyyyy/status/2124196097795203536) / `@0xSero` | `sybil-solutions/inference-optimisation-skill` (`Turbo-DSpark k=3`, draft shortlist, K-query attention) | Tested on M1 Max: depth sweep `d3=32.89`, `d4=35.95`, `d5=36.71`, `d6=48.74` to `49.43`, `d7=37.10`. `markov_topk=256` dropped acceptance (`142/156` to `128/165`, `34.66 tok/s`). Custom Metal `m1_attn` cut verify time `1.8` to `6.2x`. | Kept depth 6, `m1_attn`, and `fast_decode`. Rejected depth 3 and `markov_topk=256`. |
| [@bebuu_ai](https://x.com/bebuu_ai/status/2124175718804557991) / `loom-fun` | `loom-fun/unsloth-Inkling-2-Mini-loom` (`ROCm/hrx-system`, `42.7` to `44.8 tok/s` on Strix Halo) | `hrx-loom` targets AMD RDNA, CDNA, XDNA, x86-64, Vulkan, and Wasm with zero Metal or ARM64 backend. Ported 4-bit draft head (`43.83 tok/s`, `8.08 s`) and 4-bit attention (`43.72 tok/s`, `8.20 s`) lost speed vs 8-bit (`49.43 tok/s`). | Cannot run natively on M1 Max. Ported tricks regressed. |
