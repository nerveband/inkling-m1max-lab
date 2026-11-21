# Inkling 2 Mini on 24-core M1 Max

## 2026-11-21, preflight

Primary outcome is warm c1 generated tokens/s on actual Pi turns. Secondary outcome is wall time to a correct ordered read, exact edit, test and final answer. Use an isolated Pi 1.6.0 fixture, loopback only, deterministic sampling, 512 tokens per agent response and the full three-pass loop. Direct tests target 2,048 and 8,192 input IDs and 256 outputs. Context target is 32,768. Qualify with a separate held-out task, half/full-context retrieval and a sustained session. Compare candidate legs against both adjacent controls with six samples per leg, then check greedy equivalence and the 3% regression limit.

The host reported macOS 27.0.1 (26A434), 24 GPU cores, 64 GiB RAM, AC power, normal pressure and zero swap at preflight. No thermal warning was recorded. Desktop GPU use fluctuated between 13% and 46% in preflight snapshots. A separate existing inference service was present. The guard checks its GPU-time counters for competing inference. Hardware microbenchmarks include the normal desktop compositor.

Unsloth macOS arm64 b12588-mix-c0c3de8 archive SHA-256 is 229def18e27ed740ec4f3a6508f084b25630e08d8a87ada40b00622fc496ceb8. The binary reports build 12588 commit 883286076. Weight revision ae9327c5fbc7333116ba32eee78bb5d2bbefae4f. Baseline model SHA-256 f90c5a4df44d335ac9f9eea2a15c31d6c5a377b1b3cd9ff7fc2cf123f3cdf7e1. DSpark uses the separate Q8_0 GGUF from the same revision, SHA-256 2ec78510da837e8fc3be54a14757a0c5fbd2d8d9e85a0d92853fafdf674f3773.

Measured device rates with device.m. Two warmups, seven retained observations per metric. See THEORY.md and raw/device.jsonl. Methodology pin is fda968e7fc1996aa23c67139c755fe541c9d4be0.

The first MLX install attempt specified an absent Python 3.13. Retried with 3.12.16. A uv metadata build on the external filesystem failed during egg-info relocation. Switched to pip with local transient build files and external download cache.
