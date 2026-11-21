# Device ceiling estimates

Median timed Metal blit traffic was 309.46 GB/s (read plus write). Median 4096-square MPS FP16 GEMM was 5.462 TFLOP/s. Each median uses seven warmed GPU command-buffer timings. Copy rate is a proxy for streaming reads. MPS dense GEMM is a proxy for quantized MoE compute.

| Context | Weight reads per ID, GB | KV reads, GB | Decode ceiling, tok/s | Prefill at half measured GEMM, tok/s |
|---:|---:|---:|---:|---:|
| 2048 | 2.25 to 4.06 | 0.235 | 72.1 to 124.5 | 265.0 |
| 8192 | 2.25 to 4.06 | 0.436 | 68.8 to 115.1 | 255.2 |
| 32768 | 2.25 to 4.06 | 1.242 | 58.4 to 88.6 | 228.8 |

The tensor offsets include quantization scales, blocks and alignment. Per visit, charge 8/128 of each expert bank, dense attention, routing and one full vocabulary projection per token. Count one input embedding row. The low-traffic bound assumes perfect reuse of the same expert selection and dense weights across repeated visits. The high-traffic case streams weights afresh on all 48 visits. Actual routed expert reuse and physical DRAM traffic are unknown.

For FP16 KV, 40 local visits read at most 1,024 positions and eight global visits read the whole context. Each position has eight 128-wide K and V heads. GQA reuse is assumed. Activation traffic, cache writes, kernel launches and quantization conversion reduce attainable rates. Prefill uses twice the active tensor elements per visit plus causal attention dot products over the average span. The 50% efficiency factor is an assumption. These ranges are planning estimates, not measured model speeds.

The 2K blank-slate visible-code screen reached 18.50 tok/s on copied code and 18.67 tok/s on fresh code. Its measured prefill rate was about 170 tok/s. The gap to the traffic estimate includes Metal command-buffer waits observed in short host samples. The hardware counters available to this experiment do not identify the fraction of the gap due to physical DRAM transfers. Proposal acceptance changes the work per emitted token substantially.

For the five model responses in the 64K profile’s warm coding task, native oMLX verified 154 of 167 proposed IDs (92.2%). It counted 223 emissions including five stop IDs. The API counted 218 generated IDs. Instrumented component totals were 6,145.8 ms for target-backbone work, 116.1 ms for drafting, 2.5 ms for sampling and 12.3 ms for cache operations. The target accounted for 97.9% of summed component time. GPU work can overlap these timers. Prompt evaluation was a separate cost and first-turn time to first token reached roughly 5.4 s. This profile locates the cost in target evaluation. Physical DRAM throughput, per-expert cache misses and individual Metal kernel durations remain unmeasured.

At fixed depth 6, the first scored independent process accepted 176 of 190 proposals in the five-response Pi task. It counted 223 emissions with stops and 218 API-generated IDs. Summed native timers were 5,432.6 ms target, 175.3 ms head, 1.7 ms sampling and 12.8 ms cache work. Target evaluation took 96.6% of that instrumented sum. The deeper draft spent 59 ms more in the head but saved roughly 713 ms on target evaluation compared with the depth-3 diagnostic above. These comparisons are from separate processes.
