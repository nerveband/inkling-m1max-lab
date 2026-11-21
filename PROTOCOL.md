# Running the measurements

Use a 64 GiB Apple Silicon Mac on AC, Python 3.12, Pi 1.6.0 on PATH, and local copies of the pinned runtime and model. The run directory must be new. The guard polls macOS memory pressure, swap, power and per-process GPU use by inference clients. It aborts on new swap use, elevated pressure, AC loss or competing inference. Device-wide GPU residency includes desktop applications. RSS and device residency overlap in unified memory.

The benchmark provider file in configs/models.json points at the loopback measurement proxy. For everyday Pi use, copy configs/pi-direct.json to models.json in a **new** Pi agent directory and set PI_CODING_AGENT_DIR to that directory. The live server uses 127.0.0.1:18089. Benchmark tasks run with synthetic temporary homes, disposable Git worktrees and a local-only provider.

For one Pi control leg, start from this checkout. Set MODEL_DIR and LLAMA_SERVER first.

```sh
python3 run.py plain measurements/plain-a python3 primary.py measurements/plain-a
python3 run.py dspark-6 measurements/draft-b python3 primary.py measurements/draft-b
python3 run.py plain measurements/plain-c python3 primary.py measurements/plain-c
python3 compare.py measurements/plain-a measurements/draft-b measurements/plain-c
```

A leg warms the endpoint with one complete agent task and scores the next six. The task must read calc.py, apply the exact required change through edit, run python3 -B check.py through bash and deliver the exact final reply. Benign inspection of the two synthetic files is allowed. The separate held-out task changes the function, edit, test and expected response. A failed tool invalidates the task.

For native llama.cpp timings, count generated token IDs after the first ID in each response and divide their sum by the summed generation intervals. MLX-LM uses the same clock in the included patch. oMLX reports its generation duration and completion IDs. Its centisecond precision and interval definition introduce a small cross-engine difference. Server-prefill speed and client TTFT include different operations. A structured tool call may arrive as a single SSE event after generation, so Pi tool-call event rate cannot measure native decode. Cache hits are recorded per response.

The direct panels use exact-size token arrays. BENCH_VISIBLE=1 opens the assistant content channel so copy and fresh-code prompts actually produce visible code within the budget. Without it, the pinned template starts by allowing a thinking channel. The direct panels intentionally measure both behaviors.

```sh
BENCH_VISIBLE=1 python3 run.py plain measurements/direct-a python3 direct_pair.py measurements/direct-a
python3 run.py dspark-6 measurements/qual32 python3 qualify_llama.py measurements/qual32 32768
```

Cold readiness starts before process launch and ends at the first successful readiness response. For lazy MLX loads, retain the first request's model-load duration separately. A fresh process may read weights from a warm filesystem cache. Prefill on a new prefix, a warmed model, and a prefix-cache hit are recorded separately. The three-point half/full-context tests require an exact ordered answer. The sustained tests use one accumulating HTTP conversation and a separate Pi session reopened by its session ID.

For the oMLX Pi bracket, each trial uses a fresh model server and a new experiment-only KV directory. A 32-ID unrelated request warms inference before each task. The actual Pi first request must report zero cached IDs. This prevents the next trial from benefiting from a completed copy of the same task while preserving genuine within-task prefix hits. The first whole trial in each leg is excluded. With OMLX_APP pointing at the mounted release bundle:

```sh
python3.12 omlx_leg.py --weights "$MODEL_DIR/omlx-oq4" --cache-root "$MODEL_DIR/transient-kv" --depth 0 measurements/oq-a
python3.12 omlx_leg.py --weights "$MODEL_DIR/omlx-oq4" --cache-root "$MODEL_DIR/transient-kv" --depth 3 measurements/oq-b
python3.12 omlx_leg.py --weights "$MODEL_DIR/omlx-oq4" --cache-root "$MODEL_DIR/transient-kv" --depth 0 measurements/oq-c
python3.12 compare.py measurements/oq-a measurements/oq-b measurements/oq-c
```

For a direct-provider smoke test, run an isolated server first. With `BENCH_BYPASS_PROXY=1`, the fixture invokes the normal Pi provider on port 18089. Add `--stock-provider` to remove the benchmark extension. That check uses the standard Pi request, a disposable worktree and the same independent diff, test and final-answer assertions. Reserve it for the synthetic fixture. Its timings are outside the controlled leaderboard.

For the decoded-ID copy comparison, set `BENCH_VISIBLE=1`, `MODEL_DIR` and `LLAMA_SERVER`. Run these in order from the checkout, using new directories. Each leg uses one 64-ID warmup and six independently tokenized 2,048-ID copy inputs with 256 output IDs requested. The second argument to `direct_compare.py` is the first leg’s directory.

```sh
export BENCH_VISIBLE=1
python3.12 run.py plain measurements/copy-a python3.12 copy_panel.py measurements/copy-a
python3.12 run.py lookup256-dspark6 measurements/copy-b python3.12 copy_panel.py measurements/copy-b
python3.12 run.py plain measurements/copy-c python3.12 copy_panel.py measurements/copy-c
python3.12 direct_compare.py copy-panel.jsonl measurements/copy-a measurements/copy-b measurements/copy-c
```

The GGUF raw panel disables target prompt-KV reuse. In the lookup implementation, the n-gram table belongs to the server process and is shared across sequences. Each new input also trains it from the supplied prompt. Warm repeated-code measurements can use patterns carried from earlier requests. The 108.45 tok/s screen measured the first copy request after an unrelated fresh-code warmup. The separate six-copy bracket measures a warmed repeated-code workload.
