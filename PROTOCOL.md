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
