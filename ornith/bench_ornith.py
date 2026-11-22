"""HTTP benchmark runner for Ornith-1.5-35B-A3B on oMLX / Splash (CPU-only)."""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import http.client
import json
import os
import pathlib
import re
import statistics
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

LAB_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_URL = "http://127.0.0.1:18090/v1"
DEFAULT_MODEL_DIR = (
    pathlib.Path.home()
    / ".omlx"
    / "models"
    / "PocketAiHub"
    / "Ornith-1.5-35B-A3B-Abliterated-MLX-4bit"
)

NEEDLE_ALPHA = "ORBIT-QUARTZ-4819-KESTREL"
NEEDLE_OMEGA = "BASALT-FALCON-9264-VORTEX"


class TokenCounter:
    """Count tokens on CPU via tokenizers/transformers or deterministic fallback."""

    def __init__(self, model_dir: pathlib.Path | None) -> None:
        self.backend = "approx"
        self._impl: Any = None
        if model_dir is not None:
            tok_json = model_dir / "tokenizer.json"
            if tok_json.is_file():
                try:
                    from tokenizers import Tokenizer  # type: ignore

                    self._impl = Tokenizer.from_file(str(tok_json))
                    self.backend = "tokenizers"
                    return
                except Exception:
                    pass
                try:
                    from transformers import AutoTokenizer  # type: ignore

                    self._impl = AutoTokenizer.from_pretrained(
                        str(model_dir), local_files_only=True
                    )
                    self.backend = "transformers"
                    return
                except Exception:
                    pass

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self.backend == "tokenizers":
            return len(self._impl.encode(text, add_special_tokens=False).ids)
        if self.backend == "transformers":
            return len(self._impl.encode(text, add_special_tokens=False))
        # Deterministic CPU-only BPE-like fallback when no tokenizer wheel is installed.
        pieces = re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]|\s+", text)
        est = 0
        for piece in pieces:
            if piece.isspace():
                est += max(1, len(piece) // 4)
            elif len(piece) <= 4:
                est += 1
            else:
                est += (len(piece) + 3) // 4
        return est


def read_memory_telemetry() -> dict[str, Any]:
    """Capture macOS swapusage and memory_pressure without touching GPU."""
    info: dict[str, Any] = {
        "swap_total_mb": None,
        "swap_used_mb": None,
        "swap_free_mb": None,
        "memory_pressure_level": None,
        "memory_pressure_free_pct": None,
    }
    try:
        res = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0:
            txt = res.stdout.strip()
            for key, field in (
                ("total", "swap_total_mb"),
                ("used", "swap_used_mb"),
                ("free", "swap_free_mb"),
            ):
                m = re.search(rf"{key}\s*=\s*([\d.]+)([MG])", txt)
                if m:
                    val = float(m.group(1))
                    if m.group(2) == "G":
                        val *= 1024.0
                    info[field] = round(val, 2)
    except Exception:
        pass
    try:
        res_lvl = subprocess.run(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res_lvl.returncode == 0:
            code = res_lvl.stdout.strip()
            info["memory_pressure_level"] = {
                "1": "normal",
                "2": "warning",
                "4": "critical",
            }.get(code, code)
    except Exception:
        pass
    try:
        res = subprocess.run(
            ["memory_pressure", "-Q"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if res.returncode == 0:
            out = res.stdout
            m_free = re.search(r"System-wide memory free percentage:\s*(\d+)%", out)
            if m_free:
                info["memory_pressure_free_pct"] = int(m_free.group(1))
    except Exception:
        pass
    return info


def check_repetition_collapse(text: str) -> tuple[bool, str]:
    """Detect degenerate looping in generated text."""
    stripped = text.strip()
    if len(stripped) < 40:
        return False, "output_too_short"
    words = stripped.split()
    if len(words) >= 48:
        ngrams = [tuple(words[i : i + 6]) for i in range(len(words) - 5)]
        counts: dict[tuple[str, ...], int] = {}
        for ng in ngrams:
            counts[ng] = counts.get(ng, 0) + 1
        max_rep = max(counts.values())
        if max_rep >= 6 and max_rep / max(1, len(ngrams)) > 0.15:
            return False, f"ngram_repeat(max={max_rep},total={len(ngrams)})"
    lines = [ln.strip() for ln in stripped.splitlines() if len(ln.strip()) >= 20]
    if len(lines) >= 12:
        line_counts: dict[str, int] = {}
        for ln in lines:
            line_counts[ln] = line_counts.get(ln, 0) + 1
        top_line = max(line_counts.values())
        if top_line >= 6 and top_line / len(lines) > 0.35:
            return False, f"line_repeat(max={top_line},lines={len(lines)})"
    return True, "ok"


def _extract_python_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)(?:```|$)", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _check_python_structure(text: str, required_symbols: list[str]) -> tuple[bool, str]:
    ok_rep, rep_msg = check_repetition_collapse(text)
    if not ok_rep:
        return False, rep_msg
    code = _extract_python_code(text)
    if not any(kw in code for kw in ("def ", "class ")):
        return False, "missing_python_defs"
    for sym in required_symbols:
        if sym not in code:
            return False, f"missing_symbol:{sym}"
    # If output was not cut off by max_tokens, validate full AST syntax.
    if "```" in text and text.count("```") >= 2:
        try:
            ast.parse(code)
        except SyntaxError as exc:
            return False, f"syntax_error:{exc.msg}"
    return True, "ok"


def _build_code_edit_source() -> str:
    """Deterministic ~2.5K-token Python codebase for rewrite/copy-spec workload."""
    return '''"""Concurrent task runtime with DAG dependencies and rate controls."""

from __future__ import annotations

import collections
import heapq
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(order=True)
class ScheduledItem:
    ready_at: float
    priority: int
    task_id: str = field(compare=False)
    payload: Any = field(default=None, compare=False)


class RetryPolicy:
    def __init__(self, max_attempts: int = 3, base_delay: float = 0.05, factor: float = 2.0) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.factor = factor

    def next_delay(self, attempt: int) -> float:
        if attempt < 0:
            raise ValueError("attempt must be >= 0")
        return self.base_delay * (self.factor ** attempt)

    def should_retry(self, attempt: int, exc: BaseException) -> bool:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            return False
        return attempt + 1 < self.max_attempts


class RateLimiter:
    def __init__(self, capacity: int, refill_per_sec: float) -> None:
        if capacity <= 0 or refill_per_sec <= 0.0:
            raise ValueError("capacity and refill_per_sec must be positive")
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.refill_per_sec = float(refill_per_sec)
        self.last_refill = time.monotonic()

    def _refill(self, now: float) -> None:
        elapsed = max(0.0, now - self.last_refill)
        if elapsed > 0.0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_sec)
            self.last_refill = now

    def try_acquire(self, cost: float = 1.0) -> bool:
        if cost <= 0.0:
            raise ValueError("cost must be positive")
        self._refill(time.monotonic())
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False


class DependencyGraph:
    def __init__(self) -> None:
        self._deps: dict[str, set[str]] = {}
        self._rev: dict[str, set[str]] = collections.defaultdict(set)

    def add_node(self, task_id: str, deps: list[str] | None = None) -> None:
        dep_set = set(deps or ())
        if task_id in dep_set:
            raise ValueError(f"self dependency on {task_id}")
        self._deps[task_id] = dep_set
        for parent in dep_set:
            self._deps.setdefault(parent, set())
            self._rev[parent].add(task_id)

    def ready_nodes(self, completed: set[str]) -> list[str]:
        ready = [
            tid
            for tid, reqs in self._deps.items()
            if tid not in completed and reqs.issubset(completed)
        ]
        ready.sort()
        return ready

    def mark_done(self, task_id: str) -> list[str]:
        unlocked: list[str] = []
        for child in sorted(self._rev.get(task_id, ())):
            self._deps[child].discard(task_id)
            if not self._deps[child]:
                unlocked.append(child)
        return unlocked

    def has_cycle(self) -> bool:
        indeg = {k: len(v) for k, v in self._deps.items()}
        q = collections.deque(k for k, d in indeg.items() if d == 0)
        visited = 0
        while q:
            cur = q.popleft()
            visited += 1
            for nxt in self._rev.get(cur, ()):
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    q.append(nxt)
        return visited != len(indeg)


class MetricsCollector:
    def __init__(self) -> None:
        self._counters: dict[str, int] = collections.defaultdict(int)
        self._durations_ms: dict[str, list[float]] = collections.defaultdict(list)

    def inc(self, name: str, delta: int = 1) -> None:
        self._counters[name] += delta

    def observe_ms(self, name: str, elapsed_ms: float) -> None:
        self._durations_ms[name].append(float(elapsed_ms))

    def snapshot(self) -> dict[str, Any]:
        summary: dict[str, Any] = {"counters": dict(self._counters), "latencies": {}}
        for name, samples in self._durations_ms.items():
            ordered = sorted(samples)
            if not ordered:
                continue
            mid = ordered[len(ordered) // 2]
            summary["latencies"][name] = {
                "count": len(ordered),
                "min_ms": round(ordered[0], 3),
                "median_ms": round(mid, 3),
                "max_ms": round(ordered[-1], 3),
            }
        return summary


class TaskScheduler:
    def __init__(
        self,
        retry_policy: RetryPolicy | None = None,
        limiter: RateLimiter | None = None,
    ) -> None:
        self.retry_policy = retry_policy or RetryPolicy()
        self.limiter = limiter
        self.graph = DependencyGraph()
        self.metrics = MetricsCollector()
        self._heap: list[ScheduledItem] = []
        self._handlers: dict[str, Callable[[Any], Any]] = {}
        self._attempts: dict[str, int] = {}
        self._completed: set[str] = set()
        self._results: dict[str, Any] = {}
        self._errors: dict[str, str] = {}

    def submit(
        self,
        task_id: str,
        handler: Callable[[Any], Any],
        payload: Any = None,
        *,
        deps: list[str] | None = None,
        priority: int = 10,
        delay_s: float = 0.0,
    ) -> None:
        self.graph.add_node(task_id, deps=deps)
        if self.graph.has_cycle():
            raise ValueError(f"cycle introduced by {task_id}")
        self._handlers[task_id] = handler
        self._attempts[task_id] = 0
        if not deps or set(deps).issubset(self._completed):
            heapq.heappush(
                self._heap,
                ScheduledItem(time.monotonic() + max(0.0, delay_s), priority, task_id, payload),
            )
        self.metrics.inc("submitted")

    def step(self) -> str | None:
        if not self._heap:
            return None
        now = time.monotonic()
        if self._heap[0].ready_at > now:
            return None
        if self.limiter is not None and not self.limiter.try_acquire():
            self.metrics.inc("rate_limited")
            return None
        item = heapq.heappop(self._heap)
        tid = item.task_id
        if tid in self._completed:
            return None
        t0 = time.perf_counter()
        try:
            res = self._handlers[tid](item.payload)
        except Exception as exc:
            attempt = self._attempts.get(tid, 0)
            if self.retry_policy.should_retry(attempt, exc):
                self._attempts[tid] = attempt + 1
                backoff = self.retry_policy.next_delay(attempt)
                heapq.heappush(
                    self._heap,
                    ScheduledItem(time.monotonic() + backoff, item.priority, tid, item.payload),
                )
                self.metrics.inc("retried")
            else:
                self._errors[tid] = repr(exc)
                self.metrics.inc("failed")
            return tid
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.metrics.observe_ms("task_exec", elapsed_ms)
        self.metrics.inc("succeeded")
        self._completed.add(tid)
        self._results[tid] = res
        for child in self.graph.mark_done(tid):
            if child in self._handlers and child not in self._completed:
                heapq.heappush(self._heap, ScheduledItem(time.monotonic(), item.priority, child, None))
        return tid

    def pending_count(self) -> int:
        return sum(1 for item in self._heap if item.task_id not in self._completed)

    def result_of(self, task_id: str) -> Any:
        if task_id in self._errors:
            raise RuntimeError(self._errors[task_id])
        return self._results.get(task_id)

    def error_of(self, task_id: str) -> str | None:
        return self._errors.get(task_id)

    def drain(self, max_steps: int = 1000) -> dict[str, Any]:
        steps = 0
        while self._heap and steps < max_steps:
            progressed = self.step()
            steps += 1
            if progressed is None:
                break
        return {
            "steps": steps,
            "completed": sorted(self._completed),
            "failed": dict(self._errors),
            "metrics": self.metrics.snapshot(),
        }


class CircuitBreaker:
    """Protect downstream handlers after consecutive failures."""

    def __init__(self, failure_threshold: int = 5, reset_after_s: float = 30.0) -> None:
        if failure_threshold < 1 or reset_after_s <= 0.0:
            raise ValueError("invalid circuit breaker bounds")
        self.failure_threshold = failure_threshold
        self.reset_after_s = reset_after_s
        self._failures = 0
        self._opened_at: float | None = None

    def allow(self) -> bool:
        if self._opened_at is None:
            return True
        if time.monotonic() - self._opened_at >= self.reset_after_s:
            self._opened_at = None
            self._failures = 0
            return True
        return False

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold and self._opened_at is None:
            self._opened_at = time.monotonic()


def build_default_scheduler(rate_per_sec: float = 50.0) -> TaskScheduler:
    policy = RetryPolicy(max_attempts=4, base_delay=0.02, factor=2.0)
    limiter = RateLimiter(capacity=max(1, int(rate_per_sec)), refill_per_sec=rate_per_sec)
    return TaskScheduler(retry_policy=policy, limiter=limiter)
'''


def _build_warm_pi_prefix() -> str:
    """Build deterministic ~8K-token shared engineering context for warm_pi."""
    modules = []
    domains = [
        ("kv_ring", "RingKVStore", "allocate_slot", "evict_oldest", "lookup_prefix"),
        ("spec_tree", "DraftVerifier", "build_mask", "score_branch", "commit_prefix"),
        ("moe_router", "ExpertRouter", "partition_tokens", "dispatch_batch", "combine_outputs"),
        ("gdn_cache", "DeltaStatePool", "snapshot_state", "restore_state", "advance_conv"),
        ("http_stream", "SSEWriter", "encode_delta", "flush_chunk", "finalize_usage"),
        ("ast_guard", "SyntaxChecker", "parse_module", "collect_imports", "verify_calls"),
        ("swap_watch", "MemoryMonitor", "sample_pressure", "read_swap_mb", "enforce_budget"),
        ("bench_stats", "TrialAggregator", "compute_median", "bootstrap_ci", "append_row"),
    ]
    for idx, (mod_name, cls_name, m1, m2, m3) in enumerate(domains):
        lines = [
            f"# === Module {idx + 1}: {mod_name}.py ===",
            f"class {cls_name}:",
            f'    """Deterministic {mod_name} subsystem component #{idx + 1}."""',
            f"    VERSION = {idx + 10}",
            f"    DEFAULT_CAPACITY = {1024 * (idx + 1)}",
            "",
            "    def __init__(self, capacity: int | None = None, strict: bool = True) -> None:",
            "        self.capacity = int(capacity or self.DEFAULT_CAPACITY)",
            "        self.strict = bool(strict)",
            "        self._entries: dict[str, dict[str, int]] = {}",
            "        self._order: list[str] = []",
            "        self._checksum_seed = 0x9E3779B1 ^ (self.VERSION * 131)",
            "",
        ]
        for mid, meth in enumerate((m1, m2, m3)):
            lines.extend([
                f"    def {meth}(self, key: str, value: int, flags: int = 0) -> dict[str, int]:",
                f'        """Execute {meth} operation on {mod_name} slot."""',
                "        if self.strict and not key:",
                f'            raise ValueError("{cls_name}.{meth}: empty key")',
                f"        tag = (hash(key) ^ self._checksum_seed ^ ({mid + 1} * 4099)) & 0xFFFFFFFF",
                "        record = {",
                '            "value": int(value),',
                '            "flags": int(flags),',
                '            "tag": tag,',
                f'            "op_code": {idx * 10 + mid},',
                "        }",
                "        if key not in self._entries:",
                "            if len(self._order) >= self.capacity:",
                "                victim = self._order.pop(0)",
                "                self._entries.pop(victim, None)",
                "            self._order.append(key)",
                "        self._entries[key] = record",
                "        return dict(record)",
                "",
            ])
        for helper_i in range(4):
            lines.extend([
                f"    def validate_rule_{helper_i}(self, payload: list[int], limit: int = {64 * (helper_i + 1)}) -> int:",
                f'        """Deterministic checksum rule {helper_i} for {cls_name}."""',
                "        acc = self._checksum_seed",
                "        for pos, item in enumerate(payload):",
                f"            acc = ((acc * 33) + (item ^ (pos + {helper_i + 1}))) & 0xFFFFFFFF",
                "        if self.strict and len(payload) > limit:",
                f'            raise OverflowError("{cls_name}: payload exceeds limit")',
                "        return acc",
                "",
            ])
        modules.append("\n".join(lines))
    return "\n\n".join(modules)


def _build_needle_32k_prompt(counter: TokenCounter, target_tokens: int = 31500) -> str:
    """Build non-trivial ~31.5K-token context with two unique needles at ~20% and ~75%."""
    subsystems = [
        "gdn-recurrent-state",
        "full-attn-kv-ring",
        "moe-expert-router",
        "dflash-context-proj",
        "metal-command-queue",
        "unified-memory-pool",
        "paged-prefix-index",
        "tokenizer-bpe-stream",
    ]
    actions = [
        "verified checksum alignment across contiguous cache pages",
        "drained pending Metal command buffers prior to watermark check",
        "rebalanced active expert histograms across 256 routing buckets",
        "flushed transient conv1d window states for idle sequence slots",
        "recomputed rotary offset deltas after ring wraparound boundary",
        "audited group_size=64 affine scale tables for finiteness",
        "recorded non-zero tail latency quantiles under loopback load",
        "inspected resident KV extent occupancy without disk spill",
    ]

    header = (
        "Below is a chronological engineering telemetry ledger. Read every entry carefully.\n"
        "At the end, extract the two exact secret passkeys embedded in the ledger.\n\n"
    )
    footer = (
        "\n\nReturn ONLY a compact JSON object with keys \"alpha\" and \"omega\" containing "
        "the exact passkey values for ALPHA_NEEDLE_7492 and OMEGA_NEEDLE_3158. No markdown."
    )

    # Estimate tokens per block dynamically on the first 20 blocks.
    sample_blocks = []
    for i in range(20):
        sub = subsystems[i % len(subsystems)]
        act = actions[(i * 3) % len(actions)]
        sig = (i * 2654435761) & 0xFFFFFFFF
        sample_blocks.append(
            f"[ENTRY-{i:05d}] ts=2026-11-21T{(i % 24):02d}:{(i % 60):02d}:00Z "
            f"sub={sub} seq={1000 + i * 7} sig={sig:08x} note={act}."
        )
    sample_text = "\n".join(sample_blocks)
    sample_toks = max(1, counter.count(sample_text))
    toks_per_block = sample_toks / len(sample_blocks)
    overhead = counter.count(header) + counter.count(footer) + 80
    desired_blocks = max(120, int((target_tokens - overhead) / toks_per_block))

    alpha_idx = int(desired_blocks * 0.20)
    omega_idx = int(desired_blocks * 0.75)

    blocks: list[str] = []
    for i in range(desired_blocks):
        if i == alpha_idx:
            blocks.append(
                f"[ENTRY-{i:05d}] SECURITY AUDIT OVERRIDE: "
                f"ALPHA_NEEDLE_7492 = \"{NEEDLE_ALPHA}\" (verified exact passkey #1)."
            )
            continue
        if i == omega_idx:
            blocks.append(
                f"[ENTRY-{i:05d}] SECURITY AUDIT OVERRIDE: "
                f"OMEGA_NEEDLE_3158 = \"{NEEDLE_OMEGA}\" (verified exact passkey #2)."
            )
            continue
        sub = subsystems[i % len(subsystems)]
        act = actions[(i * 5 + (i // 7)) % len(actions)]
        sig = ((i + 17) * 2654435761) & 0xFFFFFFFF
        blocks.append(
            f"[ENTRY-{i:05d}] ts=2026-11-21T{(i % 24):02d}:{(i % 60):02d}:{(i * 13 % 60):02d}Z "
            f"sub={sub} seq={1000 + i * 7} sig={sig:08x} note={act}."
        )

    prompt = header + "\n".join(blocks) + footer
    # Fine-tune length if exact tokenizer count is slightly under/over 31.5K.
    cur = counter.count(prompt)
    while cur < target_tokens - 500:
        i = len(blocks)
        sub = subsystems[i % len(subsystems)]
        act = actions[(i * 5 + (i // 7)) % len(actions)]
        sig = ((i + 17) * 2654435761) & 0xFFFFFFFF
        blocks.append(
            f"[ENTRY-{i:05d}] ts=2026-11-21T{(i % 24):02d}:{(i % 60):02d}:00Z "
            f"sub={sub} seq={1000 + i * 7} sig={sig:08x} note={act}."
        )
        if len(blocks) % 10 == 0:
            prompt = header + "\n".join(blocks) + footer
            cur = counter.count(prompt)
    return header + "\n".join(blocks) + footer


def parse_server_log_slice(text: str) -> dict[str, Any]:
    """Parse oMLX MTP[...] lines and Splash Done lines from newly logged lines."""
    omlx_rows: list[dict[str, Any]] = []
    splash_rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if "MTP[" in line and "timing[" in line:
            m = re.search(
                r"tokens=(\d+)\s+cycles=(\d+)\s+tok/cycle=([\d.]+)\s+accept=(\d+)/(\d+)\s+\(([\d.]+%|n/a)\)",
                line,
            )
            tm = re.search(r"timing\[([^\]]+)\]", line)
            if m and tm:
                ms = {
                    k: float(v)
                    for k, v in re.findall(r"([a-z_]+)=([\d.]+)ms", tm.group(1))
                }
                tokens = int(m.group(1))
                cycles = int(m.group(2))
                tpc = float(m.group(3))
                accepted = int(m.group(4))
                proposed = int(m.group(5))
                total_ms = sum(ms.values())
                native_tps = (tokens * 1000.0 / total_ms) if total_ms > 0 else None
                omlx_rows.append(
                    {
                        "engine": "omlx",
                        "tokens": tokens,
                        "cycles": cycles,
                        "tok_per_cycle": tpc,
                        "accepted": accepted,
                        "proposed": proposed,
                        "accept_rate": (accepted / proposed) if proposed > 0 else None,
                        "ms": ms,
                        "native_decode_tps": round(native_tps, 3)
                        if native_tps is not None
                        else None,
                    }
                )
        elif "Done" in line and "request " in line and "tok/s" in line:
            m_in = re.search(r"input\s+([\d,]+)", line)
            m_ca = re.search(r"cached\s+([\d,]+)", line)
            m_out = re.search(r"output\s+([\d,]+)", line)
            m_ttft = re.search(r"TTFT\s+([\d.]+)s", line)
            m_tps = re.search(r"([\d.]+)\s+tok/s", line)
            m_tpc = re.search(r"([\d.]+)\s+tok/cycle", line)
            if m_out and m_tps:
                splash_rows.append(
                    {
                        "engine": "splash",
                        "input_tokens": int(m_in.group(1).replace(",", "")) if m_in else None,
                        "cached_tokens": int(m_ca.group(1).replace(",", "")) if m_ca else None,
                        "output_tokens": int(m_out.group(1).replace(",", "")),
                        "ttft_ms": float(m_ttft.group(1)) * 1000.0 if m_ttft else None,
                        "decode_tps": float(m_tps.group(1)),
                        "tok_per_cycle": float(m_tpc.group(1)) if m_tpc else None,
                    }
                )
    return {"omlx_mtp": omlx_rows, "splash_done": splash_rows}


def _discover_model(base_url: str, api_key: str | None, timeout: float) -> str:
    url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(url, method="GET")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    models = data.get("data") or []
    if not models:
        raise RuntimeError(f"No models returned from {url}")
    return str(models[0]["id"])


def _chat_stream(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float,
    thinking: str,
    api_key: str | None,
    timeout: float,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if thinking == "off":
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    elif thinking == "on":
        payload["chat_template_kwargs"] = {"enable_thinking": True}

    body = json.dumps(payload).encode("utf-8")
    parsed = urllib.parse.urlparse(url)
    conn_cls = (
        http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    )
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    t_start = time.perf_counter()
    first_tok_t: float | None = None
    last_tok_t: float | None = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    chunk_count = 0
    usage: dict[str, Any] | None = None
    timings: dict[str, Any] | None = None
    finish_reason: str | None = None

    conn = conn_cls(parsed.hostname or "127.0.0.1", parsed.port, timeout=timeout)
    try:
        path = parsed.path or "/v1/chat/completions"
        if parsed.query:
            path += "?" + parsed.query
        conn.request("POST", path, body=body, headers=headers)
        resp = conn.getresponse()
        if resp.status != 200:
            err_body = resp.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {resp.status}: {err_body[:400]}")

        while True:
            raw_line = resp.readline()
            if not raw_line:
                break
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                evt = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            now = time.perf_counter()
            if isinstance(evt.get("usage"), dict):
                usage = evt["usage"]
            if isinstance(evt.get("timings"), dict):
                timings = evt["timings"]
            choices = evt.get("choices") or []
            if choices:
                c0 = choices[0]
                if c0.get("finish_reason"):
                    finish_reason = str(c0["finish_reason"])
                delta = c0.get("delta") or {}
                txt = delta.get("content") or ""
                rtxt = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if txt or rtxt:
                    if first_tok_t is None:
                        first_tok_t = now
                    last_tok_t = now
                    chunk_count += 1
                    if txt:
                        content_parts.append(txt)
                    if rtxt:
                        reasoning_parts.append(rtxt)
    finally:
        conn.close()

    t_done = time.perf_counter()
    wall_ms = (t_done - t_start) * 1000.0
    ttft_ms = ((first_tok_t - t_start) * 1000.0) if first_tok_t is not None else wall_ms
    decode_ms = (
        ((last_tok_t - first_tok_t) * 1000.0)
        if (first_tok_t is not None and last_tok_t is not None and last_tok_t > first_tok_t)
        else 0.0
    )
    return {
        "content": "".join(content_parts),
        "reasoning": "".join(reasoning_parts),
        "finish_reason": finish_reason,
        "chunk_count": chunk_count,
        "client_wall_ms": round(wall_ms, 2),
        "client_ttft_ms": round(ttft_ms, 2),
        "client_decode_ms": round(decode_ms, 2),
        "usage": usage,
        "timings": timings,
    }


def _run_turn(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float,
    thinking: str,
    api_key: str | None,
    timeout: float,
    counter: TokenCounter,
    server_log: pathlib.Path | None,
) -> dict[str, Any]:
    log_offset = 0
    if server_log is not None and server_log.is_file():
        log_offset = server_log.stat().st_size

    resp = _chat_stream(
        base_url,
        model,
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        thinking=thinking,
        api_key=api_key,
        timeout=timeout,
    )

    log_metrics: dict[str, Any] = {}
    if server_log is not None and server_log.is_file():
        end_size = server_log.stat().st_size
        if end_size > log_offset:
            with server_log.open("rb") as f:
                f.seek(log_offset)
                delta_text = f.read(end_size - log_offset).decode("utf-8", errors="replace")
            log_metrics = parse_server_log_slice(delta_text)

    usage = resp.get("usage") or {}
    timings = resp.get("timings") or {}
    combined_out = (resp["reasoning"] + "\n" + resp["content"]).strip()
    local_out_toks = counter.count(combined_out)
    local_in_toks = sum(counter.count(m["content"]) for m in messages)

    prompt_tokens = usage.get("prompt_tokens") or timings.get("prompt_n") or local_in_toks
    completion_tokens = (
        usage.get("completion_tokens") or timings.get("predicted_n") or local_out_toks
    )
    cached_tokens = (
        (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        if isinstance(usage.get("prompt_tokens_details"), dict)
        else None
    )
    if cached_tokens is None:
        cached_tokens = timings.get("cache_n")

    decode_ms = resp["client_decode_ms"]
    wall_ms = resp["client_wall_ms"]
    client_decode_tps = (
        round((max(1, completion_tokens - 1)) / (decode_ms / 1000.0), 3)
        if decode_ms > 0 and completion_tokens > 1
        else None
    )
    client_wall_tps = (
        round(completion_tokens / (wall_ms / 1000.0), 3) if wall_ms > 0 else None
    )

    api_decode_tps = timings.get("predicted_per_second")
    api_prefill_tps = timings.get("prompt_per_second")
    draft_n = timings.get("draft_n")
    draft_acc = timings.get("draft_n_accepted")

    server_tpc = None
    server_accept_rate = None
    server_native_tps = None
    if log_metrics.get("omlx_mtp"):
        last_mtp = log_metrics["omlx_mtp"][-1]
        server_tpc = last_mtp.get("tok_per_cycle")
        server_accept_rate = last_mtp.get("accept_rate")
        server_native_tps = last_mtp.get("native_decode_tps")
    elif log_metrics.get("splash_done"):
        last_sp = log_metrics["splash_done"][-1]
        server_tpc = last_sp.get("tok_per_cycle")
        server_native_tps = last_sp.get("decode_tps")
        if cached_tokens is None:
            cached_tokens = last_sp.get("cached_tokens")
    if server_accept_rate is None and draft_n:
        server_accept_rate = round(float(draft_acc or 0) / float(draft_n), 4)

    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "local_tok_estimate": int(local_out_toks),
        "local_tok_backend": counter.backend,
        "cached_tokens": int(cached_tokens) if cached_tokens is not None else None,
        "finish_reason": resp["finish_reason"],
        "chunk_count": resp["chunk_count"],
        "client_wall_ms": wall_ms,
        "client_ttft_ms": resp["client_ttft_ms"],
        "client_decode_ms": decode_ms,
        "client_decode_tps": client_decode_tps,
        "client_wall_tps": client_wall_tps,
        "api_decode_tps": round(float(api_decode_tps), 3) if api_decode_tps else None,
        "api_prefill_tps": round(float(api_prefill_tps), 3) if api_prefill_tps else None,
        "server_native_tps": server_native_tps,
        "server_tok_per_cycle": server_tpc,
        "server_accept_rate": server_accept_rate,
        "log_metrics": log_metrics,
        "content": resp["content"],
        "reasoning": resp["reasoning"],
    }


def run_workload(
    name: str,
    args: argparse.Namespace,
    model: str,
    counter: TokenCounter,
) -> dict[str, Any]:
    """Execute one deterministic workload and verify output quality."""
    server_log = pathlib.Path(args.server_log) if args.server_log else None
    mem_before = read_memory_telemetry()
    turns: list[dict[str, Any]] = []
    passed = True
    check_reason = "ok"

    if name == "code_gen":
        messages = [
            {
                "role": "system",
                "content": "You are a senior Python systems engineer. Output clean code.",
            },
            {
                "role": "user",
                "content": (
                    "Write a self-contained Python 3.12 module implementing an augmented AVL "
                    "`IntervalTree` supporting `insert(start, end, label)`, "
                    "`query_overlap(start, end) -> list[tuple[int, int, str]]`, "
                    "`merge_all() -> list[tuple[int, int]]`, and `delete(start, end) -> bool`, "
                    "plus a short `run_tests()` function with deterministic assertions. "
                    "Use only the Python standard library."
                ),
            },
        ]
        res = _run_turn(
            args.url,
            model,
            messages,
            max_tokens=512,
            temperature=args.temperature,
            thinking=args.thinking,
            api_key=args.api_key,
            timeout=args.timeout,
            counter=counter,
            server_log=server_log,
        )
        passed, check_reason = _check_python_structure(res["content"], [])
        if passed and "IntervalTree" not in res["content"]:
            passed, check_reason = False, "missing_symbol:IntervalTree"
        turns.append(res)

    elif name == "code_edit":
        src = _build_code_edit_source()
        messages = [
            {
                "role": "system",
                "content": "Refactor the provided Python module cleanly, preserving all existing signatures.",
            },
            {
                "role": "user",
                "content": (
                    "Rewrite the entire Python module below to add thread-safe `threading.RLock` "
                    "synchronization to `RateLimiter`, `DependencyGraph`, `MetricsCollector`, and "
                    "`TaskScheduler`, add `cancel(task_id: str) -> bool`, and enforce an optional "
                    "`timeout_s: float | None = None` on `TaskScheduler.submit`. Return the full "
                    "updated module in one ```python code block.\n\n```python\n"
                    + src
                    + "\n```"
                ),
            },
        ]
        res = _run_turn(
            args.url,
            model,
            messages,
            max_tokens=512,
            temperature=args.temperature,
            thinking=args.thinking,
            api_key=args.api_key,
            timeout=args.timeout,
            counter=counter,
            server_log=server_log,
        )
        passed, check_reason = _check_python_structure(
            res["content"], ["RetryPolicy", "RateLimiter", "RLock"]
        )
        turns.append(res)

    elif name == "open_prose":
        messages = [
            {
                "role": "system",
                "content": "You are a systems architecture writer. Be precise and concise.",
            },
            {
                "role": "user",
                "content": (
                    "Write an analytical technical essay comparing hybrid Gated DeltaNet "
                    "(linear-attention) layers plus grouped-query softmax attention against pure "
                    "softmax attention on unified-memory Apple Silicon GPUs. Cover KV cache growth, "
                    "fixed recurrent state bandwidth, speculative draft verification batch sizes, "
                    "and fp16 versus bf16 ALU bottlenecks."
                ),
            },
        ]
        res = _run_turn(
            args.url,
            model,
            messages,
            max_tokens=350,
            temperature=args.temperature,
            thinking=args.thinking,
            api_key=args.api_key,
            timeout=args.timeout,
            counter=counter,
            server_log=server_log,
        )
        passed, check_reason = check_repetition_collapse(res["content"])
        if passed and res["completion_tokens"] < 100:
            passed, check_reason = False, f"prose_too_short({res['completion_tokens']})"
        turns.append(res)

    elif name == "warm_pi":
        prefix = _build_warm_pi_prefix()
        sys_msg = {
            "role": "system",
            "content": (
                "You are an embedded repository coding assistant. Study the codebase below and "
                "answer each follow-up edit concisely in Python.\n\n" + prefix
            ),
        }
        prompts = [
            "Turn 1 (cold): Add a method `evict_expired(self, min_value: int) -> list[str]` to `RingKVStore` that removes entries with `value < min_value` and returns removed keys in insertion order.",
            "Turn 2 (warm): Now write a standalone helper `merge_scheduler_snapshots(snapshots: list[dict]) -> dict` compatible with `TrialAggregator` outputs.",
            "Turn 3 (warm): Finally write 4 concise pytest unit tests covering `RingKVStore.allocate_slot` eviction and `ExpertRouter.partition_tokens`.",
        ]
        history: list[dict[str, str]] = [sys_msg]
        reasons: list[str] = []
        for idx, ptxt in enumerate(prompts):
            history.append({"role": "user", "content": ptxt})
            res = _run_turn(
                args.url,
                model,
                history,
                max_tokens=256,
                temperature=args.temperature,
                thinking=args.thinking,
                api_key=args.api_key,
                timeout=args.timeout,
                counter=counter,
                server_log=server_log,
            )
            ok_t, msg_t = check_repetition_collapse(res["content"])
            if not ok_t:
                passed = False
                reasons.append(f"turn{idx + 1}:{msg_t}")
            turns.append(res)
            history.append({"role": "assistant", "content": res["content"]})
        if reasons:
            check_reason = ";".join(reasons)

    elif name == "needle_32k":
        prompt_32k = _build_needle_32k_prompt(counter, target_tokens=31500)
        messages = [
            {
                "role": "system",
                "content": "You are a precise needle-retrieval verifier. Answer exactly.",
            },
            {"role": "user", "content": prompt_32k},
        ]
        res = _run_turn(
            args.url,
            model,
            messages,
            max_tokens=256,
            temperature=args.temperature,
            thinking=args.thinking,
            api_key=args.api_key,
            timeout=args.timeout,
            counter=counter,
            server_log=server_log,
        )
        ok_rep, rep_msg = check_repetition_collapse(res["content"])
        has_alpha = NEEDLE_ALPHA in res["content"]
        has_omega = NEEDLE_OMEGA in res["content"]
        passed = bool(ok_rep and has_alpha and has_omega)
        if not ok_rep:
            check_reason = rep_msg
        elif not (has_alpha and has_omega):
            check_reason = f"needle_miss(alpha={has_alpha},omega={has_omega})"
        turns.append(res)

    else:
        raise ValueError(f"Unknown workload: {name!r}")

    mem_after = read_memory_telemetry()

    # Aggregate turn metrics for summary row.
    primary = turns[-1] if name != "warm_pi" else turns[0]
    warm_turns = turns[1:] if name == "warm_pi" else []
    warm_ttft_med = (
        round(statistics.median(t["client_ttft_ms"] for t in warm_turns), 2)
        if warm_turns
        else None
    )
    warm_tps_vals = [
        t["client_decode_tps"] for t in warm_turns if t["client_decode_tps"] is not None
    ]
    warm_tps_med = round(statistics.median(warm_tps_vals), 3) if warm_tps_vals else None

    return {
        "workload": name,
        "passed": passed,
        "check_reason": check_reason,
        "turns": turns,
        "mem_before": mem_before,
        "mem_after": mem_after,
        "summary": {
            "prompt_tokens": primary["prompt_tokens"],
            "completion_tokens": sum(t["completion_tokens"] for t in turns),
            "cached_tokens": (
                warm_turns[-1]["cached_tokens"]
                if warm_turns
                else primary["cached_tokens"]
            ),
            "cold_ttft_ms": primary["client_ttft_ms"],
            "warm_ttft_ms": warm_ttft_med,
            "cold_decode_tps": primary["client_decode_tps"],
            "warm_decode_tps": warm_tps_med,
            "client_wall_ms": round(sum(t["client_wall_ms"] for t in turns), 2),
            "client_decode_tps": warm_tps_med or primary["client_decode_tps"],
            "api_decode_tps": primary["api_decode_tps"],
            "server_native_tps": primary["server_native_tps"],
            "server_tok_per_cycle": primary["server_tok_per_cycle"],
            "server_accept_rate": primary["server_accept_rate"],
        },
    }


def append_ledger(ledger_path: pathlib.Path, row: dict[str, Any]) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Ornith-1.5-35B-A3B HTTP benchmarks and record ledger rows."
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="OpenAI-compatible /v1 base URL.")
    parser.add_argument("--model", default=None, help="Served model ID (auto-discovered if omitted).")
    parser.add_argument(
        "--model-dir",
        type=pathlib.Path,
        default=DEFAULT_MODEL_DIR,
        help="Local model directory containing tokenizer.json.",
    )
    parser.add_argument("--tag", default=None, help="Run tag for work/ornith-lab/raw/<tag>/.")
    parser.add_argument("--leg", default="candidate", help="Leg label (e.g., A, B, C, stock, overlay).")
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=LAB_DIR / "raw",
        help="Root directory for raw JSON results.",
    )
    parser.add_argument(
        "--ledger",
        type=pathlib.Path,
        default=LAB_DIR / "ledger.jsonl",
        help="Path to append JSONL summary rows.",
    )
    parser.add_argument("--server-log", default=None, help="Optional path to server stdout/stderr log.")
    parser.add_argument(
        "--temp",
        "--temperature",
        dest="temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0).",
    )
    parser.add_argument(
        "--thinking",
        choices=("off", "on", "auto"),
        default="off",
        help="Reasoning mode via chat_template_kwargs.enable_thinking (default: off).",
    )
    parser.add_argument(
        "--workloads",
        default="all",
        help="Comma-separated workloads or 'all' (code_gen,code_edit,open_prose,warm_pi,needle_32k).",
    )
    parser.add_argument("--mode", choices=("single", "aba"), default="single")
    parser.add_argument("--repeats", type=int, default=1, help="Repetitions for single mode.")
    parser.add_argument("--trials", type=int, default=6, help="Trials for aba mode (default: 6).")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY") or os.environ.get("SPLASH_API_KEY"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    all_workloads = ("code_gen", "code_edit", "open_prose", "warm_pi", "needle_32k")
    if args.workloads == "all":
        workloads = list(all_workloads)
    else:
        workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]
        for w in workloads:
            if w not in all_workloads:
                raise SystemExit(f"Unknown workload {w!r}; expected {all_workloads}")

    tag = args.tag or dt.datetime.now(dt.timezone.utc).strftime("ornith-%Y%m%d-%H%M%S")
    raw_dir = args.out_dir / tag
    raw_dir.mkdir(parents=True, exist_ok=True)

    counter = TokenCounter(args.model_dir)
    model = args.model or _discover_model(args.url, args.api_key, args.timeout)
    n_trials = args.trials if args.mode == "aba" else args.repeats

    print(
        f"Benchmarking model={model} url={args.url} mode={args.mode} "
        f"trials={n_trials} leg={args.leg} tokenizer={counter.backend}",
        flush=True,
    )

    if not args.skip_warmup:
        warm = _run_turn(
            args.url,
            model,
            [{"role": "user", "content": "Reply with exactly: READY"}],
            max_tokens=16,
            temperature=0.0,
            thinking=args.thinking,
            api_key=args.api_key,
            timeout=args.timeout,
            counter=counter,
            server_log=pathlib.Path(args.server_log) if args.server_log else None,
        )
        print(
            f"Warmup: wall={warm['client_wall_ms']}ms toks={warm['completion_tokens']}",
            flush=True,
        )

    all_records: list[dict[str, Any]] = []
    for trial in range(1, n_trials + 1):
        for wl in workloads:
            res = run_workload(wl, args, model, counter)
            trial_id = f"t{trial:02d}-{args.leg}-{wl}"
            out_file = raw_dir / f"{trial_id}.json"
            out_file.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")

            s = res["summary"]
            row = {
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                "tag": tag,
                "mode": args.mode,
                "leg": args.leg,
                "trial": trial,
                "url": args.url,
                "model": model,
                "workload": wl,
                "passed": res["passed"],
                "check_reason": res["check_reason"],
                "prompt_tokens": s["prompt_tokens"],
                "completion_tokens": s["completion_tokens"],
                "cached_tokens": s["cached_tokens"],
                "cold_ttft_ms": s["cold_ttft_ms"],
                "warm_ttft_ms": s["warm_ttft_ms"],
                "cold_decode_tps": s["cold_decode_tps"],
                "warm_decode_tps": s["warm_decode_tps"],
                "client_wall_ms": s["client_wall_ms"],
                "client_decode_tps": s["client_decode_tps"],
                "api_decode_tps": s["api_decode_tps"],
                "server_native_tps": s["server_native_tps"],
                "server_tok_per_cycle": s["server_tok_per_cycle"],
                "server_accept_rate": s["server_accept_rate"],
                "swap_used_mb": res["mem_after"].get("swap_used_mb"),
                "memory_pressure": res["mem_after"].get("memory_pressure_level"),
                "raw_file": str(out_file),
            }
            append_ledger(args.ledger, row)
            all_records.append(row)
            print(
                f"[{trial}/{n_trials}] {wl:10s} pass={res['passed']} "
                f"in={s['prompt_tokens']} out={s['completion_tokens']} "
                f"ttft={s['cold_ttft_ms']}ms decode_tps={s['client_decode_tps']} "
                f"tpc={s['server_tok_per_cycle']} ({res['check_reason']})",
                flush=True,
            )

    aggregates: dict[str, Any] = {}
    for wl in workloads:
        wrows = [r for r in all_records if r["workload"] == wl]
        tps_vals = [r["client_decode_tps"] for r in wrows if r["client_decode_tps"] is not None]
        wall_vals = [r["client_wall_ms"] for r in wrows if r["client_wall_ms"] is not None]
        ttft_vals = [r["cold_ttft_ms"] for r in wrows if r["cold_ttft_ms"] is not None]
        aggregates[wl] = {
            "n": len(wrows),
            "passed": sum(1 for r in wrows if r["passed"]),
            "median_decode_tps": round(statistics.median(tps_vals), 3) if tps_vals else None,
            "min_decode_tps": round(min(tps_vals), 3) if tps_vals else None,
            "max_decode_tps": round(max(tps_vals), 3) if tps_vals else None,
            "median_wall_ms": round(statistics.median(wall_vals), 2) if wall_vals else None,
            "median_cold_ttft_ms": round(statistics.median(ttft_vals), 2) if ttft_vals else None,
        }

    summary_path = raw_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "tag": tag,
                "mode": args.mode,
                "leg": args.leg,
                "trials": n_trials,
                "model": model,
                "aggregates": aggregates,
                "records": all_records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0 if all(r["passed"] for r in all_records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
