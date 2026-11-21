"""Aligned primed Pi speed screen runner for M1 Max kernel experiments."""
import json
import os
import pathlib
import re
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
WORK = ROOT / "work"
LAB = ROOT / "work/lab"
SCRATCH = ROOT / "work/scratch"
RESOURCES = WORK / "mount-omlx-082/oMLX.app/Contents/Resources"
MODEL_DIR = pathlib.Path(os.environ.get("MODEL_DIR", "/Volumes/Extreme SSD/Models/Inkling-2-Mini"))
PY312 = "/opt/homebrew/bin/python3.12"

sys.path.insert(0, str(LAB))
from guard import guarded
from m1_overlay import PATCHES, build_overlay as _build_overlay_dir
from summary import summarize


def build_overlay(dest: pathlib.Path, features: set[str]) -> pathlib.Path:
    """Build a minimal symlink overlay overriding only modified oMLX files."""
    return _build_overlay_dir(dest / "overlay", features, RESOURCES)


def extract_mtp(dest: pathlib.Path) -> list[dict]:
    rows = []
    for line in (dest / "server.log").read_text(errors="replace").splitlines():
        if " INFO - MTP[" not in line or " timing[" not in line:
            continue
        details = line.split(" INFO - ", 1)[1]
        if "/" in details.split("MTP[", 1)[0]:
            continue
        totals = re.search(r"tokens=(\d+) cycles=(\d+).*?accept=(\d+)/(\d+)", details)
        if totals:
            accounted = re.search(r"timing\[([^\]]+)\]", details)[1]
            ms = {k: float(v) for k, v in re.findall(r"([a-z_]+)=([\d.]+)ms", accounted)}
            tokens, cycles, accepted, proposed = map(int, totals.groups())
            rows.append(dict(tokens=tokens, cycles=cycles, accepted=accepted, proposed=proposed, ms=ms, detail=details))
    (dest / "mtp.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def run_screen(tag: str, quant: str, features: set[str], depth: int = 6, context: int = 32768):
    dest = SCRATCH / f"speed-{tag}"
    dest.mkdir(parents=True)
    state = dest / "state"
    state.mkdir()
    models = dest / "models"
    models.mkdir()
    (models / "inkling").symlink_to(MODEL_DIR / f"omlx-{quant}", target_is_directory=True)
    entry = {"max_context_window": context, "mtp_enabled": depth > 0}
    if depth > 0:
        entry["mtp_fixed_depth"] = depth
    (state / "model_settings.json").write_text(json.dumps({"version": 2, "models": {"inkling": entry}}))

    overlay_dir = build_overlay(dest, features)
    paths = [
        overlay_dir,
        LAB / "patches",
        RESOURCES,
        RESOURCES / "Python/framework-mlx-base/lib/python3.11/site-packages",
    ]
    os.environ.update(
        BENCH_MODEL_ID="inkling",
        BENCH_TOKENIZER="omlx",
        HF_HUB_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
        HF_HOME=str(MODEL_DIR / "hf-cache"),
    )
    cmd = [
        "env",
        "PYTHONPATH=" + os.pathsep.join(map(str, paths)),
        str(RESOURCES / "Python/cpython-3.11/bin/python3.11"),
        "-m",
        "omlx.cli",
        "serve",
        "--base-path",
        str(state),
        "--model-dir",
        str(models),
        "--host",
        "127.0.0.1",
        "--port",
        "18089",
        "--max-concurrent-requests",
        "1",
        "--headless",
    ]
    kv_parent = MODEL_DIR / "hf-cache/experiment-kv"
    kv_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"speed-{tag}-", dir=kv_parent) as cache:
        cmd += ["--paged-ssd-cache-dir", cache, "--paged-ssd-cache-max-size", "2GB", "--hot-cache-max-size", "1GB"]
        guarded(cmd, [PY312, "-u", str(WORK / "aligned-pi.py"), str(dest)], dest, ready_path="/v1/models")

    scores = summarize(dest)
    (dest / "pi-scores.json").write_text(json.dumps(scores, indent=2) + "\n")
    mtp = extract_mtp(dest)
    s0 = scores[0]
    pi_mtp = mtp[2:] if len(mtp) >= 3 else mtp
    bb_ms = sum(r["ms"].get("backbone", 0.0) for r in pi_mtp)
    head_ms = sum(r["ms"].get("mtp", 0.0) for r in pi_mtp)
    cyc = sum(r["cycles"] for r in pi_mtp)
    acc = sum(r["accepted"] for r in pi_mtp)
    prop = sum(r["proposed"] for r in pi_mtp)
    prime = json.loads((dest / "prefix-prime.json").read_text())
    cached0 = s0["turns"][0]["usage"]["prompt_tokens_details"]["cached_tokens"]
    print(
        f"[{tag}] correct={s0['correct']} tps={s0['decode_tps']:.2f} wall={s0['wall']:.2f}s "
        f"gen={s0['generated_ids']} ttft={s0['first_ttft']:.2f}s cached0={cached0} "
        f"prime={prime['prime_s']:.2f}s cyc={cyc} acc={acc}/{prop} bb={bb_ms:.1f}ms head={head_ms:.1f}ms",
        flush=True,
    )


if __name__ == "__main__":
    # Usage: screen_m1.py <tag> <oq3|oq4> <comma-separated features> [depth]
    tag = sys.argv[1]
    quant = sys.argv[2]
    feats = set(sys.argv[3].split(",")) if len(sys.argv) > 3 and sys.argv[3] else set()
    depth = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    run_screen(tag, quant, feats, depth=depth)
