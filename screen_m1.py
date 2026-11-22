"""Aligned primed Pi speed screen runner for M1 Max kernel experiments."""
import argparse
import json
import os
import pathlib
import plistlib
import re
import shutil
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
WORK = ROOT / "work"
LAB = ROOT / "work/lab"
SCRATCH = ROOT / "work/scratch"
MODEL_DIR = pathlib.Path(os.environ.get("MODEL_DIR", "/Volumes/Extreme SSD/Models/Inkling-2-Mini"))
PY312 = "/opt/homebrew/bin/python3.12"


def _resolve_resources() -> pathlib.Path:
    """Resolve pinned oMLX 0.8.2 Resources from mount-omlx-082 or /Applications."""
    mounted = WORK / "mount-omlx-082/oMLX.app/Contents/Resources"
    if mounted.is_dir():
        return mounted
    app_res = pathlib.Path("/Applications/oMLX.app/Contents/Resources")
    plist_path = app_res.parent / "Info.plist"
    if plist_path.is_file():
        with plist_path.open("rb") as fh:
            info = plistlib.load(fh)
        if info.get("CFBundleShortVersionString") == "0.8.2" and app_res.is_dir():
            return app_res
    raise FileNotFoundError(
        "Pinned oMLX 0.8.2 Resources not found under work/mount-omlx-082 or /Applications/oMLX.app"
    )


RESOURCES = _resolve_resources()

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


def _load_settings_override(raw: str) -> dict:
    """Parse --settings JSON string or JSON file into a model_settings entry dict."""
    text = raw.strip()
    data = json.loads(text) if text.startswith("{") else json.loads(pathlib.Path(raw).read_text())
    if not isinstance(data, dict):
        raise ValueError("--settings must decode to a JSON object")
    if "models" in data and isinstance(data["models"], dict):
        models = data["models"]
        if "inkling" in models:
            return dict(models["inkling"])
        if len(models) == 1:
            return dict(next(iter(models.values())))
    return data


def _parse_env_items(items: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--env expects KEY=VAL, got: {item!r}")
        key, val = item.split("=", 1)
        if not key:
            raise ValueError(f"--env expects non-empty key, got: {item!r}")
        parsed[key] = val
    return parsed


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aligned primed Pi speed screen runner for M1 Max kernel experiments."
    )
    parser.add_argument("tag", help="Run tag (creates work/scratch/speed-<tag>)")
    parser.add_argument("quant", help="Quantization directory suffix under MODEL_DIR (e.g. oq3, oq4)")
    parser.add_argument("features", nargs="?", default="", help="Comma-separated overlay feature names")
    parser.add_argument("depth", nargs="?", type=int, default=6, help="MTP fixed draft depth (default: 6)")
    parser.add_argument("--context", type=int, default=32768, help="max_context_window setting (default: 32768)")
    parser.add_argument("--port", type=int, default=18089, help="oMLX listen port (default: 18089)")
    parser.add_argument(
        "--enable-thinking-off",
        action="store_true",
        help="Set enable_thinking=false in model_settings.json",
    )
    parser.add_argument(
        "--settings",
        default=None,
        help="JSON object or path to JSON file merged into model_settings.json entry",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VAL",
        help="Extra environment variable passed to the server process (repeatable)",
    )
    return parser.parse_args(argv)


def run_screen(
    tag: str,
    quant: str,
    features: set[str],
    depth: int = 6,
    context: int = 32768,
    port: int = 18089,
    enable_thinking: bool | None = None,
    settings_override: dict | None = None,
    extra_env: dict[str, str] | None = None,
):
    dest = SCRATCH / f"speed-{tag}"
    dest.mkdir(parents=True)
    state = dest / "state"
    state.mkdir()
    models = dest / "models"
    models.mkdir()
    (models / "inkling").symlink_to(MODEL_DIR / f"omlx-{quant}", target_is_directory=True)
    entry: dict = {"max_context_window": context, "mtp_enabled": depth > 0}
    if depth > 0:
        entry["mtp_fixed_depth"] = depth
    if enable_thinking is not None:
        entry["enable_thinking"] = enable_thinking
    if settings_override:
        entry.update(settings_override)
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
    env_assignments = ["PYTHONPATH=" + os.pathsep.join(map(str, paths))]
    if extra_env:
        env_assignments.extend(f"{k}={v}" for k, v in extra_env.items())
    cmd = [
        "env",
        *env_assignments,
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
        str(port),
        "--max-concurrent-requests",
        "1",
        "--headless",
    ]
    kv_parent = SCRATCH / "kv"
    kv_parent.mkdir(parents=True, exist_ok=True)
    for stale in kv_parent.glob(f"speed-{tag}-*"):
        shutil.rmtree(stale, ignore_errors=True)
    cache = pathlib.Path(tempfile.mkdtemp(prefix=f"speed-{tag}-", dir=kv_parent))
    try:
        cmd += ["--paged-ssd-cache-dir", str(cache), "--paged-ssd-cache-max-size", "2GB", "--hot-cache-max-size", "1GB"]
        guarded(cmd, [PY312, "-u", str(WORK / "aligned-pi.py"), str(dest)], dest, port=port, ready_path="/v1/models")
    finally:
        shutil.rmtree(cache, ignore_errors=True)

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
    # Usage: screen_m1.py <tag> <oq3|oq4> [comma-separated features] [depth] [--enable-thinking-off] [--env KEY=VAL] [--settings JSON]
    args = _parse_args()
    feats = {f for f in args.features.split(",") if f}
    enable_thinking = False if args.enable_thinking_off else None
    settings_override = _load_settings_override(args.settings) if args.settings else None
    extra_env = _parse_env_items(args.env) if args.env else None
    run_screen(
        args.tag,
        args.quant,
        feats,
        depth=args.depth,
        context=args.context,
        port=args.port,
        enable_thinking=enable_thinking,
        settings_override=settings_override,
        extra_env=extra_env,
    )
