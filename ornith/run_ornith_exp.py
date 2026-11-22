"""Unified server launcher + bench_ornith.py runner for Track B (Splash 1.4, Splash 1.7, oMLX 0.8.2)."""
import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

LAB = pathlib.Path(__file__).resolve().parent
ROOT = LAB.parents[1]
RESOURCES = pathlib.Path("/Applications/oMLX.app/Contents/Resources")
SITE = str(RESOURCES / "Python/framework-mlx-base/lib/python3.11/site-packages")
PY = str(RESOURCES / "Python/cpython-3.11/bin/python3.11")

ORNITH_DIR = pathlib.Path.home() / ".omlx/models/PocketAiHub/Ornith-1.5-35B-A3B-Abliterated-MLX-4bit"
DRAFT_QWEN36 = pathlib.Path.home() / ".omlx/models/incoai/Qwen3.6-35B-A3B-DFlash2"
DRAFT_FELKHAYER = pathlib.Path.home() / ".omlx/models/felkhayer/Ornith-1.5-35B-A3B-DFlash2-Splash"
DRAFT_JZINNO = pathlib.Path.home() / ".omlx/models/jzinno/Ornith-1.5-35B-A3B-DFlash2"

SPLASH14 = pathlib.Path("/opt/homebrew/bin/splash-m1")
SPLASH17 = LAB / "splash-1.7.0-m1/splash-1.7.0-m1"


def wait_ready(url: str, proc: subprocess.Popen, timeout: float = 240.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(f"{url}/v1/models", timeout=2.0) as r:
                if r.status == 200:
                    data = json.loads(r.read())
                    if data.get("data"):
                        return
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for {url}/v1/models")


def stage_splash_model(hub_dir: pathlib.Path) -> str:
    sys.path.insert(0, str(RESOURCES))
    sys.path.insert(0, SITE)
    from omlx.engine.splash import stage_model

    hub_dir.mkdir(parents=True, exist_ok=True)
    repo_id = "omlx/Ornith-1.5-35B-A3B-Abliterated-MLX-4bit"
    stage_model(ORNITH_DIR, repo_id, hub_dir)
    return repo_id


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("tag")
    p.add_argument("--backend", choices=("splash14", "splash17", "omlx"), required=True)
    p.add_argument("--leg", default=None)
    p.add_argument("--port", type=int, default=18090)
    p.add_argument("--draft", default="qwen36", help="qwen36|felkhayer|jzinno|none|PATH")
    p.add_argument("--max-memory", default=None)
    p.add_argument("--expert-memory", default=None)
    p.add_argument("--kv-format", default="int8")
    p.add_argument("--state-format", default=None)
    p.add_argument("--max-context", type=int, default=65536)
    p.add_argument("--language-only", action="store_true")
    p.add_argument("--splash-extra", action="append", default=[])
    # oMLX options
    p.add_argument("--features", default="none")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--dflash-quant", action="store_true")
    p.add_argument("--dflash-verify-mode", default="adaptive")
    p.add_argument("--dflash-block-size", type=int, default=None)
    p.add_argument("--hot-cache-max-size", default="12GB")
    p.add_argument("--env", action="append", default=[])
    # Bench options
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--thinking", choices=("off", "on", "auto"), default="off")
    p.add_argument("--workloads", default="all")
    p.add_argument("--mode", choices=("single", "aba"), default="single")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--trials", type=int, default=6)
    p.add_argument("--timeout", type=float, default=600.0)
    return p.parse_args()


def resolve_draft(name: str) -> pathlib.Path | None:
    if name == "none":
        return None
    mapping = {
        "qwen36": DRAFT_QWEN36,
        "felkhayer": DRAFT_FELKHAYER,
        "jzinno": DRAFT_JZINNO,
    }
    return mapping.get(name, pathlib.Path(name).expanduser())


def main() -> int:
    args = parse_args()
    raw_dir = LAB / "raw" / args.tag
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_path = raw_dir / "server.log"
    leg = args.leg or args.tag
    draft_path = resolve_draft(args.draft)

    ov_tmp_ctx = None
    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    for item in args.env:
        k, v = item.split("=", 1)
        env[k] = v

    if args.backend in ("splash14", "splash17"):
        binary = SPLASH14 if args.backend == "splash14" else SPLASH17
        hub_dir = ROOT / "work/scratch/splash-hub"
        repo_id = stage_splash_model(hub_dir)
        env["HF_HUB_CACHE"] = str(hub_dir)
        cmd = [
            str(binary),
            "serve",
            "--model",
            repo_id,
            "--offline",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--no-webui",
            "--idle-release",
            "off",
            "--kv-format",
            args.kv_format,
            "--max-cache-disk",
            "0",
            "--max-context",
            str(args.max_context),
        ]
        if draft_path is not None:
            cmd += ["--draft-model", str(draft_path)]
        if args.max_memory:
            cmd += ["--max-memory", str(args.max_memory)]
        if args.expert_memory:
            cmd += ["--expert-memory", str(args.expert_memory)]
        if args.state_format:
            cmd += ["--state-format", str(args.state_format)]
        if args.language_only:
            cmd.append("--language-only")
        cmd += args.splash_extra
    else:
        scratch_dir = ROOT / "work/scratch" / f"ornith-omlx-{args.tag}"
        shutil.rmtree(scratch_dir, ignore_errors=True)
        state = scratch_dir / "state"
        kv_dir = scratch_dir / "kv"
        models = state / "models"
        models.mkdir(parents=True)
        kv_dir.mkdir(parents=True)
        (models / "ornith").symlink_to(ORNITH_DIR, target_is_directory=True)

        ms_cfg: dict[str, object] = {
            "max_context_window": args.max_context,
            "activation_fp16_enabled": bool(args.fp16),
            "temperature": float(args.temp),
        }
        if draft_path is not None:
            ms_cfg.update(
                {
                    "dflash_enabled": True,
                    "dflash_draft_model": str(draft_path),
                    "dflash_verify_mode": args.dflash_verify_mode,
                    "dflash_draft_quant_enabled": bool(args.dflash_quant),
                }
            )
            if args.dflash_quant:
                ms_cfg["dflash_draft_quant_weight_bits"] = 4
                ms_cfg["dflash_draft_quant_group_size"] = 64
            if args.dflash_block_size is not None:
                ms_cfg["dflash_block_size"] = int(args.dflash_block_size)
        else:
            ms_cfg["dflash_enabled"] = False
        (state / "model_settings.json").write_text(
            json.dumps({"version": 1, "models": {"ornith": ms_cfg}}, indent=2) + "\n"
        )
        ov_tmp_ctx = tempfile.TemporaryDirectory(prefix=f"ornith-ov-{args.tag}-")
        ov_dir = pathlib.Path(ov_tmp_ctx.name)
        sys.path.insert(0, str(LAB))
        import ornith_m1_overlay as oo

        oo.build_overlay(ov_dir, [args.features], RESOURCES)
        env["HOME"] = str(state)
        env["OMLX_BASE_PATH"] = str(state)
        env["PYTHONPATH"] = os.pathsep.join([str(ov_dir), str(RESOURCES), SITE])
        cmd = [
            PY,
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
            str(args.port),
            "--max-concurrent-requests",
            "1",
            "--headless",
            "--paged-ssd-cache-dir",
            str(kv_dir),
            "--paged-ssd-cache-max-size",
            "2GB",
            "--hot-cache-max-size",
            str(args.hot_cache_max_size),
        ]

    with log_path.open("w") as logf:
        proc = subprocess.Popen(
            cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True
        )
        try:
            base_url = f"http://127.0.0.1:{args.port}/v1"
            wait_ready(f"http://127.0.0.1:{args.port}", proc)
            bcmd = [
                PY,
                "-u",
                str(LAB / "bench_ornith.py"),
                "--url",
                base_url,
                "--model-dir",
                str(ORNITH_DIR),
                "--tag",
                args.tag,
                "--leg",
                leg,
                "--server-log",
                str(log_path),
                "--temp",
                str(args.temp),
                "--thinking",
                args.thinking,
                "--workloads",
                args.workloads,
                "--mode",
                args.mode,
                "--repeats",
                str(args.repeats),
                "--trials",
                str(args.trials),
                "--timeout",
                str(args.timeout),
            ]
            return subprocess.call(bcmd)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            if ov_tmp_ctx is not None:
                ov_tmp_ctx.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
