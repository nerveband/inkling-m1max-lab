"""Symlink overlay builder against oMLX 0.8.2 Resources for Ornith-1.5-35B-A3B on M1 Max."""

from __future__ import annotations

import argparse
import pathlib
import shutil
from typing import Iterable

DEFAULT_RESOURCES = pathlib.Path("/Applications/oMLX.app/Contents/Resources")

ALL_FEATURES: tuple[str, ...] = (
    "m1_caps",
    "add_rms_2048",
    "min_n_512",
    "sorted_moe_verify",
    "fp16_fast_router",
    "m1_qmv_moe",
    "suffix_policy",
)

FEATURE_TARGETS: dict[str, tuple[str, ...]] = {
    "m1_caps": ("omlx/patches/mlx_lm_mtp/batch_generator.py",),
    "add_rms_2048": ("omlx/patches/qwen35_verify_qmm.py",),
    "min_n_512": (
        "omlx/patches/m1_verify_qmv.py",
        "omlx/patches/m1_verify_mma.py",
    ),
    "sorted_moe_verify": ("omlx/patches/qwen35_moe_routed_decode.py",),
    "fp16_fast_router": ("omlx/patches/qwen35_moe_router.py",),
    "m1_qmv_moe": (
        "omlx/patches/qwen35_moe_router.py",
        "omlx/patches/qwen35_moe_routed_decode.py",
    ),
    "suffix_policy": (
        "omlx/patches/mlx_lm_mtp/batch_generator.py",
        "omlx/speculative/dflash/drafter.py",
    ),
}

_LAB_PATCHES = pathlib.Path(__file__).resolve().parents[1] / "lab" / "patches"


def _patch_qwen35_router_m1_qmv_moe(text: str) -> str:
    text = _replace_once(
        text,
        "from functools import wraps\n",
        "import os\nfrom functools import wraps\n",
        label="qwen35_moe_router m1_qmv_moe os",
    )
    text = _replace_once(
        text,
        "def _ensure_vlm_verify_patch() -> None:\n",
        "_M1_QMV_MOE_MIN_ROWS = max(1, int(os.environ.get('OMLX_M1_QMV_MOE_MIN_ROWS', '2')))\n"
        "_M1_QMV_MOE_MAX_ROWS = min(16, int(os.environ.get('OMLX_M1_QMV_MOE_MAX_ROWS', '16')))\n\n\n"
        "def _ensure_vlm_verify_patch() -> None:\n",
        label="qwen35_moe_router m1_qmv_moe config",
    )
    return _replace_once(
        text,
        "        shared = self._feed_forward(feed_forward.shared_expert, x)\n"
        "        shared = mx.sigmoid(self._linear(feed_forward.shared_expert_gate, x)) * shared\n"
        "        routed = self._switch_glu(feed_forward.switch_mlp, x, indices)\n"
        "        return (routed * scores[..., None]).sum(axis=-2) + shared\n",
        "        shared = self._feed_forward(feed_forward.shared_expert, x)\n"
        "        shared_g = self._linear(feed_forward.shared_expert_gate, x)\n"
        "        rows = x.size // x.shape[-1]\n"
        "        if _M1_QMV_MOE_MIN_ROWS <= rows <= _M1_QMV_MOE_MAX_ROWS:\n"
        "            from .m1_qmv_moe import (\n"
        "                eligible_switch as _qmv_eligible,\n"
        "                m1_qmv_moe_switch as _qmv_switch,\n"
        "                prepare_switch_moe as _prepare_qmv_switch,\n"
        "            )\n"
        "            if _qmv_eligible(x, feed_forward):\n"
        "                _prepare_qmv_switch(feed_forward)\n"
        "                if not globals().get('_QMV_ENGAGED_LOGGED'):\n"
        "                    globals()['_QMV_ENGAGED_LOGGED'] = True\n"
        "                    print('BENCH engaged m1_qmv_moe', flush=True)\n"
        "                return _qmv_switch(\n"
        "                    x,\n"
        "                    indices,\n"
        "                    scores,\n"
        "                    feed_forward.switch_mlp,\n"
        "                    shared_out=shared,\n"
        "                    shared_gate=shared_g,\n"
        "                )\n"
        "        routed = self._switch_glu(feed_forward.switch_mlp, x, indices)\n"
        "        return (routed * scores[..., None]).sum(axis=-2) + mx.sigmoid(shared_g) * shared\n",
        label="qwen35_moe_router m1_qmv_moe verify",
    )


def _patch_routed_decode_m1_qmv_moe(text: str) -> str:
    text = _replace_once(
        text,
        "import logging\n",
        "import logging\nimport os\n",
        label="qwen35_moe_routed_decode m1_qmv_moe os",
    )
    text = _replace_once(
        text,
        "def _ensure_verify_window_patch(cls) -> None:\n",
        "_QMV_VERIFY_MIN_ROWS = max(1, int(os.environ.get('OMLX_M1_QMV_MOE_MIN_ROWS', '2')))\n"
        "_QMV_VERIFY_MAX_ROWS = min(16, int(os.environ.get('OMLX_M1_QMV_MOE_MAX_ROWS', '16')))\n\n\n"
        "def _ensure_verify_window_patch(cls) -> None:\n",
        label="qwen35_moe_routed_decode m1_qmv_moe bounds",
    )
    return _replace_once(
        text,
        "    @wraps(original)\n"
        "    def verify_window(self, feed_forward, x):\n"
        "        armed = is_row_exact_armed() or is_verify_armed()\n"
        "        if armed and isinstance(feed_forward, cls):\n"
        "            y = routed_verify_window(feed_forward, x)\n"
        "            if y is not None:\n"
        "                return y.reshape(x.shape)\n"
        "        return original(self, feed_forward, x)\n",
        "    @wraps(original)\n"
        "    def verify_window(self, feed_forward, x):\n"
        "        armed = is_row_exact_armed() or is_verify_armed()\n"
        "        if armed and isinstance(feed_forward, cls):\n"
        "            rows = x.size // x.shape[-1] if x.ndim >= 2 else 0\n"
        "            if not (_QMV_VERIFY_MIN_ROWS <= rows <= _QMV_VERIFY_MAX_ROWS):\n"
        "                y = routed_verify_window(feed_forward, x)\n"
        "                if y is not None:\n"
        "                    return y.reshape(x.shape)\n"
        "        return original(self, feed_forward, x)\n",
        label="qwen35_moe_routed_decode m1_qmv_moe verify_window",
    )


def _patch_bg_suffix_policy(text: str) -> str:
    import sys

    if str(_LAB_PATCHES) not in sys.path:
        sys.path.insert(0, str(_LAB_PATCHES))
    from inkling2_pr_patches import patch_batch_generator_suffix_policy

    return patch_batch_generator_suffix_policy(text)


def _patch_drafter_suffix_policy(text: str) -> str:
    import sys

    if str(_LAB_PATCHES) not in sys.path:
        sys.path.insert(0, str(_LAB_PATCHES))
    from inkling2_pr_patches import patch_drafter

    return patch_drafter(text, {"suffix_policy"})

_FP16_ROUTER_SPECIALIZATION = """
template <int NE>
struct omlx_router_topk<half, NE> {
  enum : uint { PER = uint(NE) / 32 };

  static METAL_FUNC void keys(const device half* logits, uint lane, thread uint* key) {
    float vals[PER];
    omlx_router_softmax_row<half, NE>(logits, lane, vals);
    for (uint k = 0; k < PER; ++k) {
      const uint expert = ((k / 4) * 32 + lane) * 4 + (k % 4);
      if (metal::isnan(vals[k])) {
        key[k] = 0u;
      } else {
        const uint hbits = (vals[k] == 0.0f)
            ? 0u
            : uint(as_type<ushort>(static_cast<half>(vals[k])));
        key[k] = ((hbits + 1u) << 16) | expert;
      }
    }
  }

  static METAL_FUNC void top3(thread const uint* key, uint below, thread uint* head) {
    uint a = 0u, b = 0u, c = 0u;
    for (uint k = 0; k < PER; ++k) {
      const uint v = key[k] < below ? key[k] : 0u;
      const bool ga = v > a, gb = v > b, gc = v > c;
      c = gb ? b : (gc ? v : c);
      b = ga ? a : (gb ? v : b);
      a = ga ? v : a;
    }
    head[0] = a;
    head[1] = b;
    head[2] = c;
  }

  static METAL_FUNC uint step(thread const uint* key, thread uint* head, thread int& left) {
    const uint sel = simd_max(head[0]);
    if (sel != 0u && head[0] == sel) {
      head[0] = head[1];
      head[1] = head[2];
      head[2] = 0u;
      if (--left == 0) {
        top3(key, sel, head);
        left = 3;
      }
    }
    return sel;
  }

  static METAL_FUNC uint nth(const device half* logits, uint lane, int n) {
    uint key[PER];
    keys(logits, lane, key);
    uint head[3];
    top3(key, 0xffffffffu, head);
    int left = 3;
    uint sel = 0u;
    for (int j = 0; j <= n; ++j) {
      sel = step(key, head, left);
    }
    return sel == 0u ? uint(NE) : (sel & 0xffffu);
  }

  template <int K>
  static METAL_FUNC void write(
      const device half* logits, uint lane, device uint32_t* indices,
      device half* scores) {
    uint key[PER];
    keys(logits, lane, key);
    uint head[3];
    top3(key, 0xffffffffu, head);
    int left = 3;
    float sel_p[K];
    uint sel_i[K];
    for (int j = 0; j < K; ++j) {
      const uint sel = step(key, head, left);
      sel_p[j] = sel == 0u
          ? -INFINITY
          : float(as_type<half>(ushort((sel >> 16) - 1u)));
      sel_i[j] = sel == 0u ? uint(NE) : (sel & 0xffffu);
    }
    omlx_router_topk_store<half, K>(sel_p, sel_i, lane, indices, scores);
  }
};
"""


def _replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"Expected 1 occurrence of {label!r}, found {count}")
    return text.replace(old, new)


def _patch_m1_caps(text: str) -> str:
    old = '    return architecture.startswith("applegpu_g15")\n'
    new = '    return architecture.startswith(("applegpu_g15", "applegpu_g13"))\n'
    return _replace_once(text, old, new, label="m1_caps")


def _patch_add_rms_2048(text: str) -> str:
    old_launch = (
        "    D = int(a.shape[-1])\n"
        "    rows = a.size // D\n"
        '    return _KERNEL_CACHE["add_rms"](\n'
        "        inputs=[a, b, norm.weight, mx.array([norm.eps], dtype=mx.float32)],\n"
        '        template=[("T", a.dtype), ("D", D)],\n'
        "        grid=(1024 * rows, 1, 1),\n"
        "        threadgroup=(1024, 1, 1),\n"
    )
    new_launch = (
        "    D = int(a.shape[-1])\n"
        "    rows = a.size // D\n"
        "    n_threads = 512 if D <= 2048 else 1024\n"
        '    return _KERNEL_CACHE["add_rms"](\n'
        "        inputs=[a, b, norm.weight, mx.array([norm.eps], dtype=mx.float32)],\n"
        '        template=[("T", a.dtype), ("D", D)],\n'
        "        grid=(n_threads * rows, 1, 1),\n"
        "        threadgroup=(n_threads, 1, 1),\n"
    )
    text = _replace_once(text, old_launch, new_launch, label="add_rms_norm_threads")
    old_elig = "        and 4096 < a.shape[-1] <= 16384\n"
    new_elig = "        and 2048 <= a.shape[-1] <= 16384\n"
    return _replace_once(text, old_elig, new_elig, label="add_rms_eligible_2048")


def _patch_min_n_512(text: str) -> str:
    return _replace_once(text, "_MIN_N = 1024\n", "_MIN_N = 512\n", label="min_n_512")


def _patch_sorted_moe_verify(text: str) -> str:
    old_imp = "import logging\n"
    new_imp = "import logging\nimport os\n"
    text = _replace_once(text, old_imp, new_imp, label="sorted_moe_import_os")

    old_cfg = "_WINDOW_PROVEN = False\n"
    new_cfg = (
        "_WINDOW_PROVEN = False\n"
        '_SORTED_MOE_VERIFY_MIN_ROWS = int(os.environ.get("OMLX_SORTED_MOE_VERIFY_MIN_ROWS", "4"))\n'
    )
    text = _replace_once(text, old_cfg, new_cfg, label="sorted_moe_env_flag")

    old_wrap = (
        "    @wraps(original)\n"
        "    def verify_window(self, feed_forward, x):\n"
        "        armed = is_row_exact_armed() or is_verify_armed()\n"
        "        if armed and isinstance(feed_forward, cls):\n"
        "            y = routed_verify_window(feed_forward, x)\n"
        "            if y is not None:\n"
        "                return y.reshape(x.shape)\n"
        "        return original(self, feed_forward, x)\n"
    )
    new_wrap = (
        "    @wraps(original)\n"
        "    def verify_window(self, feed_forward, x):\n"
        "        armed = is_row_exact_armed() or is_verify_armed()\n"
        "        if armed and isinstance(feed_forward, cls):\n"
        "            rows = x.size // x.shape[-1] if x.ndim >= 2 else 0\n"
        "            if (\n"
        "                _SORTED_MOE_VERIFY_MIN_ROWS <= 0\n"
        "                or not (_SORTED_MOE_VERIFY_MIN_ROWS <= rows <= 8)\n"
        "            ):\n"
        "                y = routed_verify_window(feed_forward, x)\n"
        "                if y is not None:\n"
        "                    return y.reshape(x.shape)\n"
        "        return original(self, feed_forward, x)\n"
    )
    return _replace_once(text, old_wrap, new_wrap, label="sorted_moe_verify_window")


def _patch_fp16_fast_router(text: str) -> str:
    old = (
        "    omlx_router_topk_store<bfloat16_t, K>(sel_p, sel_i, lane, indices, scores);\n"
        "  }\n"
        "};\n"
        '"""\n'
    )
    new = (
        "    omlx_router_topk_store<bfloat16_t, K>(sel_p, sel_i, lane, indices, scores);\n"
        "  }\n"
        "};\n"
        + _FP16_ROUTER_SPECIALIZATION
        + '"""\n'
    )
    return _replace_once(text, old, new, label="fp16_fast_router")


def normalize_features(features: Iterable[str]) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()
    for raw in features:
        for item in str(raw).split(","):
            name = item.strip()
            if not name or name == "none":
                continue
            if name == "all":
                for feat in ALL_FEATURES:
                    if feat not in seen:
                        seen.add(feat)
                        resolved.append(feat)
                continue
            if name not in FEATURE_TARGETS:
                raise ValueError(
                    f"Unknown feature {name!r}; expected one of {ALL_FEATURES} or 'all'"
                )
            if name not in seen:
                seen.add(name)
                resolved.append(name)
    return resolved


def build_overlay(
    overlay_dir: pathlib.Path,
    features: Iterable[str],
    resources: pathlib.Path = DEFAULT_RESOURCES,
    *,
    clean: bool = True,
) -> list[pathlib.Path]:
    """Build a symlink overlay over ``resources`` containing only patched files."""
    active = set(normalize_features(features))
    base = pathlib.Path(overlay_dir)
    if clean and base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)

    targets: set[str] = set()
    for feat in active:
        targets.update(FEATURE_TARGETS[feat])

    if not targets:
        # Still mirror top-level entries cleanly when building a baseline overlay.
        for child in resources.iterdir():
            if child.name == "__pycache__":
                continue
            dst = base / child.name
            if not dst.exists() and not dst.is_symlink():
                dst.symlink_to(child, target_is_directory=child.is_dir())
        return []

    dir_map: dict[pathlib.Path, set[str]] = {}
    for rel in sorted(targets):
        p = pathlib.Path(rel)
        for i in range(len(p.parts)):
            parent = pathlib.Path(*p.parts[:i]) if i else pathlib.Path(".")
            dir_map.setdefault(parent, set()).add(p.parts[i])

    for rel_dir, keep_names in sorted(dir_map.items()):
        src_dir = resources / rel_dir
        dst_dir = base / rel_dir
        dst_dir.mkdir(parents=True, exist_ok=True)
        for child in src_dir.iterdir():
            if child.name == "__pycache__" or child.name in keep_names:
                continue
            dst = dst_dir / child.name
            if not dst.exists() and not dst.is_symlink():
                dst.symlink_to(child, target_is_directory=child.is_dir())

    written: list[pathlib.Path] = []
    patch_map: dict[str, list[Any]] = {
        "omlx/patches/mlx_lm_mtp/batch_generator.py": [
            fn
            for feat, fn in (
                ("m1_caps", _patch_m1_caps),
                ("suffix_policy", _patch_bg_suffix_policy),
            )
            if feat in active
        ],
        "omlx/speculative/dflash/drafter.py": (
            [_patch_drafter_suffix_policy] if "suffix_policy" in active else []
        ),
        "omlx/patches/qwen35_verify_qmm.py": (
            [_patch_add_rms_2048] if "add_rms_2048" in active else []
        ),
        "omlx/patches/m1_verify_qmv.py": (
            [_patch_min_n_512] if "min_n_512" in active else []
        ),
        "omlx/patches/m1_verify_mma.py": (
            [_patch_min_n_512] if "min_n_512" in active else []
        ),
        "omlx/patches/qwen35_moe_routed_decode.py": (
            [_patch_routed_decode_m1_qmv_moe]
            if "m1_qmv_moe" in active
            else ([_patch_sorted_moe_verify] if "sorted_moe_verify" in active else [])
        ),
        "omlx/patches/qwen35_moe_router.py": [
            fn
            for feat, fn in (
                ("fp16_fast_router", _patch_fp16_fast_router),
                ("m1_qmv_moe", _patch_qwen35_router_m1_qmv_moe),
            )
            if feat in active
        ],
    }
    for rel in sorted(targets):
        src_file = resources / rel
        dst_file = base / rel
        text = src_file.read_text()
        for fn in patch_map[rel]:
            text = fn(text)
        dst_file.write_text(text)
        written.append(dst_file)

    if "m1_qmv_moe" in active:
        dst_qmv = base / "omlx/patches/m1_qmv_moe.py"
        dst_qmv.write_text((_LAB_PATCHES / "m1_qmv_moe.py").read_text())
        written.append(dst_qmv)
    if "suffix_policy" in active:
        for src_name, dst_rel in (
            ("omlx_suffix_drafter.py", "omlx/speculative/suffix_drafter.py"),
            ("omlx_draft_policy.py", "omlx/speculative/draft_policy.py"),
        ):
            dst_extra = base / dst_rel
            dst_extra.write_text((_LAB_PATCHES / src_name).read_text())
            written.append(dst_extra)
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an oMLX 0.8.2 M1 Max overlay for Ornith-1.5-35B-A3B."
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        required=True,
        help="Destination directory for the symlink overlay.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=["all"],
        help=f"Feature flags ({', '.join(ALL_FEATURES)}, 'all', or 'none').",
    )
    parser.add_argument(
        "--resources",
        type=pathlib.Path,
        default=DEFAULT_RESOURCES,
        help="Path to pinned oMLX Resources directory.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not wipe --out before populating.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    written = build_overlay(
        args.out,
        args.features,
        args.resources,
        clean=not args.no_clean,
    )
    feats = normalize_features(args.features)
    print(f"Overlay built at {args.out} (features={feats}, patched={len(written)}):")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
