"""Self-contained oMLX 0.8.2 overlay builder for M1 Max Inkling 2 Mini optimizations."""
from __future__ import annotations

import pathlib
import sys
from typing import Iterable

LAB = pathlib.Path(__file__).resolve().parent
PATCHES = LAB / "patches"
if str(PATCHES) not in sys.path:
    sys.path.insert(0, str(PATCHES))

from inkling2_pr_patches import (
    patch_batch_generator_early_emit,
    patch_batch_generator_suffix_policy,
    patch_drafter,
    patch_dspark,
    patch_kernels,
    patch_output_parser,
    patch_prefill_adapter,
    patch_processing,
    patch_request,
    patch_scheduler,
    patch_selection_dspark_2bit,
    patch_server,
    patch_vlm_engine,
)

M1_FAST_V1_FEATURES: tuple[str, ...] = (
    "sort32",
    "caps",
    "m1_attn",
    "verify_compile",
    "split_in_proj",
    "skip_lm_head",
    "fast_decode",
    "fast_greedy",
    "prefill_pad64",
)

M1_FAST_FEATURES: tuple[str, ...] = (
    "sort32",
    "caps",
    "m1_attn",
    "verify_compile",
    "split_in_proj",
    "skip_lm_head",
    "fast_decode",
    "fast_greedy",
    "prefill_pad64",
    "verify_flush",
    "m1_qmv_moe",
    "predraft_fix",
    "dspark_2bit",
    "draft4",
    "fuse_draft_qkv",
    "bounded_cache",
    "bf16_loop_ctrl",
    "prefill_fast",
    "ttft_fixes",
)

INKLING2_PR_FEATURES: tuple[str, ...] = (
    "verify_flush",
    "predraft_fix",
    "dspark_2bit",
    "bounded_cache",
    "bf16_loop_ctrl",
    "fp8_kv_full",
    "prefill_fast",
    "suffix_policy",
    "ttft_fixes",
)

_AFFINE4_CODE = """        for layer, module in enumerate(self.model.layers):
            experts = module.mlp.experts
            for which in ("w13", "w2"):
                old = getattr(experts, which)
                if old.mode != "nvfp4":
                    continue
                key = f"model.layers.{layer}.mlp.experts.{which}."
                packed, factors = out[key + "weight"], out[key + "scales"]
                outer = out.pop(key + "global_scale").astype(mx.float32) / _NVFP4_DENOMINATOR
                parts = [[], [], []]
                for start in range(0, packed.shape[0], 16):
                    sl = slice(start, start + 16)
                    values = mx.dequantize(packed[sl], factors[sl], group_size=16, bits=4, mode="nvfp4", dtype=mx.float32)
                    values = (values * outer[sl, None, None]).astype(mx.bfloat16)
                    q, scales, biases = mx.quantize(values, group_size=64, bits=4, mode="affine")
                    mx.eval(q, scales, biases)
                    for bucket, value in zip(parts, (q, scales, biases)):
                        bucket.append(value)
                tensors = [mx.concatenate(bucket) for bucket in parts]
                mx.eval(*tensors)
                for field, value in zip(("weight", "scales", "biases"), tensors):
                    out[key + field] = value
                replaced = ExpertProj(packed.shape[0], packed.shape[1], packed.shape[2] * 8, {"mode": "affine", "group_size": 64, "bits": 4})
                setattr(experts, which, replaced)
                mx.clear_cache()
        print("BENCH affine expert conversion complete", flush=True)
"""


_ATTN4_CODE = """        for layer, module in enumerate(self.model.layers):
            attn = module.attn
            for name in ("in_proj", "o_proj", "q_proj", "k_proj", "v_proj"):
                proj = getattr(attn, name, None)
                if not isinstance(proj, nn.QuantizedLinear) or proj.mode != "affine" or proj.bits <= 4:
                    continue
                key = f"model.layers.{layer}.attn.{name}."
                if key + "weight" not in out:
                    continue
                vals = mx.dequantize(
                    out[key + "weight"],
                    out[key + "scales"],
                    out.get(key + "biases"),
                    group_size=proj.group_size,
                    bits=proj.bits,
                    mode=proj.mode,
                ).astype(mx.bfloat16)
                q4, s4, b4 = mx.quantize(vals, group_size=64, bits=4, mode="affine")
                mx.eval(q4, s4, b4)
                out[key + "weight"], out[key + "scales"], out[key + "biases"] = q4, s4, b4
                out_dims = int(q4.shape[0])
                in_dims = int(vals.shape[1])
                has_bias = "bias" in proj
                replaced = nn.QuantizedLinear(in_dims, out_dims, bias=has_bias, group_size=64, bits=4, mode="affine")
                setattr(attn, name, replaced)
                mx.clear_cache()
        print("BENCH attn4 conversion complete", flush=True)
"""


def _replace_once(text: str, old: str, new: str, label: str = "") -> str:
    count = text.count(old)
    if count != 1:
        tag = f" ({label})" if label else ""
        raise AssertionError(f"Expected 1 occurrence{tag}, found {count}: {old[:100]!r}")
    return text.replace(old, new)


def _expand_features(features: Iterable[str]) -> set[str]:
    eff = set(features)
    if "dspark_fast" in eff:
        eff |= {"draft4", "fuse_draft_qkv", "markov_topk", "fast_greedy"}
    if "ttft_fixes" in eff:
        eff |= {"early_first_emit", "enable_thinking_fix", "prompt_tail_cache"}
    if "dspark_2bit" in eff and ({"markov_topk", "loom_draft_head"} & eff):
        raise ValueError("Cannot combine dspark_2bit with markov_topk or loom_draft_head")
    return eff


def _build_layer_call(features: set[str]) -> str:
    """Build composed Inkling2Layer.__call__ cleanly for any feature subset."""
    bounded = "bounded_cache" in features
    vc = "verify_compile" in features
    sp = "split_in_proj" in features
    moe_prep = bool({"m1_moe", "subwarp_moe", "m1_qmv_moe"} & features)
    bf16_lc = "bf16_loop_ctrl" in features
    pf = "prefill_fast" in features
    fd = "fast_decode" in features
    co = "conv_opt" in features and not pf

    lines = [
        "    def __call__(self, h, blocks, cache=None, bank=None):",
        '        """``h``: partial block sum [B, L, H] bf16. ``blocks``: list of unit-RMS',
        "        FP32 completed blocks; extended in place at a block boundary.",
        '        ``bank``: the previous loop pass\'s ``(blocks, partial_n)`` messages."""',
        "        b, length, hidden = h.shape",
    ]
    if bounded:
        lines += [
            "        kv, conv = cache if cache is not None else (None, None)",
            "        base = windows.CONV if kv is None else 0",
        ]
        slot_expr = "conv[base + slot]"
    else:
        lines += [
            "        kv = cache[0] if cache is not None else None",
            "        conv = cache[1] if cache is not None else None",
        ]
        slot_expr = "conv[slot]"
    lines += [
        "        dkv = self.attn.dkv",
        "        states = []",
        "        for slot, width in ((0, dkv), (1, dkv), (2, hidden), (3, hidden)):",
        f"            s = {slot_expr} if conv is not None else None",
        "            if s is None:",
        "                s = mx.zeros((b, self._conv_k - 1, width), dtype=mx.float32)",
        "            states.append(s)",
        "        tau = self.attn.log_tau(kv, length)",
    ]
    if sp:
        lines += [
            '        if bank is not None and self.attn.sliding and self.attn.__dict__.get("_m1_split_proj") is None:',
            "            from inkling2_m1_kernels import ensure_split_in_proj",
            "            ensure_split_in_proj(self.attn)",
        ]
    if moe_prep:
        if "m1_qmv_moe" in features:
            lines += [
                '        if self.mlp.experts.w13.__dict__.get("_omlx_m1_alias") is None:',
                "            from ..m1_qmv_moe import prepare_m1_moe as _prepare_qmv_moe",
                "            _prepare_qmv_moe(self.mlp.experts)",
            ]
        else:
            lines += [
                '        if self.mlp.experts.w13.__dict__.get("_omlx_m1_alias") is None:',
                "            from inkling2_m1_kernels import prepare_m1_moe",
                "            prepare_m1_moe(self.mlp.experts)",
            ]
    if vc:
        lines += [
            "        compiled = length <= _COMPILED_MAX_TOKENS",
            "        if compiled:",
            "            from ..qwen35_verify_qmm import _is_armed as _vq_armed",
            "            _arm = _vq_armed()",
            "            pre = self._pre_verify_compiled if _arm else self._pre_compiled",
            "            post = self._post_verify_compiled if _arm else self._post_compiled",
            "        else:",
            "            pre = self._pre",
            "            post = self._post",
        ]
    else:
        lines += [
            "        compiled = length <= _COMPILED_MAX_TOKENS",
            "        pre = self._pre_compiled if compiled else self._pre",
            "        post = self._post_compiled if compiled else self._post",
        ]
    lines.append("        messages = self.loop_message(*bank) if bank is not None else None")

    if bf16_lc:
        lines += [
            "        down_t = gf_t = None",
            "        if messages is not None and b * length <= kernels._LOOP_FUSED_MAX_ROWS:",
            "            down_t, gf_t = self._packed_loop()",
        ]
        pre_args = "h, blocks, states[0], states[1], tau, messages, down_t, gf_t"
        post_args = "out, gate, h, blocks, states[2], states[3], messages, down_t, gf_t"
    else:
        pre_args = "h, blocks, states[0], states[1], tau, messages"
        post_args = "out, gate, h, blocks, states[2], states[3], messages"

    attend_call = "self.attn.attend(q, k, v, rel, sink, kv, conv)" if bounded else "self.attn.attend(q, k, v, rel, sink, kv)"

    if pf:
        lines += [
            f"        pre_out = pre({pre_args})",
            "        if len(pre_out) == 11:",
            "            q, k, v, rel, gate, sink, ks, vs, rk, rv, h_n = pre_out",
            "        else:",
            "            q, k, v, rel, gate, sink, ks, vs, h_n = pre_out",
            "            rk = rv = None",
            f"        out = {attend_call}",
            "        if self.is_boundary:",
            "            blocks.append(h_n)",
            f"        post_out = post({post_args})",
            "        if len(post_out) == 5:",
            "            h, s2, s3, ra, rm = post_out",
            "        else:",
            "            h, s2, s3 = post_out",
            "            ra = rm = None",
            "        if conv is not None:",
            "            next_states = (ks, vs, s2, s3)",
            "            raws = (rk, rv, ra, rm)",
            "            if 1 < length <= _COMPILED_MAX_TOKENS:",
            "                conv._omlx_verify_xp = {",
            "                    slot: ((states[slot], raw) if raw is not None else nxt)",
            "                    for slot, (nxt, raw) in enumerate(zip(next_states, raws))",
            "                }",
            "            keep = self._conv_k - 1",
            "            for slot, (nxt, raw) in enumerate(zip(next_states, raws)):",
        ]
        if fd:
            lines += [
                "                if raw is not None or nxt.shape[1] == keep:",
                f"                    {slot_expr} = nxt",
                "                else:",
                f"                    {slot_expr} = nxt[:, -keep:] * 1.0",
            ]
        else:
            lines += [
                "                if raw is not None:",
                f"                    {slot_expr} = nxt",
                "                else:",
                f"                    {slot_expr} = nxt[:, -keep:] * 1.0",
            ]
    elif co:
        lines += [
            f"        q, k, v, rel, gate, sink, ks, vs, h_n, ks_next, vs_next = pre({pre_args})",
            f"        out = {attend_call}",
            "        if self.is_boundary:",
            "            blocks.append(h_n)",
            f"        h, s2, s3, s2_next, s3_next = post({post_args})",
            "        if conv is not None:",
            "            if 1 < length <= _COMPILED_MAX_TOKENS:",
            "                conv._omlx_verify_xp = {0: ks, 1: vs, 2: s2, 3: s3}",
            f"            {'conv[base + 0], conv[base + 1], conv[base + 2], conv[base + 3]' if bounded else 'conv[0], conv[1], conv[2], conv[3]'} = ks_next, vs_next, s2_next, s3_next",
        ]
    else:
        lines += [
            f"        q, k, v, rel, gate, sink, ks, vs, h_n = pre({pre_args})",
            f"        out = {attend_call}",
            "        if self.is_boundary:",
            "            blocks.append(h_n)",
            f"        h, s2, s3 = post({post_args})",
            "        if conv is not None:",
            "            padded = (ks, vs, s2, s3)",
            "            if 1 < length <= _COMPILED_MAX_TOKENS:",
            "                conv._omlx_verify_xp = dict(enumerate(padded))",
            "            keep = self._conv_k - 1",
            "            for slot, window in enumerate(padded):",
        ]
        if fd:
            lines += [
                "                if window.shape[1] == keep:",
                f"                    {slot_expr} = window",
                "                    continue",
                f"                {slot_expr} = window[:, -keep:] * 1.0",
            ]
        else:
            lines.append(f"                {slot_expr} = window[:, -keep:] * 1.0")

    lines.append("        return h\n")
    return "\n".join(lines)


def _patch_model(model_txt: str, features: set[str]) -> str:
    sort_thresh = 32 if "sort32" in features else 64

    if "fp8_kv_full" in features:
        model_txt = _replace_once(
            model_txt,
            "import math\n",
            "import math\nimport os\n",
            "model import os",
        )
        model_txt = _replace_once(
            model_txt,
            "    loop_config: dict | None = None\n",
            "    loop_config: dict | None = None\n"
            "    fp8_kv: str | bool = False\n",
            "model TextArgs fp8_kv",
        )
        model_txt = _replace_once(
            model_txt,
            '        if "conv_kernel_size" in params:\n'
            '            args.sconv_kernel_size = int(params["conv_kernel_size"])\n'
            "        return args\n",
            '        if "conv_kernel_size" in params:\n'
            '            args.sconv_kernel_size = int(params["conv_kernel_size"])\n'
            '        if "fp8_kv" not in params:\n'
            '            env = os.environ.get("OMLX_INKLING2_FP8_KV", "full").strip().lower()\n'
            '            if env in ("1", "true", "yes", "all"):\n'
            '                args.fp8_kv = "all"\n'
            '            elif env == "full":\n'
            '                args.fp8_kv = "full"\n'
            "        return args\n\n"
            "    def uses_fp8_kv(self, layer_idx: int) -> bool:\n"
            "        mode = self.fp8_kv\n"
            "        if not mode:\n"
            "            return False\n"
            "        if isinstance(mode, str):\n"
            "            m = mode.strip().lower()\n"
            '            if m in ("", "0", "false", "no", "off"):\n'
            "                return False\n"
            '            if m == "full":\n'
            "                return not self.is_sliding(layer_idx)\n"
            "            return True\n"
            "        return True\n",
            "model TextArgs.from_dict fp8_kv",
        )
        model_txt = _replace_once(
            model_txt,
            "        self.sliding = args.is_sliding(layer_idx)\n"
            "        self.head_dim = args.swa_head_dim if self.sliding else args.head_dim\n",
            "        self.sliding = args.is_sliding(layer_idx)\n"
            "        self.fp8_kv = args.uses_fp8_kv(layer_idx)\n"
            "        self.head_dim = args.swa_head_dim if self.sliding else args.head_dim\n",
            "model Inkling2Attention fp8_kv",
        )
        model_txt = _replace_once(
            model_txt,
            "    def sanitize(self, weights: dict) -> dict:\n",
            "    def set_fp8_kv(self, mode: bool | str = True) -> None:\n"
            "        self.args.text.fp8_kv = mode\n"
            "        for idx, layer in enumerate(self.model.layers):\n"
            "            layer.attn.fp8_kv = self.args.text.uses_fp8_kv(idx)\n\n"
            "    def sanitize(self, weights: dict) -> dict:\n",
            "model set_fp8_kv",
        )

    if "bounded_cache" in features:
        model_txt = _replace_once(
            model_txt,
            "from mlx_lm.models.cache import ArraysCache, CacheList, KVCache\n",
            "from mlx_lm.models.cache import ArraysCache, KVCache\n",
            "model cache import",
        )
        model_txt = _replace_once(
            model_txt,
            "from . import kernels\n",
            "from . import cache as windows\nfrom . import kernels\n",
            "model windows import",
        )

    if "fast_decode" in features:
        model_txt = _replace_once(
            model_txt,
            "from . import kernels\n",
            "from . import fast_decode as _fd\nfrom . import kernels\n",
            "model fast_decode import",
        )

    if "m1_qmv_moe" in features:
        model_txt = _replace_once(
            model_txt,
            "from . import kernels\n",
            "from . import kernels\nfrom ..m1_qmv_moe import eligible_inkling as _qmv_eligible, m1_qmv_moe_inkling as _qmv_moe\n",
            "model m1_qmv_moe import",
        )

    if "sort32" in features:
        model_txt = _replace_once(model_txt, "do_sort = inds.size >= 64", "do_sort = inds.size >= 32", "model sort32")

    if "affine4" in features or "attn4" in features:
        prev = "        return out\n\n    @property\n    def cast_predicate"
        extra = (_AFFINE4_CODE if "affine4" in features else "") + (_ATTN4_CODE if "attn4" in features else "")
        model_txt = _replace_once(model_txt, prev, extra + prev, "model affine4/attn4")

    if "prefill_fast" in features:
        model_txt = _replace_once(
            model_txt,
            "        self._gcap = mx.array(args.mlp_gate_softcap, dtype=mx.float32)\n"
            "        self._ucap = mx.array(args.mlp_up_softcap, dtype=mx.float32)\n",
            "        self._gcap = mx.array(args.mlp_gate_softcap, dtype=mx.float32)\n"
            "        self._ucap = mx.array(args.mlp_up_softcap, dtype=mx.float32)\n"
            "        self._inv_nvfp4 = mx.array([1.0 / _NVFP4_DENOMINATOR], dtype=mx.float32)\n"
            "        self._swiglu_params = mx.array(\n"
            "            [args.mlp_gate_softcap, args.mlp_up_softcap, 1.0 / _NVFP4_DENOMINATOR],\n"
            "            dtype=mx.float32,\n"
            "        )\n",
            "model prefill_fast experts init",
        )
        model_txt = _replace_once(
            model_txt,
            "        y = self.w13(xe, idx, do_sort)\n"
            "        s13 = self.w13.post_scale(idx)\n",
            "        y = self.w13(xe, idx, do_sort)\n"
            "        if mx.default_device() == mx.gpu:\n"
            '            g13 = self.w13.global_scale if self.w13.mode == "nvfp4" else None\n'
            "            act = kernels.expert_softcapped_swiglu(y, idx, g13, self._swiglu_params)\n"
            "            z = self.w2(act, idx, do_sort)\n"
            "            if do_sort:\n"
            "                z = _scatter_unsort(z, inv, inds.shape)\n"
            "            z = z.squeeze(-2)\n"
            '            g2 = self.w2.global_scale if self.w2.mode == "nvfp4" else None\n'
            "            return kernels.expert_weighted_combine(\n"
            "                z, inds, weights, g2, self._inv_nvfp4\n"
            "            )\n"
            "        s13 = self.w13.post_scale(idx)\n",
            "model prefill_fast experts call",
        )

    if "fast_decode" in features:
        model_txt = _replace_once(
            model_txt,
            "        self._gcap = mx.array(args.mlp_gate_softcap, dtype=mx.float32)\n"
            "        self._ucap = mx.array(args.mlp_up_softcap, dtype=mx.float32)\n",
            "        self._gcap = mx.array(args.mlp_gate_softcap, dtype=mx.float32)\n"
            "        self._ucap = mx.array(args.mlp_up_softcap, dtype=mx.float32)\n"
            "        self._gcap_f = float(args.mlp_gate_softcap)\n"
            "        self._ucap_f = float(args.mlp_up_softcap)\n",
            "model fast_decode experts init",
        )
        model_txt = _replace_once(
            model_txt,
            "    def __call__(self, x: mx.array, inds: mx.array, weights: mx.array) -> mx.array:\n"
            "        xe = mx.expand_dims(x, (-2, -3))\n"
            f"        do_sort = inds.size >= {sort_thresh}\n",
            "    def __call__(self, x: mx.array, inds: mx.array, weights: mx.array) -> mx.array:\n"
            '        if _fd.part("moe") and x.shape[0] <= 16:\n'
            "            return self._call_fused(x, inds, weights)\n"
            "        xe = mx.expand_dims(x, (-2, -3))\n"
            f"        do_sort = inds.size >= {sort_thresh}\n",
            "model fast_decode experts call",
        )
        model_txt = _replace_once(
            model_txt,
            "        out = mx.sum(z.astype(mx.float32) * weights[..., None], axis=-2)\n"
            "        return out.astype(x.dtype)\n\n\nclass Inkling2MoE",
            "        out = mx.sum(z.astype(mx.float32) * weights[..., None], axis=-2)\n"
            "        return out.astype(x.dtype)\n\n"
            "    def _fused_parts(self, x: mx.array, inds: mx.array, weights: mx.array):\n"
            "        rows, k = inds.shape\n"
            f"        do_sort = inds.size >= {sort_thresh}\n"
            "        xe = mx.expand_dims(x, (-2, -3))\n"
            "        idx = inds\n"
            "        inv = None\n"
            "        if do_sort:\n"
            "            xe, idx, inv = _gather_sort(xe, inds)\n"
            "            slots = (xe.shape[0], 1)\n"
            "            y = self.w13(xe, idx, True)\n"
            "            y = y.reshape(slots[0], 1, y.shape[-1])\n"
            "        else:\n"
            "            slots = (rows, k)\n"
            "            y = self.w13(xe, idx, False)\n"
            "            y = y.reshape(slots[0], slots[1], y.shape[-1])\n"
            "        s13 = self.w13.post_scale(idx)\n"
            "        s13 = mx.ones(slots, dtype=mx.float32) if s13 is None else s13.reshape(slots)\n"
            "        act = _fd.swiglu_scaled(y, s13, self._gcap_f, self._ucap_f)\n"
            "        if do_sort:\n"
            "            z = self.w2(act, idx, True)\n"
            "            z = z.reshape(slots[0], z.shape[-1])\n"
            "            z = _scatter_unsort(z, inv, inds.shape)\n"
            "        else:\n"
            "            z = self.w2(act[:, :, None, :], idx, False)\n"
            "            z = z.reshape(rows, k, z.shape[-1])\n"
            "        s2 = self.w2.post_scale(inds)\n"
            "        if s2 is None:\n"
            "            s2 = mx.ones((rows, k), dtype=mx.float32)\n"
            "        return z, s2, weights\n\n"
            "    def _call_fused(self, x: mx.array, inds: mx.array, weights: mx.array):\n"
            "        if (\n"
            "            x.shape[0] == 1\n"
            f"            and inds.size < {sort_thresh}\n"
            '            and _fd.part("nvfp4_moe")\n'
            "            and _fd.nvfp4_decode_eligible(self)\n"
            "        ):\n"
            "            return _fd.moe_decode_nvfp4(x, self, inds, weights)\n"
            "        return _fd.moe_combine(*self._fused_parts(x, inds, weights))\n\n\nclass Inkling2MoE",
            "model fast_decode _fused_parts",
        )
        if "fuse_moe_conv_res" in features:
            model_txt = _replace_once(
                model_txt,
                "    def __call__(self, x: mx.array) -> mx.array:\n"
                "        shape = x.shape\n"
                "        flat = x.reshape(-1, shape[-1])\n"
                "        inds, w = self.route(flat)\n"
                "        return self.experts(flat, inds, w).reshape(shape)\n",
                "    def __call__(self, x: mx.array) -> mx.array:\n"
                "        shape = x.shape\n"
                "        flat = x.reshape(-1, shape[-1])\n"
                "        inds, w = self.route(flat)\n"
                "        return self.experts(flat, inds, w).reshape(shape)\n"
                "\n    def forward_parts(self, x: mx.array):\n"
                "        flat = x.reshape(-1, x.shape[-1])\n"
                "        inds, w = self.route(flat)\n"
                "        return self.experts._fused_parts(flat, inds, w)\n",
                "model fuse_moe_conv_res forward_parts",
            )

    if "m1_moe" in features:
        model_txt = _replace_once(
            model_txt,
            "    def __call__(self, x: mx.array, inds: mx.array, weights: mx.array) -> mx.array:\n",
            "    def __call__(self, x: mx.array, inds: mx.array, weights: mx.array) -> mx.array:\n"
            "        if (\n"
            "            x.ndim == 2\n"
            "            and 1 <= x.shape[0] <= 8\n"
            "            and x.dtype == mx.bfloat16\n"
            "            and self.w13.mode == 'affine'\n"
            "            and self.w2.mode == 'affine'\n"
            "            and self.w13.bits in (3, 4)\n"
            "            and self.w2.bits in (3, 4)\n"
            "            and self.w13.group_size == 64\n"
            "            and self.w2.group_size == 64\n"
            "        ):\n"
            "            from inkling2_m1_kernels import grouped_inkling2_moe\n"
            "            return grouped_inkling2_moe(\n"
            "                x, inds, weights, self.w13, self.w2, 4.0, 25.0,\n"
            "            )\n",
            "model m1_moe",
        )

    if "subwarp_moe" in features:
        if "self._gcap_f" not in model_txt:
            model_txt = _replace_once(
                model_txt,
                "        self._gcap = mx.array(args.mlp_gate_softcap, dtype=mx.float32)\n"
                "        self._ucap = mx.array(args.mlp_up_softcap, dtype=mx.float32)\n",
                "        self._gcap = mx.array(args.mlp_gate_softcap, dtype=mx.float32)\n"
                "        self._ucap = mx.array(args.mlp_up_softcap, dtype=mx.float32)\n"
                "        self._gcap_f = float(args.mlp_gate_softcap)\n"
                "        self._ucap_f = float(args.mlp_up_softcap)\n",
                "model subwarp_moe init",
            )
        model_txt = _replace_once(
            model_txt,
            "    def __call__(self, x: mx.array, inds: mx.array, weights: mx.array) -> mx.array:\n",
            "    def __call__(self, x: mx.array, inds: mx.array, weights: mx.array) -> mx.array:\n"
            "        if x.shape[0] >= 2 and x.shape[0] <= 16 and self.w13.post_scale(inds) is None and int(self.w13.bits) in (3, 4):\n"
            "            from inkling2_m1_kernels import subwarp_inkling2_moe\n"
            "            return subwarp_inkling2_moe(x, inds, weights, self.w13, self.w2, self._gcap_f, self._ucap_f)\n",
            "model subwarp_moe",
        )

    if "m1_qmv_moe" in features:
        if "self._gcap_f" not in model_txt:
            model_txt = _replace_once(
                model_txt,
                "        self._gcap = mx.array(args.mlp_gate_softcap, dtype=mx.float32)\n"
                "        self._ucap = mx.array(args.mlp_up_softcap, dtype=mx.float32)\n",
                "        self._gcap = mx.array(args.mlp_gate_softcap, dtype=mx.float32)\n"
                "        self._ucap = mx.array(args.mlp_up_softcap, dtype=mx.float32)\n"
                "        self._gcap_f = float(args.mlp_gate_softcap)\n"
                "        self._ucap_f = float(args.mlp_up_softcap)\n",
                "model m1_qmv_moe init",
            )
        model_txt = _replace_once(
            model_txt,
            "    def __call__(self, x: mx.array, inds: mx.array, weights: mx.array) -> mx.array:\n",
            "    def __call__(self, x: mx.array, inds: mx.array, weights: mx.array) -> mx.array:\n"
            "        if _qmv_eligible(x, self):\n"
            '            if not getattr(Inkling2Experts, "_omlx_qmv_logged", False):\n'
            "                Inkling2Experts._omlx_qmv_logged = True\n"
            '                print("BENCH engaged m1_qmv_moe", flush=True)\n'
            "            return _qmv_moe(\n"
            "                x, inds, weights, self.w13, self.w2, gcap=self._gcap_f, ucap=self._ucap_f,\n"
            "            )\n",
            "model m1_qmv_moe",
        )

    if "prefill_pad64" in features:
        model_txt = _replace_once(
            model_txt,
            "    def __call__(self, x: mx.array, inds: mx.array, weights: mx.array) -> mx.array:\n",
            "    def __call__(self, x: mx.array, inds: mx.array, weights: mx.array) -> mx.array:\n"
            "        if 28 <= x.shape[0] < 64:\n"
            "            _n0 = x.shape[0]\n"
            "            _p = 64 - _n0\n"
            "            _xp = mx.concatenate([x, mx.zeros((_p, x.shape[1]), dtype=x.dtype)], axis=0)\n"
            "            _ip = mx.concatenate([inds, mx.zeros((_p, inds.shape[1]), dtype=inds.dtype)], axis=0)\n"
            "            _wp = mx.concatenate([weights, mx.zeros((_p, weights.shape[1]), dtype=weights.dtype)], axis=0)\n"
            "            return self._call_unpadded(_xp, _ip, _wp)[:_n0]\n"
            "        return self._call_unpadded(x, inds, weights)\n\n"
            "    def _call_unpadded(self, x: mx.array, inds: mx.array, weights: mx.array) -> mx.array:\n",
            "model prefill_pad64",
        )

    if "top6" in features:
        model_txt = _replace_once(
            model_txt,
            "        self.top_k = args.num_experts_per_tok\n",
            "        self.top_k = 6\n",
            "model top6",
        )

    if "bf16_ctrl" in features:
        old_cast = '                or key.endswith("loop_down")\n                or key.endswith("loop_gain_forget")\n'
        model_txt = _replace_once(model_txt, old_cast, "", "model bf16_ctrl cast")
        model_txt = _replace_once(
            model_txt,
            "    z = z.astype(mx.bfloat16).astype(mx.float32).reshape(2 * n, rows, hidden)\n"
            "    low = (z @ down.reshape(2 * n, hidden, -1)).reshape(n, 2, rows, -1).sum(axis=1)\n"
            "    low = low.astype(mx.bfloat16).astype(mx.float32)\n"
            "    gates = (low @ gain_forget).reshape(n, b, length, 2 * hidden)",
            "    wd = down.dtype\n"
            "    z = z.astype(mx.bfloat16).astype(wd).reshape(2 * n, rows, hidden)\n"
            "    low = (z @ down.reshape(2 * n, hidden, -1)).reshape(n, 2, rows, -1).sum(axis=1)\n"
            "    low = low.astype(mx.bfloat16).astype(gain_forget.dtype)\n"
            "    gates = (low @ gain_forget).astype(mx.float32).reshape(n, b, length, 2 * hidden)",
            "model bf16_ctrl mix",
        )

    if "fast_decode" in features:
        model_txt = _replace_once(
            model_txt,
            "    n, b, length, hidden = x.shape\n"
            "    rows = b * length\n",
            "    n, b, length, hidden = x.shape\n"
            "    rows = b * length\n"
            '    if _fd.part("msg") and rows <= 16:\n'
            "        return _messages_fused(x, message, down, gain_forget)\n",
            "model fast_decode _mix_messages",
        )
        model_txt = _replace_once(
            model_txt,
            "    return (1.0 - forget) * x + gain * message\n\n\ndef _static_conv",
            "    return (1.0 - forget) * x + gain * message\n\n\n"
            "def _messages_fused(x, message, down, gain_forget):\n"
            "    n, b, length, hidden = x.shape\n"
            "    rows = b * length\n"
            "    z = _fd.messages_prep(\n"
            "        x.reshape(n, rows, hidden), message.reshape(n, rows, hidden)\n"
            "    ).astype(down.dtype)\n"
            "    low = _fd.messages_low(\n"
            "        z.reshape(2 * n, rows, hidden) @ down.reshape(2 * n, hidden, -1), rows\n"
            "    ).astype(gain_forget.dtype)\n"
            "    gates = low.reshape(n, rows, -1) @ gain_forget\n"
            "    return _fd.messages_apply(\n"
            "        x.reshape(n, rows, hidden), message.reshape(n, rows, hidden), gates\n"
            "    ).reshape(n, b, length, hidden)\n\n\ndef _static_conv",
            "model fast_decode _messages_fused",
        )

    if "bf16_loop_ctrl" in features:
        model_txt = _replace_once(
            model_txt,
            "def _mix_messages(x, message, down, gain_forget):\n"
            '    """Loop controllers: ``(1 - tanh(forget)) * x + tanh(gain) * message``.\n\n'
            "    ``x`` and ``message`` are [N, B, L, H] FP32, one row per controlled input.\n"
            "    ``down`` is [N, 2, H, r] (the x and message halves) and ``gain_forget``\n"
            "    [N, r, 2H], both FP32 copies of BF16 weights.\n"
            '    """\n',
            "def _mix_messages(x, message, down, gain_forget, down_t=None, gf_t=None):\n"
            '    """Loop controllers: ``(1 - tanh(forget)) * x + tanh(gain) * message``.\n\n'
            "    ``x`` and ``message`` are [N, B, L, H] FP32, one row per controlled input.\n"
            "    ``down`` is [N, 2, H, r] (the x and message halves) and ``gain_forget``\n"
            "    [N, r, 2H], both FP32 copies of BF16 weights.\n"
            '    """\n'
            "    if mx.default_device() == mx.gpu:\n"
            "        return kernels.mix_messages_gpu(x, message, down, gain_forget, down_t, gf_t)\n",
            "model bf16_loop_ctrl _mix_messages",
        )

    if "prefill_fast" in features:
        model_txt = _replace_once(
            model_txt,
            "def _static_conv(x, w, state, scale):\n"
            '    """Depthwise causal conv (taps: ``w[:, K-1]`` is the current token).\n\n'
            "    Returns the output and the padded input window ``[state, x]``; its last\n"
            "    ``K - 1`` rows are the next state.\n"
            '    """\n'
            "    xf = x.astype(mx.float32)\n"
            "    k = w.shape[-1]\n"
            "    xp = mx.concatenate([state, xf], axis=1)\n"
            "    length = xf.shape[1]\n"
            "    out = w[:, 0] * xp[:, 0:length]\n"
            "    for j in range(1, k):\n"
            "        out = out + w[:, j] * xp[:, j : j + length]\n"
            "    # The reference rounds the depthwise conv to the weight dtype before the\n"
            "    # residual add.\n"
            "    out = (out.astype(w.dtype).astype(mx.float32) + xf) * scale\n"
            "    return out.astype(x.dtype), xp\n",
            "def _static_conv(x, w, state, scale, scale_arr=None):\n"
            '    """Depthwise causal conv (taps: ``w[:, K-1]`` is the current token)."""\n'
            "    k = w.shape[-1]\n"
            "    if k == 4 and mx.default_device() == mx.gpu:\n"
            "        sc = scale_arr if scale_arr is not None else mx.array([scale], dtype=mx.float32)\n"
            "        out, next_state = kernels.static_conv4_fused(x, w, state, sc)\n"
            "        return out, next_state, x\n"
            "    xf = x.astype(mx.float32)\n"
            "    xp = mx.concatenate([state, xf], axis=1)\n"
            "    length = xf.shape[1]\n"
            "    out = w[:, 0] * xp[:, 0:length]\n"
            "    for j in range(1, k):\n"
            "        out = out + w[:, j] * xp[:, j : j + length]\n"
            "    out = (out.astype(w.dtype).astype(mx.float32) + xf) * scale\n"
            "    return out.astype(x.dtype), xp, None\n",
            "model prefill_fast _static_conv",
        )
        model_txt = _replace_once(
            model_txt,
            "class Inkling2Attention(nn.Module):\n",
            "def _linear_slice(proj: nn.Module, x: mx.array, start: int, stop: int | None) -> mx.array:\n"
            '    """Apply rows ``[start:stop]`` of a Linear or QuantizedLinear without copying."""\n'
            "    if isinstance(proj, nn.QuantizedLinear):\n"
            '        biases = proj.get("biases")\n'
            "        y = mx.quantized_matmul(\n"
            "            x,\n"
            "            proj.weight[start:stop],\n"
            "            proj.scales[start:stop],\n"
            "            None if biases is None else biases[start:stop],\n"
            "            transpose=True,\n"
            "            group_size=proj.group_size,\n"
            "            bits=proj.bits,\n"
            '            mode=getattr(proj, "mode", "affine"),\n'
            "        )\n"
            "    else:\n"
            "        y = x @ proj.weight[start:stop].T\n"
            '    bias = proj.get("bias")\n'
            "    if bias is not None:\n"
            "        y = y + bias[start:stop]\n"
            "    return y\n\n\n"
            "class Inkling2Attention(nn.Module):\n",
            "model prefill_fast _linear_slice",
        )
        model_txt = _replace_once(
            model_txt,
            "        self._sconv_scale = 1.0 / math.sqrt(self.conv_k + 1)\n",
            "        self._sconv_scale = 1.0 / math.sqrt(self.conv_k + 1)\n"
            "        self._sconv_scale_arr = mx.array([self._sconv_scale], dtype=mx.float32)\n",
            "model prefill_fast sconv_scale_arr",
        )
        model_txt = _replace_once(
            model_txt,
            "        offset = kv.offset if kv is not None else 0\n"
            "        if isinstance(offset, mx.array):\n",
            "        offset = kv.offset if kv is not None else 0\n"
            "        if not isinstance(offset, mx.array) and int(offset) + length <= self.log_floor:\n"
            "            return None\n"
            "        if isinstance(offset, mx.array):\n",
            "model prefill_fast log_tau",
        )
        model_txt = _replace_once(
            model_txt,
            "            else:\n"
            "                # Looped passes give q, k and v their own inputs; one stacked\n"
            "                # projection serves all three.\n"
            "                pq, pk, pv = mx.split(\n"
            "                    self.in_proj(mx.concatenate([xq, xk, xv], axis=0)), 3, axis=0\n"
            "                )\n"
            "                q, _, _, r, gate, sink = mx.split(pq, self._cuts, axis=-1)\n"
            "                k = pk[..., self._cuts[0] : self._cuts[1]]\n"
            "                v = pv[..., self._cuts[1] : self._cuts[2]]\n",
            "            else:\n"
            "                c0, c1, c2, c3, c4 = self._cuts\n"
            "                q = _linear_slice(self.in_proj, xq, 0, c0)\n"
            "                k = _linear_slice(self.in_proj, xk, c0, c1)\n"
            "                v = _linear_slice(self.in_proj, xv, c1, c2)\n"
            "                rgs = _linear_slice(self.in_proj, xq, c2, None)\n"
            "                r, gate, sink = mx.split(rgs, [c3 - c2, c4 - c2], axis=-1)\n",
            "model prefill_fast project slice",
        )
        model_txt = _replace_once(
            model_txt,
            "            k, ks = _static_conv(k, self.k_conv, ks, self._sconv_scale)\n"
            "            v, vs = _static_conv(v, self.v_conv, vs, self._sconv_scale)\n",
            "            k, ks, rk = _static_conv(k, self.k_conv, ks, self._sconv_scale, self._sconv_scale_arr)\n"
            "            v, vs, rv = _static_conv(v, self.v_conv, vs, self._sconv_scale, self._sconv_scale_arr)\n",
            "model prefill_fast project sconv",
        )
        model_txt = _replace_once(
            model_txt,
            "                self._dconv_scale,\n"
            "            )\n"
            "        q = _l2_clip(\n",
            "                self._dconv_scale,\n"
            "            )\n"
            "            rk = rv = None\n"
            "        q = _l2_clip(\n",
            "model prefill_fast project rk_rv none",
        )
        model_txt = _replace_once(
            model_txt,
            "        return (\n"
            "            q.transpose(0, 2, 1, 3),\n"
            "            k.transpose(0, 2, 1, 3),\n"
            "            v.transpose(0, 2, 1, 3),\n"
            "            rel,\n"
            "            gate,\n"
            "            sink,\n"
            "            ks,\n"
            "            vs,\n"
            "        )\n",
            "        base_out = (\n"
            "            q.transpose(0, 2, 1, 3),\n"
            "            k.transpose(0, 2, 1, 3),\n"
            "            v.transpose(0, 2, 1, 3),\n"
            "            rel,\n"
            "            gate,\n"
            "            sink,\n"
            "            ks,\n"
            "            vs,\n"
            "        )\n"
            "        if rk is not None:\n"
            "            return (*base_out, rk, rv)\n"
            "        return base_out\n",
            "model prefill_fast project ret",
        )

    if "split_in_proj" in features:
        if "prefill_fast" in features:
            old_proj = (
                "            else:\n"
                "                c0, c1, c2, c3, c4 = self._cuts\n"
                "                q = _linear_slice(self.in_proj, xq, 0, c0)\n"
                "                k = _linear_slice(self.in_proj, xk, c0, c1)\n"
                "                v = _linear_slice(self.in_proj, xv, c1, c2)\n"
                "                rgs = _linear_slice(self.in_proj, xq, c2, None)\n"
                "                r, gate, sink = mx.split(rgs, [c3 - c2, c4 - c2], axis=-1)\n"
            )
        else:
            old_proj = (
                "            else:\n"
                "                # Looped passes give q, k and v their own inputs; one stacked\n"
                "                # projection serves all three.\n"
                "                pq, pk, pv = mx.split(\n"
                "                    self.in_proj(mx.concatenate([xq, xk, xv], axis=0)), 3, axis=0\n"
                "                )\n"
                "                q, _, _, r, gate, sink = mx.split(pq, self._cuts, axis=-1)\n"
                "                k = pk[..., self._cuts[0] : self._cuts[1]]\n"
                "                v = pv[..., self._cuts[1] : self._cuts[2]]\n"
            )
        inner = "".join("    " + ln + "\n" for ln in old_proj.splitlines()[1:])
        new_proj = (
            "            else:\n"
            '                _sp = self.__dict__.get("_m1_split_proj")\n'
            "                if _sp:\n"
            "                    qrg_lin, k_lin, v_lin, qrg_cuts = _sp\n"
            "                    q, r, gate, sink = mx.split(qrg_lin(xq), qrg_cuts, axis=-1)\n"
            "                    k = k_lin(xk)\n"
            "                    v = v_lin(xv)\n"
            "                else:\n"
            + inner
        )
        model_txt = _replace_once(model_txt, old_proj, new_proj, "model split_in_proj")

    if "fast_decode" in features:
        model_txt = _replace_once(
            model_txt,
            "        self._qk_clip = mx.array(args.qk_l2_clip_threshold, dtype=mx.float32)\n"
            "        self._r_clip = mx.array(args.rel_l2_clip_threshold, dtype=mx.float32)\n",
            "        self._qk_clip = mx.array(args.qk_l2_clip_threshold, dtype=mx.float32)\n"
            "        self._r_clip = mx.array(args.rel_l2_clip_threshold, dtype=mx.float32)\n"
            "        self._qk_clip_f = float(args.qk_l2_clip_threshold)\n"
            "        self._r_clip_f = float(args.rel_l2_clip_threshold)\n",
            "model fast_decode attn init",
        )
        model_txt = _replace_once(
            model_txt,
            "        b, length, _ = xq.shape\n        dkv = self.dkv\n",
            "        b, length, _ = xq.shape\n        dkv = self.dkv\n"
            '        fast = _fd.part("conv") and length <= _SHORT_WINDOW and b == 1\n',
            "model fast_decode project fast",
        )
        if "prefill_fast" in features:
            old_kv_sc = (
                "            k, ks, rk = _static_conv(k, self.k_conv, ks, self._sconv_scale, self._sconv_scale_arr)\n"
                "            v, vs, rv = _static_conv(v, self.v_conv, vs, self._sconv_scale, self._sconv_scale_arr)\n"
            )
            new_kv_sc = (
                "            if fast:\n"
                "                k, ks_t, ks_w = _fd.conv_short(k, self.k_conv, ks, self._sconv_scale)\n"
                "                v, vs_t, vs_w = _fd.conv_short(v, self.v_conv, vs, self._sconv_scale)\n"
                "                ks = ks_t if length == 1 else ks_w\n"
                "                vs = vs_t if length == 1 else vs_w\n"
                "                rk = rv = None\n"
                "            else:\n"
                "                k, ks, rk = _static_conv(k, self.k_conv, ks, self._sconv_scale, self._sconv_scale_arr)\n"
                "                v, vs, rv = _static_conv(v, self.v_conv, vs, self._sconv_scale, self._sconv_scale_arr)\n"
            )
        else:
            old_kv_sc = (
                "            k, ks = _static_conv(k, self.k_conv, ks, self._sconv_scale)\n"
                "            v, vs = _static_conv(v, self.v_conv, vs, self._sconv_scale)\n"
            )
            new_kv_sc = (
                "            if fast:\n"
                "                k, ks_t, ks_w = _fd.conv_short(k, self.k_conv, ks, self._sconv_scale)\n"
                "                v, vs_t, vs_w = _fd.conv_short(v, self.v_conv, vs, self._sconv_scale)\n"
                "                ks = ks_t if length == 1 else ks_w\n"
                "                vs = vs_t if length == 1 else vs_w\n"
                "            else:\n"
                "                k, ks = _static_conv(k, self.k_conv, ks, self._sconv_scale)\n"
                "                v, vs = _static_conv(v, self.v_conv, vs, self._sconv_scale)\n"
            )
        model_txt = _replace_once(model_txt, old_kv_sc, new_kv_sc, "model fast_decode project sconv")
        old_clips = (
            "        q = _l2_clip(\n"
            "            q.reshape(b, length, self.n_heads, self.head_dim),\n"
            "            self.q_scale,\n"
            "            self._qk_clip,\n"
            "        )\n"
            "        k = _l2_clip(\n"
            "            k.reshape(b, length, self.n_kv, self.head_dim), self._one, self._qk_clip\n"
            "        )\n"
            "        v = _unit_rms(v.reshape(b, length, self.n_kv, self.head_dim).astype(mx.float32))\n"
            "        v = v.astype(xq.dtype)\n"
            "        r = _l2_clip(\n"
            "            r.reshape(b, length, self.n_heads, self.d_rel), self.r_scale, self._r_clip\n"
            "        )\n"
        )
        new_clips = (
            '        if fast and _fd.part("clip"):\n'
            "            rows = b * length\n"
            "            q = _fd.l2_clip_heads(q.reshape(rows, self.n_heads, self.head_dim), self.q_scale, self._qk_clip_f).reshape(b, length, self.n_heads, self.head_dim)\n"
            "            k = _fd.l2_clip_heads(k.reshape(rows, self.n_kv, self.head_dim), self._one, self._qk_clip_f).reshape(b, length, self.n_kv, self.head_dim)\n"
            "            v = _fd.unit_rms_heads(v.reshape(rows, self.n_kv, self.head_dim)).reshape(b, length, self.n_kv, self.head_dim)\n"
            "            r = _fd.l2_clip_heads(r.reshape(rows, self.n_heads, self.d_rel), self.r_scale, self._r_clip_f).reshape(b, length, self.n_heads, self.d_rel)\n"
            "        else:\n"
            + "".join("    " + ln + "\n" for ln in old_clips.rstrip("\n").splitlines())
        )
        model_txt = _replace_once(model_txt, old_clips, new_clips, "model fast_decode clips")

    # Inkling2Attention.attend
    if "bounded_cache" in features:
        model_txt = _replace_once(
            model_txt,
            "    def attend(self, q, k, v, rel, sink, kv):\n"
            "        if kv is not None:\n"
            "            k, v = kv.update_and_fetch(k, v)\n"
            '        left_padding = getattr(kv, "left_padding", None) if kv is not None else None\n',
            "    def attend(self, q, k, v, rel, sink, kv, state=None):\n"
            "        left_padding = None\n"
            "        if kv is not None:\n"
            "            k, v = kv.update_and_fetch(k, v)\n"
            '            left_padding = getattr(kv, "left_padding", None)\n'
            "        elif state is not None:\n"
            "            k, v, left_padding = windows.append(state, k, v)\n",
            "model attend bounded_cache",
        )
    if "fp8_kv_full" in features:
        state_chk = (
            "            elif state is not None and state.cache[windows.KEYS] is not None:\n"
            "                use_fp8 = state.cache[windows.KEYS].dtype == mx.uint8\n"
            if "bounded_cache" in features
            else ""
        )
        old_att_head = (
            "    def attend(self, q, k, v, rel, sink, kv, state=None):\n"
            "        left_padding = None\n"
            if "bounded_cache" in features
            else "    def attend(self, q, k, v, rel, sink, kv):\n"
        )
        model_txt = _replace_once(
            model_txt,
            old_att_head,
            old_att_head
            + "        use_fp8 = self.fp8_kv\n"
            + "        if not use_fp8:\n"
            + '            if kv is not None and getattr(kv, "keys", None) is not None:\n'
            + "                use_fp8 = kv.keys.dtype == mx.uint8\n"
            + state_chk
            + "        if use_fp8:\n"
            + "            k = kernels.fp8_e4m3_quantize(k)\n"
            + "            v = kernels.fp8_e4m3_quantize(v)\n",
            "model attend fp8",
        )
    if "fast_decode" in features:
        model_txt = _replace_once(
            model_txt,
            "        return kernels.banded_sink_attention(\n",
            '        if _fd.part("mask") and left_padding is None and q.shape[0] == 1 and q.shape[2] == 1 and k.dtype != mx.uint8:\n'
            "            lo = max(0, k.shape[2] - self.window) if self.window else 0\n"
            "            if lo:\n"
            "                k = k[:, :, lo:, :]\n"
            "                v = v[:, :, lo:, :]\n"
            "            mask = _fd.decode_mask(rel, k.shape[2])\n"
            "            return mx.fast.scaled_dot_product_attention(\n"
            "                q, k, v, scale=self.scale, mask=mask, sinks=sink[0, 0].astype(q.dtype)\n"
            "            )\n"
            "        return kernels.banded_sink_attention(\n",
            "model attend fast_decode",
        )

    # _COMPILED_MAX_TOKENS constants
    cmax_extra = "_COMPILED_MAX_TOKENS = 16\n"
    if "fast_decode" in features:
        cmax_extra += "_SHORT_WINDOW = 16\n"
    if "verify_flush" in features:
        cmax_extra += "_FLUSH_LAYERS = 8\n"
    if cmax_extra != "_COMPILED_MAX_TOKENS = 16\n":
        model_txt = _replace_once(model_txt, "_COMPILED_MAX_TOKENS = 16\n", cmax_extra, "model constants")

    # Inkling2Layer.__init__ / _packed_loop / _pre / _post
    if "bf16_loop_ctrl" in features:
        model_txt = _replace_once(
            model_txt,
            "            self.loop_gain_forget = mx.zeros((4, rank, 2 * h), dtype=mx.float32)\n",
            "            self.loop_gain_forget = mx.zeros((4, rank, 2 * h), dtype=mx.float32)\n"
            "            self._loop_packed = None\n",
            "model bf16_loop_ctrl init",
        )
    if "prefill_fast" in features:
        model_txt = _replace_once(
            model_txt,
            "        self._conv_scale = 1.0 / math.sqrt(args.sconv_kernel_size + 1)\n",
            "        self._conv_scale = 1.0 / math.sqrt(args.sconv_kernel_size + 1)\n"
            "        self._conv_scale_arr = mx.array([self._conv_scale], dtype=mx.float32)\n",
            "model prefill_fast conv_scale_arr",
        )
    if "verify_compile" in features:
        old_cmp = (
            "        self._pre_compiled = mx.compile(lambda *a: Inkling2Layer._pre(layer, *a))\n"
            "        self._post_compiled = mx.compile(lambda *a: Inkling2Layer._post(layer, *a))\n"
        )
        new_cmp = (
            old_cmp
            + "        self._pre_verify_compiled = mx.compile(lambda *a: Inkling2Layer._pre(layer, *a))\n"
            + "        self._post_verify_compiled = mx.compile(lambda *a: Inkling2Layer._post(layer, *a))\n"
        )
        model_txt = _replace_once(model_txt, old_cmp, new_cmp, "model verify_compile")

    if "bf16_loop_ctrl" in features:
        model_txt = _replace_once(
            model_txt,
            "    def _pre(self, h, blocks, ks, vs, tau=None, messages=None):\n",
            "    def _packed_loop(self):\n"
            "        if mx.default_device() != mx.gpu:\n"
            "            return None, None\n"
            "        packed = self._loop_packed\n"
            "        key = (id(self.loop_down), id(self.loop_gain_forget))\n"
            "        if packed is not None and packed[0] == key:\n"
            "            return packed[1], packed[2]\n"
            "        down_t, gf_t = kernels.pack_loop_controller(\n"
            "            self.loop_down, self.loop_gain_forget\n"
            "        )\n"
            "        self._loop_packed = (key, down_t, gf_t)\n"
            "        return down_t, gf_t\n\n"
            "    def _pre(self, h, blocks, ks, vs, tau=None, messages=None, down_t=None, gf_t=None):\n",
            "model bf16_loop_ctrl _pre",
        )
        model_txt = _replace_once(
            model_txt,
            "                self.loop_down[:3],\n"
            "                self.loop_gain_forget[:3],\n"
            "            )\n",
            "                self.loop_down[:3],\n"
            "                self.loop_gain_forget[:3],\n"
            "                None if down_t is None else down_t[:3],\n"
            "                None if gf_t is None else gf_t[:3],\n"
            "            )\n",
            "model bf16_loop_ctrl _pre mix",
        )
        model_txt = _replace_once(
            model_txt,
            "    def _post(self, out, gate, h, mlp_blocks, s2, s3, messages=None):\n",
            "    def _post(self, out, gate, h, mlp_blocks, s2, s3, messages=None, down_t=None, gf_t=None):\n",
            "model bf16_loop_ctrl _post",
        )
        model_txt = _replace_once(
            model_txt,
            "                self.loop_down[3:],\n"
            "                self.loop_gain_forget[3:],\n"
            "            )[0]\n",
            "                self.loop_down[3:],\n"
            "                self.loop_gain_forget[3:],\n"
            "                None if down_t is None else down_t[3:],\n"
            "                None if gf_t is None else gf_t[3:],\n"
            "            )[0]\n",
            "model bf16_loop_ctrl _post mix",
        )

    if "prefill_fast" in features:
        model_txt = _replace_once(
            model_txt,
            "        a, s2 = _static_conv(a, self.attn_conv, s2, self._conv_scale)\n",
            "        a, s2, ra = _static_conv(a, self.attn_conv, s2, self._conv_scale, self._conv_scale_arr)\n",
            "model prefill_fast _post c1",
        )
        model_txt = _replace_once(
            model_txt,
            "        m = self.mlp(self.mlp_norm(y).astype(h.dtype))\n"
            "        m, s3 = _static_conv(m, self.mlp_conv, s3, self._conv_scale)\n"
            "        return (residual + m.astype(mx.float32)).astype(h.dtype), s2, s3\n",
            "        m = self.mlp(self.mlp_norm(y).astype(h.dtype))\n"
            "        m, s3, rm = _static_conv(m, self.mlp_conv, s3, self._conv_scale, self._conv_scale_arr)\n"
            "        out_h = (residual + m.astype(mx.float32)).astype(h.dtype)\n"
            "        if ra is not None or rm is not None:\n"
            "            return out_h, s2, s3, ra, rm\n"
            "        return out_h, s2, s3\n",
            "model prefill_fast _post c2",
        )

    if "fast_decode" in features:
        if "prefill_fast" in features:
            old_c1 = "        a, s2, ra = _static_conv(a, self.attn_conv, s2, self._conv_scale, self._conv_scale_arr)\n"
            new_c1 = (
                '        if _fd.part("conv") and a.shape[1] <= _SHORT_WINDOW:\n'
                "            a, s2t, s2w = _fd.conv_short(a, self.attn_conv, s2, self._conv_scale)\n"
                "            s2 = s2t if a.shape[1] == 1 else s2w\n"
                "            ra = None\n"
                "        else:\n"
                "            a, s2, ra = _static_conv(a, self.attn_conv, s2, self._conv_scale, self._conv_scale_arr)\n"
            )
            old_c2 = (
                "        m = self.mlp(self.mlp_norm(y).astype(h.dtype))\n"
                "        m, s3, rm = _static_conv(m, self.mlp_conv, s3, self._conv_scale, self._conv_scale_arr)\n"
            )
            new_c2 = (
                "        m = self.mlp(self.mlp_norm(y).astype(h.dtype))\n"
                '        if _fd.part("conv") and m.shape[1] <= _SHORT_WINDOW:\n'
                "            m, s3t, s3w = _fd.conv_short(m, self.mlp_conv, s3, self._conv_scale)\n"
                "            s3 = s3t if m.shape[1] == 1 else s3w\n"
                "            rm = None\n"
                "        else:\n"
                "            m, s3, rm = _static_conv(m, self.mlp_conv, s3, self._conv_scale, self._conv_scale_arr)\n"
            )
        else:
            old_c1 = "        a, s2 = _static_conv(a, self.attn_conv, s2, self._conv_scale)\n"
            new_c1 = (
                '        if _fd.part("conv") and a.shape[1] <= _SHORT_WINDOW:\n'
                "            a, s2t, s2w = _fd.conv_short(a, self.attn_conv, s2, self._conv_scale)\n"
                "            s2 = s2t if a.shape[1] == 1 else s2w\n"
                "        else:\n"
                "            a, s2 = _static_conv(a, self.attn_conv, s2, self._conv_scale)\n"
            )
            old_c2 = "        m = self.mlp(self.mlp_norm(y).astype(h.dtype))\n        m, s3 = _static_conv(m, self.mlp_conv, s3, self._conv_scale)\n"
            new_c2 = (
                "        m = self.mlp(self.mlp_norm(y).astype(h.dtype))\n"
                '        if _fd.part("conv") and m.shape[1] <= _SHORT_WINDOW:\n'
                "            m, s3t, s3w = _fd.conv_short(m, self.mlp_conv, s3, self._conv_scale)\n"
                "            s3 = s3t if m.shape[1] == 1 else s3w\n"
                "        else:\n"
                "            m, s3 = _static_conv(m, self.mlp_conv, s3, self._conv_scale)\n"
            )
        model_txt = _replace_once(model_txt, old_c1, new_c1, "model fast_decode _post c1")
        model_txt = _replace_once(model_txt, old_c2, new_c2, "model fast_decode _post c2")

        if "fuse_moe_conv_res" in features:
            old_mlp_blk = (
                "        m = self.mlp(self.mlp_norm(y).astype(h.dtype))\n"
                '        if _fd.part("conv") and m.shape[1] <= _SHORT_WINDOW:\n'
                "            m, s3t, s3w = _fd.conv_short(m, self.mlp_conv, s3, self._conv_scale)\n"
                "            s3 = s3t if m.shape[1] == 1 else s3w\n"
            )
            new_mlp_blk = (
                "        _mx = self.mlp_norm(y).astype(h.dtype)\n"
                "        if (\n"
                '            _fd.part("moe")\n'
                '            and _fd.part("conv")\n'
                "            and _mx.shape[0] == 1\n"
                "            and _mx.shape[1] <= _SHORT_WINDOW\n"
                "            and not (\n"
                "                _mx.shape[1] == 1\n"
                '                and _fd.part("nvfp4_moe")\n'
                "                and _fd.nvfp4_decode_eligible(self.mlp.experts)\n"
                "            )\n"
                "        ):\n"
                "            _z, _s2, _w = self.mlp.forward_parts(_mx)\n"
                "            _hout, _s3t, _s3w = _fd.moe_combine_conv_res(\n"
                "                _z, _s2, _w, self.mlp_conv, s3, self._conv_scale, residual\n"
                "            )\n"
                "            s3 = _s3t if _mx.shape[1] == 1 else _s3w\n"
                "            return _hout, s2, s3\n"
                "        m = self.mlp(_mx)\n"
                '        if _fd.part("conv") and m.shape[1] <= _SHORT_WINDOW:\n'
                "            m, s3t, s3w = _fd.conv_short(m, self.mlp_conv, s3, self._conv_scale)\n"
                "            s3 = s3t if m.shape[1] == 1 else s3w\n"
            )
            model_txt = _replace_once(model_txt, old_mlp_blk, new_mlp_blk, "model fuse_moe_conv_res")

    if "conv_opt" in features and "prefill_fast" not in features:
        model_txt = _replace_once(
            model_txt,
            "        return (*self.attn.project(xq, xk, xv, ks, vs, tau), h_n)\n",
            "        q, k, v, rel, gate, sink, ks, vs = self.attn.project(xq, xk, xv, ks, vs, tau)\n"
            "        _kkeep = self._conv_k - 1\n"
            "        return (q, k, v, rel, gate, sink, ks, vs, h_n, mx.contiguous(ks[:, -_kkeep:]), mx.contiguous(vs[:, -_kkeep:]))\n",
            "model conv_opt _pre",
        )
        model_txt = _replace_once(
            model_txt,
            "        return (residual + m.astype(mx.float32)).astype(h.dtype), s2, s3\n",
            "        _kkeep = self._conv_k - 1\n"
            "        return (residual + m.astype(mx.float32)).astype(h.dtype), s2, s3, mx.contiguous(s2[:, -_kkeep:]), mx.contiguous(s3[:, -_kkeep:])\n",
            "model conv_opt _post",
        )

    # Inkling2Layer.__call__ composed cleanly
    old_layer_call = (
        "    def __call__(self, h, blocks, cache=None, bank=None):\n"
        '        """``h``: partial block sum [B, L, H] bf16. ``blocks``: list of unit-RMS\n'
        "        FP32 completed blocks; extended in place at a block boundary.\n"
        '        ``bank``: the previous loop pass\'s ``(blocks, partial_n)`` messages."""\n'
        "        b, length, hidden = h.shape\n"
        "        kv = cache[0] if cache is not None else None\n"
        "        conv = cache[1] if cache is not None else None\n"
        "        dkv = self.attn.dkv\n"
        "        states = []\n"
        "        for slot, width in ((0, dkv), (1, dkv), (2, hidden), (3, hidden)):\n"
        "            s = conv[slot] if conv is not None else None\n"
        "            if s is None:\n"
        "                s = mx.zeros((b, self._conv_k - 1, width), dtype=mx.float32)\n"
        "            states.append(s)\n"
        "        tau = self.attn.log_tau(kv, length)\n"
        "        compiled = length <= _COMPILED_MAX_TOKENS\n"
        "        pre = self._pre_compiled if compiled else self._pre\n"
        "        post = self._post_compiled if compiled else self._post\n"
        "        messages = self.loop_message(*bank) if bank is not None else None\n"
        "        q, k, v, rel, gate, sink, ks, vs, h_n = pre(\n"
        "            h, blocks, states[0], states[1], tau, messages\n"
        "        )\n"
        "        out = self.attn.attend(q, k, v, rel, sink, kv)\n"
        "        if self.is_boundary:\n"
        "            blocks.append(h_n)\n"
        "        h, s2, s3 = post(out, gate, h, blocks, states[2], states[3], messages)\n"
        "        if conv is not None:\n"
        "            windows = (ks, vs, s2, s3)\n"
        "            if 1 < length <= _COMPILED_MAX_TOKENS:\n"
        "                # Speculative verify rollback re-cuts the states from these.\n"
        "                conv._omlx_verify_xp = dict(enumerate(windows))\n"
        "            keep = self._conv_k - 1\n"
        "            for slot, window in enumerate(windows):\n"
        "                # A small copy, so the state does not pin the chunk's window.\n"
        "                conv[slot] = window[:, -keep:] * 1.0\n"
        "        return h\n"
    )
    model_txt = _replace_once(model_txt, old_layer_call, _build_layer_call(features), "model Inkling2Layer.__call__")

    if "bounded_cache" in features:
        model_txt = _replace_once(
            model_txt,
            "    @property\n    def num_visits(self) -> int:\n        if self._core_start is None:\n            # The prefill helper keeps only the first layers of a plain model.\n            return len(self.layers)\n        return len(self._schedule)\n",
            "    @property\n    def num_visits(self) -> int:\n        if self._core_start is None:\n            return len(self.layers)\n        return len(self._schedule)\n\n"
            "    def cache_layout(self) -> list[tuple[int | None, int]]:\n"
            "        layout, entry = [], 0\n"
            "        for idx, _ in self._schedule[: self.num_visits]:\n"
            "            if self.layers[idx].sliding:\n"
            "                layout.append((None, entry))\n"
            "                entry += 1\n"
            "            else:\n"
            "                layout.append((entry, entry + 1))\n"
            "                entry += 2\n"
            "        return layout\n\n"
            "    def visit_caches(self, cache: list) -> list[tuple]:\n"
            "        layout = self.cache_layout()\n"
            "        entries = layout[-1][1] + 1 if layout else 0\n"
            "        if len(cache) != entries:\n"
            "            raise ValueError(\n"
            '                f"Inkling 2 needs {entries} cache entries for its "\n'
            '                f"{len(layout)} layer visits, got {len(cache)}"\n'
            "            )\n"
            "        window = self.args.sliding_window_size\n"
            "        return [\n"
            "            (None, windows.ArraysCache.adopt(cache[state], window))\n"
            "            if kv is None\n"
            "            else (cache[kv], cache[state])\n"
            "            for kv, state in layout\n"
            "        ]\n",
            "model cache_layout/visit_caches",
        )
        model_txt = _replace_once(
            model_txt,
            "        if cache is None:\n"
            "            cache = [None] * self.num_visits\n"
            "        elif len(cache) != self.num_visits:\n"
            "            # A per-layer cache would silently skip the later loop passes.\n"
            "            raise ValueError(\n"
            '                f"Inkling 2 needs one cache entry per layer visit "\n'
            '                f"({self.num_visits}), got {len(cache)}"\n'
            "            )\n"
            "        blocks: list = []\n"
            "        prelude = bank = None\n"
            "        for visit, ((idx, loop_pass), c) in enumerate(zip(self._schedule, cache)):\n",
            "        visits = [None] * self.num_visits if cache is None else self.visit_caches(cache)\n"
            "        blocks: list = []\n"
            "        prelude = bank = None\n"
            "        for visit, ((idx, loop_pass), c) in enumerate(zip(self._schedule, visits)):\n",
            "model Inkling2TextModel.__call__ visits",
        )

    if "verify_flush" in features:
        model_txt = _replace_once(
            model_txt,
            "            h = self.layers[idx](h, blocks, cache=c, bank=bank if loop_pass else None)\n",
            "            h = self.layers[idx](h, blocks, cache=c, bank=bank if loop_pass else None)\n"
            "            if (\n"
            "                h.shape[1] <= _COMPILED_MAX_TOKENS\n"
            "                and (visit + 1) % _FLUSH_LAYERS == 0\n"
            "                and visit + 1 < len(self._schedule)\n"
            "            ):\n"
            "                mx.async_eval(h, *blocks)\n",
            "model verify_flush",
        )

    if "skip_lm_head" in features:
        model_txt = _replace_once(
            model_txt,
            "        if not capture_layer_ids and not return_hidden:\n"
            "            return self.logits_from_hidden(self.model(inputs, cache, input_embeddings))\n"
            "        capture = {int(i): None for i in capture_layer_ids or ()}\n"
            "        h = self.model(inputs, cache, input_embeddings, capture=capture)\n"
            "        logits = self.logits_from_hidden(h)\n",
            '        _skip = bool(kwargs.get("skip_logits") or kwargs.get("skip_lm_head"))\n'
            "        _b = inputs.shape[0] if inputs is not None else input_embeddings.shape[0]\n"
            "        if not capture_layer_ids and not return_hidden:\n"
            "            _h = self.model(inputs, cache, input_embeddings)\n"
            "            return mx.zeros((_b, 1, 1), dtype=_h.dtype) if _skip else self.logits_from_hidden(_h)\n"
            "        capture = {int(i): None for i in capture_layer_ids or ()}\n"
            "        h = self.model(inputs, cache, input_embeddings, capture=capture)\n"
            "        logits = mx.zeros((_b, 1, 1), dtype=h.dtype) if _skip else self.logits_from_hidden(h)\n",
            "model skip_lm_head",
        )

    if "bounded_cache" in features:
        model_txt = _replace_once(
            model_txt,
            "class Model(nn.Module):\n",
            "class Model(nn.Module):\n    dense_kv_attention = True\n\n",
            "model dense_kv_attention",
        )
        old_rb = (
            "        for entry in cache:\n"
            "            kv, conv = entry[0], entry[1]\n"
            '            windows = getattr(conv, "_omlx_verify_xp", None)\n'
            "            if windows is None or kv.trim(rejected) != rejected:\n"
            "                return False\n"
            "            for slot, window in windows.items():\n"
            "                conv[slot] = window[:, keep : keep + width] * 1.0\n"
            "            conv._omlx_verify_xp = None\n"
        )
        new_rb = (
            "        for kv, conv in self.model.visit_caches(cache):\n"
            '            padded = getattr(conv, "_omlx_verify_xp", None)\n'
            "            if padded is None:\n"
            "                return False\n"
            "            if kv is None:\n"
            "                if not windows.trim(conv, rejected):\n"
            "                    return False\n"
            "            elif kv.trim(rejected) != rejected:\n"
            "                return False\n"
            "            base = windows.CONV if kv is None else 0\n"
            "            for slot, xp in padded.items():\n"
            "                if isinstance(xp, tuple):\n"
            "                    prev_st, raw_x = xp\n"
            "                    if keep >= width:\n"
            "                        conv[base + slot] = raw_x[:, keep - width : keep].astype(mx.float32)\n"
            "                    else:\n"
            "                        conv[base + slot] = mx.concatenate(\n"
            "                            [prev_st[:, keep:], raw_x[:, :keep].astype(mx.float32)], axis=1\n"
            "                        )\n"
            "                else:\n"
            "                    conv[base + slot] = xp[:, keep : keep + width] * 1.0\n"
            "            conv._omlx_verify_xp = None\n"
        )
        model_txt = _replace_once(model_txt, old_rb, new_rb, "model rollback bounded_cache")
        old_mc = (
            "    def make_cache(self):\n"
            "        # One entry per layer visit: each loop pass keeps its own KV and conv\n"
            "        # states.\n"
            "        return [\n"
            "            CacheList(KVCache(), ArraysCache(4)) for _ in range(self.model.num_visits)\n"
            "        ]\n"
        )
        new_mc = (
            "    def make_cache(self):\n"
            "        window = self.args.text.sliding_window_size\n"
            "        cache = []\n"
            "        for kv, _ in self.model.cache_layout():\n"
            "            if kv is None:\n"
            "                cache.append(windows.ArraysCache(window=window))\n"
            "            else:\n"
            "                cache += [KVCache(), ArraysCache(4)]\n"
            "        return cache\n\n"
            "    def restore_cache(self, cache: list) -> list:\n"
            "        self.model.visit_caches(cache)\n"
            "        return cache\n"
        )
        model_txt = _replace_once(model_txt, old_mc, new_mc, "model make_cache/restore_cache")
    elif "prefill_fast" in features:
        old_rb = (
            "            for slot, window in windows.items():\n"
            "                conv[slot] = window[:, keep : keep + width] * 1.0\n"
        )
        new_rb = (
            "            for slot, window in windows.items():\n"
            "                if isinstance(window, tuple):\n"
            "                    prev_st, raw_x = window\n"
            "                    if keep >= width:\n"
            "                        conv[slot] = raw_x[:, keep - width : keep].astype(mx.float32)\n"
            "                    else:\n"
            "                        conv[slot] = mx.concatenate(\n"
            "                            [prev_st[:, keep:], raw_x[:, :keep].astype(mx.float32)], axis=1\n"
            "                        )\n"
            "                else:\n"
            "                    conv[slot] = window[:, keep : keep + width] * 1.0\n"
        )
        model_txt = _replace_once(model_txt, old_rb, new_rb, "model rollback prefill_fast")

    model_txt += f'\nprint("BENCH loaded model features={sorted(features)}", flush=True)\n'
    return model_txt


def _patch_batch_generator(bg_txt: str, features: set[str]) -> str:
    if "caps" in features or "caps200" in features:
        if "caps" in features:
            bg_txt = _replace_once(
                bg_txt,
                "_SPEC_BUFFER_CAPS = (200, 512)",
                "_SPEC_BUFFER_CAPS = (400, 4096)",
                "bg caps",
            )
        bg_txt = _replace_once(
            bg_txt,
            '"qwen4_exp_text",\n    }',
            '"qwen4_exp_text",\n        "inkling2",\n        "inkling2_loop",\n    }',
            "bg caps model_types",
        )
        bg_txt = _replace_once(
            bg_txt,
            'return architecture.startswith("applegpu_g15")',
            'return architecture.startswith("applegpu_g15") or architecture.startswith("applegpu_g13")',
            "bg caps arch",
        )
        bg_txt += '\nprint("BENCH loaded M1 spec buffer caps", flush=True)\n'
    if "fast_greedy" in features:
        old_lp = "    combined_lp = rows - mx.logsumexp(rows, axis=-1, keepdims=True)  # (k+1, V)\n"
        new_lp = (
            '    _want_lp = bool(getattr(gen_batch, "logprobs", None))\n'
            "    if k > 0 and (is_greedy or follower) and verify_procs is None and not _want_lp:\n"
            "        combined_lp = rows\n"
            "    else:\n"
            "        combined_lp = rows - mx.logsumexp(rows, axis=-1, keepdims=True)  # (k+1, V)\n"
        )
        bg_txt = _replace_once(bg_txt, old_lp, new_lp, "bg fast_greedy")
        bg_txt += '\nprint("BENCH loaded fast_greedy", flush=True)\n'
    if "margin30" in features:
        old_gr = (
            "        if greedy_result is None:\n"
            "            targets = _greedy_targets(combined_lp)  # (k+1,)\n"
            "            matches = (targets[:k] == state.drafts.astype(mx.int32)).astype(mx.int32)\n"
            "            m_arr = mx.cumprod(matches).sum().reshape(1)\n"
            "            host_arr = mx.concatenate([m_arr, targets, state.drafts.astype(mx.int32)])\n"
        )
        new_gr = (
            "        if greedy_result is None:\n"
            "            targets = _greedy_targets(combined_lp)  # (k+1,)\n"
            "            drafts_i32 = state.drafts.astype(mx.int32)\n"
            "            max_l = mx.take_along_axis(combined_lp[:k], targets[:k, None], axis=-1).squeeze(-1).astype(mx.float32)\n"
            "            dft_l = mx.take_along_axis(combined_lp[:k], drafts_i32[:, None], axis=-1).squeeze(-1).astype(mx.float32)\n"
            "            matches = ((targets[:k] == drafts_i32) | (dft_l >= (max_l - 0.30))).astype(mx.int32)\n"
            "            m_arr = mx.cumprod(matches).sum().reshape(1)\n"
            "            host_arr = mx.concatenate([m_arr, targets, drafts_i32])\n"
        )
        bg_txt = _replace_once(bg_txt, old_gr, new_gr, "bg margin30")
        bg_txt += '\nprint("BENCH loaded margin30", flush=True)\n'
    if "early_first_emit" in features:
        bg_txt = patch_batch_generator_early_emit(bg_txt)
        bg_txt += '\nprint("BENCH loaded early_first_emit", flush=True)\n'
    if "suffix_policy" in features:
        bg_txt = patch_batch_generator_suffix_policy(bg_txt)
        bg_txt += '\nprint("BENCH loaded suffix_policy", flush=True)\n'
    return bg_txt


def _patch_dspark_and_selection(dsp_txt: str, sel_txt: str, features: set[str]) -> tuple[str, str]:
    if "dspark_2bit" in features:
        dsp_txt = patch_dspark(dsp_txt)
        sel_txt = patch_selection_dspark_2bit(sel_txt)
        dsp_txt += '\nprint("BENCH loaded dspark_2bit", flush=True)\n'

    if "draft4" in features:
        old_dq = '_DRAFT_QUANT = {"group_size": 64, "bits": 8}'
        new_dq = '_DRAFT_QUANT = {"group_size": 64, "bits": 4}'
        dsp_txt = _replace_once(dsp_txt, old_dq, new_dq, "dspark draft4")
        dsp_txt += '\nprint("BENCH loaded draft4", flush=True)\n'

    if "fuse_draft_qkv" in features:
        old_pctx = (
            "    def project_context_kv(self, context: mx.array):\n"
            "        mixed = mix_rmsnorm(context, self._alpha, self._mix_weight, self._eps)\n"
            "        return self.k_proj_ctx(mixed), self.v_proj_ctx(mixed)\n"
        )
        new_pctx = (
            "    def project_context_kv(self, context: mx.array):\n"
            "        mixed = mix_rmsnorm(context, self._alpha, self._mix_weight, self._eps)\n"
            '        kvp = self.__dict__.get("_kv_ctx_proj")\n'
            "        if kvp is not None:\n"
            "            kv = kvp(mixed)\n"
            "            dkv = self.n_kv_heads * self.head_dim\n"
            "            return kv[..., :dkv], kv[..., dkv:]\n"
            "        return self.k_proj_ctx(mixed), self.v_proj_ctx(mixed)\n"
        )
        dsp_txt = _replace_once(dsp_txt, old_pctx, new_pctx, "dspark fuse_draft_qkv ctx")

        old_dmlp = (
            "    def __call__(self, x):\n"
            "        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))\n"
        )
        new_dmlp = (
            "    def __call__(self, x):\n"
            '        gup = self.__dict__.get("_gate_up_proj")\n'
            "        if gup is not None:\n"
            "            gu = gup(x)\n"
            "            half = gu.shape[-1] // 2\n"
            "            return self.down_proj(nn.silu(gu[..., :half]) * gu[..., half:])\n"
            "        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))\n"
        )
        dsp_txt = _replace_once(dsp_txt, old_dmlp, new_dmlp, "dspark fuse_draft_qkv mlp")

        old_load_ret = "    mx.eval(model.parameters())\n    return model\n"
        new_load_ret = (
            "    mx.eval(model.parameters())\n"
            "    def _concat_qlinears(lins):\n"
            "        l0 = lins[0]\n"
            "        in_d = int(l0.weight.shape[1]) * 32 // int(l0.bits)\n"
            "        out_d = sum(int(l.weight.shape[0]) for l in lins)\n"
            "        q = nn.QuantizedLinear(in_d, out_d, bias=False, group_size=int(l0.group_size), bits=int(l0.bits), mode=l0.mode)\n"
            "        q.weight = mx.concatenate([l.weight for l in lins], axis=0)\n"
            "        q.scales = mx.concatenate([l.scales for l in lins], axis=0)\n"
            "        q.biases = mx.concatenate([l.biases for l in lins], axis=0)\n"
            "        mx.eval(q.weight, q.scales, q.biases)\n"
            "        return q\n"
            "    for lyr in model.layers:\n"
            "        sa = lyr.self_attn\n"
            '        object.__setattr__(sa, "_qkv_proj", _concat_qlinears([sa.q_proj, sa.k_proj, sa.v_proj]))\n'
            '        object.__setattr__(sa, "_kv_ctx_proj", _concat_qlinears([sa.k_proj_ctx, sa.v_proj_ctx]))\n'
            "        mlp = lyr.mlp\n"
            '        object.__setattr__(mlp, "_gate_up_proj", _concat_qlinears([mlp.gate_proj, mlp.up_proj]))\n'
            "    from ...speculative.dflash import variants as _df_variants\n"
            '    if not getattr(_df_variants.project_qkv, "_m1_qkv_patched", False):\n'
            "        _orig_pqkv = _df_variants.project_qkv\n"
            "        def _fused_pqkv(attn, x, context):\n"
            '            qkvp = attn.__dict__.get("_qkv_proj")\n'
            "            if qkvp is not None:\n"
            "                qkv = qkvp(x)\n"
            "                dq = attn.n_heads * attn.head_dim\n"
            "                dkv = attn.n_kv_heads * attn.head_dim\n"
            "                return (qkv[..., :dq], *attn.project_context_kv(context), qkv[..., dq : dq + dkv], qkv[..., dq + dkv :])\n"
            "            return _orig_pqkv(attn, x, context)\n"
            "        _fused_pqkv._m1_qkv_patched = True\n"
            "        _df_variants.project_qkv = _fused_pqkv\n"
            "    return model\n"
        )
        dsp_txt = _replace_once(dsp_txt, old_load_ret, new_load_ret, "dspark fuse_draft_qkv load")
        dsp_txt += '\nprint("BENCH loaded fuse_draft_qkv", flush=True)\n'

    if "markov_topk" in features:
        old_mh_init = "        self._softcap = config.final_logit_softcapping\n"
        new_mh_init = (
            "        self._softcap = config.final_logit_softcapping\n"
            "        self._unpadded = int(config.unpadded_vocab_size or config.vocab_size)\n"
        )
        dsp_txt = _replace_once(dsp_txt, old_mh_init, new_mh_init, "dspark markov_topk init")
        old_mh_end = (
            "        if self._vocab_bias is not None:\n"
            "            logits = logits + self._vocab_bias\n"
            "        return logits\n"
        )
        new_mh_end = (
            old_mh_end
            + "\n"
            + "    def apply_step_greedy_topk(self, base_logits: mx.array, prev_tokens: mx.array, topk: int = 256) -> mx.array:\n"
            + "        state = self._state\n"
            + "        dtype = state.dtype\n"
            + "        inputs = mx.concatenate(\n"
            + "            [\n"
            + "                state,\n"
            + "                self.markov_w1(prev_tokens.reshape(-1)).astype(dtype),\n"
            + "                self._hidden[:, self._step].astype(dtype),\n"
            + "                self._rows_f.astype(dtype),\n"
            + "            ],\n"
            + "            axis=-1,\n"
            + "        )\n"
            + "        proj = self.rnn_proj(inputs).astype(mx.float32)\n"
            + "        r = self.rank\n"
            + "        gate = mx.sigmoid(proj[:, :r]).astype(dtype)\n"
            + "        kept = (gate * state).astype(dtype)\n"
            + "        blend = ((1.0 - gate) * mx.tanh(proj[:, r : 2 * r]).astype(dtype)).astype(dtype)\n"
            + "        self._state = kept + blend\n"
            + "        latent = mx.tanh(proj[:, 2 * r :]).astype(dtype)\n"
            + "        self._step += 1\n"
            + "        unpadded = self._unpadded\n"
            + "        cand = base_logits[:, :unpadded]\n"
            + "        klim = min(int(topk), unpadded)\n"
            + "        top_ids = mx.argpartition(-cand, kth=klim - 1, axis=-1)[:, :klim]\n"
            + "        fw = self._fold_weight[top_ids]\n"
            + "        fb = self._fold_bias[top_ids]\n"
            + "        bias = mx.sum(fw.astype(mx.float32) * latent.astype(mx.float32)[:, None, :], axis=-1) + fb.astype(mx.float32)\n"
            + "        scores = mx.take_along_axis(cand, top_ids, axis=-1).astype(mx.float32) + bias\n"
            + "        best = mx.argmax(scores, axis=-1, keepdims=True)\n"
            + "        return mx.take_along_axis(top_ids, best, axis=-1).squeeze(-1)\n"
        )
        dsp_txt = _replace_once(dsp_txt, old_mh_end, new_mh_end, "dspark markov_topk method")
        dsp_txt += '\nprint("BENCH loaded markov_topk=256", flush=True)\n'
        old_sel = (
            "    for step in range(logits.shape[1]):\n"
            "        step_logits = markov.apply_step_logits(logits[:, step], prev)\n"
            "        if all(sampler is None for sampler in samplers):\n"
            "            prev = mx.argmax(step_logits, axis=-1)\n"
            "        else:\n"
        )
        new_sel = (
            "    for step in range(logits.shape[1]):\n"
            '        if all(sampler is None for sampler in samplers) and hasattr(markov, "apply_step_greedy_topk"):\n'
            "            prev = markov.apply_step_greedy_topk(logits[:, step], prev, 256)\n"
            "        elif all(sampler is None for sampler in samplers):\n"
            "            prev = mx.argmax(markov.apply_step_logits(logits[:, step], prev), axis=-1)\n"
            "        else:\n"
            "            step_logits = markov.apply_step_logits(logits[:, step], prev)\n"
        )
        sel_txt = _replace_once(sel_txt, old_sel, new_sel, "selection markov_topk")

    if "loom_draft_head" in features:
        old_mh_init = "        self._fold_weight = None\n        self._fold_bias = None\n"
        new_mh_init = (
            "        self._fold_weight = None\n"
            "        self._fold_bias = None\n"
            "        self._unpadded = int(config.unpadded_vocab_size or config.vocab_size)\n"
            "        self._fw_q = None\n"
            "        self._fw_s = None\n"
            "        self._fw_b = None\n"
            "        self._fb_unp = None\n"
        )
        dsp_txt = _replace_once(dsp_txt, old_mh_init, new_mh_init, "dspark loom init")
        old_fold_end = (
            "        self._fold_weight = mx.concatenate(rows, axis=0)\n"
            "        self._fold_bias = mx.concatenate(biases, axis=0)\n"
            "        mx.eval(self._fold_weight, self._fold_bias)\n"
        )
        new_fold_end = (
            old_fold_end
            + "        fw_unp = self._fold_weight[: self._unpadded]\n"
            + '        self._fw_q, self._fw_s, self._fw_b = mx.quantize(\n'
            + '            fw_unp, group_size=64, bits=4, mode="affine"\n'
            + "        )\n"
            + "        self._fb_unp = self._fold_bias[: self._unpadded].astype(mx.float32)\n"
            + "        mx.eval(self._fw_q, self._fw_s, self._fw_b, self._fb_unp)\n"
        )
        dsp_txt = _replace_once(dsp_txt, old_fold_end, new_fold_end, "dspark loom fold")
        old_mh_end = (
            "        if self._vocab_bias is not None:\n"
            "            logits = logits + self._vocab_bias\n"
            "        return logits\n"
        )
        new_mh_end = (
            old_mh_end
            + "\n"
            + "    def apply_step_greedy(self, base_logits: mx.array, prev_tokens: mx.array) -> mx.array:\n"
            + "        state = self._state\n"
            + "        dtype = state.dtype\n"
            + "        inputs = mx.concatenate(\n"
            + "            [\n"
            + "                state,\n"
            + "                self.markov_w1(prev_tokens.reshape(-1)).astype(dtype),\n"
            + "                self._hidden[:, self._step].astype(dtype),\n"
            + "                self._rows_f.astype(dtype),\n"
            + "            ],\n"
            + "            axis=-1,\n"
            + "        )\n"
            + "        proj = self.rnn_proj(inputs).astype(mx.float32)\n"
            + "        r = self.rank\n"
            + "        gate = mx.sigmoid(proj[:, :r]).astype(dtype)\n"
            + "        kept = (gate * state).astype(dtype)\n"
            + "        blend = ((1.0 - gate) * mx.tanh(proj[:, r : 2 * r]).astype(dtype)).astype(dtype)\n"
            + "        self._state = kept + blend\n"
            + "        latent = mx.tanh(proj[:, 2 * r :]).astype(dtype)\n"
            + "        bias = (\n"
            + "            mx.quantized_matmul(\n"
            + "                latent,\n"
            + "                self._fw_q,\n"
            + "                self._fw_s,\n"
            + "                self._fw_b,\n"
            + "                transpose=True,\n"
            + "                group_size=64,\n"
            + "                bits=4,\n"
            + '                mode="affine",\n'
            + "            ).astype(mx.float32)\n"
            + "            + self._fb_unp\n"
            + "        )\n"
            + "        self._step += 1\n"
            + "        scores = base_logits[:, : self._unpadded].astype(mx.float32) + bias\n"
            + "        return mx.argmax(scores, axis=-1)\n"
        )
        dsp_txt = _replace_once(dsp_txt, old_mh_end, new_mh_end, "dspark loom greedy")
        old_bind_head = (
            "        lm_head = target.lm_head\n"
            "        self._lm_head_fn = lambda hidden: lm_head(hidden)\n"
        )
        new_bind_head = (
            "        lm_head = target.lm_head\n"
            "        if isinstance(lm_head, nn.QuantizedLinear) and lm_head.bits > 4:\n"
            "            wq, ws, wb = [], [], []\n"
            "            for start in range(0, lm_head.weight.shape[0], 16384):\n"
            "                w_chunk = mx.dequantize(\n"
            "                    lm_head.weight[start : start + 16384],\n"
            "                    lm_head.scales[start : start + 16384],\n"
            '                    lm_head.get("biases")[start : start + 16384] if lm_head.get("biases") is not None else None,\n'
            "                    lm_head.group_size,\n"
            "                    lm_head.bits,\n"
            '                    mode=getattr(lm_head, "mode", "affine"),\n'
            "                ).astype(mx.bfloat16)\n"
            '                q4, s4, b4 = mx.quantize(w_chunk, group_size=64, bits=4, mode="affine")\n'
            "                mx.eval(q4, s4, b4)\n"
            "                wq.append(q4)\n"
            "                ws.append(s4)\n"
            "                wb.append(b4)\n"
            "            _w4 = mx.concatenate(wq, axis=0)\n"
            "            _s4 = mx.concatenate(ws, axis=0)\n"
            "            _b4 = mx.concatenate(wb, axis=0)\n"
            "            mx.eval(_w4, _s4, _b4)\n"
            "            mx.clear_cache()\n"
            "            self._lm_head_4bit = (_w4, _s4, _b4)\n"
            "            self._lm_head_fn = lambda hidden, w=_w4, s=_s4, b=_b4: mx.quantized_matmul(\n"
            '                hidden.astype(s.dtype), w, s, b, transpose=True, group_size=64, bits=4, mode="affine"\n'
            "            )\n"
            '            print("BENCH loaded 4-bit Loom-style draft proposal head + Markov matrix", flush=True)\n'
            "        else:\n"
            "            self._lm_head_fn = lambda hidden: lm_head(hidden)\n"
        )
        dsp_txt = _replace_once(dsp_txt, old_bind_head, new_bind_head, "dspark loom bind")
        old_sel = (
            "    for step in range(logits.shape[1]):\n"
            "        step_logits = markov.apply_step_logits(logits[:, step], prev)\n"
            "        if all(sampler is None for sampler in samplers):\n"
            "            prev = mx.argmax(step_logits, axis=-1)\n"
            "        else:\n"
        )
        new_sel = (
            "    for step in range(logits.shape[1]):\n"
            '        if all(sampler is None for sampler in samplers) and hasattr(markov, "apply_step_greedy"):\n'
            "            prev = markov.apply_step_greedy(logits[:, step], prev)\n"
            "        elif all(sampler is None for sampler in samplers):\n"
            "            prev = mx.argmax(markov.apply_step_logits(logits[:, step], prev), axis=-1)\n"
            "        else:\n"
            "            step_logits = markov.apply_step_logits(logits[:, step], prev)\n"
        )
        sel_txt = _replace_once(sel_txt, old_sel, new_sel, "selection loom")

    if "ctx_copy" in features:
        old_wr = "    if depth:\n        drafter.depth = max(1, min(drafter.depth, int(depth)))\n"
        new_wr = "    drafter.context_copy = True\n" + old_wr
        dsp_txt = _replace_once(dsp_txt, old_wr, new_wr, "dspark ctx_copy")
        dsp_txt += '\nprint("BENCH loaded ctx_copy", flush=True)\n'

    return dsp_txt, sel_txt


def build_overlay(overlay_dir: pathlib.Path, features: Iterable[str], resources: pathlib.Path) -> pathlib.Path:
    """Populate ``overlay_dir`` overriding only modified oMLX modules via symlinks."""
    features = _expand_features(features)

    if not (PATCHES / "inkling2_m1_kernels.py").is_file():
        raise FileNotFoundError(f"Missing M1 kernels patch at {PATCHES / 'inkling2_m1_kernels.py'}")
    if "fast_decode" in features and not (PATCHES / "inkling2_fast_decode.py").is_file():
        raise FileNotFoundError(f"Missing fast_decode patch at {PATCHES / 'inkling2_fast_decode.py'}")
    if "m1_qmv_moe" in features and not (PATCHES / "m1_qmv_moe.py").is_file():
        raise FileNotFoundError(f"Missing m1_qmv_moe patch at {PATCHES / 'm1_qmv_moe.py'}")

    base = pathlib.Path(overlay_dir)
    base.mkdir(parents=True, exist_ok=True)
    targets = {
        "omlx/patches/inkling2/model.py",
        "omlx/patches/inkling2/kernels.py",
        "omlx/patches/inkling2/dspark.py",
        "omlx/patches/mlx_lm_mtp/batch_generator.py",
        "omlx/speculative/dflash/selection.py",
        "omlx/models/vlm.py",
    }
    if "bounded_cache" in features:
        targets.add("omlx/patches/inkling2/cache.py")
        targets.add("omlx/prefill_helper/adapters/inkling2.py")
    if {"bounded_cache", "prompt_tail_cache"} & features:
        targets.add("omlx/scheduler.py")
    if {"predraft_fix", "dspark_2bit", "suffix_policy"} & features:
        targets.add("omlx/speculative/dflash/drafter.py")
    if "suffix_policy" in features:
        targets.add("omlx/speculative/suffix_drafter.py")
        targets.add("omlx/speculative/draft_policy.py")
    if "enable_thinking_fix" in features:
        targets.add("omlx/engine/vlm.py")
    if "prompt_tail_cache" in features:
        targets.update({
            "omlx/patches/inkling2/opener.py",
            "omlx/patches/inkling2/processing.py",
            "omlx/adapter/output_parser.py",
            "omlx/request.py",
            "omlx/server.py",
        })

    dir_map: dict[pathlib.Path, set[str]] = {}
    for rel in targets:
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

    if "fast_decode" in features:
        (base / "omlx/patches/inkling2/fast_decode.py").write_text(
            (PATCHES / "inkling2_fast_decode.py").read_text()
        )
    if "m1_qmv_moe" in features:
        (base / "omlx/patches/m1_qmv_moe.py").write_text(
            (PATCHES / "m1_qmv_moe.py").read_text()
        )
    if "bounded_cache" in features:
        (base / "omlx/patches/inkling2/cache.py").write_text(
            (PATCHES / "inkling2_cache.py").read_text()
        )
        (base / "omlx/prefill_helper/adapters/inkling2.py").write_text(
            patch_prefill_adapter((resources / "omlx/prefill_helper/adapters/inkling2.py").read_text())
        )
    if {"bounded_cache", "prompt_tail_cache"} & features:
        (base / "omlx/scheduler.py").write_text(
            patch_scheduler((resources / "omlx/scheduler.py").read_text(), features)
        )
    if {"predraft_fix", "dspark_2bit", "suffix_policy"} & features:
        (base / "omlx/speculative/dflash/drafter.py").write_text(
            patch_drafter((resources / "omlx/speculative/dflash/drafter.py").read_text(), features)
        )
    if "suffix_policy" in features:
        (base / "omlx/speculative/suffix_drafter.py").write_text(
            (PATCHES / "omlx_suffix_drafter.py").read_text()
        )
        (base / "omlx/speculative/draft_policy.py").write_text(
            (PATCHES / "omlx_draft_policy.py").read_text()
        )
    if "enable_thinking_fix" in features:
        (base / "omlx/engine/vlm.py").write_text(
            patch_vlm_engine((resources / "omlx/engine/vlm.py").read_text())
        )
    if "prompt_tail_cache" in features:
        (base / "omlx/patches/inkling2/opener.py").write_text(
            (PATCHES / "inkling2_opener.py").read_text()
        )
        (base / "omlx/patches/inkling2/processing.py").write_text(
            patch_processing((resources / "omlx/patches/inkling2/processing.py").read_text())
        )
        (base / "omlx/adapter/output_parser.py").write_text(
            patch_output_parser((resources / "omlx/adapter/output_parser.py").read_text())
        )
        (base / "omlx/request.py").write_text(
            patch_request((resources / "omlx/request.py").read_text())
        )
        (base / "omlx/server.py").write_text(
            patch_server((resources / "omlx/server.py").read_text())
        )

    # 1. kernels.py
    kernels_txt = patch_kernels((resources / "omlx/patches/inkling2/kernels.py").read_text(), features)
    (base / "omlx/patches/inkling2/kernels.py").write_text(kernels_txt)

    # 2. model.py
    model_txt = _patch_model((resources / "omlx/patches/inkling2/model.py").read_text(), features)
    (base / "omlx/patches/inkling2/model.py").write_text(model_txt)

    # 3. batch_generator.py
    bg_txt = _patch_batch_generator(
        (resources / "omlx/patches/mlx_lm_mtp/batch_generator.py").read_text(), features
    )
    (base / "omlx/patches/mlx_lm_mtp/batch_generator.py").write_text(bg_txt)

    # 4. vlm.py (skip_lm_head)
    vlm_txt = (resources / "omlx/models/vlm.py").read_text()
    if "skip_lm_head" in features:
        old_sl = '_SKIP_LOGITS_MODEL_TYPES = frozenset({"qwen4_exp", "hy_v4_vl", "hy_v4_flash"})'
        new_sl = '_SKIP_LOGITS_MODEL_TYPES = frozenset({"qwen4_exp", "hy_v4_vl", "hy_v4_flash", "inkling2", "inkling2_loop"})'
        vlm_txt = _replace_once(vlm_txt, old_sl, new_sl, "vlm skip_lm_head")
        vlm_txt += '\nprint("BENCH loaded skip_lm_head for inkling2", flush=True)\n'
    (base / "omlx/models/vlm.py").write_text(vlm_txt)

    # 5. dspark.py & selection.py
    dsp_txt, sel_txt = _patch_dspark_and_selection(
        (resources / "omlx/patches/inkling2/dspark.py").read_text(),
        (resources / "omlx/speculative/dflash/selection.py").read_text(),
        features,
    )
    (base / "omlx/patches/inkling2/dspark.py").write_text(dsp_txt)
    (base / "omlx/speculative/dflash/selection.py").write_text(sel_txt)
    return base
