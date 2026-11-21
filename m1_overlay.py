"""Self-contained oMLX 0.8.2 overlay builder for M1 Max Inkling 2 Mini optimizations."""
from __future__ import annotations

import pathlib
from typing import Iterable

LAB = pathlib.Path(__file__).resolve().parent
PATCHES = LAB / "patches"

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


def build_overlay(overlay_dir: pathlib.Path, features: Iterable[str], resources: pathlib.Path) -> pathlib.Path:
    """Populate ``overlay_dir`` overriding only modified oMLX modules via symlinks."""
    eff = set(features)
    if "dspark_fast" in eff:
        eff |= {"draft4", "fuse_draft_qkv", "markov_topk", "fast_greedy"}
    features = eff

    if not (PATCHES / "inkling2_m1_kernels.py").is_file():
        raise FileNotFoundError(f"Missing M1 kernels patch at {PATCHES / 'inkling2_m1_kernels.py'}")
    if "fast_decode" in features and not (PATCHES / "inkling2_fast_decode.py").is_file():
        raise FileNotFoundError(f"Missing fast_decode patch at {PATCHES / 'inkling2_fast_decode.py'}")

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
        fd_src = PATCHES / "inkling2_fast_decode.py"
        (base / "omlx/patches/inkling2/fast_decode.py").write_text(fd_src.read_text())

    # 1. kernels.py
    kernels_txt = (resources / "omlx/patches/inkling2/kernels.py").read_text()
    if "m1_attn" in features or "m1_attn_long" in features:
        cond = (
            "b == 1 and 1 < length <= 16"
            if "m1_attn" in features
            else "b == 1 and 1 < length <= 16 and k.shape[2] > 4096"
        )
        old = (
            "    if b == 1 and length == 1 and left_padding is None:\n"
            "        return _decode_attention(q, k, v, rel, sink, scale, window)\n"
        )
        new = (
            "    if b == 1 and length == 1 and left_padding is None:\n"
            "        return _decode_attention(q, k, v, rel, sink, scale, window)\n"
            "    if (\n"
            f"        {cond}\n"
            "        and q.dtype == mx.bfloat16\n"
            "        and q.shape[1] == 24\n"
            "        and k.shape[1] == 8\n"
            "        and q.shape[3] == 128\n"
            "    ):\n"
            "        from inkling2_m1_kernels import verify_banded_sink_attention\n"
            "        return verify_banded_sink_attention(\n"
            "            q, k, v, rel, sink, scale=scale, window=window, left_padding=left_padding\n"
            "        )\n"
        )
        assert kernels_txt.count(old) == 1
        kernels_txt = kernels_txt.replace(old, new)
        kernels_txt += f'\nprint("BENCH loaded M1 verify attention kernel ({cond})", flush=True)\n'
    (base / "omlx/patches/inkling2/kernels.py").write_text(kernels_txt)

    # 2. model.py
    model_txt = (resources / "omlx/patches/inkling2/model.py").read_text()
    if "sort32" in features:
        assert model_txt.count("do_sort = inds.size >= 64") == 1
        model_txt = model_txt.replace("do_sort = inds.size >= 64", "do_sort = inds.size >= 32")
    if "affine4" in features or "attn4" in features:
        prev = "        return out\n\n    @property\n    def cast_predicate"
        extra = (_AFFINE4_CODE if "affine4" in features else "") + (_ATTN4_CODE if "attn4" in features else "")
        assert model_txt.count(prev) == 1
        model_txt = model_txt.replace(prev, extra + prev)
    if "split_in_proj" in features:
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
        new_proj = (
            "            else:\n"
            "                _sp = self.__dict__.get(\"_m1_split_proj\")\n"
            "                if _sp:\n"
            "                    qrg_lin, k_lin, v_lin, qrg_cuts = _sp\n"
            "                    q, r, gate, sink = mx.split(qrg_lin(xq), qrg_cuts, axis=-1)\n"
            "                    k = k_lin(xk)\n"
            "                    v = v_lin(xv)\n"
            "                else:\n"
            "                    pq, pk, pv = mx.split(\n"
            "                        self.in_proj(mx.concatenate([xq, xk, xv], axis=0)), 3, axis=0\n"
            "                    )\n"
            "                    q, _, _, r, gate, sink = mx.split(pq, self._cuts, axis=-1)\n"
            "                    k = pk[..., self._cuts[0] : self._cuts[1]]\n"
            "                    v = pv[..., self._cuts[1] : self._cuts[2]]\n"
        )
        assert model_txt.count(old_proj) == 1
        model_txt = model_txt.replace(old_proj, new_proj)
    if "m1_moe" in features:
        sort_t = 32 if "sort32" in features else 64
        old_moe = (
            "    def __call__(self, x: mx.array, inds: mx.array, weights: mx.array) -> mx.array:\n"
            "        xe = mx.expand_dims(x, (-2, -3))\n"
            f"        do_sort = inds.size >= {sort_t}\n"
        )
        hook_moe = (
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
            "            )\n"
            + old_moe.split("\n", 1)[1]
        )
        assert model_txt.count(old_moe) == 1
        model_txt = model_txt.replace(old_moe, hook_moe)
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
        assert model_txt.count(old_cmp) == 1
        model_txt = model_txt.replace(old_cmp, new_cmp)
    if {"verify_compile", "split_in_proj", "m1_moe", "subwarp_moe"} & features:
        old_disp = (
            "        compiled = length <= _COMPILED_MAX_TOKENS\n"
            "        pre = self._pre_compiled if compiled else self._pre\n"
            "        post = self._post_compiled if compiled else self._post\n"
        )
        prep_lines = []
        if "split_in_proj" in features:
            prep_lines.append(
                "        if bank is not None and self.attn.sliding and self.attn.__dict__.get(\"_m1_split_proj\") is None:\n"
                "            from inkling2_m1_kernels import ensure_split_in_proj\n"
                "            ensure_split_in_proj(self.attn)\n"
            )
        if {"m1_moe", "subwarp_moe"} & features:
            prep_lines.append(
                "        if self.mlp.experts.w13.__dict__.get(\"_omlx_m1_alias\") is None:\n"
                "            from inkling2_m1_kernels import prepare_m1_moe\n"
                "            prepare_m1_moe(self.mlp.experts)\n"
            )
        if "verify_compile" in features:
            disp_lines = (
                "        compiled = length <= _COMPILED_MAX_TOKENS\n"
                "        if compiled:\n"
                "            from ..qwen35_verify_qmm import _is_armed as _vq_armed\n"
                "            _arm = _vq_armed()\n"
                "            pre = self._pre_verify_compiled if _arm else self._pre_compiled\n"
                "            post = self._post_verify_compiled if _arm else self._post_compiled\n"
                "        else:\n"
                "            pre = self._pre\n"
                "            post = self._post\n"
            )
        else:
            disp_lines = old_disp
        new_disp = "".join(prep_lines) + disp_lines
        assert model_txt.count(old_disp) == 1
        model_txt = model_txt.replace(old_disp, new_disp)
    if "conv_opt" in features:
        old_post_ret = "        return (residual + m.astype(mx.float32)).astype(h.dtype), s2, s3\n"
        old_call_tail = (
            "        if conv is not None:\n"
            "            windows = (ks, vs, s2, s3)\n"
            "            if 1 < length <= _COMPILED_MAX_TOKENS:\n"
            "                # Speculative verify rollback re-cuts the states from these.\n"
            "                conv._omlx_verify_xp = dict(enumerate(windows))\n"
            "            keep = self._conv_k - 1\n"
            "            for slot, window in enumerate(windows):\n"
            "                # A small copy, so the state does not pin the chunk's window.\n"
            "                conv[slot] = window[:, -keep:] * 1.0\n"
        )
        old_pre_ret = "        return (*self.attn.project(xq, xk, xv, ks, vs, tau), h_n)\n"
        new_pre_ret = (
            "        q, k, v, rel, gate, sink, ks, vs = self.attn.project(xq, xk, xv, ks, vs, tau)\n"
            "        _kkeep = self._conv_k - 1\n"
            "        return (q, k, v, rel, gate, sink, ks, vs, h_n, mx.contiguous(ks[:, -_kkeep:]), mx.contiguous(vs[:, -_kkeep:]))\n"
        )
        new_post_ret = (
            "        _kkeep = self._conv_k - 1\n"
            "        return (residual + m.astype(mx.float32)).astype(h.dtype), s2, s3, mx.contiguous(s2[:, -_kkeep:]), mx.contiguous(s3[:, -_kkeep:])\n"
        )
        old_unpack = (
            "        q, k, v, rel, gate, sink, ks, vs, h_n = pre(\n"
            "            h, blocks, states[0], states[1], tau, messages\n"
            "        )\n"
            "        out = self.attn.attend(q, k, v, rel, sink, kv)\n"
            "        if self.is_boundary:\n"
            "            blocks.append(h_n)\n"
            "        h, s2, s3 = post(out, gate, h, blocks, states[2], states[3], messages)\n"
        )
        new_unpack = (
            "        q, k, v, rel, gate, sink, ks, vs, h_n, ks_next, vs_next = pre(\n"
            "            h, blocks, states[0], states[1], tau, messages\n"
            "        )\n"
            "        out = self.attn.attend(q, k, v, rel, sink, kv)\n"
            "        if self.is_boundary:\n"
            "            blocks.append(h_n)\n"
            "        h, s2, s3, s2_next, s3_next = post(out, gate, h, blocks, states[2], states[3], messages)\n"
        )
        new_call_tail = (
            "        if conv is not None:\n"
            "            if 1 < length <= _COMPILED_MAX_TOKENS:\n"
            "                conv._omlx_verify_xp = {0: ks, 1: vs, 2: s2, 3: s3}\n"
            "            conv[0], conv[1], conv[2], conv[3] = ks_next, vs_next, s2_next, s3_next\n"
        )
        assert all(model_txt.count(x) == 1 for x in (old_pre_ret, old_post_ret, old_unpack, old_call_tail))
        model_txt = (
            model_txt.replace(old_pre_ret, new_pre_ret)
            .replace(old_post_ret, new_post_ret)
            .replace(old_unpack, new_unpack)
            .replace(old_call_tail, new_call_tail)
        )
    if "bf16_ctrl" in features:
        old_cast = '                or key.endswith("loop_down")\n                or key.endswith("loop_gain_forget")\n'
        assert model_txt.count(old_cast) == 1
        model_txt = model_txt.replace(old_cast, "")
        old_mix = (
            "    z = z.astype(mx.bfloat16).astype(mx.float32).reshape(2 * n, rows, hidden)\n"
            "    low = (z @ down.reshape(2 * n, hidden, -1)).reshape(n, 2, rows, -1).sum(axis=1)\n"
            "    low = low.astype(mx.bfloat16).astype(mx.float32)\n"
            "    gates = (low @ gain_forget).reshape(n, b, length, 2 * hidden)"
        )
        new_mix = (
            "    wd = down.dtype\n"
            "    z = z.astype(mx.bfloat16).astype(wd).reshape(2 * n, rows, hidden)\n"
            "    low = (z @ down.reshape(2 * n, hidden, -1)).reshape(n, 2, rows, -1).sum(axis=1)\n"
            "    low = low.astype(mx.bfloat16).astype(gain_forget.dtype)\n"
            "    gates = (low @ gain_forget).astype(mx.float32).reshape(n, b, length, 2 * hidden)"
        )
        assert model_txt.count(old_mix) == 1
        model_txt = model_txt.replace(old_mix, new_mix)
    if "fast_decode" in features:
        assert model_txt.count("from . import kernels\n") == 1
        model_txt = model_txt.replace("from . import kernels\n", "from . import fast_decode as _fd\nfrom . import kernels\n")

        old_exp_init = (
            "        self._gcap = mx.array(args.mlp_gate_softcap, dtype=mx.float32)\n"
            "        self._ucap = mx.array(args.mlp_up_softcap, dtype=mx.float32)\n"
        )
        new_exp_init = (
            old_exp_init
            + "        self._gcap_f = float(args.mlp_gate_softcap)\n"
            + "        self._ucap_f = float(args.mlp_up_softcap)\n"
        )
        assert model_txt.count(old_exp_init) == 1
        model_txt = model_txt.replace(old_exp_init, new_exp_init)

        sort_thresh = 32 if "sort32" in features else 64
        old_exp_call = f"    def __call__(self, x: mx.array, inds: mx.array, weights: mx.array) -> mx.array:\n        xe = mx.expand_dims(x, (-2, -3))\n        do_sort = inds.size >= {sort_thresh}\n"
        new_exp_call = (
            "    def __call__(self, x: mx.array, inds: mx.array, weights: mx.array) -> mx.array:\n"
            "        if _fd.part(\"moe\") and x.shape[0] <= 8:\n"
            "            return self._call_fused(x, inds, weights)\n"
            "        xe = mx.expand_dims(x, (-2, -3))\n"
            f"        do_sort = inds.size >= {sort_thresh}\n"
        )
        assert model_txt.count(old_exp_call) == 1
        model_txt = model_txt.replace(old_exp_call, new_exp_call)

        old_exp_end = (
            "        out = mx.sum(z.astype(mx.float32) * weights[..., None], axis=-2)\n"
            "        return out.astype(x.dtype)\n\n\nclass Inkling2MoE"
        )
        fused_exp_method = (
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
            "            and _fd.part(\"nvfp4_moe\")\n"
            "            and _fd.nvfp4_decode_eligible(self)\n"
            "        ):\n"
            "            return _fd.moe_decode_nvfp4(x, self, inds, weights)\n"
            "        return _fd.moe_combine(*self._fused_parts(x, inds, weights))\n\n\nclass Inkling2MoE"
        )
        assert model_txt.count(old_exp_end) == 1
        model_txt = model_txt.replace(old_exp_end, fused_exp_method)

        if "fuse_moe_conv_res" in features:
            old_moe_cls = (
                "    def __call__(self, x: mx.array) -> mx.array:\n"
                "        shape = x.shape\n"
                "        flat = x.reshape(-1, shape[-1])\n"
                "        inds, w = self.route(flat)\n"
                "        return self.experts(flat, inds, w).reshape(shape)\n"
            )
            new_moe_cls = (
                old_moe_cls
                + "\n    def forward_parts(self, x: mx.array):\n"
                + "        flat = x.reshape(-1, x.shape[-1])\n"
                + "        inds, w = self.route(flat)\n"
                + "        return self.experts._fused_parts(flat, inds, w)\n"
            )
            assert model_txt.count(old_moe_cls) == 1
            model_txt = model_txt.replace(old_moe_cls, new_moe_cls)

        old_msg_head = (
            "def _mix_messages(x, message, down, gain_forget):\n"
            "    \"\"\"Loop controllers: ``(1 - tanh(forget)) * x + tanh(gain) * message``.\n\n"
            "    ``x`` and ``message`` are [N, B, L, H] FP32, one row per controlled input.\n"
            "    ``down`` is [N, 2, H, r] (the x and message halves) and ``gain_forget``\n"
            "    [N, r, 2H], both FP32 copies of BF16 weights.\n"
            "    \"\"\"\n"
            "    n, b, length, hidden = x.shape\n"
            "    rows = b * length\n"
        )
        new_msg_head = (
            old_msg_head
            + "    if _fd.part(\"msg\") and rows <= 8:\n"
            + "        return _messages_fused(x, message, down, gain_forget)\n"
        )
        assert model_txt.count(old_msg_head) == 1
        model_txt = model_txt.replace(old_msg_head, new_msg_head)

        old_msg_tail = "    return (1.0 - forget) * x + gain * message\n\n\ndef _static_conv"
        new_msg_tail = (
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
            "    ).reshape(n, b, length, hidden)\n\n\ndef _static_conv"
        )
        assert model_txt.count(old_msg_tail) == 1
        model_txt = model_txt.replace(old_msg_tail, new_msg_tail)

        old_attn_init = (
            "        self._qk_clip = mx.array(args.qk_l2_clip_threshold, dtype=mx.float32)\n"
            "        self._r_clip = mx.array(args.rel_l2_clip_threshold, dtype=mx.float32)\n"
        )
        new_attn_init = (
            old_attn_init
            + "        self._qk_clip_f = float(args.qk_l2_clip_threshold)\n"
            + "        self._r_clip_f = float(args.rel_l2_clip_threshold)\n"
        )
        assert model_txt.count(old_attn_init) == 1
        model_txt = model_txt.replace(old_attn_init, new_attn_init)

        old_proj_start = "        b, length, _ = xq.shape\n        dkv = self.dkv\n"
        new_proj_start = old_proj_start + "        fast = _fd.part(\"conv\") and length <= _SHORT_WINDOW and b == 1\n"
        assert model_txt.count(old_proj_start) == 1
        model_txt = model_txt.replace(old_proj_start, new_proj_start)

        old_kv_sconv = (
            "            k, ks = _static_conv(k, self.k_conv, ks, self._sconv_scale)\n"
            "            v, vs = _static_conv(v, self.v_conv, vs, self._sconv_scale)\n"
        )
        new_kv_sconv = (
            "            if fast:\n"
            "                k, ks_t, ks_w = _fd.conv_short(k, self.k_conv, ks, self._sconv_scale)\n"
            "                v, vs_t, vs_w = _fd.conv_short(v, self.v_conv, vs, self._sconv_scale)\n"
            "                ks = ks_t if length == 1 else ks_w\n"
            "                vs = vs_t if length == 1 else vs_w\n"
            "            else:\n"
            "                k, ks = _static_conv(k, self.k_conv, ks, self._sconv_scale)\n"
            "                v, vs = _static_conv(v, self.v_conv, vs, self._sconv_scale)\n"
        )
        assert model_txt.count(old_kv_sconv) == 1
        model_txt = model_txt.replace(old_kv_sconv, new_kv_sconv)

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
            "        if fast and _fd.part(\"clip\"):\n"
            "            rows = b * length\n"
            "            q = _fd.l2_clip_heads(q.reshape(rows, self.n_heads, self.head_dim), self.q_scale, self._qk_clip_f).reshape(b, length, self.n_heads, self.head_dim)\n"
            "            k = _fd.l2_clip_heads(k.reshape(rows, self.n_kv, self.head_dim), self._one, self._qk_clip_f).reshape(b, length, self.n_kv, self.head_dim)\n"
            "            v = _fd.unit_rms_heads(v.reshape(rows, self.n_kv, self.head_dim)).reshape(b, length, self.n_kv, self.head_dim)\n"
            "            r = _fd.l2_clip_heads(r.reshape(rows, self.n_heads, self.d_rel), self.r_scale, self._r_clip_f).reshape(b, length, self.n_heads, self.d_rel)\n"
            "        else:\n"
            + "".join("    " + ln + "\n" for ln in old_clips.rstrip("\n").splitlines())
        )
        assert model_txt.count(old_clips) == 1
        model_txt = model_txt.replace(old_clips, new_clips)

        old_att = "        left_padding = getattr(kv, \"left_padding\", None) if kv is not None else None\n"
        new_att = (
            old_att
            + "        if _fd.part(\"mask\") and left_padding is None and q.shape[0] == 1 and q.shape[2] == 1:\n"
            + "            lo = max(0, k.shape[2] - self.window) if self.window else 0\n"
            + "            if lo:\n"
            + "                k = k[:, :, lo:, :]\n"
            + "                v = v[:, :, lo:, :]\n"
            + "            mask = _fd.decode_mask(rel, k.shape[2])\n"
            + "            return mx.fast.scaled_dot_product_attention(\n"
            + "                q, k, v, scale=self.scale, mask=mask, sinks=sink[0, 0].astype(q.dtype)\n"
            + "            )\n"
        )
        assert model_txt.count(old_att) == 1
        model_txt = model_txt.replace(old_att, new_att)

        old_cmax = "_COMPILED_MAX_TOKENS = 16\n"
        new_cmax = "_COMPILED_MAX_TOKENS = 16\n_SHORT_WINDOW = 8\n"
        assert model_txt.count(old_cmax) == 1
        model_txt = model_txt.replace(old_cmax, new_cmax)

        old_post_c1 = "        a, s2 = _static_conv(a, self.attn_conv, s2, self._conv_scale)\n"
        new_post_c1 = (
            "        if _fd.part(\"conv\") and a.shape[1] <= _SHORT_WINDOW:\n"
            "            a, s2t, s2w = _fd.conv_short(a, self.attn_conv, s2, self._conv_scale)\n"
            "            s2 = s2t if a.shape[1] == 1 else s2w\n"
            "        else:\n"
            "            a, s2 = _static_conv(a, self.attn_conv, s2, self._conv_scale)\n"
        )
        assert model_txt.count(old_post_c1) == 1
        model_txt = model_txt.replace(old_post_c1, new_post_c1)

        old_post_c2 = "        m, s3 = _static_conv(m, self.mlp_conv, s3, self._conv_scale)\n"
        new_post_c2 = (
            "        if _fd.part(\"conv\") and m.shape[1] <= _SHORT_WINDOW:\n"
            "            m, s3t, s3w = _fd.conv_short(m, self.mlp_conv, s3, self._conv_scale)\n"
            "            s3 = s3t if m.shape[1] == 1 else s3w\n"
            "        else:\n"
            "            m, s3 = _static_conv(m, self.mlp_conv, s3, self._conv_scale)\n"
        )
        assert model_txt.count(old_post_c2) == 1
        model_txt = model_txt.replace(old_post_c2, new_post_c2)

        if "fuse_moe_conv_res" in features:
            old_post_full = (
                "        m = self.mlp(self.mlp_norm(y).astype(h.dtype))\n"
                "        if _fd.part(\"conv\") and m.shape[1] <= _SHORT_WINDOW:\n"
                "            m, s3t, s3w = _fd.conv_short(m, self.mlp_conv, s3, self._conv_scale)\n"
                "            s3 = s3t if m.shape[1] == 1 else s3w\n"
                "        else:\n"
                "            m, s3 = _static_conv(m, self.mlp_conv, s3, self._conv_scale)\n"
                "        return (residual + m.astype(mx.float32)).astype(h.dtype), s2, s3\n"
            )
            new_post_full = (
                "        _mx = self.mlp_norm(y).astype(h.dtype)\n"
                "        if (\n"
                "            _fd.part(\"moe\")\n"
                "            and _fd.part(\"conv\")\n"
                "            and _mx.shape[0] == 1\n"
                "            and _mx.shape[1] <= _SHORT_WINDOW\n"
                "            and not (\n"
                "                _mx.shape[1] == 1\n"
                "                and _fd.part(\"nvfp4_moe\")\n"
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
                "        if _fd.part(\"conv\") and m.shape[1] <= _SHORT_WINDOW:\n"
                "            m, s3t, s3w = _fd.conv_short(m, self.mlp_conv, s3, self._conv_scale)\n"
                "            s3 = s3t if m.shape[1] == 1 else s3w\n"
                "        else:\n"
                "            m, s3 = _static_conv(m, self.mlp_conv, s3, self._conv_scale)\n"
                "        return (residual + m.astype(mx.float32)).astype(h.dtype), s2, s3\n"
            )
            assert model_txt.count(old_post_full) == 1
            model_txt = model_txt.replace(old_post_full, new_post_full)

        old_conv_loop = (
            "            for slot, window in enumerate(windows):\n"
            "                # A small copy, so the state does not pin the chunk's window.\n"
            "                conv[slot] = window[:, -keep:] * 1.0\n"
        )
        new_conv_loop = (
            "            for slot, window in enumerate(windows):\n"
            "                if window.shape[1] == keep:\n"
            "                    conv[slot] = window\n"
            "                    continue\n"
            "                # A small copy, so the state does not pin the chunk's window.\n"
            "                conv[slot] = window[:, -keep:] * 1.0\n"
        )
        assert model_txt.count(old_conv_loop) == 1
        model_txt = model_txt.replace(old_conv_loop, new_conv_loop)

    if "skip_lm_head" in features:
        old_fwd = (
            "        if not capture_layer_ids and not return_hidden:\n"
            "            return self.logits_from_hidden(self.model(inputs, cache, input_embeddings))\n"
            "        capture = {int(i): None for i in capture_layer_ids or ()}\n"
            "        h = self.model(inputs, cache, input_embeddings, capture=capture)\n"
            "        logits = self.logits_from_hidden(h)\n"
        )
        new_fwd = (
            "        _skip = bool(kwargs.get(\"skip_logits\") or kwargs.get(\"skip_lm_head\"))\n"
            "        _b = inputs.shape[0] if inputs is not None else input_embeddings.shape[0]\n"
            "        if not capture_layer_ids and not return_hidden:\n"
            "            _h = self.model(inputs, cache, input_embeddings)\n"
            "            return mx.zeros((_b, 1, 1), dtype=_h.dtype) if _skip else self.logits_from_hidden(_h)\n"
            "        capture = {int(i): None for i in capture_layer_ids or ()}\n"
            "        h = self.model(inputs, cache, input_embeddings, capture=capture)\n"
            "        logits = mx.zeros((_b, 1, 1), dtype=h.dtype) if _skip else self.logits_from_hidden(h)\n"
        )
        assert model_txt.count(old_fwd) == 1
        model_txt = model_txt.replace(old_fwd, new_fwd)
    if "subwarp_moe" in features:
        if "self._gcap_f" not in model_txt:
            old_ei = (
                "        self._gcap = mx.array(args.mlp_gate_softcap, dtype=mx.float32)\n"
                "        self._ucap = mx.array(args.mlp_up_softcap, dtype=mx.float32)\n"
            )
            new_ei = (
                old_ei
                + "        self._gcap_f = float(args.mlp_gate_softcap)\n"
                + "        self._ucap_f = float(args.mlp_up_softcap)\n"
            )
            assert model_txt.count(old_ei) == 1
            model_txt = model_txt.replace(old_ei, new_ei)
        old_sw_call = "    def __call__(self, x: mx.array, inds: mx.array, weights: mx.array) -> mx.array:\n"
        new_sw_call = (
            "    def __call__(self, x: mx.array, inds: mx.array, weights: mx.array) -> mx.array:\n"
            "        if x.shape[0] >= 2 and x.shape[0] <= 16 and self.w13.post_scale(inds) is None and int(self.w13.bits) in (3, 4):\n"
            "            from inkling2_m1_kernels import subwarp_inkling2_moe\n"
            "            return subwarp_inkling2_moe(x, inds, weights, self.w13, self.w2, self._gcap_f, self._ucap_f)\n"
        )
        assert model_txt.count(old_sw_call) == 1
        model_txt = model_txt.replace(old_sw_call, new_sw_call)
    if "prefill_pad64" in features:
        old_exp_def = "    def __call__(self, x: mx.array, inds: mx.array, weights: mx.array) -> mx.array:\n"
        new_exp_def = (
            "    def __call__(self, x: mx.array, inds: mx.array, weights: mx.array) -> mx.array:\n"
            "        if 28 <= x.shape[0] < 64:\n"
            "            _n0 = x.shape[0]\n"
            "            _p = 64 - _n0\n"
            "            _xp = mx.concatenate([x, mx.zeros((_p, x.shape[1]), dtype=x.dtype)], axis=0)\n"
            "            _ip = mx.concatenate([inds, mx.zeros((_p, inds.shape[1]), dtype=inds.dtype)], axis=0)\n"
            "            _wp = mx.concatenate([weights, mx.zeros((_p, weights.shape[1]), dtype=weights.dtype)], axis=0)\n"
            "            return self._call_unpadded(_xp, _ip, _wp)[:_n0]\n"
            "        return self._call_unpadded(x, inds, weights)\n\n"
            "    def _call_unpadded(self, x: mx.array, inds: mx.array, weights: mx.array) -> mx.array:\n"
        )
        assert model_txt.count(old_exp_def) == 1
        model_txt = model_txt.replace(old_exp_def, new_exp_def)
    if "top6" in features:
        old_tk = "        self.top_k = args.num_experts_per_tok\n"
        new_tk = "        self.top_k = 6\n"
        assert model_txt.count(old_tk) == 1
        model_txt = model_txt.replace(old_tk, new_tk)
    model_txt += f'\nprint("BENCH loaded model features={sorted(features)}", flush=True)\n'
    (base / "omlx/patches/inkling2/model.py").write_text(model_txt)

    # 3. batch_generator.py
    bg_txt = (resources / "omlx/patches/mlx_lm_mtp/batch_generator.py").read_text()
    if "caps" in features or "caps200" in features:
        if "caps" in features:
            assert bg_txt.count("_SPEC_BUFFER_CAPS = (200, 512)") == 1
            bg_txt = bg_txt.replace("_SPEC_BUFFER_CAPS = (200, 512)", "_SPEC_BUFFER_CAPS = (400, 4096)")
        bg_txt = bg_txt.replace(
            '"qwen4_exp_text",\n    }',
            '"qwen4_exp_text",\n        "inkling2",\n        "inkling2_loop",\n    }',
        )
        bg_txt = bg_txt.replace(
            'return architecture.startswith("applegpu_g15")',
            'return architecture.startswith("applegpu_g15") or architecture.startswith("applegpu_g13")',
        )
        bg_txt += '\nprint("BENCH loaded M1 spec buffer caps", flush=True)\n'
    if "fast_greedy" in features:
        old_lp = "    combined_lp = rows - mx.logsumexp(rows, axis=-1, keepdims=True)  # (k+1, V)\n"
        new_lp = (
            "    _want_lp = bool(getattr(gen_batch, \"logprobs\", None))\n"
            "    if k > 0 and (is_greedy or follower) and verify_procs is None and not _want_lp:\n"
            "        combined_lp = rows\n"
            "    else:\n"
            "        combined_lp = rows - mx.logsumexp(rows, axis=-1, keepdims=True)  # (k+1, V)\n"
        )
        assert bg_txt.count(old_lp) == 1
        bg_txt = bg_txt.replace(old_lp, new_lp)
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
        assert bg_txt.count(old_gr) == 1
        bg_txt = bg_txt.replace(old_gr, new_gr)
        bg_txt += '\nprint("BENCH loaded margin30", flush=True)\n'
    (base / "omlx/patches/mlx_lm_mtp/batch_generator.py").write_text(bg_txt)

    # 4. vlm.py (skip_lm_head)
    vlm_txt = (resources / "omlx/models/vlm.py").read_text()
    if "skip_lm_head" in features:
        old_sl = '_SKIP_LOGITS_MODEL_TYPES = frozenset({"qwen4_exp", "hy_v4_vl", "hy_v4_flash"})'
        new_sl = '_SKIP_LOGITS_MODEL_TYPES = frozenset({"qwen4_exp", "hy_v4_vl", "hy_v4_flash", "inkling2", "inkling2_loop"})'
        assert vlm_txt.count(old_sl) == 1
        vlm_txt = vlm_txt.replace(old_sl, new_sl)
        vlm_txt += '\nprint("BENCH loaded skip_lm_head for inkling2", flush=True)\n'
    (base / "omlx/models/vlm.py").write_text(vlm_txt)

    # 5. dspark.py & selection.py
    dsp_txt = (resources / "omlx/patches/inkling2/dspark.py").read_text()
    sel_txt = (resources / "omlx/speculative/dflash/selection.py").read_text()
    if "draft4" in features:
        old_dq = '_DRAFT_QUANT = {"group_size": 64, "bits": 8}'
        new_dq = '_DRAFT_QUANT = {"group_size": 64, "bits": 4}'
        assert dsp_txt.count(old_dq) == 1
        dsp_txt = dsp_txt.replace(old_dq, new_dq)
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
            "        kvp = self.__dict__.get(\"_kv_ctx_proj\")\n"
            "        if kvp is not None:\n"
            "            kv = kvp(mixed)\n"
            "            dkv = self.n_kv_heads * self.head_dim\n"
            "            return kv[..., :dkv], kv[..., dkv:]\n"
            "        return self.k_proj_ctx(mixed), self.v_proj_ctx(mixed)\n"
        )
        assert dsp_txt.count(old_pctx) == 1
        dsp_txt = dsp_txt.replace(old_pctx, new_pctx)

        old_dmlp = (
            "    def __call__(self, x):\n"
            "        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))\n"
        )
        new_dmlp = (
            "    def __call__(self, x):\n"
            "        gup = self.__dict__.get(\"_gate_up_proj\")\n"
            "        if gup is not None:\n"
            "            gu = gup(x)\n"
            "            half = gu.shape[-1] // 2\n"
            "            return self.down_proj(nn.silu(gu[..., :half]) * gu[..., half:])\n"
            "        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))\n"
        )
        assert dsp_txt.count(old_dmlp) == 1
        dsp_txt = dsp_txt.replace(old_dmlp, new_dmlp)

        old_load_ret = (
            "    mx.eval(model.parameters())\n"
            "    return model\n"
        )
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
            "        object.__setattr__(sa, \"_qkv_proj\", _concat_qlinears([sa.q_proj, sa.k_proj, sa.v_proj]))\n"
            "        object.__setattr__(sa, \"_kv_ctx_proj\", _concat_qlinears([sa.k_proj_ctx, sa.v_proj_ctx]))\n"
            "        mlp = lyr.mlp\n"
            "        object.__setattr__(mlp, \"_gate_up_proj\", _concat_qlinears([mlp.gate_proj, mlp.up_proj]))\n"
            "    from ...speculative.dflash import variants as _df_variants\n"
            "    if not getattr(_df_variants.project_qkv, \"_m1_qkv_patched\", False):\n"
            "        _orig_pqkv = _df_variants.project_qkv\n"
            "        def _fused_pqkv(attn, x, context):\n"
            "            qkvp = attn.__dict__.get(\"_qkv_proj\")\n"
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
        assert dsp_txt.count(old_load_ret) == 1
        dsp_txt = dsp_txt.replace(old_load_ret, new_load_ret)
        dsp_txt += '\nprint("BENCH loaded fuse_draft_qkv", flush=True)\n'
    if "markov_topk" in features:
        old_mh_init = "        self._softcap = config.final_logit_softcapping\n"
        new_mh_init = (
            "        self._softcap = config.final_logit_softcapping\n"
            "        self._unpadded = int(config.unpadded_vocab_size or config.vocab_size)\n"
        )
        assert dsp_txt.count(old_mh_init) == 1
        dsp_txt = dsp_txt.replace(old_mh_init, new_mh_init)

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
        assert dsp_txt.count(old_mh_end) == 1
        dsp_txt = dsp_txt.replace(old_mh_end, new_mh_end)
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
            "        if all(sampler is None for sampler in samplers) and hasattr(markov, \"apply_step_greedy_topk\"):\n"
            "            prev = markov.apply_step_greedy_topk(logits[:, step], prev, 256)\n"
            "        elif all(sampler is None for sampler in samplers):\n"
            "            prev = mx.argmax(markov.apply_step_logits(logits[:, step], prev), axis=-1)\n"
            "        else:\n"
            "            step_logits = markov.apply_step_logits(logits[:, step], prev)\n"
        )
        assert sel_txt.count(old_sel) == 1
        sel_txt = sel_txt.replace(old_sel, new_sel)
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
        assert dsp_txt.count(old_mh_init) == 1
        dsp_txt = dsp_txt.replace(old_mh_init, new_mh_init)

        old_fold_end = (
            "        self._fold_weight = mx.concatenate(rows, axis=0)\n"
            "        self._fold_bias = mx.concatenate(biases, axis=0)\n"
            "        mx.eval(self._fold_weight, self._fold_bias)\n"
        )
        new_fold_end = (
            old_fold_end
            + "        fw_unp = self._fold_weight[: self._unpadded]\n"
            + "        self._fw_q, self._fw_s, self._fw_b = mx.quantize(\n"
            + "            fw_unp, group_size=64, bits=4, mode=\"affine\"\n"
            + "        )\n"
            + "        self._fb_unp = self._fold_bias[: self._unpadded].astype(mx.float32)\n"
            + "        mx.eval(self._fw_q, self._fw_s, self._fw_b, self._fb_unp)\n"
        )
        assert dsp_txt.count(old_fold_end) == 1
        dsp_txt = dsp_txt.replace(old_fold_end, new_fold_end)

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
            + "                mode=\"affine\",\n"
            + "            ).astype(mx.float32)\n"
            + "            + self._fb_unp\n"
            + "        )\n"
            + "        self._step += 1\n"
            + "        scores = base_logits[:, : self._unpadded].astype(mx.float32) + bias\n"
            + "        return mx.argmax(scores, axis=-1)\n"
        )
        assert dsp_txt.count(old_mh_end) == 1
        dsp_txt = dsp_txt.replace(old_mh_end, new_mh_end)

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
            "                    lm_head.get(\"biases\")[start : start + 16384] if lm_head.get(\"biases\") is not None else None,\n"
            "                    lm_head.group_size,\n"
            "                    lm_head.bits,\n"
            "                    mode=getattr(lm_head, \"mode\", \"affine\"),\n"
            "                ).astype(mx.bfloat16)\n"
            "                q4, s4, b4 = mx.quantize(w_chunk, group_size=64, bits=4, mode=\"affine\")\n"
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
            "                hidden.astype(s.dtype), w, s, b, transpose=True, group_size=64, bits=4, mode=\"affine\"\n"
            "            )\n"
            "            print(\"BENCH loaded 4-bit Loom-style draft proposal head + Markov matrix\", flush=True)\n"
            "        else:\n"
            "            self._lm_head_fn = lambda hidden: lm_head(hidden)\n"
        )
        assert dsp_txt.count(old_bind_head) == 1
        dsp_txt = dsp_txt.replace(old_bind_head, new_bind_head)

        old_sel = (
            "    for step in range(logits.shape[1]):\n"
            "        step_logits = markov.apply_step_logits(logits[:, step], prev)\n"
            "        if all(sampler is None for sampler in samplers):\n"
            "            prev = mx.argmax(step_logits, axis=-1)\n"
            "        else:\n"
        )
        new_sel = (
            "    for step in range(logits.shape[1]):\n"
            "        if all(sampler is None for sampler in samplers) and hasattr(markov, \"apply_step_greedy\"):\n"
            "            prev = markov.apply_step_greedy(logits[:, step], prev)\n"
            "        elif all(sampler is None for sampler in samplers):\n"
            "            prev = mx.argmax(markov.apply_step_logits(logits[:, step], prev), axis=-1)\n"
            "        else:\n"
            "            step_logits = markov.apply_step_logits(logits[:, step], prev)\n"
        )
        assert sel_txt.count(old_sel) == 1
        sel_txt = sel_txt.replace(old_sel, new_sel)
    if "ctx_copy" in features:
        old_wr = "    if depth:\n        drafter.depth = max(1, min(drafter.depth, int(depth)))\n"
        new_wr = "    drafter.context_copy = True\n" + old_wr
        assert dsp_txt.count(old_wr) == 1
        dsp_txt = dsp_txt.replace(old_wr, new_wr)
        dsp_txt += '\nprint("BENCH loaded ctx_copy", flush=True)\n'
    (base / "omlx/patches/inkling2/dspark.py").write_text(dsp_txt)
    (base / "omlx/speculative/dflash/selection.py").write_text(sel_txt)
    return base
