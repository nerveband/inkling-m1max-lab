# SPDX-License-Identifier: Apache-2.0
"""Modular PR backport transforms for the oMLX 0.8.2 M1 overlay.

Backports independent opt-in optimizations from jundot/omlx PRs #5505-#5511
while keeping full composability with M1_FAST_FEATURES and optional lab flags.
"""
from __future__ import annotations


def _replace_once(text: str, old: str, new: str, label: str = "") -> str:
    count = text.count(old)
    if count != 1:
        tag = f" ({label})" if label else ""
        raise AssertionError(f"Expected 1 occurrence{tag}, found {count}: {old[:100]!r}")
    return text.replace(old, new)


# ---------------------------------------------------------------------------
# PR #5511 (kernels.py): FP8 KV, FlashDecoding, prefill_fast, bf16_loop_ctrl
# ---------------------------------------------------------------------------

_FP8_HELPERS_SRC = '''

def _build_fp8_e4m3_lut_header() -> str:
    """256-entry exact OCP FP8 E4M3 lookup table in Metal constant memory."""
    vals = []
    for c in range(256):
        if (c & 0x7F) == 0x7F:
            v = 0.0
        else:
            s = -1.0 if (c & 0x80) else 1.0
            e = (c >> 3) & 0xF
            m = c & 0x7
            v = s * (
                (m / 8.0) * (2.0**-6)
                if e == 0
                else (1.0 + m / 8.0) * (2.0 ** (e - 7))
            )
        vals.append(f"half({v:.8e}f)")
    return (
        "constant half FP8_E4M3_LUT[256] = {"
        + ",".join(vals)
        + "};\\n"
        + """
inline half fp8_e4m3_to_half(uchar code) {
    return FP8_E4M3_LUT[code];
}

inline uchar float_to_fp8_e4m3(float x) {
    uint u = as_type<uint>(x);
    uint sign = (u >> 24) & 0x80u;
    float ax = metal::fabs(x);
    if (!(ax < 448.0f)) {
        return (ax >= 448.0f) ? uchar(sign | 0x7Eu) : uchar(0u);
    }
    if (ax < 0.015625f) {
        uint q = uint(metal::rint(ax * 512.0f));
        return uchar(sign | q);
    }
    uint ua = as_type<uint>(ax);
    uint lsb = (ua >> 20) & 1u;
    uint rounded = ua + 0x7FFFFu + lsb;
    uint e = (rounded >> 23) - 120u;
    uint m = (rounded >> 20) & 0x7u;
    uint code = (e << 3) | m;
    if (code > 0x7Eu) code = 0x7Eu;
    return uchar(sign | code);
}
"""
    )


# Unit-scale IEEE/OCP FP8 E4M3 (1 sign, 4 exp bias 7, 3 mantissa; max +-448.0,
# no inf; overflow clamps to +-448.0 so NaN codes 0x7F/0xFF are never emitted).
_FP8_E4M3_HEADER = _build_fp8_e4m3_lut_header()

_FP8_QUANTIZE_SOURCE = """
    uint idx = thread_position_in_grid.x;
    if (VEC4) {
        uint base = idx * 4;
        if (base + 3 < N) {
            uchar4 out4;
            out4.x = float_to_fp8_e4m3(float(x[base + 0]));
            out4.y = float_to_fp8_e4m3(float(x[base + 1]));
            out4.z = float_to_fp8_e4m3(float(x[base + 2]));
            out4.w = float_to_fp8_e4m3(float(x[base + 3]));
            *((device uchar4*)(out + base)) = out4;
        }
    } else if (idx < N) {
        out[idx] = float_to_fp8_e4m3(float(x[idx]));
    }
"""

_FP8_DEQUANTIZE_SOURCE = """
    uint idx = thread_position_in_grid.x;
    if (idx < N) {
        out[idx] = OT(fp8_e4m3_to_half(x[idx]));
    }
"""

_fp8_quantize_kernel = None
_fp8_dequantize_kernel = None


def fp8_e4m3_quantize(x: mx.array) -> mx.array:
    """Quantize ``x`` to unit-scale OCP FP8 E4M3 codes (``uint8``, clamped to +-448)."""
    global _fp8_quantize_kernel
    if mx.default_device() == mx.gpu and x.size > 0:
        if _fp8_quantize_kernel is None:
            _fp8_quantize_kernel = mx.fast.metal_kernel(
                name="omlx_inkling2_fp8_quantize",
                input_names=["x"],
                output_names=["out"],
                source=_FP8_QUANTIZE_SOURCE,
                header=_FP8_E4M3_HEADER,
            )
        xc = mx.contiguous(x)
        n = xc.size
        vec4 = 1 if (n % 4 == 0) else 0
        grid_x = (n // 4) if vec4 else n
        return _fp8_quantize_kernel(
            inputs=[xc],
            template=[("T", xc.dtype), ("N", n), ("VEC4", vec4)],
            grid=(grid_x, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[xc.shape],
            output_dtypes=[mx.uint8],
        )[0]
    # CPU reference: exact IEEE/OCP FP8 E4M3 round-to-nearest-even.
    u32 = lambda val: mx.array(val, dtype=mx.uint32)
    xf = x.astype(mx.float32)
    u = xf.view(mx.uint32)
    sign = (u >> u32(24)) & u32(0x80)
    ax = mx.abs(xf)
    is_nan = mx.isnan(xf)
    ax_safe = mx.where(is_nan, mx.array(0.0, dtype=mx.float32), ax)
    sub = mx.round(mx.minimum(ax_safe, 0.015625) * 512.0).astype(mx.uint32)
    ua = mx.maximum(ax_safe, 0.015625).view(mx.uint32)
    lsb = (ua >> u32(20)) & u32(1)
    rounded = ua + u32(0x7FFFF) + lsb
    e = (rounded >> u32(23)) - u32(120)
    m = (rounded >> u32(20)) & u32(0x7)
    norm = mx.minimum((e << u32(3)) | m, u32(0x7E))
    mag = mx.where(ax_safe < 0.015625, sub, norm)
    mag = mx.where(ax_safe >= 448.0, u32(0x7E), mag)
    code = mx.where(is_nan, u32(0), sign | mag)
    return code.astype(mx.uint8)


def fp8_e4m3_dequantize(x: mx.array, dtype=mx.bfloat16) -> mx.array:
    """Dequantize unit-scale OCP FP8 E4M3 ``uint8`` codes to ``dtype``."""
    global _fp8_dequantize_kernel
    if mx.default_device() == mx.gpu and x.size > 0:
        if _fp8_dequantize_kernel is None:
            _fp8_dequantize_kernel = mx.fast.metal_kernel(
                name="omlx_inkling2_fp8_dequantize",
                input_names=["x"],
                output_names=["out"],
                source=_FP8_DEQUANTIZE_SOURCE,
                header=_FP8_E4M3_HEADER,
            )
        xc = mx.contiguous(x)
        return _fp8_dequantize_kernel(
            inputs=[xc],
            template=[("OT", dtype), ("N", xc.size)],
            grid=(xc.size, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[xc.shape],
            output_dtypes=[dtype],
        )[0]
    u32 = lambda val: mx.array(val, dtype=mx.uint32)
    code = x.astype(mx.uint32)
    sgn = (code & u32(0x80)) << u32(8)
    mag = code & u32(0x7F)
    norm_h = (sgn | ((mag + u32(0x40)) << u32(7))).astype(mx.uint16).view(mx.float16)
    zero_h = sgn.astype(mx.uint16).view(mx.float16)
    sub_h = (
        (sgn | u32(0x2400) | (mag << u32(7))).astype(mx.uint16).view(mx.float16)
        - (sgn | u32(0x2400)).astype(mx.uint16).view(mx.float16)
    )
    return mx.where(mag >= 8, norm_h, mx.where(mag == 0, zero_h, sub_h)).astype(dtype)
'''

_GQA_FLASH_DECODE_SRC = '''
# GQA-packed Split-K FlashDecoding kernel for short query blocks (1 <= L < 32:
# speculative verify and multi-stream decode). Packs all ``G * L`` queries
# sharing one KV head into rows ``0 .. G*L - 1`` so K and V are read once per
# KV head via 16-byte loads without copying K/V for a dummy sink row or
# materializing an additive mask.
_GQA_DECODE_PART_SOURCE = """
    constexpr int BK = 16;
    constexpr int NT = NSG * 32;
    constexpr float LOG2E = 1.4426950408889634f;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const int sp = threadgroup_position_in_grid.x;
    const int hkv_qt = threadgroup_position_in_grid.y;
    const int b = threadgroup_position_in_grid.z;

    const int L = dims[0];
    const int S = dims[1];
    const int NS = dims[2];
    const int lp = dims[3 + b];

    const int n_rows = G * L;
    const int BQ = NSG * 8;
    const int nqt = (n_rows + BQ - 1) / BQ;
    const int hkv = hkv_qt / nqt;
    const int qt = hkv_qt % nqt;

    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    const int packed_row = qt * BQ + int(sg) * 8 + fm;
    const bool row_ok = packed_row < n_rows;
    const int g_idx = row_ok ? (packed_row / L) : 0;
    const int q_pos = row_ok ? (packed_row % L) : 0;
    const int h = hkv * G + g_idx;
    const int a_row = S - L + q_pos;
    const float qscale = scale[0] * LOG2E;

    const int a_first = S - L;
    const int a_last = S - 1;
    int lo = WINDOW > 0 ? max(0, a_first - WINDOW + 1) : 0;
    lo = max(lo, lp);
    const int nk = max(0, a_last + 1 - lo);
    const int per_sp = ((nk + NS - 1) / NS + BK - 1) / BK * BK;
    const int k_start = lo + sp * per_sp;
    const int k_end = min(a_last + 1, k_start + per_sp);

    constexpr int LDK = BK + 8;
    constexpr int LDV = D + 8;
    threadgroup half Ks[D * LDK];
    threadgroup half Vs[BK * LDV];

    simdgroup_half8x8 Qf[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        float2 qv = float2(0.0f);
        if (row_ok) {
            const device T* qp = q + b * q_strides[0] + h * q_strides[1]
                + q_pos * q_strides[2] + (db * 8 + fn) * q_strides[3];
            qv = float2(float(qp[0]), float(qp[q_strides[3]]));
        }
        Qf[db].thread_elements()[0] = half(qv.x);
        Qf[db].thread_elements()[1] = half(qv.y);
    }
    simdgroup_float8x8 Of[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        Of[db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float m = (SINGLE && row_ok) ? float(sink[(b * L + q_pos) * HQ + h]) * LOG2E : -INFINITY;
    float l = SINGLE ? 1.0f : 0.0f;
    const device T* relp = rel + ((size_t)(b * L + q_pos) * HQ + h) * R;

    const device KT* kb_ptr = k + b * k_strides[0] + hkv * k_strides[1];
    const device KT* vb_ptr = v + b * v_strides[0] + hkv * v_strides[1];
    const bool vec_kv = FP8_KV
        ? (k_strides[3] == 1 && v_strides[3] == 1
           && (k_strides[2] % 16 == 0) && (v_strides[2] % 16 == 0))
        : (k_strides[3] == 1 && v_strides[3] == 1
           && (k_strides[2] % 8 == 0) && (v_strides[2] % 8 == 0));

    for (int k0 = k_start; k0 < k_end; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (FP8_KV) {
            if (vec_kv) {
                for (int c = int(tid); c < BK * D / 16; c += NT) {
                    const int r = c / (D / 16);
                    const int col = (c % (D / 16)) * 16;
                    const int key = k0 + r;
                    uint4 kw = uint4(0);
                    uint4 vw = uint4(0);
                    if (key < k_end) {
                        kw = *(const device uint4*)(kb_ptr + key * k_strides[2] + col);
                        vw = *(const device uint4*)(vb_ptr + key * v_strides[2] + col);
                    }
                    threadgroup half* kd = Ks + col * LDK + r;
                    threadgroup half* vd = Vs + r * LDV + col;
                    for (int j = 0; j < 4; ++j) {
                        uint kword = kw[j];
                        uint vword = vw[j];
                        kd[(4 * j + 0) * LDK] = fp8_e4m3_to_half(uchar(kword & 0xFFu));
                        kd[(4 * j + 1) * LDK] = fp8_e4m3_to_half(uchar((kword >> 8) & 0xFFu));
                        kd[(4 * j + 2) * LDK] = fp8_e4m3_to_half(uchar((kword >> 16) & 0xFFu));
                        kd[(4 * j + 3) * LDK] = fp8_e4m3_to_half(uchar((kword >> 24) & 0xFFu));
                        vd[4 * j + 0] = fp8_e4m3_to_half(uchar(vword & 0xFFu));
                        vd[4 * j + 1] = fp8_e4m3_to_half(uchar((vword >> 8) & 0xFFu));
                        vd[4 * j + 2] = fp8_e4m3_to_half(uchar((vword >> 16) & 0xFFu));
                        vd[4 * j + 3] = fp8_e4m3_to_half(uchar((vword >> 24) & 0xFFu));
                    }
                }
            } else {
                for (int c = int(tid); c < BK * D; c += NT) {
                    const int r = c / D;
                    const int col = c % D;
                    const int key = k0 + r;
                    half kv = half(0.0h);
                    half vv = half(0.0h);
                    if (key < k_end) {
                        kv = fp8_e4m3_to_half(uchar(kb_ptr[key * k_strides[2] + col * k_strides[3]]));
                        vv = fp8_e4m3_to_half(uchar(vb_ptr[key * v_strides[2] + col * v_strides[3]]));
                    }
                    Ks[col * LDK + r] = kv;
                    Vs[r * LDV + col] = vv;
                }
            }
        } else if (vec_kv) {
            for (int c = int(tid); c < BK * D / 8; c += NT) {
                const int r = c / (D / 8);
                const int col = (c % (D / 8)) * 8;
                const int key = k0 + r;
                uint4 kw = uint4(0);
                uint4 vw = uint4(0);
                if (key < k_end) {
                    kw = *(const device uint4*)(kb_ptr + key * k_strides[2] + col);
                    vw = *(const device uint4*)(vb_ptr + key * v_strides[2] + col);
                }
                threadgroup half* kd = Ks + col * LDK + r;
                threadgroup half* vd = Vs + r * LDV + col;
                for (int j = 0; j < 4; ++j) {
                    kd[(2 * j) * LDK] = half(as_type<float>(kw[j] << 16));
                    kd[(2 * j + 1) * LDK] = half(as_type<float>(kw[j] & 0xFFFF0000u));
                    vd[2 * j] = half(as_type<float>(vw[j] << 16));
                    vd[2 * j + 1] = half(as_type<float>(vw[j] & 0xFFFF0000u));
                }
            }
        } else {
            for (int c = int(tid); c < BK * D; c += NT) {
                const int r = c / D;
                const int col = c % D;
                const int key = k0 + r;
                float kv = 0.0f;
                float vv = 0.0f;
                if (key < k_end) {
                    kv = float(kb_ptr[key * k_strides[2] + col * k_strides[3]]);
                    vv = float(vb_ptr[key * v_strides[2] + col * v_strides[3]]);
                }
                Ks[col * LDK + r] = half(kv);
                Vs[r * LDV + col] = half(vv);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (qt * BQ + int(sg) * 8 >= n_rows) {
            continue;
        }

        simdgroup_float8x8 Sf[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            Sf[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int db = 0; db < D / 8; ++db) {
                simdgroup_half8x8 Kt;
                simdgroup_load(Kt, Ks + db * 8 * LDK + kb * 8, LDK);
                simdgroup_multiply_accumulate(Sf[kb], Qf[db], Kt, Sf[kb]);
            }
        }

        float rmax = -INFINITY;
        if (WINDOW == 0 && k0 >= lp && k0 + BK <= k_end && k0 + BK <= a_first - R + 1) {
            for (int kb = 0; kb < BK / 8; ++kb) {
                float s0 = row_ok ? Sf[kb].thread_elements()[0] * qscale : -INFINITY;
                float s1 = row_ok ? Sf[kb].thread_elements()[1] * qscale : -INFINITY;
                Sf[kb].thread_elements()[0] = s0;
                Sf[kb].thread_elements()[1] = s1;
                rmax = max(rmax, max(s0, s1));
            }
        } else {
            for (int kb = 0; kb < BK / 8; ++kb) {
                for (int e = 0; e < 2; ++e) {
                    const int key = k0 + kb * 8 + fn + e;
                    const int d = a_row - key;
                    const bool vis = row_ok && key < k_end && d >= 0 && key >= lp
                        && (WINDOW == 0 || d < WINDOW);
                    float s = Sf[kb].thread_elements()[e] * qscale;
                    if (vis) {
                        if (d < R) {
                            s += float(relp[d]) * LOG2E;
                        }
                    } else {
                        s = -INFINITY;
                    }
                    Sf[kb].thread_elements()[e] = s;
                    rmax = max(rmax, s);
                }
            }
        }
        rmax = max(rmax, simd_shuffle_xor(rmax, 1));
        rmax = max(rmax, simd_shuffle_xor(rmax, 8));
        const float m_new = max(m, rmax);
        const float factor = (m == -INFINITY || m_new == -INFINITY)
            ? 0.0f : fast::exp2(m - m_new);
        float rsum = 0.0f;
        simdgroup_half8x8 Pf[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const float sv = Sf[kb].thread_elements()[e];
                const float p = (sv == -INFINITY) ? 0.0f : fast::exp2(sv - m_new);
                Pf[kb].thread_elements()[e] = half(p);
                rsum += p;
            }
        }
        rsum += simd_shuffle_xor(rsum, 1);
        rsum += simd_shuffle_xor(rsum, 8);
        l = l * factor + rsum;
        m = m_new;
        for (int db = 0; db < D / 8; ++db) {
            Of[db].thread_elements()[0] *= factor;
            Of[db].thread_elements()[1] *= factor;
            for (int kb = 0; kb < BK / 8; ++kb) {
                simdgroup_half8x8 Vm;
                simdgroup_load(Vm, Vs + kb * 8 * LDV + db * 8, LDV);
                simdgroup_multiply_accumulate(Of[db], Pf[kb], Vm, Of[db]);
            }
        }
    }

    if (row_ok) {
        if (SINGLE) {
            const float inv = 1.0f / l;
            size_t base_o = (((size_t)b * HQ + h) * L + q_pos) * D;
            for (int db = 0; db < D / 8; ++db) {
                out_o[base_o + db * 8 + fn] = OT(Of[db].thread_elements()[0] * inv);
                out_o[base_o + db * 8 + fn + 1] = OT(Of[db].thread_elements()[1] * inv);
            }
        } else {
            size_t base_ml = (((size_t)b * HQ + h) * L + q_pos) * NS + sp;
            if (fn == 0) {
                out_m[base_ml] = m;
                out_l[base_ml] = l;
            }
            size_t base_o = base_ml * D;
            for (int db = 0; db < D / 8; ++db) {
                out_o[base_o + db * 8 + fn] = OT(Of[db].thread_elements()[0]);
                out_o[base_o + db * 8 + fn + 1] = OT(Of[db].thread_elements()[1]);
            }
        }
    }
"""

_GQA_DECODE_RED_SOURCE = """
    constexpr float LOG2E = 1.4426950408889634f;
    const uint lane = thread_index_in_simdgroup;
    const int idx = threadgroup_position_in_grid.x;
    const int q_pos = idx % L;
    const int bh = idx / L;
    const int h = bh % HQ;
    const int b = bh / HQ;

    constexpr int D4 = D / 32;
    float m = float(sink[(b * L + q_pos) * HQ + h]) * LOG2E;
    float l = 1.0f;
    float ov[D4];
    for (int i = 0; i < D4; ++i) ov[i] = 0.0f;

    size_t ml_base = (size_t)idx * NS;
    for (int sp = 0; sp < NS; ++sp) {
        float lp = part_l[ml_base + sp];
        if (lp == 0.0f) continue;
        float mp = part_m[ml_base + sp];
        float m_new = max(m, mp);
        float a0 = fast::exp2(m - m_new);
        float a1 = fast::exp2(mp - m_new);
        l = l * a0 + lp * a1;
        m = m_new;
        const device float* pp = part_o + (ml_base + sp) * D;
        for (int i = 0; i < D4; ++i) {
            ov[i] = ov[i] * a0 + pp[int(lane) * D4 + i] * a1;
        }
    }
    float inv = 1.0f / l;
    device T* dst = out + (size_t)idx * D;
    for (int i = 0; i < D4; ++i) {
        dst[int(lane) * D4 + i] = T(ov[i] * inv);
    }
"""

# Register-streaming Split-K GQA decode kernel for L == 1 (1..6 query heads per
# KV head): zero threadgroup KV staging and zero barriers inside the key loop,
# reading each KV vector once across all G query heads via coalesced 4-byte
# FP8 E4M3 loads (or BF16 loads).
_GQA_DIRECT_DECODE_PART_SOURCE = """
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr int D4 = D / 32;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const int sp = threadgroup_position_in_grid.x;
    const int hkv = threadgroup_position_in_grid.y;
    const int b = threadgroup_position_in_grid.z;

    const int S = dims[0];
    const int NS = dims[1];
    const int lp = dims[2 + b];
    const float qscale = scale[0] * LOG2E;

    const int a_row = S - 1;
    int lo = WINDOW > 0 ? max(0, a_row - WINDOW + 1) : 0;
    lo = max(lo, lp);
    const int nk = max(0, a_row + 1 - lo);
    const int per_sp = (nk + NS - 1) / NS;
    const int k_start = lo + sp * per_sp;
    const int k_end = min(a_row + 1, k_start + per_sp);

    float qv[G][D4];
    float ov[G][D4];
    float mv[G];
    float lv[G];
    const device T* relp[G];
    const int d_off = int(lane) * D4;
    for (int g_idx = 0; g_idx < G; ++g_idx) {
        int h = hkv * G + g_idx;
        const device T* qp = q + b * q_strides[0] + h * q_strides[1]
            + d_off * q_strides[3];
        for (int i = 0; i < D4; ++i) {
            qv[g_idx][i] = float(qp[i * q_strides[3]]) * qscale;
            ov[g_idx][i] = 0.0f;
        }
        mv[g_idx] = -INFINITY;
        lv[g_idx] = 0.0f;
        relp[g_idx] = rel + ((size_t)b * HQ + h) * R;
    }

    const device KT* kb_ptr = k + b * k_strides[0] + hkv * k_strides[1];
    const device KT* vb_ptr = v + b * v_strides[0] + hkv * v_strides[1];
    const size_t ks2 = k_strides[2];
    const size_t vs2 = v_strides[2];
    const size_t ks3 = k_strides[3];
    const size_t vs3 = v_strides[3];
    const bool vec_kv = FP8_KV && ks3 == 1 && vs3 == 1 && (D4 == 4)
        && (ks2 % 4 == 0) && (vs2 % 4 == 0);

    for (int key = k_start + int(sg); key < k_end; key += NSG) {
        float kv[D4];
        float vv[D4];
        if (FP8_KV && vec_kv) {
            uchar4 kw = *(const device uchar4*)(kb_ptr + key * ks2 + d_off);
            uchar4 vw = *(const device uchar4*)(vb_ptr + key * vs2 + d_off);
            kv[0] = float(FP8_E4M3_LUT[kw.x]);
            kv[1] = float(FP8_E4M3_LUT[kw.y]);
            kv[2] = float(FP8_E4M3_LUT[kw.z]);
            kv[3] = float(FP8_E4M3_LUT[kw.w]);
            vv[0] = float(FP8_E4M3_LUT[vw.x]);
            vv[1] = float(FP8_E4M3_LUT[vw.y]);
            vv[2] = float(FP8_E4M3_LUT[vw.z]);
            vv[3] = float(FP8_E4M3_LUT[vw.w]);
        } else if (FP8_KV) {
            const device KT* kp = kb_ptr + key * ks2;
            const device KT* vp = vb_ptr + key * vs2;
            for (int i = 0; i < D4; ++i) {
                kv[i] = float(FP8_E4M3_LUT[uchar(kp[(d_off + i) * ks3])]);
                vv[i] = float(FP8_E4M3_LUT[uchar(vp[(d_off + i) * vs3])]);
            }
        } else {
            const device KT* kp = kb_ptr + key * ks2;
            const device KT* vp = vb_ptr + key * vs2;
            for (int i = 0; i < D4; ++i) {
                kv[i] = float(kp[(d_off + i) * ks3]);
                vv[i] = float(vp[(d_off + i) * vs3]);
            }
        }
        const int d = a_row - key;
        const bool has_rel = d < R;
        for (int g_idx = 0; g_idx < G; ++g_idx) {
            float dot = 0.0f;
            for (int i = 0; i < D4; ++i) dot += kv[i] * qv[g_idx][i];
            dot = simd_sum(dot);
            if (has_rel) dot += float(relp[g_idx][d]) * LOG2E;
            float m_new = max(mv[g_idx], dot);
            float f_old = (mv[g_idx] == -INFINITY) ? 0.0f : fast::exp2(mv[g_idx] - m_new);
            float p = fast::exp2(dot - m_new);
            lv[g_idx] = lv[g_idx] * f_old + p;
            mv[g_idx] = m_new;
            for (int i = 0; i < D4; ++i) {
                ov[g_idx][i] = ov[g_idx][i] * f_old + p * vv[i];
            }
        }
    }

    threadgroup float tg_m[NSG][G];
    threadgroup float tg_l[NSG][G];
    threadgroup float tg_o[NSG][G][D];
    for (int g_idx = 0; g_idx < G; ++g_idx) {
        if (lane == 0) {
            tg_m[sg][g_idx] = mv[g_idx];
            tg_l[sg][g_idx] = lv[g_idx];
        }
        for (int i = 0; i < D4; ++i) {
            tg_o[sg][g_idx][d_off + i] = ov[g_idx][i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        for (int g_idx = 0; g_idx < G; ++g_idx) {
            int h = hkv * G + g_idx;
            float m = -INFINITY;
            float l = 0.0f;
            float oo[D4];
            for (int i = 0; i < D4; ++i) oo[i] = 0.0f;
            for (int s = 0; s < NSG; ++s) {
                float ls = tg_l[s][g_idx];
                if (ls == 0.0f) continue;
                float ms = tg_m[s][g_idx];
                float mn = max(m, ms);
                float a0 = (m == -INFINITY) ? 0.0f : fast::exp2(m - mn);
                float a1 = fast::exp2(ms - mn);
                l = l * a0 + ls * a1;
                m = mn;
                for (int i = 0; i < D4; ++i) {
                    oo[i] = oo[i] * a0 + tg_o[s][g_idx][d_off + i] * a1;
                }
            }
            size_t base_ml = ((size_t)b * HQ + h) * NS + sp;
            if (lane == 0) {
                out_m[base_ml] = m;
                out_l[base_ml] = l;
            }
            size_t base_o = base_ml * D;
            for (int i = 0; i < D4; ++i) {
                out_o[base_o + d_off + i] = oo[i];
            }
        }
    }
"""

_gqa_decode_part_kernel = None
_gqa_direct_part_kernel = None
_gqa_decode_red_kernel = None


def gqa_flash_decode_attention(
    q, k, v, rel, sink, *, scale, window, left_padding=None
):
    """Fused Split-K GQA-packed banded-sink attention for BF16 verify/decode."""
    global _gqa_decode_part_kernel, _gqa_direct_part_kernel, _gqa_decode_red_kernel
    b, hq, length, d = q.shape
    hkv, s = k.shape[1], k.shape[2]
    g = hq // hkv
    r = rel.shape[-1]
    fp8_kv = k.dtype == mx.uint8
    if _gqa_decode_red_kernel is None:
        _gqa_decode_red_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_gqa_decode_red",
            input_names=["part_o", "part_m", "part_l", "sink"],
            output_names=["out"],
            source=_GQA_DECODE_RED_SOURCE,
        )
    lp = (
        left_padding.astype(mx.int32)
        if left_padding is not None
        else mx.zeros((b,), dtype=mx.int32)
    )
    sc = mx.array([scale], dtype=mx.float32)
    sink_f32 = mx.contiguous(sink.astype(mx.float32))
    rel_c = mx.contiguous(rel)
    nk_est = min(s, window + length - 1) if window else s
    if length == 1 and g <= 6 and d <= 128 and (fp8_kv or nk_est >= 2048):
        if _gqa_direct_part_kernel is None:
            _gqa_direct_part_kernel = mx.fast.metal_kernel(
                name="omlx_inkling2_gqa_direct_part",
                input_names=["q", "k", "v", "rel", "dims", "scale"],
                output_names=["out_o", "out_m", "out_l"],
                source=_GQA_DIRECT_DECODE_PART_SOURCE,
                header=_FP8_E4M3_HEADER,
                ensure_row_contiguous=False,
            )
        nsg = 4
        base_tgs = b * hkv
        ns = (
            1
            if nk_est <= 256
            else min(32, max(1, (nk_est + 255) // 256))
        )
        if nk_est >= 2048 and ns < 16 and base_tgs <= 16:
            ns = 16
        if nk_est >= 8192 and ns < 32 and base_tgs <= 16:
            ns = 32
        dims = mx.concatenate([mx.array([s, ns], dtype=mx.int32), lp])
        po, pm, pl = _gqa_direct_part_kernel(
            inputs=[q, k, v, rel_c, dims, sc],
            template=[
                ("T", q.dtype),
                ("KT", k.dtype),
                ("D", d),
                ("HQ", hq),
                ("G", g),
                ("R", r),
                ("WINDOW", int(window)),
                ("NSG", nsg),
                ("FP8_KV", 1 if fp8_kv else 0),
            ],
            grid=(ns * nsg * 32, hkv, b),
            threadgroup=(nsg * 32, 1, 1),
            output_shapes=[
                (b, hq, 1, ns, d),
                (b, hq, 1, ns),
                (b, hq, 1, ns),
            ],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
        )
        return _gqa_decode_red_kernel(
            inputs=[po, pm, pl, sink_f32],
            template=[("T", q.dtype), ("D", d), ("HQ", hq), ("L", 1), ("NS", ns)],
            grid=(b * hq * 32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(b, hq, 1, d)],
            output_dtypes=[q.dtype],
        )[0]
    if _gqa_decode_part_kernel is None:
        _gqa_decode_part_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_gqa_decode_part",
            input_names=["q", "k", "v", "rel", "sink", "dims", "scale"],
            output_names=["out_o", "out_m", "out_l"],
            source=_GQA_DECODE_PART_SOURCE,
            header=_FP8_E4M3_HEADER,
            ensure_row_contiguous=False,
        )
    n_rows = g * length
    nsg = min(4, max(1, (n_rows + 7) // 8))
    bq = nsg * 8
    nqt = (n_rows + bq - 1) // bq
    base_tgs = b * hkv * nqt
    ns = (
        1
        if nk_est <= 128
        else min(
            32,
            max(1, min((nk_est + 63) // 64, (64 + base_tgs - 1) // base_tgs)),
        )
    )
    if nk_est >= 512 and ns < 8 and base_tgs <= 16:
        ns = 8
    if nk_est >= 2048 and ns < 16 and base_tgs <= 16:
        ns = 16
    if nk_est >= 8192 and ns < 32 and base_tgs <= 16:
        ns = 32
    dims = mx.concatenate([mx.array([length, s, ns], dtype=mx.int32), lp])
    tpl = [
        ("T", q.dtype),
        ("KT", k.dtype),
        ("OT", q.dtype if ns == 1 else mx.float32),
        ("D", d),
        ("HQ", hq),
        ("G", g),
        ("R", r),
        ("WINDOW", int(window)),
        ("NSG", nsg),
        ("SINGLE", 1 if ns == 1 else 0),
        ("FP8_KV", 1 if fp8_kv else 0),
    ]
    nt = nsg * 32
    if ns == 1:
        out, _, _ = _gqa_decode_part_kernel(
            inputs=[q, k, v, rel_c, sink_f32, dims, sc],
            template=tpl,
            grid=(nt, hkv * nqt, b),
            threadgroup=(nt, 1, 1),
            output_shapes=[(b, hq, length, d), (1,), (1,)],
            output_dtypes=[q.dtype, mx.float32, mx.float32],
        )
        return out
    po, pm, pl = _gqa_decode_part_kernel(
        inputs=[q, k, v, rel_c, sink_f32, dims, sc],
        template=tpl,
        grid=(ns * nt, hkv * nqt, b),
        threadgroup=(nt, 1, 1),
        output_shapes=[
            (b, hq, length, ns, d),
            (b, hq, length, ns),
            (b, hq, length, ns),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    return _gqa_decode_red_kernel(
        inputs=[po, pm, pl, sink_f32],
        template=[("T", q.dtype), ("D", d), ("HQ", hq), ("L", length), ("NS", ns)],
        grid=(b * hq * length * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(b, hq, length, d)],
        output_dtypes=[q.dtype],
    )[0]
'''

_LOOP_CTRL_KERNELS_SRC = '''

_NORM_PAIR_SOURCE = """
    uint row_idx = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint total_rows = N * ROWS;
    uint is_msg = row_idx >= total_rows ? 1u : 0u;
    uint nr = is_msg ? (row_idx - total_rows) : row_idx;
    uint n = nr / ROWS;
    uint r = nr % ROWS;
    const device float* src = (is_msg ? msg : x) + (size_t)nr * H;
    float ss = 0.0f;
    for (uint h = tid; h < H; h += TG) {
        float v = src[h];
        ss += v * v;
    }
    threadgroup float red[TG / 32];
    ss = simd_sum(ss);
    if (lane == 0) red[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < TG / 32; ++i) total += red[i];
    float inv = metal::rsqrt(total / float(H) + 1e-6f);
    device OT* dst = out + ((size_t)(n * 2 + is_msg) * ROWS + r) * H;
    for (uint h = tid; h < H; h += TG) {
        bfloat bv = bfloat(src[h] * inv);
        dst[h] = OT(bv);
    }
"""

_LOOP_DOWN_SOURCE = """
    uint nr = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    uint n = nr / R;
    uint r = nr % R;
    float acc[ROWS];
    for (uint i = 0; i < ROWS; ++i) acc[i] = 0.0f;
    for (uint part = 0; part < 2; ++part) {
        const device WT* wp = down_t + ((size_t)(n * 2 + part) * R + r) * H;
        const device bfloat* zp = z + (size_t)(n * 2 + part) * ROWS * H;
        for (uint h = lane * 4; h < H; h += 128) {
            float w0 = float(wp[h + 0]);
            float w1 = float(wp[h + 1]);
            float w2 = float(wp[h + 2]);
            float w3 = float(wp[h + 3]);
            for (uint row = 0; row < ROWS; ++row) {
                const device bfloat* zr = zp + (size_t)row * H + h;
                acc[row] += float(zr[0]) * w0 + float(zr[1]) * w1
                    + float(zr[2]) * w2 + float(zr[3]) * w3;
            }
        }
    }
    for (uint row = 0; row < ROWS; ++row) {
        float s = simd_sum(acc[row]);
        if (lane == 0) {
            low[((size_t)n * ROWS + row) * R + r] = bfloat(s);
        }
    }
"""

_LOOP_GMIX_SOURCE = """
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint nh = threadgroup_position_in_grid.x * 4 + sg;
    if (nh >= N * H) return;
    uint n = nh / H;
    uint h = nh % H;
    constexpr uint PER = (R + 31) / 32;
    float wg[PER], wf[PER];
    const device WT* gp = gf_t + ((size_t)nh * 2 + 0) * R;
    const device WT* fp = gf_t + ((size_t)nh * 2 + 1) * R;
    for (uint j = 0; j < PER; ++j) {
        uint idx = lane + j * 32;
        wg[j] = idx < R ? float(gp[idx]) : 0.0f;
        wf[j] = idx < R ? float(fp[idx]) : 0.0f;
    }
    for (uint row = 0; row < ROWS; ++row) {
        const device bfloat* lp = low + ((size_t)n * ROWS + row) * R;
        float dg = 0.0f;
        float df = 0.0f;
        for (uint j = 0; j < PER; ++j) {
            uint idx = lane + j * 32;
            float lv = idx < R ? float(lp[idx]) : 0.0f;
            dg += lv * wg[j];
            df += lv * wf[j];
        }
        dg = simd_sum(dg);
        df = simd_sum(df);
        if (lane == 0) {
            float g = metal::tanh(metal::clamp(dg, -20.0f, 20.0f));
            float f = metal::tanh(metal::clamp(df, -20.0f, 20.0f));
            size_t pos = ((size_t)n * ROWS + row) * H + h;
            out[pos] = (1.0f - f) * x[pos] + g * msg[pos];
        }
    }
"""

_TANH_MIX_SOURCE = """
    uint idx = thread_position_in_grid.x;
    if (idx >= N * ROWS * H) return;
    uint nr = idx / H;
    uint h = idx % H;
    size_t gbase = (size_t)nr * (2 * H);
    float g = metal::tanh(metal::clamp(float(gates[gbase + h]), -20.0f, 20.0f));
    float f = metal::tanh(metal::clamp(float(gates[gbase + H + h]), -20.0f, 20.0f));
    out[idx] = (1.0f - f) * x[idx] + g * msg[idx];
"""

_norm_pair_kernel = None
_loop_down_kernel = None
_loop_gmix_kernel = None
_tanh_mix_kernel = None
_LOOP_FUSED_MAX_ROWS = 16


def pack_loop_controller(down: mx.array, gain_forget: mx.array):
    """Transpose loop controller weights for coalesced decode/verify reads."""
    n, _, h, r = down.shape
    exact_bf16 = (
        mx.array_equal(down.astype(mx.bfloat16).astype(mx.float32), down).item()
        and mx.array_equal(
            gain_forget.astype(mx.bfloat16).astype(mx.float32), gain_forget
        ).item()
    )
    dtype = mx.bfloat16 if exact_bf16 else down.dtype
    down_t = mx.contiguous(down.astype(dtype).transpose(0, 1, 3, 2))
    gf_t = mx.contiguous(
        gain_forget.astype(dtype).reshape(n, r, 2, h).transpose(0, 3, 2, 1)
    )
    mx.eval(down_t, gf_t)
    return down_t, gf_t


def mix_messages_gpu(
    x: mx.array,
    message: mx.array,
    down: mx.array,
    gain_forget: mx.array,
    down_t: mx.array | None = None,
    gf_t: mx.array | None = None,
) -> mx.array:
    """Fused GPU loop-controller message mixing; exact FP32 accumulation."""
    global _norm_pair_kernel, _loop_down_kernel, _loop_gmix_kernel, _tanh_mix_kernel
    n, b, length, hidden = x.shape
    rows = b * length
    rank = down.shape[-1]
    if _norm_pair_kernel is None:
        _norm_pair_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_norm_pair",
            input_names=["x", "msg"],
            output_names=["out"],
            source=_NORM_PAIR_SOURCE,
        )
    if (
        rows <= _LOOP_FUSED_MAX_ROWS
        and hidden % 4 == 0
        and down_t is not None
        and gf_t is not None
    ):
        if _loop_down_kernel is None:
            _loop_down_kernel = mx.fast.metal_kernel(
                name="omlx_inkling2_loop_down",
                input_names=["z", "down_t"],
                output_names=["low"],
                source=_LOOP_DOWN_SOURCE,
            )
            _loop_gmix_kernel = mx.fast.metal_kernel(
                name="omlx_inkling2_loop_gmix",
                input_names=["x", "msg", "low", "gf_t"],
                output_names=["out"],
                source=_LOOP_GMIX_SOURCE,
            )
        z = _norm_pair_kernel(
            inputs=[x.astype(mx.float32), message.astype(mx.float32)],
            template=[
                ("OT", mx.bfloat16),
                ("N", n),
                ("ROWS", rows),
                ("H", hidden),
                ("TG", 256),
            ],
            grid=(2 * n * rows * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(n, 2, rows, hidden)],
            output_dtypes=[mx.bfloat16],
        )[0]
        low = _loop_down_kernel(
            inputs=[z, down_t],
            template=[
                ("WT", down_t.dtype),
                ("R", rank),
                ("H", hidden),
                ("ROWS", rows),
            ],
            grid=(n * rank * 32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(n, rows, rank)],
            output_dtypes=[mx.bfloat16],
        )[0]
        return _loop_gmix_kernel(
            inputs=[x.astype(mx.float32), message.astype(mx.float32), low, gf_t],
            template=[
                ("WT", gf_t.dtype),
                ("N", n),
                ("H", hidden),
                ("R", rank),
                ("ROWS", rows),
            ],
            grid=(((n * hidden + 3) // 4) * 128, 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(n, b, length, hidden)],
            output_dtypes=[mx.float32],
        )[0]

    if _tanh_mix_kernel is None:
        _tanh_mix_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_tanh_mix",
            input_names=["x", "msg", "gates"],
            output_names=["out"],
            source=_TANH_MIX_SOURCE,
        )
    xf = x.astype(mx.float32)
    mf = message.astype(mx.float32)
    z = _norm_pair_kernel(
        inputs=[xf, mf],
        template=[
            ("OT", mx.float32),
            ("N", n),
            ("ROWS", rows),
            ("H", hidden),
            ("TG", 256),
        ],
        grid=(2 * n * rows * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(2 * n, rows, hidden)],
        output_dtypes=[mx.float32],
    )[0]
    low = (z @ down.reshape(2 * n, hidden, rank)).reshape(n, 2, rows, rank).sum(axis=1)
    low = low.astype(mx.bfloat16).astype(mx.float32)
    gates = low @ gain_forget
    return _tanh_mix_kernel(
        inputs=[xf, mf, gates],
        template=[("N", n), ("ROWS", rows), ("H", hidden)],
        grid=(n * rows * hidden, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n, b, length, hidden)],
        output_dtypes=[mx.float32],
    )[0]
'''

_PREFILL_FAST_KERNELS_SRC = '''

_EXPERT_SWIGLU_SOURCE = """
    uint idx = thread_position_in_grid.x;
    if (idx >= M * I) return;
    uint m = idx / I;
    uint i = idx % I;
    size_t base = (size_t)m * (2 * I) + 2 * i;
    float g = float(y[base]);
    float u = float(y[base + 1]);
    if (HAS_SCALE) {
        T sc = T(gscale[eidx[m]] * params[2]);
        g = float(T(g * float(sc)));
        u = float(T(u * float(sc)));
    }
    float gcap = params[0];
    float ucap = params[1];
    float eg = metal::precise::exp(metal::clamp(-g, -80.0f, 80.0f));
    float sig = (g < -80.0f) ? 0.0f : (1.0f / (1.0f + eg));
    float tg = metal::precise::tanh(metal::clamp(g / gcap, -20.0f, 20.0f)) * gcap;
    float tu = metal::precise::tanh(metal::clamp(u / ucap, -20.0f, 20.0f)) * ucap;
    out[idx] = T(tg * sig * tu);
"""

_EXPERT_COMBINE_SOURCE = """
    uint idx = thread_position_in_grid.x;
    if (idx >= ROWS * H) return;
    uint row = idx / H;
    uint h = idx % H;
    float acc = 0.0f;
    for (uint k = 0; k < K; ++k) {
        size_t rk = (size_t)row * K + k;
        float w = weights[rk];
        if (HAS_SCALE) {
            w *= gscale[inds[rk]] * inv_denom[0];
        }
        acc += float(z[rk * H + h]) * w;
    }
    out[idx] = T(acc);
"""

_STATIC_CONV4_SOURCE = """
    uint idx = thread_position_in_grid.x;
    if (idx >= B * L * C) return;
    uint c = idx % C;
    uint bl = idx / C;
    uint l = bl % L;
    uint b = bl / L;
    const device float* st = state + (size_t)b * 3 * C;
    const device T* xb = x + (size_t)b * L * C;
    const device WT* wp = w + (size_t)c * 4;
    float x0 = (l >= 3) ? float(xb[(size_t)(l - 3) * C + c]) : st[(size_t)l * C + c];
    float x1 = (l >= 2) ? float(xb[(size_t)(l - 2) * C + c]) : st[(size_t)(l + 1) * C + c];
    float x2 = (l >= 1) ? float(xb[(size_t)(l - 1) * C + c]) : st[(size_t)(l + 2) * C + c];
    float x3 = float(xb[(size_t)l * C + c]);
    float conv = float(wp[0]) * x0 + float(wp[1]) * x1
        + float(wp[2]) * x2 + float(wp[3]) * x3;
    float y = (float(WT(conv)) + x3) * scale[0];
    out[idx] = T(y);
    if (L >= 3) {
        if (l >= L - 3) {
            next_state[((size_t)b * 3 + (l - (L - 3))) * C + c] = x3;
        }
    } else if (l == L - 1) {
        device float* nst = next_state + (size_t)b * 3 * C;
        if (L == 1) {
            nst[0 * C + c] = st[1 * C + c];
            nst[1 * C + c] = st[2 * C + c];
            nst[2 * C + c] = x3;
        } else {
            nst[0 * C + c] = st[2 * C + c];
            nst[1 * C + c] = x2;
            nst[2 * C + c] = x3;
        }
    }
"""

_expert_swiglu_kernel = None
_expert_combine_kernel = None
_static_conv4_kernel = None
_DUMMY_F32 = None


def _dummy_f32() -> mx.array:
    global _DUMMY_F32
    if _DUMMY_F32 is None:
        _DUMMY_F32 = mx.zeros((1,), dtype=mx.float32)
    return _DUMMY_F32


def expert_softcapped_swiglu(
    y: mx.array,
    idx: mx.array,
    gscale: mx.array | None,
    params: mx.array,
) -> mx.array:
    """Fused optional NVFP4 post-scale, de-interleave, and softcapped SwiGLU."""
    global _expert_swiglu_kernel
    if _expert_swiglu_kernel is None:
        _expert_swiglu_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_expert_swiglu",
            input_names=["y", "eidx", "gscale", "params"],
            output_names=["out"],
            source=_EXPERT_SWIGLU_SOURCE,
        )
    inter = y.shape[-1] // 2
    m = idx.size
    has_scale = 1 if gscale is not None else 0
    return _expert_swiglu_kernel(
        inputs=[y, idx.astype(mx.uint32), gscale if has_scale else _dummy_f32(), params],
        template=[("T", y.dtype), ("M", m), ("I", inter), ("HAS_SCALE", has_scale)],
        grid=(m * inter, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(*y.shape[:-1], inter)],
        output_dtypes=[y.dtype],
    )[0]


def expert_weighted_combine(
    z: mx.array,
    inds: mx.array,
    weights: mx.array,
    gscale: mx.array | None,
    inv_denom: mx.array,
) -> mx.array:
    """Fused optional NVFP4 post-scale and FP32 weighted reduction over top-k experts."""
    global _expert_combine_kernel
    if _expert_combine_kernel is None:
        _expert_combine_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_expert_combine",
            input_names=["z", "inds", "weights", "gscale", "inv_denom"],
            output_names=["out"],
            source=_EXPERT_COMBINE_SOURCE,
        )
    rows, top_k, hidden = z.shape
    has_scale = 1 if gscale is not None else 0
    return _expert_combine_kernel(
        inputs=[
            z,
            inds.astype(mx.uint32),
            weights.astype(mx.float32),
            gscale if has_scale else _dummy_f32(),
            inv_denom,
        ],
        template=[
            ("T", z.dtype),
            ("ROWS", rows),
            ("H", hidden),
            ("K", top_k),
            ("HAS_SCALE", has_scale),
        ],
        grid=(rows * hidden, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, hidden)],
        output_dtypes=[z.dtype],
    )[0]


def static_conv4_fused(
    x: mx.array,
    w: mx.array,
    state: mx.array,
    scale: mx.array,
) -> tuple[mx.array, mx.array]:
    """Fused K=4 depthwise causal conv, residual add, scale, and next state."""
    global _static_conv4_kernel
    if _static_conv4_kernel is None:
        _static_conv4_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_static_conv4",
            input_names=["x", "w", "state", "scale"],
            output_names=["out", "next_state"],
            source=_STATIC_CONV4_SOURCE,
        )
    b, length, channels = x.shape
    out, next_state = _static_conv4_kernel(
        inputs=[x, w, state.astype(mx.float32), scale],
        template=[
            ("T", x.dtype),
            ("WT", w.dtype),
            ("B", b),
            ("L", length),
            ("C", channels),
        ],
        grid=(b * length * channels, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(b, length, channels), (b, 3, channels)],
        output_dtypes=[x.dtype, mx.float32],
    )
    return out, next_state
'''


def patch_kernels(txt: str, features: set[str]) -> str:
    if "fp8_kv_full" in features:
        txt = _replace_once(
            txt,
            "    return mx.fast.scaled_dot_product_attention(\n"
            "        q, k, v, scale=scale, mask=mask, sinks=sink[0, 0].astype(q.dtype)\n"
            "    )\n",
            "    return mx.fast.scaled_dot_product_attention(\n"
            "        q, k, v, scale=scale, mask=mask, sinks=sink[0, 0].astype(q.dtype)\n"
            "    )\n"
            + _FP8_HELPERS_SRC,
            "kernels fp8 helpers",
        )
        txt = _replace_once(
            txt,
            "    const device T* kb_ptr = k + b * k_strides[0] + hk * k_strides[1];\n"
            "    const device T* vb_ptr = v + b * v_strides[0] + hk * v_strides[1];\n"
            "    const bool vec_kv = k_strides[3] == 1 && v_strides[3] == 1;\n\n"
            "    for (int k0 = k_begin; k0 <= a_last; k0 += BK) {\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "        if (vec_kv) {\n",
            "    const device KT* kb_ptr = k + b * k_strides[0] + hk * k_strides[1];\n"
            "    const device KT* vb_ptr = v + b * v_strides[0] + hk * v_strides[1];\n"
            "    const bool vec_kv = FP8_KV\n"
            "        ? (k_strides[3] == 1 && v_strides[3] == 1\n"
            "           && (k_strides[2] % 16 == 0) && (v_strides[2] % 16 == 0))\n"
            "        : (k_strides[3] == 1 && v_strides[3] == 1);\n\n"
            "    for (int k0 = k_begin; k0 <= a_last; k0 += BK) {\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "        if (FP8_KV) {\n"
            "            if (vec_kv) {\n"
            "                for (int c = tid; c < BK * D / 16; c += NT) {\n"
            "                    const int r = c / (D / 16);\n"
            "                    const int col = (c % (D / 16)) * 16;\n"
            "                    const int key = k0 + r;\n"
            "                    uint4 kw = uint4(0);\n"
            "                    uint4 vw = uint4(0);\n"
            "                    if (key < S) {\n"
            "                        kw = *(const device uint4*)(kb_ptr + key * k_strides[2] + col);\n"
            "                        vw = *(const device uint4*)(vb_ptr + key * v_strides[2] + col);\n"
            "                    }\n"
            "                    threadgroup half* kd = Ks + col * LDK + r;\n"
            "                    threadgroup half* vd = Vs + r * LDV + col;\n"
            "                    for (int j = 0; j < 4; ++j) {\n"
            "                        uint kword = kw[j];\n"
            "                        uint vword = vw[j];\n"
            "                        kd[(4 * j + 0) * LDK] = fp8_e4m3_to_half(uchar(kword & 0xFFu));\n"
            "                        kd[(4 * j + 1) * LDK] = fp8_e4m3_to_half(uchar((kword >> 8) & 0xFFu));\n"
            "                        kd[(4 * j + 2) * LDK] = fp8_e4m3_to_half(uchar((kword >> 16) & 0xFFu));\n"
            "                        kd[(4 * j + 3) * LDK] = fp8_e4m3_to_half(uchar((kword >> 24) & 0xFFu));\n"
            "                        vd[4 * j + 0] = fp8_e4m3_to_half(uchar(vword & 0xFFu));\n"
            "                        vd[4 * j + 1] = fp8_e4m3_to_half(uchar((vword >> 8) & 0xFFu));\n"
            "                        vd[4 * j + 2] = fp8_e4m3_to_half(uchar((vword >> 16) & 0xFFu));\n"
            "                        vd[4 * j + 3] = fp8_e4m3_to_half(uchar((vword >> 24) & 0xFFu));\n"
            "                    }\n"
            "                }\n"
            "            } else {\n"
            "                for (int c = tid; c < BK * D; c += NT) {\n"
            "                    const int r = c / D;\n"
            "                    const int col = c % D;\n"
            "                    const int key = k0 + r;\n"
            "                    half kv = half(0.0h);\n"
            "                    half vv = half(0.0h);\n"
            "                    if (key < S) {\n"
            "                        kv = fp8_e4m3_to_half(uchar(kb_ptr[key * k_strides[2] + col * k_strides[3]]));\n"
            "                        vv = fp8_e4m3_to_half(uchar(vb_ptr[key * v_strides[2] + col * v_strides[3]]));\n"
            "                    }\n"
            "                    Ks[col * LDK + r] = kv;\n"
            "                    Vs[r * LDV + col] = vv;\n"
            "                }\n"
            "            }\n"
            "        } else if (vec_kv) {\n",
            "kernels flash fp8 load",
        )
        txt = _replace_once(
            txt,
            "_FLASH_MIN_QUERIES = 32\n",
            "_FLASH_MIN_QUERIES = 32\n" + _GQA_FLASH_DECODE_SRC,
            "kernels gqa decode",
        )
        txt = _replace_once(
            txt,
            "    b, hq, length, d = q.shape\n"
            "    hkv, s = k.shape[1], k.shape[2]\n"
            "    if _flash_kernel is None:\n"
            "        _flash_kernel = mx.fast.metal_kernel(\n"
            "            name=\"omlx_inkling2_flash_banded\",\n"
            "            input_names=[\"q\", \"k\", \"v\", \"rel\", \"sink\", \"dims\", \"scale\"],\n"
            "            output_names=[\"out\"],\n"
            "            source=_FLASH_SOURCE,\n"
            "            ensure_row_contiguous=False,\n"
            "        )\n",
            "    b, hq, length, d = q.shape\n"
            "    hkv, s = k.shape[1], k.shape[2]\n"
            "    fp8_kv = k.dtype == mx.uint8\n"
            "    if _flash_kernel is None:\n"
            "        _flash_kernel = mx.fast.metal_kernel(\n"
            "            name=\"omlx_inkling2_flash_banded\",\n"
            "            input_names=[\"q\", \"k\", \"v\", \"rel\", \"sink\", \"dims\", \"scale\"],\n"
            "            output_names=[\"out\"],\n"
            "            source=_FLASH_SOURCE,\n"
            "            header=_FP8_E4M3_HEADER,\n"
            "            ensure_row_contiguous=False,\n"
            "        )\n",
            "kernels flash init fp8",
        )
        txt = _replace_once(
            txt,
            "        template=[\n"
            "            (\"T\", q.dtype),\n"
            "            (\"D\", d),\n"
            "            (\"HQ\", hq),\n"
            "            (\"HKV\", hkv),\n"
            "            (\"R\", rel.shape[-1]),\n"
            "            (\"WINDOW\", int(window)),\n"
            "        ],\n",
            "        template=[\n"
            "            (\"T\", q.dtype),\n"
            "            (\"KT\", k.dtype),\n"
            "            (\"D\", d),\n"
            "            (\"HQ\", hq),\n"
            "            (\"HKV\", hkv),\n"
            "            (\"R\", rel.shape[-1]),\n"
            "            (\"WINDOW\", int(window)),\n"
            "            (\"FP8_KV\", 1 if fp8_kv else 0),\n"
            "        ],\n",
            "kernels flash template fp8",
        )
        txt = _replace_once(
            txt,
            "    b, length = q.shape[0], q.shape[2]\n"
            "    if b == 1 and length == 1 and left_padding is None:\n"
            "        return _decode_attention(q, k, v, rel, sink, scale, window)\n"
            "    if q.dtype == mx.bfloat16 and _use_flash(length, window):\n"
            "        return flash_banded_attention(\n"
            "            q, k, v, rel, sink, scale=scale, window=window, left_padding=left_padding\n"
            "        )\n"
            "    return _masked_attention(q, k, v, rel, sink, scale, window, left_padding)\n",
            "    b, hq, length, d = q.shape\n"
            "    fp8_kv = k.dtype == mx.uint8\n"
            "    if b == 1 and length == 1 and left_padding is None and not fp8_kv:\n"
            "        return _decode_attention(q, k, v, rel, sink, scale, window)\n"
            "    if q.dtype == mx.bfloat16 and (\n"
            "        _use_flash(length, window)\n"
            "        or (fp8_kv and length >= _FLASH_MIN_QUERIES and mx.default_device() == mx.gpu)\n"
            "    ):\n"
            "        return flash_banded_attention(\n"
            "            q, k, v, rel, sink, scale=scale, window=window, left_padding=left_padding\n"
            "        )\n"
            "    if (\n"
            "        q.dtype == mx.bfloat16\n"
            "        and length < _FLASH_MIN_QUERIES\n"
            "        and mx.default_device() == mx.gpu\n"
            "        and d % 32 == 0\n"
            "        and hq % k.shape[1] == 0\n"
            "    ):\n"
            "        return gqa_flash_decode_attention(\n"
            "            q, k, v, rel, sink, scale=scale, window=window, left_padding=left_padding\n"
            "        )\n"
            "    return _masked_attention(q, k, v, rel, sink, scale, window, left_padding)\n",
            "kernels banded_sink_attention",
        )
        txt = _replace_once(
            txt,
            "def _masked_attention(q, k, v, rel, sink, scale, window, left_padding):\n"
            '    """MLX SDPA over an additive mask that carries the bias band and sinks."""\n',
            "def _masked_attention(q, k, v, rel, sink, scale, window, left_padding):\n"
            '    """MLX SDPA over an additive mask that carries the bias band and sinks."""\n'
            "    if k.dtype == mx.uint8:\n"
            "        k = fp8_e4m3_dequantize(k, q.dtype)\n"
            "        v = fp8_e4m3_dequantize(v, q.dtype)\n",
            "kernels masked dequant",
        )

    if "prefill_fast" in features:
        txt = _replace_once(
            txt,
            "        float rmax = -INFINITY;\n"
            "        for (int kb = 0; kb < BK / 8; ++kb) {\n"
            "            for (int e = 0; e < 2; ++e) {\n"
            "                const int key = k0 + kb * 8 + fn + e;\n"
            "                const int d = a_row - key;\n"
            "                const bool vis = row_ok && d >= 0 && key >= lp\n"
            "                    && (WINDOW == 0 || d < WINDOW);\n"
            "                float s = Sf[kb].thread_elements()[e] * qscale;\n"
            "                if (vis) {\n"
            "                    if (d < R) {\n"
            "                        s += float(relp[d]) * LOG2E;\n"
            "                    }\n"
            "                } else {\n"
            "                    s = -INFINITY;\n"
            "                }\n"
            "                Sf[kb].thread_elements()[e] = s;\n"
            "                rmax = max(rmax, s);\n"
            "            }\n"
            "        }\n",
            "        float rmax = -INFINITY;\n"
            "        if (WINDOW == 0 && k0 >= lp && k0 + BK <= a_first - R + 1) {\n"
            "            for (int kb = 0; kb < BK / 8; ++kb) {\n"
            "                float s0 = Sf[kb].thread_elements()[0] * qscale;\n"
            "                float s1 = Sf[kb].thread_elements()[1] * qscale;\n"
            "                Sf[kb].thread_elements()[0] = s0;\n"
            "                Sf[kb].thread_elements()[1] = s1;\n"
            "                rmax = max(rmax, max(s0, s1));\n"
            "            }\n"
            "        } else {\n"
            "            for (int kb = 0; kb < BK / 8; ++kb) {\n"
            "                for (int e = 0; e < 2; ++e) {\n"
            "                    const int key = k0 + kb * 8 + fn + e;\n"
            "                    const int d = a_row - key;\n"
            "                    const bool vis = row_ok && d >= 0 && key >= lp\n"
            "                        && (WINDOW == 0 || d < WINDOW);\n"
            "                    float s = Sf[kb].thread_elements()[e] * qscale;\n"
            "                    if (vis) {\n"
            "                        if (d < R) {\n"
            "                            s += float(relp[d]) * LOG2E;\n"
            "                        }\n"
            "                    } else {\n"
            "                        s = -INFINITY;\n"
            "                    }\n"
            "                    Sf[kb].thread_elements()[e] = s;\n"
            "                    rmax = max(rmax, s);\n"
            "                }\n"
            "            }\n"
            "        }\n",
            "kernels flash bias skip",
        )

    if "m1_attn" in features or "m1_attn_long" in features:
        cond = (
            "b == 1 and 1 < length <= 16"
            if "m1_attn" in features
            else "b == 1 and 1 < length <= 16 and k.shape[2] > 4096"
        )
        anchor = "        return _decode_attention(q, k, v, rel, sink, scale, window)\n"
        txt = _replace_once(
            txt,
            anchor,
            anchor
            + "    if (\n"
            + f"        {cond}\n"
            + "        and q.dtype == mx.bfloat16\n"
            + "        and k.dtype == mx.bfloat16\n"
            + "        and v.dtype == mx.bfloat16\n"
            + "        and q.shape[1] == 24\n"
            + "        and k.shape[1] == 8\n"
            + "        and q.shape[3] == 128\n"
            + "    ):\n"
            + "        from inkling2_m1_kernels import verify_banded_sink_attention\n"
            + "        return verify_banded_sink_attention(\n"
            + "            q, k, v, rel, sink, scale=scale, window=window, left_padding=left_padding\n"
            + "        )\n",
            "kernels m1_attn",
        )
        txt += f'\nprint("BENCH loaded M1 verify attention kernel ({cond})", flush=True)\n'

    if "bf16_loop_ctrl" in features:
        txt += _LOOP_CTRL_KERNELS_SRC
    if "prefill_fast" in features:
        txt += _PREFILL_FAST_KERNELS_SRC
    return txt


# ---------------------------------------------------------------------------
# PR #5510 (dspark.py, selection.py) + PR #5508/#5509 (drafter.py)
# ---------------------------------------------------------------------------

_DSPARK_2BIT_KERNELS_SRC = '''
_MARKOV_Q8_ARGMAX_P1_SOURCE = """
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint tg = threadgroup_position_in_grid.x;
    const uint b = threadgroup_position_in_grid.y;
    const uint num_tg = threadgroups_per_grid.x;

    constexpr uint PER = R / 32;
    const device bfloat* lp = latent + (size_t)b * R + lane * PER;
    float lv[PER];
    float lsum = 0.0f;
    for (uint i = 0; i < PER; ++i) {
        lv[i] = float(lp[i]);
        lsum += lv[i];
    }
    const uint gi = (lane * PER) / GS;
    constexpr uint K_W = R / 4;
    constexpr uint K_GS = R / GS;

    const uint chunk = (V_UNPAD + num_tg - 1) / num_tg;
    const uint v_begin = tg * chunk;
    const uint v_end = min((uint)V_UNPAD, v_begin + chunk);
    const device bfloat* blp = base_logits + (size_t)b * V_FULL;

    float best_v = -INFINITY;
    uint best_i = 0;

    for (uint v = v_begin + sg; v < v_end; v += 8) {
        float sc = float(scales[(size_t)v * K_GS + gi]);
        float bi = float(biases[(size_t)v * K_GS + gi]);
        float d0 = 0.0f;
        const device uint* wp = w_q + (size_t)v * K_W + lane * (PER / 4);
        for (uint w = 0; w < PER / 4; ++w) {
            uint pk = wp[w];
            d0 += float((pk >> 0) & 0xFFu) * lv[w * 4 + 0]
                + float((pk >> 8) & 0xFFu) * lv[w * 4 + 1]
                + float((pk >> 16) & 0xFFu) * lv[w * 4 + 2]
                + float((pk >> 24) & 0xFFu) * lv[w * 4 + 3];
        }
        float dot = simd_sum(d0 * sc + lsum * bi);
        if (lane == 0) {
            float score = float(blp[v]) + float(fold_bias[v]) + dot;
            if (score > best_v || (score == best_v && v < best_i)) {
                best_v = score;
                best_i = v;
            }
        }
    }
    threadgroup float tg_v[8];
    threadgroup uint tg_i[8];
    if (lane == 0) {
        tg_v[sg] = best_v;
        tg_i[sg] = best_i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {
        float bv = tg_v[0];
        uint bi_idx = tg_i[0];
        for (int i = 1; i < 8; ++i) {
            if (tg_v[i] > bv || (tg_v[i] == bv && tg_i[i] < bi_idx)) {
                bv = tg_v[i];
                bi_idx = tg_i[i];
            }
        }
        part_v[(size_t)b * num_tg + tg] = bv;
        part_i[(size_t)b * num_tg + tg] = bi_idx;
    }
"""

_MARKOV_ARGMAX_P2_SOURCE = """
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_position_in_threadgroup.x;
    const uint b = threadgroup_position_in_grid.x;

    float bv = -INFINITY;
    uint bi = 0;
    const device float* pv = part_v + (size_t)b * NTG;
    const device uint* pi = part_i + (size_t)b * NTG;
    for (uint i = tid; i < NTG; i += 128) {
        float v = pv[i];
        uint idx = pi[i];
        if (v > bv || (v == bv && idx < bi)) {
            bv = v;
            bi = idx;
        }
    }
    float sm = simd_max(bv);
    uint win_i = simd_min((bv == sm) ? bi : 0xFFFFFFFFu);
    threadgroup float red_v[4];
    threadgroup uint red_i[4];
    if (lane == 0) {
        red_v[sg] = sm;
        red_i[sg] = win_i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {
        float fv = red_v[0];
        uint fi = red_i[0];
        for (int i = 1; i < 4; ++i) {
            if (red_v[i] > fv || (red_v[i] == fv && red_i[i] < fi)) {
                fv = red_v[i];
                fi = red_i[i];
            }
        }
        out_tok[b] = fi;
    }
"""

_markov_q8_p1_kernel = None
_markov_argmax_p2_kernel = None
_MARKOV_FOLD_GROUP_SIZE = 64
_MARKOV_FOLD_BITS = 8
_MARKOV_ARGMAX_NTG = 256

_SCREEN_K_PER = 2
_SCREEN_K_MAX_CAND = 64
_SCREEN_DELTA = 6.0

_DSPARK_LUT2_HEADER = """
constant half2 DS_LUT2[16] = {
    half2(-1.5h,-1.5h), half2(-0.5h,-1.5h), half2( 0.5h,-1.5h), half2( 1.5h,-1.5h),
    half2(-1.5h,-0.5h), half2(-0.5h,-0.5h), half2( 0.5h,-0.5h), half2( 1.5h,-0.5h),
    half2(-1.5h, 0.5h), half2(-0.5h, 0.5h), half2( 0.5h, 0.5h), half2( 1.5h, 0.5h),
    half2(-1.5h, 1.5h), half2(-0.5h, 1.5h), half2( 0.5h, 1.5h), half2( 1.5h, 1.5h)
};
"""

_DSPARK_SCREEN_PACK_SOURCE = """
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tg = threadgroup_position_in_grid.x;
    const uint g = threadgroup_position_in_grid.y;
    const uint tile = tg * 4 + sg;
    if (tile >= T_TILES) return;

    const uint v = lane * T_TILES + tile;
    const size_t out_idx = ((size_t)tile * N_GROUPS + g) * 32 + lane;
    if (v >= V_UNPAD) {
        out_wq[out_idx * 2 + 0] = 0u;
        out_wq[out_idx * 2 + 1] = 0u;
        out_sc[out_idx] = half(0.0f);
        return;
    }

    float w[32];
    if (SRC_BITS == 8) {
        const uint k0 = g * 32;
        const uint gi = k0 / SRC_GS;
        const float sc = float(w_scales[(size_t)v * (N_IN / SRC_GS) + gi]);
        const float bi = float(w_biases[(size_t)v * (N_IN / SRC_GS) + gi]);
        for (uint j = 0; j < 8; ++j) {
            uint pk = w_q[(size_t)v * (N_IN / 4) + (k0 / 4) + j];
            w[j * 4 + 0] = float((pk >> 0) & 0xFFu) * sc + bi;
            w[j * 4 + 1] = float((pk >> 8) & 0xFFu) * sc + bi;
            w[j * 4 + 2] = float((pk >> 16) & 0xFFu) * sc + bi;
            w[j * 4 + 3] = float((pk >> 24) & 0xFFu) * sc + bi;
        }
    } else if (SRC_BITS == 6) {
        const uint k0 = g * 32;
        const uint gi = k0 / SRC_GS;
        const float sc = float(w_scales[(size_t)v * (N_IN / SRC_GS) + gi]);
        const float bi = float(w_biases[(size_t)v * (N_IN / SRC_GS) + gi]);
        const size_t w_base = (size_t)v * ((size_t)N_IN * 6 / 32) + (size_t)g * 6;
        for (uint c = 0; c < 2; ++c) {
            const uint u0 = w_q[w_base + c * 3 + 0];
            const uint u1 = w_q[w_base + c * 3 + 1];
            const uint u2 = w_q[w_base + c * 3 + 2];
            const uint bo = c * 16;
            w[bo + 0]  = float((u0 >> 0)  & 0x3Fu) * sc + bi;
            w[bo + 1]  = float((u0 >> 6)  & 0x3Fu) * sc + bi;
            w[bo + 2]  = float((u0 >> 12) & 0x3Fu) * sc + bi;
            w[bo + 3]  = float((u0 >> 18) & 0x3Fu) * sc + bi;
            w[bo + 4]  = float((u0 >> 24) & 0x3Fu) * sc + bi;
            w[bo + 5]  = float(((u0 >> 30) & 0x03u) | ((u1 & 0x0Fu) << 2)) * sc + bi;
            w[bo + 6]  = float((u1 >> 4)  & 0x3Fu) * sc + bi;
            w[bo + 7]  = float((u1 >> 10) & 0x3Fu) * sc + bi;
            w[bo + 8]  = float((u1 >> 16) & 0x3Fu) * sc + bi;
            w[bo + 9]  = float((u1 >> 22) & 0x3Fu) * sc + bi;
            w[bo + 10] = float(((u1 >> 28) & 0x0Fu) | ((u2 & 0x03u) << 4)) * sc + bi;
            w[bo + 11] = float((u2 >> 2)  & 0x3Fu) * sc + bi;
            w[bo + 12] = float((u2 >> 8)  & 0x3Fu) * sc + bi;
            w[bo + 13] = float((u2 >> 14) & 0x3Fu) * sc + bi;
            w[bo + 14] = float((u2 >> 20) & 0x3Fu) * sc + bi;
            w[bo + 15] = float((u2 >> 26) & 0x3Fu) * sc + bi;
        }
    } else if (SRC_BITS == 4) {
        const uint k0 = g * 32;
        const uint gi = k0 / SRC_GS;
        const float sc = float(w_scales[(size_t)v * (N_IN / SRC_GS) + gi]);
        const float bi = float(w_biases[(size_t)v * (N_IN / SRC_GS) + gi]);
        const size_t w_base = (size_t)v * (N_IN / 8) + (size_t)g * 4;
        for (uint j = 0; j < 4; ++j) {
            uint pk = w_q[w_base + j];
            for (uint t = 0; t < 8; ++t) {
                w[j * 8 + t] = float((pk >> (4 * t)) & 0xFu) * sc + bi;
            }
        }
    } else {
        const size_t base_w = (size_t)v * N_IN + g * 32;
        for (uint j = 0; j < 32; ++j) {
            w[j] = float(w_dense[base_w + j]);
        }
    }

    float amax = 1e-9f;
    for (uint j = 0; j < 32; ++j) {
        amax = max(amax, metal::fabs(w[j]));
    }
    float best_s = amax / 1.5f;
    float best_e = INFINITY;
    for (int t = 0; t < 6; ++t) {
        float s = (amax / 1.5f) * (1.0f - 0.1f * float(t));
        float sh = float(half(s));
        float inv_sh = (sh > 0.0f) ? (1.0f / sh) : 0.0f;
        float e = 0.0f;
        for (uint j = 0; j < 32; ++j) {
            int lv = clamp(int(metal::floor(w[j] * inv_sh + 2.0f)), 0, 3);
            float r = (float(lv) - 1.5f) * sh - w[j];
            e += r * r;
        }
        if (e < best_e) {
            best_e = e;
            best_s = s;
        }
    }

    half hs = half(best_s);
    float sh = float(hs);
    float inv_sh = (sh > 0.0f) ? (1.0f / sh) : 0.0f;

    uint word0 = 0u;
    uint word1 = 0u;
    float ww = 0.0f;
    float wr = 0.0f;
    for (uint k = 0; k < 8; ++k) {
        int lv0 = clamp(int(metal::floor(w[2 * k] * inv_sh + 2.0f)), 0, 3);
        int lv1 = clamp(int(metal::floor(w[2 * k + 1] * inv_sh + 2.0f)), 0, 3);
        word0 |= (uint(lv0) | (uint(lv1) << 2)) << (4 * k);
        ww += w[2 * k] * w[2 * k] + w[2 * k + 1] * w[2 * k + 1];
        wr += w[2 * k] * ((float(lv0) - 1.5f) * sh)
            + w[2 * k + 1] * ((float(lv1) - 1.5f) * sh);

        int lv2 = clamp(int(metal::floor(w[16 + 2 * k] * inv_sh + 2.0f)), 0, 3);
        int lv3 = clamp(int(metal::floor(w[16 + 2 * k + 1] * inv_sh + 2.0f)), 0, 3);
        word1 |= (uint(lv2) | (uint(lv3) << 2)) << (4 * k);
        ww += w[16 + 2 * k] * w[16 + 2 * k] + w[16 + 2 * k + 1] * w[16 + 2 * k + 1];
        wr += w[16 + 2 * k] * ((float(lv2) - 1.5f) * sh)
            + w[16 + 2 * k + 1] * ((float(lv3) - 1.5f) * sh);
    }
    if (wr > 0.0f) {
        hs = half(sh * metal::sqrt(ww / wr));
    }
    out_wq[out_idx * 2 + 0] = word0;
    out_wq[out_idx * 2 + 1] = word1;
    out_sc[out_idx] = hs;
"""

_DSPARK_MARKOV_STEP_SOURCE = """
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint r = threadgroup_position_in_grid.x * 4 + sg;
    const uint b = threadgroup_position_in_grid.y;
    if (r >= R) return;

    const uint prev_tok = prev_tokens[b];
    const device bfloat* st_b = state_in + (size_t)b * R;
    const device bfloat* w1_b = markov_w1 + (size_t)prev_tok * R;

    const device bfloat* wg = w_sp + (size_t)r * (2 * R);
    const device bfloat* wc = w_sp + (size_t)(r + R) * (2 * R);
    const device bfloat* wl = w_sp + (size_t)(r + 2 * R) * (2 * R);

    float dg = 0.0f;
    float dc = 0.0f;
    float dl = 0.0f;
    for (uint k = lane; k < R; k += 32) {
        float sv = float(st_b[k]);
        float tv = float(w1_b[k]);
        dg += float(wg[k]) * sv + float(wg[R + k]) * tv;
        dc += float(wc[k]) * sv + float(wc[R + k]) * tv;
        dl += float(wl[k]) * sv + float(wl[R + k]) * tv;
    }
    dg = simd_sum(dg);
    dc = simd_sum(dc);
    dl = simd_sum(dl);

    if (lane == 0) {
        const size_t base = (size_t)b * (3 * R);
        const float raw_g = float(ctx_step[base + r]) + dg;
        const float raw_c = float(ctx_step[base + R + r]) + dc;
        const float raw_l = float(ctx_step[base + 2 * R + r]) + dl;
        const float gate_f = 1.0f / (1.0f + metal::exp(metal::clamp(-raw_g, -80.0f, 80.0f)));
        const bfloat gate = bfloat(gate_f);
        const bfloat cand = bfloat(metal::tanh(metal::clamp(raw_c, -20.0f, 20.0f)));
        const bfloat st = st_b[r];
        const bfloat kept = bfloat(float(gate) * float(st));
        const bfloat blend = bfloat((1.0f - float(gate)) * float(cand));
        state_out[(size_t)b * R + r] = bfloat(float(kept) + float(blend));
        const bfloat lat = bfloat(metal::tanh(metal::clamp(raw_l, -20.0f, 20.0f)));
        lat_bf16[(size_t)b * R + r] = lat;
        lat_f16[(size_t)b * R + r] = half(float(lat));
    }
"""

_DSPARK_HEAD_BASE_SOURCE = """
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint tile = threadgroup_position_in_grid.x;
    const uint b = threadgroup_position_in_grid.y;

    constexpr uint GP = (N_GROUPS + 3) / 4;
    const uint g_begin = sg * GP;
    const uint g_end = min((uint)N_GROUPS, g_begin + GP);

    const device half2* hb = (const device half2*)(h_f16 + (size_t)b * M * N_IN);
    const device uint2* wq = (const device uint2*)head_wq;

    float acc[M];
    for (uint m = 0; m < M; ++m) acc[m] = 0.0f;

    const size_t t_base = (size_t)tile * N_GROUPS;
    for (uint g = g_begin; g < g_end; ++g) {
        const size_t idx = (t_base + g) * 32 + lane;
        const uint2 pk = wq[idx];
        const float s = float(head_sc[idx]);
        const uint h2_base = g * 16;

        half2 w0 = DS_LUT2[pk.x & 0xFu];
        half2 w1 = DS_LUT2[(pk.x >> 4) & 0xFu];
        half2 w2 = DS_LUT2[(pk.x >> 8) & 0xFu];
        half2 w3 = DS_LUT2[(pk.x >> 12) & 0xFu];
        half2 w4 = DS_LUT2[(pk.x >> 16) & 0xFu];
        half2 w5 = DS_LUT2[(pk.x >> 20) & 0xFu];
        half2 w6 = DS_LUT2[(pk.x >> 24) & 0xFu];
        half2 w7 = DS_LUT2[(pk.x >> 28) & 0xFu];

        float psum[M];
        for (uint m = 0; m < M; ++m) {
            const device half2* xm = hb + (size_t)m * (N_IN / 2) + h2_base;
            half2 gacc = w0 * xm[0];
            gacc = fma(w1, xm[1], gacc);
            gacc = fma(w2, xm[2], gacc);
            gacc = fma(w3, xm[3], gacc);
            gacc = fma(w4, xm[4], gacc);
            gacc = fma(w5, xm[5], gacc);
            gacc = fma(w6, xm[6], gacc);
            gacc = fma(w7, xm[7], gacc);
            psum[m] = float(gacc.x) + float(gacc.y);
        }

        w0 = DS_LUT2[pk.y & 0xFu];
        w1 = DS_LUT2[(pk.y >> 4) & 0xFu];
        w2 = DS_LUT2[(pk.y >> 8) & 0xFu];
        w3 = DS_LUT2[(pk.y >> 12) & 0xFu];
        w4 = DS_LUT2[(pk.y >> 16) & 0xFu];
        w5 = DS_LUT2[(pk.y >> 20) & 0xFu];
        w6 = DS_LUT2[(pk.y >> 24) & 0xFu];
        w7 = DS_LUT2[(pk.y >> 28) & 0xFu];

        for (uint m = 0; m < M; ++m) {
            const device half2* xm = hb + (size_t)m * (N_IN / 2) + h2_base + 8;
            half2 gacc = w0 * xm[0];
            gacc = fma(w1, xm[1], gacc);
            gacc = fma(w2, xm[2], gacc);
            gacc = fma(w3, xm[3], gacc);
            gacc = fma(w4, xm[4], gacc);
            gacc = fma(w5, xm[5], gacc);
            gacc = fma(w6, xm[6], gacc);
            gacc = fma(w7, xm[7], gacc);
            acc[m] += s * (psum[m] + float(gacc.x) + float(gacc.y));
        }
    }

    threadgroup float red[4][32];
    const uint sr = tile * 32 + lane;
    for (uint m = 0; m < M; ++m) {
        red[sg][lane] = acc[m];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            float tot = red[0][lane] + red[1][lane] + red[2][lane] + red[3][lane];
            approx_out[((size_t)b * M + m) * (T_TILES * 32) + sr] = half(tot);
        }
        if (m + 1 < M) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
"""

_DSPARK_FOLD_SCREEN_SOURCE = """
    constexpr float LOG2E = 1.4426950408889634f;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint tile = threadgroup_position_in_grid.x * 4 + sg;
    const uint b = threadgroup_position_in_grid.y;
    if (tile >= T_TILES) return;

    const uint sr = tile * 32 + lane;
    const uint v_id = lane * T_TILES + tile;
    const device half2* lb = (const device half2*)(lat_f16 + (size_t)b * R);
    const device uint2* wq = (const device uint2*)fold_wq;

    float score = -INFINITY;
    if (v_id < V_UNPAD) {
        float acc = 0.0f;
        const size_t t_base = (size_t)tile * R_GROUPS;
        for (uint g = 0; g < R_GROUPS; ++g) {
            const size_t idx = (t_base + g) * 32 + lane;
            const uint2 pk = wq[idx];
            const float s = float(fold_sc[idx]);
            const device half2* xm = lb + g * 16;
            half2 gacc = DS_LUT2[pk.x & 0xFu] * xm[0];
            gacc = fma(DS_LUT2[(pk.x >> 4) & 0xFu], xm[1], gacc);
            gacc = fma(DS_LUT2[(pk.x >> 8) & 0xFu], xm[2], gacc);
            gacc = fma(DS_LUT2[(pk.x >> 12) & 0xFu], xm[3], gacc);
            gacc = fma(DS_LUT2[(pk.x >> 16) & 0xFu], xm[4], gacc);
            gacc = fma(DS_LUT2[(pk.x >> 20) & 0xFu], xm[5], gacc);
            gacc = fma(DS_LUT2[(pk.x >> 24) & 0xFu], xm[6], gacc);
            gacc = fma(DS_LUT2[(pk.x >> 28) & 0xFu], xm[7], gacc);
            gacc = fma(DS_LUT2[pk.y & 0xFu], xm[8], gacc);
            gacc = fma(DS_LUT2[(pk.y >> 4) & 0xFu], xm[9], gacc);
            gacc = fma(DS_LUT2[(pk.y >> 8) & 0xFu], xm[10], gacc);
            gacc = fma(DS_LUT2[(pk.y >> 12) & 0xFu], xm[11], gacc);
            gacc = fma(DS_LUT2[(pk.y >> 16) & 0xFu], xm[12], gacc);
            gacc = fma(DS_LUT2[(pk.y >> 20) & 0xFu], xm[13], gacc);
            gacc = fma(DS_LUT2[(pk.y >> 24) & 0xFu], xm[14], gacc);
            gacc = fma(DS_LUT2[(pk.y >> 28) & 0xFu], xm[15], gacc);
            acc += s * (float(gacc.x) + float(gacc.y));
        }
        float raw = float(approx_base[(size_t)b * (T_TILES * 32) + sr])
                  + float(fold_bp[sr]) + acc;
        float cap = cap_arr[0];
        if (cap > 0.0f) {
            score = cap * fast::tanh(metal::clamp(raw / cap, -20.0f, 20.0f));
        } else {
            score = raw;
        }
    }

    float tmax = simd_max(score);
    float e = (score > -INFINITY) ? fast::exp2((score - tmax) * LOG2E) : 0.0f;
    float tsum = simd_sum(e);
    if (lane == 0) {
        const size_t p_idx = ((size_t)b * T_TILES + tile) * 2;
        part_out[p_idx + 0] = tmax;
        part_out[p_idx + 1] = tsum;
    }

    float cur = score;
    const size_t c_base = ((size_t)b * T_TILES + tile) * K_PER;
    for (uint k = 0; k < K_PER; ++k) {
        float bv = simd_max(cur);
        uint win_lane = simd_min((cur == bv) ? lane : 32u);
        if (lane == 0) {
            cand_v[c_base + k] = bv;
            cand_r[c_base + k] = tile * 32 + (win_lane < 32u ? win_lane : 0u);
        }
        if (lane == win_lane) {
            cur = -INFINITY;
        }
    }
"""

_DSPARK_HEAD_PICK_SOURCE = """
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint b = threadgroup_position_in_grid.x;

    const device float* cv = cand_v + (size_t)b * N_CAND_IN;
    const device uint* cr = cand_r + (size_t)b * N_CAND_IN;

    float local_m = -INFINITY;
    for (uint i = tid; i < N_CAND_IN; i += 256) {
        local_m = max(local_m, cv[i]);
    }
    local_m = simd_max(local_m);
    threadgroup float tg_red[8];
    if (lane == 0) tg_red[sg] = local_m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gmax = tg_red[0];
    for (uint i = 1; i < 8; ++i) gmax = max(gmax, tg_red[i]);

    if (!(gmax > -INFINITY)) {
        if (tid == 0) {
            out_tok[b] = 0u;
            out_conf[b] = 1.0f;
        }
        return;
    }

    threadgroup atomic_uint tg_cnt;
    threadgroup uint tg_vocab[K_MAX];
    threadgroup float tg_lp[K_MAX];
    threadgroup float tg_ex[K_MAX];
    if (tid == 0) atomic_store_explicit(&tg_cnt, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float cut0 = gmax - DELTA_VAL;
    for (uint i = tid; i < N_CAND_IN; i += 256) {
        float sc = cv[i];
        if (sc >= cut0) {
            uint pos = atomic_fetch_add_explicit(&tg_cnt, 1u, memory_order_relaxed);
            if (pos < K_MAX) {
                uint sr = cr[i];
                uint v = (sr & 31u) * T_TILES + (sr >> 5);
                tg_vocab[pos] = v;
                tg_lp[pos] = sc;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint total = atomic_load_explicit(&tg_cnt, memory_order_relaxed);
    if (total > K_MAX) {
        float lo = cut0;
        float hi = gmax;
        threadgroup uint tg_bs[8];
        for (int it = 0; it < 14; ++it) {
            float mid = 0.5f * (lo + hi);
            uint c = 0u;
            for (uint i = tid; i < N_CAND_IN; i += 256) {
                if (cv[i] >= mid) c += 1u;
            }
            c = simd_sum(c);
            if (lane == 0) tg_bs[sg] = c;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint tot_c = 0u;
            for (uint j = 0; j < 8; ++j) tot_c += tg_bs[j];
            if (tot_c > K_MAX) {
                lo = mid;
            } else {
                hi = mid;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) atomic_store_explicit(&tg_cnt, 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < N_CAND_IN; i += 256) {
            float sc = cv[i];
            if (sc > hi) {
                uint pos = atomic_fetch_add_explicit(&tg_cnt, 1u, memory_order_relaxed);
                if (pos < K_MAX) {
                    uint sr = cr[i];
                    tg_vocab[pos] = (sr & 31u) * T_TILES + (sr >> 5);
                    tg_lp[pos] = sc;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < N_CAND_IN; i += 256) {
            float sc = cv[i];
            if (sc >= lo && sc <= hi) {
                uint pos = atomic_fetch_add_explicit(&tg_cnt, 1u, memory_order_relaxed);
                if (pos < K_MAX) {
                    uint sr = cr[i];
                    tg_vocab[pos] = (sr & 31u) * T_TILES + (sr >> 5);
                    tg_lp[pos] = sc;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        total = min((uint)K_MAX, atomic_load_explicit(&tg_cnt, memory_order_relaxed));
    }
    const uint n_cand = total;

    threadgroup float tg_h[H];
    threadgroup float tg_lat[R];
    const device bfloat* hp = h_step + (size_t)b * H;
    const device bfloat* lp = lat_step + (size_t)b * R;
    for (uint i = tid; i < H; i += 256) tg_h[i] = float(hp[i]);
    for (uint i = tid; i < R; i += 256) tg_lat[i] = float(lp[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr uint H_KW = H / 4;
    constexpr uint H_KGS = H / HEAD_GS;
    constexpr uint R_KW = R / 4;
    constexpr uint R_KGS = R / FOLD_GS;
    const float cap = cap_arr[0];

    for (uint c = sg; c < n_cand; c += 8) {
        const uint v = tg_vocab[c];
        float hdot = 0.0f;
        if (HEAD_BITS == 8) {
            for (uint k4 = lane; k4 < H_KW; k4 += 32) {
                uint k0 = k4 * 4;
                uint gi = k0 / HEAD_GS;
                float sc = float(head_s[(size_t)v * H_KGS + gi]);
                float bi = float(head_b[(size_t)v * H_KGS + gi]);
                uint pk = head_wq[(size_t)v * H_KW + k4];
                hdot += (float((pk >> 0) & 0xFFu) * sc + bi) * tg_h[k0 + 0]
                      + (float((pk >> 8) & 0xFFu) * sc + bi) * tg_h[k0 + 1]
                      + (float((pk >> 16) & 0xFFu) * sc + bi) * tg_h[k0 + 2]
                      + (float((pk >> 24) & 0xFFu) * sc + bi) * tg_h[k0 + 3];
            }
        } else if (HEAD_BITS == 6) {
            constexpr uint H_KW6 = (H * 6) / 32;
            constexpr uint H_K16 = H / 16;
            for (uint k16 = lane; k16 < H_K16; k16 += 32) {
                uint k0 = k16 * 16;
                uint gi = k0 / HEAD_GS;
                float sc = float(head_s[(size_t)v * H_KGS + gi]);
                float bi = float(head_b[(size_t)v * H_KGS + gi]);
                size_t wb = (size_t)v * H_KW6 + (size_t)k16 * 3;
                uint u0 = head_wq[wb + 0];
                uint u1 = head_wq[wb + 1];
                uint u2 = head_wq[wb + 2];
                float qs = 0.0f;
                float hs = 0.0f;
                qs += float((u0 >> 0)  & 0x3Fu) * tg_h[k0 + 0];  hs += tg_h[k0 + 0];
                qs += float((u0 >> 6)  & 0x3Fu) * tg_h[k0 + 1];  hs += tg_h[k0 + 1];
                qs += float((u0 >> 12) & 0x3Fu) * tg_h[k0 + 2];  hs += tg_h[k0 + 2];
                qs += float((u0 >> 18) & 0x3Fu) * tg_h[k0 + 3];  hs += tg_h[k0 + 3];
                qs += float((u0 >> 24) & 0x3Fu) * tg_h[k0 + 4];  hs += tg_h[k0 + 4];
                qs += float(((u0 >> 30) & 0x03u) | ((u1 & 0x0Fu) << 2)) * tg_h[k0 + 5]; hs += tg_h[k0 + 5];
                qs += float((u1 >> 4)  & 0x3Fu) * tg_h[k0 + 6];  hs += tg_h[k0 + 6];
                qs += float((u1 >> 10) & 0x3Fu) * tg_h[k0 + 7];  hs += tg_h[k0 + 7];
                qs += float((u1 >> 16) & 0x3Fu) * tg_h[k0 + 8];  hs += tg_h[k0 + 8];
                qs += float((u1 >> 22) & 0x3Fu) * tg_h[k0 + 9];  hs += tg_h[k0 + 9];
                qs += float(((u1 >> 28) & 0x0Fu) | ((u2 & 0x03u) << 4)) * tg_h[k0 + 10]; hs += tg_h[k0 + 10];
                qs += float((u2 >> 2)  & 0x3Fu) * tg_h[k0 + 11]; hs += tg_h[k0 + 11];
                qs += float((u2 >> 8)  & 0x3Fu) * tg_h[k0 + 12]; hs += tg_h[k0 + 12];
                qs += float((u2 >> 14) & 0x3Fu) * tg_h[k0 + 13]; hs += tg_h[k0 + 13];
                qs += float((u2 >> 20) & 0x3Fu) * tg_h[k0 + 14]; hs += tg_h[k0 + 14];
                qs += float((u2 >> 26) & 0x3Fu) * tg_h[k0 + 15]; hs += tg_h[k0 + 15];
                hdot += qs * sc + hs * bi;
            }
        } else if (HEAD_BITS == 4) {
            constexpr uint H_KW4 = H / 8;
            for (uint k8 = lane; k8 < H_KW4; k8 += 32) {
                uint k0 = k8 * 8;
                uint gi = k0 / HEAD_GS;
                float sc = float(head_s[(size_t)v * H_KGS + gi]);
                float bi = float(head_b[(size_t)v * H_KGS + gi]);
                uint pk = head_wq[(size_t)v * H_KW4 + k8];
                float qs = 0.0f;
                float hs = 0.0f;
                for (uint t = 0; t < 8; ++t) {
                    float xv = tg_h[k0 + t];
                    qs += float((pk >> (4 * t)) & 0xFu) * xv;
                    hs += xv;
                }
                hdot += qs * sc + hs * bi;
            }
        } else {
            const size_t base_w = (size_t)v * H;
            for (uint k = lane; k < H; k += 32) {
                hdot += float(head_dense[base_w + k]) * tg_h[k];
            }
        }
        hdot = simd_sum(hdot);

        float fdot = 0.0f;
        for (uint k4 = lane; k4 < R_KW; k4 += 32) {
            uint k0 = k4 * 4;
            uint gi = k0 / FOLD_GS;
            float sc = float(fold_s[(size_t)v * R_KGS + gi]);
            float bi = float(fold_b[(size_t)v * R_KGS + gi]);
            uint pk = fold_wq[(size_t)v * R_KW + k4];
            fdot += (float((pk >> 0) & 0xFFu) * sc + bi) * tg_lat[k0 + 0]
                  + (float((pk >> 8) & 0xFFu) * sc + bi) * tg_lat[k0 + 1]
                  + (float((pk >> 16) & 0xFFu) * sc + bi) * tg_lat[k0 + 2]
                  + (float((pk >> 24) & 0xFFu) * sc + bi) * tg_lat[k0 + 3];
        }
        fdot = simd_sum(fdot);

        if (lane == 0) {
            float raw = hdot + fdot + float(fold_bias[v]);
            tg_ex[c] = (cap > 0.0f)
                ? (cap * metal::tanh(metal::clamp(raw / cap, -20.0f, 20.0f)))
                : raw;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup uint tg_best_i;
    threadgroup float tg_best_v;
    if (sg == 0) {
        float bv = -INFINITY;
        uint bi = 0xFFFFFFFFu;
        for (uint c = lane; c < n_cand; c += 32) {
            float ev = tg_ex[c];
            uint vi = tg_vocab[c];
            if (ev > bv || (ev == bv && vi < bi)) {
                bv = ev;
                bi = vi;
            }
        }
        float top = simd_max(bv);
        uint win = simd_min((bv == top) ? bi : 0xFFFFFFFFu);
        if (lane == 0) {
            tg_best_v = top;
            tg_best_i = (win < 0xFFFFFFFFu) ? win : 0u;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr float LOG2E = 1.4426950408889634f;
    const float bv = tg_best_v;
    const uint bi = tg_best_i;
    const device float* pp = part_in + (size_t)b * T_TILES * 2;
    float sum = 0.0f;
    for (uint t = tid; t < T_TILES; t += 256) {
        float pm = pp[t * 2 + 0];
        float ps = pp[t * 2 + 1];
        if (ps > 0.0f) {
            sum += ps * fast::exp2((pm - bv) * LOG2E);
        }
    }
    for (uint c = tid; c < n_cand; c += 256) {
        sum += fast::exp2((tg_ex[c] - bv) * LOG2E) - fast::exp2((tg_lp[c] - bv) * LOG2E);
    }
    sum = simd_sum(sum);
    if (lane == 0) tg_red[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total_sum = 0.0f;
        for (uint i = 0; i < 8; ++i) total_sum += tg_red[i];
        total_sum = max(total_sum, 1.0f);
        out_tok[b] = bi;
        out_conf[b] = 1.0f / total_sum;
    }
"""

_dspark_screen_pack_kernel = None
_dspark_head_base_kernel = None
_dspark_fold_screen_kernel = None
_dspark_head_pick_kernel = None
_dspark_markov_step_kernel = None
_DUMMY_U32 = None
_DUMMY_BF16 = None


def _dummy_u32() -> mx.array:
    global _DUMMY_U32
    if _DUMMY_U32 is None:
        _DUMMY_U32 = mx.zeros((4,), dtype=mx.uint32)
    return _DUMMY_U32


def _dummy_bf16() -> mx.array:
    global _DUMMY_BF16
    if _DUMMY_BF16 is None:
        _DUMMY_BF16 = mx.zeros((4,), dtype=mx.bfloat16)
    return _DUMMY_BF16


def _dspark_screen_pack(
    *,
    v_unpad: int,
    n_in: int,
    w_q: mx.array | None = None,
    w_scales: mx.array | None = None,
    w_biases: mx.array | None = None,
    src_gs: int = 64,
    src_bits: int = 8,
    w_dense: mx.array | None = None,
) -> tuple[mx.array, mx.array, int]:
    """Pack a Q4/Q6/Q8-affine or BF16 matrix into 2-bit tile-transposed screen copy."""
    global _dspark_screen_pack_kernel
    if _dspark_screen_pack_kernel is None:
        _dspark_screen_pack_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_dspark_screen_pack",
            input_names=["w_q", "w_scales", "w_biases", "w_dense"],
            output_names=["out_wq", "out_sc"],
            source=_DSPARK_SCREEN_PACK_SOURCE,
        )
    t_tiles = (v_unpad + 31) // 32
    n_groups = n_in // 32
    bits = int(src_bits) if w_q is not None else 0
    is_q = bits > 0
    wt = w_dense.dtype if w_dense is not None else mx.bfloat16
    ntg_x = (t_tiles + 3) // 4
    out_wq, out_sc = _dspark_screen_pack_kernel(
        inputs=[
            w_q if is_q else _dummy_u32(),
            w_scales if is_q else _dummy_bf16(),
            w_biases if is_q else _dummy_bf16(),
            w_dense if not is_q else _dummy_bf16(),
        ],
        template=[
            ("WT", wt),
            ("T_TILES", t_tiles),
            ("N_GROUPS", n_groups),
            ("N_IN", n_in),
            ("V_UNPAD", v_unpad),
            ("SRC_GS", src_gs),
            ("SRC_BITS", bits),
        ],
        grid=(ntg_x * 128, n_groups, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(t_tiles, n_groups, 32, 2), (t_tiles, n_groups, 32)],
        output_dtypes=[mx.uint32, mx.float16],
    )
    return out_wq, out_sc, t_tiles


def _dspark_screen_order(by_token: mx.array, v_unpad: int, t_tiles: int) -> mx.array:
    """Permute a [V] vector into tile-transposed screen row order [32 * T_tiles] fp16."""
    n_screen = t_tiles * 32
    sr = mx.arange(n_screen, dtype=mx.uint32)
    v = (sr & mx.array(31, dtype=mx.uint32)) * mx.array(t_tiles, dtype=mx.uint32) + (
        sr >> mx.array(5, dtype=mx.uint32)
    )
    valid = v < v_unpad
    v_safe = mx.where(valid, v, mx.array(0, dtype=mx.uint32))
    gathered = by_token[v_safe].astype(mx.float16)
    return mx.where(valid, gathered, mx.array(0.0, dtype=mx.float16))


def _dspark_head_base(
    h_bf16: mx.array, head_wq: mx.array, head_sc: mx.array, t_tiles: int
) -> mx.array:
    """Stage 1A: 2-bit screened head dot product across all M block rows at once."""
    global _dspark_head_base_kernel
    if _dspark_head_base_kernel is None:
        _dspark_head_base_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_dspark_head_base",
            input_names=["h_f16", "head_wq", "head_sc"],
            output_names=["approx_out"],
            source=_DSPARK_HEAD_BASE_SOURCE,
            header=_DSPARK_LUT2_HEADER,
        )
    b, m, n_in = h_bf16.shape
    h_f16 = mx.contiguous(h_bf16.astype(mx.float16))
    return _dspark_head_base_kernel(
        inputs=[h_f16, head_wq, head_sc],
        template=[
            ("T_TILES", t_tiles),
            ("N_GROUPS", n_in // 32),
            ("N_IN", n_in),
            ("M", m),
        ],
        grid=(t_tiles * 128, b, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(b, m, t_tiles * 32)],
        output_dtypes=[mx.float16],
    )[0]


def _dspark_markov_step_fused(
    state_in: mx.array,
    prev_tokens: mx.array,
    ctx_step: mx.array,
    w_sp: mx.array,
    markov_w1: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """Fused Markov RNN recurrence: returns ``(state_out, lat_bf16, lat_f16)``."""
    global _dspark_markov_step_kernel
    if _dspark_markov_step_kernel is None:
        _dspark_markov_step_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_dspark_markov_step",
            input_names=["state_in", "prev_tokens", "ctx_step", "w_sp", "markov_w1"],
            output_names=["state_out", "lat_bf16", "lat_f16"],
            source=_DSPARK_MARKOV_STEP_SOURCE,
        )
    b, r = state_in.shape
    ntg_x = (r + 3) // 4
    return _dspark_markov_step_kernel(
        inputs=[
            state_in,
            prev_tokens.reshape(-1).astype(mx.uint32),
            ctx_step,
            w_sp,
            markov_w1,
        ],
        template=[("R", r)],
        grid=(ntg_x * 128, b, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(b, r), (b, r), (b, r)],
        output_dtypes=[mx.bfloat16, mx.bfloat16, mx.float16],
    )


def _dspark_screened_pick(
    *,
    h_step: mx.array,
    lat_step: mx.array,
    lat_f16: mx.array | None = None,
    approx_base: mx.array,
    fold_sc_wq: mx.array,
    fold_sc_sc: mx.array,
    fold_bp: mx.array,
    head_wq: mx.array | None,
    head_s: mx.array | None,
    head_b: mx.array | None,
    head_dense: mx.array | None,
    head_gs: int,
    head_bits: int,
    fold_wq: mx.array,
    fold_s: mx.array,
    fold_b: mx.array,
    fold_bias: mx.array,
    v_unpad: int,
    t_tiles: int,
    cap_arr: mx.array,
) -> tuple[mx.array, mx.array]:
    """Stage 1B + Stage 2: 2-bit fold screen -> exact Q4/Q6/Q8/BF16 rescore -> (tok, conf)."""
    global _dspark_fold_screen_kernel, _dspark_head_pick_kernel
    if _dspark_fold_screen_kernel is None:
        _dspark_fold_screen_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_dspark_fold_screen",
            input_names=[
                "lat_f16",
                "fold_wq",
                "fold_sc",
                "fold_bp",
                "approx_base",
                "cap_arr",
            ],
            output_names=["part_out", "cand_v", "cand_r"],
            source=_DSPARK_FOLD_SCREEN_SOURCE,
            header=_DSPARK_LUT2_HEADER,
        )
        _dspark_head_pick_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_dspark_head_pick",
            input_names=[
                "cand_v",
                "cand_r",
                "part_in",
                "h_step",
                "lat_step",
                "head_wq",
                "head_s",
                "head_b",
                "head_dense",
                "fold_wq",
                "fold_s",
                "fold_b",
                "fold_bias",
                "cap_arr",
            ],
            output_names=["out_tok", "out_conf"],
            source=_DSPARK_HEAD_PICK_SOURCE,
        )
    b, r = lat_step.shape
    h_dim = h_step.shape[-1]
    if lat_f16 is None:
        lat_f16 = mx.contiguous(lat_step.astype(mx.float16))
    ntg_x = (t_tiles + 3) // 4
    part, cand_v, cand_r = _dspark_fold_screen_kernel(
        inputs=[lat_f16, fold_sc_wq, fold_sc_sc, fold_bp, approx_base, cap_arr],
        template=[
            ("T_TILES", t_tiles),
            ("R", r),
            ("R_GROUPS", r // 32),
            ("V_UNPAD", v_unpad),
            ("K_PER", _SCREEN_K_PER),
        ],
        grid=(ntg_x * 128, b, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[
            (b, t_tiles, 2),
            (b, t_tiles * _SCREEN_K_PER),
            (b, t_tiles * _SCREEN_K_PER),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.uint32],
    )
    h_bits = int(head_bits) if head_wq is not None else 0
    is_q = h_bits > 0
    wt = head_dense.dtype if head_dense is not None else mx.bfloat16
    tok, conf = _dspark_head_pick_kernel(
        inputs=[
            cand_v,
            cand_r,
            part,
            h_step if h_step.dtype == mx.bfloat16 else h_step.astype(mx.bfloat16),
            lat_step if lat_step.dtype == mx.bfloat16 else lat_step.astype(mx.bfloat16),
            head_wq if is_q else _dummy_u32(),
            head_s if is_q else _dummy_bf16(),
            head_b if is_q else _dummy_bf16(),
            head_dense if not is_q else _dummy_bf16(),
            fold_wq,
            fold_s,
            fold_b,
            fold_bias,
            cap_arr,
        ],
        template=[
            ("WT", wt),
            ("T_TILES", t_tiles),
            ("N_CAND_IN", t_tiles * _SCREEN_K_PER),
            ("K_MAX", _SCREEN_K_MAX_CAND),
            ("DELTA_VAL", int(_SCREEN_DELTA)),
            ("H", h_dim),
            ("R", r),
            ("HEAD_GS", head_gs),
            ("FOLD_GS", _MARKOV_FOLD_GROUP_SIZE),
            ("HEAD_BITS", h_bits),
        ],
        grid=(b * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(b,), (b,)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return tok, conf


def _markov_q8_greedy_argmax(
    latent: mx.array,
    w_q: mx.array,
    scales: mx.array,
    biases: mx.array,
    fold_bias: mx.array,
    base_logits: mx.array,
    v_unpad: int,
) -> mx.array:
    """Fused Q8 affine Markov projection + base_logits add + greedy argmax."""
    global _markov_q8_p1_kernel, _markov_argmax_p2_kernel
    if _markov_q8_p1_kernel is None:
        _markov_q8_p1_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_markov_q8_p1",
            input_names=[
                "latent",
                "w_q",
                "scales",
                "biases",
                "fold_bias",
                "base_logits",
            ],
            output_names=["part_v", "part_i"],
            source=_MARKOV_Q8_ARGMAX_P1_SOURCE,
        )
        _markov_argmax_p2_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_markov_argmax_p2",
            input_names=["part_v", "part_i"],
            output_names=["out_tok"],
            source=_MARKOV_ARGMAX_P2_SOURCE,
        )
    b, r = latent.shape
    v_full = base_logits.shape[-1]
    ntg = _MARKOV_ARGMAX_NTG
    pv, pi = _markov_q8_p1_kernel(
        inputs=[latent, w_q, scales, biases, fold_bias, base_logits],
        template=[
            ("R", r),
            ("GS", _MARKOV_FOLD_GROUP_SIZE),
            ("V_UNPAD", v_unpad),
            ("V_FULL", v_full),
        ],
        grid=(ntg * 256, b, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(b, ntg), (b, ntg)],
        output_dtypes=[mx.float32, mx.uint32],
    )
    return _markov_argmax_p2_kernel(
        inputs=[pv, pi],
        template=[("NTG", ntg)],
        grid=(b * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(b,)],
        output_dtypes=[mx.uint32],
    )[0]
'''

_DSPARK_DRAUGHT_CLS_SRC = '''

_GLOBAL_KV_STEP = 256
_GLOBAL_PROJ_CHUNK = 1024


def _make_inkling2_dspark_drafter_cls():
    from ...speculative.dflash import DFlashDrafter, selection, variants
    from ...speculative.dflash.drafter import _Cohort, _Predraft

    class Inkling2DSparkDrafter(DFlashDrafter):
        """DSpark drafter with per-layer local/global context and 2-bit screened head."""

        def __init__(self, model: Inkling2DSparkModel, **kwargs):
            super().__init__(model, **kwargs)
            n_layers = len(model.layers)
            self._local_layers = {
                int(i) for i in (getattr(model.config, "local_layer_ids", None) or ())
            }
            self._global_layers = {
                i for i in range(n_layers) if i not in self._local_layers
            }
            self._global_rows: dict[Any, dict[int, list]] = {}
            self._global_start: dict[Any, int] = {}
            self._predraft_global: dict[int, tuple[mx.array, mx.array, list[int]]] | None = None
            self._predraft_conf: list | None = None
            self._last_conf: dict[Any, list[mx.array]] = {}

        @property
        def states_chances(self) -> bool:
            markov = getattr(self.model, "markov_head", None)
            return bool(getattr(markov, "screen_enabled", False))

        def _bound_pending(self, row) -> None:
            if self._global_layers:
                return
            super()._bound_pending(row)

        def release(self, uids) -> None:
            uids = list(uids)
            self._predraft_global = None
            self._predraft_conf = None
            for uid in uids:
                self._global_rows.pop(uid, None)
                self._global_start.pop(uid, None)
                self._last_conf.pop(uid, None)
            super().release(uids)

        def reset_rows(self) -> None:
            self._predraft_global = None
            self._predraft_conf = None
            self._global_rows.clear()
            self._global_start.clear()
            self._last_conf.clear()
            super().reset_rows()

        def discard_predraft(self) -> None:
            self._predraft_global = None
            self._predraft_conf = None
            super().discard_predraft()

        def adopt_predraft(self, state, count: int) -> None:
            pg = self._predraft_global
            pc = self._predraft_conf
            self._predraft_global = None
            self._predraft_conf = None
            super().adopt_predraft(state, count)
            if pg is not None:
                uid = state.uid
                gmap = self._global_rows.setdefault(uid, {})
                for layer_idx, (gk, gv, lengths) in pg.items():
                    gmap[layer_idx] = [gk[0], gv[0], lengths[0]]
            if pc is not None:
                n_keep = self._proposal_count(state)
                state.draft_conf = pc[0][:n_keep]
                self._last_conf[state.uid] = state.draft_conf

        def _commit_proposals(self, state, tokens, accept_lps, conf=None) -> None:
            super()._commit_proposals(state, tokens, accept_lps)
            confs = self._last_conf.get(state.uid)
            if confs is not None and getattr(self, "confidence_windows", False):
                count = int(state.drafts.shape[0])
                state.draft_conf = confs[:count]

        def _append_global_kv(
            self,
            uid: Any,
            layer_idx: int,
            k_new: mx.array,
            v_new: mx.array,
            start_pos: int,
            n_write: int,
            commit: bool,
        ) -> tuple[mx.array, mx.array, int]:
            gmap = self._global_rows.setdefault(uid, {})
            entry = gmap.get(layer_idx)
            if uid not in self._global_start:
                self._global_start[uid] = start_pos
            g_start = self._global_start[uid]
            offset = max(0, start_pos - g_start)
            end = offset + n_write
            n_kv, _, head_dim = k_new.shape
            if entry is None:
                cap = max(_GLOBAL_KV_STEP, ((end + _GLOBAL_KV_STEP - 1) // _GLOBAL_KV_STEP) * _GLOBAL_KV_STEP)
                k_buf = mx.zeros((n_kv, cap, head_dim), dtype=k_new.dtype)
                v_buf = mx.zeros((n_kv, cap, head_dim), dtype=v_new.dtype)
                cur_len = 0
            else:
                k_buf, v_buf, cur_len = entry
                if not commit:
                    k_buf = mx.stop_gradient(k_buf)
                    v_buf = mx.stop_gradient(v_buf)
                cap = int(k_buf.shape[1])
                if end > cap:
                    new_cap = ((end + _GLOBAL_KV_STEP - 1) // _GLOBAL_KV_STEP) * _GLOBAL_KV_STEP
                    nk = mx.zeros((n_kv, new_cap, head_dim), dtype=k_buf.dtype)
                    nv = mx.zeros((n_kv, new_cap, head_dim), dtype=v_buf.dtype)
                    if cur_len > 0:
                        nk[:, :cur_len, :] = k_buf[:, :cur_len, :]
                        nv[:, :cur_len, :] = v_buf[:, :cur_len, :]
                    k_buf, v_buf = nk, nv
            if n_write > 0:
                k_buf[:, offset:end, :] = k_new[:, :n_write, :]
                v_buf[:, offset:end, :] = v_new[:, :n_write, :]
            new_len = max(cur_len, end)
            if commit:
                gmap[layer_idx] = [k_buf, v_buf, new_len]
            return k_buf, v_buf, new_len

        def _project_ctx_kv_chunked(
            self, attn, ctx_row: mx.array, base_pos: int, model
        ) -> tuple[mx.array, mx.array]:
            n = int(ctx_row.shape[1])
            if n == 0:
                return (
                    mx.zeros((attn.n_kv_heads, 0, attn.head_dim), dtype=ctx_row.dtype),
                    mx.zeros((attn.n_kv_heads, 0, attn.head_dim), dtype=ctx_row.dtype),
                )
            ctx_k_norm = getattr(attn, "k_norm_ctx", attn.k_norm)
            ks, vs = [], []
            for s0 in range(0, n, _GLOBAL_PROJ_CHUNK):
                s1 = min(n, s0 + _GLOBAL_PROJ_CHUNK)
                chunk = ctx_row[:, s0:s1]
                ck, cv = attn.project_context_kv(chunk)
                ck = ctx_k_norm(
                    ck.reshape(1, s1 - s0, attn.n_kv_heads, attn.head_dim)
                ).transpose(0, 2, 1, 3)
                cv = cv.reshape(
                    1, s1 - s0, attn.n_kv_heads, attn.head_dim
                ).transpose(0, 2, 1, 3)
                if self.value_scale is not None:
                    cv = cv * self.value_scale
                ck = model.rope(ck, offset=base_pos + s0)
                ks.append(ck[0])
                vs.append(cv[0])
            if len(ks) == 1:
                return ks[0], vs[0]
            return mx.concatenate(ks, axis=1), mx.concatenate(vs, axis=1)

        def _draft_batched(self, rows, counts=None, commit=True):
            model = self.model
            uids = [state.uid for state, *_ in rows]
            slots = self.ring_slots
            capacity = self.capacity
            batch = len(rows)
            block = self.block_size
            dtype = getattr(model, "context_dtype", mx.bfloat16)

            full_contexts, full_bases, full_lengths = [], [], []
            local_segments, local_bases, local_lengths = [], [], []
            for _state, row, context, _anchor, _sampler in rows:
                prior = row.restored if commit else None
                if commit:
                    row.restored = None
                full_ctx = (
                    mx.concatenate([prior.astype(context.dtype), context], axis=1)
                    if prior is not None
                    else context
                )
                n_full = int(full_ctx.shape[1])
                full_contexts.append(full_ctx)
                full_bases.append(row.fed)
                full_lengths.append(n_full)

                skip_local = max(0, n_full - slots)
                loc = full_ctx[:, skip_local:] if skip_local else full_ctx
                local_segments.append(loc)
                local_bases.append(row.fed + skip_local)
                local_lengths.append(int(loc.shape[1]))

            loc_width = max(1, max(local_lengths))
            h_ctx_local = mx.concatenate(
                [
                    (
                        seg
                        if seg.shape[1] == loc_width
                        else mx.concatenate(
                            [
                                seg,
                                mx.zeros(
                                    (1, loc_width - seg.shape[1], seg.shape[2]),
                                    dtype=seg.dtype,
                                ),
                            ],
                            axis=1,
                        )
                    )
                    for seg in local_segments
                ],
                axis=0,
            )

            anchors = mx.concatenate([anchor for _, _, _, anchor, _ in rows]).astype(
                mx.int32
            )
            markov = getattr(model, "markov_head", None)
            anchor_slot = markov is not None and bool(
                getattr(model.config, "sample_from_anchor", False)
            )
            if anchor_slot:
                block -= 1
            masks = mx.full(
                (batch, block - 1), int(model.config.mask_token_id), dtype=mx.int32
            )
            inputs = mx.concatenate([anchors[:, None], masks], axis=1)
            h = model._embed_input_tokens(inputs)
            if self.mask_embedding is not None:
                fill = self.mask_embedding.astype(h.dtype)
                h = mx.concatenate(
                    [h[:, :1], mx.broadcast_to(fill, (batch, block - 1, h.shape[-1]))],
                    axis=1,
                )

            cohort = self._assemble_cohort(uids, dtype)
            local_base_arr = mx.array(local_bases, dtype=mx.int32)
            full_base_arr = mx.array(full_bases, dtype=mx.int32)
            if counts is None:
                totals = mx.array(
                    [b + n for b, n in zip(full_bases, full_lengths)], dtype=mx.int32
                )
            else:
                totals = full_base_arr + mx.concatenate(counts).astype(mx.int32)
            query_offsets = totals

            write_slots = [
                mx.array([(b + j) % capacity for j in range(n)], dtype=mx.int32)
                for b, n in zip(local_bases, local_lengths)
            ]
            pos = mx.stop_gradient(cohort.pos)
            for index, (b, n, slots_b) in enumerate(
                zip(local_bases, local_lengths, write_slots)
            ):
                if not n:
                    continue
                pos[index, slots_b] = mx.arange(b, b + n, dtype=mx.int32)
            ring_valid = (
                (pos >= 0)
                & (pos >= (totals - slots)[:, None])
                & (pos < totals[:, None])
            )
            local_mask = mx.concatenate(
                [ring_valid, mx.ones((batch, block), dtype=mx.bool_)], axis=1
            )[:, None, None, :]

            new_keys, new_values = [], []
            predraft_global: dict[int, tuple[mx.array, mx.array, list[int]]] = {}
            for layer_index, layer in enumerate(model.layers):
                attn = layer.self_attn
                residual = h
                x = layer.input_layernorm(h)

                qkvp = attn.__dict__.get("_qkv_proj")
                if qkvp is not None:
                    qkv = qkvp(x)
                    dq = attn.n_heads * attn.head_dim
                    dkv = attn.n_kv_heads * attn.head_dim
                    queries = qkv[..., :dq]
                    prop_keys = qkv[..., dq : dq + dkv]
                    prop_values = qkv[..., dq + dkv :]
                else:
                    queries = attn.q_proj(x)
                    prop_keys = attn.k_proj(x)
                    prop_values = attn.v_proj(x)
                queries = attn.q_norm(
                    queries.reshape(batch, block, attn.n_heads, -1)
                ).transpose(0, 2, 1, 3)
                prop_keys = attn.k_norm(
                    prop_keys.reshape(batch, block, attn.n_kv_heads, -1)
                ).transpose(0, 2, 1, 3)
                prop_values = prop_values.reshape(
                    batch, block, attn.n_kv_heads, -1
                ).transpose(0, 2, 1, 3)
                if self.value_scale is not None:
                    prop_values = prop_values * self.value_scale
                queries = model.rope(queries, offset=query_offsets)
                prop_keys = model.rope(prop_keys, offset=query_offsets)

                if layer_index in self._global_layers:
                    row_ks, row_vs, row_lens, row_starts = [], [], [], []
                    for index, (uid, fctx, fb, fn) in enumerate(
                        zip(uids, full_contexts, full_bases, full_lengths)
                    ):
                        ck_row, cv_row = self._project_ctx_kv_chunked(
                            attn, fctx, fb, model
                        )
                        k_buf, v_buf, vlen = self._append_global_kv(
                            uid, layer_index, ck_row, cv_row, fb, fn, commit
                        )
                        row_ks.append(k_buf)
                        row_vs.append(v_buf)
                        row_lens.append(vlen)
                        row_starts.append(self._global_start[uid])

                    max_len = max(row_lens) if row_lens else 0
                    if batch == 1:
                        ctx_k = row_ks[0][None, :, :max_len, :]
                        ctx_v = row_vs[0][None, :, :max_len, :]
                        if not commit:
                            predraft_global[layer_index] = (
                                row_ks[0][None],
                                row_vs[0][None],
                                row_lens,
                            )
                    else:
                        padded_k, padded_v = [], []
                        for kb, vb, vl in zip(row_ks, row_vs, row_lens):
                            ks_i = kb[:, :vl, :]
                            vs_i = vb[:, :vl, :]
                            if vl < max_len:
                                pad = mx.zeros(
                                    (kb.shape[0], max_len - vl, kb.shape[2]),
                                    dtype=kb.dtype,
                                )
                                ks_i = mx.concatenate([ks_i, pad], axis=1)
                                vs_i = mx.concatenate([vs_i, pad], axis=1)
                            padded_k.append(ks_i)
                            padded_v.append(vs_i)
                        ctx_k = mx.stack(padded_k, axis=0)
                        ctx_v = mx.stack(padded_v, axis=0)
                        if not commit:
                            predraft_global[layer_index] = (ctx_k, ctx_v, row_lens)

                    if batch == 1 and counts is None:
                        layer_mask = None
                    else:
                        g_pos = (
                            mx.array(row_starts, dtype=mx.int32)[:, None]
                            + mx.arange(max_len, dtype=mx.int32)[None, :]
                        )
                        g_valid = (
                            (
                                mx.arange(max_len, dtype=mx.int32)[None, :]
                                < mx.array(row_lens, dtype=mx.int32)[:, None]
                            )
                            & (g_pos < totals[:, None])
                        )
                        layer_mask = mx.concatenate(
                            [g_valid, mx.ones((batch, block), dtype=mx.bool_)], axis=1
                        )[:, None, None, :]
                    keys = mx.concatenate([ctx_k, prop_keys], axis=2)
                    values = mx.concatenate([ctx_v, prop_values], axis=2)
                    new_keys.append(cohort.keys[layer_index])
                    new_values.append(cohort.values[layer_index])
                else:
                    ctx_keys, ctx_values = attn.project_context_kv(h_ctx_local)
                    ctx_k_norm = getattr(attn, "k_norm_ctx", attn.k_norm)
                    ctx_keys = ctx_k_norm(
                        ctx_keys.reshape(
                            batch, loc_width, attn.n_kv_heads, attn.head_dim
                        )
                    ).transpose(0, 2, 1, 3)
                    ctx_values = ctx_values.reshape(
                        batch, loc_width, attn.n_kv_heads, attn.head_dim
                    ).transpose(0, 2, 1, 3)
                    if self.value_scale is not None:
                        ctx_values = ctx_values * self.value_scale
                    ctx_keys = model.rope(ctx_keys, offset=local_base_arr)

                    ring_keys = mx.stop_gradient(cohort.keys[layer_index])
                    ring_values = mx.stop_gradient(cohort.values[layer_index])
                    for index, (n, slots_b) in enumerate(
                        zip(local_lengths, write_slots)
                    ):
                        if not n:
                            continue
                        ring_keys[index, :, slots_b, :] = ctx_keys[index, :, :n, :]
                        ring_values[index, :, slots_b, :] = ctx_values[index, :, :n, :]
                    new_keys.append(ring_keys)
                    new_values.append(ring_values)
                    keys = mx.concatenate([ring_keys, prop_keys], axis=2)
                    values = mx.concatenate([ring_values, prop_values], axis=2)
                    layer_mask = local_mask

                attended = mx.fast.scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    scale=attn.scale,
                    mask=layer_mask,
                    sinks=getattr(attn, "attention_sink_bias", None),
                )
                attended = attended.transpose(0, 2, 1, 3).reshape(batch, block, -1)
                attended = attn.o_proj(variants.gate_output(attn, x, attended))
                h = residual + attended

                residual = h
                x = layer.post_attention_layernorm(h)
                x = variants.gated_mlp(layer.mlp, x)
                h = residual + x

            if commit:
                cohort.keys = new_keys
                cohort.values = new_values
                cohort.pos = pos
                for index, ((_state, row, *_), fb, fn, ln) in enumerate(
                    zip(rows, full_bases, full_lengths, local_lengths)
                ):
                    row.fed = fb + fn
                    self._remember(row, h_ctx_local[index : index + 1, :ln])

            draft_hidden = model.norm(h) if anchor_slot else model.norm(h)[:, 1:]
            if commit and markov is not None:
                max_proposals = max(
                    (self._proposal_count(state) for state, *_ in rows), default=0
                )
                if 0 < max_proposals < draft_hidden.shape[1]:
                    draft_hidden = draft_hidden[:, :max_proposals]

            samplers = [sampler for *_, sampler in rows]
            greedy = all(sampler is None for sampler in samplers)
            if markov is not None:
                newest, held = h_ctx_local, local_lengths
                if counts is not None:
                    last = totals - 1 - local_base_arr
                    newest = mx.take_along_axis(
                        h_ctx_local, last[:, None, None], axis=1
                    )
                    held = [1] * batch
                markov.prepare_block(draft_hidden, newest, held, uids)
                if greedy and markov.screen_enabled and mx.default_device() == mx.gpu:
                    approx_base = markov.screen_base(draft_hidden)
                    prev = anchors.reshape(-1)
                    tokens_steps, conf_steps = [], []
                    for step in range(approx_base.shape[1]):
                        prev, conf = markov.apply_step_greedy_with_conf(
                            approx_base[:, step], prev
                        )
                        tokens_steps.append(prev)
                        conf_steps.append(conf)
                    proposals = [
                        ([tok[index : index + 1] for tok in tokens_steps], [])
                        for index in range(batch)
                    ]
                    row_confs = [
                        [c[index : index + 1] for c in conf_steps]
                        for index in range(batch)
                    ]
                    if commit:
                        for uid, rc in zip(uids, row_confs):
                            self._last_conf[uid] = rc
                    else:
                        self._predraft_conf = row_confs
                else:
                    logits = model._logits(draft_hidden)
                    proposals = selection.select_markov(markov, logits, anchors, samplers)
            else:
                logits = model._logits(draft_hidden)
                proposals = selection.sample_positions(logits, samplers)

            if not commit:
                self._predraft = _Predraft(
                    uids=tuple(uids),
                    keys=new_keys,
                    values=new_values,
                    pos=pos,
                    bases=full_bases,
                    unfed=[],
                    proposals=proposals,
                    contexts=[
                        h_ctx_local[i : i + 1, :n] for i, n in enumerate(local_lengths)
                    ],
                )
                self._predraft_global = predraft_global if predraft_global else None
            return proposals

    return Inkling2DSparkDrafter
'''


def patch_dspark(txt: str) -> str:
    """Apply PR #5510 (2-bit screened head, split-RNN Markov, global layer-1 KV)
    with extended bits in (4, 6, 8) affine head screening."""
    txt = _replace_once(
        txt,
        "Every layer, including the full-attention one, sees only the newest\n"
        "``sliding_window_size - 1`` context positions.\n",
        "Local layers (``local_layer_ids``, e.g. ``[0, 2]``) attend to the newest\n"
        "``sliding_window_size - 1`` context positions in a bounded ring, while global\n"
        "layers (layer ``1``) attend to the full committed context history.\n",
        "dspark doc",
    )
    txt = _replace_once(
        txt,
        "            final_logit_softcapping=float(text.get(\"final_logit_softcapping\") or 0),\n"
        "        )\n",
        "            final_logit_softcapping=float(text.get(\"final_logit_softcapping\") or 0),\n"
        "        )\n"
        + _DSPARK_2BIT_KERNELS_SRC,
        "dspark 2bit kernels",
    )
    txt = _replace_once(
        txt,
        "def mix_rmsnorm(context: mx.array, alpha: mx.array, weight: mx.array, eps: float):",
        "@mx.compile\ndef mix_rmsnorm(context: mx.array, alpha: mx.array, weight: mx.array, eps: float):",
        "dspark mix_rmsnorm compile",
    )
    txt = _replace_once(
        txt,
        "        self._softcap = config.final_logit_softcapping\n",
        "        self._softcap = config.final_logit_softcapping\n"
        "        self._unpadded_vocab_size = int(config.unpadded_vocab_size or config.vocab_size)\n",
        "dspark unpadded_vocab_size",
    )
    txt = _replace_once(
        txt,
        "        self._fold_weight = None\n        self._fold_bias = None\n",
        "        self._fold_weight = None\n"
        "        self._fold_wq = None\n"
        "        self._fold_scales = None\n"
        "        self._fold_biases = None\n"
        "        self._fold_bias = None\n"
        "        self._head_wq = None\n"
        "        self._head_scales = None\n"
        "        self._head_biases = None\n"
        "        self._head_dense = None\n"
        "        self._head_gs = 64\n"
        "        self._head_bits = 8\n"
        "        self._head_sc_wq = None\n"
        "        self._head_sc_sc = None\n"
        "        self._fold_sc_wq = None\n"
        "        self._fold_sc_sc = None\n"
        "        self._fold_bp = None\n"
        "        self._screen_tiles = 0\n"
        "        self._cap_arr = mx.array([self._softcap], dtype=mx.float32)\n"
        "        self._rnn_w_sp = None\n"
        "        self._rnn_w_h = None\n"
        "        self._rnn_w_f = None\n"
        "        self._rnn_ctx = None\n",
        "dspark head init",
    )
    txt = _replace_once(
        txt,
        "    def fold(self, lm_head: nn.Module) -> None:\n"
        '        """Fold the target LM head into the latent projection."""\n',
        "    @property\n"
        "    def screen_enabled(self) -> bool:\n"
        "        return self._head_sc_wq is not None and self._fold_sc_wq is not None\n\n"
        "    def fold(self, lm_head: nn.Module) -> None:\n"
        '        """Fold the target LM head into the latent projection and build 2-bit screen copies."""\n'
        "        lm_bits = int(getattr(lm_head, \"bits\", 0))\n"
        "        is_q_affine = (\n"
        "            isinstance(lm_head, nn.QuantizedLinear)\n"
        "            and getattr(lm_head, \"mode\", \"affine\") == \"affine\"\n"
        "            and lm_bits in (4, 6, 8)\n"
        "            and lm_head.get(\"biases\") is not None\n"
        "            and int(lm_head.group_size) % 32 == 0\n"
        "        )\n",
        "dspark fold head",
    )
    txt = _replace_once(
        txt,
        "        self._fold_weight = mx.concatenate(rows, axis=0)\n"
        "        self._fold_bias = mx.concatenate(biases, axis=0)\n"
        "        mx.eval(self._fold_weight, self._fold_bias)\n",
        "        folded = mx.concatenate(rows, axis=0)\n"
        "        self._fold_bias = mx.concatenate(biases, axis=0)\n"
        "        self._head_wq = self._head_scales = self._head_biases = self._head_dense = None\n"
        "        self._head_sc_wq = self._head_sc_sc = None\n"
        "        self._fold_sc_wq = self._fold_sc_sc = self._fold_bp = None\n"
        "        self._screen_tiles = 0\n"
        "        if (\n"
        "            folded.shape[0] >= 1024\n"
        "            and self.rank % 128 == 0\n"
        "            and self.rank >= _MARKOV_FOLD_GROUP_SIZE\n"
        "        ):\n"
        "            self._fold_wq, self._fold_scales, self._fold_biases = mx.quantize(\n"
        "                folded,\n"
        "                group_size=_MARKOV_FOLD_GROUP_SIZE,\n"
        "                bits=_MARKOV_FOLD_BITS,\n"
        "                mode=\"affine\",\n"
        "            )\n"
        "            self._fold_weight = None\n"
        "            eval_list = [\n"
        "                self._fold_wq,\n"
        "                self._fold_scales,\n"
        "                self._fold_biases,\n"
        "                self._fold_bias,\n"
        "            ]\n"
        "            h_dim = int(weight.shape[1])\n"
        "            can_screen = (\n"
        "                mx.default_device() == mx.gpu\n"
        "                and h_dim % 64 == 0\n"
        "                and self.rank % 64 == 0\n"
        "                and (\n"
        "                    is_q_affine\n"
        "                    or (\n"
        "                        not isinstance(lm_head, nn.QuantizedLinear)\n"
        "                        and lm_head.weight.dtype in (mx.bfloat16, mx.float16, mx.float32)\n"
        "                    )\n"
        "                )\n"
        "            )\n"
        "            if can_screen:\n"
        "                if is_q_affine:\n"
        "                    self._head_wq = lm_head.weight\n"
        "                    self._head_scales = lm_head.scales\n"
        "                    self._head_biases = lm_head.biases\n"
        "                    self._head_gs = int(lm_head.group_size)\n"
        "                    self._head_bits = lm_bits\n"
        "                    self._head_sc_wq, self._head_sc_sc, t_tiles = _dspark_screen_pack(\n"
        "                        v_unpad=self._unpadded_vocab_size,\n"
        "                        n_in=h_dim,\n"
        "                        w_q=self._head_wq,\n"
        "                        w_scales=self._head_scales,\n"
        "                        w_biases=self._head_biases,\n"
        "                        src_gs=self._head_gs,\n"
        "                        src_bits=self._head_bits,\n"
        "                    )\n"
        "                else:\n"
        "                    self._head_dense = lm_head.weight\n"
        "                    self._head_gs = 64\n"
        "                    self._head_bits = 0\n"
        "                    self._head_sc_wq, self._head_sc_sc, t_tiles = _dspark_screen_pack(\n"
        "                        v_unpad=self._unpadded_vocab_size,\n"
        "                        n_in=h_dim,\n"
        "                        w_dense=self._head_dense,\n"
        "                    )\n"
        "                self._fold_sc_wq, self._fold_sc_sc, _ = _dspark_screen_pack(\n"
        "                    v_unpad=self._unpadded_vocab_size,\n"
        "                    n_in=self.rank,\n"
        "                    w_q=self._fold_wq,\n"
        "                    w_scales=self._fold_scales,\n"
        "                    w_biases=self._fold_biases,\n"
        "                    src_gs=_MARKOV_FOLD_GROUP_SIZE,\n"
        "                    src_bits=_MARKOV_FOLD_BITS,\n"
        "                )\n"
        "                self._fold_bp = _dspark_screen_order(\n"
        "                    self._fold_bias, self._unpadded_vocab_size, t_tiles\n"
        "                )\n"
        "                self._screen_tiles = t_tiles\n"
        "                rw = self.rnn_proj.weight.astype(mx.bfloat16)\n"
        "                r2 = 2 * self.rank\n"
        "                self._rnn_w_sp = mx.contiguous(rw[:, :r2])\n"
        "                self._rnn_w_h = mx.contiguous(rw[:, r2 : r2 + h_dim])\n"
        "                self._rnn_w_f = mx.contiguous(rw[:, r2 + h_dim :])\n"
        "                eval_list.extend(\n"
        "                    [\n"
        "                        self._head_sc_wq,\n"
        "                        self._head_sc_sc,\n"
        "                        self._fold_sc_wq,\n"
        "                        self._fold_sc_sc,\n"
        "                        self._fold_bp,\n"
        "                        self._cap_arr,\n"
        "                        self._rnn_w_sp,\n"
        "                        self._rnn_w_h,\n"
        "                        self._rnn_w_f,\n"
        "                    ]\n"
        "                )\n"
        "            mx.eval(eval_list)\n"
        "        else:\n"
        "            self._fold_weight = folded\n"
        "            self._fold_wq = self._fold_scales = self._fold_biases = None\n"
        "            mx.eval(self._fold_weight, self._fold_bias)\n\n"
        "    def screen_base(self, draft_hidden: mx.array) -> mx.array:\n"
        "        if not self.screen_enabled:\n"
        "            raise RuntimeError(\"2-bit screened head is not initialized\")\n"
        "        return _dspark_head_base(\n"
        "            draft_hidden, self._head_sc_wq, self._head_sc_sc, self._screen_tiles\n"
        "        )\n\n"
        "    def _is_screened_base(self, base_logits: mx.array | None) -> bool:\n"
        "        return (\n"
        "            self.screen_enabled\n"
        "            and base_logits is not None\n"
        "            and base_logits.dtype == mx.float16\n"
        "            and base_logits.shape[-1] == self._screen_tiles * 32\n"
        "        )\n\n"
        "    def apply_step_greedy_with_conf(\n"
        "        self, base_logits: mx.array, prev_tokens: mx.array\n"
        "    ) -> tuple[mx.array, mx.array]:\n"
        "        if (\n"
        "            self._is_screened_base(base_logits)\n"
        "            and mx.default_device() == mx.gpu\n"
        "            and self._hidden is not None\n"
        "        ):\n"
        "            step_idx = self._step\n"
        "            h_step = self._hidden[:, step_idx]\n"
        "            latent, lat_f16 = self._step_latent_pair(prev_tokens)\n"
        "            return _dspark_screened_pick(\n"
        "                h_step=h_step,\n"
        "                lat_step=latent,\n"
        "                lat_f16=lat_f16,\n"
        "                approx_base=base_logits,\n"
        "                fold_sc_wq=self._fold_sc_wq,\n"
        "                fold_sc_sc=self._fold_sc_sc,\n"
        "                fold_bp=self._fold_bp,\n"
        "                head_wq=self._head_wq,\n"
        "                head_s=self._head_scales,\n"
        "                head_b=self._head_biases,\n"
        "                head_dense=self._head_dense,\n"
        "                head_gs=self._head_gs,\n"
        "                head_bits=self._head_bits,\n"
        "                fold_wq=self._fold_wq,\n"
        "                fold_s=self._fold_scales,\n"
        "                fold_b=self._fold_biases,\n"
        "                fold_bias=self._fold_bias,\n"
        "                v_unpad=self._unpadded_vocab_size,\n"
        "                t_tiles=self._screen_tiles,\n"
        "                cap_arr=self._cap_arr,\n"
        "            )\n"
        "        logits = self.apply_step_logits(base_logits, prev_tokens)\n"
        "        tok = mx.argmax(logits, axis=-1).astype(mx.uint32)\n"
        "        m = mx.max(logits, axis=-1, keepdims=True)\n"
        "        conf = mx.exp(mx.zeros_like(m[..., 0])) / mx.sum(mx.exp(logits - m), axis=-1)\n"
        "        return tok, conf.astype(mx.float32)\n",
        "dspark fold body",
    )
    txt = _replace_once(
        txt,
        "        self._hidden = draft_hidden\n"
        "        self._state = mx.zeros((len(uids), self.rank), dtype=draft_hidden.dtype)\n"
        "        self._step = 0\n",
        "        self._hidden = draft_hidden\n"
        "        self._state = mx.zeros((len(uids), self.rank), dtype=draft_hidden.dtype)\n"
        "        self._step = 0\n"
        "        if (\n"
        "            self._rnn_w_sp is not None\n"
        "            and mx.default_device() == mx.gpu\n"
        "            and draft_hidden.dtype == mx.bfloat16\n"
        "        ):\n"
        "            ctx_f = (\n"
        "                self._rows_f.astype(mx.bfloat16) @ self._rnn_w_f.T\n"
        "                + self.rnn_proj.bias.astype(mx.bfloat16)\n"
        "            )\n"
        "            ctx_h = draft_hidden @ self._rnn_w_h.T\n"
        "            self._rnn_ctx = (ctx_h.astype(mx.float32) + ctx_f[:, None, :].astype(mx.float32))\n"
        "        else:\n"
        "            self._rnn_ctx = None\n",
        "dspark prepare_block",
    )
    txt = _replace_once(
        txt,
        "    def apply_step_logits(self, base_logits: mx.array, prev_tokens: mx.array):\n"
        "        state = self._state\n",
        "    def _step_latent_pair(self, prev_tokens: mx.array) -> tuple[mx.array, mx.array | None]:\n"
        "        if (\n"
        "            self._rnn_w_sp is not None\n"
        "            and mx.default_device() == mx.gpu\n"
        "            and self._state.dtype == mx.bfloat16\n"
        "            and self.markov_w1.weight.dtype == mx.bfloat16\n"
        "        ):\n"
        "            if self._rnn_ctx is None:\n"
        "                ctx_f = (\n"
        "                    self._rows_f.astype(mx.bfloat16) @ self._rnn_w_f.T\n"
        "                    + self.rnn_proj.bias.astype(mx.bfloat16)\n"
        "                )\n"
        "                ctx_h = self._hidden.astype(mx.bfloat16) @ self._rnn_w_h.T\n"
        "                self._rnn_ctx = (\n"
        "                    ctx_h.astype(mx.float32) + ctx_f[:, None, :].astype(mx.float32)\n"
        "                )\n"
        "            ctx_step = self._rnn_ctx[:, self._step]\n"
        "            self._state, lat_bf16, lat_f16 = _dspark_markov_step_fused(\n"
        "                self._state,\n"
        "                prev_tokens,\n"
        "                ctx_step,\n"
        "                self._rnn_w_sp,\n"
        "                self.markov_w1.weight,\n"
        "            )\n"
        "            self._step += 1\n"
        "            return lat_bf16, lat_f16\n"
        "        return self._step_latent(prev_tokens), None\n\n"
        "    def _step_latent(self, prev_tokens: mx.array) -> mx.array:\n"
        "        state = self._state\n",
        "dspark step_latent",
    )
    txt = _replace_once(
        txt,
        "        self._state = kept + blend\n"
        "        latent = mx.tanh(proj[:, 2 * r :]).astype(dtype)\n"
        "        bias = latent @ self._fold_weight.T + self._fold_bias\n"
        "        self._step += 1\n",
        "        self._state = kept + blend\n"
        "        self._step += 1\n"
        "        return mx.tanh(proj[:, 2 * r :]).astype(dtype)\n\n"
        "    def apply_step_greedy(\n"
        "        self, base_logits: mx.array, prev_tokens: mx.array\n"
        "    ) -> mx.array:\n"
        "        if (\n"
        "            self._is_screened_base(base_logits)\n"
        "            and mx.default_device() == mx.gpu\n"
        "            and self._hidden is not None\n"
        "        ):\n"
        "            return self.apply_step_greedy_with_conf(base_logits, prev_tokens)[0]\n"
        "        if (\n"
        "            self._fold_wq is not None\n"
        "            and mx.default_device() == mx.gpu\n"
        "            and base_logits.dtype == mx.bfloat16\n"
        "            and self._state.dtype == mx.bfloat16\n"
        "            and self.rank % 128 == 0\n"
        "        ):\n"
        "            latent = self._step_latent(prev_tokens)\n"
        "            return _markov_q8_greedy_argmax(\n"
        "                latent,\n"
        "                self._fold_wq,\n"
        "                self._fold_scales,\n"
        "                self._fold_biases,\n"
        "                self._fold_bias,\n"
        "                base_logits,\n"
        "                self._unpadded_vocab_size,\n"
        "            )\n"
        "        return mx.argmax(self.apply_step_logits(base_logits, prev_tokens), axis=-1)\n\n"
        "    def apply_step_logits(self, base_logits: mx.array, prev_tokens: mx.array):\n"
        "        latent = self._step_latent(prev_tokens)\n"
        "        if self._fold_wq is not None:\n"
        "            bias = (\n"
        "                mx.quantized_matmul(\n"
        "                    latent,\n"
        "                    self._fold_wq,\n"
        "                    self._fold_scales,\n"
        "                    self._fold_biases,\n"
        "                    transpose=True,\n"
        "                    group_size=_MARKOV_FOLD_GROUP_SIZE,\n"
        "                    bits=_MARKOV_FOLD_BITS,\n"
        "                    mode=\"affine\",\n"
        "                )\n"
        "                + self._fold_bias\n"
        "            )\n"
        "        else:\n"
        "            bias = latent @ self._fold_weight.T + self._fold_bias\n",
        "dspark apply_step_greedy",
    )
    txt = _replace_once(
        txt,
        "    mx.eval(model.parameters())\n"
        "    return model\n\n\n"
        "def wrap_inkling2_dspark(",
        "    mx.eval(model.parameters())\n"
        "    return model\n"
        + _DSPARK_DRAUGHT_CLS_SRC
        + "\n\ndef wrap_inkling2_dspark(",
        "dspark drafter cls",
    )
    txt = _replace_once(
        txt,
        "    from ...speculative.dflash import DFlashDrafter\n\n"
        "    drafter = DFlashDrafter(\n",
        "    drafter_cls = _make_inkling2_dspark_drafter_cls()\n"
        "    drafter = drafter_cls(\n",
        "dspark wrap",
    )
    return txt


def patch_selection_dspark_2bit(txt: str) -> str:
    return _replace_once(
        txt,
        "    prev = anchors.reshape(-1)\n"
        "    tokens, accepts = [], [[] for _ in range(batch)]\n"
        "    for step in range(logits.shape[1]):\n"
        "        step_logits = markov.apply_step_logits(logits[:, step], prev)\n"
        "        if all(sampler is None for sampler in samplers):\n"
        "            prev = mx.argmax(step_logits, axis=-1)\n"
        "        else:\n"
        "            lp = step_logits - mx.logsumexp(step_logits, axis=-1, keepdims=True)\n"
        "            picks = []\n"
        "            for index, sampler in enumerate(samplers):\n"
        "                if sampler is None:\n"
        "                    picks.append(mx.argmax(step_logits[index : index + 1], axis=-1))\n"
        "                    continue\n"
        "                token, accept_lp = bg._sample_draft_with_logprobs(\n"
        "                    sampler, lp[index : index + 1]\n"
        "                )\n"
        "                picks.append(token.reshape(1))\n"
        "                accepts[index].append(accept_lp.reshape(-1))\n"
        "            prev = mx.concatenate(picks)\n",
        "    prev = anchors.reshape(-1)\n"
        "    tokens, accepts = [], [[] for _ in range(batch)]\n"
        "    greedy = all(sampler is None for sampler in samplers)\n"
        "    greedy_step = getattr(markov, \"apply_step_greedy\", None) if greedy else None\n"
        "    for step in range(logits.shape[1]):\n"
        "        if greedy_step is not None:\n"
        "            prev = greedy_step(logits[:, step], prev)\n"
        "        else:\n"
        "            step_logits = markov.apply_step_logits(logits[:, step], prev)\n"
        "            if greedy:\n"
        "                prev = mx.argmax(step_logits, axis=-1)\n"
        "            else:\n"
        "                lp = step_logits - mx.logsumexp(step_logits, axis=-1, keepdims=True)\n"
        "                picks = []\n"
        "                for index, sampler in enumerate(samplers):\n"
        "                    if sampler is None:\n"
        "                        picks.append(mx.argmax(step_logits[index : index + 1], axis=-1))\n"
        "                        continue\n"
        "                    token, accept_lp = bg._sample_draft_with_logprobs(\n"
        "                        sampler, lp[index : index + 1]\n"
        "                    )\n"
        "                    picks.append(token.reshape(1))\n"
        "                    accepts[index].append(accept_lp.reshape(-1))\n"
        "                prev = mx.concatenate(picks)\n",
        "selection select_markov",
    )


def patch_drafter(txt: str, features: set[str]) -> str:
    if "suffix_policy" in features:
        txt = _replace_once(
            txt,
            "        # A context copy that fills the block may replace the drafted block\n"
            "        # (Lightning MTP's prompt lookup); the target still verifies it.\n"
            "        self.context_copy = False\n",
            "        # A context copy that fills the block may replace the drafted block\n"
            "        # (Lightning MTP's prompt lookup); the target still verifies it.\n"
            "        self.context_copy = False\n"
            "        self.confidence_windows = False\n",
            "drafter init conf",
        )
        txt = _replace_once(
            txt,
            "    def _proposal_count(self, state) -> int:\n"
            '        """Drafts the row\'s depth controller asks to verify next."""\n'
            "        count = min(self.depth, int(getattr(state, \"depth\", 0) or self.depth))\n",
            "    def _block_confidence(self):\n"
            "        hook = getattr(\n"
            "            getattr(self.model, \"markov_head\", None), \"block_confidence\", None\n"
            "        )\n"
            "        return hook if callable(hook) else None\n\n"
            "    states_chances = False\n\n"
            "    def _proposal_count(self, state) -> int:\n"
            '        """Drafts the row\'s depth controller asks to verify next."""\n'
            "        count = min(self.depth, int(getattr(state, \"depth\", 0) or self.depth))\n"
            "        if self.confidence_windows and (\n"
            "            self.states_chances or self._block_confidence() is not None\n"
            "        ):\n"
            "            return count\n",
            "drafter proposal_count",
        )
        txt = _replace_once(
            txt,
            "    def _commit_proposals(self, state, tokens, accept_lps) -> None:\n"
            '        """Hand the row its drafts and their draft distributions."""\n'
            "        if isinstance(tokens, list):\n"
            "            # DSpark keeps one lazy token per position; only the prefix the\n"
            "            # row's depth controller verifies next is evaluated.\n"
            "            count = self._proposal_count(state)\n"
            "            tokens = (\n"
            "                mx.concatenate(tokens[:count])\n"
            "                if count\n"
            "                else mx.zeros((0,), dtype=mx.uint32)\n"
            "            )\n"
            "            accept_lps = accept_lps[:count]\n"
            "        state.drafts = tokens.reshape(-1).astype(mx.uint32)\n"
            "        state.draft_lps = []\n"
            "        state.draft_accept_lps = accept_lps\n",
            "    def _commit_proposals(self, state, tokens, accept_lps, conf=None) -> None:\n"
            '        """Hand the row its drafts, their draft distributions and conf."""\n'
            "        if isinstance(tokens, list):\n"
            "            count = self._proposal_count(state)\n"
            "            tokens = (\n"
            "                mx.concatenate(tokens[:count])\n"
            "                if count\n"
            "                else mx.zeros((0,), dtype=mx.uint32)\n"
            "            )\n"
            "            accept_lps = accept_lps[:count]\n"
            "            if conf is not None:\n"
            "                conf = conf.reshape(-1)[:count]\n"
            "        state.drafts = tokens.reshape(-1).astype(mx.uint32)\n"
            "        state.draft_lps = []\n"
            "        state.draft_accept_lps = accept_lps\n"
            "        state.draft_conf = conf if self.confidence_windows else None\n",
            "drafter commit_proposals",
        )
        txt = _replace_once(
            txt,
            "        rows = []\n"
            "        for gen_batch, state, captured, committed, _prev_buf in jobs:\n",
            "        if not jobs:\n"
            "            return\n"
            "        rows = []\n"
            "        for gen_batch, state, captured, committed, _prev_buf in jobs:\n",
            "drafter draft empty",
        )
        txt = _replace_once(
            txt,
            "        for (state, *_), (tokens, accept_lps) in zip(rows, proposals):\n"
            "            self._commit_proposals(state, tokens, accept_lps)\n",
            "        for (state, *_), proposal in zip(rows, proposals):\n"
            "            self._commit_proposals(state, *proposal)\n",
            "drafter draft unpack",
        )
        txt = _replace_once(
            txt,
            "        self._predraft.unfed = [unfed]\n"
            "        tokens, accept_lps = proposals[0]\n",
            "        self._predraft.unfed = [unfed]\n"
            "        tokens, accept_lps, *_ = proposals[0]\n",
            "drafter predraft unpack",
        )
        txt = _replace_once(
            txt,
            "        tokens, accept_lps = pre.proposals[0]\n"
            "        self._commit_proposals(state, tokens, accept_lps)\n",
            "        self._commit_proposals(state, *pre.proposals[0])\n",
            "drafter adopt_predraft",
        )

    if "dspark_2bit" in features:
        txt = _replace_once(
            txt,
            "        draft_hidden = model.norm(h) if anchor_slot else model.norm(h)[:, 1:]\n"
            "        if wide:\n"
            "            draft_hidden = draft_hidden.astype(dtype)\n"
            "        logits = model._logits(draft_hidden)\n",
            "        draft_hidden = model.norm(h) if anchor_slot else model.norm(h)[:, 1:]\n"
            "        if wide:\n"
            "            draft_hidden = draft_hidden.astype(dtype)\n"
            "        if commit and markov is not None:\n"
            "            max_proposals = max(\n"
            "                (self._proposal_count(state) for state, *_ in rows), default=0\n"
            "            )\n"
            "            if 0 < max_proposals < draft_hidden.shape[1]:\n"
            "                draft_hidden = draft_hidden[:, :max_proposals]\n"
            "        logits = model._logits(draft_hidden)\n",
            "drafter markov tail skip",
        )

    if {"predraft_fix", "dspark_2bit", "suffix_policy"} & features:
        txt = _replace_once(
            txt,
            "            if prepare is not None:\n"
            "                # Recurrent Markov heads start each block from the newest\n"
            "                # committed context row.\n"
            "                prepare(draft_hidden, h_ctx, lengths, uids)\n",
            "            if prepare is not None:\n"
            "                newest, held = h_ctx, lengths\n"
            "                if counts is not None:\n"
            "                    last = mx.concatenate(counts).astype(mx.int32) - 1\n"
            "                    newest = mx.take_along_axis(h_ctx, last[:, None, None], axis=1)\n"
            "                    held = [1] * batch\n"
            "                prepare(draft_hidden, newest, held, uids)\n",
            "drafter predraft_fix",
        )

    if "suffix_policy" in features:
        txt = _replace_once(
            txt,
            "            proposals = selection.select_markov(markov, logits, anchors, samplers)\n",
            "            proposals = selection.select_markov(markov, logits, anchors, samplers)\n"
            "            confidence = self._block_confidence()\n"
            "            if confidence is not None and self.confidence_windows:\n"
            "                chances = confidence()\n"
            "                proposals = [\n"
            "                    (tokens, accept_lps, chances[index])\n"
            "                    for index, (tokens, accept_lps) in enumerate(proposals)\n"
            "                ]\n",
            "drafter block_confidence",
        )
    return txt


# ---------------------------------------------------------------------------
# PR #5507 + PR #5509 (batch_generator.py)
# ---------------------------------------------------------------------------

_EARLY_FIRST_HELPERS_SRC = '''

def _drop_invalid_early_first_token(gen_batch: Any) -> None:
    """Clear a pending early first-token marker when its row is no longer sole owner."""
    emitted = getattr(gen_batch, "_omlx_first_token_emitted", None)
    if emitted is None:
        return
    uids = getattr(gen_batch, "uids", None)
    if not uids or tuple(uids) != (emitted[0],):
        try:
            delattr(gen_batch, "_omlx_first_token_emitted")
        except AttributeError:
            pass


def _reconcile_early_first_token_to_standard(gen_batch: Any) -> None:
    """Advance ``gen_batch`` past an already-streamed first token before standard decode."""
    emitted = getattr(gen_batch, "_omlx_first_token_emitted", None)
    if emitted is None:
        return
    try:
        delattr(gen_batch, "_omlx_first_token_emitted")
    except AttributeError:
        pass
    uids = getattr(gen_batch, "uids", None)
    if not uids or tuple(uids) != (emitted[0],) or gen_batch._next_tokens is None:
        return
    saved_len = len(gen_batch.tokens[0])
    gen_batch._step()
    if len(gen_batch.tokens[0]) > saved_len:
        del gen_batch.tokens[0][saved_len:]


def _early_first_emit(gen_batch: Any) -> Optional[List[Any]]:
    """Stream the prompt-sampled first token before MTP init and first draft."""
    if getattr(gen_batch, "_omlx_mtp_state", None) is not None:
        return None
    if gen_batch._next_tokens is None or len(gen_batch.uids) != 1:
        return None
    if gen_batch._num_tokens[0] != 0:
        return None
    if _mtp_park_state_for_batch(gen_batch) is not None:
        return None
    if _cluster_mtp(gen_batch.model) is not None:
        return None
    align = int(getattr(gen_batch.model, "_omlx_mtp_commit_align", 0) or 0)
    if align > 0 and (len(gen_batch.tokens[0]) + 1) % align == 0:
        return None
    main_tok = _ensure_uint32(gen_batch._next_tokens)
    token_id = int(main_tok.tolist()[0])
    result = _emit_response(gen_batch, token_id, gen_batch._next_logprobs[0])
    if result[0].finish_reason is None:
        gen_batch._omlx_first_token_emitted = (gen_batch.uids[0], token_id)
    return result


def _take_early_first_token(gen_batch: Any, state: Optional[_MtpState]) -> None:
    """Drop the init queue entry of a token ``_early_first_emit`` streamed."""
    emitted = getattr(gen_batch, "_omlx_first_token_emitted", None)
    if emitted is None:
        return
    try:
        delattr(gen_batch, "_omlx_first_token_emitted")
    except AttributeError:
        pass
    if state is None or not state.queue:
        return
    token_id, _lp, source = state.queue[0]
    if emitted == (state.uid, token_id) and source == "init":
        state.queue.popleft()
        _bump_emit_stat(state, "init")
'''

_DRAFT_POLICY_HELPERS_SRC = '''

_DRAFT_POLICY_ENV = "OMLX_MTP_DRAFT_POLICY"


def _draft_policy_for(model: Any) -> Optional[Any]:
    """The model's draft policy (enabled by default in suffix_policy overlay unless env=0)."""
    hosts = [
        host
        for host in (
            model,
            getattr(model, "_language_model", None),
            getattr(model, "language_model", None),
        )
        if host is not None
    ]
    flag = None
    for host in hosts:
        flag = getattr(host, "_omlx_mtp_draft_policy", None)
        if flag is True or flag is False:
            break
        flag = None
    if flag is None:
        env = os.environ.get(_DRAFT_POLICY_ENV)
        if env is None or env.strip() == "":
            flag = True
        else:
            flag = env.strip().lower() in ("1", "true", "on")
    if not flag:
        return None
    policy = getattr(hosts[0], "_omlx_mtp_draft_policy_state", None)
    if not isinstance(policy, _draft_policy.DraftPolicy):
        policy = _draft_policy.DraftPolicy(_draft_policy.K_MAX)
        hosts[0]._omlx_mtp_draft_policy_state = policy
    return policy


def _stated_chances(state: _MtpState) -> Optional[List[float]]:
    """The draft head's chance per pending draft, read to the host (one sync)."""
    import mlx.core as mx

    conf = state.draft_conf
    if conf is None:
        return None
    if isinstance(conf, (list, tuple)):
        if not conf:
            return None
        conf = mx.concatenate([c.reshape(-1) for c in conf])
    return [float(c) for c in conf.reshape(-1).tolist()]


def _drafter_window(state: _MtpState) -> int:
    """Drafts of the draft head the next verify takes (its depth controller's)."""
    return int(state.controller.cur if state.controller is not None else state.depth)


def _lookup_window(model: Any, state: _MtpState, drafter: Optional[Any]) -> int:
    """Most lookup drafts one verify window of this target takes."""
    cap = _max_verify_rows(model)
    if drafter is not None:
        if cap is None:
            return int(drafter.depth)
        return max(int(drafter.depth), min(cap - 1, _draft_policy.K_MAX))
    if _dspark_host(model) is not None:
        return int(state.depth)
    if _row_exact_verify(model) and (cap is None or cap > _context_copy.MAX_COPY):
        return _context_copy.MAX_COPY
    return _context_copy._NARROW_COPY


def _policy_lookup(
    gen_batch: Any,
    state: _MtpState,
    policy: Any,
    committed_ids: List[int],
    drafter: Optional[Any],
    drafted: bool = False,
) -> List[int]:
    """The next window from the suffix lookup, when the draft policy takes it."""
    if state.suffix is None:
        state.suffix = SuffixDrafter()
    suffix = state.suffix
    state.copy_untimed = suffix.sync(gen_batch.tokens[0])
    suffix.append(committed_ids)
    state.lookup_tail = []
    state.lookup_drafted = False
    room = gen_batch.max_tokens[0] - gen_batch._num_tokens[0] - len(committed_ids) - 1
    cap = min(room, _lookup_window(gen_batch.model, state, drafter))
    if cap < 1:
        return []
    proposal, match = suffix.propose(2 * cap + 1)
    if not proposal:
        return []
    depth = _drafter_window(state)
    choice = policy.choose(
        lookup_k=min(cap, len(proposal)),
        match=match,
        drafter_k=depth,
        drafter_window=depth or None,
        drafted=drafted,
    )
    if choice.source is not _draft_policy.Source.LOOKUP:
        return []
    state.lookup_match = match
    state.lookup_tail = proposal[choice.k :]
    state.lookup_drafted = drafted
    return proposal[: choice.k]


def _lookup_continues(
    gen_batch: Any, state: _MtpState, policy: Any, drafter: Any, k: int
) -> bool:
    """Whether the window after this lookup window would be a lookup's too."""
    tail = state.lookup_tail
    if len(tail) < 2:
        return False
    depth = _drafter_window(state)
    cap = _lookup_window(gen_batch.model, state, drafter)
    choice = policy.choose(
        lookup_k=min(cap, len(tail) - 1),
        match=state.lookup_match + k + 1,
        drafter_k=depth,
        drafter_window=depth or None,
    )
    return choice.source is _draft_policy.Source.LOOKUP
'''


def patch_batch_generator_early_emit(txt: str) -> str:
    txt = _replace_once(
        txt,
        "            if _is_mtp_eligible(self):\n"
        "                handed_off = False\n"
        "                if _singleton_mtp_handoff_ready(self):\n"
        "                    # A prefill is waiting on this batch generator; hand the\n"
        "                    # singleton back to the standard step at the drained-queue\n"
        "                    # boundary so the late join merges this very call (#2515).\n"
        "                    handed_off = _handoff_mtp_for_late_join(self, self._omlx_mtp_state)\n"
        "                if not handed_off:\n"
        "                    try:\n"
        "                        state = _prepare_mtp_state_for_next(self)\n"
        "                        if state is not None:\n"
        "                            with _spec_command_buffers(model):\n"
        "                                return _mtp_next(self, state)\n"
        "                    except _MtpStepFallback as exc:\n",
        "            emitted = getattr(self, \"_omlx_first_token_emitted\", None)\n"
        "            if emitted is not None and tuple(self.uids) != (emitted[0],):\n"
        "                delattr(self, \"_omlx_first_token_emitted\")\n"
        "                emitted = None\n"
        "            if emitted is not None or _is_mtp_eligible(self):\n"
        "                handed_off = False\n"
        "                if _singleton_mtp_handoff_ready(self):\n"
        "                    # A prefill is waiting on this batch generator; hand the\n"
        "                    # singleton back to the standard step at the drained-queue\n"
        "                    # boundary so the late join merges this very call (#2515).\n"
        "                    handed_off = _handoff_mtp_for_late_join(self, self._omlx_mtp_state)\n"
        "                if not handed_off and emitted is None:\n"
        "                    early = _early_first_emit(self)\n"
        "                    if early is not None:\n"
        "                        return early\n"
        "                if not handed_off:\n"
        "                    try:\n"
        "                        state = _prepare_mtp_state_for_next(self)\n"
        "                        if state is not None:\n"
        "                            with _spec_command_buffers(model):\n"
        "                                return _mtp_next(self, state)\n"
        "                        if emitted is not None:\n"
        "                            _reconcile_early_first_token_to_standard(self)\n"
        "                    except _MtpStepFallback as exc:\n",
        "bg early_emit patched_next",
    )
    txt = _replace_once(
        txt,
        "                                    pass\n"
        "                        _drop_mtp_state(self, \"step-fallback\")\n",
        "                                    pass\n"
        "                        elif emitted is not None:\n"
        "                            _reconcile_early_first_token_to_standard(self)\n"
        "                        _drop_mtp_state(self, \"step-fallback\")\n",
        "bg early_emit fallback",
    )
    txt = _replace_once(
        txt,
        "            _drop_mtp_batch_state(batch, \"donor-extended\")\n\n"
        "            host_state = getattr(self, \"_omlx_mtp_state\", None)\n",
        "            _drop_mtp_batch_state(batch, \"donor-extended\")\n\n"
        "            if getattr(self, \"_omlx_first_token_emitted\", None) is not None:\n"
        "                _reconcile_early_first_token_to_standard(self)\n"
        "            host_state = getattr(self, \"_omlx_mtp_state\", None)\n",
        "bg early_emit extend",
    )
    txt = _replace_once(
        txt,
        "            _release_drafter_uids(self.model, set(old_uids) - set(self.uids))\n"
        "            _drop_invalid_mtp_state(self, \"filter\", log_empty=True)\n",
        "            _release_drafter_uids(self.model, set(old_uids) - set(self.uids))\n"
        "            _drop_invalid_early_first_token(self)\n"
        "            _drop_invalid_mtp_state(self, \"filter\", log_empty=True)\n",
        "bg early_emit filter",
    )
    txt = _replace_once(
        txt,
        "        getattr(gen_batch, \"_omlx_mtp_state\", None) is not None\n"
        "        or getattr(gen_batch, \"_omlx_mtp_batch_state\", None) is not None\n",
        "        getattr(gen_batch, \"_omlx_mtp_state\", None) is not None\n"
        "        or getattr(gen_batch, \"_omlx_mtp_batch_state\", None) is not None\n"
        "        or getattr(gen_batch, \"_omlx_first_token_emitted\", None) is not None\n",
        "bg early_emit active_mtp",
    )
    txt = _replace_once(
        txt,
        "def _prepare_mtp_state_for_next(gen_batch: Any) -> Optional[_MtpState]:\n",
        _EARLY_FIRST_HELPERS_SRC + "\n\ndef _prepare_mtp_state_for_next(gen_batch: Any) -> Optional[_MtpState]:\n",
        "bg early_emit helpers",
    )
    txt = _replace_once(
        txt,
        "    if not _mtp_state_valid_for_batch(gen_batch, state):\n"
        "        _drop_mtp_state(gen_batch, \"post-init-invalid\")\n"
        "        return None\n",
        "    if not _mtp_state_valid_for_batch(gen_batch, state):\n"
        "        _drop_mtp_state(gen_batch, \"post-init-invalid\")\n"
        "        return None\n"
        "    _take_early_first_token(gen_batch, state)\n",
        "bg early_emit take",
    )
    return txt


def patch_batch_generator_suffix_policy(txt: str) -> str:
    txt = _replace_once(
        txt,
        "import math\nimport threading\n",
        "import math\nimport os\nimport threading\n",
        "bg suffix os",
    )
    txt = _replace_once(
        txt,
        "from omlx.prefill_progress import get_prefill_tracker\n",
        "from omlx.prefill_progress import get_prefill_tracker\n"
        "from omlx.speculative import draft_policy as _draft_policy\n"
        "from omlx.speculative.suffix_drafter import SuffixDrafter\n",
        "bg suffix imports",
    )
    txt = _replace_once(
        txt,
        "    copy_drafts: bool = False\n"
        "    copy_untimed: bool = False\n",
        "    copy_drafts: bool = False\n"
        "    copy_untimed: bool = False\n"
        "    suffix: Optional[Any] = None\n"
        "    lookup_match: int = 0\n"
        "    lookup_tail: List[int] = field(default_factory=list)\n"
        "    lookup_drafted: bool = False\n"
        "    draft_conf: Optional[Any] = None\n",
        "bg suffix _MtpState",
    )
    txt = _replace_once(
        txt,
        "        committed_ids = [int(t) for t in committed.tolist()]\n"
        "        copied = _context_copy_drafts(gen_batch, state, committed_ids)\n",
        "        committed_ids = [int(t) for t in committed.tolist()]\n"
        "        policy = _draft_policy_for(gen_batch.model)\n"
        "        copied = (\n"
        "            _context_copy_drafts(gen_batch, state, committed_ids)\n"
        "            if policy is None\n"
        "            else _policy_lookup(gen_batch, state, policy, committed_ids, None)\n"
        "        )\n",
        "bg suffix dspark_draft_jobs",
    )
    txt = _replace_once(
        txt,
        "            drafter.draft([(gen_batch, state, captured, state.next_main, prev_buf)])\n",
        "            drafter.confidence_windows = _draft_policy_for(gen_batch.model) is not None\n"
        "            drafter.draft([(gen_batch, state, captured, state.next_main, prev_buf)])\n",
        "bg suffix post_init_mtp",
    )
    txt = _replace_once(
        txt,
        "    if _drafter_for(gen_batch.model) is not None or _shared_verify_width(gen_batch.model):\n"
        "        # Clip before grouping so rows that share a window verify together. A shared\n"
        "        # verify host also drops drafts made deeper than the policy's (capped) depth.\n"
        "        for state in states:\n"
        "            if state.chain:\n"
        "                _clip_block_drafts(\n"
        "                    state,\n"
        "                    (\n"
        "                        state.controller.cur\n"
        "                        if state.controller is not None\n"
        "                        else state.depth\n"
        "                    ),\n"
        "                )\n"
        "    _clip_to_shared_window(states)\n",
        "    draft_policy = _draft_policy_for(gen_batch.model)\n"
        "    if _drafter_for(gen_batch.model) is not None or _shared_verify_width(gen_batch.model):\n"
        "        for state in states:\n"
        "            if state.chain and not (draft_policy is not None and state.copy_drafts):\n"
        "                _clip_block_drafts(\n"
        "                    state,\n"
        "                    (\n"
        "                        state.controller.cur\n"
        "                        if state.controller is not None\n"
        "                        else state.depth\n"
        "                    ),\n"
        "                )\n"
        "    _clip_to_shared_window(states, draft_policy)\n",
        "bg suffix batched verify",
    )
    txt = _replace_once(
        txt,
        "def _clip_to_shared_window(states: List[_MtpState]) -> None:\n"
        '    """Clip the rows verifying this cycle to their shortest proposal.\n\n'
        "    A copied window can be longer than the head's drafts. Rows of different\n"
        "    lengths verify per row from extracted caches, and merging those back copies\n"
        "    the whole batch KV. A chain or copy prefix is still a complete proposal.\n"
        '    """\n'
        "    lengths = [\n"
        "        int(state.drafts.shape[0])\n"
        "        for state in states\n"
        "        if state.chain and not state.queue and state.drafts is not None\n"
        "    ]\n"
        "    if len(set(lengths)) < 2 or min(lengths) < 1:\n"
        "        return\n"
        "    limit = min(lengths)\n"
        "    for state in states:\n"
        "        if state.chain and not state.queue and state.drafts is not None:\n"
        "            _clip_block_drafts(state, limit)\n",
        "def _clip_to_shared_window(\n"
        "    states: List[_MtpState], draft_policy: Optional[Any] = None\n"
        ") -> None:\n"
        '    """Clip the rows verifying this cycle to their shortest proposal."""\n'
        "    rows = [\n"
        "        state\n"
        "        for state in states\n"
        "        if state.chain and not state.queue and state.drafts is not None\n"
        "    ]\n"
        "    lengths = [int(state.drafts.shape[0]) for state in rows]\n"
        "    if not lengths or min(lengths) < 1:\n"
        "        return\n"
        "    limit = min(lengths)\n"
        "    if draft_policy is not None and any(state.copy_drafts for state in rows):\n"
        "        survivals = [\n"
        "            (\n"
        "                draft_policy.lookup_survival(limit, state.lookup_match)\n"
        "                if state.copy_drafts\n"
        "                else draft_policy.drafter_survival(limit)\n"
        "            )\n"
        "            for state in rows\n"
        "        ]\n"
        "        limit = max(\n"
        "            1,\n"
        "            _draft_policy.allocate_batch_verify_budget(\n"
        "                survivals, draft_policy.cost_ms, rectangular=True\n"
        "            )[0],\n"
        "        )\n"
        "    elif len(set(lengths)) < 2:\n"
        "        return\n"
        "    for state in rows:\n"
        "        _clip_block_drafts(state, limit)\n",
        "bg suffix clip_to_shared_window",
    )
    txt = _replace_once(
        txt,
        "_MISSING_CYCLE_INPUTS = \"chain cycle entered without next_main / drafts\"\n",
        _DRAFT_POLICY_HELPERS_SRC
        + "\n\n_MISSING_CYCLE_INPUTS = \"chain cycle entered without next_main / drafts\"\n",
        "bg suffix policy helpers",
    )
    txt = _replace_once(
        txt,
        "    drafter = _drafter_for(gen_batch.model)\n"
        "    if drafter is not None and verify_result is None:\n"
        "        # A shared verify was already built from the drafts it was given.\n"
        "        _clip_block_drafts(\n"
        "            state,\n"
        "            state.controller.cur if state.controller is not None else state.depth,\n"
        "        )\n",
        "    drafter = _drafter_for(gen_batch.model)\n"
        "    policy = None if follower else _draft_policy_for(gen_batch.model)\n"
        "    own_forward = verify_result is None\n"
        "    cycle_conf = None\n"
        "    if drafter is not None and own_forward and policy is not None:\n"
        "        if not state.copy_drafts:\n"
        "            cycle_conf = _stated_chances(state)\n"
        "        if cycle_conf:\n"
        "            _clip_block_drafts(state, policy.window(cycle_conf) - 1)\n"
        "        elif not state.copy_drafts:\n"
        "            cycle_conf = None\n"
        "            _clip_block_drafts(state, _drafter_window(state))\n"
        "    elif drafter is not None and own_forward:\n"
        "        _clip_block_drafts(state, _drafter_window(state))\n",
        "bg suffix run_verify_cycle_chain clip",
    )
    txt = _replace_once(
        txt,
        "    k = int(state.drafts.shape[0])\n"
        "    copy_cycle = state.copy_drafts\n"
        "    cycle_t0 = time.perf_counter()\n",
        "    k = int(state.drafts.shape[0])\n"
        "    copy_cycle = state.copy_drafts\n"
        "    cycle_match, cycle_lookup_drafted = state.lookup_match, state.lookup_drafted\n"
        "    if cycle_conf is not None:\n"
        "        cycle_conf = cycle_conf[:k]\n"
        "    cycle_t0 = time.perf_counter()\n",
        "bg suffix run_verify_cycle_chain cycle_match",
    )
    txt = _replace_once(
        txt,
        "    can_predraft = (\n"
        "        drafter is not None\n"
        "        and procs is None\n"
        "        and draft_jobs is None\n"
        "        and commit_cache is None\n"
        "        and not defer_commit\n"
        "    )\n"
        "    predrafted = False\n",
        "    can_predraft = (\n"
        "        drafter is not None\n"
        "        and procs is None\n"
        "        and draft_jobs is None\n"
        "        and commit_cache is None\n"
        "        and not defer_commit\n"
        "    )\n"
        "    if can_predraft and policy is not None and copy_cycle:\n"
        "        if k > drafter.depth or _lookup_continues(\n"
        "            gen_batch, state, policy, drafter, k\n"
        "        ):\n"
        "            can_predraft = False\n"
        "    predrafted = False\n",
        "bg suffix can_predraft",
    )
    txt = _replace_once(
        txt,
        "        state.stats.copy_accepted += m\n"
        "        state.context_copy.observe(m, k)\n",
        "        state.stats.copy_accepted += m\n"
        "        if state.context_copy is not None:\n"
        "            state.context_copy.observe(m, k)\n",
        "bg suffix copy_observe",
    )
    txt = _replace_once(
        txt,
        "        state.copy_drafts = False\n"
        "        # A block drafter's row may verify a copy of a repeated stretch of the\n"
        "        # context instead of its block; only a copy that fills the block does.\n"
        "        copied = []\n"
        "        if (\n"
        "            drafter is not None\n"
        "            and draft_jobs is None\n"
        "            and getattr(drafter, \"context_copy\", False)\n"
        "        ):\n"
        "            copied = _context_copy_drafts(\n"
        "                gen_batch, state, committed_ids, block=drafter.depth\n"
        "            )\n",
        "        state.copy_drafts = False\n"
        "        state.draft_conf = None\n"
        "        copied = []\n"
        "        if policy is not None:\n"
        "            if drafter is not None or draft_jobs is None:\n"
        "                copied = _policy_lookup(\n"
        "                    gen_batch,\n"
        "                    state,\n"
        "                    policy,\n"
        "                    committed_ids,\n"
        "                    drafter,\n"
        "                    drafted=drafter is not None and predrafted and m == m_gpu,\n"
        "                )\n"
        "        elif (\n"
        "            drafter is not None\n"
        "            and draft_jobs is None\n"
        "            and getattr(drafter, \"context_copy\", False)\n"
        "        ):\n"
        "            copied = _context_copy_drafts(\n"
        "                gen_batch, state, committed_ids, block=drafter.depth\n"
        "            )\n",
        "bg suffix draft_next lookup",
    )
    txt = _replace_once(
        txt,
        "        elif draft_jobs is None:\n"
        "            copied = _context_copy_drafts(gen_batch, state, committed_ids)\n"
        "            _chain_next_drafts(\n",
        "        elif draft_jobs is None:\n"
        "            if policy is None:\n"
        "                copied = _context_copy_drafts(gen_batch, state, committed_ids)\n"
        "            _chain_next_drafts(\n",
        "bg suffix chain fallback",
    )
    txt = _replace_once(
        txt,
        "            state.draft_accept_lps = (\n"
        "                [] if is_greedy else _copy_draft_q(copied, combined_lp.shape[-1])\n"
        "            )\n"
        "            state.copy_drafts = True\n",
        "            state.draft_accept_lps = (\n"
        "                [] if is_greedy else _copy_draft_q(copied, combined_lp.shape[-1])\n"
        "            )\n"
        "            state.draft_conf = None\n"
        "            state.copy_drafts = True\n",
        "bg suffix clear draft_conf",
    )
    txt = _replace_once(
        txt,
        "        untimed, state.copy_untimed = state.copy_untimed, False\n"
        "        if state.controller is not None and not copy_cycle:\n"
        "            was_warmup = bool(state.controller._warmup)\n"
        "            keepalive = bool(getattr(state.mtp_cache, \"fold_keepalive\", False))\n"
        "            if keepalive:\n"
        "                state.mtp_cache.fold_keepalive = False\n"
        "            state.controller.observe(\n"
        "                k,\n"
        "                m,\n"
        "                accept_ms\n"
        "                + (time.perf_counter() - finish_t0) * 1000\n"
        "                + verify_ms\n"
        "                + shared_commit_ms,\n"
        "                time_sample=not keepalive and not untimed,\n"
        "            )\n"
        "            _maybe_finish_mtp_reentry_probe(\n"
        "                gen_batch,\n"
        "                state,\n"
        "                was_warmup=was_warmup,\n"
        "            )\n",
        "        untimed, state.copy_untimed = state.copy_untimed, False\n"
        "        cycle_ms = (\n"
        "            accept_ms\n"
        "            + (time.perf_counter() - finish_t0) * 1000\n"
        "            + verify_ms\n"
        "            + shared_commit_ms\n"
        "        )\n"
        "        keepalive = False\n"
        "        if state.controller is not None and not copy_cycle:\n"
        "            was_warmup = bool(state.controller._warmup)\n"
        "            keepalive = bool(getattr(state.mtp_cache, \"fold_keepalive\", False))\n"
        "            if keepalive:\n"
        "                state.mtp_cache.fold_keepalive = False\n"
        "            state.controller.observe(\n"
        "                k,\n"
        "                m,\n"
        "                cycle_ms,\n"
        "                time_sample=not keepalive and not untimed,\n"
        "            )\n"
        "            _maybe_finish_mtp_reentry_probe(\n"
        "                gen_batch,\n"
        "                state,\n"
        "                was_warmup=was_warmup,\n"
        "            )\n"
        "        if policy is not None:\n"
        "            source = _draft_policy.Source.NONE\n"
        "            if copy_cycle:\n"
        "                source = _draft_policy.Source.LOOKUP\n"
        "            elif k:\n"
        "                source = _draft_policy.Source.DRAFTER\n"
        "            policy.observe(\n"
        "                _draft_policy.Choice(source, k, match=cycle_match),\n"
        "                m_gpu,\n"
        "                cycle_ms if own_forward and not untimed and not keepalive else None,\n"
        "                conf=cycle_conf,\n"
        "                drafted=cycle_lookup_drafted if copy_cycle else k > 0,\n"
        "            )\n",
        "bg suffix finish observe",
    )
    return txt


# ---------------------------------------------------------------------------
# PR #5505 (prefill_helper/adapters/inkling2.py, scheduler.py)
# ---------------------------------------------------------------------------

def patch_prefill_adapter(txt: str) -> str:
    txt = _replace_once(
        txt,
        "from ...patches.inkling2 import model as inkling2\n",
        "from ...patches.inkling2 import cache as windows\n"
        "from ...patches.inkling2 import model as inkling2\n",
        "adapter import",
    )
    txt = _replace_once(
        txt,
        "def _cache_state(i: int, item: Any, since: int) -> Arrays:\n"
        "    kv, conv = item[0], item[1]\n"
        "    out = {\n"
        "        f\"{i}.k\": kv.keys[..., since : kv.offset, :],\n"
        "        f\"{i}.v\": kv.values[..., since : kv.offset, :],\n"
        "    }\n"
        "    for slot in range(_CONV_SLOTS):\n"
        "        if conv[slot] is not None:\n"
        "            out[f\"{i}.c{slot}\"] = conv[slot]\n"
        "    return out\n",
        "def _cache_state(i: int, visit: tuple, rows: int | None) -> Arrays:\n"
        "    kv, conv = visit\n"
        "    if kv is None:\n"
        "        base = windows.CONV\n"
        "        if rows is None:\n"
        "            keys, values = conv[windows.KEYS], conv[windows.VALUES]\n"
        "        else:\n"
        "            keys, values = windows.newest(conv, rows)\n"
        "    else:\n"
        "        base = 0\n"
        "        start = 0 if rows is None else kv.offset - rows\n"
        "        keys = kv.keys[..., start : kv.offset, :]\n"
        "        values = kv.values[..., start : kv.offset, :]\n"
        "    out = {f\"{i}.k\": keys, f\"{i}.v\": values}\n"
        "    for slot in range(_CONV_SLOTS):\n"
        "        if conv[base + slot] is not None:\n"
        "            out[f\"{i}.c{slot}\"] = conv[base + slot]\n"
        "    return out\n",
        "adapter _cache_state",
    )
    txt = _replace_once(
        txt,
        "        before = [c[0].offset for c in cache]\n"
        "        h = lm.model.embed_norm(lm.model.embed(tokens))\n"
        "        h, entered = _run_layers(h, [], _layers(lm), cache, capture=capture)\n"
        "        boundary = {\"h\": h, **{f\"blk.{i}\": v for i, v in entered.items()}}\n"
        "        state: Arrays = {}\n"
        "        for i, c in enumerate(cache):\n"
        "            state.update(_cache_state(i, c, before[i]))\n",
        "        visits = lm.model.visit_caches(cache)\n"
        "        h = lm.model.embed_norm(lm.model.embed(tokens))\n"
        "        h, entered = _run_layers(h, [], _layers(lm), visits, capture=capture)\n"
        "        boundary = {\"h\": h, **{f\"blk.{i}\": v for i, v in entered.items()}}\n"
        "        state: Arrays = {}\n"
        "        for i, visit in enumerate(visits):\n"
        "            state.update(_cache_state(i, visit, tokens.shape[1]))\n",
        "adapter run_first",
    )
    txt = _replace_once(
        txt,
        "        h, _ = _run_layers(boundary[\"h\"], blocks, layers[k:], cache[k:], k, capture)\n",
        "        visits = lm.model.visit_caches(cache)[k:]\n"
        "        h, _ = _run_layers(boundary[\"h\"], blocks, layers[k:], visits, k, capture)\n",
        "adapter run_rest",
    )
    txt = _replace_once(
        txt,
        "    def apply_state(self, cache: list, k: int, state: Arrays) -> None:\n"
        "        for i in range(k):\n"
        "            kv, conv = cache[i][0], cache[i][1]\n"
        "            keys = state[f\"{i}.k\"]\n"
        "            if keys.shape[2]:\n"
        "                kv.update_and_fetch(keys, state[f\"{i}.v\"])\n"
        "            for slot in range(_CONV_SLOTS):\n"
        "                value = state.get(f\"{i}.c{slot}\")\n"
        "                if value is not None:\n"
        "                    conv[slot] = value\n\n"
        "    def prefix_state(self, cache: list, k: int) -> Arrays:\n"
        "        state: Arrays = {}\n"
        "        for i in range(k):\n"
        "            state.update(_cache_state(i, cache[i], 0))\n"
        "        mx.eval(list(state.values()))\n"
        "        return state\n\n"
        "    def load_prefix(self, cache: list, state: Arrays) -> None:\n"
        "        self.apply_state(cache, len(cache), state)\n",
        "    def apply_state(self, cache: list, k: int, state: Arrays) -> None:\n"
        "        for i, (kv, conv) in enumerate(windows.visits(cache)[:k]):\n"
        "            keys = state.get(f\"{i}.k\")\n"
        "            if keys is not None and keys.shape[2]:\n"
        "                if kv is None:\n"
        "                    windows.append(conv, keys, state[f\"{i}.v\"])\n"
        "                else:\n"
        "                    kv.update_and_fetch(keys, state[f\"{i}.v\"])\n"
        "            base = windows.CONV if kv is None else 0\n"
        "            for slot in range(_CONV_SLOTS):\n"
        "                value = state.get(f\"{i}.c{slot}\")\n"
        "                if value is not None:\n"
        "                    conv[base + slot] = value\n\n"
        "    def prefix_state(self, cache: list, k: int) -> Arrays:\n"
        "        state: Arrays = {}\n"
        "        for i, visit in enumerate(windows.visits(cache)[:k]):\n"
        "            held = _cache_state(i, visit, None)\n"
        "            state.update({n: v for n, v in held.items() if v is not None})\n"
        "        mx.eval(list(state.values()))\n"
        "        return state\n\n"
        "    def load_prefix(self, cache: list, state: Arrays) -> None:\n"
        "        self.apply_state(cache, len(windows.visits(cache)), state)\n",
        "adapter state methods",
    )
    return txt


def patch_scheduler(txt: str, features: set[str]) -> str:
    if "bounded_cache" in features:
        txt = _replace_once(
            txt,
            "        if self._model_uses_mla():\n"
            "            return False\n"
            "        if (\n"
            "            getattr(self, \"_turboquant_kv_scheme\", \"turboquant\")\n",
            "        if self._model_uses_mla():\n"
            "            return False\n"
            "        if getattr(getattr(self, \"model\", None), \"dense_kv_attention\", False) is True:\n"
            "            return False\n"
            "        if (\n"
            "            getattr(self, \"_turboquant_kv_scheme\", \"turboquant\")\n",
            "scheduler dense_kv_attention",
        )
    if "prompt_tail_cache" in features:
        txt = _replace_once(
            txt,
            "        hot_cache_write_back: bool = True,\n"
            "        system_store: tuple[list[int], list[Any], Any, Any] | None = None,\n",
            "        hot_cache_write_back: bool = True,\n"
            "        shared_stores: list[tuple[list[int], list[Any], Any, Any]] | None = None,\n",
            "scheduler worker sig",
        )
        txt = _replace_once(
            txt,
            "                if system_store is not None:\n"
            "                    # Before the release below, so the shared blocks stay pinned.\n"
            "                    self._store_system_prefix(\n"
            "                        request_id,\n"
            "                        system_store,\n",
            "                for shared_store in shared_stores or ():\n"
            "                    # Before the release below, so the shared blocks stay pinned.\n"
            "                    self._store_system_prefix(\n"
            "                        request_id,\n"
            "                        shared_store,\n",
            "scheduler worker loop",
        )
        txt = _replace_once(
            txt,
            "    def _prepare_system_prefix_store(\n"
            "        self, request: \"Request\", full_cache: list[Any]\n"
            "    ) -> tuple[list[int], list[Any], Any, Any] | None:\n"
            '        """Build the shared-tail payload (system prefix or branch point)."""\n'
            "        snapshots = self._boundary_cache_snapshots.get(request.request_id) or {}\n"
            "        ends = (\n"
            "            getattr(request, \"system_prefix_end\", 0) or 0,\n"
            "            getattr(request, \"junction_end\", 0) or 0,\n"
            "        )\n"
            "        system_end = next((end for end in ends if end > 0 and end in snapshots), 0)\n"
            "        if system_end <= 0:\n"
            "            return None\n"
            "        try:\n"
            "            override = self._get_boundary_store_override(\n"
            "                request.request_id,\n"
            "                list(request.prompt_token_ids[:system_end]),\n"
            "                system_tail=True,\n"
            "            )\n"
            "        except Exception as e:\n"
            "            logger.debug(\n"
            "                \"Shared tail snapshot unavailable for %s: %s\", request.request_id, e\n"
            "            )\n"
            "            return None\n"
            "        if override is None:\n"
            "            return None\n"
            "        tokens, boundary_cache, model_cache_config, provider = override\n"
            "        if getattr(provider, \"tail_terminal_token_count\", None) != system_end:\n"
            "            return None\n"
            "        return (\n"
            "            tokens,\n"
            "            self._merge_boundary_with_full_cache(boundary_cache, full_cache),\n"
            "            model_cache_config or getattr(request, \"_model_cache_config\", None),\n"
            "            provider,\n"
            "        )\n",
            "    def _superseded_tail_end(\n"
            "        self, request: \"Request | None\", snapshots: dict[int, Any]\n"
            "    ) -> int:\n"
            "        tail = getattr(request, \"prompt_tail_end\", 0) or 0\n"
            "        block_size = self.config.paged_cache_block_size\n"
            "        if tail <= 0 or block_size <= 0 or tail not in snapshots:\n"
            "            return 0\n"
            "        marker = getattr(request, \"generation_prompt_start\", 0) or 0\n"
            "        if marker > 0 and not getattr(request, \"generation_prompt_persists\", False):\n"
            "            return 0\n"
            "        if any(tc > tail and tc % block_size == 0 for tc in snapshots):\n"
            "            return tail\n"
            "        return 0\n\n"
            "    def _prepare_shared_tail_stores(\n"
            "        self, request: \"Request\", full_cache: list[Any]\n"
            "    ) -> list[tuple[list[int], list[Any], Any, Any]]:\n"
            "        snapshots = self._boundary_cache_snapshots.get(request.request_id) or {}\n"
            "        ends = (\n"
            "            getattr(request, \"system_prefix_end\", 0) or 0,\n"
            "            getattr(request, \"junction_end\", 0) or 0,\n"
            "        )\n"
            "        shared_ends = [\n"
            "            next((end for end in ends if end > 0 and end in snapshots), 0),\n"
            "            self._superseded_tail_end(request, snapshots),\n"
            "        ]\n"
            "        stores = []\n"
            "        for shared_end in sorted({end for end in shared_ends if end > 0}):\n"
            "            try:\n"
            "                override = self._get_boundary_store_override(\n"
            "                    request.request_id,\n"
            "                    list(request.prompt_token_ids[:shared_end]),\n"
            "                    system_tail=True,\n"
            "                )\n"
            "            except Exception as e:\n"
            "                logger.debug(\n"
            "                    \"Shared tail snapshot unavailable for %s: %s\",\n"
            "                    request.request_id,\n"
            "                    e,\n"
            "                )\n"
            "                continue\n"
            "            if override is None:\n"
            "                continue\n"
            "            tokens, boundary_cache, model_cache_config, provider = override\n"
            "            if getattr(provider, \"tail_terminal_token_count\", None) != shared_end:\n"
            "                continue\n"
            "            stores.append(\n"
            "                (\n"
            "                    tokens,\n"
            "                    self._merge_boundary_with_full_cache(boundary_cache, full_cache),\n"
            "                    model_cache_config\n"
            "                    or getattr(request, \"_model_cache_config\", None),\n"
            "                    provider,\n"
            "                )\n"
            "            )\n"
            "        return stores\n",
            "scheduler prepare_shared_tail_stores",
        )
        txt = _replace_once(
            txt,
            "                parser_session = self._output_parser_factory.create_session(\n"
            "                    self.tokenizer\n"
            "                )\n"
            "            if request is not None and getattr(request, \"needs_think_prefix\", False):\n",
            "                parser_session = self._output_parser_factory.create_session(\n"
            "                    self.tokenizer\n"
            "                )\n"
            "            notify_tail = getattr(parser_session, \"notify_prompt_tail\", None)\n"
            "            if (\n"
            "                callable(notify_tail)\n"
            "                and request is not None\n"
            "                and request.prompt_token_ids\n"
            "            ):\n"
            "                notify_tail(request.prompt_token_ids)\n"
            "            if request is not None and getattr(request, \"needs_think_prefix\", False):\n",
            "scheduler notify_prompt_tail",
        )
        txt = _replace_once(
            txt,
            "        self._on_prefill_boundary_snapshot(\n"
            "            request.request_id, snapshot_cache, total_tokens, source=\"prefill_tail\"\n"
            "        )\n"
            "        _mtp_priming.capture_tail_boundary(self.model, request.request_id, total_tokens)\n",
            "        self._on_prefill_boundary_snapshot(\n"
            "            request.request_id, snapshot_cache, total_tokens, source=\"prefill_tail\"\n"
            "        )\n"
            "        if total_tokens not in (\n"
            "            getattr(request, \"system_prefix_end\", 0) or 0,\n"
            "            getattr(request, \"junction_end\", 0) or 0,\n"
            "        ):\n"
            "            request.prompt_tail_end = total_tokens\n"
            "        _mtp_priming.capture_tail_boundary(self.model, request.request_id, total_tokens)\n",
            "scheduler emit_prefill_tail_snapshot",
        )
        txt = _replace_once(
            txt,
            "        shared_ends = {\n"
            "            getattr(request, \"system_prefix_end\", 0) or 0,\n"
            "            getattr(request, \"junction_end\", 0) or 0,\n"
            "        }\n",
            "        shared_ends = {\n"
            "            getattr(request, \"system_prefix_end\", 0) or 0,\n"
            "            getattr(request, \"junction_end\", 0) or 0,\n"
            "            self._superseded_tail_end(request, snapshots),\n"
            "        }\n",
            "scheduler shared_ends",
        )
        txt = _replace_once(
            txt,
            "                                    system_store = self._prepare_system_prefix_store(\n"
            "                                        request, cache_to_store\n"
            "                                    )\n",
            "                                    shared_stores = self._prepare_shared_tail_stores(\n"
            "                                        request, cache_to_store\n"
            "                                    )\n",
            "scheduler cleanup prepare",
        )
        txt = _replace_once(
            txt,
            "                                    if system_store is not None:\n"
            "                                        pre_eval_arrays.extend(\n"
            "                                            self._collect_arrays_from_extracted_cache(\n"
            "                                                system_store[1]\n"
            "                                            )\n"
            "                                        )\n",
            "                                    for shared_store in shared_stores:\n"
            "                                        pre_eval_arrays.extend(\n"
            "                                            self._collect_arrays_from_extracted_cache(\n"
            "                                                shared_store[1]\n"
            "                                            )\n"
            "                                        )\n",
            "scheduler cleanup collect",
        )
        txt = _replace_once(
            txt,
            "                                        hot_cache_write_back,\n"
            "                                        system_store,\n"
            "                                    )\n"
            "                                except BaseException:\n"
            "                                    if gate is not None:\n"
            "                                        gate.note_done()\n"
            "                                    raise\n"
            "                                self._inflight_store_futures[request_id] = store_future\n"
            "                                self._inflight_store_info[request_id] = (\n"
            "                                    _InflightStoreInfo(\n"
            "                                        tokens=list(token_sequence_to_store),\n"
            "                                        extra_keys=request.vlm_extra_keys_for_cache,\n"
            "                                        extra_key_token_start=(\n"
            "                                            request.vlm_extra_key_token_start_for_cache\n"
            "                                        ),\n"
            "                                        extra_key_ranges=(\n"
            "                                            request.vlm_extra_key_ranges_for_cache\n"
            "                                        ),\n"
            "                                        shared_prefix_len=(\n"
            "                                            len(system_store[0]) if system_store else 0\n"
            "                                        ),\n"
            "                                    )\n"
            "                                )\n",
            "                                        hot_cache_write_back,\n"
            "                                        shared_stores,\n"
            "                                    )\n"
            "                                except BaseException:\n"
            "                                    if gate is not None:\n"
            "                                        gate.note_done()\n"
            "                                    raise\n"
            "                                self._inflight_store_futures[request_id] = store_future\n"
            "                                self._inflight_store_info[request_id] = (\n"
            "                                    _InflightStoreInfo(\n"
            "                                        tokens=list(token_sequence_to_store),\n"
            "                                        extra_keys=request.vlm_extra_keys_for_cache,\n"
            "                                        extra_key_token_start=(\n"
            "                                            request.vlm_extra_key_token_start_for_cache\n"
            "                                        ),\n"
            "                                        extra_key_ranges=(\n"
            "                                            request.vlm_extra_key_ranges_for_cache\n"
            "                                        ),\n"
            "                                        shared_prefix_len=max(\n"
            "                                            (len(st[0]) for st in shared_stores),\n"
            "                                            default=0,\n"
            "                                        ),\n"
            "                                    )\n"
            "                                )\n",
            "scheduler cleanup submit",
        )
        txt = _replace_once(
            txt,
            "                                self._async_store_cache_worker(\n"
            "                                    request_id,\n"
            "                                    token_sequence_to_store,\n"
            "                                    cache_to_store,\n"
            "                                    model_cache_config,\n"
            "                                    intermediate_snapshots,\n"
            "                                    request.vlm_extra_keys_for_cache,\n"
            "                                    request.vlm_extra_key_token_start_for_cache,\n"
            "                                    request.vlm_extra_key_ranges_for_cache,\n"
            "                                    hot_cache_write_back,\n"
            "                                    system_store,\n"
            "                                )\n",
            "                                self._async_store_cache_worker(\n"
            "                                    request_id,\n"
            "                                    token_sequence_to_store,\n"
            "                                    cache_to_store,\n"
            "                                    model_cache_config,\n"
            "                                    intermediate_snapshots,\n"
            "                                    request.vlm_extra_keys_for_cache,\n"
            "                                    request.vlm_extra_key_token_start_for_cache,\n"
            "                                    request.vlm_extra_key_ranges_for_cache,\n"
            "                                    hot_cache_write_back,\n"
            "                                    shared_stores,\n"
            "                                )\n",
            "scheduler cleanup sync fallback",
        )
    return txt


# ---------------------------------------------------------------------------
# PR #5506 (engine/vlm.py, output_parser.py, request.py, processing.py, server.py)
# ---------------------------------------------------------------------------

def patch_vlm_engine(txt: str) -> str:
    txt = _replace_once(
        txt,
        "def _apply_template_thinking_mode(\n"
        "    model_type: str | None,\n"
        "    template_kwargs: dict[str, Any],\n"
        ") -> None:\n"
        '    """Translate oMLX thinking switches into model-specific template kwargs."""\n'
        "    _apply_minimax_m3_thinking_mode(model_type, template_kwargs)\n"
        "    _apply_hy_v4_reasoning_mode(model_type, template_kwargs)\n",
        "def _apply_inkling2_reasoning_mode(\n"
        "    model_type: str | None,\n"
        "    template_kwargs: dict[str, Any],\n"
        ") -> None:\n"
        '    """Map oMLX thinking switches to Inkling 2 reasoning_effort."""\n'
        "    if model_type not in INKLING2_MODEL_TYPES:\n"
        "        return\n"
        "    effort = template_kwargs.get(\"reasoning_effort\")\n"
        "    off = isinstance(effort, str) and effort.strip().lower() in (\"off\", \"no_think\")\n"
        "    if off or (effort is None and template_kwargs.get(\"enable_thinking\") is False):\n"
        "        template_kwargs[\"reasoning_effort\"] = \"none\"\n\n\n"
        "def _apply_template_thinking_mode(\n"
        "    model_type: str | None,\n"
        "    template_kwargs: dict[str, Any],\n"
        ") -> None:\n"
        '    """Translate oMLX thinking switches into model-specific template kwargs."""\n'
        "    _apply_minimax_m3_thinking_mode(model_type, template_kwargs)\n"
        "    _apply_hy_v4_reasoning_mode(model_type, template_kwargs)\n"
        "    _apply_inkling2_reasoning_mode(model_type, template_kwargs)\n",
        "vlm engine thinking mode",
    )
    return txt


def patch_output_parser(txt: str) -> str:
    txt = _replace_once(
        txt,
        "from collections.abc import Callable\n",
        "from collections.abc import Callable, Sequence\n",
        "output_parser import",
    )
    txt = _replace_once(
        txt,
        "        elif marker == _INKLING_END_SAMPLING:\n"
        "            self.stopped = True\n"
        "            self._channel = None\n"
        "        return stream, visible\n\n"
        "    def feed(self, text: str) -> tuple[str, str]:\n",
        "        elif marker == _INKLING_END_SAMPLING:\n"
        "            self.stopped = True\n"
        "            self._channel = None\n"
        "        return stream, visible\n\n"
        "    def open_block(self, marker: str) -> None:\n"
        '        """Enter the block ``marker`` opens without emitting anything for it."""\n'
        "        self._handle_marker(marker)\n"
        "        self._head = \"\"\n\n"
        "    def feed(self, text: str) -> tuple[str, str]:\n",
        "output_parser open_block",
    )
    txt = _replace_once(
        txt,
        "    def _decode_token(self, token_id: int) -> str:\n"
        "        return _decode_output_token(self._tokenizer, self._detokenizer, token_id)\n\n"
        "    def process_token(self, token_id: int) -> OutputParserTokenResult:\n",
        "    def _decode_token(self, token_id: int) -> str:\n"
        "        return _decode_output_token(self._tokenizer, self._detokenizer, token_id)\n\n"
        "    def notify_prefilled_thought(self) -> None:\n"
        "        self._splitter.open_block(_INKLING_CONTENT_THINKING)\n\n"
        "    def notify_prompt_tail(self, token_ids: Sequence[int]) -> None:\n"
        "        convert = getattr(self._tokenizer, \"convert_tokens_to_ids\", None)\n"
        "        if not callable(convert):\n"
        "            return\n"
        "        markers = {}\n"
        "        for marker in _INKLING_MARKERS:\n"
        "            token_id = convert(marker)\n"
        "            if isinstance(token_id, int) and token_id >= 0:\n"
        "                markers[token_id] = marker\n"
        "        tail: list[int] = []\n"
        "        for token_id in reversed(token_ids):\n"
        "            marker = markers.get(int(token_id))\n"
        "            if marker is None:\n"
        "                tail.append(int(token_id))\n"
        "                continue\n"
        "            if marker in (\n"
        "                _INKLING_CONTENT_TEXT,\n"
        "                _INKLING_CONTENT_XML,\n"
        "                _INKLING_CONTENT_TOOL_JSON,\n"
        "            ):\n"
        "                self._splitter.open_block(marker)\n"
        "                if marker == _INKLING_CONTENT_TOOL_JSON:\n"
        "                    self._raw_text = marker + self._tokenizer.decode(tail[::-1])\n"
        "            return\n\n"
        "    def process_token(self, token_id: int) -> OutputParserTokenResult:\n",
        "output_parser notify_prompt_tail",
    )
    return txt


def patch_request(txt: str) -> str:
    return _replace_once(
        txt,
        "    junction_end: int = 0  # Where this prompt branches off a recent stored sequence\n",
        "    junction_end: int = 0  # Where this prompt branches off a recent stored sequence\n"
        "    prompt_tail_end: int = 0  # Where the prefill's own tail snapshot was taken\n",
        "request prompt_tail_end",
    )


def patch_processing(txt: str) -> str:
    txt = _replace_once(
        txt,
        "from transformers import PreTrainedTokenizerFast\n",
        "from transformers import PreTrainedTokenizerFast\n\n"
        "from .opener import install_generation_opener\n",
        "processing import",
    )
    txt = _replace_once(
        txt,
        "        self.tokenizer = tokenizer\n"
        "        self.image_processor = image_processor or Inkling2ImageProcessor()\n",
        "        self.tokenizer = tokenizer\n"
        "        install_generation_opener(tokenizer)\n"
        "        self.image_processor = image_processor or Inkling2ImageProcessor()\n",
        "processing init",
    )
    return txt


def patch_server(txt: str) -> str:
    txt = _replace_once(
        txt,
        "from .models.decision import DecisionContextLengthError, DecisionRequestError\n",
        "from .models.decision import DecisionContextLengthError, DecisionRequestError\n"
        "from .patches.inkling2 import INKLING2_MODEL_TYPES\n",
        "server import",
    )
    txt = _replace_once(
        txt,
        "            else _compile_grammar_for_request(\n"
        "                engine,\n"
        "                structured_outputs=structured_outputs,\n"
        "                response_format=response_format,\n"
        "                chat_template_kwargs=merged_ct_kwargs or None,\n"
        "                reasoning_parser=reasoning_parser,\n"
        "            )\n"
        "        )\n",
        "            else _compile_grammar_for_request(\n"
        "                engine,\n"
        "                structured_outputs=structured_outputs,\n"
        "                response_format=response_format,\n"
        "                chat_template_kwargs=merged_ct_kwargs or None,\n"
        "                reasoning_parser=reasoning_parser,\n"
        "            )\n"
        "        )\n"
        "        merged_ct_kwargs = _grammar_opener_kwargs(\n"
        "            engine, compiled_grammar, merged_ct_kwargs\n"
        "        )\n",
        "server chat completions",
    )
    txt = _replace_once(
        txt,
        "    return reasoning_parser\n\n\n"
        "def _compile_with_structural_tag(",
        "    return reasoning_parser\n\n\n"
        "def _grammar_opener_kwargs(\n"
        "    engine, compiled_grammar, chat_template_kwargs: dict | None\n"
        ") -> dict | None:\n"
        "    if (\n"
        "        compiled_grammar is None\n"
        "        or not chat_template_kwargs\n"
        "        or chat_template_kwargs.get(\"generation_opener\") != \"content\"\n"
        "        or getattr(engine, \"model_type\", None) not in INKLING2_MODEL_TYPES\n"
        "        or getattr(compiled_grammar, \"_omlx_has_thinking_phase\", False) is True\n"
        "    ):\n"
        "        return chat_template_kwargs\n"
        "    return {**chat_template_kwargs, \"generation_opener_answer\": True}\n\n\n"
        "def _compile_with_structural_tag(",
        "server helper",
    )
    txt = _replace_once(
        txt,
        "                    compiled_grammar = _compile_grammar_for_request(\n"
        "                        engine,\n"
        "                        response_format=rf,\n"
        "                        chat_template_kwargs=merged_ct_kwargs or None,\n"
        "                        reasoning_parser=reasoning_parser,\n"
        "                    )\n",
        "                    compiled_grammar = _compile_grammar_for_request(\n"
        "                        engine,\n"
        "                        response_format=rf,\n"
        "                        chat_template_kwargs=merged_ct_kwargs or None,\n"
        "                        reasoning_parser=reasoning_parser,\n"
        "                    )\n"
        "                    merged_ct_kwargs = _grammar_opener_kwargs(\n"
        "                        engine, compiled_grammar, merged_ct_kwargs\n"
        "                    )\n",
        "server responses",
    )
    return txt
