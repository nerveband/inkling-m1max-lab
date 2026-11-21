# SPDX-License-Identifier: Apache-2.0
"""M1 Max (applegpu_g13) Metal kernels for Inkling 2 Mini speculative verify."""

from __future__ import annotations

from functools import cache

import mlx.core as mx

# Pass 1: Split-K GQA-packed multi-row banded+sink attention.
# Packs all GQA * L query rows for one KV head into NSG 8-row simdgroup tiles.
# Reads K and V directly from cache in place (no kz/vz copy, no [B,H,L,S] mask).
# Sink logit is folded into split 0's initial online-softmax state (m = sink*log2e, l = 1).
_ATTN_P1_COMMON = r"""
    constexpr int BK = 16;
    constexpr int D = 128;
    constexpr float LOG2E = 1.4426950408889634f;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const int hk = int(threadgroup_position_in_grid.x);
    const int split_idx = int(threadgroup_position_in_grid.y);
    const int S = dims[0];
    const int lp = dims[1];
    const int chunk = dims[2];
    const int k_begin = dims[3];

    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    const int row = int(sg) * 8 + int(fm);
    const bool row_ok = row < NROWS;
    const int qi = row_ok ? (row / GQA) : 0;
    const int h = hk * GQA + (row_ok ? (row % GQA) : 0);
    const int a_row = S - L + qi;
    const float qscale = scale[0] * LOG2E;

    constexpr int LDK = BK + 8;
    constexpr int LDV = D + 8;
    threadgroup half Ks[D * LDK];
    threadgroup half Vs[BK * LDV];

    simdgroup_half8x8 Qf[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        float2 qv = float2(0.0f);
        if (row_ok) {
            const device T* qp = q + h * q_strides[1] + qi * q_strides[2] + (db * 8 + fn) * q_strides[3];
            qv = float2(float(qp[0]), float(qp[q_strides[3]]));
        }
        Qf[db].thread_elements()[0] = half(qv.x);
        Qf[db].thread_elements()[1] = half(qv.y);
    }
    simdgroup_float8x8 Of[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        Of[db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    float m = (split_idx == 0 && row_ok) ? (float(sink[qi * HQ + h]) * LOG2E) : -1e30f;
    float l = (split_idx == 0) ? 1.0f : 0.0f;
    const device T* relp = rel + (size_t(qi) * HQ + h) * R;

    const int split_start = k_begin + split_idx * chunk;
    const int split_end = min(S, split_start + chunk);
    const device T* kb_ptr = k + hk * k_strides[1];
    const device T* vb_ptr = v + hk * v_strides[1];
    const bool vec_kv = (k_strides[3] == 1 && v_strides[3] == 1);

    for (int k0 = split_start; k0 < split_end; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (vec_kv) {
            for (int c = int(tid); c < BK * D / 8; c += NT) {
                const int r = c / (D / 8);
                const int col = (c % (D / 8)) * 8;
                const int key = k0 + r;
                uint4 kw = uint4(0);
                uint4 vw = uint4(0);
                if (key < split_end) {
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
                if (key < split_end) {
                    kv = float(kb_ptr[key * k_strides[2] + col * k_strides[3]]);
                    vv = float(vb_ptr[key * v_strides[2] + col * v_strides[3]]);
                }
                Ks[col * LDK + r] = half(kv);
                Vs[r * LDV + col] = half(vv);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

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
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const int key = k0 + kb * 8 + fn + e;
                const int d = a_row - key;
                const bool vis = row_ok && key < split_end && d >= 0 && key >= lp
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
        rmax = max(rmax, simd_shuffle_xor(rmax, 1));
        rmax = max(rmax, simd_shuffle_xor(rmax, 8));
        const float m_new = max(m, rmax);
        const float factor = (l > 0.0f) ? fast::exp2(m - m_new) : 0.0f;
        float rsum = 0.0f;
        simdgroup_half8x8 Pf[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const float sv = Sf[kb].thread_elements()[e];
                const float p = (sv > -1e29f) ? fast::exp2(sv - m_new) : 0.0f;
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
"""

_ATTN_P1_SINGLE_TAIL = r"""
    if (row_ok) {
        const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
        device T* op = out + (size_t(h) * L + qi) * D;
        for (int db = 0; db < D / 8; ++db) {
            op[db * 8 + fn] = T(Of[db].thread_elements()[0] * inv);
            op[db * 8 + fn + 1] = T(Of[db].thread_elements()[1] * inv);
        }
    }
"""

_ATTN_P1_SPLIT_TAIL = r"""
    if (row_ok) {
        const size_t base = ((size_t(hk) * NSPLITS + split_idx) * NROWS + row);
        device float* op = o_part + base * D;
        for (int db = 0; db < D / 8; ++db) {
            op[db * 8 + fn] = Of[db].thread_elements()[0];
            op[db * 8 + fn + 1] = Of[db].thread_elements()[1];
        }
        if (fn == 0) {
            m_part[base] = m;
            l_part[base] = l;
        }
    }
"""

# Pass 2: Reduce NSPLITS online-softmax partials per (hk, row).
# One threadgroup (32 threads = 1 simdgroup) per (hk, row); threads stride D=128 (4 floats each).
_ATTN_P2_SOURCE = r"""
    const int idx = int(threadgroup_position_in_grid.x);
    const uint lane = thread_index_in_simdgroup;
    const int hk = idx / NROWS;
    const int row = idx % NROWS;
    const int qi = row / GQA;
    const int h = hk * GQA + (row % GQA);

    float mmax = -1e30f;
    for (int s = int(lane); s < NSPLITS; s += 32) {
        const size_t base = (size_t(hk) * NSPLITS + s) * NROWS + row;
        if (l_part[base] > 0.0f) {
            mmax = max(mmax, m_part[base]);
        }
    }
    mmax = simd_max(mmax);

    float lsum = 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    const int c0 = int(lane);
    const int c1 = c0 + 32;
    const int c2 = c0 + 64;
    const int c3 = c0 + 96;
    for (int s = 0; s < NSPLITS; ++s) {
        const size_t base = (size_t(hk) * NSPLITS + s) * NROWS + row;
        const float ls = l_part[base];
        if (ls > 0.0f) {
            const float w = fast::exp2(m_part[base] - mmax);
            lsum += w * ls;
            const device float* op = o_part + base * 128;
            o0 += w * op[c0];
            o1 += w * op[c1];
            o2 += w * op[c2];
            o3 += w * op[c3];
        }
    }
    const float inv = (lsum > 0.0f) ? (1.0f / lsum) : 0.0f;
    device T* outp = out + (size_t(h) * L + qi) * 128;
    outp[c0] = T(o0 * inv);
    outp[c1] = T(o1 * inv);
    outp[c2] = T(o2 * inv);
    outp[c3] = T(o3 * inv);
"""


@cache
def _attn_p1_single_kernel():
    return mx.fast.metal_kernel(
        name="omlx_inkling2_verify_attn_p1_single",
        input_names=["q", "k", "v", "rel", "sink", "dims", "scale"],
        output_names=["out"],
        source=_ATTN_P1_COMMON + _ATTN_P1_SINGLE_TAIL,
        ensure_row_contiguous=False,
    )


@cache
def _attn_p1_split_kernel():
    return mx.fast.metal_kernel(
        name="omlx_inkling2_verify_attn_p1_split",
        input_names=["q", "k", "v", "rel", "sink", "dims", "scale"],
        output_names=["o_part", "m_part", "l_part"],
        source=_ATTN_P1_COMMON + _ATTN_P1_SPLIT_TAIL,
        ensure_row_contiguous=False,
    )


@cache
def _attn_p2_kernel():
    return mx.fast.metal_kernel(
        name="omlx_inkling2_verify_attn_p2",
        input_names=["o_part", "m_part", "l_part"],
        output_names=["out"],
        source=_ATTN_P2_SOURCE,
    )


_SCALE_CACHE: dict[float, mx.array] = {}


def _scale_arr(scale: float) -> mx.array:
    arr = _SCALE_CACHE.get(scale)
    if arr is None:
        arr = mx.array([scale], dtype=mx.float32)
        _SCALE_CACHE[scale] = arr
    return arr


def _choose_chunk(total_keys: int, hkv: int = 8) -> int:
    # M1 Max (24 cores) peaks with >= 64 threadgroups across the 8 KV heads.
    if total_keys <= 2048:
        return 64
    if total_keys <= 8192:
        return 128
    # Keep nsplits <= 64 at 32K-64K so split-K buffers stay small while maintaining <=512 TGs.
    return max(256, (((total_keys + 63) // 64 + 15) // 16) * 16)


def verify_banded_sink_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    rel: mx.array,
    sink: mx.array,
    *,
    scale: float,
    window: int = 0,
    left_padding=None,
    chunk_override: int | None = None,
) -> mx.array:
    """Fast Split-K GQA-packed banded+sink attention for B=1, 1 < L <= 16, D=128."""
    b, hq, length, d = q.shape
    hkv, s = k.shape[1], k.shape[2]
    gqa = hq // hkv
    nrows = gqa * length
    nsg = (nrows + 7) // 8
    nt = nsg * 32

    lp = int(left_padding[0].item()) if left_padding is not None else 0
    lo = max(0, s - length - window + 1) if window else 0
    lo = max(lo, lp)
    k_begin = (lo // 16) * 16
    total_keys = max(1, s - k_begin)
    chunk = chunk_override if chunk_override else _choose_chunk(total_keys, hkv)
    nsplits = max(1, (total_keys + chunk - 1) // chunk)

    dims = mx.array([s, lp, chunk, k_begin], dtype=mx.int32)
    rel_c = mx.contiguous(rel)
    sink_c = mx.contiguous(sink)
    scale_arr = _scale_arr(float(scale))

    tpl = [
        ("T", q.dtype),
        ("HQ", hq),
        ("GQA", gqa),
        ("L", length),
        ("NROWS", nrows),
        ("NT", nt),
        ("R", int(rel.shape[-1])),
        ("WINDOW", int(window)),
    ]

    if nsplits == 1:
        return _attn_p1_single_kernel()(
            inputs=[q, k, v, rel_c, sink_c, dims, scale_arr],
            template=tpl,
            grid=(hkv * nt, 1, 1),
            threadgroup=(nt, 1, 1),
            output_shapes=[(b, hq, length, d)],
            output_dtypes=[q.dtype],
        )[0]

    o_part, m_part, l_part = _attn_p1_split_kernel()(
        inputs=[q, k, v, rel_c, sink_c, dims, scale_arr],
        template=tpl + [("NSPLITS", nsplits)],
        grid=(hkv * nt, nsplits, 1),
        threadgroup=(nt, 1, 1),
        output_shapes=[
            (hkv, nsplits, nrows, d),
            (hkv, nsplits, nrows),
            (hkv, nsplits, nrows),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    return _attn_p2_kernel()(
        inputs=[o_part, m_part, l_part],
        template=[
            ("T", q.dtype),
            ("GQA", gqa),
            ("L", length),
            ("NROWS", nrows),
            ("NSPLITS", nsplits),
        ],
        grid=(hkv * nrows * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(b, hq, length, d)],
        output_dtypes=[q.dtype],
    )[0]


# ---------------------------------------------------------------------------
# M1 Grouped-Expert MoE Verify Kernels (1 <= M <= 8, top_k=8, E=128, gs=64)
# ---------------------------------------------------------------------------

_GROUP_EXPERTS_SOURCE = r"""
    const uint e = thread_position_in_threadgroup.x;
    threadgroup uint tg_inds[64];
    if (e < N_PAIRS) {
        tg_inds[e] = inds[e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint cnt = 0;
    uint toks[8];
    uint pairs[8];
    for (uint p = 0; p < N_PAIRS; ++p) {
        if (tg_inds[p] == e) {
            if (cnt < 8) {
                toks[cnt] = p / TOPK;
                pairs[cnt] = p;
            }
            cnt += 1;
        }
    }

    const uint sg = e / 32;
    const uint lane = e % 32;
    const uint active = (cnt > 0) ? 1u : 0u;
    const uint sg_prefix = simd_prefix_exclusive_sum(active);
    const uint sg_total = simd_sum(active);

    threadgroup uint sg_totals[4];
    if (lane == 0) {
        sg_totals[sg] = sg_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint base = 0;
    for (uint s = 0; s < sg; ++s) {
        base += sg_totals[s];
    }
    const uint u = base + sg_prefix;

    if (active) {
        u_exp[u] = e;
        u_cnt[u] = cnt;
        for (uint j = 0; j < 8; ++j) {
            u_toks[u * 8 + j] = (j < cnt) ? toks[j] : 0u;
            u_pairs[u * 8 + j] = (j < cnt) ? pairs[j] : 0u;
        }
    }
    if (e == 127) {
        u_num[0] = u + active;
    }
"""

_MOE_MMA_HEADER = r"""
template <typename T>
inline float moe_x_at(uint w, int e) {
  if (metal::is_same_v<T, bfloat>) {
    return as_type<float>((e & 1) ? (w & 0xFFFF0000u) : (w << 16));
  }
  return float(as_type<half2>(w)[e & 1]);
}
"""

# Fused gate+up + softcapped_swiglu for grouped experts via simdgroup_matrix MMA (RT=1).
_MOE_W13_SWIGLU_SOURCE = r"""
    const uint u = threadgroup_position_in_grid.y;
    if (u >= u_num[0]) {
        return;
    }
    const uint cnt = u_cnt[u];
    const size_t expert = size_t(u_exp[u]);
    const int lane = int(thread_index_in_simdgroup);
    const int c = int(simdgroup_index_in_threadgroup);
    const int fm = ((lane >> 4) << 2) + ((lane >> 1) & 3);
    const int fn = (((lane >> 3) & 1) << 2) + ((lane & 1) << 1);
    const int n0 = int(threadgroup_position_in_grid.x) * (8 * NT);

    constexpr int G = K / 64;
    constexpr int WPR = K * BITS / 32;
    constexpr size_t EXP_WORDS = size_t(N) * WPR;
    constexpr size_t EXP_GROUPS = size_t(N) * G;

    int nrow[NT];
    for (int t = 0; t < NT; ++t) {
        nrow[t] = min(n0 + 8 * t + fm, N - 1);
    }

    uint toks[8];
    uint pairs[8];
    for (uint t = 0; t < cnt; ++t) {
        toks[t] = u_toks[u * 8 + t];
        pairs[t] = u_pairs[u * 8 + t];
    }
    const uint t0 = toks[min(uint(fn), cnt - 1)];
    const uint t1 = toks[min(uint(fn + 1), cnt - 1)];
    const device T* x0 = X + size_t(t0) * K;
    const device T* x1 = X + size_t(t1) * K;

    const device uint* w_exp = (const device uint*)Wq + expert * EXP_WORDS;
    const device T* s_exp = SC + expert * EXP_GROUPS;
    const device T* b_exp = BI + expert * EXP_GROUPS;

    float acc[NT][2];
    for (int t = 0; t < NT; ++t) {
        acc[t][0] = 0.0f;
        acc[t][1] = 0.0f;
    }

    for (int g = c; g < G; g += S) {
        simdgroup_matrix<float, 8, 8> P[NT];
        for (int t = 0; t < NT; ++t) {
            P[t] = simdgroup_matrix<float, 8, 8>(0.0f);
        }

        uint w_a[NT], w_b[NT];
        if constexpr (BITS == 4) {
            for (int t = 0; t < NT; ++t) {
                uint2 v = *(const device uint2*)(w_exp + size_t(nrow[t]) * WPR + 8 * g + fn);
                w_a[t] = v.x;
                w_b[t] = v.y;
            }
        } else {
            const int woff = 3 * (fn >> 2) + ((fn & 2) ? 1 : 0);
            const bool hi_pair = (fn & 2) != 0;
            for (int t = 0; t < NT; ++t) {
                const device uint* p = w_exp + size_t(nrow[t]) * WPR + 6 * g + woff;
                const uint w0 = p[0], w1 = p[1];
                w_a[t] = hi_pair ? ((w0 >> 16) | (w1 << 16)) : w0;
                w_b[t] = hi_pair ? (w1 >> 8) : ((w0 >> 24) | (w1 << 8));
            }
        }

        const int xo = 64 * g + 8 * fm;
        const uint4 xa = *(const device uint4*)(x0 + xo);
        const uint4 xb = *(const device uint4*)(x1 + xo);
        float sum0 = 0.0f, sum1 = 0.0f;
        for (int e = 0; e < 8; ++e) {
            sum0 += moe_x_at<T>(xa[e / 2], e);
            sum1 += moe_x_at<T>(xb[e / 2], e);
        }

        #pragma clang loop unroll(full)
        for (int s = 0; s < 8; ++s) {
            const uint mask = (BITS == 4 ? 0xFu : 0x7u) << (BITS * s);
            const float pre = as_type<float>(uint(127 - BITS * s) << 23);
            simdgroup_matrix<float, 8, 8> b;
            b.thread_elements()[0] = moe_x_at<T>(xa[s / 2], s) * pre;
            b.thread_elements()[1] = moe_x_at<T>(xb[s / 2], s) * pre;
            for (int t = 0; t < NT; ++t) {
                simdgroup_matrix<float, 8, 8> a;
                a.thread_elements()[0] = float(w_a[t] & mask);
                a.thread_elements()[1] = float(w_b[t] & mask);
                simdgroup_multiply_accumulate(P[t], a, b, P[t]);
            }
        }

        for (int d = 2; d <= 16; d <<= 1) {
            if (d == 8) continue;
            sum0 += simd_shuffle_xor(sum0, ushort(d));
            sum1 += simd_shuffle_xor(sum1, ushort(d));
        }

        for (int t = 0; t < NT; ++t) {
            const float sc = float(s_exp[size_t(nrow[t]) * G + g]);
            const float bi = float(b_exp[size_t(nrow[t]) * G + g]);
            acc[t][0] = fma(bi, sum0, fma(sc, P[t].thread_elements()[0], acc[t][0]));
            acc[t][1] = fma(bi, sum1, fma(sc, P[t].thread_elements()[1], acc[t][1]));
        }
    }

    threadgroup float part[S * NT * 64];
    for (int t = 0; t < NT; ++t) {
        part[(c * NT + t) * 64 + fm * 8 + fn] = acc[t][0];
        part[(c * NT + t) * 64 + fm * 8 + fn + 1] = acc[t][1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr float gcap_v = float(GCAP_Q10) * (1.0f / 1024.0f);
    constexpr float ucap_v = float(UCAP_Q10) * (1.0f / 1024.0f);
    for (int i = c * 32 + lane; i < NT * 32; i += S * 32) {
        const int t = i / 32;
        const int p = (i % 32) / 8;
        const int row_idx = i % 8;
        float sg_sum = 0.0f;
        float su_sum = 0.0f;
        for (int q = 0; q < S; ++q) {
            sg_sum += part[(q * NT + t) * 64 + (2 * p) * 8 + row_idx];
            su_sum += part[(q * NT + t) * 64 + (2 * p + 1) * 8 + row_idx];
        }
        const int out_col = (n0 / 2) + 4 * t + p;
        if (row_idx < int(cnt) && out_col < (N / 2)) {
            const float gv = float(T(sg_sum));
            const float uv = float(T(su_sum));
            const float sig = 1.0f / (1.0f + metal::exp(-gv));
            const float act = metal::tanh(gv / gcap_v) * gcap_v * sig * (metal::tanh(uv / ucap_v) * ucap_v);
            OUT[size_t(pairs[row_idx]) * (N / 2) + out_col] = T(act);
        }
    }
"""

# Grouped w2 projection via simdgroup_matrix MMA (RT=1).
_MOE_W2_SOURCE = r"""
    const uint u = threadgroup_position_in_grid.y;
    if (u >= u_num[0]) {
        return;
    }
    const uint cnt = u_cnt[u];
    const size_t expert = size_t(u_exp[u]);
    const int lane = int(thread_index_in_simdgroup);
    const int c = int(simdgroup_index_in_threadgroup);
    const int fm = ((lane >> 4) << 2) + ((lane >> 1) & 3);
    const int fn = (((lane >> 3) & 1) << 2) + ((lane & 1) << 1);
    const int n0 = int(threadgroup_position_in_grid.x) * (8 * NT);

    constexpr int G = K / 64;
    constexpr int WPR = K * BITS / 32;
    constexpr size_t EXP_WORDS = size_t(N) * WPR;
    constexpr size_t EXP_GROUPS = size_t(N) * G;

    int nrow[NT];
    for (int t = 0; t < NT; ++t) {
        nrow[t] = min(n0 + 8 * t + fm, N - 1);
    }

    uint pairs[8];
    for (uint t = 0; t < cnt; ++t) {
        pairs[t] = u_pairs[u * 8 + t];
    }
    const uint p0 = pairs[min(uint(fn), cnt - 1)];
    const uint p1 = pairs[min(uint(fn + 1), cnt - 1)];
    const device T* x0 = X + size_t(p0) * K;
    const device T* x1 = X + size_t(p1) * K;

    const device uint* w_exp = (const device uint*)Wq + expert * EXP_WORDS;
    const device T* s_exp = SC + expert * EXP_GROUPS;
    const device T* b_exp = BI + expert * EXP_GROUPS;

    float acc[NT][2];
    for (int t = 0; t < NT; ++t) {
        acc[t][0] = 0.0f;
        acc[t][1] = 0.0f;
    }

    for (int g = c; g < G; g += S) {
        simdgroup_matrix<float, 8, 8> P[NT];
        for (int t = 0; t < NT; ++t) {
            P[t] = simdgroup_matrix<float, 8, 8>(0.0f);
        }

        uint w_a[NT], w_b[NT];
        if constexpr (BITS == 4) {
            for (int t = 0; t < NT; ++t) {
                uint2 v = *(const device uint2*)(w_exp + size_t(nrow[t]) * WPR + 8 * g + fn);
                w_a[t] = v.x;
                w_b[t] = v.y;
            }
        } else {
            const int woff = 3 * (fn >> 2) + ((fn & 2) ? 1 : 0);
            const bool hi_pair = (fn & 2) != 0;
            for (int t = 0; t < NT; ++t) {
                const device uint* p = w_exp + size_t(nrow[t]) * WPR + 6 * g + woff;
                const uint w0 = p[0], w1 = p[1];
                w_a[t] = hi_pair ? ((w0 >> 16) | (w1 << 16)) : w0;
                w_b[t] = hi_pair ? (w1 >> 8) : ((w0 >> 24) | (w1 << 8));
            }
        }

        const int xo = 64 * g + 8 * fm;
        const uint4 xa = *(const device uint4*)(x0 + xo);
        const uint4 xb = *(const device uint4*)(x1 + xo);
        float sum0 = 0.0f, sum1 = 0.0f;
        for (int e = 0; e < 8; ++e) {
            sum0 += moe_x_at<T>(xa[e / 2], e);
            sum1 += moe_x_at<T>(xb[e / 2], e);
        }

        #pragma clang loop unroll(full)
        for (int s = 0; s < 8; ++s) {
            const uint mask = (BITS == 4 ? 0xFu : 0x7u) << (BITS * s);
            const float pre = as_type<float>(uint(127 - BITS * s) << 23);
            simdgroup_matrix<float, 8, 8> b;
            b.thread_elements()[0] = moe_x_at<T>(xa[s / 2], s) * pre;
            b.thread_elements()[1] = moe_x_at<T>(xb[s / 2], s) * pre;
            for (int t = 0; t < NT; ++t) {
                simdgroup_matrix<float, 8, 8> a;
                a.thread_elements()[0] = float(w_a[t] & mask);
                a.thread_elements()[1] = float(w_b[t] & mask);
                simdgroup_multiply_accumulate(P[t], a, b, P[t]);
            }
        }

        for (int d = 2; d <= 16; d <<= 1) {
            if (d == 8) continue;
            sum0 += simd_shuffle_xor(sum0, ushort(d));
            sum1 += simd_shuffle_xor(sum1, ushort(d));
        }

        for (int t = 0; t < NT; ++t) {
            const float sc = float(s_exp[size_t(nrow[t]) * G + g]);
            const float bi = float(b_exp[size_t(nrow[t]) * G + g]);
            acc[t][0] = fma(bi, sum0, fma(sc, P[t].thread_elements()[0], acc[t][0]));
            acc[t][1] = fma(bi, sum1, fma(sc, P[t].thread_elements()[1], acc[t][1]));
        }
    }

    threadgroup float part[S * NT * 64];
    for (int t = 0; t < NT; ++t) {
        part[(c * NT + t) * 64 + fm * 8 + fn] = acc[t][0];
        part[(c * NT + t) * 64 + fm * 8 + fn + 1] = acc[t][1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = c * 32 + lane; i < NT * 64; i += S * 32) {
        const int t = i / 64;
        const int e = i % 64;
        const int row_idx = e % 8;
        const int local_n = e / 8;
        float v = 0.0f;
        for (int q = 0; q < S; ++q) {
            v += part[(q * NT + t) * 64 + e];
        }
        const int out_col = n0 + 8 * t + local_n;
        if (row_idx < int(cnt) && out_col < N) {
            OUT[size_t(pairs[row_idx]) * N + out_col] = T(v);
        }
    }
"""

_MOE_COMBINE_SOURCE = r"""
    const uint m = threadgroup_position_in_grid.y;
    const uint h = threadgroup_position_in_grid.x * 256 + thread_position_in_threadgroup.x;
    if (h >= H_DIM) {
        return;
    }
    const device T* zp = z + size_t(m) * TOPK * H_DIM + h;
    const device float* wp = weights + size_t(m) * TOPK;
    float sum = 0.0f;
    for (int k = 0; k < TOPK; ++k) {
        sum = fma(float(zp[k * H_DIM]), wp[k], sum);
    }
    out[size_t(m) * H_DIM + h] = T(sum);
"""


@cache
def _group_experts_kernel():
    return mx.fast.metal_kernel(
        name="omlx_inkling2_group_experts",
        input_names=["inds"],
        output_names=["u_exp", "u_cnt", "u_toks", "u_pairs", "u_num"],
        source=_GROUP_EXPERTS_SOURCE,
    )


@cache
def _moe_w13_swiglu_kernel(bits: int):
    return mx.fast.metal_kernel(
        name=f"omlx_inkling2_moe_mma_w13_swiglu_b{bits}",
        input_names=["X", "Wq", "SC", "BI", "u_exp", "u_cnt", "u_toks", "u_pairs", "u_num"],
        output_names=["OUT"],
        header=_MOE_MMA_HEADER,
        source=_MOE_W13_SWIGLU_SOURCE,
    )


@cache
def _moe_w2_kernel(bits: int):
    return mx.fast.metal_kernel(
        name=f"omlx_inkling2_moe_mma_w2_b{bits}",
        input_names=["X", "Wq", "SC", "BI", "u_exp", "u_cnt", "u_pairs", "u_num"],
        output_names=["OUT"],
        header=_MOE_MMA_HEADER,
        source=_MOE_W2_SOURCE,
    )


@cache
def _moe_combine_kernel():
    return mx.fast.metal_kernel(
        name="omlx_inkling2_moe_combine",
        input_names=["z", "weights"],
        output_names=["out"],
        source=_MOE_COMBINE_SOURCE,
    )


def prepare_m1_moe(experts) -> None:
    """Pre-compute 1-expert alias views outside mx.compile to avoid 50MB MLX buffer caps."""
    for proj in (experts.w13, experts.w2):
        if proj.__dict__.get("_omlx_m1_alias") is not None:
            continue
        import numpy as np

        w, s, b = proj.weight, proj.scales, proj.biases
        vw, vs, vb = w[:1], s[:1], b[:1]
        mx.eval(vw, vs, vb)

        def same_ptr(a, v):
            return np.frombuffer(memoryview(a), dtype=np.uint8).ctypes.data == np.frombuffer(memoryview(v), dtype=np.uint8).ctypes.data

        if same_ptr(w, vw) and same_ptr(s, vs) and same_ptr(b, vb):
            res = (vw, vs, vb)
        else:
            res = (w, s, b)
        object.__setattr__(proj, "_omlx_m1_alias", res)


def _expert_alias_views(proj) -> tuple[mx.array, mx.array, mx.array]:
    """Return pre-cached 1-expert alias views without calling mx.eval inside mx.compile."""
    cached = proj.__dict__.get("_omlx_m1_alias")
    if cached is not None:
        return cached
    return (proj.weight, proj.scales, proj.biases)


def ensure_split_in_proj(attn) -> None:
    """Split sliding quantized in_proj once (outside mx.compile) into qrg, k, v linears.

    During looped core passes (xk is not None), stock Inkling2Attention calls
    in_proj(concat([xq, xk, xv], axis=0)), creating 3*M = 21 rows > 16 which misses
    m1_verify_mma and wastes 2/3 of the k/v/qrg rows computed. Splitting into
    qrg [6552, K], k [1024, K], v [1024, K] keeps M = 7 in m1_verify_mma.
    """
    if attn.__dict__.get("_m1_split_proj") is not None:
        return
    import mlx.nn as nn

    proj = getattr(attn, "in_proj", None)
    if (
        not getattr(attn, "sliding", False)
        or not isinstance(proj, nn.QuantizedLinear)
        or getattr(proj, "mode", "affine") != "affine"
    ):
        object.__setattr__(attn, "_m1_split_proj", False)
        return

    cuts = attn._cuts  # [dq, dq+dkv, dq+2*dkv, +dr, +dq] = [3072, 4096, 5120, 5504, 8576]
    k_packed = int(proj.weight.shape[1])
    in_dims = k_packed * 32 // int(proj.bits)

    def make_sub(out_dims: int, w: mx.array, s: mx.array, b: mx.array) -> nn.QuantizedLinear:
        lin = nn.QuantizedLinear(
            in_dims, out_dims, bias=False, group_size=int(proj.group_size), bits=int(proj.bits), mode=proj.mode
        )
        lin.weight = mx.contiguous(w)
        lin.scales = mx.contiguous(s)
        lin.biases = mx.contiguous(b)
        return lin

    c0, c1, c2, c3, c4 = cuts
    kv_span = c2 - c0
    qrg_cuts = [c0, c3 - kv_span, c4 - kv_span]

    qrg_w = mx.concatenate([proj.weight[:c0], proj.weight[c2:]], axis=0)
    qrg_s = mx.concatenate([proj.scales[:c0], proj.scales[c2:]], axis=0)
    qrg_b = mx.concatenate([proj.biases[:c0], proj.biases[c2:]], axis=0)
    qrg_lin = make_sub(int(qrg_w.shape[0]), qrg_w, qrg_s, qrg_b)
    k_lin = make_sub(c1 - c0, proj.weight[c0:c1], proj.scales[c0:c1], proj.biases[c0:c1])
    v_lin = make_sub(c2 - c1, proj.weight[c1:c2], proj.scales[c1:c2], proj.biases[c1:c2])
    mx.eval(
        qrg_lin.weight, qrg_lin.scales, qrg_lin.biases,
        k_lin.weight, k_lin.scales, k_lin.biases,
        v_lin.weight, v_lin.scales, v_lin.biases,
    )
    object.__setattr__(attn, "_m1_split_proj", (qrg_lin, k_lin, v_lin, qrg_cuts))


def grouped_inkling2_moe(
    x: mx.array,
    inds: mx.array,
    weights: mx.array,
    w13,
    w2,
    gcap: float = 4.0,
    ucap: float = 25.0,
    w13_geom: tuple[int, int] | None = None,
    w2_geom: tuple[int, int] | None = None,
) -> mx.array:
    """Grouped-expert verify MoE via M1 simdgroup_matrix MMA for 1 <= M <= 8, affine 3/4-bit gs=64."""
    if w13_geom is None:
        w13_geom = (6, 2)
    if w2_geom is None:
        w2_geom = (4, 2) if int(w2.bits) == 3 else (6, 2)
    x = mx.contiguous(x)
    inds = mx.contiguous(inds)
    m, k_h = int(x.shape[0]), int(x.shape[1])
    top_k = int(inds.shape[1])
    n_pairs = m * top_k
    inter = int(w2.scales.shape[-1]) * int(w2.group_size)  # 1152

    flat_inds = inds.reshape(-1).astype(mx.uint32)
    u_exp, u_cnt, u_toks, u_pairs, u_num = _group_experts_kernel()(
        inputs=[flat_inds],
        template=[("N_PAIRS", n_pairs), ("TOPK", top_k)],
        grid=(128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(n_pairs,), (n_pairs,), (n_pairs * 8,), (n_pairs * 8,), (1,)],
        output_dtypes=[mx.uint32, mx.uint32, mx.uint32, mx.uint32, mx.uint32],
    )

    w13_w, w13_s, w13_b = _expert_alias_views(w13)
    nt1, s1 = w13_geom
    n13 = 2 * inter
    (act,) = _moe_w13_swiglu_kernel(int(w13.bits))(
        inputs=[x, w13_w, w13_s, w13_b, u_exp, u_cnt, u_toks, u_pairs, u_num],
        template=[
            ("T", x.dtype),
            ("K", k_h),
            ("N", n13),
            ("BITS", int(w13.bits)),
            ("NT", nt1),
            ("S", s1),
            ("GCAP_Q10", int(round(float(gcap) * 1024.0))),
            ("UCAP_Q10", int(round(float(ucap) * 1024.0))),
        ],
        grid=(((n13 + 8 * nt1 - 1) // (8 * nt1)) * s1 * 32, n_pairs, 1),
        threadgroup=(s1 * 32, 1, 1),
        output_shapes=[(n_pairs, inter)],
        output_dtypes=[x.dtype],
    )

    w2_w, w2_s, w2_b = _expert_alias_views(w2)
    nt2, s2 = w2_geom
    (z,) = _moe_w2_kernel(int(w2.bits))(
        inputs=[act, w2_w, w2_s, w2_b, u_exp, u_cnt, u_pairs, u_num],
        template=[
            ("T", x.dtype),
            ("K", inter),
            ("N", k_h),
            ("BITS", int(w2.bits)),
            ("NT", nt2),
            ("S", s2),
        ],
        grid=(((k_h + 8 * nt2 - 1) // (8 * nt2)) * s2 * 32, n_pairs, 1),
        threadgroup=(s2 * 32, 1, 1),
        output_shapes=[(n_pairs, k_h)],
        output_dtypes=[x.dtype],
    )

    (out,) = _moe_combine_kernel()(
        inputs=[z, weights.astype(mx.float32)],
        template=[("T", x.dtype), ("H_DIM", k_h), ("TOPK", top_k)],
        grid=(((k_h + 255) // 256) * 256, m, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(m, k_h)],
        output_dtypes=[x.dtype],
    )
    return out


# ---------------------------------------------------------------------------
# Step 2a: Sub-warp lpr=8 grouped MoE (handles cnt=1..4 with zero 8x8 waste)
# ---------------------------------------------------------------------------

from functools import lru_cache as _lru_cache

_SUBWARP_LOAD_HDR = r"""
template <typename T>
inline float4 _sw_load4(const device T* p) {
    return float4(float(p[0]), float(p[1]), float(p[2]), float(p[3]));
}
template <typename T>
inline float4 _sw_load4(const constant T* p) {
    return float4(float(p[0]), float(p[1]), float(p[2]), float(p[3]));
}
template <>
inline float4 _sw_load4<bfloat16_t>(const device bfloat16_t* p) {
    return float4(*(const device bfloat4*)p);
}
template <>
inline float4 _sw_load4<bfloat16_t>(const constant bfloat16_t* p) {
    return float4(*(const constant bfloat4*)p);
}
template <>
inline float4 _sw_load4<half>(const device half* p) {
    return float4(*(const device half4*)p);
}
template <>
inline float4 _sw_load4<half>(const constant half* p) {
    return float4(*(const constant half4*)p);
}

template <typename T>
inline T _sw_cast(float v) {
    return T(v);
}
template <>
inline bfloat16_t _sw_cast<bfloat16_t>(float v) {
    uint bits = as_type<uint>(v);
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) return bfloat16_t(v);
    bits += 0x7FFFu + ((bits >> 16u) & 1u);
    return as_type<bfloat16_t>(ushort(bits >> 16u));
}

template <int BITS>
inline void _sw_dot16(
    const device uint* w,
    ulong row_wpr_off,
    uint sub_idx,
    thread const float4& x0,
    thread const float4& x1,
    thread const float4& x2,
    thread const float4& x3,
    thread float& dw,
    thread float& sx
) {
    if (BITS == 4) {
        const device uint2* wp = (const device uint2*)(w + row_wpr_off) + sub_idx;
        uint2 q = wp[0];
        uint q0 = q.x, q1 = q.y;
        float4 w0 = float4(float(q0 & 15u), float((q0 >> 4u) & 15u), float((q0 >> 8u) & 15u), float((q0 >> 12u) & 15u));
        float4 w1 = float4(float((q0 >> 16u) & 15u), float((q0 >> 20u) & 15u), float((q0 >> 24u) & 15u), float((q0 >> 28u) & 15u));
        float4 w2 = float4(float(q1 & 15u), float((q1 >> 4u) & 15u), float((q1 >> 8u) & 15u), float((q1 >> 12u) & 15u));
        float4 w3 = float4(float((q1 >> 16u) & 15u), float((q1 >> 20u) & 15u), float((q1 >> 24u) & 15u), float((q1 >> 28u) & 15u));
        dw += dot(x0, w0) + dot(x1, w1) + dot(x2, w2) + dot(x3, w3);
    } else {
        const device ushort* wp = (const device ushort*)(w + row_wpr_off) + sub_idx * 3u;
        uint p0 = uint(wp[0]), p1 = uint(wp[1]), p2 = uint(wp[2]);
        uint u0 = p0 | (p1 << 16u);
        float4 w0 = float4(float(u0 & 7u), float((u0 >> 3u) & 7u), float((u0 >> 6u) & 7u), float((u0 >> 9u) & 7u));
        float4 w1 = float4(float((u0 >> 12u) & 7u), float((u0 >> 15u) & 7u), float((u0 >> 18u) & 7u), float((u0 >> 21u) & 7u));
        float4 w2 = float4(float((u0 >> 24u) & 7u), float((u0 >> 27u) & 7u), float(((u0 >> 30u) | ((p2 & 1u) << 2u)) & 7u), float((p2 >> 1u) & 7u));
        float4 w3 = float4(float((p2 >> 4u) & 7u), float((p2 >> 7u) & 7u), float((p2 >> 10u) & 7u), float((p2 >> 13u) & 7u));
        dw += dot(x0, w0) + dot(x1, w1) + dot(x2, w2) + dot(x3, w3);
    }
    float4 s4 = x0 + x1 + x2 + x3;
    sx += s4.x + s4.y + s4.z + s4.w;
}

template <int BITS, int C, int RPS1, typename T, typename PX, typename PS, typename PI>
inline void _sw_w13_exec(
    PX x,
    const device uint* w,
    PS s,
    PS b,
    PI c_toks,
    PI c_pairs,
    device T* act,
    uint u,
    uint eid,
    uint r0,
    uint hl,
    uint K,
    uint N,
    uint gcap_q10,
    uint ucap_q10
) {
    const uint inter = N >> 1;
    if (r0 >= inter) return;
    const uint wpr = (K * uint(BITS)) >> 5;
    const uint n_grp = K >> 6;
    const uint steps = K >> 7;

    uint toks[C];
    for (uint c = 0u; c < C; c++) toks[c] = c_toks[u * 4u + c];

    float ag[RPS1][C], au[RPS1][C];
    for (uint rr = 0u; rr < RPS1; rr++) {
        for (uint c = 0u; c < C; c++) { ag[rr][c] = 0.0f; au[rr][c] = 0.0f; }
    }

    for (uint it = 0u; it < steps; it++) {
        const uint sub = it * 8u + hl;
        const uint col0 = sub * 16u;
        const uint g = it * 2u + (hl >> 2u);
        for (uint c = 0u; c < C; c++) {
            PX xp = x + (ulong)toks[c] * K + col0;
            float4 x0 = _sw_load4<T>(xp);
            float4 x1 = _sw_load4<T>(xp + 4u);
            float4 x2 = _sw_load4<T>(xp + 8u);
            float4 x3 = _sw_load4<T>(xp + 12u);
            #pragma unroll
            for (uint rr = 0u; rr < RPS1; rr++) {
                const uint r = r0 + rr;
                if (r < inter) {
                    const ulong off0 = (ulong)eid * N + 2u * r;
                    const ulong off1 = off0 + 1u;
                    const float sc0 = float(s[off0 * n_grp + g]), bi0 = float(b[off0 * n_grp + g]);
                    const float sc1 = float(s[off1 * n_grp + g]), bi1 = float(b[off1 * n_grp + g]);
                    float dw0 = 0.0f, sx0 = 0.0f;
                    _sw_dot16<BITS>(w, off0 * wpr, sub, x0, x1, x2, x3, dw0, sx0);
                    ag[rr][c] += dw0 * sc0 + sx0 * bi0;
                    float dw1 = 0.0f, sx1 = 0.0f;
                    _sw_dot16<BITS>(w, off1 * wpr, sub, x0, x1, x2, x3, dw1, sx1);
                    au[rr][c] += dw1 * sc1 + sx1 * bi1;
                }
            }
        }
    }

    const float gcap = float(gcap_q10) / 1024.0f;
    const float ucap = float(ucap_q10) / 1024.0f;
    #pragma unroll
    for (uint rr = 0u; rr < RPS1; rr++) {
        const uint r = r0 + rr;
        if (r < inter) {
            for (uint c = 0u; c < C; c++) {
                float sgv = ag[rr][c], suv = au[rr][c];
                sgv += simd_shuffle_xor(sgv, 4u);
                sgv += simd_shuffle_xor(sgv, 2u);
                sgv += simd_shuffle_xor(sgv, 1u);
                suv += simd_shuffle_xor(suv, 4u);
                suv += simd_shuffle_xor(suv, 2u);
                suv += simd_shuffle_xor(suv, 1u);
                if (hl == 0u) {
                    float gt = metal::precise::tanh(sgv / gcap) * gcap;
                    float sig = 1.0f / (1.0f + metal::precise::exp(-sgv));
                    float ut = metal::precise::tanh(suv / ucap) * ucap;
                    uint p_idx = c_pairs[u * 4u + c];
                    act[(ulong)p_idx * inter + r] = _sw_cast<T>((gt * sig) * ut);
                }
            }
        }
    }
}

template <int BITS, int C, int RPS, typename T, typename PX, typename PS, typename PI>
inline void _sw_w2_exec(
    PX act,
    const device uint* w,
    PS s,
    PS b,
    PI c_pairs,
    device T* z,
    uint u,
    uint eid,
    uint r0,
    uint hl,
    uint K,
    uint N
) {
    const uint wpr = (K * uint(BITS)) >> 5;
    const uint n_grp = K >> 6;
    const uint steps = K >> 7;
    uint pairs[C];
    for (uint c = 0u; c < C; c++) pairs[c] = c_pairs[u * 4u + c];

    float acc[RPS][C];
    for (uint rr = 0u; rr < RPS; rr++) {
        for (uint c = 0u; c < C; c++) acc[rr][c] = 0.0f;
    }

    for (uint it = 0u; it < steps; it++) {
        const uint sub = it * 8u + hl;
        const uint col0 = sub * 16u;
        const uint g = it * 2u + (hl >> 2u);
        for (uint c = 0u; c < C; c++) {
            PX xp = act + (ulong)pairs[c] * K + col0;
            float4 x0 = _sw_load4<T>(xp);
            float4 x1 = _sw_load4<T>(xp + 4u);
            float4 x2 = _sw_load4<T>(xp + 8u);
            float4 x3 = _sw_load4<T>(xp + 12u);
            #pragma unroll
            for (uint rr = 0u; rr < RPS; rr++) {
                const uint row = r0 + rr;
                if (row < N) {
                    const ulong off = (ulong)eid * N + row;
                    const float sc = float(s[off * n_grp + g]), bi = float(b[off * n_grp + g]);
                    float dw = 0.0f, sx = 0.0f;
                    _sw_dot16<BITS>(w, off * wpr, sub, x0, x1, x2, x3, dw, sx);
                    acc[rr][c] += dw * sc + sx * bi;
                }
            }
        }
    }

    #pragma unroll
    for (uint rr = 0u; rr < RPS; rr++) {
        const uint row = r0 + rr;
        if (row < N) {
            for (uint c = 0u; c < C; c++) {
                float sv = acc[rr][c];
                sv += simd_shuffle_xor(sv, 4u);
                sv += simd_shuffle_xor(sv, 2u);
                sv += simd_shuffle_xor(sv, 1u);
                if (hl == 0u) {
                    z[(ulong)pairs[c] * N + row] = _sw_cast<T>(sv);
                }
            }
        }
    }
}
"""


@_lru_cache(maxsize=4)
def _plan_experts_kernel():
    source = r"""
        const uint tid = thread_index_in_threadgroup;
        const uint simd = simdgroup_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;

        uint my_cnt = 0u;
        for (uint p = 0u; p < N_PAIRS; p++) {
            if (flat_inds[p] == tid) my_cnt++;
        }
        const uint my_chunks = (my_cnt + 3u) >> 2;
        const uint s_pref = simd_prefix_exclusive_sum(my_chunks);
        threadgroup uint sg_tot[4];
        if (lane == 31u) sg_tot[simd] = s_pref + my_chunks;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            uint run = 0u;
            for (uint s = 0u; s < 4u; s++) {
                uint v = sg_tot[s];
                sg_tot[s] = run;
                run += v;
            }
            c_num[0] = run;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (my_cnt > 0u) {
            uint u = sg_tot[simd] + s_pref;
            uint k = 0u;
            for (uint p = 0u; p < N_PAIRS; p++) {
                if (flat_inds[p] == tid) {
                    c_toks[u * 4u + k] = p / TOPK;
                    c_pairs[u * 4u + k] = p;
                    k++;
                    if (k == 4u) {
                        c_exp[u] = tid;
                        c_cnt[u] = 4u;
                        u++;
                        k = 0u;
                    }
                }
            }
            if (k > 0u) {
                c_exp[u] = tid;
                c_cnt[u] = k;
            }
        }
    """
    return mx.fast.metal_kernel(
        name="inkling2_plan_experts_u4",
        input_names=["flat_inds"],
        output_names=["c_exp", "c_cnt", "c_toks", "c_pairs", "c_num"],
        source=source,
    )


@_lru_cache(maxsize=8)
def _subwarp_moe_w13_kernel(bits: int):
    source = r"""
        const uint u = threadgroup_position_in_grid.y;
        if (u >= c_num[0]) return;
        const uint simd = simdgroup_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        const uint hw = lane >> 3u;
        const uint hl = lane & 7u;
        const uint sw_id = simd * 4u + hw;
        const uint r0 = (threadgroup_position_in_grid.x * (SG * 4u) + sw_id) * RPS1;
        const uint eid = c_exp[u];
        const uint cnt = c_cnt[u];
        if (cnt == 1u) {
            _sw_w13_exec<BITS, 1, RPS1, T>(x, w, s, b, c_toks, c_pairs, act, u, eid, r0, hl, K, N, GCAP_Q10, UCAP_Q10);
        } else if (cnt == 2u) {
            _sw_w13_exec<BITS, 2, RPS1, T>(x, w, s, b, c_toks, c_pairs, act, u, eid, r0, hl, K, N, GCAP_Q10, UCAP_Q10);
        } else if (cnt == 3u) {
            _sw_w13_exec<BITS, 3, RPS1, T>(x, w, s, b, c_toks, c_pairs, act, u, eid, r0, hl, K, N, GCAP_Q10, UCAP_Q10);
        } else {
            _sw_w13_exec<BITS, 4, RPS1, T>(x, w, s, b, c_toks, c_pairs, act, u, eid, r0, hl, K, N, GCAP_Q10, UCAP_Q10);
        }
    """
    return mx.fast.metal_kernel(
        name=f"inkling2_sw_moe_w13_b{bits}",
        input_names=["x", "w", "s", "b", "c_exp", "c_cnt", "c_toks", "c_pairs", "c_num"],
        output_names=["act"],
        header=_SUBWARP_LOAD_HDR,
        source=source,
    )


@_lru_cache(maxsize=8)
def _subwarp_moe_w2_kernel(bits: int):
    source = r"""
        const uint u = threadgroup_position_in_grid.y;
        if (u >= c_num[0]) return;
        const uint simd = simdgroup_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        const uint hw = lane >> 3u;
        const uint hl = lane & 7u;
        const uint sw_id = simd * 4u + hw;
        const uint r0 = (threadgroup_position_in_grid.x * (SG * 4u) + sw_id) * RPS;
        const uint eid = c_exp[u];
        const uint cnt = c_cnt[u];
        if (cnt == 1u) {
            _sw_w2_exec<BITS, 1, RPS, T>(act, w, s, b, c_pairs, z, u, eid, r0, hl, K, N);
        } else if (cnt == 2u) {
            _sw_w2_exec<BITS, 2, RPS, T>(act, w, s, b, c_pairs, z, u, eid, r0, hl, K, N);
        } else if (cnt == 3u) {
            _sw_w2_exec<BITS, 3, RPS, T>(act, w, s, b, c_pairs, z, u, eid, r0, hl, K, N);
        } else {
            _sw_w2_exec<BITS, 4, RPS, T>(act, w, s, b, c_pairs, z, u, eid, r0, hl, K, N);
        }
    """
    return mx.fast.metal_kernel(
        name=f"inkling2_sw_moe_w2_b{bits}",
        input_names=["act", "w", "s", "b", "c_exp", "c_cnt", "c_pairs", "c_num"],
        output_names=["z"],
        header=_SUBWARP_LOAD_HDR,
        source=source,
    )


def subwarp_inkling2_moe(
    x: mx.array,
    inds: mx.array,
    weights: mx.array,
    w13,
    w2,
    gcap: float = 4.0,
    ucap: float = 25.0,
    sg1: int = 4,
    rps1: int = 1,
    sg2: int = 4,
    rps2: int = 4,
) -> mx.array:
    """Sub-warp lpr=8 grouped MoE for M=2..16 (zero 8x8 padding waste on cnt=1..4)."""
    x = mx.contiguous(x)
    inds = mx.contiguous(inds)
    m, k_h = int(x.shape[0]), int(x.shape[1])
    top_k = int(inds.shape[1])
    n_pairs = m * top_k
    inter = int(w2.scales.shape[-1]) * int(w2.group_size)
    n13 = 2 * inter

    flat_inds = inds.reshape(-1).astype(mx.uint32)
    c_exp, c_cnt, c_toks, c_pairs, c_num = _plan_experts_kernel()(
        inputs=[flat_inds],
        template=[("N_PAIRS", n_pairs), ("TOPK", top_k)],
        grid=(128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(n_pairs,), (n_pairs,), (n_pairs * 4,), (n_pairs * 4,), (1,)],
        output_dtypes=[mx.uint32, mx.uint32, mx.uint32, mx.uint32, mx.uint32],
    )

    w13_w, w13_s, w13_b = _expert_alias_views(w13)
    rows_per_tg1 = sg1 * 4 * rps1
    (act,) = _subwarp_moe_w13_kernel(int(w13.bits))(
        inputs=[x, w13_w, w13_s, w13_b, c_exp, c_cnt, c_toks, c_pairs, c_num],
        template=[
            ("T", x.dtype),
            ("K", k_h),
            ("N", n13),
            ("BITS", int(w13.bits)),
            ("SG", sg1),
            ("RPS1", rps1),
            ("GCAP_Q10", int(round(float(gcap) * 1024.0))),
            ("UCAP_Q10", int(round(float(ucap) * 1024.0))),
        ],
        grid=(((inter + rows_per_tg1 - 1) // rows_per_tg1) * sg1 * 32, n_pairs, 1),
        threadgroup=(sg1 * 32, 1, 1),
        output_shapes=[(n_pairs, inter)],
        output_dtypes=[x.dtype],
    )

    w2_w, w2_s, w2_b = _expert_alias_views(w2)
    rows_per_tg2 = sg2 * 4 * rps2
    (z,) = _subwarp_moe_w2_kernel(int(w2.bits))(
        inputs=[act, w2_w, w2_s, w2_b, c_exp, c_cnt, c_pairs, c_num],
        template=[
            ("T", x.dtype),
            ("K", inter),
            ("N", k_h),
            ("BITS", int(w2.bits)),
            ("SG", sg2),
            ("RPS", rps2),
        ],
        grid=(((k_h + rows_per_tg2 - 1) // rows_per_tg2) * sg2 * 32, n_pairs, 1),
        threadgroup=(sg2 * 32, 1, 1),
        output_shapes=[(n_pairs, k_h)],
        output_dtypes=[x.dtype],
    )

    (out,) = _moe_combine_kernel()(
        inputs=[z, weights.astype(mx.float32)],
        template=[("T", x.dtype), ("H_DIM", k_h), ("TOPK", top_k)],
        grid=(((k_h + 255) // 256) * 256, m, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(m, k_h)],
        output_dtypes=[x.dtype],
    )
    return out


# ---------------------------------------------------------------------------
# Step 2b: m1_attn_v2 (1-pass 4-SG GQA verify + inline r @ rel_proj_t dot)
# ---------------------------------------------------------------------------

_ATTN_V2_HDR = r"""
template <typename T>
inline float4 _av2_load4(const device T* p) {
    return float4(float(p[0]), float(p[1]), float(p[2]), float(p[3]));
}
template <typename T>
inline float4 _av2_load4(const constant T* p) {
    return float4(float(p[0]), float(p[1]), float(p[2]), float(p[3]));
}
template <>
inline float4 _av2_load4<bfloat16_t>(const device bfloat16_t* p) {
    return float4(*(const device bfloat4*)p);
}
template <>
inline float4 _av2_load4<bfloat16_t>(const constant bfloat16_t* p) {
    return float4(*(const constant bfloat4*)p);
}
template <>
inline float4 _av2_load4<half>(const device half* p) {
    return float4(*(const device half4*)p);
}
template <>
inline float4 _av2_load4<half>(const constant half* p) {
    return float4(*(const constant half4*)p);
}
"""


@_lru_cache(maxsize=4)
def _attn_v2_1pass_kernel(has_lp: bool):
    lp_expr = "uint(max(0, lp[0]))" if has_lp else "0u"
    source = f"""
        const uint ht = threadgroup_position_in_grid.y;
        const uint t = ht / N_HEADS;
        const uint h = ht - t * N_HEADS;
        const uint kvh = (h * N_KV) / N_HEADS;
        const uint simd = simdgroup_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;

        const uint q_pos = (uint(S_LEN) - uint(L_LEN)) + t;
        const uint lp_u = {lp_expr};
        const uint win_lo = (uint(WIN) > 0u && q_pos >= uint(WIN)) ? (q_pos - uint(WIN) + 1u) : 0u;
        const uint j_lo = max(lp_u, win_lo);
        const uint j_hi = min(uint(S_LEN), q_pos + 1u);

        const ulong q_off = (ulong)h * uint(q_strides[1]) + (ulong)t * uint(q_strides[2]) + lane * 4u;
        const float4 qv = _av2_load4<T>(q + q_off) * scale_buf[0];

        const ulong r_off = ((ulong)t * N_HEADS + h) * 16u;
        const float4 rs0 = _av2_load4<T>(r + r_off);
        const float4 rs1 = _av2_load4<T>(r + r_off + 4u);
        const float4 rs2 = _av2_load4<T>(r + r_off + 8u);
        const float4 rs3 = _av2_load4<T>(r + r_off + 12u);

        const ulong k_h_off = (ulong)kvh * uint(k_strides[1]);
        const ulong v_h_off = (ulong)kvh * uint(v_strides[1]);
        const uint k_s_stride = uint(k_strides[2]);
        const uint v_s_stride = uint(v_strides[2]);

        float m_loc = -1e30f;
        float l_loc = 0.0f;
        float4 acc = float4(0.0f);

        for (uint j = j_lo + simd; j < j_hi; j += 4u) {{
            const float4 kv = _av2_load4<T>(k + k_h_off + (ulong)j * k_s_stride + lane * 4u);
            float score = simd_sum(dot(qv, kv));
            const uint dist = q_pos - j;
            if (dist < 1024u) {{
                const ulong rp_off = (ulong)dist * 16u;
                score += dot(rs0, _av2_load4<T>(rel_proj_t + rp_off))
                       + dot(rs1, _av2_load4<T>(rel_proj_t + rp_off + 4u))
                       + dot(rs2, _av2_load4<T>(rel_proj_t + rp_off + 8u))
                       + dot(rs3, _av2_load4<T>(rel_proj_t + rp_off + 12u));
            }}
            const float m_new = max(m_loc, score);
            const float alpha = (l_loc > 0.0f) ? metal::precise::exp(m_loc - m_new) : 0.0f;
            const float p = metal::precise::exp(score - m_new);
            l_loc = l_loc * alpha + p;
            const float4 vv = _av2_load4<T>(v + v_h_off + (ulong)j * v_s_stride + lane * 4u);
            acc = acc * alpha + p * vv;
            m_loc = m_new;
        }}

        threadgroup float tg_m[4], tg_l[4];
        threadgroup float4 tg_a[4][32];
        if (lane == 0u) {{ tg_m[simd] = m_loc; tg_l[simd] = l_loc; }}
        tg_a[simd][lane] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd == 0u) {{
            const float sk = float(sink[t * N_HEADS + h]);
            float m_all = sk;
            for (uint s = 0u; s < 4u; s++) {{
                if (tg_l[s] > 0.0f) m_all = max(m_all, tg_m[s]);
            }}
            float l_all = metal::precise::exp(sk - m_all);
            float4 a_all = float4(0.0f);
            for (uint s = 0u; s < 4u; s++) {{
                if (tg_l[s] > 0.0f) {{
                    const float sc = metal::precise::exp(tg_m[s] - m_all);
                    l_all += tg_l[s] * sc;
                    a_all += tg_a[s][lane] * sc;
                }}
            }}
            const float4 out_v = a_all * (1.0f / l_all);
            const ulong o_off = ((ulong)h * L_LEN + t) * 128u + lane * 4u;
            out[o_off] = T(out_v.x);
            out[o_off + 1u] = T(out_v.y);
            out[o_off + 2u] = T(out_v.z);
            out[o_off + 3u] = T(out_v.w);
        }}
    """
    return mx.fast.metal_kernel(
        name=f"inkling2_attn_v2_1pass_lp{int(has_lp)}",
        input_names=["q", "k", "v", "r", "rel_proj_t", "sink", "lp", "scale_buf"],
        output_names=["out"],
        header=_ATTN_V2_HDR,
        source=source,
        ensure_row_contiguous=False,
    )


@_lru_cache(maxsize=4)
def _attn_v2_split_pass1_kernel(has_lp: bool):
    lp_expr = "uint(max(0, lp[0]))" if has_lp else "0u"
    source = f"""
        const uint ht = threadgroup_position_in_grid.y;
        const uint sp = threadgroup_position_in_grid.z;
        const uint t = ht / N_HEADS;
        const uint h = ht - t * N_HEADS;
        const uint kvh = (h * N_KV) / N_HEADS;
        const uint simd = simdgroup_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;

        const uint q_pos = (uint(S_LEN) - uint(L_LEN)) + t;
        const uint lp_u = {lp_expr};
        const uint win_lo = (uint(WIN) > 0u && q_pos >= uint(WIN)) ? (q_pos - uint(WIN) + 1u) : 0u;
        const uint j_lo = max(lp_u, win_lo);
        const uint j_hi = min(uint(S_LEN), q_pos + 1u);
        const uint span = (j_hi > j_lo) ? (j_hi - j_lo) : 0u;
        const uint j_beg = j_lo + (span * sp) / N_SPLITS;
        const uint j_end = j_lo + (span * (sp + 1u)) / N_SPLITS;

        const ulong q_off = (ulong)h * uint(q_strides[1]) + (ulong)t * uint(q_strides[2]) + lane * 4u;
        const float4 qv = _av2_load4<T>(q + q_off) * scale_buf[0];

        const ulong r_off = ((ulong)t * N_HEADS + h) * 16u;
        const float4 rs0 = _av2_load4<T>(r + r_off);
        const float4 rs1 = _av2_load4<T>(r + r_off + 4u);
        const float4 rs2 = _av2_load4<T>(r + r_off + 8u);
        const float4 rs3 = _av2_load4<T>(r + r_off + 12u);

        const ulong k_h_off = (ulong)kvh * uint(k_strides[1]);
        const ulong v_h_off = (ulong)kvh * uint(v_strides[1]);
        const uint k_s_stride = uint(k_strides[2]);
        const uint v_s_stride = uint(v_strides[2]);

        float m_loc = -1e30f;
        float l_loc = 0.0f;
        float4 acc = float4(0.0f);

        for (uint j = j_beg + simd; j < j_end; j += 4u) {{
            const float4 kv = _av2_load4<T>(k + k_h_off + (ulong)j * k_s_stride + lane * 4u);
            float score = simd_sum(dot(qv, kv));
            const uint dist = q_pos - j;
            if (dist < 1024u) {{
                const ulong rp_off = (ulong)dist * 16u;
                score += dot(rs0, _av2_load4<T>(rel_proj_t + rp_off))
                       + dot(rs1, _av2_load4<T>(rel_proj_t + rp_off + 4u))
                       + dot(rs2, _av2_load4<T>(rel_proj_t + rp_off + 8u))
                       + dot(rs3, _av2_load4<T>(rel_proj_t + rp_off + 12u));
            }}
            const float m_new = max(m_loc, score);
            const float alpha = (l_loc > 0.0f) ? metal::precise::exp(m_loc - m_new) : 0.0f;
            const float p = metal::precise::exp(score - m_new);
            l_loc = l_loc * alpha + p;
            const float4 vv = _av2_load4<T>(v + v_h_off + (ulong)j * v_s_stride + lane * 4u);
            acc = acc * alpha + p * vv;
            m_loc = m_new;
        }}

        threadgroup float tg_m[4], tg_l[4];
        threadgroup float4 tg_a[4][32];
        if (lane == 0u) {{ tg_m[simd] = m_loc; tg_l[simd] = l_loc; }}
        tg_a[simd][lane] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd == 0u) {{
            float m_sp = -1e30f;
            float l_sp = 0.0f;
            float4 a_sp = float4(0.0f);
            for (uint s = 0u; s < 4u; s++) {{
                if (tg_l[s] > 0.0f) m_sp = max(m_sp, tg_m[s]);
            }}
            for (uint s = 0u; s < 4u; s++) {{
                if (tg_l[s] > 0.0f) {{
                    const float sc = metal::precise::exp(tg_m[s] - m_sp);
                    l_sp += tg_l[s] * sc;
                    a_sp += tg_a[s][lane] * sc;
                }}
            }}
            const ulong sp_idx = (ulong)ht * N_SPLITS + sp;
            if (lane == 0u) {{
                pm[sp_idx] = m_sp;
                pl[sp_idx] = l_sp;
            }}
            device float4* pa4 = (device float4*)(pacc + sp_idx * 128u);
            pa4[lane] = a_sp;
        }}
    """
    return mx.fast.metal_kernel(
        name=f"inkling2_attn_v2_p1_lp{int(has_lp)}",
        input_names=["q", "k", "v", "r", "rel_proj_t", "lp", "scale_buf"],
        output_names=["pm", "pl", "pacc"],
        header=_ATTN_V2_HDR,
        source=source,
        ensure_row_contiguous=False,
    )


@_lru_cache(maxsize=2)
def _attn_v2_split_pass2_kernel():
    source = r"""
        const uint ht = threadgroup_position_in_grid.y;
        const uint t = ht / N_HEADS;
        const uint h = ht - t * N_HEADS;
        const uint lane = thread_index_in_simdgroup;

        const float sk = float(sink[t * N_HEADS + h]);
        float m_all = sk;
        const ulong base = (ulong)ht * N_SPLITS;
        for (uint sp = 0u; sp < N_SPLITS; sp++) {
            if (pl[base + sp] > 0.0f) m_all = max(m_all, pm[base + sp]);
        }
        float l_all = metal::precise::exp(sk - m_all);
        float4 a_all = float4(0.0f);
        for (uint sp = 0u; sp < N_SPLITS; sp++) {
            const float ls = pl[base + sp];
            if (ls > 0.0f) {
                const float sc = metal::precise::exp(pm[base + sp] - m_all);
                l_all += ls * sc;
                const device float4* pa4 = (const device float4*)(pacc + (base + sp) * 128u);
                a_all += pa4[lane] * sc;
            }
        }
        const float4 out_v = a_all * (1.0f / l_all);
        const ulong o_off = ((ulong)h * L_LEN + t) * 128u + lane * 4u;
        out[o_off] = T(out_v.x);
        out[o_off + 1u] = T(out_v.y);
        out[o_off + 2u] = T(out_v.z);
        out[o_off + 3u] = T(out_v.w);
    """
    return mx.fast.metal_kernel(
        name="inkling2_attn_v2_p2",
        input_names=["pm", "pl", "pacc", "sink"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


def verify_banded_sink_attention_v2(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    r: mx.array,
    rel_proj_t: mx.array,
    sink: mx.array,
    scale: float,
    window: int = 0,
    left_padding=None,
    n_splits: int = 16,
) -> mx.array:
    """1-pass / split-K GQA verify attention with inline rel_proj_t dot product (skips r @ rel_proj)."""
    _, n_heads, length, _ = q.shape
    n_kv = int(k.shape[1])
    s_len = int(k.shape[2])
    win = int(window or 0)
    r = mx.contiguous(r.astype(q.dtype))
    rel_proj_t = mx.contiguous(rel_proj_t.astype(q.dtype))
    sink = mx.contiguous(sink.astype(q.dtype))
    has_lp = left_padding is not None
    lp_arr = mx.contiguous(left_padding.reshape(-1).astype(mx.int32)) if has_lp else mx.zeros((1,), dtype=mx.int32)
    scale_buf = mx.array([float(scale)], dtype=mx.float32)

    if win > 0 or s_len <= 4096:
        (out,) = _attn_v2_1pass_kernel(has_lp)(
            inputs=[q, k, v, r, rel_proj_t, sink, lp_arr, scale_buf],
            template=[
                ("T", q.dtype),
                ("N_HEADS", n_heads),
                ("N_KV", n_kv),
                ("L_LEN", length),
                ("S_LEN", s_len),
                ("WIN", win),
            ],
            grid=(128, length * n_heads, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(1, n_heads, length, 128)],
            output_dtypes=[q.dtype],
        )
        return out

    pm, pl, pacc = _attn_v2_split_pass1_kernel(has_lp)(
        inputs=[q, k, v, r, rel_proj_t, lp_arr, scale_buf],
        template=[
            ("T", q.dtype),
            ("N_HEADS", n_heads),
            ("N_KV", n_kv),
            ("L_LEN", length),
            ("S_LEN", s_len),
            ("WIN", win),
            ("N_SPLITS", n_splits),
        ],
        grid=(128, length * n_heads, n_splits),
        threadgroup=(128, 1, 1),
        output_shapes=[(length * n_heads, n_splits), (length * n_heads, n_splits), (length * n_heads, n_splits, 128)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    (out,) = _attn_v2_split_pass2_kernel()(
        inputs=[pm, pl, pacc, sink],
        template=[
            ("T", q.dtype),
            ("N_HEADS", n_heads),
            ("L_LEN", length),
            ("N_SPLITS", n_splits),
        ],
        grid=(32, length * n_heads, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(1, n_heads, length, 128)],
        output_dtypes=[q.dtype],
    )
    return out


