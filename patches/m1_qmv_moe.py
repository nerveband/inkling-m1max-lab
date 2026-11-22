# SPDX-License-Identifier: Apache-2.0
"""M1 Max (applegpu_g13) coalesced QMV MoE kernels for Inkling 2 Mini and Ornith.

Implements a 5-stage zero-CPU-sync affine 4-bit (group_size=64) MoE pipeline
designed around M1 Max's 24 GPU cores and 400 GB/s unified memory:

0. ``m1_moe_plan_dual``: 1 threadgroup classifies experts into Bucket 1
   (``cnt == 1``, including the single leftover when ``cnt % 4 == 1``) and
   Bucket 2 (chunks of ``cnt in {2, 3, 4}``), outputting disjoint compact
   work queues plus a per-pair bucket indicator ``pair_is_b2``.
1A/1B. ``m1_moe_w13_c1`` & ``m1_moe_w13_cm``: Fused gate+up + SwiGLU using
   32-lane coalesced ``BLOCK_K = 512`` (``uint2`` weight loads + ``uint4``
   128-bit activation loads) and branchless ``TAIL_LANES`` for non-512-aligned
   K (e.g. Inkling ``K=2304``). Supports both Inkling interleaved ``w13``
   (with softcaps) and Ornith separate ``gate_proj``/``up_proj`` or
   concatenated ``gate_up_proj`` (standard SwiGLU). ``w13_cm`` loads and unpacks
   weights once via ``0x64006400u`` into ``half4`` registers and streams
   ``cnt in {2, 3, 4}`` tokens inner.
2A/2B. ``m1_moe_w2_c1`` & ``m1_moe_w2_cm``: Down projection with routing weight
   multiplication in the epilogue. ``w2_cm`` keeps weights outer in
   ``half4 wf[RPSG][4]`` and streams ``cnt`` activated rows inner.
3. ``m1_moe_combine_vec``: Vectorized FP32 deterministic ``0..K_TOP-1`` sum over
   ``Z1``/``Z2``, with optional Ornith ``shared_out`` / ``shared_gate`` fused.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any

import mlx.core as mx
import numpy as np

_MAX_M = 16
_GROUP_SIZE = 64
_BITS = 4

# Default M1 Max geometries (overrideable via TuneConfig for sweeps).
DEFAULT_W13_C1 = (4, 2)  # NSG=4, RPSG=2 -> 8 INTER outputs/TG
DEFAULT_W13_CM = (4, 2)  # NSG=4, RPSG=2 -> 8 INTER outputs/TG (strict C==2)
DEFAULT_W2_C1 = (4, 4)   # NSG=4, RPSG=4 -> 16 hidden outputs/TG
DEFAULT_W2_CM = (4, 4)   # NSG=4, RPSG=4 -> 16 hidden outputs/TG (strict C==2)
DEFAULT_COMBINE_VEC = 4  # 4 elements/thread (uint2 / vec<T,4>)


@dataclass(frozen=True)
class TuneConfig:
    """Optional geometry override for kernel tuning sweeps."""

    w13_c1: tuple[int, int] = DEFAULT_W13_C1
    w13_cm: tuple[int, int] = DEFAULT_W13_CM
    w2_c1: tuple[int, int] = DEFAULT_W2_C1
    w2_cm: tuple[int, int] = DEFAULT_W2_CM
    combine_vec: int = DEFAULT_COMBINE_VEC


_DEFAULT_TUNE = TuneConfig()

# Bit-level proof of MLX 4-bit affine packing + 0x64006400u half2 unpack:
# In MLX quantized.h, uint32 word q stores 8 consecutive 4-bit nibbles in
# little-endian order: element j sits at bits [4*j + 3 : 4*j] for j = 0..7.
# IEEE-754 half 0x6400 encodes +1024.0 (sign=0, exp=25 -> 2^(25-15)=1024,
# mantissa=0, 1 ULP = 1.0). For any 4-bit nibble n in 0..15,
# as_type<half>(ushort(0x6400 | n)) == half(1024 + n) exactly.
# On little-endian Apple GPU, as_type<half2>(u).x reads bits [15:0] and
# as_type<half2>(u).y reads bits [31:16]. Therefore:
#   ((q >> 0)  & 0x000F000Fu) | 0x64006400u -> half2(1024 + el0, 1024 + el4)
#   ((q >> 4)  & 0x000F000Fu) | 0x64006400u -> half2(1024 + el1, 1024 + el5)
#   ((q >> 8)  & 0x000F000Fu) | 0x64006400u -> half2(1024 + el2, 1024 + el6)
#   ((q >> 12) & 0x000F000Fu) | 0x64006400u -> half2(1024 + el3, 1024 + el7)
# Subtracting half2(1024.0f) and grouping (.x) and (.y) yields exact
# half4(el0..el3) and half4(el4..el7) matching x0 and x1 from uint4 loads.
_COMMON_HEADER = r"""
#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

inline void moe_unpack_u2_half4(
    uint2 q,
    thread half4& w0,
    thread half4& w1,
    thread half4& w2,
    thread half4& w3) {
  const half2 k1024 = half2(1024.0f);

  const uint p04 = (q.x & 0x000F000Fu) | 0x64006400u;
  const uint p15 = ((q.x >> 4u) & 0x000F000Fu) | 0x64006400u;
  const uint p26 = ((q.x >> 8u) & 0x000F000Fu) | 0x64006400u;
  const uint p37 = ((q.x >> 12u) & 0x000F000Fu) | 0x64006400u;
  const half2 h04 = as_type<half2>(p04) - k1024;
  const half2 h15 = as_type<half2>(p15) - k1024;
  const half2 h26 = as_type<half2>(p26) - k1024;
  const half2 h37 = as_type<half2>(p37) - k1024;
  w0 = half4(h04.x, h15.x, h26.x, h37.x);
  w1 = half4(h04.y, h15.y, h26.y, h37.y);

  const uint r04 = (q.y & 0x000F000Fu) | 0x64006400u;
  const uint r15 = ((q.y >> 4u) & 0x000F000Fu) | 0x64006400u;
  const uint r26 = ((q.y >> 8u) & 0x000F000Fu) | 0x64006400u;
  const uint r37 = ((q.y >> 12u) & 0x000F000Fu) | 0x64006400u;
  const half2 g04 = as_type<half2>(r04) - k1024;
  const half2 g15 = as_type<half2>(r15) - k1024;
  const half2 g26 = as_type<half2>(r26) - k1024;
  const half2 g37 = as_type<half2>(r37) - k1024;
  w2 = half4(g04.x, g15.x, g26.x, g37.x);
  w3 = half4(g04.y, g15.y, g26.y, g37.y);
}

template <int P_FP16>
inline void moe_load_x8(uint4 v, thread float4& lo, thread float4& hi) {
  if constexpr (P_FP16 != 0) {
    const half2 a = as_type<half2>(v.x);
    const half2 b = as_type<half2>(v.y);
    const half2 c = as_type<half2>(v.z);
    const half2 d = as_type<half2>(v.w);
    lo = float4(float(a.x), float(a.y), float(b.x), float(b.y));
    hi = float4(float(c.x), float(c.y), float(d.x), float(d.y));
  } else {
    lo = float4(
        as_type<float>(v.x << 16u),
        as_type<float>(v.x & 0xFFFF0000u),
        as_type<float>(v.y << 16u),
        as_type<float>(v.y & 0xFFFF0000u));
    hi = float4(
        as_type<float>(v.z << 16u),
        as_type<float>(v.z & 0xFFFF0000u),
        as_type<float>(v.w << 16u),
        as_type<float>(v.w & 0xFFFF0000u));
  }
}

template <int P_FP16>
inline void moe_load_x8_sum(uint4 v, thread float4& lo, thread float4& hi, thread float& xsum) {
  moe_load_x8<P_FP16>(v, lo, hi);
  const float4 s4 = lo + hi;
  xsum = (s4.x + s4.y) + (s4.z + s4.w);
}

template <int P_FP16>
inline float4 moe_load_x4(uint2 v) {
  if constexpr (P_FP16 != 0) {
    const half2 a = as_type<half2>(v.x);
    const half2 b = as_type<half2>(v.y);
    return float4(float(a.x), float(a.y), float(b.x), float(b.y));
  } else {
    return float4(
        as_type<float>(v.x << 16u),
        as_type<float>(v.x & 0xFFFF0000u),
        as_type<float>(v.y << 16u),
        as_type<float>(v.y & 0xFFFF0000u));
  }
}

template <int P_SOFTCAP, int P_GCAP_Q12, int P_UCAP_Q12>
inline float moe_swiglu(float g, float u) {
  const float sig = 1.0f / (1.0f + metal::exp(-clamp(g, -80.0f, 80.0f)));
  if constexpr (P_SOFTCAP != 0) {
    const float gcap = float(P_GCAP_Q12) * (1.0f / 4096.0f);
    const float ucap = float(P_UCAP_Q12) * (1.0f / 4096.0f);
    const float og = metal::tanh(clamp(g / gcap, -10.0f, 10.0f)) * gcap;
    const float ou = metal::tanh(clamp(u / ucap, -10.0f, 10.0f)) * ucap;
    return (og * sig) * ou;
  } else {
    return (g * sig) * u;
  }
}

inline float moe_unpack_dot(uint2 q, float4 p0, float4 p1, float4 p2, float4 p3) {
  half4 w0, w1, w2, w3;
  moe_unpack_u2_half4(q, w0, w1, w2, w3);
  return (dot(p0, float4(w0)) + dot(p1, float4(w1))) + (dot(p2, float4(w2)) + dot(p3, float4(w3)));
}
"""

_PLAN_DUAL_SOURCE = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    constexpr uint NSG = uint(THREADS) / 32u;

    threadgroup uint tg_inds[N_PAIRS];
    for (uint p = tid; p < uint(N_PAIRS); p += uint(THREADS)) {
      tg_inds[p] = uint(inds[p]);
    }
    threadgroup uint sg_tot1[8];
    threadgroup uint sg_tot2[8];
    threadgroup uint tg_off1;
    threadgroup uint tg_off2;
    if (tid == 0u) {
      tg_off1 = 0u;
      tg_off2 = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint base_e = 0u; base_e < uint(E); base_e += uint(THREADS)) {
      const uint e = base_e + tid;
      uint cnt = 0u;
      if (e < uint(E)) {
        for (uint p = 0u; p < uint(N_PAIRS); ++p) {
          if (tg_inds[p] == e) {
            cnt += 1u;
          }
        }
      }

      const uint n_b2 = cnt >> 1u;
      const uint n_b1 = cnt & 1u;

      const uint p1 = simd_prefix_exclusive_sum(n_b1);
      const uint p2 = simd_prefix_exclusive_sum(n_b2);
      if (lane == 31u) {
        sg_tot1[sg] = p1 + n_b1;
        sg_tot2[sg] = p2 + n_b2;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (tid == 0u) {
        uint r1 = tg_off1;
        uint r2 = tg_off2;
        for (uint s = 0u; s < NSG; ++s) {
          const uint v1 = sg_tot1[s];
          const uint v2 = sg_tot2[s];
          sg_tot1[s] = r1;
          sg_tot2[s] = r2;
          r1 += v1;
          r2 += v2;
        }
        tg_off1 = r1;
        tg_off2 = r2;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (cnt > 0u) {
        const uint u1 = sg_tot1[sg] + p1;
        uint u2 = sg_tot2[sg] + p2;
        const uint b2_limit = cnt - n_b1;
        uint seen = 0u;
        uint first_tok = 0u;
        uint first_pair = 0u;
        for (uint p = 0u; p < uint(N_PAIRS); ++p) {
          if (tg_inds[p] == e) {
            const uint tok = p / uint(K_TOP);
            if (seen < b2_limit) {
              if ((seen & 1u) == 0u) {
                first_tok = tok;
                first_pair = p;
              } else {
                b2_exp[u2] = e;
                b2_tok0[u2] = first_tok;
                b2_tok1[u2] = tok;
                b2_pair0[u2] = first_pair;
                b2_pair1[u2] = p;
                u2 += 1u;
              }
              pair_is_b2[p] = uint8_t(1);
            } else {
              b1_exp[u1] = e;
              b1_tok[u1] = tok;
              b1_pair[u1] = p;
              pair_is_b2[p] = uint8_t(0);
            }
            seen += 1u;
          }
        }
      }
      if constexpr (uint(E) > uint(THREADS)) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }

    if (tid == 0u) {
      b1_num[0] = tg_off1;
      b2_num[0] = tg_off2;
    }
"""

_W13_C1_SOURCE = r"""
    const uint u = threadgroup_position_in_grid.y;
    size_t eid, tok, pair;
    if constexpr (DIRECT_M1 != 0) {
      if (u >= uint(K_TOP)) {
        return;
      }
      eid = size_t(b1_exp[u]);
      tok = 0u;
      pair = size_t(u);
    } else {
      if (u >= b1_num[0]) {
        return;
      }
      eid = size_t(b1_exp[u]);
      tok = size_t(b1_tok[u]);
      pair = size_t(b1_pair[u]);
    }
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint r_base = threadgroup_position_in_grid.x * (uint(NSG) * uint(RPSG)) + sg * uint(RPSG);
    if (r_base >= uint(INTER)) {
      return;
    }

    constexpr uint KU = uint(K);
    constexpr uint IU = uint(INTER);
    constexpr uint NWU = uint(N_W);
    constexpr uint WPR = KU >> 3u;
    constexpr uint N_GRP = KU >> 6u;
    constexpr uint K_FULL = (KU / 512u) * 512u;
    constexpr uint TAIL_LANES = (KU - K_FULL) >> 4u;

    const device T* xp = X + tok * KU + lane * 16u;
    const uint r_eff = min(r_base, IU - 1u);
    const uint g_lane = lane >> 2u;
    const uint w_lane = lane * 2u;

    const size_t rg0 = eid * NWU + size_t(r_eff) * ((SEPARATE_GU != 0) ? 1u : uint(ROW_STRIDE));
    const device uint* w0 = Wg + rg0 * WPR + w_lane;
    const device TS* s0 = Sg + rg0 * N_GRP + g_lane;
    const device TS* b0 = Bg + rg0 * N_GRP + g_lane;
    const device uint* w0u = Wu + rg0 * WPR + w_lane;
    const device TS* s0u = Su + rg0 * N_GRP + g_lane;
    const device TS* b0u = Bu + rg0 * N_GRP + g_lane;

    float ag[RPSG];
    float au[RPSG];
    #pragma unroll
    for (uint rr = 0u; rr < uint(RPSG); ++rr) {
      ag[rr] = 0.0f;
      au[rr] = 0.0f;
    }

    for (uint k = 0u; k < K_FULL; k += 512u) {
      const uint kw = k >> 3u;
      const uint kg = k >> 6u;
      const device uint4* x4 = (const device uint4*)(xp + k);
      float4 x0, x1, x2, x3;
      float xs0, xs1;
      moe_load_x8_sum<IS_FP16>(x4[0], x0, x1, xs0);
      moe_load_x8_sum<IS_FP16>(x4[1], x2, x3, xs1);
      const float xsum = xs0 + xs1;

      #pragma unroll
      for (uint rr = 0u; rr < uint(RPSG); ++rr) {
        if constexpr (SEPARATE_GU != 0) {
          const uint ro = (r_base + rr < IU) ? rr : 0u;
          const size_t woff = size_t(ro) * WPR + kw;
          const size_t soff = size_t(ro) * N_GRP + kg;
          const float dg = moe_unpack_dot(*(const device uint2*)(w0 + woff), x0, x1, x2, x3);
          ag[rr] = fma(dg, float(s0[soff]), fma(xsum, float(b0[soff]), ag[rr]));
          const float du = moe_unpack_dot(*(const device uint2*)(w0u + woff), x0, x1, x2, x3);
          au[rr] = fma(du, float(s0u[soff]), fma(xsum, float(b0u[soff]), au[rr]));
        } else {
          const uint ro = (r_base + rr < IU) ? (rr * uint(ROW_STRIDE)) : 0u;
          const size_t wg_off = size_t(ro) * WPR + kw;
          const size_t sg_off = size_t(ro) * N_GRP + kg;
          const size_t wu_off = size_t(ro + uint(UP_ROW_OFF)) * WPR + kw;
          const size_t su_off = size_t(ro + uint(UP_ROW_OFF)) * N_GRP + kg;
          const float dg = moe_unpack_dot(*(const device uint2*)(w0 + wg_off), x0, x1, x2, x3);
          ag[rr] = fma(dg, float(s0[sg_off]), fma(xsum, float(b0[sg_off]), ag[rr]));
          const float du = moe_unpack_dot(*(const device uint2*)(w0 + wu_off), x0, x1, x2, x3);
          au[rr] = fma(du, float(s0[su_off]), fma(xsum, float(b0[su_off]), au[rr]));
        }
      }
    }

    if constexpr (TAIL_LANES > 0u) {
      if (lane < TAIL_LANES) {
        constexpr uint kw = K_FULL >> 3u;
        constexpr uint kg = K_FULL >> 6u;
        const device uint4* x4 = (const device uint4*)(xp + K_FULL);
        float4 x0, x1, x2, x3;
        float xs0, xs1;
        moe_load_x8_sum<IS_FP16>(x4[0], x0, x1, xs0);
        moe_load_x8_sum<IS_FP16>(x4[1], x2, x3, xs1);
        const float xsum = xs0 + xs1;

        #pragma unroll
        for (uint rr = 0u; rr < uint(RPSG); ++rr) {
          if constexpr (SEPARATE_GU != 0) {
            const uint ro = (r_base + rr < IU) ? rr : 0u;
            const size_t woff = size_t(ro) * WPR + kw;
            const size_t soff = size_t(ro) * N_GRP + kg;
            const float dg = moe_unpack_dot(*(const device uint2*)(w0 + woff), x0, x1, x2, x3);
            ag[rr] = fma(dg, float(s0[soff]), fma(xsum, float(b0[soff]), ag[rr]));
            const float du = moe_unpack_dot(*(const device uint2*)(w0u + woff), x0, x1, x2, x3);
            au[rr] = fma(du, float(s0u[soff]), fma(xsum, float(b0u[soff]), au[rr]));
          } else {
            const uint ro = (r_base + rr < IU) ? (rr * uint(ROW_STRIDE)) : 0u;
            const size_t wg_off = size_t(ro) * WPR + kw;
            const size_t sg_off = size_t(ro) * N_GRP + kg;
            const size_t wu_off = size_t(ro + uint(UP_ROW_OFF)) * WPR + kw;
            const size_t su_off = size_t(ro + uint(UP_ROW_OFF)) * N_GRP + kg;
            const float dg = moe_unpack_dot(*(const device uint2*)(w0 + wg_off), x0, x1, x2, x3);
            ag[rr] = fma(dg, float(s0[sg_off]), fma(xsum, float(b0[sg_off]), ag[rr]));
            const float du = moe_unpack_dot(*(const device uint2*)(w0 + wu_off), x0, x1, x2, x3);
            au[rr] = fma(du, float(s0[su_off]), fma(xsum, float(b0[su_off]), au[rr]));
          }
        }
      }
    }

    #pragma unroll
    for (uint rr = 0u; rr < uint(RPSG); ++rr) {
      const float g = simd_sum(ag[rr]);
      const float uv = simd_sum(au[rr]);
      const uint r = r_base + rr;
      if (lane == 0u && r < IU) {
        ACT[pair * IU + r] = T(moe_swiglu<SOFTCAP, GCAP_Q12, UCAP_Q12>(g, uv));
      }
    }
"""

_W13_CM_SOURCE = r"""
    const uint u = threadgroup_position_in_grid.y;
    if (u >= b2_num[0]) {
      return;
    }
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint r_base = threadgroup_position_in_grid.x * (uint(NSG) * uint(RPSG)) + sg * uint(RPSG);
    if (r_base >= uint(INTER)) {
      return;
    }

    constexpr uint KU = uint(K);
    constexpr uint IU = uint(INTER);
    constexpr uint NWU = uint(N_W);
    constexpr uint WPR = KU >> 3u;
    constexpr uint N_GRP = KU >> 6u;
    constexpr uint K_FULL = (KU / 512u) * 512u;
    constexpr uint TAIL_LANES = (KU - K_FULL) >> 4u;

    const size_t eid = size_t(b2_exp[u]);
    const size_t tok0 = size_t(b2_tok0[u]);
    const size_t tok1 = size_t(b2_tok1[u]);
    const size_t p0_idx = size_t(b2_pair0[u]);
    const size_t p1_idx = size_t(b2_pair1[u]);
    const device T* xp0 = X + tok0 * KU + lane * 16u;
    const device T* xp1 = X + tok1 * KU + lane * 16u;

    const uint r_eff = min(r_base, IU - 1u);
    const uint g_lane = lane >> 2u;
    const uint w_lane = lane * 2u;
    const size_t rg0 = eid * NWU + size_t(r_eff) * ((SEPARATE_GU != 0) ? 1u : uint(ROW_STRIDE));
    const device uint* w0 = Wg + rg0 * WPR + w_lane;
    const device TS* s0 = Sg + rg0 * N_GRP + g_lane;
    const device TS* b0 = Bg + rg0 * N_GRP + g_lane;
    const device uint* w0u = Wu + rg0 * WPR + w_lane;
    const device TS* s0u = Su + rg0 * N_GRP + g_lane;
    const device TS* b0u = Bu + rg0 * N_GRP + g_lane;

    float ag0[RPSG], ag1[RPSG], au0[RPSG], au1[RPSG];
    #pragma unroll
    for (uint rr = 0u; rr < uint(RPSG); ++rr) {
      ag0[rr] = 0.0f;
      ag1[rr] = 0.0f;
      au0[rr] = 0.0f;
      au1[rr] = 0.0f;
    }

    for (uint k = 0u; k < K_FULL; k += 512u) {
      const uint kw = k >> 3u;
      const uint kg = k >> 6u;
      uint2 wgv[RPSG], wuv[RPSG];
      float sgv[RPSG], bgv[RPSG], suv[RPSG], buv[RPSG];
      #pragma unroll
      for (uint rr = 0u; rr < uint(RPSG); ++rr) {
        if constexpr (SEPARATE_GU != 0) {
          const uint ro = (r_base + rr < IU) ? rr : 0u;
          const size_t woff = size_t(ro) * WPR + kw;
          const size_t soff = size_t(ro) * N_GRP + kg;
          wgv[rr] = *(const device uint2*)(w0 + woff);
          sgv[rr] = float(s0[soff]);
          bgv[rr] = float(b0[soff]);
          wuv[rr] = *(const device uint2*)(w0u + woff);
          suv[rr] = float(s0u[soff]);
          buv[rr] = float(b0u[soff]);
        } else {
          const uint ro = (r_base + rr < IU) ? (rr * uint(ROW_STRIDE)) : 0u;
          const size_t wg_off = size_t(ro) * WPR + kw;
          const size_t sg_off = size_t(ro) * N_GRP + kg;
          const size_t wu_off = size_t(ro + uint(UP_ROW_OFF)) * WPR + kw;
          const size_t su_off = size_t(ro + uint(UP_ROW_OFF)) * N_GRP + kg;
          wgv[rr] = *(const device uint2*)(w0 + wg_off);
          sgv[rr] = float(s0[sg_off]);
          bgv[rr] = float(b0[sg_off]);
          wuv[rr] = *(const device uint2*)(w0 + wu_off);
          suv[rr] = float(s0[su_off]);
          buv[rr] = float(b0[su_off]);
        }
      }
      {
        const device uint4* x4 = (const device uint4*)(xp0 + k);
        float4 p0, p1, p2, p3;
        float xs0, xs1;
        moe_load_x8_sum<IS_FP16>(x4[0], p0, p1, xs0);
        moe_load_x8_sum<IS_FP16>(x4[1], p2, p3, xs1);
        const float xs = xs0 + xs1;
        #pragma unroll
        for (uint rr = 0u; rr < uint(RPSG); ++rr) {
          ag0[rr] = fma(moe_unpack_dot(wgv[rr], p0, p1, p2, p3), sgv[rr], fma(xs, bgv[rr], ag0[rr]));
          au0[rr] = fma(moe_unpack_dot(wuv[rr], p0, p1, p2, p3), suv[rr], fma(xs, buv[rr], au0[rr]));
        }
      }
      {
        const device uint4* x4 = (const device uint4*)(xp1 + k);
        float4 p0, p1, p2, p3;
        float xs0, xs1;
        moe_load_x8_sum<IS_FP16>(x4[0], p0, p1, xs0);
        moe_load_x8_sum<IS_FP16>(x4[1], p2, p3, xs1);
        const float xs = xs0 + xs1;
        #pragma unroll
        for (uint rr = 0u; rr < uint(RPSG); ++rr) {
          ag1[rr] = fma(moe_unpack_dot(wgv[rr], p0, p1, p2, p3), sgv[rr], fma(xs, bgv[rr], ag1[rr]));
          au1[rr] = fma(moe_unpack_dot(wuv[rr], p0, p1, p2, p3), suv[rr], fma(xs, buv[rr], au1[rr]));
        }
      }
    }

    if constexpr (TAIL_LANES > 0u) {
      if (lane < TAIL_LANES) {
        constexpr uint kw = K_FULL >> 3u;
        constexpr uint kg = K_FULL >> 6u;
        uint2 wgv[RPSG], wuv[RPSG];
        float sgv[RPSG], bgv[RPSG], suv[RPSG], buv[RPSG];
        #pragma unroll
        for (uint rr = 0u; rr < uint(RPSG); ++rr) {
          if constexpr (SEPARATE_GU != 0) {
            const uint ro = (r_base + rr < IU) ? rr : 0u;
            const size_t woff = size_t(ro) * WPR + kw;
            const size_t soff = size_t(ro) * N_GRP + kg;
            wgv[rr] = *(const device uint2*)(w0 + woff);
            sgv[rr] = float(s0[soff]);
            bgv[rr] = float(b0[soff]);
            wuv[rr] = *(const device uint2*)(w0u + woff);
            suv[rr] = float(s0u[soff]);
            buv[rr] = float(b0u[soff]);
          } else {
            const uint ro = (r_base + rr < IU) ? (rr * uint(ROW_STRIDE)) : 0u;
            const size_t wg_off = size_t(ro) * WPR + kw;
            const size_t sg_off = size_t(ro) * N_GRP + kg;
            const size_t wu_off = size_t(ro + uint(UP_ROW_OFF)) * WPR + kw;
            const size_t su_off = size_t(ro + uint(UP_ROW_OFF)) * N_GRP + kg;
            wgv[rr] = *(const device uint2*)(w0 + wg_off);
            sgv[rr] = float(s0[sg_off]);
            bgv[rr] = float(b0[sg_off]);
            wuv[rr] = *(const device uint2*)(w0 + wu_off);
            suv[rr] = float(s0[su_off]);
            buv[rr] = float(b0[su_off]);
          }
        }
        {
          const device uint4* x4 = (const device uint4*)(xp0 + K_FULL);
          float4 p0, p1, p2, p3;
          float xs0, xs1;
          moe_load_x8_sum<IS_FP16>(x4[0], p0, p1, xs0);
          moe_load_x8_sum<IS_FP16>(x4[1], p2, p3, xs1);
          const float xs = xs0 + xs1;
          #pragma unroll
          for (uint rr = 0u; rr < uint(RPSG); ++rr) {
            ag0[rr] = fma(moe_unpack_dot(wgv[rr], p0, p1, p2, p3), sgv[rr], fma(xs, bgv[rr], ag0[rr]));
            au0[rr] = fma(moe_unpack_dot(wuv[rr], p0, p1, p2, p3), suv[rr], fma(xs, buv[rr], au0[rr]));
          }
        }
        {
          const device uint4* x4 = (const device uint4*)(xp1 + K_FULL);
          float4 p0, p1, p2, p3;
          float xs0, xs1;
          moe_load_x8_sum<IS_FP16>(x4[0], p0, p1, xs0);
          moe_load_x8_sum<IS_FP16>(x4[1], p2, p3, xs1);
          const float xs = xs0 + xs1;
          #pragma unroll
          for (uint rr = 0u; rr < uint(RPSG); ++rr) {
            ag1[rr] = fma(moe_unpack_dot(wgv[rr], p0, p1, p2, p3), sgv[rr], fma(xs, bgv[rr], ag1[rr]));
            au1[rr] = fma(moe_unpack_dot(wuv[rr], p0, p1, p2, p3), suv[rr], fma(xs, buv[rr], au1[rr]));
          }
        }
      }
    }

    #pragma unroll
    for (uint rr = 0u; rr < uint(RPSG); ++rr) {
      const float g0 = simd_sum(ag0[rr]);
      const float u0 = simd_sum(au0[rr]);
      const float g1 = simd_sum(ag1[rr]);
      const float u1 = simd_sum(au1[rr]);
      const uint r = r_base + rr;
      if (lane == 0u && r < IU) {
        ACT[p0_idx * IU + r] = T(moe_swiglu<SOFTCAP, GCAP_Q12, UCAP_Q12>(g0, u0));
        ACT[p1_idx * IU + r] = T(moe_swiglu<SOFTCAP, GCAP_Q12, UCAP_Q12>(g1, u1));
      }
    }
"""

_W2_C1_SOURCE = r"""
    const uint u = threadgroup_position_in_grid.y;
    size_t eid, pair;
    if constexpr (DIRECT_M1 != 0) {
      if (u >= uint(K_TOP)) {
        return;
      }
      eid = size_t(b1_exp[u]);
      pair = size_t(u);
    } else {
      if (u >= b1_num[0]) {
        return;
      }
      eid = size_t(b1_exp[u]);
      pair = size_t(b1_pair[u]);
    }
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint row_base = threadgroup_position_in_grid.x * (uint(NSG) * uint(RPSG)) + sg * uint(RPSG);
    if (row_base >= uint(N)) {
      return;
    }

    constexpr uint KU = uint(K);
    constexpr uint NU = uint(N);
    constexpr uint WPR = KU >> 3u;
    constexpr uint N_GRP = KU >> 6u;
    constexpr uint K_FULL = (KU / 512u) * 512u;
    constexpr uint TAIL_LANES = (KU - K_FULL) >> 4u;

    const device T* xp = ACT + pair * KU + lane * 16u;
    const size_t r0 = eid * NU + size_t(min(row_base, NU - 1u));
    const device uint* w0 = W2 + r0 * WPR + lane * 2u;
    const device TS* s0 = S2 + r0 * N_GRP + (lane >> 2u);
    const device TS* b0 = B2 + r0 * N_GRP + (lane >> 2u);

    float acc[RPSG];
    #pragma unroll
    for (uint rr = 0u; rr < uint(RPSG); ++rr) {
      acc[rr] = 0.0f;
    }

    for (uint k = 0u; k < K_FULL; k += 512u) {
      const uint kw = k >> 3u;
      const uint kg = k >> 6u;
      const device uint4* x4 = (const device uint4*)(xp + k);
      float4 x0, x1, x2, x3;
      float xs0, xs1;
      moe_load_x8_sum<IS_FP16>(x4[0], x0, x1, xs0);
      moe_load_x8_sum<IS_FP16>(x4[1], x2, x3, xs1);
      const float xsum = xs0 + xs1;

      #pragma unroll
      for (uint rr = 0u; rr < uint(RPSG); ++rr) {
        const uint ro = (row_base + rr < NU) ? rr : 0u;
        const size_t wo = size_t(ro) * WPR + kw;
        const size_t so = size_t(ro) * N_GRP + kg;
        const float dw = moe_unpack_dot(*(const device uint2*)(w0 + wo), x0, x1, x2, x3);
        acc[rr] = fma(dw, float(s0[so]), fma(xsum, float(b0[so]), acc[rr]));
      }
    }

    if constexpr (TAIL_LANES > 0u) {
      if (lane < TAIL_LANES) {
        constexpr uint kw = K_FULL >> 3u;
        constexpr uint kg = K_FULL >> 6u;
        const device uint4* x4 = (const device uint4*)(xp + K_FULL);
        float4 x0, x1, x2, x3;
        float xs0, xs1;
        moe_load_x8_sum<IS_FP16>(x4[0], x0, x1, xs0);
        moe_load_x8_sum<IS_FP16>(x4[1], x2, x3, xs1);
        const float xsum = xs0 + xs1;

        #pragma unroll
        for (uint rr = 0u; rr < uint(RPSG); ++rr) {
          const uint ro = (row_base + rr < NU) ? rr : 0u;
          const size_t wo = size_t(ro) * WPR + kw;
          const size_t so = size_t(ro) * N_GRP + kg;
          const float dw = moe_unpack_dot(*(const device uint2*)(w0 + wo), x0, x1, x2, x3);
          acc[rr] = fma(dw, float(s0[so]), fma(xsum, float(b0[so]), acc[rr]));
        }
      }
    }

    const float rw = WEIGHTS[pair];
    #pragma unroll
    for (uint rr = 0u; rr < uint(RPSG); ++rr) {
      const float v = simd_sum(acc[rr]);
      const uint row = row_base + rr;
      if (lane == 0u && row < NU) {
        Z[pair * NU + row] = T(v * rw);
      }
    }
"""

_W2_CM_SOURCE = r"""
    const uint u = threadgroup_position_in_grid.y;
    if (u >= b2_num[0]) {
      return;
    }
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint row_base = threadgroup_position_in_grid.x * (uint(NSG) * uint(RPSG)) + sg * uint(RPSG);
    if (row_base >= uint(N)) {
      return;
    }

    constexpr uint KU = uint(K);
    constexpr uint NU = uint(N);
    constexpr uint WPR = KU >> 3u;
    constexpr uint N_GRP = KU >> 6u;
    constexpr uint K_FULL = (KU / 512u) * 512u;
    constexpr uint TAIL_LANES = (KU - K_FULL) >> 4u;

    const size_t eid = size_t(b2_exp[u]);
    const size_t p0_idx = size_t(b2_pair0[u]);
    const size_t p1_idx = size_t(b2_pair1[u]);
    const device T* xp0 = ACT + p0_idx * KU + lane * 16u;
    const device T* xp1 = ACT + p1_idx * KU + lane * 16u;
    const size_t r0 = eid * NU + size_t(min(row_base, NU - 1u));
    const device uint* w0 = W2 + r0 * WPR + lane * 2u;
    const device TS* s0 = S2 + r0 * N_GRP + (lane >> 2u);
    const device TS* b0 = B2 + r0 * N_GRP + (lane >> 2u);

    float acc0[RPSG], acc1[RPSG];
    #pragma unroll
    for (uint rr = 0u; rr < uint(RPSG); ++rr) {
      acc0[rr] = 0.0f;
      acc1[rr] = 0.0f;
    }

    for (uint k = 0u; k < K_FULL; k += 512u) {
      const uint kw = k >> 3u;
      const uint kg = k >> 6u;
      uint2 wv[RPSG];
      float sv[RPSG], bv[RPSG];
      #pragma unroll
      for (uint rr = 0u; rr < uint(RPSG); ++rr) {
        const uint ro = (row_base + rr < NU) ? rr : 0u;
        wv[rr] = *(const device uint2*)(w0 + size_t(ro) * WPR + kw);
        sv[rr] = float(s0[size_t(ro) * N_GRP + kg]);
        bv[rr] = float(b0[size_t(ro) * N_GRP + kg]);
      }
      {
        const device uint4* x4 = (const device uint4*)(xp0 + k);
        float4 p0, p1, p2, p3;
        float xs0, xs1;
        moe_load_x8_sum<IS_FP16>(x4[0], p0, p1, xs0);
        moe_load_x8_sum<IS_FP16>(x4[1], p2, p3, xs1);
        const float xs = xs0 + xs1;
        #pragma unroll
        for (uint rr = 0u; rr < uint(RPSG); ++rr) {
          acc0[rr] = fma(moe_unpack_dot(wv[rr], p0, p1, p2, p3), sv[rr], fma(xs, bv[rr], acc0[rr]));
        }
      }
      {
        const device uint4* x4 = (const device uint4*)(xp1 + k);
        float4 p0, p1, p2, p3;
        float xs0, xs1;
        moe_load_x8_sum<IS_FP16>(x4[0], p0, p1, xs0);
        moe_load_x8_sum<IS_FP16>(x4[1], p2, p3, xs1);
        const float xs = xs0 + xs1;
        #pragma unroll
        for (uint rr = 0u; rr < uint(RPSG); ++rr) {
          acc1[rr] = fma(moe_unpack_dot(wv[rr], p0, p1, p2, p3), sv[rr], fma(xs, bv[rr], acc1[rr]));
        }
      }
    }

    if constexpr (TAIL_LANES > 0u) {
      if (lane < TAIL_LANES) {
        constexpr uint kw = K_FULL >> 3u;
        constexpr uint kg = K_FULL >> 6u;
        uint2 wv[RPSG];
        float sv[RPSG], bv[RPSG];
        #pragma unroll
        for (uint rr = 0u; rr < uint(RPSG); ++rr) {
          const uint ro = (row_base + rr < NU) ? rr : 0u;
          wv[rr] = *(const device uint2*)(w0 + size_t(ro) * WPR + kw);
          sv[rr] = float(s0[size_t(ro) * N_GRP + kg]);
          bv[rr] = float(b0[size_t(ro) * N_GRP + kg]);
        }
        {
          const device uint4* x4 = (const device uint4*)(xp0 + K_FULL);
          float4 p0, p1, p2, p3;
          float xs0, xs1;
          moe_load_x8_sum<IS_FP16>(x4[0], p0, p1, xs0);
          moe_load_x8_sum<IS_FP16>(x4[1], p2, p3, xs1);
          const float xs = xs0 + xs1;
          #pragma unroll
          for (uint rr = 0u; rr < uint(RPSG); ++rr) {
            acc0[rr] = fma(moe_unpack_dot(wv[rr], p0, p1, p2, p3), sv[rr], fma(xs, bv[rr], acc0[rr]));
          }
        }
        {
          const device uint4* x4 = (const device uint4*)(xp1 + K_FULL);
          float4 p0, p1, p2, p3;
          float xs0, xs1;
          moe_load_x8_sum<IS_FP16>(x4[0], p0, p1, xs0);
          moe_load_x8_sum<IS_FP16>(x4[1], p2, p3, xs1);
          const float xs = xs0 + xs1;
          #pragma unroll
          for (uint rr = 0u; rr < uint(RPSG); ++rr) {
            acc1[rr] = fma(moe_unpack_dot(wv[rr], p0, p1, p2, p3), sv[rr], fma(xs, bv[rr], acc1[rr]));
          }
        }
      }
    }

    const float rw0 = WEIGHTS[p0_idx];
    const float rw1 = WEIGHTS[p1_idx];
    #pragma unroll
    for (uint rr = 0u; rr < uint(RPSG); ++rr) {
      const float v0 = simd_sum(acc0[rr]);
      const float v1 = simd_sum(acc1[rr]);
      const uint row = row_base + rr;
      if (lane == 0u && row < NU) {
        Z[p0_idx * NU + row] = T(v0 * rw0);
        Z[p1_idx * NU + row] = T(v1 * rw1);
      }
    }
"""

_COMBINE_VEC_SOURCE = r"""
    const uint m = threadgroup_position_in_grid.y;
    const uint idx = threadgroup_position_in_grid.x * 256u + thread_position_in_threadgroup.x;
    const uint h = idx * uint(VEC);
    if (h >= uint(H_DIM)) {
      return;
    }

    const uint pair_base = m * uint(K_TOP);
    float sg_scale = 1.0f;
    if constexpr (HAS_SHARED == 2) {
      const float g = float(shared_gate[m]);
      sg_scale = 1.0f / (1.0f + metal::exp(-clamp(g, -80.0f, 80.0f)));
    }

    if constexpr (VEC == 8) {
      float4 acc0 = float4(0.0f);
      float4 acc1 = float4(0.0f);
      for (uint k = 0u; k < uint(K_TOP); ++k) {
        const uint pair = pair_base + k;
        const device T* src = (ONLY_B1 == 0 && pair_is_b2[pair] != 0u) ? Z2 : Z1;
        const device uint4* zp = (const device uint4*)(src + size_t(pair) * uint(H_DIM) + h);
        float4 v0, v1;
        moe_load_x8<IS_FP16>(*zp, v0, v1);
        acc0 += v0;
        acc1 += v1;
      }
      if constexpr (HAS_SHARED != 0) {
        const device TSH* sp = shared_out + size_t(m) * uint(H_DIM) + h;
        const float4 s0 = float4(float(sp[0]), float(sp[1]), float(sp[2]), float(sp[3]));
        const float4 s1 = float4(float(sp[4]), float(sp[5]), float(sp[6]), float(sp[7]));
        acc0 = (HAS_SHARED == 2) ? fma(s0, float4(sg_scale), acc0) : (acc0 + s0);
        acc1 = (HAS_SHARED == 2) ? fma(s1, float4(sg_scale), acc1) : (acc1 + s1);
      }
      device vec<T, 4>* op = (device vec<T, 4>*)(OUT + size_t(m) * uint(H_DIM) + h);
      op[0] = vec<T, 4>(acc0);
      op[1] = vec<T, 4>(acc1);
    } else if constexpr (VEC == 4) {
      float4 acc = float4(0.0f);
      for (uint k = 0u; k < uint(K_TOP); ++k) {
        const uint pair = pair_base + k;
        const device T* src = (ONLY_B1 == 0 && pair_is_b2[pair] != 0u) ? Z2 : Z1;
        const device uint2* zp = (const device uint2*)(src + size_t(pair) * uint(H_DIM) + h);
        acc += moe_load_x4<IS_FP16>(*zp);
      }
      if constexpr (HAS_SHARED != 0) {
        const device TSH* sp = shared_out + size_t(m) * uint(H_DIM) + h;
        const float4 sv = float4(float(sp[0]), float(sp[1]), float(sp[2]), float(sp[3]));
        acc = (HAS_SHARED == 2) ? fma(sv, float4(sg_scale), acc) : (acc + sv);
      }
      *(device vec<T, 4>*)(OUT + size_t(m) * uint(H_DIM) + h) = vec<T, 4>(acc);
    } else {
      float acc = 0.0f;
      for (uint k = 0u; k < uint(K_TOP); ++k) {
        const uint pair = pair_base + k;
        const device T* src = (ONLY_B1 == 0 && pair_is_b2[pair] != 0u) ? Z2 : Z1;
        acc += float(src[size_t(pair) * uint(H_DIM) + h]);
      }
      if constexpr (HAS_SHARED != 0) {
        const float sv = float(shared_out[size_t(m) * uint(H_DIM) + h]);
        acc = (HAS_SHARED == 2) ? fma(sv, sg_scale, acc) : (acc + sv);
      }
      OUT[size_t(m) * uint(H_DIM) + h] = T(acc);
    }
"""


@cache
def _plan_dual_kernel():
    return mx.fast.metal_kernel(
        name="omlx_m1_qmv_moe_plan_dual",
        input_names=["inds"],
        output_names=[
            "b1_exp",
            "b1_tok",
            "b1_pair",
            "b1_num",
            "b2_exp",
            "b2_tok0",
            "b2_tok1",
            "b2_pair0",
            "b2_pair1",
            "b2_num",
            "pair_is_b2",
        ],
        header=_COMMON_HEADER,
        source=_PLAN_DUAL_SOURCE,
    )


@cache
def _w13_c1_kernel():
    return mx.fast.metal_kernel(
        name="omlx_m1_qmv_moe_w13_c1",
        input_names=[
            "X",
            "Wg",
            "Sg",
            "Bg",
            "Wu",
            "Su",
            "Bu",
            "b1_exp",
            "b1_tok",
            "b1_pair",
            "b1_num",
        ],
        output_names=["ACT"],
        header=_COMMON_HEADER,
        source=_W13_C1_SOURCE,
    )


@cache
def _w13_cm_kernel():
    return mx.fast.metal_kernel(
        name="omlx_m1_qmv_moe_w13_cm",
        input_names=[
            "X",
            "Wg",
            "Sg",
            "Bg",
            "Wu",
            "Su",
            "Bu",
            "b2_exp",
            "b2_tok0",
            "b2_tok1",
            "b2_pair0",
            "b2_pair1",
            "b2_num",
        ],
        output_names=["ACT"],
        header=_COMMON_HEADER,
        source=_W13_CM_SOURCE,
    )


@cache
def _w2_c1_kernel():
    return mx.fast.metal_kernel(
        name="omlx_m1_qmv_moe_w2_c1",
        input_names=["ACT", "W2", "S2", "B2", "WEIGHTS", "b1_exp", "b1_pair", "b1_num"],
        output_names=["Z"],
        header=_COMMON_HEADER,
        source=_W2_C1_SOURCE,
    )


@cache
def _w2_cm_kernel():
    return mx.fast.metal_kernel(
        name="omlx_m1_qmv_moe_w2_cm",
        input_names=[
            "ACT",
            "W2",
            "S2",
            "B2",
            "WEIGHTS",
            "b2_exp",
            "b2_pair0",
            "b2_pair1",
            "b2_num",
        ],
        output_names=["Z"],
        header=_COMMON_HEADER,
        source=_W2_CM_SOURCE,
    )


@cache
def _combine_vec_kernel():
    return mx.fast.metal_kernel(
        name="omlx_m1_qmv_moe_combine_vec",
        input_names=["Z1", "Z2", "pair_is_b2", "shared_out", "shared_gate"],
        output_names=["OUT"],
        header=_COMMON_HEADER,
        source=_COMBINE_VEC_SOURCE,
    )


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _address(arr: mx.array) -> int:
    return np.frombuffer(memoryview(arr), dtype=np.uint8).ctypes.data


def _get_field(obj: Any, name: str, default: Any = None) -> Any:
    if hasattr(obj, "get"):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _first_expert_alias(arr: mx.array) -> mx.array:
    """Return ``arr[:1]`` when it shares ``arr``'s buffer at offset 0, else ``arr``."""
    view = arr[:1]
    mx.eval(arr, view)
    whole, first = memoryview(arr), memoryview(view)
    if whole.c_contiguous and first.c_contiguous and _address(view) == _address(arr):
        return view
    return arr


def _prepare_proj(proj: Any) -> None:
    """Pre-compute 1-expert alias views for one quantized projection outside mx.compile."""
    if proj is None or not hasattr(proj, "__dict__"):
        return
    w = _get_field(proj, "weight")
    s = _get_field(proj, "scales")
    b = _get_field(proj, "biases")
    if w is None or s is None or b is None:
        return
    cached = proj.__dict__.get("_omlx_m1_alias")
    src = proj.__dict__.get("_omlx_m1_alias_src")
    if cached is not None and src is not None and src[0] is w and src[1] is s and src[2] is b:
        return
    res = (_first_expert_alias(w), _first_expert_alias(s), _first_expert_alias(b))
    object.__setattr__(proj, "_omlx_m1_alias", res)
    object.__setattr__(proj, "_omlx_m1_alias_src", (w, s, b))


def prepare_m1_moe(experts: Any) -> None:
    """Pre-compute 1-expert alias views for Inkling2Experts outside mx.compile."""
    for name in ("w13", "w2"):
        proj = getattr(experts, name, None)
        if proj is not None:
            _prepare_proj(proj)


def prepare_switch_moe(switch_or_block: Any) -> None:
    """Pre-compute 1-expert alias views for SwitchGLU or Qwen3_5MoeSparseMoeBlock."""
    switch_mlp = getattr(switch_or_block, "switch_mlp", switch_or_block)
    for name in ("gate_up_proj", "gate_proj", "up_proj", "down_proj"):
        proj = getattr(switch_mlp, name, None)
        if proj is not None:
            _prepare_proj(proj)


def _expert_alias_views(proj: Any) -> tuple[mx.array, mx.array, mx.array]:
    """Return pre-cached 1-expert alias views without calling mx.eval inside mx.compile."""
    w, s, b = _get_field(proj, "weight"), _get_field(proj, "scales"), _get_field(proj, "biases")
    if hasattr(proj, "__dict__"):
        cached = proj.__dict__.get("_omlx_m1_alias")
        if cached is not None:
            src = proj.__dict__.get("_omlx_m1_alias_src")
            if src is None or (src[0] is w and src[1] is s and src[2] is b):
                return cached
    return (w, s, b)


def _proj_ok(proj: Any) -> bool:
    if proj is None:
        return False
    w = _get_field(proj, "weight")
    s = _get_field(proj, "scales")
    b = _get_field(proj, "biases")
    has_bias = _get_field(proj, "bias") is not None
    has_gs = _get_field(proj, "global_scale") is not None
    if w is None or s is None or b is None or has_bias or has_gs:
        return False
    return (
        getattr(proj, "mode", None) == "affine"
        and int(getattr(proj, "bits", 0)) == _BITS
        and int(getattr(proj, "group_size", 0)) == _GROUP_SIZE
        and w.ndim == 3
        and s.dtype in (mx.bfloat16, mx.float16)
        and b.dtype == s.dtype
    )


def eligible_inkling(x: mx.array, experts: Any) -> bool:
    """True when ``experts`` (Inkling2Experts) and ``x`` match this kernel."""
    if x.dtype not in (mx.bfloat16, mx.float16) or x.ndim not in (2, 3):
        return False
    m = x.shape[0] if x.ndim == 2 else x.shape[0] * x.shape[1]
    if not 1 <= m <= _MAX_M:
        return False
    h = int(x.shape[-1])
    w13 = getattr(experts, "w13", None)
    w2 = getattr(experts, "w2", None)
    if not (_proj_ok(w13) and _proj_ok(w2)):
        return False
    inter = int(_get_field(w2, "scales").shape[-1]) * _GROUP_SIZE
    return (
        h % _GROUP_SIZE == 0
        and inter % _GROUP_SIZE == 0
        and int(_get_field(w13, "weight").shape[1]) == 2 * inter
        and int(_get_field(w2, "weight").shape[1]) == h
    )


def eligible_switch(x: mx.array, switch_or_block: Any) -> bool:
    """True when ``switch_or_block`` (SwitchGLU or SparseMoeBlock) matches this kernel."""
    if x.dtype not in (mx.bfloat16, mx.float16) or x.ndim not in (2, 3):
        return False
    m = x.shape[0] if x.ndim == 2 else x.shape[0] * x.shape[1]
    if not 1 <= m <= _MAX_M:
        return False
    h = int(x.shape[-1])
    sw = getattr(switch_or_block, "switch_mlp", switch_or_block)
    down = getattr(sw, "down_proj", None)
    if not _proj_ok(down):
        return False
    inter = int(_get_field(down, "scales").shape[-1]) * _GROUP_SIZE
    if h % _GROUP_SIZE != 0 or inter % _GROUP_SIZE != 0 or int(_get_field(down, "weight").shape[1]) != h:
        return False
    gu = getattr(sw, "gate_up_proj", None)
    if gu is not None:
        return _proj_ok(gu) and int(_get_field(gu, "weight").shape[1]) == 2 * inter
    gp = getattr(sw, "gate_proj", None)
    up = getattr(sw, "up_proj", None)
    return (
        _proj_ok(gp)
        and _proj_ok(up)
        and int(_get_field(gp, "weight").shape[1]) == inter
        and int(_get_field(up, "weight").shape[1]) == inter
        and _get_field(gp, "scales").dtype == _get_field(up, "scales").dtype
    )


def m1_moe_plan_dual(inds: mx.array, num_experts: int, top_k: int):
    """Stage 0: split expert pairs into Bucket 1 (rem 1) and Bucket 2 (strict C==2)."""
    flat_inds = mx.contiguous(inds.reshape(-1))
    if flat_inds.dtype not in (mx.uint32, mx.int32):
        flat_inds = flat_inds.astype(mx.uint32)
    n_pairs = int(flat_inds.shape[0])
    max_b2 = max(1, n_pairs // 2)
    threads = 128 if num_experts <= 128 else 256
    return _plan_dual_kernel()(
        inputs=[flat_inds],
        template=[
            ("TI", flat_inds.dtype),
            ("E", int(num_experts)),
            ("N_PAIRS", n_pairs),
            ("K_TOP", int(top_k)),
            ("THREADS", threads),
        ],
        grid=(threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[
            (n_pairs,),
            (n_pairs,),
            (n_pairs,),
            (1,),
            (max_b2,),
            (max_b2,),
            (max_b2,),
            (max_b2,),
            (max_b2,),
            (1,),
            (n_pairs,),
        ],
        output_dtypes=[
            mx.uint32,
            mx.uint32,
            mx.uint32,
            mx.uint32,
            mx.uint32,
            mx.uint32,
            mx.uint32,
            mx.uint32,
            mx.uint32,
            mx.uint32,
            mx.uint8,
        ],
    )


def m1_moe_combine_vec(
    z1: mx.array,
    z2: mx.array,
    pair_is_b2: mx.array,
    m: int,
    top_k: int,
    hidden_dim: int,
    *,
    shared_out: mx.array | None = None,
    shared_gate: mx.array | None = None,
    vec: int = DEFAULT_COMBINE_VEC,
    only_b1: bool = False,
) -> mx.array:
    """Stage 3: vectorized deterministic FP32 sum over pre-weighted Z1/Z2 rows."""
    if shared_out is None:
        has_shared = 0
        sh_arr = mx.zeros((1,), dtype=z1.dtype)
        sg_arr = mx.zeros((1,), dtype=mx.float32)
    elif shared_gate is None:
        has_shared = 1
        sh_arr = mx.contiguous(shared_out.reshape(m, hidden_dim))
        sg_arr = mx.zeros((1,), dtype=mx.float32)
    else:
        has_shared = 2
        sh_arr = mx.contiguous(shared_out.reshape(m, hidden_dim))
        sg_arr = mx.contiguous(shared_gate.reshape(m))

    v = int(vec) if hidden_dim % int(vec) == 0 else (4 if hidden_dim % 4 == 0 else 1)
    n_vec = hidden_dim // v
    return _combine_vec_kernel()(
        inputs=[z1, z2, pair_is_b2, sh_arr, sg_arr],
        template=[
            ("T", z1.dtype),
            ("TSH", sh_arr.dtype),
            ("TSG", sg_arr.dtype),
            ("IS_FP16", 1 if z1.dtype == mx.float16 else 0),
            ("H_DIM", int(hidden_dim)),
            ("K_TOP", int(top_k)),
            ("HAS_SHARED", has_shared),
            ("VEC", v),
            ("ONLY_B1", 1 if only_b1 else 0),
        ],
        grid=(_ceil_div(n_vec, 256) * 256, int(m), 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(int(m), int(hidden_dim))],
        output_dtypes=[z1.dtype],
    )[0]


def _run_moe_pipeline(
    x: mx.array,
    inds: mx.array,
    weights: mx.array,
    wg_proj: Any,
    wu_proj: Any | None,
    w2_proj: Any,
    *,
    interleaved_w13: bool,
    softcap: bool,
    gcap: float,
    ucap: float,
    shared_out: mx.array | None = None,
    shared_gate: mx.array | None = None,
    tune: TuneConfig | None = None,
) -> mx.array:
    cfg = _DEFAULT_TUNE if tune is None else tune
    orig_shape = x.shape
    k_h = int(orig_shape[-1])
    m = int(x.size // k_h)
    top_k = int(inds.shape[-1])
    n_pairs = m * top_k
    max_b2 = max(1, n_pairs // 2)

    x_flat = mx.contiguous(x.reshape(m, k_h))
    weights_flat = mx.contiguous(weights.reshape(n_pairs).astype(mx.float32))

    wg_w, wg_s, wg_b = _expert_alias_views(wg_proj)
    w2_w, w2_s, w2_b = _expert_alias_views(w2_proj)
    num_experts = int(_get_field(wg_proj, "weight").shape[0])
    inter = int(_get_field(w2_proj, "scales").shape[-1]) * _GROUP_SIZE

    if wu_proj is not None:
        wu_w, wu_s, wu_b = _expert_alias_views(wu_proj)
        separate_gu = 1
        row_stride = 1
        up_row_off = 0
        n_w = inter
    elif interleaved_w13:
        wu_w, wu_s, wu_b = wg_w, wg_s, wg_b
        separate_gu = 0
        row_stride = 2
        up_row_off = 1
        n_w = 2 * inter
    else:
        # Concatenated [gate; up] along axis 1 (e.g. gate_up_proj).
        wu_w, wu_s, wu_b = wg_w, wg_s, wg_b
        separate_gu = 0
        row_stride = 1
        up_row_off = inter
        n_w = 2 * inter

    is_fp16 = 1 if x_flat.dtype == mx.float16 else 0
    gcap_q12 = int(round(float(gcap) * 4096.0))
    ucap_q12 = int(round(float(ucap) * 4096.0))
    softcap_i = 1 if softcap else 0
    direct_m1 = 1 if m == 1 else 0

    w13_tpl_base = [
        ("T", x_flat.dtype),
        ("TS", wg_s.dtype),
        ("IS_FP16", is_fp16),
        ("K", k_h),
        ("INTER", inter),
        ("SEPARATE_GU", separate_gu),
        ("ROW_STRIDE", row_stride),
        ("UP_ROW_OFF", up_row_off),
        ("N_W", n_w),
        ("SOFTCAP", softcap_i),
        ("GCAP_Q12", gcap_q12),
        ("UCAP_Q12", ucap_q12),
        ("DIRECT_M1", direct_m1),
        ("K_TOP", top_k),
    ]
    w2_tpl_base = [
        ("T", x_flat.dtype),
        ("TS", w2_s.dtype),
        ("IS_FP16", is_fp16),
        ("K", inter),
        ("N", k_h),
        ("DIRECT_M1", direct_m1),
        ("K_TOP", top_k),
    ]

    nsg1_c1, rpsg1_c1 = cfg.w13_c1
    nsg2_c1, rpsg2_c1 = cfg.w2_c1

    if m == 1:
        b1_exp = mx.contiguous(inds.reshape(-1))
        if b1_exp.dtype != mx.uint32:
            b1_exp = b1_exp.astype(mx.uint32)
        (act1,) = _w13_c1_kernel()(
            inputs=[x_flat, wg_w, wg_s, wg_b, wu_w, wu_s, wu_b, b1_exp, b1_exp, b1_exp, b1_exp],
            template=w13_tpl_base + [("NSG", nsg1_c1), ("RPSG", rpsg1_c1)],
            grid=(_ceil_div(inter, nsg1_c1 * rpsg1_c1) * nsg1_c1 * 32, n_pairs, 1),
            threadgroup=(nsg1_c1 * 32, 1, 1),
            output_shapes=[(n_pairs, inter)],
            output_dtypes=[x_flat.dtype],
        )
        (z1,) = _w2_c1_kernel()(
            inputs=[act1, w2_w, w2_s, w2_b, weights_flat, b1_exp, b1_exp, b1_exp],
            template=w2_tpl_base + [("NSG", nsg2_c1), ("RPSG", rpsg2_c1)],
            grid=(_ceil_div(k_h, nsg2_c1 * rpsg2_c1) * nsg2_c1 * 32, n_pairs, 1),
            threadgroup=(nsg2_c1 * 32, 1, 1),
            output_shapes=[(n_pairs, k_h)],
            output_dtypes=[x_flat.dtype],
        )
        out = m1_moe_combine_vec(
            z1,
            z1,
            b1_exp.view(mx.uint8),
            m,
            top_k,
            k_h,
            shared_out=shared_out,
            shared_gate=shared_gate,
            vec=cfg.combine_vec,
            only_b1=True,
        )
        return out.reshape(orig_shape)

    (
        b1_exp,
        b1_tok,
        b1_pair,
        b1_num,
        b2_exp,
        b2_tok0,
        b2_tok1,
        b2_pair0,
        b2_pair1,
        b2_num,
        pair_is_b2,
    ) = m1_moe_plan_dual(inds, num_experts, top_k)

    (act1,) = _w13_c1_kernel()(
        inputs=[x_flat, wg_w, wg_s, wg_b, wu_w, wu_s, wu_b, b1_exp, b1_tok, b1_pair, b1_num],
        template=w13_tpl_base + [("NSG", nsg1_c1), ("RPSG", rpsg1_c1)],
        grid=(_ceil_div(inter, nsg1_c1 * rpsg1_c1) * nsg1_c1 * 32, n_pairs, 1),
        threadgroup=(nsg1_c1 * 32, 1, 1),
        output_shapes=[(n_pairs, inter)],
        output_dtypes=[x_flat.dtype],
    )

    nsg1_cm, rpsg1_cm = cfg.w13_cm
    (act2,) = _w13_cm_kernel()(
        inputs=[
            x_flat,
            wg_w,
            wg_s,
            wg_b,
            wu_w,
            wu_s,
            wu_b,
            b2_exp,
            b2_tok0,
            b2_tok1,
            b2_pair0,
            b2_pair1,
            b2_num,
        ],
        template=w13_tpl_base + [("NSG", nsg1_cm), ("RPSG", rpsg1_cm)],
        grid=(_ceil_div(inter, nsg1_cm * rpsg1_cm) * nsg1_cm * 32, max_b2, 1),
        threadgroup=(nsg1_cm * 32, 1, 1),
        output_shapes=[(n_pairs, inter)],
        output_dtypes=[x_flat.dtype],
    )

    (z1,) = _w2_c1_kernel()(
        inputs=[act1, w2_w, w2_s, w2_b, weights_flat, b1_exp, b1_pair, b1_num],
        template=w2_tpl_base + [("NSG", nsg2_c1), ("RPSG", rpsg2_c1)],
        grid=(_ceil_div(k_h, nsg2_c1 * rpsg2_c1) * nsg2_c1 * 32, n_pairs, 1),
        threadgroup=(nsg2_c1 * 32, 1, 1),
        output_shapes=[(n_pairs, k_h)],
        output_dtypes=[x_flat.dtype],
    )

    nsg2_cm, rpsg2_cm = cfg.w2_cm
    (z2,) = _w2_cm_kernel()(
        inputs=[act2, w2_w, w2_s, w2_b, weights_flat, b2_exp, b2_pair0, b2_pair1, b2_num],
        template=w2_tpl_base + [("NSG", nsg2_cm), ("RPSG", rpsg2_cm)],
        grid=(_ceil_div(k_h, nsg2_cm * rpsg2_cm) * nsg2_cm * 32, max_b2, 1),
        threadgroup=(nsg2_cm * 32, 1, 1),
        output_shapes=[(n_pairs, k_h)],
        output_dtypes=[x_flat.dtype],
    )

    out = m1_moe_combine_vec(
        z1,
        z2,
        pair_is_b2,
        m,
        top_k,
        k_h,
        shared_out=shared_out,
        shared_gate=shared_gate,
        vec=cfg.combine_vec,
    )
    return out.reshape(orig_shape)


def m1_qmv_moe_inkling(
    x: mx.array,
    inds: mx.array,
    weights: mx.array,
    w13: Any,
    w2: Any,
    *,
    gcap: float = 4.0,
    ucap: float = 25.0,
    tune: TuneConfig | None = None,
) -> mx.array:
    """Coalesced QMV MoE forward for Inkling 2 Mini (interleaved ``w13``, softcaps)."""
    return _run_moe_pipeline(
        x,
        inds,
        weights,
        w13,
        None,
        w2,
        interleaved_w13=True,
        softcap=True,
        gcap=gcap,
        ucap=ucap,
        tune=tune,
    )


def m1_qmv_moe_switch(
    x: mx.array,
    inds: mx.array,
    weights: mx.array,
    switch_mlp: Any,
    *,
    shared_out: mx.array | None = None,
    shared_gate: mx.array | None = None,
    tune: TuneConfig | None = None,
) -> mx.array:
    """Coalesced QMV MoE forward for Ornith SwitchGLU (separate or fused gate/up)."""
    gu = getattr(switch_mlp, "gate_up_proj", None)
    if gu is not None:
        wg_proj = gu
        wu_proj = None
    else:
        wg_proj = switch_mlp.gate_proj
        wu_proj = switch_mlp.up_proj
    return _run_moe_pipeline(
        x,
        inds,
        weights,
        wg_proj,
        wu_proj,
        switch_mlp.down_proj,
        interleaved_w13=False,
        softcap=False,
        gcap=0.0,
        ucap=0.0,
        shared_out=shared_out,
        shared_gate=shared_gate,
        tune=tune,
    )


def m1_qmv_moe(
    x: mx.array,
    inds: mx.array,
    weights: mx.array,
    *,
    w13: Any = None,
    w2: Any = None,
    gate_proj: Any = None,
    up_proj: Any = None,
    down_proj: Any = None,
    gate_up_proj: Any = None,
    interleaved_w13: bool = True,
    softcap: bool = False,
    gcap: float = 4.0,
    ucap: float = 25.0,
    shared_out: mx.array | None = None,
    shared_gate: mx.array | None = None,
    tune: TuneConfig | None = None,
) -> mx.array:
    """Unified entry point supporting both Inkling ``w13``/``w2`` and SwitchGLU layouts."""
    if w13 is not None and w2 is not None:
        return _run_moe_pipeline(
            x,
            inds,
            weights,
            w13,
            None,
            w2,
            interleaved_w13=interleaved_w13,
            softcap=softcap,
            gcap=gcap,
            ucap=ucap,
            shared_out=shared_out,
            shared_gate=shared_gate,
            tune=tune,
        )
    if gate_up_proj is not None and down_proj is not None:
        return _run_moe_pipeline(
            x,
            inds,
            weights,
            gate_up_proj,
            None,
            down_proj,
            interleaved_w13=False,
            softcap=softcap,
            gcap=gcap,
            ucap=ucap,
            shared_out=shared_out,
            shared_gate=shared_gate,
            tune=tune,
        )
    if gate_proj is not None and up_proj is not None and down_proj is not None:
        return _run_moe_pipeline(
            x,
            inds,
            weights,
            gate_proj,
            up_proj,
            down_proj,
            interleaved_w13=False,
            softcap=softcap,
            gcap=gcap,
            ucap=ucap,
            shared_out=shared_out,
            shared_gate=shared_gate,
            tune=tune,
        )
    raise ValueError("Provide either (w13, w2), (gate_up_proj, down_proj), or (gate_proj, up_proj, down_proj).")


__all__ = [
    "TuneConfig",
    "eligible_inkling",
    "eligible_switch",
    "m1_moe_combine_vec",
    "m1_moe_plan_dual",
    "m1_qmv_moe",
    "m1_qmv_moe_inkling",
    "m1_qmv_moe_switch",
    "prepare_m1_moe",
    "prepare_switch_moe",
]
