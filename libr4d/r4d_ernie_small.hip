/* r4d_ernie_small.hip -- ERNIE 5.5 Flash's small device operations (epic 4-3). Written from ERNIE's own equations (libref's ref_ernie_situ_mul, the pinned vLLM-ERNIE
 * `situ_and_mul`) and checked against ERNIE's oracles and an independent double-precision one; not adapted from upstream PR 266.
 * (The op is `ernie_situ_mul`: upstream v2.3.0 has its own, unrelated `situ_mul` with gate_cap/up_cap.)
 * SPDX-License-Identifier: Apache-2.0
 *
 * SiTU gated product (ernie_situ_mul): y[r, i] = beta * tanh(g / beta) * sigmoid(g) * linear_beta * tanh(u / linear_beta), with g = gate_up[r, i] and u = gate_up[r, n + i]: the GATE IS THE
 * FIRST HALF of silu_mul's fused [M, 2n] plane. Both clips are finite and strictly positive (ERNIE 4 and 25); anything else is refused before a byte is written.
 * f32 arithmetic, one rounding to bf16 at the store (round to nearest even), as libref rounds once to y's dtype. A large finite gate is safe: tanh saturates to +-1 and
 * the sigmoid to 0 or 1, so the product stays within +-(beta * linear_beta). The product is formed gate * up in f32 as libref's ref_ernie_situ_mul forms it, so with clip
 * constants far from ERNIE's (4 and 25), say below 2^-20, a gate factor can underflow before the multiply; the supported range is ERNIE's.
 */
#include <hip/hip_runtime.h>

#include <climits>
#include <cmath>
#include <cstdint>

#include "r4d_args.h"
#include "r4d_plugin.h"
#include "r4d_launch.h"
#include "rad_plugin.h"
#include <cstring>
#ifndef R4D_ARGS   /* as r4d_ernie_common.h: every byte of a kernel-argument struct, padding included, is defined (memset) */
#define R4D_ARGS(T, name) T name; std::memset(&(name), 0, sizeof(name))
#endif

namespace r4d_ernie_small {

struct SituArgs {
    const uint16_t* gu; long long gp;      /* [M, 2n] row pitch */
    uint16_t*       y;  long long yp;      /* [M, n] row pitch */
    int             n;
    float           beta, lbeta;
};

__device__ __forceinline__ float bf2f(uint16_t h) { return __uint_as_float((unsigned)h << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) {
    const unsigned u = __float_as_uint(f);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0x7fffffu) ? 0x40u : 0u));   /* inf stays, NaN stays quiet */
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

__device__ __forceinline__ uint16_t situ_elem(uint16_t gh, uint16_t uh, float beta, float lbeta) {
    const float g = bf2f(gh);
    const float u = bf2f(uh);
    /* the stable logistic: expf of a non-positive argument only, so the tail of a very negative gate (below -88.7, where expf(-g) overflows) keeps its true value instead of 0 */
    const float eg = g < 0.0f ? expf(g) : 0.0f;
    const float sg = g >= 0.0f ? 1.0f / (1.0f + expf(-g)) : eg / (1.0f + eg);
    const float gate = beta * tanhf(g / beta) * sg;
    const float up = lbeta * tanhf(u / lbeta);
    return f2bf(gate * up);
}

__global__ __launch_bounds__(256) void situ_mul_kernel(SituArgs A) {
    const long long r = blockIdx.y;
    const int i = (int)blockIdx.x * 256 + (int)threadIdx.x;
    if (i >= A.n) return;
    A.y[r * A.yp + i] = situ_elem(A.gu[r * A.gp + i], A.gu[r * A.gp + A.n + i], A.beta, A.lbeta);
}

/* Vectorized uint4 fast path: 8 BF16 elements (16B) per thread when n, gp, yp are multiples of 8
 * and gu, y are 16B aligned. Exact same per-element f32 arithmetic and bf16 rounding. */
__global__ __launch_bounds__(256) void situ_mul_vec8_kernel(SituArgs A) {
    const long long r = blockIdx.y;
    const int i8 = (int)blockIdx.x * 256 + (int)threadIdx.x;
    const int n8 = A.n >> 3;
    if (i8 >= n8) return;
    const int i = i8 << 3;
    const uint4 gv = *(const uint4*)(A.gu + r * A.gp + i);
    const uint4 uv = *(const uint4*)(A.gu + r * A.gp + A.n + i);
    uint4 yv;
    yv.x = (uint32_t)situ_elem((uint16_t)gv.x, (uint16_t)uv.x, A.beta, A.lbeta) |
           ((uint32_t)situ_elem((uint16_t)(gv.x >> 16), (uint16_t)(uv.x >> 16), A.beta, A.lbeta) << 16);
    yv.y = (uint32_t)situ_elem((uint16_t)gv.y, (uint16_t)uv.y, A.beta, A.lbeta) |
           ((uint32_t)situ_elem((uint16_t)(gv.y >> 16), (uint16_t)(uv.y >> 16), A.beta, A.lbeta) << 16);
    yv.z = (uint32_t)situ_elem((uint16_t)gv.z, (uint16_t)uv.z, A.beta, A.lbeta) |
           ((uint32_t)situ_elem((uint16_t)(gv.z >> 16), (uint16_t)(uv.z >> 16), A.beta, A.lbeta) << 16);
    yv.w = (uint32_t)situ_elem((uint16_t)gv.w, (uint16_t)uv.w, A.beta, A.lbeta) |
           ((uint32_t)situ_elem((uint16_t)(gv.w >> 16), (uint16_t)(uv.w >> 16), A.beta, A.lbeta) << 16);
    *(uint4*)(A.y + r * A.yp + i) = yv;
}

/* ernie_router_topk: ERNIE's routing between its two router projections and the latent expert dispatch (the pinned Baidu `ernie5_5_routing`, libref ref_ernie_router_topk).
 * score[e] = sigmoid(logits[e]) + sigmoid(logits[experts + e]); the correction bias SELECTS (descending score + bias, ties to the lower expert id); the UNCORRECTED score WEIGHTS;
 * with renormalize the kept scores are first divided by their own sum (clamped to at least 1e-20), and the per-expert scale is ALWAYS applied after the division, indexed by the
 * logical expert id. Everything is f32, the sigmoid is libref's 1 / (1 + expf(-x)), the kept scores are summed in pick order.
 * One wave32 per row, register-resident (0 LDS, 0 __syncthreads): each lane holds OWN <= 16 experts (e = lane + 32*t), validates finiteness via __ballot, and runs k wave32
 * __shfl_xor reductions with exact lower-id tie-breaking. At most 512 experts and top-k <= 32. */
struct RouterArgs {
    const float* lg; long long lp;      /* [M, 2 * experts], row pitch */
    const float* bias; const float* scale;
    int32_t* ids; long long ip;         /* [M, k] */
    float* w; long long wp;
    int M, ne, k, renorm;
};

template <int OWN>
__global__ __launch_bounds__(128) void ernie_router_topk_kernel(RouterArgs A) {
    const int lane = (int)(threadIdx.x & 31);
    const long long r = (long long)blockIdx.x * (long long)(blockDim.x >> 5) + (long long)(threadIdx.x >> 5);
    if (r >= A.M) return;
    const float* l = A.lg + r * A.lp;

    float sc[OWN], sel[OWN];
    int bad = 0;
#pragma unroll
    for (int t = 0; t < OWN; ++t) {
        const int e = lane + 32 * t;
        if (e < A.ne) {
            const float l0 = l[e], l1 = l[A.ne + e], b = A.bias[e], sv = A.scale[e];
            if (!isfinite(l0) || !isfinite(l1) || !isfinite(b) || !isfinite(sv)) {
                bad = 1;
                sc[t] = 0.0f;
                sel[t] = -INFINITY;
            } else {
                const float s = (1.0f / (1.0f + expf(-l0))) + (1.0f / (1.0f + expf(-l1)));
                sc[t] = s;
                sel[t] = s + b;
            }
        } else {
            sc[t] = 0.0f;
            sel[t] = -INFINITY;
        }
    }
    if (__ballot(bad) != 0) return;     /* leave sentinel outputs untouched on non-finite input (no host sync allowed) */

    int my_id = 0;
    float my_sc = 0.0f;
    float sum = 0.0f;
    for (int j = 0; j < A.k; ++j) {
        float bv = -INFINITY;
        int bi = -1;
#pragma unroll
        for (int t = 0; t < OWN; ++t) {
            const float v = sel[t];
            if (v > -INFINITY && (bi < 0 || v > bv)) {      /* strictly greater: lower id wins ties */
                bv = v;
                bi = lane + 32 * t;
            }
        }
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            const float ov = __shfl_xor(bv, mask, 32);
            const int oi = __shfl_xor(bi, mask, 32);
            if (oi >= 0 && (bi < 0 || ov > bv || (ov == bv && oi < bi))) {
                bv = ov;
                bi = oi;
            }
        }
        if (bi < 0) break;
        const int win_t = bi >> 5;
        const int win_lane = bi & 31;
        float chosen_sc = 0.0f;
#pragma unroll
        for (int t = 0; t < OWN; ++t) {
            if (t == win_t) {
                chosen_sc = sc[t];
                if (lane == win_lane) sel[t] = -INFINITY;
            }
        }
        chosen_sc = __shfl(chosen_sc, win_lane, 32);
        if (lane == j) {
            my_id = bi;
            my_sc = chosen_sc;
        }
        sum += chosen_sc;
    }
    const float denom = A.renorm ? (sum > 1e-20f ? sum : 1e-20f) : 1.0f;
    if (lane < A.k) {
        A.ids[r * A.ip + lane] = my_id;
        A.w[r * A.wp + lane] = (my_sc / denom) * A.scale[my_id];
    }
}

}  /* namespace r4d_ernie_small */

extern "C" {

/* operands: gate_up [M, 2n] bf16, y [M, n] bf16 (out); params M, n, beta, linear_beta, dtype */
int r4d_rad_ernie_situ_mul(const RadArgs* a, RadStream stream) {
    using namespace r4d_ernie_small;
    if (!a || !a->t || a->n_t < 2) return RAD_E_INVAL;
    const RadTensor &gu = a->t[0], &y = a->t[1];
    const long long n = r4d_geti_or(a, "n", 0);
    const float beta = (float)r4d_getf(a, "beta", 0.0), lbeta = (float)r4d_getf(a, "linear_beta", 0.0);
    if (!(beta > 0.0f) || !(lbeta > 0.0f) || !std::isfinite(beta) || !std::isfinite(lbeta)) return RAD_E_INVAL;
    if (gu.dtype != RAD_BF16 || y.dtype != RAD_BF16) return RAD_E_DTYPE;
    if (n < 1 || n > INT_MAX / 2 || !gu.data || !y.data || gu.rank != 2 || y.rank != 2 || y.shape[0] < 1 || y.shape[0] > 65535 || gu.shape[0] != y.shape[0] ||
        gu.shape[1] != 2 * n || y.shape[1] != n || gu.stride[1] != 1 || y.stride[1] != 1 || gu.stride[0] < 2 * n || y.stride[0] < n)
        return RAD_E_SHAPE;
    if (r4d_geti_or(a, "M", y.shape[0]) < y.shape[0]) return RAD_E_SHAPE;      /* the declared row count is the tensors' */
    R4D_ARGS(SituArgs, P);
    P.gu = (const uint16_t*)gu.data; P.gp = gu.stride[0];
    P.y = (uint16_t*)y.data;         P.yp = y.stride[0];
    P.n = (int)n; P.beta = beta; P.lbeta = lbeta;
    const bool vec8 = ((n & 7) == 0) && ((gu.stride[0] & 7) == 0) && ((y.stride[0] & 7) == 0) &&
                      (((uintptr_t)gu.data & 15u) == 0) && (((uintptr_t)y.data & 15u) == 0);
    if (vec8) {
        const long long n8 = n >> 3;
        R4D_KLAUNCH(R4D_KNAME(situ_mul_vec8_kernel), dim3((unsigned)((n8 + 255) / 256), (unsigned)y.shape[0]), dim3(256), (hipStream_t)stream, P);
    } else {
        R4D_KLAUNCH(R4D_KNAME(situ_mul_kernel), dim3((unsigned)((n + 255) / 256), (unsigned)y.shape[0]), dim3(256), (hipStream_t)stream, P);
    }
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

/* operands: logits [M, 2*experts] f32, bias [experts] f32, scale [experts] f32, ids [M, k] i32 (out), weights [M, k] f32 (out); params M, experts, topk, renormalize */
int r4d_rad_ernie_router_topk(const RadArgs* a, RadStream stream) {
    using namespace r4d_ernie_small;
    if (!a || a->n_t < 5) return RAD_E_INVAL;
    const RadTensor &lg = a->t[0], &bias = a->t[1], &scale = a->t[2], &ids = a->t[3], &w = a->t[4];
    const long long ne = r4d_geti_or(a, "experts", 0), k = r4d_geti_or(a, "topk", 0), renorm = r4d_geti_or(a, "renormalize", -1);
    if (ne < 1 || ne > 512 || k < 1 || k > ne || k > 32 || (renorm != 0 && renorm != 1)) return RAD_E_SHAPE;
    if (lg.dtype != RAD_F32 || bias.dtype != RAD_F32 || scale.dtype != RAD_F32 || ids.dtype != RAD_I32 || w.dtype != RAD_F32) return RAD_E_DTYPE;
    if (!lg.data || !bias.data || !scale.data || !ids.data || !w.data || lg.rank != 2 || ids.rank != 2 || w.rank != 2 || bias.rank != 1 || scale.rank != 1) return RAD_E_SHAPE;
    const long long M = lg.shape[0];
    if (M < 1 || M > 65535 || lg.shape[1] != 2 * ne || bias.shape[0] != ne || scale.shape[0] != ne || ids.shape[0] != M || w.shape[0] != M || ids.shape[1] != k || w.shape[1] != k ||
        lg.stride[1] != 1 || ids.stride[1] != 1 || w.stride[1] != 1 || bias.stride[0] != 1 || scale.stride[0] != 1 || lg.stride[0] < 2 * ne || ids.stride[0] < k || w.stride[0] < k)
        return RAD_E_SHAPE;
    /* finite inputs, across every operand and before a byte is written (host-accessible emulator check; device VRAM pointers cannot be host-dereferenced under HIP) */
    const float* l = (const float*)lg.data;
#if !defined(__HIP__) && !defined(__HIP_PLATFORM_AMD__)
    for (long long m = 0; m < M; ++m)
        for (long long e = 0; e < 2 * ne; ++e)
            if (!std::isfinite(l[m * lg.stride[0] + e])) return RAD_E_SHAPE;
    for (long long e = 0; e < ne; ++e)
        if (!std::isfinite(((const float*)bias.data)[e]) || !std::isfinite(((const float*)scale.data)[e])) return RAD_E_SHAPE;
#endif
    R4D_ARGS(RouterArgs, P);
    P.lg = l; P.lp = lg.stride[0]; P.bias = (const float*)bias.data; P.scale = (const float*)scale.data;
    P.ids = (int32_t*)ids.data; P.ip = ids.stride[0]; P.w = (float*)w.data; P.wp = w.stride[0];
    P.M = (int)M; P.ne = (int)ne; P.k = (int)k; P.renorm = (int)renorm;
    const dim3 grid = M == 1 ? dim3(1) : dim3((unsigned)((M + 3) / 4));
    const dim3 block = M == 1 ? dim3(32) : dim3(128);
    if (ne <= 256) {
        R4D_KLAUNCH(R4D_KNAME(ernie_router_topk_kernel<8>), grid, block, (hipStream_t)stream, P);
    } else {
        R4D_KLAUNCH(R4D_KNAME(ernie_router_topk_kernel<16>), grid, block, (hipStream_t)stream, P);
    }
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

}  /* extern "C" */

namespace r4d_ernie_small {

/* ---- software fallback for host r4d_emu + hardware v_cvt_pk_fp8_f32 on gfx1201 ---- */
__device__ __forceinline__ uint8_t f32_to_e4m3_sw(float f) {
    if (f > 448.0f) f = 448.0f;
    if (f < -448.0f) f = -448.0f;
    uint32_t u = __float_as_uint(f);
    const uint32_t sign = (u >> 24) & 0x80u;
    const uint32_t exp = (u >> 23) & 0xffu;
    const uint32_t man = u & 0x7fffffu;
    if (exp == 0xffu) return (uint8_t)(sign | 0x7fu);
    int E = (int)exp - 120;
    if (E >= 1) {
        uint32_t keep = man >> 20;
        const uint32_t rem = man & 0xfffffu;
        if (rem > 0x80000u || (rem == 0x80000u && (keep & 1u))) keep += 1u;
        if (keep == 8u) { keep = 0u; E += 1; }
        if (E > 15) { E = 15; keep = 6u; }
        return (uint8_t)(sign | ((uint32_t)E << 3) | keep);
    }
    if (exp < 110u) return (uint8_t)sign;
    const int sh = 141 - (int)exp;
    const uint32_t num = (1u << 23) | man;
    uint32_t q = num >> sh;
    const uint32_t half = 1u << (sh - 1);
    const uint32_t rem = num & ((1u << sh) - 1u);
    if (rem > half || (rem == half && (q & 1u))) q += 1u;
    if (q == 8u) return (uint8_t)(sign | (1u << 3));
    return (uint8_t)(sign | q);
}

__device__ __forceinline__ uint32_t pack4_e4m3(float a, float b, float c, float d) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__clang__)
    int pk = 0;
    pk = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(a, -448.0f), 448.0f),
                                         fminf(fmaxf(b, -448.0f), 448.0f), pk, false);
    pk = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(c, -448.0f), 448.0f),
                                         fminf(fmaxf(d, -448.0f), 448.0f), pk, true);
    return (uint32_t)pk;
#else
    return (uint32_t)f32_to_e4m3_sw(a) | ((uint32_t)f32_to_e4m3_sw(b) << 8) |
           ((uint32_t)f32_to_e4m3_sw(c) << 16) | ((uint32_t)f32_to_e4m3_sw(d) << 24);
#endif
}

__device__ __forceinline__ float wave_max_f32(float v) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) v = fmaxf(v, __shfl_xor(v, mask, 32));
    return v;
}

/* ---- 1. situ_mul_quant_fp8: fused SiTU activation + group=128 E4M3 quantization ---- */
struct SituQuantArgs {
    const uint16_t* gu; long long gp;
    uint8_t*        q;  long long qp;
    float*          s;  long long sp;
    int             n, nkb;
    float           beta, lbeta;
};
static_assert(sizeof(SituQuantArgs) <= 512, "SituQuantArgs exceeds AQL kernarg limit");

__global__ __launch_bounds__(256) void situ_mul_quant_fp8_kernel(SituQuantArgs A) {
    const long long r = blockIdx.x;
    const int lane = (int)threadIdx.x & 31, wave = (int)threadIdx.x >> 5;
    const int nwaves = (int)(blockDim.x >> 5);
    const uint16_t* grow = A.gu + r * A.gp;
    uint8_t* qrow = A.q + r * A.qp;
    float* srow = A.s + r * A.sp;
    for (int kb = wave; kb < A.nkb; kb += nwaves) {
        const int c0 = kb * 128 + lane * 4;
        float v[4], m = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = c0 + j;
            v[j] = 0.0f;
            if (c < A.n) {
                const uint16_t h = situ_elem(grow[c], grow[A.n + c], A.beta, A.lbeta);
                v[j] = bf2f(h);
            }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) m = fmaxf(m, fabsf(v[j]));
        m = wave_max_f32(m);
        const float sc = m > 0.0f ? m * (1.0f / 448.0f) : 1.0f;
        if (lane == 0) srow[kb] = sc;
        const float inv = 1.0f / sc;
        const uint32_t pk = pack4_e4m3(v[0] * inv, v[1] * inv, v[2] * inv, v[3] * inv);
        if (c0 < A.n) *(uint32_t*)(qrow + c0) = pk;
    }
}

/* ---- 2. ernie_router_topk_scatter (M == 1 single-wave fused routing + scatter) ---- */
struct RouterScatterArgs {
    const float *lg, *bias, *scale;
    int32_t *ids, *sorted, *offsets, *counts;
    float *w;
    long long lp, ip, wp;
    int ne, k, renorm;
};
static_assert(sizeof(RouterScatterArgs) <= 512, "RouterScatterArgs exceeds AQL kernarg limit");

template <int SLOTS>
__global__ __launch_bounds__(32) void ernie_router_topk_scatter_m1_kernel(RouterScatterArgs A) {
    __shared__ int s_eid[32], s_slot[32];
    __shared__ float s_kept[32];
    const int lane = (int)threadIdx.x;
    const float* l = A.lg;
    float sc[SLOTS], sel[SLOTS];
    bool bad = false;
#pragma unroll
    for (int s = 0; s < SLOTS; ++s) {
        const int e = lane + s * 32;
        if (e < A.ne) {
            const float a = l[e], b = l[A.ne + e];
            const float bi = A.bias[e], si = A.scale[e];
            if (!isfinite(a) || !isfinite(b) || !isfinite(bi) || !isfinite(si)) bad = true;
            const float p = (1.0f / (1.0f + expf(-a))) + (1.0f / (1.0f + expf(-b)));
            sc[s] = p;
            sel[s] = p + bi;
        } else {
            sc[s] = 0.0f;
            sel[s] = -INFINITY;
        }
    }
    if (__ballot(bad) != 0ull) return;
    float sum = 0.0f;
    for (int j = 0; j < A.k; ++j) {
        float best = -INFINITY;
        int best_e = INT_MAX, best_s = -1;
#pragma unroll
        for (int s = 0; s < SLOTS; ++s) {
            const int e = lane + s * 32;
            if (e < A.ne && sel[s] > -INFINITY && (best_s < 0 || sel[s] > best)) {
                best = sel[s]; best_e = e; best_s = s;
            }
        }
        float win = best;
        int win_e = best_e;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor(win, off, 32);
            const int oe = __shfl_xor(win_e, off, 32);
            if (ov > win || (ov == win && oe < win_e)) { win = ov; win_e = oe; }
        }
        float win_sc = 0.0f;
        if (best_s >= 0 && best_e == win_e) {
            win_sc = sc[best_s];
            sel[best_s] = -INFINITY;
        }
        win_sc = __shfl(win_sc, win_e & 31, 32);
        if (lane == 0) {
            s_eid[j] = win_e;
            s_slot[j] = j;
            s_kept[j] = win_sc;
        }
        sum += win_sc;
    }
    __syncthreads();
    const float den = sum > 1e-20f ? sum : 1e-20f;
    if (lane < A.k) {
        const int e = s_eid[lane];
        A.ids[lane] = e;
        A.w[lane] = (A.renorm ? s_kept[lane] / den : s_kept[lane]) * A.scale[e];
    }
    __syncthreads();
    if (lane == 0) {
        for (int i = 1; i < A.k; ++i) {
            const int ke = s_eid[i], ks = s_slot[i];
            int p = i - 1;
            while (p >= 0 && s_eid[p] > ke) {
                s_eid[p + 1] = s_eid[p];
                s_slot[p + 1] = s_slot[p];
                --p;
            }
            s_eid[p + 1] = ke;
            s_slot[p + 1] = ks;
        }
        A.offsets[A.ne] = A.k;
    }
    __syncthreads();
    if (lane < A.k) A.sorted[lane] = s_slot[lane];
    for (int e = lane; e < A.ne; e += 32) {
        int off = 0, cnt = 0;
        for (int j = 0; j < A.k; ++j) {
            const int v = s_eid[j];
            if (v < e) ++off;
            else if (v == e) cnt = 1;
        }
        A.offsets[e] = off;
        A.counts[e] = cnt;
    }
}

/* ---- 3. ernie_add_mhc_post: fused BF16 add + mhc_post (exact BF16 intermediate rounding) ---- */
struct AddMhcPostArgs {
    const uint16_t *a, *b, *h;
    const float *post, *comb;
    uint16_t *out;
    long long ap, bp, hp, pp, cp, op;
    int n, hc;
};
static_assert(sizeof(AddMhcPostArgs) <= 512, "AddMhcPostArgs exceeds AQL kernarg limit");

__global__ __launch_bounds__(256) void ernie_add_mhc_post_vec4_kernel(AddMhcPostArgs A) {
    const long long r = blockIdx.y;
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i >= A.n) return;
    auto unpack = [](const uint16_t* p, float* v) {
        const uint4 q = *reinterpret_cast<const uint4*>(p);
        const uint32_t w[4] = { q.x, q.y, q.z, q.w };
#pragma unroll
        for (int e = 0; e < 8; ++e) v[e] = bf2f((uint16_t)(w[e >> 1] >> (16 * (e & 1))));
    };
    float av[8], bv[8], xv[8], hv[4][8];
    unpack(A.a + r * A.ap + i, av);
    unpack(A.b + r * A.bp + i, bv);
#pragma unroll
    for (int e = 0; e < 8; ++e) xv[e] = bf2f(f2bf(av[e] + bv[e]));
#pragma unroll
    for (int j = 0; j < 4; ++j) unpack(A.h + r * A.hp + (long long)j * A.n + i, hv[j]);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        float c[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) c[j] = A.comb[r * A.cp + j * 4 + k];
        const float pk = A.post[r * A.pp + k];
        uint32_t o[4];
#pragma unroll
        for (int e = 0; e < 8; e += 2) {
            float v0 = 0.0f, v1 = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                v0 += c[j] * hv[j][e];
                v1 += c[j] * hv[j][e + 1];
            }
            v0 += pk * xv[e];
            v1 += pk * xv[e + 1];
            o[e >> 1] = (uint32_t)f2bf(v0) | ((uint32_t)f2bf(v1) << 16);
        }
        *reinterpret_cast<uint4*>(A.out + r * A.op + (long long)k * A.n + i) =
            make_uint4(o[0], o[1], o[2], o[3]);
    }
}

__global__ __launch_bounds__(256) void ernie_add_mhc_post_kernel(AddMhcPostArgs A) {
    const long long r = blockIdx.y;
    const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= A.n * A.hc) return;
    const int k = idx / A.n, i = idx - k * A.n;
    const float xv = bf2f(f2bf(bf2f(A.a[r * A.ap + i]) + bf2f(A.b[r * A.bp + i])));
    float v = 0.0f;
    for (int j = 0; j < A.hc; ++j)
        v += A.comb[r * A.cp + j * A.hc + k] * bf2f(A.h[r * A.hp + (long long)j * A.n + i]);
    v += A.post[r * A.pp + k] * xv;
    A.out[r * A.op + (long long)k * A.n + i] = f2bf(v);
}

}  /* namespace r4d_ernie_small */

extern "C" {

int r4d_rad_situ_mul_quant_fp8(const RadArgs* a, RadStream stream) {
    using namespace r4d_ernie_small;
    if (!a || !a->t || a->n_t < 3) return RAD_E_INVAL;
    const RadTensor &gu = a->t[0], &q = a->t[1], &sc = a->t[2];
    const long long n = r4d_geti_or(a, "n", 0), nkb = (n + 127) / 128;
    const float beta = (float)r4d_getf(a, "beta", 0.0), lbeta = (float)r4d_getf(a, "linear_beta", 0.0);
    if (!(beta > 0.0f) || !(lbeta > 0.0f) || !std::isfinite(beta) || !std::isfinite(lbeta)) return RAD_E_INVAL;
    if (gu.dtype != RAD_BF16 || q.dtype != RAD_F8E4M3 || sc.dtype != RAD_F32) return RAD_E_DTYPE;
    if (r4d_geti_or(a, "group", 128) != 128 || n < 4 || (n % 4) != 0 || n > INT_MAX / 2 ||
        !gu.data || !q.data || !sc.data || gu.rank != 2 || q.rank != 2 || sc.rank != 2 ||
        gu.shape[0] < 1 || gu.shape[0] > 65535 || gu.shape[1] != 2 * n ||
        q.shape[0] != gu.shape[0] || q.shape[1] != n || sc.shape[0] != gu.shape[0] || sc.shape[1] != nkb ||
        gu.stride[1] != 1 || q.stride[1] != 1 || sc.stride[1] != 1 ||
        gu.stride[0] < 2 * n || q.stride[0] < n || sc.stride[0] < nkb ||
        (((uintptr_t)q.data & 3u) != 0) || ((q.stride[0] & 3) != 0))
        return RAD_E_SHAPE;
    if (r4d_geti_or(a, "M", gu.shape[0]) < gu.shape[0]) return RAD_E_SHAPE;
    R4D_ARGS(SituQuantArgs, P);
    P.gu = (const uint16_t*)gu.data; P.gp = gu.stride[0];
    P.q = (uint8_t*)q.data;          P.qp = q.stride[0];
    P.s = (float*)sc.data;           P.sp = sc.stride[0];
    P.n = (int)n; P.nkb = (int)nkb; P.beta = beta; P.lbeta = lbeta;
    const unsigned thr = (unsigned)(nkb < 8 ? nkb : 8) * 32u;
    R4D_KLAUNCH(R4D_KNAME(situ_mul_quant_fp8_kernel), dim3((unsigned)gu.shape[0]), dim3(thr), (hipStream_t)stream, P);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

int r4d_rad_ernie_router_topk_scatter(const RadArgs* a, RadStream stream) {
    using namespace r4d_ernie_small;
    if (!a || a->n_t < 8) return RAD_E_INVAL;
    const RadTensor &lg = a->t[0], &bias = a->t[1], &scale = a->t[2], &ids = a->t[3],
                    &w = a->t[4], &sorted = a->t[5], &offsets = a->t[6], &counts = a->t[7];
    const long long ne = r4d_geti_or(a, "experts", 0), k = r4d_geti_or(a, "topk", 0),
                    renorm = r4d_geti_or(a, "renormalize", -1);
    if (ne < 1 || ne > 512 || k < 1 || k > ne || k > 32 || (renorm != 0 && renorm != 1)) return RAD_E_SHAPE;
    if (lg.dtype != RAD_F32 || bias.dtype != RAD_F32 || scale.dtype != RAD_F32 ||
        ids.dtype != RAD_I32 || w.dtype != RAD_F32 || sorted.dtype != RAD_I32 ||
        offsets.dtype != RAD_I32 || counts.dtype != RAD_I32)
        return RAD_E_DTYPE;
    if (!lg.data || !bias.data || !scale.data || !ids.data || !w.data ||
        !sorted.data || !offsets.data || !counts.data)
        return RAD_E_SHAPE;
    if (lg.rank != 2 || ids.rank != 2 || w.rank != 2 || bias.rank != 1 || scale.rank != 1 ||
        sorted.rank != 1 || offsets.rank != 1 || counts.rank != 1)
        return RAD_E_SHAPE;
    const long long M = lg.shape[0];
    if (M != 1) return RAD_E_UNSUPPORTED;
    if (lg.shape[1] != 2 * ne || bias.shape[0] != ne || scale.shape[0] != ne ||
        ids.shape[0] != 1 || w.shape[0] != 1 || ids.shape[1] != k || w.shape[1] != k ||
        sorted.shape[0] != k || offsets.shape[0] != ne + 1 || counts.shape[0] != ne ||
        lg.stride[1] != 1 || ids.stride[1] != 1 || w.stride[1] != 1 || bias.stride[0] != 1 ||
        scale.stride[0] != 1 || sorted.stride[0] != 1 || offsets.stride[0] != 1 || counts.stride[0] != 1 ||
        lg.stride[0] < 2 * ne || ids.stride[0] < k || w.stride[0] < k)
        return RAD_E_SHAPE;
    const float* l = (const float*)lg.data;
#if !defined(__HIP__) && !defined(__HIP_PLATFORM_AMD__)
    for (long long e = 0; e < 2 * ne; ++e)
        if (!std::isfinite(l[e])) return RAD_E_SHAPE;
    for (long long e = 0; e < ne; ++e)
        if (!std::isfinite(((const float*)bias.data)[e]) || !std::isfinite(((const float*)scale.data)[e]))
            return RAD_E_SHAPE;
#endif
    R4D_ARGS(RouterScatterArgs, P);
    P.lg = l; P.bias = (const float*)bias.data; P.scale = (const float*)scale.data;
    P.ids = (int32_t*)ids.data; P.w = (float*)w.data;
    P.sorted = (int32_t*)sorted.data; P.offsets = (int32_t*)offsets.data; P.counts = (int32_t*)counts.data;
    P.lp = lg.stride[0]; P.ip = ids.stride[0]; P.wp = w.stride[0];
    P.ne = (int)ne; P.k = (int)k; P.renorm = (int)renorm;
    if (ne <= 256) {
        R4D_KLAUNCH(R4D_KNAME(ernie_router_topk_scatter_m1_kernel<8>), dim3(1), dim3(32), (hipStream_t)stream, P);
    } else {
        R4D_KLAUNCH(R4D_KNAME(ernie_router_topk_scatter_m1_kernel<16>), dim3(1), dim3(32), (hipStream_t)stream, P);
    }
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

int r4d_rad_ernie_add_mhc_post(const RadArgs* a, RadStream stream) {
    using namespace r4d_ernie_small;
    if (!a || !a->t || a->n_t < 6) return RAD_E_INVAL;
    const RadTensor &ta = a->t[0], &tb = a->t[1], &h = a->t[2], &p = a->t[3], &c = a->t[4], &o = a->t[5];
    const long long n = r4d_geti_or(a, "n", 0), hc = r4d_geti_or(a, "hc", 0);
    if (n < 1 || n > 16384 || hc < 2 || hc > 8) return RAD_E_SHAPE;
    if (ta.dtype != RAD_BF16 || tb.dtype != RAD_BF16 || h.dtype != RAD_BF16 ||
        p.dtype != RAD_F32 || c.dtype != RAD_F32 || o.dtype != RAD_BF16)
        return RAD_E_DTYPE;
    if (!ta.data || !tb.data || !h.data || !p.data || !c.data || !o.data || o.data == h.data)
        return RAD_E_SHAPE;
    if (ta.rank != 2 || tb.rank != 2 || h.rank != 2 || p.rank != 2 || c.rank != 2 || o.rank != 2)
        return RAD_E_SHAPE;
    const long long rows = ta.shape[0];
    if (rows < 1 || rows > 65535 || ta.shape[1] != n || tb.shape[0] != rows || tb.shape[1] != n ||
        h.shape[0] != rows || h.shape[1] != hc * n || p.shape[0] != rows || p.shape[1] != hc ||
        c.shape[0] != rows || c.shape[1] != hc * hc || o.shape[0] != rows || o.shape[1] != hc * n ||
        ta.stride[1] != 1 || tb.stride[1] != 1 || h.stride[1] != 1 || p.stride[1] != 1 ||
        c.stride[1] != 1 || o.stride[1] != 1 || ta.stride[0] < n || tb.stride[0] < n ||
        h.stride[0] < hc * n || p.stride[0] < hc || c.stride[0] < hc * hc || o.stride[0] < hc * n)
        return RAD_E_SHAPE;
    R4D_ARGS(AddMhcPostArgs, P);
    P.a = (const uint16_t*)ta.data; P.b = (const uint16_t*)tb.data; P.h = (const uint16_t*)h.data;
    P.post = (const float*)p.data;  P.comb = (const float*)c.data;  P.out = (uint16_t*)o.data;
    P.ap = ta.stride[0]; P.bp = tb.stride[0]; P.hp = h.stride[0];
    P.pp = p.stride[0];  P.cp = c.stride[0];  P.op = o.stride[0];
    P.n = (int)n; P.hc = (int)hc;
    auto aligned = [](const RadTensor& t) {
        return (((uintptr_t)t.data & 15u) == 0u) && ((t.stride[0] & 7) == 0);
    };
    if (hc == 4 && (n & 7) == 0 && aligned(ta) && aligned(tb) && aligned(h) && aligned(o)) {
        R4D_KLAUNCH(R4D_KNAME(ernie_add_mhc_post_vec4_kernel),
                    dim3((unsigned)((n / 8 + 255) / 256), (unsigned)rows), dim3(256), (hipStream_t)stream, P);
    } else {
        R4D_KLAUNCH(R4D_KNAME(ernie_add_mhc_post_kernel),
                    dim3((unsigned)((n * hc + 255) / 256), (unsigned)rows), dim3(256), (hipStream_t)stream, P);
    }
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

}  /* extern "C" */
