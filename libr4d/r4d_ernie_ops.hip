/* r4d_ernie_ops.hip -- ERNIE 5.5 Flash small & attention-prep device ops (story 4-3c):
 *   ernie_multimax, ernie_row_req, ernie_head_slice, ernie_rope_tail, ernie_hca_q_prep, ernie_vha_mix.
 * Written cleanly from ERNIE 5.5 Flash equations and libref references.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cstddef>
#include "r4d_common.h"
#include "r4d_ernie_common.h"
#include "r4d_ernie_verdict.h"
#include "r4d_plugin.h"

namespace r4d_ernie_ops {

using namespace r4d_ernie;

/* libref refuses a launch with a position below 0 (rope_tail, hca_q_prep) before its first store. A card launch cannot refuse: the kernel of a row whose position is
 * negative returns before its first store (the row is left as it was) and one lane raises an event into this log, which r4d_ernie_ops_take_error() hands to the
 * host (the engine polls it where it already waits for the step). Bit 2: a negative position. */
#define R4D_OPS_NEG_POS(A, m, first) do { if ((A).vc.log && (A).pos[(long long)(m) * (A).ps] < 0) { if (first) r4d_verdict::raise((A).vc.log, 2u); return; } } while (0)

static constexpr double kTwoPi = 6.283185307179586476925286766559;
static constexpr double kInvTwoPi = 0.15915494309189533576888376337251;

__device__ __forceinline__ float bf2f_lo(uint32_t u) {
    return __uint_as_float(u << 16);
}
__device__ __forceinline__ float bf2f_hi(uint32_t u) {
    return __uint_as_float(u & 0xffff0000u);
}
__device__ __forceinline__ uint32_t pack_bf2(float a, float b) {
    return (uint32_t)f2bf(a) | ((uint32_t)f2bf(b) << 16);
}
__device__ __forceinline__ void rope_cos_sin_fast(double p, double inv_freq, float* cs, float* sn) {
    const double ang = p * inv_freq;
    const float red = (float)(ang - floor(ang * kInvTwoPi) * kTwoPi);
    sincosf(red, sn, cs);
}

/* ------------------------------------------------------------------ ernie_multimax */
struct MultimaxArgs {
    const float *x, *rg, *cf;
    float *y;
    long long xp, xs, rgs, cfs, yp, ys;
    int rows, n;
};
static_assert(sizeof(MultimaxArgs) <= 512, "MultimaxArgs exceeds 512B AQL limit");

__global__ void ernie_multimax_kernel(MultimaxArgs A) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int r = (int)blockIdx.y;
    if (i >= A.n || r >= A.rows) return;
    const float r0 = A.rg[0 * A.rgs], r1 = A.rg[1 * A.rgs], r2 = A.rg[2 * A.rgs], r3 = A.rg[3 * A.rgs];
    const float t0 = A.cf[0 * A.cfs], t1 = A.cf[1 * A.cfs], t2 = A.cf[2 * A.cfs], t3 = A.cf[3 * A.cfs];
    const float v = A.x[(long long)r * A.xp + (long long)i * A.xs];
    const float h0 = fmaxf(r0 - v, 0.0f), h1 = fmaxf(v - r1, 0.0f);
    const float h2 = fmaxf(r2 - v, 0.0f), h3 = fmaxf(v - r3, 0.0f);
    const float q2 = h2 * h2, q3 = h3 * h3;
    A.y[(long long)r * A.yp + (long long)i * A.ys] = v + t0 * h0 + t1 * h1 + t2 * q2 + t3 * q3;
}

/* ------------------------------------------------------------------ ernie_row_req */
struct RowReqArgs {
    const int32_t *cu;
    int32_t *req;
    long long cus, rs;
    int S, M;
};
static_assert(sizeof(RowReqArgs) <= 512, "RowReqArgs exceeds 512B AQL limit");

__global__ void ernie_row_req_kernel(RowReqArgs A) {
    const int r = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= A.M) return;
    int lo = 0, hi = A.S;
    while (lo + 1 < hi) {
        const int mid = (lo + hi) >> 1;
        if (A.cu[(long long)mid * A.cus] <= r) lo = mid;
        else hi = mid;
    }
    A.req[(long long)r * A.rs] = lo;
}

/* ------------------------------------------------------------------ ernie_head_slice */
struct HeadSliceArgs {
    const uint16_t *x;
    uint16_t *y;
    long long xp, xs, yp, ys;
    int M, H, W, off, cnt;
};
static_assert(sizeof(HeadSliceArgs) <= 512, "HeadSliceArgs exceeds 512B AQL limit");

__global__ void ernie_head_slice_kernel(HeadSliceArgs A) {
    const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int m = (int)blockIdx.y;
    if (m >= A.M || idx >= A.H * A.cnt) return;
    const int h = idx / A.cnt, j = idx % A.cnt;
    A.y[(long long)m * A.yp + (long long)idx * A.ys] =
        A.x[(long long)m * A.xp + (long long)(h * A.W + A.off + j) * A.xs];
}

/* ------------------------------------------------------------------ ernie_rope_tail */
struct RopeTailArgs {
    uint16_t *x;
    const int32_t *pos;
    long long xp, xs, ps;
    int M, H, W, R, bf16_table, inverse;
    RopeInvTable tab;
    r4d_verdict::Vc vc;   /* this launch's event log and gate word (device memory the library owns; constant across replays) */
};
static_assert(sizeof(RopeTailArgs) <= 512, "RopeTailArgs exceeds 512B AQL limit");

struct RopeTailFast64Args {
    uint16_t *x;
    const int32_t *pos;
    long long xp, xs, ps;
    int M, H, W, R, bf16_table, inverse;
    double inv[32];
    r4d_verdict::Vc vc;   /* this launch's event log and gate word (device memory the library owns; constant across replays) */
};
static_assert(sizeof(RopeTailFast64Args) <= 512, "RopeTailFast64Args exceeds 512B AQL limit");

__global__ void ernie_rope_tail_fast64_kernel(RopeTailFast64Args A) {
    const int half = A.R >> 1;
    const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int m = (int)blockIdx.y;
    if (m >= A.M || idx >= A.H * half) return;
    R4D_OPS_NEG_POS(A, m, idx == 0);
    const int h = idx / half, i = idx % half;
    const double p = (double)A.pos[(long long)m * A.ps];
    float c, s;
    rope_cos_sin_fast(p, A.inv[i], &c, &s);
    if (A.bf16_table) { c = bfr(c); s = bfr(s); }
    if (A.inverse) s = -s;
    const long long o0 = (long long)m * A.xp + (long long)(h * A.W + (A.W - A.R) + 2 * i) * A.xs;
    if (A.xs == 1 && (((uintptr_t)(A.x + o0) & 3u) == 0u)) {
        const uint32_t pair = *(const uint32_t*)(A.x + o0);
        const float x0 = bf2f_lo(pair), x1 = bf2f_hi(pair);
        *(uint32_t*)(A.x + o0) = pack_bf2(x0 * c - x1 * s, x1 * c + x0 * s);
    } else {
        const long long o1 = o0 + A.xs;
        const float x0 = bf2f(A.x[o0]), x1 = bf2f(A.x[o1]);
        A.x[o0] = f2bf(x0 * c - x1 * s);
        A.x[o1] = f2bf(x1 * c + x0 * s);
    }
}

__global__ void ernie_rope_tail_kernel(RopeTailArgs A) {
    const int half = A.R >> 1;
    const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int m = (int)blockIdx.y;
    if (m >= A.M || idx >= A.H * half) return;
    R4D_OPS_NEG_POS(A, m, idx == 0);
    const int h = idx / half, i = idx % half;
    const double p = (double)A.pos[(long long)m * A.ps];
    float c, s;
    rope_cos_sin_fast(p, A.tab.inv[i], &c, &s);
    if (A.bf16_table) { c = bfr(c); s = bfr(s); }
    if (A.inverse) s = -s;
    const long long o0 = (long long)m * A.xp + (long long)(h * A.W + (A.W - A.R) + 2 * i) * A.xs;
    const long long o1 = o0 + A.xs;
    const float x0 = bf2f(A.x[o0]), x1 = bf2f(A.x[o1]);
    A.x[o0] = f2bf(x0 * c - x1 * s);
    A.x[o1] = f2bf(x1 * c + x0 * s);
}

/* ------------------------------------------------------------------ ernie_hca_q_prep */
struct HcaQPrepArgs {
    uint16_t *x;
    const int32_t *pos;
    long long xp, xs, ps;
    int M, H, D, R;
    float eps;
    RopeInvTable tab;
    r4d_verdict::Vc vc;   /* this launch's event log and gate word (device memory the library owns; constant across replays) */
};
static_assert(sizeof(HcaQPrepArgs) <= 512, "HcaQPrepArgs exceeds 512B AQL limit");

struct HcaQPrepFast64Args {
    uint16_t *x;
    const int32_t *pos;
    long long xp, ps;
    int M, H;
    float eps;
    double inv[32];
    r4d_verdict::Vc vc;   /* this launch's event log and gate word (device memory the library owns; constant across replays) */
};
static_assert(sizeof(HcaQPrepFast64Args) <= 512, "HcaQPrepFast64Args exceeds 512B AQL limit");

/* Fast D == 512, R == 64 wave32 register-resident kernel:
 * 4 waves per block (1 wave per head), 0 LDS, 0 __syncthreads(), single uint4 R/W pass, host-precomputed inv[32]. */
__global__ void ernie_hca_q_prep_fast512_kernel(HcaQPrepFast64Args A) {
    const int lane = (int)(threadIdx.x & 31);
    const int wave = (int)(threadIdx.x >> 5);
    const int h = (int)blockIdx.x * (int)(blockDim.x >> 5) + wave;
    const int m = (int)blockIdx.y;
    if (h >= A.H || m >= A.M) return;
    R4D_OPS_NEG_POS(A, m, h == 0 && lane == 0);

    uint16_t* row = A.x + (long long)m * A.xp + (long long)(h * 512);
    const uint4 u0 = *(const uint4*)(row + lane * 8);
    const uint4 u1 = *(const uint4*)(row + 256 + lane * 8);

    float v0[8], v1[8];
    v0[0] = bf2f_lo(u0.x); v0[1] = bf2f_hi(u0.x);
    v0[2] = bf2f_lo(u0.y); v0[3] = bf2f_hi(u0.y);
    v0[4] = bf2f_lo(u0.z); v0[5] = bf2f_hi(u0.z);
    v0[6] = bf2f_lo(u0.w); v0[7] = bf2f_hi(u0.w);
    v1[0] = bf2f_lo(u1.x); v1[1] = bf2f_hi(u1.x);
    v1[2] = bf2f_lo(u1.y); v1[3] = bf2f_hi(u1.y);
    v1[4] = bf2f_lo(u1.z); v1[5] = bf2f_hi(u1.z);
    v1[6] = bf2f_lo(u1.w); v1[7] = bf2f_hi(u1.w);

    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        ss = fmaf(v0[i], v0[i], ss);
        ss = fmaf(v1[i], v1[i], ss);
    }
    const float tot = r4d_xor_sum<32>(ss);
    const float rinv = rsqrtf(tot * (1.0f / 512.0f) + A.eps);

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        v0[i] *= rinv;
        v1[i] *= rinv;
    }

    uint4 o0;
    o0.x = pack_bf2(v0[0], v0[1]);
    o0.y = pack_bf2(v0[2], v0[3]);
    o0.z = pack_bf2(v0[4], v0[5]);
    o0.w = pack_bf2(v0[6], v0[7]);
    *(uint4*)(row + lane * 8) = o0;

    /* Lanes 24..31 own trailing 64 elements (448..511) = 32 contiguous GPT-J RoPE pairs (4 pairs/lane). */
    if (lane >= 24) {
        const double p = (double)A.pos[(long long)m * A.ps];
        const int p0 = (lane - 24) * 4;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            float cs, sn;
            rope_cos_sin_fast(p, A.inv[p0 + k], &cs, &sn);
            const float ev = v1[2 * k], od = v1[2 * k + 1];
            v1[2 * k]     = ev * cs - od * sn;
            v1[2 * k + 1] = ev * sn + od * cs;
        }
    }

    uint4 o1;
    o1.x = pack_bf2(v1[0], v1[1]);
    o1.y = pack_bf2(v1[2], v1[3]);
    o1.z = pack_bf2(v1[4], v1[5]);
    o1.w = pack_bf2(v1[6], v1[7]);
    *(uint4*)(row + 256 + lane * 8) = o1;
}

/* Fast D == 128, R == 64 wave32 register-resident kernel (0 LDS, 0 barriers, single uint2 R/W pass). */
__global__ void ernie_hca_q_prep_fast128_kernel(HcaQPrepFast64Args A) {
    const int lane = (int)(threadIdx.x & 31);
    const int wave = (int)(threadIdx.x >> 5);
    const int h = (int)blockIdx.x * (int)(blockDim.x >> 5) + wave;
    const int m = (int)blockIdx.y;
    if (h >= A.H || m >= A.M) return;
    R4D_OPS_NEG_POS(A, m, h == 0 && lane == 0);

    uint16_t* row = A.x + (long long)m * A.xp + (long long)(h * 128);
    const uint2 u = *(const uint2*)(row + lane * 4);
    float v[4];
    v[0] = bf2f_lo(u.x); v[1] = bf2f_hi(u.x);
    v[2] = bf2f_lo(u.y); v[3] = bf2f_hi(u.y);

    const float ss = fmaf(v[0], v[0], fmaf(v[1], v[1], fmaf(v[2], v[2], v[3] * v[3])));
    const float tot = r4d_xor_sum<32>(ss);
    const float rinv = rsqrtf(tot * (1.0f / 128.0f) + A.eps);

#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] *= rinv;

    /* Lanes 16..31 own trailing 64 elements (64..127) = 32 contiguous GPT-J RoPE pairs (2 pairs/lane). */
    if (lane >= 16) {
        const double p = (double)A.pos[(long long)m * A.ps];
        const int p0 = (lane - 16) * 2;
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            float cs, sn;
            rope_cos_sin_fast(p, A.inv[p0 + k], &cs, &sn);
            const float ev = v[2 * k], od = v[2 * k + 1];
            v[2 * k]     = ev * cs - od * sn;
            v[2 * k + 1] = ev * sn + od * cs;
        }
    }

    uint2 o;
    o.x = pack_bf2(v[0], v[1]);
    o.y = pack_bf2(v[2], v[3]);
    *(uint2*)(row + lane * 4) = o;
}

__global__ void ernie_hca_q_prep_kernel(HcaQPrepArgs A) {
    __shared__ float smem[8];
    const int h = (int)blockIdx.x, m = (int)blockIdx.y, tid = (int)threadIdx.x;
    if (h >= A.H || m >= A.M) return;
    R4D_OPS_NEG_POS(A, m, h == 0 && tid == 0);
    const long long base = (long long)m * A.xp + (long long)(h * A.D) * A.xs;
    float ss = 0.0f;
    for (int c = tid; c < A.D; c += 256) {
        const float v = bf2f(A.x[base + (long long)c * A.xs]);
        ss += v * v;
    }
    const float tot = block_sum(ss, smem, 256);
    const float rinv = 1.0f / sqrtf(tot / (float)A.D + A.eps);
    const int nope = A.D - A.R;
    for (int c = tid; c < nope; c += 256) {
        const float v = bf2f(A.x[base + (long long)c * A.xs]) * rinv;
        A.x[base + (long long)c * A.xs] = f2bf(v);
    }
    const double p = (double)A.pos[(long long)m * A.ps];
    const int half = A.R >> 1;
    for (int i = tid; i < half; i += 256) {
        const long long o0 = base + (long long)(nope + 2 * i) * A.xs;
        const long long o1 = o0 + A.xs;
        const float ev = bf2f(A.x[o0]) * rinv;
        const float od = bf2f(A.x[o1]) * rinv;
        float cs, sn;
        rope_cos_sin_fast(p, A.tab.inv[i], &cs, &sn);
        A.x[o0] = f2bf(ev * cs - od * sn);
        A.x[o1] = f2bf(ev * sn + od * cs);
    }
}

/* ------------------------------------------------------------------ ernie_vha_mix */
struct VhaMixArgs {
    const uint16_t *o;
    const void *u, *v;
    uint16_t *out;
    long long op, os, up, us, vp, vs, outp, outs;
    int M, H, Hloc, h0, D, R, u_f32, v_f32, fold_bf16;
};
static_assert(sizeof(VhaMixArgs) <= 512, "VhaMixArgs exceeds 512B AQL limit");

__device__ __forceinline__ float vha_mix_coeff(const VhaMixArgs& A, int h, int g) {
    float s = 0.0f;
    for (int r = 0; r < A.R; ++r) {
        const long long vi = (long long)h * A.vp + (long long)r * A.vs;
        const long long ui = (long long)g * A.up + (long long)r * A.us;
        const float vv = A.v_f32 ? ((const float*)A.v)[vi] : bf2f(((const uint16_t*)A.v)[vi]);
        const float uu = A.u_f32 ? ((const float*)A.u)[ui] : bf2f(((const uint16_t*)A.u)[ui]);
        s += vv * uu;
    }
    const float e = (h == g ? 1.0f : 0.0f) + s;
    return A.fold_bf16 ? bfr(e) : e;
}

static constexpr int kVhaMixU4Threads = 64;
static constexpr int kVhaMixU4Heads = 2;

/* Decode: each block covers 512 columns for two output heads with 16 FP32 accumulators per lane. */
template <int HLOC>
__global__ __launch_bounds__(kVhaMixU4Threads) void ernie_vha_mix_u4_kernel(VhaMixArgs A) {
    constexpr int HB = kVhaMixU4Heads, NTHR = kVhaMixU4Threads;
    constexpr int MaxHeads = HLOC * 64; /* Host validation limits world_size to 64. */
    __shared__ float mmat[HB * MaxHeads];
    const int tid = (int)threadIdx.x, m = (int)blockIdx.y;
    const int hh0 = (int)blockIdx.x % (HLOC / HB) * HB;
    const int c0 = ((int)blockIdx.x / (HLOC / HB) * NTHR + tid) * 8;
    const int H = A.H;

    for (int i = tid; i < HB * H; i += NTHR)
        mmat[i] = vha_mix_coeff(A, A.h0 + hh0 + i / H, i % H);
    __syncthreads();

    if (c0 >= A.D) return;
    float acc[HB][8] = {};
    for (int g0 = 0; g0 < H; g0 += HLOC) {
        const long long row = (long long)(g0 / HLOC) * A.M + m;
        const uint16_t* orow = A.o + row * A.op + c0;
        for (int gh = 0; gh < HLOC; ++gh) {
            const uint4 ov = *reinterpret_cast<const uint4*>(orow + (long long)gh * A.D);
            const uint32_t words[4] = {ov.x, ov.y, ov.z, ov.w};
#pragma unroll
            for (int hh = 0; hh < HB; ++hh) {
                const float w = mmat[hh * H + g0 + gh];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    acc[hh][2 * e]     += w * bf2f_lo(words[e]);
                    acc[hh][2 * e + 1] += w * bf2f_hi(words[e]);
                }
            }
        }
    }

    uint16_t* yrow = A.out + (long long)m * A.outp + c0;
#pragma unroll
    for (int hh = 0; hh < HB; ++hh) {
        *reinterpret_cast<uint4*>(yrow + (long long)(hh0 + hh) * A.D) =
            make_uint4(pack_bf2(acc[hh][0], acc[hh][1]), pack_bf2(acc[hh][2], acc[hh][3]),
                       pack_bf2(acc[hh][4], acc[hh][5]), pack_bf2(acc[hh][6], acc[hh][7]));
    }
}

/* Fast multi-head VHA mix kernel for H <= 128, Hloc in {4, 8, 16}, D even, unit inner strides:
 * - Computes ALL Hloc rows of M = I + V @ U^T ONCE in small LDS (2 KB for live Hloc=8, H=64),
 *   reused across TM tokens per block!
 * - Reads each input head o[g, c..c+1] ONCE via coalesced uint32_t load and broadcasts across all Hloc
 *   output head accumulators in registers (8x less global memory traffic for live Hloc=8!).
 * - Exact summation order and exact fold_bf16 rounding -> bit-exact parity with libref! */
template <int HLOC, int TM, int NTHR>
__global__ void ernie_vha_mix_fast_kernel(VhaMixArgs A) {
    __shared__ float mmat[HLOC * 128];
    const int tid = (int)threadIdx.x;
    const int c0 = ((int)blockIdx.x * NTHR + tid) * 2;
    const int m0 = (int)blockIdx.y * TM;
    if (m0 >= A.M) return;

    const int H = A.H;
    const int total_m = HLOC * H;
    for (int idx = tid; idx < total_m; idx += NTHR) {
        const int hh = idx / H;
        const int g = idx - hh * H;
        mmat[idx] = vha_mix_coeff(A, A.h0 + hh, g);
    }
    __syncthreads();

    if (c0 >= A.D) return;
    const bool aligned_u32 = (((uintptr_t)A.o & 3u) == 0u) && (((uintptr_t)A.out & 3u) == 0u) &&
                             ((A.op & 1) == 0) && ((A.outp & 1) == 0);

    for (int tm = 0; tm < TM; ++tm) {
        const int m = m0 + tm;
        if (m >= A.M) break;

        float acc0[HLOC], acc1[HLOC];
#pragma unroll
        for (int hh = 0; hh < HLOC; ++hh) {
            acc0[hh] = 0.0f;
            acc1[hh] = 0.0f;
        }

        int p_rk = 0, gh = 0;
        for (int g = 0; g < H; ++g) {
            const long long row_idx = (long long)p_rk * A.M + m;
            const uint16_t* op_ptr = A.o + row_idx * A.op + (long long)(gh * A.D + c0);
            float o0, o1;
            if (aligned_u32) {
                const uint32_t u = *(const uint32_t*)op_ptr;
                o0 = bf2f_lo(u);
                o1 = bf2f_hi(u);
            } else {
                o0 = bf2f(op_ptr[0]);
                o1 = bf2f(op_ptr[1]);
            }
#pragma unroll
            for (int hh = 0; hh < HLOC; ++hh) {
                const float w = mmat[hh * H + g];
                acc0[hh] += w * o0;
                acc1[hh] += w * o1;
            }
            if (++gh == HLOC) { gh = 0; ++p_rk; }
        }

        uint16_t* yrow = A.out + (long long)m * A.outp;
#pragma unroll
        for (int hh = 0; hh < HLOC; ++hh) {
            uint16_t* yp = yrow + (long long)(hh * A.D + c0);
            if (aligned_u32) {
                *(uint32_t*)yp = pack_bf2(acc0[hh], acc1[hh]);
            } else {
                yp[0] = f2bf(acc0[hh]);
                yp[1] = f2bf(acc1[hh]);
            }
        }
    }
}

__global__ void ernie_vha_mix_kernel(VhaMixArgs A) {
    /* General fallback: one block per (m, hh); computes global row h = h0 + hh of M_mat = I + V @ U^T in LDS (up to H=4096). */
    __shared__ float mrow[4096];
    const int hh = (int)blockIdx.x, m = (int)blockIdx.y, tid = (int)threadIdx.x;
    if (hh >= A.Hloc || m >= A.M) return;
    const int h = A.h0 + hh;
    for (int g = tid; g < A.H; g += 256) mrow[g] = vha_mix_coeff(A, h, g);
    __syncthreads();
    const long long ybase = (long long)m * A.outp + (long long)(hh * A.D) * A.outs;
    for (int c = tid; c < A.D; c += 256) {
        float sum = 0.0f;
        for (int g = 0; g < A.H; ++g) {
            const int p_rk = g / A.Hloc, gh = g % A.Hloc;
            const long long row_idx = (long long)p_rk * A.M + m;
            sum += mrow[g] * bf2f(A.o[row_idx * A.op + (long long)(gh * A.D + c) * A.os]);
        }
        A.out[ybase + (long long)c * A.outs] = f2bf(sum);
    }
}


/* ------------------------------------------------------------------ ernie_vha_proj */
struct VhaProjArgs {
    const uint16_t *o;
    const void *u;
    uint16_t *z;
    long long op, os, up, us, zp, zs;
    int M, H, Hloc, h0, D, R, u_f32;
};
static_assert(sizeof(VhaProjArgs) <= 512, "VhaProjArgs exceeds 512B AQL limit");

/* Fast vectorized 16-byte (8x bf16) VHA low-rank projection kernel for unit inner strides,
 * 16B-aligned o/z, (D & 7) == 0, Hloc <= 16, R <= 32.
 * - Stages U_loc[Hloc][R] in LDS (<= 2 KiB), reused across TM tokens per block.
 * - Streams hh = 0..Hloc-1 loading 1 r4d_u32x4 (16B coalesced) per head into registers and
 *   accumulating RB rank outputs in f32, then stores RB coalesced r4d_u32x4s to z.
 * - When D_LANES > 0 (e.g. D_LANES=16 for D=128), packs NTHR/D_LANES rank groups inside one
 *   block so all wave32 lanes stay 100% active and share O_loc in L1.
 * - Exact hh = 0..Hloc-1 summation order + bf16x2_rne_nan -> bit-exact parity with libref. */
template <int RB, int D_LANES, int TM, int NTHR>
__global__ void ernie_vha_proj_fast_kernel(VhaProjArgs A) {
    __shared__ float s_u[16 * 32];
    const int tid = (int)threadIdx.x;
    const int m0 = (int)blockIdx.z * TM;
    if (m0 >= A.M) return;

    const int total_u = A.Hloc * A.R;
    for (int idx = tid; idx < total_u; idx += NTHR) {
        const int hh = idx / A.R;
        const int r = idx - hh * A.R;
        const int h = A.h0 + hh;
        const long long off = (long long)h * A.up + (long long)r * A.us;
        s_u[hh * 32 + r] = A.u_f32 ? ((const float*)A.u)[off]
                                   : bf2f(((const uint16_t*)A.u)[off]);
    }
    __syncthreads();

    int c0, r0;
    if constexpr (D_LANES > 0) {
        constexpr int R_LANES = NTHR / D_LANES;
        c0 = (tid & (D_LANES - 1)) * 8;
        r0 = ((int)blockIdx.x * R_LANES + (tid / D_LANES)) * RB;
    } else {
        c0 = ((int)blockIdx.y * NTHR + tid) * 8;
        r0 = (int)blockIdx.x * RB;
    }
    if (c0 >= A.D || r0 >= A.R) return;

    const bool full_rb = (r0 + RB <= A.R);
    for (int tm = 0; tm < TM; ++tm) {
        const int m = m0 + tm;
        if (m >= A.M) break;

        float acc[RB][8];
#pragma unroll
        for (int rb = 0; rb < RB; ++rb) {
#pragma unroll
            for (int k = 0; k < 8; ++k) acc[rb][k] = 0.0f;
        }

        const uint16_t* obase = A.o + (long long)m * A.op + c0;
        for (int hh = 0; hh < A.Hloc; ++hh) {
            const r4d_u32x4 v = *(const r4d_u32x4*)(obase + (long long)hh * A.D);
            const float ov0 = bf16w_lo(v[0]), ov1 = bf16w_hi(v[0]);
            const float ov2 = bf16w_lo(v[1]), ov3 = bf16w_hi(v[1]);
            const float ov4 = bf16w_lo(v[2]), ov5 = bf16w_hi(v[2]);
            const float ov6 = bf16w_lo(v[3]), ov7 = bf16w_hi(v[3]);
            const float* urow = &s_u[hh * 32 + r0];
            if (full_rb) {
#pragma unroll
                for (int rb = 0; rb < RB; ++rb) {
                    const float w = urow[rb];
                    acc[rb][0] += ov0 * w; acc[rb][1] += ov1 * w;
                    acc[rb][2] += ov2 * w; acc[rb][3] += ov3 * w;
                    acc[rb][4] += ov4 * w; acc[rb][5] += ov5 * w;
                    acc[rb][6] += ov6 * w; acc[rb][7] += ov7 * w;
                }
            } else {
#pragma unroll
                for (int rb = 0; rb < RB; ++rb) {
                    if (r0 + rb < A.R) {
                        const float w = urow[rb];
                        acc[rb][0] += ov0 * w; acc[rb][1] += ov1 * w;
                        acc[rb][2] += ov2 * w; acc[rb][3] += ov3 * w;
                        acc[rb][4] += ov4 * w; acc[rb][5] += ov5 * w;
                        acc[rb][6] += ov6 * w; acc[rb][7] += ov7 * w;
                    }
                }
            }
        }

        uint16_t* zbase = A.z + (long long)m * A.zp + c0;
#pragma unroll
        for (int rb = 0; rb < RB; ++rb) {
            if (full_rb || r0 + rb < A.R) {
                r4d_u32x4 outv;
                outv[0] = bf16x2_rne_nan(acc[rb][0], acc[rb][1]);
                outv[1] = bf16x2_rne_nan(acc[rb][2], acc[rb][3]);
                outv[2] = bf16x2_rne_nan(acc[rb][4], acc[rb][5]);
                outv[3] = bf16x2_rne_nan(acc[rb][6], acc[rb][7]);
                *(r4d_u32x4*)(zbase + (long long)(r0 + rb) * A.D) = outv;
            }
        }
    }
}

__global__ void ernie_vha_proj_kernel(VhaProjArgs A) {
    /* General fallback for arbitrary Hloc, R, D, and non-contiguous strides. */
    __shared__ float ucol[1024];
    const int m = (int)blockIdx.x, tid = (int)threadIdx.x;
    if (m >= A.M) return;
    const bool use_lds = (A.Hloc <= 1024);
    const long long orow = (long long)m * A.op;
    const long long zrow = (long long)m * A.zp;
    for (int r = (int)blockIdx.y; r < A.R; r += (int)gridDim.y) {
        if (use_lds) {
            for (int hh = tid; hh < A.Hloc; hh += 256) {
                const long long off = (long long)(A.h0 + hh) * A.up + (long long)r * A.us;
                ucol[hh] = A.u_f32 ? ((const float*)A.u)[off] : bf2f(((const uint16_t*)A.u)[off]);
            }
            __syncthreads();
        }
        for (int c = tid; c < A.D; c += 256) {
            float sum = 0.0f;
            for (int hh = 0; hh < A.Hloc; ++hh) {
                const float uu = use_lds
                    ? ucol[hh]
                    : (A.u_f32 ? ((const float*)A.u)[(long long)(A.h0 + hh) * A.up + (long long)r * A.us]
                               : bf2f(((const uint16_t*)A.u)[(long long)(A.h0 + hh) * A.up + (long long)r * A.us]));
                sum += bf2f(A.o[orow + (long long)(hh * A.D + c) * A.os]) * uu;
            }
            A.z[zrow + (long long)(r * A.D + c) * A.zs] = f2bf(sum);
        }
        if (use_lds) __syncthreads();
    }
}

/* ------------------------------------------------------------------ ernie_vha_apply */
struct VhaApplyArgs {
    const uint16_t *o, *z;
    const void *v;
    uint16_t *out;
    long long op, os, zp, zs, vp, vs, outp, outs;
    int M, H, Hloc, h0, D, R, v_f32;
};
static_assert(sizeof(VhaApplyArgs) <= 512, "VhaApplyArgs exceeds 512B AQL limit");

/* Fast vectorized 16-byte (8x bf16) VHA low-rank reconstruction + residual kernel for unit inner
 * strides, 16B-aligned o/z/out, (D & 7) == 0, Hloc <= 16, R <= 32.
 * - Stages V_loc[R][Hloc] in LDS (<= 2 KiB, transposed for contiguous head reads), reused across TM tokens.
 * - Streams r = 0..R-1 loading 1 r4d_u32x4 (16B coalesced) of z per rank and accumulating HB heads
 *   in f32; then loads 1 r4d_u32x4 of o per head, adds residual, packs and stores to out.
 * - Exact r = 0..R-1 summation order + (o + sum) + bf16x2_rne_nan -> bit-exact parity with libref. */
template <int HB, int D_LANES, int TM, int NTHR>
__global__ void ernie_vha_apply_fast_kernel(VhaApplyArgs A) {
    __shared__ float s_v[32 * 16];
    const int tid = (int)threadIdx.x;
    const int m0 = (int)blockIdx.z * TM;
    if (m0 >= A.M) return;

    const int total_v = A.Hloc * A.R;
    for (int idx = tid; idx < total_v; idx += NTHR) {
        const int hh = idx / A.R;
        const int r = idx - hh * A.R;
        const int h = A.h0 + hh;
        const long long off = (long long)h * A.vp + (long long)r * A.vs;
        s_v[r * 16 + hh] = A.v_f32 ? ((const float*)A.v)[off]
                                   : bf2f(((const uint16_t*)A.v)[off]);
    }
    __syncthreads();

    int c0, hh0;
    if constexpr (D_LANES > 0) {
        constexpr int H_LANES = NTHR / D_LANES;
        c0 = (tid & (D_LANES - 1)) * 8;
        hh0 = ((int)blockIdx.x * H_LANES + (tid / D_LANES)) * HB;
    } else {
        c0 = ((int)blockIdx.y * NTHR + tid) * 8;
        hh0 = (int)blockIdx.x * HB;
    }
    if (c0 >= A.D || hh0 >= A.Hloc) return;

    const bool full_hb = (hh0 + HB <= A.Hloc);
    for (int tm = 0; tm < TM; ++tm) {
        const int m = m0 + tm;
        if (m >= A.M) break;

        float acc[HB][8];
#pragma unroll
        for (int hb = 0; hb < HB; ++hb) {
#pragma unroll
            for (int k = 0; k < 8; ++k) acc[hb][k] = 0.0f;
        }

        const uint16_t* zbase = A.z + (long long)m * A.zp + c0;
        for (int r = 0; r < A.R; ++r) {
            const r4d_u32x4 zv = *(const r4d_u32x4*)(zbase + (long long)r * A.D);
            const float z0 = bf16w_lo(zv[0]), z1 = bf16w_hi(zv[0]);
            const float z2 = bf16w_lo(zv[1]), z3 = bf16w_hi(zv[1]);
            const float z4 = bf16w_lo(zv[2]), z5 = bf16w_hi(zv[2]);
            const float z6 = bf16w_lo(zv[3]), z7 = bf16w_hi(zv[3]);
            const float* vrow = &s_v[r * 16 + hh0];
            if (full_hb) {
#pragma unroll
                for (int hb = 0; hb < HB; ++hb) {
                    const float w = vrow[hb];
                    acc[hb][0] += z0 * w; acc[hb][1] += z1 * w;
                    acc[hb][2] += z2 * w; acc[hb][3] += z3 * w;
                    acc[hb][4] += z4 * w; acc[hb][5] += z5 * w;
                    acc[hb][6] += z6 * w; acc[hb][7] += z7 * w;
                }
            } else {
#pragma unroll
                for (int hb = 0; hb < HB; ++hb) {
                    if (hh0 + hb < A.Hloc) {
                        const float w = vrow[hb];
                        acc[hb][0] += z0 * w; acc[hb][1] += z1 * w;
                        acc[hb][2] += z2 * w; acc[hb][3] += z3 * w;
                        acc[hb][4] += z4 * w; acc[hb][5] += z5 * w;
                        acc[hb][6] += z6 * w; acc[hb][7] += z7 * w;
                    }
                }
            }
        }

        const uint16_t* obase = A.o + (long long)m * A.op + c0;
        uint16_t* ybase = A.out + (long long)m * A.outp + c0;
#pragma unroll
        for (int hb = 0; hb < HB; ++hb) {
            if (full_hb || hh0 + hb < A.Hloc) {
                const long long hoff = (long long)(hh0 + hb) * A.D;
                const r4d_u32x4 ov = *(const r4d_u32x4*)(obase + hoff);
                r4d_u32x4 outv;
                outv[0] = bf16x2_rne_nan(bf16w_lo(ov[0]) + acc[hb][0], bf16w_hi(ov[0]) + acc[hb][1]);
                outv[1] = bf16x2_rne_nan(bf16w_lo(ov[1]) + acc[hb][2], bf16w_hi(ov[1]) + acc[hb][3]);
                outv[2] = bf16x2_rne_nan(bf16w_lo(ov[2]) + acc[hb][4], bf16w_hi(ov[2]) + acc[hb][5]);
                outv[3] = bf16x2_rne_nan(bf16w_lo(ov[3]) + acc[hb][6], bf16w_hi(ov[3]) + acc[hb][7]);
                *(r4d_u32x4*)(ybase + hoff) = outv;
            }
        }
    }
}

__global__ void ernie_vha_apply_kernel(VhaApplyArgs A) {
    /* General fallback for arbitrary Hloc, R, D, and non-contiguous strides. */
    __shared__ float vrow[1024];
    const int m = (int)blockIdx.x, hh = (int)blockIdx.y, tid = (int)threadIdx.x;
    if (m >= A.M || hh >= A.Hloc) return;
    const int h = A.h0 + hh;
    const bool use_lds = (A.R <= 1024);
    if (use_lds) {
        for (int r = tid; r < A.R; r += 256) {
            const long long off = (long long)h * A.vp + (long long)r * A.vs;
            vrow[r] = A.v_f32 ? ((const float*)A.v)[off] : bf2f(((const uint16_t*)A.v)[off]);
        }
        __syncthreads();
    }
    const long long obase = (long long)m * A.op + (long long)(hh * A.D) * A.os;
    const long long ybase = (long long)m * A.outp + (long long)(hh * A.D) * A.outs;
    const long long zrow = (long long)m * A.zp;
    for (int c = tid; c < A.D; c += 256) {
        float sum = 0.0f;
        for (int r = 0; r < A.R; ++r) {
            const float vv = use_lds
                ? vrow[r]
                : (A.v_f32 ? ((const float*)A.v)[(long long)h * A.vp + (long long)r * A.vs]
                           : bf2f(((const uint16_t*)A.v)[(long long)h * A.vp + (long long)r * A.vs]));
            sum += bf2f(A.z[zrow + (long long)(r * A.D + c) * A.zs]) * vv;
        }
        const float ov = bf2f(A.o[obase + (long long)c * A.os]);
        A.out[ybase + (long long)c * A.outs] = f2bf(ov + sum);
    }
}

} /* namespace r4d_ernie_ops */

extern "C" {

int r4d_rad_ernie_multimax(const RadArgs* a, RadStream stream) {
    using namespace r4d_ernie;
    using namespace r4d_ernie_ops;
    if (!a || !a->t || a->n_t < 4) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &rg = a->t[1], &cf = a->t[2], &y = a->t[3];
    if (x.dtype != RAD_F32 || rg.dtype != RAD_F32 || cf.dtype != RAD_F32 || y.dtype != RAD_F32) return RAD_E_DTYPE;
    if (!x.data || !rg.data || !cf.data || !y.data) return RAD_E_INVAL;
    const long long band = r4d_geti_or(a, "M", 0);
    const long long n = r4d_geti_or(a, "n", y.rank ? y.shape[y.rank - 1] : 0);
    if (band <= 0 || n <= 0 || n > INT_MAX) return RAD_E_SHAPE;
    if (numel_t(rg) != 4 || numel_t(cf) != 4) return RAD_E_SHAPE;
    if (x.rank < 1 || y.rank < 1 || !pos_strides(x) || !pos_strides(y) || !pos_strides(rg) || !pos_strides(cf))
        return RAD_E_SHAPE;
    if (x.shape[x.rank - 1] != n || y.shape[y.rank - 1] != n) return RAD_E_SHAPE;
    const long long rows = (x.rank == 1) ? 1 : (numel_t(x) / n);
    if (rows <= 0 || rows > band || rows > 65535 || numel_t(y) < rows * n) return RAD_E_SHAPE;
    R4D_ARGS(MultimaxArgs, P);
    P.x = (const float*)x.data; P.rg = (const float*)rg.data; P.cf = (const float*)cf.data; P.y = (float*)y.data;
    P.xp = (x.rank >= 2) ? x.stride[x.rank - 2] : n; P.xs = x.stride[x.rank - 1];
    P.yp = (y.rank >= 2) ? y.stride[y.rank - 2] : n; P.ys = y.stride[y.rank - 1];
    P.rgs = rg.stride[rg.rank - 1]; P.cfs = cf.stride[cf.rank - 1];
    P.rows = (int)rows; P.n = (int)n;
    R4D_KLAUNCH(R4D_KNAME(ernie_multimax_kernel), dim3((unsigned)((n + 255) / 256), (unsigned)rows), dim3(256), (hipStream_t)stream, P);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

int r4d_rad_ernie_row_req(const RadArgs* a, RadStream stream) {
    using namespace r4d_ernie;
    using namespace r4d_ernie_ops;
    if (!a || !a->t || a->n_t < 2) return RAD_E_INVAL;
    const RadTensor &cu = a->t[0], &req = a->t[1];
    if (cu.rank != 1 || req.rank != 1 || cu.shape[0] < 2 || req.shape[0] < 1 || !pos_strides(cu) || !pos_strides(req))
        return RAD_E_SHAPE;
    if (cu.dtype != RAD_I32 || req.dtype != RAD_I32) return RAD_E_DTYPE;
    if (!cu.data || !req.data) return RAD_E_INVAL;
    const long long S = cu.shape[0] - 1, M = req.shape[0];
    if (S > INT_MAX || M > INT_MAX) return RAD_E_SHAPE;
#if !defined(__HIP__) && !defined(__HIP_PLATFORM_AMD__)
    const int32_t* cp = (const int32_t*)cu.data;
    int64_t prev = cp[0];
    if (prev != 0) return RAD_E_SHAPE;
    for (long long s = 1; s <= S; ++s) {
        const int64_t c = cp[s * cu.stride[0]];
        if (c < prev) return RAD_E_SHAPE;
        prev = c;
    }
    if (prev != M) return RAD_E_SHAPE;
#endif
    R4D_ARGS(RowReqArgs, P);
    P.cu = (const int32_t*)cu.data; P.req = (int32_t*)req.data;
    P.cus = cu.stride[0]; P.rs = req.stride[0]; P.S = (int)S; P.M = (int)M;
    R4D_KLAUNCH(R4D_KNAME(ernie_row_req_kernel), dim3((unsigned)((M + 255) / 256)), dim3(256), (hipStream_t)stream, P);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

int r4d_rad_ernie_head_slice(const RadArgs* a, RadStream stream) {
    using namespace r4d_ernie;
    using namespace r4d_ernie_ops;
    if (!a || !a->t || a->n_t < 2) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &y = a->t[1];
    const long long H = r4d_geti_or(a, "heads", 0), W = r4d_geti_or(a, "width", 0),
                    off = r4d_geti_or(a, "offset", -1), cnt = r4d_geti_or(a, "count", 0);
    if (H < 1 || W < 1 || off < 0 || cnt < 1 || off + cnt > W || x.rank != 2 || y.rank != 2 || x.shape[0] < 1 ||
        x.shape[0] > 65535 || x.shape[1] != H * W || y.shape[0] != x.shape[0] || y.shape[1] != H * cnt ||
        !pos_strides(x) || !pos_strides(y) || x.stride[0] < x.shape[1] * x.stride[1] || y.stride[0] < y.shape[1] * y.stride[1])
        return RAD_E_SHAPE;
    if (x.dtype != RAD_BF16 || y.dtype != RAD_BF16) return RAD_E_DTYPE;
    if (!x.data || !y.data) return RAD_E_INVAL;
    if (operands_overlap(x, y)) return RAD_E_INVAL;
    R4D_ARGS(HeadSliceArgs, P);
    P.x = (const uint16_t*)x.data; P.y = (uint16_t*)y.data;
    P.xp = x.stride[0]; P.xs = x.stride[1]; P.yp = y.stride[0]; P.ys = y.stride[1];
    P.M = (int)x.shape[0]; P.H = (int)H; P.W = (int)W; P.off = (int)off; P.cnt = (int)cnt;
    R4D_KLAUNCH(R4D_KNAME(ernie_head_slice_kernel), dim3((unsigned)((H * cnt + 255) / 256), (unsigned)P.M), dim3(256), (hipStream_t)stream, P);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

int r4d_rad_ernie_rope_tail(const RadArgs* a, RadStream stream) {
    using namespace r4d_ernie;
    using namespace r4d_ernie_ops;
    if (!a || !a->t || a->n_t < 2) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &pos = a->t[1];
    const long long H = r4d_geti_or(a, "heads", 0), W = r4d_geti_or(a, "width", 0), R = r4d_geti_or(a, "rope_dim", 0);
    const double theta = r4d_getf(a, "theta", 0.0);
    if (H < 1 || W < 1 || R < 2 || (R & 1) || R > W || R > 2 * kMaxRopePairs || !std::isfinite(theta) || theta <= 0.0 ||
        x.rank != 2 || pos.rank != 1 || x.shape[0] < 1 || x.shape[0] > 65535 || x.shape[1] != H * W ||
        pos.shape[0] != x.shape[0] || !pos_strides(x) || !pos_strides(pos) || x.stride[0] < x.shape[1] * x.stride[1])
        return RAD_E_SHAPE;
    if (x.dtype != RAD_BF16 || pos.dtype != RAD_I32) return RAD_E_DTYPE;
    if (!x.data || !pos.data) return RAD_E_INVAL;
    const char* table = r4d_gets(a, "rope_table");
    const bool bf16_table = table && !std::strcmp(table, "bf16");
    if (table && !bf16_table && std::strcmp(table, "f32")) return RAD_E_SHAPE;
    const long long inverse = r4d_geti_or(a, "inverse", 0);
    ErnieYarn yarn;
    if ((inverse != 0 && inverse != 1) || !parse_yarn(a, &yarn)) return RAD_E_SHAPE;
    RopeInvTable tab{};
    if (!fill_rope_inv(theta, R, yarn, &tab)) return RAD_E_SHAPE;
#if !defined(__HIP__) && !defined(__HIP_PLATFORM_AMD__)
    if (!r4d_verdict::card_form())
    for (long long m = 0; m < x.shape[0]; ++m)
        if (((const int32_t*)pos.data)[m * pos.stride[0]] < 0) return RAD_E_SHAPE;
#endif
    const long long total = H * (R / 2);
    if (R <= 64) {
        R4D_ARGS(RopeTailFast64Args, F);
        R4D_VC(F, r4d_verdict::kFamOps);
        F.x = (uint16_t*)x.data; F.pos = (const int32_t*)pos.data;
        F.xp = x.stride[0]; F.xs = x.stride[1]; F.ps = pos.stride[0];
        F.M = (int)x.shape[0]; F.H = (int)H; F.W = (int)W; F.R = (int)R;
        F.bf16_table = bf16_table ? 1 : 0; F.inverse = (int)inverse;
        for (int i = 0; i < (int)(R / 2); ++i) F.inv[i] = tab.inv[i];
        R4D_KLAUNCH(R4D_KNAME(ernie_rope_tail_fast64_kernel), dim3((unsigned)((total + 255) / 256), (unsigned)F.M), dim3(256), (hipStream_t)stream, F);
        return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
    }
    R4D_ARGS(RopeTailArgs, P);
    R4D_VC(P, r4d_verdict::kFamOps);
    P.tab = tab;
    P.x = (uint16_t*)x.data; P.pos = (const int32_t*)pos.data;
    P.xp = x.stride[0]; P.xs = x.stride[1]; P.ps = pos.stride[0];
    P.M = (int)x.shape[0]; P.H = (int)H; P.W = (int)W; P.R = (int)R;
    P.bf16_table = bf16_table ? 1 : 0; P.inverse = (int)inverse;
    R4D_KLAUNCH(R4D_KNAME(ernie_rope_tail_kernel), dim3((unsigned)((total + 255) / 256), (unsigned)P.M), dim3(256), (hipStream_t)stream, P);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

int r4d_rad_ernie_hca_q_prep(const RadArgs* a, RadStream stream) {
    using namespace r4d_ernie;
    using namespace r4d_ernie_ops;
    if (!a || !a->t || a->n_t < 2) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &pos = a->t[1];
    const long long H = r4d_geti_or(a, "heads", 0), D = r4d_geti_or(a, "head_dim", 0), R = r4d_geti_or(a, "rope_dim", 0);
    const double eps = r4d_getf(a, "eps", -1.0), theta = r4d_getf(a, "theta", 0.0);
    ErnieYarn yarn;
    if (!parse_yarn(a, &yarn)) return RAD_E_SHAPE;
    if (H < 1 || H > 65535 || D < 2 || D > 4096 || R < 2 || (R & 1) || R > D || R > 2 * kMaxRopePairs ||
        !(eps >= 0.0) || !std::isfinite(eps) || !std::isfinite(theta) || theta <= 0.0 ||
        x.rank != 2 || pos.rank != 1 || x.shape[0] < 1 || x.shape[0] > 65535 || x.shape[1] != H * D ||
        pos.shape[0] != x.shape[0] || !pos_strides(x) || !pos_strides(pos) || x.stride[0] < x.shape[1] * x.stride[1])
        return RAD_E_SHAPE;
    if (x.dtype != RAD_BF16 || pos.dtype != RAD_I32) return RAD_E_DTYPE;
    if (!x.data || !pos.data) return RAD_E_INVAL;
    RopeInvTable tab{};
    if (!fill_rope_inv(theta, R, yarn, &tab)) return RAD_E_SHAPE;
#if !defined(__HIP__) && !defined(__HIP_PLATFORM_AMD__)
    if (!r4d_verdict::card_form())
    for (long long m = 0; m < x.shape[0]; ++m)
        if (((const int32_t*)pos.data)[m * pos.stride[0]] < 0) return RAD_E_SHAPE;
#endif
    if (D == 512 && R == 64 && x.stride[1] == 1 && (((uintptr_t)x.data & 15u) == 0u) && ((x.stride[0] & 7) == 0)) {
        R4D_ARGS(HcaQPrepFast64Args, F);
        R4D_VC(F, r4d_verdict::kFamOps);
        F.x = (uint16_t*)x.data; F.pos = (const int32_t*)pos.data;
        F.xp = x.stride[0]; F.ps = pos.stride[0];
        F.M = (int)x.shape[0]; F.H = (int)H; F.eps = (float)eps;
        for (int i = 0; i < 32; ++i) F.inv[i] = tab.inv[i];
        if (F.M <= 4) {
            R4D_KLAUNCH(R4D_KNAME(ernie_hca_q_prep_fast512_kernel), dim3((unsigned)H, (unsigned)F.M), dim3(32), (hipStream_t)stream, F);
        } else {
            R4D_KLAUNCH(R4D_KNAME(ernie_hca_q_prep_fast512_kernel), dim3((unsigned)((H + 3) / 4), (unsigned)F.M), dim3(128), (hipStream_t)stream, F);
        }
        return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
    }
    if (D == 128 && R == 64 && x.stride[1] == 1 && (((uintptr_t)x.data & 7u) == 0u) && ((x.stride[0] & 3) == 0)) {
        R4D_ARGS(HcaQPrepFast64Args, F);
        R4D_VC(F, r4d_verdict::kFamOps);
        F.x = (uint16_t*)x.data; F.pos = (const int32_t*)pos.data;
        F.xp = x.stride[0]; F.ps = pos.stride[0];
        F.M = (int)x.shape[0]; F.H = (int)H; F.eps = (float)eps;
        for (int i = 0; i < 32; ++i) F.inv[i] = tab.inv[i];
        R4D_KLAUNCH(R4D_KNAME(ernie_hca_q_prep_fast128_kernel), dim3((unsigned)((H + 3) / 4), (unsigned)F.M), dim3(128), (hipStream_t)stream, F);
        return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
    }
    R4D_ARGS(HcaQPrepArgs, P);
    R4D_VC(P, r4d_verdict::kFamOps);
    P.tab = tab;
    P.x = (uint16_t*)x.data; P.pos = (const int32_t*)pos.data;
    P.xp = x.stride[0]; P.xs = x.stride[1]; P.ps = pos.stride[0];
    P.M = (int)x.shape[0]; P.H = (int)H; P.D = (int)D; P.R = (int)R; P.eps = (float)eps;
    R4D_KLAUNCH(R4D_KNAME(ernie_hca_q_prep_kernel), dim3((unsigned)H, (unsigned)P.M), dim3(256), (hipStream_t)stream, P);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

int r4d_rad_ernie_vha_mix(const RadArgs* a, RadStream stream) {
    using namespace r4d_ernie;
    using namespace r4d_ernie_ops;
    if (!a || !a->t || a->n_t < 4) return RAD_E_INVAL;
    const RadTensor &o = a->t[0], &u = a->t[1], &v = a->t[2], &out = a->t[3];
    const long long H = r4d_geti_or(a, "heads", 0), D = r4d_geti_or(a, "dim", 0), R = r4d_geti_or(a, "rank", 0);
    const long long W = r4d_geti_or(a, "world_size", 1), rk = r4d_geti_or(a, "tp_rank", 0);
    const char* fold = r4d_gets(a, "fold");
    const bool fold_bf16 = fold && !std::strcmp(fold, "bf16");
    if (fold && !fold_bf16 && std::strcmp(fold, "f32")) return RAD_E_UNSUPPORTED;
    if (H < 1 || H > 4096 || D < 1 || R < 1 || W < 1 || W > 64 || rk < 0 || rk >= W || (H % W) != 0)
        return RAD_E_SHAPE;
    const long long Hloc = H / W;
    if (o.rank != 2 || u.rank != 2 || v.rank != 2 || out.rank != 2 ||
        out.shape[0] < 1 || out.shape[0] > 65535 || o.shape[0] != W * out.shape[0] || o.shape[1] != Hloc * D ||
        u.shape[0] != H || u.shape[1] != R || v.shape[0] != H || v.shape[1] != R || out.shape[1] != Hloc * D ||
        !pos_strides(o) || !pos_strides(u) || !pos_strides(v) || !pos_strides(out) ||
        out.stride[0] < out.shape[1] * out.stride[1])
        return RAD_E_SHAPE;
    if (o.dtype != RAD_BF16 || out.dtype != RAD_BF16 || (u.dtype != RAD_BF16 && u.dtype != RAD_F32) ||
        (v.dtype != RAD_BF16 && v.dtype != RAD_F32))
        return RAD_E_DTYPE;
    if (!o.data || !u.data || !v.data || !out.data) return RAD_E_INVAL;
    if (operands_overlap(o, out)) return RAD_E_INVAL;
    R4D_ARGS(VhaMixArgs, P);
    P.o = (const uint16_t*)o.data; P.u = u.data; P.v = v.data; P.out = (uint16_t*)out.data;
    P.op = o.stride[0]; P.os = o.stride[1]; P.up = u.stride[0]; P.us = u.stride[1];
    P.vp = v.stride[0]; P.vs = v.stride[1]; P.outp = out.stride[0]; P.outs = out.stride[1];
    P.M = (int)out.shape[0]; P.H = (int)H; P.Hloc = (int)Hloc; P.h0 = (int)(rk * Hloc); P.D = (int)D; P.R = (int)R;
    P.u_f32 = (u.dtype == RAD_F32) ? 1 : 0; P.v_f32 = (v.dtype == RAD_F32) ? 1 : 0;
    P.fold_bf16 = fold_bf16 ? 1 : 0;

    const bool packed_u4 = P.os == 1 && P.outs == 1 && P.op == o.shape[1] && P.outp == out.shape[1] &&
                           (((uintptr_t)P.o | (uintptr_t)P.out) & 15u) == 0u && (D & 7) == 0;
    if (P.M <= 8 && (P.Hloc == 8 || P.Hloc == 4) && packed_u4) {
        constexpr int TileCols = kVhaMixU4Threads * 8;
        const unsigned tiles = (unsigned)((D + TileCols - 1) / TileCols);
        const dim3 grid(tiles * (unsigned)(P.Hloc / kVhaMixU4Heads), (unsigned)P.M);
        if (P.Hloc == 8) {
            R4D_KLAUNCH(R4D_KNAME(ernie_vha_mix_u4_kernel<8>), grid, dim3(kVhaMixU4Threads), (hipStream_t)stream, P);
        } else {
            R4D_KLAUNCH(R4D_KNAME(ernie_vha_mix_u4_kernel<4>), grid, dim3(kVhaMixU4Threads), (hipStream_t)stream, P);
        }
        return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
    }

    if (P.H <= 128 && (P.D & 1) == 0 && P.os == 1 && P.outs == 1 && (P.Hloc == 8 || P.Hloc == 4 || P.Hloc == 16)) {
        if (P.M >= 8) {
            constexpr int TM = 4, NTHR = 64;
            const unsigned bx = (unsigned)((P.D + 2 * NTHR - 1) / (2 * NTHR));
            const unsigned by = (unsigned)((P.M + TM - 1) / TM);
            if (P.Hloc == 8)       R4D_KLAUNCH(R4D_KNAME(ernie_vha_mix_fast_kernel<8, TM, NTHR>),  dim3(bx, by), dim3(NTHR), (hipStream_t)stream, P);
            else if (P.Hloc == 4)  R4D_KLAUNCH(R4D_KNAME(ernie_vha_mix_fast_kernel<4, TM, NTHR>),  dim3(bx, by), dim3(NTHR), (hipStream_t)stream, P);
            else                   R4D_KLAUNCH(R4D_KNAME(ernie_vha_mix_fast_kernel<16, TM, NTHR>), dim3(bx, by), dim3(NTHR), (hipStream_t)stream, P);
        } else {
            /* Decode / small M: 32 threads (1 wave) per 64 columns for high CU occupancy (8 blocks for D=512). */
            constexpr int TM = 1, NTHR = 32;
            const unsigned bx = (unsigned)((P.D + 2 * NTHR - 1) / (2 * NTHR));
            const unsigned by = (unsigned)P.M;
            if (P.Hloc == 8)       R4D_KLAUNCH(R4D_KNAME(ernie_vha_mix_fast_kernel<8, TM, NTHR>),  dim3(bx, by), dim3(NTHR), (hipStream_t)stream, P);
            else if (P.Hloc == 4)  R4D_KLAUNCH(R4D_KNAME(ernie_vha_mix_fast_kernel<4, TM, NTHR>),  dim3(bx, by), dim3(NTHR), (hipStream_t)stream, P);
            else                   R4D_KLAUNCH(R4D_KNAME(ernie_vha_mix_fast_kernel<16, TM, NTHR>), dim3(bx, by), dim3(NTHR), (hipStream_t)stream, P);
        }
        return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
    }
    R4D_KLAUNCH(R4D_KNAME(ernie_vha_mix_kernel), dim3((unsigned)Hloc, (unsigned)P.M), dim3(256), (hipStream_t)stream, P);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}


int r4d_rad_ernie_vha_proj(const RadArgs* a, RadStream stream) {
    using namespace r4d_ernie;
    using namespace r4d_ernie_ops;
    if (!a || !a->t || a->n_t < 3) return RAD_E_INVAL;
    const RadTensor &o = a->t[0], &u = a->t[1], &z = a->t[2];
    const long long H = r4d_geti_or(a, "heads", 0), D = r4d_geti_or(a, "dim", 0), R = r4d_geti_or(a, "rank", 0);
    const long long W = r4d_geti_or(a, "world_size", 1), rk = r4d_geti_or(a, "tp_rank", 0);
    if (H < 1 || H > 4096 || D < 1 || D > INT_MAX || R < 1 || R > INT_MAX ||
        W < 1 || W > 64 || rk < 0 || rk >= W || (H % W) != 0)
        return RAD_E_SHAPE;
    const long long Hloc = H / W;
    if (o.rank != 2 || u.rank != 2 || z.rank != 2 ||
        o.shape[0] < 1 || o.shape[0] > INT_MAX || o.shape[1] != Hloc * D ||
        u.shape[0] != H || u.shape[1] != R || z.shape[0] != o.shape[0] || z.shape[1] != R * D ||
        !pos_strides(o) || !pos_strides(u) || !pos_strides(z) ||
        z.stride[0] < z.shape[1] * z.stride[1])
        return RAD_E_SHAPE;
    if (o.dtype != RAD_BF16 || z.dtype != RAD_BF16 || (u.dtype != RAD_BF16 && u.dtype != RAD_F32))
        return RAD_E_DTYPE;
    if (!o.data || !u.data || !z.data) return RAD_E_INVAL;
    if (operands_overlap(o, z) || operands_overlap(u, z)) return RAD_E_INVAL;

    R4D_ARGS(VhaProjArgs, P);
    P.o = (const uint16_t*)o.data; P.u = u.data; P.z = (uint16_t*)z.data;
    P.op = o.stride[0]; P.os = o.stride[1]; P.up = u.stride[0]; P.us = u.stride[1];
    P.zp = z.stride[0]; P.zs = z.stride[1];
    P.M = (int)o.shape[0]; P.H = (int)H; P.Hloc = (int)Hloc; P.h0 = (int)(rk * Hloc);
    P.D = (int)D; P.R = (int)R; P.u_f32 = (u.dtype == RAD_F32) ? 1 : 0;

    const bool fast_ok = (P.Hloc <= 16 && P.R <= 32 && (P.D & 7) == 0 &&
                          P.os == 1 && P.zs == 1 &&
                          (((uintptr_t)P.o & 15u) == 0u) && (((uintptr_t)P.z & 15u) == 0u) &&
                          ((P.op & 7) == 0) && ((P.zp & 7) == 0));
    if (fast_ok) {
        if (P.D == 128) {
            if (P.M >= 8 && P.M <= 65535 * 4) {
                constexpr int RB = 4, D_LANES = 16, TM = 4, NTHR = 64;
                const unsigned gr = (unsigned)((P.R + (NTHR / D_LANES) * RB - 1) / ((NTHR / D_LANES) * RB));
                const unsigned gm = (unsigned)((P.M + TM - 1) / TM);
                R4D_KLAUNCH(R4D_KNAME(ernie_vha_proj_fast_kernel<RB, D_LANES, TM, NTHR>),
                            dim3(gr, 1u, gm), dim3(NTHR), (hipStream_t)stream, P);
                return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
            }
            if (P.M < 8) {
                constexpr int RB = 1, D_LANES = 16, TM = 1, NTHR = 32;
                const unsigned gr = (unsigned)((P.R + (NTHR / D_LANES) * RB - 1) / ((NTHR / D_LANES) * RB));
                R4D_KLAUNCH(R4D_KNAME(ernie_vha_proj_fast_kernel<RB, D_LANES, TM, NTHR>),
                            dim3(gr, 1u, (unsigned)P.M), dim3(NTHR), (hipStream_t)stream, P);
                return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
            }
        } else {
            const int d8 = P.D >> 3;
            if (P.M >= 8 && P.M <= 65535 * 4 && (d8 + 63) / 64 <= 65535) {
                constexpr int TM = 4, NTHR = 64;
                const unsigned gd = (unsigned)((d8 + NTHR - 1) / NTHR);
                const unsigned gm = (unsigned)((P.M + TM - 1) / TM);
                if (P.R >= 8) {
                    constexpr int RB = 8;
                    const unsigned gr = (unsigned)((P.R + RB - 1) / RB);
                    R4D_KLAUNCH(R4D_KNAME(ernie_vha_proj_fast_kernel<RB, 0, TM, NTHR>),
                                dim3(gr, gd, gm), dim3(NTHR), (hipStream_t)stream, P);
                } else {
                    constexpr int RB = 4;
                    const unsigned gr = (unsigned)((P.R + RB - 1) / RB);
                    R4D_KLAUNCH(R4D_KNAME(ernie_vha_proj_fast_kernel<RB, 0, TM, NTHR>),
                                dim3(gr, gd, gm), dim3(NTHR), (hipStream_t)stream, P);
                }
                return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
            }
            if (P.M < 8 && (d8 + 31) / 32 <= 65535) {
                constexpr int RB = 2, TM = 1, NTHR = 32;
                const unsigned gr = (unsigned)((P.R + RB - 1) / RB);
                const unsigned gd = (unsigned)((d8 + NTHR - 1) / NTHR);
                R4D_KLAUNCH(R4D_KNAME(ernie_vha_proj_fast_kernel<RB, 0, TM, NTHR>),
                            dim3(gr, gd, (unsigned)P.M), dim3(NTHR), (hipStream_t)stream, P);
                return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
            }
        }
    }
    const unsigned gr_fb = (unsigned)(P.R > 65535 ? 65535 : P.R);
    R4D_KLAUNCH(R4D_KNAME(ernie_vha_proj_kernel), dim3((unsigned)P.M, gr_fb), dim3(256), (hipStream_t)stream, P);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

int r4d_rad_ernie_vha_apply(const RadArgs* a, RadStream stream) {
    using namespace r4d_ernie;
    using namespace r4d_ernie_ops;
    if (!a || !a->t || a->n_t < 4) return RAD_E_INVAL;
    const RadTensor &o = a->t[0], &z = a->t[1], &v = a->t[2], &out = a->t[3];
    const long long H = r4d_geti_or(a, "heads", 0), D = r4d_geti_or(a, "dim", 0), R = r4d_geti_or(a, "rank", 0);
    const long long W = r4d_geti_or(a, "world_size", 1), rk = r4d_geti_or(a, "tp_rank", 0);
    if (H < 1 || H > 4096 || D < 1 || D > INT_MAX || R < 1 || R > INT_MAX ||
        W < 1 || W > 64 || rk < 0 || rk >= W || (H % W) != 0)
        return RAD_E_SHAPE;
    const long long Hloc = H / W;
    if (o.rank != 2 || z.rank != 2 || v.rank != 2 || out.rank != 2 ||
        o.shape[0] < 1 || o.shape[0] > INT_MAX || o.shape[1] != Hloc * D ||
        z.shape[0] != o.shape[0] || z.shape[1] != R * D ||
        v.shape[0] != H || v.shape[1] != R ||
        out.shape[0] != o.shape[0] || out.shape[1] != Hloc * D ||
        !pos_strides(o) || !pos_strides(z) || !pos_strides(v) || !pos_strides(out) ||
        out.stride[0] < out.shape[1] * out.stride[1])
        return RAD_E_SHAPE;
    if (o.dtype != RAD_BF16 || z.dtype != RAD_BF16 || out.dtype != RAD_BF16 ||
        (v.dtype != RAD_BF16 && v.dtype != RAD_F32))
        return RAD_E_DTYPE;
    if (!o.data || !z.data || !v.data || !out.data) return RAD_E_INVAL;
    if (operands_overlap(z, out) || operands_overlap(v, out)) return RAD_E_INVAL;
    const bool exact_inplace =
        (out.data == o.data && out.stride[0] == o.stride[0] && out.stride[1] == o.stride[1]);
    if (!exact_inplace && operands_overlap(o, out)) return RAD_E_INVAL;

    R4D_ARGS(VhaApplyArgs, P);
    P.o = (const uint16_t*)o.data; P.z = (const uint16_t*)z.data; P.v = v.data; P.out = (uint16_t*)out.data;
    P.op = o.stride[0]; P.os = o.stride[1]; P.zp = z.stride[0]; P.zs = z.stride[1];
    P.vp = v.stride[0]; P.vs = v.stride[1]; P.outp = out.stride[0]; P.outs = out.stride[1];
    P.M = (int)o.shape[0]; P.H = (int)H; P.Hloc = (int)Hloc; P.h0 = (int)(rk * Hloc);
    P.D = (int)D; P.R = (int)R; P.v_f32 = (v.dtype == RAD_F32) ? 1 : 0;

    const bool fast_ok = (P.Hloc <= 16 && P.R <= 32 && (P.D & 7) == 0 &&
                          P.os == 1 && P.zs == 1 && P.outs == 1 &&
                          (((uintptr_t)P.o & 15u) == 0u) && (((uintptr_t)P.z & 15u) == 0u) &&
                          (((uintptr_t)P.out & 15u) == 0u) &&
                          ((P.op & 7) == 0) && ((P.zp & 7) == 0) && ((P.outp & 7) == 0));
    if (fast_ok) {
        if (P.D == 128) {
            if (P.M >= 8 && P.M <= 65535 * 4) {
                constexpr int HB = 2, D_LANES = 16, TM = 4, NTHR = 64;
                const unsigned gh = (unsigned)((P.Hloc + (NTHR / D_LANES) * HB - 1) / ((NTHR / D_LANES) * HB));
                const unsigned gm = (unsigned)((P.M + TM - 1) / TM);
                R4D_KLAUNCH(R4D_KNAME(ernie_vha_apply_fast_kernel<HB, D_LANES, TM, NTHR>),
                            dim3(gh, 1u, gm), dim3(NTHR), (hipStream_t)stream, P);
                return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
            }
            if (P.M < 8) {
                constexpr int HB = 1, D_LANES = 16, TM = 1, NTHR = 32;
                const unsigned gh = (unsigned)((P.Hloc + (NTHR / D_LANES) * HB - 1) / ((NTHR / D_LANES) * HB));
                R4D_KLAUNCH(R4D_KNAME(ernie_vha_apply_fast_kernel<HB, D_LANES, TM, NTHR>),
                            dim3(gh, 1u, (unsigned)P.M), dim3(NTHR), (hipStream_t)stream, P);
                return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
            }
        } else {
            const int d8 = P.D >> 3;
            if (P.M >= 8 && P.M <= 65535 * 4 && (d8 + 63) / 64 <= 65535) {
                constexpr int TM = 4, NTHR = 64;
                const unsigned gd = (unsigned)((d8 + NTHR - 1) / NTHR);
                const unsigned gm = (unsigned)((P.M + TM - 1) / TM);
                if (P.Hloc >= 8) {
                    constexpr int HB = 8;
                    const unsigned gh = (unsigned)((P.Hloc + HB - 1) / HB);
                    R4D_KLAUNCH(R4D_KNAME(ernie_vha_apply_fast_kernel<HB, 0, TM, NTHR>),
                                dim3(gh, gd, gm), dim3(NTHR), (hipStream_t)stream, P);
                } else {
                    constexpr int HB = 4;
                    const unsigned gh = (unsigned)((P.Hloc + HB - 1) / HB);
                    R4D_KLAUNCH(R4D_KNAME(ernie_vha_apply_fast_kernel<HB, 0, TM, NTHR>),
                                dim3(gh, gd, gm), dim3(NTHR), (hipStream_t)stream, P);
                }
                return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
            }
            if (P.M < 8 && (d8 + 31) / 32 <= 65535) {
                constexpr int HB = 2, TM = 1, NTHR = 32;
                const unsigned gh = (unsigned)((P.Hloc + HB - 1) / HB);
                const unsigned gd = (unsigned)((d8 + NTHR - 1) / NTHR);
                R4D_KLAUNCH(R4D_KNAME(ernie_vha_apply_fast_kernel<HB, 0, TM, NTHR>),
                            dim3(gh, gd, (unsigned)P.M), dim3(NTHR), (hipStream_t)stream, P);
                return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
            }
        }
    }
    R4D_KLAUNCH(R4D_KNAME(ernie_vha_apply_kernel), dim3((unsigned)P.M, (unsigned)P.Hloc), dim3(256), (hipStream_t)stream, P);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

} /* extern "C" */

/* The events the position checks above raised since the last call (bit 2), 0 if none or if the validators are off (R4D_ERNIE_VALIDATE=0); -1 if the copy failed. */
extern "C" int r4d_ernie_ops_take_error(void) {
    return r4d_verdict::registry().take(r4d_verdict::kFamOps);
}

/* The one poll the engine makes (core/plugin/registry.h, rad_kernel_poll_error): the events of every ERNIE validator since the previous poll, OR-ed. Nonzero = a launch libref
 * would have refused ran on the card (bits as in the families' take_error functions; 0x40000000 = some event's bits were not readable); -1 = a copy failed. One device-to-host
 * copy per device that has run a launch (all four logs in one), read by the pointer the kernels were given. */
extern "C" int rad_kernel_poll_error(void) {
    return r4d_verdict::registry().take_all();
}

/* The engine's own readback of the verdict logs (core/plugin/registry.h): where the log of `device` lives and how to read a copy of it. The engine copies the log through that
 * rank's own queue behind the rank's step work, waits for that rank's event, and hands the copy back to fold. source returns 0 with validation off (nothing to copy). */
extern "C" int rad_kernel_poll_source(int device, void** dev_ptr, int64_t* bytes) {
    return r4d_verdict::registry().log_source(device, dev_ptr, bytes);
}
extern "C" int rad_kernel_poll_fold(int device, const void* copy, int64_t bytes) {
    return r4d_verdict::registry().fold_copy(device, copy, bytes);
}
