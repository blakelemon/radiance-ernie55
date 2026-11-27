/* r4d_ernie_mla.hip -- ERNIE 5.5 Flash MLA-DSA & indexer device ops (stories 4-3f, 4-3g):
 *   ernie_idx_query, ernie_idx_key, ernie_idx_logits, ernie_dsa_select, ernie_mla_project, ernie_mla_sparse_attn.
 * Written cleanly from ERNIE 5.5 Flash equations and libref/ref_ernie_attn.cpp + libref/ref_dsa.cpp.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cstddef>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include "r4d_common.h"
#include "r4d_ernie_common.h"
#include "r4d_ernie_verdict.h"
#include "r4d_fp8_wmma.h"
#include "r4d_fwht.h"
#include "r4d_plugin.h"
#include <cstdlib>

namespace r4d_ernie_mla {

using namespace r4d_ernie;

/* libref refuses ernie_idx_query for a position below 0 and ernie_idx_key for a position below 0 or a slot outside [-1, slots) (ref_ernie_attn.cpp:494, :544) before its
 * first store. A card launch cannot: the kernel of such a row returns before its first store (the row is left as it was) and one lane raises an event into this log,
 * which r4d_ernie_mla_take_error() hands to the host (rad_kernel_poll_error ORs it with the other ERNIE logs). Bit 1: a slot outside [-1, slots); bit 2: a negative
 * position. Emulator test: tests/r4d_emu/test_ernie_hca_state.cpp (idx_pos_suite), in the card form. */

static constexpr float kLog2E = 1.44269504088896340736f;
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
__device__ __forceinline__ uint32_t load_u32_unaligned(const uint8_t* p) {
    if (((uintptr_t)p & 3u) == 0u) return *(const uint32_t*)p;
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
__device__ __forceinline__ uint32_t pack_fp8_4(float a, float b, float c, float d, float inv_sc, float qmax) {
    const float fa = fminf(fmaxf(a * inv_sc, -qmax), qmax);
    const float fb = fminf(fmaxf(b * inv_sc, -qmax), qmax);
    const float fc = fminf(fmaxf(c * inv_sc, -qmax), qmax);
    const float fd = fminf(fmaxf(d * inv_sc, -qmax), qmax);
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1201__)
    int pk = 0;
    pk = __builtin_amdgcn_cvt_pk_fp8_f32(fa, fb, pk, false);
    pk = __builtin_amdgcn_cvt_pk_fp8_f32(fc, fd, pk, true);
    return (uint32_t)pk;
#else
    return (uint32_t)f32_to_fp8e4m3_dev(fa) |
           ((uint32_t)f32_to_fp8e4m3_dev(fb) << 8) |
           ((uint32_t)f32_to_fp8e4m3_dev(fc) << 16) |
           ((uint32_t)f32_to_fp8e4m3_dev(fd) << 24);
#endif
}

__device__ __forceinline__ void rope_cos_sin_fast(int pos, double inv_freq, int bf16_table, float* cs, float* sn) {
    const double ang = (double)pos * inv_freq;
    const float red = (float)(ang - floor(ang * kInvTwoPi) * kTwoPi);
    float s, c;
    sincosf(red, &s, &c);
    if (bf16_table) { c = bfr(c); s = bfr(s); }
    *cs = c;
    *sn = s;
}

static inline bool idx_geometry(long long D, long long R) {
    return D >= 4 && D <= 256 && is_pow2(D) && R >= 2 && !(R & 1) && R <= D;
}

static inline bool idx_table(const RadArgs* a, bool* bf16_table) {
    const char* t = r4d_gets(a, "rope_table");
    if (!t || !std::strcmp(t, "bf16")) { *bf16_table = true; return true; }
    if (!std::strcmp(t, "f32")) { *bf16_table = false; return true; }
    return false;
}

struct IdxQuant { float floor; float qmax; };

static inline bool idx_rule(const RadArgs* a, IdxQuant* r) {
    const char* t = r4d_gets(a, "quant_rule");
    if (!t || !std::strcmp(t, "nvidia")) { *r = {1e-10f, 448.0f}; return true; }
    if (!std::strcmp(t, "rocm")) { *r = {1e-4f, 448.0f}; return true; }
    if (!std::strcmp(t, "rocm_fnuz")) { *r = {1e-4f, 224.0f}; return true; }
    return false;
}

static inline RadTensor idx_cache_view(const RadTensor& c) {
    RadTensor v = c;
    if (c.rank == 4 && c.shape[1] == 1) {
        v.rank = 3;
        v.shape[1] = c.shape[2]; v.shape[2] = c.shape[3];
        v.stride[1] = c.stride[2]; v.stride[2] = c.stride[3];
    }
    return v;
}

/* Fallback device helper: NeoX leading-R RoPE -> bf16 -> orthonormal FWHT over D -> bf16 -> power-of-2 E4M3 quant. */
__device__ void idx_transform_and_quant(double* v, int D, int R, int pos, const RopeInvTable& tab,
                                        int bf16_table, float qfloor, float qmax,
                                        uint8_t* codes_out, long long code_stride, float* scale_out) {
    const int tid = (int)threadIdx.x;
    const int half = R >> 1;
    if (tid < half) {
        const double ang = (double)pos * tab.inv[tid];
        float c = (float)cos(ang), s = (float)sin(ang);
        if (bf16_table) { c = bfr(c); s = bfr(s); }
        const double x0 = v[tid], x1 = v[tid + half];
        v[tid]        = (double)bfr((float)(x0 * (double)c - x1 * (double)s));
        v[tid + half] = (double)bfr((float)(x1 * (double)c + x0 * (double)s));
    }
    __syncthreads();
    /* Orthonormal FWHT over D (power of 2, D <= 256) in LDS. */
    for (int h = 1; h < D; h <<= 1) {
        if (tid < (D >> 1)) {
            const int blk = tid / h, j = tid % h;
            const int i0 = blk * (2 * h) + j, i1 = i0 + h;
            const double x = v[i0], y = v[i1];
            v[i0] = x + y;
            v[i1] = x - y;
        }
        __syncthreads();
    }
    const double norm = 1.0 / sqrt((double)D);
    if (tid < D) v[tid] = (double)bfr((float)(v[tid] * norm));
    __syncthreads();

    __shared__ float smem[8];
    const float local_am = (tid < D) ? fabsf((float)v[tid]) : 0.0f;
    const float am = fmaxf(qfloor, block_max(local_am, smem, 256));
    const uint32_t bits = __float_as_uint(am / qmax);
    const int e2 = (int)((bits >> 23) & 0xffu) - 127 + ((bits & 0x7fffffu) ? 1 : 0);
    const float sc = __uint_as_float((uint32_t)(e2 + 127) << 23);
    if (tid < D) {
        const float qv = fminf(fmaxf((float)v[tid] / sc, -qmax), qmax);
        codes_out[(long long)tid * code_stride] = f32_to_fp8e4m3_dev(qv);
    }
    if (tid == 0) *scale_out = sc;
    __syncthreads();
}

/* ------------------------------------------------------------------ ernie_idx_query */
struct IdxQueryArgs {
    const uint16_t *q;
    const int32_t *pos;
    uint8_t *qf;
    float *qs;
    long long qp, qs_in, ps, qfp, qfs, qsp, qss;
    int M, H, D, R, bf16_table;
    float qfloor, qmax;
    RopeInvTable tab;
    r4d_verdict::Vc vc;   /* this launch's event log and gate word (device memory the library owns; constant across replays) */
};
static_assert(sizeof(IdxQueryArgs) <= 512, "IdxQueryArgs exceeds 512B AQL limit");

/* Fast D==128, R==64 args with host-precomputed inv[32] (eliminates GPU FP64 pow()). */
struct IdxQueryFast128Args {
    const uint16_t *q;
    const int32_t *pos;
    uint8_t *qf;
    float *qs;
    long long qp, ps, qfp, qsp, qss;
    int M, H, bf16_table, heads_per_block;
    float qfloor, qmax;
    double inv[32];
    r4d_verdict::Vc vc;   /* this launch's event log and gate word (device memory the library owns; constant across replays) */
};
static_assert(sizeof(IdxQueryFast128Args) <= 512, "IdxQueryFast128Args exceeds 512B AQL limit");

/* Wave32 in-register D==128, R==64 indexer query:
 * - Precomputed cos/sin computed ONCE per block in LDS (shared across all heads in the block).
 * - 1 wave = 32 lanes × 4 floats = 128 floats in registers.
 * - NeoX leading-64 RoPE via r4d_xor_lane<8>, FWHT via r4d_fwht_wave<128>, hardware v_cvt_pk_fp8_f32.
 * - Zero barriers inside the per-head transform! */
__global__ void ernie_idx_query_fast128_kernel(IdxQueryFast128Args A) {
    __shared__ float c_sh[32];
    __shared__ float s_sh[32];
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31, wave = tid >> 5;
    const int nwaves = (int)(blockDim.x >> 5);
    const int m = (int)blockIdx.y;
    if (m >= A.M) return;
    if (A.vc.log && A.pos[(long long)m * A.ps] < 0) { if (blockIdx.x == 0 && tid == 0) r4d_verdict::raise(A.vc.log, 2u); return; }

    const int p = A.pos[(long long)m * A.ps];
    if (tid < 32) {
        float c, s;
        rope_cos_sin_fast(p, A.inv[tid], A.bf16_table, &c, &s);
        c_sh[tid] = c;
        s_sh[tid] = s;
    }
    __syncthreads();

    const int h_start = (int)blockIdx.x * A.heads_per_block;
    const int h_end = (h_start + A.heads_per_block < A.H) ? (h_start + A.heads_per_block) : A.H;
    const uint16_t* q_row = A.q + (long long)m * A.qp;
    uint8_t* qf_row = A.qf + (long long)m * A.qfp;
    float* qs_row = A.qs + (long long)m * A.qsp;

    for (int h = h_start + wave; h < h_end; h += nwaves) {
        const uint16_t* qh = q_row + (long long)h * 128 + lane * 4;
        float v[4];
        if (((uintptr_t)qh & 7u) == 0u) {
            const uint2 u = *(const uint2*)qh;
            v[0] = bf2f_lo(u.x); v[1] = bf2f_hi(u.x);
            v[2] = bf2f_lo(u.y); v[3] = bf2f_hi(u.y);
        } else {
            v[0] = bf2f(qh[0]); v[1] = bf2f(qh[1]);
            v[2] = bf2f(qh[2]); v[3] = bf2f(qh[3]);
        }

        /* NeoX leading-64 RoPE: lanes 0..7 pair with lanes 8..15 via lane ^ 8. */
        const int ridx0 = (lane & 7) * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float partner = r4d_xor_lane<8>(v[j]);
            if (lane < 16) {
                const float c = c_sh[ridx0 + j], s = s_sh[ridx0 + j];
                const float rot = (lane < 8) ? (v[j] * c - partner * s) : (v[j] * c + partner * s);
                v[j] = bfr(rot);
            }
        }

        /* Orthonormal FWHT over D=128 in registers (0 LDS, 0 barriers). */
        r4d_fwht_wave<128>(v, lane);

        constexpr float kNorm128 = 0.08838834764831845f; /* 1 / sqrt(128) */
        float am = A.qfloor;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v[j] = bfr(v[j] * kNorm128);
            am = fmaxf(am, fabsf(v[j]));
        }
        am = r4d_wave_max(am);
        const uint32_t bits = __float_as_uint(am / A.qmax);
        const int e2 = (int)((bits >> 23) & 0xffu) - 127 + ((bits & 0x7fffffu) ? 1 : 0);
        const float sc = __uint_as_float((uint32_t)(e2 + 127) << 23);
        const float inv_sc = 1.0f / sc; /* exact power of 2 */

        const uint32_t pk = pack_fp8_4(v[0], v[1], v[2], v[3], inv_sc, A.qmax);
        uint8_t* dst = qf_row + (long long)h * 128 + lane * 4;
        if (((uintptr_t)dst & 3u) == 0u) {
            *(uint32_t*)dst = pk;
        } else {
            dst[0] = (uint8_t)(pk & 0xffu);
            dst[1] = (uint8_t)((pk >> 8) & 0xffu);
            dst[2] = (uint8_t)((pk >> 16) & 0xffu);
            dst[3] = (uint8_t)((pk >> 24) & 0xffu);
        }
        if (lane == 0) qs_row[(long long)h * A.qss] = sc;
    }
}

__global__ void ernie_idx_query_kernel(IdxQueryArgs A) {
    __shared__ double v[256];
    const int h = (int)blockIdx.x, m = (int)blockIdx.y, tid = (int)threadIdx.x;
    if (h >= A.H || m >= A.M) return;
    if (A.vc.log && A.pos[(long long)m * A.ps] < 0) { if (h == 0 && tid == 0) r4d_verdict::raise(A.vc.log, 2u); return; }
    if (tid < A.D) v[tid] = (double)bf2f(A.q[(long long)m * A.qp + (long long)(h * A.D + tid) * A.qs_in]);
    __syncthreads();
    const int p = A.pos[(long long)m * A.ps];
    uint8_t* dst = A.qf + (long long)m * A.qfp + (long long)(h * A.D) * A.qfs;
    float* sc_dst = A.qs + (long long)m * A.qsp + (long long)h * A.qss;
    idx_transform_and_quant(v, A.D, A.R, p, A.tab, A.bf16_table, A.qfloor, A.qmax, dst, A.qfs, sc_dst);
}

/* ------------------------------------------------------------------ ernie_idx_key */
struct IdxKeyArgs {
    const uint16_t *k, *gm, *bt;
    const int32_t *pos, *slot;
    uint8_t *cache;
    long long kp, ks, gms, bts, ps, sls;
    int M, D, R, bf16_table, slots;
    double eps;
    float qfloor, qmax;
    RopeInvTable tab;
    r4d_verdict::Vc vc;   /* this launch's event log and gate word (device memory the library owns; constant across replays) */
};
static_assert(sizeof(IdxKeyArgs) <= 512, "IdxKeyArgs exceeds 512B AQL limit");

struct IdxKeyFast128Args {
    const uint16_t *k, *gm, *bt;
    const int32_t *pos, *slot;
    uint8_t *cache;
    long long kp, ps, sls;
    int M, bf16_table, slots;
    float eps, qfloor, qmax;
    double inv[32];
    r4d_verdict::Vc vc;   /* this launch's event log and gate word (device memory the library owns; constant across replays) */
};
static_assert(sizeof(IdxKeyFast128Args) <= 512, "IdxKeyFast128Args exceeds 512B AQL limit");

__device__ __forceinline__ void idx_key_fast128_one_row(const IdxKeyFast128Args& A, int m, int lane) {
    const int sl = A.slot[(long long)m * A.sls];
    if (!A.vc.log && (sl < 0 || sl >= A.slots)) return;   /* validation off: the old kernel (a negative position is computed, a slot -1 or outside the pool is skipped silently) */
    if (sl < -1 || sl >= A.slots) { if (lane == 0) r4d_verdict::raise(A.vc.log, 1u); return; }
    const int p = A.pos[(long long)m * A.ps];
    if (A.vc.log && p < 0) { if (lane == 0) r4d_verdict::raise(A.vc.log, 2u); return; }   /* libref (ref_ernie_attn.cpp idx_key) refuses a negative position on EVERY row, a dropped (slot -1) row included */
    if (sl < 0) return;

    const uint16_t* kr = A.k + (long long)m * A.kp + lane * 4;
    const uint16_t* gr = A.gm + lane * 4;
    const uint16_t* br = A.bt + lane * 4;
    float v[4], g[4], b[4];
    if (((uintptr_t)kr & 7u) == 0u) {
        const uint2 u = *(const uint2*)kr;
        v[0] = bf2f_lo(u.x); v[1] = bf2f_hi(u.x);
        v[2] = bf2f_lo(u.y); v[3] = bf2f_hi(u.y);
    } else {
        v[0] = bf2f(kr[0]); v[1] = bf2f(kr[1]); v[2] = bf2f(kr[2]); v[3] = bf2f(kr[3]);
    }
    if (((uintptr_t)gr & 7u) == 0u) {
        const uint2 u = *(const uint2*)gr;
        g[0] = bf2f_lo(u.x); g[1] = bf2f_hi(u.x);
        g[2] = bf2f_lo(u.y); g[3] = bf2f_hi(u.y);
    } else {
        g[0] = bf2f(gr[0]); g[1] = bf2f(gr[1]); g[2] = bf2f(gr[2]); g[3] = bf2f(gr[3]);
    }
    if (((uintptr_t)br & 7u) == 0u) {
        const uint2 u = *(const uint2*)br;
        b[0] = bf2f_lo(u.x); b[1] = bf2f_hi(u.x);
        b[2] = bf2f_lo(u.y); b[3] = bf2f_hi(u.y);
    } else {
        b[0] = bf2f(br[0]); b[1] = bf2f(br[1]); b[2] = bf2f(br[2]); b[3] = bf2f(br[3]);
    }

    /* LayerNorm over D=128 in wave32 registers (0 LDS, 0 barriers). */
    const float sum_l = (v[0] + v[1]) + (v[2] + v[3]);
    const float mean = r4d_xor_sum<32>(sum_l) * (1.0f / 128.0f);
    float var_l = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float d = v[j] - mean;
        var_l += d * d;
    }
    const float rstd = rsqrtf(r4d_xor_sum<32>(var_l) * (1.0f / 128.0f) + A.eps);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = bfr((v[j] - mean) * rstd * g[j] + b[j]);

    /* NeoX leading-64 RoPE via lane ^ 8. */
    const int ridx0 = (lane & 7) * 4;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float partner = r4d_xor_lane<8>(v[j]);
        if (lane < 16) {
            float c, s;
            rope_cos_sin_fast(p, A.inv[ridx0 + j], A.bf16_table, &c, &s);
            const float rot = (lane < 8) ? (v[j] * c - partner * s) : (v[j] * c + partner * s);
            v[j] = bfr(rot);
        }
    }

    /* Orthonormal FWHT over D=128 in registers. */
    r4d_fwht_wave<128>(v, lane);

    constexpr float kNorm128 = 0.08838834764831845f;
    float am = A.qfloor;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j] = bfr(v[j] * kNorm128);
        am = fmaxf(am, fabsf(v[j]));
    }
    am = r4d_wave_max(am);
    const uint32_t bits = __float_as_uint(am / A.qmax);
    const int e2 = (int)((bits >> 23) & 0xffu) - 127 + ((bits & 0x7fffffu) ? 1 : 0);
    const float sc = __uint_as_float((uint32_t)(e2 + 127) << 23);
    const float inv_sc = 1.0f / sc;

    uint8_t* row = A.cache + (long long)sl * 132;
    const uint32_t pk = pack_fp8_4(v[0], v[1], v[2], v[3], inv_sc, A.qmax);
    uint8_t* dst = row + lane * 4;
    if (((uintptr_t)dst & 3u) == 0u) {
        *(uint32_t*)dst = pk;
    } else {
        dst[0] = (uint8_t)(pk & 0xffu);
        dst[1] = (uint8_t)((pk >> 8) & 0xffu);
        dst[2] = (uint8_t)((pk >> 16) & 0xffu);
        dst[3] = (uint8_t)((pk >> 24) & 0xffu);
    }
    if (lane == 0) {
        const uint32_t u = __float_as_uint(sc);
        if (((uintptr_t)(row + 128) & 3u) == 0u) {
            *(uint32_t*)(row + 128) = u;
        } else {
            row[128] = (uint8_t)(u & 0xffu);
            row[129] = (uint8_t)((u >> 8) & 0xffu);
            row[130] = (uint8_t)((u >> 16) & 0xffu);
            row[131] = (uint8_t)((u >> 24) & 0xffu);
        }
    }
}

__global__ void ernie_idx_key_fast128_seq_kernel(IdxKeyFast128Args A) {
    const int lane = (int)(threadIdx.x & 31);
    for (int m = 0; m < A.M; ++m) idx_key_fast128_one_row(A, m, lane);
}

__global__ void ernie_idx_key_fast128_par_kernel(IdxKeyFast128Args A) {
    const int lane = (int)(threadIdx.x & 31);
    const int m = (int)blockIdx.x * (int)(blockDim.x >> 5) + (int)(threadIdx.x >> 5);
    if (m < A.M) idx_key_fast128_one_row(A, m, lane);
}

__device__ void idx_key_one_row(const IdxKeyArgs& A, int m, double* v) {
    const int tid = (int)threadIdx.x;
    const int sl = A.slot[(long long)m * A.sls];
    if (!A.vc.log && (sl < 0 || sl >= A.slots)) return;   /* validation off: the old kernel */
    if (sl < -1 || sl >= A.slots) { if (tid == 0) r4d_verdict::raise(A.vc.log, 1u); return; }
    const int p = A.pos[(long long)m * A.ps];
    if (A.vc.log && p < 0) { if (tid == 0) r4d_verdict::raise(A.vc.log, 2u); return; }   /* as libref: every row, a dropped (slot -1) row included */
    if (sl < 0) return;
    if (tid < A.D) v[tid] = (double)bf2f(A.k[(long long)m * A.kp + (long long)tid * A.ks]);
    __syncthreads();
    /* Exact LayerNorm in double (D <= 256). */
    __shared__ double ln_stat[2];
    if (tid == 0) {
        double mean = 0.0;
        for (int d = 0; d < A.D; ++d) mean += v[d];
        mean /= (double)A.D;
        double var = 0.0;
        for (int d = 0; d < A.D; ++d) { const double t = v[d] - mean; var += t * t; }
        ln_stat[0] = mean;
        ln_stat[1] = 1.0 / sqrt(var / (double)A.D + A.eps);
    }
    __syncthreads();
    if (tid < A.D) {
        const double g = (double)bf2f(A.gm[(long long)tid * A.gms]);
        const double b = (double)bf2f(A.bt[(long long)tid * A.bts]);
        v[tid] = (double)bfr((float)((v[tid] - ln_stat[0]) * ln_stat[1] * g + b));
    }
    __syncthreads();
    __shared__ float sc_val;
    uint8_t* row = A.cache + (long long)sl * (A.D + 4);
    idx_transform_and_quant(v, A.D, A.R, p, A.tab, A.bf16_table, A.qfloor, A.qmax, row, 1, &sc_val);
    if (tid == 0) {
        const uint32_t u = __float_as_uint(sc_val);
        row[A.D + 0] = (uint8_t)(u & 0xffu);
        row[A.D + 1] = (uint8_t)((u >> 8) & 0xffu);
        row[A.D + 2] = (uint8_t)((u >> 16) & 0xffu);
        row[A.D + 3] = (uint8_t)((u >> 24) & 0xffu);
    }
    __syncthreads();
}

__global__ void ernie_idx_key_seq_kernel(IdxKeyArgs A) {
    __shared__ double v[256];
    for (int m = 0; m < A.M; ++m) idx_key_one_row(A, m, v);
}

__global__ void ernie_idx_key_par_kernel(IdxKeyArgs A) {
    __shared__ double v[256];
    const int m = (int)blockIdx.x;
    if (m < A.M) idx_key_one_row(A, m, v);
}

/* ------------------------------------------------------------------ ernie_idx_logits */
/* max(x, 0) that keeps a NaN a NaN, as libref's std::max(x, 0.0f) does: fmaxf returns the other operand, which turned a NaN score into 0 and hid it from the
 * non-finite refusal of ernie_dsa_topk_tiled. For every other x it is fmaxf. */
__device__ __forceinline__ float relu_keep_nan(float x) { return (x != x) ? x : fmaxf(x, 0.0f); }

struct IdxLogitsArgs {
    const uint8_t *qf, *cache;
    const float *qs;
    const uint16_t *w;
    const int32_t *bt, *req, *pos;
    float *out;
    long long qfp, qfs, qsp, qss, wp, ws, btp, bts, rs, ps, outp, outs;
    int M, H, D, L, S, B, block, pages, window;
    int col0;   /* col_start: logical id of output column 0 (the tiled path scores [col0, col0 + L)); 0 = the whole row */
    float wscale;
    int keep_nan;   /* 1: a NaN score stays a NaN (relu_keep_nan: the tiled op refuses it; validation on); 0: fmaxf, as base d2258ef8 */
};
static_assert(sizeof(IdxLogitsArgs) <= 512, "IdxLogitsArgs exceeds 512B AQL limit");

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1201__)
/* WMMA FP8 indexer logits kernel for D == 128, qfs == 1:
 * 4 waves (128 threads) per block cover 64 heads per pass across K_KEYS = 16 * K_TILES keys.
 * Uses v_wmma_f32_16x16x16_fp8_fp8_w32_gfx12 (emulated bit-for-bit in r4d_emu). */
template <int K_TILES>
__global__ void ernie_idx_logits_wmma128_kernel(IdxLogitsArgs A) {
    constexpr int K_KEYS = 16 * K_TILES;
    __shared__ __align__(16) uint8_t qa[64 * 136];
    __shared__ __align__(16) uint8_t kb[K_KEYS * 136];
    __shared__ float scales[K_KEYS];
    __shared__ float coeffs[64];
    __shared__ float partial[4][K_KEYS];
    __shared__ int slots_sh[K_KEYS];

    const int tid = (int)threadIdx.x;
    const int lane = tid & 31, wave = tid >> 5;
    const int j0 = (int)blockIdx.x * K_KEYS;   /* output column of this block's first key */
    const int g0 = A.col0 + j0;                /* its logical id */
    const int m = (int)blockIdx.y;
    if (m >= A.M) return;

    const int p = A.pos[(long long)m * A.ps];
    const int r = A.req[(long long)m * A.rs];
    /* a request outside the block table scores nothing: -inf everywhere, and no table access (libref refuses it; ernie_dsa_topk_tiled does too, after scoring) */
    const int end = (r >= 0 && r < A.S && p + 1 - A.window > 0) ? (p + 1 - A.window) : 0;

    if (g0 >= end) {
        for (int i = tid; i < K_KEYS; i += 128) {
            if (j0 + i < A.L)
                A.out[(long long)m * A.outp + (long long)(j0 + i) * A.outs] = -INFINITY;
        }
        return;
    }

    for (int i = tid; i < K_KEYS; i += 128) {
        const int j = g0 + i;
        const int b = j / A.block;
        int slot = -1;
        float ks = 0.0f;
        if (j < end && j0 + i < A.L && b < A.B) {
            const int pg = A.bt[(long long)r * A.btp + (long long)b * A.bts];
            if (pg >= 0 && pg < A.pages) {
                slot = pg * A.block + (j % A.block);
                const uint8_t* row = A.cache + (long long)slot * 132;
                ks = __uint_as_float(load_u32_unaligned(row + 128));
            }
        }
        slots_sh[i] = slot;
        scales[i] = ks;
    }
    __syncthreads();

    /* Stage K_KEYS keys of 128 bytes into kb (pitch 136B for bank-conflict-free ds_read_b64). */
    for (int idx = tid; idx < K_KEYS * 32; idx += 128) {
        const int ki = idx >> 5;
        const int d4 = (idx & 31) << 2;
        const int slot = slots_sh[ki];
        const uint32_t w = (slot >= 0) ? load_u32_unaligned(A.cache + (long long)slot * 132 + d4) : 0u;
        *(uint32_t*)(kb + ki * 136 + d4) = w;
    }

    const uint8_t* q_row = A.qf + (long long)m * A.qfp;
    const int h_chunks = (A.H + 63) >> 6;
    for (int hc = 0; hc < h_chunks; ++hc) {
        const int h0 = hc * 64;
        if (tid < 64) {
            const int h = h0 + tid;
            coeffs[tid] = (h < A.H)
                ? ((bf2f(A.w[(long long)m * A.wp + (long long)h * A.ws]) *
                    A.qs[(long long)m * A.qsp + (long long)h * A.qss]) * A.wscale)
                : 0.0f;
        }
        for (int idx = tid; idx < 64 * 32; idx += 128) {
            const int lh = idx >> 5;
            const int d4 = (idx & 31) << 2;
            const int h = h0 + lh;
            const uint32_t w = (h < A.H) ? load_u32_unaligned(q_row + (long long)h * 128 + d4) : 0u;
            *(uint32_t*)(qa + lh * 136 + d4) = w;
        }
        __syncthreads();

        r4d_fp8::f8v acc[K_TILES];
#pragma unroll
        for (int t = 0; t < K_TILES; ++t) acc[t] = r4d_fp8::acc_zero();

#pragma unroll
        for (int d = 0; d < 128; d += 16) {
            const r4d_fp8::i2v a = r4d_fp8::frag<128>(qa, (uint32_t)(wave * 16), (uint32_t)d, (uint32_t)lane);
#pragma unroll
            for (int t = 0; t < K_TILES; ++t) {
                const r4d_fp8::i2v b = r4d_fp8::frag<128>(kb, (uint32_t)(t * 16), (uint32_t)d, (uint32_t)lane);
                acc[t] = r4d_fp8::mma(a, b, acc[t]);
            }
        }

        const int h_row_base = wave * 16 + ((lane >> 4) << 3);
        const int col = lane & 15;
#pragma unroll
        for (int t = 0; t < K_TILES; ++t) {
            const float k_sc = scales[t * 16 + col];
            float val = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                val += (A.keep_nan ? relu_keep_nan(acc[t][j] * k_sc) : fmaxf(acc[t][j] * k_sc, 0.0f)) * coeffs[h_row_base + j];
            }
            val += r4d_xor_lane<16>(val);
            if (lane < 16) {
                const int ki = t * 16 + col;
                partial[wave][ki] = (hc == 0) ? val : (partial[wave][ki] + val);
            }
        }
        __syncthreads();
    }

    for (int i = tid; i < K_KEYS; i += 128) {
        if (j0 + i < A.L) {
            const float tot = (slots_sh[i] >= 0)
                ? ((partial[0][i] + partial[1][i]) + (partial[2][i] + partial[3][i]))
                : -INFINITY;
            A.out[(long long)m * A.outp + (long long)(j0 + i) * A.outs] = tot;
        }
    }
}
#endif

__global__ void ernie_idx_logits_kernel(IdxLogitsArgs A) {
    /* Fallback: 128 threads = 4 waves; each wave computes 1 output score (m, j) with fast E4M3 decode. */
    const int lane = (int)(threadIdx.x & 31), wave = (int)(threadIdx.x >> 5);
    const int jo = (int)blockIdx.x * 4 + wave;   /* output column */
    const int m = (int)blockIdx.y;
    if (m >= A.M || jo >= A.L) return;
    const int j = A.col0 + jo;                   /* logical id */
    const int p = A.pos[(long long)m * A.ps], r = A.req[(long long)m * A.rs];
    const int end = (r >= 0 && r < A.S && p + 1 - A.window > 0) ? (p + 1 - A.window) : 0;   /* a request outside the table scores nothing */
    const int b = j / A.block;
    if (j >= end || b >= A.B) {
        if (lane == 0) A.out[(long long)m * A.outp + (long long)jo * A.outs] = -INFINITY;
        return;
    }
    const int page = A.bt[(long long)r * A.btp + (long long)b * A.bts];
    if (page < 0 || page >= A.pages) {
        if (lane == 0) A.out[(long long)m * A.outp + (long long)jo * A.outs] = -INFINITY;
        return;
    }
    const uint8_t* row = A.cache + (long long)((long long)page * A.block + (j % A.block)) * (A.D + 4);
    const float ks = __uint_as_float(load_u32_unaligned(row + A.D));
    float kv[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        const int d = lane + t * 32;
        kv[t] = (d < A.D) ? r4d_fp8::e4m3((uint32_t)row[d]) : 0.0f;
    }
    float result = 0.0f;
    for (int h = 0; h < A.H; ++h) {
        float dot = 0.0f;
        const long long qbase = (long long)m * A.qfp + (long long)(h * A.D) * A.qfs;
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int d = lane + t * 32;
            if (d < A.D) dot += r4d_fp8::e4m3((uint32_t)A.qf[qbase + (long long)d * A.qfs]) * kv[t];
        }
        dot = r4d_xor_sum<32>(dot);
        if (lane == 0) {
            const float wt = (bf2f(A.w[(long long)m * A.wp + (long long)h * A.ws]) *
                              A.qs[(long long)m * A.qsp + (long long)h * A.qss]) * A.wscale;
            result += (A.keep_nan ? relu_keep_nan(dot * ks) : fmaxf(dot * ks, 0.0f)) * wt;
        }
    }
    if (lane == 0) A.out[(long long)m * A.outp + (long long)jo * A.outs] = result;
}

/* ------------------------------------------------------------------ ernie_dsa_select */
struct DsaSelectArgs {
    const float *sc;
    const int32_t *pos, *req, *bt;
    int32_t *ids, *slots;
    long long scp, scs, ps, rs, btp, bts, idsp, idss, slp, sls;
    int M, L, S, B, topk, window, bs, pages, out_w;
    int nb, pass;   /* multi-block path: blocks a row is split over, and the radix pass of the launch */
    /* tiled route (ernie_dsa_topk_tiled with one tile): the scratch is the op's `kept` operand, not the output rows, and a verdict word per row decides the whole launch */
    int tiled, bound;
    int32_t *scr, *flags;   /* scr: row m's scratch at scr + m * scrp (1024 + 2 nb histogram / count words, then the topk-word gather list); flags: one verdict word per row */
    long long scrp;
    r4d_verdict::Vc vc;   /* this launch's event log (device memory the library owns; constant across replays) */
};
static_assert(sizeof(DsaSelectArgs) <= 512, "DsaSelectArgs exceeds 512B AQL limit");

__device__ __forceinline__ uint32_t f32_order_key(float f) {
    if (f == 0.0f) f = 0.0f;   /* canonicalize -0.0f to +0.0f so ties break by lowest id */
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ int dsa_page(const DsaSelectArgs& A, int r, int id) {
    const int b = id / A.bs;
    if (b >= A.B) return -1;
    const int e = A.bt[(long long)r * A.btp + (long long)b * A.bts];
    return (e >= 0 && e < A.pages) ? e : -1;
}

/* The tail every selection kernel shares: the local window ids seq - min(seq, window) .. seq - 1 (ascending, minus any on an uninitialised page) after the `kept`
 * selected entries, then -1 up to the row width. Whole block. */
__device__ __forceinline__ void dsa_window_tail(const DsaSelectArgs& A, int m, int r, int seq, int kept, int* ismem) {
    const int tid = (int)threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int wlo = seq - (seq < A.window ? seq : A.window);
    int total_written = kept;
    for (int id0 = wlo; id0 < seq; id0 += 256) {
        const int id = id0 + tid;
        const int pg = (id < seq) ? dsa_page(A, r, id) : -1;
        const uint32_t valid = (pg >= 0) ? 1u : 0u;
        const uint32_t w_inc = r4d_wave_prefix_u32(valid, lane);
        if (lane == 31) ismem[wave] = (int)w_inc;
        __syncthreads();
        int w_off = 0, chunk_tot = 0;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            if (w < wave) w_off += ismem[w];
            chunk_tot += ismem[w];
        }
        if (valid) {
            const int pos_out = total_written + w_off + (int)w_inc - 1;
            A.ids[(long long)m * A.idsp + (long long)pos_out * A.idss] = id;
            A.slots[(long long)m * A.slp + (long long)pos_out * A.sls] = pg * A.bs + (id % A.bs);
        }
        total_written += chunk_tot;
        __syncthreads();
    }

    for (int j = total_written + tid; j < A.out_w; j += 256) {
        A.ids[(long long)m * A.idsp + (long long)j * A.idss] = -1;
        A.slots[(long long)m * A.slp + (long long)j * A.sls] = -1;
    }
}

__global__ void ernie_dsa_select_kernel(DsaSelectArgs A) {
    /* 1 block of 256 threads per row m. Parallel prefix-sum compaction for n_cand<=topk, tie-break & window append. */
    __shared__ int ismem[8];
    __shared__ uint32_t hist[256];
    __shared__ uint32_t top_keys[2048];
    __shared__ int32_t  top_ids[2048];
    __shared__ int32_t  gt_cnt;

    const int m = (int)blockIdx.x, tid = (int)threadIdx.x;
    const int lane = tid & 31, wave = tid >> 5;
    if (m >= A.M) return;
    const int p = A.pos[(long long)m * A.ps], r = A.req[(long long)m * A.rs];
    /* libref refuses all of these for the whole launch; the card cannot, so the row stores nothing and raises an event the engine polls (g_dsa_log: 1 a NaN or +inf score
     * in [0, end), 2 a bad position or request, 4 a score range past the plane). The engine fails the step, so the rows stored before or after are never consumed. */
    if (p < 0 || r < 0 || r >= A.S) { if (tid == 0) r4d_verdict::raise(A.vc.log, 2u); return; }
    const long long seq64 = (long long)p + 1;
    const long long end64 = (seq64 - (long long)A.window > 0) ? (seq64 - (long long)A.window) : 0;
    if (end64 > (long long)A.L) { if (tid == 0) r4d_verdict::raise(A.vc.log, 4u); return; }   /* refuse OOB score read before touching scores or outputs */
    const int seq = (int)seq64;
    const int end = (int)end64;
    const float* row = A.sc + (long long)m * A.scp;

    /* Count valid candidates in [0, end). */
    int local_cand = 0, local_bad = 0;
    for (int id = tid; id < end; id += 256) {
        const float v = row[(long long)id * A.scs];
        if (A.vc.log) local_bad |= (v != v) || (v == INFINITY);   /* libref scans the score before its page filter, so a bad score on an unallocated page refuses too */
        if (v > -INFINITY && dsa_page(A, r, id) >= 0) ++local_cand;
    }
    const int n_cand = block_sum_i(local_cand, ismem, 256);
    /* validation off (a card default): as base d2258ef8, a NaN is no candidate and +inf is the largest key; the row is selected, nothing is skipped or raised */
    if (A.vc.log && block_sum_i(local_bad, ismem, 256) > 0) { if (tid == 0) r4d_verdict::raise(A.vc.log, 1u); return; }

    int kept = 0;
    if (n_cand == 0) {
        kept = 0;
    } else if (n_cand <= A.topk) {
        /* Parallel prefix-sum compaction across all 256 threads preserving ascending id order. */
        int base = 0;
        for (int id0 = 0; id0 < end; id0 += 256) {
            const int id = id0 + tid;
            const int pg = (id < end && row[(long long)id * A.scs] > -INFINITY) ? dsa_page(A, r, id) : -1;
            const uint32_t valid = (pg >= 0) ? 1u : 0u;
            const uint32_t w_inc = r4d_wave_prefix_u32(valid, lane);
            if (lane == 31) ismem[wave] = (int)w_inc;
            __syncthreads();
            int w_off = 0, chunk_tot = 0;
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                if (w < wave) w_off += ismem[w];
                chunk_tot += ismem[w];
            }
            if (valid) {
                const int pos_out = base + w_off + (int)w_inc - 1;
                A.ids[(long long)m * A.idsp + (long long)pos_out * A.idss] = id;
                A.slots[(long long)m * A.slp + (long long)pos_out * A.sls] = pg * A.bs + (id % A.bs);
            }
            base += chunk_tot;
            __syncthreads();
        }
        kept = base;
    } else {
        /* 4-pass 256-bin MSB-first radix threshold to find exact k-th largest score key T. */
        uint32_t prefix = 0u;
        int rem = A.topk;
        for (int pass = 0; pass < 4; ++pass) {
            const int shift = 24 - pass * 8;
            const uint32_t mask = (pass == 0) ? 0u : (~0u << (shift + 8));
            hist[tid] = 0u;
            __syncthreads();
            for (int id = tid; id < end; id += 256) {
                const float v = row[(long long)id * A.scs];
                if (v > -INFINITY && dsa_page(A, r, id) >= 0) {
                    const uint32_t k = f32_order_key(v);
                    if ((k & mask) == prefix) {
                        const uint32_t bin = (k >> shift) & 0xffu;
                        atomicAdd(&hist[bin], 1u);
                    }
                }
            }
            __syncthreads();
            /* Parallel 256-bin suffix sum across all 256 threads (bin tid = 0..255). */
            const uint32_t c_bin = hist[tid];
            const uint32_t w_suf = r4d_wave_suffix_u32(c_bin, lane);
            if (lane == 0) ismem[wave] = (int)w_suf;
            __syncthreads();
            uint32_t hi_sum = 0u;
#pragma unroll
            for (int w = 7; w > 0; --w) {
                if (w > wave) hi_sum += (uint32_t)ismem[w];
            }
            const uint32_t suf_inc = w_suf + hi_sum;       /* sum of hist[tid..255] */
            const uint32_t suf_exc = suf_inc - c_bin;      /* sum of hist[tid+1..255] */
            __syncthreads();
            if (suf_inc >= (uint32_t)rem && suf_exc < (uint32_t)rem) {
                ismem[0] = tid;
                ismem[1] = rem - (int)suf_exc;
            }
            __syncthreads();
            prefix |= ((uint32_t)ismem[0] << shift);
            rem = ismem[1];
            __syncthreads();
        }
        const uint32_t thresh = prefix;

        /* Collect all strictly-greater keys in parallel. */
        if (tid == 0) gt_cnt = 0;
        __syncthreads();
        for (int id = tid; id < end; id += 256) {
            const float v = row[(long long)id * A.scs];
            if (v > -INFINITY && dsa_page(A, r, id) >= 0) {
                const uint32_t k = f32_order_key(v);
                if (k > thresh) {
                    const int idx = atomicAdd(&gt_cnt, 1);
                    if (idx < 2048) { top_keys[idx] = k; top_ids[idx] = id; }
                }
            }
        }
        __syncthreads();

        /* Parallel lowest-id tie-break for keys == thresh. */
        int pos_eq = gt_cnt;
        int need = A.topk - pos_eq;
        for (int id0 = 0; id0 < end && need > 0; id0 += 256) {
            const int id = id0 + tid;
            uint32_t match = 0u;
            if (id < end) {
                const float v = row[(long long)id * A.scs];
                if (v > -INFINITY && dsa_page(A, r, id) >= 0 && f32_order_key(v) == thresh)
                    match = 1u;
            }
            const uint32_t w_inc = r4d_wave_prefix_u32(match, lane);
            if (lane == 31) ismem[wave] = (int)w_inc;
            __syncthreads();
            int w_off = 0, chunk_tot = 0;
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                if (w < wave) w_off += ismem[w];
                chunk_tot += ismem[w];
            }
            if (match) {
                const int rank_in_chunk = w_off + (int)w_inc - 1;
                if (rank_in_chunk < need && pos_eq + rank_in_chunk < 2048) {
                    top_keys[pos_eq + rank_in_chunk] = thresh;
                    top_ids[pos_eq + rank_in_chunk] = id;
                }
            }
            const int taken = (chunk_tot < need) ? chunk_tot : need;
            pos_eq += taken;
            need -= taken;
            __syncthreads();
        }

        /* Pad up to next power of 2 <= 2048 with sentinel (key=0, id=INT_MAX) and parallel bitonic sort descending by (key, -id). */
        int P2 = 1;
        while (P2 < A.topk) P2 <<= 1;
        for (int i = A.topk + tid; i < P2; i += 256) { top_keys[i] = 0u; top_ids[i] = INT_MAX; }
        __syncthreads();
        for (int size = 2; size <= P2; size <<= 1) {
            for (int stride = size >> 1; stride > 0; stride >>= 1) {
                for (int i = tid; i < P2; i += 256) {
                    const int ixj = i ^ stride;
                    if (ixj > i) {
                        const bool asc = ((i & size) == 0);
                        const uint32_t ka = top_keys[i], kb = top_keys[ixj];
                        const int32_t  ia = top_ids[i],  ib = top_ids[ixj];
                        const bool a_better = (ka > kb) || (ka == kb && ia < ib);
                        const bool swap = asc ? !a_better : a_better;
                        if (swap) {
                            top_keys[i] = kb; top_ids[i] = ib;
                            top_keys[ixj] = ka; top_ids[ixj] = ia;
                        }
                    }
                }
                __syncthreads();
            }
        }
        for (int j = tid; j < A.topk; j += 256) {
            const int32_t id = top_ids[j];
            const int pg = dsa_page(A, r, id);
            A.ids[(long long)m * A.idsp + (long long)j * A.idss] = id;
            A.slots[(long long)m * A.slp + (long long)j * A.sls] = pg * A.bs + (id % A.bs);
        }
        kept = A.topk;
        __syncthreads();
    }

    dsa_window_tail(A, m, r, seq, kept, ismem);
}

/* ------------------------------------------------------------------ ernie_dsa_select, multi-block form
 * The same selection as ernie_dsa_select_kernel (candidates, order, tie-break, window tail, padding: bit for bit), split over `nb` blocks per row so that a decode row is not
 * one block of one compute unit. Launch sequence (grid (nb, M); a block past the row's candidate end returns at once):
 *   init       zero the scratch                                                         (grid M)
 *   pass 0..3  per-block 256-bin histogram of one key byte over the candidates matching the prefix found so far, added into a global histogram: the radix threshold
 *              of the single-block kernel, MSB first, one launch per byte (a launch boundary is the grid-wide barrier)
 *   count      per block: candidates strictly above the threshold key T and equal to it
 *   gather     compaction in id order: strictly-greater ids by block offset, then the first `need = topk - #greater` ties in id order (the lowest ids, as the single-block kernel)
 *   finalize   one block per row: re-reads the keys of the list, bitonic sort by (key desc, id asc) when more than topk candidates, the window tail
 * When the candidates fit (n_cand <= topk) the passes after 0 return at once, T = 0, nothing is tied and the gather list is the answer in ascending id order.
 * Scratch lives in rows the op owns anyway (its outputs, written in full by the finalize kernel; a row that refuses is never touched, as in the single-block kernel):
 *   slots row: [0, 1024) the four histograms, [1024, 1024 + nb) per-block "greater" counts, [1024 + nb, 1024 + 2 nb) per-block tie counts
 *   ids row:   [0, topk) the gather list
 * The slice of block b is [b * sl, min(end, (b + 1) * sl)) with sl = ceil(end / nb) rounded up to 256, a function of the row's own end only. */
/* the verdict of the last tiled launch (bits 1 non-finite score, 2 bad position or request, 4 range past the host bound) for a synchronous launcher; see ernie_dsa_finalize_kernel */
constexpr int kDselBins = 1024;
constexpr int kDselMaxBlocks = 64;

/* the row's own refusal bits (2 bad position or request, 4 candidate range past the plane / the host bound), as the single-block kernel and the merge kernel set them */
__device__ __forceinline__ bool dsa_row_ok(const DsaSelectArgs& A, int m, int& seq, int& end, int& r, unsigned* bits = nullptr) {
    const int p = A.pos[(long long)m * A.ps];
    r = A.req[(long long)m * A.rs];
    if (p < 0 || r < 0 || r >= A.S) { if (bits) *bits = 2u; return false; }
    const long long seq64 = (long long)p + 1;
    const long long end64 = (seq64 - (long long)A.window > 0) ? (seq64 - (long long)A.window) : 0;
    if (end64 > (A.tiled ? (long long)A.bound : (long long)A.L)) { if (bits) *bits = 4u; return false; }
    seq = (int)seq64;
    end = (int)end64;
    if (bits) *bits = 0u;
    return true;
}
__device__ __forceinline__ int* dsa_scr(const DsaSelectArgs& A, int m, int i) {
    return A.tiled ? A.scr + (long long)m * A.scrp + i : A.slots + (long long)m * A.slp + (long long)i * A.sls;
}
__device__ __forceinline__ int* dsa_lst(const DsaSelectArgs& A, int m, int i) {
    return A.tiled ? A.scr + (long long)m * A.scrp + (kDselBins + 2 * A.nb) + i : A.ids + (long long)m * A.idsp + (long long)i * A.idss;
}
__device__ __forceinline__ int dsa_slice(const DsaSelectArgs& A, int end) { return (((end + A.nb - 1) / A.nb) + 255) & ~255; }
/* a candidate: score above -inf (NaN is not) on an initialised page; key = the order key of the score */
__device__ __forceinline__ bool dsa_cand(const DsaSelectArgs& A, const float* row, int r, int id, uint32_t& key) {
    const float v = row[(long long)id * A.scs];
    if (!(v > -INFINITY) || dsa_page(A, r, id) < 0) return false;
    key = f32_order_key(v);
    return true;
}

/* n_cand, and the radix state after `upto` passes (prefix of the threshold key, rank still to find inside it). Whole block, reads the global histograms. */
__device__ void dsa_sel_derive(const DsaSelectArgs& A, int m, int upto, int* ismem, uint32_t& prefix, int& rem, int& ncand) {
    const int tid = (int)threadIdx.x, lane = tid & 31, wave = tid >> 5;
    ncand = block_sum_i(*dsa_scr(A, m, tid), ismem, 256);
    prefix = 0u;
    rem = A.topk;
    if (ncand <= A.topk) return;
    for (int pass = 0; pass < upto; ++pass) {
        const uint32_t c_bin = (uint32_t)*dsa_scr(A, m, pass * 256 + tid);
        const uint32_t w_suf = r4d_wave_suffix_u32(c_bin, lane);
        if (lane == 0) ismem[wave] = (int)w_suf;
        __syncthreads();
        uint32_t hi_sum = 0u;
#pragma unroll
        for (int w = 7; w > 0; --w) {
            if (w > wave) hi_sum += (uint32_t)ismem[w];
        }
        const uint32_t suf_inc = w_suf + hi_sum;
        const uint32_t suf_exc = suf_inc - c_bin;
        __syncthreads();
        if (suf_inc >= (uint32_t)rem && suf_exc < (uint32_t)rem) {
            ismem[0] = tid;
            ismem[1] = rem - (int)suf_exc;
        }
        __syncthreads();
        prefix |= ((uint32_t)ismem[0] << (24 - pass * 8));
        rem = ismem[1];
        __syncthreads();
    }
}

__global__ void ernie_dsa_sel_init_kernel(DsaSelectArgs A) {
    const int m = (int)blockIdx.x;
    if (m >= A.M) return;
    int seq, end, r;
    unsigned bits = 0u;
    const bool ok = dsa_row_ok(A, m, seq, end, r, &bits);
    if (A.tiled) { if (threadIdx.x == 0) A.flags[m] = (int)bits; }   /* every row's verdict word starts here; pass 0 adds bit 1 */
    else if (!ok && threadIdx.x == 0) r4d_verdict::raise(A.vc.log, bits);   /* as the single-block kernel: the row stores nothing, one event */
    if (!ok) return;
    for (int i = (int)threadIdx.x; i < kDselBins + 2 * A.nb; i += 256) *dsa_scr(A, m, i) = 0;
}

__global__ void ernie_dsa_sel_pass_kernel(DsaSelectArgs A) {
    __shared__ int ismem[8];
    __shared__ uint32_t hist[256];
    const int b = (int)blockIdx.x, m = (int)blockIdx.y, tid = (int)threadIdx.x;
    if (m >= A.M) return;
    int seq, end, r;
    if (!dsa_row_ok(A, m, seq, end, r)) return;
    const long long lo64 = (long long)b * dsa_slice(A, end);
    if (lo64 >= end) return;
    const int lo = (int)lo64;
    const int hi = (int)((lo64 + dsa_slice(A, end) < (long long)end) ? lo64 + dsa_slice(A, end) : (long long)end);
    uint32_t prefix = 0u;
    int rem = A.topk, ncand = 0;
    if (A.pass > 0) {
        dsa_sel_derive(A, m, A.pass, ismem, prefix, rem, ncand);
        if (ncand <= A.topk) return;
    }
    const int shift = 24 - A.pass * 8;
    const uint32_t mask = (A.pass == 0) ? 0u : (~0u << (shift + 8));
    const float* row = A.sc + (long long)m * A.scp;
    hist[tid] = 0u;
    __syncthreads();
    for (int id = lo + tid; id < hi; id += 256) {
        uint32_t k;
        if (A.tiled && A.pass == 0) {   /* the tiled op refuses a NaN or +inf in [0, end) (libref's rule, any page): latch it in the row's verdict word; only the finalize kernel reads it */
            const float v = row[(long long)id * A.scs];
            if (v != v || v == INFINITY) *(volatile int*)(A.flags + m) = 1;
        }
        if (dsa_cand(A, row, r, id, k) && (k & mask) == prefix) atomicAdd(&hist[(k >> shift) & 0xffu], 1u);
    }
    __syncthreads();
    if (hist[tid]) atomicAdd((unsigned*)dsa_scr(A, m, A.pass * 256 + tid), hist[tid]);
}

__global__ void ernie_dsa_sel_count_kernel(DsaSelectArgs A) {
    __shared__ int ismem[8];
    const int b = (int)blockIdx.x, m = (int)blockIdx.y, tid = (int)threadIdx.x;
    if (m >= A.M) return;
    int seq, end, r;
    if (!dsa_row_ok(A, m, seq, end, r)) return;
    const int sl = dsa_slice(A, end);
    const long long lo64 = (long long)b * sl;
    if (lo64 >= end) {
        if (tid == 0) { *dsa_scr(A, m, kDselBins + b) = 0; *dsa_scr(A, m, kDselBins + A.nb + b) = 0; }
        return;
    }
    const int lo = (int)lo64, hi = (int)((lo64 + sl < (long long)end) ? lo64 + sl : (long long)end);
    uint32_t prefix;
    int rem, ncand;
    dsa_sel_derive(A, m, 4, ismem, prefix, rem, ncand);
    const uint32_t T = (ncand > A.topk) ? prefix : 0u;
    const float* row = A.sc + (long long)m * A.scp;
    int gt = 0, eq = 0;
    for (int id = lo + tid; id < hi; id += 256) {
        uint32_t k;
        if (dsa_cand(A, row, r, id, k)) { gt += (k > T) ? 1 : 0; eq += (k == T) ? 1 : 0; }
    }
    gt = block_sum_i(gt, ismem, 256);
    eq = block_sum_i(eq, ismem, 256);
    if (tid == 0) { *dsa_scr(A, m, kDselBins + b) = gt; *dsa_scr(A, m, kDselBins + A.nb + b) = eq; }
}

__global__ void ernie_dsa_sel_gather_kernel(DsaSelectArgs A) {
    __shared__ int ismem[16];
    const int b = (int)blockIdx.x, m = (int)blockIdx.y, tid = (int)threadIdx.x;
    const int lane = tid & 31, wave = tid >> 5;
    if (m >= A.M) return;
    int seq, end, r;
    if (!dsa_row_ok(A, m, seq, end, r)) return;
    const int sl = dsa_slice(A, end);
    const long long lo64 = (long long)b * sl;
    if (lo64 >= end) return;
    const int lo = (int)lo64, hi = (int)((lo64 + sl < (long long)end) ? lo64 + sl : (long long)end);
    uint32_t prefix;
    int rem, ncand;
    dsa_sel_derive(A, m, 4, ismem, prefix, rem, ncand);
    const bool over = ncand > A.topk;
    const uint32_t T = over ? prefix : 0u;
    int gt_base = 0, eq_base = 0, gt_total = 0;
    for (int i = 0; i < A.nb; ++i) {
        const int g = *dsa_scr(A, m, kDselBins + i), e = *dsa_scr(A, m, kDselBins + A.nb + i);
        if (i < b) { gt_base += g; eq_base += e; }
        gt_total += g;
    }
    const int need = over ? A.topk - gt_total : 0;
    const float* row = A.sc + (long long)m * A.scp;
    for (int c = lo; c < hi; c += 256) {
        const int id = c + tid;
        uint32_t k = 0u;
        const bool cand = (id < hi) && dsa_cand(A, row, r, id, k);
        const uint32_t g = (cand && k > T) ? 1u : 0u, e = (cand && k == T) ? 1u : 0u;
        const uint32_t gi = r4d_wave_prefix_u32(g, lane), ei = r4d_wave_prefix_u32(e, lane);
        if (lane == 31) { ismem[wave] = (int)gi; ismem[8 + wave] = (int)ei; }
        __syncthreads();
        int g_off = 0, e_off = 0, g_tot = 0, e_tot = 0;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            if (w < wave) { g_off += ismem[w]; e_off += ismem[8 + w]; }
            g_tot += ismem[w];
            e_tot += ismem[8 + w];
        }
        if (g) *dsa_lst(A, m, gt_base + g_off + (int)gi - 1) = id;
        if (e) {
            const int er = eq_base + e_off + (int)ei - 1;
            if (er < need) *dsa_lst(A, m, gt_total + er) = id;
        }
        gt_base += g_tot;
        eq_base += e_tot;
        __syncthreads();
    }
}

__global__ void ernie_dsa_sel_final_kernel(DsaSelectArgs A) {
    __shared__ int ismem[8];
    __shared__ uint32_t top_keys[2048];
    __shared__ int32_t  top_ids[2048];
    const int m = (int)blockIdx.x, tid = (int)threadIdx.x;
    if (m >= A.M) return;
    if (A.tiled) {
        /* the tiled op's rule (as ernie_dsa_finalize_kernel): every block reduces the same M verdict words, so all blocks agree; any bit set anywhere and no block stores;
         * block 0 publishes the verdict (the synchronous launcher's word and the event) */
        int any = 0;
        for (int i = tid; i < A.M; i += 256) any |= *(volatile const int32_t*)(A.flags + i);
        int verdict = 0;
        for (int bit = 1; bit <= 4; bit <<= 1) verdict |= (block_sum_i((any & bit) ? 1 : 0, ismem, 256) > 0) ? bit : 0;
        if (m == 0 && tid == 0) {
            if (verdict != 0) r4d_verdict::raise(A.vc.log, (unsigned)verdict);
        }
        if (verdict != 0) return;
    }
    int seq, end, r;
    if (!dsa_row_ok(A, m, seq, end, r)) return;
    const float* row = A.sc + (long long)m * A.scp;
    const int ncand = block_sum_i(*dsa_scr(A, m, tid), ismem, 256);
    const int kept = (ncand > A.topk) ? A.topk : ncand;
    for (int j = tid; j < kept; j += 256) {
        const int id = *dsa_lst(A, m, j);
        top_ids[j] = id;
        top_keys[j] = f32_order_key(row[(long long)id * A.scs]);
    }
    __syncthreads();
    if (ncand > A.topk) {
        /* sort by (key desc, id asc), padded to a power of two with a sentinel that sorts last: the single-block kernel's network */
        int P2 = 1;
        while (P2 < A.topk) P2 <<= 1;
        for (int i = A.topk + tid; i < P2; i += 256) { top_keys[i] = 0u; top_ids[i] = INT_MAX; }
        __syncthreads();
        for (int size = 2; size <= P2; size <<= 1) {
            for (int stride = size >> 1; stride > 0; stride >>= 1) {
                for (int i = tid; i < P2; i += 256) {
                    const int ixj = i ^ stride;
                    if (ixj > i) {
                        const bool asc = ((i & size) == 0);
                        const uint32_t ka = top_keys[i], kb = top_keys[ixj];
                        const int32_t  ia = top_ids[i],  ib = top_ids[ixj];
                        const bool a_better = (ka > kb) || (ka == kb && ia < ib);
                        const bool swap = asc ? !a_better : a_better;
                        if (swap) {
                            top_keys[i] = kb; top_ids[i] = ib;
                            top_keys[ixj] = ka; top_ids[ixj] = ia;
                        }
                    }
                }
                __syncthreads();
            }
        }
    }
    for (int j = tid; j < kept; j += 256) {
        const int32_t id = top_ids[j];
        const int pg = dsa_page(A, r, id);
        A.ids[(long long)m * A.idsp + (long long)j * A.idss] = id;
        A.slots[(long long)m * A.slp + (long long)j * A.sls] = pg * A.bs + (id % A.bs);
    }
    __syncthreads();
    dsa_window_tail(A, m, r, seq, kept, ismem);
}


/* ------------------------------------------------------------------ ernie_dsa_topk_tiled (device)
 * The tiled selection (docs/ernie55/score-tiling.md): per context tile one ernie_idx_logits launch with col_start, one merge kernel, and after the
 * last tile a finalize kernel. libref's contract is that nothing is stored unless every tile of every row is valid (a NaN or +inf score in a row's
 * candidate range, a bad position or request, or a candidate range past the host bound refuses the launch before the first store), so the running
 * top-k of a row lives in the op's own scratch operand (`kept`, flat i32), never in `ids` / `slots`, and the finalize kernel is the only writer of
 * the outputs. Layout of `kept` for a launch of M rows, KW = 2 * topk + 2:
 *   kept[m * KW + 0 .. topk)        order keys (f32_order_key) of row m's kept candidates
 *   kept[m * KW + topk .. 2 * topk) their logical ids
 *   kept[m * KW + 2 * topk]         number of kept candidates, kept[m * KW + 2 * topk + 1] candidates seen over all tiles
 *   kept[M * KW + m]                row m's verdict bits (1 non-finite score, 2 bad position or request, 4 candidate range past the host bound)
 * The verdict words are contiguous so the finalize kernel's blocks each read all M of them (4 M bytes, from L2) and take the same decision: any bit
 * set, anywhere, and no block stores. While at most topk candidates have been seen the kept list is in ascending id order; the first merge that
 * crosses topk cuts it to the topk best, sorted (key descending, id ascending), and it stays so. That is the output order of ernie_dsa_select in both
 * cases. The selection code below is the select kernel's (radix threshold, strictly-greater gather, lowest-id tie take, bitonic sort on (key, -id))
 * run over the union of the kept list and the tile's candidates: kept ids are all below the tile's, and equal keys stay in ascending id order inside
 * the kept list, so scanning the kept list first and the tile second takes the lowest ids among equal keys, as the select kernel does over a whole
 * row. */
struct DsaMergeArgs {
    const float *plane;
    const int32_t *pos, *req, *bt;
    int32_t *ids, *slots, *kept;
    long long planep, planes, ps, rs, btp, bts, idsp, idss, slp, sls;
    int M, S, B, topk, window, bs, pages, col0, tile, out_w, bound;   /* bound: the host bound on a candidate range [0, end) */

    r4d_verdict::Vc vc;   /* this launch's event log (device memory the library owns; constant across replays) */
};
static_assert(sizeof(DsaMergeArgs) <= 512, "DsaMergeArgs exceeds 512B AQL limit");

/* The verdict of the last finalize (bits as in the layout above) for a synchronous launcher, and the event log the host polls: a refused launch raises ONE event
 * carrying its bits (1 non-finite score, 2 bad position or request, 4 candidate range past the host bound) into g_dsa_log, the same monotonic log every other ERNIE
 * validator uses (r4d_ernie_verdict.h; take never clears device state, the engine polls it through rad_kernel_poll_error). */

__device__ __forceinline__ int merge_page(const DsaMergeArgs& A, int r, int id) {
    const int b = id / A.bs;
    if (b >= A.B) return -1;
    const int e = A.bt[(long long)r * A.btp + (long long)b * A.bts];
    return (e >= 0 && e < A.pages) ? e : -1;
}

/* Element i of the union: i < n_state is a kept candidate (`kk`, `ki` the row's key and id lists), the rest are the tile's columns. Returns whether it is a candidate. */
__device__ __forceinline__ bool merge_elem(const DsaMergeArgs& A, const int32_t* kk, const int32_t* ki, int m, int r, int n_state, int end, int i, uint32_t* key, int* id) {
    if (i < n_state) {
        *key = (uint32_t)kk[i];
        *id = ki[i];
        return true;
    }
    const int jj = i - n_state, gid = A.col0 + jj;
    if (gid >= end || jj >= A.tile) return false;
    const float v = A.plane[(long long)m * A.planep + (long long)jj * A.planes];
    if (!(v > -INFINITY) || merge_page(A, r, gid) < 0) return false;
    *key = f32_order_key(v);
    *id = gid;
    return true;
}

__global__ void ernie_dsa_merge_kernel(DsaMergeArgs A) {
    __shared__ int ismem[8];
    __shared__ uint32_t hist[256];
    __shared__ uint32_t top_keys[2048];
    __shared__ int32_t  top_ids[2048];
    __shared__ int32_t  gt_cnt;
    __shared__ int      st_n, st_cnt;

    const int m = (int)blockIdx.x, tid = (int)threadIdx.x;
    const int lane = tid & 31, wave = tid >> 5;
    if (m >= A.M) return;
    const long long KW = 2LL * A.topk + 2;
    int32_t* kk = A.kept + (long long)m * KW;           /* keys */
    int32_t* ki = kk + A.topk;                          /* ids */
    int32_t* kn = kk + 2LL * A.topk;                    /* [0] kept count, [1] candidates seen */
    int32_t* flag = A.kept + (long long)A.M * KW + m;   /* this row's verdict */
    const int p = A.pos[(long long)m * A.ps], r = A.req[(long long)m * A.rs];
    int f = 0;
    long long end64 = 0;
    if (p < 0 || r < 0 || r >= A.S) f = 2;
    else {
        end64 = ((long long)p + 1 - (long long)A.window > 0) ? ((long long)p + 1 - (long long)A.window) : 0;
        if (end64 > (long long)A.bound) f = 4;
    }
    if (A.col0 == 0) { if (tid == 0) *flag = f; }       /* the first tile starts the row's verdict (the tile loop has at least one tile) */
    else f |= *(volatile const int32_t*)flag;           /* a later tile: the sticky verdict; written only by this block */
    if (f != 0) return;                                 /* an invalid row stores nothing and scores nothing further; the finalize kernel sees the flag */
    const int end = (int)end64;
    int cols = end - A.col0;
    if (cols > A.tile) cols = A.tile;
    if (cols < 0) cols = 0;
    if (A.col0 > 0 && cols == 0) return;   /* this row has nothing in the tile: the kept list stays as it is */
    if (tid == 0) {
        st_n = (A.col0 > 0) ? kn[0] : 0;
        st_cnt = (A.col0 > 0) ? kn[1] : 0;
    }
    __syncthreads();
    const int n_state = st_n, cnt_prev = st_cnt;

    /* candidates in the tile, and the scores libref refuses (NaN, +inf) in the same pass over the plane */
    int local = 0, bad = 0;
    for (int jj = tid; jj < cols; jj += 256) {
        const float v = A.plane[(long long)m * A.planep + (long long)jj * A.planes];
        if (v != v || v == INFINITY) bad = 1;
        uint32_t k; int id;
        if (merge_elem(A, kk, ki, m, r, n_state, end, n_state + jj, &k, &id)) ++local;
    }
    const int n_tile = block_sum_i(local, ismem, 256);
    const int n_bad = block_sum_i(bad, ismem, 256);
    if (n_bad > 0) {                                    /* the row is refused: latch it and stop merging */
        if (tid == 0) *flag = 1;
        return;
    }
    const int total = n_state + n_tile;
    const int cnt_total = cnt_prev + n_tile;

    if (total <= A.topk) {
        /* nothing is cut: append the tile's candidates, in id order, behind the kept ones */
        int base = n_state;
        for (int j0 = 0; j0 < cols; j0 += 256) {
            const int jj = j0 + tid;
            uint32_t k = 0; int id = 0;
            const bool valid = (jj < cols) && merge_elem(A, kk, ki, m, r, n_state, end, n_state + jj, &k, &id);
            const uint32_t w_inc = r4d_wave_prefix_u32(valid ? 1u : 0u, lane);
            if (lane == 31) ismem[wave] = (int)w_inc;
            __syncthreads();
            int w_off = 0, chunk_tot = 0;
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                if (w < wave) w_off += ismem[w];
                chunk_tot += ismem[w];
            }
            if (valid) {
                const int o = base + w_off + (int)w_inc - 1;
                ki[o] = id;
                kk[o] = (int32_t)k;
            }
            base += chunk_tot;
            __syncthreads();
        }
    } else {
        /* more than topk in the union: 4-pass radix threshold for the topk-th largest key, as ernie_dsa_select, over kept list + tile */
        const int U = n_state + cols;
        uint32_t prefix = 0u;
        int rem = A.topk;
        for (int pass = 0; pass < 4; ++pass) {
            const int shift = 24 - pass * 8;
            const uint32_t mask = (pass == 0) ? 0u : (~0u << (shift + 8));
            hist[tid] = 0u;
            __syncthreads();
            for (int i = tid; i < U; i += 256) {
                uint32_t k; int id;
                if (merge_elem(A, kk, ki, m, r, n_state, end, i, &k, &id) && (k & mask) == prefix) atomicAdd(&hist[(k >> shift) & 0xffu], 1u);
            }
            __syncthreads();
            const uint32_t c_bin = hist[tid];
            const uint32_t w_suf = r4d_wave_suffix_u32(c_bin, lane);
            if (lane == 0) ismem[wave] = (int)w_suf;
            __syncthreads();
            uint32_t hi_sum = 0u;
#pragma unroll
            for (int w = 7; w > 0; --w) {
                if (w > wave) hi_sum += (uint32_t)ismem[w];
            }
            const uint32_t suf_inc = w_suf + hi_sum;
            const uint32_t suf_exc = suf_inc - c_bin;
            __syncthreads();
            if (suf_inc >= (uint32_t)rem && suf_exc < (uint32_t)rem) {
                ismem[0] = tid;
                ismem[1] = rem - (int)suf_exc;
            }
            __syncthreads();
            prefix |= ((uint32_t)ismem[0] << shift);
            rem = ismem[1];
            __syncthreads();
        }
        const uint32_t thresh = prefix;
        if (tid == 0) gt_cnt = 0;
        __syncthreads();
        for (int i = tid; i < U; i += 256) {
            uint32_t k; int id;
            if (merge_elem(A, kk, ki, m, r, n_state, end, i, &k, &id) && k > thresh) {
                const int idx = atomicAdd(&gt_cnt, 1);
                if (idx < 2048) { top_keys[idx] = k; top_ids[idx] = id; }
            }
        }
        __syncthreads();
        int pos_eq = gt_cnt;
        int need = A.topk - pos_eq;
        for (int i0 = 0; i0 < U && need > 0; i0 += 256) {
            const int i = i0 + tid;
            uint32_t k = 0; int id = 0;
            const bool match = (i < U) && merge_elem(A, kk, ki, m, r, n_state, end, i, &k, &id) && k == thresh;
            const uint32_t w_inc = r4d_wave_prefix_u32(match ? 1u : 0u, lane);
            if (lane == 31) ismem[wave] = (int)w_inc;
            __syncthreads();
            int w_off = 0, chunk_tot = 0;
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                if (w < wave) w_off += ismem[w];
                chunk_tot += ismem[w];
            }
            if (match) {
                const int rank_in_chunk = w_off + (int)w_inc - 1;
                if (rank_in_chunk < need && pos_eq + rank_in_chunk < 2048) {
                    top_keys[pos_eq + rank_in_chunk] = thresh;
                    top_ids[pos_eq + rank_in_chunk] = id;
                }
            }
            const int taken = (chunk_tot < need) ? chunk_tot : need;
            pos_eq += taken;
            need -= taken;
            __syncthreads();
        }
        int P2 = 1;
        while (P2 < A.topk) P2 <<= 1;
        for (int i = A.topk + tid; i < P2; i += 256) { top_keys[i] = 0u; top_ids[i] = INT_MAX; }
        __syncthreads();
        for (int size = 2; size <= P2; size <<= 1) {
            for (int stride = size >> 1; stride > 0; stride >>= 1) {
                for (int i = tid; i < P2; i += 256) {
                    const int ixj = i ^ stride;
                    if (ixj > i) {
                        const bool asc = ((i & size) == 0);
                        const uint32_t ka = top_keys[i], kb = top_keys[ixj];
                        const int32_t  ia = top_ids[i],  ib = top_ids[ixj];
                        const bool a_better = (ka > kb) || (ka == kb && ia < ib);
                        const bool swap = asc ? !a_better : a_better;
                        if (swap) {
                            top_keys[i] = kb; top_ids[i] = ib;
                            top_keys[ixj] = ka; top_ids[ixj] = ia;
                        }
                    }
                }
                __syncthreads();
            }
        }
        /* every read of the kept list is behind this barrier: overwrite it with the new top-k */
        for (int j = tid; j < A.topk; j += 256) {
            ki[j] = top_ids[j];
            kk[j] = (int32_t)top_keys[j];
        }
        __syncthreads();
    }
    if (tid == 0) {
        kn[0] = (total <= A.topk) ? total : A.topk;
        kn[1] = cnt_total;
    }
}

/* After the last tile: the launch's verdict, then (only if no row of the launch was refused) slots of the kept ids, the window, -1 padding. Every block
 * reduces the same M verdict words, so all blocks agree and a refused launch stores nothing anywhere; block 0 publishes the verdict. */
__global__ void ernie_dsa_finalize_kernel(DsaMergeArgs A) {
    __shared__ int ismem[8];
    __shared__ int st_n;
    const int m = (int)blockIdx.x, tid = (int)threadIdx.x;
    const int lane = tid & 31, wave = tid >> 5;
    if (m >= A.M) return;
    const long long KW = 2LL * A.topk + 2;
    const int32_t* flags = A.kept + (long long)A.M * KW;
    int any = 0;
    for (int i = tid; i < A.M; i += 256) any |= *(volatile const int32_t*)(flags + i);
    /* OR over the block: a bit at a time through the integer reduction (3 bits, 3 reductions, on the M-word read above) */
    int verdict = 0;
    for (int bit = 1; bit <= 4; bit <<= 1) verdict |= (block_sum_i((any & bit) ? 1 : 0, ismem, 256) > 0) ? bit : 0;
    if (m == 0 && tid == 0) {
        if (verdict != 0) r4d_verdict::raise(A.vc.log, (unsigned)verdict);
    }
    if (verdict != 0) return;
    const int seq = A.pos[(long long)m * A.ps] + 1;
    const int rr = A.req[(long long)m * A.rs];
    const int32_t* ki = A.kept + (long long)m * KW + A.topk;
    int32_t* idrow = A.ids + (long long)m * A.idsp;
    int32_t* slrow = A.slots + (long long)m * A.slp;
    if (tid == 0) st_n = A.kept[(long long)m * KW + 2LL * A.topk];
    __syncthreads();
    const int kept = st_n;
    for (int j = tid; j < kept; j += 256) {
        const int id = ki[j];
        const int pg = merge_page(A, rr, id);
        idrow[(long long)j * A.idss] = id;
        slrow[(long long)j * A.sls] = pg * A.bs + (id % A.bs);
    }
    __syncthreads();
    const int wlo = seq - (seq < A.window ? seq : A.window);
    int total_written = kept;
    for (int id0 = wlo; id0 < seq; id0 += 256) {
        const int id = id0 + tid;
        const int pg = (id < seq) ? merge_page(A, rr, id) : -1;
        const uint32_t valid = (pg >= 0) ? 1u : 0u;
        const uint32_t w_inc = r4d_wave_prefix_u32(valid, lane);
        if (lane == 31) ismem[wave] = (int)w_inc;
        __syncthreads();
        int w_off = 0, chunk_tot = 0;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            if (w < wave) w_off += ismem[w];
            chunk_tot += ismem[w];
        }
        if (valid) {
            const int pos_out = total_written + w_off + (int)w_inc - 1;
            idrow[(long long)pos_out * A.idss] = id;
            slrow[(long long)pos_out * A.sls] = pg * A.bs + (id % A.bs);
        }
        total_written += chunk_tot;
        __syncthreads();
    }
    for (int j = total_written + tid; j < A.out_w; j += 256) {
        idrow[(long long)j * A.idss] = -1;
        slrow[(long long)j * A.sls] = -1;
    }
}

/* ------------------------------------------------------------------ ernie_mla_project */
struct MlaProjArgs {
    const uint16_t *x, *w;
    uint16_t *y;
    long long xp, xs, wp, ws, yp, ys;
    int M, H, kd, vd, L, xw, yw, absorb;
};
static_assert(sizeof(MlaProjArgs) <= 512, "MlaProjArgs exceeds 512B AQL limit");

/* Multi-row tiled absorb: adjacent threads own adjacent `col` (100% coalesced weight reads!),
 * reusing every weight element across R tokens (x is uniform across the block). */
template <int R>
__global__ void ernie_mla_project_absorb_rows(MlaProjArgs A) {
    const int col = (int)blockIdx.x * 128 + (int)threadIdx.x;
    const int h = (int)blockIdx.y;
    const int r0 = (int)blockIdx.z * R;
    if (h >= A.H || col >= A.L) return;

    const uint16_t* xr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int rm = (r0 + r < A.M) ? (r0 + r) : (A.M - 1);
        xr[r] = A.x + (long long)rm * A.xp + (long long)(h * A.xw);
    }
    float sum[R];
#pragma unroll
    for (int r = 0; r < R; ++r) sum[r] = 0.0f;

    const uint16_t* wcol = A.w + (long long)(h * (A.kd + A.vd)) * A.wp + col;
    for (int j = 0; j < A.kd; ++j) {
        const float wv = bf2f(wcol[(long long)j * A.wp]);
#pragma unroll
        for (int r = 0; r < R; ++r) sum[r] = fmaf(bf2f(xr[r][j]), wv, sum[r]);
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (r0 + r < A.M)
            A.y[(long long)(r0 + r) * A.yp + (long long)(h * A.L + col)] = f2bf(sum[r]);
    }
}

/* Multi-row tiled expand: 4 waves (128 threads) per block, 1 wave computes 1 output col across R tokens.
 * Uses coalesced uint4 (8 BF16s) loads when L % 256 == 0 and aligned, else coalesced scalar loop. */
template <int R>
__global__ void ernie_mla_project_expand_rows(MlaProjArgs A) {
    const int lane = (int)(threadIdx.x & 31);
    const int col = (int)blockIdx.x * 4 + (int)(threadIdx.x >> 5);
    const int h = (int)blockIdx.y;
    const int r0 = (int)blockIdx.z * R;
    if (h >= A.H || col >= A.vd) return;

    const uint16_t* xr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int rm = (r0 + r < A.M) ? (r0 + r) : (A.M - 1);
        xr[r] = A.x + (long long)rm * A.xp + (long long)(h * A.xw);
    }
    const uint16_t* wr = A.w + (long long)(h * (A.kd + A.vd) + A.kd + col) * A.wp;

    float sum[R];
#pragma unroll
    for (int r = 0; r < R; ++r) sum[r] = 0.0f;

    const bool vec8 = ((A.L & 255) == 0) && (((uintptr_t)wr & 15u) == 0u) &&
                      (((uintptr_t)A.x & 15u) == 0u) && ((A.xp & 7) == 0) && ((A.xw & 7) == 0);
    if (vec8) {
        for (int l0 = 0; l0 < A.L; l0 += 256) {
            const uint4 wu = *(const uint4*)(wr + l0 + lane * 8);
            const float w0 = bf2f_lo(wu.x), w1 = bf2f_hi(wu.x);
            const float w2 = bf2f_lo(wu.y), w3 = bf2f_hi(wu.y);
            const float w4 = bf2f_lo(wu.z), w5 = bf2f_hi(wu.z);
            const float w6 = bf2f_lo(wu.w), w7 = bf2f_hi(wu.w);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint4 xu = *(const uint4*)(xr[r] + l0 + lane * 8);
                float acc = sum[r];
                acc = fmaf(bf2f_lo(xu.x), w0, acc);
                acc = fmaf(bf2f_hi(xu.x), w1, acc);
                acc = fmaf(bf2f_lo(xu.y), w2, acc);
                acc = fmaf(bf2f_hi(xu.y), w3, acc);
                acc = fmaf(bf2f_lo(xu.z), w4, acc);
                acc = fmaf(bf2f_hi(xu.z), w5, acc);
                acc = fmaf(bf2f_lo(xu.w), w6, acc);
                acc = fmaf(bf2f_hi(xu.w), w7, acc);
                sum[r] = acc;
            }
        }
    } else {
        for (int l = lane; l < A.L; l += 32) {
            const float wv = bf2f(wr[l]);
#pragma unroll
            for (int r = 0; r < R; ++r) sum[r] = fmaf(bf2f(xr[r][l]), wv, sum[r]);
        }
    }

    float mine = 0.0f;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float s = r4d_xor_sum<32>(sum[r]);
        if (lane == r) mine = s;
    }
    if (lane < R && r0 + lane < A.M)
        A.y[(long long)(r0 + lane) * A.yp + (long long)(h * A.vd + col)] = f2bf(mine);
}

__global__ void ernie_mla_project_kernel(MlaProjArgs A) {
    /* General strided fallback: 128 threads = 4 waves; each wave computes 1 output element (m, h, col). */
    const int lane = (int)(threadIdx.x & 31), wave = (int)(threadIdx.x >> 5);
    const int col = (int)blockIdx.x * 4 + wave;
    const int h = (int)blockIdx.y, m = (int)blockIdx.z;
    if (m >= A.M || h >= A.H || col >= A.yw) return;
    float sum = 0.0f;
    if (A.absorb) {
        const long long xb = (long long)m * A.xp + (long long)(h * A.xw) * A.xs;
        const int wrow0 = h * (A.kd + A.vd);
        for (int j = lane; j < A.kd; j += 32)
            sum += bf2f(A.x[xb + (long long)j * A.xs]) * bf2f(A.w[(long long)(wrow0 + j) * A.wp + (long long)col * A.ws]);
    } else {
        const long long xb = (long long)m * A.xp + (long long)(h * A.xw) * A.xs;
        const long long wb = (long long)(h * (A.kd + A.vd) + A.kd + col) * A.wp;
        for (int l = lane; l < A.L; l += 32)
            sum += bf2f(A.x[xb + (long long)l * A.xs]) * bf2f(A.w[wb + (long long)l * A.ws]);
    }
    sum = r4d_xor_sum<32>(sum);
    if (lane == 0) A.y[(long long)m * A.yp + (long long)(h * A.yw + col) * A.ys] = f2bf(sum);
}

/* ------------------------------------------------------------------ ernie_mla_sparse_attn */
struct MlaSparseArgs {
    const uint16_t *q, *qr, *kv;
    const int32_t *slots;
    const float *sink;
    uint16_t *out;
    long long qp, qs, qrp, qrs, kvp0, kvp2, kvs, slp, sls, sks, outp, outs;
    int M, H, L, R, bs, pages, W;
    float scale;
    int fp8, qb;   /* kv_layout of the u8 latent row, qb latent values to a scale: 1 fp8_ds_mla (E4M3 codes), 2 i8_ds_mla (int8 codes, v2), 3 i10_ds_mla, 4 i12_ds_mla (int8 high bytes + packed low bits); 0 = the bf16 row */
};
static_assert(sizeof(MlaSparseArgs) <= 512, "MlaSparseArgs exceeds 512B AQL limit");

/* Helper: load 8 BF16s from 16B-aligned pointer into 8 floats. */
__device__ __forceinline__ void load_bf8(const uint16_t* p, float out[8]) {
    const uint4 u = *(const uint4*)p;
    out[0] = bf2f_lo(u.x); out[1] = bf2f_hi(u.x);
    out[2] = bf2f_lo(u.y); out[3] = bf2f_hi(u.y);
    out[4] = bf2f_lo(u.z); out[5] = bf2f_hi(u.z);
    out[6] = bf2f_lo(u.w); out[7] = bf2f_hi(u.w);
}

__device__ __forceinline__ void store_bf8(uint16_t* p, const float in[8]) {
    uint4 u;
    u.x = pack_bf2(in[0], in[1]);
    u.y = pack_bf2(in[2], in[3]);
    u.z = pack_bf2(in[4], in[5]);
    u.w = pack_bf2(in[6], in[7]);
    *(uint4*)p = u;
}

/* The fp8_ds_mla latent row (libref/ref_ernie_attn.cpp, kv-fp8-latent): [L E4M3 codes | nt = L / qb little-endian f32 scales | R bf16 rope]. Eight
 * codes from byte c0 (a multiple of 8, so the 8-byte load is aligned when the row is 16-aligned) times the scale of their tile, an exact product: the f32
 * values are those of the bf16 row the codes decode to. */
/* bytes of packed low bits of an integer row: L / 4 (i10_ds_mla), L / 2 (i12_ds_mla), else 0 */
__device__ __host__ __forceinline__ int latent_lo_bytes(int L, int fmt) { return fmt == 3 ? L / 4 : fmt == 4 ? L / 2 : fmt == 5 ? L : 0; }   /* 5: bf16_ds_mla (diagnostic identity row), the bf16 low bytes */
__device__ __forceinline__ void load_fp8x8(const uint8_t* row, int c0, int L, int qb, float out[8], int fmt = 1) {
    const uint2 raw = *(const uint2*)(row + c0);
    const float sc = *(const float*)(row + L + latent_lo_bytes(L, fmt) + 4 * (c0 / qb));
    if (fmt == 5) {   /* bf16_ds_mla: the high bytes here, the low bytes L further, times the stored scale (1.0) */
        const uint2 lo8 = *(const uint2*)(row + L + c0);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            out[i]     = __uint_as_float((((raw.x >> (8 * i)) & 0xffu) << 24) | (((lo8.x >> (8 * i)) & 0xffu) << 16)) * sc;
            out[4 + i] = __uint_as_float((((raw.y >> (8 * i)) & 0xffu) << 24) | (((lo8.y >> (8 * i)) & 0xffu) << 16)) * sc;
        }
        return;
    }
    if (fmt >= 2) {   /* integer rows (v2): int8 high byte (<< lb) plus the low bits, times the exact f32 tile scale, one f32 product each (libref's decode) */
        const int lb = fmt == 3 ? 2 : fmt == 4 ? 4 : 0;
        uint32_t lo = 0;   /* the 8 elements' low bits: lb * 8 bits from byte L + c0 / (8 / lb) (2 or 4 bytes; element i at bit lb * i) */
        if (fmt == 3) lo = *(const uint16_t*)(row + L + (c0 >> 2));
        else if (fmt == 4) lo = *(const uint32_t*)(row + L + (c0 >> 1));
        const uint32_t mk = (1u << lb) - 1u;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            out[i]     = (float)(((int)(int8_t)((raw.x >> (8 * i)) & 0xffu) * (1 << lb)) + (int)((lo >> (lb * i)) & mk)) * sc;   /* * 2^lb, not <<: a negative code (C++17 card build) */
            out[4 + i] = (float)(((int)(int8_t)((raw.y >> (8 * i)) & 0xffu) * (1 << lb)) + (int)((lo >> (lb * (4 + i))) & mk)) * sc;
        }
        return;
    }
    const v2f a = r4d_fp8::e4m3_pk<false>(raw.x), b = r4d_fp8::e4m3_pk<true>(raw.x),
              c = r4d_fp8::e4m3_pk<false>(raw.y), d = r4d_fp8::e4m3_pk<true>(raw.y);
    out[0] = a.x * sc; out[1] = a.y * sc; out[2] = b.x * sc; out[3] = b.y * sc;
    out[4] = c.x * sc; out[5] = c.y * sc; out[6] = d.x * sc; out[7] = d.y * sc;
}
/* One element, any alignment (the general kernel). */
__device__ __forceinline__ float fp8_ds_mla_elem(const uint8_t* row, int c, int L, int qb, int fmt = 1) {
    if (c < L) {
        const int lo = latent_lo_bytes(L, fmt);
        const uint8_t* sp = row + L + lo + 4 * (c / qb);
        const uint32_t u = (uint32_t)sp[0] | ((uint32_t)sp[1] << 8) | ((uint32_t)sp[2] << 16) | ((uint32_t)sp[3] << 24);
        if (fmt == 5) return __uint_as_float(((uint32_t)row[c] << 24) | ((uint32_t)row[L + c] << 16)) * __uint_as_float(u);
        if (fmt >= 2) {
            const int lb = fmt == 3 ? 2 : fmt == 4 ? 4 : 0;
            int v = (int)(int8_t)row[c];
            if (lb) { const int per = 8 / lb; v = v * (1 << lb) + (int)((row[L + c / per] >> ((c % per) * lb)) & ((1u << lb) - 1u)); }   /* * 2^lb: v may be negative */
            return (float)v * __uint_as_float(u);
        }
        return r4d_fp8::e4m3((uint32_t)row[c]) * __uint_as_float(u);
    }
    const uint8_t* rp = row + L + latent_lo_bytes(L, fmt) + 4 * (L / qb) + 2 * (c - L);
    return __uint_as_float(((uint32_t)rp[0] | ((uint32_t)rp[1] << 8)) << 16);
}

/* Sparse MLA attention, L in {256, 512}, R == 64: ONE arithmetic for every row count (the row-identity checks).
 * The row's softmax runs over a fixed 8-way partition of its W slots, independent of the launch:
 *   partition w = the slot chunks j0 = 32 * (w + 8 * i) (32 slots each, i = 0, 1, ...), valid slots in order; partition 0 is seeded with the sink
 *   (mx = sink, den = 1), the others start empty; each runs the online-softmax chain (mla_part)
 *   merge = the 8 partition states folded in partition order (mla_merge), an empty partition (den == 0) skipped.
 * The cooperative decode kernel (M <= R4D_MLA_COOP_MAX rows, W >= 64) runs one partition per wave and folds on all waves through LDS; the wave kernel
 * runs the 8 partitions one after the other in one wave. Both call the two functions below. */
template <int L, bool FP8 = false>
__device__ __forceinline__ void mla_part(const MlaSparseArgs& A, const float (&q_reg)[L / 32], float qr0, float qr1, const int32_t* sl_row, int lane, int part,
                                         int h, float& mx, float& den, float (&acc)[L / 32]) {
    constexpr int CHUNKS = L / 256;
    constexpr int VPL = CHUNKS * 8;
    const int max_slot = A.pages * A.bs;
    const float scale = A.scale;
#pragma unroll
    for (int i = 0; i < VPL; ++i) acc[i] = 0.0f;
    mx = -INFINITY;
    den = 0.0f;
    if (part == 0) {
        const float sk = A.sink[(long long)h * A.sks];
        mx = sk;
        den = (sk == -INFINITY) ? 0.0f : 1.0f;
    }
    for (int j0 = part * 32; j0 < A.W; j0 += 8 * 32) {
        const int j = j0 + lane;
        const int s_lane = (j < A.W) ? sl_row[j] : -1;
        uint32_t mask = __builtin_amdgcn_ballot_w32(s_lane >= 0 && s_lane < max_slot);
        while (mask) {
            const int bit = __builtin_ctz(mask);
            mask &= (mask - 1u);
            const int s = __shfl(s_lane, bit, 32);
            const long long kvoff = (long long)(s / A.bs) * A.kvp0 + (long long)(s % A.bs) * A.kvp2;

            float kv_reg[VPL];
            float dot = 0.0f;
            uint32_t kr_u;
            if constexpr (FP8) {
                /* fp8_ds_mla row (kv-fp8-latent): E4M3 codes times the tile's f32 scale, decoded to the f32 the bf16 row would hold */
                const uint8_t* kv_row = (const uint8_t*)A.kv + kvoff;
#pragma unroll
                for (int ch = 0; ch < CHUNKS; ++ch) {
                    load_fp8x8(kv_row, ch * 256 + lane * 8, L, A.qb, &kv_reg[ch * 8], A.fp8);
#pragma unroll
                    for (int i = 0; i < 8; ++i) dot = fmaf(q_reg[ch * 8 + i], kv_reg[ch * 8 + i], dot);
                }
                kr_u = *(const uint32_t*)(kv_row + L + latent_lo_bytes(L, A.fp8) + 4 * (L / A.qb) + lane * 4);
            } else {
                const uint16_t* kv_row = A.kv + kvoff;
#pragma unroll
                for (int ch = 0; ch < CHUNKS; ++ch) {
                    load_bf8(kv_row + ch * 256 + lane * 8, &kv_reg[ch * 8]);
#pragma unroll
                    for (int i = 0; i < 8; ++i) dot = fmaf(q_reg[ch * 8 + i], kv_reg[ch * 8 + i], dot);
                }
                kr_u = *(const uint32_t*)(kv_row + L + lane * 2);
            }
            dot = fmaf(qr0, bf2f_lo(kr_u), dot);
            dot = fmaf(qr1, bf2f_hi(kr_u), dot);
            const float sc = r4d_xor_sum<32>(dot) * scale;

            if (sc > mx) {
                const float alpha = (mx == -INFINITY) ? 0.0f : ex2((mx - sc) * kLog2E);
                mx = sc;
                den = den * alpha + 1.0f;
#pragma unroll
                for (int i = 0; i < VPL; ++i) acc[i] = fmaf(acc[i], alpha, kv_reg[i]);
            } else {
                const float w = ex2((sc - mx) * kLog2E);
                den += w;
#pragma unroll
                for (int i = 0; i < VPL; ++i) acc[i] = fmaf(w, kv_reg[i], acc[i]);
            }
        }
    }
}

/* Fold partition state (mw, dw, aw) into the running state (M, D, Acc) in partition order; an empty partition (dw == 0) changes nothing. */
template <int N>
__device__ __forceinline__ void mla_merge(float& M, float& D, float (&Acc)[N], float mw, float dw, const float (&aw)[N]) {
    if (!(dw > 0.0f)) return;
    const float nm = fmaxf(M, mw);
    const float alpha = (M == -INFINITY) ? 0.0f : ex2((M - nm) * kLog2E);
    const float b = ex2((mw - nm) * kLog2E);
    D = fmaf(dw, b, D * alpha);
#pragma unroll
    for (int i = 0; i < N; ++i) Acc[i] = fmaf(aw[i], b, Acc[i] * alpha);
    M = nm;
}

/* Cooperative decode kernel: NWAVES (8) waves per (h, m) block, wave w runs partition w; one barrier, every wave then folds the 8 states of its own
 * lanes' columns (the same fold order, so every wave holds the same bits) and writes its share of the output. */
template <int L, int NWAVES, bool FP8 = false>
__global__ void ernie_mla_sparse_attn_coop_kernel(MlaSparseArgs A) {
    static_assert(L == 256 || L == 512, "L must be 256 or 512");
    static_assert(NWAVES == 8, "the partition count is 8");
    constexpr int CHUNKS = L / 256; /* 1 for L=256 (8 vals/lane), 2 for L=512 (16 vals/lane) */
    constexpr int VPL = CHUNKS * 8;
    __shared__ float sh_mx[NWAVES];
    __shared__ float sh_den[NWAVES];
    __shared__ float sh_acc[NWAVES][L];

    const int h = (int)blockIdx.x, m = (int)blockIdx.y;
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31, wave = tid >> 5;
    if (h >= A.H || m >= A.M) return;

    const uint16_t* q_ptr = A.q + (long long)m * A.qp + (long long)(h * L);
    const uint16_t* qr_ptr = A.qr + (long long)m * A.qrp + (long long)(h * 64);

    float q_reg[VPL];
#pragma unroll
    for (int ch = 0; ch < CHUNKS; ++ch) load_bf8(q_ptr + ch * 256 + lane * 8, &q_reg[ch * 8]);
    const uint32_t qr_u = *(const uint32_t*)(qr_ptr + lane * 2);
    const float qr0 = bf2f_lo(qr_u), qr1 = bf2f_hi(qr_u);

    float mx, den, acc[VPL];
    mla_part<L, FP8>(A, q_reg, qr0, qr1, A.slots + (long long)m * A.slp, lane, wave, h, mx, den, acc);

    if (lane == 0) {
        sh_mx[wave] = mx;
        sh_den[wave] = den;
    }
#pragma unroll
    for (int ch = 0; ch < CHUNKS; ++ch) {
#pragma unroll
        for (int i = 0; i < 8; ++i) sh_acc[wave][ch * 256 + lane * 8 + i] = acc[ch * 8 + i];
    }
    __syncthreads();

    /* Every thread folds the 8 states in partition order: (M, D) once, then its columns c = tid, tid + 256 ... with the same per-column arithmetic as
     * mla_merge in the wave kernel (a = fmaf(acc_w, b_w, a * alpha_w)). */
    float M = -INFINITY, D = 0.0f;
    float sw_den[NWAVES];
#pragma unroll
    for (int w = 0; w < NWAVES; ++w) sw_den[w] = sh_den[w];
    uint16_t* out_ptr = A.out + (long long)m * A.outp + (long long)(h * L);
    constexpr int NTHR = NWAVES * 32;
    /* the (M, D) fold is column independent: run it once per thread, then the per-column accumulation */
    float bw[NWAVES], alw[NWAVES];
#pragma unroll
    for (int w = 0; w < NWAVES; ++w) {
        alw[w] = 0.0f; bw[w] = 0.0f;
        if (sw_den[w] > 0.0f) {
            const float nm = fmaxf(M, sh_mx[w]);
            alw[w] = (M == -INFINITY) ? 0.0f : ex2((M - nm) * kLog2E);
            bw[w] = ex2((sh_mx[w] - nm) * kLog2E);
            D = fmaf(sw_den[w], bw[w], D * alw[w]);
            M = nm;
        }
    }
    const float inv_den = (D > 0.0f) ? (1.0f / D) : 0.0f;
    for (int c = tid; c < L; c += NTHR) {
        float a = 0.0f;
#pragma unroll
        for (int w = 0; w < NWAVES; ++w) {
            if (sw_den[w] > 0.0f) a = fmaf(sh_acc[w][c], bw[w], a * alw[w]);
        }
        out_ptr[c] = f2bf(a * inv_den);
    }
}

/* Wave kernel (any row count): 4 waves per block handle 4 heads of the SAME token m (sharing KV rows in L1), the 8 partitions one after the other. */
template <int L, bool FP8 = false>
__global__ void ernie_mla_sparse_attn_wave_kernel(MlaSparseArgs A) {
    static_assert(L == 256 || L == 512, "L must be 256 or 512");
    constexpr int CHUNKS = L / 256;
    constexpr int VPL = CHUNKS * 8;

    const int lane = (int)(threadIdx.x & 31);
    const int wave = (int)(threadIdx.x >> 5);
    const int h = (int)blockIdx.x * 4 + wave;
    const int m = (int)blockIdx.y;
    if (h >= A.H || m >= A.M) return;

    const uint16_t* q_ptr = A.q + (long long)m * A.qp + (long long)(h * L);
    const uint16_t* qr_ptr = A.qr + (long long)m * A.qrp + (long long)(h * 64);

    float q_reg[VPL];
#pragma unroll
    for (int ch = 0; ch < CHUNKS; ++ch) load_bf8(q_ptr + ch * 256 + lane * 8, &q_reg[ch * 8]);
    const uint32_t qr_u = *(const uint32_t*)(qr_ptr + lane * 2);
    const float qr0 = bf2f_lo(qr_u), qr1 = bf2f_hi(qr_u);
    const int32_t* sl_row = A.slots + (long long)m * A.slp;

    float M = -INFINITY, D = 0.0f, Acc[VPL];
#pragma unroll
    for (int i = 0; i < VPL; ++i) Acc[i] = 0.0f;
    for (int part = 0; part < 8; ++part) {
        float mx, den, acc[VPL];
        mla_part<L, FP8>(A, q_reg, qr0, qr1, sl_row, lane, part, h, mx, den, acc);
        mla_merge<VPL>(M, D, Acc, mx, den, acc);
    }

    const float inv_den = (D > 0.0f) ? (1.0f / D) : 0.0f;
    uint16_t* out_ptr = A.out + (long long)m * A.outp + (long long)(h * L);
#pragma unroll
    for (int ch = 0; ch < CHUNKS; ++ch) {
        float res[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) res[i] = Acc[ch * 8 + i] * inv_den;
        store_bf8(out_ptr + ch * 256 + lane * 8, res);
    }
}

template <bool FP8 = false>
__global__ void ernie_mla_sparse_attn_kernel(MlaSparseArgs A) {
    /* General fallback: 128 threads per (m, h). Supports L <= 2048, R <= 512 using LDS for q/qr/acc. */
    __shared__ float smem[4];
    __shared__ float q_sh[2048];
    __shared__ float qr_sh[512];
    __shared__ float acc_sh[2048];
    const int h = (int)blockIdx.x, m = (int)blockIdx.y, tid = (int)threadIdx.x;
    if (h >= A.H || m >= A.M) return;

    const long long qb = (long long)m * A.qp + (long long)(h * A.L) * A.qs;
    for (int c = tid; c < A.L; c += 128) {
        q_sh[c] = bf2f(A.q[qb + (long long)c * A.qs]);
        acc_sh[c] = 0.0f;
    }
    const long long qrb = (long long)m * A.qrp + (long long)(h * A.R) * A.qrs;
    for (int c = tid; c < A.R; c += 128)
        qr_sh[c] = bf2f(A.qr[qrb + (long long)c * A.qrs]);
    __syncthreads();

    const float sk = A.sink[(long long)h * A.sks];
    float mx = sk;
    float den = (sk == -INFINITY) ? 0.0f : 1.0f;
    const int max_slot = A.pages * A.bs;

    for (int j = 0; j < A.W; ++j) {
        const int s = A.slots[(long long)m * A.slp + (long long)j * A.sls];
        if (s < 0 || s >= max_slot) continue;
        const long long kvb = (long long)(s / A.bs) * A.kvp0 + (long long)(s % A.bs) * A.kvp2;
        float dot_l = 0.0f;
        const uint8_t* row8 = (const uint8_t*)A.kv + kvb;
        for (int c = tid; c < A.L; c += 128)
            dot_l += q_sh[c] * (FP8 ? fp8_ds_mla_elem(row8, c, A.L, A.qb, A.fp8) : bf2f(A.kv[kvb + (long long)c * A.kvs]));
        for (int c = tid; c < A.R; c += 128)
            dot_l += qr_sh[c] * (FP8 ? fp8_ds_mla_elem(row8, A.L + c, A.L, A.qb, A.fp8) : bf2f(A.kv[kvb + (long long)(A.L + c) * A.kvs]));
        const float sc = block_sum(dot_l, smem, 128) * A.scale;
        const float nmx = fmaxf(mx, sc);
        const float alpha = (mx == -INFINITY) ? 0.0f : expf(mx - nmx);
        const float w = expf(sc - nmx);
        den = den * alpha + w;
        for (int c = tid; c < A.L; c += 128)
            acc_sh[c] = acc_sh[c] * alpha + w * (FP8 ? fp8_ds_mla_elem(row8, c, A.L, A.qb, A.fp8) : bf2f(A.kv[kvb + (long long)c * A.kvs]));
        mx = nmx;
        __syncthreads();
    }

    const long long outb = (long long)m * A.outp + (long long)(h * A.L) * A.outs;
    for (int c = tid; c < A.L; c += 128) {
        const float res = (den > 0.0f) ? (acc_sh[c] / den) : 0.0f;
        A.out[outb + (long long)c * A.outs] = f2bf(res);
    }
}

/* ernie_dsa_topk_tiled: words of the kept-list scratch (operand 11, flat i32) a launch of `rows` rows needs: 2 * topk + 2 a row, one verdict word a row. */
static inline long long dsa_kept_words(long long rows, long long topk) { return rows * (2 * topk + 3); }

#if !defined(__HIP__) && !defined(__HIP_PLATFORM_AMD__)
/* R4D_DSA_VALIDATE (default on, host builds only): 0 turns off the launcher's scan of positions, requests and the context bound, which is the situation of a
 * card build (those operands are device memory) so a test can drive the kernels' own verdict. Read at every call where a launch completes before it
 * returns (R4D_SYNC_LAUNCH). */
static inline bool dsa_host_scan_on() {
#ifdef R4D_SYNC_LAUNCH
    const char* e = std::getenv("R4D_DSA_VALIDATE");
    return !(e && *e == '0') && !r4d_verdict::card_form();
#else
    static const bool on = [] { const char* e = std::getenv("R4D_DSA_VALIDATE"); return !(e && *e == '0') && !r4d_verdict::card_form(); }();
    return on;
#endif
}
#endif

} /* namespace r4d_ernie_mla */

extern "C" {

int r4d_rad_ernie_idx_query(const RadArgs* a, RadStream stream) {
    using namespace r4d_ernie;
    using namespace r4d_ernie_mla;
    if (!a || !a->t || a->n_t < 4) return RAD_E_INVAL;
    const RadTensor &q = a->t[0], &pos = a->t[1], &qf = a->t[2], &qs = a->t[3];
    const long long H = r4d_geti_or(a, "heads", 0), D = r4d_geti_or(a, "head_dim", 0), R = r4d_geti_or(a, "rope_dim", 0);
    const double theta = r4d_getf(a, "theta", 0.0);
    bool bf16_table = true;
    IdxQuant qr{1e-10f, 448.0f};
    if (q.rank != 2 || q.shape[0] < 1 || q.shape[0] > 65535) return RAD_E_SHAPE;
    if (H < 1 || H > 256 || !idx_geometry(D, R) || !std::isfinite(theta) || theta <= 0.0 ||
        !idx_table(a, &bf16_table) || !idx_rule(a, &qr))
        return RAD_E_SHAPE;
    const long long M = q.shape[0];
    if (q.shape[1] != H * D || pos.rank != 1 || pos.shape[0] != M || qf.rank != 2 || qf.shape[0] != M ||
        qf.shape[1] != H * D || qs.rank != 2 || qs.shape[0] != M || qs.shape[1] != H || !pos_strides(q) ||
        !pos_strides(pos) || !pos_strides(qf) || !pos_strides(qs) || q.stride[0] < q.shape[1] * q.stride[1] ||
        qf.stride[0] < qf.shape[1] * qf.stride[1] || qs.stride[0] < qs.shape[1] * qs.stride[1])
        return RAD_E_SHAPE;
    if (q.dtype != RAD_BF16 || pos.dtype != RAD_I32 || qf.dtype != RAD_F8E4M3 || qs.dtype != RAD_F32) return RAD_E_DTYPE;
    if (!q.data || !pos.data || !qf.data || !qs.data) return RAD_E_INVAL;
    ErnieYarn plain{};
    RopeInvTable tab{};
    if (!fill_rope_inv(theta, R, plain, &tab)) return RAD_E_SHAPE;
#if !defined(__HIP__) && !defined(__HIP_PLATFORM_AMD__)
    if (!r4d_verdict::card_form())   /* the card form (R4D_ERNIE_CARD_FORM=1, the emulator) leaves the check to the kernel and the poll */
    for (long long m = 0; m < M; ++m)
        if (((const int32_t*)pos.data)[m * pos.stride[0]] < 0) return RAD_E_SHAPE;
#endif
    if (D == 128 && R == 64 && q.stride[1] == 1 && qf.stride[1] == 1) {
        R4D_ARGS(IdxQueryFast128Args, F);
        R4D_VC(F, r4d_verdict::kFamMla);
        F.q = (const uint16_t*)q.data; F.pos = (const int32_t*)pos.data;
        F.qf = (uint8_t*)qf.data; F.qs = (float*)qs.data;
        F.qp = q.stride[0]; F.ps = pos.stride[0]; F.qfp = qf.stride[0];
        F.qsp = qs.stride[0]; F.qss = qs.stride[1];
        F.M = (int)M; F.H = (int)H; F.bf16_table = bf16_table ? 1 : 0;
        F.qfloor = qr.floor; F.qmax = qr.qmax;
        for (int i = 0; i < 32; ++i) F.inv[i] = tab.inv[i];
        if (M >= 16) {
            /* Prefill: 1 block of 8 waves (256 threads) per token m shares cos/sin across all H heads. */
            F.heads_per_block = (int)H;
            R4D_KLAUNCH(R4D_KNAME(ernie_idx_query_fast128_kernel), dim3(1, (unsigned)M), dim3(256), (hipStream_t)stream, F);
        } else {
            /* Decode: 4 heads (4 waves = 128 threads) per block for high CU spread. */
            F.heads_per_block = 4;
            R4D_KLAUNCH(R4D_KNAME(ernie_idx_query_fast128_kernel), dim3((unsigned)((H + 3) / 4), (unsigned)M), dim3(128), (hipStream_t)stream, F);
        }
        return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
    }
    R4D_ARGS(IdxQueryArgs, P);
    R4D_VC(P, r4d_verdict::kFamMla);
    P.tab = tab;
    P.q = (const uint16_t*)q.data; P.pos = (const int32_t*)pos.data;
    P.qf = (uint8_t*)qf.data; P.qs = (float*)qs.data;
    P.qp = q.stride[0]; P.qs_in = q.stride[1]; P.ps = pos.stride[0];
    P.qfp = qf.stride[0]; P.qfs = qf.stride[1]; P.qsp = qs.stride[0]; P.qss = qs.stride[1];
    P.M = (int)M; P.H = (int)H; P.D = (int)D; P.R = (int)R;
    P.bf16_table = bf16_table ? 1 : 0; P.qfloor = qr.floor; P.qmax = qr.qmax;
    R4D_KLAUNCH(R4D_KNAME(ernie_idx_query_kernel), dim3((unsigned)H, (unsigned)M), dim3(256), (hipStream_t)stream, P);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

int r4d_rad_ernie_idx_key(const RadArgs* a, RadStream stream) {
    using namespace r4d_ernie;
    using namespace r4d_ernie_mla;
    if (!a || !a->t || a->n_t < 6) return RAD_E_INVAL;
    const RadTensor &k = a->t[0], &gm = a->t[1], &bt = a->t[2], &pos = a->t[3], &slot = a->t[4];
    const RadTensor cache = idx_cache_view(a->t[5]);
    const long long D = r4d_geti_or(a, "head_dim", 0), R = r4d_geti_or(a, "rope_dim", 0);
    const double theta = r4d_getf(a, "theta", 0.0), eps = r4d_getf(a, "eps", 1e-6);
    bool bf16_table = true;
    IdxQuant qr{1e-10f, 448.0f};
    if (k.rank != 2 || k.shape[0] < 1 || k.shape[0] > 65535) return RAD_E_SHAPE;
    if (!idx_geometry(D, R) || !std::isfinite(theta) || theta <= 0.0 || !(eps >= 0.0) || !std::isfinite(eps) ||
        !idx_table(a, &bf16_table) || !idx_rule(a, &qr))
        return RAD_E_SHAPE;
    const long long M = k.shape[0];
    if (k.shape[1] != D || gm.rank != 1 || gm.shape[0] != D || bt.rank != 1 || bt.shape[0] != D || pos.rank != 1 ||
        pos.shape[0] != M || slot.rank != 1 || slot.shape[0] != M || cache.rank != 3 || cache.shape[0] < 1 ||
        cache.shape[1] < 1 || cache.shape[2] != D + 4 || !pos_strides(k) || !pos_strides(gm) || !pos_strides(bt) ||
        !pos_strides(pos) || !pos_strides(slot) || cache.stride[2] != 1 || cache.stride[1] != D + 4 ||
        cache.stride[0] != cache.shape[1] * (D + 4))
        return RAD_E_SHAPE;
    if (k.dtype != RAD_BF16 || gm.dtype != RAD_BF16 || bt.dtype != RAD_BF16 || pos.dtype != RAD_I32 ||
        slot.dtype != RAD_I32 || cache.dtype != RAD_U8)
        return RAD_E_DTYPE;
    if (!k.data || !gm.data || !bt.data || !pos.data || !slot.data || !cache.data) return RAD_E_INVAL;
    const long long slots = cache.shape[0] * cache.shape[1];
    ErnieYarn plain{};
    RopeInvTable tab{};
    if (!fill_rope_inv(theta, R, plain, &tab)) return RAD_E_SHAPE;
#if !defined(__HIP__) && !defined(__HIP_PLATFORM_AMD__)
    if (!r4d_verdict::card_form())
    for (long long m = 0; m < M; ++m) {
        const long long sl = ((const int32_t*)slot.data)[m * slot.stride[0]];
        if (((const int32_t*)pos.data)[m * pos.stride[0]] < 0 || sl < -1 || sl >= slots) return RAD_E_SHAPE;
    }
#endif
    if (D == 128 && R == 64 && k.stride[1] == 1 && gm.stride[0] == 1 && bt.stride[0] == 1) {
        R4D_ARGS(IdxKeyFast128Args, F);
        R4D_VC(F, r4d_verdict::kFamMla);
        F.k = (const uint16_t*)k.data; F.gm = (const uint16_t*)gm.data; F.bt = (const uint16_t*)bt.data;
        F.pos = (const int32_t*)pos.data; F.slot = (const int32_t*)slot.data; F.cache = (uint8_t*)cache.data;
        F.kp = k.stride[0]; F.ps = pos.stride[0]; F.sls = slot.stride[0];
        F.M = (int)M; F.bf16_table = bf16_table ? 1 : 0; F.slots = (int)slots;
        F.eps = (float)eps; F.qfloor = qr.floor; F.qmax = qr.qmax;
        for (int i = 0; i < 32; ++i) F.inv[i] = tab.inv[i];
        if (M <= 64) {
            R4D_KLAUNCH(R4D_KNAME(ernie_idx_key_fast128_seq_kernel), dim3(1), dim3(32), (hipStream_t)stream, F);
        } else {
            R4D_KLAUNCH(R4D_KNAME(ernie_idx_key_fast128_par_kernel), dim3((unsigned)((M + 3) / 4)), dim3(128), (hipStream_t)stream, F);
        }
        return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
    }
    R4D_ARGS(IdxKeyArgs, P);
    R4D_VC(P, r4d_verdict::kFamMla);
    P.tab = tab;
    P.k = (const uint16_t*)k.data; P.gm = (const uint16_t*)gm.data; P.bt = (const uint16_t*)bt.data;
    P.pos = (const int32_t*)pos.data; P.slot = (const int32_t*)slot.data; P.cache = (uint8_t*)cache.data;
    P.kp = k.stride[0]; P.ks = k.stride[1]; P.gms = gm.stride[0]; P.bts = bt.stride[0];
    P.ps = pos.stride[0]; P.sls = slot.stride[0];
    P.M = (int)M; P.D = (int)D; P.R = (int)R; P.bf16_table = bf16_table ? 1 : 0; P.slots = (int)slots;
    P.eps = eps; P.qfloor = qr.floor; P.qmax = qr.qmax;
    if (P.M <= 64) {
        R4D_KLAUNCH(R4D_KNAME(ernie_idx_key_seq_kernel), dim3(1), dim3(256), (hipStream_t)stream, P);
    } else {
        R4D_KLAUNCH(R4D_KNAME(ernie_idx_key_par_kernel), dim3((unsigned)P.M), dim3(256), (hipStream_t)stream, P);
    }
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

int r4d_rad_ernie_idx_logits(const RadArgs* a, RadStream stream) {
    using namespace r4d_ernie;
    using namespace r4d_ernie_mla;
    if (!a || !a->t || a->n_t < 8) return RAD_E_INVAL;
    const RadTensor cache = idx_cache_view(a->t[3]);
    const RadTensor &qf = a->t[0], &qs = a->t[1], &w = a->t[2], &bt = a->t[4], &req = a->t[5],
                    &pos = a->t[6], &out = a->t[7];
    const long long H = r4d_geti_or(a, "heads", 0), D = r4d_geti_or(a, "head_dim", 0), window = r4d_geti_or(a, "window", 128);
    if (qf.rank != 2 || qf.shape[0] < 1 || qf.shape[0] > 65535) return RAD_E_SHAPE;
    if (H < 1 || H > 256 || D < 4 || D > 256 || !is_pow2(D) || window < 1 || window > INT32_MAX / 2) return RAD_E_SHAPE;
    const long long M = qf.shape[0];
    /* col_start (optional, default 0): the output's column j is logical id col_start + j; the tiled selection scores a context in tiles with it. */
    const long long col0 = r4d_geti_or(a, "col_start", 0);
    if (col0 < 0 || col0 > INT32_MAX / 2) return RAD_E_SHAPE;
    if (qf.shape[1] != H * D || qs.rank != 2 || qs.shape[0] != M || qs.shape[1] != H || w.rank != 2 ||
        w.shape[0] != M || w.shape[1] != H || cache.rank != 3 || cache.shape[0] < 1 || cache.shape[1] < 1 ||
        cache.shape[2] != D + 4 || bt.rank != 2 || bt.shape[0] < 1 || bt.shape[1] < 1 || req.rank != 1 ||
        req.shape[0] != M || pos.rank != 1 || pos.shape[0] != M || out.rank != 2 || out.shape[0] != M ||
        out.shape[1] < 1 || !pos_strides(qf) || !pos_strides(qs) || !pos_strides(w) || !pos_strides(bt) ||
        !pos_strides(req) || !pos_strides(pos) || !pos_strides(out) || cache.stride[2] != 1 ||
        cache.stride[1] != D + 4 || cache.stride[0] != cache.shape[1] * (D + 4) ||
        out.stride[0] < out.shape[1] * out.stride[1])
        return RAD_E_SHAPE;
    if ((find_p(a, "tokens") && r4d_geti_or(a, "tokens", 0) != out.shape[1]) ||
        (find_p(a, "seqs") && r4d_geti_or(a, "seqs", 0) != bt.shape[0]) ||
        (find_p(a, "blocks") && r4d_geti_or(a, "blocks", 0) != bt.shape[1]) ||
        (find_p(a, "block_size") && r4d_geti_or(a, "block_size", 0) != cache.shape[1]) ||
        (find_p(a, "num_pages") && r4d_geti_or(a, "num_pages", 0) != cache.shape[0]))
        return RAD_E_SHAPE;
    if (qf.dtype != RAD_F8E4M3 || qs.dtype != RAD_F32 || w.dtype != RAD_BF16 || cache.dtype != RAD_U8 ||
        bt.dtype != RAD_I32 || req.dtype != RAD_I32 || pos.dtype != RAD_I32 || out.dtype != RAD_F32)
        return RAD_E_DTYPE;
    if (!qf.data || !qs.data || !w.data || !cache.data || !bt.data || !req.data || !pos.data || !out.data)
        return RAD_E_INVAL;
    R4D_ARGS(IdxLogitsArgs, P);
    P.qf = (const uint8_t*)qf.data; P.qs = (const float*)qs.data; P.w = (const uint16_t*)w.data;
    P.cache = (const uint8_t*)cache.data; P.bt = (const int32_t*)bt.data;
    P.req = (const int32_t*)req.data; P.pos = (const int32_t*)pos.data; P.out = (float*)out.data;
    P.qfp = qf.stride[0]; P.qfs = qf.stride[1]; P.qsp = qs.stride[0]; P.qss = qs.stride[1];
    P.wp = w.stride[0]; P.ws = w.stride[1]; P.btp = bt.stride[0]; P.bts = bt.stride[1];
    P.rs = req.stride[0]; P.ps = pos.stride[0]; P.outp = out.stride[0]; P.outs = out.stride[1];
    P.M = (int)M; P.H = (int)H; P.D = (int)D; P.L = (int)out.shape[1];
    P.S = (int)bt.shape[0]; P.B = (int)bt.shape[1]; P.block = (int)cache.shape[1]; P.pages = (int)cache.shape[0];
    P.window = (int)window; P.wscale = (float)(std::pow((double)D, -0.5) * std::pow((double)H, -0.5));
    P.col0 = (int)col0;
    /* the tiled op (it passes col_start) always refuses a NaN score; the untiled pair keeps the base fmaxf (a NaN becomes 0) unless validation is on */
    P.keep_nan = (find_p(a, "col_start") || r4d_verdict::master_on()) ? 1 : 0;
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1201__)
    if (P.D == 128 && P.qfs == 1) {
        /* R4D_MLA_IDXL_SMALL_MAX (default 4): largest row count served by the 2-tile form; 0 pins the 4-tile form. */
        static const int idxl_small_max = r4d_ernie::r4d_pin_env("R4D_MLA_IDXL_SMALL_MAX", r4d_ernie::pin::kMlaIdxlSmallMax);
        if (r4d_ernie::pin::idxl_small(P.M, idxl_small_max) && P.L <= 2048) {
            R4D_KLAUNCH(R4D_KNAME(ernie_idx_logits_wmma128_kernel<2>), dim3((unsigned)((P.L + 31) / 32), (unsigned)P.M), dim3(128), (hipStream_t)stream, P);
        } else {
            R4D_KLAUNCH(R4D_KNAME(ernie_idx_logits_wmma128_kernel<4>), dim3((unsigned)((P.L + 63) / 64), (unsigned)P.M), dim3(128), (hipStream_t)stream, P);
        }
        return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
    }
#endif
    R4D_KLAUNCH(R4D_KNAME(ernie_idx_logits_kernel), dim3((unsigned)((P.L + 3) / 4), (unsigned)P.M), dim3(128), (hipStream_t)stream, P);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

/* Blocks a row of ernie_dsa_select is split over; 0 = the single-block kernel. R4D_DSA_SELECT_BLOCKS: unset or "auto" = 64 blocks when the score plane is at least 16384
 * wide, else one block; "0" = always the single-block kernel; "n" = n blocks (1..64), for tests and A/B runs. Both forms select the same ids and slots, bit for bit. The
 * multi-block form keeps its scratch in the rows of the outputs (slots row: 1024 + 2 nb words), so an output row too narrow for it takes the single-block kernel. */
static int dsel_blocks(long long L, long long slots_w) {
    using r4d_ernie_mla::kDselBins;
    using r4d_ernie_mla::kDselMaxBlocks;
    long long nb = L >= 16384 ? kDselMaxBlocks : 0;
    if (const char* e = std::getenv("R4D_DSA_SELECT_BLOCKS")) {
        char* z = nullptr;
        const long v = std::strtol(e, &z, 10);
        if (z != e && *z == 0 && v >= 0) nb = v > kDselMaxBlocks ? kDselMaxBlocks : v;
    }
    if (nb > 0 && slots_w < kDselBins + 2 * nb) nb = 0;
    return (int)nb;
}

/* Which select a launch took, for the log and for tests: the first launch of each (op, route) prints one line on stderr (an operator can see that the production shape
 * took the multi-block path, or why it did not); g_dsel_last is the last launch's choice (nb 0 = the single-block kernel / the merge). `why` is null when nb > 0. */
struct DselRoute { const char* op; int nb; const char* why; };
static DselRoute g_dsel_last = {nullptr, 0, nullptr};
extern "C" int r4d_ernie_dsa_select_last_nb(void) { return g_dsel_last.nb; }   /* tests: the block count the last launch of either op used (0 = the single-block kernel / the merge) */
static void dsel_note_route(const char* op, int nb, const char* why) {
    g_dsel_last = {op, nb, why};
    static std::atomic<unsigned> seen{0};
    const unsigned bit = (op[0] == 't' ? 1u : 2u) << (nb > 0 ? 0 : 2);
    if (seen.fetch_or(bit) & bit) return;
    if (nb > 0) std::fprintf(stderr, "r4d: ernie_dsa_%s: multi-block select, %d blocks a row\n", op[0] == 't' ? "topk_tiled (one tile)" : "select", nb);
    else std::fprintf(stderr, "r4d: ernie_dsa_%s: %s\n", op[0] == 't' ? "topk_tiled (one tile)" : "select", why ? why : "old kernel");
}

/* The multi-block select over operands already in `P` (scores, positions, requests, table, outputs, geometry): init, four radix passes, count, gather, finalize. */
static int run_dsel_multi(r4d_ernie_mla::DsaSelectArgs P, int nb, RadStream stream) {
    using namespace r4d_ernie_mla;
    P.nb = nb; P.pass = 0;
    const dim3 g1((unsigned)P.M), gn((unsigned)nb, (unsigned)P.M);
    R4D_KLAUNCH(R4D_KNAME(ernie_dsa_sel_init_kernel), g1, dim3(256), (hipStream_t)stream, P);
    for (int pass = 0; pass < 4; ++pass) {
        P.pass = pass;
        R4D_KLAUNCH(R4D_KNAME(ernie_dsa_sel_pass_kernel), gn, dim3(256), (hipStream_t)stream, P);
    }
    R4D_KLAUNCH(R4D_KNAME(ernie_dsa_sel_count_kernel), gn, dim3(256), (hipStream_t)stream, P);
    R4D_KLAUNCH(R4D_KNAME(ernie_dsa_sel_gather_kernel), gn, dim3(256), (hipStream_t)stream, P);
    R4D_KLAUNCH(R4D_KNAME(ernie_dsa_sel_final_kernel), g1, dim3(256), (hipStream_t)stream, P);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

int r4d_rad_ernie_dsa_select(const RadArgs* a, RadStream stream) {
    using namespace r4d_ernie;
    using namespace r4d_ernie_mla;
    if (!a || !a->t || a->n_t < 6) return RAD_E_INVAL;
    const RadTensor &sc = a->t[0], &pos = a->t[1], &req = a->t[2], &bt = a->t[3], &ids = a->t[4], &slots = a->t[5];
    const long long band = r4d_geti_or(a, "M", 0), topk = r4d_geti_or(a, "topk", 0),
                    window = r4d_geti_or(a, "window", 0), bs = r4d_geti_or(a, "block_size", 0);
    const long long pages = find_p(a, "num_pages") ? r4d_geti_or(a, "num_pages", 0) : (bs >= 1 ? INT32_MAX / bs : 0);
    if (band <= 0 || topk < 1 || topk > 2048 || window < 1 || bs < 1 || pages < 1 || window > INT32_MAX / 2 ||
        pages > INT32_MAX / bs)
        return RAD_E_SHAPE;
    if (!sc.data || !pos.data || !req.data || !bt.data || !ids.data || !slots.data) return RAD_E_SHAPE;
    if (sc.dtype != RAD_F32 || pos.dtype != RAD_I32 || req.dtype != RAD_I32 || bt.dtype != RAD_I32 ||
        ids.dtype != RAD_I32 || slots.dtype != RAD_I32)
        return RAD_E_SHAPE;
    for (const RadTensor* t : {&sc, &pos, &req, &bt, &ids, &slots})
        if (!pos_strides(*t)) return RAD_E_SHAPE;
    if (sc.rank != 2 || pos.rank != 1 || req.rank != 1 || bt.rank != 2 || ids.rank != 2 || slots.rank != 2)
        return RAD_E_SHAPE;
    const long long M = sc.shape[0], L = sc.shape[1], S = bt.shape[0], B = bt.shape[1], width = topk + window;
    if (M < 1 || M > band || M > 65535 || L < 1 || S < 1 || B < 1 || ids.shape[1] < width || slots.shape[1] != ids.shape[1] ||
        pos.shape[0] != M || req.shape[0] != M || ids.shape[0] != M || slots.shape[0] != M)
        return RAD_E_SHAPE;
    if ((find_p(a, "tokens") && r4d_geti_or(a, "tokens", 0) != L) ||
        (find_p(a, "seqs") && r4d_geti_or(a, "seqs", 0) != S) ||
        (find_p(a, "blocks") && r4d_geti_or(a, "blocks", 0) != B))
        return RAD_E_SHAPE;
    R4D_ARGS(DsaSelectArgs, P);
    R4D_VC(P, r4d_verdict::kFamDsa);
    P.sc = (const float*)sc.data; P.pos = (const int32_t*)pos.data; P.req = (const int32_t*)req.data;
    P.bt = (const int32_t*)bt.data; P.ids = (int32_t*)ids.data; P.slots = (int32_t*)slots.data;
    P.scp = sc.stride[0]; P.scs = sc.stride[1]; P.ps = pos.stride[0]; P.rs = req.stride[0];
    P.btp = bt.stride[0]; P.bts = bt.stride[1]; P.idsp = ids.stride[0]; P.idss = ids.stride[1];
    P.slp = slots.stride[0]; P.sls = slots.stride[1];
    P.M = (int)M; P.L = (int)L; P.S = (int)S; P.B = (int)B; P.topk = (int)topk; P.window = (int)window;
    P.bs = (int)bs; P.pages = (int)pages; P.out_w = (int)ids.shape[1];
    /* The multi-block form has no whole-row refusal of a NaN / +inf score (its scratch lives in the output rows, which a refused row must not touch), so with validation on
     * (the emulator default, R4D_ERNIE_VALIDATE=1 on a card) the single-block kernel runs and every refusal is the old kernel's. With validation off (a card's default) both
     * forms do what base d2258ef8 did: a NaN is no candidate, +inf the largest key. */
    int nb = dsel_blocks(L, INT32_MAX);
    if (nb > 0 && P.vc.log) { dsel_note_route("select", 0, "validation on: the single-block kernel keeps the whole-row NaN / +inf refusal"); nb = 0; }
    else if (nb > 0 && slots.shape[1] < r4d_ernie_mla::kDselBins + 2 * nb) { dsel_note_route("select", 0, "slots row narrower than 1024 + 2 nb words: the single-block kernel"); nb = 0; }
    else if (nb > 0) dsel_note_route("select", nb, nullptr);
    if (nb > 0) return run_dsel_multi(P, nb, stream);
    R4D_KLAUNCH(R4D_KNAME(ernie_dsa_select_kernel), dim3((unsigned)P.M), dim3(256), (hipStream_t)stream, P);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}


/* ernie_dsa_topk_tiled: operands as libref's (qf, qs, w, cache, block_table_idx, req, positions, block_table_sel, scores scratch, ids, slots, kept-list
 * scratch, optional seqused). The scratch is [M, tile] or flat [numel]; a launch with M rows scores numel / M columns a tile. The number of tiles comes
 * from the host-side bound on the context: RadArgs::kv_used_max_h when the engine proved one (the optional `seqused` operand of the KV group makes it do
 * so), else the `tokens` parameter, the smaller of the two when both exist; with neither the launch is refused. Tiles past a row's own range cost a launch
 * and no work.
 *
 * Refusals (libref's, see ref_ernie_dsa_topk_tiled): a NaN or +inf score in a row's candidate range, a bad position or request, a candidate range past the
 * bound. libref returns RAD_E_SHAPE and stores nothing. Here the outputs have ONE writer, the finalize kernel, which stores only if no row of the launch
 * latched a verdict in the kept-list scratch. Where the operands are host memory (host builds, the emulator) the launcher also scans positions, requests
 * and the bound before the first launch and returns the refusal; on a card it cannot read them without a synchronisation, so the launch returns RAD_OK, stores
 * nothing, and the verdict is latched for r4d_ernie_dsa_take_error(). Where a launch completes before it returns (R4D_SYNC_LAUNCH, the emulator) it returns
 * RAD_E_SHAPE too. */
int r4d_rad_ernie_dsa_topk_tiled(const RadArgs* a, RadStream stream) {
    using namespace r4d_ernie;
    using namespace r4d_ernie_mla;
    if (!a || !a->t || a->n_t < 12) return RAD_E_INVAL;
    const RadTensor cache = idx_cache_view(a->t[3]);
    const RadTensor &qf = a->t[0], &qs = a->t[1], &w = a->t[2], &ibt = a->t[4], &req = a->t[5], &pos = a->t[6], &lbt = a->t[7],
                    &plane = a->t[8], &ids = a->t[9], &slots = a->t[10], &kept = a->t[11];
    const long long H = r4d_geti_or(a, "heads", 0), D = r4d_geti_or(a, "head_dim", 0), window = r4d_geti_or(a, "window", 128),
                    topk = r4d_geti_or(a, "topk", 0), bs = r4d_geti_or(a, "block_size", 0);
    const long long pages_sel = find_p(a, "num_pages") ? r4d_geti_or(a, "num_pages", 0) : (bs >= 1 ? INT32_MAX / bs : 0);
    if (qf.rank != 2 || qf.shape[0] < 1 || qf.shape[0] > 65535) return RAD_E_SHAPE;
    if (H < 1 || H > 256 || D < 4 || D > 256 || !is_pow2(D) || window < 1 || window > INT32_MAX / 2 || topk < 1 || topk > 2048 || bs < 1 ||
        pages_sel < 1 || pages_sel > INT32_MAX / bs)
        return RAD_E_SHAPE;
    const long long M = qf.shape[0], W = topk + window;
    if (pos.rank != 1 || pos.shape[0] != M || req.rank != 1 || req.shape[0] != M || ibt.rank != 2 || lbt.rank != 2 || lbt.shape[0] < 1 ||
        lbt.shape[1] < 1 || cache.rank != 3 || cache.shape[1] != bs || (plane.rank != 2 && plane.rank != 1) ||
        (plane.rank == 2 && plane.shape[0] != M) || (plane.rank == 1 && plane.shape[0] < M) || ids.rank != 2 || slots.rank != 2 ||
        ids.shape[0] != M || slots.shape[0] != M || ids.shape[1] < W || slots.shape[1] != ids.shape[1] || kept.rank != 1 ||
        kept.shape[0] < dsa_kept_words(M, topk) || !pos_strides(ids) || !pos_strides(slots) || !pos_strides(lbt) || !pos_strides(plane) ||
        !pos_strides(pos) || !pos_strides(req) || !pos_strides(kept) || kept.stride[0] != 1)
        return RAD_E_SHAPE;
    const long long tile = plane.rank == 2 ? plane.shape[1] : plane.shape[0] / M;
    if (tile < 1 || tile > INT32_MAX / 2) return RAD_E_SHAPE;
    if ((find_p(a, "tile") && r4d_geti_or(a, "tile", 0) != tile) || (find_p(a, "seqs") && r4d_geti_or(a, "seqs", 0) != ibt.shape[0]) ||
        (find_p(a, "blocks") && r4d_geti_or(a, "blocks", 0) != ibt.shape[1]))
        return RAD_E_SHAPE;
    if (plane.dtype != RAD_F32 || ids.dtype != RAD_I32 || slots.dtype != RAD_I32 || lbt.dtype != RAD_I32 || pos.dtype != RAD_I32 ||
        req.dtype != RAD_I32 || kept.dtype != RAD_I32)
        return RAD_E_DTYPE;
    if (!plane.data || !ids.data || !slots.data || !lbt.data || !pos.data || !req.data || !kept.data) return RAD_E_INVAL;
    const long long prow = plane.rank == 2 ? plane.stride[0] : tile * plane.stride[0], pcol = plane.rank == 2 ? plane.stride[1] : plane.stride[0];
    const bool have_tokens = find_p(a, "tokens") != nullptr;
    const long long tokens = have_tokens ? r4d_geti_or(a, "tokens", 0) : 0;
    if (have_tokens && tokens < 1) return RAD_E_SHAPE;
    long long bound = -1;   /* host bound on the candidate range [0, end) */
    if (a->kv_used_max_h > 0) bound = std::max<long long>((long long)a->kv_used_max_h - window, 0);
    if (have_tokens) bound = bound < 0 ? tokens : std::min(bound, tokens);
    if (bound < 0) return RAD_E_INVAL;   /* no host-side bound on the context: refuse, never scan an unbounded range */
    if (bound > INT32_MAX) return RAD_E_SHAPE;
    const long long S = std::min<long long>(ibt.shape[0], lbt.shape[0]);   /* a request must index both tables */
#if !defined(__HIP__) && !defined(__HIP_PLATFORM_AMD__)
    /* host build: positions and requests are readable, so the refusal is returned before the first launch */
    if (dsa_host_scan_on()) {
        for (long long m = 0; m < M; ++m) {
            const long long p = ((const int32_t*)pos.data)[m * pos.stride[0]], r = ((const int32_t*)req.data)[m * req.stride[0]];
            if (p < 0 || r < 0 || r >= S) return RAD_E_SHAPE;
            if (std::max<long long>(p + 1 - window, 0) > bound) return RAD_E_SHAPE;
        }
    }
#endif
    /* Never score a wider tile than the context bound: the logits kernel writes -inf over its whole output width, so a decode step with few
     * rows (flat scratch: numel / M columns) would otherwise write tens of megabytes of -inf a layer. At least one tile runs: the first tile's merge
     * starts every row's verdict, which the finalize kernel reads. */
    const long long tl = std::min<long long>(tile, std::max<long long>(bound, 1));
    const long long ntiles = std::max<long long>((bound + tl - 1) / tl, 1);
    /* scores of one tile through the existing logits launch (col_start) */
    RadTensor pv{};
    pv.data = plane.data; pv.dtype = RAD_F32; pv.rank = 2; pv.shape[0] = M; pv.shape[1] = tl; pv.stride[0] = plane.rank == 2 ? prow : tl * pcol; pv.stride[1] = pcol;
    R4D_ARGS(DsaMergeArgs, P);
    R4D_VC(P, r4d_verdict::kFamDsa);
    P.plane = (const float*)plane.data; P.pos = (const int32_t*)pos.data; P.req = (const int32_t*)req.data; P.bt = (const int32_t*)lbt.data;
    P.ids = (int32_t*)ids.data; P.slots = (int32_t*)slots.data; P.kept = (int32_t*)kept.data;
    P.planep = pv.stride[0]; P.planes = pcol; P.ps = pos.stride[0]; P.rs = req.stride[0]; P.btp = lbt.stride[0]; P.bts = lbt.stride[1];
    P.idsp = ids.stride[0]; P.idss = ids.stride[1]; P.slp = slots.stride[0]; P.sls = slots.stride[1];
    P.M = (int)M; P.S = (int)S; P.B = (int)lbt.shape[1]; P.topk = (int)topk; P.window = (int)window; P.bs = (int)bs;
    P.pages = (int)pages_sel; P.tile = (int)tl; P.out_w = (int)ids.shape[1]; P.bound = (int)bound;
    /* One tile = the whole context (a decode step, the plane being wider than the context): the merge over an empty kept list is the select over the row, so the
     * multi-block select does it, with its scratch in `kept` (the finalize rule of this op: the only stores to ids / slots are the last kernel's, and none if any row
     * of the launch latched a verdict). Same ids and slots, bit for bit (R4D_DSA_SELECT_BLOCKS as for ernie_dsa_select). The scratch needs 1024 + 2 nb + topk words of
     * the row's 2 topk + 2; a smaller topk takes the merge. */
    int nb_one = 0;
    if (ntiles == 1) {
        nb_one = dsel_blocks(tl, INT32_MAX);
        if (nb_one > 0 && (long long)kDselBins + 2LL * nb_one + topk > 2 * topk + 2) { dsel_note_route("tiled", 0, "kept row narrower than 1024 + 2 nb + topk"); nb_one = 0; }
        else if (nb_one > 0) dsel_note_route("tiled", nb_one, nullptr);
    }
    for (long long t0 = 0, n = 0; n < ntiles; ++n, t0 += tl) {
        RadTensor lt[8] = {qf, qs, w, a->t[3], ibt, req, pos, pv};
        RadParam lp[5] = {RAD_INT("M", M), RAD_INT("heads", H), RAD_INT("head_dim", D), RAD_INT("window", window), RAD_INT("col_start", t0)};
        auto la = *a;   /* RadArgs copy (a host struct, not a kernel argument) */
        la.t = lt; la.n_t = 8; la.p = lp; la.n_p = 5;
        const int rc = r4d_rad_ernie_idx_logits(&la, stream);
        if (rc != RAD_OK) return rc;
        if (nb_one > 0) {
            R4D_ARGS(DsaSelectArgs, Q);
            Q.vc = P.vc;
            Q.sc = (const float*)plane.data; Q.pos = (const int32_t*)pos.data; Q.req = (const int32_t*)req.data; Q.bt = (const int32_t*)lbt.data;
            Q.ids = (int32_t*)ids.data; Q.slots = (int32_t*)slots.data;
            Q.scp = pv.stride[0]; Q.scs = pcol; Q.ps = pos.stride[0]; Q.rs = req.stride[0]; Q.btp = lbt.stride[0]; Q.bts = lbt.stride[1];
            Q.idsp = ids.stride[0]; Q.idss = ids.stride[1]; Q.slp = slots.stride[0]; Q.sls = slots.stride[1];
            Q.M = (int)M; Q.L = (int)tl; Q.S = (int)S; Q.B = (int)lbt.shape[1]; Q.topk = (int)topk; Q.window = (int)window; Q.bs = (int)bs;
            Q.pages = (int)pages_sel; Q.out_w = (int)ids.shape[1];
            Q.scr = (int32_t*)kept.data; Q.scrp = 2LL * topk + 2; Q.flags = (int32_t*)kept.data + (long long)M * Q.scrp; Q.tiled = 1; Q.bound = (int)bound;
            const int rc2 = run_dsel_multi(Q, nb_one, stream);
            if (rc2 != RAD_OK) return rc2;
            break;
        }
        P.col0 = (int)t0;
        R4D_KLAUNCH(R4D_KNAME(ernie_dsa_merge_kernel), dim3((unsigned)P.M), dim3(256), (hipStream_t)stream, P);
        if (hipGetLastError() != hipSuccess) return RAD_E_DEVICE;
    }
    if (nb_one == 0) {
        R4D_KLAUNCH(R4D_KNAME(ernie_dsa_finalize_kernel), dim3((unsigned)P.M), dim3(256), (hipStream_t)stream, P);
        if (hipGetLastError() != hipSuccess) return RAD_E_DEVICE;
    }
#ifdef R4D_SYNC_LAUNCH
    {   /* the launch has completed: its verdict is readable */
        int bad = 0;   /* the emulator: the operands are host memory; the verdict is the OR of the launch's per-row verdict words (the `kept` operand's tail), which every kernel of the op reduces */
        { const int32_t* fl = (const int32_t*)kept.data + (long long)M * (2LL * topk + 2); for (long long i = 0; i < M; ++i) bad |= fl[i] & 7; }
        /* card form (R4D_ERNIE_CARD_FORM=1): the launch returns OK as on a card and the refusal is the polled event */
        if (bad && !r4d_verdict::card_form()) return RAD_E_SHAPE;
    }
#endif
    return RAD_OK;
}

int r4d_rad_ernie_mla_project(const RadArgs* a, RadStream stream) {
    using namespace r4d_ernie;
    using namespace r4d_ernie_mla;
    if (!a || !a->t || a->n_t < 3) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &w = a->t[1], &y = a->t[2];
    const long long H = r4d_geti_or(a, "heads", 0), kd = r4d_geti_or(a, "k_dim", 0),
                    vd = r4d_geti_or(a, "v_dim", 0), L = r4d_geti_or(a, "latent_dim", 0);
    const char* mode = r4d_gets(a, "mode");
    const bool absorb = mode && !std::strcmp(mode, "absorb");
    const bool expand = mode && !std::strcmp(mode, "expand");
    const long long qw = r4d_geti_or(a, "q_width", kd);
    if ((!absorb && !expand) || H < 1 || H > 65535 || kd < 1 || vd < 1 || L < 1 ||
        (absorb && qw < kd) || (expand && qw != kd))
        return RAD_E_SHAPE;
    const long long xw = absorb ? qw : L, yw = absorb ? L : vd;
    if (x.rank != 2 || w.rank != 2 || y.rank != 2 || x.shape[0] < 1 || x.shape[0] > 65535 ||
        x.shape[1] != H * xw || w.shape[0] != H * (kd + vd) || w.shape[1] != L ||
        y.shape[0] != x.shape[0] || y.shape[1] != H * yw || !pos_strides(x) || !pos_strides(w) ||
        !pos_strides(y) || y.stride[0] < y.shape[1] * y.stride[1])
        return RAD_E_SHAPE;
    if (x.dtype != RAD_BF16 || w.dtype != RAD_BF16 || y.dtype != RAD_BF16) return RAD_E_DTYPE;
    if (!x.data || !w.data || !y.data) return RAD_E_INVAL;
    R4D_ARGS(MlaProjArgs, P);
    P.x = (const uint16_t*)x.data; P.w = (const uint16_t*)w.data; P.y = (uint16_t*)y.data;
    P.xp = x.stride[0]; P.xs = x.stride[1]; P.wp = w.stride[0]; P.ws = w.stride[1];
    P.yp = y.stride[0]; P.ys = y.stride[1];
    P.M = (int)x.shape[0]; P.H = (int)H; P.kd = (int)kd; P.vd = (int)vd; P.L = (int)L;
    P.xw = (int)xw; P.yw = (int)yw; P.absorb = absorb ? 1 : 0;

    if (P.xs == 1 && P.ws == 1 && P.ys == 1) {
        /* R4D_MLA_PROJ_ROWS (default 0 = the rule: 1/2/4/8/16 rows a block by M): 1 pins one row a block for every M. */
        static const int proj_rows = r4d_ernie::r4d_pin_env("R4D_MLA_PROJ_ROWS", r4d_ernie::pin::kMlaProjRows);
        const int R = r4d_ernie::pin::mla_proj_rows(P.M, proj_rows);
        if (absorb) {
            const unsigned bx = (unsigned)((P.L + 127) / 128);
            const unsigned bz = (unsigned)((P.M + R - 1) / R);
            if (R == 16)      R4D_KLAUNCH(R4D_KNAME(ernie_mla_project_absorb_rows<16>), dim3(bx, (unsigned)P.H, bz), dim3(128), (hipStream_t)stream, P);
            else if (R == 8)  R4D_KLAUNCH(R4D_KNAME(ernie_mla_project_absorb_rows<8>),  dim3(bx, (unsigned)P.H, bz), dim3(128), (hipStream_t)stream, P);
            else if (R == 4)  R4D_KLAUNCH(R4D_KNAME(ernie_mla_project_absorb_rows<4>),  dim3(bx, (unsigned)P.H, bz), dim3(128), (hipStream_t)stream, P);
            else if (R == 2)  R4D_KLAUNCH(R4D_KNAME(ernie_mla_project_absorb_rows<2>),  dim3(bx, (unsigned)P.H, bz), dim3(128), (hipStream_t)stream, P);
            else              R4D_KLAUNCH(R4D_KNAME(ernie_mla_project_absorb_rows<1>),  dim3(bx, (unsigned)P.H, bz), dim3(128), (hipStream_t)stream, P);
        } else {
            const unsigned bx = (unsigned)((P.vd + 3) / 4);
            const unsigned bz = (unsigned)((P.M + R - 1) / R);
            if (R == 16)      R4D_KLAUNCH(R4D_KNAME(ernie_mla_project_expand_rows<16>), dim3(bx, (unsigned)P.H, bz), dim3(128), (hipStream_t)stream, P);
            else if (R == 8)  R4D_KLAUNCH(R4D_KNAME(ernie_mla_project_expand_rows<8>),  dim3(bx, (unsigned)P.H, bz), dim3(128), (hipStream_t)stream, P);
            else if (R == 4)  R4D_KLAUNCH(R4D_KNAME(ernie_mla_project_expand_rows<4>),  dim3(bx, (unsigned)P.H, bz), dim3(128), (hipStream_t)stream, P);
            else if (R == 2)  R4D_KLAUNCH(R4D_KNAME(ernie_mla_project_expand_rows<2>),  dim3(bx, (unsigned)P.H, bz), dim3(128), (hipStream_t)stream, P);
            else              R4D_KLAUNCH(R4D_KNAME(ernie_mla_project_expand_rows<1>),  dim3(bx, (unsigned)P.H, bz), dim3(128), (hipStream_t)stream, P);
        }
        return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
    }
    R4D_KLAUNCH(R4D_KNAME(ernie_mla_project_kernel), dim3((unsigned)((yw + 3) / 4), (unsigned)H, (unsigned)P.M), dim3(128), (hipStream_t)stream, P);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

int r4d_rad_ernie_mla_sparse_attn(const RadArgs* a, RadStream stream) {
    using namespace r4d_ernie;
    using namespace r4d_ernie_mla;
    if (!a || !a->t || a->n_t < 6) return RAD_E_INVAL;
    const RadTensor &q = a->t[0], &qr = a->t[1], &kv = a->t[2], &slots = a->t[3], &sink = a->t[4], &out = a->t[5];
    const long long H = r4d_geti_or(a, "heads", 0), L = r4d_geti_or(a, "latent_dim", 0),
                    R = r4d_geti_or(a, "rope_dim", 0), bs = r4d_geti_or(a, "block_size", 0), W = r4d_geti_or(a, "width", 0);
    const long long pages = find_p(a, "num_pages") ? r4d_geti_or(a, "num_pages", 0) : (kv.rank == 4 ? kv.shape[0] : 0);
    const float scale = (float)r4d_getf(a, "scale", 0.0);
    /* `kv_layout` fp8_ds_mla (kv-fp8-latent): u8 rows of L + 4 * (L / qb) + 2 * R bytes; absent or bf16 = the bf16 row. */
    const char* layout = r4d_gets(a, "kv_layout");
    const int lfmt = !layout ? 0 : !std::strcmp(layout, "fp8_ds_mla") ? 1 : !std::strcmp(layout, "i8_ds_mla") ? 2 : !std::strcmp(layout, "i10_ds_mla") ? 3 : !std::strcmp(layout, "i12_ds_mla") ? 4 : !std::strcmp(layout, "bf16_ds_mla") ? 5 : 0;   /* v2: integer rows; 5 the diagnostic identity row */
    const bool fp8 = lfmt != 0;
    if (layout && !fp8 && std::strcmp(layout, "bf16")) return RAD_E_SHAPE;
    const long long qb = find_p(a, "quant_block") ? r4d_geti_or(a, "quant_block", 0) : 128;
    if (fp8) {
        if (kv.dtype != RAD_U8) return RAD_E_DTYPE;
        if (qb < 1 || L < 1 || L % qb) return RAD_E_SHAPE;
    }
    const long long klo = latent_lo_bytes((int)L, lfmt);   /* packed low-bit bytes of an i10 / i12 row */
    if (fp8 && lfmt >= 3 && lfmt <= 4 && (qb * (lfmt == 3 ? 2 : 4)) % 8) return RAD_E_SHAPE;   /* a tile owns whole bytes of low bits */
    const long long kvw = fp8 ? L + klo + 4 * (L / qb) + 2 * R : L + R;
    if (H < 1 || H > 65535 || L < 1 || L > 2048 || R < 1 || R > 512 || bs < 1 || pages < 1 || W < 1 ||
        pages > INT32_MAX / bs || !std::isfinite(scale) || scale <= 0.0f || q.rank != 2 || qr.rank != 2 ||
        kv.rank != 4 || slots.rank != 2 || sink.rank != 1 || out.rank != 2 || q.shape[0] < 1 || q.shape[0] > 65535 ||
        q.shape[1] != H * L || qr.shape[0] != q.shape[0] || qr.shape[1] != H * R || kv.shape[0] != pages ||
        kv.shape[1] != 1 || kv.shape[2] != bs || kv.shape[3] != kvw || (fp8 && kv.stride[3] != 1) || slots.shape[0] != q.shape[0] ||
        slots.shape[1] != W || sink.shape[0] != H || out.shape[0] != q.shape[0] || out.shape[1] != H * L ||
        !pos_strides(q) || !pos_strides(qr) || !pos_strides(kv) || !pos_strides(slots) || !pos_strides(sink) ||
        !pos_strides(out) || out.stride[0] < out.shape[1] * out.stride[1])
        return RAD_E_SHAPE;
    if (q.dtype != RAD_BF16 || qr.dtype != RAD_BF16 || kv.dtype != (fp8 ? RAD_U8 : RAD_BF16) || out.dtype != RAD_BF16 ||
        slots.dtype != RAD_I32 || sink.dtype != RAD_F32)
        return RAD_E_DTYPE;
    if (!q.data || !qr.data || !kv.data || !slots.data || !sink.data || !out.data) return RAD_E_INVAL;
    R4D_ARGS(MlaSparseArgs, P);
    P.q = (const uint16_t*)q.data; P.qr = (const uint16_t*)qr.data; P.kv = (const uint16_t*)kv.data;
    P.slots = (const int32_t*)slots.data; P.sink = (const float*)sink.data; P.out = (uint16_t*)out.data;
    P.qp = q.stride[0]; P.qs = q.stride[1]; P.qrp = qr.stride[0]; P.qrs = qr.stride[1];
    P.kvp0 = kv.stride[0]; P.kvp2 = kv.stride[2]; P.kvs = kv.stride[3];
    P.slp = slots.stride[0]; P.sls = slots.stride[1]; P.sks = sink.stride[0];
    P.outp = out.stride[0]; P.outs = out.stride[1];
    P.M = (int)q.shape[0]; P.H = (int)H; P.L = (int)L; P.R = (int)R; P.bs = (int)bs;
    P.pages = (int)pages; P.W = (int)W; P.scale = scale;
    P.fp8 = lfmt; P.qb = fp8 ? (int)qb : 0;

    /* fp8_ds_mla rows take the fast kernels when every load stays aligned: the 8-byte code loads need qb a multiple of 8 and rows 16 bytes apart */
    const bool fp8_ok = !fp8 || ((qb % 8) == 0 && (kvw % 16) == 0 && (P.kvp0 % 16) == 0 && (P.kvp2 % 16) == 0);
    const bool fast_aligned = fp8_ok && (P.R == 64) && (P.L == 512 || P.L == 256) &&
                              P.qs == 1 && P.qrs == 1 && P.kvs == 1 && P.sls == 1 && P.outs == 1 &&
                              (((uintptr_t)P.q & 15u) == 0u) && (((uintptr_t)P.qr & 3u) == 0u) &&
                              (((uintptr_t)P.kv & 15u) == 0u) && (((uintptr_t)P.out & 15u) == 0u) &&
                              ((P.qp & 7) == 0) && ((P.qrp & 1) == 0) &&
                              ((P.kvp0 & 7) == 0) && ((P.kvp2 & 7) == 0) && ((P.outp & 7) == 0);
    if (fast_aligned) {
        /* R4D_MLA_COOP_MAX (default 8): largest row count served by the 8-wave cooperative kernel; 0 pins the wave kernel. */
        static const int mla_coop_max = r4d_ernie::r4d_pin_env("R4D_MLA_COOP_MAX", r4d_ernie::pin::kMlaCoopMax);
        if (r4d_ernie::pin::mla_coop(P.M, P.W, mla_coop_max)) {
            /* Decode / small-M: 8 waves (256 threads) per head cooperating across W with 0 inner barriers. */
            if (fp8 && P.L == 512)
                R4D_KLAUNCH(R4D_KNAME(ernie_mla_sparse_attn_coop_kernel<512, 8, true>), dim3((unsigned)P.H, (unsigned)P.M), dim3(256), (hipStream_t)stream, P);
            else if (fp8)
                R4D_KLAUNCH(R4D_KNAME(ernie_mla_sparse_attn_coop_kernel<256, 8, true>), dim3((unsigned)P.H, (unsigned)P.M), dim3(256), (hipStream_t)stream, P);
            else if (P.L == 512)
                R4D_KLAUNCH(R4D_KNAME(ernie_mla_sparse_attn_coop_kernel<512, 8>), dim3((unsigned)P.H, (unsigned)P.M), dim3(256), (hipStream_t)stream, P);
            else
                R4D_KLAUNCH(R4D_KNAME(ernie_mla_sparse_attn_coop_kernel<256, 8>), dim3((unsigned)P.H, (unsigned)P.M), dim3(256), (hipStream_t)stream, P);
        } else {
            /* Prefill / short-W: 4 waves per block (4 heads of same token sharing KV in L1), 0 LDS, 0 barriers. */
            if (fp8 && P.L == 512)
                R4D_KLAUNCH(R4D_KNAME(ernie_mla_sparse_attn_wave_kernel<512, true>), dim3((unsigned)((P.H + 3) / 4), (unsigned)P.M), dim3(128), (hipStream_t)stream, P);
            else if (fp8)
                R4D_KLAUNCH(R4D_KNAME(ernie_mla_sparse_attn_wave_kernel<256, true>), dim3((unsigned)((P.H + 3) / 4), (unsigned)P.M), dim3(128), (hipStream_t)stream, P);
            else if (P.L == 512)
                R4D_KLAUNCH(R4D_KNAME(ernie_mla_sparse_attn_wave_kernel<512>), dim3((unsigned)((P.H + 3) / 4), (unsigned)P.M), dim3(128), (hipStream_t)stream, P);
            else
                R4D_KLAUNCH(R4D_KNAME(ernie_mla_sparse_attn_wave_kernel<256>), dim3((unsigned)((P.H + 3) / 4), (unsigned)P.M), dim3(128), (hipStream_t)stream, P);
        }
        return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
    }
    if (fp8)
        R4D_KLAUNCH(R4D_KNAME(ernie_mla_sparse_attn_kernel<true>), dim3((unsigned)H, (unsigned)P.M), dim3(128), (hipStream_t)stream, P);
    else
        R4D_KLAUNCH(R4D_KNAME(ernie_mla_sparse_attn_kernel<false>), dim3((unsigned)H, (unsigned)P.M), dim3(128), (hipStream_t)stream, P);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

/* ernie_mla_latent_store (libref/ref_ernie_attn.cpp): one block a row. A tile's absmax -> the power-of-two scale (exponent arithmetic on the bit pattern, no
 * division, as libref), E4M3 codes of x * 2^-k, the rope slice copied as bytes. At scale 2^120 a code is clamped to 240 (0x77): a larger one (256 and up) times 2^120 is not
 * finite in f32 or bf16, and libref clamps the same way.
 * libref refuses what has no defined row: a slot outside [-1, pages * block_size) and a non-finite latent value (NaN or inf has no scale), before any store. A card cannot refuse
 * before it stores (the slot map and the values are device data), so the kernel skips such a row entirely (no byte of it is stored, no out-of-range write) and raises event bit 1 of
 * the MLA family log. The skipped slot keeps its previous bytes, which later attention would read, so this log is ALWAYS ON (R4D_VC_ALWAYS: kept and polled by the engine
 * before the step's commit with validation off too; no extra launch); the host launcher refuses it where its operands are host memory
 * (host builds and the emulator) unless R4D_MLA_VALIDATE=0 (a test of the card form). Rows before a bad one in the same launch are stored: detection after the fact.
 * PRECONDITION, NOT CHECKED: the slots of one launch are distinct (except -1). One block a row, no ordering between blocks: two rows aimed at one slot race (codes, scales and
 * rope may mix), where libref, sequential, keeps the last row. The engine never issues a repeated slot (core/sched/batch.cpp: KVGeom::slots maps a sequence's consecutive
 * positions through its own block table; core/mem/kv.cpp: blocks come from grab_block, prefix-cache adopt takes whole blocks only and writing starts past them). */
struct LatentStoreArgs {
    const uint16_t* src;
    const int32_t* slots;
    uint8_t* cache;
    long long sp, ss, cp0, cp2;   /* src row pitch (elements), slot stride, cache page / row strides (bytes) */
    int M, L, R, qb, bs, pages;
    int flip;   /* DIAGNOSTIC (bf16_ds_mla only, RADIANCE_DIAG_ID_FLIP): 1 = flip the low bit of each tile's first value, 2 = only tile 0 of slots 7 mod 61 */
    r4d_verdict::Vc vc;   /* this launch's event log (device memory the library owns; constant across replays) */
    int i8;   /* integer row (v2): 0 = fp8_ds_mla, 2 i8_ds_mla, 3 i10_ds_mla, 4 i12_ds_mla (the MlaSparseArgs fmt numbering); exact f32 scale absmax / (2^(7 + lb) - 1) */
};
/* correctly rounded f32 division on the card (libref divides with IEEE '/'; the emulator compiles this as host code) */
__device__ __forceinline__ float r4d_div_rn(float a, float b) {
#if defined(__HIP_DEVICE_COMPILE__)
    return __fdiv_rn(a, b);
#else
    return a / b;
#endif
}
namespace r4d_ernie_mla {
constexpr int kLatentTopScaleExp = 120;
constexpr unsigned kLatentTopCode = 0x77u;
__global__ void ernie_mla_latent_store_kernel(LatentStoreArgs A) {
    __shared__ float smem[8];
    __shared__ int ismem[8];
    const int m = (int)blockIdx.x, tid = (int)threadIdx.x;
    if (m >= A.M) return;
    const int s = A.slots[(long long)m * A.ss];
    if (s < -1 || s >= A.pages * A.bs) { if (tid == 0) r4d_verdict::raise(A.vc.log, 1u); return; }   /* libref refuses; the card stores nothing for the row and raises event bit 1 (g_mla_log) */
    if (s < 0) return;
    uint8_t* row = A.cache + (long long)(s / A.bs) * A.cp0 + (long long)(s % A.bs) * A.cp2;
    const uint16_t* x = A.src + (long long)m * A.sp;
    const int nt = A.L / A.qb;
    {   /* a non-finite latent value has no scale: libref refuses the launch before any store; here the row is skipped whole, before its first byte, and latched */
        int bad = 0;
        for (int c = tid; c < A.L; c += 128) bad |= !(fabsf(__uint_as_float((uint32_t)x[c] << 16)) <= 3.4028234663852886e38f);
        if (block_sum_i(bad, ismem, 128)) { if (tid == 0) r4d_verdict::raise(A.vc.log, 1u); return; }
    }
    for (int t = 0; t < nt; ++t) {
        float amax = 0.0f;
        for (int c = t * A.qb + tid; c < (t + 1) * A.qb; c += 128) amax = fmaxf(amax, fabsf(__uint_as_float((uint32_t)x[c] << 16)));
        amax = block_max(amax, smem, 128);
        if (!(amax >= 1e-30f)) amax = 1e-30f;
        if (A.i8 == 5) {   /* bf16_ds_mla (diagnostic identity row): the bf16 bits as a high and a low byte plane, scale 1.0 */
            if (tid == 0) {
                uint8_t* sp = row + A.L + A.L + 4 * t;
                sp[0] = 0; sp[1] = 0; sp[2] = 0x80; sp[3] = 0x3f;
            }
            for (int c = t * A.qb + tid; c < (t + 1) * A.qb; c += 128) {
                row[c] = (uint8_t)(x[c] >> 8);
                row[A.L + c] = (uint8_t)(x[c] & 0xff);
            }
            if (tid == 0 && (A.flip == 1 || (A.flip == 2 && t == 0 && s % 61 == 7))) row[A.L + t * A.qb] ^= 1u;   /* same thread wrote that byte above (c = t * qb) */
            continue;
        }
        if (A.i8) {   /* integer rows (libref latent_scale_i8 / latent_code_int): s = amax / (2^(7 + lb) - 1), code = rint(x / s) clamped to +-(2^(7 + lb) - 1), stored as an int8 high byte (code >> lb) and lb packed low bits */
            const int lb = A.i8 == 3 ? 2 : A.i8 == 4 ? 4 : 0;
            const float cmax = (float)((1 << (7 + lb)) - 1);
            const float s8 = r4d_div_rn(amax, cmax);
            const uint32_t sb8 = __float_as_uint(s8);
            if (tid == 0) {
                uint8_t* sp = row + A.L + latent_lo_bytes(A.L, A.i8) + 4 * t;
                sp[0] = (uint8_t)(sb8 & 0xff); sp[1] = (uint8_t)((sb8 >> 8) & 0xff); sp[2] = (uint8_t)((sb8 >> 16) & 0xff); sp[3] = (uint8_t)(sb8 >> 24);
            }
            for (int c = t * A.qb + tid; c < (t + 1) * A.qb; c += 128) {
                float q = rintf(r4d_div_rn(__uint_as_float((uint32_t)x[c] << 16), s8));
                q = fminf(cmax, fmaxf(-cmax, q));
                row[c] = (uint8_t)(int8_t)((int)q >> lb);
            }
            if (lb) {   /* the packed low bits, one thread per output byte (no shared byte is written by two threads; qb * lb % 8 == 0, so a tile owns whole bytes) */
                const int per = 8 / lb;
                for (int jb = tid; jb < A.qb / per; jb += 128) {
                    const int c0 = t * A.qb + jb * per;
                    uint32_t byte = 0;
                    for (int i = 0; i < per; ++i) {
                        float q = rintf(r4d_div_rn(__uint_as_float((uint32_t)x[c0 + i] << 16), s8));
                        q = fminf(cmax, fmaxf(-cmax, q));
                        byte |= ((uint32_t)(int)q & ((1u << lb) - 1u)) << (lb * i);
                    }
                    row[A.L + c0 / per] = (uint8_t)byte;
                }
            }
            continue;
        }
        const uint32_t ab = __float_as_uint(amax);
        const int e = (int)(ab >> 23) - 127;
        const int k = (ab & 0x7fffffu) <= 0x600000u ? e - 8 : e - 7;   /* amax <= 1.75 * 2^e */
        const uint32_t sb = (uint32_t)(k + 127) << 23;
        const float inv = __uint_as_float((uint32_t)(127 - k) << 23);   /* 2^-k, exact */
        if (tid == 0) {
            uint8_t* sp = row + A.L + 4 * t;
            sp[0] = (uint8_t)(sb & 0xff); sp[1] = (uint8_t)((sb >> 8) & 0xff); sp[2] = (uint8_t)((sb >> 16) & 0xff); sp[3] = (uint8_t)(sb >> 24);
        }
        for (int c = t * A.qb + tid; c < (t + 1) * A.qb; c += 128) {
            const float v = __uint_as_float((uint32_t)x[c] << 16);
            uint8_t code = f32_to_fp8e4m3_dev(v * inv);
            if (k >= kLatentTopScaleExp && (code & 0x7fu) > kLatentTopCode) code = (uint8_t)((code & 0x80u) | kLatentTopCode);   /* never a decoded value past the finite range */
            row[c] = code;
        }
    }
    uint8_t* rp = row + A.L + latent_lo_bytes(A.L, A.i8) + 4 * nt;
    for (int c = tid; c < A.R; c += 128) {
        const uint16_t bits = x[A.L + c];
        rp[2 * c] = (uint8_t)(bits & 0xff);
        rp[2 * c + 1] = (uint8_t)(bits >> 8);
    }
}
} /* namespace r4d_ernie_mla */

#if !defined(__HIP__) && !defined(__HIP_PLATFORM_AMD__)
/* R4D_MLA_VALIDATE (default on): 0 skips the latent store launcher's host scan. Read at every call where a launch completes before it returns (R4D_SYNC_LAUNCH). */
static inline bool r4d_mla_scan_on() {
#ifdef R4D_SYNC_LAUNCH
    const char* e = std::getenv("R4D_MLA_VALIDATE");
    return !(e && *e == '0') && !r4d_verdict::card_form();
#else
    static const bool on = [] { const char* e = std::getenv("R4D_MLA_VALIDATE"); return !(e && *e == '0') && !r4d_verdict::card_form(); }();
    return on;
#endif
}
#endif

int r4d_rad_ernie_mla_latent_store(const RadArgs* a, RadStream stream) {
    using namespace r4d_ernie;
    using namespace r4d_ernie_mla;
    if (!a || !a->t || a->n_t < 3) return RAD_E_INVAL;
    const RadTensor &src = a->t[0], &sm = a->t[1], &cache = a->t[2];
    const long long L = r4d_geti_or(a, "latent_dim", 0), R = r4d_geti_or(a, "rope_dim", 0), bs = r4d_geti_or(a, "block_size", 0),
                    qb = find_p(a, "quant_block") ? r4d_geti_or(a, "quant_block", 0) : 128;
    const char* slay = r4d_gets(a, "kv_layout");   /* v2: fp8_ds_mla (default), i8_ds_mla, i10_ds_mla or i12_ds_mla */
    const int sfmt = !slay || !std::strcmp(slay, "fp8_ds_mla") ? 0 : !std::strcmp(slay, "i8_ds_mla") ? 2 : !std::strcmp(slay, "i10_ds_mla") ? 3 : !std::strcmp(slay, "i12_ds_mla") ? 4 : !std::strcmp(slay, "bf16_ds_mla") ? 5 : -1;
    if (sfmt < 0) return RAD_E_SHAPE;
    const int slb = sfmt == 3 ? 2 : sfmt == 4 ? 4 : sfmt == 5 ? 8 : 0;
    const long long slo = L > 0 && L <= 2048 ? latent_lo_bytes((int)L, sfmt) : 0;
    if (slb && (qb * slb) % 8) return RAD_E_SHAPE;
    if (L < 1 || R < 1 || R > (1 << 20) || bs < 1 || qb < 1 || L % qb || L > 2048 || src.rank != 2 || sm.rank != 1 || cache.rank != 4 ||
        src.shape[0] < 1 || src.shape[0] > 65535 || src.shape[1] != L + R || sm.shape[0] != src.shape[0] || cache.shape[0] < 1 ||
        cache.shape[1] != 1 || cache.shape[2] != bs || cache.shape[3] != L + slo + 4 * (L / qb) + 2 * R || !pos_strides(src) || !pos_strides(sm) ||
        !pos_strides(cache) || cache.stride[3] != 1 || src.stride[1] != 1 || cache.shape[0] > INT32_MAX / bs)
        return RAD_E_SHAPE;
    if (src.dtype != RAD_BF16 || cache.dtype != RAD_U8 || sm.dtype != RAD_I32) return RAD_E_DTYPE;
    if (!src.data || !sm.data || !cache.data) return RAD_E_INVAL;
#if !defined(__HIP__) && !defined(__HIP_PLATFORM_AMD__)
    /* where the operands are host memory (host builds, the emulator) refuse as libref does, before any store; a card build compiles this out and the kernel latches instead */
    if (r4d_mla_scan_on()) {
        const long long slots_n = cache.shape[0] * bs;
        for (long long m = 0; m < src.shape[0]; ++m) {
            const long long s = ((const int32_t*)sm.data)[m * sm.stride[0]];
            if (s < -1 || s >= slots_n) return RAD_E_SHAPE;
            if (s < 0) continue;
            const uint16_t* x = (const uint16_t*)src.data + m * src.stride[0];
            for (long long c = 0; c < L; ++c)
                if ((x[c] & 0x7f80u) == 0x7f80u) return RAD_E_SHAPE;   /* bf16 NaN and infinity: an all-ones exponent */
        }
    }
#endif
    R4D_ARGS(LatentStoreArgs, P);
    R4D_VC_ALWAYS(P, r4d_verdict::kFamMla);   /* always on: a refused row leaves a stale slot (see the kernel's comment) */
    P.src = (const uint16_t*)src.data; P.slots = (const int32_t*)sm.data; P.cache = (uint8_t*)cache.data;
    P.sp = src.stride[0]; P.ss = sm.stride[0]; P.cp0 = cache.stride[0]; P.cp2 = cache.stride[2];
    P.M = (int)src.shape[0]; P.L = (int)L; P.R = (int)R; P.qb = (int)qb; P.bs = (int)bs; P.pages = (int)cache.shape[0];
    P.i8 = sfmt;
    if (sfmt == 5) {   /* DIAGNOSTIC: the identity row with a 1-ulp perturbation (libref reads the same variable) */
        const char* fe = std::getenv("RADIANCE_DIAG_ID_FLIP");
        P.flip = fe && *fe == '1' ? 1 : fe && *fe == '2' ? 2 : 0;
    }
    R4D_KLAUNCH(R4D_KNAME(ernie_mla_latent_store_kernel), dim3((unsigned)P.M), dim3(128), (hipStream_t)stream, P);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

} /* extern "C" */

/* The events the idx_query / idx_key checks raised since the last call; see g_mla_log. 0 if none or the validators are off (R4D_ERNIE_VALIDATE=0); -1 if the copy failed. */
extern "C" int r4d_ernie_mla_take_error(void) {
    return r4d_verdict::registry().take(r4d_verdict::kFamMla);
}

/* The events ernie_dsa_topk_tiled's finalize kernel raised since the last call (bits: 1 non-finite score, 2 bad position or request, 4 candidate range past the
 * host bound; see g_dsa_log). Same mechanism as every other ERNIE validator (r4d_ernie_verdict.h): the log is never cleared, the host remembers what it took, and
 * rad_kernel_poll_error (r4d_ernie_ops.hip) folds this log into the engine's poll. 0 if none or the validators are off (R4D_ERNIE_VALIDATE=0); -1 if the copy
 * failed. The launch that was refused stored nothing. */
extern "C" int r4d_ernie_dsa_take_error(void) {
    return r4d_verdict::registry().take(r4d_verdict::kFamDsa);
}
