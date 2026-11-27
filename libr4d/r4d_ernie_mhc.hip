/* r4d_ernie_mhc.hip -- ERNIE 5.5 Flash mHC device ops (story 4-3d):
 *   ernie_mhc_pre, ernie_mhc_pre_rms, ernie_mhc_head.
 * Written cleanly from ERNIE 5.5 Flash equations and libref/ref_mhc.cpp.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "r4d_ernie_common.h"
#include "r4d_plugin.h"

namespace r4d_ernie_mhc {

using namespace r4d_ernie;

struct MhcPreArgs {
    const void *h;
    const float *fn, *scale, *base;
    const uint16_t *gain;
    void *x;
    float *post, *comb;
    long long hp, hs, fnp, fns, ss, bs, gs, xp, xs, postp, posts, combp, combs;
    int rows, n, hc, iters, h_f32, fused, chunks, decode;
    float eps, pre_eps, sinkhorn_eps, post_mult, rms_eps;
};
static_assert(sizeof(MhcPreArgs) <= 512, "MhcPreArgs exceeds AQL kernarg limit");

struct MhcPartialArgs {
    const uint16_t *h;
    const float *fn;
    uint16_t *x;
    long long hp, fnp, xp;
    int rows, width, mix, chunks, decode_cols;
};
static_assert(sizeof(MhcPartialArgs) <= 512, "MhcPartialArgs exceeds AQL kernarg limit");

static constexpr int kChunkCols = 2048;
static constexpr int kDecodeChunkCols = 1024;

__device__ __forceinline__ float* mhc_partials(uint16_t* x, long long row, long long xp) {
    return reinterpret_cast<float*>((reinterpret_cast<uintptr_t>(x + row * xp) + 3u) & ~uintptr_t(3u));
}

__device__ __forceinline__ float load_h(const MhcPreArgs& A, long long idx) {
    return A.h_f32 ? ((const float*)A.h)[idx] : bf2f(((const uint16_t*)A.h)[idx]);
}

__device__ __forceinline__ void store_x(const MhcPreArgs& A, long long idx, float v) {
    if (A.h_f32) ((float*)A.x)[idx] = v;
    else ((uint16_t*)A.x)[idx] = f2bf(v);
}

/* ------------------------------------------------------------------ 2-stage fast path (bf16 contiguous) */

/* Stage 1 (decode / rows < 4): reuse each h load across four coefficients.
 * At hc=4, n=4096 the vector path has 16 chunks x 7 coefficient groups. */
template <bool Vec8>
__global__ __launch_bounds__(128) void ernie_mhc_partial_decode4_kernel(MhcPartialArgs A) {
    __shared__ float smem[4][4];
    const int chunk = (int)blockIdx.x, j0 = (int)blockIdx.y * 4, row = (int)blockIdx.z;
    const int tid = (int)threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int col0 = chunk * A.decode_cols;
    const int cols = A.width - col0 < A.decode_cols ? A.width - col0 : A.decode_cols;
    const uint16_t* hr = A.h + (long long)row * A.hp + col0;
    float sum[4] = {};
    if constexpr (Vec8) {
        for (int i = tid * 8; i < cols; i += 128 * 8) {
            const uint4 hq = *reinterpret_cast<const uint4*>(hr + i);
            const float h0 = bf2f((uint16_t)(hq.x & 0xffffu));
            const float h1 = bf2f((uint16_t)(hq.x >> 16));
            const float h2 = bf2f((uint16_t)(hq.y & 0xffffu));
            const float h3 = bf2f((uint16_t)(hq.y >> 16));
            const float h4 = bf2f((uint16_t)(hq.z & 0xffffu));
            const float h5 = bf2f((uint16_t)(hq.z >> 16));
            const float h6 = bf2f((uint16_t)(hq.w & 0xffffu));
            const float h7 = bf2f((uint16_t)(hq.w >> 16));
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int j = j0 + q;
                if (j < A.mix) {
                    const float* fnr = A.fn + (long long)j * A.fnp + col0 + i;
                    const float4 w0 = *reinterpret_cast<const float4*>(fnr);
                    const float4 w1 = *reinterpret_cast<const float4*>(fnr + 4);
                    sum[q] += h0 * w0.x + h1 * w0.y + h2 * w0.z + h3 * w0.w +
                              h4 * w1.x + h5 * w1.y + h6 * w1.z + h7 * w1.w;
                } else if (j == A.mix) {
                    sum[q] += h0 * h0 + h1 * h1 + h2 * h2 + h3 * h3 +
                              h4 * h4 + h5 * h5 + h6 * h6 + h7 * h7;
                }
            }
        }
    } else {
        for (int i = tid; i < cols; i += 128) {
            const float v = bf2f(hr[i]);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int j = j0 + q;
                if (j <= A.mix)
                    sum[q] += v * ((j == A.mix) ? v : A.fn[(long long)j * A.fnp + col0 + i]);
            }
        }
    }
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        if (j0 + q <= A.mix) {
            const float s = wave_sum(sum[q]);
            if (lane == 0) smem[wave][q] = s;
        }
    }
    __syncthreads();
    if (tid < 4 && j0 + tid <= A.mix) {
        /* Chunk-major so adjacent lanes read adjacent coefficients in the finish kernel. */
        mhc_partials(A.x, row, A.xp)[chunk * (A.mix + 1) + j0 + tid] =
            smem[0][tid] + smem[1][tid] + smem[2][tid] + smem[3][tid];
    }
}

/* Stage 1 (prefill / rows >= 4): 4 token rows x 4 coefficients per block,
 * reading fn once per 4 rows and h once per 4 coefficients. */
template <bool Vec4>
__global__ __launch_bounds__(128) void ernie_mhc_partial_rows4_coeff4_kernel(MhcPartialArgs A) {
    __shared__ float sums[4][4][4];
    const int chunk = (int)blockIdx.x;
    const int first_j = (int)blockIdx.y * 4;
    const int first_r = (int)blockIdx.z * 4;
    if (chunk >= A.chunks || first_j > A.mix || first_r >= A.rows) return;
    const int tid = (int)threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int col0 = chunk * kChunkCols;
    const int col1 = (col0 + kChunkCols < A.width) ? (col0 + kChunkCols) : A.width;
    float sum[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
    if (Vec4) {
        for (int i = col0 + tid * 4; i < col1; i += 128 * 4) {
            float4 w[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                w[q] = (first_j + q < A.mix)
                           ? *reinterpret_cast<const float4*>(A.fn + (long long)(first_j + q) * A.fnp + i)
                           : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                if (first_r + r < A.rows) {
                    const uint2 hv = *reinterpret_cast<const uint2*>(A.h + (long long)(first_r + r) * A.hp + i);
                    const float v0 = bf2f((uint16_t)(hv.x & 0xffffu));
                    const float v1 = bf2f((uint16_t)(hv.x >> 16));
                    const float v2 = bf2f((uint16_t)(hv.y & 0xffffu));
                    const float v3 = bf2f((uint16_t)(hv.y >> 16));
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        if (first_j + q < A.mix) {
                            sum[q][r] += v0 * w[q].x + v1 * w[q].y + v2 * w[q].z + v3 * w[q].w;
                        } else if (first_j + q == A.mix) {
                            sum[q][r] += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
                        }
                    }
                }
            }
        }
    } else {
        for (int i = col0 + tid; i < col1; i += 128) {
            float w[4];
#pragma unroll
            for (int q = 0; q < 4; ++q)
                w[q] = (first_j + q < A.mix) ? A.fn[(long long)(first_j + q) * A.fnp + i] : 0.0f;
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                if (first_r + r < A.rows) {
                    const float v = bf2f(A.h[(long long)(first_r + r) * A.hp + i]);
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        if (first_j + q <= A.mix)
                            sum[q][r] += v * ((first_j + q == A.mix) ? v : w[q]);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        if (first_j + q <= A.mix) {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const float ws = wave_sum(sum[q][r]);
                if (lane == 0) sums[q][r][wave] = ws;
            }
        }
    }
    __syncthreads();
    if (tid < 16) {
        const int q = tid >> 2, r = tid & 3;
        if (first_j + q <= A.mix && first_r + r < A.rows) {
            mhc_partials(A.x, first_r + r, A.xp)[(first_j + q) * A.chunks + chunk] =
                sums[q][r][0] + sums[q][r][1] + sums[q][r][2] + sums[q][r][3];
        }
    }
}

__device__ __forceinline__ void mhc_collapse_add8(float (&v)[8], const uint16_t* h, float pre) {
    const uint4 hq = *reinterpret_cast<const uint4*>(h);
    const uint32_t w[4] = {hq.x, hq.y, hq.z, hq.w};
#pragma unroll
    for (int e = 0; e < 8; ++e)
        v[e] += pre * bf2f((uint16_t)(w[e >> 1] >> (16 * (e & 1))));
}

__device__ __forceinline__ void mhc_collapse8(float (&v)[8], const uint16_t* h,
                                              int n, int hc, const float* pre) {
    if (hc == 4) {
#pragma unroll
        for (int j = 0; j < 4; ++j) mhc_collapse_add8(v, h + (long long)j * n, pre[j]);
    } else {
        for (int j = 0; j < hc; ++j) mhc_collapse_add8(v, h + (long long)j * n, pre[j]);
    }
}

__device__ __forceinline__ float mhc_finish_rms_inv(float xsq, float* wave_sums, int n, float eps) {
    const int tid = (int)threadIdx.x;
    const float sum = wave_sum(xsq);
    if ((tid & 31) == 0) wave_sums[tid >> 5] = sum;
    __syncthreads();
    /* Broadcast the eight wave sums, in the same order as lane 0 of block_sum. */
    const float even = (wave_sums[0] + wave_sums[4]) + (wave_sums[2] + wave_sums[6]);
    const float odd = (wave_sums[1] + wave_sums[5]) + (wave_sums[3] + wave_sums[7]);
    return 1.0f / sqrtf((even + odd) / (float)n + eps);
}

/* Stage 2: reduce partials -> ERNIE norm (eps OUTSIDE sqrt) -> Sinkhorn (wave0 warp-shuffle for hc=4)
 * -> vectorized uint4 collapse (+ single-pass register-cached fused RMSNorm for n <= 4096). */
__global__ __launch_bounds__(256) void ernie_mhc_finish_kernel(MhcPreArgs A) {
    __shared__ float smem[8];
    __shared__ float coeff_s[80];
    __shared__ float pre_s[8];
    __shared__ float inv_norm_s;
    const int r = (int)blockIdx.x, tid = (int)threadIdx.x;
    if (r >= A.rows) return;
    const int hc = A.hc, mix = hc * (hc + 2), width = hc * A.n;
    const float* part = mhc_partials((uint16_t*)A.x, r, A.xp);

    /* Read all partials out of x before any thread writes final output to x. */
    float partial = 0.0f;
    if (tid <= mix) {
        const int step = A.decode ? mix + 1 : 1;
        const int base = A.decode ? tid : tid * A.chunks;
        for (int c = 0; c < A.chunks; ++c) partial += part[base + c * step];
    }

    if (hc == 4) {
        /* Wave 0 computes pre, post, and 4x4 Sinkhorn in registers via wave32 shuffles. */
        if (tid < 32) {
            const float sq = __shfl(partial, 24, 32);
            const float inv = 1.0f / (sqrtf(sq / (float)width) + A.eps);
            const float s0 = A.scale[0], s1 = A.scale[1], s2 = A.scale[2];
            const float c_j = (tid < 24) ? (partial * inv) : 0.0f;
            const float post_c = __shfl(c_j, (tid + 4) & 31, 32);
            const float mat_c = __shfl(c_j, (tid + 8) & 31, 32);
            if (tid < 4) {
                pre_s[tid] = sigmoid_f(c_j * s0 + A.base[tid]) + A.pre_eps;
                A.post[(long long)r * A.postp + tid] = sigmoid_f(post_c * s1 + A.base[4 + tid]) * A.post_mult;
            }
            float val = (tid < 16) ? (mat_c * s2 + A.base[8 + tid]) : 0.0f;
            const int row_base = tid & ~3, col = tid & 3;
            float mx = -INFINITY;
#pragma unroll
            for (int k = 0; k < 4; ++k) mx = fmaxf(mx, __shfl(val, row_base + k, 32));
            val = expf(val - mx);
            float sum = 0.0f;
#pragma unroll
            for (int k = 0; k < 4; ++k) sum += __shfl(val, row_base + k, 32);
            val = val / sum + A.sinkhorn_eps;
            for (int it = 0; it < A.iters; ++it) {
                if (it) {
                    sum = 0.0f;
#pragma unroll
                    for (int k = 0; k < 4; ++k) sum += __shfl(val, row_base + k, 32);
                    val /= sum + A.sinkhorn_eps;
                }
                sum = 0.0f;
#pragma unroll
                for (int j = 0; j < 4; ++j) sum += __shfl(val, col + 4 * j, 32);
                val /= sum + A.sinkhorn_eps;
            }
            if (tid < 16) A.comb[(long long)r * A.combp + tid] = val;
        }
    } else {
        if (tid == mix) inv_norm_s = 1.0f / (sqrtf(partial / (float)width) + A.eps);
        else if (tid < mix) coeff_s[tid] = partial;
        __syncthreads();
    }
    if (hc != 4 && tid == 0) {
        const float inv = inv_norm_s;
        const float s0 = A.scale[0], s1 = A.scale[1], s2 = A.scale[2];
        float c[64];
        for (int j = 0; j < hc; ++j) {
            pre_s[j] = sigmoid_f((coeff_s[j] * inv) * s0 + A.base[j]) + A.pre_eps;
            const float z = (coeff_s[hc + j] * inv) * s1 + A.base[hc + j];
            A.post[(long long)r * A.postp + j] = sigmoid_f(z) * A.post_mult;
        }
        for (int j = 0; j < hc; ++j) {
            float mx = -INFINITY;
            for (int k = 0; k < hc; ++k) {
                const int q = 2 * hc + j * hc + k;
                mx = fmaxf(mx, (coeff_s[q] * inv) * s2 + A.base[q]);
            }
            float sum = 0.0f;
            for (int k = 0; k < hc; ++k) {
                const int q = 2 * hc + j * hc + k;
                c[j * hc + k] = expf((coeff_s[q] * inv) * s2 + A.base[q] - mx);
                sum += c[j * hc + k];
            }
            for (int k = 0; k < hc; ++k) c[j * hc + k] = c[j * hc + k] / sum + A.sinkhorn_eps;
        }
        for (int it = 0; it < A.iters; ++it) {
            if (it) {
                for (int j = 0; j < hc; ++j) {
                    float sum = 0.0f;
                    for (int k = 0; k < hc; ++k) sum += c[j * hc + k];
                    for (int k = 0; k < hc; ++k) c[j * hc + k] /= sum + A.sinkhorn_eps;
                }
            }
            for (int k = 0; k < hc; ++k) {
                float sum = 0.0f;
                for (int j = 0; j < hc; ++j) sum += c[j * hc + k];
                for (int j = 0; j < hc; ++j) c[j * hc + k] /= sum + A.sinkhorn_eps;
            }
        }
        const long long cb = (long long)r * A.combp;
        for (int j = 0; j < hc * hc; ++j) A.comb[cb + j] = c[j];
    }
    __syncthreads();

    const uint16_t* hr = (const uint16_t*)A.h + (long long)r * A.hp;
    uint16_t* xr = (uint16_t*)A.x + (long long)r * A.xp;
    const bool vec8 = (((uintptr_t)hr | (uintptr_t)xr | (A.fused ? (uintptr_t)A.gain : 0u)) & 15u) == 0u &&
                      (A.n & 7) == 0;

    if (!A.fused) {
        if (vec8) {
            for (int i = tid * 8; i < A.n; i += 256 * 8) {
                float v[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                mhc_collapse8(v, hr + i, A.n, hc, pre_s);
                uint32_t ow[4];
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    ow[e] = (uint32_t)f2bf(v[2 * e]) | ((uint32_t)f2bf(v[2 * e + 1]) << 16);
                *reinterpret_cast<uint4*>(xr + i) = make_uint4(ow[0], ow[1], ow[2], ow[3]);
            }
        } else {
            for (int i = tid; i < A.n; i += 256) {
                float v = 0.0f;
                for (int j = 0; j < hc; ++j) v += pre_s[j] * bf2f(hr[(long long)j * A.n + i]);
                xr[i] = f2bf(v);
            }
        }
        return;
    }

    /* Fused weighted RMSNorm over the collapse (bf16 only). */
    if (vec8 && A.n <= 4096) {
        /* Register-cached collapse: reads h once, 16 VGPRs/thread for n <= 4096. */
        float cv[2][8];
        float xsq = 0.0f;
#pragma unroll
        for (int blk = 0; blk < 2; ++blk) {
            const int i = (blk * 256 + tid) * 8;
            if (i < A.n) {
                float v[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                mhc_collapse8(v, hr + i, A.n, hc, pre_s);
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    cv[blk][e] = v[e];
                    xsq += v[e] * v[e];
                }
            }
        }
        const float inv = mhc_finish_rms_inv(xsq, smem, A.n, A.rms_eps);
#pragma unroll
        for (int blk = 0; blk < 2; ++blk) {
            const int i = (blk * 256 + tid) * 8;
            if (i < A.n) {
                const uint4 gq = *reinterpret_cast<const uint4*>(A.gain + i);
                const uint32_t gw[4] = {gq.x, gq.y, gq.z, gq.w};
                uint32_t ow[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float g0 = bf2f((uint16_t)(gw[e] & 0xffffu));
                    const float g1 = bf2f((uint16_t)(gw[e] >> 16));
                    const uint16_t b0 = f2bf((bfr(cv[blk][2 * e]) * g0) * inv);
                    const uint16_t b1 = f2bf((bfr(cv[blk][2 * e + 1]) * g1) * inv);
                    ow[e] = (uint32_t)b0 | ((uint32_t)b1 << 16);
                }
                *reinterpret_cast<uint4*>(xr + i) = make_uint4(ow[0], ow[1], ow[2], ow[3]);
            }
        }
        return;
    }

    if (vec8) {
        float xsq = 0.0f;
        for (int i = tid * 8; i < A.n; i += 256 * 8) {
            float v[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            mhc_collapse8(v, hr + i, A.n, hc, pre_s);
#pragma unroll
            for (int e = 0; e < 8; ++e) xsq += v[e] * v[e];
        }
        const float inv = mhc_finish_rms_inv(xsq, smem, A.n, A.rms_eps);
        for (int i = tid * 8; i < A.n; i += 256 * 8) {
            float v[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            mhc_collapse8(v, hr + i, A.n, hc, pre_s);
            const uint4 gq = *reinterpret_cast<const uint4*>(A.gain + i);
            const uint32_t gw[4] = {gq.x, gq.y, gq.z, gq.w};
            uint32_t ow[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float g0 = bf2f((uint16_t)(gw[e] & 0xffffu));
                const float g1 = bf2f((uint16_t)(gw[e] >> 16));
                const uint16_t b0 = f2bf((bfr(v[2 * e]) * g0) * inv);
                const uint16_t b1 = f2bf((bfr(v[2 * e + 1]) * g1) * inv);
                ow[e] = (uint32_t)b0 | ((uint32_t)b1 << 16);
            }
            *reinterpret_cast<uint4*>(xr + i) = make_uint4(ow[0], ow[1], ow[2], ow[3]);
        }
        return;
    }

    float xsq = 0.0f;
    for (int i = tid; i < A.n; i += 256) {
        float v = 0.0f;
        for (int j = 0; j < hc; ++j) v += pre_s[j] * bf2f(hr[(long long)j * A.n + i]);
        xsq += v * v;
    }
    const float inv = mhc_finish_rms_inv(xsq, smem, A.n, A.rms_eps);
    for (int i = tid; i < A.n; i += 256) {
        float v = 0.0f;
        for (int j = 0; j < hc; ++j) v += pre_s[j] * bf2f(hr[(long long)j * A.n + i]);
        const float num = bfr(v) * bf2f(A.gain[i]);
        xr[i] = f2bf(num * inv);
    }
}

/* ------------------------------------------------------------------ general fallback kernel */
__global__ void ernie_mhc_pre_kernel(MhcPreArgs A) {
    __shared__ float smem[8];
    __shared__ float coeff_s[80]; /* max hc=8 -> mix=80 */
    __shared__ float pre_s[8];
    const int r = (int)blockIdx.x, tid = (int)threadIdx.x;
    if (r >= A.rows) return;
    const int hc = A.hc, mix = hc * (hc + 2), width = hc * A.n;
    const long long hb = (long long)r * A.hp;

    /* Pass 1: RMS norm of h[r, :width] (eps OUTSIDE sqrt per ERNIE equation). */
    float sq = 0.0f;
    for (int i = tid; i < width; i += 256) {
        const float v = load_h(A, hb + (long long)i * A.hs);
        sq += v * v;
    }
    const float tot_sq = block_sum(sq, smem, 256);
    const float norm = sqrtf(tot_sq / (float)width) + A.eps;

    /* Pass 2: dot(h[r], fn[q]) in groups of 8 rows to bound VGPRs while reusing h loads. */
    for (int q0 = 0; q0 < mix; q0 += 8) {
        const int qn = (q0 + 8 <= mix) ? 8 : (mix - q0);
        float d[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int i = tid; i < width; i += 256) {
            const float hv = load_h(A, hb + (long long)i * A.hs);
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                if (k < qn) d[k] += hv * A.fn[(long long)(q0 + k) * A.fnp + (long long)i * A.fns];
            }
        }
        for (int k = 0; k < qn; ++k) {
            const float sum = block_sum(d[k], smem, 256);
            if (tid == 0) coeff_s[q0 + k] = sum / norm;
        }
    }
    __syncthreads();

    /* Thread 0 computes pre, post, and Sinkhorn comb. */
    if (tid == 0) {
        const float s0 = A.scale[0], s1 = A.scale[A.ss], s2 = A.scale[2 * A.ss];
        float c[64];
        for (int j = 0; j < hc; ++j) {
            pre_s[j] = sigmoid_f(coeff_s[j] * s0 + A.base[(long long)j * A.bs]) + A.pre_eps;
            const float z = coeff_s[hc + j] * s1 + A.base[(long long)(hc + j) * A.bs];
            A.post[(long long)r * A.postp + (long long)j * A.posts] = sigmoid_f(z) * A.post_mult;
        }
        for (int j = 0; j < hc; ++j) {
            float mx = -INFINITY;
            for (int k = 0; k < hc; ++k) {
                const int q = 2 * hc + j * hc + k;
                mx = fmaxf(mx, coeff_s[q] * s2 + A.base[(long long)q * A.bs]);
            }
            float sum = 0.0f;
            for (int k = 0; k < hc; ++k) {
                const int q = 2 * hc + j * hc + k;
                c[j * hc + k] = expf(coeff_s[q] * s2 + A.base[(long long)q * A.bs] - mx);
                sum += c[j * hc + k];
            }
            for (int k = 0; k < hc; ++k) c[j * hc + k] = c[j * hc + k] / sum + A.sinkhorn_eps;
        }
        for (int it = 0; it < A.iters; ++it) {
            if (it) {
                for (int j = 0; j < hc; ++j) {
                    float sum = 0.0f;
                    for (int k = 0; k < hc; ++k) sum += c[j * hc + k];
                    for (int k = 0; k < hc; ++k) c[j * hc + k] /= sum + A.sinkhorn_eps;
                }
            }
            for (int k = 0; k < hc; ++k) {
                float sum = 0.0f;
                for (int j = 0; j < hc; ++j) sum += c[j * hc + k];
                for (int j = 0; j < hc; ++j) c[j * hc + k] /= sum + A.sinkhorn_eps;
            }
        }
        const long long cb = (long long)r * A.combp;
        for (int j = 0; j < hc * hc; ++j) A.comb[cb + (long long)j * A.combs] = c[j];
    }
    __syncthreads();

    const long long xb = (long long)r * A.xp;
    if (!A.fused) {
        for (int i = tid; i < A.n; i += 256) {
            float v = 0.0f;
            for (int j = 0; j < hc; ++j) v += pre_s[j] * load_h(A, hb + (long long)(j * A.n + i) * A.hs);
            store_x(A, xb + (long long)i * A.xs, v);
        }
        return;
    }

    /* Fused weighted RMSNorm over the collapse (bf16 only). */
    float xsq = 0.0f;
    for (int i = tid; i < A.n; i += 256) {
        float v = 0.0f;
        for (int j = 0; j < hc; ++j) v += pre_s[j] * bf2f(((const uint16_t*)A.h)[hb + (long long)(j * A.n + i) * A.hs]);
        xsq += v * v;
    }
    const float tot_xsq = block_sum(xsq, smem, 256);
    const float inv = 1.0f / sqrtf(tot_xsq / (float)A.n + A.rms_eps);
    for (int i = tid; i < A.n; i += 256) {
        float v = 0.0f;
        for (int j = 0; j < hc; ++j) v += pre_s[j] * bf2f(((const uint16_t*)A.h)[hb + (long long)(j * A.n + i) * A.hs]);
        const float num = bfr(v) * bf2f(A.gain[(long long)i * A.gs]);
        ((uint16_t*)A.x)[xb + (long long)i * A.xs] = f2bf(num * inv);
    }
}

/* ------------------------------------------------------------------ ernie_mhc_head */
struct MhcHeadArgs {
    const uint16_t *h;
    const float *fn, *scale, *base;
    uint16_t *x;
    long long hp, hs, fnp, fns, bs, xp, xs;
    int rows, n, hc;
    float eps, gate_eps;
};
static_assert(sizeof(MhcHeadArgs) <= 512, "MhcHeadArgs exceeds AQL kernarg limit");

__global__ void ernie_mhc_head_kernel(MhcHeadArgs A) {
    __shared__ float smem[8];
    __shared__ float gate_s[8];
    const int r = (int)blockIdx.x, tid = (int)threadIdx.x;
    if (r >= A.rows) return;
    const int hc = A.hc, width = hc * A.n;
    const long long hb = (long long)r * A.hp;

    float sq = 0.0f;
    float d[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = tid; i < width; i += 256) {
        const float hv = bf2f(A.h[hb + (long long)i * A.hs]);
        sq += hv * hv;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (j < hc) d[j] += hv * A.fn[(long long)j * A.fnp + (long long)i * A.fns];
        }
    }
    const float tot_sq = block_sum(sq, smem, 256);
    /* eps INSIDE sqrt for ernie_mhc_head! */
    const float norm = sqrtf(tot_sq / (float)width + A.eps);
    const float s = A.scale[0];
    for (int j = 0; j < hc; ++j) {
        const float dot = block_sum(d[j], smem, 256);
        if (tid == 0) gate_s[j] = sigmoid_f(dot / norm * s + A.base[(long long)j * A.bs]) + A.gate_eps;
    }
    __syncthreads();
    const long long xb = (long long)r * A.xp;
    for (int i = tid; i < A.n; i += 256) {
        float v = 0.0f;
        for (int j = 0; j < hc; ++j) v += gate_s[j] * bf2f(A.h[hb + (long long)(j * A.n + i) * A.hs]);
        A.x[xb + (long long)i * A.xs] = f2bf(v);
    }
}

static int launch_mhc_pre(const RadArgs* a, RadStream stream, bool fused) {
    const int off = fused ? 1 : 0;
    if (!a || !a->t || a->n_t < 7 + off) return RAD_E_INVAL;
    const long long M = r4d_geti_or(a, "M", 0), n = r4d_geti_or(a, "n", 0),
                    hc = r4d_geti_or(a, "hc", 0), iters = r4d_geti_or(a, "iterations", 0);
    const float eps = (float)r4d_getf(a, "eps", 0.0), pre_eps = (float)r4d_getf(a, "pre_eps", 0.0),
                sinkhorn_eps = (float)r4d_getf(a, "sinkhorn_eps", 0.0), post_mult = (float)r4d_getf(a, "post_mult", 0.0);
    if (M <= 0 || n <= 0 || hc < 2 || hc > 8 || iters < 1 || iters > 256 || !std::isfinite(eps) || eps <= 0.0f ||
        !std::isfinite(pre_eps) || pre_eps < 0.0f || !std::isfinite(sinkhorn_eps) || sinkhorn_eps <= 0.0f ||
        !std::isfinite(post_mult) || post_mult <= 0.0f)
        return RAD_E_SHAPE;
    const RadTensor &h = a->t[0], &fn = a->t[1], &scale = a->t[2], &base = a->t[3];
    const RadTensor *gain = fused ? &a->t[4] : nullptr;
    const RadTensor &x = a->t[4 + off], &post = a->t[5 + off], &comb = a->t[6 + off];
    float rms_eps = 0.0f;
    if (fused) {
        rms_eps = (float)r4d_getf(a, "rms_eps", 0.0);
        if (!std::isfinite(rms_eps) || rms_eps <= 0.0f) return RAD_E_SHAPE;
    }
    if (!h.data || !fn.data || !scale.data || !base.data || !x.data || !post.data || !comb.data || (gain && !gain->data))
        return RAD_E_SHAPE;
    if (fused && (h.dtype != RAD_BF16 || gain->dtype != RAD_BF16 || !pos_strides(*gain) || gain->rank != 1 || numel_t(*gain) != n))
        return RAD_E_SHAPE;
    if ((h.dtype != RAD_F32 && h.dtype != RAD_BF16) || x.dtype != h.dtype || fn.dtype != RAD_F32 ||
        scale.dtype != RAD_F32 || base.dtype != RAD_F32 || post.dtype != RAD_F32 || comb.dtype != RAD_F32)
        return RAD_E_SHAPE;
    if (!pos_strides(h) || !pos_strides(fn) || !pos_strides(scale) || !pos_strides(base) || !pos_strides(x) ||
        !pos_strides(post) || !pos_strides(comb))
        return RAD_E_SHAPE;
    if (h.rank != 2 || fn.rank != 2 || x.rank != 2 || post.rank != 2 || comb.rank != 2 || scale.rank != 1 || base.rank != 1)
        return RAD_E_SHAPE;
    const long long width = hc * n, mix = hc * (hc + 2), rows = h.shape[0];
    if (rows < 1 || rows > M || rows > 65535 || width > INT_MAX || h.shape[1] != width ||
        fn.shape[0] != mix || fn.shape[1] != width || numel_t(scale) != 3 || numel_t(base) != mix ||
        x.shape[0] != rows || x.shape[1] != n || post.shape[0] != rows || post.shape[1] != hc ||
        comb.shape[0] != rows || comb.shape[1] != hc * hc)
        return RAD_E_SHAPE;

    R4D_ARGS(MhcPreArgs, P);
    P.h = h.data; P.fn = (const float*)fn.data; P.scale = (const float*)scale.data; P.base = (const float*)base.data;
    P.gain = gain ? (const uint16_t*)gain->data : nullptr;
    P.x = x.data; P.post = (float*)post.data; P.comb = (float*)comb.data;
    P.hp = h.stride[0]; P.hs = h.stride[1]; P.fnp = fn.stride[0]; P.fns = fn.stride[1];
    P.ss = scale.stride[0]; P.bs = base.stride[0]; P.gs = gain ? gain->stride[0] : 1;
    P.xp = x.stride[0]; P.xs = x.stride[1]; P.postp = post.stride[0]; P.posts = post.stride[1];
    P.combp = comb.stride[0]; P.combs = comb.stride[1];
    P.rows = (int)rows; P.n = (int)n; P.hc = (int)hc; P.iters = (int)iters;
    P.h_f32 = (h.dtype == RAD_F32) ? 1 : 0; P.fused = fused ? 1 : 0;
    P.eps = eps; P.pre_eps = pre_eps; P.sinkhorn_eps = sinkhorn_eps; P.post_mult = post_mult; P.rms_eps = rms_eps;

    /* The partial sums are chunked and then summed in an order that depends on this flag, so a row's mHC pre-collapse differs in the
     * last f32 bits between a step of fewer than four rows and one of four or more. R4D_MHC_DECODE_ROWS (libr4d default 0: the many-row form for every step; the ERNIE plugin exports 4 at model build when
     * speculation is off, 0 when on) is that bound. */
    static const int mhc_decode_rows = r4d_ernie::r4d_pin_env("R4D_MHC_DECODE_ROWS", r4d_ernie::pin::kMhcDecodeRows);
    const bool decode = r4d_ernie::pin::mhc_decode((int)rows, mhc_decode_rows);
    P.decode = decode ? 1 : 0;
    const bool vec8 = decode && (((uintptr_t)h.data & 15u) == 0u) && ((P.hp & 7) == 0) &&
                      (((uintptr_t)fn.data & 15u) == 0u) && ((P.fnp & 3) == 0) && ((width & 7) == 0);
    long long chunk_cols = vec8 ? kDecodeChunkCols : kChunkCols;
    long long chunks = (width + chunk_cols - 1) / chunk_cols;
    /* Reserve two bytes for aligning a BF16 x row up to float alignment. */
    const long long scratch_chunks = (n * 2 - 2) / ((mix + 1) * 4);
    const long long max_chunks = scratch_chunks < 65535 ? scratch_chunks : 65535;
    if (decode && max_chunks > 0 && chunks > max_chunks) {
        /* Widen the chunks to cover the full row within the scratch and grid limits. */
        chunk_cols = 8 * ((width + 8 * max_chunks - 1) / (8 * max_chunks));
        chunks = (width + chunk_cols - 1) / chunk_cols;
    }
    const bool contig = (h.dtype == RAD_BF16) &&
                        P.hs == 1 && P.fns == 1 && P.xs == 1 && P.ss == 1 && P.bs == 1 &&
                        P.posts == 1 && P.combs == 1 && (!fused || P.gs == 1) &&
                        P.xp >= n && P.hp >= width && P.fnp >= width &&
                        !operands_overlap(x, h) && !operands_overlap(x, fn) &&
                        !operands_overlap(x, scale) && !operands_overlap(x, base) &&
                        (!fused || !operands_overlap(x, *gain));
    const bool use_2stage = contig && width >= kChunkCols && chunks <= max_chunks;

    if (use_2stage) {
        P.chunks = (int)chunks;
        R4D_ARGS(MhcPartialArgs, PA);
        PA.h = (const uint16_t*)h.data;
        PA.fn = (const float*)fn.data;
        PA.x = (uint16_t*)x.data;
        PA.hp = P.hp; PA.fnp = P.fnp; PA.xp = P.xp;
        PA.rows = (int)rows; PA.width = (int)width; PA.mix = (int)mix; PA.chunks = (int)chunks;

        if (decode) {
            PA.decode_cols = (int)chunk_cols;
            const dim3 grid((unsigned)chunks, (unsigned)((mix + 4) / 4), (unsigned)rows);
            if (vec8) {
                R4D_KLAUNCH(R4D_KNAME(ernie_mhc_partial_decode4_kernel<true>), grid, dim3(128),
                            (hipStream_t)stream, PA);
            } else {
                R4D_KLAUNCH(R4D_KNAME(ernie_mhc_partial_decode4_kernel<false>), grid, dim3(128),
                            (hipStream_t)stream, PA);
            }
        } else {
            const bool vec4 = (((uintptr_t)PA.h & 7u) == 0u) && ((PA.hp & 3) == 0) &&
                              (((uintptr_t)PA.fn & 15u) == 0u) && ((PA.fnp & 3) == 0) &&
                              ((PA.width & 3) == 0);
            const dim3 grid((unsigned)chunks, (unsigned)((mix + 4) / 4), (unsigned)((rows + 3) / 4));
            if (vec4) {
                R4D_KLAUNCH(R4D_KNAME(ernie_mhc_partial_rows4_coeff4_kernel<true>),
                            grid, dim3(128), (hipStream_t)stream, PA);
            } else {
                R4D_KLAUNCH(R4D_KNAME(ernie_mhc_partial_rows4_coeff4_kernel<false>),
                            grid, dim3(128), (hipStream_t)stream, PA);
            }
        }
        if (hipGetLastError() != hipSuccess) return RAD_E_DEVICE;
        R4D_KLAUNCH(R4D_KNAME(ernie_mhc_finish_kernel), dim3((unsigned)rows), dim3(256), (hipStream_t)stream, P);
        return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
    }

    R4D_KLAUNCH(R4D_KNAME(ernie_mhc_pre_kernel), dim3((unsigned)rows), dim3(256), (hipStream_t)stream, P);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

} /* namespace r4d_ernie_mhc */

extern "C" {

int r4d_rad_ernie_mhc_pre(const RadArgs* a, RadStream stream) {
    return r4d_ernie_mhc::launch_mhc_pre(a, stream, false);
}

int r4d_rad_ernie_mhc_pre_rms(const RadArgs* a, RadStream stream) {
    return r4d_ernie_mhc::launch_mhc_pre(a, stream, true);
}

int r4d_rad_ernie_mhc_head(const RadArgs* a, RadStream stream) {
    using namespace r4d_ernie;
    using namespace r4d_ernie_mhc;
    if (!a || !a->t || a->n_t < 5) return RAD_E_INVAL;
    const long long M = r4d_geti_or(a, "M", 0), n = r4d_geti_or(a, "n", 0), hc = r4d_geti_or(a, "hc", 0);
    const float eps = (float)r4d_getf(a, "eps", 0.0), gate_eps = (float)r4d_getf(a, "gate_eps", 0.0);
    if (M <= 0 || n <= 0 || hc < 2 || hc > 8 || !std::isfinite(eps) || eps <= 0.0f ||
        !std::isfinite(gate_eps) || gate_eps < 0.0f)
        return RAD_E_SHAPE;
    const RadTensor &h = a->t[0], &fn = a->t[1], &scale = a->t[2], &base = a->t[3], &x = a->t[4];
    if (!h.data || !fn.data || !scale.data || !base.data || !x.data) return RAD_E_SHAPE;
    if (h.dtype != RAD_BF16 || x.dtype != RAD_BF16 || fn.dtype != RAD_F32 || scale.dtype != RAD_F32 || base.dtype != RAD_F32)
        return RAD_E_SHAPE;
    if (!pos_strides(h) || !pos_strides(fn) || !pos_strides(scale) || !pos_strides(base) || !pos_strides(x))
        return RAD_E_SHAPE;
    const long long width = hc * n;
    if (h.rank != 2 || fn.rank != 2 || x.rank != 2 || scale.rank != 1 || base.rank != 1) return RAD_E_SHAPE;
    const long long rows = h.shape[0];
    if (rows < 1 || rows > M || rows > 65535 || width > INT_MAX || h.shape[1] != width ||
        fn.shape[0] != hc || fn.shape[1] != width || numel_t(scale) != 1 || numel_t(base) != hc ||
        x.shape[0] != rows || x.shape[1] != n)
        return RAD_E_SHAPE;
    R4D_ARGS(MhcHeadArgs, P);
    P.h = (const uint16_t*)h.data; P.fn = (const float*)fn.data; P.scale = (const float*)scale.data; P.base = (const float*)base.data;
    P.x = (uint16_t*)x.data;
    P.hp = h.stride[0]; P.hs = h.stride[1]; P.fnp = fn.stride[0]; P.fns = fn.stride[1];
    P.bs = base.stride[0]; P.xp = x.stride[0]; P.xs = x.stride[1];
    P.rows = (int)rows; P.n = (int)n; P.hc = (int)hc; P.eps = eps; P.gate_eps = gate_eps;
    R4D_KLAUNCH(R4D_KNAME(ernie_mhc_head_kernel), dim3((unsigned)rows), dim3(256), (hipStream_t)stream, P);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

} /* extern "C" */
