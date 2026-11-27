#include <hip/hip_runtime.h>
#include "r4d_args.h"
#include "r4d_chain.h"
#include "r4d_common.h"
#include "r4d_emit.h"
#include "r4d_moe_sort.h"
#include <climits>
#include <cmath>
#include <cstdlib>
#include "r4d_ernie_pins.h"

namespace {
// One wave computes R rows by C columns, so each loaded weight serves R rows and each input
// C columns. Every output has the same arithmetic: lane l accumulates k = l, l+32, ... in
// ascending order with fmaf, then the xor butterfly over 16, 8, 4, 2, 1. A lane loads U steps of
// k before their FMAs: at a decode pass's few rows a wave streams its weight row with no other
// work to hide the load behind, and one step in flight left the router at 150 GB/s. Tail rows and
// columns read a clamped valid address and are never stored.
template <int R, int C, int U>
__device__ __forceinline__ void
    gemm_nt_f32_tile_body(const uint16_t* x, const float* w, float* y, long xp, long wp, long yp,
                          int m, int n, int k) {
    const int lane = threadIdx.x & 31, c0 = (blockIdx.x * 4 + (threadIdx.x >> 5)) * C,
              r0 = blockIdx.y * R;
    if (c0 >= n) return;
    const uint16_t* xr[R];
    const float* wc[C];
#pragma unroll
    for (int q = 0; q < R; ++q) xr[q] = x + (long)(r0 + q < m ? r0 + q : m - 1) * xp;
#pragma unroll
    for (int c = 0; c < C; ++c) wc[c] = w + (long)(c0 + c < n ? c0 + c : n - 1) * wp;
    float sum[R][C];
#pragma unroll
    for (int q = 0; q < R; ++q)
#pragma unroll
        for (int c = 0; c < C; ++c) sum[q][c] = 0;
    int j = lane;
    for (; j + 32 * (U - 1) < k; j += 32 * U) {
        float weight[U][C];
        uint16_t in[U][R];
#pragma unroll
        for (int u = 0; u < U; ++u)
#pragma unroll
            for (int c = 0; c < C; ++c) weight[u][c] = wc[c][j + 32 * u];
#pragma unroll
        for (int u = 0; u < U; ++u)
#pragma unroll
            for (int q = 0; q < R; ++q) in[u][q] = xr[q][j + 32 * u];
#pragma unroll
        for (int u = 0; u < U; ++u)
#pragma unroll
            for (int q = 0; q < R; ++q) {
                const float v = bf16_to_f32(in[u][q]);
#pragma unroll
                for (int c = 0; c < C; ++c) sum[q][c] = fmaf(v, weight[u][c], sum[q][c]);
            }
    }
    for (; j < k; j += 32) {
#pragma unroll
        for (int q = 0; q < R; ++q) {
            const float v = bf16_to_f32(xr[q][j]);
#pragma unroll
            for (int c = 0; c < C; ++c) sum[q][c] = fmaf(v, wc[c][j], sum[q][c]);
        }
    }
#pragma unroll
    for (int q = 0; q < R; ++q)
#pragma unroll
        for (int c = 0; c < C; ++c) {
#pragma unroll
            for (int mask = 16; mask; mask >>= 1) sum[q][c] += __shfl_xor(sum[q][c], mask, 32);
        }
    if (lane) return;
#pragma unroll
    for (int q = 0; q < R; ++q)
        if (r0 + q < m)
#pragma unroll
            for (int c = 0; c < C; ++c)
                if (c0 + c < n) y[(long)(r0 + q) * yp + c0 + c] = sum[q][c];
}

template <int R, int C, int U, class CH = R4dUnchained>
__global__ void __launch_bounds__(128)
    gemm_nt_f32_tile_kernel(const uint16_t* x, const float* w, float* y, long xp, long wp, long yp,
                            int m, int n, int k) {
    if (CH::on) r4d_chain_wait();
    gemm_nt_f32_tile_body<R, C, U>(x, w, y, xp, wp, yp, m, n, k);
    if (CH::on) r4d_chain_done();
}

template <int R, int C, int U = 1>
int gemm_f32_tile(RadStream stream, const RadTensor& x, const RadTensor& w, const RadTensor& y,
                  long n, long k, bool chain) {
    const long m = x.shape[0];
    return R4D_EMIT(stream,
                    (chain ? gemm_nt_f32_tile_kernel<R, C, U, r4dchain>
                           : gemm_nt_f32_tile_kernel<R, C, U, R4dUnchained>),
                    dim3((unsigned)((n + 4 * C - 1) / (4 * C)), (unsigned)((m + R - 1) / R)),
                    dim3(128), 0, (const uint16_t*)x.data, (const float*)w.data, (float*)y.data,
                    x.stride[0], w.stride[0], y.stride[0], (int)m, (int)n, (int)k);
}
// BF16-weight decode / fallback kernel: 4 waves (128 threads), 1 output column per wave, RM rows
// accumulated simultaneously so each BF16 weight row is read once across up to 8 tokens.
// With VEC8=true, each lane loads 8 BF16 (16B r4d_u32x4) per step (256 cols/wave/iter) with U
// weight loads in flight to saturate GDDR6 bandwidth; scalar tail handles non-multiple-of-8 K.
template <int RM, int U, bool VEC8>
__global__ void __launch_bounds__(128)
    gemm_nt_f32_bf16_vec_kernel(const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
                                float* __restrict__ y, long xp, long wp, long yp, int m, int n,
                                int k) {
    const int lane = (int)(threadIdx.x & 31);
    const int c = (int)blockIdx.x * 4 + (int)(threadIdx.x >> 5);
    const int r0 = (int)blockIdx.y * RM;
    if (c >= n) return;
    const uint16_t* wr = w + (long)c * wp;
    const uint16_t* xr[RM];
#pragma unroll
    for (int q = 0; q < RM; ++q) xr[q] = x + (long)(r0 + q < m ? r0 + q : m - 1) * xp;
    float sum[RM];
#pragma unroll
    for (int q = 0; q < RM; ++q) sum[q] = 0.0f;

    if constexpr (VEC8) {
        const int k8 = k & ~7;
        int j = lane * 8;
        if constexpr (U > 1) {
            for (; j + 256 * (U - 1) < k8; j += 256 * U) {
                r4d_u32x4 wv[U];
#pragma unroll
                for (int u = 0; u < U; ++u) wv[u] = *(const r4d_u32x4*)(wr + j + 256 * u);
                if constexpr (RM == 1) {
                    r4d_u32x4 xv[U];
#pragma unroll
                    for (int u = 0; u < U; ++u) xv[u] = *(const r4d_u32x4*)(xr[0] + j + 256 * u);
#pragma unroll
                    for (int u = 0; u < U; ++u) {
#pragma unroll
                        for (int t = 0; t < 4; ++t) {
                            sum[0] = fmaf(__uint_as_float(xv[u][t] << 16),
                                          __uint_as_float(wv[u][t] << 16), sum[0]);
                            sum[0] = fmaf(__uint_as_float(xv[u][t] & 0xffff0000u),
                                          __uint_as_float(wv[u][t] & 0xffff0000u), sum[0]);
                        }
                    }
                } else {
#pragma unroll
                    for (int u = 0; u < U; ++u) {
#pragma unroll
                        for (int q = 0; q < RM; ++q) {
                            const r4d_u32x4 xv = *(const r4d_u32x4*)(xr[q] + j + 256 * u);
#pragma unroll
                            for (int t = 0; t < 4; ++t) {
                                sum[q] = fmaf(__uint_as_float(xv[t] << 16),
                                              __uint_as_float(wv[u][t] << 16), sum[q]);
                                sum[q] = fmaf(__uint_as_float(xv[t] & 0xffff0000u),
                                              __uint_as_float(wv[u][t] & 0xffff0000u), sum[q]);
                            }
                        }
                    }
                }
            }
        }
        for (; j < k8; j += 256) {
            const r4d_u32x4 wv = *(const r4d_u32x4*)(wr + j);
#pragma unroll
            for (int q = 0; q < RM; ++q) {
                const r4d_u32x4 xv = *(const r4d_u32x4*)(xr[q] + j);
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    sum[q] = fmaf(__uint_as_float(xv[t] << 16), __uint_as_float(wv[t] << 16),
                                  sum[q]);
                    sum[q] = fmaf(__uint_as_float(xv[t] & 0xffff0000u),
                                  __uint_as_float(wv[t] & 0xffff0000u), sum[q]);
                }
            }
        }
        const int jt = k8 + lane;
        if (jt < k) {
            const float wv = bf16_to_f32(wr[jt]);
#pragma unroll
            for (int q = 0; q < RM; ++q) sum[q] = fmaf(bf16_to_f32(xr[q][jt]), wv, sum[q]);
        }
    } else {
        for (int j = lane; j < k; j += 32) {
            const float wv = bf16_to_f32(wr[j]);
#pragma unroll
            for (int q = 0; q < RM; ++q) sum[q] = fmaf(bf16_to_f32(xr[q][j]), wv, sum[q]);
        }
    }
#pragma unroll
    for (int q = 0; q < RM; ++q) {
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) sum[q] += __shfl_xor(sum[q], mask, 32);
    }
    if (lane) return;
#pragma unroll
    for (int q = 0; q < RM; ++q)
        if (r0 + q < m) y[(long)(r0 + q) * yp + c] = sum[q];
}

template <int RM, int U, bool VEC8>
int gemm_f32_bf16_vec(RadStream stream, const RadTensor& x, const RadTensor& w, const RadTensor& y,
                      long n, long k) {
    const long m = x.shape[0];
    return R4D_EMIT(stream, (gemm_nt_f32_bf16_vec_kernel<RM, U, VEC8>),
                    dim3((unsigned)((n + 3) / 4), (unsigned)((m + RM - 1) / RM)), dim3(128), 0,
                    (const uint16_t*)x.data, (const uint16_t*)w.data, (float*)y.data, x.stride[0],
                    w.stride[0], y.stride[0], (int)m, (int)n, (int)k);
}

__device__ __forceinline__ void gemm_bf16_rb2_accumulate(r4d_u32x4 x, r4d_u32x4 w0, r4d_u32x4 w1,
                                                        float& sum0, float& sum1) {
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        const float lo = bf16w_lo(x[t]), hi = bf16w_hi(x[t]);
        sum0 = fmaf(lo, bf16w_lo(w0[t]), sum0);
        sum0 = fmaf(hi, bf16w_hi(w0[t]), sum0);
        sum1 = fmaf(lo, bf16w_lo(w1[t]), sum1);
        sum1 = fmaf(hi, bf16w_hi(w1[t]), sum1);
    }
}

/* Single-token BF16 GEMV. Each wave reuses its input vector across two weight rows. */
__global__ void __launch_bounds__(64)
    gemm_nt_f32_bf16_m1_rb2_kernel(const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
                                   float* __restrict__ y, long wp, int n, int k) {
    const int lane = (int)threadIdx.x & 31;
    const int c0 = (int)blockIdx.x * 4 + ((int)threadIdx.x >> 5) * 2;
    if (c0 >= n) return;
    const bool has_second = c0 + 1 < n;
    const uint16_t* w0 = w + (long)c0 * wp;
    const uint16_t* w1 = has_second ? w0 + wp : w0;
    float sum0 = 0.0f, sum1 = 0.0f;
    const int k8 = k & ~7, full_k = k8 & ~255;
    int j = lane * 8;

    if (full_k != 0) {
        r4d_u32x4 a = *reinterpret_cast<const r4d_u32x4*>(x + j);
        r4d_u32x4 b0 = *reinterpret_cast<const r4d_u32x4*>(w0 + j);
        r4d_u32x4 b1 = *reinterpret_cast<const r4d_u32x4*>(w1 + j);
        /* Load the next 256-element wave step while consuming the current one (U=2). */
        for (; j + 256 < full_k; j += 256) {
            const r4d_u32x4 next0 = *reinterpret_cast<const r4d_u32x4*>(w0 + j + 256);
            const r4d_u32x4 next1 = *reinterpret_cast<const r4d_u32x4*>(w1 + j + 256);
            const r4d_u32x4 next_x = *reinterpret_cast<const r4d_u32x4*>(x + j + 256);
            gemm_bf16_rb2_accumulate(a, b0, b1, sum0, sum1);
            a = next_x;
            b0 = next0;
            b1 = next1;
        }
        gemm_bf16_rb2_accumulate(a, b0, b1, sum0, sum1);
        j += 256;
    }

    if (j < k8) {
        const r4d_u32x4 a = *reinterpret_cast<const r4d_u32x4*>(x + j);
        const r4d_u32x4 b0 = *reinterpret_cast<const r4d_u32x4*>(w0 + j);
        const r4d_u32x4 b1 = *reinterpret_cast<const r4d_u32x4*>(w1 + j);
        gemm_bf16_rb2_accumulate(a, b0, b1, sum0, sum1);
    }
    const int jt = k8 + lane;
    if (jt < k) {
        const float a = bf16_to_f32(x[jt]);
        sum0 = fmaf(a, bf16_to_f32(w0[jt]), sum0);
        sum1 = fmaf(a, bf16_to_f32(w1[jt]), sum1);
    }
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) {
        sum0 += __shfl_xor(sum0, mask, 32);
        sum1 += __shfl_xor(sum1, mask, 32);
    }
    if (lane == 0) {
        y[c0] = sum0;
        if (has_second) y[c0 + 1] = sum1;
    }
}

#if defined(__clang__) || defined(__HIP_DEVICE_COMPILE__)
// BF16-weight prefill kernel: LDS-tiled RDNA4 gfx1201 BF16 WMMA with F32 accumulation and direct
// coalesced F32 output store to pitched y[m * yp + n].
// Tile shape: BM = WM * TM * 16, BN = WN * TN * 16, BK = 32, LDS pitch = 40 shorts (80 B: 16B
// aligned, zero bank conflicts on ds_load_b128).
// Both A (X) and B (W) load 8 contiguous BF16 per lane at k = s*16 + 8*(lane>>4) + 0..7, which
// matches both gfx1201 hardware (identical K bijection on A and B) and tests/r4d_emu.
// Output D[8*(lane>>4) + e, lane & 15] writes two 64B coalesced lines per wave per e.
// KC > 0 (the row-identity form, gemm_nt_f32_pair): K is summed in chunks of KC. Each chunk is one WMMA chain from zero, and the
// chunk partials are added in chunk order into a total that starts at zero: y = ((0 + c0) + c1) + ... The small-row kernel
// (gemm_nt_f32_bf16_kc_small_kernel) computes the same chunks with the same WMMA chain and adds them in the same order, so a row's
// value does not depend on how many rows share the launch or which kernel served it. KC == 0 is the plain chain over K.
template <int WM, int WN, int TM, int TN, int KC = 0>
__global__ void __launch_bounds__(WM * WN * 32)
    gemm_nt_f32_bf16_wmma_kernel(const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
                                 float* __restrict__ y, long xp, long wp, long yp, int m, int n,
                                 int k) {
    constexpr int BK = 32;
    constexpr int PITCH = BK + 8;
    constexpr int NTH = WM * WN * 32;
    constexpr int BM = WM * TM * 16;
    constexpr int BN = WN * TN * 16;
    constexpr int CPR = BK / 8;
    constexpr int AL = BM * CPR / NTH;
    constexpr int WL = BN * CPR / NTH;
    static_assert(AL >= 1 && WL >= 1 && BM * CPR % NTH == 0 && BN * CPR % NTH == 0,
                  "WMMA tile must divide cleanly into 16B chunks");

    __shared__ __attribute__((aligned(16))) uint16_t sA[BM * PITCH];
    __shared__ __attribute__((aligned(16))) uint16_t sW[BN * PITCH];

    const int tid = (int)threadIdx.x;
    const int wave = tid >> 5, lane = tid & 31;
    const int wm = wave / WN, wn = wave % WN;

    constexpr unsigned kGroupM = 8;
    const unsigned per = kGroupM * gridDim.x;
    const unsigned lin = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned first = (lin / per) * kGroupM;
    const unsigned rem = lin % per;
    const unsigned gm = gridDim.y - first < kGroupM ? gridDim.y - first : kGroupM;
    const int m0 = (int)(first + rem % gm) * BM;
    const int n0 = (int)(rem / gm) * BN;

    const uint16_t* aptr[AL];
    const uint16_t* wptr[WL];
    uint16_t* adst[AL];
    uint16_t* wdst[WL];
#pragma unroll
    for (int i = 0; i < AL; ++i) {
        const int c = tid + NTH * i, r = c / CPR, e = (c % CPR) * 8;
        const int gr = m0 + r < m ? m0 + r : m - 1;
        aptr[i] = x + (long)gr * xp + e;
        adst[i] = sA + r * PITCH + e;
    }
#pragma unroll
    for (int i = 0; i < WL; ++i) {
        const int c = tid + NTH * i, r = c / CPR, e = (c % CPR) * 8;
        const int gr = n0 + r < n ? n0 + r : n - 1;
        wptr[i] = w + (long)gr * wp + e;
        wdst[i] = sW + r * PITCH + e;
    }

    const uint16_t* fa = sA + (wm * TM * 16 + (lane & 15)) * PITCH + 8 * (lane >> 4);
    const uint16_t* fw = sW + (wn * TN * 16 + (lane & 15)) * PITCH + 8 * (lane >> 4);

    v8f acc[TM][TN];
    v8f tot[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) { acc[i][j] = (v8f)(0.0f); tot[i][j] = (v8f)(0.0f); }

    r4d_u32x4 ra[AL], rw[WL];
#pragma unroll
    for (int i = 0; i < AL; ++i) ra[i] = *(const r4d_u32x4*)aptr[i];
#pragma unroll
    for (int i = 0; i < WL; ++i) rw[i] = *(const r4d_u32x4*)wptr[i];

    const int nk = k / BK;
    for (int kt = 0; kt < nk; ++kt) {
        if (kt > 0) __syncthreads();
#pragma unroll
        for (int i = 0; i < AL; ++i) *(r4d_u32x4*)adst[i] = ra[i];
#pragma unroll
        for (int i = 0; i < WL; ++i) *(r4d_u32x4*)wdst[i] = rw[i];
        __syncthreads();
        if (kt + 1 < nk) {
            const long k1 = (long)(kt + 1) * BK;
#pragma unroll
            for (int i = 0; i < AL; ++i) ra[i] = *(const r4d_u32x4*)(aptr[i] + k1);
#pragma unroll
            for (int i = 0; i < WL; ++i) rw[i] = *(const r4d_u32x4*)(wptr[i] + k1);
        }
#pragma unroll
        for (int s = 0; s < BK / 16; ++s) {
            v8s af[TM], bf[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) af[i] = *(const v8s*)(fa + i * 16 * PITCH + s * 16);
#pragma unroll
            for (int j = 0; j < TN; ++j) bf[j] = *(const v8s*)(fw + j * 16 * PITCH + s * 16);
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] =
                        __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(af[i], bf[j], acc[i][j]);
#if defined(__HIP_DEVICE_COMPILE__)
            __builtin_amdgcn_sched_group_barrier(0x100, TM + TN, 0);
            __builtin_amdgcn_sched_group_barrier(0x008, TM * TN, 0);
#endif
        }
        if (KC > 0 && (((kt + 1) * BK) % KC == 0 || kt + 1 == nk)) {
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) { tot[i][j] = tot[i][j] + acc[i][j]; acc[i][j] = (v8f)(0.0f); }
        }
    }
    if (KC > 0) {
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
            for (int j = 0; j < TN; ++j) acc[i][j] = tot[i][j];
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int mb = m0 + (wm * TM + i) * 16 + 8 * (lane >> 4);
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int col = n0 + (wn * TN + j) * 16 + (lane & 15);
            if (col < n) {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int row = mb + e;
                    if (row < m) y[(long)row * yp + col] = acc[i][j][e];
                }
            }
        }
    }
}

template <int WM, int WN, int TM, int TN, int KC = 0>
int gemm_f32_bf16_wmma(RadStream stream, const RadTensor& x, const RadTensor& w, const RadTensor& y,
                       long n, long k) {
    constexpr int BM = WM * TM * 16;
    constexpr int BN = WN * TN * 16;
    const long m = x.shape[0];
    return R4D_EMIT(stream, (gemm_nt_f32_bf16_wmma_kernel<WM, WN, TM, TN, KC>),
                    dim3((unsigned)((n + BN - 1) / BN), (unsigned)((m + BM - 1) / BM)),
                    dim3(WM * WN * 32), 0, (const uint16_t*)x.data, (const uint16_t*)w.data,
                    (float*)y.data, x.stride[0], w.stride[0], y.stride[0], (int)m, (int)n, (int)k);
}

// The chunk of the row-identity form (gemm_nt_f32_pair, bf16 weights, K % 32 == 0).
constexpr int kPairKC = 256;
constexpr int kPairSmallWaves = 16;

// Small-row form of the chunked WMMA sum (KC = kPairKC): one block per 16 x 16 output tile, its waves take the K chunks
// (wave w: chunks w, w + W, ...), each chunk one WMMA chain from zero over the chunk's 16-wide k steps in order, operands loaded
// straight from global memory in the fragment layout the LDS-tiled kernel reads (A lane l: row l & 15, k = 16 s + 8 (l >> 4) + 0..7;
// B the same over the columns). Wave 0 adds the chunk partials in chunk order into a total from zero: the tiled kernel's arithmetic,
// row by row (the rows of a WMMA tile do not mix), with all the chunks of a tile in flight at once instead of one serial chain.
// Rows past m read row m - 1 (as the tiled kernel does) and are not stored.
__global__ void __launch_bounds__(kPairSmallWaves * 32)
    gemm_nt_f32_bf16_kc_small_kernel(const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
                                     float* __restrict__ y, long xp, long wp, long yp, int m, int n, int k) {
    constexpr int KC = kPairKC, W = kPairSmallWaves;
    __shared__ v8f part[W][32];
    const int tid = (int)threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int n0 = (int)blockIdx.x * 16, m0 = (int)blockIdx.y * 16;
    const int ar = m0 + (lane & 15) < m ? m0 + (lane & 15) : m - 1;
    const int br = n0 + (lane & 15) < n ? n0 + (lane & 15) : n - 1;
    const uint16_t* xa = x + (long)ar * xp + 8 * (lane >> 4);
    const uint16_t* wb = w + (long)br * wp + 8 * (lane >> 4);
    const int nch = (k + KC - 1) / KC;
    v8f tot = (v8f)(0.0f);
    for (int base = 0; base < nch; base += W) {
        const int c = base + wave;
        if (c < nch) {
            const int k0 = c * KC, k1 = k0 + KC < k ? k0 + KC : k;
            v8f acc = (v8f)(0.0f);
            for (int kk = k0; kk < k1; kk += 16) {
                const v8s af = *(const v8s*)(xa + kk);
                const v8s bf = *(const v8s*)(wb + kk);
                acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(af, bf, acc);
            }
            part[wave][lane] = acc;
        }
        __syncthreads();
        if (wave == 0) {
            const int cnt = nch - base < W ? nch - base : W;
            for (int i = 0; i < cnt; ++i) tot = tot + part[i][lane];
        }
        __syncthreads();
    }
    if (wave == 0) {
        const int col = n0 + (lane & 15);
        if (col < n) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int row = m0 + 8 * (lane >> 4) + e;
                if (row < m) y[(long)row * yp + col] = tot[e];
            }
        }
    }
}

// The row-identity GEMM of gemm_nt_f32_pair: every row count takes the chunked WMMA sum, so a row's f32 output is the same bits
// alone, in a decode step, a verify step or a prefill chunk. Up to kPairSmallMax rows the small-row kernel (a block per 16 x 16 tile,
// the chunks in parallel), above it the LDS-tiled kernel with the tile configs of gemm_f32_bf16_dispatch. The bound is speed only.
constexpr long kPairSmallMax = 64;
int gemm_f32_bf16_kc(RadStream stream, const RadTensor& x, const RadTensor& w, const RadTensor& y, long n, long k) {
    const long rows = x.shape[0];
    if (rows <= kPairSmallMax)
        return R4D_EMIT(stream, gemm_nt_f32_bf16_kc_small_kernel,
                        dim3((unsigned)((n + 15) / 16), (unsigned)((rows + 15) / 16)), dim3(kPairSmallWaves * 32), 0,
                        (const uint16_t*)x.data, (const uint16_t*)w.data, (float*)y.data, x.stride[0], w.stride[0],
                        y.stride[0], (int)rows, (int)n, (int)k);
    if (rows >= 1024 && n >= 256) return gemm_f32_bf16_wmma<2, 4, 2, 2, kPairKC>(stream, x, w, y, n, k);
    return gemm_f32_bf16_wmma<2, 2, 2, 2, kPairKC>(stream, x, w, y, n, k);
}
#endif

int gemm_f32_bf16_dispatch(RadStream stream, const RadTensor& x, const RadTensor& w,
                           const RadTensor& y, long n, long k) {
    const long rows = x.shape[0];
    const bool vec8 = (((uintptr_t)x.data & 15u) == 0) && (((uintptr_t)w.data & 15u) == 0) &&
                      ((x.stride[0] & 7) == 0) && ((w.stride[0] & 7) == 0);
    if (rows <= 8) {
        if (vec8) {
            if (rows == 1 && n >= 256)
                return R4D_EMIT(stream, (gemm_nt_f32_bf16_m1_rb2_kernel),
                                dim3((unsigned)((n + 3) / 4)), dim3(64), 0,
                                (const uint16_t*)x.data, (const uint16_t*)w.data, (float*)y.data,
                                w.stride[0], (int)n, (int)k);
            if (rows == 1) return gemm_f32_bf16_vec<1, 4, true>(stream, x, w, y, n, k);
            if (rows == 2) return gemm_f32_bf16_vec<2, 2, true>(stream, x, w, y, n, k);
            if (rows <= 4) return gemm_f32_bf16_vec<4, 2, true>(stream, x, w, y, n, k);
            return gemm_f32_bf16_vec<8, 2, true>(stream, x, w, y, n, k);
        }
        return gemm_f32_bf16_vec<8, 1, false>(stream, x, w, y, n, k);
    }
#if defined(__clang__) || defined(__HIP_DEVICE_COMPILE__)
    if (vec8 && k >= 32 && (k & 31) == 0) {
        if (rows >= 1024 && n >= 256)
            return gemm_f32_bf16_wmma<2, 4, 2, 2>(stream, x, w, y, n, k);
        if (rows > 32) return gemm_f32_bf16_wmma<2, 2, 2, 2>(stream, x, w, y, n, k);
        return gemm_f32_bf16_wmma<2, 2, 1, 2>(stream, x, w, y, n, k);
    }
#endif
    if (vec8) return gemm_f32_bf16_vec<8, 2, true>(stream, x, w, y, n, k);
    return gemm_f32_bf16_vec<8, 1, false>(stream, x, w, y, n, k);
}

int gemm_f32_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 3) return RAD_E_INVAL;
    const auto &x = a->t[0], &w = a->t[1], &y = a->t[2];
    const long n = r4d_geti_or(a, "N", 0), k = r4d_geti_or(a, "K", 0);
    if (x.dtype != RAD_BF16 || (w.dtype != RAD_F32 && w.dtype != RAD_BF16) || y.dtype != RAD_F32)
        return RAD_E_DTYPE;
    if (!x.data || !w.data || !y.data) return RAD_E_INVAL;
    /* Any width: the grid spans n (a wave a C-column tile), so the bound is not the router's 512.
     * A compressor's fused kv|gate projection is 1024 wide. */
    if (n < 1 || n > 65536 || k < 1 || k > 16384 || x.rank != 2 || w.rank != 2 || y.rank != 2 ||
        x.shape[0] < 1 || x.shape[0] > 65535 || x.shape[1] != k || w.shape[0] != n ||
        w.shape[1] != k || y.shape[0] != x.shape[0] || y.shape[1] != n || x.stride[1] != 1 ||
        w.stride[1] != 1 || y.stride[1] != 1 || x.stride[0] < k || w.stride[0] < k ||
        y.stride[0] < n)
        return RAD_E_SHAPE;
    if (w.dtype == RAD_BF16) return gemm_f32_bf16_dispatch(stream, x, w, y, n, k);
    /* The tile form, rows x columns per wave and steps of k in flight, by M; all forms are
     * byte-identical, so the choice is speed only. Measured on gfx1201 (r4d_selftest --perff32,
     * cold) at N 4/32/36/288, K 4096: 16x4 gains 3-5 % at N 288/32 above 64 rows but loses 16 % at
     * N 36; 8x4 holds at every width. Below 64 rows the steps of k in flight decide: 2x2 with 16
     * steps is 2 to 2.5 times as fast as one step at N 288 up to 16 rows, and about 3 times at N 32
     * at every count below 64; at N 288 from 17 rows 8x1 with 8 steps is faster.
     * A projection of a few columns has too few waves for wide tiles at any M. */
    const long rows = x.shape[0];
    const bool chain = r4d_geti_or(a, "chain", 0) != 0;
    if (n <= 8) return gemm_f32_tile<4, 1>(stream, x, w, y, n, k, chain);
    if (rows <= 16 || (rows < 64 && n < 128))
        return gemm_f32_tile<2, 2, 16>(stream, x, w, y, n, k, chain);
    if (rows < 64) return gemm_f32_tile<8, 1, 8>(stream, x, w, y, n, k, chain);
    return gemm_f32_tile<8, 4>(stream, x, w, y, n, k, chain);
}

/* Single-token paired BF16 GEMV: each wave computes y0[c] = x . w0[c] and y1[c] = x . w1[c]
 * sharing every uint4 activation load with U=2 prefetch. 2 waves/block -> 256 blocks at N=512. */
__global__ void __launch_bounds__(64)
    gemm_nt_f32_bf16_pair_m1_kernel(const uint16_t* __restrict__ x,
                                    const uint16_t* __restrict__ w0,
                                    const uint16_t* __restrict__ w1,
                                    float* __restrict__ y0, float* __restrict__ y1,
                                    long w0p, long w1p, int n, int k) {
    const int lane = (int)threadIdx.x & 31;
    const int c = (int)blockIdx.x * 2 + ((int)threadIdx.x >> 5);
    if (c >= n) return;
    const uint16_t* r0 = w0 + (long)c * w0p;
    const uint16_t* r1 = w1 + (long)c * w1p;
    float sum0 = 0.0f, sum1 = 0.0f;
    const int k8 = k & ~7, full_k = k8 & ~255;
    int j = lane * 8;

    if (full_k != 0) {
        r4d_u32x4 a = *reinterpret_cast<const r4d_u32x4*>(x + j);
        r4d_u32x4 b0 = *reinterpret_cast<const r4d_u32x4*>(r0 + j);
        r4d_u32x4 b1 = *reinterpret_cast<const r4d_u32x4*>(r1 + j);
        for (; j + 256 < full_k; j += 256) {
            const r4d_u32x4 next0 = *reinterpret_cast<const r4d_u32x4*>(r0 + j + 256);
            const r4d_u32x4 next1 = *reinterpret_cast<const r4d_u32x4*>(r1 + j + 256);
            const r4d_u32x4 next_x = *reinterpret_cast<const r4d_u32x4*>(x + j + 256);
            gemm_bf16_rb2_accumulate(a, b0, b1, sum0, sum1);
            a = next_x;
            b0 = next0;
            b1 = next1;
        }
        gemm_bf16_rb2_accumulate(a, b0, b1, sum0, sum1);
        j += 256;
    }
    if (j < k8) {
        const r4d_u32x4 a = *reinterpret_cast<const r4d_u32x4*>(x + j);
        const r4d_u32x4 b0 = *reinterpret_cast<const r4d_u32x4*>(r0 + j);
        const r4d_u32x4 b1 = *reinterpret_cast<const r4d_u32x4*>(r1 + j);
        gemm_bf16_rb2_accumulate(a, b0, b1, sum0, sum1);
    }
    const int jt = k8 + lane;
    if (jt < k) {
        const float a = bf16_to_f32(x[jt]);
        sum0 = fmaf(a, bf16_to_f32(r0[jt]), sum0);
        sum1 = fmaf(a, bf16_to_f32(r1[jt]), sum1);
    }
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) {
        sum0 += __shfl_xor(sum0, mask, 32);
        sum1 += __shfl_xor(sum1, mask, 32);
    }
    if (lane == 0) {
        y0[c] = sum0;
        y1[c] = sum1;
    }
}

template <int RM, int U, bool VEC8>
__global__ void __launch_bounds__(128)
    gemm_nt_f32_bf16_pair_vec_kernel(const uint16_t* __restrict__ x,
                                     const uint16_t* __restrict__ w0,
                                     const uint16_t* __restrict__ w1,
                                     float* __restrict__ y0, float* __restrict__ y1,
                                     long xp, long w0p, long w1p, long y0p, long y1p,
                                     int m, int n, int k) {
    const int lane = (int)threadIdx.x & 31;
    const int c = (int)blockIdx.x * 4 + ((int)threadIdx.x >> 5);
    const int r0 = (int)blockIdx.y * RM;
    const int half = (int)blockIdx.z;
    if (c >= n || r0 >= m) return;
    const uint16_t* wrow = (half ? w1 : w0) + (long)c * (half ? w1p : w0p);
    float* yout = half ? y1 : y0;
    const long yp = half ? y1p : y0p;
    float sum[RM] = {};
    if constexpr (VEC8) {
        const int k8 = k & ~7;
        for (int j = lane * 8; j < k8; j += 32 * 8 * U) {
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int jj = j + u * 256;
                if (jj < k8) {
                    const r4d_u32x4 bw = *reinterpret_cast<const r4d_u32x4*>(wrow + jj);
                    float wf[8];
#pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        wf[2 * t]     = bf16w_lo(bw[t]);
                        wf[2 * t + 1] = bf16w_hi(bw[t]);
                    }
#pragma unroll
                    for (int r = 0; r < RM; ++r) {
                        if (r0 + r < m) {
                            const r4d_u32x4 aw =
                                *reinterpret_cast<const r4d_u32x4*>(x + (long)(r0 + r) * xp + jj);
#pragma unroll
                            for (int t = 0; t < 4; ++t) {
                                sum[r] = fmaf(bf16w_lo(aw[t]), wf[2 * t], sum[r]);
                                sum[r] = fmaf(bf16w_hi(aw[t]), wf[2 * t + 1], sum[r]);
                            }
                        }
                    }
                }
            }
        }
        const int jt = k8 + lane;
        if (jt < k) {
            const float wf = bf16_to_f32(wrow[jt]);
#pragma unroll
            for (int r = 0; r < RM; ++r)
                if (r0 + r < m) sum[r] = fmaf(bf16_to_f32(x[(long)(r0 + r) * xp + jt]), wf, sum[r]);
        }
    } else {
        for (int j = lane; j < k; j += 32) {
            const float wf = bf16_to_f32(wrow[j]);
#pragma unroll
            for (int r = 0; r < RM; ++r)
                if (r0 + r < m) sum[r] = fmaf(bf16_to_f32(x[(long)(r0 + r) * xp + j]), wf, sum[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < RM; ++r) {
        if (r0 + r < m) {
            float s = sum[r];
#pragma unroll
            for (int mask = 16; mask; mask >>= 1) s += __shfl_xor(s, mask, 32);
            if (lane == 0) yout[(long)(r0 + r) * yp + c] = s;
        }
    }
}

template <int RM, int U, bool VEC8>
int gemm_f32_bf16_pair_vec(RadStream stream, const RadTensor& x, const RadTensor& w0,
                           const RadTensor& w1, const RadTensor& y0, const RadTensor& y1,
                           long n, long k) {
    const long m = x.shape[0];
    return R4D_EMIT(stream, (gemm_nt_f32_bf16_pair_vec_kernel<RM, U, VEC8>),
                    dim3((unsigned)((n + 3) / 4), (unsigned)((m + RM - 1) / RM), 2u), dim3(128), 0,
                    (const uint16_t*)x.data, (const uint16_t*)w0.data, (const uint16_t*)w1.data,
                    (float*)y0.data, (float*)y1.data, x.stride[0], w0.stride[0], w1.stride[0],
                    y0.stride[0], y1.stride[0], (int)m, (int)n, (int)k);
}

int gemm_f32_pair_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 5) return RAD_E_INVAL;
    const auto &x = a->t[0], &w0 = a->t[1], &w1 = a->t[2], &y0 = a->t[3], &y1 = a->t[4];
    const long n = r4d_geti_or(a, "N", 0), k = r4d_geti_or(a, "K", 0);
    if (x.dtype != RAD_BF16 || (w0.dtype != RAD_F32 && w0.dtype != RAD_BF16) ||
        w1.dtype != w0.dtype || y0.dtype != RAD_F32 || y1.dtype != RAD_F32)
        return RAD_E_DTYPE;
    if (!x.data || !w0.data || !w1.data || !y0.data || !y1.data) return RAD_E_INVAL;
    const long max_n = w0.dtype == RAD_BF16 ? 1024 : 512;
    if (n < 1 || n > max_n || k < 1 || k > 16384 || x.rank != 2 || w0.rank != 2 || w1.rank != 2 ||
        y0.rank != 2 || y1.rank != 2 || x.shape[0] < 1 || x.shape[0] > 65535 || x.shape[1] != k ||
        w0.shape[0] != n || w0.shape[1] != k || w1.shape[0] != n || w1.shape[1] != k ||
        y0.shape[0] != x.shape[0] || y0.shape[1] != n || y1.shape[0] != x.shape[0] ||
        y1.shape[1] != n || x.stride[1] != 1 || w0.stride[1] != 1 || w1.stride[1] != 1 ||
        y0.stride[1] != 1 || y1.stride[1] != 1 || x.stride[0] < k || w0.stride[0] < k ||
        w1.stride[0] < k || y0.stride[0] < n || y1.stride[0] < n)
        return RAD_E_SHAPE;
    if (w0.dtype == RAD_BF16) {
        const long rows = x.shape[0];
        const bool vec8 = (((uintptr_t)x.data & 15u) == 0) && (((uintptr_t)w0.data & 15u) == 0) &&
                          (((uintptr_t)w1.data & 15u) == 0) && ((x.stride[0] & 7) == 0) &&
                          ((w0.stride[0] & 7) == 0) && ((w1.stride[0] & 7) == 0);
#if defined(__clang__) || defined(__HIP_DEVICE_COMPILE__)
        /* Row identity (the row-identity checks): with 16-byte rows and K % 32 == 0 (every ERNIE use) every row count takes the
         * chunked WMMA sum (gemm_f32_bf16_kc), so a row's f32 logits (the router's) and kv/score projection (the HCA compressor's, which
         * enters persistent state) do not depend on how many rows share its step. The vector kernels and the plain WMMA chain sum K in other
         * orders; they served 1..8 rows and more than 8 rows before, which made the value depend on the step's row count (R4D_PAIR_VEC_MAX
         * pinned the vector form for every count, at the cost of the tensor cores at prefill). */
        if (vec8 && k >= 32 && (k & 31) == 0) {
            int rc = gemm_f32_bf16_kc(stream, x, w0, y0, n, k);
            if (rc < 0) return rc;
            return gemm_f32_bf16_kc(stream, x, w1, y1, n, k);
        }
#endif
        /* Other layouts: the vector kernels for every row count (they agree with each other bit for bit, emu band_pair). */
        if (vec8) {
            if (rows == 2) return gemm_f32_bf16_pair_vec<2, 2, true>(stream, x, w0, w1, y0, y1, n, k);
            if (rows <= 4) return gemm_f32_bf16_pair_vec<4, 2, true>(stream, x, w0, w1, y0, y1, n, k);
            return gemm_f32_bf16_pair_vec<8, 2, true>(stream, x, w0, w1, y0, y1, n, k);
        }
        return gemm_f32_bf16_pair_vec<8, 1, false>(stream, x, w0, w1, y0, y1, n, k);
    }
    const long rows = x.shape[0];
    auto run_f32 = [&](const RadTensor& w, const RadTensor& y) {
        if (n <= 8) return gemm_f32_tile<4, 1>(stream, x, w, y, n, k);
        if (rows <= 16 || (rows < 64 && n < 128)) return gemm_f32_tile<2, 2, 16>(stream, x, w, y, n, k);
        if (rows < 64) return gemm_f32_tile<8, 1, 8>(stream, x, w, y, n, k);
        return gemm_f32_tile<8, 4>(stream, x, w, y, n, k);
    };
    int rc = run_f32(w0, y0);
    if (rc < 0) return rc;
    return run_f32(w1, y1);
}
// router_sigmoid_topk in one wave per row: p = sigmoid(logit), the top_k largest ranks p + bias
// in descending order, equal ranks to the lower expert, as a serial scan with a strict `>` takes
// them. Lane l holds experts l, l+32, ... as 64-bit keys: the rank's order-preserving bits (-0
// folded to +0) over ~e, so equal ranks order to the lower expert. Round j takes the largest key
// below round j-1's winner and parks it on lane j (top_k <= 32). `total` sums the selected p in
// round order, mult = norm ? factor / (total + 1e-20) : factor, and a weight is p * mult. A NaN
// rank is never selected.
// OWN is a template bound because the keys are registers; a lane-derived trip count would index
// them divergently, which gfx12 lowers to a v_movrels loop.
__device__ __forceinline__ unsigned long long router_rank_key(float rank, int e) {
    uint32_t u = __float_as_uint(rank);
    if (u == 0x80000000u) u = 0;
    const uint32_t k = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return rank == rank ? ((unsigned long long)k << 32) | (uint32_t)~e : 0ull;
}
__device__ __forceinline__ unsigned long long router_wave_max(unsigned long long b) {
    unsigned long long o;
    o = r4d_xor_lane_u64<16>(b);
    b = b > o ? b : o;
    o = r4d_xor_lane_u64<8>(b);
    b = b > o ? b : o;
    o = r4d_xor_lane_u64<4>(b);
    b = b > o ? b : o;
    o = r4d_xor_lane_u64<2>(b);
    b = b > o ? b : o;
    o = r4d_xor_lane_u64<1>(b);
    b = b > o ? b : o;
    return b;
}
// One row in one wave, from any grid: the standalone select and the fused one below run this. A
// lane below `k` returns its slot's expert as written to `ids`.
template <int OWN>
__device__ __forceinline__ int router_sigmoid_row(const float* logits, const float* bias, int* ids,
                                                  float* weights, long lp, long bp, long ip,
                                                  long wp, int r, int ne, int k, int norm,
                                                  float factor, int lane) {
    float p[OWN];
    unsigned long long key[OWN];
#pragma unroll
    for (int i = 0; i < OWN; ++i) {
        const int e = lane + 32 * i, s = e < ne ? e : 0;
        p[i] = 1.0f / (1.0f + expf(-logits[(long)r * lp + s]));
        key[i] = e < ne ? router_rank_key(p[i] + bias[(long)s * bp], e) : 0ull;
    }
    unsigned long long below = ~0ull, mine = 0;
    for (int j = 0; j < k; ++j) {
        unsigned long long best = 0;
#pragma unroll
        for (int i = 0; i < OWN; ++i) {
            const unsigned long long c = key[i] < below ? key[i] : 0ull;
            best = c > best ? c : best;
        }
        best = router_wave_max(best);
        if (lane == j) mine = best;
        below = best;
    }
    const int e = mine ? (int)~(uint32_t)mine : 0;
    float chosen = 0;
#pragma unroll
    for (int i = 0; i < OWN; ++i) {
        const float v = __shfl(p[i], e & 31, 32);
        if ((e >> 5) == i) chosen = v;
    }
    float total = 0;
    for (int j = 0; j < k; ++j)
        total += __int_as_float(__builtin_amdgcn_readlane(__float_as_int(chosen), j));
    const float mult = norm ? factor / (total + 1e-20f) : factor;
    if (lane < k) {
        ids[(long)r * ip + lane] = e;
        weights[(long)r * wp + lane] = chosen * mult;
    }
    return e;
}
// router_sigmoid_group_topk in one wave per row: router_sigmoid_row's keys, but only the experts of
// the `kg` best of `ng` equal groups are selectable. A group ranks by the sum of its two highest
// ranks; lane g holds group g's sum (ng <= 32), and the groups are taken as the experts are, the
// largest first, equal sums to the lower group, a group with no finite pair never. Then the
// selection, the weights and the stores are router_sigmoid_row's.
template <int OWN>
__global__ void __launch_bounds__(128)
    router_group_wave_kernel(const float* logits, const float* bias, int* ids, float* weights,
                             long lp, long bp, long ip, long wp, int m, int ne, int k, int ng,
                             int kg, int norm, float factor) {
    const int lane = threadIdx.x & 31, r = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (r >= m) return;
    const int gs = ne / ng;
    float p[OWN], rank[OWN];
#pragma unroll
    for (int i = 0; i < OWN; ++i) {
        const int e = lane + 32 * i, s = e < ne ? e : 0;
        p[i] = 1.0f / (1.0f + expf(-logits[(long)r * lp + s]));
        rank[i] = e < ne ? p[i] + bias[(long)s * bp] : -INFINITY;
    }
    float group = -INFINITY;
    for (int g = 0; g < ng; ++g) {
        float t0 = -INFINITY, t1 = -INFINITY;
#pragma unroll
        for (int i = 0; i < OWN; ++i) {
            const int e = lane + 32 * i;
            if (e < ne && e / gs == g) {
                const float v = rank[i];
                if (v > t0) {
                    t1 = t0;
                    t0 = v;
                } else if (v > t1)
                    t1 = v;
            }
        }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            const float o0 = __shfl_xor(t0, mask, 32), o1 = __shfl_xor(t1, mask, 32);
            const float hi = fmaxf(t0, o0), lo = fmaxf(fminf(t0, o0), fmaxf(t1, o1));
            t0 = hi;
            t1 = lo;
        }
        if (lane == g) group = t0 + t1;
    }
    unsigned long long gkey = lane < ng && group > -INFINITY ? router_rank_key(group, lane) : 0ull;
    unsigned kept = 0;   // bit g: group g is kept
    unsigned long long below = ~0ull;
    for (int j = 0; j < kg; ++j) {
        const unsigned long long best = router_wave_max(gkey < below ? gkey : 0ull);
        if (best) kept |= 1u << (int)~(uint32_t)best;
        below = best;
    }
    unsigned long long key[OWN];
#pragma unroll
    for (int i = 0; i < OWN; ++i) {
        const int e = lane + 32 * i;
        key[i] = e < ne && ((kept >> (e / gs)) & 1u) ? router_rank_key(rank[i], e) : 0ull;
    }
    unsigned long long mine = 0;
    below = ~0ull;
    for (int j = 0; j < k; ++j) {
        unsigned long long best = 0;
#pragma unroll
        for (int i = 0; i < OWN; ++i) {
            const unsigned long long c = key[i] < below ? key[i] : 0ull;
            best = c > best ? c : best;
        }
        best = router_wave_max(best);
        if (lane == j) mine = best;
        below = best;
    }
    const int e = mine ? (int)~(uint32_t)mine : 0;
    float chosen = 0;
#pragma unroll
    for (int i = 0; i < OWN; ++i) {
        const float v = __shfl(p[i], e & 31, 32);
        if ((e >> 5) == i) chosen = v;
    }
    float total = 0;
    for (int j = 0; j < k; ++j)
        total += __int_as_float(__builtin_amdgcn_readlane(__float_as_int(chosen), j));
    const float mult = norm ? factor / (total + 1e-20f) : factor;
    if (lane < k) {
        ids[(long)r * ip + lane] = e;
        weights[(long)r * wp + lane] = chosen * mult;
    }
}
template <int OWN, class CH = R4dUnchained>
__global__ void __launch_bounds__(128)
    router_sigmoid_wave_kernel(const float* logits, const float* bias, int* ids, float* weights,
                               long lp, long bp, long ip, long wp, int m, int ne, int k, int norm,
                               float factor) {
    if (CH::on) r4d_chain_wait();
    const int lane = threadIdx.x & 31, r = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (r < m)
        (void)router_sigmoid_row<OWN>(logits, bias, ids, weights, lp, bp, ip, wp, r, ne, k, norm,
                                      factor, lane);
    if (CH::on) r4d_chain_done();
}
// router_sigmoid_topk, then moe_scatter over its ids, then the `fill` of an ungated shared arm's
// gate with 1, in one workgroup. Rows wrap over the NW waves; each leaves its slots in the sort's
// LDS list instead of re-reading `ids`. A slot carries the expert its row wrote, including the
// expert 0 of a row with fewer finite ranks than top_k, so every output equals the three launches'.
// INV also writes the inverse, as moe_scatter_inv does.
template <int OWN, uint32_t NW, bool INV, class CH = R4dUnchained>
__global__ void __launch_bounds__(NW * 32)
    router_sigmoid_scatter_kernel(const float* logits, const float* bias, int* ids, float* weights,
                                  int32_t* sorted, int32_t* off, int32_t* cnt, uint16_t* gate,
                                  int32_t* inv, long lp, long bp, long ip, long wp, long gp, int m,
                                  int ne, int n_local, int k, int norm, float factor, int base,
                                  r4d_moesort::Protect prot) {
    if (CH::on) r4d_chain_wait();
    __shared__ r4d_moesort::ScatterLds<NW> sl;
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    for (int e = threadIdx.x; e < n_local; e += (int)(NW * 32)) sl.cnt[e] = 0;
    for (int r = wave; r < m; r += (int)NW) {
        const int e = router_sigmoid_row<OWN>(logits, bias, ids, weights, lp, bp, ip, wp, r, ne, k,
                                              norm, factor, lane) -
                      base;
        if (lane < k)
            sl.e[r * k + lane] =
                e >= 0 && e < n_local ? r4d_moesort::sort_place(e, (uint32_t)n_local, prot) : -1;
        if (gate && !lane) gate[(long)r * gp] = f32_to_bf16(1.0f);
    }
    lds_barrier();
    r4d_moesort::scatter_small_place<NW, INV>(sorted, off, cnt, (uint32_t)(m * k),
                                              (uint32_t)n_local, sl, inv);
    if (CH::on) r4d_chain_done();
}
int router_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 4) return RAD_E_INVAL;
    const auto &lg = a->t[0], &bias = a->t[1], &ids = a->t[2], &w = a->t[3];
    const long ne = r4d_geti_or(a, "n_expert", 0), k = r4d_geti_or(a, "top_k", 0);
    const float factor = (float)r4d_getf(a, "route_scale", 0);
    if (lg.dtype != RAD_F32 || bias.dtype != RAD_F32 || ids.dtype != RAD_I32 || w.dtype != RAD_F32)
        return RAD_E_DTYPE;
    if (ne < 1 || ne > 512 || k < 1 || k > 32 || k > ne || !std::isfinite(factor) || factor <= 0 ||
        !lg.data || !bias.data || !ids.data || !w.data || lg.rank != 2 || lg.shape[0] < 1 ||
        lg.shape[0] > 65535 || lg.shape[1] != ne || bias.rank != 1 || bias.shape[0] != ne ||
        ids.rank != 2 || w.rank != 2 || ids.shape[0] != lg.shape[0] || w.shape[0] != lg.shape[0] ||
        ids.shape[1] != k || w.shape[1] != k || lg.stride[1] != 1 || ids.stride[1] != 1 ||
        w.stride[1] != 1 || lg.stride[0] < ne || ids.stride[0] < k || w.stride[0] < k ||
        bias.stride[0] < 1)
        return RAD_E_SHAPE;
    const int norm = (int)(r4d_geti_or(a, "norm", 0) != 0);
    const bool chain = r4d_geti_or(a, "chain", 0) != 0;
    const dim3 grid((unsigned)((lg.shape[0] + 3) / 4));
    if (ne <= 9 * 32)
        return R4D_EMIT(stream,
                        (chain ? router_sigmoid_wave_kernel<9, r4dchain>
                               : router_sigmoid_wave_kernel<9, R4dUnchained>),
                        grid, dim3(128), 0,
                        (const float*)lg.data, (const float*)bias.data, (int*)ids.data,
                        (float*)w.data, lg.stride[0], bias.stride[0], ids.stride[0], w.stride[0],
                        (int)lg.shape[0], (int)ne, (int)k, norm, factor);
    return R4D_EMIT(stream,
                    (chain ? router_sigmoid_wave_kernel<16, r4dchain>
                           : router_sigmoid_wave_kernel<16, R4dUnchained>),
                    grid, dim3(128), 0,
                    (const float*)lg.data, (const float*)bias.data, (int*)ids.data, (float*)w.data,
                    lg.stride[0], bias.stride[0], ids.stride[0], w.stride[0], (int)lg.shape[0],
                    (int)ne, (int)k, norm, factor);
}
int router_group_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 4) return RAD_E_INVAL;
    const auto &lg = a->t[0], &bias = a->t[1], &ids = a->t[2], &w = a->t[3];
    const long ne = r4d_geti_or(a, "n_expert", 0), k = r4d_geti_or(a, "top_k", 0),
               ng = r4d_geti_or(a, "n_group", 0), kg = r4d_geti_or(a, "topk_group", 0);
    const float factor = (float)r4d_getf(a, "route_scale", 0);
    if (lg.dtype != RAD_F32 || bias.dtype != RAD_F32 || ids.dtype != RAD_I32 || w.dtype != RAD_F32)
        return RAD_E_DTYPE;
    if (ne < 1 || ne > 512 || k < 1 || k > 32 || ng < 1 || ng > 32 || ne % ng || ne / ng < 2 ||
        kg < 1 || kg > ng || k > kg * (ne / ng) || !std::isfinite(factor) || factor <= 0 ||
        !lg.data || !bias.data || !ids.data || !w.data || lg.rank != 2 || lg.shape[0] < 1 ||
        lg.shape[0] > 65535 || lg.shape[1] != ne || bias.rank != 1 || bias.shape[0] != ne ||
        ids.rank != 2 || w.rank != 2 || ids.shape[0] != lg.shape[0] || w.shape[0] != lg.shape[0] ||
        ids.shape[1] != k || w.shape[1] != k || lg.stride[1] != 1 || ids.stride[1] != 1 ||
        w.stride[1] != 1 || lg.stride[0] < ne || ids.stride[0] < k || w.stride[0] < k ||
        bias.stride[0] < 1)
        return RAD_E_SHAPE;
    const int norm = (int)(r4d_geti_or(a, "norm", 0) != 0);
    const dim3 grid((unsigned)((lg.shape[0] + 3) / 4));
    if (ne <= 9 * 32)
        return R4D_EMIT(stream, router_group_wave_kernel<9>, grid, dim3(128), 0,
                        (const float*)lg.data, (const float*)bias.data, (int*)ids.data,
                        (float*)w.data, lg.stride[0], bias.stride[0], ids.stride[0], w.stride[0],
                        (int)lg.shape[0], (int)ne, (int)k, (int)ng, (int)kg, norm, factor);
    return R4D_EMIT(stream, router_group_wave_kernel<16>, grid, dim3(128), 0,
                    (const float*)lg.data, (const float*)bias.data, (int*)ids.data, (float*)w.data,
                    lg.stride[0], bias.stride[0], ids.stride[0], w.stride[0], (int)lg.shape[0],
                    (int)ne, (int)k, (int)ng, (int)kg, norm, factor);
}
/* Operands: router_sigmoid_topk's four, moe_scatter's three outputs, then the optional gate and the
 * optional inverse. The sort is the single-workgroup form, so M * top_k must fit it; a larger pass
 * is refused rather than served at another cost. */
int router_scatter_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 7) return RAD_E_INVAL;
    const auto &lg = a->t[0], &bias = a->t[1], &ids = a->t[2], &w = a->t[3];
    const RadTensor *sorted = &a->t[4], *off = &a->t[5], *cnt = &a->t[6];
    const RadTensor* gate = a->n_t > 7 ? r4d_opt(a, 7) : nullptr;
    const RadTensor* inv = a->n_t > 8 ? r4d_opt(a, 8) : nullptr;
    const long ne = r4d_geti_or(a, "n_expert", 0), k = r4d_geti_or(a, "top_k", 0);
    const long nl = r4d_geti_or(a, "n_local", ne), base = r4d_geti_or(a, "expert_base", 0);
    const float factor = (float)r4d_getf(a, "route_scale", 0);
    if (lg.dtype != RAD_F32 || bias.dtype != RAD_F32 || ids.dtype != RAD_I32 ||
        w.dtype != RAD_F32 || sorted->dtype != RAD_I32 || off->dtype != RAD_I32 ||
        cnt->dtype != RAD_I32 || (gate && gate->dtype != RAD_BF16) ||
        (inv && inv->dtype != RAD_I32))
        return RAD_E_DTYPE;
    if (ne < 1 || ne > 512 || k < 1 || k > 32 || k > ne || !std::isfinite(factor) || factor <= 0 ||
        !lg.data || !bias.data || !ids.data || !w.data || !sorted->data || !off->data ||
        !cnt->data || lg.rank != 2 || lg.shape[0] < 1 || lg.shape[1] != ne || bias.rank != 1 ||
        bias.shape[0] != ne || ids.rank != 2 || w.rank != 2 || ids.shape[0] != lg.shape[0] ||
        w.shape[0] != lg.shape[0] || ids.shape[1] != k || w.shape[1] != k || lg.stride[1] != 1 ||
        ids.stride[1] != 1 || w.stride[1] != 1 || lg.stride[0] < ne || ids.stride[0] < k ||
        w.stride[0] < k || bias.stride[0] < 1)
        return RAD_E_SHAPE;
    const long m = lg.shape[0], T = m * k;
    if (T > (long)r4d_moesort::kScatterSmall || nl < 1 || nl > ne || base < 0 || base + nl > ne ||
        !r4d_contig(sorted) || !r4d_contig(off) || !r4d_contig(cnt) || r4d_numel(sorted) < T ||
        r4d_numel(off) < nl + 1 || r4d_numel(cnt) < nl ||
        (inv && (!r4d_contig(inv) || r4d_numel(inv) < T)))
        return RAD_E_SHAPE;
    long long grows = 0;
    const long long gp = gate ? r4d_rows_pitch(gate, 1, &grows) : 0;
    if (gate && (gp < 1 || grows < m)) return RAD_E_SHAPE;
    r4d_moesort::Protect prot;
    const int prc = r4d_moesort::protect_of(a, nl, &prot);
    if (prc != RAD_OK) return prc;
    const int norm = (int)(r4d_geti_or(a, "norm", 0) != 0);
    // A wave a row up to 32 rows, as router_topk_scatter: the rows a wave serves run back to back.
    const int nw = m <= 8 ? 8 : m <= 16 ? 16 : 32;
    const bool chain = r4d_geti_or(a, "chain", 0) != 0;
#define R4D_RSS_I(OWN, NW, INV)                                                                    \
    return R4D_EMIT(stream,                                                                        \
                    (chain ? router_sigmoid_scatter_kernel<OWN, NW, INV, r4dchain>                 \
                           : router_sigmoid_scatter_kernel<OWN, NW, INV, R4dUnchained>),           \
                    dim3(1), dim3((NW) * 32), 0, (const float*)lg.data, (const float*)bias.data,   \
                    (int*)ids.data, (float*)w.data, (int32_t*)sorted->data, (int32_t*)off->data,   \
                    (int32_t*)cnt->data, gate ? (uint16_t*)gate->data : (uint16_t*)nullptr,        \
                    inv ? (int32_t*)inv->data : (int32_t*)nullptr, lg.stride[0], bias.stride[0],   \
                    ids.stride[0], w.stride[0], (long)gp, (int)m, (int)ne, (int)nl, (int)k, norm,  \
                    factor, (int)base, prot)
#define R4D_RSS(OWN, NW)                                                                           \
    do {                                                                                           \
        if (inv) R4D_RSS_I(OWN, NW, true);                                                         \
        R4D_RSS_I(OWN, NW, false);                                                                 \
    } while (0)
    if (ne <= 9 * 32) {
        if (nw == 8) R4D_RSS(9, 8u);
        if (nw == 16) R4D_RSS(9, 16u);
        R4D_RSS(9, 32u);
    }
    if (nw == 8) R4D_RSS(16, 8u);
    if (nw == 16) R4D_RSS(16, 16u);
    R4D_RSS(16, 32u);
#undef R4D_RSS
#undef R4D_RSS_I
}
__global__ void swiglu_clamped_kernel(const uint16_t* gu, uint16_t* y, long gp, long yp, int n,
                                      float limit) {
    const long r = blockIdx.x;
    const int i = blockIdx.y * 256 + threadIdx.x;
    if (i >= n) return;
    const float g = fminf(bf16_to_f32(gu[r * gp + i]), limit);
    const float u = fminf(fmaxf(bf16_to_f32(gu[r * gp + n + i]), -limit), limit);
    y[r * yp + i] = f32_to_bf16((g / (1.0f + expf(-g))) * u);
}
// swiglu_step: the SiLU first, its value clipped above (Ling-3.0; vLLM's SwigluStepAndMul).
__global__ void swiglu_step_kernel(const uint16_t* gu, uint16_t* y, long gp, long yp, int n,
                                   float limit) {
    const long r = blockIdx.x;
    const int i = blockIdx.y * 256 + threadIdx.x;
    if (i >= n) return;
    const float x = bf16_to_f32(gu[r * gp + i]);
    const float g = fminf(x / (1.0f + expf(-x)), limit);
    const float u = fminf(fmaxf(bf16_to_f32(gu[r * gp + n + i]), -limit), limit);
    y[r * yp + i] = f32_to_bf16(g * u);
}
int clamped_body(const RadArgs* a, RadStream stream, bool step = false) {
    if (!a || a->n_t < 2) return RAD_E_INVAL;
    const auto &gu = a->t[0], &y = a->t[1];
    const long n = r4d_geti_or(a, "n", 0);
    const float limit = (float)r4d_getf(a, "limit", 0);
    if (gu.dtype != RAD_BF16 || y.dtype != RAD_BF16) return RAD_E_DTYPE;
    if (n < 1 || n > INT_MAX / 2 || !std::isfinite(limit) || limit <= 0 || !gu.data || !y.data ||
        gu.rank != 2 || y.rank != 2 || y.shape[0] < 1 || y.shape[0] > 65535 ||
        gu.shape[0] != y.shape[0] || gu.shape[1] != 2 * n || y.shape[1] != n || gu.stride[1] != 1 ||
        y.stride[1] != 1 || gu.stride[0] < 2 * n || y.stride[0] < n)
        return RAD_E_SHAPE;
    if (step)
        return R4D_EMIT(stream, swiglu_step_kernel, dim3(y.shape[0], (n + 255) / 256), dim3(256), 0,
                        (const uint16_t*)gu.data, (uint16_t*)y.data, gu.stride[0], y.stride[0],
                        (int)n, limit);
    return R4D_EMIT(stream, swiglu_clamped_kernel, dim3(y.shape[0], (n + 255) / 256), dim3(256), 0,
                    (const uint16_t*)gu.data, (uint16_t*)y.data, gu.stride[0], y.stride[0], (int)n,
                    limit);
}
// swiglu_clamped, then quant_act_fp8 over its product, in one pass: a wave per (row, 128 columns),
// four columns a lane, as the quantiser maps them. Each element is swiglu_clamped_kernel's
// expression and each group quant_act_fp8_kernel's. The product is rounded to bf16 and read back
// before the absmax because the separate quantiser reads the stored bf16 bytes.
__global__ void __launch_bounds__(256)
    swiglu_clamped_quant_kernel(const uint16_t* gu, uint16_t* y, uint8_t* q, float* s, long gp,
                                long yp, long qp, long sp, int n, int nkb, float limit) {
    const long r = blockIdx.x;
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    for (int kb = wave; kb < nkb; kb += (int)(blockDim.x >> 5)) {
        const int c0 = kb * 128 + lane * 4;
        float v[4], m = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = c0 + j;
            v[j] = 0.0f;
            if (c < n) {
                const float g = fminf(bf16_to_f32(gu[r * gp + c]), limit);
                const float u = fminf(fmaxf(bf16_to_f32(gu[r * gp + n + c]), -limit), limit);
                const uint16_t h = f32_to_bf16((g / (1.0f + expf(-g))) * u);
                if (y) y[r * yp + c] = h;
                v[j] = bf16_to_f32(h);
            }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) m = fmaxf(m, fabsf(v[j]));
        m = r4d_wave_max(m);
        const float sc = m > 0.0f ? m * (1.0f / 448.0f) : 1.0f;
        if (!lane) s[r * sp + kb] = sc;
        const float inv = 1.0f / sc;
        int pk = 0;
        pk = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(v[0] * inv, -448.0f), 448.0f),
                                             fminf(fmaxf(v[1] * inv, -448.0f), 448.0f), pk, false);
        pk = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(v[2] * inv, -448.0f), 448.0f),
                                             fminf(fmaxf(v[3] * inv, -448.0f), 448.0f), pk, true);
        if (c0 < n) *(uint32_t*)(q + r * qp + c0) = (uint32_t)pk;
    }
}
/* Operands: gate_up [M, 2n], q [M, n] E4M3, scale [M, ceil(n/128)] f32, out_bf16 [M, n] optional.
 * n a multiple of 4 and group 128, as quant_act_fp8 requires. */
int clamped_quant_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 3) return RAD_E_INVAL;
    const auto &gu = a->t[0], &q = a->t[1], &sc = a->t[2];
    const RadTensor* y = a->n_t > 3 ? r4d_opt(a, 3) : nullptr;
    const long n = r4d_geti_or(a, "n", 0), nkb = (n + 127) / 128;
    const float limit = (float)r4d_getf(a, "limit", 0);
    if (gu.dtype != RAD_BF16 || q.dtype != RAD_F8E4M3 || sc.dtype != RAD_F32 ||
        (y && y->dtype != RAD_BF16))
        return RAD_E_DTYPE;
    if (r4d_geti_or(a, "group", 128) != 128 || n < 4 || n % 4 || n > INT_MAX / 2 ||
        !std::isfinite(limit) || limit <= 0 || !gu.data || !q.data || !sc.data || gu.rank != 2 ||
        q.rank != 2 || sc.rank != 2 || gu.shape[0] < 1 || gu.shape[0] > 65535 ||
        gu.shape[1] != 2 * n || q.shape[0] != gu.shape[0] || q.shape[1] != n ||
        sc.shape[0] != gu.shape[0] || sc.shape[1] != nkb || gu.stride[1] != 1 || q.stride[1] != 1 ||
        sc.stride[1] != 1 || gu.stride[0] < 2 * n || q.stride[0] < n || sc.stride[0] < nkb ||
        (y && (y->rank != 2 || y->shape[0] != gu.shape[0] || y->shape[1] != n ||
               y->stride[1] != 1 || y->stride[0] < n)))
        return RAD_E_SHAPE;
    // A wave a group, at most eight: a row of GLM's TP8 expert width is two groups.
    const unsigned thr = (unsigned)(nkb < 8 ? nkb : 8) * 32u;
    return R4D_EMIT(stream, swiglu_clamped_quant_kernel, dim3((unsigned)gu.shape[0]), dim3(thr), 0,
                    (const uint16_t*)gu.data, y ? (uint16_t*)y->data : (uint16_t*)nullptr,
                    (uint8_t*)q.data, (float*)sc.data, gu.stride[0], y ? (long)y->stride[0] : 0L,
                    q.stride[0], sc.stride[0], (int)n, (int)nkb, limit);
}
}
extern "C" int r4d_rad_swiglu_clamped(const RadArgs* a, RadStream stream) {
    return clamped_body(a, stream);
}
extern "C" int r4d_rad_swiglu_step(const RadArgs* a, RadStream stream) {
    return clamped_body(a, stream, true);
}
extern "C" int r4d_rad_router_sigmoid_group_topk(const RadArgs* a, RadStream stream) {
    return router_group_body(a, stream);
}
extern "C" int r4d_rad_gemm_nt_f32(const RadArgs* a, RadStream stream) {
    return gemm_f32_body(a, stream);
}
extern "C" int r4d_rad_gemm_nt_f32_pair(const RadArgs* a, RadStream stream) {
    return gemm_f32_pair_body(a, stream);
}
extern "C" int r4d_rad_router_sigmoid_topk(const RadArgs* a, RadStream stream) {
    return router_body(a, stream);
}
extern "C" int r4d_rad_router_sigmoid_topk_scatter(const RadArgs* a, RadStream stream) {
    return router_scatter_body(a, stream);
}
