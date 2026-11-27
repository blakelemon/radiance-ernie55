/* Canonical E4M3 weights with exact F32 block scales: native FP8 WMMA and F32 scale folding, no
 * checkpoint scale narrowing. The legacy dense form computes one 16x64 output tile a block; the
 * dense decode and prefill forms below it produce the same bytes faster, and the grouped form runs
 * expert-major over the same per-element sequence. */
#include <hip/hip_runtime.h>
#include "r4d_args.h"
#include "r4d_emit.h"
#include "r4d_common.h"
#include "r4d_fp8_wmma.h"
#include "r4d_launch.h"
#include "r4d_ernie_pins.h"
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace {
struct Meta {
    long ar, asr, yr, wr, sr, wstep[2], sstep[2];
    int T, N, K, experts, topk, sorted_a;
};
__device__ const uint8_t* entry(const void* p, long step, int e) {
    return step ? (const uint8_t*)p + step * e : ((const uint8_t* const*)p)[e];
}
/* The legacy dense form: a block per 16 rows by 64 columns, the operands staged through LDS a
 * dword at a time, so it serves any row pitch and alignment. */
__global__ void fp8_f32scale(const uint8_t* a, const float* as, const uint8_t* w, const float* ws,
                             uint16_t* y, Meta m) {
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, n0 = blockIdx.x * 64;
    __shared__ uint8_t qa[16 * 136], wb[64 * 136];
    __shared__ int source[16];
    const int first = (int)blockIdx.y * 16, end = min(first + 16, m.T);
    if (tid < 16) {
        const int r = first + tid;
        source[tid] = r < end ? r : -1;
    }
    __syncthreads();
    auto acc = r4d_fp8::acc_zero();
    for (int k0 = 0; k0 < m.K; k0 += 128) {
        for (int j = tid; j < 16 * 32; j += 128) {
            const int r = j / 32, c = (j % 32) * 4, src = source[r];
            uint32_t v = 0;
            if (src >= 0) {
                const auto* p = a + (long)src * m.ar + k0 + c;
                v = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
                    ((uint32_t)p[3] << 24);
            }
            *(uint32_t*)(qa + r * 136 + c) = v;
        }
        for (int j = tid; j < 64 * 32; j += 128) {
            const int r = j / 32, c = (j % 32) * 4;
            const auto* p = w + (long)(n0 + r) * m.wr + k0 + c;
            *(uint32_t*)(wb + r * 136 + c) = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                                             ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
        }
        __syncthreads();
        auto raw = r4d_fp8::acc_zero();
#pragma unroll
        for (int k = 0; k < 128; k += 16)
            raw = r4d_fp8::mma(r4d_fp8::frag<128>(qa, 0, k, lane),
                               r4d_fp8::frag<128>(wb, wave * 16, k, lane), raw);
        const float weight_scale = ws[(long)(n0 / 128) * m.sr + k0 / 128];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int src = source[(lane >> 4) * 8 + j];
            const float scale = src >= 0 ? as[(long)src * m.asr + k0 / 128] * weight_scale : 0;
            acc[j] += raw[j] * scale;
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int r = first + (lane >> 4) * 8 + j, c = n0 + wave * 16 + (lane & 15);
        if (r < end) y[(long)r * m.yr + c] = f32_to_bf16(acc[j]);
    }
}
/* The grouped form expert-major: a block per (column span, expert) takes every row of its
 * expert in 16-row tiles, so an expert slice is read from memory once however its rows fall in
 * the sorted order (16-row tiles of the sorted order read it once per tile its run spans). Each
 * element keeps the sequence above: per 128 of K in order a fresh chain of eight WMMAs, folded as
 * raw * (activation scale * weight scale). A block walks ColTiles column tiles of 64, each over K
 * in stages of StageK, as one stage sequence with the next stage loaded into registers while the
 * current one computes from LDS. Block row `experts` zeroes the dropped rows past
 * offsets[experts]. */
typedef unsigned f8_u4 __attribute__((ext_vector_type(4)));
template <int StageK>
struct F8Stage {
    static constexpr int KB = StageK / 128;
    f8_u4 a[StageK / 128], w[StageK / 32];
    float wsc[KB], sa;
};
template <int StageK, int ColTiles>
__global__ __launch_bounds__(128) void fp8_f32scale_experts(const uint8_t* a, const float* as,
                                                            const void* w0, const void* s0,
                                                            const void* w1, const void* s1,
                                                            const int* sorted, const int* offsets,
                                                            uint16_t* y, Meta m) {
    using St = F8Stage<StageK>;
    constexpr int KB = St::KB, P = StageK + 8;
    __shared__ __align__(16) uint8_t qa[16 * P], wb[64 * P];
    __shared__ float sa[16 * KB];
    __shared__ int source[16];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, e = blockIdx.y,
              colb = blockIdx.x * 64 * ColTiles;
    if (e == m.experts) {
        const int r0 = offsets[m.experts];
        for (long i = tid; i < (long)(m.T - r0) * 64 * ColTiles; i += 128) {
            const int c = colb + (int)(i % (64 * ColTiles));
            if (c < m.N) y[(long)(r0 + i / (64 * ColTiles)) * m.yr + c] = 0;
        }
        return;
    }
    const int lo = offsets[e], hi = offsets[e + 1];
    if (lo >= hi) return;
    const int parity = e & 1;
    const auto* w = entry(parity ? w1 : w0, m.wstep[parity], e / 2);
    const auto* ws = (const float*)entry(parity ? s1 : s0, m.sstep[parity], e / 2);
    const int ksn = (m.K + StageK - 1) / StageK, tiles = min(ColTiles, (m.N - colb + 63) / 64),
              S = tiles * ksn;
    const int rowbase = (lane >> 4) * 8;
    for (int r0 = lo; r0 < hi; r0 += 16) {
        __syncthreads();
        if (tid < 16) {
            const int r = r0 + tid, flat = r < hi ? sorted[r] : -1;
            source[tid] = r < hi && flat >= 0 ? (m.sorted_a ? r : flat / m.topk) : -1;
        }
        __syncthreads();
        bool rv[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) rv[j] = source[rowbase + j] >= 0;
        auto load = [&](int s, St& R) {
            const int col0 = colb + (s / ksn) * 64, k0 = (s % ksn) * StageK;
#pragma unroll
            for (int i = 0; i < StageK / 128; ++i) {
                const int j = tid + i * 128, r = j / (StageK / 16), c = (j % (StageK / 16)) * 16,
                          src = source[r];
                R.a[i] = f8_u4{ 0, 0, 0, 0 };
                if (src >= 0 && k0 + c < m.K)
                    __builtin_memcpy(&R.a[i], a + (long)src * m.ar + k0 + c, 16);
            }
#pragma unroll
            for (int i = 0; i < StageK / 32; ++i) {
                const int j = tid + i * 128, r = j / (StageK / 16), c = (j % (StageK / 16)) * 16;
                R.w[i] = f8_u4{ 0, 0, 0, 0 };
                if (k0 + c < m.K)
                    __builtin_memcpy(&R.w[i], w + (long)(col0 + r) * m.wr + k0 + c, 16);
            }
#pragma unroll
            for (int kb = 0; kb < KB; ++kb)
                R.wsc[kb] =
                    k0 + kb * 128 < m.K ? ws[(long)(col0 / 128) * m.sr + (k0 + kb * 128) / 128] : 0;
            if (tid < 16 * KB) {
                const int src = source[tid / KB], kb = k0 + (tid % KB) * 128;
                R.sa = src >= 0 && kb < m.K ? as[(long)src * m.asr + kb / 128] : 0;
            }
        };
        auto store = [&](const St& R) {
#pragma unroll
            for (int i = 0; i < StageK / 128; ++i) {
                const int j = tid + i * 128, r = j / (StageK / 16), c = (j % (StageK / 16)) * 16;
                *reinterpret_cast<f8_u4*>(qa + r * P + c) = R.a[i];
            }
#pragma unroll
            for (int i = 0; i < StageK / 32; ++i) {
                const int j = tid + i * 128, r = j / (StageK / 16), c = (j % (StageK / 16)) * 16;
                *reinterpret_cast<f8_u4*>(wb + r * P + c) = R.w[i];
            }
            if (tid < 16 * KB) sa[tid] = R.sa;
        };
        auto acc = r4d_fp8::acc_zero();
        St R;
        load(0, R);
        for (int s = 0; s < S; ++s) {
            store(R);
            float wsc[KB];
#pragma unroll
            for (int kb = 0; kb < KB; ++kb) wsc[kb] = R.wsc[kb];
            __syncthreads();
            if (s + 1 < S) load(s + 1, R);
            const int k0 = (s % ksn) * StageK;
#pragma unroll
            for (int kb = 0; kb < KB; ++kb) {
                if (k0 + kb * 128 >= m.K) break;
                auto raw = r4d_fp8::acc_zero();
#pragma unroll
                for (int k = kb * 128; k < (kb + 1) * 128; k += 16)
                    raw = r4d_fp8::mma(r4d_fp8::frag<StageK>(qa, 0, k, lane),
                                       r4d_fp8::frag<StageK>(wb, wave * 16, k, lane), raw);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float scale = rv[j] ? sa[(rowbase + j) * KB + kb] * wsc[kb] : 0;
                    acc[j] += raw[j] * scale;
                }
            }
            if (s % ksn == ksn - 1) {
                const int c = colb + (s / ksn) * 64 + wave * 16 + (lane & 15);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int r = r0 + rowbase + j;
                    if (r < hi) y[(long)r * m.yr + c] = f32_to_bf16(acc[j]);
                }
                acc = r4d_fp8::acc_zero();
            }
            __syncthreads();
        }
    }
}
/* The small-M grouped form at GLM's MTP gate_up slice, with fp8_f32scale_experts' per-element
 * sequence and so its bytes. A draft pass routes one token to eight experts, and the four column
 * tiles a block of the form above leave sixteen blocks, each a chain of 64 stages of 256 K. Here a
 * block takes one column tile of the y-th touched expert (the grid has min(experts, rows) + 1
 * rows, as at the MXFP4 small-M form), in stages of 512 K, with loads unconditional from clamped
 * addresses and masked at the store, barriers that order LDS only, and a 128 of K's eight fragment
 * pairs loaded before its WMMA chain. */
typedef unsigned f8_g16 __attribute__((ext_vector_type(4), aligned(1)));
__device__ __forceinline__ f8_u4 f8_ldg16(const uint8_t* p) {
    return *(const __attribute__((address_space(1))) f8_g16*)p;
}
__device__ __forceinline__ float f8_ldgf(const float* p) {
    return *(const __attribute__((address_space(1))) float*)p;
}
template <int StageK>
struct F8Small {
    static constexpr int Threads = 128, KB = StageK / 128, P = StageK + 8;
    static constexpr int AChunks = StageK / Threads, WChunks = 64 * StageK / 16 / Threads;
    f8_u4 a[AChunks], w[WChunks];
    float wsc[KB], sa;
    uint32_t ok;
};
template <int StageK>
__global__ __launch_bounds__(128) void fp8_f32scale_small(const uint8_t* a, const float* as,
                                                          const void* w0, const void* s0,
                                                          const void* w1, const void* s1,
                                                          const int* sorted, const int* offsets,
                                                          uint16_t* y, Meta m) {
    using St = F8Small<StageK>;
    constexpr int KB = St::KB, P = St::P, Threads = St::Threads;
    static_assert(St::AChunks <= 8 && St::WChunks <= 16 && KB <= 4 && 16 * KB <= Threads &&
                      64 * P >= 4 * 513,
                  "stage chunks must fit the load mask and the offsets the weight tile");
    __shared__ __align__(16) uint8_t qa[16 * P], wb[64 * P];
    __shared__ float sa[16 * KB];
    __shared__ int source[16];
    __shared__ int pick;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, col0 = blockIdx.x * 64;
    if ((int)blockIdx.y == (int)gridDim.y - 1) {
        const int r0 = offsets[m.experts];
        for (long i = tid; i < (long)(m.T - r0) * 64; i += Threads) {
            const int c = col0 + (int)(i % 64);
            if (c < m.N) y[(long)(r0 + i / 64) * m.yr + c] = 0;
        }
        return;
    }
    {
        // The offsets in the weight tile, whose bytes are free until the first stage.
        int* offs = reinterpret_cast<int*>(wb);
#pragma unroll
        for (int q = 0; q < 5; ++q) {
            const int x = tid + q * Threads;
            if (x <= m.experts) offs[x] = offsets[x];
        }
        lds_barrier();
        if (wave == 0) {
            int want = blockIdx.y, found = -1;
            for (int base = 0; base < m.experts; base += 32) {
                const int x = base + lane;
                const bool live = x < m.experts && offs[x + 1] > offs[x];
                const uint32_t mask = __builtin_amdgcn_ballot_w32(live);
                const int n = __builtin_popcount(mask);
                if (want < n) {
                    const uint32_t below = mask & ((1u << lane) - 1u);
                    found = base + __builtin_ctz(__builtin_amdgcn_ballot_w32(
                                       live && __builtin_popcount(below) == want));
                    break;
                }
                want -= n;
            }
            if (lane == 0) pick = found;
        }
        lds_barrier();
    }
    const int e = pick;
    if (e < 0) return;
    const int lo = offsets[e], hi = offsets[e + 1];
    const int parity = e & 1;
    const auto* w = entry(parity ? w1 : w0, m.wstep[parity], e / 2);
    const auto* ws = (const float*)entry(parity ? s1 : s0, m.sstep[parity], e / 2);
    const int S = (m.K + StageK - 1) / StageK, rowbase = (lane >> 4) * 8;
    for (int r0 = lo; r0 < hi; r0 += 16) {
        lds_barrier();
        if (tid < 16) {
            const int r = r0 + tid, flat = r < hi ? sorted[r] : -1;
            source[tid] = flat >= 0 ? (m.sorted_a ? r : flat / m.topk) : -1;
        }
        lds_barrier();
        bool rv[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) rv[j] = source[rowbase + j] >= 0;
        auto load = [&](int s, St& L) {
            const int k0 = s * StageK;
            uint32_t ok = 0;
#pragma unroll
            for (int i = 0; i < St::AChunks; ++i) {
                const int j = tid + i * Threads, r = j / (StageK / 16), c = (j % (StageK / 16)) * 16,
                          src = source[r];
                const bool v = src >= 0 && k0 + c < m.K;
                __builtin_memcpy(&L.a[i], a + (v ? (long)src * m.ar + k0 + c : 0), 16);
                ok |= (uint32_t)v << i;
            }
#pragma unroll
            for (int i = 0; i < St::WChunks; ++i) {
                const int j = tid + i * Threads, r = j / (StageK / 16), c = (j % (StageK / 16)) * 16;
                const bool v = k0 + c < m.K;
                L.w[i] = f8_ldg16(w + (v ? (long)(col0 + r) * m.wr + k0 + c : 0));
                ok |= (uint32_t)v << (8 + i);
            }
#pragma unroll
            for (int kb = 0; kb < KB; ++kb) {
                const bool v = k0 + kb * 128 < m.K;
                L.wsc[kb] =
                    f8_ldgf(ws + (v ? (long)(col0 / 128) * m.sr + (k0 + kb * 128) / 128 : 0));
                ok |= (uint32_t)v << (24 + kb);
            }
            {
                const int src = source[(tid / KB) & 15], kb = k0 + (tid % KB) * 128;
                const bool v = tid < 16 * KB && src >= 0 && kb < m.K;
                L.sa = as[v ? (long)src * m.asr + kb / 128 : 0];
                ok |= (uint32_t)v << 28;
            }
            L.ok = ok;
        };
        auto store = [&](const St& L) {
#pragma unroll
            for (int i = 0; i < St::AChunks; ++i) {
                const int j = tid + i * Threads, r = j / (StageK / 16), c = (j % (StageK / 16)) * 16;
                *reinterpret_cast<f8_u4*>(qa + r * P + c) =
                    (L.ok >> i) & 1u ? L.a[i] : f8_u4{ 0, 0, 0, 0 };
            }
#pragma unroll
            for (int i = 0; i < St::WChunks; ++i) {
                const int j = tid + i * Threads, r = j / (StageK / 16), c = (j % (StageK / 16)) * 16;
                *reinterpret_cast<f8_u4*>(wb + r * P + c) =
                    (L.ok >> (8 + i)) & 1u ? L.w[i] : f8_u4{ 0, 0, 0, 0 };
            }
            if (tid < 16 * KB) sa[(tid % KB) * 16 + tid / KB] = (L.ok >> 28) & 1u ? L.sa : 0.f;
        };
        auto acc = r4d_fp8::acc_zero();
        St L;
        load(0, L);
        for (int s = 0; s < S; ++s) {
            store(L);
            float wsc[KB];
#pragma unroll
            for (int kb = 0; kb < KB; ++kb) wsc[kb] = (L.ok >> (24 + kb)) & 1u ? L.wsc[kb] : 0.f;
            lds_barrier();
            if (s + 1 < S) load(s + 1, L);
            const int k0 = s * StageK;
#pragma unroll
            for (int kb = 0; kb < KB; ++kb) {
                if (k0 + kb * 128 >= m.K) break;
                r4d_fp8::i2v fa[8], fb[8];
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    fa[t] = r4d_fp8::frag<StageK>(qa, 0, kb * 128 + 16 * t, lane);
                    fb[t] = r4d_fp8::frag<StageK>(wb, wave * 16, kb * 128 + 16 * t, lane);
                }
                auto raw = r4d_fp8::acc_zero();
#pragma unroll
                for (int t = 0; t < 8; ++t) raw = r4d_fp8::mma(fa[t], fb[t], raw);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float scale = rv[j] ? sa[kb * 16 + rowbase + j] * wsc[kb] : 0;
                    acc[j] += raw[j] * scale;
                }
            }
            lds_barrier();
        }
        const int c = col0 + wave * 16 + (lane & 15);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int r = r0 + rowbase + j;
            if (r < hi) y[(long)r * m.yr + c] = f32_to_bf16(acc[j]);
        }
    }
}
// Both forms below keep fp8_f32scale's arithmetic for every output: per 128-K block the same
// eight-mma chain from zero over the same fragments (the bytes at k + 8*(lane/16) of the block's
// row segment, which the legacy form copies through LDS unchanged), then acc += raw * (a_scale *
// w_scale) in ascending block order. A WMMA output depends only on its own A row, B column and
// C, so which rows and columns share a tile does not change it.

// Decode: one wave C adjacent 16-column tiles, S waves of a block on S consecutive K blocks at
// once, the fragments loaded straight from global memory; R row tiles share each weight fragment
// and C column tiles each activation fragment. Each wave scales its own partial products (the
// a_scale x w_scale product and the multiply are the legacy fold's own two roundings) into LDS;
// then every wave folds a share of the outputs, adding the round's products in block order to the
// output's running sum, which lives in LDS between rounds. The block's activation scales are
// staged in LDS when they fit (`stage`).
template <int R, int C>
__global__ void fp8_f32scale_decode(const uint8_t* a, const float* as, const uint8_t* w,
                                    const float* ws, uint16_t* y, Meta m, int stage) {
    constexpr int SL = R * C * 8;
    R4D_DYN_LDS(float, lds);
    const int lane = threadIdx.x & 31, s = threadIdx.x >> 5, S = blockDim.x >> 5, h = lane >> 4;
    const int c0 = blockIdx.x * 16 * C, first = (int)blockIdx.y * R * 16, KB = m.K / 128;
    float* part = lds;
    float* sums = part + S * SL * 32;
    float* scales = sums + SL * 32;
    if (stage)
        for (int i = threadIdx.x; i < R * 16 * KB; i += blockDim.x) {
            const int r = first + i / KB;
            scales[i] = r < m.T ? as[(long)r * m.asr + i % KB] : 0;
        }
    for (int i = threadIdx.x; i < SL * 32; i += blockDim.x) sums[i] = 0;
    const uint8_t* arow[R];
    bool alive[R];
#pragma unroll
    for (int t = 0; t < R; ++t) {
        const int r = first + t * 16 + (lane & 15);
        alive[t] = r < m.T;
        arow[t] = a + (long)(alive[t] ? r : 0) * m.ar + h * 8;
    }
    __syncthreads();
    for (int k0 = 0; k0 < KB; k0 += S) {
        const int kb = k0 + s;
        if (kb < KB) {
            r4d_fp8::i2v b[C][8];
            float weight_scale[C];
#pragma unroll
            for (int c = 0; c < C; ++c) {
                const uint8_t* wrow =
                    w + (long)(c0 + c * 16 + (lane & 15)) * m.wr + h * 8 + (long)kb * 128;
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    const uint2 v = *(const uint2*)(wrow + q * 16);
                    b[c][q][0] = (int)v.x;
                    b[c][q][1] = (int)v.y;
                }
                weight_scale[c] = ws[(long)((c0 + c * 16) / 128) * m.sr + kb];
            }
#pragma unroll
            for (int t = 0; t < R; ++t) {
                r4d_fp8::i2v f[8];
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    uint2 v = make_uint2(0, 0);
                    if (alive[t]) v = *(const uint2*)(arow[t] + (long)kb * 128 + q * 16);
                    f[q][0] = (int)v.x;
                    f[q][1] = (int)v.y;
                }
                float act[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int rr = t * 16 + h * 8 + j, r = first + rr;
                    act[j] =
                        stage ? scales[rr * KB + kb] : (r < m.T ? as[(long)r * m.asr + kb] : 0);
                }
#pragma unroll
                for (int c = 0; c < C; ++c) {
                    auto raw = r4d_fp8::acc_zero();
#pragma unroll
                    for (int q = 0; q < 8; ++q) raw = r4d_fp8::mma(f[q], b[c][q], raw);
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int r = first + t * 16 + h * 8 + j;
                        const float scale = r < m.T ? act[j] * weight_scale[c] : 0;
                        part[((long)s * SL + (t * C + c) * 8 + j) * 32 + lane] = raw[j] * scale;
                    }
                }
            }
        }
        __syncthreads();
        for (int slot = s; slot < SL; slot += S) {
            float acc = sums[slot * 32 + lane];
            for (int q = 0; q < S && k0 + q < KB; ++q)
                acc += part[((long)q * SL + slot) * 32 + lane];
            sums[slot * 32 + lane] = acc;
        }
        __syncthreads();
    }
    for (int slot = s; slot < SL; slot += S) {
        const int t = slot / (C * 8), c = slot / 8 % C, j = slot % 8,
                  r = first + t * 16 + h * 8 + j;
        if (r < m.T)
            y[(long)r * m.yr + c0 + c * 16 + (lane & 15)] = f32_to_bf16(sums[slot * 32 + lane]);
    }
}

// Fast prefill: 8-wave (256-thread) uint4-vectorized FP8 WMMA kernel over canonical row-major
// E4M3 operands with exact F32 per-128-K block scale folding (bit-identical to fp8_f32scale).
// Waves are arranged 2(M) x 4(N), each computing PTM x PTN 16x16 tiles (PTN = 2 -> BN = 128,
// matching one 128-row weight scale block). W is the WMMA A operand and A is the WMMA B operand,
// so each lane holds 8 consecutive output columns of one token row: one activation scale per M
// tile and a single coalesced 16-byte BF16 store when output row pitch and base are 16B-aligned.
// K is staged in 64-byte halves (two per 128-K scale block, 72-byte bank-conflict-free LDS pitch)
// with register prefetch overlapping the next half's global loads with the current half's WMMAs.
template <int PTM, int PTN>
__global__ __launch_bounds__(256) R4D_WAVES(PTM >= 4 ? 6 : 8) void fp8_f32scale_prefill(
    const uint8_t* __restrict__ a, const float* __restrict__ as,
    const uint8_t* __restrict__ w, const float* __restrict__ ws,
    uint16_t* __restrict__ y, Meta m) {
    constexpr int kPWA = 2, kPWW = 4, kThreads = 256;
    constexpr int BM = kPWA * PTM * 16, BN = kPWW * PTN * 16;
    constexpr int BK = 64, P = BK + 8;
    constexpr int AV = BM * (BK / 16) / kThreads;
    constexpr int WV = BN * (BK / 16) / kThreads;
    static_assert(BN == 128, "block N must span one 128-row weight scale block");
    static_assert(BM * (BK / 16) % kThreads == 0 && BN * (BK / 16) % kThreads == 0,
                  "staged tiles must divide evenly across 256 threads");

    __shared__ __align__(16) uint8_t qa[BM * P], wb[BN * P];
    __shared__ float sa[BM];

    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int wa = wave / kPWW, ww = wave % kPWW;
    const int arow0 = wa * PTM * 16, wrow0 = ww * PTN * 16;

    // L2-friendly block rasterization: group 8 M-tiles and sweep across N.
    const int tiles_n = m.N / BN;
    const int tiles_m = (m.T + BM - 1) / BM;
    const int lin = (int)blockIdx.y * tiles_n + (int)blockIdx.x;
    constexpr int kGroupM = 8;
    const int tiles_per_group = kGroupM * tiles_n;
    const int group_id = lin / tiles_per_group;
    const int first_m = group_id * kGroupM;
    const int gm = min(kGroupM, tiles_m - first_m);
    const int rem = lin % tiles_per_group;
    const int tm = first_m + rem % gm;
    const int tn = rem / gm;
    const int m0 = tm * BM, n0 = tn * BN;

    uint4 av[AV], wv[WV];
    float sa_reg = 0.0f;
    auto load_half = [&](int k0, bool load_sa, int kb_sa) {
#pragma unroll
        for (int i = 0; i < AV; ++i) {
            const int j = tid + i * kThreads, r = j >> 2, c = (j & 3) * 16;
            av[i] = (m0 + r < m.T)
                        ? *(const uint4*)(a + (long)(m0 + r) * m.ar + k0 + c)
                        : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int i = 0; i < WV; ++i) {
            const int j = tid + i * kThreads, r = j >> 2, c = (j & 3) * 16;
            wv[i] = *(const uint4*)(w + (long)(n0 + r) * m.wr + k0 + c);
        }
        if (load_sa && tid < BM) {
            const int r = m0 + tid;
            sa_reg = r < m.T ? as[(long)r * m.asr + kb_sa] : 0.0f;
        }
    };
    auto put16 = [](uint8_t* p, const uint4& v) {
        ((uint2*)p)[0] = make_uint2(v.x, v.y);
        ((uint2*)p)[1] = make_uint2(v.z, v.w);
    };
    auto store_half = [&](bool store_sa) {
#pragma unroll
        for (int i = 0; i < AV; ++i) {
            const int j = tid + i * kThreads;
            put16(qa + (j >> 2) * P + (j & 3) * 16, av[i]);
        }
#pragma unroll
        for (int i = 0; i < WV; ++i) {
            const int j = tid + i * kThreads;
            put16(wb + (j >> 2) * P + (j & 3) * 16, wv[i]);
        }
        if (store_sa && tid < BM) sa[tid] = sa_reg;
    };

    r4d_fp8::f8v acc[PTN][PTM];
#pragma unroll
    for (int i = 0; i < PTN; ++i)
#pragma unroll
        for (int j = 0; j < PTM; ++j) acc[i][j] = r4d_fp8::acc_zero();

    const int KB = m.K / 128;
    load_half(0, true, 0);
    for (int kb = 0; kb < KB; ++kb) {
        r4d_fp8::f8v raw[PTN][PTM];

        // Half 0: k = kb * 128 + [0..63]
        store_half(true);
        lds_barrier();
        load_half(kb * 128 + 64, false, 0);
        const float ws_val = ws[(long)(n0 / 128) * m.sr + kb];
#pragma unroll
        for (int ks = 0; ks < BK; ks += 16) {
            r4d_fp8::i2v wf[PTN], af[PTM];
#pragma unroll
            for (int i = 0; i < PTN; ++i) wf[i] = r4d_fp8::frag<BK>(wb, wrow0 + i * 16, ks, lane);
#pragma unroll
            for (int j = 0; j < PTM; ++j) af[j] = r4d_fp8::frag<BK>(qa, arow0 + j * 16, ks, lane);
#pragma unroll
            for (int i = 0; i < PTN; ++i)
#pragma unroll
                for (int j = 0; j < PTM; ++j)
                    raw[i][j] = r4d_fp8::mma(wf[i], af[j], ks == 0 ? r4d_fp8::acc_zero() : raw[i][j]);
        }
        lds_barrier();

        // Half 1: k = kb * 128 + [64..127]
        store_half(false);
        lds_barrier();
        if (kb + 1 < KB) load_half((kb + 1) * 128, true, kb + 1);
#pragma unroll
        for (int ks = 0; ks < BK; ks += 16) {
            r4d_fp8::i2v wf[PTN], af[PTM];
#pragma unroll
            for (int i = 0; i < PTN; ++i) wf[i] = r4d_fp8::frag<BK>(wb, wrow0 + i * 16, ks, lane);
#pragma unroll
            for (int j = 0; j < PTM; ++j) af[j] = r4d_fp8::frag<BK>(qa, arow0 + j * 16, ks, lane);
#pragma unroll
            for (int i = 0; i < PTN; ++i)
#pragma unroll
                for (int j = 0; j < PTM; ++j) raw[i][j] = r4d_fp8::mma(wf[i], af[j], raw[i][j]);
        }
#pragma unroll
        for (int j = 0; j < PTM; ++j) {
            const float scale = sa[arow0 + j * 16 + (lane & 15)] * ws_val;
#pragma unroll
            for (int i = 0; i < PTN; ++i)
#pragma unroll
                for (int t = 0; t < 8; ++t) acc[i][j][t] += raw[i][j][t] * scale;
        }
        lds_barrier();
    }

    const bool y16 = (((uintptr_t)y & 15u) == 0u) && ((m.yr & 7L) == 0L);
    const int lc = lane & 15, lh = (lane >> 4) * 8;
#pragma unroll
    for (int j = 0; j < PTM; ++j) {
        const int r = m0 + arow0 + j * 16 + lc;
        if (r >= m.T) continue;
        uint16_t* const yrow = y + (long)r * m.yr;
#pragma unroll
        for (int i = 0; i < PTN; ++i) {
            const int c = n0 + wrow0 + i * 16 + lh;
            const auto& v = acc[i][j];
            if (y16) {
                *(uint4*)(yrow + c) = make_uint4(
                    bf16x2_rne(v[0], v[1]), bf16x2_rne(v[2], v[3]),
                    bf16x2_rne(v[4], v[5]), bf16x2_rne(v[6], v[7]));
            } else {
#pragma unroll
                for (int t = 0; t < 8; ++t) yrow[c + t] = f32_to_bf16(v[t]);
            }
        }
    }
}

// ---- Coalesced uint4 matvec kernels for M <= 8 (zero LDS, zero barriers) ------------------------
// 32 lanes x 16 bytes = 512 contiguous columns per pass = 4 scale blocks of 128 (lanes 0..7 share
// kb+0, 8..15 kb+1, 16..23 kb+2, 24..31 kb+3). RB divides 128 and r0 is RB-aligned, so all RB rows
// owned by a wave share one F32 weight scale row.

__device__ __forceinline__ void f8_decode16(const uint4& packed, float (&out)[16]) {
    const uint32_t pw[4] = { packed.x, packed.y, packed.z, packed.w };
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const v2f e01 = r4d_fp8::e4m3_pk<false>(pw[k]);
        const v2f e23 = r4d_fp8::e4m3_pk<true>(pw[k]);
        out[k * 4 + 0] = e01.x;
        out[k * 4 + 1] = e01.y;
        out[k * 4 + 2] = e23.x;
        out[k * 4 + 3] = e23.y;
    }
}

__device__ __forceinline__ void bf16_widen16(const uint4& x0, const uint4& x1, float (&out)[16]) {
    const uint32_t xw[8] = { x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w };
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        out[2 * k]     = __uint_as_float(xw[k] << 16);
        out[2 * k + 1] = __uint_as_float(xw[k] & 0xffff0000u);
    }
}

__device__ __forceinline__ float f8_dot16_stream(const uint4& w_packed, const float (&xv)[16]) {
    const uint32_t pw[4] = { w_packed.x, w_packed.y, w_packed.z, w_packed.w };
    float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const v2f e01 = r4d_fp8::e4m3_pk<false>(pw[k]);
        const v2f e23 = r4d_fp8::e4m3_pk<true>(pw[k]);
        a0 = fmaf(e01.x, xv[k * 4 + 0], a0);
        a1 = fmaf(e01.y, xv[k * 4 + 1], a1);
        a0 = fmaf(e23.x, xv[k * 4 + 2], a0);
        a1 = fmaf(e23.y, xv[k * 4 + 3], a1);
    }
    return a0 + a1;
}

__device__ __forceinline__ float f8_dot16_stream(const uint4& w_packed, const uint4& x0,
                                                const uint4& x1) {
    const uint32_t pw[4] = { w_packed.x, w_packed.y, w_packed.z, w_packed.w };
    const uint32_t xw[8] = { x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w };
    float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const v2f e01 = r4d_fp8::e4m3_pk<false>(pw[k]);
        a0 = fmaf(e01.x, bf16w_lo(xw[2 * k]), a0);
        a1 = fmaf(e01.y, bf16w_hi(xw[2 * k]), a1);
        const v2f e23 = r4d_fp8::e4m3_pk<true>(pw[k]);
        a0 = fmaf(e23.x, bf16w_lo(xw[2 * k + 1]), a0);
        a1 = fmaf(e23.y, bf16w_hi(xw[2 * k + 1]), a1);
    }
    return a0 + a1;
}

// fp8a8f32 decode M == 1: E4M3 activation + F32 a_scale x E4M3 weight + F32 b_scale -> BF16.
template <uint32_t RB, uint32_t WAVES>
__global__ __launch_bounds__(WAVES * 32) void fp8a8_f32scale_m1(
    const uint8_t* __restrict__ a, const float* __restrict__ as,
    const uint8_t* __restrict__ w, const float* __restrict__ ws,
    uint16_t* __restrict__ y, Meta m) {
    static_assert(128u % RB == 0u, "RB must divide 128");
    const uint32_t tid = threadIdx.x, lane = tid & 31u, wave = tid >> 5;
    const uint32_t rows = (uint32_t)m.N, cols = (uint32_t)m.K;
    const uint32_t stride = 32u * 16u;
    const uint32_t full = (cols / stride) * stride;
    const uint32_t step = (uint32_t)gridDim.x * WAVES * RB;

    for (uint32_t r0 = ((uint32_t)blockIdx.x * WAVES + wave) * RB; r0 < rows; r0 += step) {
        const uint8_t* wrow[RB];
#pragma unroll
        for (uint32_t r = 0; r < RB; ++r)
            wrow[r] = w + (size_t)(r0 + r < rows ? r0 + r : rows - 1u) * (size_t)m.wr;
        const float* srow = ws + (size_t)(r0 / 128u) * (size_t)m.sr;
        float acc[RB];
#pragma unroll
        for (uint32_t r = 0; r < RB; ++r) acc[r] = 0.0f;

        uint4 wn[RB], an;
        float wsn = 0.0f, asn = 0.0f;
        auto load = [&](uint32_t c) {
#pragma unroll
            for (uint32_t r = 0; r < RB; ++r) wn[r] = *(const uint4*)(wrow[r] + c);
            an = *(const uint4*)(a + c);
            const uint32_t kb = c >> 7;
            wsn = srow[kb];
            asn = as[kb];
        };
        if (lane * 16u < full) load(lane * 16u);
        for (uint32_t c = lane * 16u; c < full; c += stride) {
            uint4 wc[RB];
#pragma unroll
            for (uint32_t r = 0; r < RB; ++r) wc[r] = wn[r];
            const uint4 ac = an;
            const float sc = asn * wsn;
            if (c + stride < full) load(c + stride);
            float af[16];
            f8_decode16(ac, af);
#pragma unroll
            for (uint32_t r = 0; r < RB; ++r) acc[r] = fmaf(f8_dot16_stream(wc[r], af), sc, acc[r]);
        }
        if (full < cols) {
            const uint32_t tc = full + lane * 16u;
            if (tc < cols) {
                uint4 wc[RB];
#pragma unroll
                for (uint32_t r = 0; r < RB; ++r) wc[r] = *(const uint4*)(wrow[r] + tc);
                const uint4 ac = *(const uint4*)(a + tc);
                const uint32_t kb = tc >> 7;
                const float sc = as[kb] * srow[kb];
                float af[16];
                f8_decode16(ac, af);
#pragma unroll
                for (uint32_t r = 0; r < RB; ++r)
                    acc[r] = fmaf(f8_dot16_stream(wc[r], af), sc, acc[r]);
            }
        }
#pragma unroll
        for (uint32_t r = 0; r < RB; ++r) acc[r] = r4d_wave_sum(acc[r]);
        if (lane == 0u) {
#pragma unroll
            for (uint32_t r = 0; r < RB; ++r)
                if (r0 + r < rows) y[r0 + r] = f32_to_bf16(acc[r]);
        }
    }
}

// fp8a8f32 decode M = 2..8: reads weights once across all MT tokens.
template <uint32_t MT, uint32_t RB, uint32_t WAVES>
__global__ __launch_bounds__(WAVES * 32) void fp8a8_f32scale_mt(
    const uint8_t* __restrict__ a, const float* __restrict__ as,
    const uint8_t* __restrict__ w, const float* __restrict__ ws,
    uint16_t* __restrict__ y, Meta m) {
    static_assert(128u % RB == 0u, "RB must divide 128");
    const uint32_t tid = threadIdx.x, lane = tid & 31u, wave = tid >> 5;
    const uint32_t rows = (uint32_t)m.N, cols = (uint32_t)m.K;
    const uint32_t stride = 32u * 16u;
    const uint32_t full = (cols / stride) * stride;
    const uint32_t step = (uint32_t)gridDim.x * WAVES * RB;
    constexpr bool kPre = MT <= 2u;

    for (uint32_t r0 = ((uint32_t)blockIdx.x * WAVES + wave) * RB; r0 < rows; r0 += step) {
        const uint8_t* wrow[RB];
#pragma unroll
        for (uint32_t r = 0; r < RB; ++r)
            wrow[r] = w + (size_t)(r0 + r < rows ? r0 + r : rows - 1u) * (size_t)m.wr;
        const float* srow = ws + (size_t)(r0 / 128u) * (size_t)m.sr;
        float acc[RB * MT];
#pragma unroll
        for (uint32_t i = 0; i < RB * MT; ++i) acc[i] = 0.0f;

        [[maybe_unused]] uint4 wn[kPre ? RB : 1];
        [[maybe_unused]] float wsn = 0.0f;
        [[maybe_unused]] uint4 an[kPre ? MT : 1];
        [[maybe_unused]] float asn[kPre ? MT : 1];
        auto load = [&](uint32_t c) {
            const uint32_t kb = c >> 7;
#pragma unroll
            for (uint32_t r = 0; r < RB; ++r) wn[r] = *(const uint4*)(wrow[r] + c);
            wsn = srow[kb];
            if constexpr (kPre) {
#pragma unroll
                for (uint32_t t = 0; t < MT; ++t) {
                    an[t] = *(const uint4*)(a + (size_t)t * (size_t)m.ar + c);
                    asn[t] = as[(size_t)t * (size_t)m.asr + kb];
                }
            }
        };
        if constexpr (kPre)
            if (lane * 16u < full) load(lane * 16u);
        for (uint32_t c = lane * 16u; c < full; c += stride) {
            uint4 wc[RB];
            float wsc;
            [[maybe_unused]] uint4 ac[kPre ? MT : 1];
            [[maybe_unused]] float asc[kPre ? MT : 1];
            if constexpr (kPre) {
#pragma unroll
                for (uint32_t r = 0; r < RB; ++r) wc[r] = wn[r];
                wsc = wsn;
#pragma unroll
                for (uint32_t t = 0; t < MT; ++t) { ac[t] = an[t]; asc[t] = asn[t]; }
                if (c + stride < full) load(c + stride);
            } else {
#pragma unroll
                for (uint32_t r = 0; r < RB; ++r) wc[r] = *(const uint4*)(wrow[r] + c);
                wsc = srow[c >> 7];
            }
            float wf[RB][16];
#pragma unroll
            for (uint32_t r = 0; r < RB; ++r) f8_decode16(wc[r], wf[r]);
            const uint32_t kb = c >> 7;
#pragma unroll
            for (uint32_t t = 0; t < MT; ++t) {
                uint4 av;
                float asv;
                if constexpr (kPre) {
                    av = ac[t];
                    asv = asc[t];
                } else {
                    av = *(const uint4*)(a + (size_t)t * (size_t)m.ar + c);
                    asv = as[(size_t)t * (size_t)m.asr + kb];
                }
                float af[16];
                f8_decode16(av, af);
                const float sc = asv * wsc;
#pragma unroll
                for (uint32_t r = 0; r < RB; ++r) {
                    float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
                    for (int k = 0; k < 16; k += 2) {
                        a0 = fmaf(wf[r][k],     af[k],     a0);
                        a1 = fmaf(wf[r][k + 1], af[k + 1], a1);
                    }
                    acc[r * MT + t] = fmaf(a0 + a1, sc, acc[r * MT + t]);
                }
            }
        }
        if (full < cols) {
            const uint32_t tc = full + lane * 16u;
            if (tc < cols) {
                float wf[RB][16];
#pragma unroll
                for (uint32_t r = 0; r < RB; ++r) f8_decode16(*(const uint4*)(wrow[r] + tc), wf[r]);
                const uint32_t kb = tc >> 7;
                const float wsc = srow[kb];
#pragma unroll
                for (uint32_t t = 0; t < MT; ++t) {
                    float af[16];
                    f8_decode16(*(const uint4*)(a + (size_t)t * (size_t)m.ar + tc), af);
                    const float sc = as[(size_t)t * (size_t)m.asr + kb] * wsc;
#pragma unroll
                    for (uint32_t r = 0; r < RB; ++r) {
                        float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
                        for (int k = 0; k < 16; k += 2) {
                            a0 = fmaf(wf[r][k],     af[k],     a0);
                            a1 = fmaf(wf[r][k + 1], af[k + 1], a1);
                        }
                        acc[r * MT + t] = fmaf(a0 + a1, sc, acc[r * MT + t]);
                    }
                }
            }
        }
#pragma unroll
        for (uint32_t i = 0; i < RB * MT; ++i) acc[i] = r4d_wave_sum(acc[i]);
        if (lane == 0u) {
#pragma unroll
            for (uint32_t r = 0; r < RB; ++r) {
                if (r0 + r >= rows) break;
#pragma unroll
                for (uint32_t t = 0; t < MT; ++t)
                    y[(size_t)t * (size_t)m.yr + r0 + r] = f32_to_bf16(acc[r * MT + t]);
            }
        }
    }
}

// fp8a16f32 M == 1: BF16 activation directly (no a_scale) x E4M3 weight + F32 b_scale -> BF16.
template <uint32_t RB, uint32_t WAVES>
__global__ __launch_bounds__(WAVES * 32) void fp8a16_f32scale_m1(
    const uint16_t* __restrict__ a, const uint8_t* __restrict__ w,
    const float* __restrict__ ws, uint16_t* __restrict__ y,
    uint32_t rows, uint32_t cols, long w_ld, long sw_ld) {
    static_assert(128u % RB == 0u, "RB must divide 128");
    const uint32_t tid = threadIdx.x, lane = tid & 31u, wave = tid >> 5;
    const uint32_t stride = 32u * 16u;
    const uint32_t full = (cols / stride) * stride;
    const uint32_t full16 = (cols / 16u) * 16u;
    const uint32_t step = (uint32_t)gridDim.x * WAVES * RB;

    for (uint32_t r0 = ((uint32_t)blockIdx.x * WAVES + wave) * RB; r0 < rows; r0 += step) {
        const uint8_t* wrow[RB];
#pragma unroll
        for (uint32_t r = 0; r < RB; ++r)
            wrow[r] = w + (size_t)(r0 + r < rows ? r0 + r : rows - 1u) * (size_t)w_ld;
        const float* srow = ws + (size_t)(r0 / 128u) * (size_t)sw_ld;
        float acc[RB];
#pragma unroll
        for (uint32_t r = 0; r < RB; ++r) acc[r] = 0.0f;

        // Double-buffered weight/activation prefetch with packed BF16 widening (low VGPRs + full GDDR6 latency hiding).
        uint4 wn[RB], xn0, xn1;
        float sn = 0.0f;
        auto load = [&](uint32_t c) {
#pragma unroll
            for (uint32_t r = 0; r < RB; ++r) wn[r] = *(const uint4*)(wrow[r] + c);
            const uint4* xp = (const uint4*)(a + c);
            xn0 = xp[0];
            xn1 = xp[1];
            sn = srow[c >> 7];
        };
        const uint32_t c0 = lane * 16u;
        if (c0 < full) load(c0);
        for (uint32_t c = c0; c < full; c += stride) {
            uint4 wc[RB];
#pragma unroll
            for (uint32_t r = 0; r < RB; ++r) wc[r] = wn[r];
            const uint4 x0 = xn0, x1 = xn1;
            const float sc = sn;
            if (c + stride < full) load(c + stride);
#pragma unroll
            for (uint32_t r = 0; r < RB; ++r)
                acc[r] = fmaf(f8_dot16_stream(wc[r], x0, x1), sc, acc[r]);
        }
        if (full + c0 < full16) {
            const uint32_t tc = full + c0;
            uint4 wc[RB];
#pragma unroll
            for (uint32_t r = 0; r < RB; ++r) wc[r] = *(const uint4*)(wrow[r] + tc);
            const uint4* xp = (const uint4*)(a + tc);
            const uint4 x0 = xp[0], x1 = xp[1];
            const float sc = srow[tc >> 7];
#pragma unroll
            for (uint32_t r = 0; r < RB; ++r)
                acc[r] = fmaf(f8_dot16_stream(wc[r], x0, x1), sc, acc[r]);
        }
        for (uint32_t c = full16 + lane; c < cols; c += 32u) {
            const float xv = bf16_to_f32(a[c]), sc = srow[c >> 7];
#pragma unroll
            for (uint32_t r = 0; r < RB; ++r)
                acc[r] = fmaf(r4d_fp8::e4m3(wrow[r][c]) * xv, sc, acc[r]);
        }
#pragma unroll
        for (uint32_t r = 0; r < RB; ++r) acc[r] = r4d_wave_sum(acc[r]);
        if (lane == 0u) {
#pragma unroll
            for (uint32_t r = 0; r < RB; ++r)
                if (r0 + r < rows) y[r0 + r] = f32_to_bf16(acc[r]);
        }
    }
}

// K == 256: each half-wave handles a full row, with eight lanes per 128-column scale block.
template <uint32_t RB, uint32_t WAVES>
__global__ __launch_bounds__(WAVES * 32) void fp8a16_f32scale_m1_k256(
    const uint16_t* __restrict__ a, const uint8_t* __restrict__ w,
    const float* __restrict__ ws, uint16_t* __restrict__ y,
    uint32_t rows, long w_ld, long sw_ld) {
    static_assert(RB % 2u == 0u && 128u % RB == 0u, "RB must be even and divide 128");
    const uint32_t tid = threadIdx.x, lane = tid & 31u, wave = tid >> 5;
    const uint32_t half_lane = lane & 15u, half = lane >> 4;
    const uint32_t c = half_lane * 16u;
    const uint32_t step = (uint32_t)gridDim.x * WAVES * RB;
    const uint4* xp = (const uint4*)(a + c);
    const uint4 x0 = xp[0], x1 = xp[1];

    for (uint32_t r0 = ((uint32_t)blockIdx.x * WAVES + wave) * RB; r0 < rows; r0 += step) {
        const float sc = ws[(size_t)(r0 / 128u) * (size_t)sw_ld + (c >> 7)];
#pragma unroll
        for (uint32_t r = 0; r < RB; r += 2u) {
            const uint32_t row = r0 + r + half;
            const uint8_t* wrow = w + (size_t)(row < rows ? row : rows - 1u) * (size_t)w_ld;
            const uint4 packed = *(const uint4*)(wrow + c);
            float acc = fmaf(f8_dot16_stream(packed, x0, x1), sc, 0.0f);
            acc = r4d_xor_sum<16>(acc);
            // The original full-wave sum added +0 from the idle half, including for signed zero.
            if (half_lane == 0u && row < rows) y[row] = f32_to_bf16(acc + 0.0f);
        }
    }
}

// fp8a16f32 M = 2..8: reads weights once across all MT tokens, supports pitched a/b/y.
template <uint32_t MT, uint32_t RB, uint32_t WAVES>
__global__ __launch_bounds__(WAVES * 32) void fp8a16_f32scale_mt(
    const uint16_t* __restrict__ a, const uint8_t* __restrict__ w,
    const float* __restrict__ ws, uint16_t* __restrict__ y,
    uint32_t rows, uint32_t cols, long a_ld, long w_ld, long sw_ld, long y_ld) {
    static_assert(128u % RB == 0u, "RB must divide 128");
    const uint32_t tid = threadIdx.x, lane = tid & 31u, wave = tid >> 5;
    const uint32_t stride = 32u * 16u;
    const uint32_t full = (cols / stride) * stride;
    const uint32_t full16 = (cols / 16u) * 16u;
    const uint32_t step = (uint32_t)gridDim.x * WAVES * RB;
    constexpr bool kPre = MT <= 2u;

    for (uint32_t r0 = ((uint32_t)blockIdx.x * WAVES + wave) * RB; r0 < rows; r0 += step) {
        const uint8_t* wrow[RB];
#pragma unroll
        for (uint32_t r = 0; r < RB; ++r)
            wrow[r] = w + (size_t)(r0 + r < rows ? r0 + r : rows - 1u) * (size_t)w_ld;
        const float* srow = ws + (size_t)(r0 / 128u) * (size_t)sw_ld;
        float acc[RB * MT];
#pragma unroll
        for (uint32_t i = 0; i < RB * MT; ++i) acc[i] = 0.0f;

        [[maybe_unused]] uint4 wn[kPre ? RB : 1];
        [[maybe_unused]] float sn = 0.0f;
        [[maybe_unused]] uint4 xn[kPre ? MT : 1][2];
        auto load = [&](uint32_t c) {
#pragma unroll
            for (uint32_t r = 0; r < RB; ++r) wn[r] = *(const uint4*)(wrow[r] + c);
            sn = srow[c >> 7];
            if constexpr (kPre) {
#pragma unroll
                for (uint32_t t = 0; t < MT; ++t) {
                    const uint4* xp = (const uint4*)(a + (size_t)t * (size_t)a_ld + c);
                    xn[t][0] = xp[0];
                    xn[t][1] = xp[1];
                }
            }
        };
        if constexpr (kPre)
            if (lane * 16u < full) load(lane * 16u);
        for (uint32_t c = lane * 16u; c < full; c += stride) {
            uint4 wc[RB];
            float sc;
            [[maybe_unused]] uint4 xc[kPre ? MT : 1][2];
            if constexpr (kPre) {
#pragma unroll
                for (uint32_t r = 0; r < RB; ++r) wc[r] = wn[r];
                sc = sn;
#pragma unroll
                for (uint32_t t = 0; t < MT; ++t) { xc[t][0] = xn[t][0]; xc[t][1] = xn[t][1]; }
                if (c + stride < full) load(c + stride);
            } else {
#pragma unroll
                for (uint32_t r = 0; r < RB; ++r) wc[r] = *(const uint4*)(wrow[r] + c);
                sc = srow[c >> 7];
            }
            float wf[RB][16];
#pragma unroll
            for (uint32_t r = 0; r < RB; ++r) f8_decode16(wc[r], wf[r]);
#pragma unroll
            for (uint32_t t = 0; t < MT; ++t) {
                uint4 x0, x1;
                if constexpr (kPre) {
                    x0 = xc[t][0];
                    x1 = xc[t][1];
                } else {
                    const uint4* xp = (const uint4*)(a + (size_t)t * (size_t)a_ld + c);
                    x0 = xp[0];
                    x1 = xp[1];
                }
                float xv[16];
                bf16_widen16(x0, x1, xv);
#pragma unroll
                for (uint32_t r = 0; r < RB; ++r) {
                    float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
                    for (int k = 0; k < 16; k += 2) {
                        a0 = fmaf(wf[r][k],     xv[k],     a0);
                        a1 = fmaf(wf[r][k + 1], xv[k + 1], a1);
                    }
                    acc[r * MT + t] = fmaf(a0 + a1, sc, acc[r * MT + t]);
                }
            }
        }
        if (full < full16) {
            const uint32_t tc = full + lane * 16u;
            if (tc < full16) {
                float wf[RB][16];
#pragma unroll
                for (uint32_t r = 0; r < RB; ++r) f8_decode16(*(const uint4*)(wrow[r] + tc), wf[r]);
                const float sc = srow[tc >> 7];
#pragma unroll
                for (uint32_t t = 0; t < MT; ++t) {
                    const uint4* xp = (const uint4*)(a + (size_t)t * (size_t)a_ld + tc);
                    float xv[16];
                    bf16_widen16(xp[0], xp[1], xv);
#pragma unroll
                    for (uint32_t r = 0; r < RB; ++r) {
                        float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
                        for (int k = 0; k < 16; k += 2) {
                            a0 = fmaf(wf[r][k],     xv[k],     a0);
                            a1 = fmaf(wf[r][k + 1], xv[k + 1], a1);
                        }
                        acc[r * MT + t] = fmaf(a0 + a1, sc, acc[r * MT + t]);
                    }
                }
            }
        }
        for (uint32_t c = full16 + lane; c < cols; c += 32u) {
            const float sc = srow[c >> 7];
#pragma unroll
            for (uint32_t r = 0; r < RB; ++r) {
                const float wv = r4d_fp8::e4m3(wrow[r][c]);
#pragma unroll
                for (uint32_t t = 0; t < MT; ++t)
                    acc[r * MT + t] =
                        fmaf(wv * bf16_to_f32(a[(size_t)t * (size_t)a_ld + c]), sc, acc[r * MT + t]);
            }
        }
#pragma unroll
        for (uint32_t i = 0; i < RB * MT; ++i) acc[i] = r4d_wave_sum(acc[i]);
        if (lane == 0u) {
#pragma unroll
            for (uint32_t r = 0; r < RB; ++r) {
                if (r0 + r >= rows) break;
#pragma unroll
                for (uint32_t t = 0; t < MT; ++t)
                    y[(size_t)t * (size_t)y_ld + r0 + r] = f32_to_bf16(acc[r * MT + t]);
            }
        }
    }
}

// Fallback for fp8a16f32 when base pointers or row pitches are not 16-byte aligned.
__global__ __launch_bounds__(128) void fp8a16_f32scale_fallback(
    const uint16_t* __restrict__ a, const uint8_t* __restrict__ w,
    const float* __restrict__ ws, uint16_t* __restrict__ y,
    uint32_t M, uint32_t rows, uint32_t cols, long a_ld, long w_ld, long sw_ld, long y_ld) {
    const uint32_t tid = threadIdx.x, lane = tid & 31u, wave = tid >> 5;
    const uint32_t row = (uint32_t)blockIdx.x * 4u + wave;
    if (row >= rows) return;
    const uint8_t* wrow = w + (size_t)row * (size_t)w_ld;
    const float* srow = ws + (size_t)(row / 128u) * (size_t)sw_ld;
    float acc[8];
#pragma unroll
    for (uint32_t t = 0; t < 8u; ++t) acc[t] = 0.0f;
    for (uint32_t c = lane; c < cols; c += 32u) {
        const float wv = r4d_fp8::e4m3(wrow[c]), sc = srow[c >> 7];
        for (uint32_t t = 0; t < M; ++t)
            acc[t] = fmaf(wv * bf16_to_f32(a[(size_t)t * (size_t)a_ld + c]), sc, acc[t]);
    }
    for (uint32_t t = 0; t < M; ++t) {
        const float sum = r4d_wave_sum(acc[t]);
        if (lane == 0u) y[(size_t)t * (size_t)y_ld + row] = f32_to_bf16(sum);
    }
}

constexpr int kLdsCap = 64 * 1024;
long decode_lds(int R, int S, int C, long KB, bool stage) {
    return ((long)S * R * C * 8 * 32 + (long)R * C * 8 * 32 + (stage ? (long)R * 16 * KB : 0)) * 4;
}

int dense_mvm_fp8a8(RadStream stream, const uint8_t* xa, const float* xsa, const uint8_t* wa,
                    const float* wsa, uint16_t* ya, const Meta& m) {
    constexpr uint32_t W = 4u;
    const uint32_t N = (uint32_t)m.N;
    if (m.T == 1) {
        const uint32_t rb = N <= 512u ? 1u : (N <= 1536u ? 2u : 4u);
        const uint32_t blocks = min(2048u, (N + W * rb - 1u) / (W * rb));
        if (rb == 4u)
            return R4D_EMIT(stream, (fp8a8_f32scale_m1<4u, W>), dim3(blocks), dim3(W * 32u), 0,
                            xa, xsa, wa, wsa, ya, m);
        if (rb == 2u)
            return R4D_EMIT(stream, (fp8a8_f32scale_m1<2u, W>), dim3(blocks), dim3(W * 32u), 0,
                            xa, xsa, wa, wsa, ya, m);
        return R4D_EMIT(stream, (fp8a8_f32scale_m1<1u, W>), dim3(blocks), dim3(W * 32u), 0,
                        xa, xsa, wa, wsa, ya, m);
    }
    const uint32_t rb = N <= 512u ? 1u : 2u;
    const uint32_t blocks = min(2048u, (N + W * rb - 1u) / (W * rb));
#define F8A8_MT_RB(NM, RBV)                                                                        \
    R4D_EMIT(stream, (fp8a8_f32scale_mt<NM, RBV, W>), dim3(blocks), dim3(W * 32u), 0, xa, xsa, wa, \
             wsa, ya, m)
#define F8A8_MT(NM) (rb == 2u ? F8A8_MT_RB(NM, 2u) : F8A8_MT_RB(NM, 1u))
    switch (m.T) {
        case 2: return F8A8_MT(2u);
        case 3: return F8A8_MT(3u);
        case 4: return F8A8_MT(4u);
        case 5: return F8A8_MT(5u);
        case 6: return F8A8_MT(6u);
        case 7: return F8A8_MT(7u);
        case 8: return F8A8_MT(8u);
        default: return RAD_E_SHAPE;
    }
#undef F8A8_MT
#undef F8A8_MT_RB
}

int dense_fast(RadStream stream, const RadTensor& x, const RadTensor& xs, const RadTensor& w,
               const RadTensor& ws, const RadTensor& y, const Meta& m, long long declared_mvm_max, bool* done) {
    *done = false;
    if (m.wr % 16 || m.ar % 16 ||
        (reinterpret_cast<uintptr_t>(x.data) | reinterpret_cast<uintptr_t>(w.data)) % 16)
        return RAD_OK;
    const uint8_t *xa = (const uint8_t*)x.data, *wa = (const uint8_t*)w.data;
    const float *xsa = (const float*)xs.data, *wsa = (const float*)ws.data;
    uint16_t* ya = (uint16_t*)y.data;

    // Fast decode path (M <= 8): coalesced uint4 E4M3 x E4M3 matvec, zero LDS, zero barriers.
    /* The matvec bound: the op's declared parameter "mvm_max" when it has one (ERNIE declares 0: every step takes the WMMA forms below, equal per row);
     * otherwise the generic bound, R4D_F8A8_MVM_MAX (read once) or 8. A declared value is per op and per launch: no process state. */
    static const int generic_mvm_max = r4d_ernie::pin::f8a8_mvm_max(std::getenv("R4D_F8A8_MVM_MAX"));
    const int mvm_max = r4d_ernie::pin::f8a8_mvm_bound(declared_mvm_max, generic_mvm_max);
    if (r4d_ernie::pin::f8a8_mvm((int)m.T, mvm_max)) {
        *done = true;
        return dense_mvm_fp8a8(stream, xa, xsa, wa, wsa, ya, m);
    }

    // Fast prefill path: 8-wave 128x128 or 64x128 uint4-vectorized FP8 WMMA.
    const long tiles_n = m.N / 128;
    const long tiles_128 = tiles_n * ((m.T + 127) / 128);
    const long tiles_64 = tiles_n * ((m.T + 63) / 64);
    if (m.T >= 128 && tiles_128 >= 64) {
        *done = true;
        return R4D_EMIT(stream, (fp8_f32scale_prefill<4, 2>),
                        dim3((unsigned)tiles_n, (unsigned)((m.T + 127) / 128)), dim3(256), 0, xa,
                        xsa, wa, wsa, ya, m);
    }
    if (m.T > 64 || (m.T > 32 && tiles_64 >= 64)) {
        *done = true;
        return R4D_EMIT(stream, (fp8_f32scale_prefill<2, 2>),
                        dim3((unsigned)tiles_n, (unsigned)((m.T + 63) / 64)), dim3(256), 0, xa,
                        xsa, wa, wsa, ya, m);
    }

    // Small-M (9..64 with low block count for 64x128): K-parallel WMMA decode fills all 64 CUs.
    const long kb = m.K / 128;
    const int c = m.T > 16 && m.N > 512 ? 2 : 1;
    int s = 8;
    while (s > 1 && s > kb) s /= 2;
    const int stage = decode_lds(1, s, c, kb, true) <= kLdsCap;
    const long lds = decode_lds(1, s, c, kb, stage);
    if (lds > kLdsCap || m.N % (16 * c)) return RAD_OK;
    *done = true;
    const dim3 grid(m.N / (16 * c), (unsigned)((m.T + 15) / 16)), block(32 * s);
    if (c == 2)
        return R4D_EMIT(stream, (fp8_f32scale_decode<1, 2>), grid, block, (uint32_t)lds, xa, xsa,
                        wa, wsa, ya, m, stage);
    return R4D_EMIT(stream, (fp8_f32scale_decode<1, 1>), grid, block, (uint32_t)lds, xa, xsa, wa,
                    wsa, ya, m, stage);
}

int dense_a16(const RadArgs* a, RadStream stream) {
    if (!a || !a->t || a->n_t < 5) return RAD_E_INVAL;
    // Operands: t[0]=a (BF16), t[1]=a_scale (optional/absent), t[2]=b (E4M3), t[3]=b_scale (F32),
    // t[4]=y (BF16).
    const auto &x = a->t[0], &w = a->t[2], &ws = a->t[3], &y = a->t[4];
    if (!x.data || !w.data || !ws.data || !y.data) return RAD_E_INVAL;
    if (x.dtype != RAD_BF16 || w.dtype != RAD_F8E4M3 || ws.dtype != RAD_F32 ||
        y.dtype != RAD_BF16)
        return RAD_E_DTYPE;
    if (r4d_geti_or(a, "group", 128) != 128 || r4d_geti_or(a, "pb1", 0) > 0 ||
        r4d_geti_or(a, "pb2", 0) > 0)
        return RAD_E_SHAPE;
    if (w.rank != 2 || ws.rank != 2 || x.rank < 2 || y.rank < 2) return RAD_E_SHAPE;

    const int64_t N = r4d_geti_or(a, "N", w.shape[0]);
    const int64_t K = r4d_geti_or(a, "K", w.shape[1]);
    const int64_t M = y.shape[0];
    if (M < 1 || M > 8 || N < 1 || K < 1 || N > INT_MAX || K > INT_MAX) return RAD_E_SHAPE;
    const int64_t nb = (N + 127) / 128, kb = (K + 127) / 128;
    if (w.shape[0] != N || w.shape[1] != K || ws.shape[0] != nb || ws.shape[1] != kb)
        return RAD_E_SHAPE;
    if (x.shape[0] != M || rad_tensor_numel(&x) != M * K || rad_tensor_numel(&y) != M * N)
        return RAD_E_SHAPE;

    const long long a_ld = r4d_row_pitch(&x);
    const long long w_ld = r4d_row_pitch(&w);
    const long long sw_ld = r4d_row_pitch(&ws);
    const long long y_ld = r4d_row_pitch(&y);
    if (a_ld < K || w_ld < K || sw_ld < kb || y_ld < N) return RAD_E_STRIDE;

    const uint16_t* xa = (const uint16_t*)x.data;
    const uint8_t* wa = (const uint8_t*)w.data;
    const float* wsa = (const float*)ws.data;
    uint16_t* ya = (uint16_t*)y.data;

    const bool aligned16 = r4d_aligned(x.data, 16) && r4d_aligned(w.data, 16) &&
                           (a_ld % 8 == 0) && (w_ld % 16 == 0);
    if (!aligned16) {
        const uint32_t blocks = (uint32_t)((N + 3) / 4);
        return R4D_EMIT(stream, fp8a16_f32scale_fallback, dim3(blocks), dim3(128), 0, xa, wa, wsa,
                        ya, (uint32_t)M, (uint32_t)N, (uint32_t)K, (long)a_ld, (long)w_ld,
                        (long)sw_ld, (long)y_ld);
    }

    constexpr uint32_t W = 4u;
    const uint32_t rows = (uint32_t)N, cols = (uint32_t)K;
    if (M == 1) {
        // The single activation and output rows start at their tensor base addresses.
        if (cols == 256u) {
            constexpr uint32_t rb = 4u;
            const uint32_t blocks = min(2048u, (rows + W * rb - 1u) / (W * rb));
            return R4D_EMIT(stream, (fp8a16_f32scale_m1_k256<rb, W>), dim3(blocks), dim3(W * 32u), 0,
                            xa, wa, wsa, ya, rows, (long)w_ld, (long)sw_ld);
        }
        const uint32_t rb = rows <= 512u ? 1u : (rows <= 4096u ? 2u : 4u);
        const uint32_t blocks = min(2048u, (rows + W * rb - 1u) / (W * rb));
        if (rb == 4u)
            return R4D_EMIT(stream, (fp8a16_f32scale_m1<4u, W>), dim3(blocks), dim3(W * 32u), 0,
                            xa, wa, wsa, ya, rows, cols, (long)w_ld, (long)sw_ld);
        if (rb == 2u)
            return R4D_EMIT(stream, (fp8a16_f32scale_m1<2u, W>), dim3(blocks), dim3(W * 32u), 0,
                            xa, wa, wsa, ya, rows, cols, (long)w_ld, (long)sw_ld);
        return R4D_EMIT(stream, (fp8a16_f32scale_m1<1u, W>), dim3(blocks), dim3(W * 32u), 0, xa,
                        wa, wsa, ya, rows, cols, (long)w_ld, (long)sw_ld);
    }
    const uint32_t rb = rows <= 512u ? 1u : 2u;
    const uint32_t blocks = min(2048u, (rows + W * rb - 1u) / (W * rb));
#define F8A16_MT_RB(NM, RBV)                                                                       \
    R4D_EMIT(stream, (fp8a16_f32scale_mt<NM, RBV, W>), dim3(blocks), dim3(W * 32u), 0, xa, wa,     \
             wsa, ya, rows, cols, (long)a_ld, (long)w_ld, (long)sw_ld, (long)y_ld)
#define F8A16_MT(NM) (rb == 2u ? F8A16_MT_RB(NM, 2u) : F8A16_MT_RB(NM, 1u))
    switch (M) {
        case 2: return F8A16_MT(2u);
        case 3: return F8A16_MT(3u);
        case 4: return F8A16_MT(4u);
        case 5: return F8A16_MT(5u);
        case 6: return F8A16_MT(6u);
        case 7: return F8A16_MT(7u);
        case 8: return F8A16_MT(8u);
        default: return RAD_E_SHAPE;
    }
#undef F8A16_MT
#undef F8A16_MT_RB
}

int body(const RadArgs* a, RadStream stream, bool grouped) {
    if (!a || !a->t || a->n_t < (grouped ? 9 : 5)) return RAD_E_INVAL;
    const auto &x = a->t[0], &xs = a->t[1], &w = a->t[2], &ws = a->t[3], &y = a->t[grouped ? 8 : 4];
    const int64_t N = r4d_geti_or(a, "N", 0), K = r4d_geti_or(a, "K", 0);
    if (N < 128 || N % 128 || K < 128 || K % 128 || N > INT_MAX || K > INT_MAX ||
        r4d_geti_or(a, "group", 0) != 128 || r4d_geti_or(a, "pb1", 0) > 0)
        return RAD_E_SHAPE;
    if (x.dtype != RAD_F8E4M3 || xs.dtype != RAD_F32 || w.dtype != RAD_F8E4M3 ||
        ws.dtype != RAD_F32 || y.dtype != RAD_BF16)
        return RAD_E_DTYPE;
    if (!x.data || !xs.data || !w.data || !ws.data || !y.data || x.rank != 2 || xs.rank != 2 ||
        y.rank != 2 || x.shape[1] != K || xs.shape[0] != x.shape[0] || xs.shape[1] != K / 128 ||
        y.shape[1] != N || x.stride[1] != 1 || xs.stride[1] != 1 || y.stride[1] != 1 ||
        y.shape[0] < 1 || y.shape[0] > INT_MAX)
        return RAD_E_SHAPE;
    const uint32_t rank = grouped ? 3 : 2;
    if (w.rank != rank || ws.rank != rank || w.shape[rank - 2] != N || w.shape[rank - 1] != K ||
        ws.shape[rank - 2] != N / 128 || ws.shape[rank - 1] != K / 128 || w.stride[rank - 1] != 1 ||
        ws.stride[rank - 1] != 1)
        return RAD_E_SHAPE;
    Meta m{};
    m.ar = x.stride[0];
    m.asr = xs.stride[0];
    m.yr = y.stride[0];
    m.wr = w.stride[rank - 2];
    m.sr = ws.stride[rank - 2];
    m.T = (int)y.shape[0];
    m.N = N;
    m.K = K;
    if (!grouped) {
        if (x.shape[0] != m.T) return RAD_E_SHAPE;
        bool done = false;
        const int rc = dense_fast(stream, x, xs, w, ws, y, m, r4d_geti_or(a, "mvm_max", -1), &done);
        if (done || rc < 0) return rc;
        return R4D_EMIT(stream, fp8_f32scale, dim3(N / 64, (m.T + 15) / 16), dim3(128), 0,
                        (const uint8_t*)x.data, (const float*)xs.data, (const uint8_t*)w.data,
                        (const float*)ws.data, (uint16_t*)y.data, m);
    }
    m.experts = r4d_geti_or(a, "n_expert", 0);
    m.topk = r4d_geti_or(a, "top_k", 0);
    const char* order = r4d_gets(a, "a_order");
    m.sorted_a = order && !std::strcmp(order, "sorted");
    const auto &wo = a->t[6], &so = a->t[7], &st = a->t[4], &off = a->t[5];
    if (m.experts < 2 || m.topk < 1 || r4d_geti_or(a, "N_odd", N) != N ||
        r4d_geti_or(a, "K_odd", K) != K || wo.dtype != RAD_F8E4M3 || so.dtype != RAD_F32 ||
        wo.rank != 3 || so.rank != 3 || wo.shape[1] != N || wo.shape[2] != K ||
        so.shape[1] != N / 128 || so.shape[2] != K / 128 || wo.stride[1] != m.wr ||
        so.stride[1] != m.sr || wo.stride[2] != 1 || so.stride[2] != 1 || st.dtype != RAD_I32 ||
        off.dtype != RAD_I32 || st.rank != 1 || off.rank != 1 || st.stride[0] != 1 ||
        off.stride[0] != 1 || st.shape[0] < m.T || off.shape[0] < m.experts + 1 || !wo.data ||
        !so.data || !st.data || !off.data)
        return RAD_E_SHAPE;
    m.wstep[0] = w.stride[0];
    m.wstep[1] = wo.stride[0];
    m.sstep[0] = ws.stride[0] * 4;
    m.sstep[1] = so.stride[0] * 4;
    if (w.shape[0] < (m.experts + 1) / 2 || wo.shape[0] < m.experts / 2) return RAD_E_SHAPE;
    /* The small-M form at the MTP layer's gate_up slice up to 384 routed rows: a draft pass routes
     * one token, a verify pass a few. On gfx1201 (perfmxf8, cold) it halves the expert-major time
     * at one token and gains 3 to 21 % from 2 to 16. Down's K of 256 gains nothing from it (within
     * a few percent, a quarter slower at three tokens). */
    if (m.experts == 288 && m.topk == 8 && N == 512 && K == 4096 && m.T <= 384)
        return R4D_EMIT(stream, fp8_f32scale_small<512>,
                        dim3(N / 64, (m.T < m.experts ? m.T : m.experts) + 1), dim3(128), 0,
                        (const uint8_t*)x.data, (const float*)xs.data, w.data, ws.data, wo.data,
                        so.data, (const int*)st.data, (const int*)off.data, (uint16_t*)y.data, m);
    /* The expert-major form at every routed-row count. Measured at GLM's TP8 slices: a long K
     * (gate_up's 4096) wants the 256 stage, 25 to 40 % ahead of 128, and under 128 routed rows
     * four tiles a block on top; at down's K of 256 every variant reads within a few percent of
     * the bandwidth, so it keeps 128 over four tiles. */
#define F8_EXPERTS(SK, C)                                                                          \
    R4D_EMIT(stream, (fp8_f32scale_experts<SK, C>),                                                \
             dim3((N + 64 * C - 1) / (64 * C), m.experts + 1), dim3(128), 0,                       \
             (const uint8_t*)x.data, (const float*)xs.data, w.data, ws.data, wo.data, so.data,     \
             (const int*)st.data, (const int*)off.data, (uint16_t*)y.data, m)
    if (K > 256) return m.T < 128 ? F8_EXPERTS(256, 4) : F8_EXPERTS(256, 1);
    return F8_EXPERTS(128, 4);
#undef F8_EXPERTS
}
int dense(const RadArgs* a, RadStream stream) { return body(a, stream, false); }
int grouped(const RadArgs* a, RadStream stream) { return body(a, stream, true); }
}
extern "C" int r4d_rad_fp8_f32scale(const RadArgs* a, RadStream stream) { return dense(a, stream); }
extern "C" int r4d_rad_fp8a16_f32scale(const RadArgs* a, RadStream stream) {
    return dense_a16(a, stream);
}
extern "C" int r4d_rad_moe_fp8_f32scale(const RadArgs* a, RadStream stream) {
    return grouped(a, stream);
}
extern "C" int r4d_layout_fp8_f32scale(const RadParam*, int, int operand, const RadEncoding* e,
                                       const int* sel, const RadTensor*, int n, RadLayout*) {
    if (operand != 2 && operand != 3 && operand != 6 && operand != 7) return RAD_E_UNSUPPORTED;
    if (!e || !sel || n != 1 || !rad_enc_is(e, "affine") || e->n_planes != 2 ||
        e->plane[0].dtype != RAD_F8E4M3 || e->plane[1].dtype != RAD_F32 ||
        e->plane[1].block[0] != 128 || e->plane[1].block[1] != 128 ||
        sel[0] != (operand == 3 || operand == 7 ? 1 : 0))
        return RAD_E_DTYPE;
    return RAD_E_UNSUPPORTED; // Both canonical planes are read without a relayout.
}
