/* r4d_ernie_common.h -- Shared host/device helpers for ERNIE 5.5 Flash libr4d kernels (epic 4-3).
 * Written cleanly from ERNIE 5.5 Flash equations and libref references.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef R4D_ERNIE_COMMON_H
#define R4D_ERNIE_COMMON_H

#include "r4d_ernie_pins.h"
#include <hip/hip_runtime.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "r4d_args.h"
#include "r4d_launch.h"
#include "rad_plugin.h"

/* A kernel-argument struct is declared with R4D_ARGS, never with T x{} or a braced temporary: the backend copies sizeof(T) bytes of every argument struct into the
 * kernarg block, padding included, and audits a recorded pass byte for byte against a fresh issue (core/runtime/ctx.cpp, Ctx::run_pass).  zero-initialises the members,
 * not the padding, so a struct with a tail pad could differ between two issues by whatever the stack held. memset makes every byte defined; assigning members leaves the
 * padding alone. tests/r4d_emu/ernie_no_device_globals.cmake (the argument-declaration lint) refuses the other spellings in libr4d/r4d_ernie_*.hip. */
#define R4D_ARGS(T, name) T name; std::memset(&(name), 0, sizeof(name))

namespace r4d_ernie {

/* Diagnostic row-band pins (GPU-window use). Each pin reads one env var once; unset means the shipped rule. */
static inline int r4d_pin_env(const char* name, int def) { return pin::from_env(name, def); }

static constexpr int kMaxRopePairs = 256; /* supports rope_dim up to 512 */

__device__ __forceinline__ float bf2f(uint16_t h) {
    return __uint_as_float((uint32_t)h << 16);
}

__device__ __forceinline__ uint16_t f2bf(float f) {
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7f800000u) == 0x7f800000u)
        return (uint16_t)((u >> 16) | ((u & 0x7fffffu) ? 0x40u : 0u));
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

__device__ __forceinline__ float bfr(float f) {
    return bf2f(f2bf(f));
}

__device__ __forceinline__ float sigmoid_f(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/* Exact OCP FP8 E4M3 (e4m3fn) decode/encode matching abi/rad_plugin.h bit-for-bit. */
__device__ __forceinline__ float fp8e4m3_to_f32_dev(uint8_t b) {
    const uint32_t s = (uint32_t)(b >> 7) << 31;
    const uint32_t e = (uint32_t)(b >> 3) & 0xfu;
    uint32_t m = (uint32_t)b & 0x7u;
    uint32_t u;
    if (e == 0u) {
        if (m == 0u) {
            u = s;
        } else {
            int sh = 0;
            while (!(m & 0x8u)) { m <<= 1; ++sh; }
            m &= 0x7u;
            u = s | ((uint32_t)(127 - 6 - sh) << 23) | (m << 20);
        }
    } else if (e == 0xfu && m == 0x7u) {
        u = s | 0x7fc00000u;
    } else {
        u = s | ((e - 7u + 127u) << 23) | (m << 20);
    }
    return __uint_as_float(u);
}

__device__ __forceinline__ uint8_t f32_to_fp8e4m3_dev(float f) {
    const uint32_t u = __float_as_uint(f);
    const uint8_t s = (uint8_t)((u >> 24) & 0x80u);
    const uint32_t a = u & 0x7fffffffu;
    if (a > 0x7f800000u) return (uint8_t)(s | 0x7fu);
    if (a >= 0x43e00000u) return (uint8_t)(s | 0x7eu);
    const int e = (int)(a >> 23) - 127;
    uint32_t m = a & 0x7fffffu;
    if (e < -10) return s;
    if (e < -6) {
        const int sh = -6 - e + 20;
        m |= 0x800000u;
        const uint32_t lo = m & ((1u << sh) - 1u);
        uint32_t q = m >> sh;
        const uint32_t half = 1u << (sh - 1);
        if (lo > half || (lo == half && (q & 1u))) ++q;
        return (uint8_t)(s | q);
    }
    uint32_t q = m >> 20;
    const uint32_t lo = m & 0xfffffu;
    int ee = e + 7;
    if (lo > 0x80000u || (lo == 0x80000u && (q & 1u))) {
        if (++q == 8u) { q = 0u; ++ee; }
    }
    if (ee >= 15 && q >= 7u) return (uint8_t)(s | 0x7eu);
    return (uint8_t)(s | ((uint32_t)ee << 3) | q);
}

/* Wave32 & block reductions (safe in both gfx1201 wave32 and tests/r4d_emu). */
__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) v += __shfl_xor(v, mask, 32);
    return v;
}

__device__ __forceinline__ float wave_max(float v) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) v = fmaxf(v, __shfl_xor(v, mask, 32));
    return v;
}

__device__ __forceinline__ int wave_sum_i(int v) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) v += __shfl_xor(v, mask, 32);
    return v;
}

/* Block reduction for up to 256 threads (<= 8 waves). Caller passes a __shared__ float smem[8]. */
__device__ __forceinline__ float block_sum(float v, float* smem, int nthreads) {
    const int tid = threadIdx.x;
    const int lane = tid & 31, wave = tid >> 5, nwaves = (nthreads + 31) >> 5;
    v = wave_sum(v);
    if (lane == 0) smem[wave] = v;
    __syncthreads();
    float out = 0.0f;
    if (wave == 0) {
        float w = (lane < nwaves) ? smem[lane] : 0.0f;
        w = wave_sum(w);
        if (lane == 0) smem[0] = w;
    }
    __syncthreads();
    out = smem[0];
    __syncthreads();
    return out;
}

__device__ __forceinline__ float block_max(float v, float* smem, int nthreads) {
    const int tid = threadIdx.x;
    const int lane = tid & 31, wave = tid >> 5, nwaves = (nthreads + 31) >> 5;
    v = wave_max(v);
    if (lane == 0) smem[wave] = v;
    __syncthreads();
    if (wave == 0) {
        float w = (lane < nwaves) ? smem[lane] : -INFINITY;
        w = wave_max(w);
        if (lane == 0) smem[0] = w;
    }
    __syncthreads();
    float out = smem[0];
    __syncthreads();
    return out;
}

__device__ __forceinline__ int block_sum_i(int v, int* smem, int nthreads) {
    const int tid = threadIdx.x;
    const int lane = tid & 31, wave = tid >> 5, nwaves = (nthreads + 31) >> 5;
    v = wave_sum_i(v);
    if (lane == 0) smem[wave] = v;
    __syncthreads();
    if (wave == 0) {
        int w = (lane < nwaves) ? smem[lane] : 0;
        w = wave_sum_i(w);
        if (lane == 0) smem[0] = w;
    }
    __syncthreads();
    int out = smem[0];
    __syncthreads();
    return out;
}

/* Host-side helpers for parameter/tensor validation. */
static inline const RadParam* find_p(const RadArgs* a, const char* k) {
    if (!a || !k) return nullptr;
    for (int i = 0; i < a->n_p; ++i)
        if (a->p[i].key && std::strcmp(a->p[i].key, k) == 0) return &a->p[i];
    return nullptr;
}

static inline bool pos_strides(const RadTensor& t) {
    for (uint32_t i = 0; i < t.rank; ++i)
        if (t.stride[i] < 1 || t.shape[i] < 0) return false;
    return true;
}

static inline int64_t numel_t(const RadTensor& t) {
    if (t.rank == 0) return 0;
    int64_t n = 1;
    for (uint32_t i = 0; i < t.rank; ++i) {
        if (t.shape[i] < 0) return -1;
        if (t.shape[i] == 0) return 0;
        if (n > INT64_MAX / t.shape[i]) return -1;
        n *= t.shape[i];
    }
    return n;
}

static inline bool operands_overlap(const RadTensor& a, const RadTensor& b) {
    auto extent = [](const RadTensor& t) {
        int64_t last = 0;
        for (uint32_t i = 0; i < t.rank; ++i) last += (t.shape[i] - 1) * t.stride[i];
        const int64_t eb = (rad_dtype_bits(t.dtype) + 7) / 8;
        return std::pair<const char*, const char*>((const char*)t.data, (const char*)t.data + (last + 1) * eb);
    };
    const auto ea = extent(a), eb = extent(b);
    return ea.first < eb.second && eb.first < ea.second;
}

static inline bool pool_rows(RadTensor* t) {
    if (t->rank != 4) return true;
    if (t->shape[1] != 1 || t->shape[2] < 1 || t->stride[0] != t->shape[2] * t->stride[2]) return false;
    t->rank = 2;
    t->shape[0] *= t->shape[2];
    t->shape[1] = t->shape[3];
    t->stride[0] = t->stride[2];
    t->stride[1] = t->stride[3];
    return true;
}

struct ErnieYarn {
    bool on = false;
    double factor = 0.0, beta_fast = 32.0, beta_slow = 1.0, orig = 0.0;
};

static inline bool parse_yarn(const RadArgs* a, ErnieYarn* y) {
    const RadParam *pf = find_p(a, "yarn_factor"), *pb = find_p(a, "yarn_beta_fast"),
                   *ps = find_p(a, "yarn_beta_slow"), *po = find_p(a, "yarn_original_max");
    const double f = r4d_getf(a, "yarn_factor", 0.0), bf = r4d_getf(a, "yarn_beta_fast", 32.0),
                 bs = r4d_getf(a, "yarn_beta_slow", 1.0), og = r4d_getf(a, "yarn_original_max", 0.0);
    *y = ErnieYarn{};
    if (pf && f == 0.0) return true;
    if (!pf) return !(pb || ps || po);
    if (!(f >= 1.0) || !std::isfinite(f) || !(bf > 0.0) || !std::isfinite(bf) || !(bs > 0.0) || !std::isfinite(bs) ||
        bf < bs || !po || !(og > 0.0) || !std::isfinite(og))
        return false;
    y->on = true; y->factor = f; y->beta_fast = bf; y->beta_slow = bs; y->orig = og;
    return true;
}

/* Compact on-the-fly RoPE / YaRN inverse-frequency evaluator (48 B, well under AQL 768 B kernarg slot). */
struct RopeInvEval {
    double theta = 10000.0;
    double factor = 1.0;
    double low = 0.0;
    double span = 1.0;
    int R = 64;
    int yarn_on = 0;
    __host__ __device__ __forceinline__ double operator[](int i) const {
        if (!yarn_on) return pow(theta, -2.0 * (double)i / (double)R);
        const double pf = pow(theta, (2.0 * (double)i) / (double)R);
        const double ext = 1.0 / pf;
        const double itp = 1.0 / (factor * pf);
        const double ramp = fmin(fmax(((double)i - low) / span, 0.0), 1.0);
        const double mask = 1.0 - ramp;
        return itp * (1.0 - mask) + ext * mask;
    }
};

struct RopeInvTable {
    RopeInvEval inv;
};
static_assert(sizeof(RopeInvTable) <= 64, "RopeInvTable must stay small for AQL kernarg slot");

static inline bool fill_rope_inv(double theta, int64_t R, const ErnieYarn& y, RopeInvTable* out) {
    if (R < 2 || (R & 1) || R > 2 * kMaxRopePairs || !std::isfinite(theta) || theta <= 0.0) return false;
    out->inv.theta = theta;
    out->inv.R = (int)R;
    out->inv.yarn_on = y.on ? 1 : 0;
    out->inv.factor = y.on ? y.factor : 1.0;
    if (!y.on) {
        out->inv.low = 0.0;
        out->inv.span = 1.0;
        return true;
    }
    const double kPi = 3.14159265358979323846;
    auto dim = [&](double rot) { return ((double)R * std::log(y.orig / (rot * 2.0 * kPi))) / (2.0 * std::log(theta)); };
    const double low = std::fmax(std::floor(dim(y.beta_fast)), 0.0);
    double high = std::fmin(std::ceil(dim(y.beta_slow)), (double)(R - 1));
    if (low == high) high += 0.001;
    out->inv.low = low;
    out->inv.span = high - low;
    return true;
}

static inline bool is_pow2(int64_t v) { return v > 0 && (v & (v - 1)) == 0; }

}  /* namespace r4d_ernie */

#endif /* R4D_ERNIE_COMMON_H */
