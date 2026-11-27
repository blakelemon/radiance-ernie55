/* r4d_ernie_dspark.hip -- ERNIE 5.5 DSpark device ops:
 *   ernie_mhc_stream_mean, ernie_hca_dspark_attn.
 * Written cleanly from ERNIE 5.5 DSpark equations and libref/ref_ernie_hca.cpp.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "r4d_ernie_common.h"
#include "r4d_ernie_verdict.h"
#include "r4d_plugin.h"

namespace r4d_ernie_dspark {

using namespace r4d_ernie;

__device__ __forceinline__ void unpack_uint4_bf16(uint4 q, float* dst) {
    dst[0] = bf2f((uint16_t)(q.x & 0xffffu));
    dst[1] = bf2f((uint16_t)(q.x >> 16));
    dst[2] = bf2f((uint16_t)(q.y & 0xffffu));
    dst[3] = bf2f((uint16_t)(q.y >> 16));
    dst[4] = bf2f((uint16_t)(q.z & 0xffffu));
    dst[5] = bf2f((uint16_t)(q.z >> 16));
    dst[6] = bf2f((uint16_t)(q.w & 0xffffu));
    dst[7] = bf2f((uint16_t)(q.w >> 16));
}

__device__ __forceinline__ uint4 pack_bf16_uint4(const float* src) {
    const uint32_t w0 = (uint32_t)f2bf(src[0]) | ((uint32_t)f2bf(src[1]) << 16);
    const uint32_t w1 = (uint32_t)f2bf(src[2]) | ((uint32_t)f2bf(src[3]) << 16);
    const uint32_t w2 = (uint32_t)f2bf(src[4]) | ((uint32_t)f2bf(src[5]) << 16);
    const uint32_t w3 = (uint32_t)f2bf(src[6]) | ((uint32_t)f2bf(src[7]) << 16);
    return make_uint4(w0, w1, w2, w3);
}

/* ============================================================ op 1: ernie_mhc_stream_mean */
struct MHCStreamMeanArgs {
    const uint16_t* x;
    uint16_t* out;
    long long xp, xs, outp, outs;
    int M, H, mult;
};
static_assert(sizeof(MHCStreamMeanArgs) <= 512, "MHCStreamMeanArgs exceeds AQL kernarg limit");

/* Fast path: mult == 4, H % 8 == 0, unit inner strides -> 8 bf16 (uint4) per thread. */
__global__ __launch_bounds__(256) void ernie_mhc_stream_mean_vec_kernel(MHCStreamMeanArgs A) {
    const int g = (int)blockIdx.x * 256 + (int)threadIdx.x;
    const int col = g * 8;
    const int m = (int)blockIdx.y;
    if (col >= A.H || m >= A.M) return;

    const long long row = (long long)m * A.xp + (long long)col;
    const uint4 q0 = *reinterpret_cast<const uint4*>(A.x + row);
    const uint4 q1 = *reinterpret_cast<const uint4*>(A.x + row + (long long)A.H);
    const uint4 q2 = *reinterpret_cast<const uint4*>(A.x + row + 2LL * A.H);
    const uint4 q3 = *reinterpret_cast<const uint4*>(A.x + row + 3LL * A.H);

    float v[8], s[8];
    unpack_uint4_bf16(q0, s);
    unpack_uint4_bf16(q1, v);
#pragma unroll
    for (int e = 0; e < 8; ++e) s[e] += v[e];
    unpack_uint4_bf16(q2, v);
#pragma unroll
    for (int e = 0; e < 8; ++e) s[e] += v[e];
    unpack_uint4_bf16(q3, v);
#pragma unroll
    for (int e = 0; e < 8; ++e) s[e] = 0.25f * (s[e] + v[e]);

    *reinterpret_cast<uint4*>(A.out + (long long)m * A.outp + (long long)col) = pack_bf16_uint4(s);
}

/* General path: any mult (1..16), any H, any positive strides. */
__global__ __launch_bounds__(256) void ernie_mhc_stream_mean_kernel(MHCStreamMeanArgs A) {
    const int col = (int)blockIdx.x * 256 + (int)threadIdx.x;
    const int m = (int)blockIdx.y;
    if (col >= A.H || m >= A.M) return;

    const long long row = (long long)m * A.xp;
    float s = 0.0f;
    for (int j = 0; j < A.mult; ++j)
        s += bf2f(A.x[row + (long long)(j * A.H + col) * A.xs]);
    A.out[(long long)m * A.outp + (long long)col * A.outs] = f2bf(s * (1.0f / (float)A.mult));
}

/* ============================================================ op 2: ernie_hca_dspark_attn */
struct HcaDsparkArgs {
    const uint16_t* q;
    const uint16_t* kv;
    const int32_t* bt;
    const int32_t* seqused;
    const int32_t* cu;
    const float* sink;
    uint16_t* out;
    long long qp, qs, kvp0, kvp2, kvs, btp, bts, sqs, cus, sks, outp, outs;
    int M, H, D, S, win, bs, maxb, pages;
    int check;   /* 1: the validator kernel ran before this launch and its verdict (the gate word, vc.gate) gates every store */
    r4d_verdict::Vc vc;   /* the event log and the gate word of this call site (device memory the library owns; constant across replays) */
    float scale;
};
static_assert(sizeof(HcaDsparkArgs) <= 512, "HcaDsparkArgs exceeds AQL kernarg limit");

/* On-device sequence lookup: cu[s] <= m < cu[s+1]. Never touches host memory. */
__device__ __forceinline__ int dspark_find_seq(const HcaDsparkArgs& A, int m) {
    if (A.S <= 16) {
        for (int s = 0; s < A.S; ++s)
            if (A.cu[(long long)s * A.cus] <= m && m < A.cu[(long long)(s + 1) * A.cus]) return s;
        return -1;
    }
    int lo = 0, hi = A.S - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (A.cu[(long long)mid * A.cus] <= m) lo = mid; else hi = mid - 1;
    }
    return (A.cu[(long long)lo * A.cus] <= m && m < A.cu[(long long)(lo + 1) * A.cus]) ? lo : -1;
}

/* The device-side form of libref's validation (ref_ernie_hca_dspark_attn): sink finite-or-minus-inf, cu_seqlens starts at 0, ascends and ends at M,
 * seqused >= qlen for every non-empty sequence, and every block table entry the window reads is in [-1, pages) and inside the table width. libref
 * refuses the whole launch before its first store; this kernel runs ahead of the attention kernel on the same stream (one block, the metadata is
 * S + 1 + S ints, a block table row per window, and H floats), writes its verdict into the gate word of its call site (one block, one store of the whole word; r4d_ernie_verdict.h), and every attention block of THAT launch
 * reads that word first and returns without a store when it refuses. The violation is also raised into the event log (the DSpark log of the library's device state) the host takes (r4d_ernie_dspark_take_error), since an
 * asynchronous launch cannot return one. Bit 1 sink, 2 cu_seqlens, 4 seqused, 8 page index, 16 block index past the table. */

__global__ __launch_bounds__(256) void ernie_hca_dspark_validate_kernel(HcaDsparkArgs A) {
    __shared__ int sh_flag[256];
    const int tid = (int)threadIdx.x;
    int flag = 0;
    for (int h = tid; h < A.H; h += 256) {
        const float sk = A.sink[(long long)h * A.sks];
        if (sk != sk || sk == INFINITY) flag |= 1;
    }
    if (tid == 0) {
        if (A.cu[0] != 0 || A.cu[(long long)A.S * A.cus] != A.M) flag |= 2;
    }
    for (int s = tid; s < A.S; s += 256) {
        const long long c0 = A.cu[(long long)s * A.cus], c1 = A.cu[(long long)(s + 1) * A.cus];
        if (c0 > c1) { flag |= 2; continue; }
        const long long qlen = c1 - c0;
        if (qlen == 0) continue;
        const long long total = A.seqused[(long long)s * A.sqs];
        if (total < qlen) { flag |= 4; continue; }
        long long lo = total - qlen - (long long)A.win + 1;
        if (lo < 0) lo = 0;
        for (long long blk = lo / A.bs; blk <= (total - 1) / A.bs; ++blk) {
            if (blk >= A.maxb) { flag |= 16; break; }
            const int pg = A.bt[(long long)s * A.btp + blk * A.bts];
            if (pg < -1 || pg >= A.pages) flag |= 8;
        }
    }
    sh_flag[tid] = flag;
    __syncthreads();
    if (tid == 0) {
        int all = 0;
        for (int i = 0; i < 256; ++i) all |= sh_flag[i];
        if (r4d_verdict::inject_now(A.vc.inject, A.vc.gate)) all |= (int)r4d_verdict::kInjected;
        atomicExch(A.vc.gate, r4d_verdict::kGateOpen | (unsigned)all);   /* one block: it writes the whole word, so the launch starts from nothing older */
        if (all) r4d_verdict::raise(A.vc.log, (unsigned)all);
    }
}

__device__ __forceinline__ bool dspark_refused(const HcaDsparkArgs& A) {
    if (!A.check) return false;
    return r4d_verdict::gate_refused_block(A.vc);   /* one decision for the block (every thread reaches it: the early returns before it depend on blockIdx only) */
}

/* D == 512: cooperative 8-wave/head split-KV, wave 0 merges and writes. */
template <bool Vec>
__global__ __launch_bounds__(256) void ernie_hca_dspark_attn_d512_kernel(HcaDsparkArgs A) {
    __shared__ float sh_mx[8];
    __shared__ float sh_den[8];
    __shared__ float sh_acc[8][512];

    const int tid = (int)threadIdx.x;
    const int lane = tid & 31, wave = tid >> 5;
    const int h = (int)blockIdx.x, m = (int)blockIdx.y;
    if (h >= A.H || m >= A.M) return;
    if (dspark_refused(A)) return;   /* the validator kernel refused this launch: no store */

    const int s = dspark_find_seq(A, m);
    int total_kv = 0, lo = 0;
    if (s >= 0) {
        const int qlen = A.cu[(long long)(s + 1) * A.cus] - A.cu[(long long)s * A.cus];
        total_kv = A.seqused[(long long)s * A.sqs];
        const int anchor = total_kv - qlen;
        lo = anchor - A.win + 1;
        if (lo < 0) lo = 0;
    }

    const int c0 = lane * 8;
    const int c1 = 256 + lane * 8;

    float qv[16], acc[16];
    const long long qbase = (long long)m * A.qp + (long long)(h * 512) * A.qs;
    if (Vec) {
        const uint16_t* qp = A.q + (long long)m * A.qp + (long long)h * 512;
        unpack_uint4_bf16(*reinterpret_cast<const uint4*>(qp + c0), qv);
        unpack_uint4_bf16(*reinterpret_cast<const uint4*>(qp + c1), qv + 8);
    } else {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            qv[e]     = bf2f(A.q[qbase + (long long)(c0 + e) * A.qs]);
            qv[8 + e] = bf2f(A.q[qbase + (long long)(c1 + e) * A.qs]);
        }
    }
#pragma unroll
    for (int e = 0; e < 16; ++e) acc[e] = 0.0f;

    float mx = -INFINITY;
    float den = 0.0f;
    if (wave == 0 && s >= 0) {
        const float sk = A.sink[(long long)h * A.sks];
        mx = sk;
        den = (sk == -INFINITY) ? 0.0f : 1.0f;
    }

    auto attend = [&](const uint16_t* kptr, long long ks) {
        float kv[16];
        if (Vec) {
            unpack_uint4_bf16(*reinterpret_cast<const uint4*>(kptr + c0), kv);
            unpack_uint4_bf16(*reinterpret_cast<const uint4*>(kptr + c1), kv + 8);
        } else {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                kv[e]     = bf2f(kptr[(long long)(c0 + e) * ks]);
                kv[8 + e] = bf2f(kptr[(long long)(c1 + e) * ks]);
            }
        }
        float dot_l = 0.0f;
#pragma unroll
        for (int e = 0; e < 16; ++e) dot_l += qv[e] * kv[e];
        const float sc = wave_sum(dot_l) * A.scale;
        const float nmx = fmaxf(mx, sc);
        const float alpha = (mx == -INFINITY) ? 0.0f : expf(mx - nmx);
        const float w = expf(sc - nmx);
        den = den * alpha + w;
#pragma unroll
        for (int e = 0; e < 16; ++e) acc[e] = acc[e] * alpha + w * kv[e];
        mx = nmx;
    };

    /* Non-causal window KV pass, grouped by page, strided across the 8 waves. */
    int t = lo;
    while (t < total_kv) {
        const int blk = t / A.bs;
        const int off0 = t - blk * A.bs;
        const int tend = blk * A.bs + A.bs - 1;
        const int tstop = (tend < total_kv - 1) ? tend : total_kv - 1;
        if (blk < A.maxb) {
            const int pg = A.bt[(long long)s * A.btp + (long long)blk * A.bts];
            if (pg >= 0 && pg < A.pages) {
                const uint16_t* pg_ptr = A.kv + (long long)pg * A.kvp0;
                const int cnt = tstop - t;
                const int d0 = (wave + 8 - ((t - lo) & 7)) & 7;
                for (int d = d0; d <= cnt; d += 8)
                    attend(pg_ptr + (long long)(off0 + d) * A.kvp2, A.kvs);
            }
        }
        t = tstop + 1;
    }

    if (lane == 0) {
        sh_mx[wave] = mx;
        sh_den[wave] = den;
    }
#pragma unroll
    for (int e = 0; e < 16; ++e) sh_acc[wave][e * 32 + lane] = acc[e];
    __syncthreads();

    if (wave == 0) {
        float gmx = -INFINITY;
#pragma unroll
        for (int w = 0; w < 8; ++w)
            if (sh_den[w] > 0.0f) gmx = fmaxf(gmx, sh_mx[w]);
        float gden = 0.0f;
#pragma unroll
        for (int e = 0; e < 16; ++e) acc[e] = 0.0f;
        if (gmx > -INFINITY) {
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                const float dw = sh_den[w], mw = sh_mx[w];
                if (dw > 0.0f && mw > -INFINITY) {
                    const float sc = expf(mw - gmx);
                    gden += dw * sc;
#pragma unroll
                    for (int e = 0; e < 16; ++e) acc[e] += sh_acc[w][e * 32 + lane] * sc;
                }
            }
        }
        float res[16];
        if (gden > 0.0f) {
#pragma unroll
            for (int e = 0; e < 16; ++e) res[e] = acc[e] / gden;
        } else {
#pragma unroll
            for (int e = 0; e < 16; ++e) res[e] = 0.0f;
        }
        if (Vec) {
            uint16_t* op = A.out + (long long)m * A.outp + (long long)h * 512;
            *reinterpret_cast<uint4*>(op + c0) = pack_bf16_uint4(res);
            *reinterpret_cast<uint4*>(op + c1) = pack_bf16_uint4(res + 8);
        } else {
            const long long obase = (long long)m * A.outp + (long long)(h * 512) * A.outs;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                A.out[obase + (long long)(c0 + e) * A.outs] = f2bf(res[e]);
                A.out[obase + (long long)(c1 + e) * A.outs] = f2bf(res[8 + e]);
            }
        }
    }
}

/* General path: arbitrary D in 1..4096, all 256 threads cooperate on each key via wave+LDS dot reduction. */
__global__ __launch_bounds__(256) void ernie_hca_dspark_attn_gen_kernel(HcaDsparkArgs A) {
    __shared__ float sh_wave[8];
    __shared__ float sh_sc;

    const int tid = (int)threadIdx.x;
    const int lane = tid & 31, wave = tid >> 5;
    const int h = (int)blockIdx.x, m = (int)blockIdx.y;
    if (h >= A.H || m >= A.M) return;
    if (dspark_refused(A)) return;   /* the validator kernel refused this launch: no store */

    const int s = dspark_find_seq(A, m);
    int total_kv = 0, lo = 0;
    if (s >= 0) {
        const int qlen = A.cu[(long long)(s + 1) * A.cus] - A.cu[(long long)s * A.cus];
        total_kv = A.seqused[(long long)s * A.sqs];
        const int anchor = total_kv - qlen;
        lo = anchor - A.win + 1;
        if (lo < 0) lo = 0;
    }

    float qv[16], acc[16];
#pragma unroll
    for (int e = 0; e < 16; ++e) {
        const int d = tid + (e << 8);
        qv[e] = (d < A.D) ? bf2f(A.q[(long long)m * A.qp + (long long)(h * A.D + d) * A.qs]) : 0.0f;
        acc[e] = 0.0f;
    }

    float mx = -INFINITY, den = 0.0f;
    if (s >= 0) {
        const float sk = A.sink[(long long)h * A.sks];
        mx = sk;
        den = (sk == -INFINITY) ? 0.0f : 1.0f;
    }

    for (int t = lo; t < total_kv; ++t) {
        const int blk = t / A.bs;
        if (blk >= A.maxb) continue;
        const int pg = A.bt[(long long)s * A.btp + (long long)blk * A.bts];
        if (pg < 0 || pg >= A.pages) continue;
        const uint16_t* kp = A.kv + (long long)pg * A.kvp0 + (long long)(t - blk * A.bs) * A.kvp2;
        float kv[16], dot = 0.0f;
#pragma unroll
        for (int e = 0; e < 16; ++e) {
            const int d = tid + (e << 8);
            kv[e] = (d < A.D) ? bf2f(kp[(long long)d * A.kvs]) : 0.0f;
            dot += qv[e] * kv[e];
        }
        const float wsum = wave_sum(dot);
        if (lane == 0) sh_wave[wave] = wsum;
        __syncthreads();
        if (tid == 0) {
            float tot = 0.0f;
#pragma unroll
            for (int w = 0; w < 8; ++w) tot += sh_wave[w];
            sh_sc = tot * A.scale;
        }
        __syncthreads();
        const float sc = sh_sc;
        const float nmx = fmaxf(mx, sc);
        const float alpha = (mx == -INFINITY) ? 0.0f : expf(mx - nmx);
        const float w = expf(sc - nmx);
        den = den * alpha + w;
#pragma unroll
        for (int e = 0; e < 16; ++e) acc[e] = acc[e] * alpha + w * kv[e];
        mx = nmx;
    }

    const long long obase = (long long)m * A.outp + (long long)h * A.D * A.outs;
#pragma unroll
    for (int e = 0; e < 16; ++e) {
        const int d = tid + (e << 8);
        if (d < A.D) A.out[obase + (long long)d * A.outs] = f2bf(den > 0.0f ? acc[e] / den : 0.0f);
    }
}

/* R4D_DSPARK_VALIDATE: bit 1 host scan of the metadata operands (host builds only), bit 2 validator kernel. See r4d_rad_ernie_hca_dspark_attn. Where a launch
 * completes before it returns (R4D_SYNC_LAUNCH, the emulator) the variable is read at every call so a test can toggle it; a card build reads it once. */
static inline int dspark_validate_bits() {
    /* Default: the host scan where the operands are host memory (not card form), the validator kernel where they are not (card form). R4D_ERNIE_VALIDATE=0 turns every
     * validator off (a benchmark switch). */
    if (!r4d_verdict::master_on()) return 0;
    const int dflt = r4d_verdict::card_form() ? 2 : 1;
#ifdef R4D_SYNC_LAUNCH
    const char* e = std::getenv("R4D_DSPARK_VALIDATE");
    return e && *e ? std::atoi(e) : dflt;
#else
    static const int v = [dflt] { const char* e = std::getenv("R4D_DSPARK_VALIDATE"); return e && *e ? std::atoi(e) : dflt; }();
    return v;
#endif
}

}  /* namespace r4d_ernie_dspark */

int r4d_rad_ernie_mhc_stream_mean(const RadArgs* a, RadStream stream) {
    using namespace r4d_ernie;
    using namespace r4d_ernie_dspark;
    if (!a || !a->t || a->n_t < 2) return R4D_NO(RAD_E_INVAL);
    const RadTensor &x = a->t[0], &out = a->t[1];
    const long long H = r4d_geti_or(a, "hidden", 0), mult = r4d_geti_or(a, "mult", 0);
    if (H < 1 || H > 65535 || mult < 1 || mult > 16) return R4D_NO(RAD_E_SHAPE);
    if (x.rank != 2 || out.rank != 2 || x.shape[0] < 1 || x.shape[0] > 65535 ||
        x.shape[1] != mult * H || out.shape[0] != x.shape[0] || out.shape[1] != H ||
        !pos_strides(x) || !pos_strides(out) || out.stride[0] < out.shape[1] * out.stride[1])
        return R4D_NO(RAD_E_SHAPE);
    if (x.dtype != RAD_BF16 || out.dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (!x.data || !out.data) return R4D_NO(RAD_E_INVAL);

    const long long M = x.shape[0];
    R4D_ARGS(MHCStreamMeanArgs, P);
    P.x = (const uint16_t*)x.data;
    P.out = (uint16_t*)out.data;
    P.xp = x.stride[0]; P.xs = x.stride[1];
    P.outp = out.stride[0]; P.outs = out.stride[1];
    P.M = (int)M; P.H = (int)H; P.mult = (int)mult;

    if (mult == 4 && (H & 7) == 0 && P.xs == 1 && P.outs == 1 &&
        ((((uintptr_t)P.x | (uintptr_t)P.out) & 15u) == 0u) && (P.xp & 7) == 0 && (P.outp & 7) == 0) {
        const dim3 grid((unsigned)((H / 8 + 255) / 256), (unsigned)M);
        R4D_KLAUNCH(R4D_KNAME(ernie_mhc_stream_mean_vec_kernel), grid, dim3(256), (hipStream_t)stream, P);
    } else {
        const dim3 grid((unsigned)((H + 255) / 256), (unsigned)M);
        R4D_KLAUNCH(R4D_KNAME(ernie_mhc_stream_mean_kernel), grid, dim3(256), (hipStream_t)stream, P);
    }
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

int r4d_rad_ernie_hca_dspark_attn(const RadArgs* a, RadStream stream) {
    using namespace r4d_ernie;
    using namespace r4d_ernie_dspark;
    if (!a || !a->t || a->n_t < 7) return R4D_NO(RAD_E_INVAL);
    const RadTensor &q = a->t[0], &kv = a->t[1], &bt = a->t[2], &seqused = a->t[3],
                    &cu = a->t[4], &sink = a->t[5], &out = a->t[6];
    const long long H = r4d_geti_or(a, "heads", 0), D = r4d_geti_or(a, "head_dim", 0),
                    win = r4d_geti_or(a, "window", 128);
    const double scale = r4d_getf(a, "scale", 0.0);
    if (H < 1 || H > 65535 || D < 1 || D > 4096 || win < 1 || win > INT32_MAX / 2 ||
        !std::isfinite(scale) || scale <= 0.0) return R4D_NO(RAD_E_SHAPE);
    if (q.rank != 2 || kv.rank != 4 || bt.rank != 2 || seqused.rank != 1 || cu.rank != 1 ||
        sink.rank != 1 || out.rank != 2) return R4D_NO(RAD_E_SHAPE);
    const long long M = q.shape[0], S = bt.shape[0], bs = kv.shape[2], pages = kv.shape[0], maxb = bt.shape[1];
    if (M < 1 || M > 65535 || S < 1 || S > (1LL << 20) || bs < 1 || pages < 1 || maxb < 1)
        return R4D_NO(RAD_E_SHAPE);
    if (q.shape[1] != H * D || kv.shape[1] != 1 || kv.shape[3] != D ||
        seqused.shape[0] != S || cu.shape[0] != S + 1 || sink.shape[0] != H ||
        out.shape[0] != M || out.shape[1] != H * D) return R4D_NO(RAD_E_SHAPE);
    if (!pos_strides(q) || !pos_strides(kv) || !pos_strides(bt) || !pos_strides(seqused) ||
        !pos_strides(cu) || !pos_strides(sink) || !pos_strides(out) ||
        out.stride[0] < out.shape[1] * out.stride[1]) return R4D_NO(RAD_E_SHAPE);
    if (q.dtype != RAD_BF16 || kv.dtype != RAD_BF16 || out.dtype != RAD_BF16 ||
        bt.dtype != RAD_I32 || seqused.dtype != RAD_I32 || cu.dtype != RAD_I32 || sink.dtype != RAD_F32)
        return R4D_NO(RAD_E_DTYPE);
    if (!q.data || !kv.data || !bt.data || !seqused.data || !cu.data || !sink.data || !out.data)
        return R4D_NO(RAD_E_INVAL);

    R4D_ARGS(HcaDsparkArgs, P);
    P.q = (const uint16_t*)q.data; P.kv = (const uint16_t*)kv.data;
    P.bt = (const int32_t*)bt.data; P.seqused = (const int32_t*)seqused.data;
    P.cu = (const int32_t*)cu.data; P.sink = (const float*)sink.data;
    P.out = (uint16_t*)out.data;
    P.qp = q.stride[0]; P.qs = q.stride[1];
    P.kvp0 = kv.stride[0]; P.kvp2 = kv.stride[2]; P.kvs = kv.stride[3];
    P.btp = bt.stride[0]; P.bts = bt.stride[1];
    P.sqs = seqused.stride[0]; P.cus = cu.stride[0]; P.sks = sink.stride[0];
    P.outp = out.stride[0]; P.outs = out.stride[1];
    P.M = (int)M; P.H = (int)H; P.D = (int)D;
    P.S = (int)S; P.win = (int)win; P.bs = (int)bs; P.maxb = (int)maxb; P.pages = (int)pages;
    P.scale = (float)scale;

    /* libref refuses before its first store on the contract of seqused, cu_seqlens, the sink and the window's page indices. Two forms here, chosen by
     * R4D_DSPARK_VALIDATE (a bit set; default 1 where the operands are host memory -- the host builds and the emulator -- and 0 on a card build):
     *   bit 1 (host scan): read the metadata operands on the host and refuse with RAD_E_SHAPE. Compiled out of a HIP build, whose operands are device
     *         memory the host cannot read without a synchronisation (the same arrangement as ernie_hca_state_write and ernie_hca_gather_slots).
     *   bit 2 (validator kernel): a one-block kernel ahead of the attention kernel; a violation leaves every attention block without a store, and is
     *         latched for r4d_ernie_dspark_take_error(). Where a launch completes before it returns (the emulator, R4D_SYNC_LAUNCH) the launcher also
     *         returns RAD_E_SHAPE. Cost on a card: one extra one-block launch of a few microseconds per drafter layer; UNMEASURED, so off by default. */
    const int vmode = dspark_validate_bits();
#if !defined(__HIP__) && !defined(__HIP_PLATFORM_AMD__)
    if (vmode & 1) {
        const int32_t* bt32 = (const int32_t*)bt.data;
        const int32_t* su32 = (const int32_t*)seqused.data;
        const int32_t* cu32 = (const int32_t*)cu.data;
        const float* sk32 = (const float*)sink.data;
        for (long long h = 0; h < H; ++h) {
            const float v = sk32[h * sink.stride[0]];
            if (std::isnan(v) || v == INFINITY) return R4D_NO(RAD_E_SHAPE);
        }
        if (cu32[0] != 0) return R4D_NO(RAD_E_SHAPE);
        for (long long s = 0; s < S; ++s)
            if (cu32[s * cu.stride[0]] > cu32[(s + 1) * cu.stride[0]]) return R4D_NO(RAD_E_SHAPE);
        if (cu32[S * cu.stride[0]] != M) return R4D_NO(RAD_E_SHAPE);
        for (long long s = 0; s < S; ++s) {
            const long long qlen = (long long)cu32[(s + 1) * cu.stride[0]] - cu32[s * cu.stride[0]];
            if (qlen == 0) continue;
            const long long total = su32[s * seqused.stride[0]];
            if (total < qlen) return R4D_NO(RAD_E_SHAPE);
            const long long lo = std::max<long long>(0, total - qlen - win + 1);
            for (long long blk = lo / bs; blk <= (total - 1) / bs; ++blk) {
                if (blk >= maxb) return R4D_NO(RAD_E_SHAPE);
                const long long pg = bt32[s * bt.stride[0] + blk * bt.stride[1]];
                if (pg < -1 || pg >= pages) return R4D_NO(RAD_E_SHAPE);
            }
        }
    }
#endif
    P.check = (vmode & 2) ? 1 : 0;
    {
        const r4d_verdict::Launch vl = P.check ? r4d_verdict::acquire_gated(r4d_verdict::kFamDspark, P.q, P.kv, P.bt, P.seqused, P.cu, P.sink, P.out) : r4d_verdict::acquire(r4d_verdict::kFamDspark);
        if (!vl.ok) return RAD_E_DEVICE;
        P.vc = vl.vc;
    }
    if (P.check) {
        r4d_verdict::stats().validator_launches.fetch_add(1, std::memory_order_relaxed);
        R4D_KLAUNCH(R4D_KNAME(ernie_hca_dspark_validate_kernel), dim3(1), dim3(256), (hipStream_t)stream, P);
    }

    const dim3 grid((unsigned)H, (unsigned)M);
    if (D == 512) {
        const bool vec = (P.qs == 1 && P.kvs == 1 && P.outs == 1 &&
                          ((((uintptr_t)P.q | (uintptr_t)P.kv | (uintptr_t)P.out) & 15u) == 0u) &&
                          (P.qp & 7) == 0 && (P.kvp0 & 7) == 0 && (P.kvp2 & 7) == 0 && (P.outp & 7) == 0);
        if (vec) {
            R4D_KLAUNCH(R4D_KNAME(ernie_hca_dspark_attn_d512_kernel<true>), grid, dim3(256), (hipStream_t)stream, P);
        } else {
            R4D_KLAUNCH(R4D_KNAME(ernie_hca_dspark_attn_d512_kernel<false>), grid, dim3(256), (hipStream_t)stream, P);
        }
    } else {
        R4D_KLAUNCH(R4D_KNAME(ernie_hca_dspark_attn_gen_kernel), grid, dim3(256), (hipStream_t)stream, P);
    }
    if (hipGetLastError() != hipSuccess) return RAD_E_DEVICE;
#ifdef R4D_SYNC_LAUNCH
    /* Where the launch has completed and the operands are host memory (not card form), the launcher returns libref's refusal itself; in card form it is asynchronous
     * like a card's launch, and the refusal is in the verdict the engine polls. */
    if (P.check && !r4d_verdict::card_form()) {
        const unsigned w = *(const volatile unsigned*)P.vc.gate;   /* the emulator: the operands and the state are host memory */
        if (w & ~r4d_verdict::kGateOpen) return R4D_NO(RAD_E_SHAPE);
    }
#endif
    return RAD_OK;
}

/* The events the validator kernel raised since the last call (their bits OR-ed; bits as in the validator, 0x40000000 if some were not readable), 0 if every launch
 * since was valid or the validators are off (R4D_ERNIE_VALIDATE=0). The log is never cleared; the host remembers what it took. -1 if the copy failed. */
extern "C" int r4d_ernie_dspark_take_error(void) {
    return r4d_verdict::registry().take(r4d_verdict::kFamDspark);
}
