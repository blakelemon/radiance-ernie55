/* r4d_ernie_hca.hip -- ERNIE 5.5 Flash HCA compressor & attention device ops (story 4-3e):
 *   ernie_hca_state_write, ernie_hca_gather_slots, ernie_hca_compress, ernie_hca_union_attn, ernie_hca_wo_a.
 * Written cleanly from ERNIE 5.5 Flash equations and libref/ref_ernie_hca.cpp.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cstddef>
#include <cstdlib>
#include "r4d_ernie_common.h"
#include "r4d_ernie_verdict.h"
#include "r4d_plugin.h"

namespace r4d_ernie_hca {

/* What libref refuses an ERNIE HCA op for -- an index outside what it accepts (a state slot, an out slot, a block table entry), a negative position of a live row, a
 * gather entry, a completing compress row with every gather entry masked, a non-finite pooled row, a sink that is NaN or +inf, a union row whose request, origin or
 * table entries are outside the contract -- is a refusal before the first store. A card launch cannot refuse (its operands are device memory and it is
 * asynchronous). The kernels here detect the violation on the device and do three things: the offending row's stores are skipped (no out-of-range access, no
 * state or pool row written from a row libref would have refused), one lane raises an event into the HCA log of the library's device state (r4d_ernie_verdict.h; r4d_ernie_hca_take_error() hands the events since its
 * last call to the host; the engine polls at the step's collect), and union attention is gated: ernie_hca_union_validate_kernel runs ahead of it on the same
 * stream (after ernie_hca_gate_open_kernel, which opens the call site's gate word) and leaves its verdict in that word (r4d_verdict::gate_add), which every attention block of that launch reads before its first store.
 * Event bits: 1 an index outside [-1, pool) (state slot, out slot), 2 a negative position, 4 a gather entry outside [-1, pool), 8 a completing compress row whose
 * gather list is all masked, 16 a non-finite pooled row, 32 a sink that is NaN or +inf, 64 a union row whose position, request or origin libref refuses,
 * 128 a block table entry outside [-1, pool) or past the table width. */
#define R4D_HCA_RAISE(bits) r4d_verdict::raise(A.vc.log, (unsigned)(bits))
#define R4D_HCA_PAGE_LATCH(pg, pages) do { if ((pg) < -1 || (pg) >= (pages)) R4D_HCA_RAISE(128); } while (0)

/* Where the validator kernel of union attention runs: a card launch (the host cannot read the tables), unless R4D_ERNIE_VALIDATE=0 or R4D_HCA_VALIDATE=0. Read at
 * every call where a launch completes before it returns (R4D_SYNC_LAUNCH), once on a card build. */
static inline bool hca_gate_on() {
    if (!r4d_verdict::master_on() || !r4d_verdict::card_form()) return false;
#ifdef R4D_SYNC_LAUNCH
    const char* e = std::getenv("R4D_HCA_VALIDATE");
    return !(e && *e == '0');
#else
    static const bool on = [] { const char* e = std::getenv("R4D_HCA_VALIDATE"); return !(e && *e == '0'); }();
    return on;
#endif
}

#if !defined(__HIP__) && !defined(__HIP_PLATFORM_AMD__)
/* R4D_HCA_VALIDATE (default on): 0 skips the host scans of the launchers below (state_write, compress, union attention). Read at every call where a launch completes before it returns (R4D_SYNC_LAUNCH). */
static inline bool r4d_hca_scan_on() {
#ifdef R4D_SYNC_LAUNCH
    const char* e = std::getenv("R4D_HCA_VALIDATE");
    return !(e && *e == '0') && !r4d_verdict::card_form();
#else
    static const bool on = [] { const char* e = std::getenv("R4D_HCA_VALIDATE"); return !(e && *e == '0') && !r4d_verdict::card_form(); }();
    return on;
#endif
}
#endif


using namespace r4d_ernie;

static inline bool legal_ratio(long long r) { return r == 4 || r == 128; }

static inline bool ck_rank4_bad(const RadArgs* a) { return a->n_t < 3 || a->t[2].rank != 4; }

static inline bool block_hint_ok(const RadArgs* a, const RadTensor& pool) {
    if (!find_p(a, "block_size")) return true;
    const long long b = r4d_geti_or(a, "block_size", 0);
    return b >= 1 && (pool.rank != 4 || pool.shape[2] == b);
}

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

__device__ __forceinline__ float dot_uint4_bf16(uint4 a, uint4 b) {
    return bf2f((uint16_t)(a.x & 0xffffu)) * bf2f((uint16_t)(b.x & 0xffffu)) +
           bf2f((uint16_t)(a.x >> 16))     * bf2f((uint16_t)(b.x >> 16)) +
           bf2f((uint16_t)(a.y & 0xffffu)) * bf2f((uint16_t)(b.y & 0xffffu)) +
           bf2f((uint16_t)(a.y >> 16))     * bf2f((uint16_t)(b.y >> 16)) +
           bf2f((uint16_t)(a.z & 0xffffu)) * bf2f((uint16_t)(b.z & 0xffffu)) +
           bf2f((uint16_t)(a.z >> 16))     * bf2f((uint16_t)(b.z >> 16)) +
           bf2f((uint16_t)(a.w & 0xffffu)) * bf2f((uint16_t)(b.w & 0xffffu)) +
           bf2f((uint16_t)(a.w >> 16))     * bf2f((uint16_t)(b.w >> 16));
}

/* ------------------------------------------------------------------ ernie_hca_state_write */
struct HcaStateWriteArgs {
    const float *kv, *sc, *ape;
    const int32_t *pos, *slot;
    float *st;
    long long kvp, kvs, scp, scs, apep, apes, ps, sls, stp, sts;
    int M, W, ratio, S;
    int* dd;     /* a large launch (M > kDedupeRows) with scratch: the last-writer table of the launch (see "DUPLICATE SLOTS" below); null: the block scans the later rows */
    int ddshift; /* 32 - log2(table capacity) */
    r4d_verdict::Vc vc;   /* this launch's event log and gate word (device memory the library owns; constant across replays) */
};
static_assert(sizeof(HcaStateWriteArgs) <= 512, "HcaStateWriteArgs exceeds AQL kernarg limit");

/* DUPLICATE SLOTS AT PREFILL SIZES (review m4, H1). libref stores the rows in order, a repeated slot keeps the last row; the blocks of the parallel kernels run in no promised order.
 * Up to kDedupeRows rows a block simply scans the rows after it (cheap, and it stops at the first later row with its slot). Past it that scan is O(M^2) a layer, so the launcher
 * first finds each slot's last writer from the M rows of the launch alone: an open-addressing table of `cap` = the power of two >= 2M entries (keys[cap] then rows[cap], 8 bytes an
 * entry), cleared to -1, filled by atomicCAS on the key and atomicMax on the row, and read by the store block of row m, which stores only if the table's row for its slot is m.
 * The table is the OP'S SCRATCH (ernie_hca_*_scratch, the RadScratchFn of the rows): the arena budgets it and overlays it with other ops' scratch, its size depends on M only
 * (never on the pool), its address is the arena's (constant across the replays of a pass), and the op writes every entry it reads. The last row is the maximum row of the slot,
 * so the result does not depend on the order of the atomics: the same values as the scan. Without scratch (a caller that supplied none) the launcher keeps the scan. */
constexpr int kDedupeRows = 256;
/* The capacity of the table for M rows (a power of two, load <= 1/2), its log, and the scratch bytes. */
static inline int dedupe_log2(long long M) { int k = 1; while ((1ll << k) < 2 * M && k < 30) ++k; return k; }
static inline long long dedupe_bytes(long long M) { return (8ll << dedupe_log2(M)); }
struct HcaDedupeArgs { int* tab; int cap; };
__global__ void ernie_hca_dedupe_clear_kernel(HcaDedupeArgs A) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < 2 * A.cap) A.tab[i] = -1;
}
__device__ __forceinline__ unsigned dd_home(int key, int shift) { return ((unsigned)key * 2654435761u) >> shift; }
/* Row `row` competes for `key` (>= 0). */
__device__ __forceinline__ void dd_mark(int* tab, int shift, int key, int row) {
    const unsigned mask = (1u << (32 - shift)) - 1u;
    unsigned i = dd_home(key, shift);
    for (unsigned n = 0; n <= mask; ++n, i = (i + 1u) & mask) {
        const int old = atomicCAS(&tab[i], -1, key);
        if (old == -1 || old == key) { atomicMax(&tab[mask + 1u + i], row); return; }
    }
}
/* The last row that competed for `key`; -2 if none did. */
__device__ __forceinline__ int dd_last(const int* tab, int shift, int key) {
    const unsigned mask = (1u << (32 - shift)) - 1u;
    unsigned i = dd_home(key, shift);
    for (unsigned n = 0; n <= mask; ++n, i = (i + 1u) & mask) {
        const int k = tab[i];
        if (k == key) return tab[mask + 1u + i];
        if (k == -1) return -2;
    }
    return -2;
}

/* Rows are applied in order per column (one thread owns a column of every row), so a repeated slot keeps the last row, as libref does. A row libref refuses (a slot
 * outside [-1, pool), a negative position of a live row) is not stored; the thread of column 0 raises it. */
__global__ void ernie_hca_state_write_seq_kernel(HcaStateWriteArgs A) {
    const int c = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (c >= A.W) return;
    for (int m = 0; m < A.M; ++m) {
        const int s = A.slot[(long long)m * A.sls];
        if (s < -1 || s >= A.S) { if (c == 0) R4D_HCA_RAISE(1); continue; }
        if (s < 0) continue;
        const int pp = A.pos[(long long)m * A.ps];
        if (pp < 0) { if (c == 0) R4D_HCA_RAISE(2); continue; }   /* a negative % would index before ape */
        const int ar = pp % A.ratio;
        const float k = A.kv[(long long)m * A.kvp + (long long)c * A.kvs];
        const float x = A.sc[(long long)m * A.scp + (long long)c * A.scs];
        const float b = A.ape[(long long)ar * A.apep + (long long)c * A.apes];
        float* row = A.st + (long long)s * A.stp;
        row[(long long)c * A.sts] = k;
        row[(long long)(A.W + c) * A.sts] = x + b;
    }
}

/* One block per (column chunk, row): the blocks of a grid run in no promised order, so a repeated slot would keep whichever row ran last. The block of row m
 * stores only if no later row has the same slot (the block scans the rows after it): libref's last-row-wins, whatever the order of the blocks. */
__global__ void ernie_hca_state_write_par_kernel(HcaStateWriteArgs A) {
    __shared__ int sh_dup;
    const int tid = (int)threadIdx.x;
    const int c = (int)(blockIdx.x * blockDim.x) + tid;
    const int m = (int)blockIdx.y;
    if (m >= A.M) return;
    const bool lead = blockIdx.x == 0 && tid == 0;
    const int s = A.slot[(long long)m * A.sls];
    if (s < -1 || s >= A.S) { if (lead) R4D_HCA_RAISE(1); return; }
    if (s < 0) return;
    const int pp = A.pos[(long long)m * A.ps];
    if (pp < 0) { if (lead) R4D_HCA_RAISE(2); return; }
    if (A.dd) {
        if (dd_last(A.dd, A.ddshift, s) != m) return;   /* a later row writes this slot (the same value for every thread of the block) */
    } else {
        if (tid == 0) sh_dup = 0;
        __syncthreads();
        for (int m2 = m + 1 + tid; m2 < A.M && !sh_dup; m2 += 256)   /* stops at the first later row that stores this slot; a position-addressed state has none, so a valid launch scans to its end */
            if (A.slot[(long long)m2 * A.sls] == s && A.pos[(long long)m2 * A.ps] >= 0) sh_dup = 1;   /* a refused row (negative position) stores nothing and does not take the place of this one: the rule of the last-writer pass */
        __syncthreads();
        if (sh_dup) return;
    }
    if (c >= A.W) return;
    const int ar = pp % A.ratio;
    const float k = A.kv[(long long)m * A.kvp + (long long)c * A.kvs];
    const float x = A.sc[(long long)m * A.scp + (long long)c * A.scs];
    const float b = A.ape[(long long)ar * A.apep + (long long)c * A.apes];
    float* row = A.st + (long long)s * A.stp;
    row[(long long)c * A.sts] = k;
    row[(long long)(A.W + c) * A.sts] = x + b;
}

__global__ void ernie_hca_state_last_mark_kernel(HcaStateWriteArgs A) {
    const int m = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (m >= A.M) return;
    const int s = A.slot[(long long)m * A.sls];
    if (s >= 0 && s < A.S && A.pos[(long long)m * A.ps] >= 0) dd_mark(A.dd, A.ddshift, s, m);   /* only rows that store compete: a refused row stores nothing */
}

/* ------------------------------------------------------------------ ernie_hca_gather_slots */
struct HcaGatherArgs {
    const int32_t *pos, *req, *bt, *origin;
    int32_t *out;
    long long ps, rs, btp, bts, os, outp, outs;
    int M, S, B, ratio, bs, Wn;
    r4d_verdict::Vc vc;   /* this launch's event log and gate word (device memory the library owns; constant across replays) */
};
static_assert(sizeof(HcaGatherArgs) <= 512, "HcaGatherArgs exceeds AQL kernarg limit");

/* libref's hca_gather_slots, row by row: a position below 0, a request outside [0, S), an origin that is negative, not a multiple of the block size or past the
 * position, a completing row whose window reaches a block before the origin or past the table, a page below 0 (an unallocated page where the window needs one) or
 * one whose slot would overflow int32 refuse the launch. Returns the bits of the first violation of the row m (4), 0 if the row is valid; every thread of the row
 * computes it (at most Wn table reads), so a row libref refuses is not stored by any of its threads. */
__device__ __forceinline__ int hca_gather_row_bad(const HcaGatherArgs& A, int m) {
    const long long p = A.pos[(long long)m * A.ps], r = A.req[(long long)m * A.rs];
    if (p < 0 || r < 0 || r >= A.S) return 4;
    const long long first = A.origin ? A.origin[r * A.os] : 0;
    if (first < 0 || first % A.bs || first > p) return 4;
    if ((p + 1) % A.ratio) return 0;
    for (int w = 0; w < A.Wn; ++w) {
        const long long absolute = p - A.Wn + 1 + w;
        if (absolute < 0) continue;
        const long long q = absolute - first;
        if (q < 0 || q / A.bs >= A.B) return 4;
        const long long pg = A.bt[r * A.btp + (q / A.bs) * A.bts];
        if (pg < 0 || pg > (INT32_MAX - q % A.bs) / A.bs) return 4;
    }
    return 0;
}

__global__ void ernie_hca_gather_slots_kernel(HcaGatherArgs A) {
    const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = A.M * A.Wn;
    if (idx >= total) return;
    const int m = idx / A.Wn, w = idx % A.Wn;
    if (!A.vc.log) {   /* validation off (a card's default): base d2258ef8 verbatim, a bad entry stores -1, nothing is skipped */
        const int p0 = A.pos[(long long)m * A.ps];
        const int r0 = A.req[(long long)m * A.rs];
        int32_t v0 = -1;
        if (p0 >= 0 && r0 >= 0 && r0 < A.S && ((p0 + 1ll) % A.ratio == 0)) {
            const int first = A.origin ? A.origin[(long long)r0 * A.os] : 0;
            const int start = p0 - A.Wn + 1;
            const int absolute = start + w;
            if (absolute >= first && (first == 0 || (first > 0 && first <= start && first % A.bs == 0))) {
                const int q = absolute - first;
                const int b = q / A.bs;
                if (b < A.B) {
                    const int pg = A.bt[(long long)r0 * A.btp + (long long)b * A.bts];
                    const int off = q % A.bs;
                    if (pg >= 0 && pg <= (INT32_MAX - off) / A.bs) v0 = pg * A.bs + off;
                }
            }
        }
        A.out[(long long)m * A.outp + (long long)w * A.outs] = v0;
        return;
    }
    if (hca_gather_row_bad(A, m)) { if (w == 0) R4D_HCA_RAISE(4); return; }
    const long long p = A.pos[(long long)m * A.ps];
    int32_t val = -1;
    if ((p + 1) % A.ratio == 0) {
        const long long r = A.req[(long long)m * A.rs];
        const long long first = A.origin ? A.origin[r * A.os] : 0;
        const long long absolute = p - A.Wn + 1 + w;
        if (absolute >= 0) {
            const long long q = absolute - first;
            const long long pg = A.bt[r * A.btp + (q / A.bs) * A.bts];
            val = (int32_t)(pg * A.bs + q % A.bs);
        }
    }
    A.out[(long long)m * A.outp + (long long)w * A.outs] = val;
}

/* ------------------------------------------------------------------ ernie_hca_compress */
struct HcaCompressArgs {
    const float *st;
    const int32_t *gs, *pos, *os;
    const uint16_t *nw;
    void *out;
    long long stp, sts, gsp, gss, ps, oss, nws, outp, outs;
    int M, S, Rr, ratio, D, R, qb, packed, coff, W, Wn;
    int div;   /* out slots per out row: 1 (token layout), `ratio` (cmp_layout block: one row per block, page = slot / ratio) */
    float eps;
    RopeInvTable tab;
    int* dd;     /* a large launch (M > kDedupeRows) with scratch: the last-writer table of the launch (see "DUPLICATE SLOTS"); null: the block scans the later rows */
    int ddshift; /* 32 - log2(table capacity) */
    r4d_verdict::Vc vc;   /* this launch's event log and gate word (device memory the library owns; constant across replays) */
};
static_assert(sizeof(HcaCompressArgs) <= 512, "HcaCompressArgs exceeds AQL kernarg limit");

__device__ __forceinline__ bool hca_bf_finite(float v) { return (f2bf(v) & 0x7f80u) != 0x7f80u; }

/* One row of the pooling. Everything libref refuses a launch for is judged here, before the row's first store, by every thread of the block (the same operands, the
 * same answer, so the early returns are uniform): an out slot outside [-1, pool) (-1 ignores the row altogether), a negative position, a gather entry outside
 * [-1, pool) in ANY live row (a row that does not complete included), a completing row whose gather list is entirely masked, and a pooled row that is not finite
 * once rounded to bf16 (judged on exactly the nope and rope values that are stored). A violation raises one event (thread 0) and the row stores nothing.
 * scan_dups (the one-block-per-row kernel): the blocks of a grid run in no promised order, so the block of row m stores only if no LATER completing row has the same
 * out slot (libref stores the staged rows in order, a repeated slot keeps the last). sh_flag is two shared ints. */
__device__ void hca_compress_one_row(const HcaCompressArgs& A, int m, float* sh_y, float* smem, int* sh_flag, bool scan_dups) {
    const int tid = (int)threadIdx.x;
    const int slot = A.os[(long long)m * A.oss];   /* the out slot; in the block layout (A.div > 1) a pool row of a block, o = slot / div is its page */
    if (slot < -1 || slot >= A.Rr) { if (tid == 0) R4D_HCA_RAISE(1); return; }
    if (slot < 0) return;
    const int p = A.pos[(long long)m * A.ps];
    if (p < 0) { if (tid == 0) R4D_HCA_RAISE(2); return; }
    int unmasked = 0, bad_gather = 0;
    if (A.vc.log) {   /* validation on; off, the row is judged by nothing and a gather entry outside the pool is skipped, as before the verdict series */
        for (int w = 0; w < A.Wn; ++w) {
            const int s = A.gs[(long long)m * A.gsp + (long long)w * A.gss];
            if (s < -1 || s >= A.S) bad_gather = 1;
            unmasked += s >= 0;
        }
        if (bad_gather) { if (tid == 0) R4D_HCA_RAISE(4); return; }
    }
    if ((p + 1) % A.ratio != 0) return;
    if (A.vc.log && !unmasked) { if (tid == 0) R4D_HCA_RAISE(8); return; }   /* validation off: the row pools to 0, as base */
    /* block layout: a block row is written by its last token. libref refuses a completing row at any other offset before its first store; the kernel stores nothing and raises. */
    if (A.div > 1 && slot % A.div != A.div - 1) { if (tid == 0) R4D_HCA_RAISE(256); return; }
    const int o = slot / A.div;

    if (tid == 0) { sh_flag[0] = 0; sh_flag[1] = 0; }
    __syncthreads();
    for (int c = tid; c < A.D; c += 256) {
        float mx = -INFINITY;
        for (int w = 0; w < A.Wn; ++w) {
            const int s = A.gs[(long long)m * A.gsp + (long long)w * A.gss];
            R4D_HCA_PAGE_LATCH(s, A.S);   /* a state index at or past the pool is a refusal in libref; the card skips it (no out-of-range read) and latches */
            if (s < 0 || s >= A.S) continue;
            const int off = (A.coff == 2 && w >= A.ratio) ? A.D : 0;
            const float sc = A.st[(long long)s * A.stp + (long long)(A.W + off + c) * A.sts];
            if (sc > mx) mx = sc;
        }
        float num = 0.0f, den = 0.0f;
        for (int w = 0; w < A.Wn; ++w) {
            const int s = A.gs[(long long)m * A.gsp + (long long)w * A.gss];
            R4D_HCA_PAGE_LATCH(s, A.S);   /* a state index at or past the pool is a refusal in libref; the card skips it (no out-of-range read) and latches */
            if (s < 0 || s >= A.S) continue;
            const int off = (A.coff == 2 && w >= A.ratio) ? A.D : 0;
            const float sc = A.st[(long long)s * A.stp + (long long)(A.W + off + c) * A.sts];
            const float k = A.st[(long long)s * A.stp + (long long)(off + c) * A.sts];
            const float ex = expf(sc - mx);
            num += ex * k;
            den += ex;
        }
        /* as libref: with a live entry den >= 1 unless a value is NaN, and then the row is non-finite and refused below; validation off: base's guard */
        sh_y[c] = (A.vc.log || den > 0.0f) ? (num / den) : 0.0f;
    }
    __syncthreads();

    float ss = 0.0f;
    for (int c = tid; c < A.D; c += 256) {
        const float v = sh_y[c];
        ss += v * v;
    }
    const float tot_ss = block_sum(ss, smem, 256);
    const float rinv = 1.0f / sqrtf(tot_ss / (float)A.D + A.eps);
    for (int c = tid; c < A.D; c += 256)
        sh_y[c] = (sh_y[c] * rinv) * bf2f(A.nw[(long long)c * A.nws]);
    __syncthreads();

    if (A.vc.log) {   /* libref's finiteness rule, on the rounded values that would be stored (validation off: base stores the row whatever it holds) */
        const int nope_ = A.D - A.R;
        const int start_ = (p / A.ratio) * A.ratio;
        int nf = 0;
        for (int c = tid; c < nope_; c += 256) nf |= !hca_bf_finite(sh_y[c]);
        for (int i = tid; i < A.R / 2; i += 256) {
            const double ang = (double)start_ * A.tab.inv[i];
            const float cs = (float)cos(ang), sn = (float)sin(ang);
            const float ev = sh_y[nope_ + 2 * i], od = sh_y[nope_ + 2 * i + 1];
            nf |= !hca_bf_finite(ev * cs - od * sn);
            nf |= !hca_bf_finite(ev * sn + od * cs);
        }
        if (nf) sh_flag[1] = 1;
        __syncthreads();
        if (sh_flag[1]) { if (tid == 0) R4D_HCA_RAISE(16); __syncthreads(); return; }
    }

    /* Only now, with every numerical rule judged, may a LATER completing row with the same out slot take this row's place: a row libref refuses is refused even when a later row overwrites it. */
    if (scan_dups) {
        if (A.dd) {
            if (dd_last(A.dd, A.ddshift, slot) != m) return;   /* a later completing row writes this out slot (the same value for every thread of the block) */
        } else {
            for (int m2 = m + 1 + tid; m2 < A.M && !sh_flag[0]; m2 += 256) {
                if (A.os[(long long)m2 * A.oss] != slot) continue;
                const int p2 = A.pos[(long long)m2 * A.ps];
                if (p2 >= 0 && (p2 + 1) % A.ratio == 0) sh_flag[0] = 1;
            }
            __syncthreads();
            if (sh_flag[0]) return;
        }
    }


    const int nope = A.D - A.R;
    const int start = (p / A.ratio) * A.ratio;
    if (!A.packed) {
        uint16_t* dst = (uint16_t*)A.out + (long long)o * A.outp;
        for (int c = tid; c < nope; c += 256)
            dst[(long long)c * A.outs] = f2bf(sh_y[c]);
        for (int i = tid; i < A.R / 2; i += 256) {
            const double ang = (double)start * A.tab.inv[i];
            const float cs = (float)cos(ang), sn = (float)sin(ang);
            const float ev = sh_y[nope + 2 * i], od = sh_y[nope + 2 * i + 1];
            dst[(long long)(nope + 2 * i) * A.outs] = f2bf(ev * cs - od * sn);
            dst[(long long)(nope + 2 * i + 1) * A.outs] = f2bf(ev * sn + od * cs);
        }
    } else {
        uint8_t* dst = (uint8_t*)A.out + (long long)o * A.outp;
        const int nb = nope / A.qb;
        for (int b = tid; b < nb; b += 256) {
            float amax = 0.0f;
            for (int i = 0; i < A.qb; ++i) {
                const float bv = bfr(sh_y[b * A.qb + i]);
                amax = fmaxf(amax, fabsf(bv));
            }
            const float scale = fmaxf(amax / 448.0f, 1e-4f);
            const uint32_t bits = __float_as_uint(scale);
            int e2 = (int)((bits >> 23) & 0xffu) - 127 + ((bits & 0x7fffffu) ? 1 : 0);
            dst[(long long)(nope + 2 * A.R + b) * A.outs] = (uint8_t)(e2 + 127);
            const float is = __uint_as_float((uint32_t)(127 - e2) << 23);
            for (int i = 0; i < A.qb; ++i) {
                float v = bfr(sh_y[b * A.qb + i]) * is;
                v = fminf(fmaxf(v, -448.0f), 448.0f);
                dst[(long long)(b * A.qb + i) * A.outs] = f32_to_fp8e4m3_dev(v);
            }
        }
        for (int i = tid; i < A.R / 2; i += 256) {
            const double ang = (double)start * A.tab.inv[i];
            const float cs = (float)cos(ang), sn = (float)sin(ang);
            const float ev = sh_y[nope + 2 * i], od = sh_y[nope + 2 * i + 1];
            const uint16_t h0 = f2bf(ev * cs - od * sn), h1 = f2bf(ev * sn + od * cs);
            dst[(long long)(nope + 4 * i + 0) * A.outs] = (uint8_t)(h0 & 255u);
            dst[(long long)(nope + 4 * i + 1) * A.outs] = (uint8_t)(h0 >> 8);
            dst[(long long)(nope + 4 * i + 2) * A.outs] = (uint8_t)(h1 & 255u);
            dst[(long long)(nope + 4 * i + 3) * A.outs] = (uint8_t)(h1 >> 8);
        }
        if (tid == 0) dst[(long long)(nope + 2 * A.R + nb) * A.outs] = 0u;
    }
    __syncthreads();
}

__global__ void ernie_hca_compress_seq_kernel(HcaCompressArgs A) {
    __shared__ float sh_y[4096];
    __shared__ float smem[8];
    __shared__ int sh_flag[2];
    for (int m = 0; m < A.M; ++m) hca_compress_one_row(A, m, sh_y, smem, sh_flag, false);
}

__global__ void ernie_hca_compress_last_mark_kernel(HcaCompressArgs A) {
    const int m = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (m >= A.M) return;
    const int o = A.os[(long long)m * A.oss];
    if (o < 0 || o >= A.Rr) return;
    const int p = A.pos[(long long)m * A.ps];
    if (p >= 0 && (p + 1) % A.ratio == 0) dd_mark(A.dd, A.ddshift, o, m);   /* every completing row competes, as the scan it replaces (a later row that is itself refused still takes its place: the launch is refused) */
}

__global__ void ernie_hca_compress_par_kernel(HcaCompressArgs A) {
    __shared__ float sh_y[4096];
    __shared__ float smem[8];
    __shared__ int sh_flag[2];
    const int m = (int)blockIdx.x;
    if (m < A.M) hca_compress_one_row(A, m, sh_y, smem, sh_flag, true);
}

/* ------------------------------------------------------------------ ernie_hca_union_attn */
struct HcaUnionArgs {
    const uint16_t *q, *wk, *ck;
    const int32_t *pos, *req, *wt, *ct, *origin;
    const float *sink;
    uint16_t *out;
    long long qp, qs, wkp0, wkp2, wks, ckp0, ckp2, cks, ps, rs, wtp, wts, ctp, cts, os, sks, outp, outs;
    int M, H, D, ratio, win, S, Bw, Bc, bw, bc, Pw, Pc, tok_rows;
    int check;      /* 1: the gate was opened and ernie_hca_union_validate_kernel ran ahead of this launch: the gate word (vc.gate) gates every store */
    r4d_verdict::Vc vc;   /* the event log and the gate word of this call site (device memory the library owns; constant across replays) */
    float scale;
    int pipe;   /* 1: the compressed pass of the D == 512 kernels runs a pipelined compressed pass: rows resolved and loaded ahead of the softmax chain (same arithmetic, same order) */

    int pad_ = 0;   /* explicit tail padding: the argument bytes are all named and zero, whatever the stack held (the struct is copied by value into the kernel arguments) */
};
static_assert(sizeof(HcaUnionArgs) <= 512, "HcaUnionArgs exceeds AQL kernarg limit");

/* Causality and compressed-group phase use absolute p. Bound the window in
 * those coordinates before translating it to the first retained block. */
__device__ inline bool hca_window_range(const HcaUnionArgs& A, int p, int r, unsigned& start, unsigned& end) {
    if (p < 0 || r < 0 || r >= A.S) return false;
    const int first = A.origin ? A.origin[(long long)r * A.os] : 0;
    const int earliest = p >= A.win - 1 ? p - A.win + 1 : 0;
    if (first && (first < 0 || first > earliest || first % A.bw)) return false;
    start = (unsigned)(earliest - first);
    end = (unsigned)(p - first);
    return true;
}

/* One decision for the whole block (thread 0 reads the word once, every thread of the block must reach this) or, for the wide d512 kernel whose waves do not
 * synchronise, one per wave. */
__device__ __forceinline__ bool union_refused(const HcaUnionArgs& A) {
    if (!A.check) return false;
    return r4d_verdict::gate_refused_block(A.vc);
}
__device__ __forceinline__ bool union_refused_wave(const HcaUnionArgs& A) {
    if (!A.check) return false;
    return r4d_verdict::gate_refused_wave(A.vc);
}

/* The first step of a gated launch, one thread: the call site's gate word becomes "open, nothing refused". Every launch (every replay) starts here, so a refusal
 * of one launch is never seen by the next. */
__global__ void ernie_hca_gate_open_kernel(HcaUnionArgs A) {
    if (threadIdx.x == 0) r4d_verdict::gate_open(A.vc.gate);
}

__device__ __forceinline__ bool hca_table_ok(const int32_t* tab, long long t0, long long t1, long long r, long long blk, long long width, long long pages) {
    if (blk >= width) return false;
    const long long pg = tab[r * t0 + blk * t1];
    return pg >= -1 && pg < pages;
}

/* The device form of libref's hca_union_attn validation, one block per query row (block 0 also checks the sink): a position below 0, a request outside [0, S), an
 * origin that is negative, not a multiple of the window block or past the oldest window token, a sink that is NaN or +inf, and every table entry the row reads --
 * the compressed groups' blocks and the window's blocks -- below -1, at or past the pool, or past the table width (-1 is skipped). A violating block sets its bits in
 * the gate word of the call site and raises an event; the attention blocks of the launch read the word first. Bits as at the HCA log: 32, 64, 128. */
__global__ __launch_bounds__(256) void ernie_hca_union_validate_kernel(HcaUnionArgs A) {
    __shared__ int sh_bits[256];
    const int tid = (int)threadIdx.x, m = (int)blockIdx.x;
    int bits = 0;
    if (m == 0) {
        for (int h = tid; h < A.H; h += 256) {
            const float sk = A.sink[(long long)h * A.sks];
            if (sk != sk || sk == INFINITY) bits |= 32;
        }
    }
    if (m < A.M) {
        const long long p = A.pos[(long long)m * A.ps], r = A.req[(long long)m * A.rs];
        if (p < 0 || r < 0 || r >= A.S) {
            bits |= 64;
        } else {
            const long long first = A.origin ? A.origin[r * A.os] : 0;
            const long long start = p - A.win + 1 > 0 ? p - A.win + 1 : 0;
            if (first < 0 || first % A.bw || first > start) {
                bits |= 64;
            } else {
                const long long n = (p + 1) / A.ratio;
                if (n > 0) {
                    const long long last = A.tok_rows ? (n * A.ratio - 1) / A.bc : (n - 1) / A.bc;
                    for (long long bk = tid; bk <= last; bk += 256) {
                        bool hit = true;
                        if (A.tok_rows) {   /* token rows: group j is at row (j + 1) * ratio - 1; a block is read only if some group lands in it */
                            long long jmin = (bk * A.bc + A.ratio) / A.ratio - 1;
                            if (jmin < 0) jmin = 0;
                            hit = jmin < n && (jmin + 1) * A.ratio - 1 <= bk * A.bc + A.bc - 1;
                        }
                        if (hit && !hca_table_ok(A.ct, A.ctp, A.cts, r, bk, A.Bc, A.Pc)) bits |= 128;
                    }
                }
                for (long long bk = (start - first) / A.bw + tid; bk <= (p - first) / A.bw; bk += 256)
                    if (!hca_table_ok(A.wt, A.wtp, A.wts, r, bk, A.Bw, A.Pw)) bits |= 128;
            }
        }
    }
    sh_bits[tid] = bits;
    __syncthreads();
    if (tid == 0) {
        int all = 0;
        for (int i = 0; i < 256; ++i) all |= sh_bits[i];
        if (m == 0 && r4d_verdict::inject_now(A.vc.inject, A.vc.gate)) all |= (int)r4d_verdict::kInjected;
        if (all) { r4d_verdict::gate_add(A.vc.gate, (unsigned)all); R4D_HCA_RAISE(all); }
    }
}
/* Rows a pipeline stage of the batched compressed pass holds: 8 VGPRs a row and stage, two stages live. Sized so neither kernel spills (gfx1201: decode kernel 244 VGPRs at 6,
 * wide kernel 148 at 4, 192 with 5 spilled at 6); only 64 waves run a decode step, so occupancy is not what the registers cost. */
#ifndef R4D_HCA_PIPE_ROWS_DECODE
#define R4D_HCA_PIPE_ROWS_DECODE 6
#endif
#ifndef R4D_HCA_PIPE_ROWS_WIDE
#define R4D_HCA_PIPE_ROWS_WIDE 4
#endif

/* D == 512 (live ERNIE 5.5 TP=8 HCA shape): ONE arithmetic for every row count (the row-identity checks).
 *
 * The row's softmax runs over a fixed 8-way partition of its key sequence, independent of how many rows share the launch:
 *   key sequence  = compressed rows j = 0 .. n_cmp - 1, then the window tokens wstart .. wend (k = t - wstart)
 *   partition w   = keys j == w (mod 8) of the compressed rows, then the window tokens with (t - wstart) == w (mod 8), in that order
 *   partition state (mx, den, acc[512]) = the online softmax chain over its keys, seeded with the sink in partition 0 only
 *   merge         = the 8 partition states folded in partition order 0..7 (hca512_merge), an empty partition (den == 0) skipped
 * The split-KV decode kernel (M <= R4D_HCA_DECODE_MAX rows, default 4) runs one partition per wave (8 waves a head, 64 waves at M = 1) and folds
 * the states on wave 0; the wide kernel (4 or 8 heads a block, one wave a head) runs the 8 partitions one after the other in the same wave.
 * Both call hca512_part and hca512_merge: the same expressions, with every multiply-add written as an explicit fmaf (the same bits with and
 * without the compiler's contraction, on the card and in the emulator). The dot over 512 columns is 16 sequential fmaf per lane, then wave_sum. */
template <bool Vec, int U>
__device__ __forceinline__ void hca512_part(const HcaUnionArgs& A, const float (&qv)[16], int c0, int c1, int p, int r, unsigned wstart, unsigned wend,
                                            int part, int h, float& mx, float& den, float (&acc)[16]) {
#pragma unroll
    for (int e = 0; e < 16; ++e) acc[e] = 0.0f;
    mx = -INFINITY;
    den = 0.0f;
    if (part == 0) {
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
        for (int e = 0; e < 16; ++e) dot_l = fmaf(qv[e], kv[e], dot_l);
        const float sc = wave_sum(dot_l) * A.scale;
        const float nmx = fmaxf(mx, sc);
        const float alpha = (mx == -INFINITY) ? 0.0f : expf(mx - nmx);
        const float w = expf(sc - nmx);
        den = fmaf(den, alpha, w);
#pragma unroll
        for (int e = 0; e < 16; ++e) acc[e] = fmaf(w, kv[e], acc[e] * alpha);
        mx = nmx;
    };

    /* Compressed rows j == part (mod 8) */
    const int n_cmp = (int)((p + 1ll) / A.ratio);
    bool done_cmp = false;
    if constexpr (Vec) {

    /* Software-pipelined compressed pass (A.pipe). The page-table load, the row load, the dot and the wave reduction of a row do not depend on the softmax
     * state, so they run ahead of it: three stages of batches of U rows (table entries of batch b + 2 requested, rows of batch b + 1 requested, batch b scored and
     * run through the chain). Every value, the order of the chain and the reduction trees are those of `attend`; only the memory latency of the rows overlaps. */
    auto ct_load = [&](int jb, int jstep, int (&pg)[U]) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int j = jb + jstep * u;
            int e = -1;
            if (j < n_cmp) {
                const int id = A.tok_rows ? (int)((j + 1ll) * A.ratio - 1) : j;
                const int blk = id / A.bc;
                if (blk < A.Bc) e = A.ct[(long long)r * A.ctp + (long long)blk * A.cts];
            }
            R4D_HCA_PAGE_LATCH(e, A.Pc);   /* the integration latch: each table entry read once, as the one-row loop */
            pg[u] = e;
        }
    };
    auto row_load = [&](int jb, int jstep, const int (&pg)[U], const uint16_t* (&kp)[U], uint4 (&raw)[U][2]) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int j = jb + jstep * u;
            kp[u] = nullptr;
            if (j < n_cmp && pg[u] >= 0 && pg[u] < A.Pc) {
                const int id = A.tok_rows ? (int)((j + 1ll) * A.ratio - 1) : j;
                kp[u] = A.ck + (long long)pg[u] * A.ckp0 + (long long)(id % A.bc) * A.ckp2;
                raw[u][0] = *reinterpret_cast<const uint4*>(kp[u] + c0);
                raw[u][1] = *reinterpret_cast<const uint4*>(kp[u] + c1);
            }
        }
    };
    auto consume = [&](const uint16_t* const (&kp)[U], const uint4 (&raw)[U][2]) {
        float scs[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            if (kp[u]) {
                float kv[16];
                unpack_uint4_bf16(raw[u][0], kv);
                unpack_uint4_bf16(raw[u][1], kv + 8);
                float dot_l = 0.0f;
#pragma unroll
                for (int e = 0; e < 16; ++e) dot_l = fmaf(qv[e], kv[e], dot_l);
                scs[u] = wave_sum(dot_l) * A.scale;
            }
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            if (kp[u]) {
                float kv[16];
                unpack_uint4_bf16(raw[u][0], kv);
                unpack_uint4_bf16(raw[u][1], kv + 8);
                const float sc = scs[u];
                const float nmx = fmaxf(mx, sc);
                const float alpha = (mx == -INFINITY) ? 0.0f : expf(mx - nmx);
                const float w = expf(sc - nmx);
                den = fmaf(den, alpha, w);
#pragma unroll
                for (int e = 0; e < 16; ++e) acc[e] = fmaf(w, kv[e], acc[e] * alpha);
                mx = nmx;
            }
        }
    };
    auto pipelined_cmp = [&](int first, int jstep) {
        const int bstep = jstep * U;
        int pgA[U], pgB[U], pgC[U];
        const uint16_t* kpA[U];
        const uint16_t* kpB[U];
        uint4 rawA[U][2], rawB[U][2];
        ct_load(first, jstep, pgA);
        ct_load(first + bstep, jstep, pgB);
        row_load(first, jstep, pgA, kpA, rawA);
        for (int jb = first; jb < n_cmp; jb += bstep) {
            ct_load(jb + 2 * bstep, jstep, pgC);
            row_load(jb + bstep, jstep, pgB, kpB, rawB);
            consume(kpA, rawA);
#pragma unroll
            for (int u = 0; u < U; ++u) {
                kpA[u] = kpB[u];
                rawA[u][0] = rawB[u][0];
                rawA[u][1] = rawB[u][1];
                pgB[u] = pgC[u];
            }
        }
    };
        if (A.pipe) {
            pipelined_cmp(part, 8);
            done_cmp = true;
        }
    }
    if (!done_cmp) {
        for (int j = part; j < n_cmp; j += 8) {
            const int id = A.tok_rows ? (int)((j + 1ll) * A.ratio - 1) : j;
            const int blk = id / A.bc;
            if (blk >= A.Bc) continue;
            const int pg = A.ct[(long long)r * A.ctp + (long long)blk * A.cts];
            R4D_HCA_PAGE_LATCH(pg, A.Pc);
            if (pg < 0 || pg >= A.Pc) continue;
            const uint16_t* kptr = A.ck + (long long)pg * A.ckp0 + (long long)(id % A.bc) * A.ckp2;
            attend(kptr, A.cks);
        }
    }

    /* Window tokens with (t - wstart) == part (mod 8), grouped by page to avoid repeated division and page-table loads */
    unsigned t = wstart;
    while (t <= wend) {
        const unsigned blk = t / A.bw;
        const int off0 = (int)(t % A.bw);
        const unsigned tend = (blk + 1) * A.bw - 1;
        const unsigned tstop = (tend < wend) ? tend : wend;
        if (blk < (unsigned)A.Bw) {
            const int pg = A.wt[(long long)r * A.wtp + (long long)blk * A.wts];
            R4D_HCA_PAGE_LATCH(pg, A.Pw);
            if (pg >= 0 && pg < A.Pw) {
                const uint16_t* pg_ptr = A.wk + (long long)pg * A.wkp0;
                const int cnt = tstop - t;
                const int d0 = (part + 8 - ((t - wstart) & 7)) & 7;
                for (int d = d0; d <= cnt; d += 8) {
                    attend(pg_ptr + (long long)(off0 + d) * A.wkp2, A.wks);
                }
            }
        }
        t = tstop + 1;
    }
}

/* Fold partition state (mw, dw, aw) into the running state (M, D, Acc), in partition order: an empty partition (dw == 0) changes nothing. */
__device__ __forceinline__ void hca512_merge(float& M, float& D, float (&Acc)[16], float mw, float dw, const float (&aw)[16]) {
    if (!(dw > 0.0f)) return;
    const float nm = fmaxf(M, mw);
    const float alpha = (M == -INFINITY) ? 0.0f : expf(M - nm);
    const float b = expf(mw - nm);
    D = fmaf(dw, b, D * alpha);
#pragma unroll
    for (int e = 0; e < 16; ++e) Acc[e] = fmaf(aw[e], b, Acc[e] * alpha);
    M = nm;
}

template <bool Vec>
__device__ __forceinline__ void hca512_store(const HcaUnionArgs& A, int m, int h, int c0, int c1, float D, const float (&Acc)[16]) {
    float res[16];
    if (D > 0.0f) {
#pragma unroll
        for (int e = 0; e < 16; ++e) res[e] = Acc[e] / D;
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

template <bool Vec>
__device__ __forceinline__ void hca512_load_q(const HcaUnionArgs& A, int m, int h, int c0, int c1, float (&qv)[16]) {
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
}

/* Wide kernel (any row count; the shipped rule above R4D_HCA_DECODE_MAX rows):
 *   - 32 lanes x 16 elements/lane = 512 dims in VGPRs, 0 bytes LDS, 0 barriers, uint4 16B coalesced loads
 *   - 4 or 8 waves/block (4 or 8 query heads sharing the single KV head in L1 cache), the 8 partitions one after the other in the wave */
template <bool Vec>
__global__ void ernie_hca_union_attn_d512_kernel(HcaUnionArgs A) {
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31, wave = tid >> 5, nwaves = (int)blockDim.x >> 5;
    const int h = (int)blockIdx.x * nwaves + wave;
    const int m = (int)blockIdx.y;
    if (h >= A.H || m >= A.M) return;
    if (union_refused_wave(A)) return;   /* the validator kernel refused this launch: no store */

    const int p = A.pos[(long long)m * A.ps];
    const int r = A.req[(long long)m * A.rs];
    unsigned wstart, wend;
    if (!hca_window_range(A, p, r, wstart, wend)) return;
    const int c0 = lane * 8;
    const int c1 = 256 + lane * 8;

    float qv[16];
    hca512_load_q<Vec>(A, m, h, c0, c1, qv);
    float M = -INFINITY, D = 0.0f, Acc[16];
#pragma unroll
    for (int e = 0; e < 16; ++e) Acc[e] = 0.0f;
    for (int part = 0; part < 8; ++part) {
        float mx, den, acc[16];
        hca512_part<Vec, R4D_HCA_PIPE_ROWS_WIDE>(A, qv, c0, c1, p, r, wstart, wend, part, h, mx, den, acc);
        hca512_merge(M, D, Acc, mx, den, acc);
    }
    hca512_store<Vec>(A, m, h, c0, c1, D, Acc);
}

/* Cooperative 8-wave/head split-KV decode kernel (the shipped rule up to R4D_HCA_DECODE_MAX = 4 rows) for D == 512:
 *   - Launch dim3(H, M) blocks x 256 threads (8 waves/head -> 64 waves at M=1, H=8)
 *   - Wave w runs partition w; single-barrier LDS hand-over (sh_mx[8], sh_den[8], sh_acc[8][512] = 16.06 KiB), wave 0 folds 0..7 in order */
template <bool Vec>
__global__ __launch_bounds__(256) void ernie_hca_union_attn_d512_decode_kernel(HcaUnionArgs A) {
    __shared__ float sh_mx[8];
    __shared__ float sh_den[8];
    __shared__ float sh_acc[8][512];

    const int tid = (int)threadIdx.x;
    const int lane = tid & 31, wave = tid >> 5;
    const int h = (int)blockIdx.x;
    const int m = (int)blockIdx.y;
    if (h >= A.H || m >= A.M) return;
    if (union_refused(A)) return;   /* the validator kernel refused this launch: no store */

    const int p = A.pos[(long long)m * A.ps];
    const int r = A.req[(long long)m * A.rs];
    unsigned wstart, wend;
    if (!hca_window_range(A, p, r, wstart, wend)) return;
    const int c0 = lane * 8;
    const int c1 = 256 + lane * 8;

    float qv[16];
    hca512_load_q<Vec>(A, m, h, c0, c1, qv);
    float mx, den, acc[16];
    hca512_part<Vec, R4D_HCA_PIPE_ROWS_DECODE>(A, qv, c0, c1, p, r, wstart, wend, wave, h, mx, den, acc);

    /* Publish the partition state to LDS (bank-conflict-free lane-major layout) */
    if (lane == 0) {
        sh_mx[wave] = mx;
        sh_den[wave] = den;
    }
#pragma unroll
    for (int e = 0; e < 16; ++e) sh_acc[wave][e * 32 + lane] = acc[e];
    __syncthreads();

    /* Wave 0 folds the 8 partitions in order and writes the final output */
    if (wave == 0) {
        float M = -INFINITY, D = 0.0f, Acc[16];
#pragma unroll
        for (int e = 0; e < 16; ++e) Acc[e] = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            float aw[16];
#pragma unroll
            for (int e = 0; e < 16; ++e) aw[e] = sh_acc[w][e * 32 + lane];
            hca512_merge(M, D, Acc, sh_mx[w], sh_den[w], aw);
        }
        hca512_store<Vec>(A, m, h, c0, c1, D, Acc);
    }
}

/* Fast path for D == 128: 128 threads, 1 element/thread in registers, online softmax. */
__global__ void ernie_hca_union_attn_d128_kernel(HcaUnionArgs A) {
    __shared__ float smem[4];
    const int h = (int)blockIdx.x, m = (int)blockIdx.y, tid = (int)threadIdx.x;
    if (h >= A.H || m >= A.M) return;
    if (union_refused(A)) return;   /* the validator kernel refused this launch: no store */
    const int p = A.pos[(long long)m * A.ps], r = A.req[(long long)m * A.rs];
    unsigned wstart, wend;
    if (!hca_window_range(A, p, r, wstart, wend)) return;
    const float qv = bf2f(A.q[(long long)m * A.qp + (long long)(h * 128 + tid) * A.qs]);
    const float sk = A.sink[(long long)h * A.sks];
    float mx = sk;
    float den = (sk == -INFINITY) ? 0.0f : 1.0f;
    float acc = 0.0f;

    const int n_cmp = (int)((p + 1ll) / A.ratio);
    for (int j = 0; j < n_cmp; ++j) {
        const int id = A.tok_rows ? (int)((j + 1ll) * A.ratio - 1) : j;
        const int blk = id / A.bc;
        if (blk >= A.Bc) continue;
        const int pg = A.ct[(long long)r * A.ctp + (long long)blk * A.cts];
        R4D_HCA_PAGE_LATCH(pg, A.Pc);
        if (pg < 0 || pg >= A.Pc) continue;
        const float kv = bf2f(A.ck[(long long)pg * A.ckp0 + (long long)(id % A.bc) * A.ckp2 + (long long)tid * A.cks]);
        const float dot = block_sum(qv * kv, smem, 128);
        const float sc = dot * A.scale;
        const float nmx = fmaxf(mx, sc);
        const float alpha = (mx == -INFINITY) ? 0.0f : expf(mx - nmx);
        const float w = expf(sc - nmx);
        den = den * alpha + w;
        acc = acc * alpha + w * kv;
        mx = nmx;
    }

    for (unsigned t = wstart; t <= wend; ++t) {
        const unsigned blk = t / A.bw;
        if (blk >= (unsigned)A.Bw) continue;
        const int pg = A.wt[(long long)r * A.wtp + (long long)blk * A.wts];
        R4D_HCA_PAGE_LATCH(pg, A.Pw);
        if (pg < 0 || pg >= A.Pw) continue;
        const float kv = bf2f(A.wk[(long long)pg * A.wkp0 + (long long)(t % A.bw) * A.wkp2 + (long long)tid * A.wks]);
        const float dot = block_sum(qv * kv, smem, 128);
        const float sc = dot * A.scale;
        const float nmx = fmaxf(mx, sc);
        const float alpha = (mx == -INFINITY) ? 0.0f : expf(mx - nmx);
        const float w = expf(sc - nmx);
        den = den * alpha + w;
        acc = acc * alpha + w * kv;
        mx = nmx;
    }

    const float res = (den > 0.0f) ? (acc / den) : 0.0f;
    A.out[(long long)m * A.outp + (long long)(h * 128 + tid) * A.outs] = f2bf(res);
}

/* General path for arbitrary D <= 4096: uses LDS for q and acc. */
__global__ void ernie_hca_union_attn_gen_kernel(HcaUnionArgs A) {
    __shared__ float smem[8];
    __shared__ float q_sh[4096];
    __shared__ float acc_sh[4096];
    const int h = (int)blockIdx.x, m = (int)blockIdx.y, tid = (int)threadIdx.x;
    if (h >= A.H || m >= A.M) return;
    if (union_refused(A)) return;   /* the validator kernel refused this launch: no store */
    const int p = A.pos[(long long)m * A.ps], r = A.req[(long long)m * A.rs];
    unsigned wstart, wend;
    if (!hca_window_range(A, p, r, wstart, wend)) return;
    for (int c = tid; c < A.D; c += 256) {
        q_sh[c] = bf2f(A.q[(long long)m * A.qp + (long long)(h * A.D + c) * A.qs]);
        acc_sh[c] = 0.0f;
    }
    __syncthreads();
    const float sk = A.sink[(long long)h * A.sks];
    float mx = sk;
    float den = (sk == -INFINITY) ? 0.0f : 1.0f;

    const int n_cmp = (int)((p + 1ll) / A.ratio);
    for (int j = 0; j < n_cmp; ++j) {
        const int id = A.tok_rows ? (int)((j + 1ll) * A.ratio - 1) : j;
        const int blk = id / A.bc;
        if (blk >= A.Bc) continue;
        const int pg = A.ct[(long long)r * A.ctp + (long long)blk * A.cts];
        R4D_HCA_PAGE_LATCH(pg, A.Pc);
        if (pg < 0 || pg >= A.Pc) continue;
        const long long base = (long long)pg * A.ckp0 + (long long)(id % A.bc) * A.ckp2;
        float dot_l = 0.0f;
        for (int c = tid; c < A.D; c += 256) dot_l += q_sh[c] * bf2f(A.ck[base + (long long)c * A.cks]);
        const float sc = block_sum(dot_l, smem, 256) * A.scale;
        const float nmx = fmaxf(mx, sc);
        const float alpha = (mx == -INFINITY) ? 0.0f : expf(mx - nmx);
        const float w = expf(sc - nmx);
        den = den * alpha + w;
        for (int c = tid; c < A.D; c += 256)
            acc_sh[c] = acc_sh[c] * alpha + w * bf2f(A.ck[base + (long long)c * A.cks]);
        mx = nmx;
        __syncthreads();
    }

    for (unsigned t = wstart; t <= wend; ++t) {
        const unsigned blk = t / A.bw;
        if (blk >= (unsigned)A.Bw) continue;
        const int pg = A.wt[(long long)r * A.wtp + (long long)blk * A.wts];
        R4D_HCA_PAGE_LATCH(pg, A.Pw);
        if (pg < 0 || pg >= A.Pw) continue;
        const long long base = (long long)pg * A.wkp0 + (long long)(t % A.bw) * A.wkp2;
        float dot_l = 0.0f;
        for (int c = tid; c < A.D; c += 256) dot_l += q_sh[c] * bf2f(A.wk[base + (long long)c * A.wks]);
        const float sc = block_sum(dot_l, smem, 256) * A.scale;
        const float nmx = fmaxf(mx, sc);
        const float alpha = (mx == -INFINITY) ? 0.0f : expf(mx - nmx);
        const float w = expf(sc - nmx);
        den = den * alpha + w;
        for (int c = tid; c < A.D; c += 256)
            acc_sh[c] = acc_sh[c] * alpha + w * bf2f(A.wk[base + (long long)c * A.wks]);
        mx = nmx;
        __syncthreads();
    }

    for (int c = tid; c < A.D; c += 256) {
        const float res = (den > 0.0f) ? (acc_sh[c] / den) : 0.0f;
        A.out[(long long)m * A.outp + (long long)(h * A.D + c) * A.outs] = f2bf(res);
    }
}

/* ------------------------------------------------------------------ ernie_hca_wo_a */
struct HcaWoAArgs {
    const uint16_t *o, *w;
    const int32_t *pos;
    uint16_t *out;
    long long op, os, ps, wp, ws, outp, outs;
    int M, H, D, R, G, K, hg, per;
    RopeInvTable tab;
    r4d_verdict::Vc vc;   /* this launch's event log and gate word (device memory the library owns; constant across replays) */
};
static_assert(sizeof(HcaWoAArgs) <= 512, "HcaWoAArgs exceeds AQL kernarg limit");

/* Fast decode / small-M kernel (M <= 16): dim3((K+7)/8, G, M), 256 threads (8 waves, 1 wave/row j).
 * Only the hg * R inverse-rotated tail dims are staged in LDS (512 BF16 = 1 KiB for ERNIE);
 * nope dims are read directly from o with coalesced 128-bit uint4 loads! */
template <bool Vec>
__global__ __launch_bounds__(256) void ernie_hca_wo_a_decode_kernel(HcaWoAArgs A) {
    __shared__ __align__(16) uint16_t tail_sh[4096];
    __shared__ float cs_sh[256];
    __shared__ float sn_sh[256];
    const int j_blk = (int)blockIdx.x * 8;
    const int g = (int)blockIdx.y;
    const int m = (int)blockIdx.z;
    const int tid = (int)threadIdx.x;
    if (g >= A.G || m >= A.M) return;
    if (A.vc.log && A.pos[(long long)m * A.ps] < 0) { if (blockIdx.x == 0 && g == 0 && tid == 0) R4D_HCA_RAISE(2); return; }   /* libref refuses a negative position: the row is not stored */

    const int nope = A.D - A.R;
    const int half = A.R >> 1;
    const double p = (double)A.pos[(long long)m * A.ps];
    const uint16_t* o_row = A.o + (long long)m * A.op + (long long)g * A.per;

    /* Compute cos/sin once across all hg heads (exact double angle matching libref). */
    if (tid < half) {
        const double ang = p * A.tab.inv[tid];
        cs_sh[tid] = (float)cos(ang);
        sn_sh[tid] = -(float)sin(ang);
    }
    __syncthreads();

    /* Rotate only the hg * R tail elements into tail_sh (1 iteration for hg=8, R=64). */
    const int total_pairs = A.hg * half;
    for (int idx = tid; idx < total_pairs; idx += 256) {
        const int hh = idx / half, i = idx % half;
        const int c0 = hh * A.D + nope + 2 * i;
        const float ev = bf2f(o_row[c0]);
        const float od = bf2f(o_row[c0 + 1]);
        const float cs = cs_sh[i], sn = sn_sh[i];
        tail_sh[hh * A.R + 2 * i]     = f2bf(ev * cs - od * sn);
        tail_sh[hh * A.R + 2 * i + 1] = f2bf(ev * sn + od * cs);
    }
    __syncthreads();

    const int lane = tid & 31, wave = tid >> 5;
    const int j = j_blk + wave;
    if (j >= A.K) return;

    const uint16_t* w_row = A.w + (long long)(g * A.K + j) * A.wp;
    float sum = 0.0f;
    if (Vec) {
        for (int hh = 0; hh < A.hg; ++hh) {
            const uint16_t* w_h = w_row + hh * A.D;
            const uint16_t* o_h = o_row + hh * A.D;
            for (int lc = lane * 8; lc < nope; lc += 256) {
                const uint4 wv = *reinterpret_cast<const uint4*>(w_h + lc);
                const uint4 ov = *reinterpret_cast<const uint4*>(o_h + lc);
                sum += dot_uint4_bf16(ov, wv);
            }
            const uint16_t* w_r = w_h + nope;
            const uint16_t* t_h = tail_sh + hh * A.R;
            for (int rc = lane * 8; rc < A.R; rc += 256) {
                const uint4 wv = *reinterpret_cast<const uint4*>(w_r + rc);
                const uint4 ov = *reinterpret_cast<const uint4*>(t_h + rc);
                sum += dot_uint4_bf16(ov, wv);
            }
        }
    } else {
        for (int hh = 0; hh < A.hg; ++hh) {
            const uint16_t* w_h = w_row + hh * A.D;
            const uint16_t* o_h = o_row + hh * A.D;
            for (int lc = lane; lc < nope; lc += 32)
                sum += bf2f(o_h[lc]) * bf2f(w_h[lc]);
            const uint16_t* w_r = w_h + nope;
            const uint16_t* t_h = tail_sh + hh * A.R;
            for (int rc = lane; rc < A.R; rc += 32)
                sum += bf2f(t_h[rc]) * bf2f(w_r[rc]);
        }
    }
    sum = wave_sum(sum);
    if (lane == 0)
        A.out[(long long)m * A.outp + (long long)(g * A.K + j)] = f2bf(sum);
}

/* Fast prefill LDS-tiled WMMA kernel (M > 16, D % 32 == 0, R % 32 == 0, R <= 64):
 *   - 8 waves (256 threads), BM = 64 tokens x BN = 128 output channels, BK = 32 (2 WMMA k-steps)
 *   - 100% coalesced global -> register -> LDS loads (1 uint4/thread for A_sh, 2 uint4/thread for B_sh)
 *   - Register double-buffering overlaps next BK=32 global fetch with current tile's 8 WMMAs
 *   - Inverse GPT-J RoPE fused in registers on the trailing R dims of each head (0 branches in WMMA loop)
 *   - 2 x 4 wave tiling: each wave computes 2 x 2 = 4 WMMA 16x16 tiles (32 acc VGPRs)
 *   - A_wmma = W[j, k] (B_sh), B_wmma = O_rot[m, k] (A_sh) -> D[j, m]: lane L holds 8 contiguous
 *     output channels j for token m = (L & 15) -> single coalesced uint4 store per token
 *   - LDS: A_sh[64][40] (5 KiB) + B_sh[128][40] (10 KiB) + cs_sh[64][32] (8 KiB) + sn_sh[64][32] (8 KiB)
 *          = 31.0 KiB (< 32 KiB -> 2 blocks/CU = 16 waves/CU). */
#if defined(__clang__) || defined(__HIP_DEVICE_COMPILE__)
typedef short hca_v8s __attribute__((ext_vector_type(8)));
typedef float hca_v8f __attribute__((ext_vector_type(8)));
typedef uint32_t hca_u4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ hca_v8f hca_wmma_bf16(hca_v8s a, hca_v8s b, hca_v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}

template <bool Vec>
__device__ __forceinline__ hca_u4 hca_ld_u4(const uint16_t* p) {
    if (Vec) {
        return *reinterpret_cast<const hca_u4*>(p);
    } else {
        hca_u4 v;
        v[0] = (uint32_t)p[0] | ((uint32_t)p[1] << 16);
        v[1] = (uint32_t)p[2] | ((uint32_t)p[3] << 16);
        v[2] = (uint32_t)p[4] | ((uint32_t)p[5] << 16);
        v[3] = (uint32_t)p[6] | ((uint32_t)p[7] << 16);
        return v;
    }
}

/* The chunk of the row-identity sum of wo_a: both WMMA kernels below add K (= per) in chunks of kWoAKC, each chunk one WMMA chain from
 * zero over its 16-wide k steps in order, the partials in chunk order into a total from zero. */
constexpr int kWoAKC = 256;
constexpr int kWoASmallWaves = 8;

/* Small-row form (row identity, the row-identity checks): one block per (16 channels, 16 tokens, group), its waves take the
 * K chunks (wave w: chunks w, w + 8, ...). Fragments come straight from global memory in the layout the tiled kernel reads from LDS
 * (A = W rows j, B = O rows m, lane l: row l & 15, k = 16 s + 8 (l >> 4) + 0..7); the trailing R dims of each head are inverse-rotated
 * and rounded to bf16 with the tiled kernel's expression (cos / sin of the double angle, as float). Wave 0 adds the partials in chunk
 * order, so a token's outputs are the tiled kernel's bits whatever the row count. */
template <bool Vec>
__global__ __launch_bounds__(kWoASmallWaves * 32) void ernie_hca_wo_a_wmma_small_kernel(HcaWoAArgs A) {
    __shared__ hca_v8f part[kWoASmallWaves][32];
    __shared__ float cs_sh[16][32];
    __shared__ float sn_sh[16][32];
    const int j0 = (int)blockIdx.x * 16, m0 = (int)blockIdx.y * 16, g = (int)blockIdx.z;
    const int tid = (int)threadIdx.x, wave = tid >> 5, lane = tid & 31, half_l = lane >> 4;
    const int nope = A.D - A.R;
    const int mr = m0 + (lane & 15) < A.M ? m0 + (lane & 15) : A.M - 1;
    const int jr = j0 + (lane & 15) < A.K ? j0 + (lane & 15) : A.K - 1;
    const uint16_t* orow = A.o + (long long)mr * A.op + (long long)g * A.per + 8 * half_l;
    const uint16_t* wrow = A.w + (long long)(g * A.K + jr) * A.wp + 8 * half_l;
    /* cos / sin of the 16 tokens' angles once per block (R <= 64: at most 32 pairs), as the tiled kernel stages them */
    for (int i = tid; i < 16 * 32; i += kWoASmallWaves * 32) {
        const int t = i >> 5, pr = i & 31;
        if (pr < (A.R >> 1)) {
            if (m0 + t < A.M) {
                const double ang = (double)A.pos[(long long)(m0 + t) * A.ps] * A.tab.inv[pr];
                cs_sh[t][pr] = (float)cos(ang);
                sn_sh[t][pr] = -(float)sin(ang);
            } else {
                /* a padded token (past M): its column of the product is not stored and no other column reads it; no double-precision trig for it
                 * (at M = 1 that was 15 of 16 tokens' worth, per block) */
                cs_sh[t][pr] = 0.0f;
                sn_sh[t][pr] = 0.0f;
            }
        }
    }
    __syncthreads();
    const int nch = (A.per + kWoAKC - 1) / kWoAKC;
    hca_v8f tot = (hca_v8f)(0.0f);
    for (int base = 0; base < nch; base += kWoASmallWaves) {
        const int c = base + wave;
        if (c < nch) {
            const int k0 = c * kWoAKC, k1 = k0 + kWoAKC < A.per ? k0 + kWoAKC : A.per;
            hca_v8f acc = (hca_v8f)(0.0f);
            for (int kk = k0; kk < k1; kk += 16) {
                const hca_u4 wv = hca_ld_u4<Vec>(wrow + kk);
                hca_u4 ov = hca_ld_u4<Vec>(orow + kk);
                const int hk = (kk + 8 * half_l) % A.D;
                if (hk >= nope) {
                    const int pi0 = (hk - nope) >> 1;
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const uint32_t u = ov[e];
                        const float ev = bf2f((uint16_t)(u & 0xffffu));
                        const float od = bf2f((uint16_t)(u >> 16));
                        const float cs = cs_sh[lane & 15][pi0 + e];
                        const float sn = sn_sh[lane & 15][pi0 + e];
                        const uint16_t h0 = f2bf(ev * cs - od * sn);
                        const uint16_t h1 = f2bf(ev * sn + od * cs);
                        ov[e] = (uint32_t)h0 | ((uint32_t)h1 << 16);
                    }
                }
                acc = hca_wmma_bf16(__builtin_bit_cast(hca_v8s, wv), __builtin_bit_cast(hca_v8s, ov), acc);
            }
            part[wave][lane] = acc;
        }
        __syncthreads();
        if (wave == 0) {
            const int cnt = nch - base < kWoASmallWaves ? nch - base : kWoASmallWaves;
            for (int i = 0; i < cnt; ++i) tot = tot + part[i][lane];
        }
        __syncthreads();
    }
    if (wave == 0) {
        /* D[j][m]: lane l holds token m0 + (l & 15), channels j0 + 8 (l >> 4) + 0..7 */
        const int m = m0 + (lane & 15);
        const bool neg = m < A.M && A.vc.log && A.pos[(long long)m * A.ps] < 0;   /* libref refuses a negative position: the row is not stored (validation on) */
        if (neg && half_l == 0 && blockIdx.x == 0 && g == 0) R4D_HCA_RAISE(2);
        if (m < A.M && !neg) {
            uint16_t* row_out = A.out + (long long)m * A.outp + (long long)g * A.K;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int j = j0 + 8 * half_l + e;
                if (j < A.K) row_out[j] = f2bf(tot[e]);
            }
        }
    }
}

template <bool Vec>
__global__ __launch_bounds__(256) void ernie_hca_wo_a_wmma_tiled_kernel(HcaWoAArgs A) {
    static constexpr int BM = 64;
    static constexpr int BN = 128;
    static constexpr int BK = 32;
    static constexpr int PITCH = 40; /* 80B = 20 dwords: 16B aligned, conflict-free 16-lane ds_load_b128 */

    __shared__ __align__(16) uint16_t A_sh[BM][PITCH];
    __shared__ __align__(16) uint16_t B_sh[BN][PITCH];
    __shared__ float cs_sh[BM][32];
    __shared__ float sn_sh[BM][32];

    const int j0 = (int)blockIdx.x * BN;
    const int m0 = (int)blockIdx.y * BM;
    const int g  = (int)blockIdx.z;
    const int tid = (int)threadIdx.x;
    if (g >= A.G || m0 >= A.M || j0 >= A.K) return;

    const int lane = tid & 31, wave = tid >> 5;
    const int nope = A.D - A.R;
    const int rhalf = A.R >> 1;

    /* Precompute cos/sin for all BM tokens once (R <= 64 -> rhalf <= 32, 1 pass/row per wave). */
    for (int rm = wave; rm < BM; rm += 8) {
        if (A.vc.log && m0 + rm < A.M && lane == 0 && A.pos[(long long)(m0 + rm) * A.ps] < 0 && blockIdx.x == 0 && blockIdx.z == 0) R4D_HCA_RAISE(2);
        if (m0 + rm < A.M && lane < rhalf) {
            const double ang = (double)A.pos[(long long)(m0 + rm) * A.ps] * A.tab.inv[lane];
            cs_sh[rm][lane] = (float)cos(ang);
            sn_sh[rm][lane] = -(float)sin(ang);
        }
    }
    __syncthreads();

    /* Coalesced staging coordinates: 4 consecutive threads cover 32 bf16 (64B cache line). */
    const int a_row  = tid >> 2;         /* 0..63 */
    const int a_col  = (tid & 3) * 8;    /* 0, 8, 16, 24 */
    const int b0_row = a_row;            /* 0..63 */
    const int b1_row = a_row + 64;       /* 64..127 */

    const bool a_valid  = (m0 + a_row  < A.M);
    const bool b0_valid = (j0 + b0_row < A.K);
    const bool b1_valid = (j0 + b1_row < A.K);

    const uint16_t* a_gptr  = a_valid  ? (A.o + (long long)(m0 + a_row) * A.op + (long long)g * A.per + a_col) : nullptr;
    const uint16_t* b0_gptr = b0_valid ? (A.w + (long long)(g * A.K + j0 + b0_row) * A.wp + a_col) : nullptr;
    const uint16_t* b1_gptr = b1_valid ? (A.w + (long long)(g * A.K + j0 + b1_row) * A.wp + a_col) : nullptr;

    /* 2 x 4 wave tiling */
    const int col = lane & 15, half_l = lane >> 4;
    const int wy = wave >> 2;  /* 0..1 (tokens: 32/wave) */
    const int wx = wave & 3;   /* 0..3 (channels: 32/wave) */
    const int m_wave = wy * 32;
    const int j_wave = wx * 32;

    hca_v8f acc[2][2], tot[2][2];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int b = 0; b < 2; ++b) { acc[a][b] = (hca_v8f)(0.0f); tot[a][b] = (hca_v8f)(0.0f); }

    const hca_u4 zu4 = {0u, 0u, 0u, 0u};

    /* Prologue: prefetch first BK=32 tile into registers */
    hca_u4 ra  = a_valid  ? hca_ld_u4<Vec>(a_gptr)  : zu4;
    hca_u4 rb0 = b0_valid ? hca_ld_u4<Vec>(b0_gptr) : zu4;
    hca_u4 rb1 = b1_valid ? hca_ld_u4<Vec>(b1_gptr) : zu4;

    int lc = 0;
    for (int k = 0; k < A.per; k += BK) {
        if (k > 0) __syncthreads();

        /* Apply inverse GPT-J RoPE in registers on the trailing R dims of each head (uniform scalar branch). */
        if (lc >= nope) {
            if (a_valid) {
                const int pi0 = (lc - nope + a_col) >> 1;
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const uint32_t w = ra[e];
                    const float ev = bf2f((uint16_t)(w & 0xffffu));
                    const float od = bf2f((uint16_t)(w >> 16));
                    const float cs = cs_sh[a_row][pi0 + e];
                    const float sn = sn_sh[a_row][pi0 + e];
                    const uint16_t h0 = f2bf(ev * cs - od * sn);
                    const uint16_t h1 = f2bf(ev * sn + od * cs);
                    ra[e] = (uint32_t)h0 | ((uint32_t)h1 << 16);
                }
            }
        }

        /* Stage current tile to LDS */
        *reinterpret_cast<hca_u4*>(&A_sh[a_row][a_col])  = ra;
        *reinterpret_cast<hca_u4*>(&B_sh[b0_row][a_col]) = rb0;
        *reinterpret_cast<hca_u4*>(&B_sh[b1_row][a_col]) = rb1;
        __syncthreads();

        /* Prefetch next tile from global memory while computing WMMAs on current LDS tile */
        const int kn = k + BK;
        if (kn < A.per) {
            ra  = a_valid  ? hca_ld_u4<Vec>(a_gptr  + kn) : zu4;
            rb0 = b0_valid ? hca_ld_u4<Vec>(b0_gptr + kn) : zu4;
            rb1 = b1_valid ? hca_ld_u4<Vec>(b1_gptr + kn) : zu4;
        }

        /* 2 WMMA k-steps (kk = 0, 16), 2 x 2 = 4 tiles per wave */
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            const int k_lds = kk + 8 * half_l;
            const hca_v8s af0 = *reinterpret_cast<const hca_v8s*>(&B_sh[j_wave +      col][k_lds]);
            const hca_v8s af1 = *reinterpret_cast<const hca_v8s*>(&B_sh[j_wave + 16 + col][k_lds]);
            const hca_v8s bf0 = *reinterpret_cast<const hca_v8s*>(&A_sh[m_wave +      col][k_lds]);
            const hca_v8s bf1 = *reinterpret_cast<const hca_v8s*>(&A_sh[m_wave + 16 + col][k_lds]);
            acc[0][0] = hca_wmma_bf16(af0, bf0, acc[0][0]);
            acc[0][1] = hca_wmma_bf16(af1, bf0, acc[0][1]);
            acc[1][0] = hca_wmma_bf16(af0, bf1, acc[1][0]);
            acc[1][1] = hca_wmma_bf16(af1, bf1, acc[1][1]);
#ifdef __HIP_DEVICE_COMPILE__
            __builtin_amdgcn_sched_group_barrier(0x100, 4, 0); /* DS reads */
            __builtin_amdgcn_sched_group_barrier(0x008, 4, 0); /* WMMA */
#endif
        }

        /* Row identity: K in chunks of kWoAKC, each one WMMA chain from zero, the partials added in chunk order (as the small-row kernel). */
        if (((k + BK) % kWoAKC) == 0 || k + BK >= A.per) {
#pragma unroll
            for (int a = 0; a < 2; ++a)
#pragma unroll
                for (int b = 0; b < 2; ++b) { tot[a][b] = tot[a][b] + acc[a][b]; acc[a][b] = (hca_v8f)(0.0f); }
        }

        lc += BK;
        if (lc == A.D) lc = 0;
    }
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int b = 0; b < 2; ++b) acc[a][b] = tot[a][b];

    /* Epilogue: lane L holds 8 contiguous output channels for token mr -> single uint4 store */
    const bool out_vec = Vec && (((uintptr_t)A.out & 15u) == 0u) && ((A.outp & 7) == 0) && ((A.K & 7) == 0);
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
        const int mr = m0 + m_wave + mi * 16 + col;
        if (mr < A.M && (!A.vc.log || A.pos[(long long)mr * A.ps] >= 0)) {   /* a row libref refuses (position below 0) is not stored */
            uint16_t* row_out = A.out + (long long)mr * A.outp + (long long)g * A.K;
#pragma unroll
            for (int ni = 0; ni < 2; ++ni) {
                const int out_j0 = j0 + j_wave + ni * 16 + 8 * half_l;
                if (out_j0 < A.K) {
                    uint16_t* op = row_out + out_j0;
                    if (out_vec) {
                        float vals[8];
#pragma unroll
                        for (int e = 0; e < 8; ++e) vals[e] = acc[mi][ni][e];
                        *reinterpret_cast<uint4*>(op) = pack_bf16_uint4(vals);
                    } else {
#pragma unroll
                        for (int e = 0; e < 8; ++e) {
                            if (out_j0 + e < A.K) op[e] = f2bf(acc[mi][ni][e]);
                        }
                    }
                }
            }
        }
    }
}
#endif /* defined(__clang__) || defined(__HIP_DEVICE_COMPILE__) */

/* General fallback kernel for non-contiguous / arbitrary D, R, K shapes. */
__global__ void ernie_hca_wo_a_kernel(HcaWoAArgs A) {
    /* One block of 128 threads (4 waves) per (m, g). Caches rotated bf16 input in LDS (up to per=8192 -> 16 KiB). */
    __shared__ uint16_t rot_sh[8192];
    const int g = (int)blockIdx.x, m = (int)blockIdx.y, tid = (int)threadIdx.x;
    if (g >= A.G || m >= A.M) return;
    if (A.vc.log && A.pos[(long long)m * A.ps] < 0) { if (g == 0 && tid == 0) R4D_HCA_RAISE(2); return; }   /* libref refuses a negative position: the row is not stored */
    const double p = (double)A.pos[(long long)m * A.ps];
    const int nope = A.D - A.R;
    const long long obase = (long long)m * A.op + (long long)(g * A.per) * A.os;

    /* Step 1: Inverse RoPE rounded to bf16 into rot_sh. */
    for (int idx = tid; idx < A.hg * nope; idx += 128) {
        const int hh = idx / nope, c = idx % nope;
        rot_sh[hh * A.D + c] = A.o[obase + (long long)(hh * A.D + c) * A.os];
    }
    const int half = A.R >> 1;
    for (int idx = tid; idx < A.hg * half; idx += 128) {
        const int hh = idx / half, i = idx % half;
        const double ang = p * A.tab.inv[i];
        const float cs = (float)cos(ang), sn = -(float)sin(ang);
        const int c0 = hh * A.D + nope + 2 * i;
        const float ev = bf2f(A.o[obase + (long long)c0 * A.os]);
        const float od = bf2f(A.o[obase + (long long)(c0 + 1) * A.os]);
        rot_sh[c0]     = f2bf(ev * cs - od * sn);
        rot_sh[c0 + 1] = f2bf(ev * sn + od * cs);
    }
    __syncthreads();

    /* Step 2: Grouped GEMV: 4 waves compute K output columns. */
    const int lane = tid & 31, wave = tid >> 5;
    for (int j = wave; j < A.K; j += 4) {
        const long long wb = (long long)(g * A.K + j) * A.wp;
        float sum = 0.0f;
        for (int c = lane; c < A.per; c += 32)
            sum += bf2f(rot_sh[c]) * bf2f(A.w[wb + (long long)c * A.ws]);
        sum = wave_sum(sum);
        if (lane == 0)
            A.out[(long long)m * A.outp + (long long)(g * A.K + j) * A.outs] = f2bf(sum);
    }
}

/* The launch's last-writer table: the op's scratch (RadArgs.scratch, sized by the scratch hook below for the band's upper bound of M). False if the caller gave none or too little: the
 * launch then keeps the O(M^2) scan, which stores the same rows. */
static inline bool dedupe_setup(const RadArgs* a, long long M, int** tab, int* shift) {
    const long long need = dedupe_bytes(M);
    if (!a->scratch || a->scratch_bytes < need) return false;
    *tab = (int*)a->scratch;
    *shift = 32 - dedupe_log2(M);
    return true;
}

} /* namespace r4d_ernie_hca */

extern "C" {

/* Bytes of scratch (RadScratchFn): the last-writer table for the largest M of the band; none up to kDedupeRows rows. */
int64_t r4d_rad_ernie_hca_state_write_scratch(const RadArgs* a) {
    const long long M = r4d_geti_or(a, "M", 0);
    return M > r4d_ernie_hca::kDedupeRows ? r4d_ernie_hca::dedupe_bytes(M) : 0;
}
int64_t r4d_rad_ernie_hca_compress_scratch(const RadArgs* a) {
    return r4d_rad_ernie_hca_state_write_scratch(a);
}

int r4d_rad_ernie_hca_state_write(const RadArgs* a, RadStream stream) {
    using namespace r4d_ernie;
    using namespace r4d_ernie_hca;
    if (!a || !a->t || a->n_t < 6) return RAD_E_INVAL;
    RadTensor stv = a->t[5];
    const RadTensor &kv = a->t[0], &sc = a->t[1], &ape = a->t[2], &pos = a->t[3], &slot = a->t[4];
    const long long ratio = r4d_geti_or(a, "ratio", 0), D = r4d_geti_or(a, "head_dim", 0);
    if (!legal_ratio(ratio) || D < 1 || D > 4096 || !block_hint_ok(a, a->t[5])) return RAD_E_SHAPE;
    if (!pool_rows(&stv)) return RAD_E_SHAPE;
    const RadTensor& st = stv;
    const long long coff = ratio == 4 ? 2 : 1, W = coff * D;
    if (kv.rank != 2 || sc.rank != 2 || ape.rank != 2 || pos.rank != 1 || slot.rank != 1 || st.rank != 2 ||
        kv.shape[0] < 1 || kv.shape[0] > 65535 || kv.shape[1] != W || sc.shape[0] != kv.shape[0] || sc.shape[1] != W ||
        ape.shape[0] != ratio || ape.shape[1] != W || pos.shape[0] != kv.shape[0] || slot.shape[0] != kv.shape[0] ||
        st.shape[0] < 1 || st.shape[1] != 2 * W || !pos_strides(kv) || !pos_strides(sc) || !pos_strides(ape) ||
        !pos_strides(pos) || !pos_strides(slot) || !pos_strides(st))
        return RAD_E_SHAPE;
    if (kv.dtype != RAD_F32 || sc.dtype != RAD_F32 || ape.dtype != RAD_F32 || st.dtype != RAD_F32 ||
        pos.dtype != RAD_I32 || slot.dtype != RAD_I32)
        return RAD_E_DTYPE;
    if (!kv.data || !sc.data || !ape.data || !pos.data || !slot.data || !st.data) return RAD_E_INVAL;
    for (const RadTensor* in : {&kv, &sc, &ape, &pos, &slot})
        if (operands_overlap(st, *in)) return RAD_E_INVAL;
    bool has_dup_slot = false;
#if !defined(__HIP__) && !defined(__HIP_PLATFORM_AMD__)
    if (r4d_hca_scan_on())
    for (long long m = 0; m < kv.shape[0]; ++m) {
        const long long s = ((const int32_t*)slot.data)[m * slot.stride[0]];
        if (s < -1 || s >= st.shape[0]) return RAD_E_SHAPE;
        if (s >= 0 && ((const int32_t*)pos.data)[m * pos.stride[0]] < 0) return RAD_E_SHAPE;
        if (s >= 0 && !has_dup_slot && kv.shape[0] <= 256) {
            for (long long m2 = 0; m2 < m; ++m2)
                if (((const int32_t*)slot.data)[m2 * slot.stride[0]] == s) { has_dup_slot = true; break; }
        }
    }
#endif
    R4D_ARGS(HcaStateWriteArgs, P);
    R4D_VC(P, r4d_verdict::kFamHca);
    P.kv = (const float*)kv.data; P.sc = (const float*)sc.data; P.ape = (const float*)ape.data;
    P.pos = (const int32_t*)pos.data; P.slot = (const int32_t*)slot.data; P.st = (float*)st.data;
    P.kvp = kv.stride[0]; P.kvs = kv.stride[1]; P.scp = sc.stride[0]; P.scs = sc.stride[1];
    P.apep = ape.stride[0]; P.apes = ape.stride[1]; P.ps = pos.stride[0]; P.sls = slot.stride[0];
    P.stp = st.stride[0]; P.sts = st.stride[1];
    P.M = (int)kv.shape[0]; P.W = (int)W; P.ratio = (int)ratio; P.S = (int)st.shape[0];
    if (P.M == 1 || has_dup_slot) {
        R4D_KLAUNCH(R4D_KNAME(ernie_hca_state_write_seq_kernel), dim3((unsigned)((W + 255) / 256)), dim3(256), (hipStream_t)stream, P);
    } else {
        if (P.M > kDedupeRows && dedupe_setup(a, P.M, &P.dd, &P.ddshift)) {
            R4D_ARGS(HcaDedupeArgs, D);
            D.tab = P.dd; D.cap = 1 << (32 - P.ddshift);
            R4D_KLAUNCH(R4D_KNAME(ernie_hca_dedupe_clear_kernel), dim3((unsigned)((2 * D.cap + 255) / 256)), dim3(256), (hipStream_t)stream, D);
            R4D_KLAUNCH(R4D_KNAME(ernie_hca_state_last_mark_kernel), dim3((unsigned)((P.M + 255) / 256)), dim3(256), (hipStream_t)stream, P);
            r4d_verdict::stats().dedupe_launches.fetch_add(2, std::memory_order_relaxed);
        }
        R4D_KLAUNCH(R4D_KNAME(ernie_hca_state_write_par_kernel), dim3((unsigned)((W + 255) / 256), (unsigned)P.M), dim3(256), (hipStream_t)stream, P);
    }
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

static int launch_hca_gather_slots(const RadArgs* a, RadStream stream, bool with_origin) {
    using namespace r4d_ernie;
    using namespace r4d_ernie_hca;
    if (!a || !a->t || a->n_t < (with_origin ? 5 : 4)) return RAD_E_INVAL;
    const RadTensor &pos = a->t[0], &req = a->t[1], &bt = a->t[2], &origin = a->t[3], &out = a->t[with_origin ? 4 : 3];
    const long long ratio = r4d_geti_or(a, "ratio", 0), bs = r4d_geti_or(a, "block_size", 0);
    if (!legal_ratio(ratio) || bs < 1 || bs > INT32_MAX) return RAD_E_SHAPE;
    const long long Wn = (ratio == 4 ? 2 : 1) * ratio;
    if (pos.rank != 1 || req.rank != 1 || bt.rank != 2 || out.rank != 2 || pos.shape[0] < 1 || pos.shape[0] > 65535 ||
        req.shape[0] != pos.shape[0] || bt.shape[0] < 1 || bt.shape[1] < 1 || out.shape[0] != pos.shape[0] ||
        bt.shape[0] > INT32_MAX || bt.shape[1] > INT32_MAX ||
        out.shape[1] != Wn || !pos_strides(pos) || !pos_strides(req) || !pos_strides(bt) || !pos_strides(out) ||
        (with_origin && (origin.rank != 1 || origin.shape[0] != bt.shape[0] || !pos_strides(origin))))
        return RAD_E_SHAPE;
    if (pos.dtype != RAD_I32 || req.dtype != RAD_I32 || bt.dtype != RAD_I32 || out.dtype != RAD_I32 ||
        (with_origin && origin.dtype != RAD_I32))
        return RAD_E_DTYPE;
    if (!pos.data || !req.data || !bt.data || !out.data || (with_origin && !origin.data)) return RAD_E_INVAL;
#if !defined(__HIP__) && !defined(__HIP_PLATFORM_AMD__)
    const long long M = pos.shape[0], S = bt.shape[0], B = bt.shape[1];
    if (r4d_hca_scan_on())
    for (long long m = 0; m < M; ++m) {
        const long long p = ((const int32_t*)pos.data)[m * pos.stride[0]];
        const long long r = ((const int32_t*)req.data)[m * req.stride[0]];
        if (p < 0 || r < 0 || r >= S) return RAD_E_SHAPE;
        const long long first = with_origin ? ((const int32_t*)origin.data)[r * origin.stride[0]] : 0;
        if (first < 0 || first % bs || first > p) return RAD_E_SHAPE;
        if ((p + 1) % ratio) continue;
        for (long long w = 0; w < Wn; ++w) {
            const long long absolute = p - Wn + 1 + w;
            if (absolute < 0) continue;
            const long long q = absolute - first;
            if (q < 0 || q / bs >= B) return RAD_E_SHAPE;
            const long long pg = ((const int32_t*)bt.data)[r * bt.stride[0] + (q / bs) * bt.stride[1]];
            if (pg < 0 || pg > (INT32_MAX - q % bs) / bs) return RAD_E_SHAPE;
        }
    }
#endif
    R4D_ARGS(HcaGatherArgs, P);
    R4D_VC(P, r4d_verdict::kFamHca);
    P.pos = (const int32_t*)pos.data; P.req = (const int32_t*)req.data; P.bt = (const int32_t*)bt.data; P.out = (int32_t*)out.data;
    P.ps = pos.stride[0]; P.rs = req.stride[0]; P.btp = bt.stride[0]; P.bts = bt.stride[1];
    if (with_origin) { P.origin = (const int32_t*)origin.data; P.os = origin.stride[0]; }
    P.outp = out.stride[0]; P.outs = out.stride[1];
    P.M = (int)pos.shape[0]; P.S = (int)bt.shape[0]; P.B = (int)bt.shape[1];
    P.ratio = (int)ratio; P.bs = (int)bs; P.Wn = (int)Wn;
    const long long total = (long long)P.M * Wn;
    R4D_KLAUNCH(R4D_KNAME(ernie_hca_gather_slots_kernel), dim3((unsigned)((total + 255) / 256)), dim3(256), (hipStream_t)stream, P);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

int r4d_rad_ernie_hca_gather_slots(const RadArgs* a, RadStream stream) {
    return launch_hca_gather_slots(a, stream, false);
}

int r4d_rad_ernie_hca_gather_slots_origin(const RadArgs* a, RadStream stream) {
    return launch_hca_gather_slots(a, stream, true);
}

int r4d_rad_ernie_hca_compress(const RadArgs* a, RadStream stream) {
    using namespace r4d_ernie;
    using namespace r4d_ernie_hca;
    if (!a || !a->t || a->n_t < 6) return RAD_E_INVAL;
    RadTensor stv = a->t[0], outv = a->t[5];
    const RadTensor &gs = a->t[1], &pos = a->t[2], &os = a->t[3], &nw = a->t[4];
    /* out_layout (kv-cmpblock, additive, as libref): "token" (default) stores at out_slots[m] of a per-token pool; "block" views the pool [pages, 1, B, e]
     * as [pages, B * e] rows, B = ratio, so a completing token's slot page * B + B - 1 names row `page`. */
    const char* cmpl = r4d_gets(a, "out_layout");
    const bool blk_rows = cmpl && !std::strcmp(cmpl, "block");
    if (cmpl && !blk_rows && std::strcmp(cmpl, "token")) return RAD_E_SHAPE;
    if (!pool_rows(&stv)) return RAD_E_SHAPE;
    long long slot_div = 1;
    if (blk_rows) {
        if (outv.rank != 4 || outv.shape[1] != 1 || outv.shape[2] < 1 || outv.stride[3] != 1 || outv.stride[2] != outv.shape[3]) return RAD_E_SHAPE;
        slot_div = outv.shape[2];
        outv.rank = 2;
        outv.shape[1] = slot_div * outv.shape[3];
        outv.stride[1] = 1;
    } else if (!pool_rows(&outv)) {
        return RAD_E_SHAPE;
    }
    const RadTensor &st = stv, &out = outv;
    const long long ratio = r4d_geti_or(a, "ratio", 0), D = r4d_geti_or(a, "head_dim", 0),
                    R = r4d_geti_or(a, "rope_dim", 0), qb = r4d_geti_or(a, "quant_block", 64);
    if (blk_rows && slot_div != ratio) return RAD_E_SHAPE;
    const double eps = r4d_getf(a, "eps", 1e-6), theta = r4d_getf(a, "theta", 0.0);
    const char* layout = r4d_gets(a, "layout");
    const bool packed = layout && !std::strcmp(layout, "fp8_ds_mla");
    if (layout && !packed && std::strcmp(layout, "bf16_full")) return RAD_E_SHAPE;
    if (!legal_ratio(ratio) || D < 2 || D > 4096 || R < 2 || (R & 1) || R > D || R > 2 * kMaxRopePairs ||
        !(eps >= 0.0) || !std::isfinite(eps) || !std::isfinite(theta) || theta <= 0.0)
        return RAD_E_SHAPE;
    const long long nope = D - R, coff = ratio == 4 ? 2 : 1, W = coff * D, Wn = coff * ratio;
    if (packed && (qb < 1 || nope < 1 || nope % qb)) return RAD_E_SHAPE;
    const long long nb = packed ? nope / qb : 0, rb = packed ? nope + 2 * R + nb + 1 : D;
    if (st.rank != 2 || gs.rank != 2 || pos.rank != 1 || os.rank != 1 || nw.rank != 1 || out.rank != 2 ||
        gs.shape[0] < 1 || gs.shape[0] > 65535 || st.shape[0] < 1 || st.shape[1] != 2 * W || gs.shape[1] != Wn ||
        pos.shape[0] != gs.shape[0] || os.shape[0] != gs.shape[0] || nw.shape[0] != D || out.shape[0] < 1 ||
        out.shape[1] != rb || !pos_strides(st) || !pos_strides(gs) || !pos_strides(pos) || !pos_strides(os) ||
        !pos_strides(nw) || !pos_strides(out))
        return RAD_E_SHAPE;
    if (st.dtype != RAD_F32 || gs.dtype != RAD_I32 || pos.dtype != RAD_I32 || os.dtype != RAD_I32 ||
        nw.dtype != RAD_BF16 || out.dtype != (packed ? RAD_U8 : RAD_BF16))
        return RAD_E_DTYPE;
    if (!st.data || !gs.data || !pos.data || !os.data || !nw.data || !out.data) return RAD_E_INVAL;
    for (const RadTensor* in : {&st, &gs, &pos, &os, &nw})
        if (operands_overlap(out, *in)) return RAD_E_INVAL;
    ErnieYarn yarn;
    if (!parse_yarn(a, &yarn)) return RAD_E_SHAPE;
    R4D_ARGS(HcaCompressArgs, P);
    R4D_VC(P, r4d_verdict::kFamHca);
    if (!fill_rope_inv(theta, R, yarn, &P.tab)) return RAD_E_SHAPE;
    bool has_dup_slot = false;
#if !defined(__HIP__) && !defined(__HIP_PLATFORM_AMD__)
    if (r4d_hca_scan_on())
    for (long long m = 0; m < gs.shape[0]; ++m) {
        const long long o = ((const int32_t*)os.data)[m * os.stride[0]];
        if (o < -1 || o >= out.shape[0] * slot_div) return RAD_E_SHAPE;
        if (o < 0) continue;
        const long long p = ((const int32_t*)pos.data)[m * pos.stride[0]];
        if (p < 0) return RAD_E_SHAPE;
        long long unmasked = 0;
        for (long long w = 0; w < Wn; ++w) {
            const long long s = ((const int32_t*)gs.data)[m * gs.stride[0] + w * gs.stride[1]];
            if (s < -1 || s >= st.shape[0]) return RAD_E_SHAPE;
            unmasked += (s >= 0);
        }
        if (((p + 1) % ratio == 0) && !unmasked) return RAD_E_SHAPE;
        if (blk_rows && ((p + 1) % ratio == 0) && o % slot_div != slot_div - 1) return RAD_E_SHAPE;
        if (((p + 1) % ratio == 0) && !has_dup_slot && gs.shape[0] <= 256) {
            for (long long m2 = 0; m2 < m; ++m2) {
                const long long o2 = ((const int32_t*)os.data)[m2 * os.stride[0]];
                const long long p2 = ((const int32_t*)pos.data)[m2 * pos.stride[0]];
                if (o2 == o && p2 >= 0 && ((p2 + 1) % ratio == 0)) { has_dup_slot = true; break; }
            }
        }
    }
#endif
    P.st = (const float*)st.data; P.gs = (const int32_t*)gs.data; P.pos = (const int32_t*)pos.data;
    P.os = (const int32_t*)os.data; P.nw = (const uint16_t*)nw.data; P.out = out.data;
    P.stp = st.stride[0]; P.sts = st.stride[1]; P.gsp = gs.stride[0]; P.gss = gs.stride[1];
    P.ps = pos.stride[0]; P.oss = os.stride[0]; P.nws = nw.stride[0];
    P.outp = out.stride[0]; P.outs = out.stride[1];
    P.M = (int)gs.shape[0]; P.S = (int)st.shape[0]; P.Rr = (int)(out.shape[0] * slot_div); P.div = (int)slot_div;
    P.ratio = (int)ratio; P.D = (int)D; P.R = (int)R; P.qb = (int)qb; P.packed = packed ? 1 : 0;
    P.coff = (int)coff; P.W = (int)W; P.Wn = (int)Wn; P.eps = (float)eps;
#ifdef R4D_SYNC_LAUNCH
    const unsigned ev0 = P.vc.log ? P.vc.log->cnt : 0u;   /* the emulator: the launch completes before it returns */
#endif
    if (P.M == 1 || has_dup_slot) {
        R4D_KLAUNCH(R4D_KNAME(ernie_hca_compress_seq_kernel), dim3(1), dim3(256), (hipStream_t)stream, P);
    } else {
        if (P.M > kDedupeRows && dedupe_setup(a, P.M, &P.dd, &P.ddshift)) {
            R4D_ARGS(HcaDedupeArgs, D);
            D.tab = P.dd; D.cap = 1 << (32 - P.ddshift);
            R4D_KLAUNCH(R4D_KNAME(ernie_hca_dedupe_clear_kernel), dim3((unsigned)((2 * D.cap + 255) / 256)), dim3(256), (hipStream_t)stream, D);
            R4D_KLAUNCH(R4D_KNAME(ernie_hca_compress_last_mark_kernel), dim3((unsigned)((P.M + 255) / 256)), dim3(256), (hipStream_t)stream, P);
            r4d_verdict::stats().dedupe_launches.fetch_add(2, std::memory_order_relaxed);
        }
        R4D_KLAUNCH(R4D_KNAME(ernie_hca_compress_par_kernel), dim3((unsigned)P.M), dim3(256), (hipStream_t)stream, P);
    }
    if (hipGetLastError() != hipSuccess) return RAD_E_DEVICE;
#ifdef R4D_SYNC_LAUNCH
    /* Not card form: the launcher refuses as libref does. The host scan above judges the operands in host memory; the one rule it cannot judge, a pooled row that is
     * not finite (it depends on the state's values), is the kernel's event, which the launcher turns into libref's refusal. Card form: the event is polled. */
    if (!r4d_verdict::card_form() && P.vc.log && P.vc.log->cnt != ev0) return R4D_NO(RAD_E_SHAPE);
#endif
    return RAD_OK;
}

static int launch_hca_union_attn(const RadArgs* a, RadStream stream, bool with_origin) {
    using namespace r4d_ernie;
    using namespace r4d_ernie_hca;
    if (!a || !a->t || a->n_t < (with_origin ? 10 : 9)) return RAD_E_INVAL;
    const RadTensor &q = a->t[0], &wk = a->t[1], &ck = a->t[2], &pos = a->t[3], &req = a->t[4],
                    &wt = a->t[5], &ct = a->t[6], &origin = a->t[7];
    const RadTensor &sink = a->t[with_origin ? 8 : 7], &out = a->t[with_origin ? 9 : 8];
    const long long H = r4d_geti_or(a, "heads", 0), D = r4d_geti_or(a, "head_dim", 0),
                    ratio = r4d_geti_or(a, "ratio", 0), win = r4d_geti_or(a, "window", 128);
    const double scale = r4d_getf(a, "scale", 0.0);
    const char* cl = r4d_gets(a, "cmp_layout");
    const bool tok_rows = cl && !std::strcmp(cl, "token");
    /* "block" (kv-cmpblock): cmp_kv is [pages, 1, ratio, D / ratio], a page is one row, row j is page cmp_table[req][j]. The kernels run it as the token layout
     * with the row's page at the table entry of its completing token ((j + 1) * ratio - 1) / bc = j and a zero offset inside the page (ckp2 = 0), so no kernel changes. */
    const bool blk_rows = cl && !std::strcmp(cl, "block");
    if (cl && !tok_rows && !blk_rows && std::strcmp(cl, "dense")) return RAD_E_SHAPE;
    /* optional block_size (kv-cmpblock): the compressed group's block, when given the pool's page rows */
    const bool bs_bad = find_p(a, "block_size") && (ck_rank4_bad(a) || r4d_geti_or(a, "block_size", 0) != a->t[2].shape[2]);
    if (H < 1 || H > 65535 || D < 1 || D > 4096 || !legal_ratio(ratio) || win < 1 || win > INT32_MAX / 2 ||
        !std::isfinite(scale) || scale <= 0.0 || q.rank != 2 || wk.rank != 4 || ck.rank != 4 || pos.rank != 1 ||
        req.rank != 1 || wt.rank != 2 || ct.rank != 2 || sink.rank != 1 || out.rank != 2 || q.shape[0] < 1 ||
        q.shape[0] > 65535 || q.shape[1] != H * D || wk.shape[0] < 1 || wk.shape[1] != 1 || wk.shape[2] < 1 ||
        wk.shape[3] != D || ck.shape[0] < 1 || ck.shape[1] != 1 || ck.shape[2] < 1 ||
        (blk_rows ? (ck.shape[2] != ratio || ck.shape[2] * ck.shape[3] != D || ck.stride[2] != ck.shape[3] * ck.stride[3]) : ck.shape[3] != D) || bs_bad ||
        pos.shape[0] != q.shape[0] || req.shape[0] != q.shape[0] || wt.shape[0] < 1 || wt.shape[1] < 1 ||
        ct.shape[0] != wt.shape[0] || ct.shape[1] < 1 || sink.shape[0] != H || out.shape[0] != q.shape[0] ||
        out.shape[1] != H * D || !pos_strides(q) || !pos_strides(wk) || !pos_strides(ck) || !pos_strides(pos) ||
        !pos_strides(req) || !pos_strides(wt) || !pos_strides(ct) || !pos_strides(sink) || !pos_strides(out) ||
        out.stride[0] < out.shape[1] * out.stride[1] ||
        wt.shape[0] > INT32_MAX || wt.shape[1] > INT32_MAX || ct.shape[1] > INT32_MAX || wk.shape[0] > INT32_MAX ||
        wk.shape[2] > INT32_MAX || ck.shape[0] > INT32_MAX || ck.shape[2] > INT32_MAX ||
        (with_origin && (origin.rank != 1 || origin.shape[0] != wt.shape[0] || !pos_strides(origin))))
        return RAD_E_SHAPE;
    if (q.dtype != RAD_BF16 || wk.dtype != RAD_BF16 || ck.dtype != RAD_BF16 || out.dtype != RAD_BF16 ||
        pos.dtype != RAD_I32 || req.dtype != RAD_I32 || wt.dtype != RAD_I32 || ct.dtype != RAD_I32 || sink.dtype != RAD_F32 ||
        (with_origin && origin.dtype != RAD_I32))
        return RAD_E_DTYPE;
    if (!q.data || !wk.data || !ck.data || !pos.data || !req.data || !wt.data || !ct.data || !sink.data || !out.data ||
        (with_origin && !origin.data))
        return RAD_E_INVAL;
#if !defined(__HIP__) && !defined(__HIP_PLATFORM_AMD__)
    /* libref's validation (hca_union_attn), before the first store: a sink that is NaN or +inf, a negative position, a request out of range, an origin that is
     * negative, not a multiple of the window block, or past the oldest window token, and every table entry a row reads: below -1, at or past the pool, or a
     * block past the table width. -1 is skipped. A card build cannot read the tables here (device memory); its kernels latch the page violations instead
     * (R4D_HCA_PAGE_LATCH). R4D_HCA_VALIDATE=0 turns the scan off (a test of the latch). */
    if (r4d_hca_scan_on()) {
        for (long long h = 0; h < H; ++h) {
            const float v = ((const float*)sink.data)[h * sink.stride[0]];
            if (std::isnan(v) || v == INFINITY) return R4D_NO(RAD_E_SHAPE);
        }
        const long long Sq = wt.shape[0], Bw = wt.shape[1], Bc = ct.shape[1], bw = wk.shape[2], bc = ck.shape[2], Pw = wk.shape[0], Pc = ck.shape[0];
        auto page_ok = [](const RadTensor& tab, long long r, long long blk, long long width, long long pages) {
            if (blk >= width) return false;
            const long long pg = ((const int32_t*)tab.data)[r * tab.stride[0] + blk * tab.stride[1]];
            return pg >= -1 && pg < pages;
        };
        for (long long m = 0; m < q.shape[0]; ++m) {
            const long long p = ((const int32_t*)pos.data)[m * pos.stride[0]], r = ((const int32_t*)req.data)[m * req.stride[0]];
            if (p < 0 || r < 0 || r >= Sq) return R4D_NO(RAD_E_SHAPE);
            const long long first = with_origin ? ((const int32_t*)origin.data)[r * origin.stride[0]] : 0;
            const long long start = std::max<long long>(0, p - win + 1);
            if (first < 0 || first % bw || first > start) return R4D_NO(RAD_E_SHAPE);
            const long long n = (p + 1) / ratio;
            for (long long j = 0; j < n; ++j)
                if (!page_ok(ct, r, ((tok_rows || blk_rows) ? (j + 1) * ratio - 1 : j) / bc, Bc, Pc)) return R4D_NO(RAD_E_SHAPE);
            for (long long t = start; t <= p; ++t)
                if (!page_ok(wt, r, (t - first) / bw, Bw, Pw)) return R4D_NO(RAD_E_SHAPE);
        }
    }
#endif
    R4D_ARGS(HcaUnionArgs, P);
    P.q = (const uint16_t*)q.data; P.wk = (const uint16_t*)wk.data; P.ck = (const uint16_t*)ck.data;
    P.pos = (const int32_t*)pos.data; P.req = (const int32_t*)req.data;
    P.wt = (const int32_t*)wt.data; P.ct = (const int32_t*)ct.data;
    P.sink = (const float*)sink.data; P.out = (uint16_t*)out.data;
    P.qp = q.stride[0]; P.qs = q.stride[1];
    P.wkp0 = wk.stride[0]; P.wkp2 = wk.stride[2]; P.wks = wk.stride[3];
    P.ckp0 = ck.stride[0]; P.ckp2 = blk_rows ? 0 : ck.stride[2]; P.cks = ck.stride[3];
    P.ps = pos.stride[0]; P.rs = req.stride[0];
    P.wtp = wt.stride[0]; P.wts = wt.stride[1]; P.ctp = ct.stride[0]; P.cts = ct.stride[1];
    if (with_origin) { P.origin = (const int32_t*)origin.data; P.os = origin.stride[0]; }
    P.sks = sink.stride[0]; P.outp = out.stride[0]; P.outs = out.stride[1];
    P.M = (int)q.shape[0]; P.H = (int)H; P.D = (int)D; P.ratio = (int)ratio; P.win = (int)win;
    P.S = (int)wt.shape[0]; P.Bw = (int)wt.shape[1]; P.Bc = (int)ct.shape[1];
    P.bw = (int)wk.shape[2]; P.bc = (int)ck.shape[2]; P.Pw = (int)wk.shape[0]; P.Pc = (int)ck.shape[0];
    P.tok_rows = (tok_rows || blk_rows) ? 1 : 0; P.scale = (float)scale;
    /* Card form: the validator kernel runs ahead of the attention kernel on the same stream (see ernie_hca_union_validate_kernel). Where the launch completes
     * before it returns and the operands are host memory (not card form), the host scan above has refused already. */
    P.check = hca_gate_on() ? 1 : 0;
    if (P.check) {
        /* The call site's gate word is found from the operand pointers (equal on every issue and replay of the site). */
        const r4d_verdict::Launch vl = r4d_verdict::acquire_gated(r4d_verdict::kFamHca, P.q, P.wk, P.ck, P.pos, P.req, P.wt, P.ct, P.origin, P.sink, P.out);
        if (!vl.ok) return RAD_E_DEVICE;
        P.vc = vl.vc;
        R4D_KLAUNCH(R4D_KNAME(ernie_hca_gate_open_kernel), dim3(1), dim3(32), (hipStream_t)stream, P);
        R4D_KLAUNCH(R4D_KNAME(ernie_hca_union_validate_kernel), dim3((unsigned)P.M), dim3(256), (hipStream_t)stream, P);
        r4d_verdict::stats().validator_launches.fetch_add(2, std::memory_order_relaxed);
    } else {
        const r4d_verdict::Launch vl = r4d_verdict::acquire(r4d_verdict::kFamHca);
        if (!vl.ok) return RAD_E_DEVICE;
        P.vc = vl.vc;
    }
    /* R4D_HCA_UNION_BATCH=0 selects the one-row-at-a-time compressed pass (A/B runs and the equality tests); anything else, or unset, the batched one. Same values either way. */
    { const char* e = std::getenv("R4D_HCA_UNION_BATCH"); P.pipe = (e && e[0] == '0' && e[1] == 0) ? 0 : 1; }
    /* R4D_HCA_UNION_BATCH_MIN_CTX=N (unset or 0: no floor, the branch's default): the batched pass only where the host's bound on the context (RadArgs::kv_used_max_h, a keyed bucket
     * on decode passes, so the choice is constant across replays of a pass) is at least N tokens. A dispatch knob for the short-context measurements, not a default: the code shows
     * only a fixed prologue (two table batches and one row batch before the first row is scored), which does not explain the measured 1.5 ms at 1K. Same values either way. */
    if (P.pipe && a->kv_used_max_h > 0) { const char* f = std::getenv("R4D_HCA_UNION_BATCH_MIN_CTX"); if (f && std::atoll(f) > (long long)a->kv_used_max_h) P.pipe = 0; }

    if (D == 512) {
        const bool vec = (P.qs == 1 && P.wks == 1 && P.cks == 1 && P.outs == 1 &&
                          ((((uintptr_t)P.q | (uintptr_t)P.wk | (uintptr_t)P.ck | (uintptr_t)P.out) & 15u) == 0u) &&
                          (P.qp & 7) == 0 && (P.wkp0 & 7) == 0 && (P.wkp2 & 7) == 0 &&
                          (P.ckp0 & 7) == 0 && (P.ckp2 & 7) == 0 && (P.outp & 7) == 0);
        /* R4D_HCA_DECODE_MAX (default 4, the shipped rule): the largest query-row count served by the 8-wave split-KV decode kernel
         * (16 pins every 1..16-row step to it, 0 pins every step to the wide kernel). Both kernels compute the same bits for a row (hca512_part). */
        static const int hca_decode_max = r4d_ernie::r4d_pin_env("R4D_HCA_DECODE_MAX", r4d_ernie::pin::kHcaDecodeMax);
        if (r4d_ernie::pin::hca_decode(P.M, hca_decode_max)) {
            const dim3 grid((unsigned)H, (unsigned)P.M);
            if (vec) {
                R4D_KLAUNCH(R4D_KNAME(ernie_hca_union_attn_d512_decode_kernel<true>), grid, dim3(256), (hipStream_t)stream, P);
            } else {
                R4D_KLAUNCH(R4D_KNAME(ernie_hca_union_attn_d512_decode_kernel<false>), grid, dim3(256), (hipStream_t)stream, P);
            }
        } else {
            const int nwaves = (P.M >= 16 && H >= 8) ? 8 : 4;
            const dim3 grid((unsigned)((H + nwaves - 1) / nwaves), (unsigned)P.M);
            const dim3 block((unsigned)(nwaves * 32));
            if (vec) {
                R4D_KLAUNCH(R4D_KNAME(ernie_hca_union_attn_d512_kernel<true>), grid, block, (hipStream_t)stream, P);
            } else {
                R4D_KLAUNCH(R4D_KNAME(ernie_hca_union_attn_d512_kernel<false>), grid, block, (hipStream_t)stream, P);
            }
        }
    } else if (D == 128) {
        R4D_KLAUNCH(R4D_KNAME(ernie_hca_union_attn_d128_kernel), dim3((unsigned)H, (unsigned)P.M), dim3(128), (hipStream_t)stream, P);
    } else {
        R4D_KLAUNCH(R4D_KNAME(ernie_hca_union_attn_gen_kernel), dim3((unsigned)H, (unsigned)P.M), dim3(256), (hipStream_t)stream, P);
    }
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

int r4d_rad_ernie_hca_union_attn(const RadArgs* a, RadStream stream) {
    return launch_hca_union_attn(a, stream, false);
}

int r4d_rad_ernie_hca_union_attn_origin(const RadArgs* a, RadStream stream) {
    return launch_hca_union_attn(a, stream, true);
}

int r4d_rad_ernie_hca_wo_a(const RadArgs* a, RadStream stream) {
    using namespace r4d_ernie;
    using namespace r4d_ernie_hca;
    if (!a || !a->t || a->n_t < 4) return RAD_E_INVAL;
    const RadTensor &o = a->t[0], &pos = a->t[1], &w = a->t[2], &out = a->t[3];
    const long long H = r4d_geti_or(a, "heads", 0), D = r4d_geti_or(a, "head_dim", 0), R = r4d_geti_or(a, "rope_dim", 0),
                    G = r4d_geti_or(a, "groups", 0), K = r4d_geti_or(a, "rank", 0);
    const double theta = r4d_getf(a, "theta", 0.0);
    ErnieYarn yarn;
    if (!parse_yarn(a, &yarn)) return RAD_E_SHAPE;
    if (H < 1 || D < 2 || D > 4096 || R < 2 || (R & 1) || R > D || R > 2 * kMaxRopePairs || G < 1 || G > 65535 ||
        H % G || K < 1 || !std::isfinite(theta) || theta <= 0.0)
        return RAD_E_SHAPE;
    const long long per = (H / G) * D;
    if (per > 8192) return RAD_E_SHAPE;
    if (o.rank != 2 || pos.rank != 1 || w.rank != 2 || out.rank != 2 || o.shape[0] < 1 || o.shape[0] > 65535 ||
        o.shape[1] != H * D || pos.shape[0] != o.shape[0] || w.shape[0] != G * K || w.shape[1] != per ||
        out.shape[0] != o.shape[0] || out.shape[1] != G * K || !pos_strides(o) || !pos_strides(pos) ||
        !pos_strides(w) || !pos_strides(out) || o.stride[0] < o.shape[1] * o.stride[1] || out.stride[0] < out.shape[1] * out.stride[1])
        return RAD_E_SHAPE;
    if (o.dtype != RAD_BF16 || w.dtype != RAD_BF16 || out.dtype != RAD_BF16 || pos.dtype != RAD_I32) return RAD_E_DTYPE;
    if (!o.data || !pos.data || !w.data || !out.data) return RAD_E_INVAL;
    R4D_ARGS(HcaWoAArgs, P);
    R4D_VC(P, r4d_verdict::kFamHca);
    if (!fill_rope_inv(theta, R, yarn, &P.tab)) return RAD_E_SHAPE;
#if !defined(__HIP__) && !defined(__HIP_PLATFORM_AMD__)
    if (r4d_hca_scan_on())
    for (long long m = 0; m < o.shape[0]; ++m)
        if (((const int32_t*)pos.data)[m * pos.stride[0]] < 0) return RAD_E_SHAPE;
#endif
    P.o = (const uint16_t*)o.data; P.pos = (const int32_t*)pos.data; P.w = (const uint16_t*)w.data; P.out = (uint16_t*)out.data;
    P.op = o.stride[0]; P.os = o.stride[1]; P.ps = pos.stride[0]; P.wp = w.stride[0]; P.ws = w.stride[1];
    P.outp = out.stride[0]; P.outs = out.stride[1];
    P.M = (int)o.shape[0]; P.H = (int)H; P.D = (int)D; P.R = (int)R; P.G = (int)G; P.K = (int)K;
    P.hg = (int)(H / G); P.per = (int)per;

    const bool contig = (P.os == 1 && P.ws == 1 && P.outs == 1);
    const bool vec_ow = ((((uintptr_t)P.o | (uintptr_t)P.w) & 15u) == 0u) && ((P.op & 7) == 0) && ((P.wp & 7) == 0);

    /* Row identity (the row-identity checks): the form follows the shape, never the row count, so a token's output is the same
     * bits alone, in a decode or verify step, or in a prefill chunk. ERNIE's shape (D % 32 == 0, R % 32 == 0, R <= 64) takes the chunked
     * WMMA sum for every row count: up to 64 rows the small-row kernel, above it the LDS-tiled kernel (same chunks, same order; the bound
     * is speed only). Before, 1..16 rows took the lane-strided decode kernel, which sums in another order (R4D_HCA_WOA_DECODE_MAX=65535
     * pinned it for every count, putting prefill on it). Other shapes keep one form for every row count: the decode kernel where it
     * applies, else the general kernel. R4D_HCA_WOA_DECODE_MAX is no longer read. */
#if defined(__clang__) || defined(__HIP_DEVICE_COMPILE__)
    if (contig && ((P.D & 31) == 0) && ((P.R & 31) == 0) && P.R <= 64) {
        if (P.M <= 64) {
            const dim3 grid((unsigned)((P.K + 15) / 16), (unsigned)((P.M + 15) / 16), (unsigned)P.G);
            if (vec_ow) {
                R4D_KLAUNCH(R4D_KNAME(ernie_hca_wo_a_wmma_small_kernel<true>), grid, dim3(kWoASmallWaves * 32), (hipStream_t)stream, P);
            } else {
                R4D_KLAUNCH(R4D_KNAME(ernie_hca_wo_a_wmma_small_kernel<false>), grid, dim3(kWoASmallWaves * 32), (hipStream_t)stream, P);
            }
            return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
        }
        const dim3 grid((unsigned)((P.K + 127) / 128), (unsigned)((P.M + 63) / 64), (unsigned)P.G);
        if (vec_ow) {
            R4D_KLAUNCH(R4D_KNAME(ernie_hca_wo_a_wmma_tiled_kernel<true>), grid, dim3(256), (hipStream_t)stream, P);
        } else {
            R4D_KLAUNCH(R4D_KNAME(ernie_hca_wo_a_wmma_tiled_kernel<false>), grid, dim3(256), (hipStream_t)stream, P);
        }
        return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
    }
#endif
    if (contig && (P.hg * P.R <= 4096) && ((P.D & 7) == 0) && ((P.R & 7) == 0)) {
        const dim3 grid((unsigned)((P.K + 7) / 8), (unsigned)P.G, (unsigned)P.M);
        if (vec_ow) {
            R4D_KLAUNCH(R4D_KNAME(ernie_hca_wo_a_decode_kernel<true>), grid, dim3(256), (hipStream_t)stream, P);
        } else {
            R4D_KLAUNCH(R4D_KNAME(ernie_hca_wo_a_decode_kernel<false>), grid, dim3(256), (hipStream_t)stream, P);
        }
        return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
    }

    R4D_KLAUNCH(R4D_KNAME(ernie_hca_wo_a_kernel), dim3((unsigned)G, (unsigned)P.M), dim3(128), (hipStream_t)stream, P);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

} /* extern "C" */

/* The events the HCA kernels raised since the last call (their bits OR-ed, see the HCA log in r4d_ernie_verdict.h; 0x40000000 if some were not readable), 0 if none or if the validators are off
 * (R4D_ERNIE_VALIDATE=0); -1 if the copy failed. The log is never cleared, the host remembers what it took. */
extern "C" int r4d_ernie_hca_take_error(void) {
    return r4d_verdict::registry().take(r4d_verdict::kFamHca);
}
