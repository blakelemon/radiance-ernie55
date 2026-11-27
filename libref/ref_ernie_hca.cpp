/* ERNIE 5.5 HCA compressor operations (story 3-3b), single rank, host only.
 *
 * The 37 HCA layers pool the KV stream of every `ratio` tokens into one compressed row. The 3-3 report
 * (docs/ernie55/hca-compression.md) fixes the semantics; these two ops are its state write and its group
 * compression, split the way the cache-state API splits them: every token writes a fp32 state row, and the token
 * that completes a group reads the `coff * ratio` rows ending at its own position, pools them, normalises,
 * and stores one compressed row at a slot the caller names. The slot of a row is never derived here: the plugin
 * owns the state and compressed caches (docs/ernie55/hca-compressor-ops.md).
 *
 * Every launch validates every operand and every index before the first store, so a refusal never leaves a partial
 * result; compress computes all rows first and stores afterwards. Arithmetic is fp32 with the stage roundings of
 * the report: a fixed summation order (no reduction() clauses), exp, 1/sqrt, then one bf16 rounding of nope and rope
 * (full layout) or the UE8M0 block-scaled E4M3 form (fp8_ds_mla). */
#include "ref_common.h"
#include "ref_ernie_rope.h"
#include "ref_ops.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
using namespace ref;

namespace {
bool pos_strides(const RadTensor& t) {
    for (uint32_t i = 0; i < t.rank; ++i)
        if (t.stride[i] < 1 || t.shape[i] < 0) return false;
    return true;
}
double hca_f64(const RadArgs* a, const char* k, double dflt) {
    const RadParam* p = p_find(a, k);
    return p && p->kind == RAD_P_F64 ? p->dval : p ? (double)p_f32(a, k, (float)dflt) : dflt;
}
inline int64_t at_i(const RadTensor& t, int64_t i) { return ld_int(t.dtype, t.data, i * t.stride[0]); }
/* A paged pool [pages, 1, block, W] (what the engine hands an op for a one-head KV group) as the [pages * block, W] rows the explicit slots
 * address (slot = page * block + offset), when the page axis is contiguous (story 3-4b, additive). Rank 2 is untouched. */
bool pool_rows(RadTensor* t) {
    if (t->rank != 4) return true;
    if (t->shape[1] != 1 || t->shape[2] < 1 || t->stride[0] != t->shape[2] * t->stride[2]) return false;
    t->rank = 2;
    t->shape[0] *= t->shape[2];
    t->shape[1] = t->shape[3];
    t->stride[0] = t->stride[2];
    t->stride[1] = t->stride[3];
    return true;
}
/* Optional block_size (story 3-4b: the plugin names the page size of the KV group the pool belongs to): must be >= 1 and, for a rank-4 pool, its block. */
bool block_hint_ok(const RadArgs* a, const RadTensor& pool) {
    if (!p_find(a, "block_size")) return true;
    const long long b = p_int(a, "block_size", 0);
    return b >= 1 && (pool.rank != 4 || pool.shape[2] == b);
}
/* Whether two operands share a byte: the extents their (positive) strides reach, so a conservative test (story 3-3c). The state-write loop re-reads
 * slots, positions and values while it stores, and the compress store loop re-reads the out slots, so an operand that overlaps the output would make
 * a validated index change under the op; any such launch refuses before the first store. */
bool operands_overlap(const RadTensor& a, const RadTensor& b) {
    auto extent = [](const RadTensor& t) {
        int64_t last = 0;
        for (uint32_t i = 0; i < t.rank; ++i) last += (t.shape[i] - 1) * t.stride[i];
        const int64_t eb = (rad_dtype_bits(t.dtype) + 7) / 8;
        return std::pair<const char*, const char*>((const char*)t.data, (const char*)t.data + (last + 1) * eb);
    };
    const auto ea = extent(a), eb = extent(b);
    return ea.first < eb.second && eb.first < ea.second;
}
/* The two legal periods (k01): lanes per state row, rows pooled per group. */
bool legal_ratio(int64_t r) { return r == 4 || r == 128; }
}  // namespace

/* ------------------------------------------------------------------ ernie_hca_state_write */
/* state[slots[m]] = [kv[m] | score[m] + ape[positions[m] % ratio]] (f32, k03). Rows apply in order, a repeated slot
 * keeps the last; slot -1 skips the row (its position is not read). state is [S, 2 * coff * head_dim]. */
int ref_ernie_hca_state_write(const RadArgs* a, RadStream) {
    if (!have(a, 6)) return RAD_E_INVAL;
    RadTensor stv = a->t[5];
    const RadTensor &kv = a->t[0], &sc = a->t[1], &ape = a->t[2], &pos = a->t[3], &slot = a->t[4];
    const int64_t ratio = p_int(a, "ratio", 0), D = p_int(a, "head_dim", 0);
    if (!legal_ratio(ratio) || D < 1 || D > 4096 || !block_hint_ok(a, a->t[5])) return RAD_E_SHAPE;
    if (!pool_rows(&stv)) return RAD_E_SHAPE;
    const RadTensor& st = stv;
    const int64_t coff = ratio == 4 ? 2 : 1, W = coff * D;
    if (kv.rank != 2 || sc.rank != 2 || ape.rank != 2 || pos.rank != 1 || slot.rank != 1 || st.rank != 2 || kv.shape[0] < 1 ||
        kv.shape[1] != W || sc.shape[0] != kv.shape[0] || sc.shape[1] != W || ape.shape[0] != ratio || ape.shape[1] != W ||
        pos.shape[0] != kv.shape[0] || slot.shape[0] != kv.shape[0] || st.shape[0] < 1 || st.shape[1] != 2 * W ||
        !pos_strides(kv) || !pos_strides(sc) || !pos_strides(ape) || !pos_strides(pos) || !pos_strides(slot) || !pos_strides(st))
        return RAD_E_SHAPE;
    if (kv.dtype != RAD_F32 || sc.dtype != RAD_F32 || ape.dtype != RAD_F32 || st.dtype != RAD_F32 || pos.dtype != RAD_I32 ||
        slot.dtype != RAD_I32)
        return RAD_E_DTYPE;
    if (!kv.data || !sc.data || !ape.data || !pos.data || !slot.data || !st.data) return RAD_E_INVAL;
    for (const RadTensor* in : {&kv, &sc, &ape, &pos, &slot})
        if (operands_overlap(st, *in)) return RAD_E_INVAL;   /* story 3-3c: state may not share a byte with any input */
    const int64_t M = kv.shape[0], S = st.shape[0];
    for (int64_t m = 0; m < M; ++m) {
        const int64_t s = at_i(slot, m);
        if (s < -1 || s >= S) return RAD_E_SHAPE;
        if (s >= 0 && at_i(pos, m) < 0) return RAD_E_SHAPE;
    }
    for (int64_t m = 0; m < M; ++m) {   /* in order: a repeated slot keeps the last row */
        const int64_t s = at_i(slot, m);
        if (s < 0) continue;
        const int64_t ar = at_i(pos, m) % ratio;
        float* row = (float*)st.data + s * st.stride[0];
        for (int64_t c = 0; c < W; ++c) {
            const float k = ((const float*)kv.data)[m * kv.stride[0] + c * kv.stride[1]];
            const float x = ((const float*)sc.data)[m * sc.stride[0] + c * sc.stride[1]];
            const float b = ((const float*)ape.data)[ar * ape.stride[0] + c * ape.stride[1]];
            row[c * st.stride[1]] = k;
            row[(W + c) * st.stride[1]] = x + b;
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ ernie_hca_compress */
/* For each row m whose out_slots[m] >= 0 and (positions[m] + 1) % ratio == 0 (k04): pool the window of coff * ratio
 * state rows ending at the position, listed oldest first in gather_slots[m] (-1 = masked: score -inf, kv 0, k06).
 * Channel c of window row w reads state channel c (previous group, lane 0) or D + c (current group, lane 1, only at
 * ratio 4: rows w >= ratio) of kv and of score (k07); the softmax is per channel over the unmasked rows. The pooled
 * row is RMS-normalised in f32 with the bf16 weight, the trailing rope_dim channels are rotated GPT-J interleaved at
 * the group START (position / ratio) * ratio from the un-rounded row with theta (plain RoPE; optional yarn_* parameters select the
 * YaRN table of the HCA layers, story 3-4a, k22) and rounded once to bf16. layout bf16_full (default, k20) stores nope and rope as bf16 [R, D];
 * fp8_ds_mla stores a u8 row [nope E4M3 codes | rope bf16 little endian | one UE8M0 byte per quant_block nope
 * channels (exponent + 127, scale = max(absmax / 448, 1e-4), k09) | one zero pad byte]. */
int ref_ernie_hca_compress(const RadArgs* a, RadStream) {
    if (!have(a, 6)) return RAD_E_INVAL;
    RadTensor stv = a->t[0], outv = a->t[5];
    const RadTensor &gs = a->t[1], &pos = a->t[2], &os = a->t[3], &nw = a->t[4];
    /* `out_layout` (kv-cmpblock, additive, optional; named so it sorts after head_dim: the sealed 3-3b test pins the first three sorted parameter names): "token"
     * (default, the 3-4b form) stores the row at out_slots[m] of a per-token pool; "block" views the pool [pages, 1, B, e] as [pages, B * e] rows, B = ratio, so
     * the group holds ONE row per block and a completing token's slot (page * B + B - 1) names the row page * (B * e). Only the address moves; the arithmetic
     * is the same. */
    const char* cmpl = p_str(a, "out_layout", "token");
    const bool blk_rows = !std::strcmp(cmpl, "block");
    if (!blk_rows && std::strcmp(cmpl, "token")) return RAD_E_SHAPE;
    if (!pool_rows(&stv)) return RAD_E_SHAPE;   /* story 3-4b: engine pools [pages, 1, block, W]; the schema is unchanged (sealed 3-3b) */
    int64_t slot_div = 1;   /* out slots per out row: B in the block layout */
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
    const int64_t ratio = p_int(a, "ratio", 0), D = p_int(a, "head_dim", 0), R = p_int(a, "rope_dim", 0), qb = p_int(a, "quant_block", 64);
    if (blk_rows && slot_div != ratio) return RAD_E_SHAPE;   /* one row per ratio tokens */
    const double eps = hca_f64(a, "eps", 1e-6), theta = hca_f64(a, "theta", 0.0);
    const char* layout = p_str(a, "layout", "bf16_full");
    const bool packed = !std::strcmp(layout, "fp8_ds_mla");
    if (!packed && std::strcmp(layout, "bf16_full")) return RAD_E_SHAPE;
    if (!legal_ratio(ratio) || D < 2 || D > 4096 || R < 2 || (R & 1) || R > D || !(eps >= 0.0) || !std::isfinite(eps) ||
        !std::isfinite(theta) || theta <= 0.0)
        return RAD_E_SHAPE;
    const int64_t nope = D - R, coff = ratio == 4 ? 2 : 1, W = coff * D, Wn = coff * ratio;
    if (packed && (qb < 1 || nope < 1 || nope % qb)) return RAD_E_SHAPE;
    const int64_t nb = packed ? nope / qb : 0, rb = packed ? nope + 2 * R + nb + 1 : D;
    if (st.rank != 2 || gs.rank != 2 || pos.rank != 1 || os.rank != 1 || nw.rank != 1 || out.rank != 2 || gs.shape[0] < 1 ||
        st.shape[0] < 1 || st.shape[1] != 2 * W || gs.shape[1] != Wn || pos.shape[0] != gs.shape[0] || os.shape[0] != gs.shape[0] ||
        nw.shape[0] != D || out.shape[0] < 1 || out.shape[1] != rb || !pos_strides(st) || !pos_strides(gs) || !pos_strides(pos) ||
        !pos_strides(os) || !pos_strides(nw) || !pos_strides(out))
        return RAD_E_SHAPE;
    if (st.dtype != RAD_F32 || gs.dtype != RAD_I32 || pos.dtype != RAD_I32 || os.dtype != RAD_I32 || nw.dtype != RAD_BF16 ||
        out.dtype != (packed ? RAD_U8 : RAD_BF16))
        return RAD_E_DTYPE;
    if (!st.data || !gs.data || !pos.data || !os.data || !nw.data || !out.data) return RAD_E_INVAL;
    for (const RadTensor* in : {&st, &gs, &pos, &os, &nw})
        if (operands_overlap(out, *in)) return RAD_E_INVAL;   /* story 3-3c: out may not share a byte with any input (out slots are re-read at store time) */
    const int64_t M = gs.shape[0], S = st.shape[0], Rr = out.shape[0] * slot_div;
    /* validation: a row with out slot -1 is ignored altogether; every other row is checked even when it does not complete */
    std::vector<int64_t> live;
    for (int64_t m = 0; m < M; ++m) {
        const int64_t o = at_i(os, m);
        if (o < -1 || o >= Rr) return RAD_E_SHAPE;
        if (o < 0) continue;
        const int64_t p = at_i(pos, m);
        if (p < 0) return RAD_E_SHAPE;
        int64_t unmasked = 0;
        for (int64_t w = 0; w < Wn; ++w) {
            const int64_t s = ld_int(gs.dtype, gs.data, m * gs.stride[0] + w * gs.stride[1]);
            if (s < -1 || s >= S) return RAD_E_SHAPE;
            unmasked += s >= 0;
        }
        if ((p + 1) % ratio) continue;
        if (!unmasked) return RAD_E_SHAPE;
        if (blk_rows && o % slot_div != slot_div - 1) return RAD_E_SHAPE;   /* a block's row is written by the token that completes it, its last */
        live.push_back(m);
    }
    const int64_t L = (int64_t)live.size();
    std::vector<float> nrm((size_t)D);
    for (int64_t c = 0; c < D; ++c) nrm[(size_t)c] = ldt(&nw, c * nw.stride[0]);
    ErnieYarn yarn;   /* story 3-4a, additive: the YaRN table of the HCA layers (k22); absent = plain RoPE as in 3-3b */
    if (!ernie_yarn_args(a, &yarn)) return RAD_E_SHAPE;
    const std::vector<double> inv = ernie_rope_inv(theta, R, yarn);
    const int64_t rbytes = packed ? rb : 2 * D;   /* bytes of one staged row */
    std::vector<uint8_t> rows((size_t)(L * rbytes), 0);
    std::vector<int> bad((size_t)L, 0);
#pragma omp parallel for schedule(static)
    for (int64_t j = 0; j < L; ++j) {
        const int64_t m = live[(size_t)j], p = at_i(pos, m);
        std::vector<float> pooled((size_t)D), y((size_t)D), e((size_t)Wn);
        for (int64_t c = 0; c < D; ++c) {
            float mx = -INFINITY;
            for (int64_t w = 0; w < Wn; ++w) {
                const int64_t s = ld_int(gs.dtype, gs.data, m * gs.stride[0] + w * gs.stride[1]);
                if (s < 0) continue;
                const int64_t off = (coff == 2 && w >= ratio) ? D : 0;
                const float sc = ((const float*)st.data)[s * st.stride[0] + (W + off + c) * st.stride[1]];
                if (sc > mx) mx = sc;
            }
            float num = 0.0f, den = 0.0f;
            for (int64_t w = 0; w < Wn; ++w) {
                const int64_t s = ld_int(gs.dtype, gs.data, m * gs.stride[0] + w * gs.stride[1]);
                if (s < 0) continue;
                const int64_t off = (coff == 2 && w >= ratio) ? D : 0;
                const float sc = ((const float*)st.data)[s * st.stride[0] + (W + off + c) * st.stride[1]];
                const float k = ((const float*)st.data)[s * st.stride[0] + (off + c) * st.stride[1]];
                const float ex = std::exp(sc - mx);
                num += ex * k;
                den += ex;
            }
            pooled[(size_t)c] = num / den;
        }
        float ss = 0.0f;
        for (int64_t c = 0; c < D; ++c) ss += pooled[(size_t)c] * pooled[(size_t)c];
        const float mean = ss / (float)D, rinv = 1.0f / std::sqrt(mean + (float)eps);
        for (int64_t c = 0; c < D; ++c) y[(size_t)c] = (pooled[(size_t)c] * rinv) * nrm[(size_t)c];
        const int64_t start = (p / ratio) * ratio;
        std::vector<float> rope((size_t)R);
        for (int64_t i = 0; i < R / 2; ++i) {
            const double ang = (double)start * inv[(size_t)i];
            const float cs = (float)std::cos(ang), sn = (float)std::sin(ang);
            const float ev = y[(size_t)(nope + 2 * i)], od = y[(size_t)(nope + 2 * i + 1)];
            rope[(size_t)(2 * i)] = ev * cs - od * sn;
            rope[(size_t)(2 * i + 1)] = ev * sn + od * cs;
        }
        uint8_t* row = rows.data() + (size_t)(j * rbytes);
        /* Finiteness is judged on what is stored: the bf16-ROUNDED nope and rope values (story 3-3c; a finite f32 near the top of the range rounds
         * to bf16 infinity, which both layouts would otherwise store or feed to the block scale). Rounding never turns a finite value into NaN. */
        std::vector<float> yb((size_t)nope), rb16((size_t)R);
        bool finite = true;
        for (int64_t c = 0; c < nope; ++c) { yb[(size_t)c] = bf16_to_f32(f32_to_bf16(y[(size_t)c])); finite &= std::isfinite(yb[(size_t)c]); }
        for (int64_t c = 0; c < R; ++c) { rb16[(size_t)c] = bf16_to_f32(f32_to_bf16(rope[(size_t)c])); finite &= std::isfinite(rb16[(size_t)c]); }
        if (!finite) { bad[(size_t)j] = 1; continue; }
        if (!packed) {
            uint16_t* h = (uint16_t*)row;
            for (int64_t c = 0; c < nope; ++c) h[c] = f32_to_bf16(y[(size_t)c]);
            for (int64_t c = 0; c < R; ++c) h[nope + c] = f32_to_bf16(rope[(size_t)c]);
        } else {
            std::vector<float> blk((size_t)qb);
            for (int64_t b = 0; b < nb; ++b) {
                float amax = 0.0f;
                for (int64_t i = 0; i < qb; ++i) {
                    blk[(size_t)i] = bf16_to_f32(f32_to_bf16(y[(size_t)(b * qb + i)]));
                    amax = std::fmax(amax, std::fabs(blk[(size_t)i]));
                }
                const float scale = std::fmax(amax / 448.0f, 1e-4f);
                int ex;
                const float mant = std::frexp(scale, &ex);
                const int e2 = mant == 0.5f ? ex - 1 : ex;   /* ceil(log2(scale)) */
                row[nope + 2 * R + b] = (uint8_t)(e2 + 127);
                const float is = std::ldexp(1.0f, -e2);
                for (int64_t i = 0; i < qb; ++i) {
                    float v = blk[(size_t)i] * is;
                    if (v > 448.0f) v = 448.0f;
                    if (v < -448.0f) v = -448.0f;
                    row[b * qb + i] = f32_to_fp8e4m3(v);
                }
            }
            for (int64_t c = 0; c < R; ++c) {
                const uint16_t h = f32_to_bf16(rope[(size_t)c]);
                row[nope + 2 * c] = (uint8_t)(h & 255);
                row[nope + 2 * c + 1] = (uint8_t)(h >> 8);
            }
        }
    }
    for (int64_t j = 0; j < L; ++j)
        if (bad[(size_t)j]) return RAD_E_SHAPE;   /* a non-finite row refuses before any store */
    for (int64_t j = 0; j < L; ++j) {            /* in order: a repeated slot keeps the last row */
        const int64_t o = at_i(os, live[(size_t)j]) / slot_div;
        const uint8_t* row = rows.data() + (size_t)(j * rbytes);
        if (!packed) {
            const uint16_t* h = (const uint16_t*)row;
            for (int64_t c = 0; c < D; ++c) ((uint16_t*)out.data)[o * out.stride[0] + c * out.stride[1]] = h[c];
        } else {
            for (int64_t c = 0; c < rb; ++c) ((uint8_t*)out.data)[o * out.stride[0] + c * out.stride[1]] = row[c];
        }
    }
    return RAD_OK;
}

/* ================================================================== story 3-4a: the HCA attention ops
 * The attention branch of an HCA layer (docs/ernie55/hca-attention-ops.md, facts h04 h08 h09 h10 h11 h16 h17 of the 3-7 audit):
 *   ernie_hca_q_prep     weightless per-head RMS over the head width, then GPT-J rope on the trailing slice (one bf16 rounding)
 *   ernie_hca_union_attn one softmax over [complete compressed rows | local window] and the sink, keys = values
 *   ernie_hca_wo_a       inverse rope at the query position, then the grouped wo_a
 * Every launch validates every operand and every index it will read before the first store. */

/* ------------------------------------------------------------------ ernie_hca_q_prep */
/* x [M, heads * head_dim] bf16, in place. Per head: y = x * rsqrt(mean(x^2) + eps) in f32 (no weight, h10), then the trailing `rope_dim` values
 * are rotated GPT-J interleaved at the row's absolute position with the f32 cos/sin table (YaRN if the yarn_* parameters are given) from the f32 y,
 * and everything is rounded once to bf16 (the fused q kernel of the pinned model rotates before it stores). */
int ref_ernie_hca_q_prep(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &pos = a->t[1];
    const int64_t H = p_int(a, "heads", 0), D = p_int(a, "head_dim", 0), R = p_int(a, "rope_dim", 0);
    const double eps = hca_f64(a, "eps", -1.0), theta = hca_f64(a, "theta", 0.0);
    ErnieYarn yarn;
    if (!ernie_yarn_args(a, &yarn)) return RAD_E_SHAPE;
    if (H < 1 || D < 2 || D > 4096 || R < 2 || (R & 1) || R > D || !(eps >= 0.0) || !std::isfinite(eps) || !std::isfinite(theta) ||
        theta <= 0.0 || x.rank != 2 || pos.rank != 1 || x.shape[0] < 1 || x.shape[1] != H * D || pos.shape[0] != x.shape[0] ||
        !pos_strides(x) || !pos_strides(pos) || x.stride[0] < x.shape[1] * x.stride[1])
        return RAD_E_SHAPE;
    if (x.dtype != RAD_BF16 || pos.dtype != RAD_I32) return RAD_E_DTYPE;
    if (!x.data || !pos.data) return RAD_E_INVAL;
    const int64_t M = x.shape[0];
    for (int64_t m = 0; m < M; ++m)
        if (at_i(pos, m) < 0) return RAD_E_SHAPE;
    const std::vector<double> inv = ernie_rope_inv(theta, R, yarn);
#pragma omp parallel for schedule(static)
    for (int64_t m = 0; m < M; ++m) {
        const double p = (double)at_i(pos, m);
        std::vector<float> cs((size_t)(R / 2)), sn((size_t)(R / 2)), y((size_t)D);
        for (int64_t i = 0; i < R / 2; ++i) {
            const double ang = p * inv[(size_t)i];
            cs[(size_t)i] = (float)std::cos(ang);
            sn[(size_t)i] = (float)std::sin(ang);
        }
        for (int64_t h = 0; h < H; ++h) {
            float ss = 0.0f;
            for (int64_t c = 0; c < D; ++c) {
                const float v = ldt(&x, m * x.stride[0] + (h * D + c) * x.stride[1]);
                y[(size_t)c] = v;
                ss += v * v;
            }
            const float rinv = 1.0f / std::sqrt(ss / (float)D + (float)eps);
            for (int64_t c = 0; c < D; ++c) y[(size_t)c] *= rinv;
            for (int64_t c = 0; c < D - R; ++c) stt(&x, m * x.stride[0] + (h * D + c) * x.stride[1], y[(size_t)c]);
            for (int64_t i = 0; i < R / 2; ++i) {
                const float ev = y[(size_t)(D - R + 2 * i)], od = y[(size_t)(D - R + 2 * i + 1)];
                stt(&x, m * x.stride[0] + (h * D + D - R + 2 * i) * x.stride[1], ev * cs[(size_t)i] - od * sn[(size_t)i]);
                stt(&x, m * x.stride[0] + (h * D + D - R + 2 * i + 1) * x.stride[1], ev * sn[(size_t)i] + od * cs[(size_t)i]);
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ ernie_hca_union_attn */
/* Row r (position p, request q = req[r]) attends, in ONE softmax with the per-head fp32 sink (joins the max and the denominator, no value):
 *   - the compressed rows j = 0 .. (p + 1) / ratio - 1 (complete groups only, h03), row j at cmp_kv slot
 *     cmp_table[q][j / cmp_block] * cmp_block + j % cmp_block;
 *   - the window tokens max(0, p - window + 1) .. p, token t at win_kv slot win_table[q][t / win_block] * win_block + t % win_block.
 * In the origin variant, window slot addressing uses t - win_origin[q]. The origin must cover the oldest causal window token.
 * The window and the newest complete group OVERLAP and are NOT de-duplicated: the pinned source concatenates them (h04, h08), so a token
 * inside the newest group is attended raw and again through its group row. A table entry of -1 is an uninitialised page and its keys are
 * skipped; an entry >= pages, below -1, or a block index past the table width refuses. Key = value = the cache row (h09: K = V, the rotated
 * rope channels are part of the value): score = scale * (q . row), out = sum w row / (sink term + sum w). f32 throughout, one bf16 store. */
static int hca_union_attn(const RadArgs* a, bool with_origin) {
    if (!have(a, with_origin ? 10 : 9)) return RAD_E_INVAL;
    const RadTensor &q = a->t[0], &wk = a->t[1], &ck = a->t[2], &pos = a->t[3], &req = a->t[4], &wt = a->t[5], &ct = a->t[6];
    const RadTensor& window_origin = a->t[7];
    const RadTensor &sink = a->t[with_origin ? 8 : 7], &out = a->t[with_origin ? 9 : 8];
    const int64_t H = p_int(a, "heads", 0), D = p_int(a, "head_dim", 0), ratio = p_int(a, "ratio", 0), win = p_int(a, "window", 128);
    const double scale = hca_f64(a, "scale", 0.0);
    /* `cmp_layout` (story 3-4b, additive): "dense" (default, the 3-4a form) keeps compressed row j at logical id j of cmp_table; "token" keeps it at
     * the logical id of the token that completed its group, (j + 1) * ratio - 1, so a per-token KV group holds it at the slot of that token. */
    const char* cl = p_str(a, "cmp_layout", "dense");
    const bool tok_rows = !std::strcmp(cl, "token");
    /* "block" (kv-cmpblock, additive): the compressed group holds ONE row per block of `ratio` tokens. cmp_kv is [pages, 1, ratio, D / ratio], a page is one row (D contiguous
     * elements), row j is the page cmp_table[req][j]. The rows are the token layout's, only their addresses differ. */
    const bool blk_rows = !std::strcmp(cl, "block");
    if (!tok_rows && !blk_rows && std::strcmp(cl, "dense")) return RAD_E_SHAPE;
    /* optional `block_size` (kv-cmpblock): the compressed group's block size, which the builder reads off the first op after the group that carries one; when given it
     * must be the pool's page rows */
    const bool bs_bad = p_find(a, "block_size") && (a->n_t < 3 || a->t[2].rank != 4 || p_int(a, "block_size", 0) != a->t[2].shape[2]);
    if (H < 1 || D < 1 || D > 4096 || !legal_ratio(ratio) || win < 1 || win > INT32_MAX / 2 || !std::isfinite(scale) || scale <= 0.0 ||
        q.rank != 2 || wk.rank != 4 || ck.rank != 4 || pos.rank != 1 || req.rank != 1 || wt.rank != 2 || ct.rank != 2 || sink.rank != 1 ||
        out.rank != 2 || q.shape[0] < 1 || q.shape[1] != H * D || wk.shape[0] < 1 || wk.shape[1] != 1 || wk.shape[2] < 1 || wk.shape[3] != D ||
        ck.shape[0] < 1 || ck.shape[1] != 1 || ck.shape[2] < 1 ||
        (blk_rows ? (ck.shape[2] != ratio || ck.shape[2] * ck.shape[3] != D || ck.stride[2] != ck.shape[3] * ck.stride[3]) : ck.shape[3] != D) || bs_bad || pos.shape[0] != q.shape[0] || req.shape[0] != q.shape[0] ||
        wt.shape[0] < 1 || wt.shape[1] < 1 || ct.shape[0] != wt.shape[0] || ct.shape[1] < 1 || sink.shape[0] != H || out.shape[0] != q.shape[0] ||
        out.shape[1] != H * D || !pos_strides(q) || !pos_strides(wk) || !pos_strides(ck) || !pos_strides(pos) || !pos_strides(req) ||
        !pos_strides(wt) || !pos_strides(ct) || !pos_strides(sink) || !pos_strides(out) || out.stride[0] < out.shape[1] * out.stride[1] ||
        (with_origin && (window_origin.rank != 1 || window_origin.shape[0] != wt.shape[0] || !pos_strides(window_origin))))
        return RAD_E_SHAPE;
    if (q.dtype != RAD_BF16 || wk.dtype != RAD_BF16 || ck.dtype != RAD_BF16 || out.dtype != RAD_BF16 || pos.dtype != RAD_I32 ||
        req.dtype != RAD_I32 || wt.dtype != RAD_I32 || ct.dtype != RAD_I32 || sink.dtype != RAD_F32 ||
        (with_origin && window_origin.dtype != RAD_I32))
        return RAD_E_DTYPE;
    if (!q.data || !wk.data || !ck.data || !pos.data || !req.data || !wt.data || !ct.data || !sink.data || !out.data ||
        (with_origin && !window_origin.data))
        return RAD_E_INVAL;
    const int64_t M = q.shape[0], S = wt.shape[0], Bw = wt.shape[1], Bc = ct.shape[1], bw = wk.shape[2], bc = ck.shape[2], Pw = wk.shape[0],
                  Pc = ck.shape[0];
    for (int64_t h = 0; h < H; ++h) {
        const float s = ldt(&sink, h * sink.stride[0]);
        if (std::isnan(s) || s == INFINITY) return RAD_E_SHAPE;
    }
    /* a page for logical row `i` of request `r` in a table: -1 = uninitialised (skip), -2 = refuse */
    auto page_of = [&](const RadTensor& tab, int64_t r, int64_t block_idx, int64_t width, int64_t pages) -> int64_t {
        if (block_idx >= width) return -2;
        const int64_t pg = ld_int(tab.dtype, tab.data, r * tab.stride[0] + block_idx * tab.stride[1]);
        if (pg < -1 || pg >= pages) return -2;
        return pg;
    };
    for (int64_t m = 0; m < M; ++m) {
        const int64_t p = at_i(pos, m), r = at_i(req, m);
        if (p < 0 || r < 0 || r >= S) return RAD_E_SHAPE;
        const int64_t first = with_origin ? at_i(window_origin, r) : 0;
        const int64_t start = std::max<int64_t>(0, p - win + 1);
        if (first < 0 || first % bw || first > start) return RAD_E_SHAPE;
        const int64_t n = (p + 1) / ratio;
        for (int64_t j = 0; j < n; ++j)
            if (page_of(ct, r, blk_rows ? j : (tok_rows ? (j + 1) * ratio - 1 : j) / bc, Bc, Pc) == -2) return RAD_E_SHAPE;   /* block: row j is table entry j */
        for (int64_t t = start; t <= p; ++t)
            if (page_of(wt, r, (t - first) / bw, Bw, Pw) == -2) return RAD_E_SHAPE;
    }
#pragma omp parallel for schedule(static)
    for (int64_t job = 0; job < M * H; ++job) {
        const int64_t m = job / H, h = job % H, p = at_i(pos, m), r = at_i(req, m), n = (p + 1) / ratio;
        const int64_t first = with_origin ? at_i(window_origin, r) : 0;
        std::vector<const RadTensor*> pool;
        std::vector<int64_t> base;
        std::vector<float> sc;
        auto push_key = [&](const RadTensor& kv, int64_t page, int64_t off) {
            const int64_t b = page * kv.stride[0] + off * kv.stride[2];
            float v = 0.0f;
            for (int64_t c = 0; c < D; ++c) v += ldt(&q, m * q.stride[0] + (h * D + c) * q.stride[1]) * ldt(&kv, b + c * kv.stride[3]);
            sc.push_back(v * (float)scale);
            base.push_back(b);
            pool.push_back(&kv);
        };
        for (int64_t j = 0; j < n; ++j) {
            const int64_t id = tok_rows ? (j + 1) * ratio - 1 : j, pg = page_of(ct, r, blk_rows ? j : id / bc, Bc, Pc);
            if (pg >= 0) push_key(ck, pg, blk_rows ? 0 : id % bc);
        }
        for (int64_t t = std::max<int64_t>(0, p - win + 1) - first; t <= p - first; ++t) {
            const int64_t pg = page_of(wt, r, t / bw, Bw, Pw);
            if (pg >= 0) push_key(wk, pg, t % bw);
        }
        const float sk = ldt(&sink, h * sink.stride[0]);
        float mx = sk;
        for (float s : sc) mx = std::max(mx, s);
        float den = 0.0f;
        if (sk != -INFINITY) den += ref_exp(sk - mx);
        std::vector<float> wgt(sc.size());
        for (size_t j = 0; j < sc.size(); ++j) {
            wgt[j] = ref_exp(sc[j] - mx);
            den += wgt[j];
        }
        for (int64_t c = 0; c < D; ++c) {
            float num = 0.0f;
            for (size_t j = 0; j < sc.size(); ++j) num += wgt[j] * ldt(pool[j], base[j] + c * pool[j]->stride[3]);
            stt(&out, m * out.stride[0] + (h * D + c) * out.stride[1], den > 0.0f ? num / den : 0.0f);
        }
    }
    return RAD_OK;
}

int ref_ernie_hca_union_attn(const RadArgs* a, RadStream) {
    return hca_union_attn(a, false);
}

int ref_ernie_hca_union_attn_origin(const RadArgs* a, RadStream) {
    return hca_union_attn(a, true);
}

/* ------------------------------------------------------------------ ernie_hca_wo_a */
/* o [M, heads * head_dim] bf16 (the attention output after VHA) -> out [M, groups * rank]. Step 1 (h10): the trailing rope_dim values of every head
 * are rotated by the INVERSE of the layer's rotary (sin negated, GPT-J interleaved, f32 table, YaRN if the yarn_* parameters are given) at the
 * query row's absolute position and rounded to bf16 (the pinned op writes the model dtype). Step 2 (h16): the grouped wo_a: group g takes heads
 * g * (heads / groups) .. (g + 1) * (heads / groups) - 1 (a contiguous slice of `per = heads / groups * head_dim` inputs) and weight rows
 * [g * rank, (g + 1) * rank) of wo_a [groups * rank, per]: out[m, g * rank + j] = sum_c rot[m, g * per + c] * wo_a[g * rank + j, c], f32 accumulation,
 * one bf16 store. */
int ref_ernie_hca_wo_a(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor &o = a->t[0], &pos = a->t[1], &w = a->t[2], &out = a->t[3];
    const int64_t H = p_int(a, "heads", 0), D = p_int(a, "head_dim", 0), R = p_int(a, "rope_dim", 0), G = p_int(a, "groups", 0),
                  K = p_int(a, "rank", 0);
    const double theta = hca_f64(a, "theta", 0.0);
    ErnieYarn yarn;
    if (!ernie_yarn_args(a, &yarn)) return RAD_E_SHAPE;
    if (H < 1 || D < 2 || D > 4096 || R < 2 || (R & 1) || R > D || G < 1 || H % G || K < 1 || !std::isfinite(theta) || theta <= 0.0)
        return RAD_E_SHAPE;
    const int64_t per = (H / G) * D;
    if (o.rank != 2 || pos.rank != 1 || w.rank != 2 || out.rank != 2 || o.shape[0] < 1 || o.shape[1] != H * D || pos.shape[0] != o.shape[0] ||
        w.shape[0] != G * K || w.shape[1] != per || out.shape[0] != o.shape[0] || out.shape[1] != G * K || !pos_strides(o) || !pos_strides(pos) ||
        !pos_strides(w) || !pos_strides(out) || o.stride[0] < o.shape[1] * o.stride[1] || out.stride[0] < out.shape[1] * out.stride[1])
        return RAD_E_SHAPE;
    if (o.dtype != RAD_BF16 || w.dtype != RAD_BF16 || out.dtype != RAD_BF16 || pos.dtype != RAD_I32) return RAD_E_DTYPE;
    if (!o.data || !pos.data || !w.data || !out.data) return RAD_E_INVAL;
    const int64_t M = o.shape[0];
    for (int64_t m = 0; m < M; ++m)
        if (at_i(pos, m) < 0) return RAD_E_SHAPE;
    const std::vector<double> inv = ernie_rope_inv(theta, R, yarn);
    const int64_t hg = H / G;
#pragma omp parallel for schedule(static)
    for (int64_t job = 0; job < M * G; ++job) {
        const int64_t m = job / G, g = job % G;
        const double p = (double)at_i(pos, m);
        std::vector<float> rot((size_t)per);
        for (int64_t hh = 0; hh < hg; ++hh) {
            const int64_t head = g * hg + hh;
            for (int64_t c = 0; c < D - R; ++c) rot[(size_t)(hh * D + c)] = ldt(&o, m * o.stride[0] + (head * D + c) * o.stride[1]);
            for (int64_t i = 0; i < R / 2; ++i) {
                const double ang = p * inv[(size_t)i];
                const float cs = (float)std::cos(ang), sn = -(float)std::sin(ang);
                const float ev = ldt(&o, m * o.stride[0] + (head * D + D - R + 2 * i) * o.stride[1]),
                            od = ldt(&o, m * o.stride[0] + (head * D + D - R + 2 * i + 1) * o.stride[1]);
                rot[(size_t)(hh * D + D - R + 2 * i)] = round_dt(RAD_BF16, ev * cs - od * sn);
                rot[(size_t)(hh * D + D - R + 2 * i + 1)] = round_dt(RAD_BF16, ev * sn + od * cs);
            }
        }
        for (int64_t j = 0; j < K; ++j) {
            float sum = 0.0f;
            for (int64_t c = 0; c < per; ++c) sum += rot[(size_t)c] * ldt(&w, (g * K + j) * w.stride[0] + c * w.stride[1]);
            stt(&out, m * out.stride[0] + (g * K + j) * out.stride[1], sum);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ ernie_hca_gather_slots */
/* The state slots a group completion reads (story 3-4b). Row m (position p, request req[m]) whose position completes a group, (p + 1) % ratio == 0,
 * lists the slots of the coff * ratio state rows ending at p, oldest first: logical id q = p - coff * ratio + 1 + w sits at
 * block_table[req][q / block_size] * block_size + q % block_size, and -1 where q < 0 (k06). Every other row lists -1. The state group is a per-token
 * KV group, so a state row is addressed by its position like any KV row. A table entry of -1 (unallocated page) for a needed row, an entry that
 * would overflow int32 slots, a block index past the table, a negative position or an out-of-range request refuses before any store: a state row
 * that does not exist must not be pooled as if it were a masked one. In the origin variant, the block lookup uses q - state_origin[req]. */
static int hca_gather_slots(const RadArgs* a, bool with_origin) {
    if (!have(a, with_origin ? 5 : 4)) return RAD_E_INVAL;
    const RadTensor &pos = a->t[0], &req = a->t[1], &bt = a->t[2], &origin = a->t[3], &out = a->t[with_origin ? 4 : 3];
    const int64_t ratio = p_int(a, "ratio", 0), bs = p_int(a, "block_size", 0);
    if (!legal_ratio(ratio) || bs < 1) return RAD_E_SHAPE;
    const int64_t Wn = (ratio == 4 ? 2 : 1) * ratio;
    if (pos.rank != 1 || req.rank != 1 || bt.rank != 2 || out.rank != 2 || pos.shape[0] < 1 || req.shape[0] != pos.shape[0] || bt.shape[0] < 1 ||
        bt.shape[1] < 1 || out.shape[0] != pos.shape[0] || out.shape[1] != Wn || !pos_strides(pos) || !pos_strides(req) || !pos_strides(bt) ||
        !pos_strides(out) || (with_origin && (origin.rank != 1 || origin.shape[0] != bt.shape[0] || !pos_strides(origin))))
        return RAD_E_SHAPE;
    if (pos.dtype != RAD_I32 || req.dtype != RAD_I32 || bt.dtype != RAD_I32 || out.dtype != RAD_I32 ||
        (with_origin && origin.dtype != RAD_I32))
        return RAD_E_DTYPE;
    if (!pos.data || !req.data || !bt.data || !out.data || (with_origin && !origin.data)) return RAD_E_INVAL;
    const int64_t M = pos.shape[0], S = bt.shape[0], B = bt.shape[1];
    std::vector<int32_t> slots((size_t)(M * Wn), -1);
    for (int64_t m = 0; m < M; ++m) {
        const int64_t p = at_i(pos, m), r = at_i(req, m);
        if (p < 0 || r < 0 || r >= S) return RAD_E_SHAPE;
        const int64_t first = with_origin ? at_i(origin, r) : 0;
        if (first < 0 || first % bs || first > p) return RAD_E_SHAPE;
        if ((p + 1) % ratio) continue;
        for (int64_t w = 0; w < Wn; ++w) {
            const int64_t absolute = p - Wn + 1 + w;
            if (absolute < 0) continue;
            const int64_t q = absolute - first;
            if (q < 0 || q / bs >= B) return RAD_E_SHAPE;
            const int64_t pg = ld_int(bt.dtype, bt.data, r * bt.stride[0] + (q / bs) * bt.stride[1]);
            if (pg < 0 || pg > (INT32_MAX - q % bs) / bs) return RAD_E_SHAPE;   /* story 3-4c: the slot pg * bs + q % bs must fit int32; the old bound refused valid last pages */
            slots[(size_t)(m * Wn + w)] = (int32_t)(pg * bs + q % bs);
        }
    }
    for (int64_t m = 0; m < M; ++m)
        for (int64_t w = 0; w < Wn; ++w) ((int32_t*)out.data)[m * out.stride[0] + w * out.stride[1]] = slots[(size_t)(m * Wn + w)];
    return RAD_OK;
}

int ref_ernie_hca_gather_slots(const RadArgs* a, RadStream) {
    return hca_gather_slots(a, false);
}

int ref_ernie_hca_gather_slots_origin(const RadArgs* a, RadStream) {
    return hca_gather_slots(a, true);
}

/* ------------------------------------------------------------------ ernie_mhc_stream_mean */
/* Mean-pool the `mult` mHC streams of x [M, mult * H] into one hidden stream out [M, H]: for each row m and channel c,
 * sum = sum_{s=0..mult-1} x[m, s * H + c] accumulated in f32, out[m, c] = bf16(sum * (1 / mult)). `out` may be a column
 * slice of a wider buffer (out.stride[0] >= H * out.stride[1]); every operand and index is validated before the first store. */
int ref_ernie_mhc_stream_mean(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &out = a->t[1];
    const int64_t H = p_int(a, "hidden", 0), mult = p_int(a, "mult", 0);
    if (H < 1 || mult < 1 || mult > 16 || x.rank != 2 || out.rank != 2 || x.shape[0] < 1 || out.shape[0] != x.shape[0] ||
        x.shape[1] != mult * H || out.shape[1] != H || !pos_strides(x) || !pos_strides(out) ||
        out.stride[0] < out.shape[1] * out.stride[1])
        return RAD_E_SHAPE;
    if (x.dtype != RAD_BF16 || out.dtype != RAD_BF16) return RAD_E_DTYPE;
    if (!x.data || !out.data) return RAD_E_INVAL;
    const int64_t M = x.shape[0];
    const float inv = 1.0f / (float)mult;
#pragma omp parallel for schedule(static)
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t c = 0; c < H; ++c) {
            float sum = 0.0f;
            for (int64_t s = 0; s < mult; ++s) sum += ldt(&x, m * x.stride[0] + (s * H + c) * x.stride[1]);
            stt(&out, m * out.stride[0] + c * out.stride[1], sum * inv);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ ernie_hca_dspark_attn */
/* Non-causal anchor-window sliding-window MQA K=V attention with a per-head fp32 sink for ERNIE 5.5 DSpark draft layers
 * (no compressed cache). For sequence s with qlen = cu[s+1] - cu[s] > 0, total_kv = seqused[s], anchor = total_kv - qlen,
 * lo = max(0, anchor - window + 1): every query row m in [cu[s], cu[s+1]) and head h attends in ONE f32 softmax over all
 * keys t in [lo, total_kv) whose page block_table[s, t / block_size] >= 0 (a -1 page is skipped), PLUS the per-head sink
 * sk = sink[h] (joins the max and denominator via ref_exp(sk - mx) when sk != -INFINITY, contributes no value).
 * Key = Value = kv[pg, 0, t % block_size, 0..D-1]; score = scale * (q[m,h] . row) in f32; out = bf16(num / den) (0 when
 * den <= 0). Every operand, cu_seqlens, seqused and needed page index is validated before the first store. */
int ref_ernie_hca_dspark_attn(const RadArgs* a, RadStream) {
    if (!have(a, 7)) return RAD_E_INVAL;
    const RadTensor &q = a->t[0], &kv = a->t[1], &bt = a->t[2], &su = a->t[3], &cu = a->t[4], &sink = a->t[5], &out = a->t[6];
    const int64_t H = p_int(a, "heads", 0), D = p_int(a, "head_dim", 0), win = p_int(a, "window", 128);
    const double scale = hca_f64(a, "scale", 0.0);
    if (H < 1 || D < 1 || D > 4096 || win < 1 || win > INT32_MAX / 2 || !std::isfinite(scale) || scale <= 0.0 ||
        q.rank != 2 || kv.rank != 4 || bt.rank != 2 || su.rank != 1 || cu.rank != 1 || sink.rank != 1 || out.rank != 2 ||
        q.shape[0] < 1 || q.shape[1] != H * D || kv.shape[0] < 1 || kv.shape[1] != 1 || kv.shape[2] < 1 || kv.shape[3] != D ||
        bt.shape[0] < 1 || bt.shape[1] < 1 || su.shape[0] != bt.shape[0] || cu.shape[0] != bt.shape[0] + 1 || sink.shape[0] != H ||
        out.shape[0] != q.shape[0] || out.shape[1] != H * D || !pos_strides(q) || !pos_strides(kv) || !pos_strides(bt) ||
        !pos_strides(su) || !pos_strides(cu) || !pos_strides(sink) || !pos_strides(out) ||
        out.stride[0] < out.shape[1] * out.stride[1])
        return RAD_E_SHAPE;
    if (q.dtype != RAD_BF16 || kv.dtype != RAD_BF16 || out.dtype != RAD_BF16 || bt.dtype != RAD_I32 || su.dtype != RAD_I32 ||
        cu.dtype != RAD_I32 || sink.dtype != RAD_F32)
        return RAD_E_DTYPE;
    if (!q.data || !kv.data || !bt.data || !su.data || !cu.data || !sink.data || !out.data) return RAD_E_INVAL;
    const int64_t M = q.shape[0], S = bt.shape[0], max_blocks = bt.shape[1], pages = kv.shape[0], bs = kv.shape[2];
    for (int64_t h = 0; h < H; ++h) {
        const float s = ldt(&sink, h * sink.stride[0]);
        if (std::isnan(s) || s == INFINITY) return RAD_E_SHAPE;
    }
    if (at_i(cu, 0) != 0) return RAD_E_SHAPE;
    for (int64_t s = 0; s < S; ++s)
        if (at_i(cu, s) > at_i(cu, s + 1)) return RAD_E_SHAPE;
    if (at_i(cu, S) != M) return RAD_E_SHAPE;
    std::vector<int64_t> r_seq((size_t)M), r_lo((size_t)M), r_total((size_t)M);
    for (int64_t s = 0; s < S; ++s) {
        const int64_t qlen = at_i(cu, s + 1) - at_i(cu, s);
        if (qlen == 0) continue;
        const int64_t total_kv = at_i(su, s);
        if (total_kv < qlen) return RAD_E_SHAPE;
        const int64_t anchor = total_kv - qlen, lo = std::max<int64_t>(0, anchor - win + 1);
        for (int64_t t = lo; t < total_kv; ++t) {
            if (t / bs >= max_blocks) return RAD_E_SHAPE;
            const int64_t pg = ld_int(bt.dtype, bt.data, s * bt.stride[0] + (t / bs) * bt.stride[1]);
            if (pg < -1 || pg >= pages) return RAD_E_SHAPE;
        }
        for (int64_t m = at_i(cu, s); m < at_i(cu, s + 1); ++m) {
            r_seq[(size_t)m] = s;
            r_lo[(size_t)m] = lo;
            r_total[(size_t)m] = total_kv;
        }
    }
#pragma omp parallel for schedule(static)
    for (int64_t job = 0; job < M * H; ++job) {
        const int64_t m = job / H, h = job % H, s = r_seq[(size_t)m], lo = r_lo[(size_t)m], total_kv = r_total[(size_t)m];
        std::vector<float> sc;
        std::vector<int64_t> base;
        sc.reserve((size_t)(total_kv - lo) + 1);
        base.reserve((size_t)(total_kv - lo) + 1);
        for (int64_t t = lo; t < total_kv; ++t) {
            const int64_t pg = ld_int(bt.dtype, bt.data, s * bt.stride[0] + (t / bs) * bt.stride[1]);
            if (pg < 0) continue;
            const int64_t b = pg * kv.stride[0] + (t % bs) * kv.stride[2];
            float v = 0.0f;
            for (int64_t c = 0; c < D; ++c) v += ldt(&q, m * q.stride[0] + (h * D + c) * q.stride[1]) * ldt(&kv, b + c * kv.stride[3]);
            sc.push_back(v * (float)scale);
            base.push_back(b);
        }
        const float sk = ldt(&sink, h * sink.stride[0]);
        float mx = sk;
        for (float v : sc) mx = std::max(mx, v);
        float den = 0.0f;
        if (sk != -INFINITY) den += ref_exp(sk - mx);
        std::vector<float> wgt(sc.size());
        for (size_t j = 0; j < sc.size(); ++j) {
            wgt[j] = ref_exp(sc[j] - mx);
            den += wgt[j];
        }
        for (int64_t c = 0; c < D; ++c) {
            float num = 0.0f;
            for (size_t j = 0; j < sc.size(); ++j) num += wgt[j] * ldt(&kv, base[j] + c * kv.stride[3]);
            stt(&out, m * out.stride[0] + (h * D + c) * out.stride[1], den > 0.0f ? num / den : 0.0f);
        }
    }
    return RAD_OK;
}
