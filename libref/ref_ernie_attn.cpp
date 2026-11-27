/* ERNIE 5.5 MLA-DSA attention operations (story 3-2a), single rank, host only.
 *
 * The attention output of an MLA-DSA layer is: trailing-slice rotary on q and k_pe, query absorption
 * through W_UK, sparse latent attention with a per-head sink, value expansion through W_UV, the VHA
 * head mix, the element sigmoid gate (act_mul) and the output projection. These four ops are the
 * pieces no existing op computes (docs/ernie55/attention-reference.md, reuse table); the choices
 * recorded in docs/ernie55/mla-attention-ops.md are the numerics of this reference.
 *
 * Every launch validates every operand and every index before the first store, so a refusal never
 * leaves a partial result. Activations are bf16, accumulation is f32 with the summation order fixed
 * by the loops below (no reduction() clauses), results are rounded to bf16 once, at the store. */
#include "ref_common.h"
#include "ref_ernie_rope.h"
#include "ref_ops.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <vector>
using namespace ref;

namespace {
/* A positive-stride view: the ops index by element strides, so a row of a wider buffer is legal. */
bool pos_strides(const RadTensor& t) {
    for (uint32_t i = 0; i < t.rank; ++i)
        if (t.stride[i] < 1 || t.shape[i] < 0) return false;
    return true;
}
}  // namespace

/* ------------------------------------------------------------------ ernie_rope_tail */
/* GPT-J interleaved rotary on the TRAILING `rope_dim` values of each `width`-wide head, in place,
 * at the absolute position of the row (m08): pair i is (2i, 2i + 1) of the slice and turns by
 * position * theta^(-2i / rope_dim). The leading values are not touched. The angle is formed in
 * double and its cosine and sine rounded to f32; the rotation is f32 and rounded once to bf16. */
int ref_ernie_rope_tail(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &pos = a->t[1];
    const int64_t H = p_int(a, "heads", 0), W = p_int(a, "width", 0), R = p_int(a, "rope_dim", 0);
    const double theta = (double)p_f32(a, "theta", 0.0f);
    if (H < 1 || W < 1 || R < 2 || (R & 1) || R > W || !std::isfinite(theta) || theta <= 0.0 ||
        x.rank != 2 || pos.rank != 1 || x.shape[0] < 1 || x.shape[1] != H * W ||
        pos.shape[0] != x.shape[0] || !pos_strides(x) || !pos_strides(pos) ||
        x.stride[0] < x.shape[1] * x.stride[1])
        return RAD_E_SHAPE;
    if (x.dtype != RAD_BF16 || pos.dtype != RAD_I32) return RAD_E_DTYPE;
    if (!x.data || !pos.data) return RAD_E_INVAL;
    /* `rope_table` (story 3-2b amendment, additive): f32 (default, the 3-2a form) keeps cos and sin as f32; bf16
     * rounds each to bf16, the model-dtype table of vLLM's default RotaryEmbedding the DSA layers use. */
    const char* table = p_str(a, "rope_table", "f32");
    const bool bf16_table = !std::strcmp(table, "bf16");
    if (!bf16_table && std::strcmp(table, "f32")) return RAD_E_SHAPE;
    /* Story 3-4a, additive: `inverse` 1 turns by the opposite angle (sin negated); the yarn_* parameters select the YaRN table of the HCA layers. */
    const long long inverse = p_int(a, "inverse", 0);
    ErnieYarn yarn;
    if ((inverse != 0 && inverse != 1) || !ernie_yarn_args(a, &yarn)) return RAD_E_SHAPE;
    const int64_t M = x.shape[0];
    for (int64_t m = 0; m < M; ++m)
        if (ld_int(pos.dtype, pos.data, m * pos.stride[0]) < 0) return RAD_E_SHAPE;
    const std::vector<double> inv = ernie_rope_inv(theta, R, yarn);
#pragma omp parallel for schedule(static)
    for (int64_t m = 0; m < M; ++m) {
        const double p = (double)ld_int(pos.dtype, pos.data, m * pos.stride[0]);
        for (int64_t h = 0; h < H; ++h)
            for (int64_t i = 0; i < R / 2; ++i) {
                const double ang = p * inv[(size_t)i];
                float c = (float)std::cos(ang), s = (float)std::sin(ang);
                if (bf16_table) { c = round_dt(RAD_BF16, c); s = round_dt(RAD_BF16, s); }
                if (inverse) s = -s;
                const int64_t o0 = m * x.stride[0] + (h * W + (W - R) + 2 * i) * x.stride[1],
                              o1 = o0 + x.stride[1];
                const float x0 = ldt(&x, o0), x1 = ldt(&x, o1);
                stt(&x, o0, x0 * c - x1 * s);
                stt(&x, o1, x1 * c + x0 * s);
            }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ ernie_mla_project */
/* kv_b_proj as ERNIE lays it out: per head `k_dim` key rows then `v_dim` value rows of `latent_dim`
 * columns, so w is [heads * (k_dim + v_dim), latent_dim] and the two widths differ (ref_mla_project
 * needs them equal). Mode "absorb": y[m, h, :] = x_nope[m, h, :k_dim] . W_UK[h], the nope part of a
 * q row whose head is `q_width` (>= k_dim, default k_dim) wide; the trailing columns are not read.
 * Mode "expand": y[m, h, c] = sum_l x[m, h, l] * W_UV[h][c, l]. */
int ref_ernie_mla_project(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &w = a->t[1], &y = a->t[2];
    const int64_t H = p_int(a, "heads", 0), kd = p_int(a, "k_dim", 0), vd = p_int(a, "v_dim", 0),
                  L = p_int(a, "latent_dim", 0);
    const char* mode = p_str(a, "mode", "");
    const bool absorb = !std::strcmp(mode, "absorb"), expand = !std::strcmp(mode, "expand");
    const int64_t qw = p_int(a, "q_width", kd);
    if ((!absorb && !expand) || H < 1 || kd < 1 || vd < 1 || L < 1 || (absorb && qw < kd) ||
        (expand && qw != kd))
        return RAD_E_SHAPE;
    const int64_t xw = absorb ? qw : L, yw = absorb ? L : vd;
    if (x.rank != 2 || w.rank != 2 || y.rank != 2 || x.shape[0] < 1 || x.shape[1] != H * xw ||
        w.shape[0] != H * (kd + vd) || w.shape[1] != L || y.shape[0] != x.shape[0] ||
        y.shape[1] != H * yw || !pos_strides(x) || !pos_strides(w) || !pos_strides(y) ||
        y.stride[0] < y.shape[1] * y.stride[1])
        return RAD_E_SHAPE;
    if (x.dtype != RAD_BF16 || w.dtype != RAD_BF16 || y.dtype != RAD_BF16) return RAD_E_DTYPE;
    if (!x.data || !w.data || !y.data) return RAD_E_INVAL;
    const int64_t M = x.shape[0];
#pragma omp parallel for schedule(static)
    for (int64_t job = 0; job < M * H; ++job) {
        const int64_t m = job / H, h = job % H;
        for (int64_t col = 0; col < yw; ++col) {
            float sum = 0;
            if (absorb) {
                for (int64_t j = 0; j < kd; ++j)
                    sum += ldt(&x, m * x.stride[0] + (h * xw + j) * x.stride[1]) *
                           ldt(&w, (h * (kd + vd) + j) * w.stride[0] + col * w.stride[1]);
            } else {
                for (int64_t l = 0; l < L; ++l)
                    sum += ldt(&x, m * x.stride[0] + (h * xw + l) * x.stride[1]) *
                           ldt(&w, (h * (kd + vd) + kd + col) * w.stride[0] + l * w.stride[1]);
            }
            stt(&y, m * y.stride[0] + (h * yw + col) * y.stride[1], sum);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ ernie_vha_mix */
/* Value-head mixing (m17): out[t, h, c] = sum_g m[h, g] * o[t, g, c] on the head axis, with
 * m = I + V U^T folded once from the two [heads, rank] matrices: m[h, g] = [h == g] + sum_r V[h, r] U[g, r].
 * `fold` picks the dtype of the folded matrix (m18 is unresolved in the sources): "f32" (default)
 * keeps it, "bf16" rounds each entry to bf16 first. U and V may be held bf16 or f32. */
int ref_ernie_vha_mix(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor &o = a->t[0], &u = a->t[1], &v = a->t[2], &out = a->t[3];
    const int64_t H = p_int(a, "heads", 0), D = p_int(a, "dim", 0), R = p_int(a, "rank", 0);
    const int64_t W = p_int(a, "world_size", 1), rk = p_int(a, "tp_rank", 0);
    const char* fold = p_str(a, "fold", "f32");
    const bool fold_bf16 = !std::strcmp(fold, "bf16");
    if (!fold_bf16 && std::strcmp(fold, "f32")) return RAD_E_UNSUPPORTED;
    if (H < 1 || H > 4096 || D < 1 || R < 1 || W < 1 || W > 64 || rk < 0 || rk >= W || (H % W) != 0)
        return RAD_E_SHAPE;
    const int64_t Hloc = H / W;
    if (o.rank != 2 || u.rank != 2 || v.rank != 2 || out.rank != 2 || out.shape[0] < 1 ||
        o.shape[0] != W * out.shape[0] || o.shape[1] != Hloc * D || u.shape[0] != H ||
        u.shape[1] != R || v.shape[0] != H || v.shape[1] != R || out.shape[1] != Hloc * D ||
        !pos_strides(o) || !pos_strides(u) || !pos_strides(v) || !pos_strides(out) ||
        out.stride[0] < out.shape[1] * out.stride[1])
        return RAD_E_SHAPE;
    if (o.dtype != RAD_BF16 || out.dtype != RAD_BF16 || (u.dtype != RAD_BF16 && u.dtype != RAD_F32) ||
        (v.dtype != RAD_BF16 && v.dtype != RAD_F32))
        return RAD_E_DTYPE;
    if (!o.data || !u.data || !v.data || !out.data) return RAD_E_INVAL;
    /* Operand overlap (story 3-2b amendment): the mix reads every head of a row while it writes one, so `out`
     * may not share a byte with `o` (no in-place call). Extents, so a conservative refusal. */
    {
        auto extent = [](const RadTensor& t) {
            int64_t last = 0;
            for (uint32_t i = 0; i < t.rank; ++i) last += (t.shape[i] - 1) * t.stride[i];
            return std::pair<const char*, const char*>((const char*)t.data, (const char*)t.data + (last + 1) * 2);
        };
        const auto eo = extent(o), ex = extent(out);
        if (eo.first < ex.second && ex.first < eo.second) return RAD_E_INVAL;
    }
    const int64_t h0 = rk * Hloc;
    std::vector<float> m((size_t)(Hloc * H));
    for (int64_t hh = 0; hh < Hloc; ++hh) {
        const int64_t h = h0 + hh;
        for (int64_t g = 0; g < H; ++g) {
            float s = 0;
            for (int64_t r = 0; r < R; ++r)
                s += ldt(&v, h * v.stride[0] + r * v.stride[1]) * ldt(&u, g * u.stride[0] + r * u.stride[1]);
            float e = (h == g ? 1.0f : 0.0f) + s;
            if (fold_bf16) e = round_dt(RAD_BF16, e);
            m[(size_t)(hh * H + g)] = e;
        }
    }
    const int64_t M = out.shape[0];
#pragma omp parallel for schedule(static)
    for (int64_t job = 0; job < M * Hloc; ++job) {
        const int64_t t = job / Hloc, hh = job % Hloc;
        for (int64_t c = 0; c < D; ++c) {
            float sum = 0;
            for (int64_t g = 0; g < H; ++g) {
                const int64_t p_rk = g / Hloc, gh = g % Hloc;
                sum += m[(size_t)(hh * H + g)] * ldt(&o, (p_rk * M + t) * o.stride[0] + (gh * D + c) * o.stride[1]);
            }
            stt(&out, t * out.stride[0] + (hh * D + c) * out.stride[1], sum);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ ernie_vha_proj */
/* Distributed VHA low-rank projection: each TP rank owning Hloc = H / W heads projects its local
 * attention output o [M, Hloc * D] through its slice of U [H, R] into z [M, R * D]:
 *   z[t, r * D + c] = round_bf16( sum_{hh=0..Hloc-1} bf2f(o[t, hh * D + c]) * ld(u[h0 + hh, r]) )
 * Accumulated in f32 in fixed hh = 0..Hloc-1 order, rounded once to bf16 at store. */
extern "C" int ref_ernie_vha_proj(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor &o = a->t[0], &u = a->t[1], &z = a->t[2];
    const int64_t H = p_int(a, "heads", 0), D = p_int(a, "dim", 0), R = p_int(a, "rank", 0);
    const int64_t W = p_int(a, "world_size", 1), rk = p_int(a, "tp_rank", 0);
    if (H < 1 || H > 4096 || D < 1 || R < 1 || W < 1 || W > 64 || rk < 0 || rk >= W || (H % W) != 0)
        return RAD_E_SHAPE;
    const int64_t Hloc = H / W;
    if (o.rank != 2 || u.rank != 2 || z.rank != 2 || o.shape[0] < 1 || o.shape[1] != Hloc * D ||
        u.shape[0] != H || u.shape[1] != R || z.shape[0] != o.shape[0] || z.shape[1] != R * D ||
        !pos_strides(o) || !pos_strides(u) || !pos_strides(z) ||
        z.stride[0] < z.shape[1] * z.stride[1])
        return RAD_E_SHAPE;
    if (o.dtype != RAD_BF16 || z.dtype != RAD_BF16 || (u.dtype != RAD_BF16 && u.dtype != RAD_F32))
        return RAD_E_DTYPE;
    if (!o.data || !u.data || !z.data) return RAD_E_INVAL;
    {
        auto extent = [](const RadTensor& t) {
            int64_t last = 0;
            for (uint32_t i = 0; i < t.rank; ++i) last += (t.shape[i] - 1) * t.stride[i];
            const int64_t eb = (rad_dtype_bits(t.dtype) + 7) / 8;
            return std::pair<const char*, const char*>((const char*)t.data, (const char*)t.data + (last + 1) * eb);
        };
        const auto eo = extent(o), eu = extent(u), ez = extent(z);
        if ((eo.first < ez.second && ez.first < eo.second) ||
            (eu.first < ez.second && ez.first < eu.second))
            return RAD_E_INVAL;
    }
    const int64_t M = o.shape[0];
    const int64_t h0 = rk * Hloc;
#pragma omp parallel for schedule(static)
    for (int64_t job = 0; job < M * R; ++job) {
        const int64_t t = job / R, r = job % R;
        for (int64_t c = 0; c < D; ++c) {
            float sum = 0.0f;
            for (int64_t hh = 0; hh < Hloc; ++hh)
                sum += ldt(&o, t * o.stride[0] + (hh * D + c) * o.stride[1]) *
                       ldt(&u, (h0 + hh) * u.stride[0] + r * u.stride[1]);
            stt(&z, t * z.stride[0] + (r * D + c) * z.stride[1], sum);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ ernie_vha_apply */
/* Distributed VHA low-rank reconstruction + residual: after all_reduce on z [M, R * D], each TP rank
 * applies its local slice of V [H, R] and adds o [M, Hloc * D] into out [M, Hloc * D]:
 *   out[t, hh * D + c] = round_bf16( bf2f(o[t, hh * D + c]) +
 *                                    sum_{r=0..R-1} bf2f(z[t, r * D + c]) * ld(v[h0 + hh, r]) )
 * Accumulated in f32 in fixed r = 0..R-1 order, rounded once to bf16 at store. */
extern "C" int ref_ernie_vha_apply(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor &o = a->t[0], &z = a->t[1], &v = a->t[2], &out = a->t[3];
    const int64_t H = p_int(a, "heads", 0), D = p_int(a, "dim", 0), R = p_int(a, "rank", 0);
    const int64_t W = p_int(a, "world_size", 1), rk = p_int(a, "tp_rank", 0);
    if (H < 1 || H > 4096 || D < 1 || R < 1 || W < 1 || W > 64 || rk < 0 || rk >= W || (H % W) != 0)
        return RAD_E_SHAPE;
    const int64_t Hloc = H / W;
    if (o.rank != 2 || z.rank != 2 || v.rank != 2 || out.rank != 2 || o.shape[0] < 1 ||
        o.shape[1] != Hloc * D || z.shape[0] != o.shape[0] || z.shape[1] != R * D ||
        v.shape[0] != H || v.shape[1] != R || out.shape[0] != o.shape[0] ||
        out.shape[1] != Hloc * D || !pos_strides(o) || !pos_strides(z) || !pos_strides(v) ||
        !pos_strides(out) || out.stride[0] < out.shape[1] * out.stride[1])
        return RAD_E_SHAPE;
    if (o.dtype != RAD_BF16 || z.dtype != RAD_BF16 || out.dtype != RAD_BF16 ||
        (v.dtype != RAD_BF16 && v.dtype != RAD_F32))
        return RAD_E_DTYPE;
    if (!o.data || !z.data || !v.data || !out.data) return RAD_E_INVAL;
    {
        auto extent = [](const RadTensor& t) {
            int64_t last = 0;
            for (uint32_t i = 0; i < t.rank; ++i) last += (t.shape[i] - 1) * t.stride[i];
            const int64_t eb = (rad_dtype_bits(t.dtype) + 7) / 8;
            return std::pair<const char*, const char*>((const char*)t.data, (const char*)t.data + (last + 1) * eb);
        };
        const auto eo = extent(o), ez = extent(z), ev = extent(v), ex = extent(out);
        if ((ez.first < ex.second && ex.first < ez.second) ||
            (ev.first < ex.second && ex.first < ev.second))
            return RAD_E_INVAL;
        const bool exact_inplace =
            (out.data == o.data && out.stride[0] == o.stride[0] && out.stride[1] == o.stride[1]);
        if (!exact_inplace && eo.first < ex.second && ex.first < eo.second) return RAD_E_INVAL;
    }
    const int64_t M = o.shape[0];
    const int64_t h0 = rk * Hloc;
#pragma omp parallel for schedule(static)
    for (int64_t job = 0; job < M * Hloc; ++job) {
        const int64_t t = job / Hloc, hh = job % Hloc;
        for (int64_t c = 0; c < D; ++c) {
            float sum = 0.0f;
            for (int64_t r = 0; r < R; ++r)
                sum += ldt(&z, t * z.stride[0] + (r * D + c) * z.stride[1]) *
                       ldt(&v, (h0 + hh) * v.stride[0] + r * v.stride[1]);
            stt(&out, t * out.stride[0] + (hh * D + c) * out.stride[1],
                ldt(&o, t * o.stride[0] + (hh * D + c) * o.stride[1]) + sum);
        }
    }
    return RAD_OK;
}


/* ------------------------------------------------------------------ the fp8_ds_mla latent row (kv-fp8-latent)
 * An opt-in cache row for the MLA latent (`--kv-cache-dtype fp8`): one u8 row of  L + 4 * (L / quant_block) + 2 * R  bytes (656 at
 * L 512, R 64, quant_block 128):
 *     [0, L)                    E4M3 codes of the latent values
 *     [L, L + 4 * nt)           nt = L / quant_block little-endian f32 scales, one per quant_block latent values
 *     [L + 4 * nt, + 2 * R)     the R rope values as bf16, little endian, copied bit for bit
 * A scale is the smallest power of two s with tile_absmax <= 448 * s (absmax floored at 1e-30), so x / s and code * s are exact:
 * a stored value reads back as the f32 product code * s, which is also an exact bf16 (an E4M3 value has three mantissa bits).
 * The reader's arithmetic past the load is therefore the bf16 reader's on a cache holding the decoded values (libref has one kernel for both: bit for bit; libr4d: bit for bit against the
 * bf16 reader of the same kernel, see docs/OPS.md).
 *
 * Range. A finite bf16 value is at most 255 * 2^120 (0x7f7f), and a tile near the top takes scale 2^120 with normalised values up to 255, which E4M3 rounds to code 256 from 248 up;
 * 256 * 2^120 is 2^128, not finite in f32 or bf16. The writer therefore never emits a code whose decoded value is not finite: at scale 2^120 (the only scale where it can happen:
 * 448 * 2^119 = 1.75 * 2^127 is finite) the code magnitude is clamped to 240 (0x77), the largest code not above 255. A tile with scale below 2^120 is unaffected, so every byte the
 * writer produced for any value below 248 * 2^120 is the byte it always produced. A value whose magnitude is at most 2^-10 times its tile's scale (E4M3's ties-to-even zero threshold) decodes to a signed zero; a tile with an absmax below the 1e-30 floor takes scale 2^-108, so there
 * only magnitudes up to 2^-118 vanish and a value like 2^-101 survives.
 * A non-finite latent value (NaN, +inf, -inf) has no scale: the store refuses it (RAD_E_SHAPE) before any store, as ernie_hca_compress refuses a non-finite row; the rope values are copied
 * bit for bit and not inspected, and a row sent to slot -1 is not inspected at all. */
namespace {
struct LatentRow {
    int64_t L = 0, R = 0, qb = 0, nt = 0, bytes = 0;
    bool i8 = false;   /* an integer latent row (kv-fp8-latent v2): i8_ds_mla, i10_ds_mla, i12_ds_mla */
    int lb = 0;        /* low bits below the int8 high byte: 0 (i8_ds_mla), 2 (i10_ds_mla), 4 (i12_ds_mla); the code range is +-(2^(7 + lb) - 1) */
    bool idn = false;  /* bf16_ds_mla (diagnostic identity row): element c is the bf16 (row[c] << 8) | row[L + c], lb = 8, scales 1.0 */
    int64_t lo = 0;    /* bytes of packed low bits, L * lb / 8, between the high bytes and the scales */
    bool set(int64_t L_, int64_t R_, int64_t qb_) {
        if (L_ < 1 || R_ < 1 || qb_ < 1 || L_ % qb_ || L_ > (int64_t(1) << 20) || R_ > (int64_t(1) << 20) || (qb_ * lb) % 8) return false;   /* a tile owns whole bytes of low bits */
        L = L_; R = R_; qb = qb_; nt = L_ / qb_; lo = L_ * lb / 8; bytes = L_ + lo + 4 * nt + 2 * R_;
        return true;
    }
    size_t scale_off(int64_t t) const { return (size_t)(L + lo + 4 * t); }
    size_t rope_off(int64_t c) const { return (size_t)(L + lo + 4 * nt + 2 * c); }
    /* the signed code of element c: int8 high byte at row[c], then lb bits, element c at bit (c % (8 / lb)) * lb of byte L + c / (8 / lb) */
    int code(const uint8_t* row, int64_t c) const {
        int v = (int)(int8_t)row[c];
        if (lb) { const int per = 8 / lb; v = v * (1 << lb) + ((row[L + c / per] >> ((c % per) * lb)) & ((1 << lb) - 1)); }
        return v;
    }
};
/* The smallest power of two s with amax <= 448 * s: 448 = 1.75 * 2^8, so with e = floor(log2 amax) the candidates are 2^(e-8) (when
 * amax <= 1.75 * 2^e) and 2^(e-7). Exponent arithmetic only, no division, so a device kernel reproduces it exactly. */
float latent_scale(float amax) {
    if (!(amax >= 1e-30f)) amax = 1e-30f;
    const int e = std::ilogb(amax);
    const int k = amax <= std::ldexp(1.75f, e) ? e - 8 : e - 7;
    return std::ldexp(1.0f, k);
}
/* The largest scale exponent at which a code can decode past the finite range, and the largest code magnitude allowed there (see the Range paragraph above). */
constexpr int kLatentTopScaleExp = 120;
constexpr uint8_t kLatentTopCode = 0x77;
float latent_load_f32(const uint8_t* p) { float f; std::memcpy(&f, p, 4); return f; }
/* i8_ds_mla (kv-fp8-latent v2). A tile's scale is s = max(absmax, 1e-30) / 127 (one correctly rounded f32 division), a code is
 * rint(x / s) (one correctly rounded division, ties to even) clamped to [-127, 127], and a value reads back as the f32 product code * s.
 * Every quantity is one IEEE operation, so a device kernel using correctly rounded division (__fdiv_rn) reproduces each byte. The decoded
 * magnitude is at most 127 * s <= absmax * (1 + 2^-23): finite for every finite bf16 input (absmax <= 0x7f7f = 3.39e38 < FLT_MAX), so the
 * format has no top-of-range clamp. Values with |x| < s / 2 decode to zero; the 1e-30 floor keeps s a normal f32 (>= 7.8e-33), and a
 * subnormal x then always gives code 0. The error of a value is at most s / 2 = absmax / 254 (no relative error floor: uniform steps). */
float latent_scale_i8(float amax, int lb = 0) {
    if (!(amax >= 1e-30f)) amax = 1e-30f;
    return amax / (float)((1 << (7 + lb)) - 1);
}
/* The signed code rint(x / s) clamped to +-(2^(7 + lb) - 1): lb 0 is i8_ds_mla (+-127), 2 i10_ds_mla (+-511), 4 i12_ds_mla (+-2047). It is stored as the
 * int8 high byte floor(code / 2^lb) (arithmetic shift) and the lb low bits code mod 2^lb (two's complement split). */
int latent_code_int(float x, float s, int lb = 0) {
    const float m = (float)((1 << (7 + lb)) - 1);
    float q = std::rint(x / s);
    q = std::min(m, std::max(-m, q));
    return (int)q;
}
/* the kv_layout names of the latent row; 0 = not a latent row */
int latent_layout(const char* lay) {
    if (!std::strcmp(lay, "fp8_ds_mla")) return 1;
    if (!std::strcmp(lay, "i8_ds_mla")) return 2;
    if (!std::strcmp(lay, "i10_ds_mla")) return 3;
    if (!std::strcmp(lay, "i12_ds_mla")) return 4;
    if (!std::strcmp(lay, "bf16_ds_mla")) return 5;
    return 0;
}
/* Element c of the row at `row`: a latent value as code * scale, a rope value as its bf16. */
float latent_decode(const LatentRow& g, const uint8_t* row, int64_t c) {
    if (c < g.L && g.idn) return bf16_to_f32((uint16_t)((row[c] << 8) | row[g.L + c])) * latent_load_f32(row + g.scale_off(c / g.qb));
    if (c < g.L) return (g.i8 ? (float)g.code(row, c) : fp8e4m3_lut(row[c])) * latent_load_f32(row + g.scale_off(c / g.qb));
    return bf16_to_f32((uint16_t)(row[g.rope_off(c - g.L)] | (row[g.rope_off(c - g.L) + 1] << 8)));
}
}  // namespace

/* ernie_mla_latent_store: the writer of that row. src [M, latent_dim + rope_dim] bf16 (the normed latent then the rotated k_pe, as
 * kv_store's two halves hold it), slot_mapping [M] i32 (slot = page * block_size + offset; -1 skips the row), cache [pages, 1,
 * block_size, row bytes] u8 INOUT. Only the named rows are written. Every slot and every latent value of a named row is checked before any store (a NaN or infinity is RAD_E_SHAPE; see the Range paragraph above). */
int ref_ernie_mla_latent_store(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor &src = a->t[0], &sm = a->t[1], &cache = a->t[2];
    const int64_t L = p_int(a, "latent_dim", 0), R = p_int(a, "rope_dim", 0), bs = p_int(a, "block_size", 0),
                  qb = p_find(a, "quant_block") ? p_int(a, "quant_block", 0) : 128;
    LatentRow g;
    /* `kv_layout` (v2, optional): fp8_ds_mla (default) or i8_ds_mla; same row skeleton and size */
    const int lay = latent_layout(p_str(a, "kv_layout", "fp8_ds_mla"));
    if (!lay) return RAD_E_SHAPE;
    g.i8 = lay >= 2; g.lb = lay == 3 ? 2 : lay == 4 ? 4 : lay == 5 ? 8 : 0; g.idn = lay == 5;
    if (bs < 1 || !g.set(L, R, qb) || src.rank != 2 || sm.rank != 1 || cache.rank != 4 || src.shape[0] < 1 ||
        src.shape[1] != L + R || sm.shape[0] != src.shape[0] || cache.shape[0] < 1 || cache.shape[1] != 1 ||
        cache.shape[2] != bs || cache.shape[3] != g.bytes || !pos_strides(src) || !pos_strides(sm) || !pos_strides(cache) ||
        cache.stride[3] != 1 || cache.stride[2] != g.bytes || cache.stride[0] != bs * g.bytes || cache.shape[0] > INT32_MAX / bs)
        return RAD_E_SHAPE;
    if (src.dtype != RAD_BF16 || cache.dtype != RAD_U8 || (sm.dtype != RAD_I32 && sm.dtype != RAD_I64)) return RAD_E_DTYPE;
    if (!src.data || !sm.data || !cache.data) return RAD_E_INVAL;
    const int64_t M = src.shape[0], slots = cache.shape[0] * bs;
    for (int64_t m = 0; m < M; ++m) {
        const int64_t s = ld_int(sm.dtype, sm.data, m * sm.stride[0]);
        if (s < -1 || s >= slots) return RAD_E_SHAPE;
        if (s < 0) continue;   /* a row sent nowhere is not inspected */
        for (int64_t c = 0; c < g.L; ++c)
            if (!std::isfinite(ldt(&src, m * src.stride[0] + c * src.stride[1]))) return RAD_E_SHAPE;   /* no scale for a NaN / inf: refused before any store */
    }
    uint8_t* base = (uint8_t*)cache.data;
#pragma omp parallel for schedule(static) if (M >= 256)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t s = ld_int(sm.dtype, sm.data, m * sm.stride[0]);
        if (s < 0) continue;
        uint8_t* row = base + (size_t)s * (size_t)g.bytes;
        for (int64_t t = 0; t < g.nt; ++t) {
            float amax = 0.0f;
            for (int64_t c = t * g.qb; c < (t + 1) * g.qb; ++c) amax = std::max(amax, std::fabs(ldt(&src, m * src.stride[0] + c * src.stride[1])));
            if (g.idn) {   /* the identity row: the bf16 bits split into a high and a low byte plane, scale 1.0 */
                const float one = 1.0f;
                std::memcpy(row + g.scale_off(t), &one, 4);
                for (int64_t c = t * g.qb; c < (t + 1) * g.qb; ++c) {
                    const uint16_t bits = ((const uint16_t*)src.data)[m * src.stride[0] + c * src.stride[1]];
                    row[c] = (uint8_t)(bits >> 8);
                    row[g.L + c] = (uint8_t)(bits & 0xff);
                }
                if (const char* fe = std::getenv("RADIANCE_DIAG_ID_FLIP"); fe && (*fe == '1' || (*fe == '2' && t == 0 && s % 61 == 7))) row[g.L + t * g.qb] ^= 1;   /* DIAG ONLY: 1 ulp on the tile's first value (2: only tile 0 of slots 7 mod 61) */
                continue;
            }
            if (g.i8) {
                const float s8 = latent_scale_i8(amax, g.lb);
                std::memcpy(row + g.scale_off(t), &s8, 4);
                for (int64_t c = t * g.qb; c < (t + 1) * g.qb; ++c) {
                    const int q = latent_code_int(ldt(&src, m * src.stride[0] + c * src.stride[1]), s8, g.lb);
                    row[c] = (uint8_t)(int8_t)(q >> g.lb);
                    if (g.lb) {   /* the low bits of the code, packed 8 / lb to a byte from the low end (element c at bit (c % (8 / lb)) * lb of byte L + c / (8 / lb)) */
                        const int per = 8 / g.lb;
                        uint8_t& b = row[g.L + c / per];
                        const int sh = (int)(c % per) * g.lb, mk = ((1 << g.lb) - 1) << sh;
                        b = (uint8_t)((b & ~mk) | ((q & ((1 << g.lb) - 1)) << sh));
                    }
                }
                continue;
            }
            const float sc = latent_scale(amax);
            std::memcpy(row + g.scale_off(t), &sc, 4);
            const int k = std::ilogb(sc);
            for (int64_t c = t * g.qb; c < (t + 1) * g.qb; ++c) {
                uint8_t code = f32_to_fp8e4m3(std::ldexp(ldt(&src, m * src.stride[0] + c * src.stride[1]), -k));
                if (k >= kLatentTopScaleExp && (code & 0x7f) > kLatentTopCode) code = (uint8_t)((code & 0x80) | kLatentTopCode);   /* never a decoded value past the finite range */
                row[c] = code;
            }
        }
        for (int64_t c = 0; c < g.R; ++c) {
            const uint16_t bits = ((const uint16_t*)src.data)[m * src.stride[0] + (g.L + c) * src.stride[1]];
            row[g.rope_off(c)] = (uint8_t)(bits & 0xff);
            row[g.rope_off(c) + 1] = (uint8_t)(bits >> 8);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ ernie_mla_sparse_attn */
/* Absorbed latent MQA over a selected list of cache rows. Row r attends the cache rows named by
 * slots[r, :] (physical slot = page * block_size + offset; -1 is a pad and is skipped; a slot named
 * twice is attended twice, as the pinned HCA union does). A cache row is [latent | rope]
 * (latent_dim + rope_dim wide): score = scale * (q . latent + q_rope . rope), the value is the
 * latent. The per-head sink is one fp32 logit that joins the running max and the denominator and
 * has no value (m15): out = sum_j exp(s_j - mx) latent_j / (exp(sink - mx) + sum_j exp(s_j - mx)).
 * A sink of -inf is no sink; with no key and a finite sink the output is zero. */
int ref_ernie_mla_sparse_attn(const RadArgs* a, RadStream) {
    if (!have(a, 6)) return RAD_E_INVAL;
    const RadTensor &q = a->t[0], &qr = a->t[1], &kv = a->t[2], &slots = a->t[3], &sink = a->t[4],
                    &out = a->t[5];
    const int64_t H = p_int(a, "heads", 0), L = p_int(a, "latent_dim", 0),
                  R = p_int(a, "rope_dim", 0), bs = p_int(a, "block_size", 0),
                  W = p_int(a, "width", 0);
    /* `num_pages` is optional (story 3-2b amendment): the pool size is known only after the plugin declared the
     * op, so an absent value is the cache operand's own page count. A given value must still match it. */
    const int64_t pages = p_find(a, "num_pages") ? p_int(a, "num_pages", 0) : (kv.rank == 4 ? kv.shape[0] : 0);
    const float scale = p_f32(a, "scale", 0.0f);
    /* `kv_layout` (kv-fp8-latent, additive, optional): "bf16" (default, the cache row is [latent | rope] bf16) or "fp8_ds_mla" (the u8 row
     * above, `quant_block` latent values to a scale, default 128). Absent = the old behaviour. */
    const char* layout = p_str(a, "kv_layout", "bf16");
    const int lay = latent_layout(layout);   /* v2: i8_ds_mla is the int8 row of the same skeleton (see above) */
    const bool fp8 = lay != 0;
    if (!fp8 && std::strcmp(layout, "bf16")) return RAD_E_SHAPE;
    LatentRow lg;
    lg.i8 = lay >= 2; lg.lb = lay == 3 ? 2 : lay == 4 ? 4 : lay == 5 ? 8 : 0; lg.idn = lay == 5;
    if (fp8) {
        if (kv.dtype != RAD_U8) return RAD_E_DTYPE;
        if (!lg.set(L, R, p_find(a, "quant_block") ? p_int(a, "quant_block", 0) : 128)) return RAD_E_SHAPE;
    }
    if (H < 1 || L < 1 || L > 2048 || R < 1 || bs < 1 || pages < 1 || W < 1 ||
        pages > INT32_MAX / bs || !std::isfinite(scale) || scale <= 0 || q.rank != 2 ||
        qr.rank != 2 || kv.rank != 4 || slots.rank != 2 || sink.rank != 1 || out.rank != 2 ||
        q.shape[0] < 1 || q.shape[1] != H * L || qr.shape[0] != q.shape[0] || qr.shape[1] != H * R ||
        kv.shape[0] != pages || kv.shape[1] != 1 || kv.shape[2] != bs || kv.shape[3] != (fp8 ? lg.bytes : L + R) ||
        (fp8 && kv.stride[3] != 1) ||
        slots.shape[0] != q.shape[0] || slots.shape[1] != W || sink.shape[0] != H ||
        out.shape[0] != q.shape[0] || out.shape[1] != H * L || !pos_strides(q) || !pos_strides(qr) ||
        !pos_strides(kv) || !pos_strides(slots) || !pos_strides(sink) || !pos_strides(out) ||
        out.stride[0] < out.shape[1] * out.stride[1])
        return RAD_E_SHAPE;
    if (q.dtype != RAD_BF16 || qr.dtype != RAD_BF16 || kv.dtype != (fp8 ? RAD_U8 : RAD_BF16) ||
        out.dtype != RAD_BF16 || slots.dtype != RAD_I32 || sink.dtype != RAD_F32)
        return RAD_E_DTYPE;
    if (!q.data || !qr.data || !kv.data || !slots.data || !sink.data || !out.data) return RAD_E_INVAL;
    const int64_t M = q.shape[0];
    for (int64_t h = 0; h < H; ++h) {
        const float s = ldt(&sink, h * sink.stride[0]);
        if (std::isnan(s) || s == INFINITY) return RAD_E_SHAPE;
    }
    for (int64_t r = 0; r < M; ++r)
        for (int64_t j = 0; j < W; ++j) {
            const int64_t s = ld_int(slots.dtype, slots.data, r * slots.stride[0] + j * slots.stride[1]);
            if (s < -1 || s >= pages * bs) return RAD_E_SHAPE;
        }
    /* element c of the cache row at element offset b: the bf16 value, or the decoded fp8_ds_mla value */
    auto kvld = [&](int64_t b, int64_t c) -> float {
        return fp8 ? latent_decode(lg, (const uint8_t*)kv.data + b, c) : ldt(&kv, b + c * kv.stride[3]);
    };
#pragma omp parallel for schedule(static) if (M * H * W >= 4096)   /* each (row, head) job is independent: the result does not depend on the thread count; a tiny launch is not worth waking the pool */
    for (int64_t job = 0; job < M * H; ++job) {
        const int64_t r = job / H, h = job % H;
        std::vector<float> sc;
        std::vector<int64_t> base;
        sc.reserve((size_t)W);
        base.reserve((size_t)W);
        for (int64_t j = 0; j < W; ++j) {
            const int64_t s = ld_int(slots.dtype, slots.data, r * slots.stride[0] + j * slots.stride[1]);
            if (s < 0) continue;
            const int64_t b = (s / bs) * kv.stride[0] + (s % bs) * kv.stride[2];
            float v = 0;
            for (int64_t c = 0; c < L; ++c)
                v += ldt(&q, r * q.stride[0] + (h * L + c) * q.stride[1]) * kvld(b, c);
            for (int64_t c = 0; c < R; ++c)
                v += ldt(&qr, r * qr.stride[0] + (h * R + c) * qr.stride[1]) * kvld(b, L + c);
            sc.push_back(v * scale);
            base.push_back(b);
        }
        const float sk = ldt(&sink, h * sink.stride[0]);
        float mx = sk;
        for (float s : sc) mx = std::max(mx, s);
        float den = 0;
        if (sk != -INFINITY) den += ref_exp(sk - mx);
        std::vector<float> wgt(sc.size());
        for (size_t j = 0; j < sc.size(); ++j) {
            wgt[j] = ref_exp(sc[j] - mx);
            den += wgt[j];
        }
        for (int64_t c = 0; c < L; ++c) {
            float num = 0;
            for (size_t j = 0; j < sc.size(); ++j) num += wgt[j] * kvld(base[j], c);
            stt(&out, r * out.stride[0] + (h * L + c) * out.stride[1], den > 0 ? num / den : 0.0f);
        }
    }
    return RAD_OK;
}

/* ================================================================== the DSA indexer score path
 * Story 3-2i (docs/ernie55/indexer-scores.md, facts m09/m10/m13/m24/m26 of attention-reference.md).
 * Three host ops make the per-token logits that feed ernie_dsa_select:
 *   ernie_idx_query   q projection rows -> rope-first NeoX rope -> Hadamard -> FP8 + ue8m0 scale
 *   ernie_idx_key     k projection rows -> LayerNorm -> same transform -> 132-byte cache row
 *   ernie_idx_logits  sum_h relu(q_h . k * k_scale) * (w_h * q_scale_h * head_dim^-1/2 * heads^-1/2)
 * The bf16 boundaries are the pinned model's: after the rotary, after the Hadamard, after the
 * LayerNorm. The cache row is `head_dim` E4M3 bytes then one little-endian fp32 scale. */
namespace {
float bfr(float x) { return bf16_to_f32(f32_to_bf16(x)); }

/* Shared geometry of the three ops, validated once: head_dim a power of two (the Hadamard width), an
 * even rope slice that fits. */
double idx_f64(const RadArgs* a, const char* k, double dflt) {
    const RadParam* p = p_find(a, k);
    return p && p->kind == RAD_P_F64 ? p->dval : p ? (double)p_f32(a, k, (float)dflt) : dflt;
}
bool idx_geometry(int64_t D, int64_t R) { return D >= 4 && D <= 256 && pow2(D) && R >= 2 && !(R & 1) && R <= D; }

/* The cos/sin pairs of one absolute position: angle in double, rounded to f32 and then, for the
 * default `bf16` table, to bf16 (vLLM builds the table in the model dtype; docs record the choice). */
bool idx_table(const RadArgs* a, bool* bf16_table) {
    const char* t = p_str(a, "rope_table", "bf16");
    if (!std::strcmp(t, "bf16")) { *bf16_table = true; return true; }
    if (!std::strcmp(t, "f32")) { *bf16_table = false; return true; }
    return false;
}
void idx_cos_sin(int64_t pos, const std::vector<double>& inv, bool bf16_table, float* c, float* s) {
    for (size_t i = 0; i < inv.size(); ++i) {
        const double ang = (double)pos * inv[i];
        float cc = (float)std::cos(ang), ss = (float)std::sin(ang);
        if (bf16_table) { cc = bfr(cc); ss = bfr(ss); }
        c[i] = cc;
        s[i] = ss;
    }
}
/* v holds `D` bf16 values: [rope R | nope D-R] (rope first). Rotates the leading R split-half
 * (pair i is (i, i + R/2), NeoX), leaves the rest, then the orthonormal Hadamard over all D, bf16.
 * The arithmetic between the bf16 boundaries is double: the reference is the exact value rounded where
 * the pinned model rounds, so an f32 kernel disagrees with it only by its own rounding noise. */
void idx_transform(double* v, int64_t D, int64_t R, const float* c, const float* s) {
    for (int64_t i = 0; i < R / 2; ++i) {
        const double x0 = v[i], x1 = v[i + R / 2];
        v[i] = bfr((float)(x0 * (double)c[i] - x1 * (double)s[i]));
        v[i + R / 2] = bfr((float)(x1 * (double)c[i] + x0 * (double)s[i]));
    }
    for (int64_t h = 1; h < D; h <<= 1)
        for (int64_t i = 0; i < D; i += 2 * h)
            for (int64_t j = i; j < i + h; ++j) {
                const double x = v[j], y = v[j + h];
                v[j] = x + y;
                v[j + h] = x - y;
            }
    const double norm = 1.0 / std::sqrt((double)D);
    for (int64_t d = 0; d < D; ++d) v[d] = bfr((float)(v[d] * norm));
}
/* The quantiser rule (story 4-1, `quant_rule`): the amax floor and the code range the scale is derived from. nvidia (default, the pinned
 * per_token_group_quant_fp8): floor 1e-10, range 448. rocm: floor 1e-4. rocm_fnuz: floor 1e-4 and range 224 (the FNUZ e4m3 divisor); the code bytes stay
 * E4M3 (the FNUZ byte encoding is a device-side matter, see docs/ernie55/device-dispatch.md). */
struct IdxQuant { float floor; float qmax; };
bool idx_rule(const RadArgs* a, IdxQuant* r) {
    const char* t = p_str(a, "quant_rule", "nvidia");
    if (!std::strcmp(t, "nvidia")) { *r = {1e-10f, 448.0f}; return true; }
    if (!std::strcmp(t, "rocm")) { *r = {1e-4f, 448.0f}; return true; }
    if (!std::strcmp(t, "rocm_fnuz")) { *r = {1e-4f, 224.0f}; return true; }
    return false;
}
/* One group: ue8m0 scale 2^ceil(log2(max(absmax, floor) / range)), codes = E4M3(clamp(v / scale)). */
float idx_quant(const double* v, int64_t D, uint8_t* codes, const IdxQuant& qr) {
    float am = qr.floor;
    for (int64_t d = 0; d < D; ++d) am = std::max(am, (float)std::fabs(v[d]));
    int e = 0;
    const float fr = std::frexp(am / qr.qmax, &e);
    const float sc = std::ldexp(1.0f, e - (fr == 0.5f));
    for (int64_t d = 0; d < D; ++d) codes[d] = f32_to_fp8e4m3(std::clamp((float)v[d] / sc, -qr.qmax, qr.qmax));
    return sc;
}
/* The engine hands a paged KV group's layer as [pages, n_head_kv, block, row]; the indexer's group has one head, so
 * a rank-4 cache with shape[1] == 1 is the same bytes as [pages, block, row] (story 3-2b amendment). */
RadTensor idx_cache_view(const RadTensor& c) {
    RadTensor v = c;
    if (c.rank == 4 && c.shape[1] == 1) {
        v.rank = 3;
        v.shape[1] = c.shape[2]; v.shape[2] = c.shape[3];
        v.stride[1] = c.stride[2]; v.stride[2] = c.stride[3];
    }
    return v;
}
std::vector<double> idx_inv_freq(int64_t R, double theta) {
    std::vector<double> inv((size_t)(R / 2));
    for (int64_t i = 0; i < R / 2; ++i) inv[(size_t)i] = std::pow(theta, -2.0 * (double)i / (double)R);
    return inv;
}
}  // namespace

/* ernie_idx_query: q [M, heads * head_dim] bf16 (wq_b output), positions [M] -> qf [M, heads *
 * head_dim] E4M3, qs [M, heads] f32 (one power-of-two scale per head row). */
int ref_ernie_idx_query(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor &q = a->t[0], &pos = a->t[1], &qf = a->t[2], &qs = a->t[3];
    const int64_t H = p_int(a, "heads", 0), D = p_int(a, "head_dim", 0), R = p_int(a, "rope_dim", 0);
    const double theta = idx_f64(a, "theta", 0.0);
    bool table = true;
    IdxQuant qr{1e-10f, 448.0f};
    if (q.rank != 2 || q.shape[0] < 1) return RAD_E_SHAPE;
    if (H < 1 || H > 256 || !idx_geometry(D, R) || !std::isfinite(theta) || theta <= 0.0 || !idx_table(a, &table) || !idx_rule(a, &qr))
        return RAD_E_SHAPE;
    const int64_t M = q.shape[0];
    if (q.shape[1] != H * D || pos.rank != 1 || pos.shape[0] != M || qf.rank != 2 || qf.shape[0] != M ||
        qf.shape[1] != H * D || qs.rank != 2 || qs.shape[0] != M || qs.shape[1] != H || !pos_strides(q) ||
        !pos_strides(pos) || !pos_strides(qf) || !pos_strides(qs) || q.stride[0] < q.shape[1] * q.stride[1] ||
        qf.stride[0] < qf.shape[1] * qf.stride[1] || qs.stride[0] < qs.shape[1] * qs.stride[1])
        return RAD_E_SHAPE;
    if (q.dtype != RAD_BF16 || pos.dtype != RAD_I32 || qf.dtype != RAD_F8E4M3 || qs.dtype != RAD_F32)
        return RAD_E_DTYPE;
    if (!q.data || !pos.data || !qf.data || !qs.data) return RAD_E_INVAL;
    for (int64_t m = 0; m < M; ++m)
        if (ld_int(pos.dtype, pos.data, m * pos.stride[0]) < 0) return RAD_E_SHAPE;
    const std::vector<double> inv = idx_inv_freq(R, theta);
#pragma omp parallel for schedule(static)
    for (int64_t m = 0; m < M; ++m) {
        std::vector<float> c((size_t)(R / 2)), s((size_t)(R / 2));
        std::vector<double> v((size_t)D);
        std::vector<uint8_t> codes((size_t)D);
        idx_cos_sin(ld_int(pos.dtype, pos.data, m * pos.stride[0]), inv, table, c.data(), s.data());
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t d = 0; d < D; ++d) v[(size_t)d] = (double)ldt(&q, m * q.stride[0] + (h * D + d) * q.stride[1]);
            idx_transform(v.data(), D, R, c.data(), s.data());
            const float sc = idx_quant(v.data(), D, codes.data(), qr);
            for (int64_t d = 0; d < D; ++d)
                ((uint8_t*)qf.data)[m * qf.stride[0] + (h * D + d) * qf.stride[1]] = codes[(size_t)d];
            stt(&qs, m * qs.stride[0] + h * qs.stride[1], sc);
        }
    }
    return RAD_OK;
}

/* ernie_idx_key: k [M, head_dim] bf16 (the k half of the key/weights projection), the k_norm gamma and
 * beta [head_dim] bf16, positions [M] and physical slots [M] (page * block + offset; negative is
 * padding) -> rows written into cache [pages, block, head_dim + 4] u8. Rows are applied in order, a
 * repeated slot keeps the last. */
int ref_ernie_idx_key(const RadArgs* a, RadStream) {
    if (!have(a, 6)) return RAD_E_INVAL;
    const RadTensor &k = a->t[0], &gm = a->t[1], &bt = a->t[2], &pos = a->t[3], &slot = a->t[4];
    const RadTensor cache = idx_cache_view(a->t[5]);
    const int64_t D = p_int(a, "head_dim", 0), R = p_int(a, "rope_dim", 0);
    const double theta = idx_f64(a, "theta", 0.0), eps = idx_f64(a, "eps", 1e-6);
    bool table = true;
    IdxQuant qr{1e-10f, 448.0f};
    if (k.rank != 2 || k.shape[0] < 1) return RAD_E_SHAPE;
    if (!idx_geometry(D, R) || !std::isfinite(theta) || theta <= 0.0 || !(eps >= 0.0) || !std::isfinite(eps) ||
        !idx_table(a, &table) || !idx_rule(a, &qr))
        return RAD_E_SHAPE;
    const int64_t M = k.shape[0];
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
    const int64_t slots = cache.shape[0] * cache.shape[1];
    for (int64_t m = 0; m < M; ++m) {
        const int64_t sl = ld_int(slot.dtype, slot.data, m * slot.stride[0]);
        if (ld_int(pos.dtype, pos.data, m * pos.stride[0]) < 0 || sl < -1 || sl >= slots) return RAD_E_SHAPE;
    }
    const std::vector<double> inv = idx_inv_freq(R, theta);
    std::vector<uint8_t> rows((size_t)(M * (D + 4)));
#pragma omp parallel for schedule(static)
    for (int64_t m = 0; m < M; ++m) {
        std::vector<float> c((size_t)(R / 2)), s((size_t)(R / 2));
        std::vector<double> v((size_t)D);
        idx_cos_sin(ld_int(pos.dtype, pos.data, m * pos.stride[0]), inv, table, c.data(), s.data());
        double mean = 0;
        for (int64_t d = 0; d < D; ++d) { v[(size_t)d] = (double)ldt(&k, m * k.stride[0] + d * k.stride[1]); mean += v[(size_t)d]; }
        mean /= (double)D;
        double var = 0;
        for (int64_t d = 0; d < D; ++d) { const double t = v[(size_t)d] - mean; var += t * t; }
        const double rstd = 1.0 / std::sqrt(var / (double)D + eps);
        for (int64_t d = 0; d < D; ++d)
            v[(size_t)d] = bfr((float)((v[(size_t)d] - mean) * rstd * (double)ldt(&gm, d * gm.stride[0]) + (double)ldt(&bt, d * bt.stride[0])));
        idx_transform(v.data(), D, R, c.data(), s.data());
        uint8_t* row = rows.data() + (size_t)(m * (D + 4));
        const float sc = idx_quant(v.data(), D, row, qr);
        std::memcpy(row + D, &sc, sizeof sc);
    }
    for (int64_t m = 0; m < M; ++m) {
        const int64_t sl = ld_int(slot.dtype, slot.data, m * slot.stride[0]);
        if (sl >= 0) std::memcpy((uint8_t*)cache.data + sl * (D + 4), rows.data() + (size_t)(m * (D + 4)), (size_t)(D + 4));
    }
    return RAD_OK;
}

/* ernie_idx_logits: for the query row m (position p, request req[m]) the logit of every logical key
 * id j < end = max(p + 1 - window, 0): sum_h relu((qf_h . key_j) * key_scale_j) * ((w_h * qs_h) *
 * head_dim^-1/2 * heads^-1/2), the key found through block_table[req][j / block] -> cache. Columns at
 * or past `end` are -inf, and so is a key on an uninitialised page (entry < 0, >= pages, or a block
 * past the table). scores is [M, L] indexed by LOGICAL id, what ernie_dsa_select reads. */
int ref_ernie_idx_logits(const RadArgs* a, RadStream) {
    if (!have(a, 8)) return RAD_E_INVAL;
    const RadTensor cache = idx_cache_view(a->t[3]);
    const RadTensor &qf = a->t[0], &qs = a->t[1], &w = a->t[2], &bt = a->t[4], &req = a->t[5],
                    &pos = a->t[6], &out = a->t[7];
    const int64_t H = p_int(a, "heads", 0), D = p_int(a, "head_dim", 0), window = p_int(a, "window", 128);
    /* col_start (optional, default 0): output column j is logical id col_start + j, the form the tiled selection scores a context with. */
    const bool has_col0 = p_find(a, "col_start") != nullptr;
    const int64_t col0 = p_int(a, "col_start", 0);
    if (qf.rank != 2 || qf.shape[0] < 1) return RAD_E_SHAPE;
    if (H < 1 || H > 256 || D < 4 || D > 256 || !pow2(D) || window < 1 || window > INT32_MAX / 2 || col0 < 0 || col0 > INT32_MAX / 2)
        return RAD_E_SHAPE;
    const int64_t M = qf.shape[0];
    if (qf.shape[1] != H * D || qs.rank != 2 || qs.shape[0] != M || qs.shape[1] != H || w.rank != 2 ||
        w.shape[0] != M || w.shape[1] != H || cache.rank != 3 || cache.shape[0] < 1 || cache.shape[1] < 1 ||
        cache.shape[2] != D + 4 || bt.rank != 2 || bt.shape[0] < 1 || bt.shape[1] < 1 || req.rank != 1 ||
        req.shape[0] != M || pos.rank != 1 || pos.shape[0] != M || out.rank != 2 || out.shape[0] != M ||
        out.shape[1] < 1 || !pos_strides(qf) || !pos_strides(qs) || !pos_strides(w) || !pos_strides(bt) ||
        !pos_strides(req) || !pos_strides(pos) || !pos_strides(out) || cache.stride[2] != 1 ||
        cache.stride[1] != D + 4 || cache.stride[0] != cache.shape[1] * (D + 4) ||
        out.stride[0] < out.shape[1] * out.stride[1])
        return RAD_E_SHAPE;
    if ((p_find(a, "tokens") && p_int(a, "tokens", 0) != out.shape[1]) ||
        (p_find(a, "seqs") && p_int(a, "seqs", 0) != bt.shape[0]) ||
        (p_find(a, "blocks") && p_int(a, "blocks", 0) != bt.shape[1]) ||
        (p_find(a, "block_size") && p_int(a, "block_size", 0) != cache.shape[1]) ||
        (p_find(a, "num_pages") && p_int(a, "num_pages", 0) != cache.shape[0]))
        return RAD_E_SHAPE;
    if (qf.dtype != RAD_F8E4M3 || qs.dtype != RAD_F32 || w.dtype != RAD_BF16 || cache.dtype != RAD_U8 ||
        bt.dtype != RAD_I32 || req.dtype != RAD_I32 || pos.dtype != RAD_I32 || out.dtype != RAD_F32)
        return RAD_E_DTYPE;
    if (!qf.data || !qs.data || !w.data || !cache.data || !bt.data || !req.data || !pos.data || !out.data)
        return RAD_E_INVAL;
    const int64_t L = out.shape[1], S = bt.shape[0], B = bt.shape[1], block = cache.shape[1], pages = cache.shape[0];
    for (int64_t m = 0; m < M; ++m) {
        const int64_t p = ld_int(pos.dtype, pos.data, m * pos.stride[0]), r = ld_int(req.dtype, req.data, m * req.stride[0]);
        if (p < 0 || r < 0 || r >= S || (!has_col0 && std::max<int64_t>(p + 1 - window, 0) > L)) return RAD_E_SHAPE;
    }
    const float wscale = (float)(std::pow((double)D, -0.5) * std::pow((double)H, -0.5));
#pragma omp parallel for schedule(static)
    for (int64_t job = 0; job < M * L; ++job) {
        const int64_t m = job / L, jl = job % L, j = col0 + jl;
        const int64_t p = ld_int(pos.dtype, pos.data, m * pos.stride[0]), r = ld_int(req.dtype, req.data, m * req.stride[0]);
        float result = -INFINITY;
        const int64_t end = std::max<int64_t>(p + 1 - window, 0), b = j / block;
        if (j < end && b < B) {
            const int64_t page = ld_int(bt.dtype, bt.data, r * bt.stride[0] + b * bt.stride[1]);
            if (page >= 0 && page < pages) {
                const uint8_t* row = (const uint8_t*)cache.data + (page * block + j % block) * (D + 4);
                float kv[256], ks;
                for (int64_t d = 0; d < D; ++d) kv[d] = fp8e4m3_lut(row[d]);
                std::memcpy(&ks, row + D, sizeof ks);
                result = 0;
                for (int64_t h = 0; h < H; ++h) {
                    float dot = 0;
                    for (int64_t d = 0; d < D; ++d) dot += ldt(&qf, m * qf.stride[0] + (h * D + d) * qf.stride[1]) * kv[d];
                    const float wt = (ldt(&w, m * w.stride[0] + h * w.stride[1]) * ldt(&qs, m * qs.stride[0] + h * qs.stride[1])) * wscale;
                    result += std::max(dot * ks, 0.0f) * wt;
                }
            }
        }
        stt(&out, m * out.stride[0] + jl * out.stride[1], result);
    }
    return RAD_OK;
}


/* ernie_dsa_topk_tiled: ernie_idx_logits followed by ernie_dsa_select with the score plane cut into tiles of `scores.shape[1]` logical
 * columns instead of one plane as wide as the context (docs/ernie55/score-tiling.md). Per tile every row scores its columns exactly as
 * ernie_idx_logits does (the same per-column arithmetic, the key found through block_table_idx), then merges the tile's candidates
 * (score not -inf, page initialised through block_table_sel) into a running list kept to the `topk` best under the strict total order
 * (score descending, id ascending). That order is the one ernie_dsa_select cuts a row by, so the kept set after the last tile is the
 * global top-k and the result equals ernie_dsa_select over the untiled plane bit for bit: at most `topk` candidates in ascending id
 * order, more cut to the `topk` best in descending score order with ties to the LOWER id, then the window, then -1, then the slots.
 * No store happens before every tile is scored: a NaN or +inf score in a row's candidate range, a bad position or request, or a candidate
 * range beyond the optional `tokens` bound refuses with RAD_E_SHAPE and leaves ids and slots untouched. Operands:
 *   0 qf  1 qs  2 w  3 cache  4 block_table_idx  5 req  6 positions  7 block_table_sel  8 scores [M, tile] (scratch)  9 ids  10 slots
 *   11 kept (scratch, i32, at least M * (2 * topk + 3) words: the device op's running top-k and per-row verdicts, which keep its outputs unwritten until
 *      every tile is valid; this reference keeps its running lists on the host and never touches it, but refuses one that is too small, as the device does)
 *   12 seqused (optional, the engine's host bound for the device op; unused here) */
int ref_ernie_dsa_topk_tiled(const RadArgs* a, RadStream) {
    if (!have(a, 12)) return RAD_E_INVAL;
    const RadTensor cache = idx_cache_view(a->t[3]);
    const RadTensor &qf = a->t[0], &qs = a->t[1], &w = a->t[2], &ibt = a->t[4], &req = a->t[5], &pos = a->t[6], &lbt = a->t[7],
                    &plane = a->t[8], &ids = a->t[9], &slots = a->t[10], &kept = a->t[11];
    const int64_t H = p_int(a, "heads", 0), D = p_int(a, "head_dim", 0), window = p_int(a, "window", 128), topk = p_int(a, "topk", 0),
                  bs = p_int(a, "block_size", 0);
    const int64_t pages_sel = p_find(a, "num_pages") ? p_int(a, "num_pages", 0) : (bs >= 1 ? INT32_MAX / bs : 0);
    if (qf.rank != 2 || qf.shape[0] < 1) return RAD_E_SHAPE;
    if (H < 1 || H > 256 || D < 4 || D > 256 || !pow2(D) || window < 1 || window > INT32_MAX / 2 || topk < 1 || topk > INT32_MAX / 2 ||
        bs < 1 || pages_sel < 1 || pages_sel > INT32_MAX / bs)
        return RAD_E_SHAPE;
    const int64_t M = qf.shape[0], W = topk + window;
    if (qf.shape[1] != H * D || qs.rank != 2 || qs.shape[0] != M || qs.shape[1] != H || w.rank != 2 || w.shape[0] != M ||
        w.shape[1] != H || cache.rank != 3 || cache.shape[0] < 1 || cache.shape[1] < 1 || cache.shape[2] != D + 4 || ibt.rank != 2 ||
        ibt.shape[0] < 1 || ibt.shape[1] < 1 || lbt.rank != 2 || lbt.shape[0] < 1 || lbt.shape[1] < 1 || req.rank != 1 ||
        req.shape[0] != M || pos.rank != 1 || pos.shape[0] != M || (plane.rank != 2 && plane.rank != 1) ||
        (plane.rank == 2 && (plane.shape[0] != M || plane.shape[1] < 1)) || (plane.rank == 1 && plane.shape[0] < M) ||
        ids.rank != 2 || slots.rank != 2 || ids.shape[0] != M || slots.shape[0] != M || ids.shape[1] < W || slots.shape[1] != ids.shape[1] ||
        !pos_strides(qf) || !pos_strides(qs) || !pos_strides(w) || !pos_strides(ibt) || !pos_strides(lbt) || !pos_strides(req) ||
        !pos_strides(pos) || !pos_strides(plane) || !pos_strides(ids) || !pos_strides(slots) || kept.rank != 1 ||
        kept.shape[0] < M * (2 * topk + 3) || !pos_strides(kept) || cache.stride[2] != 1 ||
        cache.stride[1] != D + 4 || cache.stride[0] != cache.shape[1] * (D + 4) ||
        (plane.rank == 2 && plane.stride[0] < plane.shape[1] * plane.stride[1]))
        return RAD_E_SHAPE;
    if (cache.shape[1] != bs) return RAD_E_SHAPE;   /* both groups page by the selection's block size */
    /* The scratch plane is [M, tile] (the tile is its width) or a flat [numel >= M] buffer, from which a launch with M rows takes
     * tile = numel / M columns a row: a decode step with few rows scores wide tiles out of the same bytes a prefill chunk cuts narrow. */
    const int64_t tile = plane.rank == 2 ? plane.shape[1] : plane.shape[0] / M;
    const int64_t prow = plane.rank == 2 ? plane.stride[0] : tile * plane.stride[0], pcol = plane.rank == 2 ? plane.stride[1] : plane.stride[0];
    if ((p_find(a, "tile") && p_int(a, "tile", 0) != tile) || (p_find(a, "seqs") && p_int(a, "seqs", 0) != ibt.shape[0]) ||
        (p_find(a, "blocks") && p_int(a, "blocks", 0) != ibt.shape[1]))
        return RAD_E_SHAPE;
    if (qf.dtype != RAD_F8E4M3 || qs.dtype != RAD_F32 || w.dtype != RAD_BF16 || cache.dtype != RAD_U8 || ibt.dtype != RAD_I32 ||
        req.dtype != RAD_I32 || pos.dtype != RAD_I32 || lbt.dtype != RAD_I32 || plane.dtype != RAD_F32 || ids.dtype != RAD_I32 ||
        slots.dtype != RAD_I32 || kept.dtype != RAD_I32)
        return RAD_E_DTYPE;
    if (!qf.data || !qs.data || !w.data || !cache.data || !ibt.data || !req.data || !pos.data || !lbt.data || !plane.data || !ids.data ||
        !slots.data || !kept.data)
        return RAD_E_INVAL;
    const bool bounded = p_find(a, "tokens") != nullptr;
    const int64_t bound = bounded ? p_int(a, "tokens", 0) : 0;
    if (bounded && bound < 1) return RAD_E_SHAPE;
    const int64_t Si = ibt.shape[0], Bi = ibt.shape[1], Sl = lbt.shape[0], Bl = lbt.shape[1], block = bs,
                  pages_idx = cache.shape[0];
    std::vector<int64_t> endv((size_t)M);
    int64_t maxend = 0;
    for (int64_t m = 0; m < M; ++m) {
        const int64_t p = ld_int(pos.dtype, pos.data, m * pos.stride[0]), r = ld_int(req.dtype, req.data, m * req.stride[0]);
        if (p < 0 || r < 0 || r >= Si || r >= Sl) return RAD_E_SHAPE;
        const int64_t end = std::max<int64_t>(p + 1 - window, 0);
        if (bounded && end > bound) return RAD_E_SHAPE;
        endv[(size_t)m] = end;
        maxend = std::max(maxend, end);
    }
    const float wscale = (float)(std::pow((double)D, -0.5) * std::pow((double)H, -0.5));
    typedef std::pair<float, int32_t> Cand;
    auto better = [](const Cand& x, const Cand& y) { return x.first > y.first || (x.first == y.first && x.second < y.second); };
    std::vector<std::vector<Cand>> run((size_t)M);
    std::vector<int64_t> ncand((size_t)M, 0);
    int bad = 0;
    auto sel_page = [&](int64_t r, int64_t id) -> int64_t {
        const int64_t b = id / bs;
        if (b >= Bl) return -1;
        const int64_t e = ld_int(lbt.dtype, lbt.data, r * lbt.stride[0] + b * lbt.stride[1]);
        return e >= 0 && e < pages_sel ? e : -1;
    };
    for (int64_t t0 = 0; t0 < maxend; t0 += tile) {
        const int64_t tw = std::min(tile, maxend - t0);
        /* score the tile: column jj of the plane is logical id t0 + jj */
#pragma omp parallel for schedule(static)
        for (int64_t job = 0; job < M * tw; ++job) {
            const int64_t m = job / tw, jj = job % tw, j = t0 + jj;
            const int64_t p = ld_int(pos.dtype, pos.data, m * pos.stride[0]), r = ld_int(req.dtype, req.data, m * req.stride[0]);
            float result = -INFINITY;
            const int64_t end = std::max<int64_t>(p + 1 - window, 0), b = j / block;
            if (j < end && b < Bi) {
                const int64_t page = ld_int(ibt.dtype, ibt.data, r * ibt.stride[0] + b * ibt.stride[1]);
                if (page >= 0 && page < pages_idx) {
                    const uint8_t* row = (const uint8_t*)cache.data + (page * block + j % block) * (D + 4);
                    float kv[256], ks;
                    for (int64_t d = 0; d < D; ++d) kv[d] = fp8e4m3_lut(row[d]);
                    std::memcpy(&ks, row + D, sizeof ks);
                    result = 0;
                    for (int64_t h = 0; h < H; ++h) {
                        float dot = 0;
                        for (int64_t d = 0; d < D; ++d) dot += ldt(&qf, m * qf.stride[0] + (h * D + d) * qf.stride[1]) * kv[d];
                        const float wt = (ldt(&w, m * w.stride[0] + h * w.stride[1]) * ldt(&qs, m * qs.stride[0] + h * qs.stride[1])) * wscale;
                        result += std::max(dot * ks, 0.0f) * wt;
                    }
                }
            }
            stt(&plane, m * prow + jj * pcol, result);
        }
        /* merge the tile into each row's running top-k */
#pragma omp parallel for schedule(static)
        for (int64_t m = 0; m < M; ++m) {
            const int64_t r = ld_int(req.dtype, req.data, m * req.stride[0]), cols = std::min(tw, endv[(size_t)m] - t0);
            std::vector<Cand>& rl = run[(size_t)m];
            for (int64_t jj = 0; jj < cols; ++jj) {
                const float v = ldt(&plane, m * prow + jj * pcol);
                if (std::isnan(v) || v == INFINITY) {
#pragma omp atomic write
                    bad = 1;
                    continue;
                }
                const int64_t id = t0 + jj;
                if (v > -INFINITY && sel_page(r, id) >= 0) {
                    rl.emplace_back(v, (int32_t)id);
                    ++ncand[(size_t)m];
                    if ((int64_t)rl.size() >= 2 * topk) {
                        std::nth_element(rl.begin(), rl.begin() + topk, rl.end(), better);
                        rl.resize((size_t)topk);
                    }
                }
            }
        }
    }
    if (bad) return RAD_E_SHAPE;
    int32_t *ip = (int32_t*)ids.data, *op = (int32_t*)slots.data;
    const int64_t os = ids.stride[1], ss = slots.stride[1];
#pragma omp parallel for schedule(static)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t seq = ld_int(pos.dtype, pos.data, m * pos.stride[0]) + 1, r = ld_int(req.dtype, req.data, m * req.stride[0]);
        std::vector<Cand>& rl = run[(size_t)m];
        if (ncand[(size_t)m] > topk) {
            std::sort(rl.begin(), rl.end(), better);
            rl.resize((size_t)topk);
        }
        std::vector<int32_t> line;
        line.reserve((size_t)W);
        for (const Cand& c : rl) line.push_back(c.second);
        for (int64_t id = seq - std::min(seq, window); id < seq; ++id)
            if (sel_page(r, id) >= 0) line.push_back((int32_t)id);
        for (int64_t j = 0; j < ids.shape[1]; ++j) {
            const int32_t id = j < (int64_t)line.size() ? line[(size_t)j] : -1;
            ip[m * ids.stride[0] + j * os] = id;
            op[m * slots.stride[0] + j * ss] = id < 0 ? -1 : (int32_t)(sel_page(r, id) * bs + id % bs);
        }
    }
    return RAD_OK;
}


/* ================================================================== story 3-2b helpers */
/* ernie_row_req: the request (sequence) index of each live row, from the step's cu_seqlens [S + 1]. The DSA ops
 * take one request id per row; the batch carries bounds. cu must start at 0, never decrease and end at the
 * number of rows. */
int ref_ernie_row_req(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor &cu = a->t[0], &req = a->t[1];
    if (cu.rank != 1 || req.rank != 1 || cu.shape[0] < 2 || req.shape[0] < 1 || !pos_strides(cu) ||
        !pos_strides(req))
        return RAD_E_SHAPE;
    if (cu.dtype != RAD_I32 || req.dtype != RAD_I32) return RAD_E_DTYPE;
    if (!cu.data || !req.data) return RAD_E_INVAL;
    const int64_t S = cu.shape[0] - 1, M = req.shape[0];
    int64_t prev = ld_int(cu.dtype, cu.data, 0);
    if (prev != 0) return RAD_E_SHAPE;
    for (int64_t s = 1; s <= S; ++s) {
        const int64_t c = ld_int(cu.dtype, cu.data, s * cu.stride[0]);
        if (c < prev) return RAD_E_SHAPE;
        prev = c;
    }
    if (prev != M) return RAD_E_SHAPE;
    for (int64_t s = 0; s < S; ++s) {
        const int64_t lo = ld_int(cu.dtype, cu.data, s * cu.stride[0]), hi = ld_int(cu.dtype, cu.data, (s + 1) * cu.stride[0]);
        for (int64_t r = lo; r < hi; ++r) ((int32_t*)req.data)[r * req.stride[0]] = (int32_t)s;
    }
    return RAD_OK;
}

/* ernie_head_slice: y[m, h, j] = x[m, h, offset + j] for j < count, from x [M, heads * width] to y [M, heads * count]
 * (bf16). It takes the rope columns of every q head (q_b's output is [nope | rope] per head) into the compact
 * [M, heads * rope] operand ernie_mla_sparse_attn reads. Operands may not overlap. */
int ref_ernie_head_slice(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &y = a->t[1];
    const int64_t H = p_int(a, "heads", 0), W = p_int(a, "width", 0), off = p_int(a, "offset", -1),
                  n = p_int(a, "count", 0);
    if (H < 1 || W < 1 || off < 0 || n < 1 || off + n > W || x.rank != 2 || y.rank != 2 || x.shape[0] < 1 ||
        x.shape[1] != H * W || y.shape[0] != x.shape[0] || y.shape[1] != H * n || !pos_strides(x) ||
        !pos_strides(y) || x.stride[0] < x.shape[1] * x.stride[1] || y.stride[0] < y.shape[1] * y.stride[1])
        return RAD_E_SHAPE;
    if (x.dtype != RAD_BF16 || y.dtype != RAD_BF16) return RAD_E_DTYPE;
    if (!x.data || !y.data) return RAD_E_INVAL;
    {
        auto extent = [](const RadTensor& t) {
            int64_t last = 0;
            for (uint32_t i = 0; i < t.rank; ++i) last += (t.shape[i] - 1) * t.stride[i];
            return std::pair<const char*, const char*>((const char*)t.data, (const char*)t.data + (last + 1) * 2);
        };
        const auto ex = extent(x), ey = extent(y);
        if (ex.first < ey.second && ey.first < ex.second) return RAD_E_INVAL;
    }
    const int64_t M = x.shape[0];
#pragma omp parallel for schedule(static)
    for (int64_t m = 0; m < M; ++m)
        for (int64_t h = 0; h < H; ++h)
            for (int64_t j = 0; j < n; ++j)
                stt(&y, m * y.stride[0] + (h * n + j) * y.stride[1],
                    ldt(&x, m * x.stride[0] + (h * W + off + j) * x.stride[1]));
    return RAD_OK;
}
