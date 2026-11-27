/* Manifold-constrained hyper-connections. The residual mixing matrix is indexed
 * [source, destination]; transposing it leaves every shape valid and changes the model.
 * The first Sinkhorn pass is softmax + eps, then column normalization. Later passes
 * add eps to both denominators. These are separate operations from Qwen's hc_read/write. */
#include "ref_common.h"
#include "ref_ops.h"

#include <algorithm>
#include <cmath>

using namespace ref;

namespace {
bool matrix(const RadTensor& t, int64_t rows, int64_t cols) {
    return t.data && t.rank == 2 && t.shape[0] == rows && t.shape[1] == cols;
}
}

namespace {
int64_t part_stride(int64_t hc) { return (hc * (hc + 2) + 1 + 3) / 4 * 4; }

bool pre_params(const RadArgs* a, int64_t* n, int64_t* hc, int64_t* iters, float* eps,
                float* heps) {
    *n = p_int(a, "n", 0);
    *hc = p_int(a, "hc", 0);
    *iters = p_int(a, "iterations", 0);
    *eps = p_f32(a, "eps", 0);
    *heps = p_f32(a, "hc_eps", 0);
    return *n > 0 && *hc >= 2 && *hc <= 8 && *iters >= 1 && *iters <= 256 && std::isfinite(*eps) &&
           std::isfinite(*heps) && *eps > 0 && *heps >= 0;
}

/* The outputs' shapes, shared by mhc_pre and mhc_pre_finish. */
bool pre_outputs(const RadTensor& h, const RadTensor& scale, const RadTensor& base,
                 const RadTensor& x, const RadTensor& post, const RadTensor& comb, int64_t rows,
                 int64_t n, int64_t hc) {
    const int64_t mix = hc * (hc + 2);
    return matrix(h, rows, hc * n) && scale.rank == 1 && scale.dtype == RAD_F32 &&
           numel(&scale) == 3 && base.rank == 1 && base.dtype == RAD_F32 && numel(&base) == mix &&
           matrix(x, rows, n) && matrix(post, rows, hc) && matrix(comb, rows, hc * hc) &&
           post.dtype == RAD_F32 && comb.dtype == RAD_F32;
}


namespace {
/* ernie_mhc_pre[_rms] summation order (the mHC libref proposal, option B). The device (libr4d
 * ernie_mhc_pre[_rms], two-stage form) sums the squares, the 24 coefficient dots and the fused RMSNorm variance in a fixed order that
 * does not depend on the step row count; libref adopts that order so device == libref bit for bit instead of within bf16 steps. It
 * applies where the device runs the two-stage form, which needs ALL of (mhc_device_form below): bf16 h; h, fn, x, scale, base, post, comb (and the
 * gain) with a last stride of 1; row pitches hp >= hc*n, fnp >= hc*n, xp >= n; x not overlapping h, fn, scale, base or gain (the op's own contract;
 * libref does not model it); hc*n >= kMhcChunkCols; and at most max_chunks chunks of 2048 columns, max_chunks = min(65535, (2n - 2) / (4 (mix + 1)))
 * (the partials are staged in x's row). Elsewhere the device runs a general fallback kernel, whose order is not part of this contract, and libref keeps
 * its plain sequential sums. The lane step of the dot/square sums (vec4 below) and of the fused variance (vec8) are chosen by ALIGNMENT, not by the
 * width alone, exactly as the device chooses them.
 *   dot/square of one row: width split into chunks of 2048 columns; 128 lanes; lane t adds, from column col0 + 4t in steps of 512, the
 *   term ((p0 + p1) + p2) + p3 of four consecutive products to its running sum when vec4 (h 8-byte aligned with a row pitch % 4 == 0, fn 16-byte
 *   aligned with a row pitch % 4 == 0, width % 4 == 0), else one product from col0 + t in steps of 128; a wave of 32 lanes reduces by xor-butterfly 16, 8, 4, 2, 1; the chunk is ((w0 + w1) + w2) + w3 of its four waves; the row is
 *   the chunks added in order from 0.
 *   fused RMSNorm variance (x = the f32 collapse): 256 lanes; lane t adds x_i^2 one element at a time over i = 8t + 2048k + e (e = 0..7,
 *   k outer) when vec8 (the row's h and x and the gain 16-byte aligned, n % 8 == 0), else i = t + 256k; a wave reduces by the same butterfly; the eight waves combine as
 *   ((w0 + w4) + (w2 + w6)) + ((w1 + w5) + (w3 + w7)). */
constexpr int64_t kMhcChunkCols = 2048;

struct MhcDeviceForm { bool chunked; bool vec4; };
/* Which of the device's forms ernie_mhc_pre[_rms] takes for this call (see the contract above). `chunked` is the two-stage form; vec4 its lane step. */
MhcDeviceForm mhc_device_form(const RadTensor& h, const RadTensor& fn, const RadTensor& scale, const RadTensor& base, const RadTensor& x,
                              const RadTensor& post, const RadTensor& comb, const RadTensor* gain, int64_t n, int64_t hc) {
    auto unit = [](const RadTensor& t) { return t.rank >= 1 && t.stride[t.rank - 1] == 1; };
    const int64_t width = hc * n, mix = hc * (hc + 2);
    const int64_t hp = h.rank == 2 ? h.stride[0] : 0, fnp = fn.rank == 2 ? fn.stride[0] : 0, xp = x.rank == 2 ? x.stride[0] : 0;
    const bool contig = h.dtype == RAD_BF16 && unit(h) && unit(fn) && unit(x) && unit(scale) && unit(base) && unit(post) && unit(comb) &&
                        (!gain || unit(*gain)) && xp >= n && hp >= width && fnp >= width;
    const int64_t chunks = (width + kMhcChunkCols - 1) / kMhcChunkCols;
    const int64_t max_chunks = std::min<int64_t>(65535, (n * 2 - 2) / ((mix + 1) * 4));
    MhcDeviceForm f{};
    f.chunked = contig && width >= kMhcChunkCols && chunks <= max_chunks;
    f.vec4 = (((uintptr_t)h.data & 7u) == 0u) && ((hp & 3) == 0) && (((uintptr_t)fn.data & 15u) == 0u) && ((fnp & 3) == 0) && ((width & 3) == 0);
    return f;
}

float mhc_wave32(const float* v32) {
    float v[32];
    for (int l = 0; l < 32; ++l) v[l] = v32[l];
    for (int mask = 16; mask > 0; mask >>= 1) {
        float nv[32];
        for (int l = 0; l < 32; ++l) nv[l] = v[l] + v[l ^ mask];
        for (int l = 0; l < 32; ++l) v[l] = nv[l];
    }
    return v[0];
}

template <class P>
float mhc_chunked_sum(int64_t width, bool vec4, P prod) {
    float row = 0;
    for (int64_t col0 = 0; col0 < width; col0 += kMhcChunkCols) {
        const int64_t col1 = std::min(col0 + kMhcChunkCols, width);
        float lane[128];
        for (int t = 0; t < 128; ++t) {
            float s = 0;
            if (vec4) {
                for (int64_t i = col0 + 4 * t; i < col1; i += 512) s += ((prod(i) + prod(i + 1)) + prod(i + 2)) + prod(i + 3);
            } else {
                for (int64_t i = col0 + t; i < col1; i += 128) s += prod(i);
            }
            lane[t] = s;
        }
        float w[4];
        for (int q = 0; q < 4; ++q) w[q] = mhc_wave32(lane + 32 * q);
        row += ((w[0] + w[1]) + w[2]) + w[3];
    }
    return row;
}

template <class V>
float mhc_rms_sumsq(int64_t n, bool vec8, V val) {
    float wv[8];
    for (int q = 0; q < 8; ++q) {
        float lane[32];
        for (int l = 0; l < 32; ++l) {
            const int t = q * 32 + l;
            float s = 0;
            if (vec8) {
                for (int64_t i = 8 * (int64_t)t; i < n; i += 2048)
                    for (int e = 0; e < 8; ++e) { const float v = val(i + e); s += v * v; }
            } else {
                for (int64_t i = t; i < n; i += 256) { const float v = val(i); s += v * v; }
            }
            lane[l] = s;
        }
        wv[q] = mhc_wave32(lane);
    }
    return ((wv[0] + wv[4]) + (wv[2] + wv[6])) + ((wv[1] + wv[5]) + (wv[3] + wv[7]));
}
}  /* namespace */

/* Row r from its RMS-scaled coefficients: gates, Sinkhorn mixing, and the collapse of h into `x`
 * (mhc_pre) or the pre gates into `pre_out` (mhc_mix); either may be null. */
void finish_row(const float* coeff, const RadTensor& h, const RadTensor& scale,
                const RadTensor& base, const RadTensor* x, const RadTensor& post,
                const RadTensor& comb, int64_t r, int64_t n, int64_t hc, int64_t iters,
                float heps, const RadTensor* pre_out = nullptr) {
    float pre[8], c[64];
    const int64_t hb = rowoff(&h, r, hc * n), hs = laststride(&h);
    for (int64_t j = 0; j < hc; ++j) {
        pre[j] =
            1.0f / (1.0f +
                    std::exp(-(coeff[j] * ldt(&scale, 0) + ldt(&base, j * laststride(&base))))) +
            heps;
        const float z = coeff[hc + j] * ldt(&scale, laststride(&scale)) +
                        ldt(&base, (hc + j) * laststride(&base));
        stt(&post, rowoff(&post, r, hc) + j * laststride(&post), 2.0f / (1.0f + std::exp(-z)));
    }
    for (int64_t j = 0; j < hc; ++j) {
        float mx = -INFINITY;
        for (int64_t k = 0; k < hc; ++k) {
            const int64_t q = 2 * hc + j * hc + k;
            c[j * hc + k] =
                coeff[q] * ldt(&scale, 2 * laststride(&scale)) + ldt(&base, q * laststride(&base));
            mx = std::max(mx, c[j * hc + k]);
        }
        float sum = 0;
        for (int64_t k = 0; k < hc; ++k) {
            c[j * hc + k] = std::exp(c[j * hc + k] - mx);
            sum += c[j * hc + k];
        }
        for (int64_t k = 0; k < hc; ++k) c[j * hc + k] = c[j * hc + k] / sum + heps;
    }
    for (int64_t it = 0; it < iters; ++it) {
        if (it)
            for (int64_t j = 0; j < hc; ++j) {
                float sum = 0;
                for (int64_t k = 0; k < hc; ++k) sum += c[j * hc + k];
                for (int64_t k = 0; k < hc; ++k) c[j * hc + k] /= sum + heps;
            }
        for (int64_t k = 0; k < hc; ++k) {
            float sum = 0;
            for (int64_t j = 0; j < hc; ++j) sum += c[j * hc + k];
            for (int64_t j = 0; j < hc; ++j) c[j * hc + k] /= sum + heps;
        }
    }
    const int64_t cb = rowoff(&comb, r, hc * hc), cs = laststride(&comb);
    for (int64_t j = 0; j < hc * hc; ++j) stt(&comb, cb + j * cs, c[j]);
    if (pre_out)
        for (int64_t j = 0; j < hc; ++j)
            stt(pre_out, rowoff(pre_out, r, hc) + j * laststride(pre_out), pre[j]);
    if (!x) return;
    const int64_t xb = rowoff(x, r, n), xs = laststride(x);
    for (int64_t i = 0; i < n; ++i) {
        float v = 0;
        for (int64_t j = 0; j < hc; ++j) v += pre[j] * ldt(&h, hb + (j * n + i) * hs);
        stt(x, xb + i * xs, v);
    }
}
}

int ref_mhc_pre(const RadArgs* a, RadStream) {
    if (!have(a, 7)) return RAD_E_INVAL;
    int64_t n, hc, iters;
    float eps, heps;
    if (!pre_params(a, &n, &hc, &iters, &eps, &heps)) return RAD_E_SHAPE;
    const auto &h = a->t[0], &fn = a->t[1], &scale = a->t[2], &base = a->t[3];
    const auto &x = a->t[4], &post = a->t[5], &comb = a->t[6];
    const int64_t width = hc * n, mix = hc * (hc + 2);
    if (h.rank != 2) return RAD_E_SHAPE;
    const int64_t rows = h.shape[0];
    if (rows < 1 || !matrix(fn, mix, width) || fn.dtype != RAD_F32 ||
        !pre_outputs(h, scale, base, x, post, comb, rows, n, hc))
        return RAD_E_SHAPE;
    /* the generic mhc_pre keeps the plain sequential sums: the two-stage order below belongs to ernie_mhc_pre[_rms] only */
#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        float coeff[80];
        const int64_t hb = rowoff(&h, r, width), hs = laststride(&h);
        float sq = 0;
        for (int64_t i = 0; i < width; ++i) {
            const float v = ldt(&h, hb + i * hs);
            sq += v * v;
        }
        const float inv = 1.0f / std::sqrt(sq / (float)width + eps);
        for (int64_t j = 0; j < mix; ++j) {
            float dot = 0;
            const int64_t fb = rowoff(&fn, j, width), fs = laststride(&fn);
            for (int64_t i = 0; i < width; ++i) dot += ldt(&h, hb + i * hs) * ldt(&fn, fb + i * fs);
            coeff[j] = dot * inv;
        }
        finish_row(coeff, h, scale, base, &x, post, comb, r, n, hc, iters, heps);
    }
    return RAD_OK;
}

int ref_mhc_pre_partial(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const int64_t n = p_int(a, "n", 0), hc = p_int(a, "hc", 0), chunk = p_int(a, "chunk", 0);
    const int64_t first = p_int(a, "first", -1), count = p_int(a, "chunks", 0);
    if (n <= 0 || hc < 2 || hc > 8 || chunk <= 0 || (hc * n) % chunk || first < 0 || count < 1 ||
        first + count > hc * n / chunk)
        return RAD_E_SHAPE;
    const auto &h = a->t[0], &fn = a->t[1], &part = a->t[2];
    const int64_t width = hc * n, mix = hc * (hc + 2), S = part_stride(hc);
    if (h.rank != 2) return RAD_E_SHAPE;
    const int64_t rows = h.shape[0];
    if (rows < 1 || !matrix(h, rows, width) || !matrix(fn, mix, width) || fn.dtype != RAD_F32 ||
        !matrix(part, rows, count * S) || part.dtype != RAD_F32)
        return RAD_E_SHAPE;
#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t hb = rowoff(&h, r, width), hs = laststride(&h);
        const int64_t pb = rowoff(&part, r, count * S), ps = laststride(&part);
        for (int64_t c = 0; c < count; ++c) {
            const int64_t lo = (first + c) * chunk;
            for (int64_t j = 0; j < S; ++j) {
                float sum = 0;
                if (j < mix) {
                    const int64_t fb = rowoff(&fn, j, width), fs = laststride(&fn);
                    for (int64_t i = lo; i < lo + chunk; ++i)
                        sum += ldt(&h, hb + i * hs) * ldt(&fn, fb + i * fs);
                } else if (j == mix) {
                    for (int64_t i = lo; i < lo + chunk; ++i) {
                        const float v = ldt(&h, hb + i * hs);
                        sum += v * v;
                    }
                }
                stt(&part, pb + (c * S + j) * ps, sum);
            }
        }
    }
    return RAD_OK;
}

int ref_mhc_pre_finish(const RadArgs* a, RadStream) {
    if (!have(a, 7)) return RAD_E_INVAL;
    int64_t n, hc, iters;
    float eps, heps;
    if (!pre_params(a, &n, &hc, &iters, &eps, &heps)) return RAD_E_SHAPE;
    const int64_t chunk = p_int(a, "chunk", 0);
    if (chunk <= 0 || (hc * n) % chunk) return RAD_E_SHAPE;
    const auto &part = a->t[0], &h = a->t[1], &scale = a->t[2], &base = a->t[3];
    const auto &x = a->t[4], &post = a->t[5], &comb = a->t[6];
    const int64_t width = hc * n, mix = hc * (hc + 2), S = part_stride(hc), chunks = width / chunk;
    if (h.rank != 2) return RAD_E_SHAPE;
    const int64_t rows = h.shape[0];
    if (rows < 1 || !matrix(part, rows, chunks * S) || part.dtype != RAD_F32 ||
        !pre_outputs(h, scale, base, x, post, comb, rows, n, hc))
        return RAD_E_SHAPE;
    /* the generic mhc_pre_finish sums the gathered partials in chunk order, as the device does: no second order here */
#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        float coeff[80];
        const int64_t pb = rowoff(&part, r, chunks * S), ps = laststride(&part);
        float sq = 0;
        for (int64_t c = 0; c < chunks; ++c) sq += ldt(&part, pb + (c * S + mix) * ps);
        const float inv = 1.0f / std::sqrt(sq / (float)width + eps);
        for (int64_t j = 0; j < mix; ++j) {
            float dot = 0;
            for (int64_t c = 0; c < chunks; ++c) dot += ldt(&part, pb + (c * S + j) * ps);
            coeff[j] = dot * inv;
        }
        finish_row(coeff, h, scale, base, &x, post, comb, r, n, hc, iters, heps);
    }
    return RAD_OK;
}

int ref_mhc_post(const RadArgs* a, RadStream) {
    if (!have(a, 5)) return RAD_E_INVAL;
    const int64_t n = p_int(a, "n", 0), hc = p_int(a, "hc", 0);
    if (n <= 0 || hc < 2 || hc > 8) return RAD_E_SHAPE;
    const auto &x = a->t[0], &h = a->t[1], &post = a->t[2], &comb = a->t[3], &out = a->t[4];
    if (x.rank != 2) return RAD_E_SHAPE;
    const int64_t rows = x.shape[0];
    if (rows < 1 || !matrix(x, rows, n) || !matrix(h, rows, hc * n) || !matrix(post, rows, hc) ||
        !matrix(comb, rows, hc * hc) || !matrix(out, rows, hc * n) || out.data == h.data ||
        post.dtype != RAD_F32 || comb.dtype != RAD_F32)
        return RAD_E_SHAPE;
#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t hb = rowoff(&h, r, hc * n), cb = rowoff(&comb, r, hc * hc);
        const int64_t pb = rowoff(&post, r, hc), xb = rowoff(&x, r, n);
        const int64_t ob = rowoff(&out, r, hc * n);
        for (int64_t k = 0; k < hc; ++k)
            for (int64_t i = 0; i < n; ++i) {
                float v = 0;
                for (int64_t j = 0; j < hc; ++j)
                    v += ldt(&comb, cb + (j * hc + k) * laststride(&comb)) *
                         ldt(&h, hb + (j * n + i) * laststride(&h));
                v += ldt(&post, pb + k * laststride(&post)) * ldt(&x, xb + i * laststride(&x));
                stt(&out, ob + (k * n + i) * laststride(&out), v);
            }
    }
    return RAD_OK;
}

int ref_ernie_add_mhc_post(const RadArgs* a, RadStream) {
    if (!have(a, 6)) return RAD_E_INVAL;
    const int64_t n = p_int(a, "n", 0), hc = p_int(a, "hc", 0);
    if (n <= 0 || hc < 2 || hc > 8) return RAD_E_SHAPE;
    const auto &ta = a->t[0], &tb = a->t[1], &h = a->t[2], &post = a->t[3], &comb = a->t[4], &out = a->t[5];
    if (ta.rank != 2 || ta.shape[0] < 1) return RAD_E_SHAPE;
    const int64_t rows = ta.shape[0];
    if (!matrix(ta, rows, n) || !matrix(tb, rows, n) || !matrix(h, rows, hc * n) ||
        !matrix(post, rows, hc) || !matrix(comb, rows, hc * hc) || !matrix(out, rows, hc * n) ||
        post.dtype != RAD_F32 || comb.dtype != RAD_F32 || out.data == h.data)
        return RAD_E_SHAPE;
    REF_PAR_FOR(rows * hc * n)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t ab = rowoff(&ta, r, n), bb = rowoff(&tb, r, n), hb = rowoff(&h, r, hc * n),
                      pb = rowoff(&post, r, hc), cb = rowoff(&comb, r, hc * hc), ob = rowoff(&out, r, hc * n);
        const int64_t as = laststride(&ta), bs = laststride(&tb), hs = laststride(&h),
                      ps = laststride(&post), cs = laststride(&comb), os = laststride(&out);
        for (int64_t i = 0; i < n; ++i) {
            float xv = ldt(&ta, ab + i * as) + ldt(&tb, bb + i * bs);
            if (out.dtype == RAD_BF16) xv = bf16_to_f32(f32_to_bf16(xv));
            for (int64_t dst = 0; dst < hc; ++dst) {
                float v = 0.0f;
                for (int64_t src = 0; src < hc; ++src)
                    v += ldt(&comb, cb + (src * hc + dst) * cs) * ldt(&h, hb + (src * n + i) * hs);
                v += ldt(&post, pb + dst * ps) * xv;
                stt(&out, ob + (dst * n + i) * os, v);
            }
        }
    }
    return RAD_OK;
}

int ref_mhc_contract(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const int64_t n = p_int(a, "n", 0), hc = p_int(a, "hc", 0);
    if (n <= 0 || hc < 2 || hc > 8) return RAD_E_SHAPE;
    const auto &h = a->t[0], &x = a->t[1];
    if (h.rank != 2 || h.shape[0] < 1 || !matrix(h, h.shape[0], hc * n) ||
        !matrix(x, h.shape[0], n))
        return RAD_E_SHAPE;
#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < h.shape[0]; ++r)
        for (int64_t i = 0; i < n; ++i) {
            float v = 0;
            for (int64_t j = 0; j < hc; ++j)
                v += ldt(&h, rowoff(&h, r, hc * n) + (j * n + i) * laststride(&h));
            stt(&x, rowoff(&x, r, n) + i * laststride(&x), v / (float)hc);
        }
    return RAD_OK;
}

/* mhc_pre without the collapse: the gates and mixing of the stream h, with `pre` written out for a
 * later collapse (mhc_collapse). DeepSeek-V4.1 collapses each sublayer's input with the pre gates
 * the PREVIOUS sublayer computed, so the two halves are separate ops there. */
int ref_mhc_mix(const RadArgs* a, RadStream) {
    if (!have(a, 7)) return RAD_E_INVAL;
    int64_t n, hc, iters;
    float eps, heps;
    if (!pre_params(a, &n, &hc, &iters, &eps, &heps)) return RAD_E_SHAPE;
    const auto &h = a->t[0], &fn = a->t[1], &scale = a->t[2], &base = a->t[3];
    const auto &pre = a->t[4], &post = a->t[5], &comb = a->t[6];
    const int64_t width = hc * n, mix = hc * (hc + 2);
    if (h.rank != 2) return RAD_E_SHAPE;
    const int64_t rows = h.shape[0];
    if (rows < 1 || !matrix(fn, mix, width) || fn.dtype != RAD_F32 || !matrix(pre, rows, hc) ||
        pre.dtype != RAD_F32 || !matrix(h, rows, width) || scale.rank != 1 ||
        scale.dtype != RAD_F32 || numel(&scale) != 3 || base.rank != 1 ||
        base.dtype != RAD_F32 || numel(&base) != mix || !matrix(post, rows, hc) ||
        !matrix(comb, rows, hc * hc) || post.dtype != RAD_F32 || comb.dtype != RAD_F32)
        return RAD_E_SHAPE;
#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        float coeff[80];
        const int64_t hb = rowoff(&h, r, width), hs = laststride(&h);
        float sq = 0;
        for (int64_t i = 0; i < width; ++i) {
            const float v = ldt(&h, hb + i * hs);
            sq += v * v;
        }
        const float inv = 1.0f / std::sqrt(sq / (float)width + eps);
        for (int64_t j = 0; j < mix; ++j) {
            float dot = 0;
            const int64_t fb = rowoff(&fn, j, width), fs = laststride(&fn);
            for (int64_t i = 0; i < width; ++i) dot += ldt(&h, hb + i * hs) * ldt(&fn, fb + i * fs);
            coeff[j] = dot * inv;
        }
        finish_row(coeff, h, scale, base, nullptr, post, comb, r, n, hc, iters, heps, &pre);
    }
    return RAD_OK;
}

/* x[i] = sum over the hc streams of pre[s] * h[s, i], accumulated in f32 and rounded once. */
int ref_mhc_collapse(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const int64_t n = p_int(a, "n", 0), hc = p_int(a, "hc", 0);
    if (n <= 0 || hc < 2 || hc > 8) return RAD_E_SHAPE;
    const auto &h = a->t[0], &pre = a->t[1], &x = a->t[2];
    if (h.rank != 2 || h.shape[0] < 1 || !matrix(h, h.shape[0], hc * n) ||
        !matrix(pre, h.shape[0], hc) || pre.dtype != RAD_F32 || !matrix(x, h.shape[0], n))
        return RAD_E_SHAPE;
#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < h.shape[0]; ++r)
        for (int64_t i = 0; i < n; ++i) {
            float v = 0;
            for (int64_t j = 0; j < hc; ++j)
                v += ldt(&pre, rowoff(&pre, r, hc) + j * laststride(&pre)) *
                     ldt(&h, rowoff(&h, r, hc * n) + (j * n + i) * laststride(&h));
            stt(&x, rowoff(&x, r, n) + i * laststride(&x), v);
        }
    return RAD_OK;
}

/* ERNIE's unfused, single-rank residual read. The operand order matches mhc_pre, but the read
 * differs in three ways the GLM reference does not have: the projection is scaled by
 * 1 / (sqrt(mean(h*h)) + eps) with the epsilon outside the root, the pre gate adds its own epsilon
 * while the mixing matrix Sinkhorns with a separate one, and the post gate carries a learned
 * multiplier instead of the fixed factor of two. The collapsed output omits the optional weighted
 * RMSNorm. comb stays indexed [source, destination]. */
namespace {
bool ernie_params(const RadArgs* a, int64_t* M, int64_t* n, int64_t* hc, int64_t* iters, float* eps,
                  float* pre_eps, float* sinkhorn_eps, float* post_mult) {
    *M = p_int(a, "M", 0);
    *n = p_int(a, "n", 0);
    *hc = p_int(a, "hc", 0);
    *iters = p_int(a, "iterations", 0);
    *eps = p_f32(a, "eps", 0);
    *pre_eps = p_f32(a, "pre_eps", 0);
    *sinkhorn_eps = p_f32(a, "sinkhorn_eps", 0);
    *post_mult = p_f32(a, "post_mult", 0);
    return *M > 0 && *n > 0 && *hc >= 2 && *hc <= 8 && *iters >= 1 && *iters <= 256 &&
           std::isfinite(*eps) && *eps > 0 && std::isfinite(*pre_eps) && *pre_eps >= 0 &&
           std::isfinite(*sinkhorn_eps) && *sinkhorn_eps >= 0 && std::isfinite(*post_mult) &&
           *post_mult > 0;
}

/* Every stride this op walks must be positive, or rowoff would land on a row it has already
 * written. The rank is checked first so a malformed descriptor cannot walk past the fixed-size
 * stride array (the same guard `router_pos_strides` applies in ref_moe.cpp). */
bool ernie_strides(const RadTensor& t) {
    if (t.rank > (uint32_t)RAD_MAX_RANK) return false; /* never walk past the stride array */
    for (uint32_t i = 0; i < t.rank; ++i)
        if (t.stride[i] <= 0) return false;
    return true;
}

/* Row r from its RMS-scaled coefficients: pre/post gates, Sinkhorn mixing and the collapse of h. */
void ernie_row(const float* coeff, const RadTensor& h, const RadTensor& scale,
               const RadTensor& base, const RadTensor& x, const RadTensor& post,
               const RadTensor& comb, int64_t r, int64_t n, int64_t hc, int64_t iters,
               float pre_eps, float sinkhorn_eps, float post_mult,
               const RadTensor* gain = nullptr, float rms_eps = 0, bool chunked = false, bool vec8 = false) {
    float pre[8], c[64];
    const int64_t ss = laststride(&scale), bs = laststride(&base);
    const float s0 = ldt(&scale, 0), s1 = ldt(&scale, ss), s2 = ldt(&scale, 2 * ss);
    for (int64_t j = 0; j < hc; ++j) {
        pre[j] = act_sigmoid(coeff[j] * s0 + ldt(&base, j * bs)) + pre_eps;
        const float z = coeff[hc + j] * s1 + ldt(&base, (hc + j) * bs);
        stt(&post, rowoff(&post, r, hc) + j * laststride(&post), act_sigmoid(z) * post_mult);
    }
    for (int64_t j = 0; j < hc; ++j) {
        float mx = -INFINITY;
        for (int64_t k = 0; k < hc; ++k) {
            const int64_t q = 2 * hc + j * hc + k;
            mx = std::max(mx, coeff[q] * s2 + ldt(&base, q * bs));
        }
        float sum = 0;
        for (int64_t k = 0; k < hc; ++k) {
            const int64_t q = 2 * hc + j * hc + k;
            c[j * hc + k] = ref_exp(coeff[q] * s2 + ldt(&base, q * bs) - mx);
            sum += c[j * hc + k];
        }
        for (int64_t k = 0; k < hc; ++k) c[j * hc + k] = c[j * hc + k] / sum + sinkhorn_eps;
    }
    for (int64_t it = 0; it < iters; ++it) {
        if (it)
            for (int64_t j = 0; j < hc; ++j) {
                float sum = 0;
                for (int64_t k = 0; k < hc; ++k) sum += c[j * hc + k];
                for (int64_t k = 0; k < hc; ++k) c[j * hc + k] /= sum + sinkhorn_eps;
            }
        for (int64_t k = 0; k < hc; ++k) {
            float sum = 0;
            for (int64_t j = 0; j < hc; ++j) sum += c[j * hc + k];
            for (int64_t j = 0; j < hc; ++j) c[j * hc + k] /= sum + sinkhorn_eps;
        }
    }
    const int64_t cb = rowoff(&comb, r, hc * hc), cs = laststride(&comb);
    for (int64_t j = 0; j < hc * hc; ++j) stt(&comb, cb + j * cs, c[j]);
    const int64_t hb = rowoff(&h, r, hc * n), hs = laststride(&h);
    const int64_t xb = rowoff(&x, r, n), xs = laststride(&x);
    const auto collapse = [&](int64_t i) {
        float v = 0;
        for (int64_t j = 0; j < hc; ++j) v += pre[j] * ldt(&h, hb + (j * n + i) * hs);
        return v;
    };
    if (!gain) {
        for (int64_t i = 0; i < n; ++i) stt(&x, xb + i * xs, collapse(i));
        return;
    }
    /* Fused weighted RMSNorm over the collapse. The variance reads the collapse as the f32 sum
     * left it; the numerator reads it back after the bf16 narrowing a stored collapse would have
     * had, times the bf16 gain; the product is rounded once on the way out. */
    float sq = 0;
    if (chunked) {
        sq = mhc_rms_sumsq(n, vec8, collapse);
    } else {
        for (int64_t i = 0; i < n; ++i) {
            const float v = collapse(i);
            sq += v * v;
        }
    }
    const float inv = 1.0f / std::sqrt(sq / (float)n + rms_eps);
    const int64_t gs = laststride(gain);
    for (int64_t i = 0; i < n; ++i) {
        const float numerator = round_dt(RAD_BF16, collapse(i)) * ldt(gain, i * gs);
        stt(&x, xb + i * xs, numerator * inv);
    }
}
}  /* namespace */

namespace {
/* ernie_mhc_pre and ernie_mhc_pre_rms share every check and every coefficient; the fused form has
 * the extra gain operand (before the outputs), the rms_eps parameter, and is bf16-only. */
int ernie_pre_impl(const RadArgs* a, bool fused) {
    const int off = fused ? 1 : 0; /* the gain sits between base and x */
    if (!have(a, 7 + off)) return RAD_E_INVAL;
    int64_t M, n, hc, iters;
    float eps, pre_eps, sinkhorn_eps, post_mult;
    if (!ernie_params(a, &M, &n, &hc, &iters, &eps, &pre_eps, &sinkhorn_eps, &post_mult))
        return RAD_E_SHAPE;
    const auto &h = a->t[0], &fn = a->t[1], &scale = a->t[2], &base = a->t[3];
    const auto &x = a->t[4 + off], &post = a->t[5 + off], &comb = a->t[6 + off];
    const RadTensor* gain = fused ? &a->t[4] : nullptr;
    float rms_eps = 0;
    if (fused) {
        rms_eps = p_f32(a, "rms_eps", 0);
        if (!std::isfinite(rms_eps) || rms_eps <= 0) return RAD_E_SHAPE;
    }
    if (!h.data || !fn.data || !scale.data || !base.data || !x.data || !post.data || !comb.data ||
        (gain && !gain->data))
        return RAD_E_SHAPE;
    if (fused && (h.dtype != RAD_BF16 || gain->dtype != RAD_BF16 || !ernie_strides(*gain) ||
                  gain->rank != 1 || numel(gain) != n))
        return RAD_E_SHAPE;
    if ((h.dtype != RAD_F32 && h.dtype != RAD_BF16) || x.dtype != h.dtype || fn.dtype != RAD_F32 ||
        scale.dtype != RAD_F32 || base.dtype != RAD_F32 || post.dtype != RAD_F32 ||
        comb.dtype != RAD_F32)
        return RAD_E_SHAPE;
    if (!ernie_strides(h) || !ernie_strides(fn) || !ernie_strides(scale) || !ernie_strides(base) ||
        !ernie_strides(x) || !ernie_strides(post) || !ernie_strides(comb))
        return RAD_E_SHAPE;
    int64_t width, mix;
    if (__builtin_mul_overflow(hc, n, &width) || width <= 0 ||
        __builtin_mul_overflow(hc, hc + 2, &mix))
        return RAD_E_SHAPE;
    if (h.rank != 2) return RAD_E_SHAPE;
    /* `M` is the row band's ceiling, not this call's row count: the live rows come from the
     * rank-two residual and are banded by it (see band_rows in ref_common.h), so a decode step reads
     * the few rows it carries out of band-sized buffers, while a call above the ceiling -- resolved
     * against constraints it does not satisfy -- is refused here, before h is read and before x,
     * post and comb are written. The outputs carry those same rows; rows past them keep padding. */
    const int64_t rows = band_rows(h.shape[0], M);
    if (rows < 1 || rows > INT64_MAX / width || !matrix(h, rows, width) || !matrix(fn, mix, width) ||
        scale.rank != 1 || numel(&scale) != 3 || base.rank != 1 || numel(&base) != mix ||
        !matrix(x, rows, n) || !matrix(post, rows, hc) || !matrix(comb, rows, hc * hc))
        return RAD_E_SHAPE;
    /* the device's form for this call (mhc_device_form) */
    const MhcDeviceForm form = mhc_device_form(h, fn, scale, base, x, post, comb, gain, n, hc);
    const bool chunked = form.chunked;
#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        float coeff[80];
        const int64_t hb = rowoff(&h, r, width), hs = laststride(&h);
        float sq = 0;
        if (chunked) {
            sq = mhc_chunked_sum(width, form.vec4, [&](int64_t i) { const float v = ldt(&h, hb + i * hs); return v * v; });
        } else {
            for (int64_t i = 0; i < width; ++i) {
                const float v = ldt(&h, hb + i * hs);
                sq += v * v;
            }
        }
        const float inv = 1.0f / (std::sqrt(sq / (float)width) + eps);
        for (int64_t j = 0; j < mix; ++j) {
            float dot = 0;
            const int64_t fb = rowoff(&fn, j, width), fs = laststride(&fn);
            if (chunked) {
                dot = mhc_chunked_sum(width, form.vec4, [&](int64_t i) { return ldt(&h, hb + i * hs) * ldt(&fn, fb + i * fs); });
            } else {
                for (int64_t i = 0; i < width; ++i) dot += ldt(&h, hb + i * hs) * ldt(&fn, fb + i * fs);
            }
            coeff[j] = dot * inv;
        }
        /* the fused variance's lane step: this row's h and x and the gain 16-byte aligned, n % 8 == 0 (the finish kernel's vec8) */
        const bool vec8 = gain && (n & 7) == 0 && ((((uintptr_t)h.data + (uintptr_t)(hb * 2)) | ((uintptr_t)x.data + (uintptr_t)(rowoff(&x, r, n) * 2)) | (uintptr_t)gain->data) & 15u) == 0u;
        ernie_row(coeff, h, scale, base, x, post, comb, r, n, hc, iters, pre_eps, sinkhorn_eps,
                  post_mult, gain, rms_eps, chunked, vec8);
    }
    return RAD_OK;
}
}  /* namespace */

int ref_ernie_mhc_pre(const RadArgs* a, RadStream) { return ernie_pre_impl(a, false); }

/* ernie_mhc_pre followed by the weighted RMSNorm ERNIE's fused layer-norm branch applies to the
 * collapse (the ordinary branch; a first-layer residual that was broadcast into hc identical
 * streams is the same equation over that input, see docs/OPS.md). */
int ref_ernie_mhc_pre_rms(const RadArgs* a, RadStream) { return ernie_pre_impl(a, true); }

/* ERNIE's final residual read: the learned collapse of the hc streams into one, the head that
 * follows ernie_mhc_pre and its post updates. Each fn row is dotted with the whole hc*n residual,
 * scaled by rsqrt(mean(h*h) + eps) -- the epsilon INSIDE the root, as in the GLM mhc_pre and unlike
 * ernie_mhc_pre, which puts it outside -- then by the scalar
 * scale plus a per-stream bias, squashed to a sigmoid gate with gate_eps added after. The gates
 * weight each stream and the streams are summed into x with one BF16 rounding; the final RMSNorm
 * stays in the graph. This is the asymmetric, learned counterpart to mhc_contract's unweighted
 * mean. M is the dispatch band ceiling; the live row count comes from h, and x must match it. */
int ref_ernie_mhc_head(const RadArgs* a, RadStream) {
    if (!have(a, 5)) return RAD_E_INVAL;
    const int64_t M = p_int(a, "M", 0), n = p_int(a, "n", 0), hc = p_int(a, "hc", 0);
    const float eps = p_f32(a, "eps", 0), gate_eps = p_f32(a, "gate_eps", 0);
    if (M <= 0 || n <= 0 || hc < 2 || hc > 8 || !std::isfinite(eps) || eps <= 0 ||
        !std::isfinite(gate_eps) || gate_eps < 0)
        return RAD_E_SHAPE;
    const auto &h = a->t[0], &fn = a->t[1], &scale = a->t[2], &base = a->t[3], &x = a->t[4];
    if (!h.data || !fn.data || !scale.data || !base.data || !x.data) return RAD_E_SHAPE;
    if (h.dtype != RAD_BF16 || x.dtype != RAD_BF16 || fn.dtype != RAD_F32 ||
        scale.dtype != RAD_F32 || base.dtype != RAD_F32)
        return RAD_E_SHAPE;
    if (!ernie_strides(h) || !ernie_strides(fn) || !ernie_strides(scale) || !ernie_strides(base) ||
        !ernie_strides(x))
        return RAD_E_SHAPE;
    int64_t width;
    if (__builtin_mul_overflow(hc, n, &width) || width <= 0) return RAD_E_SHAPE;
    if (h.rank != 2 || fn.rank != 2) return RAD_E_SHAPE;
    const int64_t rows = h.shape[0];
    if (rows < 1 || rows > M || rows > INT64_MAX / width || !matrix(h, rows, width) ||
        !matrix(fn, hc, width) || scale.rank != 1 || numel(&scale) != 1 || base.rank != 1 ||
        numel(&base) != hc || !matrix(x, rows, n))
        return RAD_E_SHAPE;
    const float s = ldt(&scale, 0);
    const int64_t bs = laststride(&base);
#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t hb = rowoff(&h, r, width), hs = laststride(&h);
        float gate[8];
        float sq = 0;
        for (int64_t i = 0; i < width; ++i) {
            const float v = ldt(&h, hb + i * hs);
            sq += v * v;
        }
        const float norm = std::sqrt(sq / (float)width + eps);
        for (int64_t j = 0; j < hc; ++j) {
            float dot = 0;
            const int64_t fb = rowoff(&fn, j, width), fs = laststride(&fn);
            for (int64_t i = 0; i < width; ++i) dot += ldt(&h, hb + i * hs) * ldt(&fn, fb + i * fs);
            gate[j] = act_sigmoid(dot / norm * s + ldt(&base, j * bs)) + gate_eps;
        }
        const int64_t xb = rowoff(&x, r, n), xs = laststride(&x);
        for (int64_t i = 0; i < n; ++i) {
            float v = 0;
            for (int64_t j = 0; j < hc; ++j) v += gate[j] * ldt(&h, hb + (j * n + i) * hs);
            stt(&x, xb + i * xs, v);
        }
    }
    return RAD_OK;
}
