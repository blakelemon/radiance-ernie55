/* Rotary frequencies shared by the ERNIE host ops that rotate a trailing slice (stories 3-3b, 3-4a): the plain
 * table theta^(-2i / R) and the YaRN blend of the HCA layers (vLLM DeepseekScalingRotaryEmbedding as the V4 builder
 * configures it: magnitude exactly 1, correction range from the ORIGINAL length). Header-only, host only. */
#pragma once
#include "ref_common.h"
#include <cmath>
#include <vector>

namespace ref {

struct ErnieYarn {
    bool on = false;
    double factor = 0.0, beta_fast = 32.0, beta_slow = 1.0, orig = 0.0;
};

static inline double ernie_rope_f64(const RadArgs* a, const char* k, double dflt, bool* present) {
    const RadParam* p = p_find(a, k);
    *present = p != nullptr;
    if (!p) return dflt;
    return p->kind == RAD_P_F64 ? p->dval : (double)p_f32(a, k, (float)dflt);
}

/* Optional parameters yarn_factor (0 or absent: plain RoPE), yarn_beta_fast (32), yarn_beta_slow (1),
 * yarn_original_max (required with a factor). false = refuse: factor not in {0} U [1, inf), a non-finite or
 * non-positive beta or length, beta_fast < beta_slow, any yarn_* parameter present without yarn_factor (story 3-6). */
static inline bool ernie_yarn_args(const RadArgs* a, ErnieYarn* y) {
    bool hf, hb, hs, ho;
    const double f = ernie_rope_f64(a, "yarn_factor", 0.0, &hf), bf = ernie_rope_f64(a, "yarn_beta_fast", 32.0, &hb),
                 bs = ernie_rope_f64(a, "yarn_beta_slow", 1.0, &hs), og = ernie_rope_f64(a, "yarn_original_max", 0.0, &ho);
    *y = ErnieYarn();
    if (hf && f == 0.0) return true;
    if (!hf) return !(hb || hs || ho);   /* story 3-6: a YaRN parameter without its factor is a mistake, not plain RoPE */
    if (!(f >= 1.0) || !std::isfinite(f) || !(bf > 0.0) || !std::isfinite(bf) || !(bs > 0.0) || !std::isfinite(bs) || bf < bs ||
        !ho || !(og > 0.0) || !std::isfinite(og))
        return false;
    y->on = true; y->factor = f; y->beta_fast = bf; y->beta_slow = bs; y->orig = og;
    return true;
}

/* inv[i] for pair i of an R-wide rotary slice (R even). Plain: theta^(-2i / R). YaRN: interpolation 1 / (factor *
 * theta^(2i / R)) blended with extrapolation 1 / theta^(2i / R) by mask = 1 - ramp(i; low, high), low = max(floor(dim(beta_fast)), 0),
 * high = min(ceil(dim(beta_slow)), R - 1), dim(r) = R ln(orig / (2 pi r)) / (2 ln theta). */
static inline std::vector<double> ernie_rope_inv(double theta, int64_t R, const ErnieYarn& y) {
    std::vector<double> inv((size_t)(R / 2));
    if (!y.on) {
        for (int64_t i = 0; i < R / 2; ++i) inv[(size_t)i] = std::pow(theta, -2.0 * (double)i / (double)R);
        return inv;
    }
    const double kPi = 3.14159265358979323846;
    auto dim = [&](double rot) { return ((double)R * std::log(y.orig / (rot * 2.0 * kPi))) / (2.0 * std::log(theta)); };
    const double low = std::fmax(std::floor(dim(y.beta_fast)), 0.0);
    double high = std::fmin(std::ceil(dim(y.beta_slow)), (double)(R - 1));
    if (low == high) high += 0.001;
    for (int64_t i = 0; i < R / 2; ++i) {
        const double pf = std::pow(theta, (2.0 * (double)i) / (double)R), ext = 1.0 / pf, itp = 1.0 / (y.factor * pf);
        const double ramp = std::fmin(std::fmax(((double)i - low) / (high - low), 0.0), 1.0), mask = 1.0 - ramp;
        inv[(size_t)i] = itp * (1.0 - mask) + ext * mask;
    }
    return inv;
}

}  // namespace ref
