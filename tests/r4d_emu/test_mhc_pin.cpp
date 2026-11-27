/* tests/r4d_emu: ernie_mhc_pre_rms (libr4d/r4d_ernie_mhc.hip) against libref, with and without R4D_MHC_DECODE_ROWS.
 *
 * The mHC pre-collapse chunks its partial sums and then sums them in an order that depends on the step's row count in the OLD rule (decode form below 4 rows,
 * many-row form from 4). The producer kernel writes the partial sums in one layout and the finish kernel reads them in the same one through the `decode` argument
 * the launcher passes both; if a pin moved one without the other the result would be garbage, not a last-bit difference.
 *   argv[2] == "default": R4D_MHC_DECODE_ROWS unset (row identity: every step on the many-row form). libref carries the device's summation order since the
 *                         row-identity amendment (libref/ref_mhc.cpp, option B of the mHC libref proposal), so x must be EQUAL BIT FOR BIT to libref (post and comb bit for bit
 *                         up to the expf/exp last-bit difference, 1e-6 relative) at T = 1..9, 16 and 64, no output may be NaN or infinite, and a row at any T equals the same row at M = 1.
 *   argv[2] == "pinned":  R4D_MHC_DECODE_ROWS=4 (ctest sets it; the old rule, which the ERNIE plugin still exports when speculation is off): from 4 rows the many-row
 *                         form, the same code path as the default, so x is EQUAL BIT FOR BIT to libref and post/comb held to 1e-6 relative at T = 4, 5, 8; below 4 rows
 *                         the decode form sums in another order than libref's (the device's many-row order), so T = 1, 3 are held to x within 4 bf16 steps and post
 *                         and comb within 1e-3 relative (the bound of the different order, not of the many-row form).
 * argv[1] is the path of libref.so. (The KNOWN FAILURE of T = 5, 8 is gone: libref adopted the device order in 1b755bad.) */
#include "rad_plugin.h"
#include "r4d_plugin.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <random>
#include <vector>

extern "C" int r4d_rad_ernie_mhc_pre_rms(const RadArgs*, RadStream);
namespace {
RadParam I(const char* k, int64_t v) { RadParam p{}; p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }
RadParam F(const char* k, double v) { RadParam p{}; p.key = k; p.kind = RAD_P_F64; p.dval = v; return p; }
RadTensor Mx(void* d, uint32_t dt, int64_t r, int64_t c) { RadTensor t{}; t.data = d; t.dtype = dt; t.rank = 2; t.shape[0] = r; t.shape[1] = c; t.stride[0] = c; t.stride[1] = 1; return t; }
RadTensor V(void* d, uint32_t dt, int64_t n) { RadTensor t{}; t.data = d; t.dtype = dt; t.rank = 1; t.shape[0] = n; t.stride[0] = 1; return t; }
RadLaunchFn find(void* lib, const char* op) {
    auto cnt = reinterpret_cast<int (*)()>(dlsym(lib, "rad_kernel_count"));
    auto at = reinterpret_cast<const RadKernelInfo* (*)(int)>(dlsym(lib, "rad_kernel_at"));
    for (int i = 0; cnt && i < cnt(); ++i) { const RadKernelInfo* k = at(i); if (k && k->op && !std::strcmp(k->op, op)) return k->launch; }
    return nullptr;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    const bool pinned = !std::strcmp(argv[2], "pinned");
    const char* env = std::getenv("R4D_MHC_DECODE_ROWS");
    if (pinned != (env && *env)) { std::fprintf(stderr, "mode %s but R4D_MHC_DECODE_ROWS is %s\n", argv[2], env ? env : "unset"); return 2; }
    void* lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) { std::fprintf(stderr, "%s\n", dlerror()); return 2; }
    RadLaunchFn ref = find(lib, "ernie_mhc_pre_rms");
    if (!ref) { std::fprintf(stderr, "libref lacks ernie_mhc_pre_rms\n"); return 2; }
    constexpr int n = 4096, hc = 4, width = n * hc, mix = hc * (hc + 2), MAXROWS = 64;
    std::mt19937 g(2);
    auto fr = [&](float s) { return s * ((int)(g() % 20001) - 10000) / 10000.0f; };
    std::vector<uint16_t> h((size_t)MAXROWS * width), gain(n);
    std::vector<float> fn((size_t)mix * width), scale = {0.5f, 0.4f, 0.3f}, base(mix);
    for (auto& v : h) v = rad_f32_to_bf16(fr(3.f));
    for (auto& v : gain) v = rad_f32_to_bf16(1.f + fr(.1f));
    for (auto& v : fn) v = fr(.02f);
    for (auto& v : base) v = fr(.5f);
    struct Out { std::vector<uint16_t> x; std::vector<float> post, comb; };
    auto run = [&](RadLaunchFn f, int first, int T, Out& o) {
        o.x.assign((size_t)T * n, 0xdead); o.post.assign((size_t)T * hc, -1.f); o.comb.assign((size_t)T * hc * hc, -1.f);
        RadTensor t[8] = {Mx(&h[(size_t)first * width], RAD_BF16, T, width), Mx(fn.data(), RAD_F32, mix, width), V(scale.data(), RAD_F32, 3), V(base.data(), RAD_F32, mix),
                          V(gain.data(), RAD_BF16, n), Mx(o.x.data(), RAD_BF16, T, n), Mx(o.post.data(), RAD_F32, T, hc), Mx(o.comb.data(), RAD_F32, T, hc * hc)};
        RadParam p[] = {I("M", MAXROWS), I("n", n), I("hc", hc), I("iterations", 20), F("eps", 1e-6), F("pre_eps", 1e-6), F("sinkhorn_eps", 1e-6), F("post_mult", 2.0), F("rms_eps", 1e-6)};
        RadArgs a{}; a.t = t; a.n_t = 8; a.p = p; a.n_p = 9;
        return f(&a, nullptr);
    };
    int bad = 0;
    auto finite = [](const Out& o) {
        for (uint16_t v : o.x) if ((v & 0x7f80u) == 0x7f80u) return false;
        for (float v : o.post) if (!std::isfinite(v)) return false;
        for (float v : o.comb) if (!std::isfinite(v)) return false;
        return true;
    };
    const std::vector<int> sizes = pinned ? std::vector<int>{1, 3, 4, 5, 8} : std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 64};
    for (int T : sizes) {
        Out d, r;
        const int rd = run(r4d_rad_ernie_mhc_pre_rms, 0, T, d), rr = run(ref, 0, T, r);
        bool ok = rd == 0 && rr == 0 && finite(d) && finite(r);
        long wx = 0, pc = 0; double wp = 0;
        if (pinned && T < 4) {
            for (size_t i = 0; ok && i < d.x.size(); ++i) { const long dd = std::abs((int)d.x[i] - (int)r.x[i]); if (dd > wx) wx = dd; if (dd > 4) ok = false; }
            for (size_t i = 0; ok && i < d.post.size(); ++i) { const double e = std::fabs(d.post[i] - r.post[i]) / (1e-6 + std::fabs(r.post[i])); if (e > wp) wp = e; if (e > 1e-3) ok = false; }
            for (size_t i = 0; ok && i < d.comb.size(); ++i) { const double e = std::fabs(d.comb[i] - r.comb[i]) / (1e-6 + std::fabs(r.comb[i])); if (e > wp) wp = e; if (e > 1e-3) ok = false; }
        } else if (ok) {
            for (size_t i = 0; i < d.x.size(); ++i) { const long dd = std::abs((int)d.x[i] - (int)r.x[i]); if (dd > wx) wx = dd; }
            for (size_t i = 0; i < d.post.size(); ++i) { const double e = std::fabs(d.post[i] - r.post[i]) / (1e-6 + std::fabs(r.post[i])); if (e > wp) wp = e; }
            for (size_t i = 0; i < d.comb.size(); ++i) { const double e = std::fabs(d.comb[i] - r.comb[i]) / (1e-6 + std::fabs(r.comb[i])); if (e > wp) wp = e; }
            /* x is bit for bit. post and comb are bit for bit except that the device calls expf and libref (double exp rounded to f32) can differ by one
             * ulp of an exp, which the Sinkhorn passes carry to a few ulps: held to 1e-6 relative, and the count of differing values is printed. */
            ok = !std::memcmp(d.x.data(), r.x.data(), d.x.size() * 2) && wp <= 1e-6;   /* also the pinned arm from 4 rows: the many-row form either way */
            for (size_t i = 0; i < d.post.size(); ++i) pc += std::memcmp(&d.post[i], &r.post[i], 4) != 0;
            for (size_t i = 0; i < d.comb.size(); ++i) pc += std::memcmp(&d.comb[i], &r.comb[i], 4) != 0;
        }
        std::printf("%s %s T=%d vs libref%s (rc %d/%d, x max %ld bf16 steps, post/comb max rel %.2g, %ld post/comb values not bit-equal)\n", ok ? "PASS" : "FAIL",
                    pinned ? "pinned " : "default", T, pinned ? "" : " (x bit for bit)", rd, rr, wx, wp, pc);
        if (!ok) ++bad;
    }
    if (!pinned) {
        std::vector<Out> alone(MAXROWS);
        for (int r = 0; r < MAXROWS; ++r) if (run(r4d_rad_ernie_mhc_pre_rms, r, 1, alone[r])) { std::printf("FAIL one-row launch refused\n"); return 1; }
        for (int T : {2, 3, 4, 5, 8, 9, 16, 64}) {
            Out d;
            if (run(r4d_rad_ernie_mhc_pre_rms, 0, T, d)) { std::printf("FAIL T=%d refused\n", T); ++bad; continue; }
            bool same = true;
            for (int r = 0; r < T && same; ++r)
                same = !std::memcmp(&d.x[(size_t)r * n], alone[r].x.data(), n * 2) && !std::memcmp(&d.post[(size_t)r * hc], alone[r].post.data(), hc * 4) &&
                       !std::memcmp(&d.comb[(size_t)r * hc * hc], alone[r].comb.data(), hc * hc * 4);
            std::printf("%s default T=%d: every row bit-identical to the same row at M=1\n", same ? "PASS" : "FAIL", T);
            if (!same) ++bad;
        }
    }
    std::printf("%d failure(s)\n", bad);
    dlclose(lib);
    return bad ? 1 : 0;
}
