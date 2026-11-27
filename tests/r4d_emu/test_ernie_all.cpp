/* tests/r4d_emu/test_ernie_all.cpp -- Comprehensive SIMT-emulated verification of all 19 ERNIE 5.5 Flash
 * device kernels (stories 4-3c, 4-3d, 4-3e, 4-3f, 4-3g) head-to-head against libref.so.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <map>
#include <random>
#include <string>
#include <vector>

static uint16_t f2b(float f) { return rad_f32_to_bf16(f); }
static float b2f(uint16_t h) { return rad_bf16_to_f32(h); }

static RadParam pi(const char* k, int64_t v) { RadParam p{}; p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }
static RadParam pf(const char* k, double v) { RadParam p{}; p.key = k; p.kind = RAD_P_F64; p.dval = v; return p; }
static RadParam ps(const char* k, const char* v) { RadParam p{}; p.key = k; p.kind = RAD_P_STR; p.sval = v; return p; }

static RadTensor t1(void* d, uint32_t dt, int64_t n) {
    RadTensor t{}; t.data = d; t.dtype = dt; t.rank = 1; t.shape[0] = n; t.stride[0] = 1; return t;
}
static RadTensor t2(void* d, uint32_t dt, int64_t r, int64_t c, int64_t ld = 0) {
    RadTensor t{}; t.data = d; t.dtype = dt; t.rank = 2; t.shape[0] = r; t.shape[1] = c;
    t.stride[0] = ld ? ld : c; t.stride[1] = 1; return t;
}
static RadTensor t4(void* d, uint32_t dt, int64_t d0, int64_t d1, int64_t d2, int64_t d3) {
    RadTensor t{}; t.data = d; t.dtype = dt; t.rank = 4;
    t.shape[0] = d0; t.shape[1] = d1; t.shape[2] = d2; t.shape[3] = d3;
    t.stride[3] = 1; t.stride[2] = d3; t.stride[1] = d2 * d3; t.stride[0] = d1 * d2 * d3;
    return t;
}

static std::map<std::string, RadLaunchFn> g_ref;

static bool load_libref(const char* path) {
    void* h = dlopen(path, RTLD_NOW);
    if (!h) { std::fprintf(stderr, "dlopen(%s) failed: %s\n", path, dlerror()); return false; }
    auto cnt_fn = (int(*)(void))dlsym(h, "rad_kernel_count");
    auto at_fn  = (const RadKernelInfo*(*)(int))dlsym(h, "rad_kernel_at");
    if (!cnt_fn || !at_fn) { std::fprintf(stderr, "missing rad_kernel_count/at\n"); return false; }
    const int n = cnt_fn();
    for (int i = 0; i < n; ++i) {
        const RadKernelInfo* k = at_fn(i);
        if (k && k->op && k->launch && !g_ref.count(k->op)) g_ref[k->op] = k->launch;
    }
    return true;
}

static int g_pass = 0, g_fail = 0;

static void check_f32(const char* name, const std::vector<float>& a, const std::vector<float>& b, float tol = 1e-4f) {
    float mx = 0.0f; int bad = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::isinf(a[i]) && std::isinf(b[i]) && std::signbit(a[i]) == std::signbit(b[i])) continue;
        float d = std::fabs(a[i] - b[i]);
        if (std::isnan(d) || d > tol * (1.0f + std::fabs(b[i]))) ++bad;
        if (d > mx) mx = d;
    }
    if (bad) { std::printf("FAIL %-28s bad=%d/%zu max_abs=%.3e\n", name, bad, a.size(), mx); ++g_fail; }
    else     { std::printf("PASS %-28s max_abs=%.3e\n", name, mx); ++g_pass; }
}

static void check_bf16(const char* name, const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, float tol = 2e-2f, int max_ulp = 2) {
    float mx = 0.0f; int bad = 0, exact = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        float fa = b2f(a[i]), fb = b2f(b[i]);
        if (!std::isfinite(fa) || !std::isfinite(fb)) { ++bad; continue; }
        if (a[i] == b[i]) { ++exact; continue; }
        float d = std::fabs(fa - fb);
        int ulp = std::abs((int)a[i] - (int)b[i]);
        if (d > mx) mx = d;
        if (std::isnan(d) || (ulp > max_ulp && d > tol * (1.0f + std::fabs(fb)))) ++bad;
    }
    if (bad) { std::printf("FAIL %-28s bad=%d/%zu exact=%d/%zu max_abs=%.3e\n", name, bad, a.size(), exact, a.size(), mx); ++g_fail; }
    else     { std::printf("PASS %-28s exact=%d/%zu max_abs=%.3e\n", name, exact, a.size(), mx); ++g_pass; }
}

template <typename T>
static void check_exact(const char* name, const std::vector<T>& a, const std::vector<T>& b) {
    int bad = 0;
    for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) ++bad;
    if (bad) { std::printf("FAIL %-28s mismatch=%d/%zu\n", name, bad, a.size()); ++g_fail; }
    else     { std::printf("PASS %-28s exact=%zu/%zu\n", name, a.size(), a.size()); ++g_pass; }
}

int main(int argc, char** argv) {
    const char* ref_path = (argc > 1) ? argv[1] : "/src/build-ernie55/libref.so";
    if (!load_libref(ref_path)) return 2;
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    /* 1. ernie_multimax */
    {
        const int M = 4, n = 513;
        std::vector<float> x(M * n), rg = {-2.5f, 2.5f, -5.0f, 5.0f}, cf = {0.1f, -0.2f, 0.05f, -0.03f}, yd(M * n), yr(M * n);
        for (auto& v : x) v = nd(rng) * 4.0f;
        RadTensor td[4] = {t2(x.data(), RAD_F32, M, n), t1(rg.data(), RAD_F32, 4), t1(cf.data(), RAD_F32, 4), t2(yd.data(), RAD_F32, M, n)};
        RadTensor tr[4] = {td[0], td[1], td[2], t2(yr.data(), RAD_F32, M, n)};
        RadParam p[3] = {pi("M", M + 2), pi("n", n), ps("dtype", "f32")};
        RadArgs ad{td, 4, p, 3}, ar{tr, 4, p, 3};
        if (r4d_rad_ernie_multimax(&ad, nullptr) != RAD_OK || g_ref["ernie_multimax"](&ar, nullptr) != RAD_OK) ++g_fail;
        else check_f32("ernie_multimax", yd, yr, 1e-6f);
    }

    /* 2. ernie_row_req */
    {
        std::vector<int32_t> cu = {0, 0, 3, 7, 7, 12}, rd(12), rr(12);
        RadTensor td[2] = {t1(cu.data(), RAD_I32, 6), t1(rd.data(), RAD_I32, 12)};
        RadTensor tr[2] = {td[0], t1(rr.data(), RAD_I32, 12)};
        RadParam p[1] = {pi("M", 12)};
        RadArgs ad{td, 2, p, 1}, ar{tr, 2, p, 1};
        if (r4d_rad_ernie_row_req(&ad, nullptr) != RAD_OK || g_ref["ernie_row_req"](&ar, nullptr) != RAD_OK) ++g_fail;
        else check_exact("ernie_row_req", rd, rr);
    }

    /* 3. ernie_head_slice */
    {
        const int M = 3, H = 8, W = 192, off = 128, cnt = 64;
        std::vector<uint16_t> x(M * H * W), yd(M * H * cnt), yr(M * H * cnt);
        for (auto& v : x) v = f2b(nd(rng));
        RadTensor td[2] = {t2(x.data(), RAD_BF16, M, H * W), t2(yd.data(), RAD_BF16, M, H * cnt)};
        RadTensor tr[2] = {td[0], t2(yr.data(), RAD_BF16, M, H * cnt)};
        RadParam p[5] = {pi("M", M), pi("heads", H), pi("width", W), pi("offset", off), pi("count", cnt)};
        RadArgs ad{td, 2, p, 5}, ar{tr, 2, p, 5};
        if (r4d_rad_ernie_head_slice(&ad, nullptr) != RAD_OK || g_ref["ernie_head_slice"](&ar, nullptr) != RAD_OK) ++g_fail;
        else check_exact("ernie_head_slice", yd, yr);
    }

    /* 4. ernie_rope_tail (plain bf16 table + YaRN inverse f32 table) */
    for (int mode : {0, 1}) {
        const int M = 3, H = 8, W = 128, R = 64;
        std::vector<uint16_t> xd(M * H * W), xr(M * H * W);
        for (auto& v : xd) v = f2b(nd(rng));
        xr = xd;
        std::vector<int32_t> pos = {0, 17, 513};
        RadTensor td[2] = {t2(xd.data(), RAD_BF16, M, H * W), t1(pos.data(), RAD_I32, M)};
        RadTensor tr[2] = {t2(xr.data(), RAD_BF16, M, H * W), td[1]};
        std::vector<RadParam> p = {pi("M", M), pi("heads", H), pi("width", W), pi("rope_dim", R), pf("theta", 10000.0)};
        if (mode == 0) { p.push_back(ps("rope_table", "bf16")); }
        else { p.push_back(ps("rope_table", "f32")); p.push_back(pi("inverse", 1)); p.push_back(pf("yarn_factor", 4.0)); p.push_back(pf("yarn_original_max", 32768.0)); }
        RadArgs ad{td, 2, p.data(), (int)p.size()}, ar{tr, 2, p.data(), (int)p.size()};
        if (r4d_rad_ernie_rope_tail(&ad, nullptr) != RAD_OK || g_ref["ernie_rope_tail"](&ar, nullptr) != RAD_OK) ++g_fail;
        else check_bf16(mode == 0 ? "ernie_rope_tail(bf16)" : "ernie_rope_tail(yarn,inv)", xd, xr, 1e-3f, 1);
    }

    /* 5. ernie_hca_q_prep */
    {
        const int M = 3, H = 8, D = 128, R = 64;
        std::vector<uint16_t> xd(M * H * D), xr(M * H * D);
        for (auto& v : xd) v = f2b(nd(rng));
        xr = xd;
        std::vector<int32_t> pos = {5, 129, 1024};
        RadTensor td[2] = {t2(xd.data(), RAD_BF16, M, H * D), t1(pos.data(), RAD_I32, M)};
        RadTensor tr[2] = {t2(xr.data(), RAD_BF16, M, H * D), td[1]};
        RadParam p[8] = {pi("M", M), pi("heads", H), pi("head_dim", D), pi("rope_dim", R), pf("eps", 1e-6), pf("theta", 10000.0), pf("yarn_factor", 4.0), pf("yarn_original_max", 32768.0)};
        RadArgs ad{td, 2, p, 8}, ar{tr, 2, p, 8};
        if (r4d_rad_ernie_hca_q_prep(&ad, nullptr) != RAD_OK || g_ref["ernie_hca_q_prep"](&ar, nullptr) != RAD_OK) ++g_fail;
        else check_bf16("ernie_hca_q_prep(yarn)", xd, xr, 2e-3f, 1);
    }

    /* 6. ernie_vha_mix (f32 and bf16 fold) */
    for (int fbf : {0, 1}) {
        const int M = 3, H = 16, D = 128, R = 8;
        std::vector<uint16_t> o(M * H * D), u(H * R), v(H * R), yd(M * H * D), yr(M * H * D);
        for (auto& e : o) e = f2b(nd(rng));
        for (auto& e : u) e = f2b(nd(rng) * 0.2f);
        for (auto& e : v) e = f2b(nd(rng) * 0.2f);
        RadTensor td[4] = {t2(o.data(), RAD_BF16, M, H * D), t2(u.data(), RAD_BF16, H, R), t2(v.data(), RAD_BF16, H, R), t2(yd.data(), RAD_BF16, M, H * D)};
        RadTensor tr[4] = {td[0], td[1], td[2], t2(yr.data(), RAD_BF16, M, H * D)};
        RadParam p[5] = {pi("M", M), pi("heads", H), pi("dim", D), pi("rank", R), ps("fold", fbf ? "bf16" : "f32")};
        RadArgs ad{td, 4, p, 5}, ar{tr, 4, p, 5};
        if (r4d_rad_ernie_vha_mix(&ad, nullptr) != RAD_OK || g_ref["ernie_vha_mix"](&ar, nullptr) != RAD_OK) ++g_fail;
        else check_bf16(fbf ? "ernie_vha_mix(bf16)" : "ernie_vha_mix(f32)", yd, yr, 2e-3f, 1);
    }
    /* 6b. ernie_vha_mix TP-sharded (world_size=4, tp_rank=0..3, bf16 fold): exact vs libref AND vs unsharded TP=1 */
    {
        const int M = 3, H = 16, W = 4, Hloc = H / W, D = 128, R = 8;
        std::vector<uint16_t> o_full(M * H * D), o_ag(W * M * Hloc * D), u(H * R), v(H * R), y_full(M * H * D);
        for (auto& e : o_full) e = f2b(nd(rng));
        for (auto& e : u) e = f2b(nd(rng) * 0.2f);
        for (auto& e : v) e = f2b(nd(rng) * 0.2f);
        for (int rk = 0; rk < W; ++rk)
            for (int m = 0; m < M; ++m)
                for (int lh = 0; lh < Hloc; ++lh)
                    for (int c = 0; c < D; ++c)
                        o_ag[((rk * M + m) * Hloc + lh) * D + c] = o_full[(m * H + rk * Hloc + lh) * D + c];
        RadTensor tf[4] = {t2(o_full.data(), RAD_BF16, M, H * D), t2(u.data(), RAD_BF16, H, R), t2(v.data(), RAD_BF16, H, R), t2(y_full.data(), RAD_BF16, M, H * D)};
        RadParam pf1[5] = {pi("M", M), pi("heads", H), pi("dim", D), pi("rank", R), ps("fold", "bf16")};
        RadArgs af{tf, 4, pf1, 5};
        if (g_ref["ernie_vha_mix"](&af, nullptr) != RAD_OK) ++g_fail;
        std::vector<uint16_t> y_cat_d(M * H * D), y_cat_r(M * H * D);
        for (int rk = 0; rk < W; ++rk) {
            std::vector<uint16_t> yd(M * Hloc * D), yr(M * Hloc * D);
            RadTensor td[4] = {t2(o_ag.data(), RAD_BF16, W * M, Hloc * D), tf[1], tf[2], t2(yd.data(), RAD_BF16, M, Hloc * D)};
            RadTensor tr[4] = {td[0], tf[1], tf[2], t2(yr.data(), RAD_BF16, M, Hloc * D)};
            RadParam pt[7] = {pi("M", M), pi("heads", H), pi("dim", D), pi("rank", R), ps("fold", "bf16"), pi("world_size", W), pi("tp_rank", rk)};
            RadArgs ad{td, 4, pt, 7}, ar{tr, 4, pt, 7};
            if (r4d_rad_ernie_vha_mix(&ad, nullptr) != RAD_OK || g_ref["ernie_vha_mix"](&ar, nullptr) != RAD_OK) ++g_fail;
            for (int m = 0; m < M; ++m)
                for (int lh = 0; lh < Hloc; ++lh)
                    for (int c = 0; c < D; ++c) {
                        y_cat_d[(m * H + rk * Hloc + lh) * D + c] = yd[(m * Hloc + lh) * D + c];
                        y_cat_r[(m * H + rk * Hloc + lh) * D + c] = yr[(m * Hloc + lh) * D + c];
                    }
        }
        check_bf16("ernie_vha_mix(tp4 vs ref)", y_cat_d, y_cat_r, 0.0f, 0);
        check_bf16("ernie_vha_mix(tp4 vs tp1)", y_cat_d, y_full, 0.0f, 0);
    }

    /* 7, 8, 9. ernie_mhc_pre, ernie_mhc_pre_rms, ernie_mhc_head */
    {
        const int M = 3, n = 256, hc = 4, W = hc * n, mix = hc * (hc + 2);
        std::vector<uint16_t> h(M * W), gain(n), xd(M * n), xr(M * n);
        std::vector<float> fn(mix * W), scale = {0.7f, 0.8f, 0.9f}, base(mix), postd(M * hc), postr(M * hc), combd(M * hc * hc), combr(M * hc * hc);
        for (auto& e : h) e = f2b(nd(rng));
        for (auto& e : gain) e = f2b(1.0f + 0.1f * nd(rng));
        for (auto& e : fn) e = nd(rng) * 0.05f;
        for (auto& e : base) e = nd(rng) * 0.3f;

        RadTensor td[7] = {t2(h.data(), RAD_BF16, M, W), t2(fn.data(), RAD_F32, mix, W), t1(scale.data(), RAD_F32, 3), t1(base.data(), RAD_F32, mix),
                           t2(xd.data(), RAD_BF16, M, n), t2(postd.data(), RAD_F32, M, hc), t2(combd.data(), RAD_F32, M, hc * hc)};
        RadTensor tr[7] = {td[0], td[1], td[2], td[3], t2(xr.data(), RAD_BF16, M, n), t2(postr.data(), RAD_F32, M, hc), t2(combr.data(), RAD_F32, M, hc * hc)};
        RadParam p[9] = {pi("M", M), pi("n", n), pi("hc", hc), pi("iterations", 20), pf("eps", 1e-6), pf("pre_eps", 1e-3), pf("sinkhorn_eps", 1e-6), pf("post_mult", 2.0), ps("dtype", "bf16")};
        RadArgs ad{td, 7, p, 9}, ar{tr, 7, p, 9};
        if (r4d_rad_ernie_mhc_pre(&ad, nullptr) != RAD_OK || g_ref["ernie_mhc_pre"](&ar, nullptr) != RAD_OK) ++g_fail;
        else { check_bf16("ernie_mhc_pre(x)", xd, xr); check_f32("ernie_mhc_pre(post)", postd, postr, 1e-5f); check_f32("ernie_mhc_pre(comb)", combd, combr, 1e-5f); }

        RadTensor tdr[8] = {td[0], td[1], td[2], td[3], t1(gain.data(), RAD_BF16, n), t2(xd.data(), RAD_BF16, M, n), t2(postd.data(), RAD_F32, M, hc), t2(combd.data(), RAD_F32, M, hc * hc)};
        RadTensor trr[8] = {td[0], td[1], td[2], td[3], tdr[4], t2(xr.data(), RAD_BF16, M, n), t2(postr.data(), RAD_F32, M, hc), t2(combr.data(), RAD_F32, M, hc * hc)};
        RadParam pr[10] = {p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], pf("rms_eps", 1e-6), ps("dtype", "bf16")};
        std::fill(postd.begin(), postd.end(), NAN); std::fill(postr.begin(), postr.end(), NAN);
        std::fill(combd.begin(), combd.end(), NAN); std::fill(combr.begin(), combr.end(), NAN);
        RadArgs adr{tdr, 8, pr, 10}, arr{trr, 8, pr, 10};
        if (r4d_rad_ernie_mhc_pre_rms(&adr, nullptr) != RAD_OK || g_ref["ernie_mhc_pre_rms"](&arr, nullptr) != RAD_OK) ++g_fail;
        else {
            check_bf16("ernie_mhc_pre_rms(x)", xd, xr);
            check_f32("ernie_mhc_pre_rms(post)", postd, postr, 1e-5f);
            check_f32("ernie_mhc_pre_rms(comb)", combd, combr, 1e-5f);
        }

        std::vector<float> fnh(hc * W), sch = {0.75f}, baseh(hc);
        for (auto& e : fnh) e = nd(rng) * 0.05f;
        for (auto& e : baseh) e = nd(rng) * 0.3f;
        RadTensor tdh[5] = {td[0], t2(fnh.data(), RAD_F32, hc, W), t1(sch.data(), RAD_F32, 1), t1(baseh.data(), RAD_F32, hc), t2(xd.data(), RAD_BF16, M, n)};
        RadTensor trh[5] = {tdh[0], tdh[1], tdh[2], tdh[3], t2(xr.data(), RAD_BF16, M, n)};
        RadParam ph[5] = {pi("M", M), pi("n", n), pi("hc", hc), pf("eps", 1e-6), pf("gate_eps", 1e-3)};
        RadArgs adh{tdh, 5, ph, 5}, arh{trh, 5, ph, 5};
        if (r4d_rad_ernie_mhc_head(&adh, nullptr) != RAD_OK || g_ref["ernie_mhc_head"](&arh, nullptr) != RAD_OK) ++g_fail;
        else check_bf16("ernie_mhc_head(x)", xd, xr);
    }

    /* 10, 11, 12. ernie_hca_state_write, ernie_hca_gather_slots, ernie_hca_compress */
    {
        const int M = 8, ratio = 4, D = 128, R = 64, coff = 2, W = coff * D, Wn = coff * ratio, S = 16;
        std::vector<float> kv(M * W), sc(M * W), ape(ratio * W), std_buf(S * 2 * W, 0.0f), str_buf(S * 2 * W, 0.0f);
        for (auto& e : kv) e = nd(rng);
        for (auto& e : sc) e = nd(rng) * 0.5f;
        for (auto& e : ape) e = nd(rng) * 0.2f;
        std::vector<int32_t> pos = {0, 1, 2, 3, 4, 5, 6, 7}, slots = {0, 1, 2, 3, 4, 5, 6, 7};
        RadTensor td[6] = {t2(kv.data(), RAD_F32, M, W), t2(sc.data(), RAD_F32, M, W), t2(ape.data(), RAD_F32, ratio, W),
                           t1(pos.data(), RAD_I32, M), t1(slots.data(), RAD_I32, M), t2(std_buf.data(), RAD_F32, S, 2 * W)};
        RadTensor tr[6] = {td[0], td[1], td[2], td[3], td[4], t2(str_buf.data(), RAD_F32, S, 2 * W)};
        RadParam p[3] = {pi("M", M), pi("ratio", ratio), pi("head_dim", D)};
        RadArgs ad{td, 6, p, 3}, ar{tr, 6, p, 3};
        if (r4d_rad_ernie_hca_state_write(&ad, nullptr) != RAD_OK || g_ref["ernie_hca_state_write"](&ar, nullptr) != RAD_OK) ++g_fail;
        else check_f32("ernie_hca_state_write", std_buf, str_buf, 1e-6f);

        std::vector<int32_t> req(M, 0), bt = {0, 1, 2, 3}, gsd(M * Wn, -9), gsr(M * Wn, -9);
        RadTensor tg_d[4] = {t1(pos.data(), RAD_I32, M), t1(req.data(), RAD_I32, M), t2(bt.data(), RAD_I32, 1, 4), t2(gsd.data(), RAD_I32, M, Wn)};
        RadTensor tg_r[4] = {tg_d[0], tg_d[1], tg_d[2], t2(gsr.data(), RAD_I32, M, Wn)};
        RadParam pg[3] = {pi("M", M), pi("ratio", ratio), pi("block_size", 4)};
        RadArgs agd{tg_d, 4, pg, 3}, agr{tg_r, 4, pg, 3};
        if (r4d_rad_ernie_hca_gather_slots(&agd, nullptr) != RAD_OK || g_ref["ernie_hca_gather_slots"](&agr, nullptr) != RAD_OK) ++g_fail;
        else check_exact("ernie_hca_gather_slots", gsd, gsr);

        std::vector<uint16_t> nw(D), od(M * D, 0), or_(M * D, 0);
        for (auto& e : nw) e = f2b(1.0f + 0.1f * nd(rng));
        std::vector<int32_t> oslots = {0, 1, 2, 3, 4, 5, 6, 7};
        RadTensor tc_d[6] = {t2(std_buf.data(), RAD_F32, S, 2 * W), t2(gsd.data(), RAD_I32, M, Wn), t1(pos.data(), RAD_I32, M),
                             t1(oslots.data(), RAD_I32, M), t1(nw.data(), RAD_BF16, D), t2(od.data(), RAD_BF16, M, D)};
        RadTensor tc_r[6] = {tc_d[0], tc_d[1], tc_d[2], tc_d[3], tc_d[4], t2(or_.data(), RAD_BF16, M, D)};
        RadParam pc[7] = {pi("M", M), pi("ratio", ratio), pi("head_dim", D), pi("rope_dim", R), pf("eps", 1e-6), pf("theta", 10000.0), ps("layout", "bf16_full")};
        RadArgs acd{tc_d, 6, pc, 7}, acr{tc_r, 6, pc, 7};
        if (r4d_rad_ernie_hca_compress(&acd, nullptr) != RAD_OK || g_ref["ernie_hca_compress"](&acr, nullptr) != RAD_OK) ++g_fail;
        else check_bf16("ernie_hca_compress(bf16)", od, or_, 2e-3f, 1);

        const int rb = (D - R) + 2 * R + (D - R) / 64 + 1;
        std::vector<uint8_t> pd(M * rb, 0), pr(M * rb, 0);
        RadTensor tcp_d[6] = {tc_d[0], tc_d[1], tc_d[2], tc_d[3], tc_d[4], t2(pd.data(), RAD_U8, M, rb)};
        RadTensor tcp_r[6] = {tc_d[0], tc_d[1], tc_d[2], tc_d[3], tc_d[4], t2(pr.data(), RAD_U8, M, rb)};
        RadParam pcp[8] = {pi("M", M), pi("ratio", ratio), pi("head_dim", D), pi("rope_dim", R), pf("eps", 1e-6), pf("theta", 10000.0), ps("layout", "fp8_ds_mla"), pi("quant_block", 64)};
        RadArgs acpd{tcp_d, 6, pcp, 8}, acpr{tcp_r, 6, pcp, 8};
        if (r4d_rad_ernie_hca_compress(&acpd, nullptr) != RAD_OK || g_ref["ernie_hca_compress"](&acpr, nullptr) != RAD_OK) ++g_fail;
        else check_exact("ernie_hca_compress(fp8)", pd, pr);
    }

    /* 13, 14. ernie_hca_union_attn (d=128 + token layout) & ernie_hca_wo_a */
    {
        const int M = 3, H = 8, D = 128, R = 64, ratio = 4, win = 8, bw = 4, bc = 4, Pw = 4, Pc = 4;
        std::vector<uint16_t> q(M * H * D), wk(Pw * bw * D), ck(Pc * bc * D), od(M * H * D), or_(M * H * D);
        for (auto& e : q) e = f2b(nd(rng) * 0.5f);
        for (auto& e : wk) e = f2b(nd(rng) * 0.5f);
        for (auto& e : ck) e = f2b(nd(rng) * 0.5f);
        std::vector<int32_t> pos = {0, 5, 11}, req = {0, 0, 0}, wt = {0, 1, 2, 3}, ct = {0, 1, 2, 3};
        std::vector<float> sink(H);
        for (int h = 0; h < H; ++h) sink[h] = (h == 0) ? -INFINITY : (0.5f * nd(rng));
        RadTensor td[9] = {t2(q.data(), RAD_BF16, M, H * D), t4(wk.data(), RAD_BF16, Pw, 1, bw, D), t4(ck.data(), RAD_BF16, Pc, 1, bc, D),
                           t1(pos.data(), RAD_I32, M), t1(req.data(), RAD_I32, M), t2(wt.data(), RAD_I32, 1, 4), t2(ct.data(), RAD_I32, 1, 4),
                           t1(sink.data(), RAD_F32, H), t2(od.data(), RAD_BF16, M, H * D)};
        RadTensor tr[9] = {td[0], td[1], td[2], td[3], td[4], td[5], td[6], td[7], t2(or_.data(), RAD_BF16, M, H * D)};
        RadParam p[7] = {pi("M", M), pi("heads", H), pi("head_dim", D), pi("ratio", ratio), pi("window", win), pf("scale", 1.0 / std::sqrt((double)D)), ps("cmp_layout", "token")};
        RadArgs ad{td, 9, p, 7}, ar{tr, 9, p, 7};
        if (r4d_rad_ernie_hca_union_attn(&ad, nullptr) != RAD_OK || g_ref["ernie_hca_union_attn"](&ar, nullptr) != RAD_OK) ++g_fail;
        else check_bf16("ernie_hca_union_attn", od, or_, 2e-3f, 1);

        const int G = 2, K = 32, per = (H / G) * D;
        std::vector<uint16_t> w(G * K * per), wod(M * G * K), wor(M * G * K);
        for (auto& e : w) e = f2b(nd(rng) * 0.05f);
        RadTensor twd[4] = {t2(od.data(), RAD_BF16, M, H * D), t1(pos.data(), RAD_I32, M), t2(w.data(), RAD_BF16, G * K, per), t2(wod.data(), RAD_BF16, M, G * K)};
        RadTensor twr[4] = {twd[0], twd[1], twd[2], t2(wor.data(), RAD_BF16, M, G * K)};
        RadParam pw[7] = {pi("M", M), pi("heads", H), pi("head_dim", D), pi("rope_dim", R), pi("groups", G), pi("rank", K), pf("theta", 10000.0)};
        RadArgs awd{twd, 4, pw, 7}, awr{twr, 4, pw, 7};
        if (r4d_rad_ernie_hca_wo_a(&awd, nullptr) != RAD_OK || g_ref["ernie_hca_wo_a"](&awr, nullptr) != RAD_OK) ++g_fail;
        else check_bf16("ernie_hca_wo_a", wod, wor, 2e-3f, 1);
    }

    /* 15, 16, 17, 18, 19. ernie_idx_query, ernie_idx_key, ernie_idx_logits, ernie_dsa_select, ernie_mla_project, ernie_mla_sparse_attn */
    {
        const int M = 4, H = 8, D = 128, R = 64, bs = 4, pages = 8, L = 32, win = 4, topk = 8;
        std::vector<uint16_t> q(M * H * D), k(M * D), gm(D), bt_ln(D), w(M * H);
        for (auto& e : q) e = f2b(nd(rng));
        for (auto& e : k) e = f2b(nd(rng));
        for (auto& e : gm) e = f2b(1.0f + 0.1f * nd(rng));
        for (auto& e : bt_ln) e = f2b(0.1f * nd(rng));
        for (auto& e : w) e = f2b(nd(rng) * 0.5f);
        std::vector<int32_t> pos = {0, 5, 15, 25}, slots = {0, 5, 15, 25};
        std::vector<uint8_t> qfd(M * H * D), qfr(M * H * D), cd(pages * bs * (D + 4), 0), cr(pages * bs * (D + 4), 0);
        std::vector<float> qsd(M * H), qsr(M * H);

        RadTensor tqd[4] = {t2(q.data(), RAD_BF16, M, H * D), t1(pos.data(), RAD_I32, M), t2(qfd.data(), RAD_F8E4M3, M, H * D), t2(qsd.data(), RAD_F32, M, H)};
        RadTensor tqr[4] = {tqd[0], tqd[1], t2(qfr.data(), RAD_F8E4M3, M, H * D), t2(qsr.data(), RAD_F32, M, H)};
        RadParam pq[7] = {pi("M", M), pi("heads", H), pi("head_dim", D), pi("rope_dim", R), pf("theta", 10000.0), ps("rope_table", "bf16"), ps("quant_rule", "nvidia")};
        RadArgs aqd{tqd, 4, pq, 7}, aqr{tqr, 4, pq, 7};
        if (r4d_rad_ernie_idx_query(&aqd, nullptr) != RAD_OK || g_ref["ernie_idx_query"](&aqr, nullptr) != RAD_OK) ++g_fail;
        else { check_exact("ernie_idx_query(qf)", qfd, qfr); check_f32("ernie_idx_query(qs)", qsd, qsr, 1e-7f); }

        RadTensor tkd[6] = {t2(k.data(), RAD_BF16, M, D), t1(gm.data(), RAD_BF16, D), t1(bt_ln.data(), RAD_BF16, D),
                            t1(pos.data(), RAD_I32, M), t1(slots.data(), RAD_I32, M), t4(cd.data(), RAD_U8, pages, 1, bs, D + 4)};
        RadTensor tkr[6] = {tkd[0], tkd[1], tkd[2], tkd[3], tkd[4], t4(cr.data(), RAD_U8, pages, 1, bs, D + 4)};
        RadParam pk[7] = {pi("M", M), pi("head_dim", D), pi("rope_dim", R), pf("theta", 10000.0), pf("eps", 1e-6), ps("rope_table", "bf16"), ps("quant_rule", "nvidia")};
        RadArgs akd{tkd, 6, pk, 7}, akr{tkr, 6, pk, 7};
        if (r4d_rad_ernie_idx_key(&akd, nullptr) != RAD_OK || g_ref["ernie_idx_key"](&akr, nullptr) != RAD_OK) ++g_fail;
        else check_exact("ernie_idx_key(cache)", cd, cr);

        /* Populate all cache slots for logits test */
        for (int s = 0; s < pages * bs; ++s) {
            for (int d = 0; d < D; ++d) cd[s * (D + 4) + d] = cr[s * (D + 4) + d] = rad_f32_to_fp8e4m3(nd(rng));
            float ks = 0.25f;
            std::memcpy(cd.data() + s * (D + 4) + D, &ks, 4);
            std::memcpy(cr.data() + s * (D + 4) + D, &ks, 4);
        }
        std::vector<int32_t> bt = {0, 1, 2, 3, 4, 5, 6, 7}, req(M, 0);
        std::vector<float> scd(M * L), scr(M * L);
        RadTensor tld[8] = {tqd[2], tqd[3], t2(w.data(), RAD_BF16, M, H), t4(cd.data(), RAD_U8, pages, 1, bs, D + 4),
                            t2(bt.data(), RAD_I32, 1, 8), t1(req.data(), RAD_I32, M), t1(pos.data(), RAD_I32, M), t2(scd.data(), RAD_F32, M, L)};
        RadTensor tlr[8] = {tld[0], tld[1], tld[2], tld[3], tld[4], tld[5], tld[6], t2(scr.data(), RAD_F32, M, L)};
        RadParam pl[4] = {pi("M", M), pi("heads", H), pi("head_dim", D), pi("window", win)};
        RadArgs ald{tld, 8, pl, 4}, alr{tlr, 8, pl, 4};
        if (r4d_rad_ernie_idx_logits(&ald, nullptr) != RAD_OK || g_ref["ernie_idx_logits"](&alr, nullptr) != RAD_OK) ++g_fail;
        else check_f32("ernie_idx_logits", scd, scr, 1e-5f);

        /* Force ties in row 3 to test lowest-id tie-breaking in ernie_dsa_select! */
        for (int j = 0; j < 20; ++j) scd[3 * L + j] = scr[3 * L + j] = (j % 3 == 0) ? 1.5f : (float)(j % 5);
        std::vector<int32_t> idsd(M * (topk + win)), idsr(M * (topk + win)), sld(M * (topk + win)), slr(M * (topk + win));
        RadTensor tsd[6] = {t2(scd.data(), RAD_F32, M, L), t1(pos.data(), RAD_I32, M), t1(req.data(), RAD_I32, M),
                            t2(bt.data(), RAD_I32, 1, 8), t2(idsd.data(), RAD_I32, M, topk + win), t2(sld.data(), RAD_I32, M, topk + win)};
        RadTensor tsr[6] = {tsd[0], tsd[1], tsd[2], tsd[3], t2(idsr.data(), RAD_I32, M, topk + win), t2(slr.data(), RAD_I32, M, topk + win)};
        RadParam pds[5] = {pi("M", M), pi("topk", topk), pi("window", win), pi("block_size", bs), pi("num_pages", pages)};
        RadArgs ads{tsd, 6, pds, 5}, ars{tsr, 6, pds, 5};
        if (r4d_rad_ernie_dsa_select(&ads, nullptr) != RAD_OK || g_ref["ernie_dsa_select"](&ars, nullptr) != RAD_OK) ++g_fail;
        else { check_exact("ernie_dsa_select(ids)", idsd, idsr); check_exact("ernie_dsa_select(slots)", sld, slr); }

        /* ernie_mla_project (absorb + expand) */
        const int kd = 128, vd = 128, lat = 256, qw = 192;
        std::vector<uint16_t> xa(M * H * qw), wm(H * (kd + vd) * lat), yad(M * H * lat), yar(M * H * lat), yed(M * H * vd), yer(M * H * vd);
        for (auto& e : xa) e = f2b(nd(rng) * 0.5f);
        for (auto& e : wm) e = f2b(nd(rng) * 0.05f);
        RadTensor tmd[3] = {t2(xa.data(), RAD_BF16, M, H * qw), t2(wm.data(), RAD_BF16, H * (kd + vd), lat), t2(yad.data(), RAD_BF16, M, H * lat)};
        RadTensor tmr[3] = {tmd[0], tmd[1], t2(yar.data(), RAD_BF16, M, H * lat)};
        RadParam pma[7] = {pi("M", M), pi("heads", H), pi("k_dim", kd), pi("v_dim", vd), pi("latent_dim", lat), ps("mode", "absorb"), pi("q_width", qw)};
        RadArgs amd{tmd, 3, pma, 7}, amr{tmr, 3, pma, 7};
        if (r4d_rad_ernie_mla_project(&amd, nullptr) != RAD_OK || g_ref["ernie_mla_project"](&amr, nullptr) != RAD_OK) ++g_fail;
        else check_bf16("ernie_mla_project(absorb)", yad, yar, 2e-3f, 1);

        RadTensor ted[3] = {t2(yad.data(), RAD_BF16, M, H * lat), tmd[1], t2(yed.data(), RAD_BF16, M, H * vd)};
        RadTensor ter[3] = {ted[0], tmd[1], t2(yer.data(), RAD_BF16, M, H * vd)};
        RadParam pme[6] = {pi("M", M), pi("heads", H), pi("k_dim", kd), pi("v_dim", vd), pi("latent_dim", lat), ps("mode", "expand")};
        RadArgs aed{ted, 3, pme, 6}, aer{ter, 3, pme, 6};
        if (r4d_rad_ernie_mla_project(&aed, nullptr) != RAD_OK || g_ref["ernie_mla_project"](&aer, nullptr) != RAD_OK) ++g_fail;
        else check_bf16("ernie_mla_project(expand)", yed, yer, 2e-3f, 1);

        /* ernie_mla_sparse_attn */
        std::vector<uint16_t> qr(M * H * R), kvp(pages * bs * (lat + R)), sad(M * H * lat), sar(M * H * lat);
        for (auto& e : qr) e = f2b(nd(rng) * 0.5f);
        for (auto& e : kvp) e = f2b(nd(rng) * 0.5f);
        std::vector<float> sink(H);
        for (int h = 0; h < H; ++h) sink[h] = (h == 1) ? -INFINITY : (0.5f * nd(rng));
        RadTensor tsad[6] = {t2(yad.data(), RAD_BF16, M, H * lat), t2(qr.data(), RAD_BF16, M, H * R),
                             t4(kvp.data(), RAD_BF16, pages, 1, bs, lat + R), t2(sld.data(), RAD_I32, M, topk + win),
                             t1(sink.data(), RAD_F32, H), t2(sad.data(), RAD_BF16, M, H * lat)};
        RadTensor tsar[6] = {tsad[0], tsad[1], tsad[2], tsad[3], tsad[4], t2(sar.data(), RAD_BF16, M, H * lat)};
        RadParam psa[7] = {pi("M", M), pi("heads", H), pi("latent_dim", lat), pi("rope_dim", R), pi("block_size", bs), pi("width", topk + win), pf("scale", 1.0 / std::sqrt(192.0))};
        RadArgs asad{tsad, 6, psa, 7}, asar{tsar, 6, psa, 7};
        if (r4d_rad_ernie_mla_sparse_attn(&asad, nullptr) != RAD_OK || g_ref["ernie_mla_sparse_attn"](&asar, nullptr) != RAD_OK) ++g_fail;
        else check_bf16("ernie_mla_sparse_attn", sad, sar, 2e-3f, 1);
    }

    /* 20. ernie_dsa_select signed-zero tie-break (-0.0f vs +0.0f) */
    {
        std::vector<float> sc = {-0.0f, +0.0f, -INFINITY};
        std::vector<int32_t> pos = {2}, req = {0}, bt = {0};
        std::vector<int32_t> idsd(2, -9), slsd(2, -9), idsr(2, -9), slsr(2, -9);
        RadTensor td[6] = {t2(sc.data(), RAD_F32, 1, 3), t1(pos.data(), RAD_I32, 1), t1(req.data(), RAD_I32, 1),
                           t2(bt.data(), RAD_I32, 1, 1), t2(idsd.data(), RAD_I32, 1, 2), t2(slsd.data(), RAD_I32, 1, 2)};
        RadTensor tr[6] = {td[0], td[1], td[2], td[3], t2(idsr.data(), RAD_I32, 1, 2), t2(slsr.data(), RAD_I32, 1, 2)};
        RadParam p[5] = {pi("M", 1), pi("topk", 1), pi("window", 1), pi("block_size", 4), pi("num_pages", 1)};
        RadArgs ad{td, 6, p, 5}, ar{tr, 6, p, 5};
        if (r4d_rad_ernie_dsa_select(&ad, nullptr) != RAD_OK || g_ref["ernie_dsa_select"](&ar, nullptr) != RAD_OK) ++g_fail;
        else {
            check_exact("ernie_dsa_select(signed0 ids)", idsd, idsr);
            check_exact("ernie_dsa_select(signed0 slots)", slsd, slsr);
        }
    }
    /* 21. M=128 (>64 parallel dispatch) & ratio=128 HCA state_write parity */
    {
        const int M = 128, ratio = 128, D = 64, coff = 1, W = coff * D, S = 128;
        std::vector<float> kv(M * W), sc(M * W), ape(ratio * W), st_d(S * 2 * W, 0.0f), st_r(S * 2 * W, 0.0f);
        for (auto& e : kv) e = nd(rng);
        for (auto& e : sc) e = nd(rng) * 0.5f;
        for (auto& e : ape) e = nd(rng) * 0.2f;
        std::vector<int32_t> pos(M), sl(M);
        for (int i = 0; i < M; ++i) { pos[i] = i; sl[i] = i; }
        RadTensor td[6] = {t2(kv.data(), RAD_F32, M, W), t2(sc.data(), RAD_F32, M, W), t2(ape.data(), RAD_F32, ratio, W),
                           t1(pos.data(), RAD_I32, M), t1(sl.data(), RAD_I32, M), t2(st_d.data(), RAD_F32, S, 2 * W)};
        RadTensor tr[6] = {td[0], td[1], td[2], td[3], td[4], t2(st_r.data(), RAD_F32, S, 2 * W)};
        RadParam p[3] = {pi("M", M), pi("ratio", ratio), pi("head_dim", D)};
        RadArgs ad{td, 6, p, 3}, ar{tr, 6, p, 3};
        if (r4d_rad_ernie_hca_state_write(&ad, nullptr) != RAD_OK || g_ref["ernie_hca_state_write"](&ar, nullptr) != RAD_OK) ++g_fail;
        else check_f32("ernie_hca_state_write(M128,r128)", st_d, st_r, 1e-6f);
    }

    std::printf("\nSUMMARY: pass=%d fail=%d\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
