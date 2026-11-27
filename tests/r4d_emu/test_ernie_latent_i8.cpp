/* tests/r4d_emu/test_ernie_latent_i8.cpp -- the i8_ds_mla latent cache (kv-fp8-latent v2): libr4d's ernie_mla_latent_store with kv_layout i8_ds_mla and the
 * i8_ds_mla form of ernie_mla_sparse_attn, run by the SIMT emulator, against libref.so.
 *   store    the device writer's bytes equal libref's, byte for byte (codes, scales, rope, untouched rows), on adversarial inputs: zeros, bf16 subnormals, the largest
 *            finite magnitude (0x7f7f and its negative), mixed signs, exact halfway inputs (ties to even), a sweep of every binade, a tile under the 1e-30 floor; and the
 *            format's own properties checked on libref's bytes against an independent definition (scale = f32 absmax / 127, |code| <= 127, |x - code * s| <= s / 2,
 *            every decoded value finite); refusals of a non-finite latent value and of a bad slot (host build: refused, cache untouched; scan off: skipped and latched)
 *   attn     the i8 reader (fast coop / wave kernels and the general kernel, picked by geometry and by alignment) equals libref's i8 reader within the 1-ulp band the
 *            existing ernie_mla_sparse_attn checks use; and the fast and general device kernels agree within the same band
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
#include <string>
#include <vector>

static uint16_t f2b(float f) { return rad_f32_to_bf16(f); }
static float b2f(uint16_t h) { return rad_bf16_to_f32(h); }
static RadParam pi(const char* k, int64_t v) { RadParam p{}; p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }
static RadParam pf(const char* k, double v) { RadParam p{}; p.key = k; p.kind = RAD_P_F64; p.dval = v; return p; }
static RadParam ps(const char* k, const char* v) { RadParam p{}; p.key = k; p.kind = RAD_P_STR; p.sval = v; return p; }
static RadTensor t1(void* d, uint32_t dt, int64_t n) { RadTensor t{}; t.data = d; t.dtype = dt; t.rank = 1; t.shape[0] = n; t.stride[0] = 1; return t; }
static RadTensor t2(void* d, uint32_t dt, int64_t r, int64_t c) { RadTensor t{}; t.data = d; t.dtype = dt; t.rank = 2; t.shape[0] = r; t.shape[1] = c; t.stride[0] = c; t.stride[1] = 1; return t; }
static RadTensor t4(void* d, uint32_t dt, int64_t d0, int64_t d1, int64_t d2, int64_t d3) {
    RadTensor t{}; t.data = d; t.dtype = dt; t.rank = 4; t.shape[0] = d0; t.shape[1] = d1; t.shape[2] = d2; t.shape[3] = d3;
    t.stride[3] = 1; t.stride[2] = d3; t.stride[1] = d2 * d3; t.stride[0] = d1 * d2 * d3; return t;
}
static std::map<std::string, RadLaunchFn> g_ref;
static bool load_libref(const char* path) {
    void* h = dlopen(path, RTLD_NOW);
    if (!h) { std::fprintf(stderr, "dlopen(%s) failed: %s\n", path, dlerror()); return false; }
    auto cnt_fn = (int(*)(void))dlsym(h, "rad_kernel_count");
    auto at_fn = (const RadKernelInfo*(*)(int))dlsym(h, "rad_kernel_at");
    if (!cnt_fn || !at_fn) return false;
    for (int i = 0, n = cnt_fn(); i < n; ++i) { const RadKernelInfo* k = at_fn(i); if (k && k->op && k->launch && !g_ref.count(k->op)) g_ref[k->op] = k->launch; }
    return true;
}
/* the layout under test (argv[2]): i8_ds_mla (default), i10_ds_mla (10-bit codes: int8 high byte + 2 packed low bits) or i12_ds_mla (12-bit: int8 + a nibble) */
static const char* g_lay = "i8_ds_mla";
static int g_lb = 0;                                   /* low bits under the int8 high byte */
static float cmaxf() { return (float)((1 << (7 + g_lb)) - 1); }   /* 127, 511, 2047: the code range and the scale divisor */
static int g_pass = 0, g_fail = 0;
static void verdict(const std::string& name0, bool ok, const char* detail = "") {
    const std::string name = std::strcmp(g_lay, "i8_ds_mla") ? std::string("[") + g_lay + "] " + name0 : name0;
    std::printf("%s %-70s %s\n", ok ? "PASS" : "FAIL", name.c_str(), detail);
    ok ? ++g_pass : ++g_fail;
}
template <typename T> static void check_exact(const std::string& name, const std::vector<T>& a, const std::vector<T>& b) {
    int bad = 0;
    for (size_t i = 0; i < a.size(); ++i) bad += a[i] != b[i];
    char d[64]; std::snprintf(d, sizeof d, "mismatch=%d/%zu", bad, a.size());
    verdict(name, a.size() == b.size() && !bad, d);
}
static void check_ulp(const std::string& name, const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, int max_ulp, float tol) {
    int bad = 0, exact = 0; float mx = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const float fa = b2f(a[i]), fb = b2f(b[i]);
        if (!std::isfinite(fa) || !std::isfinite(fb)) { ++bad; continue; }
        if (a[i] == b[i]) { ++exact; continue; }
        const float d = std::fabs(fa - fb); mx = std::max(mx, d);
        if (std::abs((int)a[i] - (int)b[i]) > max_ulp && d > tol * (1.0f + std::fabs(fb))) ++bad;
    }
    char d[96]; std::snprintf(d, sizeof d, "bad=%d exact=%d/%zu max_abs=%.3e", bad, exact, a.size(), mx);
    verdict(name, !bad, d);
}
static uint32_t g_s = 777u;
static float rnd() { g_s = g_s * 1664525u + 1013904223u; return (float)((g_s >> 8) & 0xffff) / 65535.0f * 2.0f - 1.0f; }

struct StoreGeo { int L, R, qb, bs, pages; int lo() const { return L * g_lb / 8; } int rowb() const { return L + lo() + 4 * (L / qb) + 2 * R; } };
static int run_store(RadLaunchFn fn, const StoreGeo& g, std::vector<uint16_t>& src, std::vector<int32_t>& slots, std::vector<uint8_t>& cache) {
    const int M = (int)slots.size();
    RadParam p[6] = {pi("M", M), pi("latent_dim", g.L), pi("rope_dim", g.R), pi("block_size", g.bs), pi("quant_block", g.qb), ps("kv_layout", g_lay)};
    RadTensor t[3] = {t2(src.data(), RAD_BF16, M, g.L + g.R), t1(slots.data(), RAD_I32, M), t4(cache.data(), RAD_U8, g.pages, 1, g.bs, g.rowb())};
    RadArgs a{t, 3, p, 6};
    return fn(&a, nullptr);
}
/* the format's definition, checked independently of either writer on the bytes of one stored row */
static bool row_props(const StoreGeo& g, const uint16_t* x, const uint8_t* row, std::string* why) {
    char b[160];
    for (int t = 0; t < g.L / g.qb; ++t) {
        float amax = 0;
        for (int c = t * g.qb; c < (t + 1) * g.qb; ++c) amax = std::max(amax, std::fabs(b2f(x[c])));
        const float want = (amax >= 1e-30f ? amax : 1e-30f) / cmaxf();
        float s; std::memcpy(&s, row + g.L + g.lo() + 4 * t, 4);
        if (std::memcmp(&s, &want, 4)) { std::snprintf(b, sizeof b, "tile %d scale %a != %a", t, s, want); *why = b; return false; }
        for (int c = t * g.qb; c < (t + 1) * g.qb; ++c) {
            int code = (int)(int8_t)row[c];
            if (g_lb) { const int per = 8 / g_lb; code = code * (1 << g_lb) + ((row[g.L + c / per] >> ((c % per) * g_lb)) & ((1 << g_lb) - 1)); }
            const double xv = b2f(x[c]), dv = (double)((float)code * s);
            if (code < -(int)cmaxf() || code > (int)cmaxf() || !std::isfinite((float)code * s)) { std::snprintf(b, sizeof b, "c %d code %d", c, code); *why = b; return false; }
            /* the definition: code = rint(fl(x / s)) clamped to [-127, 127] (the f32 quotient is rounded before rint, so an input within one quotient ulp of a
             * halfway point can take the far code: x = absmax / 2 has x / s = 63.5 (1 - e) with e < 2^-23, which rounds to 63.5 and then to even, 64) */
            const int want_code = (int)std::min(cmaxf(), std::max(-cmaxf(), std::rint(b2f(x[c]) / s)));
            if (code != want_code) { std::snprintf(b, sizeof b, "c %d x %a code %d != rint(x/s) %d", c, xv, code, want_code); *why = b; return false; }
            /* and therefore nearest within the quotient's rounding: |x - code s| <= s / 2 + 128 * 2^-23 * s (independent of the f32 path) */
            if (std::fabs(xv - (double)code * s) > (0.5 + ((double)cmaxf() + 1.0) * std::ldexp(1.0, -23)) * (double)s + 1e-45) { std::snprintf(b, sizeof b, "c %d x %a code %d s %a: not nearest", c, xv, code, s); *why = b; return false; }
            (void)dv;
        }
    }
    for (int c = 0; c < g.R; ++c) {
        const uint8_t* rp = row + g.L + g.lo() + 4 * (g.L / g.qb) + 2 * c;
        if ((uint16_t)(rp[0] | (rp[1] << 8)) != x[g.L + c]) { *why = "rope bits"; return false; }
    }
    return true;
}
static void store_equal(const std::string& name, const StoreGeo& g, std::vector<uint16_t> src, std::vector<int32_t> slots) {
    std::vector<uint8_t> cd((size_t)g.pages * g.bs * g.rowb(), 0xA5), cr(cd);
    const int rd = run_store(r4d_rad_ernie_mla_latent_store, g, src, slots, cd), rr = run_store(g_ref["ernie_mla_latent_store"], g, src, slots, cr);
    if (rd != RAD_OK || rr != RAD_OK) { char d[64]; std::snprintf(d, sizeof d, "device rc %d, libref rc %d", rd, rr); verdict(name, false, d); return; }
    check_exact(name + " (device bytes = libref)", cd, cr);
    std::string why; bool ok = true;
    for (size_t m = 0; m < slots.size() && ok; ++m)
        if (slots[m] >= 0) ok = row_props(g, &src[m * (g.L + g.R)], &cr[(size_t)slots[m] * g.rowb()], &why);
    verdict(name + " (libref bytes meet the i8_ds_mla definition)", ok, why.c_str());
}
static std::vector<uint16_t> rows_of(const StoreGeo& g, int M, float (*gen)(int, int)) {
    std::vector<uint16_t> s((size_t)M * (g.L + g.R));
    for (int m = 0; m < M; ++m) for (int c = 0; c < g.L + g.R; ++c) s[(size_t)m * (g.L + g.R) + c] = f2b(gen(m, c));
    return s;
}
extern "C" int r4d_ernie_mla_take_error(void);

static void store_cases() {
    const StoreGeo g{64, 8, 32, 4, 3};   /* 2 tiles a row, 12 slots */
    std::vector<int32_t> sl = {5, -1, 0, 11, 7, 2};
    const int M = (int)sl.size();
    auto bits_rows = [&](uint16_t (*gen)(int, int)) { std::vector<uint16_t> s((size_t)M * (g.L + g.R)); for (int m = 0; m < M; ++m) for (int c = 0; c < g.L + g.R; ++c) s[(size_t)m * (g.L + g.R) + c] = gen(m, c); return s; };
    store_equal("store: random mixed signs, 3 decades", g, rows_of(g, M, [](int, int c) { return rnd() * std::pow(10.0f, (float)(c % 3)); }), sl);
    store_equal("i8 store: all zeros (and -0)", g, bits_rows([](int m, int c) -> uint16_t { return (m + c) & 1 ? 0x8000 : 0x0000; }), sl);
    store_equal("i8 store: bf16 subnormals only (under the 1e-30 floor)", g, bits_rows([](int m, int c) -> uint16_t { return (uint16_t)(((c * 7 + m) % 127 + 1) | ((c & 1) << 15)); }), sl);
    store_equal("i8 store: subnormals beside normal values in one tile", g, bits_rows([](int m, int c) -> uint16_t { return c % 5 ? (uint16_t)((c + m) | 0x8000 * (c & 1)) : f2b(0.75f * (float)(c - 30)); }), sl);
    store_equal("i8 store: largest finite magnitude 0x7f7f / 0xff7f", g, bits_rows([](int m, int c) -> uint16_t { return (c + m) % 3 == 0 ? 0x7f7f : (c % 3 == 1 ? 0xff7f : f2b(rnd() * 3e38f)); }), sl);
    store_equal("i8 store: every value 0x7f7f (review's top-of-range input)", g, bits_rows([](int, int) -> uint16_t { return 0x7f7f; }), sl);
    /* tile absmax 127 -> s = 1 exactly: k + 0.5 inputs are exact ties (ties to even), and 126.5 rounds to 126 */
    store_equal("i8 store: exact halfway inputs, ties to even", g, rows_of(g, M, [](int m, int c) { return c % 32 == 0 ? (m & 1 ? -cmaxf() : cmaxf()) : ((float)((c * 5 + m) % (int)cmaxf()) + 0.5f) * (c & 2 ? -1.0f : 1.0f); }), sl);
    store_equal("i8 store: every binade from 2^-133 to 2^127", g, rows_of(g, M, [](int m, int c) { return std::ldexp(1.0f + 0.37f * rnd(), -133 + ((m * 64 + c) * 7) % 261) * (c & 1 ? -1.0f : 1.0f); }), sl);
    store_equal("i8 store: one outlier per tile (absmax 1000x the rest)", g, rows_of(g, M, [](int m, int c) { return c % 32 == (m * 3) % 32 ? 900.0f : rnd(); }), sl);
    /* the release geometry, many rows */
    const StoreGeo gr{512, 64, 32, 16, 4};
    std::vector<int32_t> slr(64); for (int i = 0; i < 64; ++i) slr[i] = (i * 13 + 5) % 64;
    std::vector<uint16_t> big((size_t)64 * 576);
    for (size_t i = 0; i < big.size(); ++i) big[i] = f2b(rnd() * 4.0f * (1.0f + (float)((i / 32) % 5)));
    store_equal("i8 store: release L512 R64 qb32, 64 rows", gr, big, slr);
    for (int qb : {128, 64, 16}) { const StoreGeo gq{512, 64, qb, 16, 4}; store_equal("i8 store: release L512 R64 qb" + std::to_string(qb), gq, big, slr); }

    /* refusals: a non-finite latent value and a bad slot are refused by libref and by the host launcher (cache untouched); scan off: skipped row, latched */
    for (uint16_t badv : {(uint16_t)0x7fc0, (uint16_t)0x7f80, (uint16_t)0xff80}) {
        auto s = rows_of(g, M, [](int, int) { return rnd(); });
        s[(size_t)2 * (g.L + g.R) + 9] = badv;
        std::vector<uint8_t> cd((size_t)g.pages * g.bs * g.rowb(), 0xA5), cr(cd), c0(cd);
        const int rd = run_store(r4d_rad_ernie_mla_latent_store, g, s, sl, cd), rr = run_store(g_ref["ernie_mla_latent_store"], g, s, sl, cr);
        char nm[96]; std::snprintf(nm, sizeof nm, "i8 store refuses latent 0x%04x (libref and host launcher, cache untouched)", badv);
        verdict(nm, rd == RAD_E_SHAPE && rr == RAD_E_SHAPE && cd == c0 && cr == c0);
        setenv("R4D_MLA_VALIDATE", "0", 1);
        (void)r4d_ernie_mla_take_error();
        std::vector<uint8_t> cs(c0);
        const int rs = run_store(r4d_rad_ernie_mla_latent_store, g, s, sl, cs);
        const int latched = r4d_ernie_mla_take_error();
        unsetenv("R4D_MLA_VALIDATE");
        /* expected: every row but the bad one stored as libref stores it alone */
        std::vector<int32_t> sl2(sl); sl2[2] = -1;
        std::vector<uint8_t> want(c0);
        run_store(g_ref["ernie_mla_latent_store"], g, s, sl2, want);
        std::snprintf(nm, sizeof nm, "i8 store scan off, latent 0x%04x: row skipped whole, others stored, latched", badv);
        verdict(nm, rs == RAD_OK && latched != 0 && cs == want);
    }
    {
        auto s = rows_of(g, M, [](int, int) { return rnd(); });
        std::vector<int32_t> bad(sl); bad[3] = 12;   /* one past the pool */
        std::vector<uint8_t> cd((size_t)g.pages * g.bs * g.rowb(), 0xA5), cr(cd), c0(cd);
        const int rd = run_store(r4d_rad_ernie_mla_latent_store, g, s, bad, cd), rr = run_store(g_ref["ernie_mla_latent_store"], g, s, bad, cr);
        verdict("i8 store refuses a slot past the pool (libref and host launcher, cache untouched)", rd == RAD_E_SHAPE && rr == RAD_E_SHAPE && cd == c0 && cr == c0);
    }
    {   /* an unknown layout name is refused by both */
        auto s = rows_of(g, M, [](int, int) { return rnd(); });
        std::vector<uint8_t> cd((size_t)g.pages * g.bs * g.rowb(), 0xA5);
        RadParam p[6] = {pi("M", M), pi("latent_dim", g.L), pi("rope_dim", g.R), pi("block_size", g.bs), pi("quant_block", g.qb), ps("kv_layout", "i4_ds_mla")};
        RadTensor t[3] = {t2(s.data(), RAD_BF16, M, g.L + g.R), t1(sl.data(), RAD_I32, M), t4(cd.data(), RAD_U8, g.pages, 1, g.bs, g.rowb())};
        RadArgs a{t, 3, p, 6};
        const int rd = r4d_rad_ernie_mla_latent_store(&a, nullptr), rr = g_ref["ernie_mla_latent_store"](&a, nullptr);
        verdict("store refuses an unknown kv_layout (device and libref)", rd != RAD_OK && rr != RAD_OK);
    }
}

struct Geo { const char* name; int H, L, R, qb, bs, pages, W, M; float sink0; };
static void run_geo(const Geo& g) {
    const StoreGeo sg{g.L, g.R, g.qb, g.bs, g.pages};
    const int rowb = sg.rowb(), n = g.pages * g.bs;
    std::vector<uint16_t> src((size_t)n * (g.L + g.R));
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < g.L + g.R; ++c) {
            float v = rnd() * 0.9f * std::pow(6.0f, (float)(((c < g.L ? c / g.qb : 0) + r) % 3));
            if (r == 5 && c < g.qb) v = 0.0f;
            if (r == 7 && c == 3) v = 7e-4f * rnd();
            src[(size_t)r * (g.L + g.R) + c] = f2b(v);
        }
    std::vector<int32_t> slots(n);
    for (int s = 0; s < n; ++s) slots[s] = (s * 7 + 3) % n;
    slots[1] = -1;
    std::vector<uint8_t> cdev((size_t)n * rowb, 0xA5), cref(cdev);
    if (run_store(r4d_rad_ernie_mla_latent_store, sg, src, slots, cdev) != RAD_OK || run_store(g_ref["ernie_mla_latent_store"], sg, src, slots, cref) != RAD_OK) {
        verdict(std::string(g.name) + " store launch", false); return;
    }
    check_exact(std::string(g.name) + " i8 store bytes (device = libref)", cdev, cref);
    std::vector<uint8_t> written(n, 0);
    for (int i = 0; i < n; ++i) if (slots[i] >= 0) written[slots[i]] = 1;
    std::vector<int32_t> sl((size_t)g.M * g.W);
    for (auto& s : sl) { const float u = rnd(); int v = u < -0.6f ? -1 : (int)((u + 1.0f) * 0.5f * (float)(n - 1)); while (v >= 0 && !written[v]) v = (v + 1) % n; s = v; }
    std::vector<uint16_t> q((size_t)g.M * g.H * g.L), qr((size_t)g.M * g.H * g.R);
    for (auto& e : q) e = f2b(rnd()); for (auto& e : qr) e = f2b(rnd());
    std::vector<float> sink(g.H);
    for (int h = 0; h < g.H; ++h) sink[h] = g.sink0 == -INFINITY ? g.sink0 : g.sink0 + 0.2f * h;
    /* a copy of the cache 2 bytes off a 16-byte boundary: the launcher must take the general kernel */
    std::vector<uint8_t> cmis(cdev.size() + 32);
    uint8_t* mis = cmis.data() + (16 - ((uintptr_t)cmis.data() % 16)) % 16 + 2;
    std::memcpy(mis, cdev.data(), cdev.size());
    std::vector<uint16_t> o8(q.size()), og(q.size()), oref(q.size());
    RadParam pa[10] = {pi("M", g.M), pi("heads", g.H), pi("latent_dim", g.L), pi("rope_dim", g.R), pi("block_size", g.bs), pi("width", g.W), pf("scale", 0.125),
                       ps("kv_layout", g_lay), pi("quant_block", g.qb), pi("num_pages", g.pages)};
    RadTensor a8[6] = {t2(q.data(), RAD_BF16, g.M, g.H * g.L), t2(qr.data(), RAD_BF16, g.M, g.H * g.R), t4(cdev.data(), RAD_U8, g.pages, 1, g.bs, rowb),
                       t2(sl.data(), RAD_I32, g.M, g.W), t1(sink.data(), RAD_F32, g.H), t2(o8.data(), RAD_BF16, g.M, g.H * g.L)};
    RadTensor ag[6] = {a8[0], a8[1], t4(mis, RAD_U8, g.pages, 1, g.bs, rowb), a8[3], a8[4], t2(og.data(), RAD_BF16, g.M, g.H * g.L)};
    RadTensor ar[6] = {a8[0], a8[1], a8[2], a8[3], a8[4], t2(oref.data(), RAD_BF16, g.M, g.H * g.L)};
    RadArgs x8{a8, 6, pa, 10}, xg{ag, 6, pa, 10}, xr{ar, 6, pa, 10};
    if (r4d_rad_ernie_mla_sparse_attn(&x8, nullptr) != RAD_OK || r4d_rad_ernie_mla_sparse_attn(&xg, nullptr) != RAD_OK || g_ref["ernie_mla_sparse_attn"](&xr, nullptr) != RAD_OK) {
        verdict(std::string(g.name) + " reader launches", false); return;
    }
    const bool fast = g.R == 64 && (g.L == 512 || g.L == 256) && g.qb % 8 == 0 && rowb % 16 == 0;   /* (launcher: fast kernels on the same condition) */
    check_ulp(std::string(g.name) + (fast ? " i8 reader (fast kernel) vs libref" : " i8 reader (general kernel) vs libref"), o8, oref, 1, 2e-3f);
    check_ulp(std::string(g.name) + " i8 reader (general kernel, unaligned cache) vs libref", og, oref, 1, 2e-3f);
    /* the reader must actually read the i8 row: the same bytes read as fp8_ds_mla must differ */
    std::vector<uint16_t> ox(q.size());
    RadParam px[10]; std::copy(pa, pa + 10, px); px[7] = ps("kv_layout", "fp8_ds_mla");
    RadTensor ax[6] = {a8[0], a8[1], a8[2], a8[3], a8[4], t2(ox.data(), RAD_BF16, g.M, g.H * g.L)};
    RadArgs xx{ax, 6, px, 10};
    if (r4d_rad_ernie_mla_sparse_attn(&xx, nullptr) == RAD_OK) {
        int diff = 0; for (size_t i = 0; i < ox.size(); ++i) diff += ox[i] != o8[i];
        verdict(std::string(g.name) + " the layout name changes the decode (fp8 reading of i8 bytes differs)", diff > 0);
    }
}

int main(int argc, char** argv) {
    const char* ref_path = argc > 1 ? argv[1] : "/src/build-ernie55/libref.so";
    if (argc > 2) g_lay = argv[2];
    g_lb = !std::strcmp(g_lay, "i10_ds_mla") ? 2 : !std::strcmp(g_lay, "i12_ds_mla") ? 4 : 0;
    if (g_lb == 0 && std::strcmp(g_lay, "i8_ds_mla")) { std::fprintf(stderr, "layout must be i8_ds_mla, i10_ds_mla or i12_ds_mla\n"); return 2; }
    if (!load_libref(ref_path)) return 2;
    store_cases();
    const Geo geos[] = {
        {"release L512 R64 qb128 coop", 4, 512, 64, 128, 8, 6, 80, 3, 0.25f},
        {"release L512 R64 qb128 wave", 5, 512, 64, 128, 8, 4, 40, 9, -INFINITY},
        {"release L512 R64 qb32 coop (704 B)", 4, 512, 64, 32, 8, 6, 80, 3, -1.0f},
        {"release L512 R64 qb32 wave (704 B)", 5, 512, 64, 32, 8, 4, 40, 9, 0.5f},
        {"release L512 R64 qb16 coop (768 B)", 4, 512, 64, 16, 8, 6, 80, 3, 0.25f},
        {"release L512 R64 qb64 wave (672 B)", 5, 512, 64, 64, 8, 4, 40, 9, -1.0f},
        {"L256 R64 qb32 coop (416 B)", 3, 256, 64, 32, 4, 5, 70, 2, 0.5f},
        {"L256 R64 qb128 (392 B: general)", 3, 256, 64, 128, 4, 5, 70, 2, 0.25f},
        {"L64 R16 qb16 general kernel", 2, 64, 16, 16, 4, 5, 9, 3, -INFINITY},
    };
    for (const auto& g : geos) run_geo(g);
    std::printf("\nSUMMARY: pass=%d fail=%d\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
