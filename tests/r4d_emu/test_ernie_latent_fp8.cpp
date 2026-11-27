/* tests/r4d_emu/test_ernie_latent_fp8.cpp -- the fp8_ds_mla latent cache (kv-fp8-latent): libr4d's ernie_mla_latent_store and the kv_layout
 * fp8_ds_mla form of ernie_mla_sparse_attn, run by the SIMT emulator, against libref.so.
 *   store    the device writer's bytes equal libref's, byte for byte (codes, scales, rope, untouched rows); its numeric edges (the clamp of the top binade, halfway
 *            inputs, a sweep of every binade) and its refusals (a non-finite latent value, a slot outside the pool: refused on a host build, skipped and latched with the scan off)
 *   attn     the fp8 reader equals the same device kernel's bf16 form on a cache holding the decoded values, bit for bit (the decode is exact, so
 *            nothing may differ), and equals libref's fp8 reader within the 1-ulp band the existing ernie_mla_sparse_attn check uses (two-pass against
 *            online softmax)
 * SPDX-License-Identifier: Apache-2.0
 */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
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
    auto at_fn = (const RadKernelInfo*(*)(int))dlsym(h, "rad_kernel_at");
    if (!cnt_fn || !at_fn) return false;
    for (int i = 0, n = cnt_fn(); i < n; ++i) { const RadKernelInfo* k = at_fn(i); if (k && k->op && k->launch && !g_ref.count(k->op)) g_ref[k->op] = k->launch; }
    return true;
}
static int g_pass = 0, g_fail = 0;
template <typename T> static void check_exact(const std::string& name, const std::vector<T>& a, const std::vector<T>& b) {
    int bad = 0;
    for (size_t i = 0; i < a.size(); ++i) bad += a[i] != b[i];
    if (a.size() != b.size() || bad) { std::printf("FAIL %-52s mismatch=%d/%zu\n", name.c_str(), bad, a.size()); ++g_fail; }
    else { std::printf("PASS %-52s exact=%zu/%zu\n", name.c_str(), a.size(), a.size()); ++g_pass; }
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
    if (bad) { std::printf("FAIL %-52s bad=%d/%zu exact=%d max_abs=%.3e\n", name.c_str(), bad, a.size(), exact, mx); ++g_fail; }
    else { std::printf("PASS %-52s exact=%d/%zu max_abs=%.3e\n", name.c_str(), exact, a.size(), mx); ++g_pass; }
}
/* Distance in representable bf16 steps on the ordered (sign and magnitude, +0 = -0) representation: 1 is exactly the neighbouring value. */
static int ord_bf16(uint16_t h) { return (h & 0x8000u) ? -(int)(h & 0x7fffu) : (int)h; }
static int ulp_dist(uint16_t a, uint16_t b) {
    if ((a & 0x7f80u) == 0x7f80u || (b & 0x7f80u) == 0x7f80u) return a == b ? 0 : 1 << 20;
    return std::abs(ord_bf16(a) - ord_bf16(b));
}
/* The bound is on the representable distance, nothing else; the histogram and the first offender are printed either way. */
static void check_true_ulp(const std::string& name, const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, int max_steps) {
    size_t hist[4] = {0, 0, 0, 0}, first = (size_t)-1;
    int mx = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const int d = ulp_dist(a[i], b[i]);
        ++hist[std::min(d, 3)];
        mx = std::max(mx, d);
        if (d > max_steps && first == (size_t)-1) first = i;
    }
    if (first != (size_t)-1) {
        std::printf("FAIL %-52s max %d step(s) > %d; 0/1/2/3+: %zu/%zu/%zu/%zu; first at %zu: 0x%04x (%g) vs 0x%04x (%g)\n", name.c_str(), mx, max_steps, hist[0], hist[1], hist[2], hist[3], first,
                    (unsigned)a[first], b2f(a[first]), (unsigned)b[first], b2f(b[first]));
        ++g_fail;
    } else {
        std::printf("PASS %-52s max %d step(s); 0/1/2/3+: %zu/%zu/%zu/%zu\n", name.c_str(), mx, hist[0], hist[1], hist[2], hist[3]);
        ++g_pass;
    }
}
/* Measurement only (no verdict): the true-step histogram of a comparison that is asserted with the file's absolute-plus-relative tolerance. */
static void info_true_ulp(const std::string& name, const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    size_t hist[4] = {0, 0, 0, 0};
    int mx = 0;
    for (size_t i = 0; i < a.size(); ++i) { const int d = ulp_dist(a[i], b[i]); ++hist[std::min(d, 3)]; mx = std::max(mx, d); }
    std::printf("INFO %-52s true steps: max %d; 0/1/2/3+: %zu/%zu/%zu/%zu\n", name.c_str(), mx, hist[0], hist[1], hist[2], hist[3]);
}
static uint32_t g_s = 12345u;
static float rnd() { g_s = g_s * 1664525u + 1013904223u; return (float)((g_s >> 8) & 0xffff) / 65535.0f * 2.0f - 1.0f; }

struct Geo { const char* name; int H, L, R, qb, bs, pages, W, M; float sink0; };

static void run_geo(const Geo& g) {
    const int nt = g.L / g.qb, rowb = g.L + 4 * nt + 2 * g.R, n = g.pages * g.bs;
    std::vector<uint16_t> src((size_t)n * (g.L + g.R));
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < g.L + g.R; ++c) {
            const int tile = c < g.L ? c / g.qb : 0;
            float v = rnd() * 0.9f * std::pow(6.0f, (float)((tile + r) % 3));
            if (r == 5 && c < g.qb) v = 0.0f;                  /* an all-zero first tile */
            if (r == 7 && c == 3) v = 7e-4f * rnd();            /* a tiny tile max */
            src[(size_t)r * (g.L + g.R) + c] = f2b(v);
        }
    std::vector<int32_t> slots(n);
    for (int s = 0; s < n; ++s) slots[s] = (s * 7 + 3) % n;
    slots[1] = -1;                                              /* a skipped row */
    std::vector<uint8_t> cdev((size_t)n * rowb, 0xA5), cref((size_t)n * rowb, 0xA5);
    RadParam pst[5] = {pi("M", n), pi("latent_dim", g.L), pi("rope_dim", g.R), pi("block_size", g.bs), pi("quant_block", g.qb)};
    RadTensor td[3] = {t2(src.data(), RAD_BF16, n, g.L + g.R), t1(slots.data(), RAD_I32, n), t4(cdev.data(), RAD_U8, g.pages, 1, g.bs, rowb)};
    RadTensor tr[3] = {td[0], td[1], t4(cref.data(), RAD_U8, g.pages, 1, g.bs, rowb)};
    RadArgs ad{td, 3, pst, 5}, ar{tr, 3, pst, 5};
    if (r4d_rad_ernie_mla_latent_store(&ad, nullptr) != RAD_OK || g_ref["ernie_mla_latent_store"](&ar, nullptr) != RAD_OK) { std::printf("FAIL %s store launch\n", g.name); ++g_fail; return; }
    check_exact(std::string(g.name) + " store bytes (device = libref)", cdev, cref);
    /* the decoded bf16 cache */
    std::vector<uint16_t> dec((size_t)n * (g.L + g.R));
    for (int s = 0; s < n; ++s) {
        const uint8_t* row = &cdev[(size_t)s * rowb];
        for (int c = 0; c < g.L + g.R; ++c) {
            float v;
            if (c < g.L) { float sc; std::memcpy(&sc, row + g.L + 4 * (c / g.qb), 4); v = rad_fp8e4m3_to_f32(row[c]) * sc; }
            else { const uint8_t* rp = row + g.L + 4 * nt + 2 * (c - g.L); v = b2f((uint16_t)(rp[0] | (rp[1] << 8))); }
            dec[(size_t)s * (g.L + g.R) + c] = f2b(v);
        }
    }
    /* the rows named by no slot keep the sentinel, and the dec of those rows are junk: attend only slots that were written */
    std::vector<uint8_t> written(n, 0);
    for (int i = 0; i < n; ++i) if (slots[i] >= 0) written[slots[i]] = 1;
    std::vector<int32_t> sl((size_t)g.M * g.W);
    for (auto& s : sl) { const float u = rnd(); int v = u < -0.6f ? -1 : (int)((u + 1.0f) * 0.5f * (float)(n - 1)); while (v >= 0 && !written[v]) v = (v + 1) % n; s = v; }
    std::vector<uint16_t> q((size_t)g.M * g.H * g.L), qr((size_t)g.M * g.H * g.R);
    for (auto& e : q) e = f2b(rnd()); for (auto& e : qr) e = f2b(rnd());
    std::vector<float> sink(g.H);
    for (int h = 0; h < g.H; ++h) sink[h] = g.sink0 == -INFINITY ? g.sink0 : g.sink0 + 0.2f * h;
    std::vector<uint16_t> o8(q.size()), ob(q.size()), or8(q.size());
    RadParam pa8[10] = {pi("M", g.M), pi("heads", g.H), pi("latent_dim", g.L), pi("rope_dim", g.R), pi("block_size", g.bs), pi("width", g.W), pf("scale", 0.125),
                        ps("kv_layout", "fp8_ds_mla"), pi("quant_block", g.qb), pi("num_pages", g.pages)};
    RadParam pab[7] = {pi("M", g.M), pi("heads", g.H), pi("latent_dim", g.L), pi("rope_dim", g.R), pi("block_size", g.bs), pi("width", g.W), pf("scale", 0.125)};
    RadTensor a8[6] = {t2(q.data(), RAD_BF16, g.M, g.H * g.L), t2(qr.data(), RAD_BF16, g.M, g.H * g.R), t4(cdev.data(), RAD_U8, g.pages, 1, g.bs, rowb),
                       t2(sl.data(), RAD_I32, g.M, g.W), t1(sink.data(), RAD_F32, g.H), t2(o8.data(), RAD_BF16, g.M, g.H * g.L)};
    RadTensor ab[6] = {a8[0], a8[1], t4(dec.data(), RAD_BF16, g.pages, 1, g.bs, g.L + g.R), a8[3], a8[4], t2(ob.data(), RAD_BF16, g.M, g.H * g.L)};
    /* the same decoded cache two bytes off a 16-byte boundary: the launcher then takes the general kernel (the fast kernels need 16-byte aligned loads), whatever the geometry */
    std::vector<uint16_t> dec_store(dec.size() + 8);
    std::copy(dec.begin(), dec.end(), dec_store.begin() + 1);
    std::vector<uint16_t> ogen(q.size());
    RadTensor ag[6] = {a8[0], a8[1], t4(dec_store.data() + 1, RAD_BF16, g.pages, 1, g.bs, g.L + g.R), a8[3], a8[4], t2(ogen.data(), RAD_BF16, g.M, g.H * g.L)};
    RadTensor ar8[6] = {a8[0], a8[1], a8[2], a8[3], a8[4], t2(or8.data(), RAD_BF16, g.M, g.H * g.L)};
    RadArgs x8{a8, 6, pa8, 10}, xb{ab, 6, pab, 7}, xr{ar8, 6, pa8, 10}, xg{ag, 6, pab, 7};
    if (r4d_rad_ernie_mla_sparse_attn(&x8, nullptr) != RAD_OK) { std::printf("FAIL %s device fp8 reader launch\n", g.name); ++g_fail; return; }
    if (r4d_rad_ernie_mla_sparse_attn(&xb, nullptr) != RAD_OK) { std::printf("FAIL %s device bf16 reader launch\n", g.name); ++g_fail; return; }
    if (g_ref["ernie_mla_sparse_attn"](&xr, nullptr) != RAD_OK) { std::printf("FAIL %s libref fp8 reader launch\n", g.name); ++g_fail; return; }
    if (r4d_rad_ernie_mla_sparse_attn(&xg, nullptr) != RAD_OK) { std::printf("FAIL %s device bf16 reader (general kernel) launch\n", g.name); ++g_fail; return; }
    /* which kernel serves which cache (r4d_rad_ernie_mla_sparse_attn): the fast kernels take R == 64, L 256 or 512 and aligned loads; an fp8 row additionally needs qb % 8 == 0 and a row
     * of a multiple of 16 bytes. Where the fp8 row is not eligible (L 256, R 64, qb 128: 392 bytes) it takes the general kernel while an aligned bf16 cache takes a fast one, and the two
     * accumulate in a different order; the bit-for-bit claim is between readers of the same kernel, so the comparison is made against the bf16 reader of the same kernel. */
    const bool fast_geo = g.R == 64 && (g.L == 512 || g.L == 256);
    const bool fp8_fast = fast_geo && g.qb % 8 == 0 && rowb % 16 == 0;
    check_exact(std::string(g.name) + (fp8_fast ? " fp8 reader (fast kernel) = bf16 reader (fast) on the decoded cache" : " fp8 reader (general kernel) = bf16 reader (general, unaligned cache) on the decoded cache"),
                o8, fp8_fast ? ob : ogen);
    if (fast_geo && !fp8_fast) check_true_ulp(std::string(g.name) + " fp8 reader (general) vs bf16 reader (fast): different kernels, true ulp", o8, ob, 1);
    check_ulp(std::string(g.name) + " fp8 reader vs libref fp8 reader", o8, or8, 1, 2e-3f);
    info_true_ulp(std::string(g.name) + " fp8 reader vs libref fp8 reader", o8, or8);
}

/* ---------------------------------------------------------------- the writer at its numeric edges and its refusals (review of e7091439)
 * The device writer must emit libref's bytes, byte for byte, for the whole range: the top binade where the codes are clamped to 240 at scale 2^120 (a code 256 times 2^120
 * overflows f32), the halfway inputs, a sweep of every binade. A non-finite latent value or a slot outside [-1, pool) is a refusal in libref, before any store; on a host build the
 * launcher refuses too (same code, cache untouched); with the host scan off (R4D_MLA_VALIDATE=0, the situation of a card build, whose operands are device memory) the kernel
 * stores nothing for that row and latches the violation (r4d_ernie_mla_take_error). */
extern "C" int r4d_ernie_mla_take_error(void);
extern "C" int rad_kernel_poll_error(void);
extern "C" int rad_kernel_poll_source(int, void**, int64_t*);
extern "C" int rad_kernel_poll_fold(int, const void*, int64_t);

struct StoreGeo { int L, R, qb, bs, pages; int rowb() const { return L + 4 * (L / qb) + 2 * R; } };

static int run_store(RadLaunchFn fn, const StoreGeo& g, std::vector<uint16_t>& src, std::vector<int32_t>& slots, std::vector<uint8_t>& cache) {
    const int M = (int)slots.size();
    RadParam p[5] = {pi("M", M), pi("latent_dim", g.L), pi("rope_dim", g.R), pi("block_size", g.bs), pi("quant_block", g.qb)};
    RadTensor t[3] = {t2(src.data(), RAD_BF16, M, g.L + g.R), t1(slots.data(), RAD_I32, M), t4(cache.data(), RAD_U8, g.pages, 1, g.bs, g.rowb())};
    RadArgs a{t, 3, p, 5};
    return fn(&a, nullptr);
}

/* device bytes == libref bytes for one launch (both must succeed) */
static void store_equal(const std::string& name, const StoreGeo& g, std::vector<uint16_t> src, std::vector<int32_t> slots) {
    std::vector<uint8_t> cd((size_t)g.pages * g.bs * g.rowb(), 0xA5), cr(cd);
    const int rd = run_store(r4d_rad_ernie_mla_latent_store, g, src, slots, cd), rr = run_store(g_ref["ernie_mla_latent_store"], g, src, slots, cr);
    if (rd != RAD_OK || rr != RAD_OK) { std::printf("FAIL %-60s device rc %d, libref rc %d\n", name.c_str(), rd, rr); ++g_fail; return; }
    check_exact(name, cd, cr);
}

static void edge_cases() {
    uint32_t s = 20261126u;
    auto rn = [&]() { s = s * 1664525u + 1013904223u; return (float)((s >> 8) & 0xffffff) / 16777216.0f; };
    /* review's reproducer: latent_dim 128, quant_block 128, rope_dim 1, every latent value 0x7f7f */
    {
        const StoreGeo g{128, 1, 128, 1, 1};
        std::vector<uint16_t> src((size_t)(g.L + g.R), 0);
        for (int c = 0; c < g.L; ++c) src[(size_t)c] = 0x7f7f;
        std::vector<int32_t> slots = {0};
        std::vector<uint8_t> cd((size_t)g.rowb(), 0xA5);
        if (run_store(r4d_rad_ernie_mla_latent_store, g, src, slots, cd) != RAD_OK) { std::printf("FAIL review reproducer launch\n"); ++g_fail; }
        else {
            bool all77 = true;
            for (int c = 0; c < g.L; ++c) all77 &= cd[(size_t)c] == 0x77;
            float sc; std::memcpy(&sc, &cd[(size_t)g.L], 4);
            const bool ok = all77 && sc == std::ldexp(1.0f, 120);
            std::printf("%s %-60s codes 0x77 everywhere, scale 2^120 (the largest decoded value 240 * 2^120 is finite in f32 and bf16)\n", ok ? "PASS" : "FAIL", "review reproducer: every latent value bf16 0x7f7f");
            ok ? ++g_pass : ++g_fail;
        }
        store_equal("review reproducer: device bytes = libref bytes", g, src, slots);
    }
    /* the top binade: rows of one tile each, values in units of 2^120 around the codes that round to 256, both signs */
    {
        const StoreGeo g{128, 1, 128, 4, 8};
        const float us[] = {224.0f, 224.5f, 239.0f, 240.0f, 244.0f, 247.9375f, 248.0f, 248.5f, 252.0f, 254.0f, 254.5f, 255.0f};
        std::vector<uint16_t> src;
        std::vector<int32_t> slots;
        int row = 0;
        for (float u : us)
            for (float sg : {1.0f, -1.0f}) {
                for (int c = 0; c < g.L; ++c) src.push_back(f2b(sg * std::ldexp(c == 0 ? u : (c % 3 == 0 ? 255.0f : u * 0.7f), 120)));
                for (int c = 0; c < g.R; ++c) src.push_back(f2b(0.25f));
                slots.push_back(row++);
            }
        StoreGeo gg = g; gg.pages = (row + g.bs - 1) / g.bs;
        store_equal("top binade of the range (24 rows): device = libref", gg, src, slots);
        /* and none of the stored codes passes 240 at that scale */
        std::vector<uint8_t> cd((size_t)gg.pages * gg.bs * gg.rowb(), 0xA5);
        run_store(r4d_rad_ernie_mla_latent_store, gg, src, slots, cd);
        bool capped = true;
        for (int r = 0; r < row; ++r) for (int c = 0; c < g.L; ++c) capped &= (cd[(size_t)r * gg.rowb() + c] & 0x7f) <= 0x77;
        std::printf("%s %-60s\n", capped ? "PASS" : "FAIL", "top binade: no code above 0x77 at scale 2^120");
        capped ? ++g_pass : ++g_fail;
    }
    /* the halfway inputs and their pinned bytes (as ref_test) */
    {
        const StoreGeo g{128, 1, 128, 1, 1};
        struct H { float in; unsigned code; };
        const H table[] = {{1.0625f, 0x38}, {1.1875f, 0x3a}, {1.3125f, 0x3a}, {1.4375f, 0x3c}, {-1.0625f, 0xb8}, {-1.1875f, 0xba}, {0.0009765625f, 0x00}, {0.0029296875f, 0x02},
                           {0.0048828125f, 0x02}, {0.0068359375f, 0x04}, {0.0146484375f, 0x08}, {-0.0009765625f, 0x80}, {-0.0029296875f, 0x82}, {272.0f, 0x78}, {304.0f, 0x7a},
                           {432.0f, 0x7e}, {-432.0f, 0xfe}, {0.0f, 0x00}, {-0.0f, 0x80}};
        std::vector<uint16_t> src((size_t)(g.L + g.R), 0);
        src[0] = f2b(448.0f);
        for (size_t i = 0; i < sizeof table / sizeof table[0]; ++i) src[i + 1] = f2b(table[i].in);
        std::vector<int32_t> slots = {0};
        std::vector<uint8_t> cd((size_t)g.rowb(), 0xA5);
        run_store(r4d_rad_ernie_mla_latent_store, g, src, slots, cd);
        bool ok = cd[0] == 0x7e;
        for (size_t i = 0; i < sizeof table / sizeof table[0]; ++i) {
            if (cd[i + 1] != table[i].code) { std::printf("  halfway %g: device 0x%02x, expected 0x%02x\n", (double)table[i].in, (unsigned)cd[i + 1], table[i].code); ok = false; }
        }
        std::printf("%s %-60s\n", ok ? "PASS" : "FAIL", "halfway inputs: device bytes are the pinned ties-to-even codes");
        ok ? ++g_pass : ++g_fail;
        store_equal("halfway inputs: device = libref", g, src, slots);
    }
    /* a sweep over every binade, three tile widths worth of mixed values */
    {
        const StoreGeo g{256, 2, 64, 4, 64};
        std::vector<uint16_t> src;
        std::vector<int32_t> slots;
        int row = 0;
        for (int e = -126; e <= 127; ++e) {
            for (int c = 0; c < g.L + g.R; ++c) {
                const int tile = c < g.L ? c / g.qb : 0, ee = std::max(-126, std::min(127, e - 3 * (tile % 4)));
                float v = (rn() < 0.5f ? -1.0f : 1.0f) * std::ldexp((1.0f + rn() * 0.999f) * (c % 7 == 0 ? 1.0f : rn()), ee);
                if (std::fabs(v) > 3.3895313892515355e38f) v = 3.3895313892515355e38f;
                src.push_back(f2b(v));
            }
            slots.push_back(row++);
        }
        store_equal("sweep of every binade (254 rows): device = libref", g, src, slots);
    }
    /* refusals with the host scan on: libref and the launcher agree on the code and leave the cache alone */
    {
        const StoreGeo g{256, 8, 128, 2, 4};
        struct Bad { const char* name; float v; int pos; int slot0, slot1; };
        const Bad bads[] = {{"NaN in tile 0", NAN, 0, 1, 2}, {"+inf in tile 0", INFINITY, 127, 1, 2}, {"-inf in tile 1", -INFINITY, 128, 1, 2}, {"NaN in the last latent value", NAN, 255, 1, 2},
                            {"slot -2", 0.5f, 3, 1, -2}, {"slot == pool", 0.5f, 3, 1, 8}, {"slot far past the pool", 0.5f, 3, 99, 2}};
        for (const Bad& b : bads) {
            std::vector<uint16_t> src((size_t)(2 * (g.L + g.R)), f2b(0.5f));
            src[(size_t)(g.L + g.R) + b.pos] = f2b(b.v);   /* the second row is the bad one */
            std::vector<int32_t> slots = {b.slot0, b.slot1};
            std::vector<uint8_t> cd((size_t)g.pages * g.bs * g.rowb(), 0x5A), cr(cd);
            const int rd = run_store(r4d_rad_ernie_mla_latent_store, g, src, slots, cd), rr = run_store(g_ref["ernie_mla_latent_store"], g, src, slots, cr);
            const bool ok = rd == RAD_E_SHAPE && rr == RAD_E_SHAPE && cd == std::vector<uint8_t>(cd.size(), 0x5A) && cr == std::vector<uint8_t>(cr.size(), 0x5A);
            std::printf("%s refusal (scan on): %-32s libref %d, device %d\n", ok ? "PASS" : "FAIL", b.name, rr, rd);
            ok ? ++g_pass : ++g_fail;
        }
    }
    /* the card form: the host scan is off. The kernel skips the bad row entirely (no store of any of its bytes), stores the others, and latches. */
    {
        setenv("R4D_MLA_VALIDATE", "0", 1);
        (void)r4d_ernie_mla_take_error();
        const StoreGeo g{256, 8, 128, 2, 4};
        struct Bad { const char* name; float v; int pos; int slot0, slot1; int latched; };
        const Bad cases[] = {{"valid", 0.5f, 3, 1, 2, 0}, {"NaN in tile 0", NAN, 0, 1, 2, 1}, {"+inf in tile 0", INFINITY, 127, 1, 2, 1}, {"-inf in tile 1", -INFINITY, 128, 1, 2, 1},
                             {"NaN in the last latent value", NAN, 255, 1, 2, 1}, {"slot -2", 0.5f, 3, 1, -2, 1}, {"slot == pool", 0.5f, 3, 1, 8, 1},
                             {"slot -1 with a NaN row (ignored altogether)", NAN, 3, 1, -1, 0}};
        for (const Bad& c : cases) {
            std::vector<uint16_t> src((size_t)(2 * (g.L + g.R)), f2b(0.5f));
            src[(size_t)(g.L + g.R) + c.pos] = f2b(c.v);
            std::vector<int32_t> slots = {c.slot0, c.slot1};
            std::vector<uint8_t> cd((size_t)g.pages * g.bs * g.rowb(), 0x5A);
            const int rc = run_store(r4d_rad_ernie_mla_latent_store, g, src, slots, cd);
            const int got = r4d_ernie_mla_take_error();
            /* the expected cache, always compared in full: libref's own launch on both rows when nothing is refused (a valid pair, a NaN row sent to slot -1), else row 0 stored as
             * libref stores it alone with the bad row's slot untouched */
            std::vector<uint8_t> want((size_t)g.pages * g.bs * g.rowb(), 0x5A);
            if (c.latched == 0) {
                run_store(g_ref["ernie_mla_latent_store"], g, src, slots, want);
            } else {
                std::vector<uint16_t> src0(src.begin(), src.begin() + (g.L + g.R));
                std::vector<int32_t> slots0 = {c.slot0};
                run_store(g_ref["ernie_mla_latent_store"], g, src0, slots0, want);
            }
            const bool ok = rc == RAD_OK && got == c.latched && r4d_ernie_mla_take_error() == 0 && cd == want;
            std::printf("%s latch (scan off): %-44s rc %d, latched %d%s\n", ok ? "PASS" : "FAIL", c.name, rc, got, cd == want ? "" : ", WRONG BYTES");
            ok ? ++g_pass : ++g_fail;
        }
        unsetenv("R4D_MLA_VALIDATE");
    }
    /* the card's DEFAULT: validation OFF (R4D_ERNIE_CARD_DEFAULTS=1) and card form (no host scan). The latent store's log is always on (R4D_VC_ALWAYS), so a refused row
     * still reaches both polls the engine makes: its own readback (rad_kernel_poll_source + _fold, what a card engine uses once the device has a log) and the library's
     * poll (rad_kernel_poll_error, the engine's fallback). Red against a store whose event goes to a null log with validation off (review r4b blocker). */
    {
        setenv("R4D_ERNIE_CARD_DEFAULTS", "1", 1);
        setenv("R4D_ERNIE_CARD_FORM", "1", 1);
        unsetenv("R4D_ERNIE_VALIDATE");
        const StoreGeo g{256, 8, 128, 2, 4};
        auto bad_launch = [&](std::vector<uint8_t>& cd) {
            std::vector<uint16_t> src((size_t)(2 * (g.L + g.R)), f2b(0.5f));
            src[(size_t)(g.L + g.R) + 5] = f2b(NAN);   /* row 1 holds a NaN; its slot 2 is valid */
            std::vector<int32_t> slots = {1, 2};
            cd.assign((size_t)g.pages * g.bs * g.rowb(), 0x5A);
            return run_store(r4d_rad_ernie_mla_latent_store, g, src, slots, cd);
        };
        auto slot_untouched = [&](const std::vector<uint8_t>& cd, int slot) {
            const size_t rb = (size_t)g.rowb();
            for (size_t i = 0; i < rb; ++i) if (cd[(size_t)slot * rb + i] != 0x5A) return false;
            return true;
        };
        (void)rad_kernel_poll_error();
        std::vector<uint8_t> cd;
        const int rc1 = bad_launch(cd);
        const bool kept1 = slot_untouched(cd, 2) && !slot_untouched(cd, 1);
        void* log = nullptr; int64_t bytes = 0;
        const int has = rad_kernel_poll_source(0, &log, &bytes);
        std::vector<char> copy((size_t)(bytes > 0 ? bytes : 0));
        if (has == 1 && log && bytes > 0) std::memcpy(copy.data(), log, (size_t)bytes);
        const int folded = has == 1 ? rad_kernel_poll_fold(0, copy.data(), bytes) : -9;
        const int rc2 = bad_launch(cd);
        const int polled = rad_kernel_poll_error();
        const int again = rad_kernel_poll_error();
        const bool ok = rc1 == RAD_OK && kept1 && has == 1 && folded > 0 && (folded & 1) && rc2 == RAD_OK && polled > 0 && (polled & 1) && again == 0;
        std::printf("%s latch (card default, validation off): rc %d/%d, source %d, engine fold 0x%x, library poll 0x%x, then 0x%x%s\n", ok ? "PASS" : "FAIL", rc1, rc2, has,
                    (unsigned)folded, (unsigned)polled, (unsigned)again, kept1 ? "" : ", WRONG BYTES");
        ok ? ++g_pass : ++g_fail;
        unsetenv("R4D_ERNIE_CARD_FORM");
        unsetenv("R4D_ERNIE_CARD_DEFAULTS");
    }
}

int main(int argc, char** argv) {
    const char* ref_path = argc > 1 ? argv[1] : "/src/build-ernie55/libref.so";
    if (!load_libref(ref_path)) return 2;
    const Geo geos[] = {
        {"release L512 R64 qb128 coop (M<=8, W>=64)", 4, 512, 64, 128, 8, 6, 80, 3, 0.25f},
        {"release L512 R64 qb128 wave (M>8)", 5, 512, 64, 128, 8, 4, 40, 9, -INFINITY},
        {"L256 R64 qb64 coop", 3, 256, 64, 64, 4, 5, 70, 2, -1.0f},
        {"L256 R64 qb32 coop (row 416 B: fast)", 3, 256, 64, 32, 4, 5, 70, 2, 0.5f},
        {"L256 R64 qb32 wave (row 416 B: fast)", 3, 256, 64, 32, 4, 4, 40, 9, -INFINITY},
        {"L256 R64 qb128 coop (row 392 B: general)", 3, 256, 64, 128, 4, 5, 70, 2, 0.25f},
        {"L256 R64 qb128 wave (row 392 B: general)", 3, 256, 64, 128, 4, 4, 40, 9, -1.0f},
        {"L64 R16 qb64 general kernel", 2, 64, 16, 64, 4, 5, 9, 3, 0.5f},
        {"L64 R16 qb16 general kernel", 2, 64, 16, 16, 4, 5, 9, 3, -INFINITY},
    };
    for (const auto& g : geos) run_geo(g);
    edge_cases();
    std::printf("\nSUMMARY: pass=%d fail=%d\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
