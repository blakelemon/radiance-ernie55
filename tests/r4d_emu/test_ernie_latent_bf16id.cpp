/* tests/r4d_emu/test_ernie_latent_bf16id.cpp -- kv-fp8-latent v2 DIAGNOSTIC: the identity latent row bf16_ds_mla (selector RADIANCE_ERNIE55_KV_LATENT_FMT=bf16id)
 * carries the bf16 latent losslessly through the quantised-row plumbing (u8 group, ernie_mla_latent_store, the kv_layout readers). Run by the SIMT emulator:
 *   store    device bytes = libref bytes; every latent value's bf16 bits are in the row (high byte plane, low byte plane), scales 1.0, rope copied
 *   attn     at each geometry (fast coop / wave kernels, the general kernel, an unaligned cache) the bf16_ds_mla reader's output is BIT-IDENTICAL to the
 *            bf16-row reader's output on the same values (device and libref): the quantised path itself adds nothing
 *   flip     RADIANCE_DIAG_ID_FLIP=1 flips the low bit of each tile's first value on device and in libref alike (the 1-ulp perturbation arm)
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
static int g_pass = 0, g_fail = 0;
static void verdict(const std::string& name0, bool ok, const char* detail = "") {
    const std::string& name = name0;
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

static int run_store(RadLaunchFn fn, int L, int R, int qb, int bs, int pages, std::vector<uint16_t>& src, std::vector<int32_t>& slots, std::vector<uint8_t>& cache) {
    const int M = (int)slots.size(), rowb = 2 * L + 4 * (L / qb) + 2 * R;
    RadParam p[6] = {pi("M", M), pi("latent_dim", L), pi("rope_dim", R), pi("block_size", bs), pi("quant_block", qb), ps("kv_layout", "bf16_ds_mla")};
    RadTensor t[3] = {t2(src.data(), RAD_BF16, M, L + R), t1(slots.data(), RAD_I32, M), t4(cache.data(), RAD_U8, pages, 1, bs, rowb)};
    RadArgs a{t, 3, p, 6};
    return fn(&a, nullptr);
}
struct Geo { const char* name; int H, L, R, qb, bs, pages, W, M; float sink0; };
static void run_geo(const Geo& g) {
    const int rowb = 2 * g.L + 4 * (g.L / g.qb) + 2 * g.R, n = g.pages * g.bs, w = g.L + g.R;
    std::vector<uint16_t> src((size_t)n * w);
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < w; ++c) src[(size_t)r * w + c] = f2b(rnd() * 0.9f * std::pow(6.0f, (float)((c / 37 + r) % 4)) * (c == 5 ? 1e-30f : 1.0f));
    std::vector<int32_t> slots(n);
    for (int s = 0; s < n; ++s) slots[s] = (s * 7 + 3) % n;
    slots[1] = -1;
    std::vector<uint8_t> cdev((size_t)n * rowb, 0xA5), cref(cdev);
    if (run_store(r4d_rad_ernie_mla_latent_store, g.L, g.R, g.qb, g.bs, g.pages, src, slots, cdev) != RAD_OK ||
        run_store(g_ref["ernie_mla_latent_store"], g.L, g.R, g.qb, g.bs, g.pages, src, slots, cref) != RAD_OK) { verdict(std::string(g.name) + " store launch", false); return; }
    check_exact(std::string(g.name) + " bf16_ds_mla store bytes (device = libref)", cdev, cref);
    /* the bf16 cache the bf16 path would hold: the same values at the same slots; unwritten rows get the decoded 0xA5 filler of the u8 cache */
    std::vector<uint16_t> cbf((size_t)n * w);
    int lossless = 1;
    for (int s = 0; s < n; ++s) {
        const uint8_t* row = cdev.data() + (size_t)s * rowb;
        for (int c = 0; c < g.L; ++c) cbf[(size_t)s * w + c] = (uint16_t)((row[c] << 8) | row[g.L + c]);
        for (int c = 0; c < g.R; ++c) cbf[(size_t)s * w + g.L + c] = (uint16_t)(row[2 * g.L + 4 * (g.L / g.qb) + 2 * c] | (row[2 * g.L + 4 * (g.L / g.qb) + 2 * c + 1] << 8));
    }
    for (int m = 0; m < n; ++m) if (slots[m] >= 0) for (int c = 0; c < w; ++c) lossless &= cbf[(size_t)slots[m] * w + c] == src[(size_t)m * w + c];
    for (int s = 0; s < n; ++s) { const uint8_t* row = cdev.data() + (size_t)s * rowb; float sc; std::memcpy(&sc, row + 2 * g.L, 4); bool hit = false; for (int m = 0; m < n; ++m) hit |= slots[m] == s; if (hit) lossless &= sc == 1.0f; }
    verdict(std::string(g.name) + " every written value is its bf16, scales 1.0", lossless);
    std::vector<int32_t> sl((size_t)g.M * g.W);
    for (auto& s : sl) { const float u = rnd(); s = u < -0.6f ? -1 : (int)((u + 1.0f) * 0.5f * (float)(n - 1)); }
    std::vector<uint16_t> q((size_t)g.M * g.H * g.L), qr((size_t)g.M * g.H * g.R);
    for (auto& e : q) e = f2b(rnd()); for (auto& e : qr) e = f2b(rnd());
    std::vector<float> sink(g.H);
    for (int h = 0; h < g.H; ++h) sink[h] = g.sink0 == -INFINITY ? g.sink0 : g.sink0 + 0.2f * h;
    std::vector<uint8_t> cmis(cdev.size() + 32);
    uint8_t* mis = cmis.data() + (16 - ((uintptr_t)cmis.data() % 16)) % 16 + 2;
    std::memcpy(mis, cdev.data(), cdev.size());
    const size_t no = q.size();
    std::vector<uint16_t> oid(no), obf(no), omis(no), obfmis(no), orid(no), orbf(no);
    std::vector<uint16_t> cbfmis(cbf.size() + 16);
    uint16_t* bmis = cbfmis.data() + ((16 - ((uintptr_t)cbfmis.data() % 16)) % 16) / 2 + 1;   /* 2 bytes off a 16-byte boundary: the general kernel */
    std::memcpy(bmis, cbf.data(), cbf.size() * 2);
    RadParam pid[10] = {pi("M", g.M), pi("heads", g.H), pi("latent_dim", g.L), pi("rope_dim", g.R), pi("block_size", g.bs), pi("width", g.W), pf("scale", 0.125),
                        ps("kv_layout", "bf16_ds_mla"), pi("quant_block", g.qb), pi("num_pages", g.pages)};
    RadParam pbf[8] = {pid[0], pid[1], pid[2], pid[3], pid[4], pid[5], pid[6], pid[9]};
    const RadTensor tq = t2(q.data(), RAD_BF16, g.M, g.H * g.L), tqr = t2(qr.data(), RAD_BF16, g.M, g.H * g.R), tsl = t2(sl.data(), RAD_I32, g.M, g.W), tsk = t1(sink.data(), RAD_F32, g.H);
    RadTensor aid[6] = {tq, tqr, t4(cdev.data(), RAD_U8, g.pages, 1, g.bs, rowb), tsl, tsk, t2(oid.data(), RAD_BF16, g.M, g.H * g.L)};
    RadTensor amis[6] = {tq, tqr, t4(mis, RAD_U8, g.pages, 1, g.bs, rowb), tsl, tsk, t2(omis.data(), RAD_BF16, g.M, g.H * g.L)};
    RadTensor abf[6] = {tq, tqr, t4(cbf.data(), RAD_BF16, g.pages, 1, g.bs, w), tsl, tsk, t2(obf.data(), RAD_BF16, g.M, g.H * g.L)};
    RadTensor abfmis[6] = {tq, tqr, t4(bmis, RAD_BF16, g.pages, 1, g.bs, w), tsl, tsk, t2(obfmis.data(), RAD_BF16, g.M, g.H * g.L)};
    RadTensor arid[6] = {aid[0], aid[1], aid[2], aid[3], aid[4], t2(orid.data(), RAD_BF16, g.M, g.H * g.L)};
    RadTensor arbf[6] = {abf[0], abf[1], abf[2], abf[3], abf[4], t2(orbf.data(), RAD_BF16, g.M, g.H * g.L)};
    RadArgs xid{aid, 6, pid, 10}, xmis{amis, 6, pid, 10}, xbf{abf, 6, pbf, 8}, xbfmis{abfmis, 6, pbf, 8}, xrid{arid, 6, pid, 10}, xrbf{arbf, 6, pbf, 8};
    if (r4d_rad_ernie_mla_sparse_attn(&xid, nullptr) != RAD_OK || r4d_rad_ernie_mla_sparse_attn(&xmis, nullptr) != RAD_OK || r4d_rad_ernie_mla_sparse_attn(&xbf, nullptr) != RAD_OK || r4d_rad_ernie_mla_sparse_attn(&xbfmis, nullptr) != RAD_OK ||
        g_ref["ernie_mla_sparse_attn"](&xrid, nullptr) != RAD_OK || g_ref["ernie_mla_sparse_attn"](&xrbf, nullptr) != RAD_OK) { verdict(std::string(g.name) + " reader launches", false); return; }
    const bool fast = g.R == 64 && (g.L == 512 || g.L == 256) && g.qb % 8 == 0 && rowb % 16 == 0;
    check_exact(std::string(g.name) + (fast ? " device bf16_ds_mla (fast) = device bf16 reader, bit for bit" : " device bf16_ds_mla (general) = device bf16 reader, bit for bit"), oid, obf);
    check_exact(std::string(g.name) + " device bf16_ds_mla (general, unaligned) = device bf16 reader (general, unaligned), bit for bit", omis, obfmis);
    check_ulp(std::string(g.name) + " device bf16_ds_mla general vs fast (1-ulp band)", omis, oid, 1, 2e-3f);
    check_exact(std::string(g.name) + " libref bf16_ds_mla = libref bf16 reader, bit for bit", orid, orbf);
    /* the flip arm: device and libref flip the same bit */
    setenv("RADIANCE_DIAG_ID_FLIP", "1", 1);
    std::vector<uint8_t> fdev((size_t)n * rowb, 0xA5), fref(fdev);
    const int r1 = run_store(r4d_rad_ernie_mla_latent_store, g.L, g.R, g.qb, g.bs, g.pages, src, slots, fdev), r2 = run_store(g_ref["ernie_mla_latent_store"], g.L, g.R, g.qb, g.bs, g.pages, src, slots, fref);
    unsetenv("RADIANCE_DIAG_ID_FLIP");
    int nd = 0; for (size_t i = 0; i < fdev.size(); ++i) nd += fdev[i] != cdev[i];
    check_exact(std::string(g.name) + " flip arm: device bytes = libref bytes", fdev, fref);
    char d[64]; std::snprintf(d, sizeof d, "changed bytes=%d", nd);
    int written = 0; for (int m = 0; m < n; ++m) written += slots[m] >= 0;
    verdict(std::string(g.name) + " flip arm changes exactly one byte per tile of each written row", r1 == RAD_OK && r2 == RAD_OK && nd == written * (g.L / g.qb), d);
}
int main(int argc, char** argv) {
    const char* ref_path = argc > 1 ? argv[1] : "/src/build-ernie55/libref.so";
    if (!load_libref(ref_path)) return 2;
    const Geo geos[] = {
        {"release L512 R64 qb128 coop", 4, 512, 64, 128, 8, 6, 80, 3, 0.25f},
        {"release L512 R64 qb128 wave", 5, 512, 64, 128, 8, 4, 40, 9, -INFINITY},
        {"release L512 R64 qb128 coop W2176", 2, 512, 64, 128, 64, 40, 2176, 2, 0.5f},
        {"release L512 R64 qb128 wave W2176", 4, 512, 64, 128, 64, 40, 2176, 12, -1.0f},
        {"L256 R64 qb128 coop", 3, 256, 64, 128, 4, 5, 70, 2, 0.5f},
        {"L64 R16 qb64 general kernel", 2, 64, 16, 64, 4, 5, 9, 3, -INFINITY},
    };
    for (const auto& g : geos) run_geo(g);
    std::printf("\nSUMMARY: pass=%d fail=%d\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
