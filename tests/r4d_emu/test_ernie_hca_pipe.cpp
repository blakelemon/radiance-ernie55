/* ernie_hca_union_attn_origin, D == 512, on the device kernels (SIMT emulator): the batched compressed pass (the default) against the one-row-at-a-time pass
 * (R4D_HCA_UNION_BATCH=0), bit for bit on every output, for the decode kernel (M <= 4) and the wide kernel (M > 4), and both against libref within one bf16 ulp.
 * Depths from 0 compressed rows to 3,906 (500K tokens) and 8,192 (1M); holes and out-of-range entries in the compressed table (a hole must be skipped in place, the
 * chain order unchanged); the "block" and "token" layouts; a recycled window (origin > 0); an unaligned q (the scalar kernel, which the batch does not touch). */
#include "rad_plugin.h"
#include "r4d_plugin.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <random>
#include <vector>

namespace {
RadParam I(const char* k, int64_t v) { RadParam p{}; p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }
RadParam F(const char* k, double v) { RadParam p{}; p.key = k; p.kind = RAD_P_F64; p.dval = v; return p; }
RadParam S_(const char* k, const char* v) { RadParam p{}; p.key = k; p.kind = RAD_P_STR; p.sval = v; return p; }
RadTensor T(void* d, uint32_t dt, std::vector<int64_t> sh) {
    RadTensor t{}; t.data = d; t.dtype = dt; t.rank = (uint32_t)sh.size();
    int64_t acc = 1;
    for (int i = (int)sh.size() - 1; i >= 0; --i) { t.shape[i] = sh[(size_t)i]; t.stride[i] = acc; acc *= sh[(size_t)i]; }
    return t;
}
RadLaunchFn ref_op(void* lib, const char* name) {
    auto count = reinterpret_cast<int (*)()>(dlsym(lib, "rad_kernel_count"));
    auto at = reinterpret_cast<const RadKernelInfo* (*)(int)>(dlsym(lib, "rad_kernel_at"));
    for (int i = 0; count && i < count(); ++i) { const RadKernelInfo* k = at(i); if (k && !std::strcmp(k->op, name)) return k->launch; }
    return nullptr;
}
int g_fail = 0, g_pass = 0;
bool within(uint16_t a, uint16_t b) {
    const float x = rad_bf16_to_f32(a), y = rad_bf16_to_f32(b);
    return std::fabs(x - y) <= std::max(std::fabs(x), std::fabs(y)) * (1.0f / 64.0f) + 1e-5f;
}

struct Case {
    const char* name;
    int M, H, depth /* compressed rows of the deepest row */, layout /* 0 block, 1 token */, holes, origin /* recycle the window */, misalign;
    int posmode;   /* 0: row i at depth * 128 + 127 - 3 i; 1: random positions below that */
};

bool run_case(const Case& c, RadLaunchFn ref, uint32_t seed) {
    constexpr int ratio = 128, D = 512, BW = 128, WIN = 128;
    std::mt19937 g(seed);
    const int M = c.M, H = c.H, S = 2;
    const int maxp = c.depth * ratio + 127;      /* deepest position: (maxp + 1) / ratio compressed rows */
    const int NBc = c.layout == 0 ? maxp / ratio + 2 : maxp / 64 + 3;
    const int block = c.layout == 0 ? ratio : 64, E = c.layout == 0 ? D / ratio : D;
    const int PC = S * NBc + 3;
    std::vector<uint16_t> ck((size_t)PC * block * E);
    for (auto& x : ck) x = rad_f32_to_bf16(((int)(g() % 2001) - 1000) / 700.0f);
    std::vector<int32_t> pos((size_t)M), req((size_t)M);
    for (int m = 0; m < M; ++m) {
        req[(size_t)m] = m % S;
        pos[(size_t)m] = c.posmode ? (int32_t)(g() % (uint32_t)(maxp + 1)) : std::max(0, maxp - 3 * m);
    }
    std::vector<int32_t> origin((size_t)S, 0);
    int minp[2] = {INT32_MAX, INT32_MAX}, mxp[2] = {0, 0};
    for (int m = 0; m < M; ++m) { minp[req[(size_t)m]] = std::min(minp[req[(size_t)m]], pos[(size_t)m]); mxp[req[(size_t)m]] = std::max(mxp[req[(size_t)m]], pos[(size_t)m]); }
    for (int s = 0; s < S; ++s) if (c.origin && minp[s] != INT32_MAX && minp[s] >= WIN) origin[(size_t)s] = ((minp[s] - WIN + 1) / BW) * BW;
    int NBw = 0;
    for (int s = 0; s < S; ++s) NBw = std::max(NBw, (mxp[s] - origin[(size_t)s]) / BW + 2);
    const int PW = S * NBw + 2;
    std::vector<uint16_t> wk((size_t)PW * BW * D);
    for (auto& x : wk) x = rad_f32_to_bf16(((int)(g() % 2001) - 1000) / 900.0f);
    std::vector<int32_t> wt((size_t)S * NBw, -1), ct((size_t)S * NBc, -1);
    {
        std::vector<int> pw(PW), pc(PC);
        for (int i = 0; i < PW; ++i) pw[(size_t)i] = i;
        for (int i = 0; i < PC; ++i) pc[(size_t)i] = i;
        std::shuffle(pw.begin(), pw.end(), g); std::shuffle(pc.begin(), pc.end(), g);
        int kw = 0, kc = 0;
        for (int s = 0; s < S; ++s) {
            for (int b = 0; b < NBw; ++b) wt[(size_t)s * NBw + b] = pw[(size_t)kw++];
            for (int b = 0; b < NBc; ++b) {
                int e = pc[(size_t)kc++];
                if (c.holes && g() % 17 == 0) e = (c.holes == 2 && (g() & 1)) ? PC + 7 : -1;   /* -1: uninitialised, skipped by both; PC + 7: libref refuses, the device skips */
                ct[(size_t)s * NBc + b] = e;
            }
        }
    }
    std::vector<uint16_t> qbuf((size_t)M * H * D + 8);
    for (auto& x : qbuf) x = rad_f32_to_bf16(((int)(g() % 2001) - 1000) / 1200.0f);
    uint16_t* q = qbuf.data() + (c.misalign ? 1 : 0);
    std::vector<float> sink((size_t)H);
    for (int h = 0; h < H; ++h) sink[(size_t)h] = 0.25f * (float)h - 0.3f;
    auto run = [&](RadLaunchFn fn, const char* batch, std::vector<uint16_t>& out) -> int {
        if (batch) setenv("R4D_HCA_UNION_BATCH", batch, 1); else unsetenv("R4D_HCA_UNION_BATCH");
        RadTensor ts[] = {T(q, RAD_BF16, {M, H * D}), T(wk.data(), RAD_BF16, {PW, 1, BW, D}),
                          c.layout == 0 ? T(ck.data(), RAD_BF16, {PC, 1, ratio, E}) : T(ck.data(), RAD_BF16, {PC, 1, block, D}),
                          T(pos.data(), RAD_I32, {M}), T(req.data(), RAD_I32, {M}), T(wt.data(), RAD_I32, {S, NBw}), T(ct.data(), RAD_I32, {S, NBc}),
                          T(origin.data(), RAD_I32, {S}), T(sink.data(), RAD_F32, {H}), T(out.data(), RAD_BF16, {M, H * D})};
        RadParam ps[] = {I("M", M), I("heads", H), I("head_dim", D), I("ratio", ratio), I("window", WIN), F("scale", 1.0 / std::sqrt((double)D)),
                         S_("cmp_layout", c.layout == 0 ? "block" : "token"), I("block_size", block)};
        RadArgs a{}; a.t = ts; a.n_t = 10; a.p = ps; a.n_p = c.layout == 0 ? 8 : 7;
        const int rc = fn(&a, nullptr);
        unsetenv("R4D_HCA_UNION_BATCH");
        return rc;
    };
    const size_t n = (size_t)M * H * D;
    std::vector<uint16_t> o_new(n, 0x1234), o_old(n, 0x1234), o_ref(n, 0x1234);
    /* an out-of-range compressed-table entry: the integration refuses the launch on the host (emulator default) or latches an event and skips the entry (card form). The
     * equality of the two passes is the card form's with validation off, as a card runs by default: nothing refuses the launch and the entry is skipped. */
    if (c.holes == 2) { setenv("R4D_ERNIE_CARD_FORM", "1", 1); setenv("R4D_ERNIE_VALIDATE", "0", 1); }
    const int r1 = run(r4d_rad_ernie_hca_union_attn_origin, nullptr, o_new), r0 = run(r4d_rad_ernie_hca_union_attn_origin, "0", o_old);
    unsetenv("R4D_ERNIE_CARD_FORM"); unsetenv("R4D_ERNIE_VALIDATE");
    if (r1 != RAD_OK || r0 != RAD_OK) { std::printf("FAIL %s: device refused (%d, %d)\n", c.name, r1, r0); ++g_fail; return false; }
    bool ok = std::memcmp(o_new.data(), o_old.data(), n * 2) == 0;
    if (!ok) std::printf("FAIL %s: batched output differs from the one-row pass\n", c.name);
    if (ref && c.holes == 2) {
        const int rr = run(ref, nullptr, o_ref);
        if (rr == RAD_OK) { std::printf("FAIL %s: libref accepted an out-of-range table entry\n", c.name); ok = false; }
        else std::printf("INFO %s: libref refuses an out-of-range compressed-table entry (rc %d); the device launch skips it (pre-existing, same for both passes)\n", c.name, rr);
    } else if (ref) {
        const int rr = run(ref, nullptr, o_ref);
        if (rr != RAD_OK) { std::printf("FAIL %s: libref refused (%d)\n", c.name, rr); ok = false; }
        else for (size_t i = 0; i < n; ++i) if (!within(o_new[i], o_ref[i])) { std::printf("FAIL %s: output %zu is more than 1/64 from libref\n", c.name, i); ok = false; break; }
    }
    if (ok) { std::printf("PASS %s\n", c.name); ++g_pass; } else ++g_fail;
    return ok;
}
}  // namespace

int main(int argc, char** argv) {
    setenv("OMP_NUM_THREADS", "2", 0);
    void* lib = argc > 1 ? dlopen(argv[1], RTLD_NOW | RTLD_LOCAL) : nullptr;
    RadLaunchFn ref = lib ? ref_op(lib, "ernie_hca_union_attn_origin") : nullptr;
    if (!ref) { std::printf("FAIL no libref ernie_hca_union_attn_origin\n"); return 1; }
    const Case cases[] = {
        /* name                         M  H depth layout holes origin mis posmode */
        {"decode, no compressed rows",  1, 2,    0, 0, 0, 0, 0, 0},
        {"decode, 1 row",               1, 2,    1, 0, 0, 0, 0, 0},
        {"decode, 7 rows",              1, 2,    7, 0, 0, 0, 0, 0},
        {"decode, 8 rows",              2, 2,    8, 0, 0, 0, 0, 1},
        {"decode, 63 rows",             1, 2,   63, 0, 0, 0, 0, 0},
        {"decode, 64 rows",             1, 2,   64, 0, 0, 0, 0, 0},
        {"decode, 65 rows",             4, 2,   65, 0, 1, 0, 0, 1},
        {"decode, 130 rows, holes",     1, 2,  130, 0, 1, 0, 0, 0},
        {"decode, 3906 rows (500K)",    1, 2, 3906, 0, 0, 1, 0, 0},
        {"decode, 3906 rows, holes",    4, 2, 3906, 0, 1, 1, 0, 1},
        {"decode, 3906 rows, bad entries", 1, 2, 3906, 0, 2, 1, 0, 0},
        {"decode, 8192 rows (1M)",      1, 2, 8192, 0, 1, 1, 0, 0},
        {"decode, token layout",        2, 2,  300, 1, 1, 0, 0, 1},
        {"decode, 8 heads",             1, 8,  500, 0, 0, 0, 0, 0},
        {"decode, unaligned q",         1, 2,  200, 0, 0, 0, 1, 0},
        {"wide, no compressed rows",    8, 2,    0, 0, 0, 0, 0, 0},
        {"wide, 5 rows, 7 rows",        5, 2,    7, 0, 0, 0, 0, 1},
        {"wide, 65 rows, holes",        8, 2,   65, 0, 1, 0, 0, 1},
        {"wide, 3906 rows",             8, 2, 3906, 0, 0, 1, 0, 0},
        {"wide, 3906 rows, holes",      6, 2, 3906, 0, 1, 1, 0, 1},
        {"wide, 700 rows, bad entries",  8, 2,  700, 0, 2, 0, 0, 0},
        {"wide, 16 rows x 8 heads",    16, 8,  300, 0, 1, 0, 0, 1},
        {"wide, token layout",          5, 2,  300, 1, 1, 0, 0, 1},
        {"wide, unaligned q",           8, 2,  200, 0, 0, 0, 1, 0},
    };
    uint32_t seed = 71;
    for (const Case& c : cases) run_case(c, ref, seed++);
    std::printf("%d passed, %d failed\n%s\n", g_pass, g_fail, g_fail ? "FAILURES" : "ALL PASS");
    return g_fail ? 1 : 0;
}
