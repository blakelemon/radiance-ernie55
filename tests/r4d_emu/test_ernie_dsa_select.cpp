/* ernie_dsa_select on the device kernels (SIMT emulator): the single-block kernel and the multi-block form (R4D_DSA_SELECT_BLOCKS = 0 / n), each against libref's ernie_dsa_select,
 * bit for bit on ids and slots (the whole output buffers, so a row the op must not touch is compared too). Adversarial score planes: ties, all equal, N < k, N not a
 * multiple of the block or slice size, -inf holes, uninitialised pages, signed zeros, no candidates, N up to 1,048,576. Where libref refuses (NaN / +inf in range) the
 * device forms are compared with each other. Launcher refusals: every malformed launch libref refuses is refused by the device launcher with the outputs untouched.
 * Built with the .hip kernels as C++ (RAD_TEST_HIP_EMU). */
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

enum Mode { RANDOM, QUANT16, ALLEQ, HOLES, SIGNED0, TIES_AT_K, NONE, NONFINITE, TWO_LEVELS, DESC_RUN };
struct Case {
    const char* name;
    int L, topk, window, bs, rows;
    Mode mode;
    int page_holes;       /* 0 none, 1 -1 entries, 2 entries >= num_pages, 3 both */
    const char* blocks;   /* nb variants to run, comma separated ("0" the single-block kernel) */
    int ends[4];          /* candidate end of each row (<= L); -1 = random */
};

bool same(const std::vector<int32_t>& a, const std::vector<int32_t>& b) { return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0; }

bool run_case(const Case& c, RadLaunchFn ref, uint32_t seed) {
    std::mt19937 g(seed);
    const int M = c.rows, S = 2, W = c.topk + c.window;
    const int B = (c.L + c.window + c.bs - 1) / c.bs + 2, pages = S * B + 3;
    std::vector<float> sc((size_t)M * c.L);
    for (int m = 0; m < M; ++m)
        for (int i = 0; i < c.L; ++i) {
            float v = 0.0f;
            switch (c.mode) {
            case RANDOM: v = (float)(g() % 1000003) / 1000003.0f; break;
            case QUANT16: v = (float)(g() % 16) * 0.125f; break;
            case ALLEQ: v = 0.75f; break;
            case HOLES: v = (g() % 10 < 3) ? -INFINITY : (float)(g() % 4000) / 997.0f; break;
            case SIGNED0: { const uint32_t r = g() % 6; v = r == 0 ? 0.0f : r == 1 ? -0.0f : r == 2 ? -1.5f : r == 3 ? 1e-40f : r == 4 ? -1e-40f : 3e38f; break; }
            case TIES_AT_K: v = (i % 7 == 0) ? 2.0f : (i % 3 == 0 ? 1.0f : 0.5f); break;
            case NONE: v = -INFINITY; break;
            case NONFINITE: { const uint32_t r = g() % 50; v = r == 0 ? NAN : r == 1 ? INFINITY : (float)(g() % 100) / 7.0f; break; }
            case TWO_LEVELS: v = (g() % 100) < 2 ? 5.0f : 1.0f; break;
            case DESC_RUN: v = (float)(c.L - i); break;
            }
            sc[(size_t)m * c.L + i] = v;
        }
    std::vector<int32_t> bt((size_t)S * B), perm((size_t)S * B), req((size_t)M), pos((size_t)M);
    for (size_t i = 0; i < perm.size(); ++i) perm[i] = (int32_t)i;
    std::shuffle(perm.begin(), perm.end(), g);
    for (size_t i = 0; i < bt.size(); ++i) {
        bt[i] = perm[i];
        const uint32_t r = g() % 8;
        if ((c.page_holes & 1) && r == 0) bt[i] = -1 - (int32_t)(g() % 3);
        if ((c.page_holes & 2) && r == 1) bt[i] = pages + (int32_t)(g() % 5);
    }
    for (int m = 0; m < M; ++m) {
        req[(size_t)m] = m % S;
        int e = c.ends[m % 4] >= 0 ? std::min(c.ends[m % 4], c.L) : (int)(g() % (uint32_t)(c.L + 1));
        pos[(size_t)m] = e + c.window - 1;   /* end = p + 1 - window */
    }
    auto launch = [&](RadLaunchFn fn, std::vector<int32_t>& ids, std::vector<int32_t>& sl) -> int {
        RadTensor ts[] = {T(sc.data(), RAD_F32, {M, c.L}), T(pos.data(), RAD_I32, {M}), T(req.data(), RAD_I32, {M}), T(bt.data(), RAD_I32, {S, B}),
                          T(ids.data(), RAD_I32, {M, W}), T(sl.data(), RAD_I32, {M, W})};
        RadParam ps[] = {I("M", M), I("topk", c.topk), I("window", c.window), I("block_size", c.bs), I("num_pages", pages)};
        RadArgs a{}; a.t = ts; a.n_t = 6; a.p = ps; a.n_p = 5;
        return fn(&a, nullptr);
    };
    const std::vector<int32_t> fill((size_t)M * W, 0x5a5a5a5a);
    std::vector<int32_t> rid = fill, rsl = fill;
    int rrc = -1;
    if (ref) rrc = launch(ref, rid, rsl);
    const bool ref_ok = ref && rrc == RAD_OK;
    if (c.mode == NONFINITE) {
        if (ref && rrc == RAD_OK) { std::printf("FAIL %s: libref accepted a NaN / +inf score\n", c.name); ++g_fail; return false; }
    } else if (ref && !ref_ok) { std::printf("FAIL %s: libref refused (rc %d)\n", c.name, rrc); ++g_fail; return false; }
    /* Two modes, as the integration has them. Validation ON (the emulator default, R4D_ERNIE_VALIDATE=1 on a card): the single-block kernel refuses a row with a NaN / +inf score in
     * [0, end) (stores nothing, raises an event) whatever R4D_DSA_SELECT_BLOCKS says. Validation OFF (a card's default): every form does what base d2258ef8 did, a NaN is no candidate and
     * +inf the largest key; the reference is libref on the plane with NaN -> -inf and +inf -> 3e38 (the same candidates, the same order keys' relative order). */
    std::vector<float> sc2 = sc;
    for (float& v : sc2) { if (v != v) v = -INFINITY; else if (v == INFINITY) v = 3e38f; }
    std::vector<int32_t> rid2 = fill, rsl2 = fill;
    bool ref2_ok = false;
    if (c.mode == NONFINITE) {
        sc.swap(sc2);   /* launch() reads sc */
        ref2_ok = ref && launch(ref, rid2, rsl2) == RAD_OK;
        sc.swap(sc2);
    }
    std::vector<char> row_bad((size_t)M, 0);   /* a NaN / +inf anywhere in [0, end) of the row (libref scans before the page filter) */
    for (int m = 0; m < M; ++m) {
        const int end = std::max(pos[(size_t)m] + 1 - c.window, 0);
        for (int i = 0; i < end; ++i) { const float v = sc[(size_t)m * c.L + i]; if (v != v || v == INFINITY) { row_bad[(size_t)m] = 1; break; } }
    }
    bool ok = true;
    for (int strict = 1; strict >= 0; --strict) {
        setenv("R4D_ERNIE_VALIDATE", strict ? "1" : "0", 1);
        std::vector<int32_t> base_id, base_sl;
        std::string list = c.blocks;
        size_t pos0 = 0;
        while (pos0 <= list.size()) {
            size_t q = list.find(',', pos0);
            if (q == std::string::npos) q = list.size();
            const std::string nbs = list.substr(pos0, q - pos0);
            pos0 = q + 1;
            setenv("R4D_DSA_SELECT_BLOCKS", nbs.c_str(), 1);
            std::vector<int32_t> did = fill, dsl = fill;
            const int rc = launch(r4d_rad_ernie_dsa_select, did, dsl);
            if (rc != RAD_OK) { std::printf("FAIL %s [validate %d, blocks %s]: device refused (rc %d)\n", c.name, strict, nbs.c_str(), rc); ok = false; continue; }
            if (c.mode != NONFINITE) {
                if (ref_ok && (!same(did, rid) || !same(dsl, rsl))) { std::printf("FAIL %s [validate %d, blocks %s]: device differs from libref (%s)\n", c.name, strict, nbs.c_str(), same(did, rid) ? "slots" : "ids"); ok = false; }
            } else if (strict) {
                for (int m = 0; m < M; ++m) {
                    const bool untouched = std::all_of(did.begin() + (long)m * W, did.begin() + (long)(m + 1) * W, [](int32_t x) { return x == 0x5a5a5a5a; }) &&
                                           std::all_of(dsl.begin() + (long)m * W, dsl.begin() + (long)(m + 1) * W, [](int32_t x) { return x == 0x5a5a5a5a; });
                    if (row_bad[(size_t)m] != (char)untouched) { std::printf("FAIL %s [validate 1, blocks %s]: row %d %s\n", c.name, nbs.c_str(), m, row_bad[(size_t)m] ? "stored although it holds a NaN / +inf" : "was not stored"); ok = false; }
                }
            } else if (ref2_ok && (!same(did, rid2) || !same(dsl, rsl2))) {
                std::printf("FAIL %s [validate 0, blocks %s]: differs from libref on the plane with NaN -> -inf, +inf -> 3e38 (the base behaviour)\n", c.name, nbs.c_str()); ok = false;
            }
            if (base_id.empty()) { base_id = did; base_sl = dsl; }
            else if (!same(did, base_id) || !same(dsl, base_sl)) { std::printf("FAIL %s [validate %d, blocks %s]: differs from [blocks %s]\n", c.name, strict, nbs.c_str(), c.blocks); ok = false; }
        }
    }
    unsetenv("R4D_DSA_SELECT_BLOCKS");
    unsetenv("R4D_ERNIE_VALIDATE");
    if (ok) { std::printf("PASS %s\n", c.name); ++g_pass; } else ++g_fail;
    return ok;
}

/* rows the select must not touch (bad position, bad request, candidate range past the plane) beside a good one: every form, both modes, same bytes, the bad rows untouched */
void invalid_row_case() {
    const int L = 20000, M = 4, S = 2, window = 128, topk = 2048, bs = 64, W = topk + window, B = (L + window) / bs + 2, pages = S * B + 3;
    std::mt19937 g(5);
    std::vector<float> sc((size_t)M * L);
    for (float& v : sc) v = (float)(g() % 1000003) / 1000003.0f;
    std::vector<int32_t> bt((size_t)S * B), req = {0, 1, 7, 0}, pos = {L, -5, L, L + window + 40};
    for (size_t i = 0; i < bt.size(); ++i) bt[i] = (int32_t)i;
    std::vector<int32_t> first;
    bool ok = true;
    for (int strict = 1; strict >= 0; --strict)
        for (const char* nbs : {"0", "64", "5"}) {
            setenv("R4D_ERNIE_VALIDATE", strict ? "1" : "0", 1);
            setenv("R4D_DSA_SELECT_BLOCKS", nbs, 1);
            std::vector<int32_t> ids((size_t)M * W, 0x5a5a5a5a), sl = ids;
            RadTensor ts[] = {T(sc.data(), RAD_F32, {M, L}), T(pos.data(), RAD_I32, {M}), T(req.data(), RAD_I32, {M}), T(bt.data(), RAD_I32, {S, B}), T(ids.data(), RAD_I32, {M, W}), T(sl.data(), RAD_I32, {M, W})};
            RadParam ps[] = {I("M", M), I("topk", topk), I("window", window), I("block_size", bs), I("num_pages", pages)};
            RadArgs a{}; a.t = ts; a.n_t = 6; a.p = ps; a.n_p = 5;
            const int rc = r4d_rad_ernie_dsa_select(&a, nullptr);
            bool good = rc == RAD_OK;
            for (int m = 1; m < M; ++m)
                for (int j = 0; j < W; ++j) good = good && ids[(size_t)m * W + j] == 0x5a5a5a5a && sl[(size_t)m * W + j] == 0x5a5a5a5a;
            bool row0 = false;
            for (int j = 0; j < W; ++j) row0 = row0 || ids[(size_t)j] != 0x5a5a5a5a;
            std::vector<int32_t> both = ids; both.insert(both.end(), sl.begin(), sl.end());
            if (first.empty()) first = both; else good = good && same(both, first);
            if (!good || !row0) { std::printf("FAIL invalid rows [validate %d, blocks %s]: rc %d, bad rows untouched / forms equal / good row stored: %d\n", strict, nbs, rc, (int)good); ok = false; }
        }
    unsetenv("R4D_DSA_SELECT_BLOCKS");
    unsetenv("R4D_ERNIE_VALIDATE");
    if (ok) { std::printf("PASS invalid rows (bad position, request, range) untouched, every form, both modes\n"); ++g_pass; } else ++g_fail;
}

/* the production shape takes the multi-block path (and its fallbacks fall back), by the route the launcher recorded (r4d_ernie_dsa_select_last_nb) */
extern "C" int r4d_ernie_dsa_select_last_nb(void);
void route_cases() {
    const int L = 70000, M = 2, S = 1, window_p = 128, topk_p = 2048, bs = 64;
    struct R { const char* name; int validate; int topk, window; int expect_nb; };
    const R rs[] = {{"production shape (topk 2048, window 128: rows 2176 wide), validation off: multi-block", 0, topk_p, window_p, 64},
                    {"production shape, validation on: single-block", 1, topk_p, window_p, 0},
                    {"row narrower than 1024 + 2 nb (topk 64, window 100): single-block", 0, 64, 100, 0},
                    {"row of exactly 1024 + 2 nb (topk 1024, window 128 = 1152): multi-block", 0, 1024, 128, 64}};
    unsetenv("R4D_DSA_SELECT_BLOCKS");
    for (const R& r : rs) {
        setenv("R4D_ERNIE_VALIDATE", r.validate ? "1" : "0", 1);
        const int W = r.topk + r.window, B = (L + r.window) / bs + 2, pages = S * B + 3;
        std::vector<float> sc((size_t)M * L, 0.5f);
        std::vector<int32_t> bt((size_t)S * B), req = {0, 0}, pos = {60000, 50000};
        for (size_t i = 0; i < bt.size(); ++i) bt[i] = (int32_t)i;
        std::vector<int32_t> ids((size_t)M * W, 0x5a5a5a5a), sl = ids;
        RadTensor ts[] = {T(sc.data(), RAD_F32, {M, L}), T(pos.data(), RAD_I32, {M}), T(req.data(), RAD_I32, {M}), T(bt.data(), RAD_I32, {S, B}), T(ids.data(), RAD_I32, {M, W}), T(sl.data(), RAD_I32, {M, W})};
        RadParam ps[] = {I("M", M), I("topk", r.topk), I("window", r.window), I("block_size", bs), I("num_pages", pages)};
        RadArgs a{}; a.t = ts; a.n_t = 6; a.p = ps; a.n_p = 5;
        const int rc = r4d_rad_ernie_dsa_select(&a, nullptr);
        const int nb = r4d_ernie_dsa_select_last_nb();
        if (rc != RAD_OK || nb != r.expect_nb) { std::printf("FAIL route '%s': rc %d, nb %d (want %d)\n", r.name, rc, nb, r.expect_nb); ++g_fail; }
        else { ++g_pass; std::printf("PASS route '%s' (nb %d)\n", r.name, nb); }
    }
    unsetenv("R4D_ERNIE_VALIDATE");
}

/* launch refusals: libref refuses -> the device launcher refuses and stores nothing */
void refusal_cases(RadLaunchFn ref) {
    const int M = 2, L = 300, S = 2, bs = 8, window = 8, topk = 16, B = 50, pages = 200, W = topk + window;
    std::vector<float> sc((size_t)M * L, 0.5f);
    std::vector<int32_t> bt((size_t)S * B), req = {0, 1}, pos = {250, 200};
    for (size_t i = 0; i < bt.size(); ++i) bt[i] = (int32_t)i;
    struct Mut { const char* name; int what; };
    const Mut muts[] = {{"topk 0", 0}, {"window 0", 1}, {"block_size 0", 2}, {"ids narrower than topk + window", 3}, {"slots width differs from ids", 4}, {"tokens hint != L", 5},
                        {"seqs hint != S", 6}, {"blocks hint != B", 7}, {"scores dtype", 8}, {"M band 0", 9}, {"positions length", 10}, {"num_pages 0", 11},
                        {"scores rank 1", 12}, {"negative stride", 13}, {"null ids", 14}};
    for (const Mut& mu : muts) {
        for (const char* nbs : {"0", "3"}) {
            setenv("R4D_DSA_SELECT_BLOCKS", nbs, 1);
            std::vector<int32_t> ids((size_t)M * W, 0x5a5a5a5a), sl = ids, ids2 = ids, sl2 = ids;
            std::vector<int32_t> pos2 = pos;
            int64_t Wi = W, Ws = W, Mp = M, topk_ = topk, window_ = window, bs_ = bs, np = pages, band = M;
            RadTensor tsc = T(sc.data(), RAD_F32, {M, L}), tpos = T(pos2.data(), RAD_I32, {Mp}), treq = T(req.data(), RAD_I32, {M}), tbt = T(bt.data(), RAD_I32, {S, B}),
                      tid = T(ids.data(), RAD_I32, {M, Wi}), tsl = T(sl.data(), RAD_I32, {M, Ws});
            std::vector<RadParam> ps;
            int64_t tokens = L, seqs = S, blocks = B;
            bool hint_t = false, hint_s = false, hint_b = false;
            switch (mu.what) {
            case 0: topk_ = 0; break;
            case 1: window_ = 0; break;
            case 2: bs_ = 0; break;
            case 3: Wi = W - 1; Ws = W - 1; tid = T(ids.data(), RAD_I32, {M, Wi}); tsl = T(sl.data(), RAD_I32, {M, Ws}); break;
            case 4: tsl = T(sl.data(), RAD_I32, {M, W - 1}); break;
            case 5: hint_t = true; tokens = L + 1; break;
            case 6: hint_s = true; seqs = S + 1; break;
            case 7: hint_b = true; blocks = B + 1; break;
            case 8: tsc.dtype = RAD_BF16; break;
            case 9: band = 0; break;
            case 10: tpos = T(pos2.data(), RAD_I32, {1}); break;
            case 11: np = 0; break;
            case 12: tsc = T(sc.data(), RAD_F32, {(int64_t)M * L}); break;
            case 13: tpos.stride[0] = -1; break;
            case 14: tid.data = nullptr; break;
            }
            ps = {I("M", band), I("topk", topk_), I("window", window_), I("block_size", bs_), I("num_pages", np)};
            if (hint_t) ps.push_back(I("tokens", tokens));
            if (hint_s) ps.push_back(I("seqs", seqs));
            if (hint_b) ps.push_back(I("blocks", blocks));
            RadTensor ts[] = {tsc, tpos, treq, tbt, tid, tsl};
            RadArgs a{}; a.t = ts; a.n_t = 6; a.p = ps.data(); a.n_p = (int)ps.size();
            const int rrc = ref ? ref(&a, nullptr) : RAD_E_SHAPE;
            tid.data = mu.what == 14 ? nullptr : ids2.data(); tsl.data = sl2.data();
            if (mu.what != 14) tid.data = ids2.data();
            RadTensor ts2[] = {tsc, tpos, treq, tbt, tid, tsl};
            a.t = ts2;
            const int drc = r4d_rad_ernie_dsa_select(&a, nullptr);
            const std::vector<int32_t> untouched((size_t)M * W, 0x5a5a5a5a);
            const bool stored = !same(ids2, untouched) || !same(sl2, untouched);
            /* every mutation is a malformed launch: libref must refuse it, the device launcher must refuse it, and nothing may be stored (each asserted on its own, so a check
             * dropped from both sides fails here instead of passing as an agreement) */
            if (rrc == RAD_OK || drc == RAD_OK || stored) { std::printf("FAIL refusal '%s' [blocks %s]: libref rc %d, device rc %d, stored %d\n", mu.name, nbs, rrc, drc, (int)stored); ++g_fail; }
            else { ++g_pass; std::printf("PASS refusal '%s' [blocks %s] (libref %d, device %d)\n", mu.name, nbs, rrc, drc); }
        }
    }
    unsetenv("R4D_DSA_SELECT_BLOCKS");
}
}  // namespace

int main(int argc, char** argv) {
    setenv("OMP_NUM_THREADS", "2", 0);
    void* lib = argc > 1 ? dlopen(argv[1], RTLD_NOW | RTLD_LOCAL) : nullptr;
    RadLaunchFn ref = lib ? ref_op(lib, "ernie_dsa_select") : nullptr;
    if (!ref) { std::printf("FAIL no libref ernie_dsa_select\n"); return 1; }
    const char* ALL = "0,1,2,3,5,64";
    /* the real shape: topk 2048, window 128 (rows 2176 wide); small topk with a long window to reach the multi-block path (it needs a row 1024 + 2 nb words wide) */
    const Case cases[] = {
        /* name                                   L     topk win   bs rows mode      holes blocks end0.. */
        {"N < topk (real shape)",                 300, 2048, 128, 64, 3, RANDOM,    0, ALL, {300, 200, 129, 0}},
        {"N = topk - 1 / topk / topk + 1",       3000, 2048, 128, 64, 3, RANDOM,    0, ALL, {2047, 2048, 2049, 100}},
        {"N not a slice multiple, ties",         5001, 2048, 128, 64, 4, QUANT16,   0, ALL, {5001, 4999, 2300, 257}},
        {"all scores equal",                     6000, 2048, 128, 64, 3, ALLEQ,     0, ALL, {6000, 2049, 2048, 5000}},
        {"-inf holes + page holes",              9000, 2048, 128, 64, 4, HOLES,     3, ALL, {9000, 8000, 3000, -1}},
        {"signed zeros and denormals",           4100, 2048, 128, 64, 3, SIGNED0,   0, ALL, {4100, 4000, 3000, 2500}},
        {"ties straddling the k-th",             8191, 2048, 128, 64, 3, TIES_AT_K, 1, ALL, {8191, 4096, 5000, 2200}},
        {"two levels, 2% above",                20000, 2048, 128, 64, 3, TWO_LEVELS,0, ALL, {20000, 12000, 7000, 2500}},
        {"descending scores",                    7000, 2048, 128, 64, 2, DESC_RUN,  0, ALL, {7000, 2100, 0, 0}},
        {"no candidates",                        5000, 2048, 128, 64, 3, NONE,      0, ALL, {5000, 100, 0, 3000}},
        {"NaN and +inf (libref refuses)",        6000, 2048, 128, 64, 3, NONFINITE, 0, ALL, {6000, 3000, 2500, 5000}},
        {"topk 64, window 1200, ties",           3000,   64, 1200,16, 3, QUANT16,   0, ALL, {3000, 2999, 65, 64}},
        {"topk 1",                               3000,    1, 1200,16, 3, RANDOM,    0, ALL, {3000, 1, 2, 1500}},
        {"topk 1000 (not a power of two)",       5000, 1000, 1200,32, 3, QUANT16,   3, ALL, {5000, 1001, 1000, 999}},
        {"window longer than the sequence",      2500, 2048, 4000,64, 3, RANDOM,    0, ALL, {2500, 100, 0, 2048}},
        {"L = 16384 (auto threshold)",          16384, 2048, 128, 64, 2, RANDOM,    2, "auto,0", {16384, 9000, 0, 0}},
        {"L = 16385 (auto multi-block)",        16385, 2048, 128, 64, 2, QUANT16,   0, "auto,0", {16385, 12000, 0, 0}},
        {"L = 70001, random ends",              70001, 2048, 128, 64, 4, RANDOM,    1, "auto,0,64", {-1, -1, -1, -1}},
        {"L = 1,048,576, ties and holes",     1048576, 2048, 128, 64, 2, HOLES,     3, "auto,0", {1048576, 700001, 0, 0}},
        {"L = 1,048,576, all equal",          1048576, 2048, 128, 64, 1, ALLEQ,     0, "auto,0", {1048576, 0, 0, 0}},
    };
    uint32_t seed = 21;
    for (const Case& c : cases) run_case(c, ref, seed++);
    refusal_cases(ref);
    invalid_row_case();
    route_cases();
    std::printf("%d passed, %d failed\n%s\n", g_pass, g_fail, g_fail ? "FAILURES" : "ALL PASS");
    return g_fail ? 1 : 0;
}
