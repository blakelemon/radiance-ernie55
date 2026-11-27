/* ernie_dsa_topk_tiled on the device kernels (SIMT emulator), against the device's untiled pair ernie_idx_logits + ernie_dsa_select
 * (bitwise) and against libref's tiled op. docs/ernie55/score-tiling.md. Built with RAD_TEST_HIP_EMU: the .hip kernels run as C++. */
#include "rad_plugin.h"
#include "r4d_plugin.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <random>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

/* The device op's latch of a refusal (bits: 1 non-finite score, 2 bad position or request, 4 candidate range past the host bound), read and cleared. */
extern "C" __attribute__((weak)) int r4d_ernie_dsa_take_error(void);
extern "C" int r4d_shape_ernie_dsa_topk_tiled(const RadParam*, int, int, RadOpdDesc*);

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
uint8_t e4(std::mt19937& g, int span) {   /* a finite E4M3 code: sign, exponent 0..span, mantissa 0 -> small dyadic values */
    const uint8_t sign = (g() & 1) ? 0x80 : 0, ex = (uint8_t)(6 + g() % (uint32_t)span), man = (g() & 1) ? 0 : 4;
    return (uint8_t)(sign | (ex << 3) | man);
}

struct Case {
    const char* name;
    int ctx, tile, topk, window, block, rows, ties /* 0 random, 1 three distinct keys, 2 all equal, 3 holes in the index table, 4 holes only in the selection table */;
    int D, H, flat /* scratch is a flat [rows * tile] buffer */, use_kv_used /* pass RadArgs.kv_used_max_h */, pos_last /* all rows at ctx - 1 */;
    int dyadic;   /* values chosen so every sum is exact: device, libref and untiled agree bit for bit on the SCORES too */
    const char* nb = nullptr;   /* R4D_DSA_SELECT_BLOCKS for the tiled launch (the untiled pair always runs the single-block select); null = unset, the default */
};
/* ties: 1 = three repeated key vectors (one prototype per KEY, one fixed key scale); 5 = one key vector, key scales 4 / 2 / 1 in groups of about 8 / 37 / 55 %
 * of the keys, all-ones query: three score levels, the topk cutoff falls inside the middle group and the group spans many tiles */
/* words of the kept-list scratch the op takes as operand 11: [rows, 2 * topk + 2] kept entries, then one flag word a row */
int64_t kept_words(int64_t rows, int64_t topk) { return rows * (2 * topk + 3); }

int differ(const std::vector<int32_t>& a, const std::vector<int32_t>& b) { return std::memcmp(a.data(), b.data(), a.size() * 4) != 0; }

bool run_case(const Case& c, RadLaunchFn ref_tiled, uint32_t seed) {
    std::mt19937 g(seed);
    const int S = 2, M = c.rows, D = c.D, H = c.H, W = c.topk + c.window;
    const int B = (c.ctx + c.block - 1) / c.block + 1, pages = S * B + 3;
    std::vector<uint8_t> qf((size_t)M * H * D), cache((size_t)pages * c.block * (D + 4));
    std::vector<float> qs((size_t)M * H);
    std::vector<uint16_t> w((size_t)M * H);
    for (auto& v : qf) v = c.ties == 5 ? 0x38 : e4(g, c.dyadic ? 3 : 6);   /* 0x38 = 1.0 */
    for (auto& v : qs) v = c.ties == 5 ? 1.0f : c.dyadic ? 0.25f * (float)(1 + g() % 4) : 0.1f + (g() % 1000) / 700.0f;
    for (auto& v : w) { const float f = c.ties == 5 ? 1.0f : c.dyadic ? 0.5f * (float)(1 + g() % 3) : 0.1f + (g() % 1000) / 600.0f; v = rad_f32_to_bf16(f); }
    std::vector<std::vector<uint8_t>> proto(3, std::vector<uint8_t>((size_t)D));
    for (auto& r : proto) for (auto& b : r) b = e4(g, c.dyadic ? 3 : 6);
    for (int pg = 0; pg < pages; ++pg)
        for (int o = 0; o < c.block; ++o) {
            uint8_t* row = cache.data() + ((size_t)pg * c.block + (size_t)o) * (D + 4);
            const size_t which = c.ties == 1 ? g() % 3 : 0;   /* ONE prototype per key, not one per dimension */
            for (int d = 0; d < D; ++d) row[d] = c.ties == 5 ? 0x38 : c.ties == 1 ? proto[which][(size_t)d] : c.ties == 2 ? proto[0][(size_t)d] : e4(g, c.dyadic ? 3 : 6);
            const uint32_t pct = c.ties == 5 ? g() % 100 : 0;
            const float ks = c.ties == 5 ? (pct < 8 ? 4.0f : pct < 45 ? 2.0f : 1.0f) : (c.ties == 2 || c.ties == 1) ? 1.0f : (c.dyadic ? (float)(1 << (g() % 3)) * 0.5f : 0.3f + (g() % 1000) / 900.0f);
            std::memcpy(row + D, &ks, 4);
        }
    std::vector<int32_t> perm((size_t)S * B), bt1((size_t)S * B), bt2, req((size_t)M), pos((size_t)M);
    for (size_t i = 0; i < perm.size(); ++i) perm[i] = (int32_t)i;
    std::shuffle(perm.begin(), perm.end(), g);
    for (size_t i = 0; i < perm.size(); ++i) bt1[i] = (c.ties == 3 && g() % 4 == 0) ? -1 : perm[i];
    bt2 = bt1;
    if (c.ties == 4) { bt1 = perm; bt2 = perm; for (auto& e : bt2) if (g() % 4 == 0) e = -1; }
    for (int m = 0; m < M; ++m) { req[(size_t)m] = m % S; pos[(size_t)m] = (c.pos_last || m == 0) ? c.ctx - 1 : (int32_t)(g() % (uint32_t)c.ctx); }
    const int plane_n = c.flat ? M * c.tile : M * c.tile;
    std::vector<float> plane_full((size_t)M * c.ctx), plane_tile((size_t)plane_n);
    std::vector<int32_t> ids_u((size_t)M * W, 0x5a5a5a5a), sl_u((size_t)M * W, 0x5a5a5a5a), ids_t = ids_u, sl_t = ids_u, ids_r = ids_u, sl_r = ids_u;

    RadTensor tqf = T(qf.data(), RAD_F8E4M3, {M, H * D}), tqs = T(qs.data(), RAD_F32, {M, H}), tw = T(w.data(), RAD_BF16, {M, H}),
              tcache = T(cache.data(), RAD_U8, {pages, 1, c.block, D + 4}), tbt1 = T(bt1.data(), RAD_I32, {S, B}), tbt2 = T(bt2.data(), RAD_I32, {S, B}),
              treq = T(req.data(), RAD_I32, {M}), tpos = T(pos.data(), RAD_I32, {M});
    /* device, untiled: logits into a plane as wide as the context, then select */
    {
        RadTensor tp = T(plane_full.data(), RAD_F32, {M, c.ctx});
        RadTensor ts[] = {tqf, tqs, tw, tcache, tbt1, treq, tpos, tp};
        RadParam ps[] = {I("M", M), I("heads", H), I("head_dim", D), I("window", c.window)};
        RadArgs a{}; a.t = ts; a.n_t = 8; a.p = ps; a.n_p = 4;
        if (r4d_rad_ernie_idx_logits(&a, nullptr) != RAD_OK) { std::printf("FAIL %s: device logits refused\n", c.name); return false; }
        RadTensor ti = T(ids_u.data(), RAD_I32, {M, W}), tl = T(sl_u.data(), RAD_I32, {M, W});
        RadTensor ts2[] = {tp, tpos, treq, tbt2, ti, tl};
        RadParam ps2[] = {I("M", M), I("topk", c.topk), I("window", c.window), I("block_size", c.block), I("num_pages", pages)};
        RadArgs a2{}; a2.t = ts2; a2.n_t = 6; a2.p = ps2; a2.n_p = 5;
        setenv("R4D_DSA_SELECT_BLOCKS", "0", 1);   /* the untiled reference is the single-block kernel */
        const int src = r4d_rad_ernie_dsa_select(&a2, nullptr);
        unsetenv("R4D_DSA_SELECT_BLOCKS");
        if (src != RAD_OK) { std::printf("FAIL %s: device select refused\n", c.name); return false; }
    }
    /* the scores themselves: every tile of ernie_idx_logits with col_start equals the same columns of the untiled plane, bit for bit */
    {
        bool same = true;
        std::vector<float> tp((size_t)M * c.tile);
        for (int t0 = 0; t0 < c.ctx; t0 += c.tile) {
            RadTensor tpt = T(tp.data(), RAD_F32, {M, c.tile});
            RadTensor ts[] = {tqf, tqs, tw, tcache, tbt1, treq, tpos, tpt};
            RadParam ps[] = {I("M", M), I("heads", H), I("head_dim", D), I("window", c.window), I("col_start", t0)};
            RadArgs a{}; a.t = ts; a.n_t = 8; a.p = ps; a.n_p = 5;
            if (r4d_rad_ernie_idx_logits(&a, nullptr) != RAD_OK) { std::printf("FAIL %s: logits with col_start refused\n", c.name); return false; }
            for (int m = 0; m < M && same; ++m)
                for (int jj = 0; jj < c.tile && t0 + jj < c.ctx; ++jj)
                    if (std::memcmp(&tp[(size_t)m * c.tile + jj], &plane_full[(size_t)m * c.ctx + t0 + jj], 4) != 0) { same = false; break; }
        }
        if (!same) { std::printf("FAIL %s: tile scores differ from the untiled plane\n", c.name); return false; }
    }
    std::vector<int32_t> kept((size_t)kept_words(M, c.topk), 0x5a5a5a5a);
    auto tiled = [&](RadLaunchFn fn, bool device, std::vector<int32_t>& ids, std::vector<int32_t>& sl) -> int {
        RadTensor tkept = T(kept.data(), RAD_I32, {(int64_t)kept.size()});
        RadTensor tp = c.flat ? T(plane_tile.data(), RAD_F32, {M * c.tile}) : T(plane_tile.data(), RAD_F32, {M, c.tile});
        RadTensor ti = T(ids.data(), RAD_I32, {M, W}), tl = T(sl.data(), RAD_I32, {M, W});
        RadTensor ts[] = {tqf, tqs, tw, tcache, tbt1, treq, tpos, tbt2, tp, ti, tl, tkept};
        RadParam ps[] = {I("M", M), I("heads", H), I("head_dim", D), I("topk", c.topk), I("window", c.window), I("block_size", c.block), I("num_pages", pages), I("tokens", c.ctx)};
        RadArgs a{}; a.t = ts; a.n_t = 12; a.p = ps; a.n_p = 8;
        a.kv_used_max_h = (device && c.use_kv_used) ? c.ctx + 1 : -1;
        return fn(&a, nullptr);
    };
    if (c.ties == 5) {   /* the fixture must really put the topk cutoff inside a tie group that spans tiles, else it tests nothing */
        const int end = c.ctx - c.window;
        std::vector<std::pair<float, int>> sc;
        for (int j = 0; j < end; ++j) sc.push_back({plane_full[(size_t)j], j});   /* row 0 sits at ctx - 1 */
        std::vector<float> srt; for (auto& e : sc) srt.push_back(e.first);
        std::sort(srt.begin(), srt.end(), std::greater<float>());
        const float kth = srt[(size_t)c.topk - 1];
        int greater = 0, equal = 0, first_eq = -1, last_eq = -1;
        for (auto& e : sc) { if (e.first > kth) ++greater; else if (e.first == kth) { ++equal; if (first_eq < 0) first_eq = e.second; last_eq = e.second; } }
        const bool straddles = equal > c.topk - greater && first_eq / c.tile != last_eq / c.tile;
        std::printf("INFO %s: %d above the cutoff score, %d tied at it (ids %d..%d, tiles %d..%d), %d of them kept\n", c.name, greater, equal, first_eq, last_eq,
                    first_eq / c.tile, last_eq / c.tile, c.topk - greater);
        if (!straddles) { std::printf("FAIL %s: fixture does not put the topk cutoff inside a tie group that spans tiles\n", c.name); return false; }
    }
    if (c.nb) setenv("R4D_DSA_SELECT_BLOCKS", c.nb, 1);
    int rc = tiled(r4d_rad_ernie_dsa_topk_tiled, true, ids_t, sl_t);
    unsetenv("R4D_DSA_SELECT_BLOCKS");
    if (rc != RAD_OK) { std::printf("FAIL %s: device tiled op refused (rc %d)\n", c.name, rc); return false; }
    bool ok = true;
    if (differ(ids_u, ids_t) || differ(sl_u, sl_t)) { std::printf("FAIL %s: device tiled differs from device untiled (%s)\n", c.name, differ(ids_u, ids_t) ? "ids" : "slots"); ok = false; }
    if (ref_tiled) {
        const int rr = tiled(ref_tiled, false, ids_r, sl_r);
        if (rr != RAD_OK) { std::printf("FAIL %s: libref tiled refused (rc %d)\n", c.name, rr); return false; }
        const bool same = !differ(ids_r, ids_t) && !differ(sl_r, sl_t);
        if (c.dyadic && !same) { std::printf("FAIL %s: device tiled differs from libref tiled on exactly representable scores\n", c.name); ok = false; }
        if (!c.dyadic) { int rows = 0; for (int m = 0; m < M; ++m) rows += std::memcmp(&ids_r[(size_t)m * W], &ids_t[(size_t)m * W], (size_t)W * 4) != 0; std::printf("INFO %s: %d of %d rows differ from libref (score reduction order, not tile related)\n", c.name, rows, M); }
    }
    if (ok) std::printf("PASS %s\n", c.name);
    return ok;
}

/* ---------------------------------------------------------------------------------------------------------------------------------------
 * Refusals. libref's contract (ref_ernie_dsa_topk_tiled): a NaN or +inf score in a row's candidate range [0, end), a bad position or request, or a
 * candidate range past the `tokens` bound refuses the whole launch with RAD_E_SHAPE and leaves ids and slots untouched. The device op must do the same
 * wherever the host can see the violation before the first store (this emulator: the operands are host memory); on a card it stores nothing and latches
 * the verdict (r4d_ernie_dsa_take_error). One row, position 39, window 2, 10 blocks of 4 tokens, every score 0.5 (the fixture of the libref refusal test). */
struct Neg {
    int M = 1, H = 1, D = 4, block = 4, ctx = 40, topk = 4, window = 2, pages = 11, B = 10, S = 1;
    std::vector<uint8_t> qf, cache;
    std::vector<float> qs;
    std::vector<uint16_t> w;
    std::vector<int32_t> bt1, bt2, req, pos;
    explicit Neg(int rows = 1, int ctx_ = 40, int topk_ = 4, int window_ = 2, int block_ = 4) : M(rows) {
        ctx = ctx_; topk = topk_; window = window_; block = block_; B = ctx / block; pages = B + 1;
        qf.assign((size_t)M * H * D, 0); qs.assign((size_t)M * H, 1.0f); w.assign((size_t)M * H, rad_f32_to_bf16(1.0f));
        for (int m = 0; m < M; ++m) { qf[(size_t)m * H * D] = 0x38; qf[(size_t)m * H * D + 1] = 0x38; }   /* [1, 1, 0, 0]: dot 1 with a key [1, 0, 0, 0] */
        cache.assign((size_t)pages * block * (D + 4), 0);
        const float one = 1.0f;
        for (int t = 0; t < pages * block; ++t) { uint8_t* r = cache.data() + (size_t)t * (D + 4); r[0] = 0x38; std::memcpy(r + D, &one, 4); }
        for (int i = 0; i < S * B; ++i) bt1.push_back(i);
        bt2 = bt1;
        for (int m = 0; m < M; ++m) { req.push_back(0); pos.push_back(ctx - 1); }
    }
    void key_scale(int token, float v) { std::memcpy(cache.data() + (size_t)token * (D + 4) + D, &v, 4); }   /* identity table: token t is cache row t */
    struct Out { std::vector<int32_t> ids, sl; };
    Out fresh(int tile, int topk_) const { (void)tile; Out o; const size_t n = (size_t)M * (topk_ + window); o.ids.assign(n, 0x5a5a5a5a); o.sl.assign(n, 0x5a5a5a5a); return o; }
    /* tokens < 0: parameter absent; kv_used < 0: no host bound; kept_n < 0: the right size */
    int32_t *bt1_over = nullptr, *bt2_over = nullptr;   /* tables placed against a PROT_NONE page (guarded_table) */
    int run(RadLaunchFn fn, int tile, int tokens, int kv_used, Out& o, int kept_n = -1) {
        std::vector<float> plane((size_t)M * tile);
        std::vector<int32_t> kept((size_t)(kept_n < 0 ? kept_words(M, topk) : kept_n), 0x5a5a5a5a);
        const int W = topk + window;
        RadTensor tqf = T(qf.data(), RAD_F8E4M3, {M, H * D}), tqs = T(qs.data(), RAD_F32, {M, H}), tw = T(w.data(), RAD_BF16, {M, H}),
                  tcache = T(cache.data(), RAD_U8, {pages, 1, block, D + 4}), tbt1 = T(bt1_over ? bt1_over : bt1.data(), RAD_I32, {S, B}), tbt2 = T(bt2_over ? bt2_over : bt2.data(), RAD_I32, {S, B}),
                  treq = T(req.data(), RAD_I32, {M}), tpos = T(pos.data(), RAD_I32, {M}), tp = T(plane.data(), RAD_F32, {M, tile}),
                  ti = T(o.ids.data(), RAD_I32, {M, W}), tl = T(o.sl.data(), RAD_I32, {M, W}), tk = T(kept.data(), RAD_I32, {(int64_t)kept.size()});
        RadTensor ts[] = {tqf, tqs, tw, tcache, tbt1, treq, tpos, tbt2, tp, ti, tl, tk};
        RadParam ps[] = {I("M", M), I("heads", H), I("head_dim", D), I("topk", topk), I("window", window), I("block_size", block), I("num_pages", pages), I("tokens", tokens)};
        RadArgs a{}; a.t = ts; a.n_t = 12; a.p = ps; a.n_p = tokens >= 0 ? 8 : 7;
        a.kv_used_max_h = kv_used;
        return fn(&a, nullptr);
    }
};

bool run_negatives(RadLaunchFn ref_tiled, int vmode) {
    /* vmode 1 (R4D_DSA_VALIDATE, the host-build default): the launcher scans positions, requests and the bound before the first launch and returns the
     * refusal, nothing latched. vmode 0 (a card build): only the device verdict exists; the kernels store nothing and latch bits 2 and 4. Bit 1 (a
     * non-finite score) is only knowable after scoring, so it is latched in both modes. */
    setenv("R4D_DSA_VALIDATE", vmode ? "1" : "0", 1);
    std::printf("-- refusals, R4D_DSA_VALIDATE=%d\n", vmode);
    bool ok = true;
    const bool have_latch = r4d_ernie_dsa_take_error != nullptr;
    if (!have_latch) { std::printf("FAIL negatives: r4d_ernie_dsa_take_error does not exist (the device has no way to report a refusal it cannot return)\n"); ok = false; }
    auto take = [&]() { return have_latch ? r4d_ernie_dsa_take_error() : 0; };
    /* what: the case; expect: rc of the device (and libref unless dev_only); latch: bits r4d_ernie_dsa_take_error must report (0: none) */
    auto check = [&](const char* what, Neg& n, int tile, int tokens, int kv_used, int expect, int latch, bool dev_only = false, int kept_n = -1) {
        (void)take();
        if (vmode) latch &= 1;
        Neg::Out d = n.fresh(tile, n.topk);
        const int rc = n.run(r4d_rad_ernie_dsa_topk_tiled, tile, tokens, kv_used, d, kept_n);
        const int got = take();
        bool good = rc == expect;
        if (!good) std::printf("FAIL %s (tile %d): device rc %d, expected %d\n", what, tile, rc, expect);
        if (expect != RAD_OK) {
            const Neg::Out f = n.fresh(tile, n.topk);
            if (d.ids != f.ids || d.sl != f.sl) { std::printf("FAIL %s (tile %d): device refused (rc %d) but ids/slots were modified\n", what, tile, rc); good = false; }
            if (have_latch && got != latch) { std::printf("FAIL %s (tile %d): latch %d, expected %d\n", what, tile, got, latch); good = false; }
        } else if (got != 0) { std::printf("FAIL %s (tile %d): valid launch latched %d\n", what, tile, got); good = false; }
        if (ref_tiled && !dev_only) {
            Neg::Out r = n.fresh(tile, n.topk);
            const int rr = n.run(ref_tiled, tile, tokens, kv_used, r, kept_n);
            if (rr != expect) { std::printf("FAIL %s (tile %d): libref rc %d, expected %d\n", what, tile, rr, expect); good = false; }
            if (expect == RAD_OK && (r.ids != d.ids || r.sl != d.sl)) { std::printf("FAIL %s (tile %d): device and libref results differ\n", what, tile); good = false; }
            if (expect != RAD_OK) { const Neg::Out f = n.fresh(tile, n.topk); if (r.ids != f.ids || r.sl != f.sl) { std::printf("FAIL %s (tile %d): libref modified ids/slots on a refusal\n", what, tile); good = false; } }
        }
        if (good) std::printf("PASS %s (tile %d)\n", what, tile);
        ok = ok && good;
    };
    const float nan = std::nanf(""), pinf = INFINITY;
    for (int tile : {1, 8}) {
        { Neg n; check("valid, tokens 40", n, tile, 40, -1, RAD_OK, 0); }
        /* the review reproducer: candidate range [0, 38) against tokens = 30, no host bound. Device scored ids 0..29 and returned success. */
        { Neg n; check("tokens bound 30 < range 38 refuses", n, tile, 30, -1, RAD_E_SHAPE, 4); }
        { Neg n; check("tokens bound 37 < range 38 refuses (one short)", n, tile, 37, -1, RAD_E_SHAPE, 4); }
        { Neg n; check("tokens bound 38 = range 38 passes", n, tile, 38, -1, RAD_OK, 0); }
        /* the host-proven bound (RadArgs::kv_used_max_h = seqused max) bounds the same range: 32 - window 2 = 30 */
        { Neg n; check("host bound 32 (-> 30) refuses, no tokens", n, tile, -1, 32, RAD_E_SHAPE, 4, true); }
        { Neg n; check("host bound 40 passes, no tokens", n, tile, -1, 40, RAD_OK, 0, true); }
        { Neg n; check("no bound at all refuses", n, tile, -1, -1, RAD_E_INVAL, 0, true); }
        /* non-finite scores, in an early tile and in the last one (merges of the earlier tiles must not leave stores behind) */
        for (int token : {2, 13, 37}) {
            { Neg n; n.key_scale(token, nan);  char b[96]; std::snprintf(b, sizeof b, "NaN score at id %d refuses", token); check(b, n, tile, 40, -1, RAD_E_SHAPE, 1); }
            { Neg n; n.key_scale(token, pinf); char b[96]; std::snprintf(b, sizeof b, "+inf score at id %d refuses", token); check(b, n, tile, 40, -1, RAD_E_SHAPE, 1); }
        }
        { Neg n; n.key_scale(38, nan);  check("NaN at id 38 (window, not scored) passes", n, tile, 40, -1, RAD_OK, 0); }
        { Neg n; n.key_scale(39, pinf); check("+inf at id 39 (window, not scored) passes", n, tile, 40, -1, RAD_OK, 0); }
        { Neg n; n.key_scale(37, -pinf); check("-inf key scale (score 0 after relu) passes", n, tile, 40, -1, RAD_OK, 0); }
        /* a NaN key on a page the SELECTION table has not initialised is still a refusal: libref scans the score before the page filter */
        { Neg n; n.key_scale(13, nan); n.bt2[3] = -1; check("NaN on a page uninitialised in the selection table refuses", n, tile, 40, -1, RAD_E_SHAPE, 1); }
        { Neg n; n.pos[0] = -1; check("negative position refuses", n, tile, 40, -1, RAD_E_SHAPE, 2); }
        { Neg n; n.req[0] = 1; check("request outside the tables refuses", n, tile, 40, -1, RAD_E_SHAPE, 2); }
        /* the whole launch refuses, rows that are valid included */
        { Neg n(3); n.pos[1] = 20; n.pos[2] = 30; n.key_scale(37, nan); n.pos[2] = 39; check("3 rows, NaN only in row 2's last tile: nothing stored for any row", n, tile, 40, -1, RAD_E_SHAPE, 1); }
        { Neg n(3); n.req[2] = 5; check("3 rows, request 5 in row 2: nothing stored for any row", n, tile, 40, -1, RAD_E_SHAPE, 2); }
        { Neg n(3); n.pos[1] = 20; check("3 rows, valid", n, tile, 40, -1, RAD_OK, 0); }
        /* the new scratch operand: too small is a shape error on both */
        { Neg n; check("kept-list scratch one word short refuses", n, tile, 40, -1, RAD_E_SHAPE, 0, false, (int)kept_words(1, 4) - 1); }
    }
    /* two refusals in one latch window OR their bits; take clears */
    if (have_latch) {
        (void)take();
        Neg a; a.key_scale(13, nan); Neg::Out o = a.fresh(8, a.topk); (void)a.run(r4d_rad_ernie_dsa_topk_tiled, 8, 40, -1, o);
        Neg b; b.pos[0] = -1;        o = b.fresh(8, b.topk);           (void)b.run(r4d_rad_ernie_dsa_topk_tiled, 8, 40, -1, o);
        const int got = take(), again = take(), want = vmode ? 1 : 3;   /* with the host scan on, the bad position is returned, not latched */
        if (got != want || again != 0) { std::printf("FAIL latch: two refusals gave %d (expected %d) then %d (expected 0)\n", got, want, again); ok = false; } else std::printf("PASS latch ORs and clears\n");
    }
    /* The untiled device pair on the same scores. libref's ernie_dsa_select refuses NaN and +inf, a bad position and a range past the plane for the whole launch. The pair
     * refuses nothing in its return code (it cannot on a card); the select kernel stores nothing for the row and raises an event (bits 1, 2) that the engine polls. This was a
     * pre-existing gap of the pair (it dropped NaN, accepted +inf, skipped bad rows silently), reported by the tiling branch and closed here with the unified mechanism. */
    {
        for (int kind = 0; kind < 3; ++kind) {
            Neg n; if (kind < 2) n.key_scale(13, kind ? pinf : nan); else n.pos[0] = -1;
            (void)take();
            std::vector<float> plane((size_t)n.M * n.ctx);
            RadTensor tqf = T(n.qf.data(), RAD_F8E4M3, {1, 4}), tqs = T(n.qs.data(), RAD_F32, {1, 1}), tw = T(n.w.data(), RAD_BF16, {1, 1}),
                      tcache = T(n.cache.data(), RAD_U8, {n.pages, 1, n.block, 8}), tbt = T(n.bt1.data(), RAD_I32, {1, n.B}), treq = T(n.req.data(), RAD_I32, {1}),
                      tpos = T(n.pos.data(), RAD_I32, {1}), tp = T(plane.data(), RAD_F32, {1, n.ctx});
            RadTensor ts[] = {tqf, tqs, tw, tcache, tbt, treq, tpos, tp};
            RadParam ps[] = {I("M", 1), I("heads", 1), I("head_dim", 4), I("window", 2)};
            RadArgs a{}; a.t = ts; a.n_t = 8; a.p = ps; a.n_p = 4;
            const int r1 = r4d_rad_ernie_idx_logits(&a, nullptr);
            Neg::Out o = n.fresh(8, 4);
            RadTensor ti = T(o.ids.data(), RAD_I32, {1, 6}), tl = T(o.sl.data(), RAD_I32, {1, 6});
            RadTensor ts2[] = {tp, tpos, treq, tbt, ti, tl};
            RadParam ps2[] = {I("M", 1), I("topk", 4), I("window", 2), I("block_size", 4), I("num_pages", 11)};
            RadArgs a2{}; a2.t = ts2; a2.n_t = 6; a2.p = ps2; a2.n_p = 5;
            const int r2 = r4d_rad_ernie_dsa_select(&a2, nullptr);
            int rl = -99;
            if (ref_tiled) { /* libref's untiled pair through the same operands is in ref_test; here only the device side */ }
            const int bits = take(), again = take();
            const int want = kind < 2 ? 1 : 2;
            bool stored = false;
            for (int32_t v : o.ids) stored |= v != 0x5a5a5a5a;
            const bool good = r2 == RAD_OK && bits == want && again == 0 && !stored;
            std::printf("%s untiled device pair on %s: logits rc %d, select rc %d, event bits %d (want %d), outputs %s (plane[13] = %g; libref ernie_dsa_select refuses with RAD_E_SHAPE)%s\n",
                        good ? "PASS" : "FAIL", kind == 0 ? "a NaN key scale" : kind == 1 ? "a +inf key scale" : "a negative position", r1, r2, bits, want, stored ? "WRITTEN" : "untouched", (double)plane[13], rl == -99 ? "" : "");
            ok = ok && good;
        }
    }
    return ok;
}

/* direct callbacks: a zero or negative divisor must come back as RAD_E_UNSUPPORTED, never a SIGFPE. Each call runs in a child so a crash is a FAIL line. */
int shape_child(const char* key, int64_t v) {
    const pid_t pid = fork();
    if (pid == 0) {
        std::vector<RadParam> p = {I("M", 1), I("heads", 1), I("head_dim", 4), I("topk", 1), I("window", 2), I("tokens", 16), I("block_size", 8)};
        bool found = false;
        for (auto& q : p) if (!std::strcmp(q.key, key)) { q.ival = v; found = true; }
        if (!found) p.push_back(I(key, v));
        RadOpdDesc d{};
        _exit(r4d_shape_ernie_dsa_topk_tiled(p.data(), (int)p.size(), 0, &d) == RAD_E_UNSUPPORTED ? 0 : 3);
    }
    int st = 0; waitpid(pid, &st, 0);
    return WIFSIGNALED(st) ? 100 + WTERMSIG(st) : WEXITSTATUS(st);
}
bool run_shape_callback() {
    bool ok = true;
    struct C { const char* key; int64_t v; };
    const C bad[] = {{"block_size", 0}, {"block_size", -1}, {"block_size", -8}, {"tile", 0}, {"tile", -3}, {"topk", 0}, {"topk", -1}, {"window", 0}, {"window", -2},
                     {"heads", 0}, {"heads", -1}, {"head_dim", 0}, {"head_dim", 3}, {"M", 0}, {"M", -1}, {"tokens", 0}, {"tokens", -5}, {"seqs", 0}, {"seqs", -1}};
    for (const C& c : bad) {
        const int rc = shape_child(c.key, c.v);
        if (rc != 0) { std::printf("FAIL shape callback %s=%lld: %s (%d)\n", c.key, (long long)c.v, rc >= 100 ? "crashed with a signal" : "did not return RAD_E_UNSUPPORTED", rc); ok = false; }
    }
    if (ok) std::printf("PASS shape callback refuses %zu invalid parameters without arithmetic on them\n", sizeof bad / sizeof bad[0]);
    return ok;
}

/* A block table against a PROT_NONE page: `at_end` puts its last word against the page after it (a request past the table reads into it), else its first word
 * against the page before it (a negative request does). A kernel that indexes a table with an unchecked request faults here instead of reading a neighbour. */
int32_t* guarded_table(const std::vector<int32_t>& v, bool at_end) {
    const size_t pg = (size_t)sysconf(_SC_PAGESIZE);
    uint8_t* mem = (uint8_t*)mmap(nullptr, 3 * pg, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    mprotect(mem, pg, PROT_NONE);
    mprotect(mem + 2 * pg, pg, PROT_NONE);
    int32_t* t = (int32_t*)(at_end ? mem + 2 * pg - v.size() * 4 : mem + pg);
    std::memcpy(t, v.data(), v.size() * 4);
    return t;
}

/* The request is what the kernels index the block tables with. libref refuses a request outside [0, tables) before it reads anything. The logits kernels ran before
 * the merge kernel validated the request and read bt[r * stride] unchecked: on a card a negative request faults or reads a neighbour. Each case runs in a child
 * (a fault is a FAIL line) with the tables against guard pages, in both validation modes (the host scan returns before any launch; the card situation does not). */
int request_child(int vmode, int req, bool at_end, bool untiled) {
    const pid_t pid = fork();
    if (pid == 0) {
        setenv("R4D_DSA_VALIDATE", vmode ? "1" : "0", 1);
        Neg n;
        n.req[0] = req;
        n.bt1_over = guarded_table(n.bt1, at_end);
        n.bt2_over = guarded_table(n.bt2, at_end);
        if (untiled) {   /* the untiled logits launch alone: every column of the row must come back -inf, whatever the request */
            std::vector<float> plane((size_t)n.ctx, 7.0f);
            RadTensor tqf = T(n.qf.data(), RAD_F8E4M3, {1, 4}), tqs = T(n.qs.data(), RAD_F32, {1, 1}), tw = T(n.w.data(), RAD_BF16, {1, 1}),
                      tcache = T(n.cache.data(), RAD_U8, {n.pages, 1, n.block, 8}), tbt = T(n.bt1_over, RAD_I32, {1, n.B}), treq = T(n.req.data(), RAD_I32, {1}),
                      tpos = T(n.pos.data(), RAD_I32, {1}), tp = T(plane.data(), RAD_F32, {1, n.ctx});
            RadTensor ts[] = {tqf, tqs, tw, tcache, tbt, treq, tpos, tp};
            RadParam ps[] = {I("M", 1), I("heads", 1), I("head_dim", 4), I("window", 2)};
            RadArgs a{}; a.t = ts; a.n_t = 8; a.p = ps; a.n_p = 4;
            const int rc = r4d_rad_ernie_idx_logits(&a, nullptr);
            for (float f : plane) if (!(f == -INFINITY)) _exit(4);
            _exit(rc == RAD_OK ? 0 : 3);
        }
        Neg::Out o = n.fresh(8, n.topk);
        const int rc = n.run(r4d_rad_ernie_dsa_topk_tiled, 8, 40, -1, o);
        const Neg::Out f = n.fresh(8, n.topk);
        _exit(rc == RAD_E_SHAPE && o.ids == f.ids && o.sl == f.sl ? 0 : 3);
    }
    int st = 0; waitpid(pid, &st, 0);
    return WIFSIGNALED(st) ? 100 + WTERMSIG(st) : WEXITSTATUS(st);
}
bool run_request_guard() {
    bool ok = true;
    struct C { const char* what; int req; bool at_end; };
    const C cs[] = {{"request -1", -1, false}, {"request INT_MIN", INT32_MIN, false}, {"request 1 (one table row)", 1, true}, {"request 5", 5, true}, {"request INT_MAX", INT32_MAX, true}};
    for (int vmode : {1, 0})
        for (const C& c : cs)
            for (int untiled = 0; untiled < 2; ++untiled) {
                if (untiled && vmode) continue;   /* the untiled launch has no host scan: once is enough */
                const int rc = request_child(vmode, c.req, c.at_end, untiled);
                if (rc != 0) { std::printf("FAIL guard pages: %s%s (R4D_DSA_VALIDATE=%d): %s (%d)\n", c.what, untiled ? ", untiled logits launch" : "", vmode, rc >= 100 ? "faulted on a table access" : "wrong result", rc); ok = false; }
            }
    if (ok) std::printf("PASS guard pages: no block-table access outside the table for a request out of range (tiled op, both modes; untiled logits launch)\n");
    return ok;
}

/* The device row's constraints are what the engine selects the kernel by: a geometry the launcher accepts must satisfy every constraint of the row, or the kernel
 * is ineligible for it and the op falls to libref (a reference implementation) or is refused at startup. The row was declared window >= 2 while the launcher,
 * the shape callback and libref accept window >= 1; the direct-launch cases above bypassed the registry. Reads the rows from libr4d.so itself. */
bool check_registry_rows(const char* libr4d_path) {
    if (!libr4d_path) {
        const char* none = std::getenv("R4D_EMU_NO_LIBR4D");
        if (none && *none == '1') { std::printf("INFO registry: R4D_EMU_NO_LIBR4D=1 (a build without libr4d): the row's constraints are not checked\n"); return true; }
        std::printf("FAIL registry: no libr4d.so path given (argv[2]); the row's constraints are not checked\n");
        return false;
    }
    void* lib = dlopen(libr4d_path, RTLD_NOW | RTLD_LOCAL);
    if (!lib) { std::printf("FAIL registry: cannot load %s: %s\n", libr4d_path, dlerror()); return false; }
    auto count = reinterpret_cast<int (*)()>(dlsym(lib, "rad_kernel_count"));
    auto at = reinterpret_cast<const RadKernelInfo* (*)(int)>(dlsym(lib, "rad_kernel_at"));
    if (!count || !at) { std::printf("FAIL registry: libr4d.so has no kernel table\n"); return false; }
    const RadKernelInfo* row = nullptr;
    int rows = 0;
    for (int i = 0; i < count(); ++i) { const RadKernelInfo* k = at(i); if (k && !std::strcmp(k->op, "ernie_dsa_topk_tiled")) { row = k; ++rows; } }
    if (rows != 1) { std::printf("FAIL registry: %d rows for ernie_dsa_topk_tiled, expected 1\n", rows); return false; }
    struct G { const char* what; int64_t M, heads, head_dim, topk, window, block_size; bool ok; };
    const G geoms[] = {
        {"window 1, the geometry the direct-launch case accepts", 3, 2, 128, 16, 1, 8, true},
        {"window 2", 3, 2, 128, 16, 2, 8, true},
        {"production: 2048 rows, 64 heads, topk 2048, window 128, block 128", 2048, 64, 128, 2048, 128, 128, true},
        {"largest legal row, head and dim counts", 65535, 256, 256, 2048, 1, 1, true},
        {"window 0 is refused", 3, 2, 128, 16, 0, 8, false},
        {"topk 2049 is refused", 3, 2, 128, 2049, 128, 8, false},
        {"head_dim 3 is refused", 3, 2, 3, 16, 128, 8, false},
    };
    bool ok = true;
    for (const G& g : geoms) {
        auto val = [&](const char* key, int64_t* v) {
            if (!std::strcmp(key, "M")) *v = g.M; else if (!std::strcmp(key, "heads")) *v = g.heads; else if (!std::strcmp(key, "head_dim")) *v = g.head_dim;
            else if (!std::strcmp(key, "topk")) *v = g.topk; else if (!std::strcmp(key, "window")) *v = g.window; else if (!std::strcmp(key, "block_size")) *v = g.block_size;
            else return false;
            return true;
        };
        bool holds = true;
        const char* why = "";
        for (int i = 0; i < row->n_constraints; ++i) {
            const RadConstraint& c = row->constraints[i];
            int64_t v = 0;
            if (!val(c.key, &v)) { std::printf("FAIL registry: the row constrains `%s`, which this test cannot evaluate\n", c.key); ok = false; continue; }
            const bool h = c.op == RAD_C_GE ? v >= c.ival : c.op == RAD_C_LE ? v <= c.ival : c.op == RAD_C_EQ ? v == c.ival : c.op == RAD_C_DIV ? (c.ival != 0 && v % c.ival == 0) : false;
            if (!h && holds) why = c.key;
            holds = holds && h;
        }
        if (holds != g.ok) { std::printf("FAIL registry: %s: the row's constraints %s, expected %s (first unmet: %s)\n", g.what, holds ? "hold" : "do not hold", g.ok ? "hold" : "not hold", why); ok = false; }
        else std::printf("PASS registry: %s\n", g.what);
    }
    return ok;
}
}  // namespace

/* A decode step through the multi-block select (one tile, scratch in the kept operand) against the merge it replaces (R4D_DSA_SELECT_BLOCKS=0): the same rc, the same ids and slots
 * byte for byte, the same latched bits, on valid and invalid launches, with the host scan on (R4D_DSA_VALIDATE=1) and off (the device verdict only), and with the card-side
 * validation of the untiled select on and off (R4D_ERNIE_VALIDATE; the tiled op refuses on the device either way). Rows 1100 candidates deep so the selection is real. */
extern "C" int r4d_ernie_dsa_select_last_nb(void);
bool run_route_equal() {
    bool ok = true;
    const float nan = std::nanf(""), pinf = INFINITY;
    struct Sc { const char* what; int rows; int ctx; int mut; };
    const Sc scs[] = {{"valid, 2500 deep", 2, 2500, 0}, {"valid, fewer candidates than topk", 2, 1000, 0}, {"NaN at id 700", 2, 2500, 1}, {"+inf at id 1900", 2, 2500, 2},
                      {"NaN in the window (not scored)", 2, 2500, 3}, {"negative position in row 1", 3, 2500, 4}, {"request outside the tables in row 2", 3, 2500, 5},
                      {"tokens bound below the range", 2, 2500, 6}, {"NaN on a page the selection table lacks", 2, 2500, 7}, {"pages missing from the selection table", 2, 2500, 8},
                      {"-inf scores (no candidate) in half the ids", 2, 2500, 9}};
    for (const Sc& sc : scs)
        for (int dv = 1; dv >= 0; --dv)
            for (int ev = 1; ev >= 0; --ev) {
                setenv("R4D_DSA_VALIDATE", dv ? "1" : "0", 1);
                setenv("R4D_ERNIE_VALIDATE", ev ? "1" : "0", 1);
                struct Res { int rc; std::vector<int32_t> ids, sl; int latch; int nb; };
                Res res[2];
                for (int variant = 0; variant < 2; ++variant) {
                    std::mt19937 g(77);
                    Neg n(sc.rows, sc.ctx, 1100, 64, 16);
                    for (int t = 0; t < n.pages * n.block; ++t) n.key_scale(t, 0.1f + (float)(g() % 1000) / 100.0f);
                    for (int m = 0; m < sc.rows; ++m) n.pos[(size_t)m] = sc.ctx - 1 - 37 * m;
                    int tokens = sc.ctx;
                    switch (sc.mut) {
                    case 1: n.key_scale(700, nan); break;
                    case 2: n.key_scale(1900, pinf); break;
                    case 3: n.key_scale(sc.ctx - 3, nan); break;
                    case 4: n.pos[1] = -1; break;
                    case 5: n.req[2] = 5; break;
                    case 6: tokens = sc.ctx - n.window - 10; break;
                    case 7: n.key_scale(700, nan); n.bt2[700 / 16] = -1; break;
                    case 8: for (int b = 0; b < n.B; b += 5) n.bt2[(size_t)b] = -1; break;
                    case 9: for (int t = 0; t < sc.ctx; t += 2) n.key_scale(t, -pinf); break;
                    }
                    setenv("R4D_DSA_SELECT_BLOCKS", variant ? "3" : "0", 1);
                    if (r4d_ernie_dsa_take_error) (void)r4d_ernie_dsa_take_error();
                    Neg::Out o = n.fresh(3000, n.topk);
                    Res& r = res[variant];
                    r.rc = n.run(r4d_rad_ernie_dsa_topk_tiled, 3000, tokens, -1, o);
                    r.latch = r4d_ernie_dsa_take_error ? r4d_ernie_dsa_take_error() : 0;
                    r.ids = o.ids; r.sl = o.sl; r.nb = r4d_ernie_dsa_select_last_nb();
                }
                unsetenv("R4D_DSA_SELECT_BLOCKS");
                const bool launched = res[1].rc == RAD_OK || !dv;   /* a host-scan refusal returns before any kernel */
                const bool same = res[0].rc == res[1].rc && res[0].ids == res[1].ids && res[0].sl == res[1].sl && res[0].latch == res[1].latch;
                const bool routed = !launched || res[1].nb == 3;
                if (!same || !routed) {
                    std::printf("FAIL multi-block route vs merge, %s [host scan %d, card validation %d]: rc %d/%d latch %d/%d ids %s slots %s, route nb %d\n", sc.what, dv, ev, res[0].rc, res[1].rc,
                                res[0].latch, res[1].latch, res[0].ids == res[1].ids ? "same" : "DIFFER", res[0].sl == res[1].sl ? "same" : "DIFFER", res[1].nb);
                    ok = false;
                } else std::printf("PASS multi-block route vs merge, %s [host scan %d, card validation %d] (rc %d, latch %d)\n", sc.what, dv, ev, res[1].rc, res[1].latch);
            }
    unsetenv("R4D_DSA_VALIDATE");
    unsetenv("R4D_ERNIE_VALIDATE");
    return ok;
}

/* Validation OFF (R4D_ERNIE_VALIDATE=0, a card's default) on the untiled pair idx_logits + dsa_select: as base d2258ef8, a NaN idx score is relu'd to 0 (fmaxf) and a candidate, nothing is
 * refused and no event is polled. A NaN key scale scores exactly as a 0 key scale. With validation on (above) the same plane refuses the row. */
bool run_untiled_validation_off() {
    bool ok = true;
    auto pair = [&](Neg& n, Neg::Out& o) {
        std::vector<float> plane((size_t)n.M * n.ctx);
        RadTensor tqf = T(n.qf.data(), RAD_F8E4M3, {1, 4}), tqs = T(n.qs.data(), RAD_F32, {1, 1}), tw = T(n.w.data(), RAD_BF16, {1, 1}),
                  tcache = T(n.cache.data(), RAD_U8, {n.pages, 1, n.block, 8}), tbt = T(n.bt1.data(), RAD_I32, {1, n.B}), treq = T(n.req.data(), RAD_I32, {1}),
                  tpos = T(n.pos.data(), RAD_I32, {1}), tp = T(plane.data(), RAD_F32, {1, n.ctx});
        RadTensor ts[] = {tqf, tqs, tw, tcache, tbt, treq, tpos, tp};
        RadParam ps[] = {I("M", 1), I("heads", 1), I("head_dim", 4), I("window", 2)};
        RadArgs a{}; a.t = ts; a.n_t = 8; a.p = ps; a.n_p = 4;
        int rc = r4d_rad_ernie_idx_logits(&a, nullptr);
        RadTensor ti = T(o.ids.data(), RAD_I32, {1, 6}), tl = T(o.sl.data(), RAD_I32, {1, 6});
        RadTensor ts2[] = {tp, tpos, treq, tbt, ti, tl};
        RadParam ps2[] = {I("M", 1), I("topk", 4), I("window", 2), I("block_size", 4), I("num_pages", 11)};
        RadArgs a2{}; a2.t = ts2; a2.n_t = 6; a2.p = ps2; a2.n_p = 5;
        return rc != RAD_OK ? rc : r4d_rad_ernie_dsa_select(&a2, nullptr);
    };
    setenv("R4D_ERNIE_VALIDATE", "0", 1);
    for (int kind = 0; kind < 2; ++kind) {   /* NaN scale == 0 scale; +inf scale == ... compared to the same value in the on mode is not defined, so only the NaN case */
        if (kind) break;
        if (r4d_ernie_dsa_take_error) (void)r4d_ernie_dsa_take_error();
        Neg a, b;
        a.key_scale(13, std::nanf("")); b.key_scale(13, 0.0f);
        Neg::Out oa = a.fresh(8, 4), ob = b.fresh(8, 4);
        const int ra = pair(a, oa), rb = pair(b, ob);
        const int got = r4d_ernie_dsa_take_error ? r4d_ernie_dsa_take_error() : 0;
        const Neg::Out f = a.fresh(8, 4);
        const bool pass = ra == RAD_OK && rb == RAD_OK && oa.ids == ob.ids && oa.sl == ob.sl && oa.ids != f.ids && got == 0;
        std::printf("%s untiled pair, validation off: a NaN idx score is relu'd to 0 and selected as a 0 score (rc %d/%d, verdict %d)\n", pass ? "PASS" : "FAIL", ra, rb, got);
        ok = ok && pass;
    }
    {   /* a negative position of the single row: base computed the row with it (the select returns without storing: base's own p < 0 check), nothing polled */
        Neg a; a.pos[0] = -1;
        Neg::Out o = a.fresh(8, 4);
        (void)pair(a, o);
        const Neg::Out f = a.fresh(8, 4);
        const bool pass = o.ids == f.ids;
        std::printf("%s untiled pair, validation off: a negative position is not selected (base's own check)\n", pass ? "PASS" : "FAIL");
        ok = ok && pass;
    }
    unsetenv("R4D_ERNIE_VALIDATE");
    return ok;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    setenv("OMP_NUM_THREADS", "2", 0);   /* libref's OpenMP spin must not eat the container's CPU budget */
    void* lib = argc > 1 ? dlopen(argv[1], RTLD_NOW | RTLD_LOCAL) : nullptr;
    RadLaunchFn ref_tiled = lib ? ref_op(lib, "ernie_dsa_topk_tiled") : nullptr;
    /* the reference comparisons are the point of this test: a missing library or op is a failure, not a quiet skip (R4D_EMU_DEVICE_ONLY=1 says so on purpose) */
    const char* device_only = std::getenv("R4D_EMU_DEVICE_ONLY");
    if (!ref_tiled && !(device_only && *device_only == '1')) {
        std::printf("FAIL no libref tiled op (argv[1] = %s): the device-vs-libref comparisons cannot run; set R4D_EMU_DEVICE_ONLY=1 to run without them\n", argc > 1 ? argv[1] : "(none)");
        return 1;
    }
    if (!ref_tiled) std::printf("INFO R4D_EMU_DEVICE_ONLY=1: device-vs-libref comparison skipped\n");
    /* D 128 takes the WMMA kernel, D 64 the fallback kernel; H 2 / D 128 gives wscale 1/16, exact */
    const Case cases[] = {
        /* name                 ctx tile topk win blk rows ties D   H flat kv  last dy */
        {"tile 32, ctx 200",     200,  32,  16,  8,  8,  3, 0, 128, 2, 0, 0, 0, 1},
        {"ctx multiple of tile", 256,  64,  16,  8,  8,  3, 0, 128, 2, 0, 0, 1, 1},
        {"one past a boundary",  257,  64,  16,  8,  8,  3, 0, 128, 2, 0, 0, 1, 1},
        {"one short",            255,  64,  16,  8,  8,  3, 0, 128, 2, 0, 0, 1, 1},
        {"tile not 16-aligned",  300,  37,  16,  8,  8,  2, 0, 128, 2, 0, 0, 0, 1},
        {"tile wider than ctx",  200, 512,  16,  8,  8,  2, 0, 128, 2, 0, 0, 0, 1},
        {"fewer than topk",       24,   8,  16,  8,  8,  2, 0, 128, 2, 0, 0, 1, 1},
        {"shorter than window",    5,   4,  16,  8,  8,  2, 0, 128, 2, 0, 0, 1, 1},
        {"tile 1",               120,   1,   4,  4,  8,  2, 0, 128, 2, 0, 0, 1, 1},
        {"tile = block",         200,   8,  16,  8,  8,  3, 0, 128, 2, 0, 0, 0, 1},
        {"heavy ties",           300,  40,  32, 16,  8,  4, 1, 128, 2, 0, 0, 0, 1},
        {"heavy ties, 64",       300,  64,  32, 16,  8,  4, 1, 128, 2, 0, 0, 1, 1},
        {"all scores equal",     300,  33,  32, 16,  8,  4, 2, 128, 2, 0, 0, 1, 1},
        {"holes (index table)",  300,  50,  32, 16,  8,  4, 3, 128, 2, 0, 0, 0, 1},
        {"holes (selection)",    300,  50,  32, 16,  8,  4, 4, 128, 2, 0, 0, 0, 1},
        {"block 16",             300,  64,  16,  8, 16,  4, 1, 128, 2, 0, 0, 1, 1},
        {"one row",              300,  16,  32, 16,  8,  1, 0, 128, 2, 0, 0, 1, 1},
        {"flat scratch, 1 row",  300,  16,  32, 16,  8,  1, 0, 128, 2, 1, 1, 1, 1},
        {"flat scratch, rows",   300,  16,  32, 16,  8,  3, 1, 128, 2, 1, 1, 0, 1},
        {"flat scratch, tile far wider than the context", 100, 4096, 16, 8, 8, 2, 0, 128, 2, 1, 1, 1, 1},
        {"flat scratch, wide, ctx 700", 700, 2048, 64, 16, 8, 3, 1, 128, 2, 1, 1, 0, 1},
        {"fallback kernel D 64", 300,  40,  32, 16,  8,  3, 0,  64, 2, 0, 0, 0, 1},
        {"many tiles, topk 2048", 5000, 700, 2048, 128, 64, 2, 0, 128, 2, 0, 1, 1, 1},
        {"random scores",        600,  97,  64, 16,  8,  3, 0, 128, 4, 0, 0, 0, 0},
        {"window 1",             200,  32,  16,  1,  8,  3, 0, 128, 2, 0, 0, 1, 1},
        {"mixed ties, cutoff mid-group",        300,  40,  32, 16,  8,  3, 5, 128, 2, 0, 0, 1, 1},
        {"mixed ties, topk 64, tile 37",        400,  37,  64, 16,  8,  2, 5, 128, 2, 0, 1, 0, 1},
        {"mixed ties, flat scratch, block 16",  500,  64, 100,  8, 16,  3, 5, 128, 2, 1, 1, 1, 1},
        {"mixed ties, tile 1",                  200,   1,  24,  8,  8,  2, 5, 128, 2, 0, 0, 1, 1},
        /* production head counts: 64 heads fill the WMMA kernel's one head chunk, 128 take two; M 2 and a tile of 500 select wmma128<2> where the untiled
         * plane (3000 columns) selects <4>, so tiled == untiled holds across the two variants at a real accumulation depth */
        {"64 heads",                            400,  90,  64, 16,  8,  3, 0, 128, 64, 0, 0, 1, 0},
        {"128 heads (two chunks), <2> vs <4>", 3000, 500,  64, 16,  8,  2, 0, 128, 128, 0, 1, 1, 1},
        {"64 heads, tile 1",                    120,   1,  16,  8,  8,  2, 0, 128, 64, 0, 0, 1, 0},
        /* a decode step: one tile covers the context, and the multi-block select does the merge (the rows are 2176 wide, as in the model) */
        {"one tile -> multi-block, flat", 5000, 8192, 2048, 128, 64, 2, 0, 128, 2, 1, 1, 1, 1, "3"},
        {"one tile -> multi-block, ties", 6000, 8192, 2048, 128, 64, 3, 1, 128, 2, 1, 1, 0, 1, "5"},
        {"one tile -> multi-block, all equal", 4500, 6000, 2048, 128, 64, 2, 2, 128, 2, 0, 1, 1, 1, "64"},
        {"one tile -> multi-block, holes", 7000, 9000, 2048, 128, 64, 3, 3, 128, 2, 1, 1, 0, 1, "7"},
        {"one tile -> multi-block, holes (selection)", 7000, 9000, 2048, 128, 64, 3, 4, 128, 2, 1, 1, 0, 1, "2"},
        {"one tile -> multi-block, N < topk", 1500, 9000, 2048, 128, 64, 3, 0, 128, 2, 1, 1, 0, 1, "3"},
        {"one tile -> multi-block, random scores", 3000, 4096, 2048, 128, 64, 3, 0, 128, 4, 0, 1, 0, 0, "4"},
    };
    bool ok = true;
    uint32_t seed = 11;
    for (const Case& c : cases) ok = run_case(c, ref_tiled, seed++) && ok;
    ok = run_negatives(ref_tiled, 1) && ok;
    ok = run_negatives(ref_tiled, 0) && ok;
    ok = run_route_equal() && ok;
    ok = run_untiled_validation_off() && ok;
    ok = run_shape_callback() && ok;
    ok = run_request_guard() && ok;
    ok = check_registry_rows(argc > 2 ? argv[2] : nullptr) && ok;
    std::printf("%s\n", ok ? "ALL PASS" : "FAILURES");
    return ok ? 0 : 1;
}
