/* tests/r4d_emu: ernie_hca_dspark_attn (libr4d/r4d_ernie_dspark.hip) on the host SIMT emulator, held to libref's contract.
 *
 * libref validates every operand, cu_seqlens, seqused and every needed page index before its first store. The device launcher must refuse
 * the same inputs with the same error class and leave the output untouched. Each negative case runs on libref and on the emulated device
 * launcher and the two must agree on the return code and on "output untouched"; a valid case must agree on the values. The negatives run
 * in both validation modes of the launcher: the host-side scan of the metadata operands (R4D_DSPARK_VALIDATE unset) and the in-kernel
 * latch (R4D_DSPARK_VALIDATE=2, the form a card build uses, where the operands live in device memory). Cases: review's reproducer
 * (seqused < qlen), a cu_seqlens that does not start at 0, descends, or does not end at M, a NaN and a +inf sink, a page below -1, a page past
 * the pool, a needed block past the table; and the controls that must still pass: page -1 (skipped), a bad page outside the window, an
 * empty sequence slot with an arbitrary seqused, a -inf sink. Run once with D=2 (general kernel) and once with D=512 (cooperative kernel). */
#include "rad_plugin.h"
#include "r4d_plugin.h"
#include "r4d_ernie_verdict.h"
#include "verdict_replay.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <functional>
#include <string>
#include <vector>
extern "C" int r4d_ernie_dspark_take_error(void);
extern void (*emu_barrier_hook)(unsigned bx, unsigned by);
extern int emu_grids;

namespace {

RadParam I(const char* k, int64_t v) { RadParam p{}; p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }
RadParam F(const char* k, double v) { RadParam p{}; p.key = k; p.kind = RAD_P_F64; p.dval = v; return p; }
RadTensor V(void* d, uint32_t dt, int64_t n) { RadTensor t{}; t.data = d; t.dtype = dt; t.rank = 1; t.shape[0] = n; t.stride[0] = 1; return t; }
RadTensor Mx(void* d, uint32_t dt, int64_t r, int64_t c) { RadTensor t{}; t.data = d; t.dtype = dt; t.rank = 2; t.shape[0] = r; t.shape[1] = c; t.stride[0] = c; t.stride[1] = 1; return t; }

RadLaunchFn find(void* lib, const char* op) {
    auto cnt = reinterpret_cast<int (*)()>(dlsym(lib, "rad_kernel_count"));
    auto at = reinterpret_cast<const RadKernelInfo* (*)(int)>(dlsym(lib, "rad_kernel_at"));
    if (!cnt || !at) return nullptr;
    for (int i = 0; i < cnt(); ++i) { const RadKernelInfo* k = at(i); if (k && k->op && !std::strcmp(k->op, op)) return k->launch; }
    return nullptr;
}

/* The case: S = 2 sequences, 2 query rows each (M = 4), block size 4, 3 pages, a table of 2 blocks per sequence, window 4.
 * Sequence 0: seqused 5 -> anchor 3, lo 0, keys 0..4 (pages bt[0][0], bt[0][1]). Sequence 1: seqused 3 -> anchor 1, lo 0, keys 0..2 (page bt[1][0]). */
struct Fix {
    int D;
    std::vector<uint16_t> q, kv, out;
    std::vector<int32_t> bt, su, cu;
    std::vector<float> sink;
    explicit Fix(int d) : D(d) {
        const int M = 4, pages = 3, bs = 4;
        q.resize((size_t)M * D); kv.resize((size_t)pages * bs * D); out.assign((size_t)M * D, 0xdead);
        for (size_t i = 0; i < q.size(); ++i) q[i] = rad_f32_to_bf16(0.0625f * (float)((int)(i % 11) - 5));
        for (size_t i = 0; i < kv.size(); ++i) kv[i] = rad_f32_to_bf16(0.125f * (float)((int)((i * 7) % 13) - 6));
        bt = {0, 1, 2, -1};
        su = {5, 3};
        cu = {0, 2, 4};
        sink = {0.25f};
    }
    int run(RadLaunchFn fn) {
        RadTensor kvt{}; kvt.data = kv.data(); kvt.dtype = RAD_BF16; kvt.rank = 4; kvt.shape[0] = 3; kvt.shape[1] = 1; kvt.shape[2] = 4; kvt.shape[3] = D;
        kvt.stride[3] = 1; kvt.stride[2] = D; kvt.stride[1] = 4 * D; kvt.stride[0] = 4 * D;
        RadTensor t[7] = {Mx(q.data(), RAD_BF16, 4, D), kvt, Mx(bt.data(), RAD_I32, 2, 2), V(su.data(), RAD_I32, (int64_t)su.size()),
                          V(cu.data(), RAD_I32, (int64_t)cu.size()), V(sink.data(), RAD_F32, 1), Mx(out.data(), RAD_BF16, 4, D)};
        RadParam p[] = {I("M", 4), I("heads", 1), I("head_dim", D), I("window", 4), F("scale", 0.5)};
        RadArgs a{}; a.t = t; a.n_t = 7; a.p = p; a.n_p = 5;
        return fn(&a, nullptr);
    }
    bool untouched() const { for (uint16_t v : out) if (v != 0xdead) return false; return true; }
};

int g_fail = 0;
void check(bool ok, const std::string& name) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", name.c_str());
    if (!ok) ++g_fail;
}


unsigned g_flip_to = 0;
int g_flip_grid = -1;
bool g_flip_fired = false;
void flip_hook(unsigned, unsigned) {   /* at the first block barrier of the first block of the chosen grid: the word changes while the other threads have not read it */
    if (emu_grids != g_flip_grid || g_flip_fired) return;
    g_flip_fired = true;
    r4d_verdict::State* st = r4d_verdict::registry().state_of(0);
    const int n = r4d_verdict::registry().gate_sites(0);
    for (int i = 0; i < n; ++i) st->gate[i] = g_flip_to;
}

struct Neg { const char* name; std::function<void(Fix&)> mutate; unsigned want; };   /* want: the verdict bits (1 sink, 2 cu_seqlens, 4 seqused, 8 page, 16 block past the table) */

const Neg g_negs[] = {
        {"seqused < qlen (review's reproducer)", [](Fix& f) { f.su[0] = 1; }, 4},
        {"seqused of the second sequence < qlen", [](Fix& f) { f.su[1] = 1; }, 4},
        {"seqused = 0 with qlen 2", [](Fix& f) { f.su[1] = 0; }, 4},
        {"cu_seqlens[0] != 0", [](Fix& f) { f.cu = {1, 2, 4}; }, 2},
        {"cu_seqlens descends", [](Fix& f) { f.cu = {0, 3, 2}; f.su = {5, 3}; }, 2},
        {"cu_seqlens[S] != M (short)", [](Fix& f) { f.cu = {0, 2, 3}; }, 2},
        {"cu_seqlens[S] != M (long)", [](Fix& f) { f.cu = {0, 2, 5}; }, 2},
        {"sink NaN", [](Fix& f) { f.sink[0] = NAN; }, 1},
        {"sink +inf", [](Fix& f) { f.sink[0] = INFINITY; }, 1},
        {"page below -1 in a needed block", [](Fix& f) { f.bt[1] = -2; }, 8},
        {"page far below -1 in the first block", [](Fix& f) { f.bt[0] = -1000; }, 8},
        {"page == pool size", [](Fix& f) { f.bt[0] = 3; }, 8},
        {"page past the pool in the last needed block", [](Fix& f) { f.bt[1] = 77; }, 8},
        {"needed block past the table (seqused 9, bs 4, 2 blocks)", [](Fix& f) { f.su[0] = 9; }, 16},
};

void suite(RadLaunchFn ref, RadLaunchFn dev, int D, const char* mode) {
    const std::string tag = std::string("D") + std::to_string(D) + " " + mode + " ";
    {   /* the valid case: values within one bf16 step, the arithmetic of the two sides is not the same instruction for instruction */
        Fix a(D), b(D);
        const int ra = a.run(ref), rb = b.run(dev);
        bool close = ra == RAD_OK && rb == RAD_OK;
        for (size_t i = 0; close && i < a.out.size(); ++i) close = std::abs((int)a.out[i] - (int)b.out[i]) <= 1;
        check(close, tag + "valid case: libref and device agree");
    }
    const Neg* negs = g_negs; const size_t n_negs = sizeof g_negs / sizeof g_negs[0];

    for (size_t ni = 0; ni < n_negs; ++ni) {
        const Neg& n = negs[ni];
        Fix a(D), b(D);
        n.mutate(a); n.mutate(b);
        const int ra = a.run(ref), rb = b.run(dev);
        check(ra == RAD_E_SHAPE && rb == RAD_E_SHAPE && a.untouched() && b.untouched(),
              tag + n.name + " (libref rc " + std::to_string(ra) + ", device rc " + std::to_string(rb) +
              (b.untouched() ? ", output untouched" : ", OUTPUT WRITTEN") + ")");
    }
    struct Pos { const char* name; std::function<void(Fix&)> mutate; };
    const Pos poss[] = {
        {"page -1 is skipped, not refused", [](Fix& f) { f.bt[1] = -1; }},
        {"a bad page outside the window is not read (window 4, seqused 9 would need it; here sequence 1 block 1)", [](Fix& f) { f.bt[3] = -50; }},
        {"an empty sequence slot with any seqused", [](Fix& f) { f.cu = {0, 4, 4}; f.su = {6, -7}; f.bt[0] = 0; f.bt[1] = 1; }},
        {"a -inf sink", [](Fix& f) { f.sink[0] = -INFINITY; }},
    };
    for (const Pos& n : poss) {
        Fix a(D), b(D);
        n.mutate(a); n.mutate(b);
        const int ra = a.run(ref), rb = b.run(dev);
        bool close = ra == RAD_OK && rb == RAD_OK;
        for (size_t i = 0; close && i < a.out.size(); ++i) close = std::abs((int)a.out[i] - (int)b.out[i]) <= 1;
        check(close, tag + n.name + " (libref rc " + std::to_string(ra) + ", device rc " + std::to_string(rb) + ")");
    }
}


/* The card contract (R4D_ERNIE_CARD_FORM=1: the host scans are off, as on a card whose operands are device memory), at the DEFAULT settings (no R4D_DSPARK_VALIDATE):
 * a launch libref refuses returns RAD_OK (it is asynchronous), writes nothing, and its refusal is in the verdict the engine polls (r4d_ernie_dspark_take_error) with
 * the bit of the violated rule; a valid launch and every control leave the verdict empty and agree with libref. The reviewer's reproducer is the first case. */
void card_suite(RadLaunchFn ref, RadLaunchFn dev, int D) {
    const std::string tag = std::string("D") + std::to_string(D) + " card form (defaults) ";
    setenv("R4D_ERNIE_CARD_FORM", "1", 1);
    unsetenv("R4D_DSPARK_VALIDATE");
    unsetenv("R4D_ERNIE_VALIDATE");
    (void)r4d_ernie_dspark_take_error();
    for (size_t ni = 0; ni < sizeof g_negs / sizeof g_negs[0]; ++ni) {
        const Neg& n = g_negs[ni];
        Fix a(D), b(D);
        n.mutate(a); n.mutate(b);
        const int ra = a.run(ref);
        (void)r4d_ernie_dspark_take_error();
        const int rb = b.run(dev);
        const int bits = r4d_ernie_dspark_take_error();
        check(ra == RAD_E_SHAPE && a.untouched() && rb == RAD_OK && b.untouched() && ((unsigned)bits & n.want) == n.want && r4d_ernie_dspark_take_error() == 0,
              tag + n.name + " (libref rc " + std::to_string(ra) + ", launch rc " + std::to_string(rb) + (b.untouched() ? ", output untouched" : ", OUTPUT WRITTEN") +
              ", verdict 0x" + [&] { char t[16]; std::snprintf(t, sizeof t, "%x", (unsigned)bits); return std::string(t); }() + ", wanted bits 0x" + [&] { char t[16]; std::snprintf(t, sizeof t, "%x", n.want); return std::string(t); }() + ")");
    }
    struct Pos { const char* name; std::function<void(Fix&)> mutate; };
    const Pos poss[] = {{"valid", [](Fix&) {}}, {"page -1 is skipped", [](Fix& f) { f.bt[1] = -1; }}, {"a bad page outside the window", [](Fix& f) { f.bt[3] = -50; }},
                        {"an empty sequence slot with any seqused", [](Fix& f) { f.cu = {0, 4, 4}; f.su = {6, -7}; f.bt[0] = 0; f.bt[1] = 1; }}, {"a -inf sink", [](Fix& f) { f.sink[0] = -INFINITY; }}};
    for (const Pos& n : poss) {
        Fix a(D), b(D);
        n.mutate(a); n.mutate(b);
        const int ra = a.run(ref), rb = b.run(dev), bits = r4d_ernie_dspark_take_error();
        bool close = ra == RAD_OK && rb == RAD_OK;
        for (size_t i = 0; close && i < a.out.size(); ++i) close = std::abs((int)a.out[i] - (int)b.out[i]) <= 1;
        check(close && bits == 0, tag + n.name + " (libref rc " + std::to_string(ra) + ", launch rc " + std::to_string(rb) + ", verdict " + std::to_string(bits) + ")");
    }
    {   /* R4D_ERNIE_VALIDATE=0 is the benchmark switch: validators off, so the invalid launch runs and nothing is polled */
        setenv("R4D_ERNIE_VALIDATE", "0", 1);
        Fix b(D); b.su[0] = 1;
        const int rb = b.run(dev), bits = r4d_ernie_dspark_take_error();
        check(rb == RAD_OK && !b.untouched() && bits == 0, tag + "R4D_ERNIE_VALIDATE=0 turns the validator off (invalid launch runs, empty verdict)");
        unsetenv("R4D_ERNIE_VALIDATE");
    }
    unsetenv("R4D_ERNIE_CARD_FORM");
}

/* Several sequences in one launch, D = 512 (the production geometry), H = 2, block 16, window 20: ragged query lengths and seqused, a window that starts mid-sequence (the
 * pages before it are -1 and must not be read), a finite sink. The device launch must equal libref (within one bf16 step) and every sequence launched alone must equal
 * its rows of the batch bit for bit: a sequence's window, pages and anchor come from its own metadata only. A mutation of the window boundary, the anchor or the sequence
 * lookup moves these values. */
void multiseq(RadLaunchFn ref, RadLaunchFn dev) {
    constexpr int D = 512, H = 2, bs = 16, pages = 12, maxb = 4, S = 5, win = 20;
    const int qlen[S] = {3, 1, 2, 4, 2}, used[S] = {37, 12, 29, 50, 9};
    int32_t bt[S * maxb] = {0, 1, 2, -1,   3, -1, -1, -1,   4, 5, -1, -1,   -1, 6, 7, 8,   9, -1, -1, -1};
    std::vector<uint16_t> kv((size_t)pages * bs * D);
    for (size_t i = 0; i < kv.size(); ++i) kv[i] = rad_f32_to_bf16(0.0625f * (float)((int)((i * 2654435761u >> 7) % 17) - 8));
    int M = 0; int32_t cu[S + 1]; cu[0] = 0;
    for (int i = 0; i < S; ++i) { M += qlen[i]; cu[i + 1] = M; }
    std::vector<uint16_t> q((size_t)M * H * D);
    for (size_t i = 0; i < q.size(); ++i) q[i] = rad_f32_to_bf16(0.03125f * (float)((int)((i * 40503u >> 5) % 13) - 6));
    float sink[H] = {0.5f, -1.0f};
    auto call = [&](RadLaunchFn fn, int s0, int ns, std::vector<uint16_t>& out) {
        /* sequences s0 .. s0 + ns - 1 as their own launch */
        int rows = 0; std::vector<int32_t> c(ns + 1, 0), su(ns), b((size_t)ns * maxb);
        for (int i = 0; i < ns; ++i) { rows += qlen[s0 + i]; c[i + 1] = rows; su[i] = used[s0 + i]; for (int j = 0; j < maxb; ++j) b[(size_t)i * maxb + j] = bt[(s0 + i) * maxb + j]; }
        out.assign((size_t)rows * H * D, 0xdead);
        RadTensor kvt{}; kvt.data = kv.data(); kvt.dtype = RAD_BF16; kvt.rank = 4; kvt.shape[0] = pages; kvt.shape[1] = 1; kvt.shape[2] = bs; kvt.shape[3] = D;
        kvt.stride[3] = 1; kvt.stride[2] = D; kvt.stride[1] = (int64_t)bs * D; kvt.stride[0] = (int64_t)bs * D;
        const size_t q0 = (size_t)cu[s0] * H * D;
        RadTensor t[7] = {Mx(q.data() + q0, RAD_BF16, rows, H * D), kvt, Mx(b.data(), RAD_I32, ns, maxb), V(su.data(), RAD_I32, ns), V(c.data(), RAD_I32, ns + 1), V(sink, RAD_F32, H),
                          Mx(out.data(), RAD_BF16, rows, H * D)};
        RadParam p[] = {I("M", rows), I("heads", H), I("head_dim", D), I("window", win), F("scale", 0.0442)};
        RadArgs a{}; a.t = t; a.n_t = 7; a.p = p; a.n_p = 5;
        return fn(&a, nullptr);
    };
    std::vector<uint16_t> r, d;
    const int rr = call(ref, 0, S, r), rd = call(dev, 0, S, d);
    bool close = rr == RAD_OK && rd == RAD_OK;
    long worst = 0;
    for (size_t i = 0; close && i < r.size(); ++i) { const long dd = std::abs((int)r[i] - (int)d[i]); if (dd > worst) worst = dd; if (dd > 1) close = false; }
    check(close, "multi-sequence D512 H2, 5 ragged sequences, window starting mid-sequence: device == libref (max bf16 step " + std::to_string(worst) + ")");
    bool alone_ok = rd == RAD_OK;
    for (int i = 0; i < S && alone_ok; ++i) {
        std::vector<uint16_t> one;
        alone_ok = call(dev, i, 1, one) == RAD_OK;
        const size_t off = (size_t)cu[i] * H * D;
        for (size_t j = 0; alone_ok && j < one.size(); ++j) alone_ok = one[j] == d[off + j];
    }
    check(alone_ok, "multi-sequence D512: every sequence alone is bit-identical to its rows of the batch");
}

/* P1 (kv_dspark WINDOW, DESIGN section 1): the batch builder stages a WINDOW group's block table from the group's first held block, so the table starts k blocks
 * later and seqused is k * bs smaller. The op must give the same bytes: t_window = t_full - k * bs is the same page and the same row offset, reached in the same
 * order. FULL (width 10, seqused = used) and REBASED (width 4, row s = the FULL row's entries k_s .., seqused_s = used_s - k_s * bs) run on the same kv, except
 * that the rebased run reads a copy with every page that only FULL columns below k_s reference set to bf16 NaN: a dropped page that were read would show. The
 * control rebases sequence 0 one block too many and must differ; so must a poisoned row at lo (the window's first row), while a poisoned row at lo - 1 must not.
 * Both validation modes of the launcher, D = 512 (vector kernel) and D = 64 (generic kernel). */
void rebased(RadLaunchFn ref, RadLaunchFn dev, int D) {
    constexpr int H = 2, bs = 16, win = 20, S = 4, pages = 40, maxb_full = 10, maxb_win = 4;
    const int qlen[S] = {7, 1, 7, 3}, used[S] = {150, 40, 77, 23};
    std::vector<int32_t> bt_full(S * maxb_full), bt_win(S * maxb_win), su_full(S), su_win(S), cu(S + 1);
    int lo[S], k[S];
    int next_page = 0;
    for (int s = 0; s < S; ++s) {
        const int nblk = (used[s] - 1) / bs;   /* sequence s owns distinct pages for blocks 0 .. nblk */
        for (int j = 0; j < maxb_full; ++j) bt_full[s * maxb_full + j] = j <= nblk ? next_page++ : -1;
        const int l = used[s] - qlen[s] - win + 1;
        lo[s] = l < 0 ? 0 : l;
        k[s] = lo[s] / bs;                     /* the whole blocks below the read range: {7, 1, 3, 0} */
        su_full[s] = used[s];
        su_win[s] = used[s] - k[s] * bs;
        for (int j = 0; j < maxb_win; ++j) bt_win[s * maxb_win + j] = k[s] + j < maxb_full ? bt_full[s * maxb_full + k[s] + j] : -1;
    }
    cu[0] = 0;
    for (int s = 0; s < S; ++s) cu[s + 1] = cu[s] + qlen[s];
    const int M = cu[S];
    std::vector<uint16_t> kv((size_t)pages * bs * D), q((size_t)M * H * D);
    for (size_t i = 0; i < kv.size(); ++i) kv[i] = rad_f32_to_bf16(0.0625f * (float)((int)((i * 2654435761u >> 7) % 17) - 8));
    for (size_t i = 0; i < q.size(); ++i) q[i] = rad_f32_to_bf16(0.03125f * (float)((int)((i * 40503u >> 5) % 13) - 6));
    float sink[H] = {0.5f, -1.0f};
    auto call = [&](RadLaunchFn fn, std::vector<uint16_t>& kva, std::vector<int32_t>& bta, int maxb, std::vector<int32_t>& sua,
                    std::vector<uint16_t>& out) {
        out.assign((size_t)M * H * D, 0xdead);
        RadTensor kvt{}; kvt.data = kva.data(); kvt.dtype = RAD_BF16; kvt.rank = 4;
        kvt.shape[0] = pages; kvt.shape[1] = 1; kvt.shape[2] = bs; kvt.shape[3] = D;
        kvt.stride[3] = 1; kvt.stride[2] = D; kvt.stride[1] = (int64_t)bs * D; kvt.stride[0] = (int64_t)bs * D;
        RadTensor t[7] = {Mx(q.data(), RAD_BF16, M, H * D), kvt, Mx(bta.data(), RAD_I32, S, maxb), V(sua.data(), RAD_I32, S),
                          V(cu.data(), RAD_I32, S + 1), V(sink, RAD_F32, H), Mx(out.data(), RAD_BF16, M, H * D)};
        RadParam p[] = {I("M", M), I("heads", H), I("head_dim", D), I("window", win), F("scale", 0.0442)};
        RadArgs a{}; a.t = t; a.n_t = 7; a.p = p; a.n_p = 5;
        return fn(&a, nullptr);
    };
    auto has_nan = [](const std::vector<uint16_t>& v) {
        for (uint16_t x : v) if ((x & 0x7f80) == 0x7f80 && (x & 0x007f) != 0) return true;
        return false;
    };
    std::vector<uint16_t> kv2 = kv;
    {   /* poison every page that only FULL-table columns below k_s reference (the pages the window dropped) */
        bool kept[pages] = {};
        for (int s = 0; s < S; ++s)
            for (int j = k[s]; j < maxb_full; ++j) { const int pg = bt_full[s * maxb_full + j]; if (pg >= 0) kept[pg] = true; }
        for (int s = 0; s < S; ++s)
            for (int j = 0; j < k[s]; ++j) {
                const int pg = bt_full[s * maxb_full + j];
                if (pg < 0 || kept[pg]) continue;
                for (int r = 0; r < bs; ++r)
                    for (int c = 0; c < D; ++c) kv2[((size_t)pg * bs + r) * D + c] = 0x7fc0;
            }
    }
    const char* modes[2] = {"host scan", "validator kernel"};
    for (int mi = 0; mi < 2; ++mi) {
        if (mi) setenv("R4D_DSPARK_VALIDATE", "2", 1); else unsetenv("R4D_DSPARK_VALIDATE");
        const std::string tag = " [" + std::string(modes[mi]) + "]";
        const std::string dtag = "D" + std::to_string(D) + " ";
        std::vector<uint16_t> rf, rr, df, dr;
        const int a1 = call(ref, kv, bt_full, maxb_full, su_full, rf), a2 = call(ref, kv2, bt_win, maxb_win, su_win, rr);
        const int b1 = call(dev, kv, bt_full, maxb_full, su_full, df), b2 = call(dev, kv2, bt_win, maxb_win, su_win, dr);
        check(a1 == RAD_OK && a2 == RAD_OK && b1 == RAD_OK && b2 == RAD_OK,
              dtag + "all four launches return RAD_OK (libref " + std::to_string(a1) + "/" + std::to_string(a2) + ", device " + std::to_string(b1) + "/" +
              std::to_string(b2) + ")" + tag);
        check(std::memcmp(rf.data(), rr.data(), rf.size() * sizeof(uint16_t)) == 0,
              "libref D=" + std::to_string(D) + ": a rebased window table gives the same bytes" + tag);
        check(std::memcmp(df.data(), dr.data(), df.size() * sizeof(uint16_t)) == 0,
              "device D=" + std::to_string(D) + ": a rebased window table gives the same bytes" + tag);
        check(!has_nan(rf) && !has_nan(rr) && !has_nan(df) && !has_nan(dr), dtag + "no output holds a NaN (the poisoned pages are never read)" + tag);
        {   /* the control: sequence 0 rebased one block too many, so its window starts where the table no longer holds the oldest rows */
            std::vector<int32_t> bt_ctl(bt_win.begin(), bt_win.begin() + S * maxb_win), su_ctl(su_win.begin(), su_win.begin() + S);
            const int k0 = k[0] + 1;
            for (int j = 0; j < maxb_win; ++j) bt_ctl[j] = k0 + j < maxb_full ? bt_full[k0 + j] : -1;
            su_ctl[0] = used[0] - k0 * bs;
            std::vector<uint16_t> dc;
            const int rc = call(ref, kv2, bt_ctl, maxb_win, su_ctl, dc);
            check(rc == RAD_OK && std::memcmp(dc.data(), rf.data(), dc.size() * sizeof(uint16_t)) != 0,
                  dtag + "control: one block too many rebased changes sequence 0 (rc " + std::to_string(rc) + ")" + tag);
        }
        bool mid_page = true;
        for (int s = 0; s < S; ++s) mid_page = mid_page && lo[s] % bs != 0;
        check(mid_page, dtag + "the window's first row is not the first of its page (lo % bs != 0), so lo - 1 is in a page the table holds" + tag);
        {   /* (a) the row just older than the window is not read: poison lo - 1 in kv2 and the FULL run must not move */
            std::vector<uint16_t> kv3 = kv2;
            for (int s = 0; s < S; ++s)
                if (lo[s] > 0)
                    for (int c = 0; c < D; ++c)
                        kv3[((size_t)bt_full[s * maxb_full + (lo[s] - 1) / bs] * bs + (lo[s] - 1) % bs) * D + c] = 0x7fc0;
            std::vector<uint16_t> af, ad;
            const int ra = call(ref, kv3, bt_full, maxb_full, su_full, af), rb = call(dev, kv3, bt_full, maxb_full, su_full, ad);
            check(ra == RAD_OK && rb == RAD_OK && std::memcmp(af.data(), rf.data(), af.size() * sizeof(uint16_t)) == 0 &&
                      std::memcmp(ad.data(), df.data(), ad.size() * sizeof(uint16_t)) == 0,
                  dtag + "a poisoned row just below the window (lo - 1) changes nothing (libref/device)" + tag);
        }
        for (int s = 0; s < S; ++s) {   /* (b) the row AT the window's first row is read: poison lo and that sequence's rows must move */
            std::vector<uint16_t> kv4 = kv2;
            const int t = lo[s], pg = bt_full[s * maxb_full + t / bs];
            for (int c = 0; c < D; ++c) kv4[((size_t)pg * bs + t % bs) * D + c] = 0x7fc0;
            std::vector<uint16_t> bf, bd;
            const int ra = call(ref, kv4, bt_full, maxb_full, su_full, bf), rb = call(dev, kv4, bt_full, maxb_full, su_full, bd);
            const size_t off = (size_t)cu[s] * H * D, n = (size_t)qlen[s] * H * D * sizeof(uint16_t);
            check(ra == RAD_OK && rb == RAD_OK && std::memcmp(bf.data() + off, rf.data() + off, n) != 0 && std::memcmp(bd.data() + off, df.data() + off, n) != 0,
                  dtag + "a poisoned row at the window's first row (lo = " + std::to_string(t) + ") changes sequence " + std::to_string(s) + " (libref/device)" + tag);
        }
    }
    unsetenv("R4D_DSPARK_VALIDATE");
}
}  // namespace


int main(int argc, char** argv) {
    if (argc != 2) return 2;
    void* lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) { std::fprintf(stderr, "%s\n", dlerror()); return 2; }
    RadLaunchFn ref = find(lib, "ernie_hca_dspark_attn");
    if (!ref) { std::fprintf(stderr, "libref has no ernie_hca_dspark_attn\n"); return 2; }
    multiseq(ref, r4d_rad_ernie_hca_dspark_attn);
    for (int D : {64, 512}) rebased(ref, r4d_rad_ernie_hca_dspark_attn, D);
    for (int D : {2, 512}) {
        unsetenv("R4D_DSPARK_VALIDATE");
        suite(ref, r4d_rad_ernie_hca_dspark_attn, D, "host scan");
        card_suite(ref, r4d_rad_ernie_hca_dspark_attn, D);
    }
    {   /* the verdict the engine polls: bit 4 (seqused < qlen) after one bad launch, taken by the read, still 0 after a good launch */
        setenv("R4D_ERNIE_CARD_FORM", "1", 1);
        Fix bad(2), good(2);
        (void)r4d_ernie_dspark_take_error();
        bad.su[0] = 1;
        (void)bad.run(r4d_rad_ernie_hca_dspark_attn);
        const int first = r4d_ernie_dspark_take_error(), second = r4d_ernie_dspark_take_error();
        (void)good.run(r4d_rad_ernie_hca_dspark_attn);
        check(first == 4 && second == 0 && r4d_ernie_dspark_take_error() == 0, "take_error: bit 4 after a bad launch, cleared by the read, 0 after a good one");
    }
    {   /* reviewer's device-global-verdict finding: a valid launch after an invalid one must not erase the invalid one's refusal (the earlier design cleared one device-wide word at every
         * validator), and the events are counted, not OR-ed into one word: two invalid launches are two events, a poll in between sees only its own */
        setenv("R4D_ERNIE_CARD_FORM", "1", 1);
        Fix bad1(2), bad2(2), good1(2), good2(2);
        (void)r4d_ernie_dspark_take_error();
        bad1.su[0] = 1;
        bad2.cu = {0, 2, 3}; bad2.su = {2, 2};
        (void)bad1.run(r4d_rad_ernie_hca_dspark_attn);
        const int rg = good1.run(r4d_rad_ernie_hca_dspark_attn);
        const int after_good = r4d_ernie_dspark_take_error();
        (void)good2.run(r4d_rad_ernie_hca_dspark_attn);
        (void)bad2.run(r4d_rad_ernie_hca_dspark_attn);
        const int after_bad2 = r4d_ernie_dspark_take_error(), again = r4d_ernie_dspark_take_error();
        check(rg == RAD_OK && good1.out == good2.out && after_good == 4 && after_bad2 != 0 && again == 0,
              "take_error: a valid launch after an invalid one does not erase its verdict; a later invalid one is its own event (bits 0x" + [&] { char t[16]; std::snprintf(t, sizeof t, "%x/%x", (unsigned)after_good, (unsigned)after_bad2); return std::string(t); }() + ")");
    }
    {   /* the gate and the event log as protocols. The word: open (bit 31) with the refusal bits below; a launch starts by opening it, so what an earlier launch left is gone. */
        using namespace r4d_verdict;
        unsigned w = 0;
        const bool never_opened = gate_read(&w) == kGateLost;                      // a word nobody opened is not a pass
        gate_open(&w);
        const bool valid_passes = gate_read(&w) == 0;
        gate_add(&w, 4u);
        gate_add(&w, 64u);
        const bool refusal_seen = gate_read(&w) == (4u | 64u);
        gate_add(&w, 0u);                                                          // a block with nothing to say changes nothing
        const bool zero_noop = gate_read(&w) == (4u | 64u);
        gate_open(&w);                                                             // the next launch (or the next replay) of the call site
        const bool reopened = gate_read(&w) == 0;
        check(never_opened && valid_passes && refusal_seen && zero_noop && reopened, "gate: an unopened word reads LOST, an opened one passes, refusal bits accumulate, the next open clears them");
        Dev d{};
        Taken t;
        raise(&d, 4u);
        const bool one = t.fold(0, d) == 4;
        d.cnt += 1u;   /* an event counted whose entry is not published yet: its ring slot still holds the entry of the event before */
        const int stale = t.fold(0, d);
        raise(&d, 2u);
        for (int i = 0; i < 70; ++i) raise(&d, 1u);
        const int wrapped = t.fold(0, d);
        check(one && (stale & (int)kUnknownBits) && !(stale & 4) && (wrapped & (int)kUnknownBits), "event log: an unpublished entry is unknown, not the previous event's bits; an overrun is unknown");
    }
    {   /* the self-test refusal (R4D_ERNIE_SELFTEST_REFUSE=N): the Nth gated launch is reported as a violation, stores nothing, and reaches the poll; the others are untouched */
        setenv("R4D_ERNIE_CARD_FORM", "1", 1);
        r4d_verdict::State* st = r4d_verdict::registry().state_of(0);
        (void)r4d_ernie_dspark_take_error();
        st->inject = 3;
        bool ok = true;
        for (int i = 1; i <= 5; ++i) {
            Fix f(2);
            const int rc = f.run(r4d_rad_ernie_hca_dspark_attn);
            const int bits = r4d_ernie_dspark_take_error();
            ok = ok && rc == RAD_OK && (i == 3 ? (bits == (int)r4d_verdict::kInjected && f.untouched()) : (bits == 0 && !f.untouched()));
        }
        check(ok && st->inject == 0, "self-test refusal: the 3rd gated launch is refused with bit 0x10000000 and stores nothing, launches 1, 2, 4, 5 are valid");
    }
    for (int D : {2, 512}) {   /* recorded pass (review B1): played past kAuditPlays with validation on; audits equal the recording; a refusal on a replay is seen once and does not stick */
        setenv("R4D_ERNIE_CARD_FORM", "1", 1);
        Fix f(D);
        verdict_replay::Case c;
        c.issue = [&] { return f.run(r4d_rad_ernie_hca_dspark_attn); };
        c.corrupt = [&] { f.su[0] = 1; };
        c.restore = [&] { f.su[0] = 5; };
        c.poison = [&] { for (uint16_t& v : f.out) v = 0xdead; };
        c.output = [&] { return f.out; };
        c.untouched = [&] { return f.untouched(); };
        c.take = [&] { return r4d_ernie_dspark_take_error(); };
        std::string detail;
        const int sites0 = r4d_verdict::registry().gate_sites(0);
        const bool ok = verdict_replay::replay_audit(c, 1100, &detail);
        const int sites = r4d_verdict::registry().gate_sites(0) - sites0;
        check(ok && sites <= 1, "dspark recorded pass D" + std::to_string(D) + ": 1100 plays, audits equal the recording, a refusal on a replay is seen once and does not stick" + (ok ? "" : ": " + detail) + " (" + std::to_string(sites) + " gate word(s))");
    }
    for (int D : {2, 512}) {   /* one decision per block (review M3): the word changes the moment thread 0 has read it */
        setenv("R4D_ERNIE_CARD_FORM", "1", 1);
        for (int refused_first = 0; refused_first < 2; ++refused_first) {
            Fix f(D), ref_f(D);
            if (refused_first) f.su[0] = 1;
            (void)ref_f.run(ref);
            (void)r4d_ernie_dspark_take_error();
            g_flip_fired = false;
            g_flip_grid = emu_grids + 2;   // the validator, then the attention grid
            g_flip_to = refused_first ? r4d_verdict::kGateOpen : (r4d_verdict::kGateOpen | 4u);
            emu_barrier_hook = flip_hook;
            const int rc = f.run(r4d_rad_ernie_hca_dspark_attn);
            emu_barrier_hook = nullptr;
            (void)r4d_ernie_dspark_take_error();
            bool whole = rc == RAD_OK && g_flip_fired;
            for (int r = 0; r < 4; ++r) {
                int w = 0;
                for (int i = 0; i < D; ++i) w += f.out[(size_t)r * D + i] != 0xdead;
                whole = whole && (w == 0 || w == D);
            }
            check(whole, "dspark one decision per block D" + std::to_string(D) + (refused_first ? ", refusing when thread 0 reads, clean afterwards" : ", clean when thread 0 reads, refusing afterwards") + ": no row stored in part");
        }
        g_flip_grid = -1;
    }
    unsetenv("R4D_ERNIE_CARD_FORM");
    std::printf("%d failure(s)\n", g_fail);
    dlclose(lib);
    return g_fail ? 1 : 0;
}
