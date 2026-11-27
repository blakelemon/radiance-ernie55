/* tests/r4d_emu: ernie_hca_state_write and ernie_hca_compress (libr4d/r4d_ernie_hca.hip) on the host SIMT emulator, held to libref's refusals.
 *
 * libref validates every index before its first store: a state slot or a compress out slot outside [-1, pool), a state index of a gather list outside [-1, pool), a
 * negative position of a live row, a live row whose whole gather list is masked. Each negative case runs on libref and on the launcher, which must return
 * the same code and leave the output untouched; the controls (-1 slots skipped, masked entries, a repeated slot keeping the last row) must agree on values.
 * Then, in the CARD FORM (R4D_ERNIE_CARD_FORM=1: no host scan, the situation of a card build, whose operands are device memory), at the default settings, the kernels
 * must not read or write out of range and must put every violation libref refuses into the verdict the engine polls (r4d_ernie_hca_take_error), and a valid launch
 * must not; and where libref ACCEPTS a repeated slot (the last row wins) the card must produce libref's state whatever order the blocks of the grid run in
 * (R4D_EMU_BLOCK_ORDER=reverse runs them last to first). */
#include "rad_plugin.h"
#include "r4d_plugin.h"
#include "verdict_replay.h"
#include "r4d_ernie_verdict.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <functional>
#include <string>
#include <vector>

extern "C" int r4d_ernie_hca_take_error(void);
extern "C" int r4d_ernie_ops_take_error(void);
extern "C" int r4d_ernie_mla_take_error(void);
extern int emu_last_block;   /* threads per block of the last emulated launch (tests/r4d_emu/emu/emu.cpp): tells which kernel a launcher chose */
/* the events of the HCA kernels and of the position checks of q_prep / rope_tail (libr4d's ops file keeps its own log) */
static int take_all() { return r4d_ernie_hca_take_error() | r4d_ernie_ops_take_error() | r4d_ernie_mla_take_error(); }

namespace {
RadParam I(const char* k, int64_t v) { RadParam p{}; p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }
RadParam F(const char* k, double v) { RadParam p{}; p.key = k; p.kind = RAD_P_F64; p.dval = v; return p; }
RadTensor V(void* d, uint32_t dt, int64_t n) { RadTensor t{}; t.data = d; t.dtype = dt; t.rank = 1; t.shape[0] = n; t.stride[0] = 1; return t; }
RadTensor Mx(void* d, uint32_t dt, int64_t r, int64_t c) { RadTensor t{}; t.data = d; t.dtype = dt; t.rank = 2; t.shape[0] = r; t.shape[1] = c; t.stride[0] = c; t.stride[1] = 1; return t; }
RadLaunchFn find(void* lib, const char* op) {
    auto cnt = reinterpret_cast<int (*)()>(dlsym(lib, "rad_kernel_count"));
    auto at = reinterpret_cast<const RadKernelInfo* (*)(int)>(dlsym(lib, "rad_kernel_at"));
    for (int i = 0; cnt && i < cnt(); ++i) { const RadKernelInfo* k = at(i); if (k && k->op && !std::strcmp(k->op, op)) return k->launch; }
    return nullptr;
}
int g_fail = 0;
void check(bool ok, const std::string& name) { std::printf("%s %s\n", ok ? "PASS" : "FAIL", name.c_str()); if (!ok) ++g_fail; }

/* ---- state_write: ratio 4, head_dim 8 (W = 16), 6 state rows, 3 input rows ---- */
struct SW {
    static constexpr int M = 3, S = 6, W = 16, ratio = 4, D = 8;
    std::vector<float> kv, sc, ape, st;
    int32_t pos[M] = {5, 6, 7}, slot[M] = {0, 3, 5};
    SW() : kv(M * W), sc(M * W), ape(ratio * W), st(S * 2 * W, -7.f) {
        for (size_t i = 0; i < kv.size(); ++i) { kv[i] = 0.25f * (int)(i % 7) - 0.5f; sc[i] = 0.125f * (int)(i % 5); }
        for (size_t i = 0; i < ape.size(); ++i) ape[i] = 0.0625f * (int)(i % 9);
    }
    int run(RadLaunchFn fn) {
        RadTensor t[6] = {Mx(kv.data(), RAD_F32, M, W), Mx(sc.data(), RAD_F32, M, W), Mx(ape.data(), RAD_F32, ratio, W), V(pos, RAD_I32, M), V(slot, RAD_I32, M), Mx(st.data(), RAD_F32, S, 2 * W)};
        RadParam p[] = {I("ratio", ratio), I("head_dim", D)};
        RadArgs a{}; a.t = t; a.n_t = 6; a.p = p; a.n_p = 2;
        return fn(&a, nullptr);
    }
    bool untouched() const { for (float v : st) if (v != -7.f) return false; return true; }
};
void state_write_suite(RadLaunchFn ref, RadLaunchFn dev) {
    struct C { const char* name; void (*mutate)(SW&); };
    const C negs[] = {{"slot -2", [](SW& f) { f.slot[1] = -2; }}, {"slot == pool", [](SW& f) { f.slot[2] = SW::S; }},
                      {"slot far past the pool", [](SW& f) { f.slot[0] = 999; }}, {"negative position of a live row", [](SW& f) { f.pos[1] = -1; }}};
    for (const C& n : negs) {
        SW a, b; n.mutate(a); n.mutate(b);
        const int ra = a.run(ref), rb = b.run(dev);
        check(ra == RAD_E_SHAPE && rb == RAD_E_SHAPE && a.untouched() && b.untouched(), std::string("state_write refusal: ") + n.name + " (libref " + std::to_string(ra) + ", device " + std::to_string(rb) + (b.untouched() ? "" : ", WRITTEN") + ")");
    }
    const C poss[] = {{"valid", [](SW&) {}}, {"slot -1 skips the row, its position is not read", [](SW& f) { f.slot[1] = -1; f.pos[1] = -5; }},
                      {"a repeated slot keeps the last row", [](SW& f) { f.slot[0] = 2; f.slot[2] = 2; }}};
    for (const C& n : poss) {
        SW a, b; n.mutate(a); n.mutate(b);
        const int ra = a.run(ref), rb = b.run(dev);
        check(ra == RAD_OK && rb == RAD_OK && a.st == b.st, std::string("state_write control: ") + n.name);
    }
    /* the card form: no host scan */
    setenv("R4D_ERNIE_CARD_FORM", "1", 1);
    (void)take_all();
    struct L { const char* name; void (*mutate)(SW&); int want; };
    const L lat[] = {{"valid", [](SW&) {}, 0}, {"slot -1", [](SW& f) { f.slot[1] = -1; f.pos[1] = -5; }, 0}, {"slot -2", [](SW& f) { f.slot[1] = -2; }, 1},
                     {"slot == pool", [](SW& f) { f.slot[2] = SW::S; }, 1}, {"negative position of a live row", [](SW& f) { f.pos[1] = -1; }, 1}};
    for (const L& c : lat) {
        SW f; c.mutate(f);
        const int rc = f.run(dev), got = take_all();
        check(rc == RAD_OK && (c.want ? got != 0 : got == 0) && take_all() == 0, std::string("state_write latch: ") + c.name + " (rc " + std::to_string(rc) + ", latched " + std::to_string(got) + ")");
    }
    unsetenv("R4D_ERNIE_CARD_FORM");
}

/* Duplicate state slots: libref accepts them and the LAST row wins. In the card form the launcher cannot scan the slots, so the kernel itself must give libref's state
 * for any number of rows and any order the blocks of the grid run in. */
void state_write_dup_suite(RadLaunchFn ref, RadLaunchFn dev) {
    setenv("R4D_ERNIE_CARD_FORM", "1", 1);
    const int pool = 64, ratio = 4, D = 8, W = 16;
    for (int M : {3, 7, 200, 700, 2048})
      for (int use_scratch : {1, 0}) {   /* the last-writer table is the op's scratch; a caller that gives none keeps the scan: both are libref's state */
        if (!use_scratch && M < 700) continue;
        for (const char* order : {"forward", "reverse"}) {
            setenv("R4D_EMU_BLOCK_ORDER", order, 1);
            std::vector<float> kv((size_t)M * W), sc((size_t)M * W), ape((size_t)ratio * W), a_st((size_t)pool * 2 * W, -7.f), b_st(a_st);
            std::vector<int32_t> pos((size_t)M), slot((size_t)M);
            for (size_t i = 0; i < kv.size(); ++i) { kv[i] = 0.25f * (int)(i % 7) - 0.5f + 0.001f * (int)(i / 7 % 13); sc[i] = 0.125f * (int)(i % 5); }
            for (size_t i = 0; i < ape.size(); ++i) ape[i] = 0.0625f * (int)(i % 9);
            for (int m = 0; m < M; ++m) { pos[m] = m % 29; slot[m] = M <= 7 ? (m * 3) % 4 : (m * 7) % 40; if (m % 11 == 5) slot[m] = -1; }
            auto call = [&](RadLaunchFn fn, std::vector<float>& st) {
                RadTensor t[6] = {Mx(kv.data(), RAD_F32, M, W), Mx(sc.data(), RAD_F32, M, W), Mx(ape.data(), RAD_F32, ratio, W), V(pos.data(), RAD_I32, M), V(slot.data(), RAD_I32, M), Mx(st.data(), RAD_F32, pool, 2 * W)};
                RadParam p[] = {I("ratio", ratio), I("head_dim", D), I("M", M)};
                RadArgs a{}; a.t = t; a.n_t = 6; a.p = p; a.n_p = 3;
                std::vector<int32_t> scr;
                if (fn == dev && use_scratch) { const int64_t n = r4d_rad_ernie_hca_state_write_scratch(&a); if (n > 0) { scr.assign((size_t)n / 4, (int32_t)0x5A5A5A5A); a.scratch = scr.data(); a.scratch_bytes = n; } }
                return fn(&a, nullptr);
            };
            (void)take_all();
            const int ra = call(ref, a_st), rb = call(dev, b_st), v = take_all();
            check(ra == RAD_OK && rb == RAD_OK && a_st == b_st && v == 0, std::string("state_write card form: repeated slots, M=") + std::to_string(M) + (use_scratch ? ", table scratch" : ", no scratch (scan)") + ", blocks " + order + ": state == libref (last row wins), verdict empty");
        }
    }
    unsetenv("R4D_EMU_BLOCK_ORDER");
    unsetenv("R4D_ERNIE_CARD_FORM");
}

/* ---- compress: ratio 4 (coff 2, Wn 8), head_dim 8, rope 4, 6 state rows, 4 out rows, 3 input rows ---- */
struct CP {
    static constexpr int M = 3, S = 6, Rr = 4, ratio = 4, D = 8, W = 16, Wn = 8;
    std::vector<float> st;
    std::vector<uint16_t> nw, out;
    int32_t gs[M * Wn], pos[M] = {3, 7, 5}, os[M] = {0, 1, 2};
    CP() : st(S * 2 * W), nw(D), out(Rr * D, 0xdead) {
        for (size_t i = 0; i < st.size(); ++i) st[i] = 0.25f * (int)((i * 3) % 11) - 1.0f;
        for (int i = 0; i < D; ++i) nw[i] = rad_f32_to_bf16(0.75f + 0.0625f * i);
        for (int m = 0; m < M; ++m) for (int w = 0; w < Wn; ++w) gs[m * Wn + w] = (m + w) % S;
    }
    int run(RadLaunchFn fn) {
        RadTensor t[6] = {Mx(st.data(), RAD_F32, S, 2 * W), Mx(gs, RAD_I32, M, Wn), V(pos, RAD_I32, M), V(os, RAD_I32, M), V(nw.data(), RAD_BF16, D), Mx(out.data(), RAD_BF16, Rr, D)};
        RadParam p[] = {I("ratio", ratio), I("head_dim", D), I("rope_dim", 4), F("eps", 1e-6), F("theta", 10000.0)};
        RadArgs a{}; a.t = t; a.n_t = 6; a.p = p; a.n_p = 5;
        return fn(&a, nullptr);
    }
    bool untouched() const { for (uint16_t v : out) if (v != 0xdead) return false; return true; }
    bool close(const CP& o) const { for (size_t i = 0; i < out.size(); ++i) if (std::abs((int)out[i] - (int)o.out[i]) > 1) return false; return true; }
};
void compress_suite(RadLaunchFn ref, RadLaunchFn dev) {
    struct C { const char* name; void (*mutate)(CP&); };
    const C negs[] = {{"gather entry -2", [](CP& f) { f.gs[1] = -2; }}, {"gather entry == pool", [](CP& f) { f.gs[9] = CP::S; }},
                      {"gather entry bad in a row that does not complete a group", [](CP& f) { f.gs[2 * CP::Wn + 3] = 77; }},
                      {"out slot -2", [](CP& f) { f.os[1] = -2; }}, {"out slot == out rows", [](CP& f) { f.os[0] = CP::Rr; }},
                      {"negative position of a live row", [](CP& f) { f.pos[1] = -1; }},
                      {"a completing row whose gather list is all masked", [](CP& f) { for (int w = 0; w < CP::Wn; ++w) f.gs[w] = -1; }},
                      {"a completing row that pools to a non-finite value", [](CP& f) { f.st[1 * 2 * CP::W + CP::W] = NAN; f.st[2 * 2 * CP::W + CP::W] = NAN; }}};
    for (const C& n : negs) {
        CP a, b; n.mutate(a); n.mutate(b);
        const int ra = a.run(ref), rb = b.run(dev);
        check(ra == RAD_E_SHAPE && rb == RAD_E_SHAPE && a.untouched() && b.untouched(), std::string("compress refusal: ") + n.name + " (libref " + std::to_string(ra) + ", device " + std::to_string(rb) + (b.untouched() ? "" : ", WRITTEN") + ")");
    }
    const C poss[] = {{"valid", [](CP&) {}}, {"masked entries (-1) are skipped", [](CP& f) { f.gs[0] = -1; f.gs[3] = -1; }},
                      {"out slot -1 ignores the row altogether", [](CP& f) { f.os[1] = -1; f.pos[1] = -9; f.gs[8] = 1234; }}};
    for (const C& n : poss) {
        CP a, b; n.mutate(a); n.mutate(b);
        const int ra = a.run(ref), rb = b.run(dev);
        check(ra == RAD_OK && rb == RAD_OK && a.close(b), std::string("compress control: ") + n.name + " (libref " + std::to_string(ra) + ", device " + std::to_string(rb) + ")");
    }
    setenv("R4D_ERNIE_CARD_FORM", "1", 1);
    (void)take_all();
    struct L { const char* name; void (*mutate)(CP&); int want; };
    const L lat[] = {{"valid", [](CP&) {}, 0}, {"masked -1", [](CP& f) { f.gs[0] = -1; }, 0}, {"gather entry -2", [](CP& f) { f.gs[1] = -2; }, 1},
                     {"gather entry == pool", [](CP& f) { f.gs[1] = CP::S; }, 1}, {"out slot -2", [](CP& f) { f.os[1] = -2; }, 1},
                     {"out slot == out rows", [](CP& f) { f.os[0] = CP::Rr; }, 1}, {"negative position", [](CP& f) { f.pos[1] = -1; }, 1},
                     {"gather entry bad in a row that does not complete a group (pos[2]=5, gs[2*8+3]=77)", [](CP& f) { f.gs[2 * CP::Wn + 3] = 77; }, 1},
                     {"a completing row whose gather list is all masked", [](CP& f) { for (int w = 0; w < CP::Wn; ++w) f.gs[w] = -1; }, 1},
                     {"a completing row that pools to a non-finite value", [](CP& f) { f.st[1 * 2 * CP::W + CP::W] = NAN; f.st[2 * 2 * CP::W + CP::W] = NAN; }, 1}};
    for (const L& c : lat) {
        CP f; c.mutate(f);
        const int rc = f.run(dev), got = take_all();
        check(rc == RAD_OK && (c.want ? got != 0 : got == 0) && take_all() == 0, std::string("compress latch: ") + c.name + " (rc " + std::to_string(rc) + ", latched " + std::to_string(got) + ")");
    }
    {   /* a refused completing row must not leave a stored row behind: an all-masked or non-finite pool is not state a later step may read */
        CP f; for (int w = 0; w < CP::Wn; ++w) f.gs[w] = -1;
        (void)f.run(dev); (void)take_all();
        bool row0 = true; for (int c = 0; c < CP::D; ++c) row0 = row0 && f.out[c] == 0xdead;
        check(row0, "compress card form: the all-masked completing row stores nothing (no zero row as apparently valid state)");
        CP g; g.st[1 * 2 * CP::W + CP::W] = NAN; g.st[2 * 2 * CP::W + CP::W] = NAN;
        (void)g.run(dev); (void)take_all();
        bool row1 = true; for (int c = 0; c < CP::D; ++c) row1 = row1 && g.out[1 * CP::D + c] == 0xdead;
        check(row1, "compress card form: a non-finite completing row stores nothing");
    }
    unsetenv("R4D_ERNIE_CARD_FORM");
}

/* Repeated out slots among completing rows: libref accepts them (the last completing row keeps the slot); the card must match in any block order, for few rows and many. */
/* review m4: past 256 rows the duplicate-slot scan (O(M^2) a layer) is replaced by a last-writer pass (clear, mark, then the store kernel): the launches of one issue are 3 at
 * M = 700 and 1 at M = 200, and the work of the kernels grows with M: the number of block-level slot reads in the scan form is counted by the emulator's grids, not timed. */
void dedupe_launch_shape(RadLaunchFn sw, RadLaunchFn cp) {
    setenv("R4D_ERNIE_CARD_FORM", "1", 1);
    const int pool = 64, ratio = 4, D = 8, W = 16;
    for (int M : {200, 256, 257, 700}) {
        std::vector<float> kv((size_t)M * W, 0.5f), sc((size_t)M * W, 0.25f), ape((size_t)ratio * W, 0.f), st((size_t)pool * 2 * W, -7.f);
        std::vector<int32_t> pos((size_t)M, 5), slot((size_t)M);
        for (int m = 0; m < M; ++m) slot[m] = (m * 7) % 40;
        std::vector<int32_t> scr;
        auto issue = [&] {
            RadTensor t[6] = {Mx(kv.data(), RAD_F32, M, W), Mx(sc.data(), RAD_F32, M, W), Mx(ape.data(), RAD_F32, ratio, W), V(pos.data(), RAD_I32, M), V(slot.data(), RAD_I32, M), Mx(st.data(), RAD_F32, pool, 2 * W)};
            RadParam p[] = {I("ratio", ratio), I("head_dim", D), I("M", M)};
            RadArgs a{}; a.t = t; a.n_t = 6; a.p = p; a.n_p = 3;
            const int64_t n = r4d_rad_ernie_hca_state_write_scratch(&a);
            if (n > 0) { scr.assign((size_t)n / 4, 0x5A5A5A5A); a.scratch = scr.data(); a.scratch_bytes = n; }
            return sw(&a, nullptr);
        };
        int rc = -1;
        const long long d0 = r4d_verdict::stats().dedupe_launches.load();
        const auto tape = verdict_replay::record(issue, &rc);
        const size_t want = M > 256 ? 3 : 1;
        check(r4d_verdict::stats().dedupe_launches.load() - d0 == (M > 256 ? 2 : 0), "state_write M=" + std::to_string(M) + ": the last-writer launches are counted (m-6)");
        check(rc == RAD_OK && tape.size() == want, "state_write M=" + std::to_string(M) + ": " + std::to_string(tape.size()) + " launch(es) (want " + std::to_string(want) + ": the last-writer pass only past 256 rows)");
        std::string why;
        const auto again = verdict_replay::record(issue);
        check(verdict_replay::agree(tape, again, &why), "state_write M=" + std::to_string(M) + ": a second issue has the same kernel-argument bytes (" + why + ")");
        (void)take_all();
    }
    (void)cp;
    unsetenv("R4D_ERNIE_CARD_FORM");
}
void compress_dup_suite(RadLaunchFn ref, RadLaunchFn dev) {
    setenv("R4D_ERNIE_CARD_FORM", "1", 1);
    const int ratio = 4, D = 8, W = 16, Wn = 8, S = 12;
    for (int M : {3, 8, 150, 600, 2048})
      for (int use_scratch : {1, 0}) {
        if (!use_scratch && M < 600) continue;
        for (const char* order : {"forward", "reverse"}) {
            setenv("R4D_EMU_BLOCK_ORDER", order, 1);
            const int Rr = M <= 8 ? 2 : 20;
            std::vector<float> st((size_t)S * 2 * W);
            for (size_t i = 0; i < st.size(); ++i) st[i] = 0.25f * (int)((i * 3) % 11) - 1.0f;
            std::vector<uint16_t> nw(D), a_out((size_t)Rr * D, 0xdead), b_out(a_out);
            for (int i = 0; i < D; ++i) nw[i] = rad_f32_to_bf16(0.75f + 0.0625f * i);
            std::vector<int32_t> gs((size_t)M * Wn), pos((size_t)M), os((size_t)M);
            for (int m = 0; m < M; ++m) {
                for (int w = 0; w < Wn; ++w) gs[(size_t)m * Wn + w] = (m * 5 + w) % S;
                pos[m] = (m % 3 == 0) ? 4 * (m % 17) + 3 : 4 * (m % 17) + 1;   /* every third row completes a group */
                os[m] = (m % 3 == 0) ? (M <= 8 ? m % 2 : (m / 3) % 7) : -1 + (m % 5 == 1 ? 2 : 0);
            }
            auto call = [&](RadLaunchFn fn, std::vector<uint16_t>& out) {
                RadTensor t[6] = {Mx(st.data(), RAD_F32, S, 2 * W), Mx(gs.data(), RAD_I32, M, Wn), V(pos.data(), RAD_I32, M), V(os.data(), RAD_I32, M), V(nw.data(), RAD_BF16, D), Mx(out.data(), RAD_BF16, Rr, D)};
                RadParam p[] = {I("ratio", ratio), I("head_dim", D), I("rope_dim", 4), F("eps", 1e-6), F("theta", 10000.0), I("M", M)};
                RadArgs a{}; a.t = t; a.n_t = 6; a.p = p; a.n_p = 6;
                std::vector<int32_t> scr;
                if (fn == dev && use_scratch) { const int64_t n = r4d_rad_ernie_hca_compress_scratch(&a); if (n > 0) { scr.assign((size_t)n / 4, (int32_t)0x5A5A5A5A); a.scratch = scr.data(); a.scratch_bytes = n; } }
                return fn(&a, nullptr);
            };
            (void)take_all();
            const int ra = call(ref, a_out), rb = call(dev, b_out), v = take_all();
            bool close = ra == RAD_OK && rb == RAD_OK;
            for (size_t i = 0; close && i < a_out.size(); ++i) close = std::abs((int)a_out[i] - (int)b_out[i]) <= 1;
            check(close && v == 0, std::string("compress card form: repeated out slots, M=") + std::to_string(M) + (use_scratch ? ", table scratch" : ", no scratch (scan)") + ", blocks " + order + ": out == libref (last completing row wins) (libref " + std::to_string(ra) + ", device " + std::to_string(rb) + ", verdict " + std::to_string(v) + ")");
        }
    }
    {   /* review: a row a later row overwrites is still judged. Rows 0 and 1 complete a group into the same out slot; row 0 pools slot 0 only (NaN key), row 1 slot 1 only (finite):
         * libref refuses the launch, the card must raise the non-finite bit whichever block order runs */
        for (const char* order : {"forward", "reverse"}) {
            setenv("R4D_EMU_BLOCK_ORDER", order, 1);
            CP a, b;
            for (CP* f : {&a, &b}) {
                for (int w = 0; w < CP::Wn; ++w) { f->gs[w] = w == 0 ? 0 : -1; f->gs[CP::Wn + w] = w == 0 ? 1 : -1; }
                f->os[0] = 0; f->os[1] = 0; f->os[2] = -1;
                f->st[0] = NAN;   /* kv channel 0 of state row 0 */
            }
            (void)take_all();
            const int ra = a.run(ref), rb = b.run(dev), v = take_all();
            check(ra == RAD_E_SHAPE && rb == RAD_OK && (v & 16), std::string("compress card form: a non-finite completing row is refused although a later row overwrites its slot, blocks ") + order +
                  " (libref " + std::to_string(ra) + ", launch " + std::to_string(rb) + ", verdict " + std::to_string(v) + ")");
        }
    }
    unsetenv("R4D_EMU_BLOCK_ORDER");
    unsetenv("R4D_ERNIE_CARD_FORM");
}

/* ---- rotary position operands: ernie_hca_q_prep, ernie_hca_wo_a, ernie_rope_tail. libref refuses a position below 0 (any row) before its first store. In the card form the
 * kernel must latch it, and must not store the row whose position is invalid (a rotary at a negative position is a value libref never produces). Valid rows are as libref. ---- */
void pos_suite(void* lib) {
    setenv("R4D_ERNIE_CARD_FORM", "1", 1);
    /* Two shapes: D 8 / R 4 (the general kernels) and D 32 / R 32, which takes the chunked WMMA small-row kernel of ernie_hca_wo_a (D % 32 == 0, R % 32 == 0, R <= 64, M <= 64) */
    for (const int shape : {0, 1}) {
    const int D = shape ? 32 : 8, R = shape ? 32 : 4;
    constexpr int M = 3, H = 2, G = 2, K = 3;
    struct Op { const char* name; const char* ref_name; RadLaunchFn dev; };
    auto bf = [](int i) { return rad_f32_to_bf16(0.125f * (float)((i * 7) % 13 - 6)); };
    for (int which = 0; which < 3; ++which) {
        const char* name = which == 0 ? "ernie_hca_q_prep" : which == 1 ? "ernie_hca_wo_a" : "ernie_rope_tail";
        RadLaunchFn ref = find(lib, name);
        RadLaunchFn dev = which == 0 ? r4d_rad_ernie_hca_q_prep : which == 1 ? r4d_rad_ernie_hca_wo_a : r4d_rad_ernie_rope_tail;
        if (!ref) { check(false, std::string("libref has ") + name); continue; }
        for (int bad = -1; bad < M; ++bad) {
            std::vector<uint16_t> x((size_t)M * H * D), xr, w((size_t)G * K * (H / G) * D), out_a((size_t)M * G * K, 0xdead), out_b(out_a);
            for (size_t i = 0; i < x.size(); ++i) x[i] = bf((int)i);
            for (size_t i = 0; i < w.size(); ++i) w[i] = bf((int)i + 3);
            int32_t pos[M] = {5, 9, 2};
            if (bad >= 0) pos[bad] = -4;
            std::vector<uint16_t> xa = x, xb = x;
            auto call = [&](RadLaunchFn fn, std::vector<uint16_t>& xx, std::vector<uint16_t>& oo) {
                if (which == 0) {
                    RadTensor t[2] = {Mx(xx.data(), RAD_BF16, M, H * D), V(pos, RAD_I32, M)};
                    RadParam p[] = {I("heads", H), I("head_dim", D), I("rope_dim", R), F("eps", 1e-6), F("theta", 10000.0)};
                    RadArgs a{}; a.t = t; a.n_t = 2; a.p = p; a.n_p = 5; return fn(&a, nullptr);
                }
                if (which == 1) {
                    RadTensor t[4] = {Mx(xx.data(), RAD_BF16, M, H * D), V(pos, RAD_I32, M), Mx(w.data(), RAD_BF16, G * K, (H / G) * D), Mx(oo.data(), RAD_BF16, M, G * K)};
                    RadParam p[] = {I("heads", H), I("head_dim", D), I("rope_dim", R), I("groups", G), I("rank", K), F("theta", 10000.0)};
                    RadArgs a{}; a.t = t; a.n_t = 4; a.p = p; a.n_p = 6; return fn(&a, nullptr);
                }
                RadTensor t[2] = {Mx(xx.data(), RAD_BF16, M, H * D), V(pos, RAD_I32, M)};
                RadParam p[] = {I("heads", H), I("width", D), I("rope_dim", R), F("theta", 10000.0)};
                RadArgs a{}; a.t = t; a.n_t = 2; a.p = p; a.n_p = 4; return fn(&a, nullptr);
            };
            (void)take_all();
            const int ra = call(ref, xa, out_a), rb = call(dev, xb, out_b), got = take_all();
            const std::string tag = std::string(name) + (shape ? " (D 32 R 32)" : "") + " card form, " + (bad < 0 ? "valid positions" : "row " + std::to_string(bad) + " has position -4");
            if (bad < 0) {
                bool close = ra == RAD_OK && rb == RAD_OK;
                for (size_t i = 0; close && i < xa.size(); ++i) close = std::abs((int)xa[i] - (int)xb[i]) <= 1;
                for (size_t i = 0; close && i < out_a.size(); ++i) close = std::abs((int)out_a[i] - (int)out_b[i]) <= 1;
                check(close && got == 0, tag + " (verdict empty, values as libref)");
            } else {
                /* the offending row is not stored: the in-place ops leave its input row, wo_a leaves its output row untouched */
                bool row_kept = true;
                if (which == 1) { for (int c = 0; c < G * K; ++c) row_kept = row_kept && out_b[(size_t)bad * G * K + c] == 0xdead; }
                else            { for (int c = 0; c < H * D; ++c) row_kept = row_kept && xb[(size_t)bad * H * D + c] == x[(size_t)bad * H * D + c]; }
                check(ra == RAD_E_SHAPE && rb == RAD_OK && got != 0 && row_kept, tag + " (libref rc " + std::to_string(ra) + ", launch rc " + std::to_string(rb) + ", verdict " + std::to_string(got) + (row_kept ? ", row not stored" : ", ROW STORED") + ")");
            }
        }
    }
    }
    unsetenv("R4D_ERNIE_CARD_FORM");
}

/* Validation OFF (R4D_ERNIE_VALIDATE=0 in the card form, a card's default): the kernels do what base d2258ef8 did. A gather entry outside [-1, pool) is masked, an all-masked
 * completing row pools to 0 and is stored, a non-finite pooled row is stored, a row with a negative position is computed; no event is polled. Validation on (above) skips them. */
void validation_off_suite(void* lib, RadLaunchFn cp_ref) {
    setenv("R4D_ERNIE_CARD_FORM", "1", 1);
    setenv("R4D_ERNIE_VALIDATE", "0", 1);
    (void)take_all();
    {   /* a bad gather entry == the same entry masked (-1) */
        for (int bad : {-2, CP::S, 77}) {
            CP f, g; f.gs[1] = bad; g.gs[1] = -1;
            const int ra = f.run(r4d_rad_ernie_hca_compress), rb = g.run(cp_ref), got = take_all();
            check(ra == RAD_OK && rb == RAD_OK && f.close(g) && got == 0, "compress validation off: gather entry " + std::to_string(bad) + " is masked, row stored as with -1, no event");
        }
        CP f, g; f.gs[2 * CP::Wn + 3] = 77;
        const int ra = f.run(r4d_rad_ernie_hca_compress), rb = g.run(cp_ref), got = take_all();
        check(ra == RAD_OK && rb == RAD_OK && f.close(g) && got == 0, "compress validation off: a bad entry in a row that completes no group changes nothing");
    }
    {   /* all masked: pools to 0; the row is stored (zeros), the others as in the control */
        CP f, g; for (int w = 0; w < CP::Wn; ++w) f.gs[w] = -1;
        const int ra = f.run(r4d_rad_ernie_hca_compress), rb = g.run(cp_ref), got = take_all();
        bool zero = true, rest = true;
        for (int c = 0; c < CP::D; ++c) zero = zero && (f.out[c] & 0x7fff) == 0;
        for (int i = CP::D; i < CP::Rr * CP::D; ++i) rest = rest && std::abs((int)f.out[i] - (int)g.out[i]) <= 1;
        check(ra == RAD_OK && rb == RAD_OK && zero && rest && got == 0, "compress validation off: the all-masked completing row stores zeros (base), the others are unchanged, no event");
    }
    {   /* a non-finite pooled row is stored (base stores whatever the pooling gave) */
        CP f; f.st[1 * 2 * CP::W + CP::W] = NAN; f.st[2 * 2 * CP::W + CP::W] = NAN;
        const int ra = f.run(r4d_rad_ernie_hca_compress), got = take_all();
        bool stored = false;
        for (int c = 0; c < CP::D; ++c) stored = stored || f.out[1 * CP::D + c] != 0xdead;
        check(ra == RAD_OK && stored && got == 0, "compress validation off: a non-finite completing row is stored, no event");
    }
    {   /* a negative position of a live compress row: base returned without storing (p < 0 check is base's) */
        CP f; f.pos[1] = -1;
        const int ra = f.run(r4d_rad_ernie_hca_compress), got = take_all();
        bool row1 = true; for (int c = 0; c < CP::D; ++c) row1 = row1 && f.out[1 * CP::D + c] == 0xdead;
        check(ra == RAD_OK && row1 && got == 0, "compress validation off: a negative position is skipped as in base (no store), no event");
    }
    constexpr int M = 3, H = 2, D = 8, R = 4, G = 2, K = 3;
    auto bf = [](int i) { return rad_f32_to_bf16(0.125f * (float)((i * 7) % 13 - 6)); };
    for (int which = 0; which < 3; ++which) {
        const char* name = which == 0 ? "ernie_hca_q_prep" : which == 1 ? "ernie_hca_wo_a" : "ernie_rope_tail";
        RadLaunchFn dev = which == 0 ? r4d_rad_ernie_hca_q_prep : which == 1 ? r4d_rad_ernie_hca_wo_a : r4d_rad_ernie_rope_tail;
        for (int bad = 0; bad < M; ++bad) {
            std::vector<uint16_t> x((size_t)M * H * D), w((size_t)G * K * (H / G) * D), out((size_t)M * G * K, 0xdead);
            for (size_t i = 0; i < x.size(); ++i) x[i] = bf((int)i);
            for (size_t i = 0; i < w.size(); ++i) w[i] = bf((int)i + 3);
            int32_t pos[M] = {5, 9, 2};
            pos[bad] = -4;
            std::vector<uint16_t> xb = x;
            int rc;
            if (which == 0) {
                RadTensor t[2] = {Mx(xb.data(), RAD_BF16, M, H * D), V(pos, RAD_I32, M)};
                RadParam p[] = {I("heads", H), I("head_dim", D), I("rope_dim", R), F("eps", 1e-6), F("theta", 10000.0)};
                RadArgs a{}; a.t = t; a.n_t = 2; a.p = p; a.n_p = 5; rc = dev(&a, nullptr);
            } else if (which == 1) {
                RadTensor t[4] = {Mx(xb.data(), RAD_BF16, M, H * D), V(pos, RAD_I32, M), Mx(w.data(), RAD_BF16, G * K, (H / G) * D), Mx(out.data(), RAD_BF16, M, G * K)};
                RadParam p[] = {I("heads", H), I("head_dim", D), I("rope_dim", R), I("groups", G), I("rank", K), F("theta", 10000.0)};
                RadArgs a{}; a.t = t; a.n_t = 4; a.p = p; a.n_p = 6; rc = dev(&a, nullptr);
            } else {
                RadTensor t[2] = {Mx(xb.data(), RAD_BF16, M, H * D), V(pos, RAD_I32, M)};
                RadParam p[] = {I("heads", H), I("width", D), I("rope_dim", R), F("theta", 10000.0)};
                RadArgs a{}; a.t = t; a.n_t = 2; a.p = p; a.n_p = 4; rc = dev(&a, nullptr);
            }
            const int got = take_all();
            bool computed = false;
            if (which == 1) { for (int c = 0; c < G * K; ++c) computed = computed || out[(size_t)bad * G * K + c] != 0xdead; }
            else            { for (int c = 0; c < H * D; ++c) computed = computed || xb[(size_t)bad * H * D + c] != x[(size_t)bad * H * D + c]; }
            check(rc == RAD_OK && computed && got == 0, std::string(name) + " validation off: row " + std::to_string(bad) + " (position -4) is computed as base, no event");
        }
    }
    (void)lib;
    unsetenv("R4D_ERNIE_VALIDATE");
    unsetenv("R4D_ERNIE_CARD_FORM");
}
/* ---- the FAST q_prep kernels (D512 with 4 or fewer rows: 32 threads, D512 wider: 128, D128: 128) and the TILED wo_a kernel (more than 16 rows): the generic kernels are
 * covered by pos_suite above; these are separate kernel bodies with their own position check. Each case asserts which kernel the launcher chose (threads per block of the
 * last launch), then, for every row in turn at position -4: libref refuses, the launch returns OK with the verdict raised, the bad row is not stored, and every other row
 * equals libref's result for the same call with all positions valid. ---- */
void pos_fast_suite(void* lib) {
    setenv("R4D_ERNIE_CARD_FORM", "1", 1);
    auto bf = [](int i) { return rad_f32_to_bf16(0.125f * (float)((i * 7) % 13 - 6)); };
    struct Q { const char* name; int D, H, M, threads; };
    const Q qs[] = {{"q_prep D512 M=3 (fast512, 32 threads)", 512, 2, 3, 32}, {"q_prep D512 M=6 H=5 (fast512 wide, 128 threads)", 512, 5, 6, 128},
                    {"q_prep D128 M=4 H=5 (fast128, 128 threads)", 128, 5, 4, 128}};
    RadLaunchFn qref = find(lib, "ernie_hca_q_prep");
    if (!qref) check(false, "libref has ernie_hca_q_prep");
    for (const Q& c : qs) {
        if (!qref) break;
        const int R = 64;
        std::vector<int32_t> good((size_t)c.M);
        for (int m = 0; m < c.M; ++m) good[(size_t)m] = 3 + 7 * m;
        std::vector<uint16_t> x0((size_t)c.M * c.H * c.D);
        for (size_t i = 0; i < x0.size(); ++i) x0[i] = bf((int)i);
        auto call = [&](RadLaunchFn fn, std::vector<uint16_t>& xx, std::vector<int32_t>& pp) {
            RadTensor t[2] = {Mx(xx.data(), RAD_BF16, c.M, (int64_t)c.H * c.D), V(pp.data(), RAD_I32, c.M)};
            RadParam p[] = {I("heads", c.H), I("head_dim", c.D), I("rope_dim", R), F("eps", 1e-6), F("theta", 10000.0)};
            RadArgs a{}; a.t = t; a.n_t = 2; a.p = p; a.n_p = 5; return fn(&a, nullptr);
        };
        std::vector<uint16_t> xr = x0; std::vector<int32_t> g2 = good;
        const int rr = call(qref, xr, g2);
        for (int bad = -1; bad < c.M; ++bad) {
            std::vector<uint16_t> xd = x0; std::vector<int32_t> pp = good;
            if (bad >= 0) pp[(size_t)bad] = -4;
            (void)take_all();
            const int rd = call(r4d_rad_ernie_hca_q_prep, xd, pp), got = take_all(), blk = emu_last_block;
            bool rows_ok = rr == RAD_OK && rd == RAD_OK;
            const size_t rw = (size_t)c.H * c.D;
            for (int m = 0; m < c.M; ++m)
                for (size_t k = 0; k < rw; ++k) {
                    const size_t i = (size_t)m * rw + k;
                    if (m == bad) rows_ok = rows_ok && xd[i] == x0[i];
                    else rows_ok = rows_ok && std::abs((int)xd[i] - (int)xr[i]) <= 1;
                }
            check(blk == c.threads && rows_ok && (bad < 0 ? got == 0 : got != 0),
                  std::string(c.name) + (bad < 0 ? ", valid positions" : ", row " + std::to_string(bad) + " at position -4") + " (launch rc " + std::to_string(rd) + ", verdict " + std::to_string(got) +
                  ", block " + std::to_string(blk) + (rows_ok ? ", bad row kept, other rows as libref" : ", ROWS DIFFER") + ")");
        }
    }
    RadLaunchFn wref = find(lib, "ernie_hca_wo_a");
    if (!wref) check(false, "libref has ernie_hca_wo_a");
    if (wref) {
        constexpr int M = 70, H = 2, D = 64, R = 32, G = 2, K = 3, per = (H / G) * D;
        for (int shift : {0, 1}) {   /* shift 1: the operand pointers are 2 bytes off a 16-byte boundary (the non-vector instantiation) */
            std::vector<uint16_t> obuf((size_t)M * H * D + 8), wbuf((size_t)G * K * per + 8);
            uint16_t* o = obuf.data() + shift; uint16_t* w = wbuf.data() + shift;
            for (size_t i = 0; i < (size_t)M * H * D; ++i) o[i] = bf((int)i);
            for (size_t i = 0; i < (size_t)G * K * per; ++i) w[i] = bf((int)i + 5);
            std::vector<int32_t> good((size_t)M);
            for (int m = 0; m < M; ++m) good[(size_t)m] = 2 + 3 * m;
            auto call = [&](RadLaunchFn fn, std::vector<int32_t>& pp, std::vector<uint16_t>& out) {
                RadTensor t[4] = {Mx(o, RAD_BF16, M, H * D), V(pp.data(), RAD_I32, M), Mx(w, RAD_BF16, G * K, per), Mx(out.data(), RAD_BF16, M, G * K)};
                RadParam p[] = {I("heads", H), I("head_dim", D), I("rope_dim", R), I("groups", G), I("rank", K), F("theta", 10000.0)};
                RadArgs a{}; a.t = t; a.n_t = 4; a.p = p; a.n_p = 6; return fn(&a, nullptr);
            };
            std::vector<uint16_t> ref_out((size_t)M * G * K, 0xdead); std::vector<int32_t> g2 = good;
            const int rr = call(wref, g2, ref_out);
            for (int bad : {-1, 0, 15, 16, 17, 63, 64, 69}) {
                std::vector<uint16_t> out((size_t)M * G * K, 0xdead); std::vector<int32_t> pp = good;
                if (bad >= 0) pp[(size_t)bad] = -4;
                (void)take_all();
                const int rd = call(r4d_rad_ernie_hca_wo_a, pp, out), got = take_all(), blk = emu_last_block;
                bool rows_ok = rr == RAD_OK && rd == RAD_OK;
                for (int m = 0; m < M; ++m)
                    for (int k = 0; k < G * K; ++k) {
                        const size_t i = (size_t)m * G * K + k;
                        if (m == bad) rows_ok = rows_ok && out[i] == 0xdead;
                        else rows_ok = rows_ok && std::abs((int)out[i] - (int)ref_out[i]) <= 1;
                    }
                check(blk == 256 && rows_ok && (bad < 0 ? got == 0 : got != 0),
                      std::string("wo_a tiled (M=70, 256 threads) ") + (shift ? "scalar" : "vector") + (bad < 0 ? ", valid positions" : ", row " + std::to_string(bad) + " at position -4") + " (launch rc " + std::to_string(rd) +
                      ", verdict " + std::to_string(got) + ", block " + std::to_string(blk) + (rows_ok ? ", bad row not stored, other rows as libref" : ", ROWS DIFFER") + ")");
            }
        }
    }
    unsetenv("R4D_ERNIE_CARD_FORM");
}
/* ---- ernie_idx_query / ernie_idx_key (libr4d/r4d_ernie_mla.hip; libref/ref_ernie_attn.cpp:494 and :544). libref refuses a negative position (query, key) and a slot outside
 * [-1, slots) (key) before its first store. In the card form the kernel must leave the offending row as it was, raise bit 2 / 1, and compute every other row as libref does. ---- */
void idx_pos_suite(void* lib) {
    setenv("R4D_ERNIE_CARD_FORM", "1", 1);
    auto bf = [](int i) { return rad_f32_to_bf16(0.125f * (float)((i * 7) % 13 - 6)); };
    RadLaunchFn qref = find(lib, "ernie_idx_query"), kref = find(lib, "ernie_idx_key");
    if (!qref || !kref) { check(false, "libref has ernie_idx_query and ernie_idx_key"); unsetenv("R4D_ERNIE_CARD_FORM"); return; }
    struct Q { const char* name; int D, R, H, M, threads; };
    const Q qs[] = {{"idx_query D128 decode (fast128, 128 threads)", 128, 64, 5, 3, 128}, {"idx_query D128 prefill (fast128, 256 threads)", 128, 64, 3, 17, 256},
                    {"idx_query D64 (generic, 256 threads)", 64, 32, 3, 4, 256}};
    for (const Q& c : qs) {
        std::vector<uint16_t> q((size_t)c.M * c.H * c.D);
        for (size_t i = 0; i < q.size(); ++i) q[i] = bf((int)i);
        std::vector<int32_t> good((size_t)c.M);
        for (int m = 0; m < c.M; ++m) good[(size_t)m] = 4 + 5 * m;
        auto call = [&](RadLaunchFn fn, std::vector<int32_t>& pp, std::vector<uint8_t>& qf, std::vector<float>& qsc) {
            RadTensor t[4] = {Mx(q.data(), RAD_BF16, c.M, (int64_t)c.H * c.D), V(pp.data(), RAD_I32, c.M), Mx(qf.data(), RAD_F8E4M3, c.M, (int64_t)c.H * c.D), Mx(qsc.data(), RAD_F32, c.M, c.H)};
            RadParam p[] = {I("heads", c.H), I("head_dim", c.D), I("rope_dim", c.R), F("theta", 10000.0)};
            RadArgs a{}; a.t = t; a.n_t = 4; a.p = p; a.n_p = 4; return fn(&a, nullptr);
        };
        const size_t nq = (size_t)c.M * c.H * c.D, ns = (size_t)c.M * c.H;
        std::vector<uint8_t> rq(nq, 0xAD); std::vector<float> rs(ns, -77.f); std::vector<int32_t> g2 = good;
        const int rr = call(qref, g2, rq, rs);
        for (int bad = -1; bad < c.M; bad += (bad < 0 ? 1 : (c.M > 8 && bad < c.M - 2 ? (c.M - 2 - bad) : 1))) {
            std::vector<uint8_t> dq(nq, 0xAD); std::vector<float> ds(ns, -77.f); std::vector<int32_t> pp = good;
            if (bad >= 0) pp[(size_t)bad] = -4;
            (void)take_all();
            const int rd = call(r4d_rad_ernie_idx_query, pp, dq, ds), got = take_all(), blk = emu_last_block;
            bool rows_ok = rr == RAD_OK && rd == RAD_OK;
            for (int m = 0; m < c.M; ++m)
                for (int h = 0; h < c.H; ++h) {
                    const size_t is = (size_t)m * c.H + h;
                    if (m == bad) { rows_ok = rows_ok && ds[is] == -77.f; for (int k = 0; k < c.D; ++k) rows_ok = rows_ok && dq[is * c.D + k] == 0xAD; }
                    else { rows_ok = rows_ok && ds[is] == rs[is]; for (int k = 0; k < c.D; ++k) rows_ok = rows_ok && dq[is * c.D + k] == rq[is * c.D + k]; }
                }
            check(blk == c.threads && rows_ok && (bad < 0 ? got == 0 : got == 2),
                  std::string(c.name) + (bad < 0 ? ", valid positions" : ", row " + std::to_string(bad) + " at position -4") + " (launch rc " + std::to_string(rd) + ", verdict " + std::to_string(got) +
                  ", block " + std::to_string(blk) + (rows_ok ? ", bad row not stored, other rows == libref" : ", ROWS DIFFER") + ")");
        }
    }
    struct K { const char* name; int D, R, M, threads; };
    const K ks[] = {{"idx_key D128 M=5 (fast128 seq, 32 threads)", 128, 64, 5, 32}, {"idx_key D128 M=70 (fast128 par, 128 threads)", 128, 64, 70, 128}, {"idx_key D64 M=5 (generic)", 64, 32, 5, -1}};
    for (const K& c : ks) {
        const int blocks = 4, bsz = 32, slots = blocks * bsz, W = c.D + 4;
        std::vector<uint16_t> k((size_t)c.M * c.D), gm((size_t)c.D), bt((size_t)c.D);
        for (size_t i = 0; i < k.size(); ++i) k[i] = bf((int)i);
        for (int i = 0; i < c.D; ++i) { gm[(size_t)i] = rad_f32_to_bf16(1.0f + 0.01f * (float)i); bt[(size_t)i] = bf(i + 2); }
        std::vector<int32_t> gpos((size_t)c.M), gslot((size_t)c.M);
        for (int m = 0; m < c.M; ++m) { gpos[(size_t)m] = 4 + 3 * m; gslot[(size_t)m] = 2 + m; }
        auto call = [&](RadLaunchFn fn, std::vector<int32_t>& pp, std::vector<int32_t>& sl, std::vector<uint8_t>& cache) {
            RadTensor cv{}; cv.data = cache.data(); cv.dtype = RAD_U8; cv.rank = 3; cv.shape[0] = blocks; cv.shape[1] = bsz; cv.shape[2] = W; cv.stride[0] = (int64_t)bsz * W; cv.stride[1] = W; cv.stride[2] = 1;
            RadTensor t[6] = {Mx(k.data(), RAD_BF16, c.M, c.D), V(gm.data(), RAD_BF16, c.D), V(bt.data(), RAD_BF16, c.D), V(pp.data(), RAD_I32, c.M), V(sl.data(), RAD_I32, c.M), cv};
            RadParam p[] = {I("head_dim", c.D), I("rope_dim", c.R), F("theta", 10000.0), F("eps", 1e-6)};
            RadArgs a{}; a.t = t; a.n_t = 6; a.p = p; a.n_p = 4; return fn(&a, nullptr);
        };
        std::vector<uint8_t> rc((size_t)slots * W, 0xAD); std::vector<int32_t> gp = gpos, gsl = gslot;
        const int rr = call(kref, gp, gsl, rc);
        struct Bad { const char* what; int row; int pos; int slot; };
        const int mid = c.M / 2;
        const Bad bads[] = {{"valid", -1, 0, 0}, {"row 0 at position -4", 0, -4, 0}, {"a middle row at position -4", mid, -4, 0}, {"the last row at position -4", c.M - 1, -4, 0},
                            {"a middle row with slot -1 AND position -4 (libref judges the position of a dropped row too)", mid, -4, -1}, {"row 0 with slot -1 AND position -4", 0, -4, -1}, {"a middle row with slot -2", mid, 0, -2}, {"a middle row with slot == slots", mid, 0, slots}, {"the last row with slot far past the cache", c.M - 1, 0, 100000}};
        for (const Bad& b : bads) {
            std::vector<uint8_t> dc((size_t)slots * W, 0xAD); std::vector<int32_t> pp = gpos, sl = gslot;
            if (b.row >= 0) { if (b.pos) pp[(size_t)b.row] = b.pos; if (b.slot) sl[(size_t)b.row] = b.slot; }
            (void)take_all();
            std::vector<uint8_t> lc((size_t)slots * W, 0xAD); std::vector<int32_t> lp = pp, ls = sl;
            const int rl = call(kref, lp, ls, lc);   /* libref on the same operands: the spec */
            const int rd = call(r4d_rad_ernie_idx_key, pp, sl, dc), got = take_all(), blk = emu_last_block;
            bool rows_ok = rr == RAD_OK && rd == RAD_OK && rl == (b.row < 0 ? RAD_OK : RAD_E_SHAPE);
            std::vector<uint8_t> want = rc;
            if (b.row >= 0) for (int x = 0; x < W; ++x) want[(size_t)gslot[(size_t)b.row] * W + x] = 0xAD;   /* the refused row's slot stays as it was */
            rows_ok = rows_ok && dc == want;
            const int expect_bit = b.row < 0 ? 0 : (b.pos ? 2 : 1);
            check((c.threads < 0 || blk == c.threads) && rows_ok && got == expect_bit,
                  std::string(c.name) + ", " + b.what + " (launch rc " + std::to_string(rd) + ", verdict " + std::to_string(got) + ", block " + std::to_string(blk) + (rows_ok ? ", cache == libref, the refused row's slot untouched" : ", CACHE DIFFERS") + ")");
        }
        {   /* slot -1 skips the row (libref accepts it, its position is still judged) */
            std::vector<uint8_t> dc((size_t)slots * W, 0xAD), ec((size_t)slots * W, 0xAD); std::vector<int32_t> pp = gpos, sl = gslot, g3 = gpos, s3 = gslot;
            sl[0] = -1; s3[0] = -1;
            const int re = call(kref, g3, s3, ec);
            (void)take_all();
            const int rd = call(r4d_rad_ernie_idx_key, pp, sl, dc), got = take_all();
            check(re == RAD_OK && rd == RAD_OK && dc == ec && got == 0, std::string(c.name) + ", slot -1 skips the row (libref " + std::to_string(re) + ", launch " + std::to_string(rd) + ", verdict " + std::to_string(got) + ")");
        }
    }
    unsetenv("R4D_ERNIE_CARD_FORM");
}
/* ---- fix series 5 -------------------------------------------------------------------------------------------------------------------------------------------------- */

/* H1: the last-writer table is the op's scratch and its size is a function of M alone. */
void table_suite(RadLaunchFn ref, RadLaunchFn dev) {
    {
        auto bytes = [](int64_t M) { RadParam p[] = {I("M", M)}; RadArgs a{}; a.p = p; a.n_p = 1; return std::make_pair(r4d_rad_ernie_hca_state_write_scratch(&a), r4d_rad_ernie_hca_compress_scratch(&a)); };
        check(bytes(256).first == 0 && bytes(256).second == 0, "scratch hook: no scratch up to 256 rows");
        check(bytes(257).first == 8192 && bytes(2048).first == 32768 && bytes(2048).second == 32768, "scratch hook: 8 bytes x the power of two >= 2M (257 rows: 8 KiB, 2048 rows: 32 KiB)");
        check(bytes(65535).first == (8ll << 17), "scratch hook: 65535 rows: 1 MiB (the table never depends on the pool)");
    }
    setenv("R4D_ERNIE_CARD_FORM", "1", 1);
    constexpr int ratio = 4, D = 8, W = 16;
    std::vector<int32_t> shared_scr;   /* one buffer reused by every launch below: stale contents from the previous launch must not matter */
    struct Pat { const char* name; int M, pool; int (*slot)(int m); };
    const Pat pats[] = {
        {"multiples of 64 (a bad hash would collide)", 2048, 4096, [](int m) { return (m * 64) % 4096; }},
        {"a pseudo-random walk with many repeats", 2048, 4096, [](int m) { return (int)(((unsigned)m * 2654435761u >> 7) % 1500u); }},
        {"all rows the same slot", 700, 8, [](int) { return 3; }},
        {"every slot distinct", 2048, 4096, [](int m) { return 2047 - m; }},
        {"slots reversed in pairs with -1 holes", 1000, 600, [](int m) { return m % 7 == 3 ? -1 : (999 - m) % 600; }},
    };
    for (const Pat& pt : pats)
        for (const char* order : {"forward", "reverse"}) {
            setenv("R4D_EMU_BLOCK_ORDER", order, 1);
            const int M = pt.M, pool = pt.pool;
            std::vector<float> kv((size_t)M * W), sc((size_t)M * W), ape((size_t)ratio * W), a_st((size_t)pool * 2 * W, -7.f), b_st(a_st);
            std::vector<int32_t> pos((size_t)M), slot((size_t)M);
            for (size_t i = 0; i < kv.size(); ++i) { kv[i] = 0.25f * (int)(i % 7) - 0.5f + 0.001f * (int)(i / 7 % 13); sc[i] = 0.125f * (int)(i % 5); }
            for (size_t i = 0; i < ape.size(); ++i) ape[i] = 0.0625f * (int)(i % 9);
            for (int m = 0; m < M; ++m) { pos[m] = m % 29; slot[m] = pt.slot(m); }
            auto call = [&](RadLaunchFn fn, std::vector<float>& st) {
                RadTensor t[6] = {Mx(kv.data(), RAD_F32, M, W), Mx(sc.data(), RAD_F32, M, W), Mx(ape.data(), RAD_F32, ratio, W), V(pos.data(), RAD_I32, M), V(slot.data(), RAD_I32, M), Mx(st.data(), RAD_F32, pool, 2 * W)};
                RadParam p[] = {I("ratio", ratio), I("head_dim", D), I("M", M)};
                RadArgs a{}; a.t = t; a.n_t = 6; a.p = p; a.n_p = 3;
                if (fn == dev) {
                    const int64_t n = r4d_rad_ernie_hca_state_write_scratch(&a);
                    if (n > 0) { if (shared_scr.size() < (size_t)n / 4) shared_scr.resize((size_t)n / 4, (int32_t)0x13572468); a.scratch = shared_scr.data(); a.scratch_bytes = n; }
                }
                return fn(&a, nullptr);
            };
            (void)take_all();
            const long long l0 = r4d_verdict::stats().dedupe_launches.load();
            const int ra = call(ref, a_st), rb = call(dev, b_st), v = take_all();
            check(ra == RAD_OK && rb == RAD_OK && a_st == b_st && v == 0 && r4d_verdict::stats().dedupe_launches.load() - l0 == 2,
                  std::string("state_write table: ") + pt.name + ", blocks " + order + ", reused scratch: state == libref, two last-writer launches");
        }
    unsetenv("R4D_EMU_BLOCK_ORDER");
    unsetenv("R4D_ERNIE_CARD_FORM");
}

/* m-2: a row libref would refuse (negative position) does not take the place of an earlier valid row with its slot, whichever path resolves the duplicates: the scan (<= 256 rows,
 * or no scratch) or the last-writer table. Both give the state of the launch with that row skipped, and raise the event. */
void refused_dup_suite(RadLaunchFn ref, RadLaunchFn dev) {
    setenv("R4D_ERNIE_CARD_FORM", "1", 1);
    constexpr int ratio = 4, D = 8, W = 16, pool = 64;
    for (int M : {200, 700})
        for (int use_scratch : {1, 0}) {
            if (M == 200 && !use_scratch) continue;
            for (const char* order : {"forward", "reverse"}) {
                setenv("R4D_EMU_BLOCK_ORDER", order, 1);
                std::vector<float> kv((size_t)M * W), sc((size_t)M * W), ape((size_t)ratio * W), a_st((size_t)pool * 2 * W, -7.f), b_st(a_st);
                std::vector<int32_t> pos((size_t)M), slot((size_t)M), want_slot, want_pos;
                for (size_t i = 0; i < kv.size(); ++i) { kv[i] = 0.25f * (int)(i % 7) - 0.5f + 0.001f * (int)(i / 7 % 13); sc[i] = 0.125f * (int)(i % 5); }
                for (size_t i = 0; i < ape.size(); ++i) ape[i] = 0.0625f * (int)(i % 9);
                for (int m = 0; m < M; ++m) { pos[m] = m % 29; slot[m] = 6 + (m * 7) % 50; }
                slot[10] = 5; pos[10] = 3;          /* a valid row ... */
                slot[150] = 5; pos[150] = -4;       /* ... and a later row with its slot that libref refuses */
                want_slot = slot; want_pos = pos; want_slot[150] = -1; want_pos[150] = 0;   /* the same launch with that row skipped: libref accepts it */
                auto call = [&](RadLaunchFn fn, std::vector<float>& st, std::vector<int32_t>& ps, std::vector<int32_t>& sl) {
                    RadTensor t[6] = {Mx(kv.data(), RAD_F32, M, W), Mx(sc.data(), RAD_F32, M, W), Mx(ape.data(), RAD_F32, ratio, W), V(ps.data(), RAD_I32, M), V(sl.data(), RAD_I32, M), Mx(st.data(), RAD_F32, pool, 2 * W)};
                    RadParam p[] = {I("ratio", ratio), I("head_dim", D), I("M", M)};
                    RadArgs a{}; a.t = t; a.n_t = 6; a.p = p; a.n_p = 3;
                    std::vector<int32_t> scr;
                    if (fn == dev && use_scratch) { const int64_t n = r4d_rad_ernie_hca_state_write_scratch(&a); if (n > 0) { scr.assign((size_t)n / 4, 0x5A5A5A5A); a.scratch = scr.data(); a.scratch_bytes = n; } }
                    return fn(&a, nullptr);
                };
                (void)take_all();
                const int ra = call(ref, a_st, want_pos, want_slot), rb = call(dev, b_st, pos, slot), v = take_all();
                check(ra == RAD_OK && rb == RAD_OK && a_st == b_st && (v & 2),
                      "state_write refused duplicate: M=" + std::to_string(M) + (use_scratch ? (M > 256 ? ", table" : ", scan") : ", no scratch (scan)") + ", blocks " + order +
                      ": the later row at a negative position does not replace the valid row (state == libref with that row skipped), event raised");
            }
        }
    unsetenv("R4D_EMU_BLOCK_ORDER");
    unsetenv("R4D_ERNIE_CARD_FORM");
}

/* m-1: kernel-argument bytes of an issue do not depend on what the stack held. The backend copies sizeof(args) bytes of every struct, padding included, and audits a replay against
 * a fresh issue. Each launcher is issued twice with the stack poisoned two ways; the recorded argument bytes must be identical. */
__attribute__((noinline)) void poison_stack(unsigned char v) {
    volatile unsigned char buf[65536];
    for (size_t i = 0; i < sizeof buf; ++i) buf[i] = v;
    __asm__ volatile("" ::: "memory");
}
void argpad_suite(void* lib) {
    (void)lib;
    setenv("R4D_ERNIE_CARD_FORM", "1", 1);
    auto bf = [](int i) { return rad_f32_to_bf16(0.125f * (float)((i * 7) % 13 - 6)); };
    auto stable = [&](const std::string& name, const std::function<int()>& issue) {
        int rc1 = -1, rc2 = -1;
        poison_stack(0xA5);
        const auto t1 = verdict_replay::record(issue, &rc1);
        poison_stack(0x5A);
        const auto t2 = verdict_replay::record(issue, &rc2);
        std::string why;
        check(rc1 == RAD_OK && rc2 == RAD_OK && !t1.empty() && verdict_replay::agree(t1, t2, &why), "argument bytes are independent of the stack: " + name + " (" + std::to_string(t1.size()) + " launches" + (why.empty() ? "" : ", " + why) + ")");
    };
    {   /* state_write past 256 rows: the clear kernel's HcaDedupeArgs has 4 tail bytes */
        const int M = 700, pool = 64, ratio = 4, D = 8, W = 16;
        std::vector<float> kv((size_t)M * W, 0.5f), sc((size_t)M * W, 0.25f), ape((size_t)ratio * W, 0.f), st((size_t)pool * 2 * W, -7.f);
        std::vector<int32_t> pos((size_t)M, 5), slot((size_t)M), scr;
        for (int m = 0; m < M; ++m) slot[m] = (m * 7) % 40;
        stable("state_write M=700 (clear, mark, store)", [&] {
            RadTensor t[6] = {Mx(kv.data(), RAD_F32, M, W), Mx(sc.data(), RAD_F32, M, W), Mx(ape.data(), RAD_F32, ratio, W), V(pos.data(), RAD_I32, M), V(slot.data(), RAD_I32, M), Mx(st.data(), RAD_F32, pool, 2 * W)};
            RadParam p[] = {I("ratio", ratio), I("head_dim", D), I("M", M)};
            RadArgs a{}; a.t = t; a.n_t = 6; a.p = p; a.n_p = 3;
            const int64_t n = r4d_rad_ernie_hca_state_write_scratch(&a);
            scr.assign((size_t)n / 4, 0); a.scratch = scr.data(); a.scratch_bytes = n;
            return r4d_rad_ernie_hca_state_write(&a, nullptr);
        });
    }
    {
        const int M = 600, S = 12, Rr = 20, ratio = 4, D = 8, W = 16, Wn = 8;
        std::vector<float> st((size_t)S * 2 * W);
        for (size_t i = 0; i < st.size(); ++i) st[i] = 0.25f * (int)((i * 3) % 11) - 1.0f;
        std::vector<uint16_t> nw(D), out((size_t)Rr * D, 0xdead);
        for (int i = 0; i < D; ++i) nw[i] = rad_f32_to_bf16(0.75f + 0.0625f * i);
        std::vector<int32_t> gs((size_t)M * Wn), pos((size_t)M), os((size_t)M), scr;
        for (int m = 0; m < M; ++m) { for (int w = 0; w < Wn; ++w) gs[(size_t)m * Wn + w] = (m * 5 + w) % S; pos[m] = (m % 3 == 0) ? 4 * (m % 17) + 3 : 4 * (m % 17) + 1; os[m] = (m % 3 == 0) ? (m / 3) % 7 : -1; }
        stable("compress M=600 (clear, mark, store)", [&] {
            RadTensor t[6] = {Mx(st.data(), RAD_F32, S, 2 * W), Mx(gs.data(), RAD_I32, M, Wn), V(pos.data(), RAD_I32, M), V(os.data(), RAD_I32, M), V(nw.data(), RAD_BF16, D), Mx(out.data(), RAD_BF16, Rr, D)};
            RadParam p[] = {I("ratio", ratio), I("head_dim", D), I("rope_dim", 4), F("eps", 1e-6), F("theta", 10000.0), I("M", M)};
            RadArgs a{}; a.t = t; a.n_t = 6; a.p = p; a.n_p = 6;
            const int64_t n = r4d_rad_ernie_hca_compress_scratch(&a);
            scr.assign((size_t)n / 4, 0); a.scratch = scr.data(); a.scratch_bytes = n;
            return r4d_rad_ernie_hca_compress(&a, nullptr);
        });
    }
    {
        constexpr int M = 3, H = 2, D = 8, R = 4, G = 2, K = 3;
        std::vector<uint16_t> x((size_t)M * H * D), w((size_t)G * K * (H / G) * D), out((size_t)M * G * K, 0xdead);
        for (size_t i = 0; i < x.size(); ++i) x[i] = bf((int)i);
        for (size_t i = 0; i < w.size(); ++i) w[i] = bf((int)i + 3);
        std::vector<int32_t> pos = {5, 9, 2};
        stable("wo_a", [&] {
            RadTensor t[4] = {Mx(x.data(), RAD_BF16, M, H * D), V(pos.data(), RAD_I32, M), Mx(w.data(), RAD_BF16, G * K, (H / G) * D), Mx(out.data(), RAD_BF16, M, G * K)};
            RadParam p[] = {I("heads", H), I("head_dim", D), I("rope_dim", R), I("groups", G), I("rank", K), F("theta", 10000.0)};
            RadArgs a{}; a.t = t; a.n_t = 4; a.p = p; a.n_p = 6; return r4d_rad_ernie_hca_wo_a(&a, nullptr);
        });
        stable("q_prep", [&] {
            RadTensor t[2] = {Mx(x.data(), RAD_BF16, M, H * D), V(pos.data(), RAD_I32, M)};
            RadParam p[] = {I("heads", H), I("head_dim", D), I("rope_dim", R), F("eps", 1e-6), F("theta", 10000.0)};
            RadArgs a{}; a.t = t; a.n_t = 2; a.p = p; a.n_p = 5; return r4d_rad_ernie_hca_q_prep(&a, nullptr);
        });
        stable("rope_tail", [&] {
            RadTensor t[2] = {Mx(x.data(), RAD_BF16, M, H * D), V(pos.data(), RAD_I32, M)};
            RadParam p[] = {I("heads", H), I("width", D), I("rope_dim", R), F("theta", 10000.0)};
            RadArgs a{}; a.t = t; a.n_t = 2; a.p = p; a.n_p = 4; return r4d_rad_ernie_rope_tail(&a, nullptr);
        });
    }
    for (const int Dq : {128, 512}) {   /* the q_prep fast paths: HcaQPrepFast64Args (4-byte gap before the double table) */
        constexpr int M = 3, H = 2, R = 64;
        std::vector<uint16_t> x((size_t)M * H * Dq);
        for (size_t i = 0; i < x.size(); ++i) x[i] = bf((int)i);
        std::vector<int32_t> pos = {5, 9, 2};
        stable("q_prep (fast, D=" + std::to_string(Dq) + ")", [&] {
            RadTensor t[2] = {Mx(x.data(), RAD_BF16, M, (int64_t)H * Dq), V(pos.data(), RAD_I32, M)};
            RadParam p[] = {I("heads", H), I("head_dim", Dq), I("rope_dim", R), F("eps", 1e-6), F("theta", 10000.0)};
            RadArgs a{}; a.t = t; a.n_t = 2; a.p = p; a.n_p = 5; return r4d_rad_ernie_hca_q_prep(&a, nullptr);
        });
    }
    {
        const int Dk = 64, R = 32, M = 5, blocks = 4, bsz = 32, slots = blocks * bsz, W = Dk + 4, H = 3;
        std::vector<uint16_t> k((size_t)M * Dk), gm((size_t)Dk), bt((size_t)Dk), q((size_t)M * H * Dk);
        for (size_t i = 0; i < k.size(); ++i) k[i] = bf((int)i);
        for (size_t i = 0; i < q.size(); ++i) q[i] = bf((int)i + 1);
        for (int i = 0; i < Dk; ++i) { gm[(size_t)i] = rad_f32_to_bf16(1.0f + 0.01f * (float)i); bt[(size_t)i] = bf(i + 2); }
        std::vector<int32_t> gpos(M), gslot(M);
        for (int m = 0; m < M; ++m) { gpos[m] = 4 + 3 * m; gslot[m] = 2 + m; }
        std::vector<uint8_t> cache((size_t)slots * W, 0xAD), qf((size_t)M * H * Dk); std::vector<float> qs((size_t)M * H);
        stable("idx_key (generic)", [&] {
            RadTensor cv{}; cv.data = cache.data(); cv.dtype = RAD_U8; cv.rank = 3; cv.shape[0] = blocks; cv.shape[1] = bsz; cv.shape[2] = W; cv.stride[0] = (int64_t)bsz * W; cv.stride[1] = W; cv.stride[2] = 1;
            RadTensor t[6] = {Mx(k.data(), RAD_BF16, M, Dk), V(gm.data(), RAD_BF16, Dk), V(bt.data(), RAD_BF16, Dk), V(gpos.data(), RAD_I32, M), V(gslot.data(), RAD_I32, M), cv};
            RadParam p[] = {I("head_dim", Dk), I("rope_dim", R), F("theta", 10000.0), F("eps", 1e-6)};
            RadArgs a{}; a.t = t; a.n_t = 6; a.p = p; a.n_p = 4; return r4d_rad_ernie_idx_key(&a, nullptr);
        });
        stable("idx_query (generic)", [&] {
            RadTensor t[4] = {Mx(q.data(), RAD_BF16, M, (int64_t)H * Dk), V(gpos.data(), RAD_I32, M), Mx(qf.data(), RAD_F8E4M3, M, (int64_t)H * Dk), Mx(qs.data(), RAD_F32, M, H)};
            RadParam p[] = {I("heads", H), I("head_dim", Dk), I("rope_dim", R), F("theta", 10000.0)};
            RadArgs a{}; a.t = t; a.n_t = 4; a.p = p; a.n_p = 4; return r4d_rad_ernie_idx_query(&a, nullptr);
        });
    }
    unsetenv("R4D_ERNIE_CARD_FORM");
}

/* The card default and the explicit off: validation OFF costs nothing and an invalid row is computed as it was before the verdict series. R4D_ERNIE_CARD_DEFAULTS=1 gives the
 * emulator the card's default (no R4D_ERNIE_VALIDATE); R4D_ERNIE_VALIDATE=1 is the opt-in and must behave as the rest of this file (a refused row stores nothing). Run FIRST, so no
 * State exists yet: with validation off none may be allocated by any launch, no validator or poll copy is issued, and nothing is polled. */
void off_suite(void* lib) {
    (void)lib;
    setenv("R4D_ERNIE_CARD_FORM", "1", 1);
    setenv("R4D_ERNIE_CARD_DEFAULTS", "1", 1);
    unsetenv("R4D_ERNIE_VALIDATE");
    r4d_verdict::Stats& stt = r4d_verdict::stats();
    const long long vl0 = stt.validator_launches.load(), cp0 = stt.poll_copies.load();
    check(r4d_verdict::registry().states() == 0, "validation off: no State exists before the first launch");
    const uint16_t kPoison = 0xdead;
    auto stored = [](const std::vector<uint16_t>& v, size_t b, size_t n) { for (size_t i = b; i < b + n; ++i) if (v[i] != 0xdead) return true; return false; };
    for (int mode = 0; mode < 2; ++mode) {   /* 0: the card default (off), 1: R4D_ERNIE_VALIDATE=1 */
        if (mode == 1) setenv("R4D_ERNIE_VALIDATE", "1", 1);
        const std::string tag = mode ? "validation on (R4D_ERNIE_VALIDATE=1): " : "validation off (card default): ";
        (void)take_all();
        {   /* gather: a position below 0 */
            const int M = 2, ratio = 4, bs = 8, Wn = 8;
            std::vector<int32_t> pos = {-3, 7}, req = {0, 0}, bt = {0, 1, 2, 3}, out((size_t)M * Wn, 0x7777);
            RadTensor t[4] = {V(pos.data(), RAD_I32, M), V(req.data(), RAD_I32, M), Mx(bt.data(), RAD_I32, 1, 4), Mx(out.data(), RAD_I32, M, Wn)};
            RadParam p[] = {I("ratio", ratio), I("block_size", bs)};
            RadArgs a{}; a.t = t; a.n_t = 4; a.p = p; a.n_p = 2;
            const int rc = r4d_rad_ernie_hca_gather_slots(&a, nullptr);
            bool all_m1 = true, untouched = true;
            for (int w = 0; w < Wn; ++w) { all_m1 = all_m1 && out[w] == -1; untouched = untouched && out[w] == 0x7777; }
            const int ev = take_all();
            check(rc == RAD_OK && (mode ? (untouched && ev != 0) : (all_m1 && ev == 0)), tag + "gather_slots, a row at position -3 " + (mode ? "stores nothing and raises" : "gets the masked row it always got, no event") + " (rc " + std::to_string(rc) + ", event " + std::to_string(ev) + ")");
        }
        {   /* compress: a completing row whose gather list is all masked */
            CP f; for (int w = 0; w < CP::Wn; ++w) f.gs[w] = -1;
            const int rc = f.run(r4d_rad_ernie_hca_compress), ev = take_all();
            bool zero = true; for (int c = 0; c < CP::D; ++c) zero = zero && f.out[c] == 0;
            check(rc == RAD_OK && (mode ? (f.out[0] == 0xdead && ev != 0) : (zero && ev == 0)), tag + "compress, an all-masked completing row " + (mode ? "stores nothing and raises" : "stores the zero row it always stored, no event") + " (event " + std::to_string(ev) + ")");
        }
        {   /* compress: a non-finite pooled row */
            CP f; f.st[1 * 2 * CP::W + CP::W] = NAN; f.st[2 * 2 * CP::W + CP::W] = NAN;
            const int rc = f.run(r4d_rad_ernie_hca_compress), ev = take_all();
            check(rc == RAD_OK && (mode ? (!stored(f.out, 1 * CP::D, CP::D) && ev != 0) : (stored(f.out, 1 * CP::D, CP::D) && ev == 0)), tag + "compress, a non-finite pooled row " + (mode ? "stores nothing and raises" : "is stored as before, no event"));
        }
        {   /* compress: a gather entry outside the pool is skipped (no out-of-range read) and not judged when off */
            CP f; f.gs[1] = CP::S + 40;
            const int rc = f.run(r4d_rad_ernie_hca_compress), ev = take_all();
            check(rc == RAD_OK && (mode ? ev != 0 : (ev == 0 && stored(f.out, 0, CP::D))), tag + "compress, a gather entry past the pool " + (mode ? "raises" : "is skipped, the row is stored, no event"));
        }
        {   /* state_write: invalid slot / position are skipped as before (same in both modes), raised only when on */
            SW f; f.slot[1] = -2; f.pos[2] = -1;
            const int rc = f.run(r4d_rad_ernie_hca_state_write), ev = take_all();
            bool row0 = false; for (int c = 0; c < 2 * SW::W; ++c) row0 = row0 || f.st[(size_t)0 * 2 * SW::W + c] != -7.f;
            check(rc == RAD_OK && row0 && (mode ? ev != 0 : ev == 0), tag + "state_write, a bad slot and a negative position are skipped" + (mode ? " and raised" : " silently") + ", the valid row is stored");
        }
        {   /* rope-position ops: a negative position */
            constexpr int M = 3, H = 2, D = 8, R = 4, G = 2, K = 3;
            auto bf = [](int i) { return rad_f32_to_bf16(0.125f * (float)((i * 7) % 13 - 6)); };
            std::vector<uint16_t> x((size_t)M * H * D), w((size_t)G * K * (H / G) * D), out((size_t)M * G * K, kPoison);
            for (size_t i = 0; i < x.size(); ++i) x[i] = bf((int)i);
            for (size_t i = 0; i < w.size(); ++i) w[i] = bf((int)i + 3);
            std::vector<int32_t> pos = {5, -4, 2};
            RadTensor t[4] = {Mx(x.data(), RAD_BF16, M, H * D), V(pos.data(), RAD_I32, M), Mx(w.data(), RAD_BF16, G * K, (H / G) * D), Mx(out.data(), RAD_BF16, M, G * K)};
            RadParam p[] = {I("heads", H), I("head_dim", D), I("rope_dim", R), I("groups", G), I("rank", K), F("theta", 10000.0)};
            RadArgs a{}; a.t = t; a.n_t = 4; a.p = p; a.n_p = 6;
            const int rc = r4d_rad_ernie_hca_wo_a(&a, nullptr), ev = take_all();
            check(rc == RAD_OK && (mode ? (!stored(out, 1 * G * K, G * K) && ev != 0) : (stored(out, 1 * G * K, G * K) && ev == 0)), tag + "wo_a, a row at position -4 " + (mode ? "stores nothing and raises" : "is computed as before, no event"));
            std::vector<uint16_t> x0 = x;
            RadTensor tq[2] = {Mx(x.data(), RAD_BF16, M, H * D), V(pos.data(), RAD_I32, M)};
            RadParam pq[] = {I("heads", H), I("head_dim", D), I("rope_dim", R), F("eps", 1e-6), F("theta", 10000.0)};
            RadArgs aq{}; aq.t = tq; aq.n_t = 2; aq.p = pq; aq.n_p = 5;
            const int rq = r4d_rad_ernie_hca_q_prep(&aq, nullptr), evq = take_all();
            bool row1_same = true; for (int c = 0; c < H * D; ++c) row1_same = row1_same && x[(size_t)1 * H * D + c] == x0[(size_t)1 * H * D + c];
            check(rq == RAD_OK && (mode ? (row1_same && evq != 0) : (!row1_same && evq == 0)), tag + "q_prep, a row at position -4 " + (mode ? "is left as it was and raises" : "is computed as before, no event"));
        }
        {   /* idx_key: a negative position on a valid slot; a slot outside the cache */
            const int Dk = 64, R = 32, M = 4, blocks = 4, bsz = 32, slots = blocks * bsz, W = Dk + 4;
            auto bf = [](int i) { return rad_f32_to_bf16(0.125f * (float)((i * 7) % 13 - 6)); };
            std::vector<uint16_t> k((size_t)M * Dk), gm((size_t)Dk), bt((size_t)Dk);
            for (size_t i = 0; i < k.size(); ++i) k[i] = bf((int)i);
            for (int i = 0; i < Dk; ++i) { gm[(size_t)i] = rad_f32_to_bf16(1.0f + 0.01f * (float)i); bt[(size_t)i] = bf(i + 2); }
            std::vector<int32_t> pos = {4, -4, 10, 13}, slot = {2, 3, 4, -2};
            std::vector<uint8_t> cache((size_t)slots * W, 0xAD);
            RadTensor cv{}; cv.data = cache.data(); cv.dtype = RAD_U8; cv.rank = 3; cv.shape[0] = blocks; cv.shape[1] = bsz; cv.shape[2] = W; cv.stride[0] = (int64_t)bsz * W; cv.stride[1] = W; cv.stride[2] = 1;
            RadTensor t[6] = {Mx(k.data(), RAD_BF16, M, Dk), V(gm.data(), RAD_BF16, Dk), V(bt.data(), RAD_BF16, Dk), V(pos.data(), RAD_I32, M), V(slot.data(), RAD_I32, M), cv};
            RadParam p[] = {I("head_dim", Dk), I("rope_dim", R), F("theta", 10000.0), F("eps", 1e-6)};
            RadArgs a{}; a.t = t; a.n_t = 6; a.p = p; a.n_p = 4;
            const int rc = r4d_rad_ernie_idx_key(&a, nullptr), ev = take_all();
            auto row_written = [&](int s) { for (int x = 0; x < W; ++x) if (cache[(size_t)s * W + x] != 0xAD) return true; return false; };
            check(rc == RAD_OK && (mode ? (!row_written(3) && row_written(2) && row_written(4) && ev == 3) : (row_written(3) && row_written(2) && row_written(4) && ev == 0)),
                  tag + "idx_key, position -4 on slot 3 and slot -2 " + (mode ? "refuse their rows and raise (bits 1|2)" : "behave as before: the negative position is computed, the bad slot skipped silently, no event"));
        }
        if (mode == 0) {
            check(r4d_verdict::registry().states() == 0, "validation off: no State was allocated by any launch above");
            check(stt.validator_launches.load() == vl0 && stt.poll_copies.load() == cp0, "validation off: no validator launch and no poll copy");
            check(r4d_verdict::registry().take_all() == 0, "validation off: the poll reads nothing (no copy, no event)");
            /* big launches still run the last-writer pass (valid duplicates are resolved with validation off too) but nothing else */
            const long long d0 = stt.dedupe_launches.load();
            const int M = 700, pool = 64, ratio = 4, D = 8, W = 16;
            std::vector<float> kv((size_t)M * W, 0.5f), sc((size_t)M * W, 0.25f), ape((size_t)ratio * W, 0.f), st((size_t)pool * 2 * W, -7.f);
            std::vector<int32_t> pos((size_t)M, 5), slot((size_t)M), scr;
            for (int m = 0; m < M; ++m) slot[m] = (m * 7) % 40;
            RadTensor t[6] = {Mx(kv.data(), RAD_F32, M, W), Mx(sc.data(), RAD_F32, M, W), Mx(ape.data(), RAD_F32, ratio, W), V(pos.data(), RAD_I32, M), V(slot.data(), RAD_I32, M), Mx(st.data(), RAD_F32, pool, 2 * W)};
            RadParam p[] = {I("ratio", ratio), I("head_dim", D), I("M", M)};
            RadArgs a{}; a.t = t; a.n_t = 6; a.p = p; a.n_p = 3;
            const int64_t n = r4d_rad_ernie_hca_state_write_scratch(&a); scr.assign((size_t)n / 4, 0); a.scratch = scr.data(); a.scratch_bytes = n;
            int rc = -1;
            const auto tape = verdict_replay::record([&] { return rc = r4d_rad_ernie_hca_state_write(&a, nullptr); }, &rc);
            check(rc == RAD_OK && tape.size() == 3 && stt.dedupe_launches.load() - d0 == 2 && r4d_verdict::registry().states() == 0 && stt.validator_launches.load() == vl0,
                  "validation off: state_write M=700 launches clear, mark, store (the duplicate pass) and nothing else, still no State");
        }
    }
    unsetenv("R4D_ERNIE_VALIDATE");
    unsetenv("R4D_ERNIE_CARD_DEFAULTS");
    unsetenv("R4D_ERNIE_CARD_FORM");
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    void* lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) { std::fprintf(stderr, "%s\n", dlerror()); return 2; }
    RadLaunchFn sw = find(lib, "ernie_hca_state_write"), cp = find(lib, "ernie_hca_compress");
    if (!sw || !cp) { std::fprintf(stderr, "libref lacks the ops\n"); return 2; }
    unsetenv("R4D_ERNIE_CARD_FORM");
    off_suite(lib);   /* first: it asserts that no verdict State exists yet */
    state_write_suite(sw, r4d_rad_ernie_hca_state_write);
    compress_suite(cp, r4d_rad_ernie_hca_compress);
    state_write_dup_suite(sw, r4d_rad_ernie_hca_state_write);
    compress_dup_suite(cp, r4d_rad_ernie_hca_compress);
    dedupe_launch_shape(r4d_rad_ernie_hca_state_write, r4d_rad_ernie_hca_compress);
    table_suite(sw, r4d_rad_ernie_hca_state_write);
    refused_dup_suite(sw, r4d_rad_ernie_hca_state_write);
    argpad_suite(lib);
    pos_suite(lib);
    validation_off_suite(lib, cp);
    pos_fast_suite(lib);
    idx_pos_suite(lib);
    std::printf("%d failure(s)\n", g_fail);
    dlclose(lib);
    return g_fail ? 1 : 0;
}
