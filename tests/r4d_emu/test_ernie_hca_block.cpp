/* tests/r4d_emu: ernie_hca_compress with out_layout "block" (libr4d/r4d_ernie_hca.hip) on the host SIMT emulator, held to libref's refusals.
 *
 * In the block layout the group keeps ONE row per block of `ratio` tokens: a completing token's out slot is page * B + B - 1, the block's last token, and libref refuses
 * a live (completing) row whose slot sits at any other offset, or past pages * B, before any store. The host launcher scans for that where its operands are host
 * memory; on a card the scan is compiled out and the kernel cannot refuse before it stores, so it must skip the row (no store, no out-of-range write) and latch the
 * violation for the host (r4d_ernie_hca_take_error), the error-latch mechanism of the other ERNIE HCA ops.
 * Each negative case runs on libref and on the launcher with its scan on (both must return RAD_E_SHAPE and leave the pool untouched); then with the scan off
 * (R4D_HCA_VALIDATE=0, the situation of a card build) the kernel must return RAD_OK, leave the pool untouched, and latch the violation. Controls: a valid launch
 * latches nothing and stores libref's rows; a row that does not complete a group is not checked for its offset, as in libref.
 * The first negative is the reproducer of the review of 1669b14d: ratio 4, head_dim 8, rope_dim 4, position 3, out pool [3, 1, 4, 2], out slot 4, gather
 * slots 0..7. */
#include "rad_plugin.h"
#include "r4d_plugin.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>

extern "C" int r4d_ernie_hca_take_error(void);
extern "C" int rad_kernel_poll_error(void);

namespace {
RadParam I(const char* k, int64_t v) { RadParam p{}; p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }
RadParam F(const char* k, double v) { RadParam p{}; p.key = k; p.kind = RAD_P_F64; p.dval = v; return p; }
RadParam T(const char* k, const char* v) { RadParam p{}; p.key = k; p.kind = RAD_P_STR; p.sval = v; return p; }
RadTensor V(void* d, uint32_t dt, int64_t n) { RadTensor t{}; t.data = d; t.dtype = dt; t.rank = 1; t.shape[0] = n; t.stride[0] = 1; return t; }
RadTensor Mx(void* d, uint32_t dt, int64_t r, int64_t c) { RadTensor t{}; t.data = d; t.dtype = dt; t.rank = 2; t.shape[0] = r; t.shape[1] = c; t.stride[0] = c; t.stride[1] = 1; return t; }
RadTensor Pool(void* d, int64_t pages, int64_t b, int64_t e) {
    RadTensor t{}; t.data = d; t.dtype = RAD_BF16; t.rank = 4;
    t.shape[0] = pages; t.shape[1] = 1; t.shape[2] = b; t.shape[3] = e;
    t.stride[3] = 1; t.stride[2] = e; t.stride[1] = b * e; t.stride[0] = b * e;
    return t;
}
RadLaunchFn find(void* lib, const char* op) {
    auto cnt = reinterpret_cast<int (*)()>(dlsym(lib, "rad_kernel_count"));
    auto at = reinterpret_cast<const RadKernelInfo* (*)(int)>(dlsym(lib, "rad_kernel_at"));
    for (int i = 0; cnt && i < cnt(); ++i) { const RadKernelInfo* k = at(i); if (k && k->op && !std::strcmp(k->op, op)) return k->launch; }
    return nullptr;
}
int g_fail = 0;
void check(bool ok, const std::string& name) { std::printf("%s %s\n", ok ? "PASS" : "FAIL", name.c_str()); if (!ok) ++g_fail; }

/* ordered representation: the distance in representable bf16 steps (+0 and -0 are the same point) */
int ord(uint16_t h) { return (h & 0x8000u) ? -(int)(h & 0x7fffu) : (int)h; }

/* ratio 4 (coff 2, Wn 8), head_dim 8 (W 16), rope 4; 8 state rows; out pool [3, 1, 4, 2] = 3 block rows of 8 bf16; up to 3 input rows */
struct CB {
    static constexpr int S = 8, ratio = 4, D = 8, W = 16, Wn = 8, Pages = 3, B = 4, E = 2, Rr = Pages * B;
    int M;
    std::vector<float> st;
    std::vector<uint16_t> nw, out;
    std::vector<int32_t> gs, pos, os;
    explicit CB(std::vector<int32_t> p, std::vector<int32_t> o) : M((int)p.size()), st(S * 2 * W), nw(D), out(Pages * B * E, 0xdead), gs((size_t)M * Wn), pos(p), os(o) {
        for (size_t i = 0; i < st.size(); ++i) st[i] = 0.25f * (int)((i * 3) % 11) - 1.0f;
        for (int i = 0; i < D; ++i) nw[i] = rad_f32_to_bf16(0.75f + 0.0625f * i);
        for (int m = 0; m < M; ++m) for (int w = 0; w < Wn; ++w) gs[(size_t)m * Wn + w] = w;   /* the valid gather slots 0..7 */
    }
    int run(RadLaunchFn fn) {
        RadTensor t[6] = {Mx(st.data(), RAD_F32, S, 2 * W), Mx(gs.data(), RAD_I32, M, Wn), V(pos.data(), RAD_I32, M), V(os.data(), RAD_I32, M), V(nw.data(), RAD_BF16, D),
                          Pool(out.data(), Pages, B, E)};
        RadParam p[] = {I("ratio", ratio), I("head_dim", D), I("rope_dim", 4), F("eps", 1e-6), F("theta", 10000.0), T("out_layout", "block")};
        RadArgs a{}; a.t = t; a.n_t = 6; a.p = p; a.n_p = 6;
        return fn(&a, nullptr);
    }
    bool untouched() const { for (uint16_t v : out) if (v != 0xdead) return false; return true; }
    /* block row `page` (8 contiguous bf16) was written */
    bool row_written(int page) const { for (int i = 0; i < B * E; ++i) if (out[(size_t)page * B * E + i] == 0xdead) return false; return true; }
    /* every element of the pool: the sentinel where `want` still holds it (exactly, not "some sentinel remains"), within one bf16 step of `want` where it holds a row */
    bool pool_equals(const CB& want) const {
        for (size_t i = 0; i < out.size(); ++i) {
            if (want.out[i] == 0xdead ? out[i] != 0xdead : (out[i] == 0xdead || std::abs(ord(out[i]) - ord(want.out[i])) > 1)) return false;
        }
        return true;
    }
    int max_ulp(const CB& o) const { int m = 0; for (size_t i = 0; i < out.size(); ++i) m = std::max(m, std::abs(ord(out[i]) - ord(o.out[i]))); return m; }
};

/* good_pos / good_os: the rows before the first bad one, which a card has stored by the time it latches (a refusal before the first store is what libref does) */
struct Neg {
    const char* name; std::vector<int32_t> pos, os, good_pos, good_os;
    int gs_row = -1, gs_col = 0, gs_val = 0;   /* a gather entry set to gs_val (-2, S: libref refuses) */
    int want_gs_val = 0;                      /* unused when the card refuses the row (every gather case: the row is not stored) */
    unsigned bits = 0;                        /* the event bits the card form raises (r4d_ernie_hca.hip, g_hca_log): 1 out slot outside [-1, pool), 2 negative position, 4 gather entry outside [-1, pool), 8 all gather entries masked, 256 completing row at a wrong block offset */
};
void poke(CB& f, const Neg& n, int val) { if (n.gs_row >= 0) f.gs[(size_t)n.gs_row * CB::Wn + n.gs_col] = val; }
}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    void* lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) { std::fprintf(stderr, "%s\n", dlerror()); return 2; }
    RadLaunchFn ref = find(lib, "ernie_hca_compress");
    if (!ref) { std::fprintf(stderr, "libref lacks ernie_hca_compress\n"); return 2; }
    RadLaunchFn dev = r4d_rad_ernie_hca_compress;
    unsetenv("R4D_HCA_VALIDATE");

    const std::vector<Neg> negs = {
        {"review reproducer: position 3, slot 4 (page 1, offset 0)", {3}, {4}, {}, {}, -1, 0, 0, 0, 256u},
        {"completing token at offset 1 (slot 5)", {3}, {5}, {}, {}, -1, 0, 0, 0, 256u},
        {"completing token at offset 2 (slot 10, page 2)", {7}, {10}, {}, {}, -1, 0, 0, 0, 256u},
        {"token completing block 1 given slot 8 (page 2, offset 0)", {7}, {8}, {}, {}, -1, 0, 0, 0, 256u},
        {"slot == pages * B", {3}, {CB::Rr}, {}, {}, -1, 0, 0, 0, 1u},
        {"slot -2", {3}, {-2}, {}, {}, -1, 0, 0, 0, 1u},
        {"valid row first, then a row at the wrong offset", {3, 7}, {3, 6}, {3}, {3}, -1, 0, 0, 0, 256u},
        {"wrong offset first, then a valid row (the valid row is still stored)", {3, 7}, {5, 7}, {7}, {7}, -1, 0, 0, 0, 256u},
        {"negative position of a live row (position -1, slot 3)", {-1}, {3}, {}, {}, -1, 0, 0, 0, 2u},
        {"gather entry -2 in the completing row (the integration's mechanism refuses the row: nothing stored)", {3}, {3}, {}, {}, 0, 1, -2, 0, 4u},
        {"gather entry == state pool in the completing row (same)", {3}, {3}, {}, {}, 0, 5, CB::S, 0, 4u},
    };
    /* refusal parity with the host scan on: libref and the launcher return the same code and store nothing */
    for (const Neg& n : negs) {
        CB a(n.pos, n.os), b(n.pos, n.os);
        poke(a, n, n.gs_val); poke(b, n, n.gs_val);
        const int ra = a.run(ref), rb = b.run(dev);
        check(ra == RAD_E_SHAPE && rb == RAD_E_SHAPE && a.untouched() && b.untouched(),
              std::string("block refusal (scan on): ") + n.name + " (libref " + std::to_string(ra) + ", device " + std::to_string(rb) + (b.untouched() ? "" : ", WRITTEN") + ")");
    }
    /* the card form (R4D_ERNIE_CARD_FORM=1): no host scan, the launch returns OK, the kernel skips the row and raises an event the engine polls. */
    setenv("R4D_ERNIE_CARD_FORM", "1", 1);
    (void)r4d_ernie_hca_take_error();
    for (const Neg& n : negs) {
        CB f(n.pos, n.os);
        poke(f, n, n.gs_val);
        const int rc = f.run(dev), got = r4d_ernie_hca_take_error();
        /* the pool must equal, element for element, the pool libref leaves after running only the rows that are not bad */
        CB want(n.good_pos, n.good_os);
        if (want.M > 0) poke(want, n, n.want_gs_val);
        if (!n.good_pos.empty()) (void)want.run(ref);
        const bool clean = f.pool_equals(want);
        check(rc == RAD_OK && got == (int)n.bits && clean && r4d_ernie_hca_take_error() == 0,
              std::string("block refusal polled (card form): ") + n.name + " (rc " + std::to_string(rc) + ", event bits " + std::to_string(got) + " want " + std::to_string(n.bits) + (clean ? "" : ", WRONG STORE") + ")");
        /* and through the one poll the engine makes (rad_kernel_poll_error): the same launch is reported there too */
        { CB g(n.pos, n.os); poke(g, n, n.gs_val); (void)g.run(dev); const int polled = rad_kernel_poll_error(); check(polled != 0 && rad_kernel_poll_error() == 0, std::string("block refusal reaches rad_kernel_poll_error: ") + n.name + " (poll " + std::to_string(polled) + ")"); }
    }
    /* controls (scan off, as on a card): valid rows store libref's values within one bf16 step and latch nothing; a row that completes no group is not checked */
    struct Ok { const char* name; std::vector<int32_t> pos, os; std::vector<int> pages; };
    const std::vector<Ok> oks = {
        {"valid completing tokens at the block's last row", {3, 7, 11}, {3, 7, 11}, {0, 1, 2}},
        {"a row that completes no group at a wrong offset is ignored (position 2, slot 4)", {2}, {4}, {}},
        {"out slot -1 ignores the row altogether", {3, 7}, {-1, 7}, {1}},
    };
    for (const Ok& c : oks) {
        CB a(c.pos, c.os), b(c.pos, c.os);
        const int ra = a.run(ref), rb = b.run(dev), got = r4d_ernie_hca_take_error();
        const bool pages_ok = b.pool_equals(a);   /* every element: libref's rows within one bf16 step, the sentinel exactly everywhere else */
        check(ra == RAD_OK && rb == RAD_OK && got == 0 && pages_ok && b.max_ulp(a) <= 1,
              std::string("block control: ") + c.name + " (libref " + std::to_string(ra) + ", device " + std::to_string(rb) + ", latched " + std::to_string(got) + ", max ulp to libref " + std::to_string(b.max_ulp(a)) + ")");
    }
    /* A completing row whose whole gather list is masked: libref refuses it. The base branch's kernel stored a zero row and latched nothing (a pinned difference); the event log's
     * judgement (bit 8, before the row's first store) closes it: the card form stores nothing and raises. */
    {
        CB a({3}, {3}), b({3}, {3});
        for (int w = 0; w < CB::Wn; ++w) { a.gs[(size_t)w] = -1; b.gs[(size_t)w] = -1; }
        unsetenv("R4D_ERNIE_CARD_FORM");
        const int ra = a.run(ref), rb = b.run(dev);
        check(ra == RAD_E_SHAPE && rb == RAD_E_SHAPE && a.untouched() && b.untouched(), "all-masked completing row (scan on): both refuse");
        setenv("R4D_ERNIE_CARD_FORM", "1", 1);
        (void)r4d_ernie_hca_take_error();
        CB f({3}, {3});
        for (int w = 0; w < CB::Wn; ++w) f.gs[(size_t)w] = -1;
        const int rc = f.run(dev), got = r4d_ernie_hca_take_error();
        check(rc == RAD_OK && got == 8 && f.untouched(), "all-masked completing row (card form): nothing stored, event bit 8 (rc " + std::to_string(rc) + ", bits " + std::to_string(got) + ")");
    }
    unsetenv("R4D_ERNIE_CARD_FORM");
    unsetenv("R4D_HCA_VALIDATE");
    std::printf("%d failure(s)\n", g_fail);
    dlclose(lib);
    return g_fail ? 1 : 0;
}
