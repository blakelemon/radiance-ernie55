/* tests/r4d_emu: row identity of the ERNIE row-banded device ops on the host SIMT emulator (the row-identity checks).
 * A row's output must be the same bits alone, in a decode step, a drafter verify step or a prefill chunk, whatever its neighbours:
 * row_identity.h places probe rows in launches of 1..2048 rows and compares them byte for byte with the same rows computed alone. Shapes are
 * small (a test has about 10 s of CPU); the band edges are row counts. argv[1] names the family:
 *   pair     gemm_nt_f32_pair, bf16 weights: the chunked WMMA sum for every row count (K = 4096 at up to 65 rows, K = 544 with a short
 *            last chunk up to 2048 rows), and |y - f64| within the bf16-product sum bound against a host f64 reference.
 *   woa      ernie_hca_wo_a: the chunked WMMA sum for every row count (ERNIE's H=8 D=512 R=64 G=2 rank=256 up to 65 rows; H=6 D=128
 *            R=32 G=2 rank=32, per = 384 with a short last chunk, up to 2048 rows); the small-row kernel (<= 64 rows) against the
 *            tiled kernel (> 64) is part of the check.
 *   mlaproj  ernie_mla_project absorb and expand (1/2/4/8/16 rows a block by step size; the per-row fmaf chain does not depend on it, so
 *            R4D_MLA_PROJ_ROWS=1 is not needed), H=2 kd=64 vd=64 L=256 up to 2048 rows.
 *   hca      ernie_hca_union_attn_origin, D=512, H=2, a launch is a contiguous run of one request (positions 64..963, window/compressed tables of 16 blocks), 1..512 rows (the wide kernel for every row count).
 *   mla      ernie_mla_sparse_attn, H=2 L=512 rope=64 (cooperative kernel up to 8 rows, wave kernel above; R4D_MLA_COOP_MAX unset: the cooperative kernel up to 8 rows, the wave kernel above; mlawide pins the wave kernel).
 *   mhc      ernie_mhc_pre_rms, hc=4: n=512 (one 2048-column chunk) and n=1024 (two chunks), the many-row form for every row count.
 *   f8a8     block-fp8 dense GEMM, fp8a8f32, the op declaring mvm_max=0 as the plugin does (the WMMA forms for every row count), at N=1024 K=512 and at the
 *            production wave counts (K=2048, 4096: s=8 waves, staged LDS). f8a8env also sets the stale R4D_F8A8_MVM_MAX=8: the declared parameter must win.
 *   moe      moe_mxfp4 at the ERNIE TP8 slices (512 experts, top 10): gate_up (N=512 K=2048, parts=2, token order) and down (N=2048 K=256, sorted order); one token
 *            alone (10 routed rows) against launches of 1, 77 (770 routed rows: gate_up P64 form from 769) and 576 (5,760 sorted rows: down P64 form from 5,760)
 *            tokens, so the small, expert-major and P64 forms are all compared. About 45 s.
 *   f8a8ctl  the same with the op declaring mvm_max=8 (the generic default): the test PASSES only if a row differs between the matvec and the WMMA
 *            forms, i.e. the harness has teeth. */
#include "rad_plugin.h"
#ifndef ROWID_DEVICE
#include "r4d_plugin.h"
#endif
#include "row_identity.h"
#include <algorithm>
#include <array>
#include <numeric>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

/* The launchers under test. In the emulator build (the default) they are linked in; with -DROWID_DEVICE the program is a plain host binary for a gfx1201
 * card: each ROWID_ENTRY looks the entry point up in libr4d.so (dlsym; $ROWID_LIBR4D or the image's /opt/radiance/share/radiance/kernels/libr4d.so) and
 * calls it on DEVICE copies of every operand (all tensors are copied to the device and back around the call), so the families, the probe rows and the
 * byte comparison are the emulator test's, and what is compared is what the card computes. */
#ifdef ROWID_DEVICE
#include <dlfcn.h>
#include <hip/hip_runtime.h>
namespace dev {
void* lib() {
    static void* l = [] {
        const char* path = std::getenv("ROWID_LIBR4D");
        void* h = dlopen(path && *path ? path : "/opt/radiance/share/radiance/kernels/libr4d.so", RTLD_NOW | RTLD_GLOBAL);
        if (!h) { std::fprintf(stderr, "dlopen libr4d.so: %s\n", dlerror()); std::exit(2); }
        return h;
    }();
    return l;
}
size_t extent_bytes(const RadTensor& t) {
    if (!t.data || t.rank == 0) return 0;
    long long el = 1;
    for (uint32_t d = 0; d < t.rank; ++d) { if (t.shape[d] <= 0) return 0; el += (t.shape[d] - 1) * t.stride[d]; }
    int bits = rad_dtype_bits(t.dtype);
    if (bits == 0 && t.dtype == RAD_FP4E2M1) bits = 4;
    return (size_t)((el * bits + 7) / 8);
}
void ck(hipError_t e, const char* what) { if (e != hipSuccess) { std::fprintf(stderr, "HIP %s: %s\n", what, hipGetErrorString(e)); std::exit(3); } }
int call(const char* name, const RadArgs* a) {
    using Fn = int (*)(const RadArgs*, RadStream);
    Fn fn = (Fn)dlsym(lib(), name);
    if (!fn) { std::fprintf(stderr, "libr4d.so has no %s\n", name); std::exit(2); }
    std::vector<RadTensor> t(a->t, a->t + a->n_t);
    std::vector<void*> dv(t.size(), nullptr);
    std::vector<size_t> by(t.size(), 0);
    for (size_t i = 0; i < t.size(); ++i) {
        by[i] = extent_bytes(t[i]);
        if (!by[i]) continue;
        ck(hipMalloc(&dv[i], by[i]), "malloc");
        ck(hipMemcpy(dv[i], t[i].data, by[i], hipMemcpyHostToDevice), "h2d");
        t[i].data = dv[i];
    }
    RadArgs b = *a;
    b.t = t.data();
    void* sc = nullptr;
    if (a->scratch && a->scratch_bytes > 0) { ck(hipMalloc(&sc, (size_t)a->scratch_bytes), "malloc scratch"); b.scratch = sc; }
    int cus = 0;
    ck(hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, 0), "attr");
    b.wgp_count = cus; b.kv_used_max_h = -1;
    const int rc = fn(&b, nullptr);
    ck(hipDeviceSynchronize(), "sync");
    for (size_t i = 0; i < t.size(); ++i) {
        if (dv[i]) { ck(hipMemcpy(const_cast<void*>(a->t[i].data), dv[i], by[i], hipMemcpyDeviceToHost), "d2h"); (void)hipFree(dv[i]); }
    }
    if (sc) (void)hipFree(sc);
    return rc;
}
}  // namespace dev
#define ROWID_ENTRY(n) static int r4d_rad_##n(const RadArgs* a, RadStream) { return dev::call("r4d_rad_" #n, a); }
#else
#define ROWID_ENTRY(n) extern "C" int r4d_rad_##n(const RadArgs*, RadStream);
#endif
ROWID_ENTRY(gemm_nt_f32_pair)
ROWID_ENTRY(ernie_hca_wo_a)
ROWID_ENTRY(ernie_mla_project)
ROWID_ENTRY(ernie_hca_union_attn_origin)
ROWID_ENTRY(ernie_mla_sparse_attn)
ROWID_ENTRY(ernie_mhc_pre_rms)
ROWID_ENTRY(fp8_f32scale)
ROWID_ENTRY(moe_mxfp4)
ROWID_ENTRY(gemm_bf16_m16)
ROWID_ENTRY(gemm_bf16_m64)
ROWID_ENTRY(gemm_bf16_tiled)
ROWID_ENTRY(ernie_idx_logits)
ROWID_ENTRY(ernie_router_topk)
ROWID_ENTRY(ernie_router_topk_scatter)
ROWID_ENTRY(moe_scatter)
ROWID_ENTRY(moe_gather)


namespace {
RadParam I(const char* k, int64_t v) { RadParam p{}; p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }
RadParam S(const char* k, const char* v) { RadParam p{}; p.key = k; p.kind = RAD_P_STR; p.sval = v; return p; }
RadTensor R2(void* d, uint32_t dt, int64_t r, int64_t c) { RadTensor t{}; t.data = d; t.dtype = dt; t.rank = 2; t.shape[0] = r; t.shape[1] = c; t.stride[0] = c; t.stride[1] = 1; return t; }
float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }

/* Hostile bf16 rows (valid finite inputs): kind 0 all zero, 1 large magnitudes (up to 2e4), 2 one spike of 3e4 over tiny (1e-3) values, 3 alternating
 * +-large. A row-banded kernel with a batch-wide scale, max or clamp would let such a row move its neighbours; the harness places them as neighbours. */
void hostile_bf16(uint16_t* row, size_t n, int kind, std::mt19937& g) {
    for (size_t i = 0; i < n; ++i) {
        float v = 0.f;
        const float u = ((int)(g() % 20001) - 10000) / 10000.0f;
        if (kind == 1) v = u * 20000.f;
        else if (kind == 2) v = u * 1e-3f;
        else if (kind == 3) v = (i & 1 ? -1.f : 1.f) * 8000.f * (0.5f + std::fabs(u));
        row[i] = rad_f32_to_bf16(v);
    }
    if (kind == 2 && n) row[g() % n] = rad_f32_to_bf16(30000.f);
}
/* the last `n` rows before the final four rows of a pool of `rows` rows are hostile: returns their indices */
std::vector<int> hostile_rows(int pool, int n = 5) {
    std::vector<int> h;
    for (int i = 0; i < n; ++i) h.push_back(pool - 4 - n + i);
    return h;
}

int pair_case(int N, int K, int pool, const std::vector<int>& sizes) {
    std::mt19937 g(6 + K);
    auto bf = [&](float s) { return rad_f32_to_bf16(s * ((int)(g() % 20001) - 10000) / 10000.0f); };
    std::vector<uint16_t> x((size_t)pool * K), w0((size_t)N * K), w1((size_t)N * K);
    for (auto& v : x) v = bf(2.f);
    for (auto& v : w0) v = bf(0.05f);
    for (auto& v : w1) v = bf(0.05f);
    const std::vector<int> hostile = hostile_rows(pool);
    for (size_t i = 0; i < hostile.size(); ++i) hostile_bf16(&x[(size_t)hostile[i] * K], K, (int)(i % 4), g);
    char name[64];
    std::snprintf(name, sizeof name, "pair N=%d K=%d", N, K);
    rowid::Family f{name, pool, (size_t)N * 8, nullptr, hostile};
    int bad = 0;
    f.run = [&](const std::vector<int>& rows, std::vector<uint8_t>& out) {
        const int T = (int)rows.size();
        std::vector<uint16_t> xb((size_t)T * K);
        for (int i = 0; i < T; ++i) std::memcpy(&xb[(size_t)i * K], &x[(size_t)rows[i] * K], K * 2);
        std::vector<float> o0((size_t)T * N, -9.f), o1((size_t)T * N, -9.f);
        RadTensor t[5] = {R2(xb.data(), RAD_BF16, T, K), R2(w0.data(), RAD_BF16, N, K), R2(w1.data(), RAD_BF16, N, K),
                          R2(o0.data(), RAD_F32, T, N), R2(o1.data(), RAD_F32, T, N)};
        RadParam p[] = {I("M", T), I("N", N), I("K", K), S("dtype", "bf16")};
        RadArgs a{}; a.t = t; a.n_t = 5; a.p = p; a.n_p = 4;
        const int rc = r4d_rad_gemm_nt_f32_pair(&a, nullptr);
        if (rc) return rc;
        /* value check on the first row of each launch: |y - f64 sum| <= K * 2^-23 * sum |x w| + tiny (any f32 summation order) */
        for (int q = 0; q < 2; ++q) {
            const std::vector<uint16_t>& w = q ? w1 : w0;
            const std::vector<float>& o = q ? o1 : o0;
            for (int c = 0; c < N; ++c) {
                double s = 0, sa = 0;
                for (int j = 0; j < K; ++j) { const double v = (double)bf2f(xb[j]) * bf2f(w[(size_t)c * K + j]); s += v; sa += std::fabs(v); }
                if (!(std::fabs(o[c] - s) <= K * 1.2e-7 * sa + 1e-30)) { if (bad < 4) std::printf("FAIL %s value col %d: %g vs %g\n", name, c, o[c], s); ++bad; }
            }
        }
        out.resize((size_t)T * N * 8);
        for (int i = 0; i < T; ++i) {
            std::memcpy(&out[(size_t)i * N * 8], &o0[(size_t)i * N], N * 4);
            std::memcpy(&out[(size_t)i * N * 8 + N * 4], &o1[(size_t)i * N], N * 4);
        }
        return 0;
    };
    return rowid::check(f, sizes) + bad;
}
RadParam F(const char* k, double v) { RadParam p{}; p.key = k; p.kind = RAD_P_F64; p.dval = v; return p; }
RadTensor R1(void* d, uint32_t dt, int64_t n) { RadTensor t{}; t.data = d; t.dtype = dt; t.rank = 1; t.shape[0] = n; t.stride[0] = 1; return t; }

int woa_case(int H, int D, int Rr, int G, int K, int pool, const std::vector<int>& sizes) {
    std::mt19937 g(4 + D);
    auto bf = [&](float s) { return rad_f32_to_bf16(s * ((int)(g() % 20001) - 10000) / 10000.0f); };
    const int per = (H / G) * D;
    std::vector<uint16_t> o((size_t)pool * H * D), w((size_t)G * K * per);
    std::vector<int32_t> pos(pool);
    for (auto& v : o) v = bf(2.f);
    for (auto& v : w) v = bf(0.05f);
    for (int m = 0; m < pool; ++m) pos[m] = 300 + 37 * m;
    const std::vector<int> hostile = hostile_rows(pool);
    for (size_t i = 0; i < hostile.size(); ++i) hostile_bf16(&o[(size_t)hostile[i] * H * D], (size_t)H * D, (int)(i % 4), g);
    char name[64];
    std::snprintf(name, sizeof name, "woa H=%d D=%d R=%d G=%d rank=%d", H, D, Rr, G, K);
    rowid::Family f{name, pool, (size_t)G * K * 2, nullptr, hostile};
    f.run = [&](const std::vector<int>& rows, std::vector<uint8_t>& out) {
        const int T = (int)rows.size();
        std::vector<uint16_t> ob((size_t)T * H * D), y((size_t)T * G * K, 0xdead);
        std::vector<int32_t> pb(T);
        for (int i = 0; i < T; ++i) { std::memcpy(&ob[(size_t)i * H * D], &o[(size_t)rows[i] * H * D], (size_t)H * D * 2); pb[i] = pos[rows[i]]; }
        RadTensor t[4] = {R2(ob.data(), RAD_BF16, T, H * D), R1(pb.data(), RAD_I32, T), R2(w.data(), RAD_BF16, G * K, per), R2(y.data(), RAD_BF16, T, G * K)};
        RadParam p[] = {I("heads", H), I("head_dim", D), I("rope_dim", Rr), I("groups", G), I("rank", K), F("theta", 160000.0)};
        RadArgs a{}; a.t = t; a.n_t = 4; a.p = p; a.n_p = 6;
        const int rc = r4d_rad_ernie_hca_wo_a(&a, nullptr);
        if (rc) return rc;
        out.resize(y.size() * 2);
        std::memcpy(out.data(), y.data(), out.size());
        return 0;
    };
    return rowid::check(f, sizes);
}
int mlaproj_case(bool absorb, int pool, const std::vector<int>& sizes) {
    const int H = 2, kd = 64, vd = 64, L = 256;
    const int xw = absorb ? kd : L, yw = absorb ? L : vd;
    std::mt19937 g(absorb ? 21 : 22);
    auto bf = [&](float s) { return rad_f32_to_bf16(s * ((int)(g() % 20001) - 10000) / 10000.0f); };
    std::vector<uint16_t> x((size_t)pool * H * xw), w((size_t)H * (kd + vd) * L);
    for (auto& v : x) v = bf(2.f);
    for (auto& v : w) v = bf(0.05f);
    const std::vector<int> hostile = hostile_rows(pool);
    for (size_t i = 0; i < hostile.size(); ++i) hostile_bf16(&x[(size_t)hostile[i] * H * xw], (size_t)H * xw, (int)(i % 4), g);
    rowid::Family f{absorb ? "mla_project absorb" : "mla_project expand", pool, (size_t)H * yw * 2, nullptr, hostile};
    f.run = [&](const std::vector<int>& rows, std::vector<uint8_t>& out) {
        const int T = (int)rows.size();
        std::vector<uint16_t> xb((size_t)T * H * xw), y((size_t)T * H * yw, 0xdead);
        for (int i = 0; i < T; ++i) std::memcpy(&xb[(size_t)i * H * xw], &x[(size_t)rows[i] * H * xw], (size_t)H * xw * 2);
        RadTensor t[3] = {R2(xb.data(), RAD_BF16, T, H * xw), R2(w.data(), RAD_BF16, H * (kd + vd), L), R2(y.data(), RAD_BF16, T, H * yw)};
        RadParam p[] = {I("heads", H), I("k_dim", kd), I("v_dim", vd), I("latent_dim", L), S("mode", absorb ? "absorb" : "expand")};
        RadArgs a{}; a.t = t; a.n_t = 3; a.p = p; a.n_p = 5;
        const int rc = r4d_rad_ernie_mla_project(&a, nullptr);
        if (rc) return rc;
        out.resize(y.size() * 2);
        std::memcpy(out.data(), y.data(), out.size());
        return 0;
    };
    return rowid::check(f, sizes);
}

int hca_case(int H, int pool, const std::vector<int>& sizes) {
    const int D = 512, ratio = 128, window = 128, block = 64, win_blocks = 16, cmp_blocks = 16, win_pitch = 16, cmp_pitch = 16, pages = 16, qw = H * D;
    std::mt19937 g(31);
    auto rnd = [&](float s) { return rad_f32_to_bf16(s * ((int)(g() % 20001) - 10000) / 10000.0f); };
    std::vector<uint16_t> q((size_t)pool * qw), local((size_t)pages * block * D), comp((size_t)pages * block * D);
    for (auto& v : q) v = rnd(2.f);
    for (auto& v : local) v = rnd(1.5f);
    for (auto& v : comp) v = rnd(1.5f);
    std::vector<float> sink(H);
    for (int h = 0; h < H; ++h) sink[h] = 0.3f - 0.1f * h;
    /* Request 0 reads positions 0 .. 64 * 16 - 1 (16 window blocks from origin 0; 16 compressed token blocks), pages shuffled over a 16-page pool. */
    const int base = 64;
    int32_t origins[] = {0, -99, 0};
    int32_t window_table[32], compressed_table[32];
    for (int i = 0; i < 16; ++i) { window_table[i] = (i * 5 + 3) % 16; compressed_table[i] = (i * 7 + 1) % 16; window_table[16 + i] = compressed_table[16 + i] = -1; }
    /* Union attention reads a chunk of one request: its rows are consecutive positions, so a launch is a contiguous run of the pool (the row's
     * neighbours are the rows around it, not random rows). */
    auto run = [&](int start, int T, std::vector<uint16_t>& y) {
        y.assign((size_t)T * qw, 0xdead);
        std::vector<int32_t> pb(T), rb(T, 0);
        for (int i = 0; i < T; ++i) pb[i] = base + start + i;
        auto pg = [&](void* d) { RadTensor t{}; t.data = d; t.dtype = RAD_BF16; t.rank = 4; t.shape[0] = pages; t.shape[1] = 1; t.shape[2] = block; t.shape[3] = D;
                                 t.stride[0] = (int64_t)block * D; t.stride[1] = (int64_t)block * D; t.stride[2] = D; t.stride[3] = 1; return t; };
        RadTensor ot{}; ot.data = origins; ot.dtype = RAD_I32; ot.rank = 1; ot.shape[0] = 2; ot.stride[0] = 2;
        RadTensor t[10] = {R2(&q[(size_t)start * qw], RAD_BF16, T, qw), pg(local.data()), pg(comp.data()), R1(pb.data(), RAD_I32, T), R1(rb.data(), RAD_I32, T),
                           R2(window_table, RAD_I32, 2, win_pitch), R2(compressed_table, RAD_I32, 2, cmp_pitch), ot, R1(sink.data(), RAD_F32, H), R2(y.data(), RAD_BF16, T, qw)};
        t[5].shape[1] = win_blocks; t[6].shape[1] = cmp_blocks;
        RadParam sc = F("scale", 1.0 / std::sqrt((double)D));
        RadParam lay = S("cmp_layout", "token");
        RadParam p[] = {I("M", T), I("heads", H), I("head_dim", D), I("ratio", ratio), I("window", window), sc, lay};
        RadArgs a{}; a.t = t; a.n_t = 10; a.p = p; a.n_p = 7;
        return r4d_rad_ernie_hca_union_attn_origin(&a, nullptr);
    };
    const int probes[3] = {pool / 2 - 3, pool / 2, pool / 2 + 5};
    std::vector<uint16_t> alone[3];
    for (int i = 0; i < 3; ++i) if (int rc = run(probes[i], 1, alone[i])) { std::printf("FAIL hca one-row launch refused rc=%d\n", rc); return 1; }
    int bad = 0;
    for (int T : sizes) {
        if (T > pool) continue;
        int differ = 0, checked = 0, refused = 0;
        for (int i = 0; i < 3; ++i) {
            for (int where = 0; where < 3; ++where) {   /* probe first, middle, last row of the launch */
                int start = where == 0 ? probes[i] : where == 1 ? probes[i] - T / 2 : probes[i] - T + 1;
                start = std::max(0, std::min(start, pool - T));
                if (start > probes[i] || start + T <= probes[i]) continue;
                std::vector<uint16_t> y;
                if (run(start, T, y)) { ++refused; continue; }
                ++checked;
                differ += std::memcmp(&y[(size_t)(probes[i] - start) * qw], alone[i].data(), (size_t)qw * 2) != 0;
            }
        }
        std::printf("%s hca_union_attn_origin D=512 T=%4d: %d of %d probe placements differ from the same row alone (%d refused)\n", differ || refused ? "FAIL" : "PASS", T, differ, checked, refused);
        bad += differ != 0 || refused != 0;
    }
    return bad;
}

/* Union attention across requests, holes and recycled windows: three requests share the page pools through different (shuffled) tables.
 *   request 0: positions 64..363, pages from the start of its window (origin 0)
 *   request 1: positions 400..699, origin 0, holes: two window blocks and two compressed blocks are -1 (skipped keys)
 *   request 2: positions 640..939, recycled window (origin 512: the first 8 blocks are gone), compressed table complete
 * A launch gathers ANY rows of the 900-row pool (so a launch holds rows of several requests in any order); a probe is compared with the
 * same row alone for launches of 1..512 rows with the shipped band (unless the caller pins R4D_HCA_DECODE_MAX first). Both compressed layouts. */
int hca_multi_case(const char* layout, const std::vector<int>& sizes, int H = 2) {
    const int D = 512, ratio = 128, window = 128, block = 64, W = 32, pitch = 32, pages = 32, qw = H * D, pool = 1200, NR = 4;
    std::mt19937 g(77);
    auto rnd = [&](float s) { return rad_f32_to_bf16(s * ((int)(g() % 20001) - 10000) / 10000.0f); };
    std::vector<uint16_t> q((size_t)pool * qw), local((size_t)pages * block * D), comp((size_t)pages * block * D);
    for (auto& v : q) v = rnd(2.f);
    for (auto& v : local) v = rnd(1.5f);
    for (auto& v : comp) v = rnd(1.5f);
    std::vector<float> sink(H);
    for (int h = 0; h < H; ++h) sink[h] = 0.3f - 0.1f * h;
    int32_t origins[NR] = {0, 0, 512, 960};
    int32_t wtab[NR * pitch], ctab[NR * pitch];
    for (int k = 0; k < NR; ++k)
        for (int i = 0; i < W; ++i) { wtab[k * pitch + i] = (i * (5 + 2 * k) + 3 + k) % pages; ctab[k * pitch + i] = (i * (7 + 4 * k) + 1 + 2 * k) % pages; }
    wtab[1 * pitch + 6] = wtab[1 * pitch + 9] = -1;
    ctab[1 * pitch + 3] = ctab[1 * pitch + 10] = -1;
    const int p0[NR] = {64, 400, 640, 1100};   /* request 3: positions 1100..1399, n_cmp 8..10: every one of the 8 key partitions holds compressed rows */
    auto row_pos = [&](int i) { return p0[i / 300] + i % 300; };
    auto row_req = [&](int i) { return i / 300; };
    std::vector<int> hostile;
    for (int i = 0; i < 6; ++i) hostile.push_back(150 + 300 * (i % 4) + i);   /* a few in each request: zero, large, spike, alternating */
    for (size_t i = 0; i < hostile.size(); ++i) hostile_bf16(&q[(size_t)hostile[i] * qw], (size_t)qw, (int)(i % 4), g);
    rowid::Family f{"hca_multi", pool, (size_t)qw * 2, nullptr, hostile};
    f.run = [&](const std::vector<int>& rows, std::vector<uint8_t>& out) {
        const int T = (int)rows.size();
        std::vector<uint16_t> qg((size_t)T * qw), y((size_t)T * qw, 0xdead);
        std::vector<int32_t> pb(T), rb(T);
        for (int i = 0; i < T; ++i) {
            std::memcpy(&qg[(size_t)i * qw], &q[(size_t)rows[i] * qw], (size_t)qw * 2);
            pb[i] = row_pos(rows[i]); rb[i] = row_req(rows[i]);
        }
        auto pg = [&](void* d) { RadTensor t{}; t.data = d; t.dtype = RAD_BF16; t.rank = 4; t.shape[0] = pages; t.shape[1] = 1; t.shape[2] = block; t.shape[3] = D;
                                 t.stride[0] = (int64_t)block * D; t.stride[1] = (int64_t)block * D; t.stride[2] = D; t.stride[3] = 1; return t; };
        RadTensor ot{}; ot.data = origins; ot.dtype = RAD_I32; ot.rank = 1; ot.shape[0] = NR; ot.stride[0] = 1;
        RadTensor t[10] = {R2(qg.data(), RAD_BF16, T, qw), pg(local.data()), pg(comp.data()), R1(pb.data(), RAD_I32, T), R1(rb.data(), RAD_I32, T),
                           R2(wtab, RAD_I32, NR, pitch), R2(ctab, RAD_I32, NR, pitch), ot, R1(sink.data(), RAD_F32, H), R2(y.data(), RAD_BF16, T, qw)};
        RadParam sc = F("scale", 1.0 / std::sqrt((double)D));
        RadParam lay = S("cmp_layout", layout);
        RadParam p[] = {I("M", T), I("heads", H), I("head_dim", D), I("ratio", ratio), I("window", window), sc, lay};
        RadArgs a{}; a.t = t; a.n_t = 10; a.p = p; a.n_p = 7;
        const int rc = r4d_rad_ernie_hca_union_attn_origin(&a, nullptr);
        out.resize(y.size() * 2);
        std::memcpy(out.data(), y.data(), out.size());
        return rc;
    };
    return rowid::check(f, sizes, 23);
}

int mla_sparse_case(int H, int W, int pool, const std::vector<int>& sizes) {
    const int L = 512, Rr = 64, bs = 4, pages = std::max(128, W / bs + 16);   /* W distinct slots a row need W / bs pages */
    std::mt19937 g(41);
    auto bf = [&](float s) { return rad_f32_to_bf16(s * ((int)(g() % 20001) - 10000) / 10000.0f); };
    std::vector<uint16_t> q((size_t)pool * H * L), qr((size_t)pool * H * Rr), kv((size_t)pages * bs * (L + Rr));
    std::vector<float> sink(H);
    std::vector<int32_t> slots((size_t)pool * W);
    for (auto& v : q) v = bf(2.f);
    for (auto& v : qr) v = bf(2.f);
    for (auto& v : kv) v = bf(1.5f);
    for (int h = 0; h < H; ++h) sink[h] = 0.2f * h - 0.5f;
    for (int m = 0; m < pool; ++m) {
        std::vector<int> sl(pages * bs);
        for (int i = 0; i < pages * bs; ++i) sl[i] = i;
        std::shuffle(sl.begin(), sl.end(), g);
        for (int j = 0; j < W; ++j) slots[(size_t)m * W + j] = (j % 17 == 5) ? -1 : sl[j];
    }
    /* hostile rows: fully masked (every slot -1), one valid slot, large queries, zero queries, a spike */
    const std::vector<int> hostile = hostile_rows(pool);
    for (size_t i = 0; i < hostile.size(); ++i) {
        const int m = hostile[i];
        if (i == 0) for (int j = 0; j < W; ++j) slots[(size_t)m * W + j] = -1;
        else if (i == 1) for (int j = 1; j < W; ++j) slots[(size_t)m * W + j] = -1;
        else if (i == 2) for (int j = 0; j < W; ++j) if (j % 8 != 3) slots[(size_t)m * W + j] = -1;   /* one partition only */
        hostile_bf16(&q[(size_t)m * H * L], (size_t)H * L, i == 0 ? 0 : (int)(1 + i % 3), g);
        hostile_bf16(&qr[(size_t)m * H * Rr], (size_t)H * Rr, (int)(1 + i % 3), g);
    }
    char name[64];
    std::snprintf(name, sizeof name, "mla_sparse_attn L=512 W=%d", W);
    rowid::Family f{name, pool, (size_t)H * L * 2, nullptr, hostile};
    f.run = [&](const std::vector<int>& rows, std::vector<uint8_t>& out) {
        const int T = (int)rows.size();
        std::vector<uint16_t> qb((size_t)T * H * L), rb((size_t)T * H * Rr), y((size_t)T * H * L, 0xdead);
        std::vector<int32_t> sb((size_t)T * W);
        for (int i = 0; i < T; ++i) {
            std::memcpy(&qb[(size_t)i * H * L], &q[(size_t)rows[i] * H * L], (size_t)H * L * 2);
            std::memcpy(&rb[(size_t)i * H * Rr], &qr[(size_t)rows[i] * H * Rr], (size_t)H * Rr * 2);
            std::memcpy(&sb[(size_t)i * W], &slots[(size_t)rows[i] * W], (size_t)W * 4);
        }
        RadTensor kt{}; kt.data = kv.data(); kt.dtype = RAD_BF16; kt.rank = 4; kt.shape[0] = pages; kt.shape[1] = 1; kt.shape[2] = bs; kt.shape[3] = L + Rr;
        kt.stride[0] = (int64_t)bs * (L + Rr); kt.stride[1] = (int64_t)bs * (L + Rr); kt.stride[2] = L + Rr; kt.stride[3] = 1;
        RadTensor t[6] = {R2(qb.data(), RAD_BF16, T, H * L), R2(rb.data(), RAD_BF16, T, H * Rr), kt, R2(sb.data(), RAD_I32, T, W), R1(sink.data(), RAD_F32, H), R2(y.data(), RAD_BF16, T, H * L)};
        RadParam p[] = {I("heads", H), I("latent_dim", L), I("rope_dim", Rr), I("block_size", bs), I("width", W), I("num_pages", pages), F("scale", 0.05)};
        RadArgs a{}; a.t = t; a.n_t = 6; a.p = p; a.n_p = 7;
        const int rc = r4d_rad_ernie_mla_sparse_attn(&a, nullptr);
        if (rc) return rc;
        out.resize(y.size() * 2);
        std::memcpy(out.data(), y.data(), out.size());
        return 0;
    };
    return rowid::check(f, sizes);
}

int mhc_case(int n, int pool, const std::vector<int>& sizes) {
    const int hc = 4, width = n * hc, mix = hc * (hc + 2);
    std::mt19937 g(51 + n);
    auto fr = [&](float s) { return s * ((int)(g() % 20001) - 10000) / 10000.0f; };
    std::vector<uint16_t> h((size_t)pool * width), gain(n);
    std::vector<float> fn((size_t)mix * width), scale = {0.5f, 0.4f, 0.3f}, base(mix);
    for (auto& v : h) v = rad_f32_to_bf16(fr(3.f));
    const std::vector<int> hostile = hostile_rows(pool);
    for (size_t i = 0; i < hostile.size(); ++i) hostile_bf16(&h[(size_t)hostile[i] * width], (size_t)width, (int)(i % 4), g);
    for (auto& v : gain) v = rad_f32_to_bf16(1.f + fr(.1f));
    for (auto& v : fn) v = fr(.02f);
    for (auto& v : base) v = fr(.5f);
    char name[48];
    std::snprintf(name, sizeof name, "mhc_pre_rms hc=4 n=%d", n);
    const size_t ob = (size_t)n * 2 + hc * 4 + hc * hc * 4;
    rowid::Family f{name, pool, ob, nullptr, hostile};
    f.run = [&](const std::vector<int>& rows, std::vector<uint8_t>& out) {
        const int T = (int)rows.size();
        std::vector<uint16_t> hb((size_t)T * width), x((size_t)T * n, 0xdead);
        std::vector<float> post((size_t)T * hc, -1.f), comb((size_t)T * hc * hc, -1.f);
        for (int i = 0; i < T; ++i) std::memcpy(&hb[(size_t)i * width], &h[(size_t)rows[i] * width], (size_t)width * 2);
        RadTensor t[8] = {R2(hb.data(), RAD_BF16, T, width), R2(fn.data(), RAD_F32, mix, width), R1(scale.data(), RAD_F32, 3), R1(base.data(), RAD_F32, mix),
                          R1(gain.data(), RAD_BF16, n), R2(x.data(), RAD_BF16, T, n), R2(post.data(), RAD_F32, T, hc), R2(comb.data(), RAD_F32, T, hc * hc)};
        RadParam p[] = {I("M", 2048), I("n", n), I("hc", hc), I("iterations", 20), F("eps", 1e-6), F("pre_eps", 1e-6), F("sinkhorn_eps", 1e-6), F("post_mult", 2.0), F("rms_eps", 1e-6)};
        RadArgs a{}; a.t = t; a.n_t = 8; a.p = p; a.n_p = 9;
        const int rc = r4d_rad_ernie_mhc_pre_rms(&a, nullptr);
        if (rc) return rc;
        out.resize((size_t)T * ob);
        for (int i = 0; i < T; ++i) {
            uint8_t* o = &out[(size_t)i * ob];
            std::memcpy(o, &x[(size_t)i * n], (size_t)n * 2);
            std::memcpy(o + (size_t)n * 2, &post[(size_t)i * hc], hc * 4);
            std::memcpy(o + (size_t)n * 2 + hc * 4, &comb[(size_t)i * hc * hc], hc * hc * 4);
        }
        return 0;
    };
    return rowid::check(f, sizes);
}

/* fp8a8 dense: rows carry their own E4M3 activations and per-128-column scales. The family reports whether any row differed, so the control arm
 * (the matvec for up to 8 rows) can be required to differ. */
int f8a8_case(int N, int K, int pool, const std::vector<int>& sizes, int64_t declared_mvm_max = -1) {
    const int kb = K / 128, nb = N / 128;
    std::mt19937 g(61);
    auto code = [&] { uint8_t c; do { c = (uint8_t)(g() & 0xff); } while ((c & 0x7f) == 0x7f); return c; };
    std::vector<uint8_t> xq((size_t)pool * K), w((size_t)N * K);
    std::vector<float> xs((size_t)pool * kb), ws((size_t)nb * kb);
    for (auto& v : xq) v = code();
    for (auto& v : w) v = code();
    for (auto& v : xs) v = 0.002f + (g() % 1000) * 1e-5f;
    for (auto& v : ws) v = 0.01f + (g() % 1000) * 1e-5f;
    /* hostile rows: all-zero codes; largest codes (0x7e = 448) under a huge scale; tiny scale; alternating extremes */
    const std::vector<int> hostile = hostile_rows(pool);
    for (size_t i = 0; i < hostile.size(); ++i) {
        uint8_t* r = &xq[(size_t)hostile[i] * K];
        float* s = &xs[(size_t)hostile[i] * kb];
        for (int j = 0; j < K; ++j) r[j] = (i % 4) == 0 ? 0 : (i % 4) == 3 ? ((j & 1) ? 0xfe : 0x7e) : 0x7e;
        for (int j = 0; j < kb; ++j) s[j] = (i % 4) == 2 ? 1e-6f : (i % 4) == 0 ? 0.5f : 40.f;
    }
    rowid::Family f{"fp8a8f32 dense", pool, (size_t)N * 2, nullptr, hostile};
    f.run = [&](const std::vector<int>& rows, std::vector<uint8_t>& out) {
        const int T = (int)rows.size();
        std::vector<uint8_t> xb((size_t)T * K);
        std::vector<float> sb((size_t)T * kb);
        std::vector<uint16_t> y((size_t)T * N, 0xdead);
        for (int i = 0; i < T; ++i) { std::memcpy(&xb[(size_t)i * K], &xq[(size_t)rows[i] * K], K); std::memcpy(&sb[(size_t)i * kb], &xs[(size_t)rows[i] * kb], kb * 4); }
        RadTensor t[5] = {R2(xb.data(), RAD_F8E4M3, T, K), R2(sb.data(), RAD_F32, T, kb), R2(w.data(), RAD_F8E4M3, N, K), R2(ws.data(), RAD_F32, nb, kb), R2(y.data(), RAD_BF16, T, N)};
        /* the plugin declares mvm_max=0 on its fp8a8f32 ops; a negative `declared_mvm_max` omits the parameter (a generic model's op) */
        RadParam p[] = {I("M", T), I("N", N), I("K", K), I("group", 128), S("dtype", "fp8a8f32"), I("mvm_max", declared_mvm_max)};
        RadArgs a{}; a.t = t; a.n_t = 5; a.p = p; a.n_p = declared_mvm_max >= 0 ? 6 : 5;
        const int rc = r4d_rad_fp8_f32scale(&a, nullptr);
        if (rc) return rc;
        out.resize(y.size() * 2);
        std::memcpy(out.data(), y.data(), out.size());
        return 0;
    };
    return rowid::check(f, sizes);
}

/* ---------------------------------------------------------------------------------------------------------------------------------------------------
 * The families of the lossless path outside the four rewritten ones (review M2): bf16 gemm_nt (the indexer ik / iw projections), the indexer logits,
 * the router (fused one-row form against the split form), and the moe_gather forms. */

/* bf16 gemm_nt as the plugin issues it for the indexer's ik (N=128) and iw (N=heads): the row the engine resolves by step size (M <= 16 the m16 row, which forwards to
 * m64; M <= 64 the m64 row; above 64 with N >= 128 the tiled row). `tuned` mimics a tune cache that gave each band its own split (the core appends sk to the
 * launch); `ri` is the op's declared row_identity. A split fixes the summation order, so differing splits across bands change bits unless row_identity ignores them. */
int gemmnt_case(int N, int K, int pool, const std::vector<int>& sizes, bool tuned, bool ri) {
    std::mt19937 g(91 + N);
    auto bf = [&](float s) { return rad_f32_to_bf16(s * ((int)(g() % 20001) - 10000) / 10000.0f); };
    std::vector<uint16_t> x((size_t)pool * K), w((size_t)N * K);
    for (auto& v : x) v = bf(2.f);
    for (auto& v : w) v = bf(0.05f);
    /* Cancellation, so the f32 summation order is visible in a bf16 output: the row's second half is the negation of its first (the weight's second half
     * repeats its first) under per-column power-of-two scales, so the exact sum is 0 and the output is the rounding residue, which depends on the order of the
     * additions (a different split changes it). */
    for (int j = 0; j < K / 2; ++j) {
        const float sc = std::ldexp(1.0f, (j * 7) % 13 - 6);
        for (int r = 0; r < pool; ++r) {
            const float v = bf2f(x[(size_t)r * K + j]) * sc;
            x[(size_t)r * K + j] = rad_f32_to_bf16(v);
            x[(size_t)r * K + j + K / 2] = rad_f32_to_bf16(-v);
        }
        for (int n = 0; n < N; ++n) w[(size_t)n * K + j + K / 2] = w[(size_t)n * K + j];
    }
    const std::vector<int> hostile = hostile_rows(pool);
    for (size_t i = 0; i < hostile.size(); ++i) hostile_bf16(&x[(size_t)hostile[i] * K], K, (int)(i % 4), g);
    char name[96];
    std::snprintf(name, sizeof name, "gemm_nt bf16 N=%d K=%d%s%s", N, K, tuned ? " per-band tuned sk" : "", ri ? " row_identity" : "");
    rowid::Family f{name, pool, (size_t)N * 2, nullptr, hostile};
    f.run = [&](const std::vector<int>& rows, std::vector<uint8_t>& out) {
        const int T = (int)rows.size();
        std::vector<uint16_t> xb((size_t)T * K), y((size_t)T * N, 0xdead);
        for (int i = 0; i < T; ++i) std::memcpy(&xb[(size_t)i * K], &x[(size_t)rows[i] * K], (size_t)K * 2);
        RadTensor t[3] = {R2(xb.data(), RAD_BF16, T, K), R2(w.data(), RAD_BF16, N, K), R2(y.data(), RAD_BF16, T, N)};
        const bool tiled = T > 64 && N >= 128;
        const int sk = !tuned ? 0 : T <= 16 ? 1 : T <= 64 ? 4 : 8;   /* the tuned split of the band; K = 2048 divides by 16 * 8 */
        RadParam p[8];
        int np = 0;
        p[np++] = I("M", T); p[np++] = I("N", N); p[np++] = I("K", K); p[np++] = S("dtype", "bf16");
        if (ri) p[np++] = I("row_identity", 1);
        if (tuned) { p[np++] = I("wv", 1); p[np++] = I("sk", sk); }
        RadArgs a{}; a.t = t; a.n_t = 3; a.p = p; a.n_p = np;
        const int rc = T <= 16 ? r4d_rad_gemm_bf16_m16(&a, nullptr) : tiled ? r4d_rad_gemm_bf16_tiled(&a, nullptr) : r4d_rad_gemm_bf16_m64(&a, nullptr);
        if (rc) return rc;
        out.resize(y.size() * 2);
        std::memcpy(out.data(), y.data(), out.size());
        return 0;
    };
    return rowid::check(f, sizes);
}

/* ernie_idx_logits (D = 128): the 2-tile form up to R4D_MLA_IDXL_SMALL_MAX (4) rows with L <= 2048, the 4-tile form above. A row is a query position of one of
 * three requests (own block table, own key rows); out is [rows, L] f32. Hostile rows: a position inside the window (every key masked), all-zero queries,
 * largest codes under a huge scale, tiny scale. */
int idxl_case(int H, int L, int pool, const std::vector<int>& sizes) {
    const int D = 128, S = 3, block = 64, B = L / block, pages = S * B, window = 128;
    std::mt19937 g(101 + H);
    auto code = [&] { uint8_t c; do { c = (uint8_t)(g() & 0xff); } while ((c & 0x7f) == 0x7f); return c; };
    std::vector<uint8_t> qf((size_t)pool * H * D), cache((size_t)pages * block * (D + 4));
    std::vector<float> qs((size_t)pool * H);
    std::vector<uint16_t> w((size_t)pool * H);
    std::vector<int32_t> bt((size_t)S * B), req(pool), pos(pool);
    for (auto& v : qf) v = code();
    for (auto& v : qs) v = 0.002f + (g() % 1000) * 1e-5f;
    for (auto& v : w) v = rad_f32_to_bf16(((int)(g() % 2001) - 1000) / 500.0f);
    for (int pg = 0; pg < pages * block; ++pg) {
        uint8_t* row = &cache[(size_t)pg * (D + 4)];
        for (int j = 0; j < D; ++j) row[j] = code();
        const float sc = 0.003f + (g() % 1000) * 1e-5f;
        std::memcpy(row + D, &sc, 4);
    }
    std::vector<int> perm(pages);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), g);
    for (int r = 0; r < S; ++r) for (int b = 0; b < B; ++b) bt[(size_t)r * B + b] = perm[(size_t)r * B + b];
    bt[(size_t)1 * B + 3] = -1;   /* a hole in request 1's table */
    for (int m = 0; m < pool; ++m) { req[m] = m % S; pos[m] = window + (int)(g() % (unsigned)(L + 1)); }   /* end = pos + 1 - window in [1, L + 1] */
    const std::vector<int> hostile = hostile_rows(pool);
    for (size_t i = 0; i < hostile.size(); ++i) {
        const int m = hostile[i];
        uint8_t* q = &qf[(size_t)m * H * D];
        float* s = &qs[(size_t)m * H];
        for (int j = 0; j < H * D; ++j) q[j] = (i % 4) == 0 ? 0 : (i % 4) == 3 ? ((j & 1) ? 0xfe : 0x7e) : 0x7e;
        for (int h = 0; h < H; ++h) s[h] = (i % 4) == 1 ? 50.f : (i % 4) == 2 ? 1e-7f : 1.f;
        if (i == 0) pos[m] = 5;   /* inside the window: every key masked */
    }
    char name[64];
    std::snprintf(name, sizeof name, "idx_logits H=%d L=%d", H, L);
    rowid::Family f{name, pool, (size_t)L * 4, nullptr, hostile};
    f.run = [&](const std::vector<int>& rows, std::vector<uint8_t>& out) {
        const int T = (int)rows.size();
        std::vector<uint8_t> qb((size_t)T * H * D);
        std::vector<float> sb((size_t)T * H), o((size_t)T * L, -7.f);
        std::vector<uint16_t> wb((size_t)T * H);
        std::vector<int32_t> rb(T), pb(T);
        for (int i = 0; i < T; ++i) {
            std::memcpy(&qb[(size_t)i * H * D], &qf[(size_t)rows[i] * H * D], (size_t)H * D);
            std::memcpy(&sb[(size_t)i * H], &qs[(size_t)rows[i] * H], (size_t)H * 4);
            std::memcpy(&wb[(size_t)i * H], &w[(size_t)rows[i] * H], (size_t)H * 2);
            rb[i] = req[rows[i]]; pb[i] = pos[rows[i]];
        }
        RadTensor ct{}; ct.data = cache.data(); ct.dtype = RAD_U8; ct.rank = 3; ct.shape[0] = pages; ct.shape[1] = block; ct.shape[2] = D + 4;
        ct.stride[0] = (int64_t)block * (D + 4); ct.stride[1] = D + 4; ct.stride[2] = 1;
        RadTensor t[8] = {R2(qb.data(), RAD_F8E4M3, T, H * D), R2(sb.data(), RAD_F32, T, H), R2(wb.data(), RAD_BF16, T, H), ct, R2(bt.data(), RAD_I32, S, B),
                          R1(rb.data(), RAD_I32, T), R1(pb.data(), RAD_I32, T), R2(o.data(), RAD_F32, T, L)};
        RadParam p[] = {I("heads", H), I("head_dim", D), I("window", window)};
        RadArgs a{}; a.t = t; a.n_t = 8; a.p = p; a.n_p = 3;
        const int rc = r4d_rad_ernie_idx_logits(&a, nullptr);
        if (rc) return rc;
        out.resize(o.size() * 4);
        std::memcpy(out.data(), o.data(), out.size());
        return 0;
    };
    return rowid::check(f, sizes);
}

/* The router: one row runs the fused ernie_router_topk_scatter (and its sorted / offsets / counts are compared with moe_scatter over the split form's ids),
 * more rows run ernie_router_topk. Logits are bf16-valued (ties occur), a quarter of the rows with only 8 distinct levels. Output per row: ids and weights. */
int router_case(int ne, int k, int renorm, int pool, const std::vector<int>& sizes) {
    std::mt19937 g(111 + ne);
    std::vector<float> lg((size_t)pool * 2 * ne), bias(ne), scale(ne);
    for (int e = 0; e < ne; ++e) { bias[e] = bf2f(rad_f32_to_bf16(((int)(g() % 2001) - 1000) / 1000.0f)); scale[e] = 1.0f + (e % 5) * 0.25f; }
    for (int m = 0; m < pool; ++m) {
        const int levels = (m % 4 == 3) ? 8 : 1000000;
        for (int e = 0; e < 2 * ne; ++e) lg[(size_t)m * 2 * ne + e] = bf2f(rad_f32_to_bf16(((int)(g() % (2 * levels + 1)) - levels) * (4.0f / levels)));
    }
    /* hostile rows: all-equal logits (every expert tied), zeros, large magnitudes, alternating extremes */
    const std::vector<int> hostile = hostile_rows(pool);
    for (size_t i = 0; i < hostile.size(); ++i)
        for (int e = 0; e < 2 * ne; ++e) {
            float v = i % 4 == 0 ? 0.f : i % 4 == 1 ? 3.f : i % 4 == 2 ? (e & 1 ? 60.f : -60.f) : 1e4f * (e % 7 - 3);
            lg[(size_t)hostile[i] * 2 * ne + e] = bf2f(rad_f32_to_bf16(v));
        }
    char name[64];
    std::snprintf(name, sizeof name, "router ne=%d k=%d renorm=%d", ne, k, renorm);
    rowid::Family f{name, pool, (size_t)k * 8, nullptr, hostile};
    f.run = [&](const std::vector<int>& rows, std::vector<uint8_t>& out) {
        const int T = (int)rows.size();
        std::vector<float> lb((size_t)T * 2 * ne), w((size_t)T * k, -7.f);
        std::vector<int32_t> ids((size_t)T * k, -7);
        for (int i = 0; i < T; ++i) std::memcpy(&lb[(size_t)i * 2 * ne], &lg[(size_t)rows[i] * 2 * ne], (size_t)2 * ne * 4);
        auto lt = [&] { return R2(lb.data(), RAD_F32, T, 2 * ne); };
        auto split = [&](std::vector<int32_t>& id, std::vector<float>& ww) {
            RadTensor t[5] = {lt(), R1(bias.data(), RAD_F32, ne), R1(scale.data(), RAD_F32, ne), R2(id.data(), RAD_I32, T, k), R2(ww.data(), RAD_F32, T, k)};
            RadParam p[] = {I("M", T), I("experts", ne), I("topk", k), I("renormalize", renorm)};
            RadArgs a{}; a.t = t; a.n_t = 5; a.p = p; a.n_p = 4;
            return r4d_rad_ernie_router_topk(&a, nullptr);
        };
        auto scatter = [&](const std::vector<int32_t>& id, std::vector<int32_t>& sorted, std::vector<int32_t>& off, std::vector<int32_t>& cnt) {
            sorted.assign((size_t)T * k, -5); off.assign(ne + 1, -5); cnt.assign(ne, -5);
            RadTensor t[4] = {R2(const_cast<int32_t*>(id.data()), RAD_I32, T, k), R1(sorted.data(), RAD_I32, (int64_t)T * k), R1(off.data(), RAD_I32, ne + 1), R1(cnt.data(), RAD_I32, ne)};
            RadParam p[] = {I("M", T), I("n_expert", ne), I("top_k", k), I("expert_base", 0)};
            RadArgs a{}; a.t = t; a.n_t = 4; a.p = p; a.n_p = 4;
            return r4d_rad_moe_scatter(&a, nullptr);
        };
        if (T == 1) {
            std::vector<int32_t> so(k, -5), of(ne + 1, -5), ct(ne, -5);
            RadTensor t[8] = {lt(), R1(bias.data(), RAD_F32, ne), R1(scale.data(), RAD_F32, ne), R2(ids.data(), RAD_I32, 1, k), R2(w.data(), RAD_F32, 1, k),
                              R1(so.data(), RAD_I32, k), R1(of.data(), RAD_I32, ne + 1), R1(ct.data(), RAD_I32, ne)};
            RadParam p[] = {I("M", 1), I("experts", ne), I("topk", k), I("renormalize", renorm)};
            RadArgs a{}; a.t = t; a.n_t = 8; a.p = p; a.n_p = 4;
            if (int rc = r4d_rad_ernie_router_topk_scatter(&a, nullptr)) return rc;
            /* the fused form's scatter outputs against the split form (topk + moe_scatter) on the same row */
            std::vector<int32_t> id2((size_t)k, -7), so2, of2, ct2;
            std::vector<float> w2((size_t)k, -7.f);
            if (int rc = split(id2, w2)) return rc;
            if (int rc = scatter(id2, so2, of2, ct2)) return rc;
            if (std::memcmp(id2.data(), ids.data(), (size_t)k * 4) || std::memcmp(w2.data(), w.data(), (size_t)k * 4) || so != so2 || of != of2 || ct != ct2) {
                std::printf("FAIL %s: the fused one-row router differs from the split form (ids %d weights %d sorted %d offsets %d counts %d)\n", name,
                            (int)!!std::memcmp(id2.data(), ids.data(), (size_t)k * 4), (int)!!std::memcmp(w2.data(), w.data(), (size_t)k * 4), (int)(so != so2), (int)(of != of2), (int)(ct != ct2));
                return 99;
            }
        } else {
            if (int rc = split(ids, w)) return rc;
        }
        out.resize((size_t)T * k * 8);
        for (int i = 0; i < T; ++i) {
            std::memcpy(&out[(size_t)i * k * 8], &ids[(size_t)i * k], (size_t)k * 4);
            std::memcpy(&out[(size_t)i * k * 8 + (size_t)k * 4], &w[(size_t)i * k], (size_t)k * 4);
        }
        return 0;
    };
    return rowid::check(f, sizes);
}

/* moe_gather as the plugin issues it: expert-output rows in sorted (expert) order, f32 per-token weights, a bf16 [token, n] result. The forms: the self-scan
 * column kernel up to kGatherSelfT slots, the prepass + column kernel above (the wide kernel needs bf16 weights, which ERNIE does not use), and with
 * `chain` 1 the chained form the core selects for decode steps of up to 64 rows. The expert rows of a token are the same whatever the order of the slots. */
int gather_case(int n, int k, int pool, const std::vector<int>& sizes, bool chain) {
    std::mt19937 g(121 + n);
    auto bf = [&](float s) { return rad_f32_to_bf16(s * ((int)(g() % 20001) - 10000) / 10000.0f); };
    std::vector<uint16_t> ye((size_t)pool * k * n);
    std::vector<float> ew((size_t)pool * k);
    for (auto& v : ye) v = bf(2.f);
    for (auto& v : ew) v = 0.05f + (g() % 1000) * 1e-3f;
    const std::vector<int> hostile = hostile_rows(pool);
    for (size_t i = 0; i < hostile.size(); ++i) {
        for (int j = 0; j < k; ++j) {
            hostile_bf16(&ye[((size_t)hostile[i] * k + j) * n], (size_t)n, (int)(i % 4), g);
            ew[(size_t)hostile[i] * k + j] = i % 4 == 0 ? 0.f : i % 4 == 1 ? 1e3f : 1e-6f;
        }
    }
    char name[64];
    std::snprintf(name, sizeof name, "moe_gather n=%d top_k=%d%s", n, k, chain ? " chain" : "");
    rowid::Family f{name, pool, (size_t)n * 2, nullptr, hostile};
    f.run = [&](const std::vector<int>& toks, std::vector<uint8_t>& out) {
        const int T = (int)toks.size(), P = T * k;
        std::vector<int32_t> sorted(P);
        std::iota(sorted.begin(), sorted.end(), 0);
        std::shuffle(sorted.begin(), sorted.end(), g);   /* slot r holds flat entry sorted[r] = token * k + j */
        std::vector<uint16_t> yb((size_t)P * n), y((size_t)T * n, 0xdead);
        std::vector<float> wb((size_t)T * k);
        for (int r = 0; r < P; ++r) {
            const int flat = sorted[r], tk = flat / k, j = flat % k;
            std::memcpy(&yb[(size_t)r * n], &ye[((size_t)toks[tk] * k + j) * n], (size_t)n * 2);
        }
        for (int i = 0; i < T; ++i) std::memcpy(&wb[(size_t)i * k], &ew[(size_t)toks[i] * k], (size_t)k * 4);
        std::vector<int32_t> scratch(P + 8);
        RadTensor t[6] = {R2(yb.data(), RAD_BF16, P, n), R2(wb.data(), RAD_F32, T, k), R1(sorted.data(), RAD_I32, P), RadTensor{}, RadTensor{}, R2(y.data(), RAD_BF16, T, n)};
        RadParam p[] = {I("M", T), I("n", n), I("top_k", k), S("dtype", "bf16"), S("act", "none"), I("chain", chain && T <= 64 ? 1 : 0)};
        RadArgs a{}; a.t = t; a.n_t = 6; a.p = p; a.n_p = 6; a.scratch = scratch.data(); a.scratch_bytes = (int64_t)scratch.size() * 4;
        const int rc = r4d_rad_moe_gather(&a, nullptr);
        if (rc) return rc;
        out.resize(y.size() * 2);
        std::memcpy(out.data(), y.data(), out.size());
        return 0;
    };
    return rowid::check(f, sizes);
}

int moe_case(bool down, int pool, const std::vector<int>& sizes) {
    const int ne = 512, top = 10;
    const int64_t N = down ? 2048 : 512, K = down ? 256 : 2048, parts = down ? 1 : 2;
    const bool sorted_a = down;
    std::mt19937 g(3);
    const int64_t half0 = (ne + 1) / 2, half1 = ne / 2;
    std::vector<uint8_t> w0(half0 * N * K / 2), w1(half1 * N * K / 2), s0(half0 * N * K / 32), s1(half1 * N * K / 32);
    for (auto& v : w0) v = (uint8_t)g();
    for (auto& v : w1) v = (uint8_t)g();
    for (auto& v : s0) v = (uint8_t)(122 + g() % 8);
    for (auto& v : s1) v = (uint8_t)(122 + g() % 8);
    std::vector<std::array<int, 10>> ids(pool);
    for (auto& r : ids) { std::vector<int> e(ne); std::iota(e.begin(), e.end(), 0); std::shuffle(e.begin(), e.end(), g); for (int k = 0; k < top; ++k) r[k] = e[k]; }
    const int64_t ar_per = sorted_a ? top : 1;
    std::vector<uint8_t> act((size_t)pool * ar_per * K);
    std::vector<float> acts((size_t)pool * ar_per * (K / 128));
    for (auto& v : act) { uint8_t c; do { c = (uint8_t)g(); } while ((c & 0x7f) == 0x7f); v = c; }
    for (auto& v : acts) v = 0.004f + (g() % 1000) * 1e-5f;
    auto T3 = [&](void* d, uint32_t dt, int64_t a, int64_t b, int64_t c) { RadTensor t{}; t.data = d; t.dtype = dt; t.rank = 3; t.shape[0] = a; t.shape[1] = b; t.shape[2] = c; t.stride[0] = b * c; t.stride[1] = c; t.stride[2] = 1; return t; };
    rowid::Family f{down ? "moe_mxfp4 down" : "moe_mxfp4 gate_up", pool, (size_t)top * N * 2, nullptr};
    f.run = [&](const std::vector<int>& toks, std::vector<uint8_t>& out) {
        const int M = (int)toks.size(), T = M * top;
        std::vector<int> flat_expert(T), sorted(T), off(ne + 1, 0);
        for (int i = 0; i < M; ++i) for (int k = 0; k < top; ++k) flat_expert[i * top + k] = ids[toks[i]][k];
        std::iota(sorted.begin(), sorted.end(), 0);
        std::stable_sort(sorted.begin(), sorted.end(), [&](int a, int b) { return flat_expert[a] < flat_expert[b]; });
        for (int q = 0; q < T; ++q) off[flat_expert[q] + 1]++;
        for (int e = 0; e < ne; ++e) off[e + 1] += off[e];
        const int64_t Ar = sorted_a ? T : M;
        std::vector<uint8_t> aq((size_t)Ar * K);
        std::vector<float> as((size_t)Ar * (K / 128));
        for (int64_t r = 0; r < Ar; ++r) {
            const int64_t src = sorted_a ? (int64_t)toks[sorted[r] / top] * top + sorted[r] % top : toks[r];
            std::memcpy(&aq[r * K], &act[src * K], K);
            std::memcpy(&as[r * (K / 128)], &acts[src * (K / 128)], (K / 128) * 4);
        }
        std::vector<uint16_t> y((size_t)T * N, 0xdead);
        RadTensor t[9] = {R2(aq.data(), RAD_F8E4M3, Ar, K), R2(as.data(), RAD_F32, Ar, K / 128), T3(w0.data(), RAD_FP4E2M1, half0, N, K), T3(s0.data(), RAD_U8, half0, N, K / 32),
                          R1(sorted.data(), RAD_I32, T), R1(off.data(), RAD_I32, ne + 1), T3(w1.data(), RAD_FP4E2M1, half1, N, K), T3(s1.data(), RAD_U8, half1, N, K / 32), R2(y.data(), RAD_BF16, T, N)};
        RadParam p[] = {I("M", M), I("N", N), I("K", K), I("group", 128), I("n_expert", ne), I("top_k", top), I("parts", parts), S("a_order", sorted_a ? "sorted" : "token")};
        RadArgs a{}; a.t = t; a.n_t = 9; a.p = p; a.n_p = 8;
        const int rc = r4d_rad_moe_mxfp4(&a, nullptr);
        if (rc) return rc;
        out.resize((size_t)T * N * 2);
        for (int r = 0; r < T; ++r) std::memcpy(&out[(size_t)sorted[r] * N * 2], &y[(size_t)r * N], N * 2);   /* back to (token, k) order */
        return 0;
    };
    return rowid::check(f, sizes);
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s pair\n", argv[0]); return 2; }
    const std::string fam = argv[1];
    /* argv[2] == "prod": the ERNIE 5.5 shapes of one TP8 rank (hidden 4096, 8 attention heads a rank of 64, 64 indexer heads, 512 experts top 10, o_lora_rank 1024)
     * and launches of up to 2048 rows; meant for the card (-DROWID_DEVICE), where a launch costs microseconds. On the emulator it takes hours. */
    const bool prod = argc > 2 && !std::strcmp(argv[2], "prod");
    const std::vector<int> big = {1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 33, 64, 65, 129, 2048};
    int bad = 0;
    if (prod && fam == "pair") {
        bad += pair_case(512, 4096, 2100, big);
        bad += pair_case(256, 4096, 2100, big);
    } else if (prod && fam == "woa") {
        bad += woa_case(8, 512, 64, 1, 1024, 2100, big);
        bad += woa_case(8, 512, 64, 2, 1024, 2100, big);
    } else if (prod && (fam == "hca" || fam == "hcawide" || fam == "hcadec")) {
        if (fam == "hcawide") setenv("R4D_HCA_DECODE_MAX", "0", 1);
        if (fam == "hcadec") setenv("R4D_HCA_DECODE_MAX", "16", 1);
        const std::vector<int> sz = {1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 64, 512};
        bad += hca_multi_case("token", sz, 8);
        bad += hca_multi_case("dense", sz, 8);
    } else if (prod && (fam == "mla" || fam == "mlawide")) {
        if (fam == "mlawide") setenv("R4D_MLA_COOP_MAX", "0", 1);
        bad += mla_sparse_case(8, 2176, 2100, big);
        bad += mla_sparse_case(8, 64, 2100, big);
    } else if (prod && fam == "mhc") {
        bad += mhc_case(4096, 2100, big);
    } else if (prod && (fam == "f8a8" || fam == "f8a8env")) {
        if (fam == "f8a8env") setenv("R4D_F8A8_MVM_MAX", "8", 1);
        const int shapes[][2] = {{4096, 4096}, {1024, 4096}, {512, 4096}, {4096, 1024}, {1536, 2048}, {2048, 8192}};
        for (const auto& nk : shapes) bad += f8a8_case(nk[0], nk[1], 2100, big, 0);
    } else if (prod && fam == "gemmnt") {
        bad += gemmnt_case(128, 4096, 2100, big, false, true);
        bad += gemmnt_case(64, 4096, 2100, big, false, true);
        bad += gemmnt_case(128, 4096, 2100, big, false, false);
    } else if (prod && fam == "idxl") {
        bad += idxl_case(64, 2048, 2100, big);
    } else if (prod && fam == "router") {
        bad += router_case(512, 10, 1, 2100, big);
        bad += router_case(288, 8, 1, 2100, big);
    } else if (prod && (fam == "gather" || fam == "gatherchain")) {
        bad += gather_case(1024, 10, 2100, big, fam == "gatherchain");
        bad += gather_case(2048, 8, 2100, big, fam == "gatherchain");
    } else if (fam == "pair") {
        bad += pair_case(32, 4096, 80, {1, 2, 3, 4, 5, 8, 9, 16, 17, 33, 64, 65});
        bad += pair_case(32, 544, 2100, {1, 2, 5, 9, 64, 65, 129, 2048});
    } else if (fam == "woa") {
        bad += woa_case(8, 512, 64, 2, 256, 80, {1, 2, 3, 4, 5, 8, 9, 16, 17, 33, 64, 65});
        bad += woa_case(6, 128, 32, 2, 32, 2100, {1, 2, 5, 16, 17, 64, 65, 129, 2048});
    } else if (fam == "mlaproj") {
        bad += mlaproj_case(true, 2100, {1, 2, 3, 4, 5, 8, 9, 15, 16, 17, 64, 65, 2048});
        bad += mlaproj_case(false, 2100, {1, 2, 3, 4, 5, 8, 9, 15, 16, 17, 64, 65, 2048});
    } else if (fam == "hca" || fam == "hcawide" || fam == "hcadec") {
        /* hca: the shipped band (R4D_HCA_DECODE_MAX unset: the decode kernel up to 4 rows, the wide kernel above). hcawide pins every step to the wide kernel,
         * hcadec pins every step of up to 16 rows to the decode kernel: all three must give the same bits as the row alone. */
        if (fam == "hcawide") setenv("R4D_HCA_DECODE_MAX", "0", 1);
        if (fam == "hcadec") setenv("R4D_HCA_DECODE_MAX", "16", 1);
        bad += hca_case(2, 900, {1, 2, 3, 4, 5, 8, 9, 17, 33, 64, 65, 129, 512});
        const std::vector<int> sz = {1, 2, 3, 4, 5, 8, 9, 16, 64, 512};
        bad += hca_multi_case("token", sz);
        bad += hca_multi_case("dense", sz);
    } else if (fam == "mla" || fam == "mlawide") {
        /* mla: the shipped band (R4D_MLA_COOP_MAX unset: the cooperative kernel up to 8 rows with W >= 64, the wave kernel above); mlawide pins the wave kernel */
        if (fam == "mlawide") setenv("R4D_MLA_COOP_MAX", "0", 1);
        bad += mla_sparse_case(2, 64, 2048, {1, 2, 3, 4, 5, 8, 9, 17, 64, 65, 129, 2048});
        bad += mla_sparse_case(2, 512, 600, {1, 2, 3, 5, 8, 9, 64, 65, 600});
        /* all 8 partitions (W >= 256 chunks of 32 over 8 partitions) and the production width: top 2048 + window 128 + 0 sink = 2176 */
        bad += mla_sparse_case(2, 256, 140, {1, 2, 3, 5, 8, 9, 64, 129});
        bad += mla_sparse_case(2, 2176, 140, {1, 2, 3, 4, 5, 8, 9, 16, 64, 129});
    } else if (fam == "mhc") {
        bad += mhc_case(512, 2048, {1, 2, 3, 4, 5, 8, 9, 17, 64, 65, 129, 2048});
        bad += mhc_case(1024, 2048, {1, 2, 5, 9, 65, 2048});
    } else if (fam == "moe") {
        bad += moe_case(false, 600, {1, 77, 576});
        bad += moe_case(true, 600, {1, 77, 576});
    } else if (fam == "gemmnt") {
        /* the indexer's ik (N = 128, tiled row above 64 rows) and iw (N = heads, the m64 row at every size), default (untuned) splits, with and without row_identity,
         * and with a tuned split that differs per band but row_identity declared: all identical */
        const std::vector<int> sz = {1, 2, 3, 4, 5, 8, 9, 16, 17, 33, 64, 65, 129};
        bad += gemmnt_case(128, 2048, 140, sz, false, true);
        bad += gemmnt_case(128, 2048, 140, sz, false, false);
        bad += gemmnt_case(64, 2048, 140, sz, false, true);
        bad += gemmnt_case(128, 2048, 140, sz, true, true);
        bad += gemmnt_case(64, 2048, 140, sz, true, true);
    } else if (fam == "gemmntctl") {
        /* control: a tune cache that splits each band differently, row_identity NOT declared: rows must differ across the bands (the harness has teeth, and the
         * guard is needed) */
        const int r0 = rowid::refused();
        const int failed = gemmnt_case(128, 2048, 140, {1, 2, 4, 5, 16, 17, 64, 65, 129}, true, false);
        const int ref_n = rowid::refused() - r0;   /* a refused launch is not a difference: every launch must run */
        std::printf("control: %d launches differ from the one-row result, expected > 0; %d refused, expected 0\n", failed, ref_n);
        return failed > 0 && ref_n == 0 ? 0 : 1;
    } else if (fam == "idxl") {
        const std::vector<int> sz = {1, 2, 3, 4, 5, 8, 9, 16, 17, 33, 64};
        bad += idxl_case(32, 1024, 70, sz);
        bad += idxl_case(96, 1024, 70, sz);   /* 96 heads: two head chunks */
    } else if (fam == "router") {
        const std::vector<int> sz = {1, 2, 3, 4, 5, 8, 9, 16, 17, 64, 129};
        bad += router_case(512, 10, 1, 140, sz);
        bad += router_case(288, 8, 1, 140, sz);
        bad += router_case(64, 6, 0, 140, sz);
    } else if (fam == "gather" || fam == "gatherchain") {
        /* top_k 10: 6 tokens fill the self-scan bound (T = M * k <= kGatherSelfT), so sizes cross it */
        const std::vector<int> sz = {1, 2, 3, 4, 5, 8, 9, 16, 17, 33, 64, 65, 129, 300};
        bad += gather_case(1024, 10, 310, sz, fam == "gatherchain");
        bad += gather_case(2048, 8, 310, sz, fam == "gatherchain");
    } else if (fam == "f8a8" || fam == "f8a8env") {
        /* The op declares mvm_max=0 (what the plugin does): no environment variable is involved. f8a8env sets the stale generic variable to 8 as
         * well: the declared parameter must win, so the result is the same (red against a launcher that reads the process variable). */
        if (fam == "f8a8env") setenv("R4D_F8A8_MVM_MAX", "8", 1);
        bad += f8a8_case(1024, 512, 2048, {1, 2, 3, 4, 5, 8, 9, 16, 17, 33, 64, 65, 129, 2048}, 0);
        /* production wave counts: K = 2048 (kb = 16) and K = 4096 (kb = 32) give s = 8 waves with the LDS-staged decode form; N = 1024 gives c = 2 above 16 rows */
        bad += f8a8_case(1024, 2048, 140, {1, 2, 3, 4, 5, 8, 9, 16, 17, 33, 64, 65, 129}, 0);
        bad += f8a8_case(512, 4096, 140, {1, 2, 3, 5, 8, 9, 16, 17, 33, 64, 65, 129}, 0);
    } else if (fam == "f8a8ctl") {
        const int r0 = rowid::refused();
        const int failed = f8a8_case(1024, 512, 2048, {1, 2, 3, 4, 5, 8, 9, 16, 17, 33}, 8);   /* control: declared 8 (the generic default) must NOT be row identical across the matvec / WMMA boundary */
        const int ref_n = rowid::refused() - r0;   /* a refused launch is not a difference: every launch must run */
        std::printf("control: %d launches differ from the one-row result, expected > 0; %d refused, expected 0\n", failed, ref_n);
        return failed > 0 && ref_n == 0 ? 0 : 1;
    } else {
        std::fprintf(stderr, "unknown family %s\n", fam.c_str());
        return 2;
    }
    std::printf("%d failure(s)\n", bad);
    return bad ? 1 : 0;
}
