/* tests/r4d_emu: gemm_nt_f32_pair (libr4d/r4d_glm.hip; the ERNIE router logits and the HCA compressor's kv/score projection, bf16 weights, f32 outputs)
 * on the host SIMT emulator: row independence of the device launcher AND its values against libref.
 *
 * Since 0b81b454 (row identity) every row count of an aligned bf16 launch with K % 32 == 0 takes the chunked WMMA sum and every other layout the vector kernels, so
 * in EVERY layout below a row of a launch of T rows must equal the same row launched alone, bit for bit. The row-identity suite (test_row_identity, family `pair`)
 * covers contiguous bf16 operands only; this test keeps the layout and dtype coverage it does not have (restored after the r4a merge deleted it, r2-r4a review).
 *   argv[1] is the path of libref.so. Every output of every launch must be within the forward error bound of the sum
 *             (1.5e-4 * sum_k |x_k w_k|: a K = 1024 f32 sum in two different orders differs by at most about 2 K 2^-24 of that) of libref's value, and finite.
 * Layouts (each against its own one-row launches): aligned, contiguous (the vec8 kernels, and the single-row kernel at M = 1); MISALIGNED (x and w0 start one element
 * into their buffers, so the 16-byte loads are off and the scalar-load kernels run); SLICED (the outputs are N columns of rows N + 3 wide: the padding must stay
 * untouched); F32 weights (the other dtype the launcher serves, the row-banded f32 tiles). N and K are small (the container gives a test 10 s of CPU); the WMMA
 * tile edge cases are the row counts, not the width. */
#include "rad_plugin.h"
#include "r4d_plugin.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <random>
#include <string>
#include <vector>

extern "C" int r4d_rad_gemm_nt_f32_pair(const RadArgs*, RadStream);

namespace {
RadParam I(const char* k, int64_t v) { RadParam p{}; p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }
RadParam S(const char* k, const char* v) { RadParam p{}; p.key = k; p.kind = RAD_P_STR; p.sval = v; return p; }
RadTensor R(void* d, uint32_t dt, int64_t r, int64_t c, int64_t pitch = -1) { RadTensor t{}; t.data = d; t.dtype = dt; t.rank = 2; t.shape[0] = r; t.shape[1] = c; t.stride[0] = pitch < 0 ? c : pitch; t.stride[1] = 1; return t; }
RadLaunchFn find(void* lib, const char* op) {
    auto cnt = reinterpret_cast<int (*)()>(dlsym(lib, "rad_kernel_count"));
    auto at = reinterpret_cast<const RadKernelInfo* (*)(int)>(dlsym(lib, "rad_kernel_at"));
    for (int i = 0; cnt && i < cnt(); ++i) { const RadKernelInfo* k = at(i); if (k && k->op && !std::strcmp(k->op, op)) return k->launch; }
    return nullptr;
}
struct Layout { const char* name; int x_off, ypitch_extra; bool f32w; int maxm; std::vector<int> rows; };
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s libref.so\n", argv[0]); return 2; }
    void* lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) { std::fprintf(stderr, "%s\n", dlerror()); return 2; }
    RadLaunchFn ref = find(lib, "gemm_nt_f32_pair");
    if (!ref) { std::fprintf(stderr, "libref lacks gemm_nt_f32_pair\n"); return 2; }
    constexpr int N = 128, K = 1024, MAXM = 130;
    const Layout layouts[] = {
        {"aligned", 0, 0, false, MAXM, {2, 3, 4, 8, 9, 16, 17, 33, 43, 65, 85, 129}},
        {"misaligned (x, w0 one element in)", 1, 0, false, 34, {2, 3, 8, 9, 17, 33}},
        {"sliced outputs (pitch N+3)", 0, 3, false, 34, {2, 3, 8, 9, 17, 33}},
        {"f32 weights", 0, 0, true, 34, {2, 3, 8, 9, 17, 33}},
    };
    int bad = 0;
    for (const Layout& L : layouts) {
        std::mt19937 g(6);
        auto fr = [&](float s) { return s * ((int)(g() % 20001) - 10000) / 10000.0f; };
        const int pitch = N + L.ypitch_extra;
        std::vector<uint16_t> x((size_t)L.maxm * K + 8), w0b((size_t)N * K + 8), w1b((size_t)N * K + 8);
        std::vector<float> w0f((size_t)N * K), w1f((size_t)N * K);
        for (auto& v : x) v = rad_f32_to_bf16(fr(2.f));
        for (auto& v : w0b) v = rad_f32_to_bf16(fr(0.05f));
        for (auto& v : w1b) v = rad_f32_to_bf16(fr(0.05f));
        for (auto& v : w0f) v = fr(0.05f);
        for (auto& v : w1f) v = fr(0.05f);
        const uint16_t* xb = x.data() + L.x_off;
        const uint16_t* w0p = w0b.data() + L.x_off;   /* the misaligned layout moves the weights too */
        const uint16_t* w1p = w1b.data() + L.x_off;
        auto xv = [&](int r, int j) { return rad_bf16_to_f32(xb[(size_t)r * K + j]); };
        auto wv = [&](int which, int c, int j) {
            if (L.f32w) return (which ? w1f : w0f)[(size_t)c * K + j];
            return rad_bf16_to_f32((which ? w1p : w0p)[(size_t)c * K + j]);
        };
        /* one launch: rows first..first+T-1, outputs with the layout's pitch; padding columns are pre-filled with -9 and must stay so */
        auto run = [&](RadLaunchFn fn, int first, int T, std::vector<float>& o0, std::vector<float>& o1) {
            o0.assign((size_t)T * pitch, -9.f); o1.assign((size_t)T * pitch, -9.f);
            RadTensor w0t = L.f32w ? R(w0f.data(), RAD_F32, N, K) : R((void*)w0p, RAD_BF16, N, K), w1t = L.f32w ? R(w1f.data(), RAD_F32, N, K) : R((void*)w1p, RAD_BF16, N, K);
            RadTensor t[5] = {R((void*)(xb + (size_t)first * K), RAD_BF16, T, K), w0t, w1t, R(o0.data(), RAD_F32, T, N, pitch), R(o1.data(), RAD_F32, T, N, pitch)};
            RadParam p[] = {I("M", T), I("N", N), I("K", K), S("dtype", L.f32w ? "f32" : "bf16")};
            RadArgs a{}; a.t = t; a.n_t = 5; a.p = p; a.n_p = 4;
            return fn(&a, nullptr);
        };
        std::vector<std::vector<float>> alone0((size_t)L.maxm), alone1((size_t)L.maxm);
        for (int r = 0; r < L.maxm; ++r)
            if (int rc = run(r4d_rad_gemm_nt_f32_pair, r, 1, alone0[r], alone1[r])) { std::printf("FAIL %s: one-row launch refused rc=%d\n", L.name, rc); return 1; }
        for (int T : L.rows) {
            std::vector<float> y0, y1, r0, r1;
            const int rd = run(r4d_rad_gemm_nt_f32_pair, 0, T, y0, y1), rr = run(ref, 0, T, r0, r1);
            if (rd || rr) { std::printf("FAIL %s T=%d refused (device %d, libref %d)\n", L.name, T, rd, rr); ++bad; continue; }
            int rows_differ = 0;
            for (int r = 0; r < T; ++r) {
                const bool d = std::memcmp(&y0[(size_t)r * pitch], alone0[r].data(), N * 4) || std::memcmp(&y1[(size_t)r * pitch], alone1[r].data(), N * 4);
                rows_differ += d;
            }
            /* values against libref, and the padding of a sliced output */
            long off_bound = 0, nonfinite = 0, padding = 0;
            double worst = 0;
            for (int r = 0; r < T; ++r)
                for (int c = 0; c < pitch; ++c) {
                    for (int which = 0; which < 2; ++which) {
                        const float d = (which ? y1 : y0)[(size_t)r * pitch + c], e = (which ? r1 : r0)[(size_t)r * pitch + c];
                        if (c >= N) { if (d != -9.f) ++padding; continue; }
                        if (!std::isfinite(d) || !std::isfinite(e)) { ++nonfinite; continue; }
                        double absum = 0;
                        for (int j = 0; j < K; ++j) absum += std::fabs((double)xv(r, j) * wv(which, c, j));
                        const double err = std::fabs((double)d - e), bound = 1.5e-4 * absum + 1e-30;
                        if (err / bound > worst) worst = err / bound;
                        if (err > bound) ++off_bound;
                    }
                }
            const bool ok = rows_differ == 0;   /* row identity in every layout */
            const bool vals = off_bound == 0 && nonfinite == 0 && padding == 0;
            std::printf("%s %s T=%3d: %d of %d rows differ from the same row at M=1; vs libref: %ld outside the bound (worst %.3g of it), %ld non-finite, %ld padding writes\n",
                        ok && vals ? "PASS" : "FAIL", L.name, T, rows_differ, T, off_bound, worst, nonfinite, padding);
            if (!ok || !vals) ++bad;
        }
    }
    std::printf("%d failure(s)\n", bad);
    dlclose(lib);
    return bad ? 1 : 0;
}
