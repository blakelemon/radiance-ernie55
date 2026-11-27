/* tests/r4d_emu: ERNIE's small device operations (r4d_ernie_small.hip) on the host SIMT emulator. situ_mul is checked against a double-precision SiTU rounded once to bf16
 * (at most one bf16 ulp apart, 90% exact), pitches and the rows past M are untouched, and every refusal leaves y exactly as it was. `t file gu.bin M n beta lb y.bin` runs one
 * case from a file for the independent oracle of the acceptance test. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"
#include "ref_host.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static int run_situ(std::vector<uint16_t>& gu, std::vector<uint16_t>& y, int M, int n, long gp, long yp, double beta, double lb, const char* dt = nullptr, int rankgu = 2, int nparam = -1, int guw = -1, uint32_t gdt = RAD_BF16, uint32_t ydt = RAD_BF16, int mparam = -1) {
    RadTensor t[2]{};
    t[0].data = gu.data(); t[0].dtype = gdt; t[0].rank = rankgu; t[0].shape[0] = M; t[0].shape[1] = guw >= 0 ? guw : 2 * n; t[0].stride[0] = gp; t[0].stride[1] = 1;
    t[1].data = y.data();  t[1].dtype = ydt; t[1].rank = 2; t[1].shape[0] = M; t[1].shape[1] = n; t[1].stride[0] = yp; t[1].stride[1] = 1;
    RadParam p[5]{}; p[0] = {"M", RAD_P_INT, mparam >= 0 ? mparam : M}; p[1] = {"n", RAD_P_INT, nparam >= 0 ? nparam : n};
    p[2].key = "beta"; p[2].kind = RAD_P_F64; p[2].dval = beta; p[3].key = "linear_beta"; p[3].kind = RAD_P_F64; p[3].dval = lb;
    p[4].key = "dtype"; p[4].kind = RAD_P_STR; p[4].sval = dt ? dt : "bf16";
    RadArgs a{}; a.t = t; a.n_t = 2; a.p = p; a.n_p = 5;
    return r4d_rad_situ_mul(&a, nullptr);
}

static double situ_ref(double g, double u, double beta, double lb) { return beta * std::tanh(g / beta) * (1.0 / (1.0 + std::exp(-g))) * lb * std::tanh(u / lb); }


/* ---- ernie_router_topk: libref's arithmetic (f32 sigmoid, selection by score + bias with ties to the lower id, kept scores summed in pick order) written out here ---- */
static float sigf(float x) { return 1.0f / (1.0f + std::exp(-x)); }
static void host_router(const std::vector<float>& lg, long lp, const std::vector<float>& bias, const std::vector<float>& scale, int M, int ne, int k, int renorm,
                        std::vector<int32_t>& ids, long ip, std::vector<float>& w, long wp) {
    for (int m = 0; m < M; ++m) {
        std::vector<float> sc(ne), sel(ne);
        for (int e = 0; e < ne; ++e) { sc[e] = sigf(lg[(size_t)m * lp + e]) + sigf(lg[(size_t)m * lp + ne + e]); sel[e] = sc[e] + bias[e]; }
        std::vector<int> pk(k); float sum = 0.0f;
        for (int j = 0; j < k; ++j) {
            int best = -1;
            for (int e = 0; e < ne; ++e) if (sel[e] > -INFINITY && (best < 0 || sel[e] > sel[best])) best = e;
            pk[j] = best; sum += sc[best]; sel[best] = -INFINITY;
        }
        const float denom = renorm ? (sum > 1e-20f ? sum : 1e-20f) : 1.0f;
        for (int j = 0; j < k; ++j) { ids[(size_t)m * ip + j] = pk[j]; w[(size_t)m * wp + j] = (sc[pk[j]] / denom) * scale[pk[j]]; }
    }
}
static int run_router(std::vector<float>& lg, long lp, std::vector<float>& bias, std::vector<float>& scale, std::vector<int32_t>& ids, long ip, std::vector<float>& w, long wp,
                      int M, int ne, int k, int renorm, uint32_t ldt = RAD_F32, int wdt_override = 0) {
    RadTensor t[5]{};
    t[0].data = lg.data(); t[0].dtype = ldt; t[0].rank = 2; t[0].shape[0] = M; t[0].shape[1] = 2 * ne; t[0].stride[0] = lp; t[0].stride[1] = 1;
    t[1].data = bias.data(); t[1].dtype = RAD_F32; t[1].rank = 1; t[1].shape[0] = ne; t[1].stride[0] = 1;
    t[2].data = scale.data(); t[2].dtype = RAD_F32; t[2].rank = 1; t[2].shape[0] = ne; t[2].stride[0] = 1;
    t[3].data = ids.data(); t[3].dtype = RAD_I32; t[3].rank = 2; t[3].shape[0] = M; t[3].shape[1] = k; t[3].stride[0] = ip; t[3].stride[1] = 1;
    t[4].data = w.data(); t[4].dtype = wdt_override ? (uint32_t)wdt_override : RAD_F32; t[4].rank = 2; t[4].shape[0] = M; t[4].shape[1] = k; t[4].stride[0] = wp; t[4].stride[1] = 1;
    RadParam p[4]{}; p[0] = {"M", RAD_P_INT, M}; p[1] = {"experts", RAD_P_INT, ne}; p[2] = {"topk", RAD_P_INT, k}; p[3] = {"renormalize", RAD_P_INT, renorm};
    RadArgs a{}; a.t = t; a.n_t = 5; a.p = p; a.n_p = 4;
    return r4d_rad_ernie_router_topk(&a, nullptr);
}
static int router_main() {
    std::mt19937 rng(23);
    std::uniform_real_distribution<float> ud(-3.f, 3.f);
    int fails = 0, cases = 0;
    for (int ne : {16, 64, 512}) for (int k : {1, 3, 10, 16}) for (int M : {1, 5}) for (int renorm : {0, 1}) for (int pad : {0, 7}) for (int mode : {0, 1, 2}) {
        if (k > ne) continue;
        const int lp = 2 * ne + pad, ip = k + (pad ? 3 : 0), wp = k + (pad ? 5 : 0), R = M + 2;     /* independent pitches */
        std::vector<float> lg((size_t)R * lp), bias(ne), scale(ne);
        for (auto& v : lg) v = ud(rng);
        for (int e = 0; e < ne; ++e) { bias[e] = mode == 0 ? 0.f : ud(rng) * 0.1f; scale[e] = 0.5f + (float)(rng() % 100) / 100.f; }
        if (mode == 2) for (auto& v : lg) v = 0.25f;                      /* exact ties everywhere: with a zero bias the lower ids win */
        if (mode == 2) std::fill(bias.begin(), bias.end(), 0.f);
        std::vector<int32_t> ids((size_t)R * ip, 0x5a5a), idh = ids; std::vector<float> w((size_t)R * wp, 7.5f), wh = w;
        const int rc = run_router(lg, lp, bias, scale, ids, ip, w, wp, M, ne, k, renorm);
        host_router(lg, lp, bias, scale, M, ne, k, renorm, idh, ip, wh, wp);
        bool ok = rc == RAD_OK;
        for (int r = 0; r < R && ok; ++r) for (int c = 0; c < std::max(ip, wp) && ok; ++c) {
            if (c < ip) ok = ids[(size_t)r * ip + c] == idh[(size_t)r * ip + c];
            if (c < wp && ok) { uint32_t a, b; std::memcpy(&a, &w[(size_t)r * wp + c], 4); std::memcpy(&b, &wh[(size_t)r * wp + c], 4); ok = a == b; }   /* bit for bit */
        }
        if (mode == 2 && ok) for (int m = 0; m < M; ++m) for (int j = 0; j < k; ++j) ok = ok && ids[(size_t)m * ip + j] == j;
        ++cases;
        if (!ok) { ++fails; std::printf("FAIL router ne %d k %d M %d renorm %d pad %d mode %d\n", ne, k, M, renorm, pad, mode); }
    }
    /* refusals leave the outputs exactly as found */
    {
        const int M = 3, ne = 16, k = 4;
        std::vector<float> lg((size_t)M * 2 * ne, 0.3f), bias(ne, 0.f), scale(ne, 1.f), w((size_t)M * k, 9.f);
        std::vector<int32_t> ids((size_t)M * k, 77);
        const auto ids0 = ids; const auto w0 = w;
        auto expect = [&](int rc, int want, const char* what) { ++cases; if (rc != want || ids != ids0 || w != w0) { ++fails; std::printf("FAIL router refusal: %s (rc %d)\n", what, rc); } };
        auto bad = lg; bad[5] = NAN;                       expect(run_router(bad, 2 * ne, bias, scale, ids, k, w, k, M, ne, k, 1), RAD_E_SHAPE, "NaN logit");
        bad = lg; bad[2 * ne + 3] = INFINITY;              expect(run_router(bad, 2 * ne, bias, scale, ids, k, w, k, M, ne, k, 1), RAD_E_SHAPE, "inf logit in a later row");
        auto b2 = bias; b2[7] = NAN;                       expect(run_router(lg, 2 * ne, b2, scale, ids, k, w, k, M, ne, k, 1), RAD_E_SHAPE, "NaN bias");
        auto s2 = scale; s2[9] = -INFINITY;                expect(run_router(lg, 2 * ne, bias, s2, ids, k, w, k, M, ne, k, 1), RAD_E_SHAPE, "inf scale");
        expect(run_router(lg, 2 * ne, bias, scale, ids, k, w, k, M, ne, 0, 1), RAD_E_SHAPE, "topk 0");
        expect(run_router(lg, 2 * ne, bias, scale, ids, k, w, k, M, ne, ne + 1, 1), RAD_E_SHAPE, "topk above experts");
        expect(run_router(lg, 2 * ne, bias, scale, ids, k, w, k, M, ne, k, 2), RAD_E_SHAPE, "renormalize 2");
        expect(run_router(lg, 2 * ne, bias, scale, ids, k, w, k, M, ne, 33, 1), RAD_E_SHAPE, "topk above 32");
        {   /* 64 experts, topk 33: within the experts, above the kernel's 32 */
            const int ne2 = 64;
            std::vector<float> lg2((size_t)M * 2 * ne2, 0.3f), b2(ne2, 0.f), s2(ne2, 1.f), w2((size_t)M * 33, 9.f);
            std::vector<int32_t> i2((size_t)M * 33, 77); const auto i20 = i2, w20 = w2 == w2 ? i2 : i2; (void)w20;
            const auto wcopy = w2;
            ++cases;
            const int rc = run_router(lg2, 2 * ne2, b2, s2, i2, 33, w2, 33, M, ne2, 33, 1);
            if (rc != RAD_E_SHAPE || i2 != i20 || w2 != wcopy) { ++fails; std::printf("FAIL topk 33 accepted\n"); }
        }
        expect(run_router(lg, 2 * ne, bias, scale, ids, k, w, k, M, ne, k, 1, RAD_BF16), RAD_E_DTYPE, "bf16 logits");
        expect(run_router(lg, 2 * ne, bias, scale, ids, k, w, k, M, ne, k, 1, RAD_F32, RAD_BF16), RAD_E_DTYPE, "bf16 weights");
        expect(run_router(lg, 2 * ne - 1, bias, scale, ids, k, w, k, M, ne, k, 1), RAD_E_SHAPE, "logit pitch shorter than 2 * experts");
        expect(run_router(lg, 2 * ne, bias, scale, ids, k - 1, w, k, M, ne, k, 1), RAD_E_SHAPE, "ids pitch shorter than k");
        expect(run_router(lg, 2 * ne, bias, scale, ids, k, w, k - 1, M, ne, k, 1), RAD_E_SHAPE, "weights pitch shorter than k");
    }
    std::printf("ernie_router_topk: %d cases, %d failure(s)\n", cases, fails);
    return fails != 0;
}

int main(int argc, char** argv) {
    if (argc == 8 && !std::strcmp(argv[1], "file")) {
        const int M = std::atoi(argv[3]), n = std::atoi(argv[4]);
        std::vector<uint16_t> gu((size_t)M * 2 * n), y((size_t)M * n, 0);
        FILE* f = std::fopen(argv[2], "rb");
        if (!f || std::fread(gu.data(), 2, gu.size(), f) != gu.size()) return 3;
        std::fclose(f);
        if (run_situ(gu, y, M, n, 2 * n, n, std::atof(argv[5]), std::atof(argv[6])) != RAD_OK) return 4;
        f = std::fopen(argv[7], "wb");
        std::fwrite(y.data(), 2, y.size(), f);
        std::fclose(f);
        return 0;
    }
    if (argc == 12 && !std::strcmp(argv[1], "file") && !std::strcmp(argv[2], "router")) {
        const int M = std::atoi(argv[6]), ne = std::atoi(argv[7]), k = std::atoi(argv[8]), rn = std::atoi(argv[9]);
        auto rd = [](const char* f, size_t n) { std::vector<float> v(n); FILE* h = std::fopen(f, "rb"); if (!h || std::fread(v.data(), 4, n, h) != n) std::exit(3); std::fclose(h); return v; };
        auto lg = rd(argv[3], (size_t)M * 2 * ne); auto bias = rd(argv[4], ne); auto scale = rd(argv[5], ne);
        std::vector<int32_t> ids((size_t)M * k); std::vector<float> w((size_t)M * k);
        if (run_router(lg, 2 * ne, bias, scale, ids, k, w, k, M, ne, k, rn) != RAD_OK) return 4;
        FILE* f = std::fopen(argv[10], "wb"); std::fwrite(ids.data(), 4, ids.size(), f); std::fclose(f);
        f = std::fopen(argv[11], "wb"); std::fwrite(w.data(), 4, w.size(), f); std::fclose(f);
        return 0;
    }
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 3.f);
    int fails = 0, cases = 0;
    for (int n : {1, 7, 128, 255, 256, 257, 700}) for (int M : {1, 3, 16}) for (int pad : {0, 5}) for (double beta : {4.0, 1.0}) {
        const int padg = pad, pady = pad == 0 ? 0 : 11 - pad;    /* input and output pitches vary independently: (0, 0) and (5, 6) */
        const int gp = 2 * n + padg, yp = n + pady, R = M + 2;
        std::vector<uint16_t> gu((size_t)R * gp), y0((size_t)R * yp);
        for (size_t i = 0; i < gu.size(); ++i) { float v = nd(rng); if (i % 53 == 0) v *= 30; gu[i] = f2b(v); }
        for (int r = 0; r < M; ++r) for (int c = 0; c < n; c += 7) { gu[(size_t)r * gp + c] = f2b(-89.f - (float)(r % 2)); gu[(size_t)r * gp + n + c] = f2b((r % 2) ? 60.f : -60.f); }   /* the tail: expf(-g) overflows, the true value is normal */
        for (auto& v : y0) v = 0x7e7e;                                   /* a sentinel: untouched cells must keep it */
        auto y = y0;
        const double lb = beta == 4.0 ? 25.0 : 10.0;       /* both clips vary: a hard-coded linear_beta shows */
        const int rc = run_situ(gu, y, M, n, gp, yp, beta, lb);
        bool ok = rc == RAD_OK;
        int exact = 0, total = 0;
        for (int r = 0; r < R && ok; ++r) for (int c = 0; c < yp && ok; ++c) {
            const size_t o = (size_t)r * yp + c;
            if (r < M && c < n) {
                const double g = b2f(gu[(size_t)r * gp + c]), u = b2f(gu[(size_t)r * gp + n + c]);
                const uint16_t want = f2b((float)situ_ref(g, u, beta, lb));
                const int d = std::abs((int)want - (int)y[o]);
                
                /* sign-aware one-ulp bound; the only extra allowance is a flush: a DENORMAL expected value (|want| < 1.1754944e-38) may come back as a zero of either sign */
                const bool both_zero = ((want | y[o]) & 0x7fff) == 0;
                const bool tiny = (std::fabs(b2f(want)) < 1.1754944e-38f) && ((y[o] & 0x7fff) == 0);
                ok = both_zero || (!((want ^ y[o]) & 0x8000) && d <= 1) || tiny;
                exact += (y[o] == want); ++total;
                if (g < -88.f && std::fabs(u) > 50 && std::fabs(b2f(want)) > 1.2e-38f && d > 1) ok = false;      /* the tail keeps its value */
            } else ok = y[o] == y0[o];
        }
        ok = ok && total > 0 && exact * 10 >= total * 9;
        ++cases;
        if (!ok) { ++fails; std::printf("FAIL n %d M %d pad %d beta %g\n", n, M, pad, beta); }
    }
    /* refusals leave y exactly as found */
    {
        const int M = 4, n = 64;
        std::vector<uint16_t> gu((size_t)M * 2 * n, f2b(1.f)), y((size_t)M * n, 0x1234), y0 = y;
        struct Bad { double beta, lb; const char* dt; int want; } bad[] = {{0, 25, nullptr, RAD_E_INVAL}, {4, 0, nullptr, RAD_E_INVAL}, {-4, 25, nullptr, RAD_E_INVAL},
                                                                          {INFINITY, 25, nullptr, RAD_E_INVAL}, {4, NAN, nullptr, RAD_E_INVAL}, {4, INFINITY, nullptr, RAD_E_INVAL}, {4, -25, nullptr, RAD_E_INVAL}};
        for (auto& b : bad) {
            ++cases;
            const int rc = run_situ(gu, y, M, n, 2 * n, n, b.beta, b.lb, b.dt);
            if (rc != b.want || y != y0) { ++fails; std::printf("FAIL refusal beta %g lb %g rc %d\n", b.beta, b.lb, rc); }
        }
        ++cases;
        if (run_situ(gu, y, M, n, 2 * n, n, 4, 25, nullptr, 2, n + 1) != RAD_E_SHAPE || y != y0) { ++fails; std::printf("FAIL n parameter that does not match the tensors accepted\n"); }   /* gate_up is [M, 2n], n says n + 1 */
        ++cases;
        if (run_situ(gu, y, M, n, 2 * n, n, 4, 25, nullptr, 2, -1, 2 * n - 2) != RAD_E_SHAPE || y != y0) { ++fails; std::printf("FAIL gate_up narrower than 2n accepted\n"); }
        ++cases;
        if (run_situ(gu, y, M, n, 2 * n, n, 4, 25, nullptr, 2, n - 1) != RAD_E_SHAPE || y != y0) { ++fails; std::printf("FAIL narrower n accepted\n"); }
        ++cases;
        if (run_situ(gu, y, M, n, 2 * n, n, 4, 25, nullptr, 2, -1, -1, RAD_F32, RAD_BF16) != RAD_E_DTYPE || y != y0) { ++fails; std::printf("FAIL f32 gate_up accepted\n"); }
        ++cases;
        if (run_situ(gu, y, M, n, 2 * n, n, 4, 25, nullptr, 2, -1, -1, RAD_BF16, RAD_F32) != RAD_E_DTYPE || y != y0) { ++fails; std::printf("FAIL f32 y accepted\n"); }
        ++cases;
        if (run_situ(gu, y, M, n, 2 * n, n, 4, 25, nullptr, 1) != RAD_E_SHAPE || y != y0) { ++fails; std::printf("FAIL rank-1 gate_up accepted\n"); }
        ++cases;
        if (run_situ(gu, y, M, n, 2 * n, n, 4, 25, nullptr, 2, -1, -1, RAD_BF16, RAD_BF16, M + 1) != RAD_E_SHAPE || y != y0) { ++fails; std::printf("FAIL an M parameter that is not the tensors' rows accepted\n"); }
        ++cases;
        if (run_situ(gu, y, M, 0, 2 * n, n, 4, 25, nullptr, 2, 0) != RAD_E_SHAPE || y != y0) { ++fails; std::printf("FAIL n = 0 accepted\n"); }
        ++cases;
        if (run_situ(gu, y, M, n, 2 * n, n - 1, 4, 25) != RAD_E_SHAPE || y != y0) { ++fails; std::printf("FAIL short output pitch accepted\n"); }
        ++cases;
        if (run_situ(gu, y, M, n, n, n, 4, 25) != RAD_E_SHAPE || y != y0) { ++fails; std::printf("FAIL short pitch accepted\n"); }
    }
    std::printf("situ_mul: %d cases, %d failure(s)\n", cases, fails);
    return (fails != 0) | router_main();
}
