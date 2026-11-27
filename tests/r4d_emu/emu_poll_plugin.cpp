/* emu_poll_plugin.cpp -- a kernel library with no kernels whose poll is the REAL one (libr4d's rad_kernel_poll_error, compiled from the HIP sources under the SIMT emulator),
 * for engine_e2e_test.
 *
 * On open it runs ONE launch libref would refuse, through the device launcher of one op family in CARD FORM (R4D_ERNIE_CARD_FORM=1: no host scan, the launch returns
 * RAD_OK and the kernel raises an event), chosen by RAD_EMU_POLL_FAMILY:
 *   hca     ernie_hca_compress, out_layout block, a completing token at a block offset other than the last (event bit 0x100)
 *   dsa     ernie_dsa_topk_tiled with a NaN key scale in the candidate range (event bit 0x1)
 *   dspark  ernie_hca_dspark_attn with a NaN sink (event bit 0x1)
 *   none    nothing (the control: the engine must run to the end)
 * The engine then polls through the exported rad_kernel_poll_error exactly as it does a card's library; this file only supplies the plugin identity and the injected launch. */
#include "rad_abi.h"
#include "rad_abi_layout.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
RadParam I(const char* k, int64_t v) { RadParam p{}; p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }
RadParam F(const char* k, double v) { RadParam p{}; p.key = k; p.kind = RAD_P_F64; p.dval = v; return p; }
RadParam S(const char* k, const char* v) { RadParam p{}; p.key = k; p.kind = RAD_P_STR; p.sval = v; return p; }
RadTensor Tn(void* d, uint32_t dt, std::vector<int64_t> sh) {
    RadTensor t{}; t.data = d; t.dtype = dt; t.rank = (uint32_t)sh.size();
    int64_t acc = 1;
    for (int i = (int)sh.size() - 1; i >= 0; --i) { t.shape[i] = sh[(size_t)i]; t.stride[i] = acc; acc *= sh[(size_t)i]; }
    return t;
}

/* ernie_hca_compress, block layout: ratio 4, head_dim 8, rope 4, out pool [3, 1, 4, 2]; position 3 completes block 0 but its out slot 4 is page 1 offset 0 (libref: RAD_E_SHAPE) */
int inject_hca() {
    std::vector<float> st(8 * 2 * 16);
    for (size_t i = 0; i < st.size(); ++i) st[i] = 0.25f * (int)((i * 3) % 11) - 1.0f;
    std::vector<int32_t> gs(8), pos = {3}, os = {4};
    for (int w = 0; w < 8; ++w) gs[(size_t)w] = w;
    std::vector<uint16_t> nw(8), out(3 * 4 * 2, 0xdead);
    for (int i = 0; i < 8; ++i) nw[(size_t)i] = rad_f32_to_bf16(0.75f + 0.0625f * i);
    RadTensor t[6] = {Tn(st.data(), RAD_F32, {8, 32}), Tn(gs.data(), RAD_I32, {1, 8}), Tn(pos.data(), RAD_I32, {1}), Tn(os.data(), RAD_I32, {1}), Tn(nw.data(), RAD_BF16, {8}),
                      Tn(out.data(), RAD_BF16, {3, 1, 4, 2})};
    RadParam p[] = {I("ratio", 4), I("head_dim", 8), I("rope_dim", 4), F("eps", 1e-6), F("theta", 10000.0), S("out_layout", "block")};
    RadArgs a{}; a.t = t; a.n_t = 6; a.p = p; a.n_p = 6;
    const int rc = r4d_rad_ernie_hca_compress(&a, nullptr);
    for (uint16_t v : out) if (v != 0xdead) return -2;   /* the refused row must store nothing */
    return rc;
}

/* ernie_dsa_topk_tiled: one row at position 39, window 2, topk 4, 10 blocks of 4 tokens; the key scale of token 13 is NaN (a non-finite score in [0, end)) */
int inject_dsa() {
    const int M = 1, H = 1, D = 4, block = 4, topk = 4, window = 2, pages = 11, B = 10, tile = 8, W = topk + window;
    std::vector<uint8_t> qf((size_t)M * H * D, 0), cache((size_t)pages * block * (D + 4), 0);
    qf[0] = 0x38; qf[1] = 0x38;
    std::vector<float> qs(1, 1.0f);
    std::vector<uint16_t> w(1, rad_f32_to_bf16(1.0f));
    const float one = 1.0f;
    for (int t = 0; t < pages * block; ++t) { uint8_t* r = cache.data() + (size_t)t * (D + 4); r[0] = 0x38; std::memcpy(r + D, &one, 4); }
    const float nan = std::nanf("");
    std::memcpy(cache.data() + (size_t)13 * (D + 4) + D, &nan, 4);
    std::vector<int32_t> bt1(B), bt2, req = {0}, pos = {39};
    for (int i = 0; i < B; ++i) bt1[(size_t)i] = i;
    bt2 = bt1;
    std::vector<float> plane((size_t)M * tile);
    std::vector<int32_t> ids((size_t)M * W, 0x5a5a5a5a), sl((size_t)M * W, 0x5a5a5a5a), kept((size_t)M * (2 * topk + 3), 0x5a5a5a5a);
    RadTensor ts[] = {Tn(qf.data(), RAD_F8E4M3, {M, H * D}), Tn(qs.data(), RAD_F32, {M, H}), Tn(w.data(), RAD_BF16, {M, H}), Tn(cache.data(), RAD_U8, {pages, 1, block, D + 4}),
                      Tn(bt1.data(), RAD_I32, {1, B}), Tn(req.data(), RAD_I32, {M}), Tn(pos.data(), RAD_I32, {M}), Tn(bt2.data(), RAD_I32, {1, B}), Tn(plane.data(), RAD_F32, {M, tile}),
                      Tn(ids.data(), RAD_I32, {M, W}), Tn(sl.data(), RAD_I32, {M, W}), Tn(kept.data(), RAD_I32, {(int64_t)kept.size()})};
    RadParam ps[] = {I("M", M), I("heads", H), I("head_dim", D), I("topk", topk), I("window", window), I("block_size", block), I("num_pages", pages), I("tokens", 40)};
    RadArgs a{}; a.t = ts; a.n_t = 12; a.p = ps; a.n_p = 8; a.kv_used_max_h = -1;
    const int rc = r4d_rad_ernie_dsa_topk_tiled(&a, nullptr);
    for (int32_t v : ids) if (v != 0x5a5a5a5a) return -2;
    return rc;
}

/* ernie_hca_dspark_attn: 2 sequences, 4 query rows, block 4, 3 pages, window 4; a NaN sink (libref: RAD_E_SHAPE) */
int inject_dspark() {
    const int D = 2;
    std::vector<uint16_t> q((size_t)4 * D), kv((size_t)3 * 4 * D), out((size_t)4 * D, 0xdead);
    for (size_t i = 0; i < q.size(); ++i) q[i] = rad_f32_to_bf16(0.0625f * (float)((int)(i % 11) - 5));
    for (size_t i = 0; i < kv.size(); ++i) kv[i] = rad_f32_to_bf16(0.125f * (float)((int)((i * 7) % 13) - 6));
    std::vector<int32_t> bt = {0, 1, 2, -1}, su = {5, 3}, cu = {0, 2, 4};
    std::vector<float> sink = {NAN};
    RadTensor t[7] = {Tn(q.data(), RAD_BF16, {4, D}), Tn(kv.data(), RAD_BF16, {3, 1, 4, D}), Tn(bt.data(), RAD_I32, {2, 2}), Tn(su.data(), RAD_I32, {2}), Tn(cu.data(), RAD_I32, {3}),
                      Tn(sink.data(), RAD_F32, {1}), Tn(out.data(), RAD_BF16, {4, D})};
    RadParam p[] = {I("M", 4), I("heads", 1), I("head_dim", D), I("window", 4), F("scale", 0.5)};
    RadArgs a{}; a.t = t; a.n_t = 7; a.p = p; a.n_p = 5;
    const int rc = r4d_rad_ernie_hca_dspark_attn(&a, nullptr);
    for (uint16_t v : out) if (v != 0xdead) return -2;
    return rc;
}
}  // namespace

extern "C" {

uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
RAD_PLUGIN_ABI_LAYOUT_EXPORT

const RadPluginInfo* rad_plugin_info(void) {
    static const RadPluginInfo info = { RAD_PLUGIN_KERNEL, "emupoll", "0.1.0", "the real libr4d poll over emulated kernels, for engine_e2e_test", "host" };
    return &info;
}
int rad_kernel_schema_count(void) { return 0; }
const RadOpSchema* rad_kernel_schema_at(int) { return nullptr; }
int rad_kernel_count(void) { return 0; }
const RadKernelInfo* rad_kernel_at(int) { return nullptr; }

/* rad_kernel_poll_error is the one libr4d exports (libr4d/r4d_ernie_ops.hip, linked in from the emulator objects): the engine polls it unchanged. */
int rad_plugin_open(void) {
    const char* f = std::getenv("RAD_EMU_POLL_FAMILY");
    if (!f || !*f || !std::strcmp(f, "none")) return RAD_OK;
    setenv("R4D_ERNIE_CARD_FORM", "1", 1);
    int rc = -1;
    if (!std::strcmp(f, "hca")) rc = inject_hca();
    else if (!std::strcmp(f, "dsa")) rc = inject_dsa();
    else if (!std::strcmp(f, "dspark")) rc = inject_dspark();
    else { std::fprintf(stderr, "emupoll: unknown family '%s'\n", f); return RAD_E_INVAL; }
    /* the launch must return OK as on a card (the refusal is the polled event), and store nothing */
    if (rc != RAD_OK) { std::fprintf(stderr, "emupoll: the injected '%s' launch returned %d, not RAD_OK\n", f, rc); return RAD_E_STATE; }
    return RAD_OK;
}

}  // extern "C"
