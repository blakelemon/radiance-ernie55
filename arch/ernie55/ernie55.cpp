/* ernie55 -- ERNIE 5.5 Flash (model_type ernie5_5_moe): architecture discovery.
 *
 * This plugin is the identity and the validated geometry of the family, and nothing else yet. It
 * claims the architecture id `ernie55moe` (HuggingFace's model_type without underscores, as
 * core/format/checkpoint.cpp derives it) unquantised; the pinned release's block-fp8 descriptor
 * reaches it through the family's single-plugin rule (core/build/startup.cpp, pick_arch).
 *
 * What it checks, from the container's free-form keys (the converted config.json), mirroring
 * tools/ernie55/audit_checkpoint.py:
 *   - model_type is ernie5_5_moe;
 *   - the target layer count (num_hidden_layers) and the draft layer count
 *     (num_nextn_predict_layers) are separate counts: the draft is never derived from the target;
 *   - compress_ratios has one entry per target layer followed by one per draft layer, each 128
 *     (HCA) or -2 (MLA-DSA), read entry by entry and never from a modulo period;
 *   - first_k_dense_replace dense feed-forward layers lead the target, the rest are routed;
 *   - the hidden width, head width, hyper-connection multiplicity and routed-MoE counts are
 *     positive and consistent (top-k within the routed experts).
 *
 * Weight coverage (story 2-5) is opt-in: `rad-convert --set ernie55_weights_only=1` declares the
 * dense and shared tensors (see declare_weights) with their checkpoint names, shapes and dtypes,
 * so the converter writes them and the container reader returns them, and declare then returns
 * RAD_OK with no ops and no logits: such a container cannot serve, and the engine refuses it at
 * startup. Without the key the behaviour is the one below and no weight is declared.
 *
 * Routed-MoE coverage (story 2-6) is a second opt-in on top of it, `--set ernie55_moe_weights=1`
 * (refused without ernie55_weights_only=1): for every routed layer it also declares both router
 * matrices, the correction bias, the learned per-expert scale, the two latent projections and the
 * latent norm, and every routed expert as three weights of its own (gate w1, up w3, down w2) at its
 * LOGICAL expert id, whichever shard the checkpoint keeps it in. See declare_routed.
 *
 * Feed-forward forward blocks (story 2-7) are a third opt-in, `--set ernie55_ffn_forward=1` (refused
 * without ernie55_weights_only=1): the container then also declares a graph that executes the dense
 * and shared SiTU-GLU feed-forward half of every layer on the host backend. It is an FFN-block probe,
 * not the model: no attention and no routed experts exist yet, so the program is token embedding,
 * the first-layer broadcast into hc streams, then per layer the ordinary mHC pre branch with the
 * fused RMSNorm, the dense (layer < first_k_dense_replace) or shared feed-forward, and the mHC post
 * update, ending in the hc streams of each sequence's mean, served as a pooled (embedding) output.
 * See declare_ffn and docs/ernie55/ffn-forward.md.
 *
 * The routed latent-MoE block (story 2-8) is a fourth opt-in, `--set ernie55_moe_forward=1` (refused
 * without weights_only, moe_weights and ffn_forward): each routed layer's program then also runs the
 * two f32 routers, ernie_router_topk, the latent down projection, the top-k experts at latent width,
 * the weighted combine, the latent norm AFTER the combine, the latent up projection and the shared
 * expert's addition. See declare_ffn and docs/ernie55/moe-forward.md.
 *
 * The target output (story 2-9) is a fifth opt-in, `--set ernie55_target_output=1` (refused without
 * weights_only and ffn_forward, and without moe_forward when the model has routed layers): the program
 * then ends in the language-model head instead of the pooled mean. On the rows the sampler asks for
 * (RadBatch::out_ids) it runs the learned mHC head collapse, the final RMSNorm, the vocabulary
 * projection and the learned MultiMax transform, and declares the result as the logits buffer, so
 * sampling receives the transformed scores. There is still no attention: each token's score row depends
 * on that token alone. See docs/ernie55/target-output.md.
 *
 * The MLA-DSA attention branch (story 3-2b) is a sixth opt-in, `--set ernie55_attention_forward=1` (refused
 * without weights_only and ffn_forward, and for any model with an HCA main layer: those layers are refused by
 * name until 3-4). Every layer then runs, before its feed-forward half, the attn-branch mHC pre (fused norm), the
 * q path (q_a, norm, q_b), kv_a to latent and rope tail, the bf16-table trailing rotary at absolute positions,
 * the latent cache write, the DSA indexer (wq_b, wk, weights_proj, ernie_idx_query/key/logits on its own FP8 key
 * cache), ernie_dsa_select, absorbed sparse attention with the sink, value expansion, the VHA mix (fold bf16),
 * the element sigmoid gate, wo_b and the attn-branch mHC post. Projections must be BF16 in the container (a
 * block-fp8 checkpoint is converted with `--quant 'blk.*.attn_<proj>.weight=cast:dtype=bf16'`, an exact
 * dequantisation); a block-fp8 projection is refused by name. See docs/ernie55/attention-forward.md.
 *
 * The HCA attention branch (story 3-4b) is a seventh opt-in, `--set ernie55_hca_forward=1` (refused without
 * ernie55_attention_forward=1; without it an HCA main layer is refused by name as in 3-2b). Every HCA layer then runs,
 * before its feed-forward half: the attn-branch mHC pre, the q path (down, norm, up, weightless per-head RMS and the
 * YaRN trailing rotary), the kv path (projection, kv_norm, rotary) into a window cache, the compressor (kv and score
 * projections, the fp32 state rows, group compression at (position + 1) % period == 0 into a compressed cache), the
 * union attention over the complete compressed rows and the 128-token window with the fp32 sink, the VHA mix, the
 * inverse rotary and grouped wo_a, the sigmoid gate on the wo_a output, wo_b and the mHC post. MLA-DSA and HCA layers
 * interleave per the schedule. See docs/ernie55/hca-forward.md.
 *
 * A valid config is summarised on stderr (one `ernie55: config` line, the loader-visible record of
 * the schedule) and then refused: no forward block exists yet, so declare returns
 * RAD_E_UNSUPPORTED naming that, rather than building a graph that would compute nothing.
 * Anything invalid is refused with the offending key named and RAD_E_INVAL.
 */
#include <arch/rad_arch.h>
#include <arch/rad_linear.h>
#include "ernie55_pins.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace ernie55 {
/* Row independence. A block-fp8 projection reads its activation two ways by the step's TOTAL row count: bf16 straight into the
 * fp8a16f32 GEMV when the step has at most eight rows, and quantised per 128 columns to E4M3 (fp8a8) above that. The two are
 * different arithmetic (the E4M3 grid costs about 2% relative error per output), so a sequence's decode or verify row changes when
 * a step-mate (another sequence's prefill chunk, another sequence's accepted drafts) pushes the step over eight rows. With
 * speculation the row count depends on how many drafts were accepted, so a sequence's logits depend on its neighbours' content.
 * ERNIE_A16_MAX selects the largest step (in rows) that may use the bf16 path. The default follows the band decision of resolve_row_bands(): 0 with speculation (the drafter in the file or a depth requested), 8 without.
 * 0 sends every row down the fp8a8 path, whose scales are per (row, 128 columns) and involve no other row: row independent.
 * Larger values are clamped to 8: the fp8a16f32 ops are declared for M <= 8 (op_a16 below) and libr4d refuses M > 8 for them
 * (cFp8A16F32Scale, dense_a16), so an all-bf16 mode cannot serve prefill-sized steps. */
/* The bands are resolved once at model build (declare(), resolve_row_bands): speculation on takes the row-identical bands, off the earlier ones. */
static bool g_row_identity = true;          /* the decision of the last declare(); true before any (row identity is the default of the pure functions) */
static int64_t g_a16_max = pin::kA16Default;
static int64_t g_vha_dec_max = pin::kVhaDecMax;
static inline int64_t a16_max_rows() { return g_a16_max; }

/* RADIANCE_ERNIE55_DSPARK_KV_WINDOW=1: the drafter's KV group (kv_dspark) is a sliding window of the drafter's attention window instead of a full-context group. The drafter's attention reads only [anchor - window + 1, total) through seqused and the block table, which the core rebases for a window group, so its output is bit-identical. Default off. */
static bool dspark_kv_window_from_env() { const char* v = getenv("RADIANCE_ERNIE55_DSPARK_KV_WINDOW"); return v && v[0] == '1' && v[1] == '\0'; }

/* Diagnostic pin, neutral unless set: ERNIE_ROUTER_SPLIT=1 routes a one-row step through ernie_router_topk + moe_scatter, the ops
 * every other row count uses, instead of the fused ernie_router_topk_scatter, so the router is not a source of M=1 versus M>1
 * difference. */
static inline bool router_split() {
    static const bool v = pin::router_split(std::getenv("ERNIE_ROUTER_SPLIT"));
    return v;
}

/* Diagnostic pin, neutral unless set: ERNIE_VHA_DEC_MAX is the largest step (rows) that mixes the heads by all-gather + ernie_vha_mix
 * when the VHA rank exceeds the heads per rank (default 0 with speculation, 8 without: resolve_row_bands()); above it the step reduces the rank-wide projection
 * (ernie_vha_proj, exact all_reduce, ernie_vha_apply). 0 sends every step down the reduce path, so a row's heads are mixed the same
 * way whatever the step size. */
static inline int64_t vha_dec_cap() { return g_vha_dec_max; }

/* The declared, logged row-band decision (see ernie55_pins.h): called first in declare(), before any use of a band. */
static void resolve_row_bands(bool has_drafter, int max_spec) {
    /* declare() runs on one thread a card (engine_bringup): the decision, the setenv of the libr4d band variables and the log happen ONCE a process under a mutex, so no
     * thread calls getenv while another calls setenv here. The bands are process-wide (libr4d reads its two variables once, the first launch); a later declare in the same
     * process (a second model, rad-tune, a test) whose decision differs is told so loudly and keeps the first decision, so the plugin's bands and libr4d's stay consistent. */
    static std::mutex mu;
    static int decided = -1;   /* -1 none yet, else the first decision (0 previous bands, 1 row identity) */
    std::lock_guard<std::mutex> lk(mu);
    const char* ov = std::getenv("ERNIE_ROW_IDENTITY");
    const bool want = pin::row_identity_on(ov, has_drafter, max_spec);
    if (decided >= 0) {
        if ((decided == 1) != want)
            fprintf(stderr, "radiance: ernie55: WARNING: this declare would choose the %s bands (drafter in file %d, max_spec %d) but this process already runs the %s bands (the decision "
                            "of its first model: the libr4d band variables are read once a process); keeping the first. Serve one ERNIE model a process, or set ERNIE_ROW_IDENTITY\n",
                    want ? "row-identical" : "previous", has_drafter ? 1 : 0, max_spec, decided == 1 ? "row-identical" : "previous");
        return;
    }
    decided = want ? 1 : 0;
    g_row_identity = want;
    g_a16_max = pin::a16_max(std::getenv("ERNIE_A16_MAX"), g_row_identity);
    g_vha_dec_max = pin::vha_dec_cap(std::getenv("ERNIE_VHA_DEC_MAX"), g_row_identity);
    pin::export_process_pins(g_row_identity);   /* the mHC decode rows only: the fp8a8 matvec bound is a declared op parameter */
    fprintf(stderr, "radiance: ernie55: row bands: %s (%s; drafter in file %d, max_spec %d): ERNIE_A16_MAX=%lld ERNIE_VHA_DEC_MAX=%lld fp8a8 mvm_max=%d (declared per op) "
            "R4D_MHC_DECODE_ROWS=%d R4D_HCA_DECODE_MAX=%d R4D_MLA_COOP_MAX=%d; pair GEMM, wo_a, D512 union attention and sparse MLA are row-identical in both modes\n",
            g_row_identity ? "row-identical (speculation)" : "previous (no speculation)",
            ov && *ov ? "ERNIE_ROW_IDENTITY override" : "automatic", has_drafter ? 1 : 0, max_spec, (long long)g_a16_max, (long long)g_vha_dec_max,
            pin::f8a8_mvm_declared(std::getenv("R4D_F8A8_MVM_MAX"), g_row_identity), pin::env_int("R4D_MHC_DECODE_ROWS", -1),
            pin::env_int("R4D_HCA_DECODE_MAX", pin::kHcaDecodeMaxLog), pin::env_int("R4D_MLA_COOP_MAX", pin::kMlaCoopMaxLog));
}

/* The fp8a8 matvec bound this model declares on each of its fp8a8f32 ops (op parameter "mvm_max"; see ernie55_pins.h): 0 in the row-identical bands, 8 in the
 * previous bands, or the R4D_F8A8_MVM_MAX diagnostic override. Read at model build, after resolve_row_bands(). */
static inline int f8a8_mvm_declared(bool* overridden = nullptr) {
    return pin::f8a8_mvm_declared(std::getenv("R4D_F8A8_MVM_MAX"), g_row_identity, overridden);
}

static constexpr const char* kModelType = "ernie5_5_moe";
static constexpr int kHca = 128;      /* compress_ratios entry for hybrid compressed attention */
static constexpr int kMlaDsa = -2;    /* compress_ratios entry for MLA with the sparse indexer */

struct Config {
    int64_t main_layers = 0, draft_layers = 0, first_dense = 0;
    int64_t hidden = 0, head_dim = 0, hc_mult = 0;
    int64_t experts = 0, shared = 0, top_k = 0, latent = 0;
    int64_t vocab = 0, dense_ff = 0, moe_ff = 0;   /* weight map only (read_weight_dims) */
    std::vector<int> modes;           /* main_layers entries, then draft_layers entries */
};

/* strtoll skips leading whitespace and takes '+'; neither is a whole decimal integer here. */
static bool starts_integer(const char* s) { return *s == '-' || (*s >= '0' && *s <= '9'); }

/* A whole decimal integer, no sign other than '-', no float or trailing text. */
static bool parse_int(const char* s, int64_t* out) {
    if (!s || !starts_integer(s)) return false;
    char* end = nullptr;
    errno = 0;
    const long long v = strtoll(s, &end, 10);
    if (errno || *end) return false;
    *out = v;
    return true;
}

static bool need(const RadModelMeta* m, const char* key, int64_t lowest, int64_t* out,
                 std::string* why) {
    const char* v = rad_meta_gets(m, key, nullptr);
    if (!v) { *why = std::string("config.json is missing ") + key; return false; }
    int64_t x = 0;
    if (!parse_int(v, &x)) {
        *why = std::string(key) + " must be an integer, got '" + v + "'";
        return false;
    }
    if (x < lowest) {
        *why = std::string(key) + " must be an integer >= " + std::to_string(lowest) + ", got " +
               std::to_string(x);
        return false;
    }
    *out = x;
    return true;
}

/* Fill *c from the container's metadata or say why not. */
static int read_config(const RadModelMeta* m, Config* c, std::string* why) {
    const char* mt = rad_meta_gets(m, "model_type", nullptr);
    if (!mt || strcmp(mt, kModelType) != 0) {
        *why = std::string("model_type must be \"") + kModelType + "\", got " +
               (mt ? std::string("\"") + mt + "\"" : std::string("none"));
        return RAD_E_UNSUPPORTED;
    }
    if (!need(m, "num_hidden_layers", 1, &c->main_layers, why) ||
        !need(m, "num_nextn_predict_layers", 0, &c->draft_layers, why) ||
        !need(m, "hidden_size", 1, &c->hidden, why) ||
        !need(m, "head_dim", 1, &c->head_dim, why) ||
        !need(m, "hc_mult", 1, &c->hc_mult, why) ||
        !need(m, "moe_latent_size", 1, &c->latent, why) ||
        !need(m, "n_routed_experts", 1, &c->experts, why) ||
        !need(m, "n_shared_experts", 1, &c->shared, why) ||
        !need(m, "num_experts_per_tok", 1, &c->top_k, why) ||
        !need(m, "first_k_dense_replace", 0, &c->first_dense, why))
        return RAD_E_INVAL;
    if (c->top_k > c->experts) {
        *why = "num_experts_per_tok (" + std::to_string(c->top_k) +
               ") is greater than n_routed_experts (" + std::to_string(c->experts) + ")";
        return RAD_E_INVAL;
    }
    if (c->first_dense > c->main_layers) {
        *why = "first_k_dense_replace (" + std::to_string(c->first_dense) +
               ") is greater than num_hidden_layers (" + std::to_string(c->main_layers) + ")";
        return RAD_E_INVAL;
    }
    const char* sched = rad_meta_gets(m, "compress_ratios", nullptr);
    if (!sched) { *why = "config.json is missing compress_ratios"; return RAD_E_INVAL; }
    /* The converter flattens an array of scalars to one space-separated string. */
    for (const char* p = sched; *p;) {
        while (*p == ' ') ++p;
        if (!*p) break;
        char* end = nullptr;
        errno = 0;
        const long long v = strtoll(p, &end, 10);
        if (!starts_integer(p) || end == p || errno || (*end && *end != ' ')) {
            *why = "compress_ratios[" + std::to_string(c->modes.size()) +
                   "] must be an integer attention mode";
            return RAD_E_INVAL;
        }
        if (v != kHca && v != kMlaDsa) {
            *why = "compress_ratios[" + std::to_string(c->modes.size()) + "] is " +
                   std::to_string(v) + "; this supported variant allows only 128 (HCA) and -2 "
                   "(MLA-DSA)";
            return RAD_E_INVAL;
        }
        c->modes.push_back((int)v);
        p = end;
    }
    int64_t want = 0;
    if (__builtin_add_overflow(c->main_layers, c->draft_layers, &want)) {
        *why = "num_hidden_layers (" + std::to_string(c->main_layers) +
               ") plus num_nextn_predict_layers (" + std::to_string(c->draft_layers) +
               ") overflows a 64-bit layer count";
        return RAD_E_INVAL;
    }
    if ((int64_t)c->modes.size() != want) {
        *why = "compress_ratios has " + std::to_string(c->modes.size()) +
               " entries, but num_hidden_layers (" + std::to_string(c->main_layers) +
               ") plus num_nextn_predict_layers (" + std::to_string(c->draft_layers) + ") is " +
               std::to_string(want);
        return RAD_E_INVAL;
    }
    return RAD_OK;
}

/* One letter a layer: H (HCA) or M (MLA-DSA), main layers then, after a '|', the draft layers. */
static std::string schedule_text(const Config& c) {
    std::string s;
    for (size_t i = 0; i < c.modes.size(); ++i) {
        if ((int64_t)i == c.main_layers) s += '|';
        s += c.modes[i] == kHca ? 'H' : 'M';
    }
    return s;
}

/* ================================================================== dense and shared weights */
/* Checkpoint tensor names are the pinned release's (model.safetensors.index.json) and shapes are
 * its shard headers' and the Baidu model's (vLLM-ERNIE b8414da, nvidia/model.py, moe.py). Declared
 * names follow the blk.N convention of the other plugins. Out of scope here, so neither declared
 * nor required: attention, routers, latent projections, routed experts and the draft layer. */
using rad::arch::Names;

/* The widths the weight map needs beyond the validated geometry. */
static int read_weight_dims(const RadModelMeta* m, Config* c, std::string* why) {
    if (!need(m, "vocab_size", 1, &c->vocab, why)) return RAD_E_INVAL;
    if (c->first_dense > 0 && !need(m, "intermediate_size", 1, &c->dense_ff, why))
        return RAD_E_INVAL;
    if (c->first_dense < c->main_layers && !need(m, "moe_intermediate_size", 1, &c->moe_ff, why))
        return RAD_E_INVAL;
    return RAD_OK;
}

static bool mul(int64_t a, int64_t b, int64_t* out) { return !__builtin_mul_overflow(a, b, out); }

static std::string dims_text(const int64_t* d, uint32_t rank) {
    std::string t = "[";
    for (uint32_t i = 0; i < rank; ++i) t += (i ? ", " : "") + std::to_string((long long)d[i]);
    return t + "]";
}

/* The shape and dtype the checkpoint (or, at load, the container) holds `name` at, against what
 * the model needs. A name the model does not hold is left to the converter, which names the
 * missing tensor. `fp8_ok`: block-fp8 codes with their scale plane stand in for BF16. The core
 * would accept a wrong extent or a narrowing dtype silently, so this is the diagnostic. */
static int expect(RadBuilder* b, const char* name, const char* src, uint32_t stored,
                  std::initializer_list<int64_t> shape, bool fp8_ok = false,
                  const char* note = nullptr, bool mxfp4_ok = false) {
    RadEncoding enc{};
    int64_t got[RAD_MAX_RANK] = {0};
    uint32_t rank = 0;
    const int rc = rad_weight_encoding(b, name, &enc, got, &rank);
    if (rc != RAD_OK) return RAD_OK;   /* absent or not plannable: the converter names it */
    int64_t want[RAD_MAX_RANK] = {0};
    uint32_t n = 0;
    for (int64_t d : shape) want[n++] = d;
    bool ok = rank == n && std::equal(got, got + n, want);
    const uint32_t have = enc.n_planes > 0 ? enc.plane[0].dtype : (uint32_t)RAD_BF16;
    if (ok) {
        const bool plain_ok = enc.n_planes == 1 && have == stored;
        const bool fp8 = fp8_ok && enc.n_planes == 2 && have == RAD_F8E4M3;
        const bool mx = mxfp4_ok && rad::arch::linear_is_mxfp4(enc);
        ok = plain_ok || fp8 || mx;
    }
    if (ok) return RAD_OK;
    fprintf(stderr,
            "radiance: ernie55: checkpoint tensor '%s' (weight '%s') is %s %s, but this model "
            "needs %s %s%s%s%s\n",
            src, name, rad_dtype_name(have), dims_text(got, rank).c_str(), rad_dtype_name(stored),
            dims_text(want, n).c_str(),
            mxfp4_ok ? " (or block-fp8 or MXFP4 of that shape)"
                     : fp8_ok ? " (or block-fp8 of that shape)" : "",
            note ? " " : "", note ? note : "");
    return RAD_E_SHAPE;
}

/* A weight stored as the checkpoint holds it, read at `declared` (an exact widening of `stored`
 * is the loader's). `src` is declared first as a name-map entry. `note` ends a shape diagnostic. */
static int plain(RadBuilder* b, const char* name, const char* src, uint32_t stored,
                 uint32_t declared, std::initializer_list<int64_t> shape, int access, int layer,
                 const char* note = nullptr, rad_weight* out = nullptr,
                 int shard = RAD_SHARD_NONE, int world = 1) {
    RAD_ARCH_TRY(rad::arch::map_copy(b, name, src));
    RAD_ARCH_TRY(expect(b, name, src, stored, shape, false, note));
    const RadWeightGroup grp = layer < 0 ? rad::arch::grp_model() : rad::arch::grp_layer(layer);
    rad_weight w = 0;
    if (shard != RAD_SHARD_NONE && world > 1) {
        int64_t d[4] = {0};
        int n = 0;
        for (int64_t v : shape) d[n++] = v;
        if (shard == RAD_SHARD_ROW) {
            if (n < 1 || d[0] % world) return RAD_E_SHAPE;
            d[0] /= world;
        } else if (shard == RAD_SHARD_COL) {
            if (n < 2 || d[n - 1] % world) return RAD_E_SHAPE;
            d[n - 1] /= world;
        }
        if (n == 1) w = rad::arch::decl_w(b, name, declared, {d[0]}, access, shard, grp);
        else if (n == 2) w = rad::arch::decl_w(b, name, declared, {d[0], d[1]}, access, shard, grp);
        else return RAD_E_INVAL;
    } else {
        w = rad::arch::decl_w(b, name, declared, shape, access, RAD_SHARD_NONE, grp);
    }
    if (out) *out = w;
    return RAD_OK;
}

/* A linear: BF16 as one weight, or the block-fp8 the release ships as a codes view and a scale
 * view of one stored weight (the layout of arch/common/rad_linear.h's LinearQ). */
static int linear(RadBuilder* b, const char* name, const char* src, int64_t rows, int64_t cols,
                  const RadWeightGroup& g, Names& nm, int access = RAD_ACCESS_PER_TOKEN,
                  const char* note = nullptr, rad_weight* out = nullptr,
                  int shard = RAD_SHARD_NONE, int world = 1) {
    using namespace rad::arch;
    RAD_ARCH_TRY(map_copy(b, name, src));
    RAD_ARCH_TRY(expect(b, name, src, RAD_BF16, {rows, cols}, true, note, g.expert >= 0));
    const int sh = world > 1 ? shard : RAD_SHARD_NONE;
    if ((sh == RAD_SHARD_ROW && rows % world) || (sh == RAD_SHARD_COL && cols % world)) return RAD_E_SHAPE;
    const int64_t lrows = (sh == RAD_SHARD_ROW) ? rows / world : rows;
    const int64_t lcols = (sh == RAD_SHARD_COL) ? cols / world : cols;
    RadEncoding enc{};
    if (!weight_enc(b, name, &enc) || enc.n_planes != 2) {
        const rad_weight w = decl_w(b, name, RAD_BF16, {lrows, lcols}, access, sh, g);
        if (out) *out = w;
        return RAD_OK;
    }
    if (linear_is_mxfp4(enc)) {
        if (lcols % RAD_MX_GROUP) {
            fprintf(stderr, "radiance: ernie55: MXFP4 weight '%s' is %lldx%lld per rank, not whole 1x%d "
                            "blocks\n", src, (long long)lrows, (long long)lcols, (int)RAD_MX_GROUP);
            return RAD_E_SHAPE;
        }
        decl_view(b, name, nullptr, "codes", RAD_FP4E2M1, {lrows, lcols}, access, sh, g);
        const std::string base = std::string(name).substr(0, std::strlen(name) - 7);   /* ".weight" */
        decl_view(b, nm.f("%s.scale", base.c_str()), name, "scale", RAD_E8M0,
                  {lrows, lcols / RAD_MX_GROUP}, access, sh, g);
        return RAD_OK;
    }
    constexpr int64_t kBlock = 128;     /* quantization_config.weight_block_size, pinned */
    if (lrows % kBlock || lcols % kBlock) {
        fprintf(stderr, "radiance: ernie55: block-fp8 weight '%s' is %lldx%lld per rank, not whole %lld "
                        "blocks\n", src, (long long)lrows, (long long)lcols, (long long)kBlock);
        return RAD_E_SHAPE;
    }
    decl_view(b, name, nullptr, "codes", RAD_F8E4M3, {lrows, lcols}, access, sh, g);
    const std::string base = std::string(name).substr(0, std::strlen(name) - 7);   /* ".weight" */
    decl_view(b, nm.f("%s.scale", base.c_str()), name, "scale", enc.plane[1].dtype,
              {lrows / kBlock, lcols / kBlock}, access, sh, g);
    return RAD_OK;
}

struct QWeight {
    rad_weight w = 0, s = 0;
    uint32_t scale_dt = 0;
    bool fp8 = false;
    bool valid() const { return w != 0 && (!fp8 || s != 0); }
};
struct QOp {
    rad_op op = 0, op_a16 = 0;
    QWeight w;
};
struct QAct {
    rad_buf q = 0, s = 0;
    int64_t K = 0;
    bool valid() const { return q != 0 && s != 0; }
};

/* Like linear(), supporting fused (concat) sources and returning QWeight for BF16 or block-FP8. */
static int linear_q(RadBuilder* b, const char* name, std::initializer_list<const char*> srcs,
                    int64_t rows, int64_t cols, const RadWeightGroup& g, rad::arch::Names& nm,
                    QWeight* qout, int access = RAD_ACCESS_PER_TOKEN, int shard = RAD_SHARD_NONE,
                    const int64_t* row_parts = nullptr, int n_parts = 0, const char* note = nullptr,
                    int world = 1) {
    using namespace rad::arch;
    const char* src0 = *srcs.begin();
    if (srcs.size() == 1) {
        RAD_ARCH_TRY(map_copy(b, name, src0));
    } else {
        RAD_ARCH_TRY(map_concat(b, name, srcs, 0));
    }
    RAD_ARCH_TRY(expect(b, name, src0, RAD_BF16, {rows, cols}, true, note, g.expert >= 0));
    const int sh = world > 1 ? shard : RAD_SHARD_NONE;
    if ((sh == RAD_SHARD_ROW && rows % world) || (sh == RAD_SHARD_COL && cols % world)) return RAD_E_SHAPE;
    const int64_t lrows = (sh == RAD_SHARD_ROW) ? rows / world : rows;
    const int64_t lcols = (sh == RAD_SHARD_COL) ? cols / world : cols;
    int64_t lparts[4] = {0};
    if (n_parts > 0) {
        if (n_parts > 4) return RAD_E_INVAL;
        for (int i = 0; i < n_parts; ++i) {
            if (sh == RAD_SHARD_ROW && (row_parts[i] % world)) return RAD_E_SHAPE;
            lparts[i] = (sh == RAD_SHARD_ROW) ? row_parts[i] / world : row_parts[i];
        }
    }
    const int64_t* rp = n_parts > 0 ? lparts : nullptr;
    RadEncoding enc{};
    if (!weight_enc(b, name, &enc) || enc.n_planes != 2) {
        const rad_weight w = decl_w_n(b, name, RAD_BF16, {lrows, lcols}, access, sh, g, 0, rp, n_parts);
        if (!w) return RAD_E_INVAL;
        if (qout) { qout->w = w; qout->s = 0; qout->scale_dt = 0; qout->fp8 = false; }
        return RAD_OK;
    }
    constexpr int64_t kBlock = 128;
    if (lrows % kBlock || lcols % kBlock) {
        fprintf(stderr, "radiance: ernie55: block-fp8 weight '%s' is %lldx%lld per rank, not whole %lld "
                        "blocks\n", src0, (long long)lrows, (long long)lcols, (long long)kBlock);
        return RAD_E_SHAPE;
    }
    int64_t s_parts[4] = {0};
    if (n_parts > 0) {
        for (int i = 0; i < n_parts; ++i) {
            if (lparts[i] % kBlock) return RAD_E_SHAPE;
            s_parts[i] = lparts[i] / kBlock;
        }
    }
    const rad_weight wc = decl_view(b, name, nullptr, "codes", RAD_F8E4M3, {lrows, lcols}, access,
                                    sh, g, rp, n_parts);
    const std::string base = std::string(name).substr(0, std::strlen(name) - 7);
    const rad_weight ws = decl_view(b, nm.f("%s.scale", base.c_str()), name, "scale",
                                    enc.plane[1].dtype, {lrows / kBlock, lcols / kBlock}, access,
                                    sh, g, n_parts ? s_parts : nullptr, n_parts);
    if (!wc || !ws) return RAD_E_INVAL;
    if (qout) { qout->w = wc; qout->s = ws; qout->scale_dt = enc.plane[1].dtype; qout->fp8 = true; }
    return RAD_OK;
}


/* One mHC block: the checkpoint keeps the mixer as [hc*hidden, mix] (in, out); the ernie_mhc ops
 * read [mix, hc*hidden]. The three scalar gains join as scale[pre, post, res]. */
struct MhcW { rad_weight fn = 0, base = 0, scale = 0; };

static int mhc(RadBuilder* b, Names& nm, const std::string& prefix, const char* dn, int layer,
               int64_t mix, int64_t width, MhcW* out = nullptr) {
    using namespace rad::arch;
    const char* fn = nm.f("%s.fn", dn);
    const char* fsrc = nm.ckpt("%s.mapping_proj.weight", prefix.c_str());
    RAD_ARCH_TRY(map_transpose(b, fn, fsrc));
    RAD_ARCH_TRY(expect(b, fn, fsrc, RAD_F32, {mix, width}));
    const rad_weight wfn = decl_w(b, fn, RAD_F32, {mix, width}, RAD_ACCESS_PER_TOKEN,
                                  RAD_SHARD_NONE, grp_layer(layer));
    if (out) out->fn = wfn;
    RAD_ARCH_TRY(plain(b, nm.f("%s.base", dn), nm.ckpt("%s.bias", prefix.c_str()), RAD_F32, RAD_F32, {mix},
                       RAD_ACCESS_PER_TOKEN, layer, nullptr, out ? &out->base : nullptr));
    const char* scale = nm.f("%s.scale", dn);
    RAD_ARCH_TRY(map_concat(b, scale,
                            {nm.ckpt("%s.alpha_pre", prefix.c_str()),
                             nm.ckpt("%s.alpha_post", prefix.c_str()),
                             nm.ckpt("%s.alpha_res", prefix.c_str())}, 0));
    RAD_ARCH_TRY(expect(b, scale, nm.ckpt("%s.alpha_{pre,post,res}", prefix.c_str()), RAD_F32,
                        {3}));
    const rad_weight ws = decl_w(b, scale, RAD_F32, {3}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                                 grp_layer(layer));
    if (out) out->scale = ws;
    return RAD_OK;
}

static bool need_real(const RadModelMeta* m, const char* key, bool zero_ok, double* out, std::string* why);

/* ================================================================== MLA-DSA attention (story 3-2b) */
/* Geometry from the release's config (hybrid_mla_*, index_*, sliding_window, rope_theta). */
struct AttnConfig {
    int64_t heads = 0, qlora = 0, lat = 0, nope = 0, rope = 0, vdim = 0, ih = 0, id = 0, topk = 0, window = 0,
            block = 64;
    double theta = 0;
    /* ernie55_idx_tile (0 = off, the default): the DSA score plane holds `idx_tile` context columns instead of max-model-len and the context is
     * scored and cut to the top-k in tiles (ernie_dsa_topk_tiled, docs/ernie55/score-tiling.md). The selection is bit-identical to the untiled one. */
    int64_t idx_tile = 0;
};
static constexpr const char* kRopeTable = "bf16";   /* DSA layers: default RotaryEmbedding, cos/sin cast to bf16 (m08) */
static constexpr const char* kVhaFold = "bf16";     /* m18: at tp = 1 the folded matrix is bf16 */

static int read_attn_config(const RadModelMeta* m, AttnConfig* a, std::string* why) {
    if (!need(m, "hybrid_mla_num_attention_heads", 1, &a->heads, why) ||
        !need(m, "hybrid_mla_q_lora_rank", 1, &a->qlora, why) ||
        !need(m, "hybrid_mla_kv_lora_rank", 1, &a->lat, why) ||
        !need(m, "hybrid_mla_qk_nope_head_dim", 1, &a->nope, why) ||
        !need(m, "hybrid_mla_qk_rope_head_dim", 2, &a->rope, why) ||
        !need(m, "hybrid_mla_v_head_dim", 1, &a->vdim, why) || !need(m, "index_n_heads", 1, &a->ih, why) ||
        !need(m, "index_head_dim", 4, &a->id, why) || !need(m, "index_topk", 1, &a->topk, why) ||
        !need(m, "sliding_window", 1, &a->window, why) || !need_real(m, "rope_theta", false, &a->theta, why))
        return RAD_E_INVAL;
    const int64_t cap = int64_t(1) << 20;
    if (a->heads > 4096 || a->qlora > cap || a->lat > cap || a->nope > cap || a->rope > cap || a->vdim > cap ||
        a->ih > 256 || a->id > 256 || a->topk > cap || a->window > cap) {
        *why = "an attention extent exceeds the supported bound (heads 4096, index heads 256, index_head_dim 256, 2^20)";
        return RAD_E_INVAL;
    }
    if ((a->rope & 1) || ((a->lat + a->rope) & 1) || (a->id & (a->id - 1)) || a->rope > a->id) {
        *why = "attention geometry: qk_rope_head_dim must be even and at most index_head_dim, kv_lora_rank + qk_rope_head_dim "
               "even, index_head_dim a power of two";
        return RAD_E_INVAL;
    }
    const int64_t blk = rad_meta_geti(m, "ernie55_attention_block", 64);
    if (blk < 1 || blk > 4096) {
        *why = "ernie55_attention_block must be 1..4096, got " + std::to_string((long long)blk);
        return RAD_E_INVAL;
    }
    a->block = blk;
    int64_t tile = rad_meta_geti(m, "ernie55_idx_tile", 0);
    /* A serve-time override of the container key, for A/B runs on one container: RADIANCE_ERNIE55_IDX_TILE=<columns>, 0 = untiled. */
    if (const char* ev = getenv("RADIANCE_ERNIE55_IDX_TILE")) if (*ev) tile = (int64_t)strtoll(ev, nullptr, 10);
    if (tile < 0 || tile > (int64_t(1) << 24)) {
        *why = "ernie55_idx_tile must be 0 (untiled) or 1..2^24 context columns, got " + std::to_string((long long)tile);
        return RAD_E_INVAL;
    }
    a->idx_tile = tile;
    return RAD_OK;
}

/* One layer's attention weights. A block-fp8 projection has no single handle (0): declare_ffn refuses it by name. */
struct AttnHandles {
    rad_weight qa = 0, qa_norm = 0, qb = 0, kva = 0, kv_norm = 0, kvb = 0, gate = 0, o = 0, u = 0, v = 0, sink = 0,
               iq = 0, ik = 0, iw = 0, ik_g = 0, ik_b = 0;
    QWeight q_qa, q_qb, q_kva, q_gate, q_o, q_iq;
    int64_t rank = 0;
    bool present = false;
};

/* ================================================================== HCA attention (story 3-4b) */
/* Geometry from the release's config. `period` is the compression period (the schedule says 128; 4 is the other legal period, k01, selectable
 * for small tests with ernie55_hca_period); rope_dim, head width, heads, q rank, o groups and o rank are the HCA layer's own keys. */
struct HcaConfig {
    int64_t heads = 0, D = 0, rope = 0, qlora = 0, groups = 0, rank = 0, window = 0, period = 128, block = 64;
    double theta = 0;
    bool yarn = true;
    /* The compressed group's layout (ernie55_hca_cmp_layout auto | block | token): true = one row per `period`-token block (block size `period`, a page is one
     * row), false = one slot per token at the block size of the other HCA groups (the CPU-reference form, kept for A/B). auto = block where head_dim is a multiple
     * of 2 * period (the release: 512 / 256), token otherwise (the sealed 3-4b fixtures use head_dim 16 at period 128). */
    bool cmp_block = true;
    double yfactor = 0, ybeta_fast = 32, ybeta_slow = 1, yorig = 0;
};
static int read_hca_config(const RadModelMeta* m, const Config& c, HcaConfig* h, std::string* why) {
    int64_t period = 128;
    if (rad_meta_gets(m, "ernie55_hca_period", nullptr)) period = rad_meta_geti(m, "ernie55_hca_period", 128);
    if (period != 4 && period != 128) {
        *why = "ernie55_hca_period must be 4 or 128 (the two legal compression periods), got " + std::to_string((long long)period);
        return RAD_E_INVAL;
    }
    h->period = period;
    const char* cml = rad_meta_gets(m, "ernie55_hca_cmp_layout", "auto");
    /* A serve-time override of the container key, for A/B runs on one container without a re-conversion: RADIANCE_ERNIE55_HCA_CMP_LAYOUT=auto|block|token. */
    if (const char* ev = getenv("RADIANCE_ERNIE55_HCA_CMP_LAYOUT")) if (*ev) cml = ev;
    if (strcmp(cml, "auto") != 0 && strcmp(cml, "block") != 0 && strcmp(cml, "token") != 0) {
        *why = std::string("ernie55_hca_cmp_layout must be auto, block or token, got \"") + cml + "\"";
        return RAD_E_INVAL;
    }
    const char* rp = rad_meta_gets(m, "ernie55_hca_rope", "yarn");
    if (strcmp(rp, "yarn") != 0 && strcmp(rp, "plain") != 0) {
        *why = std::string("ernie55_hca_rope must be yarn or plain, got \"") + rp + "\"";
        return RAD_E_INVAL;
    }
    h->yarn = strcmp(rp, "yarn") == 0;
    if (!need(m, "num_attention_heads", 1, &h->heads, why) || !need(m, "qk_rope_head_dim", 2, &h->rope, why) ||
        !need(m, "q_lora_rank", 1, &h->qlora, why) || !need(m, "o_groups", 1, &h->groups, why) ||
        !need(m, "o_lora_rank", 1, &h->rank, why) || !need(m, "sliding_window", 1, &h->window, why) ||
        !need_real(m, "compress_rope_theta", false, &h->theta, why))
        return RAD_E_INVAL;
    h->D = c.head_dim;
    const int64_t cap = int64_t(1) << 20;
    if (h->heads > 4096 || h->D > 4096 || h->qlora > cap || h->groups > h->heads || h->rank > cap || h->window > cap) {
        *why = "an HCA attention extent exceeds the supported bound (heads 4096, head_dim 4096, 2^20)";
        return RAD_E_INVAL;
    }
    if (h->D < 2 || (h->D & 1) || (h->rope & 1) || h->rope > h->D || h->heads % h->groups) {
        *why = "HCA geometry: head_dim must be even, qk_rope_head_dim even and at most head_dim, num_attention_heads a multiple of o_groups";
        return RAD_E_INVAL;
    }
    /* A block-layout page is `period` rows of head_dim / period elements, a KV decl's rows are 2 * head_dim_decl wide: head_dim must divide by 2 * period. */
    const bool block_fits = h->D % (2 * h->period) == 0;
    if (strcmp(cml, "block") == 0 && !block_fits) {
        *why = "ernie55_hca_cmp_layout=block needs head_dim to be a multiple of 2 * period (a compressed row is `period` KV rows of head_dim / period); use auto or token";
        return RAD_E_INVAL;
    }
    h->cmp_block = strcmp(cml, "token") != 0 && block_fits;
    const int64_t blk = rad_meta_geti(m, "ernie55_attention_block", 64);
    if (blk < 1 || blk > 4096) { *why = "ernie55_attention_block must be 1..4096, got " + std::to_string((long long)blk); return RAD_E_INVAL; }
    h->block = blk;
    if (h->yarn) {
        if (!need_real(m, "rope_scaling.factor", false, &h->yfactor, why) || !need_real(m, "rope_scaling.beta_fast", false, &h->ybeta_fast, why) ||
            !need_real(m, "rope_scaling.beta_slow", false, &h->ybeta_slow, why) ||
            !need_real(m, "rope_scaling.original_max_position_embeddings", false, &h->yorig, why)) {
            *why += " (ernie55_hca_rope=yarn is the default: the HCA layers rotate with the YaRN table of rope_scaling; ernie55_hca_rope=plain turns it off)";
            return RAD_E_INVAL;
        }
        if (h->yfactor < 1.0 || h->ybeta_fast < h->ybeta_slow) {
            *why = "rope_scaling.factor must be >= 1 and beta_fast >= beta_slow";
            return RAD_E_INVAL;
        }
    }
    return RAD_OK;
}

/* One HCA layer's weights. A block-fp8 projection has no single handle (0): declare_ffn refuses it by name. */
struct HcaHandles {
    rad_weight qa = 0, qa_norm = 0, qb = 0, kv = 0, kv_norm = 0, gate = 0, og = 0, o = 0, u = 0, v = 0, sink = 0, ckv = 0, cgate = 0, ape = 0, cnorm = 0;
    QWeight q_qa, q_qb, q_kv, q_gate, q_o;
    int64_t rank = 0;
    bool present = false;
};

static int declare_hca(RadBuilder* b, const Config& c, const HcaConfig& a, int l, Names& nm, HcaHandles* h, bool quant_fwd = false, int world = 1,
                       const std::string& ap_override = "", const char* dst_prefix = "blk.", int grp_id = -1, bool with_compressor = true) {
    using namespace rad::arch;
    const int gl = grp_id >= 0 ? grp_id : l;
    const std::string ap = ap_override.empty() ? ("model.layers." + std::to_string(l) + ".self_attn") : ap_override;
    const std::string cp = ap + ".core_attention.compressor";
    const int64_t H = c.hidden, GR = a.groups * a.rank, W = (a.period == 4 ? 2 : 1) * a.D;
    const RadWeightGroup g = grp_layer(gl);
    const int A = RAD_ACCESS_PER_TOKEN;
    const int sh_r = (quant_fwd && world > 1) ? RAD_SHARD_ROW : RAD_SHARD_NONE;
    const int sh_c = (quant_fwd && world > 1) ? RAD_SHARD_COL : RAD_SHARD_NONE;
    auto dname = [&](const char* suffix) { return nm.f("%s%d.%s", dst_prefix, l, suffix); };
    auto lin = [&](const char* dn, const std::string& src, int64_t rows, int64_t cols, rad_weight* out, int sh = RAD_SHARD_NONE) {
        const char* suf = std::strncmp(dn, "blk.%d.", 7) == 0 ? dn + 7 : dn;
        return linear(b, dname(suf), nm.ckpt("%s", src.c_str()), rows, cols, g, nm, A, nullptr, out, sh, world);
    };
    auto pl = [&](const char* dn, const std::string& src, uint32_t stored, uint32_t declared,
                  std::initializer_list<int64_t> shape, rad_weight* out, int sh = RAD_SHARD_NONE) {
        const char* suf = std::strncmp(dn, "blk.%d.", 7) == 0 ? dn + 7 : dn;
        return plain(b, dname(suf), nm.ckpt("%s", src.c_str()), stored, declared, shape, A, gl, nullptr, out, sh, world);
    };
    h->present = true;
    auto linq = [&](const char* dn, const std::string& src, int64_t rows, int64_t cols, rad_weight* out, QWeight* qout, int sh = RAD_SHARD_NONE) -> int {
        if (!quant_fwd) return lin(dn, src, rows, cols, out, sh);
        const char* suf = std::strncmp(dn, "blk.%d.", 7) == 0 ? dn + 7 : dn;
        RAD_ARCH_TRY(linear_q(b, dname(suf), {nm.ckpt("%s", src.c_str())}, rows, cols, g, nm, qout, A, sh, nullptr, 0, nullptr, world));
        if (out && !qout->fp8) *out = qout->w;
        return RAD_OK;
    };
    RAD_ARCH_TRY(linq("blk.%d.attn_q_a.weight", ap + ".linear_q_down_proj.weight", a.qlora, H, &h->qa, &h->q_qa));
    RAD_ARCH_TRY(pl("blk.%d.attn_q_a_norm.weight", ap + ".q_layernorm.weight", RAD_BF16, RAD_BF16, {a.qlora}, &h->qa_norm));
    RAD_ARCH_TRY(linq("blk.%d.attn_q_b.weight", ap + ".linear_q_up_proj.weight", a.heads * a.D, a.qlora, &h->qb, &h->q_qb, sh_r));
    RAD_ARCH_TRY(linq("blk.%d.attn_kv_a_mqa.weight", ap + ".linear_kv_proj.weight", a.D, H, &h->kv, &h->q_kv));
    RAD_ARCH_TRY(pl("blk.%d.attn_kv_a_norm.weight", ap + ".kv_layernorm.weight", RAD_BF16, RAD_BF16, {a.D}, &h->kv_norm));
    RAD_ARCH_TRY(linq("blk.%d.attn_gate.weight", ap + ".gate_proj.weight", GR, a.qlora, &h->gate, &h->q_gate, sh_r));
    /* the grouped wo_a is stored without a ".weight" suffix (its block scales are `linear_o_group_proj_scale_inv`) */
    RAD_ARCH_TRY(lin("blk.%d.attn_o_group.weight", ap + ".linear_o_group_proj", GR, (a.heads / a.groups) * a.D, &h->og, sh_r));
    RAD_ARCH_TRY(linq("blk.%d.attn_output.weight", ap + ".o_proj.weight", H, GR, &h->o, &h->q_o, sh_c));
    {
        const char* un = dname("attn_vha_u.weight");
        const char* us = nm.ckpt("%s.vha_postmix_U", ap.c_str());
        RAD_ARCH_TRY(map_copy(b, un, us));
        RadEncoding enc{};
        int64_t dims[RAD_MAX_RANK] = {0};
        uint32_t rk = 0;
        int64_t rank = 16;
        if (rad_weight_encoding(b, un, &enc, dims, &rk) == RAD_OK) {
            if (rk != 2 || dims[0] != a.heads || dims[1] < 1) {
                fprintf(stderr, "radiance: ernie55: checkpoint tensor '%s' (weight '%s') is %s, but this model needs "
                                "[%lld, rank]\n", us, un, dims_text(dims, rk).c_str(), (long long)a.heads);
                return RAD_E_SHAPE;
            }
            rank = dims[1];
        }
        h->rank = rank;
        RAD_ARCH_TRY(expect(b, un, us, RAD_BF16, {a.heads, rank}));
        h->u = decl_w(b, un, RAD_BF16, {a.heads, rank}, A, RAD_SHARD_NONE, g);
        RAD_ARCH_TRY(pl("blk.%d.attn_vha_v.weight", ap + ".vha_postmix_V", RAD_BF16, RAD_BF16, {a.heads, rank}, &h->v));
    }
    RAD_ARCH_TRY(pl("blk.%d.attn_sink.weight", ap + ".core_attention.attn_sink", RAD_F32, RAD_F32, {a.heads}, &h->sink, sh_r));
    if (with_compressor) {
        /* stored BF16: keep BF16 on quant_fwd (2x less VRAM/bandwidth + BF16 WMMA in gemm_nt_f32), F32 otherwise */
        const uint32_t comp_dt = quant_fwd ? (uint32_t)RAD_BF16 : (uint32_t)RAD_F32;
        RAD_ARCH_TRY(pl("blk.%d.attn_comp_kv.weight", cp + ".linear_wkv.weight", RAD_BF16, comp_dt, {W, H}, &h->ckv));
        RAD_ARCH_TRY(pl("blk.%d.attn_comp_gate.weight", cp + ".linear_wgate.weight", RAD_BF16, comp_dt, {W, H}, &h->cgate));
        RAD_ARCH_TRY(pl("blk.%d.attn_comp_ape.weight", cp + ".ape", RAD_F32, RAD_F32, {a.period, W}, &h->ape));
        RAD_ARCH_TRY(pl("blk.%d.attn_comp_norm.weight", cp + ".norm.weight", RAD_BF16, RAD_BF16, {a.D}, &h->cnorm));
    }
    return RAD_OK;
}

static int declare_attention(RadBuilder* b, const Config& c, const AttnConfig& a, int l, Names& nm, AttnHandles* h, bool quant_fwd = false, int world = 1) {
    using namespace rad::arch;
    const std::string ap = "model.layers." + std::to_string(l) + ".self_attn";
    const std::string ix = ap + ".core_attention.indexer";
    const int64_t H = c.hidden, qw = a.nope + a.rope;
    const RadWeightGroup g = grp_layer(l);
    const int A = RAD_ACCESS_PER_TOKEN;
    const int sh_r = (quant_fwd && world > 1) ? RAD_SHARD_ROW : RAD_SHARD_NONE;
    const int sh_c = (quant_fwd && world > 1) ? RAD_SHARD_COL : RAD_SHARD_NONE;
    auto lin = [&](const char* dn, const std::string& src, int64_t rows, int64_t cols, rad_weight* out, int sh = RAD_SHARD_NONE) {
        return linear(b, nm.f(dn, l), nm.ckpt("%s", src.c_str()), rows, cols, g, nm, A, nullptr, out, sh, world);
    };
    auto pl = [&](const char* dn, const std::string& src, uint32_t stored, uint32_t declared,
                  std::initializer_list<int64_t> shape, rad_weight* out, int sh = RAD_SHARD_NONE) {
        return plain(b, nm.f(dn, l), nm.ckpt("%s", src.c_str()), stored, declared, shape, A, l, nullptr, out, sh, world);
    };
    h->present = true;
    auto linq = [&](const char* dn, const std::string& src, int64_t rows, int64_t cols, rad_weight* out, QWeight* qout, int sh = RAD_SHARD_NONE) -> int {
        if (!quant_fwd) return lin(dn, src, rows, cols, out, sh);
        RAD_ARCH_TRY(linear_q(b, nm.f(dn, l), {nm.ckpt("%s", src.c_str())}, rows, cols, g, nm, qout, A, sh, nullptr, 0, nullptr, world));
        if (out && !qout->fp8) *out = qout->w;
        return RAD_OK;
    };
    RAD_ARCH_TRY(linq("blk.%d.attn_q_a.weight", ap + ".q_a_proj.weight", a.qlora, H, &h->qa, &h->q_qa));
    RAD_ARCH_TRY(pl("blk.%d.attn_q_a_norm.weight", ap + ".q_a_layernorm.weight", RAD_BF16, RAD_BF16, {a.qlora}, &h->qa_norm));
    RAD_ARCH_TRY(linq("blk.%d.attn_q_b.weight", ap + ".q_b_proj.weight", a.heads * qw, a.qlora, &h->qb, &h->q_qb, sh_r));
    RAD_ARCH_TRY(linq("blk.%d.attn_kv_a_mqa.weight", ap + ".kv_a_proj_with_mqa.weight", a.lat + a.rope, H, &h->kva, &h->q_kva));
    RAD_ARCH_TRY(pl("blk.%d.attn_kv_a_norm.weight", ap + ".kv_a_layernorm.weight", RAD_BF16, RAD_BF16, {a.lat}, &h->kv_norm));
    RAD_ARCH_TRY(lin("blk.%d.attn_kv_b.weight", ap + ".kv_b_proj.weight", a.heads * (a.nope + a.vdim), a.lat, &h->kvb, sh_r));
    RAD_ARCH_TRY(linq("blk.%d.attn_gate.weight", ap + ".gate_proj.weight", a.heads * a.vdim, a.qlora, &h->gate, &h->q_gate, sh_r));
    RAD_ARCH_TRY(linq("blk.%d.attn_output.weight", ap + ".o_proj.weight", H, a.heads * a.vdim, &h->o, &h->q_o, sh_c));
    /* VHA rank from the stored shape ([heads, rank] both), 16 in the release. */
    {
        const char* un = nm.f("blk.%d.attn_vha_u.weight", l);
        const char* us = nm.ckpt("%s.vha_postmix_U", ap.c_str());
        RAD_ARCH_TRY(map_copy(b, un, us));
        RadEncoding enc{};
        int64_t dims[RAD_MAX_RANK] = {0};
        uint32_t rk = 0;
        int64_t rank = 16;
        if (rad_weight_encoding(b, un, &enc, dims, &rk) == RAD_OK) {
            if (rk != 2 || dims[0] != a.heads || dims[1] < 1) {
                fprintf(stderr, "radiance: ernie55: checkpoint tensor '%s' (weight '%s') is %s, but this model needs "
                                "[%lld, rank]\n", us, un, dims_text(dims, rk).c_str(), (long long)a.heads);
                return RAD_E_SHAPE;
            }
            rank = dims[1];
        }
        h->rank = rank;
        RAD_ARCH_TRY(expect(b, un, us, RAD_BF16, {a.heads, rank}));
        h->u = decl_w(b, un, RAD_BF16, {a.heads, rank}, A, RAD_SHARD_NONE, g);
        RAD_ARCH_TRY(pl("blk.%d.attn_vha_v.weight", ap + ".vha_postmix_V", RAD_BF16, RAD_BF16, {a.heads, rank}, &h->v));
    }
    RAD_ARCH_TRY(pl("blk.%d.attn_sink.weight", ap + ".core_attention.softmax_offset", RAD_BF16, RAD_F32, {a.heads}, &h->sink, sh_r));
    RAD_ARCH_TRY(linq("blk.%d.attn_idx_q_b.weight", ix + ".wq_b.weight", a.ih * a.id, a.qlora, &h->iq, &h->q_iq));
    RAD_ARCH_TRY(pl("blk.%d.attn_idx_k.weight", ix + ".wk.weight", RAD_BF16, RAD_BF16, {a.id, H}, &h->ik));
    RAD_ARCH_TRY(pl("blk.%d.attn_idx_w.weight", ix + ".weights_proj.weight", RAD_BF16, RAD_BF16, {a.ih, H}, &h->iw));
    RAD_ARCH_TRY(pl("blk.%d.attn_idx_k_norm.weight", ix + ".k_norm.weight", RAD_BF16, RAD_BF16, {a.id}, &h->ik_g));
    RAD_ARCH_TRY(pl("blk.%d.attn_idx_k_norm.bias", ix + ".k_norm.bias", RAD_BF16, RAD_BF16, {a.id}, &h->ik_b));
    return RAD_OK;
}

/* ================================================================== routed latent MoE */
/* Checkpoint names are the release's (layer L, `m` = model.layers.L.block_sparse_moe) and shapes
 * its shard headers' and moe.py's, `E` n_routed_experts, `Lt` moe_latent_size, `F`
 * moe_intermediate_size. The release stores both routers BF16, the correction bias and the learned
 * scale F32, the latent projections and experts block-fp8 (BF16 here as well, as for the dense
 * linears) and the latent norm BF16. Experts are declared one weight each at their own logical id
 * (the container writes them 0..n-1 as fixed-stride units), never by position in a shard. */
static constexpr int64_t kMaxExperts = 65536;   /* ids are ints; a larger count is not a model */

/* The handles the routed program (story 2-8) reads. Routers are declared F32 (the loader widens the
 * stored BF16 exactly): the Baidu gate computes its matmul and logits in f32. A block-fp8 projection
 * has no single handle; its entry stays 0 and declare_ffn refuses it by name. The expert handles are
 * three runs (all gates, then all ups, then all downs), each consecutive in expert order, which is
 * what moe_gemm's weight table is. */
struct RoutedHandles {
    rad_weight r0 = 0, r1 = 0, bias = 0, scale = 0, in = 0, out = 0, norm = 0;
    QWeight q_in, q_out;
    bool mx = false;
    std::vector<rad_weight> ge, ue, de;
    std::vector<rad_weight> gu_c0, gu_s0, gu_c1, gu_s1;
    std::vector<rad_weight> dn_c0, dn_s0, dn_c1, dn_s1;
};

static int declare_routed(RadBuilder* b, const Config& c, int l, Names& nm,
                          RoutedHandles* rh = nullptr, bool quant_fwd = false, int world = 1,
                          const std::string& moe_override = "", const char* dst_prefix = "blk.", int grp_id = -1) {
    using namespace rad::arch;
    const int gl = grp_id >= 0 ? grp_id : l;
    const int64_t H = c.hidden, E = c.experts, Lt = c.latent, F = c.moe_ff;
    const int sh_r = (quant_fwd && world > 1) ? RAD_SHARD_ROW : RAD_SHARD_NONE;
    const int sh_c = (quant_fwd && world > 1) ? RAD_SHARD_COL : RAD_SHARD_NONE;
    const int64_t Floc = (quant_fwd && world > 1) ? F / world : F;
    const std::string m = moe_override.empty() ? ("model.layers." + std::to_string(l) + ".block_sparse_moe") : moe_override;
    const std::string lat = "(moe_latent_size " + std::to_string((long long)Lt) + ")";
    const RadWeightGroup g = grp_layer(gl);
    /* The two router halves are separate [E, H] matrices: gate.weight scores the first E logits,
     * gate.weight_1 the second E (vLLM-ERNIE fuses them as rows [0, E) and [E, 2E)). */
    const uint32_t router = (rh && !quant_fwd) ? (uint32_t)RAD_F32 : (uint32_t)RAD_BF16;   /* read dtype */
    RAD_ARCH_TRY(plain(b, nm.f("%s%d.ffn_gate_inp.weight", dst_prefix, l), nm.ckpt("%s.gate.weight", m.c_str()),
                       RAD_BF16, router, {E, H}, RAD_ACCESS_PER_TOKEN, gl, nullptr,
                       rh ? &rh->r0 : nullptr));
    RAD_ARCH_TRY(plain(b, nm.f("%s%d.ffn_gate_inp_1.weight", dst_prefix, l),
                       nm.ckpt("%s.gate.weight_1", m.c_str()), RAD_BF16, router, {E, H},
                       RAD_ACCESS_PER_TOKEN, gl, nullptr, rh ? &rh->r1 : nullptr));
    /* F32 auxiliaries stay F32 end to end: never declared narrower than they are stored. */
    RAD_ARCH_TRY(plain(b, nm.f("%s%d.exp_probs_b.bias", dst_prefix, l),
                       nm.ckpt("%s.e_score_correction_bias", m.c_str()), RAD_F32, RAD_F32, {E},
                       RAD_ACCESS_PER_TOKEN, gl, nullptr, rh ? &rh->bias : nullptr));
    RAD_ARCH_TRY(plain(b, nm.f("%s%d.ffn_gate_inp.scale", dst_prefix, l),
                       nm.ckpt("%s.gate.routed_scaling_factor_param", m.c_str()), RAD_F32, RAD_F32,
                       {E}, RAD_ACCESS_PER_TOKEN, gl, nullptr, rh ? &rh->scale : nullptr));
    if (quant_fwd && rh) {
        RAD_ARCH_TRY(linear_q(b, nm.f("%s%d.ffn_latent_in.weight", dst_prefix, l),
                              {nm.ckpt("%s.fc1_latent_proj.weight", m.c_str())}, Lt, H, g, nm,
                              &rh->q_in, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, nullptr, 0, lat.c_str()));
        if (!rh->q_in.fp8) rh->in = rh->q_in.w;
        RAD_ARCH_TRY(linear_q(b, nm.f("%s%d.ffn_latent_out.weight", dst_prefix, l),
                              {nm.ckpt("%s.fc2_latent_proj.weight", m.c_str())}, H, Lt, g, nm,
                              &rh->q_out, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, nullptr, 0, lat.c_str()));
        if (!rh->q_out.fp8) rh->out = rh->q_out.w;
    } else {
        RAD_ARCH_TRY(linear(b, nm.f("%s%d.ffn_latent_in.weight", dst_prefix, l),
                            nm.ckpt("%s.fc1_latent_proj.weight", m.c_str()), Lt, H, g, nm,
                            RAD_ACCESS_PER_TOKEN, lat.c_str(), rh ? &rh->in : nullptr));
        RAD_ARCH_TRY(linear(b, nm.f("%s%d.ffn_latent_out.weight", dst_prefix, l),
                            nm.ckpt("%s.fc2_latent_proj.weight", m.c_str()), H, Lt, g, nm,
                            RAD_ACCESS_PER_TOKEN, lat.c_str(), rh ? &rh->out : nullptr));
    }
    RAD_ARCH_TRY(plain(b, nm.f("%s%d.ffn_latent_norm.weight", dst_prefix, l),
                       nm.ckpt("%s.latent_norm.weight", m.c_str()), RAD_BF16, RAD_BF16, {Lt},
                       RAD_ACCESS_PER_TOKEN, gl, lat.c_str(), rh ? &rh->norm : nullptr));
    /* The first id past the end must not exist: a checkpoint with more experts than the config says
     * (an off-by-one count) would otherwise convert with the extras silently dropped. Each of the
     * three projections is probed (a surplus expert may lack any one). Only names can be asked for,
     * so a surplus id with a gap below it is not seen. */
    static const char* const kind[3] = {"gate", "up", "down"};
    static const char* const wn[3] = {"w1", "w3", "w2"};
    for (int k = 0; k < 3; ++k) {
        const char* over = nm.ckpt("%s.experts.%lld.%s.weight", m.c_str(), (long long)E, wn[k]);
        const char* probe = nm.f("%s%d.ffn_%s_exps.%lld.weight", dst_prefix, l, kind[k], (long long)E);
        RAD_ARCH_TRY(map_copy(b, probe, over));
        RadEncoding enc{};
        if (rad_weight_encoding(b, probe, &enc, nullptr, nullptr) == RAD_OK) {
            fprintf(stderr, "radiance: ernie55: checkpoint tensor '%s' is expert %lld of layer %d, but "
                            "n_routed_experts is %lld\n", over, (long long)E, l, (long long)E);
            return RAD_E_SHAPE;
        }
    }
    /* Expert-major, so the container sees each layer's experts as 0..n-1 units of three slots. The
     * declaration order is by projection (all gates, then all ups, then all downs): a movement unit
     * is the group (layer, expert, slot), not the handle order, and each run of handles is then
     * the consecutive weight table moe_gemm reads. */
    if (quant_fwd && rh) {
        const char* gu0_name = nm.f("%s%d.ffn_gate_up_exps.0.weight", dst_prefix, l);
        const std::string ep0 = m + ".experts.0";
        RAD_ARCH_TRY(map_concat(b, gu0_name,
                                {nm.ckpt("%s.w1.weight", ep0.c_str()), nm.ckpt("%s.w3.weight", ep0.c_str())}, 0));
        RadEncoding enc0{};
        const bool is_mx = weight_enc(b, gu0_name, &enc0) && linear_is_mxfp4(enc0);
        if (is_mx) {
            if (Lt % 128 || Floc % 128) {
                fprintf(stderr, "radiance: ernie55: MXFP4 routed experts require moe_latent_size (%lld) and "
                                "per-rank moe_intermediate_size (%lld) to be multiples of 128\n", (long long)Lt, (long long)Floc);
                return RAD_E_SHAPE;
            }
            rh->mx = true;
            for (int64_t e = 0; e < E; ++e) {
                const long long ee = (long long)e;
                const std::string ep = m + ".experts." + std::to_string(ee);
                const char* gun = nm.f("%s%d.ffn_gate_up_exps.%lld.weight", dst_prefix, l, ee);
                const char* dnn = nm.f("%s%d.ffn_down_exps.%lld.weight", dst_prefix, l, ee);
                if (e > 0)
                    RAD_ARCH_TRY(map_concat(b, gun,
                                            {nm.ckpt("%s.w1.weight", ep.c_str()), nm.ckpt("%s.w3.weight", ep.c_str())}, 0));
                RAD_ARCH_TRY(map_copy(b, dnn, nm.ckpt("%s.w2.weight", ep.c_str())));
                RAD_ARCH_TRY(expect(b, gun, nm.ckpt("%s.w1.weight", ep.c_str()), RAD_BF16, {2 * F, Lt}, false, nullptr, true));
                RAD_ARCH_TRY(expect(b, dnn, nm.ckpt("%s.w2.weight", ep.c_str()), RAD_BF16, {Lt, F}, false, nullptr, true));
                RadEncoding eg{}, ed{};
                if (!weight_enc(b, gun, &eg) || !linear_is_mxfp4(eg) || !weight_enc(b, dnn, &ed) || !linear_is_mxfp4(ed)) {
                    fprintf(stderr, "radiance: ernie55: layer %d expert %lld must be canonical MXFP4\n", l, ee);
                    return RAD_E_DTYPE;
                }
            }
            const int64_t gu_parts[2] = {Floc, Floc};
            auto decl_parity = [&](bool is_gu, std::vector<rad_weight>& c0, std::vector<rad_weight>& s0,
                                   std::vector<rad_weight>& c1, std::vector<rad_weight>& s1) -> int {
                const int64_t rows = is_gu ? 2 * Floc : Lt, cols = is_gu ? Lt : Floc;
                const int slot = is_gu ? 0 : 1;
                const int sh = is_gu ? sh_r : sh_c;
                const int64_t* rp = is_gu ? gu_parts : nullptr;
                const int np = is_gu ? 2 : 0;
                for (int par = 0; par < 2; ++par) {
                    auto& vc = par == 0 ? c0 : c1;
                    auto& vs = par == 0 ? s0 : s1;
                    for (int64_t e = par; e < E; e += 2) {
                        const char* wn = nm.f(is_gu ? "%s%d.ffn_gate_up_exps.%lld.weight"
                                                    : "%s%d.ffn_down_exps.%lld.weight", dst_prefix, l, (long long)e);
                        const rad_weight h = decl_view(b, wn, nullptr, "codes", RAD_FP4E2M1, {rows, cols},
                                                       RAD_ACCESS_CONDITIONAL, sh, grp_expert_slot(gl, (int)e, slot), rp, np);
                        if (!h) return RAD_E_INVAL;
                        vc.push_back(h);
                    }
                    for (int64_t e = par; e < E; e += 2) {
                        const char* wn = nm.f(is_gu ? "%s%d.ffn_gate_up_exps.%lld.weight"
                                                    : "%s%d.ffn_down_exps.%lld.weight", dst_prefix, l, (long long)e);
                        const char* sn = nm.f(is_gu ? "%s%d.ffn_gate_up_exps.%lld.scale"
                                                    : "%s%d.ffn_down_exps.%lld.scale", dst_prefix, l, (long long)e);
                        const rad_weight h = decl_view(b, sn, wn, "scale", RAD_E8M0, {rows, cols / RAD_MX_GROUP},
                                                       RAD_ACCESS_CONDITIONAL, sh, grp_expert_slot(gl, (int)e, slot), rp, np);
                        if (!h) return RAD_E_INVAL;
                        vs.push_back(h);
                    }
                }
                return RAD_OK;
            };
            RAD_ARCH_TRY(decl_parity(true, rh->gu_c0, rh->gu_s0, rh->gu_c1, rh->gu_s1));
            RAD_ARCH_TRY(decl_parity(false, rh->dn_c0, rh->dn_s0, rh->dn_c1, rh->dn_s1));
            return RAD_OK;
        }
    }
    if (rh) { rh->ge.assign((size_t)E, 0); rh->ue.assign((size_t)E, 0); rh->de.assign((size_t)E, 0); }
    static const char* const pn[3] = {"gate", "up", "down"};
    for (int k = 0; k < 3; ++k)
        for (int64_t e = 0; e < E; ++e) {
            const long long ee = (long long)e;
            const std::string ep = m + ".experts." + std::to_string(ee);
            const std::string at = "expert " + std::to_string(ee) + " of layer " + std::to_string(l) +
                                   " (moe_latent_size " + std::to_string((long long)Lt) +
                                   ", moe_intermediate_size " + std::to_string((long long)F) + ")";
            std::vector<rad_weight>* run = !rh ? nullptr : k == 0 ? &rh->ge : k == 1 ? &rh->ue : &rh->de;
            RAD_ARCH_TRY(linear(b, nm.f("%s%d.ffn_%s_exps.%lld.weight", dst_prefix, l, pn[k], ee),
                                nm.ckpt("%s.%s.weight", ep.c_str(), wn[k]), k == 2 ? Lt : F,
                                k == 2 ? F : Lt, grp_expert_slot(gl, (int)e, k), nm,
                                RAD_ACCESS_CONDITIONAL, at.c_str(), run ? &(*run)[(size_t)e] : nullptr,
                                k == 2 ? sh_c : sh_r, world));
        }
    return RAD_OK;
}

/* The weight handles the feed-forward graph (declare_ffn) reads, filled by declare_weights when it
 * is given one. A projection stored as block-fp8 is two views of one stored weight and has no
 * single handle: its entry stays 0 and declare_ffn refuses it by name. */
struct FfnHandles {
    struct Layer { MhcW hc; rad_weight gain = 0, gate = 0, up = 0, down = 0;
                   QWeight q_gu, q_down;
                   RoutedHandles moe;
                   MhcW attn_hc; rad_weight attn_gain = 0; AttnHandles attn; HcaHandles hca; };
    rad_weight tok = 0;
    /* The head (story 2-9): learned collapse, final norm, vocabulary projection, MultiMax params. */
    rad_weight head_fn = 0, head_base = 0, head_scale = 0, out_norm = 0, lm_head = 0, mm_r = 0, mm_t = 0;
    std::vector<Layer> layers;
};

static int declare_weights(RadBuilder* b, const Config& c, bool moe, FfnHandles* fh = nullptr,
                           bool moe_forward = false, const AttnConfig* ac = nullptr, const HcaConfig* hcc = nullptr,
                           bool quant_fwd = false, int world = 1) {
    using namespace rad::arch;
    Names nm;
    int64_t width = 0, mix = 0, head_n = 0, table = 0;
    int64_t hc2 = 0;
    if (__builtin_add_overflow(c.hc_mult, (int64_t)2, &hc2) ||
        !mul(c.hc_mult, c.hidden, &width) || !mul(hc2, c.hc_mult, &mix) ||
        !mul(c.vocab, c.hidden, &table) || !mul(mix, width, &head_n)) {
        fprintf(stderr, "radiance: ernie55: tensor extents overflow 64 bits (hidden=%lld "
                        "hc_mult=%lld vocab_size=%lld)\n", (long long)c.hidden,
                (long long)c.hc_mult, (long long)c.vocab);
        return RAD_E_INVAL;
    }
    const int64_t H = c.hidden, V = c.vocab;
    const int sh_r = (quant_fwd && world > 1) ? RAD_SHARD_ROW : RAD_SHARD_NONE;
    const int sh_c = (quant_fwd && world > 1) ? RAD_SHARD_COL : RAD_SHARD_NONE;

    RAD_ARCH_TRY(plain(b, "token_embd.weight", "model.embed_tokens.weight", RAD_BF16, RAD_BF16, {V, H},
                       RAD_ACCESS_VOCAB, -1, nullptr, fh ? &fh->tok : nullptr, sh_r, world));
    if (fh) fh->layers.resize((size_t)c.main_layers);
    /* Final head: the mHC collapse, the final norm, the vocabulary projection and MultiMax. */
    const char* hf = "head_hc.fn";
    RAD_ARCH_TRY(map_transpose(b, hf, "model.mhc_contract.hc_head_fn"));
    RAD_ARCH_TRY(expect(b, hf, "model.mhc_contract.hc_head_fn", RAD_F32, {c.hc_mult, width}));
    const rad_weight whead = decl_w(b, hf, RAD_F32, {c.hc_mult, width}, RAD_ACCESS_PER_TOKEN,
                                    RAD_SHARD_NONE, grp_model());
    if (fh) fh->head_fn = whead;
    RAD_ARCH_TRY(plain(b, "head_hc.base", "model.mhc_contract.hc_head_base", RAD_F32, RAD_F32,
                       {c.hc_mult}, RAD_ACCESS_PER_TOKEN, -1, nullptr, fh ? &fh->head_base : nullptr));
    RAD_ARCH_TRY(plain(b, "head_hc.scale", "model.mhc_contract.hc_head_scale", RAD_F32, RAD_F32, {1},
                       RAD_ACCESS_PER_TOKEN, -1, nullptr, fh ? &fh->head_scale : nullptr));
    RAD_ARCH_TRY(plain(b, "output_norm.weight", "model.norm.weight", RAD_BF16, RAD_BF16, {H},
                       RAD_ACCESS_PER_TOKEN, -1, nullptr, fh ? &fh->out_norm : nullptr));
    RAD_ARCH_TRY(plain(b, "output.weight", "lm_head.weight", RAD_BF16, RAD_BF16, {V, H}, RAD_ACCESS_VOCAB,
                       -1, nullptr, fh ? &fh->lm_head : nullptr, sh_r, world));
    /* Stored BF16, read F32 (an exact widening at load): ernie_multimax refuses any other dtype. */
    RAD_ARCH_TRY(plain(b, "multimax.ranges", "model.lm_head.multimax_ranges", RAD_BF16, RAD_F32, {4},
                       RAD_ACCESS_PER_TOKEN, -1, nullptr, fh ? &fh->mm_r : nullptr));
    RAD_ARCH_TRY(plain(b, "multimax.coefficients", "model.lm_head.multimax_ts", RAD_BF16, RAD_F32, {4},
                       RAD_ACCESS_PER_TOKEN, -1, nullptr, fh ? &fh->mm_t : nullptr));

    for (int l = 0; l < (int)c.main_layers; ++l) {
        const std::string lp = "model.layers." + std::to_string(l);
        FfnHandles::Layer* fl = fh ? &fh->layers[(size_t)l] : nullptr;
        RAD_ARCH_TRY(plain(b, nm.f("blk.%d.attn_norm.weight", l),
                           nm.ckpt("%s.input_layernorm.weight", lp.c_str()), RAD_BF16, RAD_BF16, {H},
                           RAD_ACCESS_PER_TOKEN, l, nullptr, fl ? &fl->attn_gain : nullptr));
        RAD_ARCH_TRY(plain(b, nm.f("blk.%d.ffn_norm.weight", l),
                           nm.ckpt("%s.post_attention_layernorm.weight", lp.c_str()), RAD_BF16,
                           RAD_BF16, {H}, RAD_ACCESS_PER_TOKEN, l, nullptr,
                           fl ? &fl->gain : nullptr));
        RAD_ARCH_TRY(mhc(b, nm, lp + ".self_attention_hyper_connection",
                         nm.f("blk.%d.attn_hc", l), l, mix, width, fl ? &fl->attn_hc : nullptr));
        if (ac && c.modes[(size_t)l] == kMlaDsa) RAD_ARCH_TRY(declare_attention(b, c, *ac, l, nm, fl ? &fl->attn : nullptr, quant_fwd, world));
        if (ac && hcc && c.modes[(size_t)l] == kHca) RAD_ARCH_TRY(declare_hca(b, c, *hcc, l, nm, fl ? &fl->hca : nullptr, quant_fwd, world));
        RAD_ARCH_TRY(mhc(b, nm, lp + ".mlp_hyper_connection", nm.f("blk.%d.ffn_hc", l), l, mix,
                         width, fl ? &fl->hc : nullptr));
        if (l < c.first_dense) {
            const int64_t I = c.dense_ff;
            if (quant_fwd && fl) {
                const int64_t rp[2] = {I, I};
                RAD_ARCH_TRY(linear_q(b, nm.f("blk.%d.ffn_gate_up.weight", l),
                                      {nm.ckpt("%s.mlp.gate_proj.weight", lp.c_str()),
                                       nm.ckpt("%s.mlp.up_proj.weight", lp.c_str())},
                                      2 * I, H, grp_layer(l), nm, &fl->q_gu, RAD_ACCESS_PER_TOKEN, sh_r, rp, 2, nullptr, world));
                RAD_ARCH_TRY(linear_q(b, nm.f("blk.%d.ffn_down.weight", l),
                                      {nm.ckpt("%s.mlp.down_proj.weight", lp.c_str())},
                                      H, I, grp_layer(l), nm, &fl->q_down, RAD_ACCESS_PER_TOKEN, sh_c, nullptr, 0, nullptr, world));
            } else {
                RAD_ARCH_TRY(linear(b, nm.f("blk.%d.ffn_gate.weight", l),
                                    nm.ckpt("%s.mlp.gate_proj.weight", lp.c_str()), I, H, grp_layer(l), nm,
                                    RAD_ACCESS_PER_TOKEN, nullptr, fl ? &fl->gate : nullptr));
                RAD_ARCH_TRY(linear(b, nm.f("blk.%d.ffn_up.weight", l),
                                    nm.ckpt("%s.mlp.up_proj.weight", lp.c_str()), I, H, grp_layer(l), nm,
                                    RAD_ACCESS_PER_TOKEN, nullptr, fl ? &fl->up : nullptr));
                RAD_ARCH_TRY(linear(b, nm.f("blk.%d.ffn_down.weight", l),
                                    nm.ckpt("%s.mlp.down_proj.weight", lp.c_str()), H, I, grp_layer(l), nm,
                                    RAD_ACCESS_PER_TOKEN, nullptr, fl ? &fl->down : nullptr));
            }
        } else {
            int64_t I = 0;
            if (!mul(c.moe_ff, c.shared, &I)) {
                fprintf(stderr, "radiance: ernie55: moe_intermediate_size (%lld) times "
                                "n_shared_experts (%lld) overflows 64 bits\n",
                        (long long)c.moe_ff, (long long)c.shared);
                return RAD_E_INVAL;
            }
            const std::string sp = lp + ".block_sparse_moe.shared_experts";
            if (quant_fwd && fl) {
                const int64_t rp[2] = {I, I};
                RAD_ARCH_TRY(linear_q(b, nm.f("blk.%d.ffn_gate_up_shexp.weight", l),
                                      {nm.ckpt("%s.w1.weight", sp.c_str()),
                                       nm.ckpt("%s.w3.weight", sp.c_str())},
                                      2 * I, H, grp_layer(l), nm, &fl->q_gu, RAD_ACCESS_PER_TOKEN, sh_r, rp, 2, nullptr, world));
                RAD_ARCH_TRY(linear_q(b, nm.f("blk.%d.ffn_down_shexp.weight", l),
                                      {nm.ckpt("%s.w2.weight", sp.c_str())},
                                      H, I, grp_layer(l), nm, &fl->q_down, RAD_ACCESS_PER_TOKEN, sh_c, nullptr, 0, nullptr, world));
            } else {
                RAD_ARCH_TRY(linear(b, nm.f("blk.%d.ffn_gate_shexp.weight", l),
                                    nm.ckpt("%s.w1.weight", sp.c_str()), I, H, grp_layer(l), nm,
                                    RAD_ACCESS_PER_TOKEN, nullptr, fl ? &fl->gate : nullptr));
                RAD_ARCH_TRY(linear(b, nm.f("blk.%d.ffn_up_shexp.weight", l),
                                    nm.ckpt("%s.w3.weight", sp.c_str()), I, H, grp_layer(l), nm,
                                    RAD_ACCESS_PER_TOKEN, nullptr, fl ? &fl->up : nullptr));
                RAD_ARCH_TRY(linear(b, nm.f("blk.%d.ffn_down_shexp.weight", l),
                                    nm.ckpt("%s.w2.weight", sp.c_str()), H, I, grp_layer(l), nm,
                                    RAD_ACCESS_PER_TOKEN, nullptr, fl ? &fl->down : nullptr));
            }
            if (moe) RAD_ARCH_TRY(declare_routed(b, c, l, nm, moe_forward && fl ? &fl->moe : nullptr, quant_fwd, world));
        }
    }
    return RAD_OK;
}


/* ================================================================== DSpark (3-layer block-diffusion MTP drafter) */
struct DsparkConfig {
    int64_t block_size = 7;
    int64_t markov_rank = 256;
    int64_t noise_token_id = 1024;
    std::vector<int64_t> target_layer_ids;
    int64_t n_taps = 0;
    int64_t n_layers = 0;
    int64_t window = 128;
    int64_t top_k = 32;
};

static bool dspark_parse_int_list(const char* s, std::vector<int64_t>* out) {
    out->clear();
    if (!s) return false;
    std::string cur;
    auto flush = [&]() -> bool {
        size_t a = 0, b = cur.size();
        while (a < b && std::isspace((unsigned char)cur[a])) ++a;
        while (b > a && std::isspace((unsigned char)cur[b - 1])) --b;
        if (a == b) return true;
        const std::string t = cur.substr(a, b - a);
        char* end = nullptr;
        errno = 0;
        const long long v = std::strtoll(t.c_str(), &end, 10);
        if (!end || end == t.c_str() || *end != '\0' || errno) return false;
        out->push_back((int64_t)v);
        return true;
    };
    for (const char* p = s; *p; ++p) {
        if (*p == '[' || *p == ']' || *p == ',' || std::isspace((unsigned char)*p)) {
            if (!flush()) return false;
            cur.clear();
        } else {
            cur.push_back(*p);
        }
    }
    return flush();
}

/* The drafter's per-position candidate count (the row_topk R, dflash_select's top_k): ERNIE_DSPARK_TOPK=<an even integer in 2..32>, default 32. Unset, empty, non-integer,
 * odd or out of range keeps 32 (the 8-rank all-gather of the 2K f32 row needs row*4 % 16 == 0, i.e. K even: libr4d ar_gather_wide_setup) with one warning (unset: silent); a valid override is logged once. Every shape downstream (cand/unary/pair/gath, the all_gather row,
 * the select and merge R) is declared from this one field. */
static int64_t dspark_topk_from_env() {
    const char* ev = std::getenv("ERNIE_DSPARK_TOPK");
    if (!ev || !*ev) return 32;
    static std::atomic<bool> logged{false};
    const bool first = !logged.exchange(true);
    char* end = nullptr;
    errno = 0;
    const long long v = strtoll(ev, &end, 10);
    if (!*ev || end == ev || *end || errno || v < 2 || v > 32 || (v & 1)) {
        if (first) fprintf(stderr, "radiance: ernie55: WARNING: ERNIE_DSPARK_TOPK=\"%s\" is not an even integer in 2..32; using 32\n", ev);
        return 32;
    }
    if (first) fprintf(stderr, "radiance: ernie55: ERNIE_DSPARK_TOPK=%lld: the drafter keeps its top %lld candidates a position (default 32)\n", v, v);
    return (int64_t)v;
}

static int read_dspark_config(const RadModelMeta* m, const Config& c, DsparkConfig* out, std::string* why) {
    auto bad = [&](const char* what) {
        if (why) *why = std::string("ernie55: dspark: ") + what;
        return RAD_E_INVAL;
    };
    DsparkConfig d;
    d.block_size = rad_meta_geti(m, "dspark_block_size", 7);
    if (d.block_size < 1 || d.block_size > 16) return bad("dspark_block_size must be in 1..16");
    d.markov_rank = rad_meta_geti(m, "dspark_markov_rank", 256);
    if (d.markov_rank < 32 || d.markov_rank > 512 || (d.markov_rank % 32) != 0)
        return bad("dspark_markov_rank must be a multiple of 32 in 32..512");
    d.noise_token_id = rad_meta_geti(m, "dspark_noise_token_id", 1024);
    if (d.noise_token_id < 0 || d.noise_token_id >= c.vocab)
        return bad("dspark_noise_token_id out of vocab range");
    const char* tl = rad_meta_gets(m, "dspark_target_layer_ids", nullptr);
    if (!tl || !*tl) return bad("dspark_target_layer_ids missing");
    if (!dspark_parse_int_list(tl, &d.target_layer_ids))
        return bad("dspark_target_layer_ids is not an integer list");
    d.n_taps = (int64_t)d.target_layer_ids.size();
    if (d.n_taps < 1 || d.n_taps > 8) return bad("dspark_target_layer_ids needs 1..8 ids");
    for (int64_t id : d.target_layer_ids)
        if (id < 0 || id >= c.main_layers) return bad("dspark_target_layer_ids id out of range");
    d.n_layers = d.n_taps;
    d.window = rad_meta_geti(m, "sliding_window", 128);
    if (d.window < 1) return bad("sliding_window must be >= 1");
    d.top_k = dspark_topk_from_env();
    *out = d;
    return RAD_OK;
}

struct DsparkHandles {
    QWeight feat_proj;
    rad_weight feat_norm = 0;
    rad_weight head_fn = 0, head_base = 0, head_scale = 0;
    rad_weight out_norm = 0;
    rad_weight mk1 = 0, mk2 = 0;
    std::vector<FfnHandles::Layer> layers;
};

static int declare_dspark_weights(RadBuilder* b, const Config& c, const HcaConfig& hcc,
                                  const DsparkConfig& ds, DsparkHandles* dh, int world) {
    using namespace rad::arch;
    if (!dh) return RAD_E_INVAL;
    Names nm;
    const int64_t H = c.hidden, V = c.vocab;
    int64_t width = 0, mix = 0;
    const int64_t hc2 = c.hc_mult + 2;
    if (!mul(c.hc_mult, H, &width) || !mul(hc2, c.hc_mult, &mix)) return RAD_E_INVAL;
    const int A = RAD_ACCESS_PER_TOKEN;
    const RadWeightGroup gm = grp_model();
    const int sh_r = world > 1 ? RAD_SHARD_ROW : RAD_SHARD_NONE;
    const int sh_c = world > 1 ? RAD_SHARD_COL : RAD_SHARD_NONE;
    const int64_t nt = ds.n_taps, rank = ds.markov_rank;

    RAD_ARCH_TRY(linear_q(b, "dspark.feature_proj.weight",
                          {"model.mtp.feature_projection.weight"}, H, nt * H, gm, nm,
                          &dh->feat_proj, A, RAD_SHARD_NONE, nullptr, 0, nullptr, world));
    RAD_ARCH_TRY(plain(b, "dspark.feature_norm.weight", "model.mtp.feature_norm.weight",
                       RAD_BF16, RAD_BF16, {H}, A, -1, nullptr, &dh->feat_norm));
    RAD_ARCH_TRY(map_transpose(b, "dspark.head_hc.fn", "model.mtp.mhc_contract.hc_head_fn"));
    RAD_ARCH_TRY(expect(b, "dspark.head_hc.fn", "model.mtp.mhc_contract.hc_head_fn",
                        RAD_F32, {c.hc_mult, width}));
    dh->head_fn = decl_w(b, "dspark.head_hc.fn", RAD_F32, {c.hc_mult, width}, A, RAD_SHARD_NONE, gm);
    RAD_ARCH_TRY(plain(b, "dspark.head_hc.base", "model.mtp.mhc_contract.hc_head_base",
                       RAD_F32, RAD_F32, {c.hc_mult}, A, -1, nullptr, &dh->head_base));
    RAD_ARCH_TRY(plain(b, "dspark.head_hc.scale", "model.mtp.mhc_contract.hc_head_scale",
                       RAD_F32, RAD_F32, {1}, A, -1, nullptr, &dh->head_scale));
    RAD_ARCH_TRY(plain(b, "dspark.norm.weight", "model.mtp.norm.weight",
                       RAD_BF16, RAD_BF16, {H}, A, -1, nullptr, &dh->out_norm));

    /* Markov codebooks: checkpoint stores markov_w1 as [rank, V], transpose to [V, rank]. */
    RAD_ARCH_TRY(map_transpose(b, "dspark.markov_w1", "model.mtp.markov_head.markov_w1.weight"));
    RAD_ARCH_TRY(expect(b, "dspark.markov_w1", "model.mtp.markov_head.markov_w1.weight",
                        RAD_BF16, {V, rank}));
    dh->mk1 = decl_w(b, "dspark.markov_w1", RAD_BF16, {V, rank}, RAD_ACCESS_VOCAB, RAD_SHARD_NONE, gm);
    RAD_ARCH_TRY(plain(b, "dspark.markov_w2", "model.mtp.markov_head.markov_w2.weight",
                       RAD_BF16, RAD_BF16, {V, rank}, RAD_ACCESS_VOCAB, -1, nullptr, &dh->mk2));

    dh->layers.resize((size_t)ds.n_layers);
    int64_t I = 0;
    if (!mul(c.moe_ff, c.shared, &I)) return RAD_E_INVAL;
    const int64_t rp[2] = {I, I};
    for (int i = 0; i < (int)ds.n_layers; ++i) {
        const int gl = (int)c.main_layers + i;
        const std::string lp = "model.mtp." + std::to_string(i);
        FfnHandles::Layer& dl = dh->layers[(size_t)i];
        const RadWeightGroup g = grp_layer(gl);
        RAD_ARCH_TRY(plain(b, nm.f("dspark.blk.%d.attn_norm.weight", i),
                           nm.ckpt("%s.input_layernorm.weight", lp.c_str()),
                           RAD_BF16, RAD_BF16, {H}, A, gl, nullptr, &dl.attn_gain));
        RAD_ARCH_TRY(plain(b, nm.f("dspark.blk.%d.ffn_norm.weight", i),
                           nm.ckpt("%s.post_attention_layernorm.weight", lp.c_str()),
                           RAD_BF16, RAD_BF16, {H}, A, gl, nullptr, &dl.gain));
        RAD_ARCH_TRY(mhc(b, nm, lp + ".self_attention_hyper_connection",
                         nm.f("dspark.blk.%d.attn_hc", i), gl, mix, width, &dl.attn_hc));
        RAD_ARCH_TRY(declare_hca(b, c, hcc, i, nm, &dl.hca, true, world,
                                 lp + ".self_attn", "dspark.blk.", gl, false));
        RAD_ARCH_TRY(mhc(b, nm, lp + ".mlp_hyper_connection",
                         nm.f("dspark.blk.%d.ffn_hc", i), gl, mix, width, &dl.hc));
        const std::string sp = lp + ".block_sparse_moe.shared_experts";
        RAD_ARCH_TRY(linear_q(b, nm.f("dspark.blk.%d.ffn_gate_up_shexp.weight", i),
                              {nm.ckpt("%s.w1.weight", sp.c_str()),
                               nm.ckpt("%s.w3.weight", sp.c_str())},
                              2 * I, H, g, nm, &dl.q_gu, A, sh_r, rp, 2, nullptr, world));
        RAD_ARCH_TRY(linear_q(b, nm.f("dspark.blk.%d.ffn_down_shexp.weight", i),
                              {nm.ckpt("%s.w2.weight", sp.c_str())},
                              H, I, g, nm, &dl.q_down, A, sh_c, nullptr, 0, nullptr, world));
        RAD_ARCH_TRY(declare_routed(b, c, i, nm, &dl.moe, true, world,
                                    lp + ".block_sparse_moe", "dspark.blk.", gl));
    }
    return RAD_OK;
}

/* ================================================================== feed-forward forward */
/* Story 2-7. The equations are the pinned Baidu model's (vLLM-ERNIE b8414da, nvidia/model.py and
 * nvidia/moe.py). A layer's feed-forward half reads the mHC `ffn` branch:
 *     x, post, comb = mhc_pre(h, hc_ffn_*) with the fused ffn_norm   (ernie_mhc_pre_rms)
 *     y = down(situ_and_mul(gate_up(x)))                             (dense: mlp, shared: w1/w3/w2)
 *     h = post * y + comb^T h                                        (mhc_post)
 * with the mixer's rms_eps and the fused norm's eps both rms_norm_eps, pre_eps and sinkhorn_eps both
 * hc_eps, post_mult 2.0 (`hc_post_alpha`), hc_sinkhorn_iters passes. The shared expert is the same
 * SiTU-GLU on the same normed x with no gate and no scale of its own; its output is later added to
 * the routed experts' (story 2-8), which this program does not run. The first layer's residual is the
 * embedding broadcast into hc identical streams (hc_enter). Gate and up stay two GEMMs into the two
 * halves of one [rows, 2I] plane, the layout `ernie_situ_mul` reads (gate first).
 *
 * Every layer here is the feed-forward half alone: attention does not exist yet, so the attention
 * branch of the real layer is absent and h passes unchanged into the feed-forward branch. */
struct FfnConfig {
    double rms_eps = 0, hc_eps = 0, beta = 0, linear_beta = 0;
    int64_t iterations = 0;
};

static bool need_real(const RadModelMeta* m, const char* key, bool zero_ok, double* out,
                      std::string* why) {
    const char* v = rad_meta_gets(m, key, nullptr);
    if (!v) { *why = std::string("config.json is missing ") + key; return false; }
    char* end = nullptr;
    errno = 0;
    const double x = (*v >= '0' && *v <= '9') || *v == '.' ? strtod(v, &end) : -1.0;
    if (!end || end == v || *end || errno || !std::isfinite(x) || x < 0 || (x == 0 && !zero_ok)) {
        *why = std::string(key) + " must be a finite number " + (zero_ok ? ">= 0" : "> 0") +
               ", got '" + v + "'";
        return false;
    }
    *out = x;
    return true;
}

static int read_ffn_config(const RadModelMeta* m, const Config& c, FfnConfig* f, std::string* why) {
    const char* act = rad_meta_gets(m, "hidden_act", nullptr);
    if (!act || strcmp(act, "situ") != 0) {
        *why = std::string("hidden_act must be \"situ\", got ") +
               (act ? std::string("\"") + act + "\"" : std::string("none"));
        return RAD_E_UNSUPPORTED;
    }
    if (!need_real(m, "rms_norm_eps", false, &f->rms_eps, why) ||
        !need_real(m, "hc_eps", true, &f->hc_eps, why) ||
        !need_real(m, "activation_situ_beta", false, &f->beta, why) ||
        !need_real(m, "activation_situ_linear_beta", false, &f->linear_beta, why) ||
        !need(m, "hc_sinkhorn_iters", 1, &f->iterations, why))
        return RAD_E_INVAL;
    if (f->iterations > 256) {
        *why = "hc_sinkhorn_iters must be at most 256, got " + std::to_string((long long)f->iterations);
        return RAD_E_INVAL;
    }
    if (c.hc_mult < 2 || c.hc_mult > 8) {
        *why = "hc_mult must be 2..8 for the mHC ops, got " + std::to_string((long long)c.hc_mult);
        return RAD_E_INVAL;
    }
    return RAD_OK;
}

/* The routed program's config (story 2-8): vLLM-ERNIE's Ernie5_5MoE refuses a model without split-feature
 * routing and a learnable per-expert scale, and with more than one expert group or a non-sigmoid score. */
static int read_moe_config(const RadModelMeta* m, bool* renorm, std::string* why) {
    int64_t norm = 0, split = 0, learn = 0, groups = 1, topk_groups = 1;
    if (!need(m, "norm_topk_prob", 0, &norm, why) || !need(m, "moe_split_feature_routing", 0, &split, why) ||
        !need(m, "routed_scaling_factor_learnable", 0, &learn, why))
        return RAD_E_INVAL;
    if (norm > 1) { *why = "norm_topk_prob must be 0 or 1, got " + std::to_string((long long)norm); return RAD_E_INVAL; }
    if (split != 1) { *why = "moe_split_feature_routing must be true: the two-router form is the only one routed"; return RAD_E_UNSUPPORTED; }
    if (learn != 1) { *why = "routed_scaling_factor_learnable must be true: the per-expert scale is a learned parameter"; return RAD_E_UNSUPPORTED; }
    if (rad_meta_gets(m, "n_group", nullptr) && !need(m, "n_group", 1, &groups, why)) return RAD_E_INVAL;
    if (rad_meta_gets(m, "topk_group", nullptr) && !need(m, "topk_group", 1, &topk_groups, why)) return RAD_E_INVAL;
    if (groups != 1 || topk_groups != 1) {
        *why = "n_group and topk_group must be 1: only one expert group is routed";
        return RAD_E_UNSUPPORTED;
    }
    const char* sf = rad_meta_gets(m, "scoring_func", "sigmoid");
    if (strcmp(sf, "sigmoid") != 0) {
        *why = std::string("scoring_func must be \"sigmoid\", got \"") + sf + "\"";
        return RAD_E_UNSUPPORTED;
    }
    *renorm = norm == 1;
    return RAD_OK;
}

/* A routed layer's ops, beyond the shared expert's. See declare_ffn. */
struct RoutedOps {
    RoutedHandles w;
    rad_op r0 = 0, r1 = 0, r01 = 0, topk = 0, scatter = 0, topk_scatter = 0, fc1 = 0, gate = 0, up = 0,
           act = 0, act_q = 0, down = 0, gather = 0, comb_ar = 0, ffn_ar = 0, norm = 0, fc2 = 0,
           add = 0, add_post = 0;
    QOp q_fc1, q_fc2;
    rad_op gu_q = 0, dn_q = 0, qop_xl = 0, qop_eact = 0, qop_nl = 0;
    bool need_qx = false, need_qnl = false, combine_ar = false;
};

struct AttnLayer {
    AttnHandles w;
    MhcW hc;
    rad_weight gain = 0;
    rad_op pre = 0, qa = 0, qn = 0, qb = 0, kva = 0, kvn = 0, rq = 0, rk = 0, store = 0, ik = 0, iw = 0, iq = 0,
           iqn = 0, ikey = 0, logit = 0, select = 0, topk_tiled = 0, slice = 0, absorb = 0, attn = 0, expand = 0,
           vha_ag = 0, vha = 0, vha_proj = 0, vha_ar = 0, vha_apply = 0, gate = 0,
           gmul = 0, out = 0, out_ar = 0, post = 0;
    int64_t vha_rank = 0;
    QOp q_qa, q_qb, q_kva, q_iq, q_gate, q_out;
    rad_op qop_x = 0, qop_qn = 0, qop_og = 0;
    bool need_qx = false, need_qqn = false, need_qog = false;
};

struct HcaLayer {
    HcaHandles w;
    MhcW hc;
    rad_weight gain = 0;
    rad_op pre = 0, qa = 0, qn = 0, qb = 0, qp = 0, kva = 0, kvn = 0, rk = 0, store = 0, ckv = 0, csc = 0, ckv_csc = 0, sw = 0, gs = 0, cmp = 0, att = 0,
           vha_ag = 0, vha = 0, vha_proj = 0, vha_ar = 0, vha_apply = 0, wo = 0, gate = 0, gmul = 0, out = 0, out_ar = 0, post = 0;
    int64_t vha_rank = 0;
    QOp q_qa, q_qb, q_kva, q_gate, q_out;
    rad_op qop_x = 0, qop_hqn = 0, qop_hog = 0;
    bool need_qx = false, need_qhqn = false, need_qhog = false;
};

struct FfnLayer {
    AttnLayer At;
    HcaLayer Hc;
    bool hca = false;
    rad_buf ffn_in = 0, ffn_out = 0;
    rad_weight fn = 0, scale = 0, base = 0, gain = 0, gate = 0, up = 0, down = 0;
    rad_op pre = 0, gate_op = 0, up_op = 0, act = 0, down_op = 0, post = 0;
    QOp q_gu, q_down;
    rad_op down_ar = 0, qop_x = 0, qop_act = 0, act_q = 0;
    bool need_qx = false, need_qact = false;
    int64_t width = 0;      /* the feed-forward's intermediate width */
    int kind = 0;           /* 0 dense, 1 shared: which gate_up / activation plane it uses */
    bool routed = false;    /* a routed layer (story 2-8): the shared expert plus the latent MoE */
    RoutedOps R;
};



struct DsparkProg {
    bool on = false;
    int64_t depth = 0, block = 0, n_taps = 0, n_layers = 0, window = 128, top_k = 32;
    std::vector<int64_t> target_layers;
    std::vector<rad_op> tap_mean;
    rad_kvgroup kv_ds = 0;
    rad_buf taps = 0, ctx = 0, lgt_bf16 = 0, cand = 0, unary = 0, pair = 0, gath = 0, tokens = 0;
    QAct qa_taps;
    rad_weight feat_norm = 0, head_fn = 0, head_base = 0, head_scale = 0, out_norm = 0, mk1 = 0, mk2 = 0;
    /* Context pass ops */
    bool need_qtaps = false, need_qctx = false;
    rad_op qop_taps = 0, fnorm = 0, qop_ctx = 0;
    QOp q_fproj;
    std::vector<QOp> ctx_kv;
    std::vector<rad_op> ctx_kvn, ctx_rk, ctx_store;
    std::vector<rad_weight> ctx_kv_norm;
    /* Query pass ops */
    rad_op embed = 0, embed_ar = 0, enter = 0;
    std::vector<FfnLayer> L;
    rad_op gather = 0, head = 0, norm = 0, proj = 0, lgt_cast = 0, topk = 0, gather_topk = 0, merge_topk = 0, sel = 0;
};

struct FfnProgram {
    bool attn = false;                  /* story 3-2b: the attention branch runs before each feed-forward half */
    AttnConfig A;
    int64_t lmax = 0;                   /* max-model-len (at least 1): the bound on a row's context the tiled selection is declared with */
    int fin = 0;                        /* which h buffer holds the final streams */
    rad_kvgroup kvl = 0, kvi = 0;
    /* --kv-cache-dtype fp8 (kv-fp8-latent): the MLA latent group `kvl` only is stored as fp8_ds_mla rows (E4M3 codes, f32 tile scales, bf16 rope), in
     * quant blocks of lat_qb latent values. Every other group keeps its dtype. */
    bool lat_fp8 = false;
    int64_t lat_qb = 0;
    /* kv-fp8-latent v2: RADIANCE_ERNIE55_KV_LATENT_FMT=i8gN stores int8 codes with exact f32 scales per N latent values (i8_ds_mla, same
     * row skeleton); ds_mla or unset = the fp8_ds_mla row above (the default graph is unchanged) */
    bool lat_i8 = false;
    /* the integer layouts: i8gN -> i8_ds_mla, i10gN -> i10_ds_mla (10-bit codes, 2 packed low bits), i12gN -> i12_ds_mla (12-bit, a packed nibble); lat_lo = packed low-bit bytes of a row */
    const char* lat_lay = "i8_ds_mla";
    int64_t lat_lo = 0;
    rad_buf qa = 0, qn = 0, q = 0, qr = 0, qabs = 0, kv = 0, gt = 0, ol = 0, ov = 0, om = 0, og = 0, ik = 0, iw = 0,
            iq = 0, qf = 0, qs = 0, sc = 0, topk_state = 0, req = 0, aids = 0, sl = 0;
    rad_op row_req = 0;
    /* HCA's sliding KV and compressor state groups each have their own per-sequence block table
     * origin. The full compressed group stores each summary at its completing token's slot. */
    bool hca = false;
    HcaConfig Hc;
    rad_kvgroup kvw = 0, kvs = 0, kvc = 0;
    rad_buf hwin_origin = 0, hstate_origin = 0;
    rad_buf hqa = 0, hqn = 0, hq = 0, hkv = 0, hkc = 0, hsc = 0, hgs = 0, hol = 0, hom = 0, hwo = 0, hgt = 0, hog = 0;
    bool on = false;
    int64_t T = 0, S = 0, hidden = 0, hc = 0, layers = 0;
    rad_weight tok = 0;
    rad_buf emb = 0, h[2] = {0, 0}, x = 0, post = 0, comb = 0, y = 0, mean = 0, pooled = 0;
    rad_buf gu[2] = {0, 0}, act[2] = {0, 0};
    /* Routed program (story 2-8): P = T * top_k rows, one a (token, slot). */
    int64_t E = 0, K = 0, Lt = 0, Fe = 0;
    bool moe = false;
    rad_buf ys = 0, lg = 0, ids = 0, rw = 0, sorted = 0, eoff = 0, ecnt = 0, xl = 0, egu = 0, eact = 0,
            edn = 0, comb2 = 0, ffn_ar = 0, nl = 0, ro = 0;
    rad_op embed = 0, enter = 0, pool = 0, cast = 0;
    /* Target output (story 2-9): R rows are the sampler's. */
    bool logits = false;
    int64_t R = 0;
    rad_weight head_fn = 0, head_base = 0, head_scale = 0, out_norm = 0, lm_head = 0, mm_r = 0, mm_t = 0;
    rad_buf hout = 0, hx = 0, hn = 0, lgt = 0;
    rad_op gather = 0, head = 0, norm = 0, proj = 0, mmax = 0;
    bool quant_fwd = false;
    int world = 1, rank = 0;
    int64_t Vloc = 0, Voff = 0, mla_heads_loc = 0, hca_heads_loc = 0, hca_groups_loc = 0, hca_GR_loc = 0;
    rad_buf ov_all = 0, hol_all = 0, oz = 0, hz = 0;
    int64_t mla_vha_rank = 0, hca_vha_rank = 0, mla_vha_dec_max = 0, hca_vha_dec_max = 0;
    rad_op embed_ar = 0;
    QAct qa_x, qa_act[2], qa_xl, qa_eact, qa_nl, qa_hqn, qa_hog, qa_qn, qa_og;
    std::vector<FfnLayer> L;
    DsparkProg ds;
};

static FfnProgram g_ffn[rad::arch::MAX_SLOTS];

/* ---- The collectives overlap (RADIANCE_ERNIE55_AR_OVERLAP=1, default off; the "collectives
 * overlap" amendment).
 *
 * At a prefill step of a tensor-parallel quantised forward every all_reduce sits in a hard chain:
 * the op after it reads the reduced tensor (vha_ar -> vha_apply, out_ar -> mHC post, the FFN
 * reduction -> latent norm and the shared add, the embedding reduction -> hc_enter). With the
 * switch on, a prefill step of at least `ov_min` rows is cut into two pieces of rows [0, R) and
 * [R, T); the program between two collectives (a "segment") runs a piece at a time on lane 0, and
 * every collective runs a piece at a time on lane 1, so piece 1's reduction moves while piece 2
 * computes the same segment, and piece 2's while piece 1 computes the next. The ops are the
 * serial step's ops on row ranges of the same buffers (every row lands where the serial step puts
 * it), each at its piece's row count; the attention and cache ops address rows by position and
 * request (never by cu_seqlens: row_req runs once over the whole step), so piece 2 reads piece 1's
 * cache writes, which lane 0 issued before it.
 *
 * The second lane's rules (core/runtime/ctx.h, Ctx::lane1_): every collective of the step is on
 * lane 1 (the embedding's included; the all_gather forms are decode-only and a piece never takes
 * them, ov_step checks the rows); lane 1 runs nothing but all_reduce, which carries no device-wide
 * arrival counter; every buffer the pieces touch is passed to rad_buf_concurrent, so neither lane's
 * write can land in another buffer's bytes (the planner packed them from the serial order, which
 * the piece order does not keep inside a segment), and ov_step asserts the arena ranges disjoint
 * before the first pipelined step, running serially otherwise. */
struct OvBuf {
    rad_buf h = 0;
    int64_t rows = 0, pitch = 0, esize = 0;
    int kind = 0;
};
static thread_local FfnProgram* s_ov_rec = nullptr;   /* declare_ffn's program while it declares (ranks declare on their own threads) */
static std::vector<OvBuf> g_ov_bufs[rad::arch::MAX_SLOTS];
static std::vector<int64_t> g_ov_pitch[rad::arch::MAX_SLOTS];
struct OvState {
    bool on = false, checked = false, ok = false, noted = false;
    int64_t min_rows = 0;
};
static OvState g_ov[rad::arch::MAX_SLOTS];
static thread_local int s_ov_slot = 0;

static int64_t ov_esize(uint32_t dt) {
    switch (dt) {
        case RAD_BF16: return 2;
        case RAD_F32: case RAD_I32: return 4;
        case RAD_F8E4M3: return 1;
        default: return 0;
    }
}

/* Shadows rad::arch::decl_b inside this namespace (unqualified lookup finds it before the
 * using-directive's), recording each buffer's row pitch for the pieces' row slices. Declares
 * exactly what rad::arch::decl_b declares. */
static rad_buf decl_b(RadBuilder* b, const char* name, uint32_t dtype, std::initializer_list<int64_t> shape,
                      int kind = RAD_BUF_TRANSIENT, int domain = RAD_DOMAIN_DEVICE) {
    const rad_buf h = rad::arch::decl_b(b, name, dtype, shape, kind, domain);
    if (s_ov_rec && h && shape.size() >= 1) {
        OvBuf r;
        r.h = h;
        r.rows = *shape.begin();
        r.pitch = 1;
        for (auto it = shape.begin() + 1; it != shape.end(); ++it) r.pitch *= *it;
        r.esize = ov_esize(dtype);
        r.kind = kind;
        g_ov_bufs[s_ov_slot].push_back(r);
        if (g_ov_pitch[s_ov_slot].size() <= h) g_ov_pitch[s_ov_slot].resize((size_t)h + 1, 0);
        g_ov_pitch[s_ov_slot][h] = r.pitch;
    }
    return h;
}

struct OvRecord {
    OvRecord(FfnProgram* m, int slot) {
        s_ov_rec = m;
        s_ov_slot = slot;
        g_ov_bufs[slot].clear();
        g_ov_pitch[slot].clear();
        g_ov[slot] = OvState{};
    }
    ~OvRecord() { s_ov_rec = nullptr; }
};

/* At the end of declare_ffn (target programs only): turn the overlap on if the switch asks and the
 * program is one it covers, and take every row buffer the pieces touch out of the sharing pool. */
static int ov_declare(RadBuilder* b, const RadBuildCtx* ctx, const FfnProgram& m) {
    const int slot = rad_build_slot(ctx);
    const char* ev = getenv("RADIANCE_ERNIE55_AR_OVERLAP");
    if (!ev || strcmp(ev, "1") != 0) return RAD_OK;
    auto off = [&](const char* why) {
        if (m.rank == 0) fprintf(stderr, "radiance: ernie55: collectives overlap requested but off: %s\n", why);
        return RAD_OK;
    };
    if (!m.quant_fwd || m.world < 2) return off("it needs the quantised forward at tensor parallel 2 or more");
    if (!m.attn || !m.logits || !m.embed_ar) return off("not a target program with attention");
    for (const FfnLayer& y : m.L) {
        const bool attn_ok = y.hca ? (y.Hc.vha_proj && y.Hc.vha_ar && y.Hc.vha_apply && y.Hc.out_ar)
                                   : (y.At.vha_proj && y.At.vha_ar && y.At.vha_apply && y.At.out_ar);
        const bool ffn_ok = y.routed ? (y.R.ffn_ar || (y.down_ar && y.R.comb_ar)) : (bool)y.down_ar;
        if (!attn_ok || !ffn_ok) return off("a layer lacks one of the expected all_reduces");
    }
    int64_t min_rows = 1280;
    if (const char* mv = getenv("RADIANCE_ERNIE55_AR_OVERLAP_MIN")) {
        int64_t v = 0;
        if (!parse_int(mv, &v) || v < 2) {
            fprintf(stderr, "radiance: ernie55: RADIANCE_ERNIE55_AR_OVERLAP_MIN must be an integer >= 2, got '%s'\n", mv);
            return RAD_E_INVAL;
        }
        min_rows = v;
    }
    const int64_t P = m.T * m.K;
    for (const OvBuf& r : g_ov_bufs[slot]) {
        if (r.kind != RAD_BUF_TRANSIENT) continue;
        if (r.h == m.hout || r.h == m.hx || r.h == m.hn || r.h == m.lgt) continue;   /* the head's, after both pieces */
        if (r.h == m.ov_all || r.h == m.hol_all) continue;                            /* decode-only gathers */
        if (r.rows != m.T && !(m.moe && r.rows == P)) continue;
        RAD_ARCH_TRY(rad_buf_concurrent(b, r.h));
    }
    g_ov[slot].on = true;
    g_ov[slot].min_rows = min_rows;
    if (m.rank == 0)
        fprintf(stderr, "radiance: ernie55: collectives overlap on: prefill steps from %lld rows run as two "
                        "pieces, all_reduce on the second lane\n", (long long)min_rows);
    return RAD_OK;
}

static int declare_ffn(RadBuilder* b, const RadBuildCtx* ctx, const Config& c, const FfnConfig& f,
                       const FfnHandles& fh, const bool* renorm = nullptr, bool target = false,
                       const AttnConfig* ac = nullptr, const HcaConfig* hcc = nullptr,
                       bool quant_fwd = false, const DsparkConfig* dsc = nullptr,
                       const DsparkHandles* dh = nullptr) {
    using namespace rad::arch;
    if ((!quant_fwd && ctx->world_size > 1) || ctx->world_size < 1 || ctx->rank < 0 ||
        ctx->rank >= ctx->world_size || ctx->is_draft || ctx->max_tok < 1 || ctx->max_seqs < 1 ||
        rad_build_slot(ctx) >= MAX_SLOTS) {
        fprintf(stderr, "radiance: ernie55: unsupported build context (world %d, rank %d, draft %d, quant_fwd %d)\n",
                (int)ctx->world_size, (int)ctx->rank, (int)ctx->is_draft, (int)quant_fwd);
        return RAD_E_UNSUPPORTED;
    }
    const int world = ctx->world_size, rank = ctx->rank;
    FfnProgram& m = g_ffn[rad_build_slot(ctx)];
    m = FfnProgram{};
    OvRecord ov_record(&m, rad_build_slot(ctx));
    Names nm(ctx->scope);
    const int64_t T = ctx->max_tok, S = ctx->max_seqs, H = c.hidden, hc = c.hc_mult;
    int64_t W = 0, Sh = 0;
    if (!mul(hc, H, &W) || !mul(c.moe_ff, c.shared, &Sh)) return RAD_E_INVAL;   /* checked in declare_weights */
    if (world > 1) {
        if ((c.vocab % world) != 0 ||
            (ac && (ac->heads % world) != 0) ||
            (hcc && ((hcc->heads % world) != 0 || (hcc->groups % world) != 0)) ||
            (c.first_dense > 0 && (c.dense_ff % world) != 0) ||
            (c.first_dense < c.main_layers && ((Sh % world) != 0 || (c.moe_ff % world) != 0))) {
            fprintf(stderr, "radiance: ernie55: TP=%d requires vocab, attention heads, HCA o_groups, and FFN/MoE widths divisible by world_size\n", world);
            return RAD_E_UNSUPPORTED;
        }
    }
    m.T = T; m.S = S; m.hidden = H; m.hc = hc; m.layers = c.main_layers; m.tok = fh.tok;
    m.quant_fwd = quant_fwd;
    m.world = world; m.rank = rank;
    m.Vloc = world > 1 ? c.vocab / world : c.vocab;
    m.Voff = world > 1 ? (int64_t)rank * m.Vloc : 0;
    m.fin = (int)(c.main_layers & 1);
    if (!m.tok) return RAD_E_INVAL;

    m.emb  = decl_b(b, nm.f("ffn.emb"), RAD_BF16, {T, H});
    m.h[0] = decl_b(b, nm.f("ffn.h0"), RAD_BF16, {T, W});
    m.h[1] = decl_b(b, nm.f("ffn.h1"), RAD_BF16, {T, W});
    m.x    = decl_b(b, nm.f("ffn.x"), RAD_BF16, {T, H});
    m.post = decl_b(b, nm.f("ffn.post"), RAD_F32, {T, hc});
    m.comb = decl_b(b, nm.f("ffn.comb"), RAD_F32, {T, hc * hc});
    m.y    = decl_b(b, nm.f("ffn.y"), RAD_BF16, {T, H});
    if (ac) {
        const AttnConfig& a = *ac;
        if (ctx->kv_dtype && ctx->kv_dtype != RAD_BF16 && ctx->kv_dtype != RAD_F8E4M3) {
            fprintf(stderr, "radiance: ernie55: the attention program keeps a bf16 latent cache (m25), or with --kv-cache-dtype fp8 an fp8_ds_mla "
                            "one; run with --kv-cache-dtype bf16 or fp8\n");
            return RAD_E_UNSUPPORTED;
        }
        if (ctx->kv_dtype == RAD_F8E4M3) {
            /* the fp8_ds_mla row is lat codes, 4 bytes of scale per quant block, rope bf16: lat is even (lat + rope and rope are), the block is the
             * largest of 128, 64, ... that divides lat (128 at the release's 512, so four scales) */
            int64_t qb = 128;
            while (qb > 1 && a.lat % qb) qb >>= 1;
            m.lat_fp8 = true; m.lat_qb = qb;
            if (const char* ev = getenv("RADIANCE_ERNIE55_KV_LATENT_FMT"); ev && *ev && strcmp(ev, "ds_mla") != 0) {
                int64_t g = 0;
                int bitsv = 0, gv = 0, used = 0;
                if (sscanf(ev, "i%dg%d%n", &bitsv, &gv, &used) == 2 && ev[used] == 0 && (bitsv == 8 || bitsv == 10 || bitsv == 12) &&
                    (gv == 128 || gv == 64 || gv == 32 || gv == 16))
                    g = gv;
                if (!strcmp(ev, "bf16id")) { g = qb; bitsv = 16; }   /* diagnostic: the bf16 latent losslessly through the quantised row path (bf16_ds_mla) */
                if (!g || a.lat % g) {
                    fprintf(stderr, "radiance: ernie55: RADIANCE_ERNIE55_KV_LATENT_FMT must be ds_mla, bf16id, or i8gN, i10gN, i12gN with N one of 128, 64, 32, 16 (dividing the latent width), got \"%s\"\n", ev);
                    return RAD_E_INVAL;
                }
                m.lat_i8 = true; m.lat_qb = qb = g;
                m.lat_lay = bitsv == 16 ? "bf16_ds_mla" : bitsv == 12 ? "i12_ds_mla" : bitsv == 10 ? "i10_ds_mla" : "i8_ds_mla";
                m.lat_lo = bitsv == 16 ? a.lat : bitsv == 12 ? a.lat / 2 : bitsv == 10 ? a.lat / 4 : 0;
                if (rank == 0)
                    fprintf(stderr, "radiance: ernie55: --kv-cache-dtype fp8 with RADIANCE_ERNIE55_KV_LATENT_FMT=%s: the MLA latent cache (kv_latent) is stored as %s rows (%lld bytes a token "
                                    "and layer, LOSSY: integer latent codes, one exact f32 scale per %lld values, bf16 rope); every other cache keeps its dtype\n",
                            ev, m.lat_lay, (long long)(a.lat + m.lat_lo + 4 * (a.lat / qb) + 2 * a.rope), (long long)qb);
            }
            if (rank == 0 && !m.lat_i8)
                fprintf(stderr, "radiance: ernie55: --kv-cache-dtype fp8: the MLA latent cache (kv_latent) is stored as fp8_ds_mla rows (%lld bytes a token "
                                "and layer, LOSSY: E4M3 latent, one power-of-two scale per %lld values, bf16 rope); the HCA window, state and compressed "
                                "caches, the indexer cache and any drafter cache keep their dtype\n",
                        (long long)(a.lat + 4 * (a.lat / qb) + 2 * a.rope), (long long)qb);
        }
        const int64_t Lmax = std::max<int64_t>(ctx->max_ctx, 1);
        int64_t sbytes = 0;
        const int64_t plane_cols = a.idx_tile > 0 ? std::min<int64_t>(a.idx_tile, Lmax) : Lmax;
        if (a.idx_tile > 0) {
            /* Tiled (ernie55_idx_tile): the plane is {T, tile}; the product cap applies to the tile, the context only has to fit i32 positions. */
            if (!mul(T, plane_cols, &sbytes) || sbytes > (int64_t(1) << 30) || Lmax > (int64_t(1) << 30)) {
                fprintf(stderr, "radiance: ernie55: the tiled indexer score plane is max-num-batched-tokens (%lld) x ernie55_idx_tile (%lld) "
                                "f32 and may not exceed 2^30 scores; max-model-len (%lld) may not exceed 2^30\n",
                        (long long)T, (long long)plane_cols, (long long)Lmax);
                return RAD_E_UNSUPPORTED;
            }
            fprintf(stderr, "radiance: ernie55: indexer score plane tiled: {%lld, %lld} f32 = %.2f MiB per card (untiled {%lld, %lld} would be %.2f MiB)\n",
                    (long long)T, (long long)plane_cols, (double)T * (double)plane_cols * 4.0 / 1048576.0, (long long)T, (long long)Lmax,
                    (double)T * (double)Lmax * 4.0 / 1048576.0);
        } else if (!mul(T, Lmax, &sbytes) || sbytes > (int64_t(1) << 30)) {
            fprintf(stderr, "radiance: ernie55: the indexer score plane is max-num-batched-tokens (%lld) x max-model-len (%lld) "
                            "f32; this reference program refuses more than 2^30 scores, lower either or set ernie55_idx_tile\n",
                    (long long)T, (long long)Lmax);
            return RAD_E_UNSUPPORTED;
        }
        const int64_t qw = a.nope + a.rope, W2 = a.topk + a.window;
        const int64_t ah_loc = a.heads / world;
        m.attn = true; m.A = a; m.lmax = Lmax; m.fin = 0; m.mla_heads_loc = ah_loc;
        m.qa   = decl_b(b, nm.f("attn.qa"), RAD_BF16, {T, a.qlora});
        m.qn   = decl_b(b, nm.f("attn.qn"), RAD_BF16, {T, a.qlora});
        m.q    = decl_b(b, nm.f("attn.q"), RAD_BF16, {T, ah_loc * qw});
        m.qr   = decl_b(b, nm.f("attn.qrope"), RAD_BF16, {T, ah_loc * a.rope});
        m.qabs = decl_b(b, nm.f("attn.qabs"), RAD_BF16, {T, ah_loc * a.lat});
        m.kv   = decl_b(b, nm.f("attn.kv"), RAD_BF16, {T, a.lat + a.rope});
        m.gt   = decl_b(b, nm.f("attn.gate"), RAD_BF16, {T, ah_loc * a.vdim});
        m.ol   = decl_b(b, nm.f("attn.olat"), RAD_BF16, {T, ah_loc * a.lat});
        m.ov   = decl_b(b, nm.f("attn.ov"), RAD_BF16, {T, ah_loc * a.vdim});
        int64_t mla_rk = 0;
        bool mla_same_rk = true;
        for (int64_t l = 0; l < c.main_layers; ++l) {
            if (c.modes[(size_t)l] != kHca && fh.layers[(size_t)l].attn.present) {
                const int64_t rk = fh.layers[(size_t)l].attn.rank;
                if (mla_rk == 0) mla_rk = rk;
                else if (rk != mla_rk) mla_same_rk = false;
            }
        }
        const bool mla_fac = world > 1 && quant_fwd && mla_same_rk && mla_rk > 0 && ((mla_rk * a.vdim) % 8 == 0);
        if (mla_fac) {
            m.mla_vha_rank = mla_rk;
            m.oz = decl_b(b, nm.f("attn.oz"), RAD_BF16, {T, mla_rk * a.vdim});
            if (mla_rk > ah_loc) {
                m.mla_vha_dec_max = std::min<int64_t>(T, vha_dec_cap());
                if (m.mla_vha_dec_max > 0) m.ov_all = decl_b(b, nm.f("attn.ov_all"), RAD_BF16, {world * m.mla_vha_dec_max, ah_loc * a.vdim});
            }
        } else if (world > 1) {
            m.mla_vha_dec_max = T;
            m.ov_all = decl_b(b, nm.f("attn.ov_all"), RAD_BF16, {world * T, ah_loc * a.vdim});
        }
        m.om   = decl_b(b, nm.f("attn.omix"), RAD_BF16, {T, ah_loc * a.vdim});
        m.og   = decl_b(b, nm.f("attn.ogated"), RAD_BF16, {T, ah_loc * a.vdim});
        m.ik   = decl_b(b, nm.f("attn.idx_k"), RAD_BF16, {T, a.id});
        m.iw   = decl_b(b, nm.f("attn.idx_w"), RAD_BF16, {T, a.ih});
        m.iq   = decl_b(b, nm.f("attn.idx_q"), RAD_BF16, {T, a.ih * a.id});
        m.qf   = decl_b(b, nm.f("attn.idx_qf"), RAD_F8E4M3, {T, a.ih * a.id});
        m.qs   = decl_b(b, nm.f("attn.idx_qs"), RAD_F32, {T, a.ih});
        /* Tiled: one flat scratch of T x tile scores, so a step with M rows scores tile = T * tile / M columns a row (ernie_dsa_topk_tiled). */
        m.sc   = a.idx_tile > 0 ? decl_b(b, nm.f("attn.idx_scores"), RAD_F32, {T * plane_cols})
                                : decl_b(b, nm.f("attn.idx_scores"), RAD_F32, {T, plane_cols});
        /* Tiled: the device op's running top-k and per-row verdicts (T x (2 * topk + 3) i32, 32 MiB at T 2048 and topk 2048): its outputs have one writer,
         * the last launch, which stores nothing unless every tile of every row was valid (libref's refusal contract). libref does not touch it. */
        if (a.idx_tile > 0) {
            fprintf(stderr, "radiance: ernie55: indexer top-k state: {%lld} i32 = %.2f MiB per card\n", (long long)(T * (2 * a.topk + 3)),
                    (double)(T * (2 * a.topk + 3)) * 4.0 / 1048576.0);
            m.topk_state = decl_b(b, nm.f("attn.idx_topk_state"), RAD_I32, {T * (2 * a.topk + 3)});
        }
        m.req  = decl_b(b, nm.f("attn.req"), RAD_I32, {T});
        m.aids = decl_b(b, nm.f("attn.ids"), RAD_I32, {T, W2});
        m.sl   = decl_b(b, nm.f("attn.slots"), RAD_I32, {T, W2});
        m.row_req = rw(b, RAD_OP(b, "ernie_row_req", RAD_PARAMS(RAD_RANGE("M", 1, T)), RAD_NOWEIGHTS), {}, {m.req});
        if (!m.row_req) return RAD_E_INVAL;
    }
    if (ac && hcc) {
        const HcaConfig& hh = *hcc;
        const int64_t Dh = hh.D, W = (hh.period == 4 ? 2 : 1) * Dh, Wn = (hh.period == 4 ? 2 : 1) * hh.period;
        const int64_t hh_loc = hh.heads / world, hg_loc = hh.groups / world, GR_loc = hg_loc * hh.rank;
        /* --kv-cache-dtype fp8 names the MLA latent cache only (kv-fp8-latent); the HCA caches stay bf16 */
        if (ctx->kv_dtype && ctx->kv_dtype != RAD_BF16 && ctx->kv_dtype != RAD_F8E4M3) {
            fprintf(stderr, "radiance: ernie55: the HCA program keeps bf16 window and compressed caches; run with --kv-cache-dtype bf16 (or fp8, which names the MLA latent cache only)\n");
            return RAD_E_UNSUPPORTED;
        }
        m.hca = true; m.Hc = hh; m.hca_heads_loc = hh_loc; m.hca_groups_loc = hg_loc; m.hca_GR_loc = GR_loc;
        m.hqa = decl_b(b, nm.f("hca.qa"), RAD_BF16, {T, hh.qlora});
        m.hqn = decl_b(b, nm.f("hca.qn"), RAD_BF16, {T, hh.qlora});
        m.hq  = decl_b(b, nm.f("hca.q"), RAD_BF16, {T, hh_loc * Dh});
        m.hkv = decl_b(b, nm.f("hca.kv"), RAD_BF16, {T, Dh});
        m.hkc = decl_b(b, nm.f("hca.comp_kv"), RAD_F32, {T, W});
        m.hsc = decl_b(b, nm.f("hca.comp_score"), RAD_F32, {T, W});
        m.hgs = decl_b(b, nm.f("hca.gather"), RAD_I32, {T, Wn});
        m.hol = decl_b(b, nm.f("hca.o"), RAD_BF16, {T, hh_loc * Dh});
        int64_t hca_rk = 0;
        bool hca_same_rk = true;
        for (int64_t l = 0; l < c.main_layers; ++l) {
            if (c.modes[(size_t)l] == kHca && fh.layers[(size_t)l].hca.present) {
                const int64_t rk = fh.layers[(size_t)l].hca.rank;
                if (hca_rk == 0) hca_rk = rk;
                else if (rk != hca_rk) hca_same_rk = false;
            }
        }
        const bool hca_fac = world > 1 && quant_fwd && hca_same_rk && hca_rk > 0 && ((hca_rk * Dh) % 8 == 0);
        if (hca_fac) {
            m.hca_vha_rank = hca_rk;
            m.hz = decl_b(b, nm.f("hca.z"), RAD_BF16, {T, hca_rk * Dh});
            if (hca_rk > hh_loc) {
                m.hca_vha_dec_max = std::min<int64_t>(T, vha_dec_cap());
                if (m.hca_vha_dec_max > 0) m.hol_all = decl_b(b, nm.f("hca.o_all"), RAD_BF16, {world * m.hca_vha_dec_max, hh_loc * Dh});
            }
        } else if (world > 1) {
            m.hca_vha_dec_max = T;
            m.hol_all = decl_b(b, nm.f("hca.o_all"), RAD_BF16, {world * T, hh_loc * Dh});
        }
        m.hom = decl_b(b, nm.f("hca.omix"), RAD_BF16, {T, hh_loc * Dh});
        m.hwo = decl_b(b, nm.f("hca.wo_a"), RAD_BF16, {T, GR_loc});
        m.hgt = decl_b(b, nm.f("hca.gate"), RAD_BF16, {T, GR_loc});
        m.hog = decl_b(b, nm.f("hca.ogated"), RAD_BF16, {T, GR_loc});
    }
    const int64_t widths[2] = {c.first_dense > 0 ? c.dense_ff / world : 0,
                               c.first_dense < c.main_layers ? Sh / world : 0};
    for (int k = 0; k < 2; ++k) {
        if (!widths[k]) continue;
        m.gu[k]  = decl_b(b, nm.f("ffn.gate_up%d", k), RAD_BF16, {T, 2 * widths[k]});
        m.act[k] = decl_b(b, nm.f("ffn.act%d", k), RAD_BF16, {T, widths[k]});
    }
    if (renorm) {
        int64_t P = 0, E2 = 0;
        if (!mul(T, c.top_k, &P) || !mul(2, c.experts, &E2)) return RAD_E_INVAL;
        m.moe = true; m.E = c.experts; m.K = c.top_k; m.Lt = c.latent; m.Fe = c.moe_ff / world;
        bool any_combined = false, any_uncombined = false;
        for (int64_t l = c.first_dense; l < c.main_layers; ++l) {
            const QWeight& qd = fh.layers[(size_t)l].q_down;
            /* Combine shared down_ar [T, H] and routed comb_ar [T, Lt] into one [T, H + Lt]
             * all_reduce when shared down runs on gemm_fp8a8_f32scale (supports pitched y.stride[0];
             * fp8a8 and bf16 gemm_nt currently require dense y.stride[0] == N). */
            const bool comb = world > 1 && quant_fwd && qd.fp8 && qd.scale_dt == RAD_F32 &&
                              (H % 8) == 0 && (c.latent % 8) == 0;
            any_combined |= comb;
            any_uncombined |= !comb;
        }
        if (any_uncombined) m.ys = decl_b(b, nm.f("moe.shared"), RAD_BF16, {T, H});
        m.lg    = decl_b(b, nm.f("moe.logits"), RAD_F32, {T, E2});
        m.ids   = decl_b(b, nm.f("moe.ids"), RAD_I32, {T, c.top_k});
        m.rw    = decl_b(b, nm.f("moe.w"), RAD_F32, {T, c.top_k});
        m.sorted = decl_b(b, nm.f("moe.sorted"), RAD_I32, {P});
        m.eoff  = decl_b(b, nm.f("moe.eoff"), RAD_I32, {c.experts + 1});
        /* PERSIST: the routing histogram is read by the engine, not by a declared op. */
        m.ecnt  = decl_b(b, nm.f("moe.ecnt"), RAD_I32, {c.experts}, RAD_BUF_PERSIST);
        m.xl    = decl_b(b, nm.f("moe.latent"), RAD_BF16, {T, c.latent});
        m.egu   = decl_b(b, nm.f("moe.gate_up"), RAD_BF16, {P, 2 * m.Fe});
        m.eact  = decl_b(b, nm.f("moe.act"), RAD_BF16, {P, m.Fe});
        m.edn   = decl_b(b, nm.f("moe.down"), RAD_BF16, {P, c.latent});
        if (any_uncombined) m.comb2 = decl_b(b, nm.f("moe.combined"), RAD_BF16, {T, c.latent});
        if (any_combined)   m.ffn_ar = decl_b(b, nm.f("moe.ffn_ar"), RAD_BF16, {T, H + c.latent});
        m.nl    = decl_b(b, nm.f("moe.normed"), RAD_BF16, {T, c.latent});
        m.ro    = decl_b(b, nm.f("moe.routed"), RAD_BF16, {T, H});
    }
    if (target) {
        /* One row per sampled position, as the stock programs size the logits (rad_arch.h). */
        int64_t R = S * (ctx->max_spec > 0 ? ctx->max_spec + 1 : 1);
        if (R > T) R = T;
        if (ctx->max_out_rows > R) R = ctx->max_out_rows;
        m.R = R;
        m.head_fn = fh.head_fn; m.head_base = fh.head_base; m.head_scale = fh.head_scale;
        m.out_norm = fh.out_norm; m.lm_head = fh.lm_head; m.mm_r = fh.mm_r; m.mm_t = fh.mm_t;
        if (!m.head_fn || !m.head_base || !m.head_scale || !m.out_norm || !m.lm_head || !m.mm_r || !m.mm_t)
            return RAD_E_INVAL;
        m.hout = decl_b(b, nm.f("out.h"), RAD_BF16, {R, W});
        m.hx   = decl_b(b, nm.f("out.x"), RAD_BF16, {R, H});
        m.hn   = decl_b(b, nm.f("out.normed"), RAD_BF16, {R, H});
        m.lgt  = decl_b(b, nm.f("out.logits"), RAD_F32, {R, m.Vloc});
        RAD_ARCH_TRY(rad_declare_logits(b, m.lgt));
    } else {
        m.mean   = decl_b(b, nm.f("ffn.mean"), RAD_BF16, {S, W});
        m.pooled = decl_b(b, nm.f("ffn.pooled"), RAD_F32, {S, W}, RAD_BUF_PERSIST);
        RAD_ARCH_TRY(rad_declare_pooled(b, m.pooled));
    }

    auto make_ar = [&](int64_t width, rad_buf buf) -> rad_op {
        if (world <= 1) return 0;
        return rw(b, RAD_OP(b, "all_reduce",
                            RAD_PARAMS(RAD_INT("world_size", world), RAD_RANGE("numel", width, T * width),
                                       RAD_STR("dtype", "bf16"), RAD_INT("exact", 1)),
                            RAD_NOWEIGHTS), {buf}, {buf});
    };
    auto make_ag_max = [&](int64_t width, int64_t max_rows, rad_buf src, rad_buf dst) -> rad_op {
        if (world <= 1) return 0;
        return rw(b, RAD_OP(b, "all_gather",
                            RAD_PARAMS(RAD_INT("world_size", world), RAD_RANGE("numel", width, max_rows * width),
                                       RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS), {src}, {dst});
    };
    auto make_ag = [&](int64_t width, rad_buf src, rad_buf dst) -> rad_op {
        return make_ag_max(width, T, src, dst);
    };
    m.embed = rw(b, RAD_OP(b, "embed_lookup",
                           RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n_embd", H),
                                      RAD_INT("n_vocab", m.Vloc), RAD_STR("dtype", "bf16"),
                                      RAD_INT("vocab_offset", m.Voff), RAD_F64("mult", 1.0)),
                           RAD_WEIGHTS(m.tok)), {}, {m.emb});
    if (world > 1) {
        m.embed_ar = make_ar(H, m.emb);
        if (!m.embed_ar) return RAD_E_INVAL;
    }
    m.enter = rw(b, RAD_OP(b, "hc_enter",
                           RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", H), RAD_INT("hc", hc),
                                      RAD_STR("dtype", "bf16")),
                           RAD_NOWEIGHTS), {m.emb}, {m.h[0]});
    if (!m.embed || !m.enter) return RAD_E_INVAL;

    auto make_qact = [&](const char* tag, rad_buf src, int64_t rows, int64_t width, QAct* out) -> int {
        if (!quant_fwd || !src || width <= 0 || width % 128) return RAD_OK;
        out->K = width;
        out->q = decl_b(b, nm.f("qa.%s.q", tag), RAD_F8E4M3, {rows, width});
        out->s = decl_b(b, nm.f("qa.%s.s", tag), RAD_F32, {rows, width / 128});
        return out->valid() ? RAD_OK : RAD_E_INVAL;
    };
    auto make_qop = [&](rad_buf src, int64_t rows, const QAct& qa, rad_op* out) -> int {
        if (!qa.valid()) return RAD_E_SHAPE;
        *out = rw(b, RAD_OP(b, "quant_act_fp8",
                            RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", qa.K),
                                       RAD_INT("group", 128), RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS),
                  {src}, {qa.q, qa.s});
        return *out ? RAD_OK : RAD_E_INVAL;
    };
    if (quant_fwd) {
        RAD_ARCH_TRY(make_qact("x", m.x, T, H, &m.qa_x));
        for (int k = 0; k < 2; ++k)
            if (widths[k]) RAD_ARCH_TRY(make_qact(k == 0 ? "act0" : "act1", m.act[k], T, widths[k], &m.qa_act[k]));
        if (m.moe) {
            RAD_ARCH_TRY(make_qact("xl", m.xl, T, c.latent, &m.qa_xl));
            RAD_ARCH_TRY(make_qact("eact", m.eact, T * c.top_k, m.Fe, &m.qa_eact));
            RAD_ARCH_TRY(make_qact("nl", m.nl, T, c.latent, &m.qa_nl));
        }
        if (m.hca) {
            RAD_ARCH_TRY(make_qact("hqn", m.hqn, T, m.Hc.qlora, &m.qa_hqn));
            RAD_ARCH_TRY(make_qact("hog", m.hog, T, m.hca_GR_loc, &m.qa_hog));
        }
        if (m.attn) {
            RAD_ARCH_TRY(make_qact("qn", m.qn, T, m.A.qlora, &m.qa_qn));
            RAD_ARCH_TRY(make_qact("og", m.og, T, m.mla_heads_loc * m.A.vdim, &m.qa_og));
        }
    }
    auto make_qgemm = [&](int64_t N, int64_t K, const QWeight& qw, rad_buf x_bf16, const QAct& qa,
                          rad_buf y_out, QOp* out) -> int {
        out->w = qw;
        if (!qw.valid()) return RAD_E_INVAL;
        if (!qw.fp8) {
            out->op = rw(b, RAD_OP(b, "gemm_nt",
                                   RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("N", N), RAD_INT("K", K),
                                              RAD_STR("dtype", "bf16"), RAD_INT("row_identity", 1)),
                                   RAD_WEIGHTS(qw.w)),
                         {x_bf16}, {y_out});
        } else {
            if (!qa.valid()) {
                fprintf(stderr, "radiance: ernie55: block-fp8 GEMM (%lldx%lld) requires K divisible by 128\n",
                        (long long)N, (long long)K);
                return RAD_E_SHAPE;
            }
            const char* dt = qw.scale_dt == RAD_F32 ? "fp8a8f32" : "fp8a8";
            if (qw.scale_dt == RAD_F32) {
                /* Row identity of the fp8a8 decode path: libr4d serves a step by the matvec (<= mvm_max rows), the K-parallel WMMA decode forms
                 * (<= 64 rows, c = 1 or 2) or the prefill forms; the WMMA forms agree per row, the matvec does not (so mvm_max is declared 0). The
                 * WMMA decode form falls back to the generic kernel when N % (16 * c) != 0 (c = 2 only above 16 rows, so the fallback would
                 * change with the step's row count), when its LDS exceeds 64 KiB (unstaged it is (S*C + C) * 8 * 32 * 4 = 18,432 bytes at most,
                 * independent of K), or on operand alignment (fixed per op, not per step). N % 128 == 0 is the op's own constraint
                 * (cFp8F32Scale); it is asserted here so a shape that could take the fallback at T > 16 cannot be built. */
                if (N % 128 != 0 || K % 128 != 0) {
                    fprintf(stderr, "radiance: ernie55: fp8a8f32 GEMM %lldx%lld: N and K must be multiples of 128 so the decode path cannot "
                                    "fall back to the generic kernel at T > 16 (row identity)\n", (long long)N, (long long)K);
                    return RAD_E_SHAPE;
                }
                out->op = rw(b, RAD_OP(b, "gemm_nt_q",
                                       RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("N", N), RAD_INT("K", K),
                                                  RAD_INT("group", 128), RAD_STR("dtype", dt), RAD_INT("mvm_max", f8a8_mvm_declared())),
                                       RAD_WEIGHTS(qw.w, qw.s)),
                             {qa.q, qa.s}, {y_out});
            } else
            out->op = rw(b, RAD_OP(b, "gemm_nt_q",
                                   RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("N", N), RAD_INT("K", K),
                                              RAD_INT("group", 128), RAD_STR("dtype", dt)),
                                   RAD_WEIGHTS(qw.w, qw.s)),
                         {qa.q, qa.s}, {y_out});
            if (qw.scale_dt == RAD_F32 && (N % 128) == 0 && (K % 128) == 0) {
                const int64_t M16 = std::min<int64_t>(T, 8);
                out->op_a16 = rw(b, RAD_OP(b, "gemm_nt_q",
                                           RAD_PARAMS(RAD_RANGE("M", 1, M16), RAD_INT("N", N), RAD_INT("K", K),
                                                      RAD_INT("group", 128), RAD_STR("dtype", "fp8a16f32")),
                                           RAD_WEIGHTS(qw.w, qw.s)),
                                 {x_bf16}, {y_out});
                if (!out->op_a16) return RAD_E_INVAL;
            }
        }
        return out->op ? RAD_OK : RAD_E_INVAL;
    };

    const bool ds_on = quant_fwd && target && m.hca && m.moe && dsc && dh && ctx->max_spec > 0;
    if (ds_on) {
        /* The block is `depth` positions: the anchor and depth - 1 mask rows. Scheduler::block_begin refuses a block below 2, so a depth-1 drafter would
         * run its context pass for nothing and never propose; refuse it here, by name, instead of drafting silently nothing (review, kv-context-multiseq). */
        if (ctx->max_spec < pin::kMinDraftDepth) {
            fprintf(stderr, "radiance: ernie55: the DSpark drafter needs a draft depth (--num-speculative-tokens, --max-spec) of at least %d, got %d; "
                            "use 0 to run without the drafter, or %d..%lld\n",
                    pin::kMinDraftDepth, ctx->max_spec, pin::kMinDraftDepth, (long long)dsc->block_size);
            return RAD_E_INVAL;
        }
        if (ctx->max_spec > dsc->block_size) {
            fprintf(stderr, "radiance: ernie55: max_spec (%d) exceeds dspark_block_size (%lld)\n",
                    ctx->max_spec, (long long)dsc->block_size);
            return RAD_E_INVAL;
        }
        m.ds.on = true;
        m.ds.depth = ctx->max_spec;
        m.ds.block = ctx->max_spec;
        m.ds.n_taps = dsc->n_taps;
        m.ds.n_layers = dsc->n_layers;
        m.ds.window = dsc->window;
        m.ds.top_k = dsc->top_k;
        m.ds.target_layers = dsc->target_layer_ids;
        m.ds.tap_mean.assign((size_t)dsc->n_taps, 0);
        m.ds.taps = decl_b(b, nm.f("dspark.taps"), RAD_BF16, {T, dsc->n_taps * H}, RAD_BUF_PERSIST);
        if (!m.ds.taps) return RAD_E_INVAL;
    }
    m.L.resize((size_t)c.main_layers);
    for (int64_t l = 0; l < c.main_layers; ++l) {
        FfnLayer& y = m.L[(size_t)l];
        const FfnHandles::Layer& w = fh.layers[(size_t)l];
        y.kind = l < c.first_dense ? 0 : 1;
        y.width = widths[y.kind];
        y.fn = w.hc.fn; y.scale = w.hc.scale; y.base = w.hc.base; y.gain = w.gain;
        y.gate = w.gate; y.up = w.up; y.down = w.down;
        if (!quant_fwd && (!y.gate || !y.up || !y.down)) {
            fprintf(stderr, "radiance: ernie55: layer %lld feed-forward projections are block-fp8; "
                            "the 2-7 program reads BF16 projections only\n", (long long)l);
            return RAD_E_UNSUPPORTED;
        }
        if (quant_fwd && (!w.q_gu.valid() || !w.q_down.valid())) return RAD_E_INVAL;
        if (!y.fn || !y.scale || !y.base || !y.gain) return RAD_E_INVAL;
        const rad_buf cur = m.attn ? m.h[1] : m.h[l & 1], nxt = m.attn ? m.h[0] : m.h[(l + 1) & 1];
        y.ffn_in = cur; y.ffn_out = nxt;
        const int64_t I = y.width;
        auto gemm = [&](int64_t N, int64_t K, rad_weight wt) {
            /* row_identity 1: the bf16 GEMM rows (M <= 16, <= 64, tiled) take the untuned split in every band, whatever tune cache is loaded
             * (r4d_gemm_args.h cfg_for), so the indexer's ik / iw projections give one row the same bits at any step size. */
            return RAD_OP(b, "gemm_nt",
                          RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("N", N), RAD_INT("K", K),
                                     RAD_STR("dtype", "bf16"), RAD_INT("row_identity", 1)),
                          RAD_WEIGHTS(wt));
        };
        y.hca = m.attn && c.modes[(size_t)l] == kHca;
        if (y.hca) {
            if (!m.hca) {
                fprintf(stderr, "radiance: ernie55: layer %lld is an HCA layer (compress_ratios[%lld] = 128): add --set ernie55_hca_forward=1\n",
                        (long long)l, (long long)l);
                return RAD_E_UNSUPPORTED;
            }
            const HcaConfig& hh = m.Hc;
            HcaLayer& Hc = y.Hc;
            const HcaHandles& hw = w.hca;
            Hc.w = hw; Hc.hc = w.attn_hc; Hc.gain = w.attn_gain;
            if (!quant_fwd) {
                const std::pair<const char*, rad_weight> need_bf16[] = {
                    {"attn_q_a", hw.qa}, {"attn_q_b", hw.qb}, {"attn_kv_a_mqa", hw.kv}, {"attn_gate", hw.gate}, {"attn_o_group", hw.og},
                    {"attn_output", hw.o}};
                for (const auto& pr : need_bf16)
                    if (!pr.second) {
                        fprintf(stderr, "radiance: ernie55: layer %lld projection blk.%lld.%s.weight is block-fp8; the HCA program "
                                        "reads BF16: convert with --quant 'blk.*.%s.weight=cast:dtype=bf16' (an exact "
                                        "dequantisation)\n", (long long)l, (long long)l, pr.first, pr.first);
                        return RAD_E_UNSUPPORTED;
                    }
            } else {
                if (!hw.og) {
                    fprintf(stderr, "radiance: ernie55: layer %lld blk.%lld.attn_o_group.weight must be cast to BF16 "
                                    "for ernie_hca_wo_a (--quant 'blk.*.attn_o_group.weight=cast:dtype=bf16')\n",
                            (long long)l, (long long)l);
                    return RAD_E_UNSUPPORTED;
                }
                if (!hw.q_qa.valid() || !hw.q_qb.valid() || !hw.q_kv.valid() || !hw.q_gate.valid() || !hw.q_o.valid())
                    return RAD_E_INVAL;
            }
            if (!Hc.hc.fn || !Hc.hc.scale || !Hc.hc.base || !Hc.gain || !hw.qa_norm || !hw.kv_norm || !hw.u || !hw.v || !hw.sink || !hw.ckv ||
                !hw.cgate || !hw.ape || !hw.cnorm)
                return RAD_E_INVAL;
            const int64_t Dh = hh.D, coff = hh.period == 4 ? 2 : 1, W = coff * Dh, GR = m.hca_GR_loc;
            const int64_t hh_loc = m.hca_heads_loc, hg_loc = m.hca_groups_loc;
            const double yf = hh.yarn ? hh.yfactor : 0.0;
            Hc.pre = rw(b, RAD_OP(b, "ernie_mhc_pre_rms",
                                  RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", H), RAD_INT("hc", hc),
                                             RAD_INT("iterations", f.iterations), RAD_F64("eps", f.rms_eps),
                                             RAD_F64("pre_eps", f.hc_eps), RAD_F64("sinkhorn_eps", f.hc_eps),
                                             RAD_F64("post_mult", 2.0), RAD_F64("rms_eps", f.rms_eps),
                                             RAD_STR("dtype", "bf16")),
                                  RAD_WEIGHTS(Hc.hc.fn, Hc.hc.scale, Hc.hc.base, Hc.gain)),
                        {m.h[0]}, {m.x, m.post, m.comb});
            auto rms = [&](int64_t n, rad_weight g) {
                return RAD_OP(b, "rmsnorm", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", n), RAD_F64("eps", f.rms_eps),
                                                       RAD_STR("dtype", "bf16")), RAD_WEIGHTS(g));
            };
            if (quant_fwd) {
                Hc.need_qx = hw.q_qa.fp8 || hw.q_kv.fp8;
                Hc.need_qhqn = hw.q_qb.fp8 || hw.q_gate.fp8;
                Hc.need_qhog = hw.q_o.fp8;
                if (Hc.need_qx) RAD_ARCH_TRY(make_qop(m.x, T, m.qa_x, &Hc.qop_x));
                RAD_ARCH_TRY(make_qgemm(hh.qlora, H, hw.q_qa, m.x, m.qa_x, m.hqa, &Hc.q_qa));
                Hc.qa = Hc.q_qa.op;
                Hc.qn = rw(b, rms(hh.qlora, hw.qa_norm), {m.hqa}, {m.hqn});
                if (Hc.need_qhqn) RAD_ARCH_TRY(make_qop(m.hqn, T, m.qa_hqn, &Hc.qop_hqn));
                RAD_ARCH_TRY(make_qgemm(hh_loc * Dh, hh.qlora, hw.q_qb, m.hqn, m.qa_hqn, m.hq, &Hc.q_qb));
                Hc.qb = Hc.q_qb.op;
                Hc.qp = rw(b, RAD_OP(b, "ernie_hca_q_prep",
                                     RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("heads", hh_loc), RAD_INT("head_dim", Dh),
                                                RAD_INT("rope_dim", hh.rope), RAD_F64("eps", f.rms_eps), RAD_F64("theta", hh.theta),
                                                RAD_F64("yarn_factor", yf), RAD_F64("yarn_beta_fast", hh.ybeta_fast),
                                                RAD_F64("yarn_beta_slow", hh.ybeta_slow), RAD_F64("yarn_original_max", hh.yorig)),
                                     RAD_NOWEIGHTS), {m.hq}, {m.hq});
                RAD_ARCH_TRY(make_qgemm(Dh, H, hw.q_kv, m.x, m.qa_x, m.hkv, &Hc.q_kva));
                Hc.kva = Hc.q_kva.op;
            } else {
                Hc.qa = rw(b, gemm(hh.qlora, H, hw.qa), {m.x}, {m.hqa});
                Hc.qn = rw(b, rms(hh.qlora, hw.qa_norm), {m.hqa}, {m.hqn});
                Hc.qb = rw(b, gemm(hh.heads * Dh, hh.qlora, hw.qb), {m.hqn}, {m.hq});
                Hc.qp = rw(b, RAD_OP(b, "ernie_hca_q_prep",
                                     RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("heads", hh.heads), RAD_INT("head_dim", Dh),
                                                RAD_INT("rope_dim", hh.rope), RAD_F64("eps", f.rms_eps), RAD_F64("theta", hh.theta),
                                                RAD_F64("yarn_factor", yf), RAD_F64("yarn_beta_fast", hh.ybeta_fast),
                                                RAD_F64("yarn_beta_slow", hh.ybeta_slow), RAD_F64("yarn_original_max", hh.yorig)),
                                     RAD_NOWEIGHTS), {m.hq}, {m.hq});
                Hc.kva = rw(b, gemm(Dh, H, hw.kv), {m.x}, {m.hkv});
            }
            Hc.kvn = rw(b, rms(Dh, hw.kv_norm), {m.hkv}, {m.hkv});
            Hc.rk = rw(b, RAD_OP(b, "ernie_rope_tail",
                                 RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("heads", 1), RAD_INT("width", Dh), RAD_INT("rope_dim", hh.rope),
                                            RAD_F64("theta", hh.theta), RAD_F64("yarn_factor", yf), RAD_F64("yarn_beta_fast", hh.ybeta_fast),
                                            RAD_F64("yarn_beta_slow", hh.ybeta_slow), RAD_F64("yarn_original_max", hh.yorig)),
                                 RAD_NOWEIGHTS), {m.hkv}, {m.hkv});
            auto declare_origin = [&](const char* name, rad_kvgroup group) {
                RadBufDecl d{};
                d.dtype = RAD_I32;
                d.rank = 1;
                d.shape[0] = S;
                d.kind = RAD_BUF_DERIVED;
                d.domain = RAD_DOMAIN_DEVICE;
                d.derive = "kv_block_origin";
                d.kv = group;
                return rad_decl_buffer(b, name, &d);
            };
            /* A window row is the rotated K = V vector in bf16 (the two halves kv_store takes).
             * Each paged group takes its block size from the first following op with that parameter. */
            if (!m.kvw) {
                RadKVGroupDecl d{};
                /* The KV manager reserves blocks separately for the in-flight step. */
                d.kind = RAD_KV_WINDOW; d.window = hh.window;
                d.dtype = RAD_BF16; d.n_head_kv = 1; d.head_dim = Dh / 2;
                m.kvw = rad_decl_kv_group(b, nm.f("kv_hca_window"), &d);
                if (!m.kvw) return RAD_E_INVAL;
                m.hwin_origin = declare_origin(nm.f("hca.window_origin"), m.kvw);
                if (!m.hwin_origin) return RAD_E_INVAL;
            }
            RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, m.kvw));
            Hc.store = rw(b, RAD_OP(b, "kv_store",
                                    RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", Dh / 2), RAD_INT("n_head_kv", 1),
                                               RAD_INT("block_size", hh.block), RAD_STR("kv_dtype", "bf16")),
                                    RAD_NOWEIGHTS), {m.hkv}, {});
            Hc.ckv_csc = rw(b, RAD_OP(b, "gemm_nt_f32_pair",
                                      RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("N", W), RAD_INT("K", H), RAD_STR("dtype", "bf16")),
                                      RAD_WEIGHTS(hw.ckv, hw.cgate)), {m.x}, {m.hkc, m.hsc});
            Hc.ckv = Hc.csc = Hc.ckv_csc;
            /* The compressor state: a per-token f32 group, row = [kv | score + APE[pos % period]] (2 * coff * Dh f32), so a state row is addressed by its
             * position like any KV row, a chunk split cannot change it and the prefix cache reuses it with the tokens. */
            if (!m.kvs) {
                RadKVGroupDecl d{};
                /* Compression reads up to coff * period rows for any completing token. */
                d.kind = RAD_KV_WINDOW; d.window = coff * hh.period;
                d.dtype = RAD_F32; d.n_head_kv = 1; d.head_dim = W;
                m.kvs = rad_decl_kv_group(b, nm.f("kv_hca_state"), &d);
                if (!m.kvs) return RAD_E_INVAL;
                m.hstate_origin = declare_origin(nm.f("hca.state_origin"), m.kvs);
                if (!m.hstate_origin) return RAD_E_INVAL;
            }
            RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, m.kvs));
            Hc.sw = rw(b, RAD_OP(b, "ernie_hca_state_write",
                                 RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("ratio", hh.period), RAD_INT("head_dim", Dh),
                                            RAD_INT("block_size", hh.block)),
                                 RAD_WEIGHTS(hw.ape)), {m.hkc, m.hsc}, {});
            /* The compressed group, two layouts (ernie55_hca_cmp_layout).
             * token: group j lives at its completing token's slot, (j + 1) * period - 1, in a per-token group at the other HCA groups' block size. Each cached
             *   block is then a function of the tokens through its end, at the price of period - 1 unwritten slots a row. The gather supplies the block size.
             * block (default): the group's block is `period` tokens and its page is ONE row. The decl carries head_dim / (2 * period) so a block of `period`
             *   rows of that width is exactly one `head_dim` row (the engine hands the ops [pages, 1, period, head_dim / period]); token slot page * period +
             *   period - 1 names the row of page `page`. The cached block is still a function of the tokens through its end: a block of `period` tokens holds
             *   the one row those tokens complete. The block size comes from the first op after the group that carries a block_size: the union attention, which
             *   names `period` (ernie_hca_compress cannot carry it: the sealed 3-3b test pins its first sorted parameter names), so the group is declared after the
             *   gather (which names the state block size) and the gather cannot claim it. */
            const bool cblock = hh.cmp_block;
            auto declare_kvc = [&]() -> int {
                if (!m.kvc) {
                    RadKVGroupDecl d{};
                    d.kind = RAD_KV_FULL; d.dtype = RAD_BF16; d.n_head_kv = 1; d.head_dim = cblock ? Dh / (2 * hh.period) : Dh / 2;
                    m.kvc = rad_decl_kv_group(b, nm.f("kv_hca_compressed"), &d);
                    if (!m.kvc) return RAD_E_INVAL;
                }
                return rad_bind_layer_kv(b, (int)l, m.kvc);
            };
            if (!cblock) RAD_ARCH_TRY(declare_kvc());
            Hc.gs = rw(b, RAD_OP(b, "ernie_hca_gather_slots_origin",
                                 RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("ratio", hh.period), RAD_INT("block_size", hh.block)),
                                 RAD_NOWEIGHTS), {m.req, m.hstate_origin}, {m.hgs});
            if (cblock) RAD_ARCH_TRY(declare_kvc());
#define ERNIE_HCA_CMP_COMMON RAD_RANGE("M", 1, T), RAD_INT("ratio", hh.period), RAD_INT("head_dim", Dh), RAD_INT("rope_dim", hh.rope), \
                             RAD_F64("eps", f.rms_eps), RAD_F64("theta", hh.theta), RAD_F64("yarn_factor", yf), RAD_F64("yarn_beta_fast", hh.ybeta_fast), \
                             RAD_F64("yarn_beta_slow", hh.ybeta_slow), RAD_F64("yarn_original_max", hh.yorig)
            if (cblock)
                Hc.cmp = rw(b, RAD_OP(b, "ernie_hca_compress", RAD_PARAMS(ERNIE_HCA_CMP_COMMON, RAD_STR("out_layout", "block")),
                                      RAD_WEIGHTS(hw.cnorm)), {m.hgs}, {});
            else
                Hc.cmp = rw(b, RAD_OP(b, "ernie_hca_compress", RAD_PARAMS(ERNIE_HCA_CMP_COMMON), RAD_WEIGHTS(hw.cnorm)), {m.hgs}, {});
#undef ERNIE_HCA_CMP_COMMON
#define ERNIE_HCA_UNION_COMMON RAD_RANGE("M", 1, T), RAD_INT("heads", hh_loc), RAD_INT("head_dim", Dh), RAD_INT("ratio", hh.period), \
                               RAD_INT("window", hh.window), RAD_F64("scale", 1.0 / std::sqrt((double)Dh))
            if (cblock)
                Hc.att = rw(b, RAD_OP(b, "ernie_hca_union_attn_origin",
                                      RAD_PARAMS(ERNIE_HCA_UNION_COMMON, RAD_STR("cmp_layout", "block"), RAD_INT("block_size", hh.period)),
                                      RAD_WEIGHTS(hw.sink)), {m.hq, m.req, m.hwin_origin}, {m.hol});
            else
                Hc.att = rw(b, RAD_OP(b, "ernie_hca_union_attn_origin", RAD_PARAMS(ERNIE_HCA_UNION_COMMON, RAD_STR("cmp_layout", "token")),
                                      RAD_WEIGHTS(hw.sink)), {m.hq, m.req, m.hwin_origin}, {m.hol});
#undef ERNIE_HCA_UNION_COMMON
            /* after the union attention: it is the op that claims the compressed group's block size in the block layout */
            const int64_t kvc_block = cblock ? hh.period : hh.block;
            if (rad_kv_block_size(b, m.kvw) != hh.block || rad_kv_block_size(b, m.kvs) != hh.block || rad_kv_block_size(b, m.kvc) != kvc_block) {
                fprintf(stderr, "radiance: ernie55: a KV block is not %lld tokens (window %lld, state %lld, compressed %lld, expected compressed %lld)\n",
                        (long long)hh.block, (long long)rad_kv_block_size(b, m.kvw), (long long)rad_kv_block_size(b, m.kvs),
                        (long long)rad_kv_block_size(b, m.kvc), (long long)kvc_block);
                return RAD_E_INVAL;
            }
            if (world > 1 && m.hz) {
                Hc.vha_rank = hw.rank;
                Hc.vha_proj = rw(b, RAD_OP(b, "ernie_vha_proj",
                                           RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("heads", hh.heads), RAD_INT("dim", Dh), RAD_INT("rank", hw.rank),
                                                      RAD_INT("world_size", world), RAD_INT("tp_rank", rank)),
                                           RAD_WEIGHTS(hw.u)), {m.hol}, {m.hz});
                Hc.vha_ar = make_ar(hw.rank * Dh, m.hz);
                Hc.vha_apply = rw(b, RAD_OP(b, "ernie_vha_apply",
                                            RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("heads", hh.heads), RAD_INT("dim", Dh), RAD_INT("rank", hw.rank),
                                                       RAD_INT("world_size", world), RAD_INT("tp_rank", rank)),
                                            RAD_WEIGHTS(hw.v)), {m.hol, m.hz}, {m.hom});
                if (!Hc.vha_proj || !Hc.vha_ar || !Hc.vha_apply) return RAD_E_INVAL;
                if (m.hol_all && m.hca_vha_dec_max > 0) {
                    Hc.vha_ag = make_ag_max(hh_loc * Dh, m.hca_vha_dec_max, m.hol, m.hol_all);
                    Hc.vha = rw(b, RAD_OP(b, "ernie_vha_mix",
                                          RAD_PARAMS(RAD_RANGE("M", 1, m.hca_vha_dec_max), RAD_INT("heads", hh.heads), RAD_INT("dim", Dh),
                                                     RAD_INT("rank", hw.rank), RAD_STR("fold", kVhaFold),
                                                     RAD_INT("world_size", world), RAD_INT("tp_rank", rank)),
                                          RAD_WEIGHTS(hw.u, hw.v)), {m.hol_all}, {m.hom});
                    if (!Hc.vha_ag || !Hc.vha) return RAD_E_INVAL;
                }
            } else if (world > 1) {
                Hc.vha_ag = make_ag(hh_loc * Dh, m.hol, m.hol_all);
                if (!Hc.vha_ag) return RAD_E_INVAL;
                Hc.vha = rw(b, RAD_OP(b, "ernie_vha_mix",
                                      RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("heads", hh.heads), RAD_INT("dim", Dh), RAD_INT("rank", hw.rank),
                                                 RAD_STR("fold", kVhaFold), RAD_INT("world_size", world), RAD_INT("tp_rank", rank)),
                                      RAD_WEIGHTS(hw.u, hw.v)), {m.hol_all}, {m.hom});
            } else {
                Hc.vha = rw(b, RAD_OP(b, "ernie_vha_mix",
                                      RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("heads", hh.heads), RAD_INT("dim", Dh), RAD_INT("rank", hw.rank),
                                                 RAD_STR("fold", kVhaFold)),
                                      RAD_WEIGHTS(hw.u, hw.v)), {m.hol}, {m.hom});
            }
            Hc.wo = rw(b, RAD_OP(b, "ernie_hca_wo_a",
                                 RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("heads", hh_loc), RAD_INT("head_dim", Dh), RAD_INT("rope_dim", hh.rope),
                                            RAD_INT("groups", hg_loc), RAD_INT("rank", hh.rank), RAD_F64("theta", hh.theta),
                                            RAD_F64("yarn_factor", yf), RAD_F64("yarn_beta_fast", hh.ybeta_fast),
                                            RAD_F64("yarn_beta_slow", hh.ybeta_slow), RAD_F64("yarn_original_max", hh.yorig)),
                                 RAD_WEIGHTS(hw.og)), {m.hom}, {m.hwo});
            if (quant_fwd) {
                RAD_ARCH_TRY(make_qgemm(GR, hh.qlora, hw.q_gate, m.hqn, m.qa_hqn, m.hgt, &Hc.q_gate));
                Hc.gate = Hc.q_gate.op;
                Hc.gmul = rw(b, RAD_OP(b, "act_mul",
                                       RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", GR), RAD_STR("act", "sigmoid"), RAD_STR("dtype", "bf16")),
                                       RAD_NOWEIGHTS), {m.hgt, m.hwo}, {m.hog});
                if (Hc.need_qhog) RAD_ARCH_TRY(make_qop(m.hog, T, m.qa_hog, &Hc.qop_hog));
                RAD_ARCH_TRY(make_qgemm(H, GR, hw.q_o, m.hog, m.qa_hog, m.y, &Hc.q_out));
                Hc.out = Hc.q_out.op;
                if (world > 1) {
                    Hc.out_ar = make_ar(H, m.y);
                    if (!Hc.out_ar) return RAD_E_INVAL;
                }
            } else {
                Hc.gate = rw(b, gemm(GR, hh.qlora, hw.gate), {m.hqn}, {m.hgt});
                Hc.gmul = rw(b, RAD_OP(b, "act_mul",
                                       RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", GR), RAD_STR("act", "sigmoid"), RAD_STR("dtype", "bf16")),
                                       RAD_NOWEIGHTS), {m.hgt, m.hwo}, {m.hog});
                Hc.out = rw(b, gemm(H, GR, hw.o), {m.hog}, {m.y});
            }
            Hc.post = rw(b, RAD_OP(b, "mhc_post",
                                   RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", H), RAD_INT("hc", hc), RAD_STR("dtype", "bf16")),
                                   RAD_NOWEIGHTS), {m.y, m.h[0], m.post, m.comb}, {m.h[1]});
            if (!Hc.pre || !Hc.qa || !Hc.qn || !Hc.qb || !Hc.qp || !Hc.kva || !Hc.kvn || !Hc.rk || !Hc.store || !Hc.ckv || !Hc.csc || !Hc.sw || !Hc.gs ||
                !Hc.cmp || !Hc.att || (!Hc.vha && !Hc.vha_apply) || !Hc.wo || !Hc.gate || !Hc.gmul || !Hc.out || !Hc.post)
                return RAD_E_INVAL;
        } else if (m.attn) {
            const AttnConfig& a = m.A;
            AttnLayer& At = y.At;
            const AttnHandles& aw = w.attn;
            At.w = aw; At.hc = w.attn_hc; At.gain = w.attn_gain;
            if (!aw.present) {
                fprintf(stderr, "radiance: ernie55: layer %lld is an HCA layer (compress_ratios 128); the attention program "
                                "serves MLA-DSA layers only (story 3-4 builds HCA)\n", (long long)l);
                return RAD_E_UNSUPPORTED;
            }
            if (!quant_fwd) {
                const std::pair<const char*, rad_weight> need_bf16[] = {
                    {"attn_q_a", aw.qa}, {"attn_q_b", aw.qb}, {"attn_kv_a_mqa", aw.kva}, {"attn_kv_b", aw.kvb},
                    {"attn_gate", aw.gate}, {"attn_output", aw.o}, {"attn_idx_q_b", aw.iq}};
                for (const auto& pr : need_bf16)
                    if (!pr.second) {
                        fprintf(stderr, "radiance: ernie55: layer %lld projection blk.%lld.%s.weight is block-fp8; the attention "
                                        "program reads BF16: convert with --quant 'blk.*.%s.weight=cast:dtype=bf16' (an exact "
                                        "dequantisation)\n", (long long)l, (long long)l, pr.first, pr.first);
                        return RAD_E_UNSUPPORTED;
                    }
            } else {
                if (!aw.kvb) {
                    fprintf(stderr, "radiance: ernie55: layer %lld blk.%lld.attn_kv_b.weight must be cast to BF16 "
                                    "(--quant 'blk.*.attn_kv_b.weight=cast:dtype=bf16')\n", (long long)l, (long long)l);
                    return RAD_E_UNSUPPORTED;
                }
                if (!aw.q_qa.valid() || !aw.q_qb.valid() || !aw.q_kva.valid() || !aw.q_gate.valid() ||
                    !aw.q_o.valid() || !aw.q_iq.valid())
                    return RAD_E_INVAL;
            }
            if (!At.hc.fn || !At.hc.scale || !At.hc.base || !At.gain || !aw.qa_norm || !aw.kv_norm || !aw.u || !aw.v ||
                !aw.sink || !aw.ik || !aw.iw || !aw.ik_g || !aw.ik_b)
                return RAD_E_INVAL;
            const int64_t qw = a.nope + a.rope, cw = a.lat + a.rope, ah_loc = m.mla_heads_loc;
            const double scale = 1.0 / std::sqrt((double)qw);
            At.pre = rw(b, RAD_OP(b, "ernie_mhc_pre_rms",
                                  RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", H), RAD_INT("hc", hc),
                                             RAD_INT("iterations", f.iterations), RAD_F64("eps", f.rms_eps),
                                             RAD_F64("pre_eps", f.hc_eps), RAD_F64("sinkhorn_eps", f.hc_eps),
                                             RAD_F64("post_mult", 2.0), RAD_F64("rms_eps", f.rms_eps),
                                             RAD_STR("dtype", "bf16")),
                                  RAD_WEIGHTS(At.hc.fn, At.hc.scale, At.hc.base, At.gain)),
                        {m.h[0]}, {m.x, m.post, m.comb});
            auto rms = [&](int64_t n, rad_weight g) {
                return RAD_OP(b, "rmsnorm", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", n), RAD_F64("eps", f.rms_eps),
                                                       RAD_STR("dtype", "bf16")), RAD_WEIGHTS(g));
            };
            if (quant_fwd) {
                At.need_qx = aw.q_qa.fp8 || aw.q_kva.fp8;
                At.need_qqn = aw.q_qb.fp8 || aw.q_iq.fp8 || aw.q_gate.fp8;
                At.need_qog = aw.q_o.fp8;
                if (At.need_qx) RAD_ARCH_TRY(make_qop(m.x, T, m.qa_x, &At.qop_x));
                RAD_ARCH_TRY(make_qgemm(a.qlora, H, aw.q_qa, m.x, m.qa_x, m.qa, &At.q_qa));
                At.qa = At.q_qa.op;
                At.qn = rw(b, rms(a.qlora, aw.qa_norm), {m.qa}, {m.qn});
                if (At.need_qqn) RAD_ARCH_TRY(make_qop(m.qn, T, m.qa_qn, &At.qop_qn));
                RAD_ARCH_TRY(make_qgemm(ah_loc * qw, a.qlora, aw.q_qb, m.qn, m.qa_qn, m.q, &At.q_qb));
                At.qb = At.q_qb.op;
                RAD_ARCH_TRY(make_qgemm(cw, H, aw.q_kva, m.x, m.qa_x, m.kv, &At.q_kva));
                At.kva = At.q_kva.op;
            } else {
                At.qa = rw(b, gemm(a.qlora, H, aw.qa), {m.x}, {m.qa});
                At.qn = rw(b, rms(a.qlora, aw.qa_norm), {m.qa}, {m.qn});
                At.qb = rw(b, gemm(a.heads * qw, a.qlora, aw.qb), {m.qn}, {m.q});
                At.kva = rw(b, gemm(cw, H, aw.kva), {m.x}, {m.kv});
            }
            At.kvn = rw(b, rms(a.lat, aw.kv_norm), {m.kv}, {m.kv});
            auto rope = [&](int64_t heads, int64_t width) {
                return RAD_OP(b, "ernie_rope_tail",
                              RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("heads", heads), RAD_INT("width", width),
                                         RAD_INT("rope_dim", a.rope), RAD_F64("theta", a.theta),
                                         RAD_STR("rope_table", kRopeTable)),
                              RAD_NOWEIGHTS);
            };
            At.rq = rw(b, rope(ah_loc, qw), {m.q}, {m.q});
            At.rk = rw(b, rope(1, cw), {m.kv}, {m.kv});
            /* The latent cache: one RAD_KV_FULL group, row = [normed latent | rotated k_pe] (576 bf16 in the release), as
             * bailing_hybrid's MLA keeps it. The first op after the group that carries a block size claims it. */
            if (!m.kvl) {
                RadKVGroupDecl d{};
                d.kind = RAD_KV_FULL; d.dtype = RAD_BF16; d.n_head_kv = 1; d.head_dim = cw / 2;
                /* fp8_ds_mla: u8 rows of lat + 4 * (lat / qb) + 2 * rope bytes (656 at the release), declared as head_dim = bytes / 2 (K and V halves,
                 * as the indexer's 132-byte rows are) */
                if (m.lat_fp8) { d.dtype = RAD_U8; d.head_dim = (a.lat + m.lat_lo + 4 * (a.lat / m.lat_qb) + 2 * a.rope) / 2; }
                m.kvl = rad_decl_kv_group(b, nm.f("kv_latent"), &d);
                if (!m.kvl) return RAD_E_INVAL;
            }
            RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, m.kvl));
            if (m.lat_fp8 && !m.lat_i8)
                At.store = rw(b, RAD_OP(b, "ernie_mla_latent_store",
                                        RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("latent_dim", a.lat), RAD_INT("rope_dim", a.rope),
                                                   RAD_INT("block_size", a.block), RAD_INT("quant_block", m.lat_qb)),
                                        RAD_NOWEIGHTS), {m.kv}, {});
            else if (m.lat_i8)
                At.store = rw(b, RAD_OP(b, "ernie_mla_latent_store",
                                        RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("latent_dim", a.lat), RAD_INT("rope_dim", a.rope),
                                                   RAD_INT("block_size", a.block), RAD_INT("quant_block", m.lat_qb), RAD_STR("kv_layout", m.lat_lay)),
                                        RAD_NOWEIGHTS), {m.kv}, {});
            else
            At.store = rw(b, RAD_OP(b, "kv_store",
                                    RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", cw / 2), RAD_INT("n_head_kv", 1),
                                               RAD_INT("block_size", a.block), RAD_STR("kv_dtype", "bf16")),
                                    RAD_NOWEIGHTS), {m.kv}, {});
            /* Ops are declared in execution order: the buffer planner reads liveness off the declaration order of rw(). */
            At.ik = rw(b, gemm(a.id, H, aw.ik), {m.x}, {m.ik});
            At.iw = rw(b, gemm(a.ih, H, aw.iw), {m.x}, {m.iw});
            if (quant_fwd) {
                RAD_ARCH_TRY(make_qgemm(a.ih * a.id, a.qlora, aw.q_iq, m.qn, m.qa_qn, m.iq, &At.q_iq));
                At.iq = At.q_iq.op;
            } else {
                At.iq = rw(b, gemm(a.ih * a.id, a.qlora, aw.iq), {m.qn}, {m.iq});
            }
            /* The indexer's key cache: a second group of u8 rows, 128 FP8 codes then the fp32 scale (132 bytes), one row a token. */
            if (!m.kvi) {
                RadKVGroupDecl d{};
                d.kind = RAD_KV_FULL; d.dtype = RAD_U8; d.n_head_kv = 1; d.head_dim = (a.id + 4) / 2;
                m.kvi = rad_decl_kv_group(b, nm.f("kv_dsa_index"), &d);
                if (!m.kvi) return RAD_E_INVAL;
                RAD_ARCH_TRY(rad_kv_group_dense_read(b, m.kvi));
            }
            RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, m.kvi));
            At.ikey = rw(b, RAD_OP(b, "ernie_idx_key",
                                   RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", a.id), RAD_INT("rope_dim", a.rope),
                                              RAD_F64("theta", a.theta), RAD_STR("rope_table", kRopeTable),
                                              RAD_INT("block_size", a.block)),
                                   RAD_WEIGHTS(aw.ik_g, aw.ik_b)), {m.ik}, {});
            if (!At.store || !At.ikey) return RAD_E_INVAL;
            if (rad_kv_block_size(b, m.kvl) != a.block || rad_kv_block_size(b, m.kvi) != a.block) {
                fprintf(stderr, "radiance: ernie55: a KV block is not %lld tokens (latent %lld, index %lld)\n",
                        (long long)a.block, (long long)rad_kv_block_size(b, m.kvl), (long long)rad_kv_block_size(b, m.kvi));
                return RAD_E_INVAL;
            }
            At.iqn = rw(b, RAD_OP(b, "ernie_idx_query",
                                  RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("heads", a.ih), RAD_INT("head_dim", a.id),
                                             RAD_INT("rope_dim", a.rope), RAD_F64("theta", a.theta),
                                             RAD_STR("rope_table", kRopeTable)),
                                  RAD_NOWEIGHTS), {m.iq}, {m.qf, m.qs});
            if (a.idx_tile > 0) {
                At.topk_tiled = rw(b, RAD_OP(b, "ernie_dsa_topk_tiled",
                                             RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("heads", a.ih), RAD_INT("head_dim", a.id),
                                                        RAD_INT("topk", a.topk), RAD_INT("window", a.window),
                                                        RAD_INT("block_size", a.block), RAD_INT("tokens", m.lmax)),
                                             RAD_NOWEIGHTS), {m.qf, m.qs, m.iw, m.req}, {m.sc, m.aids, m.sl, m.topk_state});
                At.logit = At.topk_tiled;
            } else {
            At.logit = rw(b, RAD_OP(b, "ernie_idx_logits",
                                    RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("heads", a.ih), RAD_INT("head_dim", a.id),
                                               RAD_INT("window", a.window)),
                                    RAD_NOWEIGHTS), {m.qf, m.qs, m.iw, m.req}, {m.sc});
            At.select = rw(b, RAD_OP(b, "ernie_dsa_select",
                                     RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("topk", a.topk), RAD_INT("window", a.window),
                                                RAD_INT("block_size", a.block)),
                                     RAD_NOWEIGHTS), {m.sc, m.req}, {m.aids, m.sl});
            }
            At.slice = rw(b, RAD_OP(b, "ernie_head_slice",
                                    RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("heads", ah_loc), RAD_INT("width", qw),
                                               RAD_INT("offset", a.nope), RAD_INT("count", a.rope)),
                                    RAD_NOWEIGHTS), {m.q}, {m.qr});
            At.absorb = rw(b, RAD_OP(b, "ernie_mla_project",
                                     RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("heads", ah_loc), RAD_INT("k_dim", a.nope),
                                                RAD_INT("v_dim", a.vdim), RAD_INT("latent_dim", a.lat),
                                                RAD_STR("mode", "absorb"), RAD_INT("q_width", qw)),
                                     RAD_WEIGHTS(aw.kvb)), {m.q}, {m.qabs});
            if (m.lat_fp8)
                At.attn = rw(b, RAD_OP(b, "ernie_mla_sparse_attn",
                                       RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("heads", ah_loc), RAD_INT("latent_dim", a.lat),
                                                  RAD_INT("rope_dim", a.rope), RAD_INT("block_size", a.block),
                                                  RAD_INT("width", a.topk + a.window), RAD_F64("scale", scale),
                                                  RAD_STR("kv_layout", m.lat_i8 ? m.lat_lay : "fp8_ds_mla"), RAD_INT("quant_block", m.lat_qb)),
                                       RAD_WEIGHTS(aw.sink)), {m.qabs, m.qr, m.sl}, {m.ol});
            else
            At.attn = rw(b, RAD_OP(b, "ernie_mla_sparse_attn",
                                   RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("heads", ah_loc), RAD_INT("latent_dim", a.lat),
                                              RAD_INT("rope_dim", a.rope), RAD_INT("block_size", a.block),
                                              RAD_INT("width", a.topk + a.window), RAD_F64("scale", scale)),
                                   RAD_WEIGHTS(aw.sink)), {m.qabs, m.qr, m.sl}, {m.ol});
            At.expand = rw(b, RAD_OP(b, "ernie_mla_project",
                                     RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("heads", ah_loc), RAD_INT("k_dim", a.nope),
                                                RAD_INT("v_dim", a.vdim), RAD_INT("latent_dim", a.lat),
                                                RAD_STR("mode", "expand")),
                                     RAD_WEIGHTS(aw.kvb)), {m.ol}, {m.ov});
            if (world > 1 && m.oz) {
                At.vha_rank = aw.rank;
                At.vha_proj = rw(b, RAD_OP(b, "ernie_vha_proj",
                                           RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("heads", a.heads), RAD_INT("dim", a.vdim),
                                                      RAD_INT("rank", aw.rank), RAD_INT("world_size", world), RAD_INT("tp_rank", rank)),
                                           RAD_WEIGHTS(aw.u)), {m.ov}, {m.oz});
                At.vha_ar = make_ar(aw.rank * a.vdim, m.oz);
                At.vha_apply = rw(b, RAD_OP(b, "ernie_vha_apply",
                                            RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("heads", a.heads), RAD_INT("dim", a.vdim),
                                                       RAD_INT("rank", aw.rank), RAD_INT("world_size", world), RAD_INT("tp_rank", rank)),
                                            RAD_WEIGHTS(aw.v)), {m.ov, m.oz}, {m.om});
                if (!At.vha_proj || !At.vha_ar || !At.vha_apply) return RAD_E_INVAL;
                if (m.ov_all && m.mla_vha_dec_max > 0) {
                    At.vha_ag = make_ag_max(ah_loc * a.vdim, m.mla_vha_dec_max, m.ov, m.ov_all);
                    At.vha = rw(b, RAD_OP(b, "ernie_vha_mix",
                                          RAD_PARAMS(RAD_RANGE("M", 1, m.mla_vha_dec_max), RAD_INT("heads", a.heads), RAD_INT("dim", a.vdim),
                                                     RAD_INT("rank", aw.rank), RAD_STR("fold", kVhaFold),
                                                     RAD_INT("world_size", world), RAD_INT("tp_rank", rank)),
                                          RAD_WEIGHTS(aw.u, aw.v)), {m.ov_all}, {m.om});
                    if (!At.vha_ag || !At.vha) return RAD_E_INVAL;
                }
            } else if (world > 1) {
                At.vha_ag = make_ag(ah_loc * a.vdim, m.ov, m.ov_all);
                if (!At.vha_ag) return RAD_E_INVAL;
                At.vha = rw(b, RAD_OP(b, "ernie_vha_mix",
                                      RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("heads", a.heads), RAD_INT("dim", a.vdim),
                                                 RAD_INT("rank", aw.rank), RAD_STR("fold", kVhaFold),
                                                 RAD_INT("world_size", world), RAD_INT("tp_rank", rank)),
                                      RAD_WEIGHTS(aw.u, aw.v)), {m.ov_all}, {m.om});
            } else {
                At.vha = rw(b, RAD_OP(b, "ernie_vha_mix",
                                      RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("heads", a.heads), RAD_INT("dim", a.vdim),
                                                 RAD_INT("rank", aw.rank), RAD_STR("fold", kVhaFold)),
                                      RAD_WEIGHTS(aw.u, aw.v)), {m.ov}, {m.om});
            }
            if (quant_fwd) {
                RAD_ARCH_TRY(make_qgemm(ah_loc * a.vdim, a.qlora, aw.q_gate, m.qn, m.qa_qn, m.gt, &At.q_gate));
                At.gate = At.q_gate.op;
                At.gmul = rw(b, RAD_OP(b, "act_mul",
                                       RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", ah_loc * a.vdim), RAD_STR("act", "sigmoid"),
                                                  RAD_STR("dtype", "bf16")),
                                       RAD_NOWEIGHTS), {m.gt, m.om}, {m.og});
                if (At.need_qog) RAD_ARCH_TRY(make_qop(m.og, T, m.qa_og, &At.qop_og));
                RAD_ARCH_TRY(make_qgemm(H, ah_loc * a.vdim, aw.q_o, m.og, m.qa_og, m.y, &At.q_out));
                At.out = At.q_out.op;
                if (world > 1) {
                    At.out_ar = make_ar(H, m.y);
                    if (!At.out_ar) return RAD_E_INVAL;
                }
            } else {
                At.gate = rw(b, gemm(a.heads * a.vdim, a.qlora, aw.gate), {m.qn}, {m.gt});
                At.gmul = rw(b, RAD_OP(b, "act_mul",
                                       RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", a.heads * a.vdim), RAD_STR("act", "sigmoid"),
                                                  RAD_STR("dtype", "bf16")),
                                       RAD_NOWEIGHTS), {m.gt, m.om}, {m.og});
                At.out = rw(b, gemm(H, a.heads * a.vdim, aw.o), {m.og}, {m.y});
            }
            At.post = rw(b, RAD_OP(b, "mhc_post",
                                   RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", H), RAD_INT("hc", hc),
                                              RAD_STR("dtype", "bf16")),
                                   RAD_NOWEIGHTS), {m.y, m.h[0], m.post, m.comb}, {m.h[1]});
            if (!At.pre || !At.qa || !At.qn || !At.qb || !At.kva || !At.kvn || !At.rq || !At.rk || !At.ik || !At.iw ||
                !At.iq || !At.iqn || !At.logit || (!At.select && !At.topk_tiled) || !At.slice || !At.absorb || !At.attn || !At.expand ||
                (!At.vha && !At.vha_apply) || !At.gate || !At.gmul || !At.out || !At.post)
                return RAD_E_INVAL;
        }
        y.pre = rw(b, RAD_OP(b, "ernie_mhc_pre_rms",
                             RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", H), RAD_INT("hc", hc),
                                        RAD_INT("iterations", f.iterations),
                                        RAD_F64("eps", f.rms_eps), RAD_F64("pre_eps", f.hc_eps),
                                        RAD_F64("sinkhorn_eps", f.hc_eps),
                                        RAD_F64("post_mult", 2.0), RAD_F64("rms_eps", f.rms_eps),
                                        RAD_STR("dtype", "bf16")),
                             RAD_WEIGHTS(y.fn, y.scale, y.base, y.gain)),
                   {cur}, {m.x, m.post, m.comb});
        y.routed = m.moe && l >= c.first_dense;
        if (quant_fwd && y.routed) {
            /* Routed quantised layer: interleave the shared arm and routed arm declarations in the
             * exact order step() issues them so buffer lifetimes cover both arms without aliasing
             * (letting Ctx::haz_mark overlap shared GEMMs with routed GEMMs on idle CUs), and when
             * world > 1 with f32-scale FP8 down_proj combine shared down_ar [T, H] + routed comb_ar
             * [T, Lt] into one [T, H + Lt] all_reduce on m.ffn_ar. */
            RoutedOps& R = y.R;
            R.w = w.moe;
            const RoutedHandles& rh = R.w;
            if (!rh.q_in.valid() || !rh.q_out.valid()) return RAD_E_INVAL;
            if (!rh.mx) {
                for (int64_t e = 0; e < c.experts; ++e)
                    if (!rh.ge[(size_t)e] || !rh.ue[(size_t)e] || !rh.de[(size_t)e])
                        return RAD_E_UNSUPPORTED;
            }
            if (!rh.r0 || !rh.r1 || !rh.bias || !rh.scale || !rh.norm) return RAD_E_INVAL;
            const int64_t E = c.experts, K = c.top_k, Lt = c.latent, Fe = m.Fe, P = T * K;
            R.combine_ar = world > 1 && w.q_down.fp8 && w.q_down.scale_dt == RAD_F32 &&
                           (H % 8) == 0 && (Lt % 8) == 0;
            const rad_buf ys_buf = R.combine_ar ? m.ffn_ar : m.ys;
            const rad_buf comb_buf = R.combine_ar ? m.ffn_ar : m.comb2;
            auto table = [&](int64_t N, int64_t Kd, const char* order, const std::vector<rad_weight>& tab,
                             rad_buf a, rad_buf out) {
                rad_op h = rad_decl_op(b, "moe_gemm",
                                       RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("N", N), RAD_INT("K", Kd),
                                                  RAD_INT("n_expert", E), RAD_INT("top_k", K),
                                                  RAD_STR("a_order", order), RAD_STR("dtype", "bf16")),
                                       tab.data(), (int)tab.size());
                return rw(b, h, {a, m.sorted, m.eoff}, {out});
            };
            auto table_mx = [&](int64_t N, int64_t Kd, const char* order, int64_t parts,
                                const std::vector<rad_weight>& c0, const std::vector<rad_weight>& s0,
                                const std::vector<rad_weight>& c1, const std::vector<rad_weight>& s1,
                                const QAct& qa, rad_buf out) {
                std::vector<rad_weight> tab;
                tab.reserve(c0.size() + s0.size() + c1.size() + s1.size());
                tab.insert(tab.end(), c0.begin(), c0.end());
                tab.insert(tab.end(), s0.begin(), s0.end());
                tab.insert(tab.end(), c1.begin(), c1.end());
                tab.insert(tab.end(), s1.begin(), s1.end());
                const RadParam ps[] = {
                    RAD_RANGE("M", 1, T), RAD_INT("N", N), RAD_INT("K", Kd),
                    RAD_INT("n_expert", E), RAD_INT("top_k", K), RAD_INT("group", 128),
                    RAD_STR("a_order", order), RAD_STR("dtype", "mxfp4a8"),
                    RAD_INT("N_odd", N), RAD_INT("K_odd", Kd), RAD_INT("parts", parts)
                };
                rad_op h = rad_decl_op(b, "moe_gemm_q", ps, (int)(sizeof(ps) / sizeof(*ps)),
                                       tab.data(), (int)tab.size());
                return rw(b, h, {qa.q, qa.s, m.sorted, m.eoff}, {out});
            };
            y.need_qx = w.q_gu.fp8 || rh.q_in.fp8;
            y.need_qact = w.q_down.fp8;
            R.need_qx = rh.q_in.fp8;
            if (y.need_qx) RAD_ARCH_TRY(make_qop(m.x, T, m.qa_x, &y.qop_x));
            R.r01 = rw(b, RAD_OP(b, "gemm_nt_f32_pair",
                                 RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("N", E), RAD_INT("K", H),
                                            RAD_STR("dtype", "bf16")),
                                 RAD_WEIGHTS(rh.r0, rh.r1)), {m.x}, {m.lg});
            R.r0 = R.r1 = R.r01;
            RAD_ARCH_TRY(make_qgemm(Lt, H, rh.q_in, m.x, m.qa_x, m.xl, &R.q_fc1));
            R.fc1 = R.q_fc1.op;
            RAD_ARCH_TRY(make_qgemm(2 * I, H, w.q_gu, m.x, m.qa_x, m.gu[y.kind], &y.q_gu));
            y.gate_op = y.up_op = y.q_gu.op;
            R.topk_scatter = rw(b, RAD_OP(b, "ernie_router_topk_scatter",
                                          RAD_PARAMS(RAD_INT("M", 1), RAD_INT("experts", E),
                                                     RAD_INT("topk", K), RAD_INT("renormalize", *renorm ? 1 : 0)),
                                          RAD_WEIGHTS(rh.bias, rh.scale)),
                                {m.lg}, {m.ids, m.rw, m.sorted, m.eoff, m.ecnt});
            R.topk = rw(b, RAD_OP(b, "ernie_router_topk",
                                  RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("experts", E),
                                             RAD_INT("topk", K), RAD_INT("renormalize", *renorm ? 1 : 0)),
                                  RAD_WEIGHTS(rh.bias, rh.scale)), {m.lg}, {m.ids, m.rw});
            R.scatter = rw(b, RAD_OP(b, "moe_scatter",
                                     RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n_expert", E),
                                                RAD_INT("top_k", K), RAD_INT("expert_base", 0)),
                                     RAD_NOWEIGHTS), {m.ids}, {m.sorted, m.eoff, m.ecnt});
            if (rh.mx) {
                if (!m.qa_xl.valid() || !m.qa_eact.valid()) return RAD_E_SHAPE;
                RAD_ARCH_TRY(make_qop(m.xl, T, m.qa_xl, &R.qop_xl));
            }
            y.act = rw(b, RAD_OP(b, "ernie_situ_mul",
                                 RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", I),
                                            RAD_STR("dtype", "bf16"), RAD_F64("beta", f.beta),
                                            RAD_F64("linear_beta", f.linear_beta)),
                                 RAD_NOWEIGHTS), {m.gu[y.kind]}, {m.act[y.kind]});
            if (y.need_qact) {
                if ((I % 4) == 0) {
                    y.act_q = rw(b, RAD_OP(b, "situ_mul_quant_fp8",
                                           RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", I), RAD_INT("group", 128),
                                                      RAD_STR("dtype", "bf16"), RAD_F64("beta", f.beta),
                                                      RAD_F64("linear_beta", f.linear_beta)),
                                           RAD_NOWEIGHTS), {m.gu[y.kind]}, {m.qa_act[y.kind].q, m.qa_act[y.kind].s});
                    if (!y.act_q) return RAD_E_INVAL;
                } else {
                    RAD_ARCH_TRY(make_qop(m.act[y.kind], T, m.qa_act[y.kind], &y.qop_act));
                }
            }
            if (rh.mx) {
                R.gu_q = table_mx(2 * Fe, Lt, "token", 2, rh.gu_c0, rh.gu_s0, rh.gu_c1, rh.gu_s1, m.qa_xl, m.egu);
                R.gate = R.up = R.gu_q;
            } else {
                R.gate = table(Fe, Lt, "token", std::vector<rad_weight>(rh.ge), m.xl, m.egu);
                R.up = table(Fe, Lt, "token", std::vector<rad_weight>(rh.ue), m.xl, m.egu);
            }
            RAD_ARCH_TRY(make_qgemm(H, I, w.q_down, m.act[y.kind], m.qa_act[y.kind], ys_buf, &y.q_down));
            y.down_op = y.q_down.op;
            if (rh.mx && (Fe % 4) == 0) {
                R.act_q = rw(b, RAD_OP(b, "situ_mul_quant_fp8",
                                       RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", Fe), RAD_INT("group", 128),
                                                  RAD_STR("dtype", "bf16"), RAD_F64("beta", f.beta),
                                                  RAD_F64("linear_beta", f.linear_beta)),
                                       RAD_NOWEIGHTS), {m.egu}, {m.qa_eact.q, m.qa_eact.s});
                R.act = R.act_q;
                R.dn_q = table_mx(Lt, Fe, "sorted", 1, rh.dn_c0, rh.dn_s0, rh.dn_c1, rh.dn_s1, m.qa_eact, m.edn);
                R.down = R.dn_q;
            } else {
                R.act = rw(b, RAD_OP(b, "ernie_situ_mul",
                                     RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", Fe), RAD_STR("dtype", "bf16"),
                                                RAD_F64("beta", f.beta), RAD_F64("linear_beta", f.linear_beta)),
                                     RAD_NOWEIGHTS), {m.egu}, {m.eact});
                if (rh.mx) {
                    RAD_ARCH_TRY(make_qop(m.eact, P, m.qa_eact, &R.qop_eact));
                    R.dn_q = table_mx(Lt, Fe, "sorted", 1, rh.dn_c0, rh.dn_s0, rh.dn_c1, rh.dn_s1, m.qa_eact, m.edn);
                    R.down = R.dn_q;
                } else {
                    R.down = table(Lt, Fe, "sorted", std::vector<rad_weight>(rh.de), m.eact, m.edn);
                }
            }
            R.gather = rw(b, RAD_OP(b, "moe_gather",
                                    RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", Lt), RAD_INT("top_k", K),
                                               RAD_STR("dtype", "bf16"), RAD_STR("act", "none")),
                                    RAD_NOWEIGHTS), {m.edn, m.rw, m.sorted}, {comb_buf});
            if (R.combine_ar) {
                R.ffn_ar = make_ar(H + Lt, m.ffn_ar);
                if (!R.ffn_ar) return RAD_E_INVAL;
            } else if (world > 1) {
                y.down_ar = make_ar(H, m.ys);
                R.comb_ar = make_ar(Lt, m.comb2);
                if (!y.down_ar || !R.comb_ar) return RAD_E_INVAL;
            }
            R.norm = rw(b, RAD_OP(b, "rmsnorm",
                                  RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", Lt),
                                             RAD_F64("eps", f.rms_eps), RAD_STR("dtype", "bf16")),
                                  RAD_WEIGHTS(rh.norm)), {comb_buf}, {m.nl});
            R.need_qnl = rh.q_out.fp8;
            if (R.need_qnl) RAD_ARCH_TRY(make_qop(m.nl, T, m.qa_nl, &R.qop_nl));
            RAD_ARCH_TRY(make_qgemm(H, Lt, rh.q_out, m.nl, m.qa_nl, m.ro, &R.q_fc2));
            R.fc2 = R.q_fc2.op;
            R.add_post = rw(b, RAD_OP(b, "ernie_add_mhc_post",
                                      RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", H), RAD_INT("hc", hc),
                                                 RAD_STR("dtype", "bf16")),
                                      RAD_NOWEIGHTS), {m.ro, ys_buf, y.ffn_in, m.post, m.comb}, {y.ffn_out});
            R.add = R.add_post;
            if (!R.r01 || !R.topk_scatter || !R.topk || !R.scatter || !R.fc1 || !R.gate || !R.up || !R.act ||
                !R.down || !R.gather || !R.norm || !R.fc2 || !R.add_post)
                return RAD_E_INVAL;
        } else {
            if (quant_fwd) {
                y.need_qx = w.q_gu.fp8;
                y.need_qact = w.q_down.fp8;
                if (y.need_qx) RAD_ARCH_TRY(make_qop(m.x, T, m.qa_x, &y.qop_x));
                RAD_ARCH_TRY(make_qgemm(2 * I, H, w.q_gu, m.x, m.qa_x, m.gu[y.kind], &y.q_gu));
                y.gate_op = y.up_op = y.q_gu.op;
                y.act = rw(b, RAD_OP(b, "ernie_situ_mul",
                                     RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", I),
                                                RAD_STR("dtype", "bf16"), RAD_F64("beta", f.beta),
                                                RAD_F64("linear_beta", f.linear_beta)),
                                     RAD_NOWEIGHTS), {m.gu[y.kind]}, {m.act[y.kind]});
                if (y.need_qact) {
                    if ((I % 4) == 0) {
                        y.act_q = rw(b, RAD_OP(b, "situ_mul_quant_fp8",
                                               RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", I), RAD_INT("group", 128),
                                                          RAD_STR("dtype", "bf16"), RAD_F64("beta", f.beta),
                                                          RAD_F64("linear_beta", f.linear_beta)),
                                               RAD_NOWEIGHTS), {m.gu[y.kind]}, {m.qa_act[y.kind].q, m.qa_act[y.kind].s});
                        if (!y.act_q) return RAD_E_INVAL;
                    } else {
                        RAD_ARCH_TRY(make_qop(m.act[y.kind], T, m.qa_act[y.kind], &y.qop_act));
                    }
                }
                RAD_ARCH_TRY(make_qgemm(H, I, w.q_down, m.act[y.kind], m.qa_act[y.kind], m.y, &y.q_down));
                y.down_op = y.q_down.op;
                if (world > 1) {
                    y.down_ar = make_ar(H, m.y);
                    if (!y.down_ar) return RAD_E_INVAL;
                }
            } else {
                y.gate_op = rw(b, gemm(I, H, y.gate), {m.x}, {m.gu[y.kind]});
                y.up_op = rw(b, gemm(I, H, y.up), {m.x}, {m.gu[y.kind]});
                y.act = rw(b, RAD_OP(b, "ernie_situ_mul",
                                     RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", I),
                                                RAD_STR("dtype", "bf16"), RAD_F64("beta", f.beta),
                                                RAD_F64("linear_beta", f.linear_beta)),
                                     RAD_NOWEIGHTS), {m.gu[y.kind]}, {m.act[y.kind]});
                y.down_op = rw(b, gemm(H, I, y.down), {m.act[y.kind]}, {y.routed ? m.ys : m.y});
            }
            if (y.routed) {
                /* The routed branch (nvidia/moe.py Ernie5_5MoE.forward) reads the same normed x as the shared
                 * expert: two f32 routers, split-sigmoid top-k, fc1 to latent width, the experts there,
                 * the weighted combine, then latent_norm and fc2 (routed_output_transform) and the sum
                 * with the shared output. */
                RoutedOps& R = y.R;
                R.w = w.moe;
                const RoutedHandles& rh = R.w;
                if (!rh.in || !rh.out) {
                    fprintf(stderr, "radiance: ernie55: layer %lld latent projections are block-fp8; the "
                                    "2-8 program reads BF16 projections only\n", (long long)l);
                    return RAD_E_UNSUPPORTED;
                }
                for (int64_t e = 0; e < c.experts; ++e)
                    if (!rh.ge[(size_t)e] || !rh.ue[(size_t)e] || !rh.de[(size_t)e]) {
                        fprintf(stderr, "radiance: ernie55: layer %lld expert %lld is block-fp8; the 2-8 "
                                        "program reads BF16 experts only\n", (long long)l, (long long)e);
                        return RAD_E_UNSUPPORTED;
                    } else if ((e && (rh.ge[(size_t)e] != rh.ge[0] + e || rh.ue[(size_t)e] != rh.ue[0] + e ||
                                      rh.de[(size_t)e] != rh.de[0] + e)))
                        return RAD_E_INVAL;
                if (!rh.r0 || !rh.r1 || !rh.bias || !rh.scale || !rh.norm) return RAD_E_INVAL;
                const int64_t E = c.experts, K = c.top_k, Lt = c.latent, Fe = m.Fe, P = T * K;
                auto router = [&](rad_weight wt) {
                    return rw(b, RAD_OP(b, "gemm_nt_f32",
                                        RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("N", E), RAD_INT("K", H),
                                                   RAD_STR("dtype", "bf16")),
                                        RAD_WEIGHTS(wt)), {m.x}, {m.lg});
                };
                auto table = [&](int64_t N, int64_t Kd, const char* order, const std::vector<rad_weight>& tab,
                                 rad_buf a, rad_buf out) {
                    rad_op h = rad_decl_op(b, "moe_gemm",
                                           RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("N", N), RAD_INT("K", Kd),
                                                      RAD_INT("n_expert", E), RAD_INT("top_k", K),
                                                      RAD_STR("a_order", order), RAD_STR("dtype", "bf16")),
                                           tab.data(), (int)tab.size());
                    return rw(b, h, {a, m.sorted, m.eoff}, {out});
                };
                R.r0 = router(rh.r0);
                R.r1 = router(rh.r1);
                R.topk = rw(b, RAD_OP(b, "ernie_router_topk",
                                      RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("experts", E),
                                                 RAD_INT("topk", K), RAD_INT("renormalize", *renorm ? 1 : 0)),
                                      RAD_WEIGHTS(rh.bias, rh.scale)), {m.lg}, {m.ids, m.rw});
                R.scatter = rw(b, RAD_OP(b, "moe_scatter",
                                         RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n_expert", E),
                                                    RAD_INT("top_k", K), RAD_INT("expert_base", 0)),
                                         RAD_NOWEIGHTS), {m.ids}, {m.sorted, m.eoff, m.ecnt});
                R.fc1 = rw(b, gemm(Lt, H, rh.in), {m.x}, {m.xl});
                R.gate = table(Fe, Lt, "token", std::vector<rad_weight>(rh.ge), m.xl, m.egu);
                R.up = table(Fe, Lt, "token", std::vector<rad_weight>(rh.ue), m.xl, m.egu);
                R.act = rw(b, RAD_OP(b, "ernie_situ_mul",
                                     RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", Fe), RAD_STR("dtype", "bf16"),
                                                RAD_F64("beta", f.beta), RAD_F64("linear_beta", f.linear_beta)),
                                     RAD_NOWEIGHTS), {m.egu}, {m.eact});
                R.down = table(Lt, Fe, "sorted", std::vector<rad_weight>(rh.de), m.eact, m.edn);
                R.gather = rw(b, RAD_OP(b, "moe_gather",
                                        RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", Lt), RAD_INT("top_k", K),
                                                   RAD_STR("dtype", "bf16"), RAD_STR("act", "none")),
                                        RAD_NOWEIGHTS), {m.edn, m.rw, m.sorted}, {m.comb2});
                R.norm = rw(b, RAD_OP(b, "rmsnorm",
                                      RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", Lt),
                                                 RAD_F64("eps", f.rms_eps), RAD_STR("dtype", "bf16")),
                                      RAD_WEIGHTS(rh.norm)), {m.comb2}, {m.nl});
                R.fc2 = rw(b, gemm(H, Lt, rh.out), {m.nl}, {m.ro});
                R.add = rw(b, RAD_OP(b, "add",
                                     RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", H), RAD_STR("dtype", "bf16")),
                                     RAD_NOWEIGHTS), {m.ro, m.ys}, {m.y});
                if (!R.r0 || !R.r1 || !R.topk || !R.scatter || !R.fc1 || !R.gate || !R.up || !R.act ||
                    !R.down || !R.gather || !R.norm || !R.fc2 || !R.add)
                    return RAD_E_INVAL;
            }
        }
        y.post = rw(b, RAD_OP(b, "mhc_post",
                              RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", H), RAD_INT("hc", hc),
                                         RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS), {m.y, cur, m.post, m.comb}, {nxt});
        if (!y.pre || !y.gate_op || !y.up_op || !y.act || !y.down_op || !y.post)
            return RAD_E_INVAL;
        if (ds_on) {
            for (int ti = 0; ti < (int)dsc->n_taps; ++ti) {
                if (dsc->target_layer_ids[(size_t)ti] == l) {
                    m.ds.tap_mean[(size_t)ti] = rw(b, RAD_OP(b, "ernie_mhc_stream_mean",
                        RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("hidden", H), RAD_INT("mult", hc)),
                        RAD_NOWEIGHTS), {nxt}, {m.ds.taps});
                    if (!m.ds.tap_mean[(size_t)ti]) return RAD_E_INVAL;
                }
            }
        }
    }
    if (target) {
        /* The head runs on the sampled rows only: it is per token, and those are the rows read. The
         * collapse eps is the final norm's (rms_norm_eps), the gate eps hc_eps (model.py: the
         * hc_head_fused kernel is called with rms_norm_eps, hc_eps). */
        const int64_t R = m.R;
        m.gather = rw(b, RAD_OP(b, "gather_rows",
                                RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", W), RAD_STR("dtype", "bf16")),
                                RAD_NOWEIGHTS), {m.h[m.fin]}, {m.hout});
        m.head = rw(b, RAD_OP(b, "ernie_mhc_head",
                              RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", H), RAD_INT("hc", hc),
                                         RAD_F64("eps", f.rms_eps), RAD_F64("gate_eps", f.hc_eps)),
                              RAD_WEIGHTS(m.head_fn, m.head_scale, m.head_base)), {m.hout}, {m.hx});
        m.norm = rw(b, RAD_OP(b, "rmsnorm",
                              RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", H), RAD_F64("eps", f.rms_eps),
                                         RAD_STR("dtype", "bf16")),
                              RAD_WEIGHTS(m.out_norm)), {m.hx}, {m.hn});
        m.proj = rw(b, RAD_OP(b, "logits_gemm",
                              RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n_vocab", m.Vloc), RAD_INT("n_embd", H),
                                         RAD_STR("dtype", "bf16")),
                              RAD_WEIGHTS(m.lm_head)), {m.hn}, {m.lgt});
        m.mmax = rw(b, RAD_OP(b, "ernie_multimax",
                              RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", m.Vloc), RAD_STR("dtype", "f32")),
                              RAD_WEIGHTS(m.mm_r, m.mm_t)), {m.lgt}, {m.lgt});
        if (!m.gather || !m.head || !m.norm || !m.proj || !m.mmax) return RAD_E_INVAL;
        if (ds_on) {
            DsparkProg& ds = m.ds;
            const HcaConfig& hh = m.Hc;
            const int64_t Dh = hh.D, hh_loc = m.hca_heads_loc, hg_loc = m.hca_groups_loc, GR = m.hca_GR_loc;
            const int64_t qrows = S * ds.block;
            const int64_t srows = S * ds.depth;
            if (qrows > T || srows > m.R) {
                fprintf(stderr, "radiance: ernie55: dspark requires max_tok (%lld) >= max_seqs*block (%lld) and R (%lld) >= srows (%lld)\n",
                        (long long)T, (long long)qrows, (long long)m.R, (long long)srows);
                return RAD_E_INVAL;
            }
            const double ds_theta = ac->theta; /* plain RoPE at rope_theta (8e6), NO YaRN */
            ds.feat_norm = dh->feat_norm;
            ds.head_fn = dh->head_fn; ds.head_base = dh->head_base; ds.head_scale = dh->head_scale;
            ds.out_norm = dh->out_norm; ds.mk1 = dh->mk1; ds.mk2 = dh->mk2;
            if (!ds.feat_norm || !ds.head_fn || !ds.head_base || !ds.head_scale || !ds.out_norm || !ds.mk1 || !ds.mk2)
                return RAD_E_INVAL;

            ds.ctx = decl_b(b, nm.f("dspark.ctx"), RAD_BF16, {T, H});
            ds.lgt_bf16 = decl_b(b, nm.f("dspark.lgt_bf16"), RAD_BF16, {srows, m.Vloc});
            ds.cand = decl_b(b, nm.f("dspark.cand"), RAD_I32, {srows, ds.top_k});
            ds.unary = decl_b(b, nm.f("dspark.unary"), RAD_F32, {srows, ds.top_k});
            if (world > 1) {
                ds.pair = decl_b(b, nm.f("dspark.pair"), RAD_F32, {srows, ds.top_k * 2});
                ds.gath = decl_b(b, nm.f("dspark.gath"), RAD_F32, {srows, (int64_t)world * ds.top_k * 2});
                if (!ds.pair || !ds.gath) return RAD_E_INVAL;
            }
            ds.tokens = decl_b(b, nm.f("dspark.tokens"), RAD_I32, {S, ds.depth}, RAD_BUF_PERSIST);
            if (!ds.ctx || !ds.lgt_bf16 || !ds.cand || !ds.unary || !ds.tokens) return RAD_E_INVAL;

            /* Dedicated paged KV group for the 3 draft layers (RAD_KV_FULL, or RAD_KV_WINDOW of ds.window under RADIANCE_ERNIE55_DSPARK_KV_WINDOW=1, bf16, 1 KV head, 512 dims as 2x256). */
            {
                const bool win = dspark_kv_window_from_env();
                if (win && ds.window < 1) {
                    fprintf(stderr, "radiance: ernie55: RADIANCE_ERNIE55_DSPARK_KV_WINDOW=1 needs a drafter window >= 1, got %lld\n", (long long)ds.window);
                    return RAD_E_INVAL;
                }
                RadKVGroupDecl d{};
                d.kind = win ? RAD_KV_WINDOW : RAD_KV_FULL;
                if (win) d.window = ds.window;
                d.dtype = RAD_BF16; d.n_head_kv = 1; d.head_dim = Dh / 2;
                if (win) {
                    static bool said = false;
                    if (!said) {
                        said = true;
                        fprintf(stderr, "radiance: ernie55: kv_dspark is a window group of %lld tokens (RADIANCE_ERNIE55_DSPARK_KV_WINDOW=1): the drafter cache no longer grows with the context\n", (long long)ds.window);
                    }
                }
                ds.kv_ds = rad_decl_kv_group(b, nm.f("kv_dspark"), &d);
                if (!ds.kv_ds) return RAD_E_INVAL;
            }
            for (int i = 0; i < (int)ds.n_layers; ++i)
                RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)c.main_layers + i, ds.kv_ds));

            /* ---- Context pass ops ---- */
            if (dh->feat_proj.fp8) {
                RAD_ARCH_TRY(make_qact("ds_taps", ds.taps, T, ds.n_taps * H, &ds.qa_taps));
                RAD_ARCH_TRY(make_qop(ds.taps, T, ds.qa_taps, &ds.qop_taps));
                ds.need_qtaps = true;
            }
            RAD_ARCH_TRY(make_qgemm(H, ds.n_taps * H, dh->feat_proj, ds.taps, ds.qa_taps, ds.ctx, &ds.q_fproj));
            ds.fnorm = rw(b, RAD_OP(b, "rmsnorm",
                                    RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", H), RAD_F64("eps", f.rms_eps),
                                               RAD_STR("dtype", "bf16")),
                                    RAD_WEIGHTS(ds.feat_norm)), {ds.ctx}, {ds.ctx});
            if (!ds.fnorm) return RAD_E_INVAL;
            for (int i = 0; i < (int)ds.n_layers; ++i)
                if (dh->layers[(size_t)i].hca.q_kv.fp8) ds.need_qctx = true;
            if (ds.need_qctx) RAD_ARCH_TRY(make_qop(ds.ctx, T, m.qa_x, &ds.qop_ctx));

            ds.ctx_kv.resize((size_t)ds.n_layers);
            ds.ctx_kvn.resize((size_t)ds.n_layers);
            ds.ctx_rk.resize((size_t)ds.n_layers);
            ds.ctx_store.resize((size_t)ds.n_layers);
            ds.ctx_kv_norm.resize((size_t)ds.n_layers);
            for (int i = 0; i < (int)ds.n_layers; ++i) {
                const HcaHandles& hw = dh->layers[(size_t)i].hca;
                ds.ctx_kv_norm[(size_t)i] = hw.kv_norm;
                RAD_ARCH_TRY(make_qgemm(Dh, H, hw.q_kv, ds.ctx, m.qa_x, m.hkv, &ds.ctx_kv[(size_t)i]));
                ds.ctx_kvn[(size_t)i] = rw(b, RAD_OP(b, "rmsnorm",
                                                     RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", Dh),
                                                                RAD_F64("eps", f.rms_eps), RAD_STR("dtype", "bf16")),
                                                     RAD_WEIGHTS(hw.kv_norm)), {m.hkv}, {m.hkv});
                ds.ctx_rk[(size_t)i] = rw(b, RAD_OP(b, "ernie_rope_tail",
                                                    RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("heads", 1), RAD_INT("width", Dh),
                                                               RAD_INT("rope_dim", hh.rope), RAD_F64("theta", ds_theta),
                                                               RAD_F64("yarn_factor", 0.0)),
                                                    RAD_NOWEIGHTS), {m.hkv}, {m.hkv});
                ds.ctx_store[(size_t)i] = rw(b, RAD_OP(b, "kv_store",
                                                       RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", Dh / 2),
                                                                  RAD_INT("n_head_kv", 1), RAD_INT("block_size", hh.block),
                                                                  RAD_STR("kv_dtype", "bf16")),
                                                       RAD_NOWEIGHTS), {m.hkv}, {});
                if (!ds.ctx_kvn[(size_t)i] || !ds.ctx_rk[(size_t)i] || !ds.ctx_store[(size_t)i]) return RAD_E_INVAL;
            }
            if (rad_kv_block_size(b, ds.kv_ds) != hh.block) return RAD_E_INVAL;

            /* ---- Query pass ops (declared in exact execution order for buffer planner liveness) ---- */
            ds.embed = rw(b, RAD_OP(b, "embed_lookup",
                                    RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n_embd", H),
                                               RAD_INT("n_vocab", m.Vloc), RAD_STR("dtype", "bf16"),
                                               RAD_INT("vocab_offset", m.Voff), RAD_F64("mult", 1.0)),
                                    RAD_WEIGHTS(m.tok)), {}, {m.emb});
            if (world > 1) {
                ds.embed_ar = make_ar(H, m.emb);
                if (!ds.embed_ar) return RAD_E_INVAL;
            }
            ds.enter = rw(b, RAD_OP(b, "hc_enter",
                                    RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", H), RAD_INT("hc", hc),
                                               RAD_STR("dtype", "bf16")),
                                    RAD_NOWEIGHTS), {m.emb}, {m.h[0]});
            if (!ds.embed || !ds.enter) return RAD_E_INVAL;

            ds.L.resize((size_t)ds.n_layers);
            for (int i = 0; i < (int)ds.n_layers; ++i) {
                FfnLayer& dy = ds.L[(size_t)i];
                const FfnHandles::Layer& dw = dh->layers[(size_t)i];
                dy.hca = true; dy.routed = true; dy.kind = 1; dy.width = widths[1];
                dy.ffn_in = m.h[1]; dy.ffn_out = m.h[0];
                dy.fn = dw.hc.fn; dy.scale = dw.hc.scale; dy.base = dw.hc.base; dy.gain = dw.gain;

                /* Uncompressed SWA HCA attention with plain RoPE at rope_theta (8e6) */
                HcaLayer& Hc = dy.Hc;
                const HcaHandles& hw = dw.hca;
                Hc.w = hw; Hc.hc = dw.attn_hc; Hc.gain = dw.attn_gain;
                if (!hw.og || !hw.q_qa.valid() || !hw.q_qb.valid() || !hw.q_kv.valid() || !hw.q_gate.valid() || !hw.q_o.valid() ||
                    !Hc.hc.fn || !Hc.hc.scale || !Hc.hc.base || !Hc.gain || !hw.qa_norm || !hw.kv_norm || !hw.u || !hw.v || !hw.sink)
                    return RAD_E_INVAL;

                Hc.pre = rw(b, RAD_OP(b, "ernie_mhc_pre_rms",
                                      RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", H), RAD_INT("hc", hc),
                                                 RAD_INT("iterations", f.iterations), RAD_F64("eps", f.rms_eps),
                                                 RAD_F64("pre_eps", f.hc_eps), RAD_F64("sinkhorn_eps", f.hc_eps),
                                                 RAD_F64("post_mult", 2.0), RAD_F64("rms_eps", f.rms_eps),
                                                 RAD_STR("dtype", "bf16")),
                                      RAD_WEIGHTS(Hc.hc.fn, Hc.hc.scale, Hc.hc.base, Hc.gain)),
                            {m.h[0]}, {m.x, m.post, m.comb});
                auto drms = [&](int64_t n, rad_weight g) {
                    return RAD_OP(b, "rmsnorm", RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", n),
                                                           RAD_F64("eps", f.rms_eps), RAD_STR("dtype", "bf16")),
                                  RAD_WEIGHTS(g));
                };
                Hc.need_qx = hw.q_qa.fp8 || hw.q_kv.fp8;
                Hc.need_qhqn = hw.q_qb.fp8 || hw.q_gate.fp8;
                Hc.need_qhog = hw.q_o.fp8;
                if (Hc.need_qx) RAD_ARCH_TRY(make_qop(m.x, T, m.qa_x, &Hc.qop_x));
                RAD_ARCH_TRY(make_qgemm(hh.qlora, H, hw.q_qa, m.x, m.qa_x, m.hqa, &Hc.q_qa));
                Hc.qa = Hc.q_qa.op;
                Hc.qn = rw(b, drms(hh.qlora, hw.qa_norm), {m.hqa}, {m.hqn});
                if (Hc.need_qhqn) RAD_ARCH_TRY(make_qop(m.hqn, T, m.qa_hqn, &Hc.qop_hqn));
                RAD_ARCH_TRY(make_qgemm(hh_loc * Dh, hh.qlora, hw.q_qb, m.hqn, m.qa_hqn, m.hq, &Hc.q_qb));
                Hc.qb = Hc.q_qb.op;
                Hc.qp = rw(b, RAD_OP(b, "ernie_hca_q_prep",
                                     RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("heads", hh_loc), RAD_INT("head_dim", Dh),
                                                RAD_INT("rope_dim", hh.rope), RAD_F64("eps", f.rms_eps),
                                                RAD_F64("theta", ds_theta), RAD_F64("yarn_factor", 0.0)),
                                     RAD_NOWEIGHTS), {m.hq}, {m.hq});
                RAD_ARCH_TRY(make_qgemm(Dh, H, hw.q_kv, m.x, m.qa_x, m.hkv, &Hc.q_kva));
                Hc.kva = Hc.q_kva.op;
                Hc.kvn = rw(b, drms(Dh, hw.kv_norm), {m.hkv}, {m.hkv});
                Hc.rk = rw(b, RAD_OP(b, "ernie_rope_tail",
                                     RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("heads", 1), RAD_INT("width", Dh),
                                                RAD_INT("rope_dim", hh.rope), RAD_F64("theta", ds_theta),
                                                RAD_F64("yarn_factor", 0.0)),
                                     RAD_NOWEIGHTS), {m.hkv}, {m.hkv});
                Hc.store = rw(b, RAD_OP(b, "kv_store",
                                        RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("head_dim", Dh / 2),
                                                   RAD_INT("n_head_kv", 1), RAD_INT("block_size", hh.block),
                                                   RAD_STR("kv_dtype", "bf16")),
                                        RAD_NOWEIGHTS), {m.hkv}, {});
                Hc.att = rw(b, RAD_OP(b, "ernie_hca_dspark_attn",
                                      RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("heads", hh_loc), RAD_INT("head_dim", Dh),
                                                 RAD_INT("window", ds.window), RAD_F64("scale", 1.0 / std::sqrt((double)Dh))),
                                      RAD_WEIGHTS(hw.sink)), {m.hq}, {m.hol});
                if (world > 1 && m.hz) {
                    Hc.vha_rank = hw.rank;
                    Hc.vha_proj = rw(b, RAD_OP(b, "ernie_vha_proj",
                                               RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("heads", hh.heads), RAD_INT("dim", Dh),
                                                          RAD_INT("rank", hw.rank), RAD_INT("world_size", world), RAD_INT("tp_rank", rank)),
                                               RAD_WEIGHTS(hw.u)), {m.hol}, {m.hz});
                    Hc.vha_ar = make_ar(hw.rank * Dh, m.hz);
                    Hc.vha_apply = rw(b, RAD_OP(b, "ernie_vha_apply",
                                                RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("heads", hh.heads), RAD_INT("dim", Dh),
                                                           RAD_INT("rank", hw.rank), RAD_INT("world_size", world), RAD_INT("tp_rank", rank)),
                                                RAD_WEIGHTS(hw.v)), {m.hol, m.hz}, {m.hom});
                    if (!Hc.vha_proj || !Hc.vha_ar || !Hc.vha_apply) return RAD_E_INVAL;
                    if (m.hol_all && m.hca_vha_dec_max > 0) {
                        Hc.vha_ag = make_ag_max(hh_loc * Dh, m.hca_vha_dec_max, m.hol, m.hol_all);
                        Hc.vha = rw(b, RAD_OP(b, "ernie_vha_mix",
                                              RAD_PARAMS(RAD_RANGE("M", 1, m.hca_vha_dec_max), RAD_INT("heads", hh.heads), RAD_INT("dim", Dh),
                                                         RAD_INT("rank", hw.rank), RAD_STR("fold", kVhaFold),
                                                         RAD_INT("world_size", world), RAD_INT("tp_rank", rank)),
                                              RAD_WEIGHTS(hw.u, hw.v)), {m.hol_all}, {m.hom});
                        if (!Hc.vha_ag || !Hc.vha) return RAD_E_INVAL;
                    }
                } else if (world > 1) {
                    Hc.vha_ag = make_ag(hh_loc * Dh, m.hol, m.hol_all);
                    if (!Hc.vha_ag) return RAD_E_INVAL;
                    Hc.vha = rw(b, RAD_OP(b, "ernie_vha_mix",
                                          RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("heads", hh.heads), RAD_INT("dim", Dh),
                                                     RAD_INT("rank", hw.rank), RAD_STR("fold", kVhaFold),
                                                     RAD_INT("world_size", world), RAD_INT("tp_rank", rank)),
                                          RAD_WEIGHTS(hw.u, hw.v)), {m.hol_all}, {m.hom});
                } else {
                    Hc.vha = rw(b, RAD_OP(b, "ernie_vha_mix",
                                          RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("heads", hh.heads), RAD_INT("dim", Dh),
                                                     RAD_INT("rank", hw.rank), RAD_STR("fold", kVhaFold)),
                                          RAD_WEIGHTS(hw.u, hw.v)), {m.hol}, {m.hom});
                }
                Hc.wo = rw(b, RAD_OP(b, "ernie_hca_wo_a",
                                     RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("heads", hh_loc), RAD_INT("head_dim", Dh),
                                                RAD_INT("rope_dim", hh.rope), RAD_INT("groups", hg_loc), RAD_INT("rank", hh.rank),
                                                RAD_F64("theta", ds_theta), RAD_F64("yarn_factor", 0.0)),
                                     RAD_WEIGHTS(hw.og)), {m.hom}, {m.hwo});
                RAD_ARCH_TRY(make_qgemm(GR, hh.qlora, hw.q_gate, m.hqn, m.qa_hqn, m.hgt, &Hc.q_gate));
                Hc.gate = Hc.q_gate.op;
                Hc.gmul = rw(b, RAD_OP(b, "act_mul",
                                       RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", GR), RAD_STR("act", "sigmoid"),
                                                  RAD_STR("dtype", "bf16")),
                                       RAD_NOWEIGHTS), {m.hgt, m.hwo}, {m.hog});
                if (Hc.need_qhog) RAD_ARCH_TRY(make_qop(m.hog, T, m.qa_hog, &Hc.qop_hog));
                RAD_ARCH_TRY(make_qgemm(H, GR, hw.q_o, m.hog, m.qa_hog, m.y, &Hc.q_out));
                Hc.out = Hc.q_out.op;
                if (world > 1) {
                    Hc.out_ar = make_ar(H, m.y);
                    if (!Hc.out_ar) return RAD_E_INVAL;
                }
                Hc.post = rw(b, RAD_OP(b, "mhc_post",
                                       RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", H), RAD_INT("hc", hc),
                                                  RAD_STR("dtype", "bf16")),
                                       RAD_NOWEIGHTS), {m.y, m.h[0], m.post, m.comb}, {m.h[1]});
                if (!Hc.pre || !Hc.qa || !Hc.qn || !Hc.qb || !Hc.qp || !Hc.kva || !Hc.kvn || !Hc.rk || !Hc.store ||
                    !Hc.att || !Hc.wo || !Hc.gate || !Hc.gmul || !Hc.out || !Hc.post)
                    return RAD_E_INVAL;

                /* Routed MXFP4 MoE + Shared Expert (same structure as trunk routed layers) */
                const int64_t I = dy.width, E = c.experts, K = c.top_k, Lt = c.latent, Fe = m.Fe, P = T * K;
                dy.pre = rw(b, RAD_OP(b, "ernie_mhc_pre_rms",
                                      RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", H), RAD_INT("hc", hc),
                                                 RAD_INT("iterations", f.iterations), RAD_F64("eps", f.rms_eps),
                                                 RAD_F64("pre_eps", f.hc_eps), RAD_F64("sinkhorn_eps", f.hc_eps),
                                                 RAD_F64("post_mult", 2.0), RAD_F64("rms_eps", f.rms_eps),
                                                 RAD_STR("dtype", "bf16")),
                                      RAD_WEIGHTS(dy.fn, dy.scale, dy.base, dy.gain)),
                            {dy.ffn_in}, {m.x, m.post, m.comb});
                RoutedOps& R = dy.R;
                R.w = dw.moe;
                const RoutedHandles& rh = R.w;
                if (!rh.r0 || !rh.r1 || !rh.bias || !rh.scale || !rh.norm || !rh.q_in.valid() || !rh.q_out.valid() ||
                    !dw.q_gu.valid() || !dw.q_down.valid())
                    return RAD_E_INVAL;
                R.combine_ar = (world > 1 && m.ffn_ar != 0);
                const rad_buf ys_buf = R.combine_ar ? m.ffn_ar : m.ys;
                const rad_buf comb_buf = R.combine_ar ? m.ffn_ar : m.comb2;
                dy.need_qx = dw.q_gu.fp8 || rh.q_in.fp8;
                dy.need_qact = dw.q_down.fp8;
                if (dy.need_qx) RAD_ARCH_TRY(make_qop(m.x, T, m.qa_x, &dy.qop_x));
                R.r01 = rw(b, RAD_OP(b, "gemm_nt_f32_pair",
                                     RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("N", E), RAD_INT("K", H),
                                                RAD_STR("dtype", "bf16")),
                                     RAD_WEIGHTS(rh.r0, rh.r1)), {m.x}, {m.lg});
                R.r0 = R.r1 = R.r01;
                RAD_ARCH_TRY(make_qgemm(Lt, H, rh.q_in, m.x, m.qa_x, m.xl, &R.q_fc1));
                R.fc1 = R.q_fc1.op;
                RAD_ARCH_TRY(make_qgemm(2 * I, H, dw.q_gu, m.x, m.qa_x, m.gu[1], &dy.q_gu));
                dy.gate_op = dy.up_op = dy.q_gu.op;
                R.topk = rw(b, RAD_OP(b, "ernie_router_topk",
                                      RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("experts", E),
                                                 RAD_INT("topk", K), RAD_INT("renormalize", *renorm ? 1 : 0)),
                                      RAD_WEIGHTS(rh.bias, rh.scale)), {m.lg}, {m.ids, m.rw});
                R.scatter = rw(b, RAD_OP(b, "moe_scatter",
                                         RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n_expert", E),
                                                    RAD_INT("top_k", K), RAD_INT("expert_base", 0)),
                                         RAD_NOWEIGHTS), {m.ids}, {m.sorted, m.eoff, m.ecnt});
                R.topk_scatter = rw(b, RAD_OP(b, "ernie_router_topk_scatter",
                                              RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("experts", E),
                                                         RAD_INT("topk", K), RAD_INT("renormalize", *renorm ? 1 : 0)),
                                              RAD_WEIGHTS(rh.bias, rh.scale)),
                                    {m.lg}, {m.ids, m.rw, m.sorted, m.eoff, m.ecnt});
                dy.act = rw(b, RAD_OP(b, "ernie_situ_mul",
                                      RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", I), RAD_STR("dtype", "bf16"),
                                                 RAD_F64("beta", f.beta), RAD_F64("linear_beta", f.linear_beta)),
                                      RAD_NOWEIGHTS), {m.gu[1]}, {m.act[1]});
                if (dy.need_qact) {
                    if ((I % 4) == 0) {
                        dy.act_q = rw(b, RAD_OP(b, "situ_mul_quant_fp8",
                                                RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", I), RAD_INT("group", 128),
                                                           RAD_STR("dtype", "bf16"), RAD_F64("beta", f.beta),
                                                           RAD_F64("linear_beta", f.linear_beta)),
                                                RAD_NOWEIGHTS), {m.gu[1]}, {m.qa_act[1].q, m.qa_act[1].s});
                        if (!dy.act_q) return RAD_E_INVAL;
                    } else {
                        RAD_ARCH_TRY(make_qop(m.act[1], T, m.qa_act[1], &dy.qop_act));
                    }
                }
                RAD_ARCH_TRY(make_qgemm(H, I, dw.q_down, m.act[1], m.qa_act[1], ys_buf, &dy.q_down));
                dy.down_op = dy.q_down.op;
                /* The trunk's table_mx (declare_ffn above): the same moe_gemm_q schema, mxfp4a8 at group 128, the
                 * even/odd parity tables (`parts` stacked outputs: 2 for gate|up, 1 for down). The drafter's
                 * experts have the trunk's geometry (gate_up 4096x2048, down 2048x2048, E8M0 per 32 of K). */
                auto dtable_mx = [&](int64_t N, int64_t Kd, const char* order, int64_t parts,
                                     const std::vector<rad_weight>& c0, const std::vector<rad_weight>& s0,
                                     const std::vector<rad_weight>& c1, const std::vector<rad_weight>& s1,
                                     const QAct& qa, rad_buf out) {
                    std::vector<rad_weight> all;
                    all.reserve(c0.size() + s0.size() + c1.size() + s1.size());
                    all.insert(all.end(), c0.begin(), c0.end());
                    all.insert(all.end(), s0.begin(), s0.end());
                    all.insert(all.end(), c1.begin(), c1.end());
                    all.insert(all.end(), s1.begin(), s1.end());
                    const RadParam ps[] = {
                        RAD_RANGE("M", 1, qrows), RAD_INT("N", N), RAD_INT("K", Kd),
                        RAD_INT("n_expert", E), RAD_INT("top_k", K), RAD_INT("group", 128),
                        RAD_STR("a_order", order), RAD_STR("dtype", "mxfp4a8"),
                        RAD_INT("N_odd", N), RAD_INT("K_odd", Kd), RAD_INT("parts", parts)
                    };
                    rad_op h = rad_decl_op(b, "moe_gemm_q", ps, (int)(sizeof(ps) / sizeof(*ps)),
                                           all.data(), (int)all.size());
                    return rw(b, h, {qa.q, qa.s, m.sorted, m.eoff}, {out});
                };
                auto dtable = [&](int64_t N, int64_t Kd, const char* order, const std::vector<rad_weight>& tab,
                                  rad_buf a, rad_buf out) {
                    rad_op h = rad_decl_op(b, "moe_gemm",
                                           RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("N", N), RAD_INT("K", Kd),
                                                      RAD_INT("n_expert", E), RAD_INT("top_k", K),
                                                      RAD_STR("a_order", order), RAD_STR("dtype", "bf16")),
                                           tab.data(), (int)tab.size());
                    return rw(b, h, {a, m.sorted, m.eoff}, {out});
                };
                if (rh.mx) {
                    RAD_ARCH_TRY(make_qop(m.xl, T, m.qa_xl, &R.qop_xl));
                    /* As the trunk (declare_ffn): the expert ops read activation codes and scales the qop calls above populate; refuse a declaration
                     * where one is missing rather than hand moe_gemm_q a null scale plane (review, kv-context fixes). */
                    if (!m.qa_xl.valid()) return RAD_E_SHAPE;
                    R.gu_q = dtable_mx(2 * Fe, Lt, "token", 2, rh.gu_c0, rh.gu_s0, rh.gu_c1, rh.gu_s1, m.qa_xl, m.egu);
                    R.gate = R.up = R.gu_q;
                } else {
                    R.gate = dtable(Fe, Lt, "token", rh.ge, m.xl, m.egu);
                    R.up = dtable(Fe, Lt, "token", rh.ue, m.xl, m.egu);
                }
                R.act = rw(b, RAD_OP(b, "ernie_situ_mul",
                                     RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", Fe), RAD_STR("dtype", "bf16"),
                                                RAD_F64("beta", f.beta), RAD_F64("linear_beta", f.linear_beta)),
                                     RAD_NOWEIGHTS), {m.egu}, {m.eact});
                if (rh.mx) {
                    if ((Fe % 4) == 0) {
                        R.act_q = rw(b, RAD_OP(b, "situ_mul_quant_fp8",
                                               RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", Fe), RAD_INT("group", 128),
                                                          RAD_STR("dtype", "bf16"), RAD_F64("beta", f.beta),
                                                          RAD_F64("linear_beta", f.linear_beta)),
                                               RAD_NOWEIGHTS), {m.egu}, {m.qa_eact.q, m.qa_eact.s});
                        if (!R.act_q) return RAD_E_INVAL;
                    } else {
                        RAD_ARCH_TRY(make_qop(m.eact, P, m.qa_eact, &R.qop_eact));
                    }
                    if (!m.qa_eact.valid()) return RAD_E_SHAPE;
                    R.dn_q = dtable_mx(Lt, Fe, "sorted", 1, rh.dn_c0, rh.dn_s0, rh.dn_c1, rh.dn_s1, m.qa_eact, m.edn);
                    R.down = R.dn_q;
                } else {
                    R.down = dtable(Lt, Fe, "sorted", rh.de, m.eact, m.edn);
                }
                R.gather = rw(b, RAD_OP(b, "moe_gather",
                                        RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", Lt), RAD_INT("top_k", K),
                                                   RAD_STR("dtype", "bf16"), RAD_STR("act", "none")),
                                        RAD_NOWEIGHTS), {m.edn, m.rw, m.sorted}, {comb_buf});
                if (R.combine_ar) {
                    R.ffn_ar = make_ar(H + Lt, m.ffn_ar);
                    if (!R.ffn_ar) return RAD_E_INVAL;
                } else if (world > 1) {
                    dy.down_ar = make_ar(H, m.ys);
                    R.comb_ar = make_ar(Lt, m.comb2);
                    if (!dy.down_ar || !R.comb_ar) return RAD_E_INVAL;
                }
                R.norm = rw(b, RAD_OP(b, "rmsnorm",
                                      RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", Lt),
                                                 RAD_F64("eps", f.rms_eps), RAD_STR("dtype", "bf16")),
                                      RAD_WEIGHTS(rh.norm)), {comb_buf}, {m.nl});
                R.need_qnl = rh.q_out.fp8;
                if (R.need_qnl) RAD_ARCH_TRY(make_qop(m.nl, T, m.qa_nl, &R.qop_nl));
                RAD_ARCH_TRY(make_qgemm(H, Lt, rh.q_out, m.nl, m.qa_nl, m.ro, &R.q_fc2));
                R.fc2 = R.q_fc2.op;
                R.add_post = rw(b, RAD_OP(b, "ernie_add_mhc_post",
                                          RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", H), RAD_INT("hc", hc),
                                                     RAD_STR("dtype", "bf16")),
                                          RAD_NOWEIGHTS), {m.ro, ys_buf, dy.ffn_in, m.post, m.comb}, {dy.ffn_out});
                R.add = R.add_post;
                if (!dy.pre || !R.r01 || !R.topk || !R.scatter || !R.topk_scatter || !R.fc1 || !dy.gate_op ||
                    !dy.act || !dy.down_op || !R.gate || !R.up || !R.act || !R.down || !R.gather || !R.norm ||
                    !R.fc2 || !R.add_post)
                    return RAD_E_INVAL;
            }

            /* ---- Final contract, shared target lm_head, top-K & Markov walk ---- */
            ds.gather = rw(b, RAD_OP(b, "gather_rows",
                                     RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("n", W), RAD_STR("dtype", "bf16")),
                                     RAD_NOWEIGHTS), {m.h[0]}, {m.hout});
            ds.head = rw(b, RAD_OP(b, "ernie_mhc_head",
                                   RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("n", H), RAD_INT("hc", hc),
                                              RAD_F64("eps", f.rms_eps), RAD_F64("gate_eps", f.hc_eps)),
                                   RAD_WEIGHTS(ds.head_fn, ds.head_scale, ds.head_base)), {m.hout}, {m.hx});
            ds.norm = rw(b, RAD_OP(b, "rmsnorm",
                                   RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("n", H), RAD_F64("eps", f.rms_eps),
                                              RAD_STR("dtype", "bf16")),
                                   RAD_WEIGHTS(ds.out_norm)), {m.hx}, {m.hn});
            ds.proj = rw(b, RAD_OP(b, "logits_gemm",
                                   RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("n_vocab", m.Vloc), RAD_INT("n_embd", H),
                                              RAD_STR("dtype", "bf16")),
                                   RAD_WEIGHTS(m.lm_head)), {m.hn}, {m.lgt});
            ds.lgt_cast = rw(b, RAD_OP(b, "cast",
                                       RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("n", m.Vloc),
                                                  RAD_STR("from", "f32"), RAD_STR("to", "bf16")),
                                       RAD_NOWEIGHTS), {m.lgt}, {ds.lgt_bf16});
            ds.topk = (world > 1)
                ? rw(b, RAD_OP(b, "row_topk",
                               RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("N", m.Vloc),
                                          RAD_INT("R", ds.top_k), RAD_INT("vocab_off", m.Voff),
                                          RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS), {ds.lgt_bf16}, {ds.cand, ds.unary, ds.pair})
                : rw(b, RAD_OP(b, "row_topk",
                               RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("N", m.Vloc),
                                          RAD_INT("R", ds.top_k), RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS), {ds.lgt_bf16}, {ds.cand, ds.unary});
            if (world > 1) {
                ds.gather_topk = rw(b, RAD_OP(b, "all_gather",
                                              RAD_PARAMS(RAD_INT("world_size", world),
                                                         RAD_RANGE("numel", ds.top_k * 2, srows * ds.top_k * 2),
                                                         RAD_STR("dtype", "f32"),
                                                         RAD_INT("row", ds.top_k * 2)),
                                              RAD_NOWEIGHTS), {ds.pair}, {ds.gath});
                ds.merge_topk = rw(b, RAD_OP(b, "row_topk_merge",
                                             RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("R", ds.top_k),
                                                        RAD_INT("world_size", world), RAD_STR("dtype", "bf16")),
                                             RAD_NOWEIGHTS), {ds.gath}, {ds.cand, ds.unary});
                if (!ds.gather_topk || !ds.merge_topk) return RAD_E_INVAL;
            }
            ds.sel = rw(b, RAD_OP(b, "dflash_select",
                                  RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("steps", ds.depth),
                                             RAD_INT("top_k", ds.top_k), RAD_INT("rank", dsc->markov_rank),
                                             RAD_INT("n_vocab", c.vocab), RAD_INT("anchor_stride", ds.block),
                                             RAD_STR("dtype", "bf16")),
                                  RAD_WEIGHTS(ds.mk1, ds.mk2)),
                        {ds.cand, ds.unary}, {ds.tokens});
            if (!ds.gather || !ds.head || !ds.norm || !ds.proj || !ds.lgt_cast || !ds.topk || !ds.sel)
                return RAD_E_INVAL;

            RadDrafterDecl dd{};
            dd.name = "dspark";
            dd.kind = RAD_DRAFT_BLOCK;
            dd.depth = (int)ds.depth;
            dd.proposal = ds.tokens;
            dd.proposal_pitch = ds.depth;
            dd.mask_token = (int32_t)dsc->noise_token_id;
            /* The drafter's own sliding window over its cache, which the core stages the block pass against (rad_builder.h): it must be the window the
             * attention above is gated with. 0 declared the cache unbounded, so after a prefix hit that left a hole in it (draft_floor > 0) the scheduler's
             * reach test (window > 0 && ...) never passed and the sequence never drafted again, and no refill was asked of the prefix cache. */
            dd.window = (int64_t)ds.window;
            dd.block = ds.block;
            RAD_ARCH_TRY(rad_declare_drafter(b, &dd));
        }
        m.logits = true;
        RAD_ARCH_TRY(ov_declare(b, ctx, m));
        m.on = true;
        return RAD_OK;
    }
    m.pool = rw(b, RAD_OP(b, "seg_mean",
                          RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", W), RAD_STR("dtype", "bf16")),
                          RAD_NOWEIGHTS), {m.h[m.fin]}, {m.mean});
    m.cast = rw(b, RAD_OP(b, "cast",
                          RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n", W), RAD_STR("from", "bf16"),
                                     RAD_STR("to", "f32")),
                          RAD_NOWEIGHTS), {m.mean}, {m.pooled});
    if (!m.pool || !m.cast) return RAD_E_INVAL;
    m.on = true;
    return RAD_OK;
}

static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    Config c;
    std::string why;
    const int s = read_config(meta, &c, &why);
    if (s < 0) return s;
    /* The drafter count is the checkpoint's own, not a function of the target's depth. Serial
     * heads run at any depth up to it, so it is not the only depth. */
    out->draft_depth = (int)c.draft_layers;
    /* --kv-cache-dtype fp8 stores the MLA latent cache as fp8_ds_mla rows (kv-fp8-latent): only the attention program has one */
    out->kv_fp8_ok = rad_meta_geti(meta, "ernie55_quant_forward", 0) == 1 ? 1 : 0;
    /* The feed-forward program is an embedding program with no drafter: reporting the checkpoint's
     * draft layers would make the engine default to speculative decoding it refuses. */
    if (rad_meta_geti(meta, "ernie55_ffn_forward", 0) == 1 ||
        rad_meta_geti(meta, "ernie55_quant_forward", 0) == 1) out->draft_depth = 0;
    if (rad_meta_geti(meta, "ernie55_quant_forward", 0) == 1 &&
        rad_meta_geti(meta, "ernie55_dspark", 0) == 1)
        out->draft_depth = (int)rad_meta_geti(meta, "dspark_block_size", 7);
    return RAD_OK;
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    resolve_row_bands(rad_meta_geti(meta, "ernie55_dspark", 0) == 1, ctx->max_spec);   /* the row bands and the libr4d band variables, before any launch or use reads them */
    Config c;
    std::string why;
    const int s = read_config(meta, &c, &why);
    if (s < 0) {
        fprintf(stderr, "radiance: ernie55: invalid ERNIE 5.5 config: %s\n", why.c_str());
        return s;
    }
    const int64_t qfwd_key = rad_meta_geti(meta, "ernie55_quant_forward", 0);
    if (qfwd_key != 0 && qfwd_key != 1) {
        fprintf(stderr, "radiance: ernie55: ernie55_quant_forward must be 0 or 1, got %lld\n",
                (long long)qfwd_key);
        return RAD_E_INVAL;
    }
    const bool quant_fwd = qfwd_key == 1;
    if (quant_fwd) {
        const int ws = read_weight_dims(meta, &c, &why);
        if (ws < 0) {
            fprintf(stderr, "radiance: ernie55: invalid ERNIE 5.5 config: %s\n", why.c_str());
            return ws;
        }
        if (c.experts > kMaxExperts) {
            fprintf(stderr, "radiance: ernie55: invalid ERNIE 5.5 config: n_routed_experts (%lld) > %lld\n",
                    (long long)c.experts, (long long)kMaxExperts);
            return RAD_E_INVAL;
        }
        FfnHandles fh;
        FfnConfig fc;
        const int fs = read_ffn_config(meta, c, &fc, &why);
        if (fs < 0) { fprintf(stderr, "radiance: ernie55: invalid ERNIE 5.5 config: %s\n", why.c_str()); return fs; }
        bool renorm = false;
        const bool has_routed = c.first_dense < c.main_layers;
        if (has_routed) {
            const int ms = read_moe_config(meta, &renorm, &why);
            if (ms < 0) { fprintf(stderr, "radiance: ernie55: invalid ERNIE 5.5 config: %s\n", why.c_str()); return ms; }
        }
        AttnConfig ac;
        const int as = read_attn_config(meta, &ac, &why);
        if (as < 0) { fprintf(stderr, "radiance: ernie55: invalid ERNIE 5.5 config: %s\n", why.c_str()); return as; }
        HcaConfig hcc;
        const int hs = read_hca_config(meta, c, &hcc, &why);
        if (hs < 0) { fprintf(stderr, "radiance: ernie55: invalid ERNIE 5.5 config: %s\n", why.c_str()); return hs; }
        if (ctx->world_size > 1) {
            const int w = ctx->world_size;
            int64_t sh_w = 0;
            if (!mul(c.moe_ff, c.shared, &sh_w) || (c.vocab % w) != 0 ||
                (ac.heads % w) != 0 || (hcc.heads % w) != 0 || (hcc.groups % w) != 0 ||
                (c.first_dense > 0 && (c.dense_ff % w) != 0) ||
                (has_routed && ((sh_w % w) != 0 || (c.moe_ff % w) != 0))) {
                fprintf(stderr, "radiance: ernie55: TP=%d requires vocab, attention heads, HCA o_groups, and FFN/MoE widths divisible by world_size\n", w);
                return RAD_E_UNSUPPORTED;
            }
        }
        const int64_t ds_key = rad_meta_geti(meta, "ernie55_dspark", 0);
        if (ds_key != 0 && ds_key != 1) {
            fprintf(stderr, "radiance: ernie55: ernie55_dspark must be 0 or 1, got %lld\n", (long long)ds_key);
            return RAD_E_INVAL;
        }
        DsparkConfig dsc;
        DsparkHandles dh;
        const bool use_dspark = (ds_key == 1 && ctx->max_spec > 0);
        if (use_dspark) {
            const int dss = read_dspark_config(meta, c, &dsc, &why);
            if (dss < 0) { fprintf(stderr, "radiance: ernie55: invalid DSpark config: %s\n", why.c_str()); return dss; }
        }
        {
            /* The activation bands in force, logged once so the deployed container's log states them (ERNIE_A16_MAX and ERNIE_VHA_DEC_MAX are
             * read from the environment at first use; the fp8a8 matvec bound is declared per op). */
            bool mvm_ov = false;
            const int mvm = f8a8_mvm_declared(&mvm_ov);
            if (mvm_ov)
                fprintf(stderr, "radiance: ernie55: WARNING: R4D_F8A8_MVM_MAX=%d in the environment overrides the declared matvec bound "
                                "(diagnosis only: a row's fp8a8 output then depends on whether its step has more or fewer than %d rows)\n", mvm, mvm);
            if (pin::f8a8_mvm_refused(mvm, g_row_identity)) {
                fprintf(stderr, "radiance: ernie55: refusing to start: R4D_F8A8_MVM_MAX=%d with the row-identical bands (drafter in the file or max_spec=%d > 0) would make decode rows "
                                "depend on the step's row count and break spec N == spec 0; unset R4D_F8A8_MVM_MAX (the model declares 0 here) or set ERNIE_ROW_IDENTITY=0\n", mvm, (int)ctx->max_spec);
                return RAD_E_INVAL;
            }
            if (a16_max_rows() > 0)
                fprintf(stderr, "radiance: ernie55: WARNING: ERNIE_A16_MAX=%lld: steps of <= %lld rows read bf16 activations and larger steps fp8, so a "
                                "row's logits depend on the step's row count (not lossless under speculation or batching)\n",
                        (long long)a16_max_rows(), (long long)a16_max_rows());
            if (vha_dec_cap() > 0)
                fprintf(stderr, "radiance: ernie55: WARNING: ERNIE_VHA_DEC_MAX=%lld: steps of <= %lld rows mix the VHA heads by all-gather, larger "
                                "steps by the reduce path (not row-identical)\n", (long long)vha_dec_cap(), (long long)vha_dec_cap());
        }
        RAD_ARCH_TRY(declare_weights(b, c, has_routed, &fh, has_routed, &ac, &hcc, true, ctx->world_size));
        if (use_dspark) RAD_ARCH_TRY(declare_dspark_weights(b, c, hcc, dsc, &dh, ctx->world_size));
        RAD_ARCH_TRY(declare_ffn(b, ctx, c, fc, fh, has_routed ? &renorm : nullptr, true, &ac, &hcc, true,
                                 use_dspark ? &dsc : nullptr, use_dspark ? &dh : nullptr));
        fprintf(stderr, "radiance: ernie55: quant-forward: %lld layers, TP=%d rank=%d (FP8 trunk + MXFP4/BF16 experts, MLA-DSA + HCA, target output%s)\n",
                (long long)c.main_layers, (int)ctx->world_size, (int)ctx->rank,
                use_dspark ? ", DSpark MTP" : "");
        return RAD_OK;
    }
    if (rad_meta_geti(meta, "ernie55_moe_weights", 0) == 1 &&
        rad_meta_geti(meta, "ernie55_weights_only", 0) != 1) {
        fprintf(stderr, "radiance: ernie55: ernie55_moe_weights=1 extends the weights-only "
                        "conversion: it needs ernie55_weights_only=1 as well\n");
        return RAD_E_INVAL;
    }
    const int64_t ffn_key = rad_meta_geti(meta, "ernie55_ffn_forward", 0);
    if (ffn_key != 0 && ffn_key != 1) {
        fprintf(stderr, "radiance: ernie55: ernie55_ffn_forward must be 0 or 1, got %lld\n",
                (long long)ffn_key);
        return RAD_E_INVAL;
    }
    const bool ffn = ffn_key == 1;
    if (ffn && rad_meta_geti(meta, "ernie55_weights_only", 0) != 1) {
        fprintf(stderr, "radiance: ernie55: ernie55_ffn_forward=1 extends the weights-only "
                        "conversion: it needs ernie55_weights_only=1 as well\n");
        return RAD_E_INVAL;
    }
    const int64_t moe_key = rad_meta_geti(meta, "ernie55_moe_forward", 0);
    if (moe_key != 0 && moe_key != 1) {
        fprintf(stderr, "radiance: ernie55: ernie55_moe_forward must be 0 or 1, got %lld\n",
                (long long)moe_key);
        return RAD_E_INVAL;
    }
    const bool moe_fwd = moe_key == 1;
    if (moe_fwd && (!ffn || rad_meta_geti(meta, "ernie55_weights_only", 0) != 1 ||
                    rad_meta_geti(meta, "ernie55_moe_weights", 0) != 1)) {
        fprintf(stderr, "radiance: ernie55: ernie55_moe_forward=1 extends the feed-forward program: it needs "
                        "ernie55_weights_only=1, ernie55_moe_weights=1 and ernie55_ffn_forward=1 as well\n");
        return RAD_E_INVAL;
    }
    const int64_t attn_key = rad_meta_geti(meta, "ernie55_attention_forward", 0);
    if (attn_key != 0 && attn_key != 1) {
        fprintf(stderr, "radiance: ernie55: ernie55_attention_forward must be 0 or 1, got %lld\n", (long long)attn_key);
        return RAD_E_INVAL;
    }
    const bool attn_fwd = attn_key == 1;
    if (attn_fwd && (!ffn || rad_meta_geti(meta, "ernie55_weights_only", 0) != 1)) {
        fprintf(stderr, "radiance: ernie55: ernie55_attention_forward=1 extends the feed-forward program: it needs "
                        "ernie55_weights_only=1 and ernie55_ffn_forward=1 as well\n");
        return RAD_E_INVAL;
    }
    const int64_t hca_key = rad_meta_geti(meta, "ernie55_hca_forward", 0);
    if (hca_key != 0 && hca_key != 1) {
        fprintf(stderr, "radiance: ernie55: ernie55_hca_forward must be 0 or 1, got %lld\n", (long long)hca_key);
        return RAD_E_INVAL;
    }
    const bool hca_fwd = hca_key == 1;
    if (hca_fwd && !attn_fwd) {
        fprintf(stderr, "radiance: ernie55: ernie55_hca_forward=1 extends the attention program: it needs ernie55_attention_forward=1 "
                        "(with ernie55_weights_only=1 and ernie55_ffn_forward=1) as well\n");
        return RAD_E_INVAL;
    }
    if (attn_fwd && !hca_fwd) {
        for (int64_t i = 0; i < c.main_layers; ++i)
            if (c.modes[(size_t)i] == kHca) {
                fprintf(stderr, "radiance: ernie55: ernie55_attention_forward=1 serves MLA-DSA layers only: layer %lld is an HCA "
                                "layer (compress_ratios[%lld] = 128), which story 3-4 builds\n", (long long)i, (long long)i);
                return RAD_E_UNSUPPORTED;
            }
    }
    const int64_t tgt_key = rad_meta_geti(meta, "ernie55_target_output", 0);
    if (tgt_key != 0 && tgt_key != 1) {
        fprintf(stderr, "radiance: ernie55: ernie55_target_output must be 0 or 1, got %lld\n",
                (long long)tgt_key);
        return RAD_E_INVAL;
    }
    const bool target = tgt_key == 1;
    if (target && (!ffn || rad_meta_geti(meta, "ernie55_weights_only", 0) != 1)) {
        fprintf(stderr, "radiance: ernie55: ernie55_target_output=1 extends the feed-forward program: it needs "
                        "ernie55_weights_only=1 and ernie55_ffn_forward=1 as well\n");
        return RAD_E_INVAL;
    }
    if (target && !moe_fwd && c.first_dense < c.main_layers) {
        fprintf(stderr, "radiance: ernie55: ernie55_target_output=1 needs ernie55_moe_forward=1 (with "
                        "ernie55_moe_weights=1) because layers %lld.. are routed: without it their experts "
                        "would be skipped and the scores would not be the model's\n", (long long)c.first_dense);
        return RAD_E_INVAL;
    }
    if (rad_meta_geti(meta, "ernie55_weights_only", 0) == 1) {
        const int ws = read_weight_dims(meta, &c, &why);
        if (ws < 0) {
            fprintf(stderr, "radiance: ernie55: invalid ERNIE 5.5 config: %s\n", why.c_str());
            return ws;
        }
        const bool moe = rad_meta_geti(meta, "ernie55_moe_weights", 0) == 1;
        if (moe && c.experts > kMaxExperts) {
            fprintf(stderr, "radiance: ernie55: invalid ERNIE 5.5 config: n_routed_experts (%lld) is "
                            "greater than the %lld routed experts this converter declares\n",
                    (long long)c.experts, (long long)kMaxExperts);
            return RAD_E_INVAL;
        }
        FfnHandles fh;
        FfnConfig fc;
        if (ffn) {
            const int fs = read_ffn_config(meta, c, &fc, &why);
            if (fs < 0) {
                fprintf(stderr, "radiance: ernie55: invalid ERNIE 5.5 config: %s\n", why.c_str());
                return fs;
            }
        }
        bool renorm = false;
        if (moe_fwd) {
            const int ms = read_moe_config(meta, &renorm, &why);
            if (ms < 0) {
                fprintf(stderr, "radiance: ernie55: invalid ERNIE 5.5 config: %s\n", why.c_str());
                return ms;
            }
        }
        AttnConfig ac;
        if (attn_fwd) {
            const int as = read_attn_config(meta, &ac, &why);
            if (as < 0) {
                fprintf(stderr, "radiance: ernie55: invalid ERNIE 5.5 config: %s\n", why.c_str());
                return as;
            }
        }
        HcaConfig hcc;
        bool any_hca = false;
        for (int64_t i = 0; i < c.main_layers; ++i) any_hca |= c.modes[(size_t)i] == kHca;
        if (hca_fwd && any_hca) {
            const int hs = read_hca_config(meta, c, &hcc, &why);
            if (hs < 0) {
                fprintf(stderr, "radiance: ernie55: invalid ERNIE 5.5 config: %s\n", why.c_str());
                return hs;
            }
        }
        const HcaConfig* hp = hca_fwd && any_hca ? &hcc : nullptr;
        const int ds = declare_weights(b, c, moe, ffn ? &fh : nullptr, moe_fwd, attn_fwd ? &ac : nullptr, hp);
        if (ds < 0) return ds;
        if (ffn) {
            const int fd = declare_ffn(b, ctx, c, fc, fh, moe_fwd ? &renorm : nullptr, target, attn_fwd ? &ac : nullptr, hp);
            if (fd < 0) return fd;
            if (hp)
                fprintf(stderr, "radiance: ernie55: hca-forward: HCA layers run the compressor (period %lld, state/compressed/window caches as per-token "
                                "KV groups, block %lld), the union attention over the complete compressed rows and a %lld-token window "
                                "(overlap kept) with the fp32 sink, VHA fold=%s, the inverse-rotary grouped wo_a (%lld groups, rank %lld), the "
                                "sigmoid gate on its output and wo_b; heads %lld, head_dim %lld, rope %lld at theta %g, %s\n",
                        (long long)hp->period, (long long)hp->block, (long long)hp->window, kVhaFold, (long long)hp->groups, (long long)hp->rank,
                        (long long)hp->heads, (long long)hp->D, (long long)hp->rope, hp->theta,
                        hp->yarn ? "YaRN rope (rope_scaling: factor/beta_fast/beta_slow/original_max)" : "plain rope (ernie55_hca_rope=plain)");
            if (attn_fwd)
                fprintf(stderr, "radiance: ernie55: attention-forward: %lld MLA-DSA layers run the attn-branch mHC, the latent "
                                "attention (heads %lld, latent %lld + rope %lld) over a bf16 latent cache and the DSA indexer "
                                "over an FP8 key cache (top-%lld, window %lld, block %lld), rope_table=%s fold=%s\n",
                        (long long)c.main_layers, (long long)ac.heads, (long long)ac.lat, (long long)ac.rope,
                        (long long)ac.topk, (long long)ac.window, (long long)ac.block, kRopeTable, kVhaFold);
            if (moe_fwd)
                fprintf(stderr, "radiance: ernie55: moe-forward: routed layers also run the f32 routers, "
                                "top-%lld of %lld experts at latent width %lld, the combine, the latent "
                                "norm and up projection after it, and the shared expert's sum\n",
                        (long long)c.top_k, (long long)c.experts, (long long)c.latent);
            if (target) {
                fprintf(stderr, "radiance: ernie55: target-output: the program ends in the learned mHC head, "
                                "the final norm, the vocabulary projection and MultiMax on the sampled rows, "
                                "and declares those scores as the logits; no attention exists, so each "
                                "token's scores depend on that token alone\n");
                return RAD_OK;
            }
            fprintf(stderr, "radiance: ernie55: ffn-forward: declared the feed-forward program "
                            "(embedding, first-layer broadcast, mHC ffn branch, dense/shared SiTU-GLU, "
                            "mHC post) of %lld layers, pooled over the hc streams; no attention or "
                            "routed experts exist, so this is a block probe, not a language model\n",
                    (long long)c.main_layers);
            return RAD_OK;
        }
        char routed[160] = "";
        if (moe)
            snprintf(routed, sizeof routed, " and routed-MoE weights (routers, bias, expert scale, "
                     "latent projections and norm, %lld experts) of %lld layers",
                     (long long)c.experts, (long long)(c.main_layers - c.first_dense));
        fprintf(stderr, "radiance: ernie55: weights-only: declared the embedding, norms, mHC, "
                        "head and dense/shared feed-forward weights of %lld layers%s; no forward "
                        "blocks exist, so this container cannot serve\n",
                (long long)c.main_layers, routed);
        return RAD_OK;
    }
    int64_t hca = 0, mla = 0;
    for (int64_t i = 0; i < c.main_layers; ++i) (c.modes[(size_t)i] == kHca ? hca : mla)++;
    fprintf(stderr,
            "radiance: ernie55: config ok: arch=ernie55moe model_type=%s target_layers=%lld "
            "draft_layers=%lld first_dense=%lld hca_layers=%lld mla_dsa_layers=%lld "
            "hidden=%lld head_dim=%lld hc_mult=%lld experts=%lld top_k=%lld latent=%lld "
            "schedule=%s\n",
            kModelType, (long long)c.main_layers, (long long)c.draft_layers,
            (long long)c.first_dense, (long long)hca, (long long)mla, (long long)c.hidden,
            (long long)c.head_dim, (long long)c.hc_mult, (long long)c.experts,
            (long long)c.top_k, (long long)c.latent, schedule_text(c).c_str());
    fprintf(stderr, "radiance: ernie55: the forward blocks (attention, mHC, feed-forward, head) are "
                    "not available in this build; refusing to declare a graph\n");
    return RAD_E_UNSUPPORTED;
}


static void emit_dspark_context(RadCtx* c, const FfnProgram& m, const RadBatch* batch) {
    using namespace rad::arch;
    const DsparkProg& ds = m.ds;
    if (!ds.on) return;
    const int64_t T = batch->n_tok;
    if (T <= 0) return;
    const RadKVGroupBatch* kw = kv_batch(batch, ds.kv_ds);
    if (!kw || !kw->slot_mapping) {
        rad_step_fail(c, "ernie55 dspark context pass requires draft KV slot_mapping");
        return;
    }
    const RadOperand pos = praw(batch->positions, RAD_I32, T);
    auto issue_qact = [&](rad_op qop, const QAct& qa, rad_buf src, int64_t rows) {
        RAD_ISSUE_N(c, qop, rows, brow_slice(src, 0, rows, qa.K),
                    brow_slice(qa.q, 0, rows, qa.K), brow_slice(qa.s, 0, rows, qa.K / 128));
    };
    auto use_a16 = [&](const QOp& qo, int64_t rows) { return pin::use_a16(qo.w.fp8, rows, a16_max_rows(), qo.op_a16 != 0); };
    auto need_a8 = [&](const QOp& qo, int64_t rows) { return qo.w.fp8 && !use_a16(qo, rows); };
    auto issue_qgemm = [&](const QOp& qo, rad_buf x_bf16, const QAct& qa, rad_buf y_out, int64_t rows) {
        if (!qo.w.fp8) {
            RAD_ISSUE_N(c, qo.op, rows, brows(x_bf16, rows), RAD_W(qo.w.w), brows(y_out, rows));
        } else if (use_a16(qo, rows)) {
            RAD_ISSUE_N(c, qo.op_a16, rows, brows(x_bf16, rows), RAD_NONE,
                        RAD_W(qo.w.w), RAD_W(qo.w.s), brows(y_out, rows), RAD_NONE, RAD_NONE);
        } else {
            RAD_ISSUE_N(c, qo.op, rows, brows(qa.q, rows), brows(qa.s, rows),
                        RAD_W(qo.w.w), RAD_W(qo.w.s), brows(y_out, rows), RAD_NONE, RAD_NONE);
        }
    };
    if (ds.need_qtaps && need_a8(ds.q_fproj, T)) issue_qact(ds.qop_taps, ds.qa_taps, ds.taps, T);
    issue_qgemm(ds.q_fproj, ds.taps, ds.qa_taps, ds.ctx, T);
    RAD_ISSUE_N(c, ds.fnorm, T, brows(ds.ctx, T), RAD_W(ds.feat_norm), brows(ds.ctx, T));

    bool any_ctx_a8 = false;
    for (int i = 0; i < (int)ds.n_layers; ++i)
        if (need_a8(ds.ctx_kv[(size_t)i], T)) any_ctx_a8 = true;
    if (ds.need_qctx && any_ctx_a8) issue_qact(ds.qop_ctx, m.qa_x, ds.ctx, T);

    const int64_t Dh = m.Hc.D;
    for (int i = 0; i < (int)ds.n_layers; ++i) {
        issue_qgemm(ds.ctx_kv[(size_t)i], ds.ctx, m.qa_x, m.hkv, T);
        RAD_ISSUE_N(c, ds.ctx_kvn[(size_t)i], T, brows(m.hkv, T), RAD_W(ds.ctx_kv_norm[(size_t)i]), brows(m.hkv, T));
        RAD_ISSUE_N(c, ds.ctx_rk[(size_t)i], T, brows(m.hkv, T), pos);
        RAD_ISSUE_N(c, ds.ctx_store[(size_t)i], T, bcol(m.hkv, 0, Dh / 2, T), bcol(m.hkv, Dh / 2, Dh / 2, T),
                    praw(kw->slot_mapping, RAD_I32, T), kv_cache(ds.kv_ds, (int)(m.layers + i)));
    }
}

static void emit_dspark_query(RadCtx* c, const FfnProgram& m, const RadBatch* batch) {
    using namespace rad::arch;
    const DsparkProg& ds = m.ds;
    if (!ds.on) return;
    const int64_t T = batch->n_tok, S = batch->n_seq, O = batch->n_out;
    if (T <= 0 || S <= 0) return;
    const RadKVGroupBatch* kw = kv_batch(batch, ds.kv_ds);
    if (!kw || !kw->slot_mapping || !kw->block_table || !kw->seqused) {
        rad_step_fail(c, "ernie55 dspark query pass requires draft KV slot_mapping, block_table, seqused");
        return;
    }
    const RadOperand pos = praw(batch->positions, RAD_I32, T);
    const RadOperand bt = praw2(kw->block_table, RAD_I32, S, kw->block_table_pitch);
    const RadOperand su = praw(kw->seqused, RAD_I32, S);
    const RadOperand cu = praw(batch->cu_seqlens, RAD_I32, S + 1);

    RAD_ISSUE_N(c, ds.embed, T, praw(batch->token_ids, RAD_I32, T), RAD_W(m.tok), brows(m.emb, T));
    if (ds.embed_ar) RAD_ISSUE_N(c, ds.embed_ar, T * m.hidden, brows(m.emb, T), RAD_NONE);
    RAD_ISSUE_N(c, ds.enter, T, brows(m.emb, T), brows(m.h[0], T));

    auto issue_qact = [&](rad_op qop, const QAct& qa, rad_buf src, int64_t rows) {
        RAD_ISSUE_N(c, qop, rows, brow_slice(src, 0, rows, qa.K),
                    brow_slice(qa.q, 0, rows, qa.K), brow_slice(qa.s, 0, rows, qa.K / 128));
    };
    auto use_a16 = [&](const QOp& qo, int64_t rows) { return pin::use_a16(qo.w.fp8, rows, a16_max_rows(), qo.op_a16 != 0); };
    auto need_a8 = [&](const QOp& qo, int64_t rows) { return qo.w.fp8 && !use_a16(qo, rows); };
    auto issue_qgemm_op = [&](const QOp& qo, rad_buf x_bf16, const QAct& qa, RadOperand y_opd, int64_t rows) {
        if (!qo.w.fp8) {
            RAD_ISSUE_N(c, qo.op, rows, brows(x_bf16, rows), RAD_W(qo.w.w), y_opd);
        } else if (use_a16(qo, rows)) {
            RAD_ISSUE_N(c, qo.op_a16, rows, brows(x_bf16, rows), RAD_NONE,
                        RAD_W(qo.w.w), RAD_W(qo.w.s), y_opd, RAD_NONE, RAD_NONE);
        } else {
            RAD_ISSUE_N(c, qo.op, rows, brows(qa.q, rows), brows(qa.s, rows),
                        RAD_W(qo.w.w), RAD_W(qo.w.s), y_opd, RAD_NONE, RAD_NONE);
        }
    };
    auto issue_qgemm = [&](const QOp& qo, rad_buf x_bf16, const QAct& qa, rad_buf y_out, int64_t rows) {
        issue_qgemm_op(qo, x_bf16, qa, brows(y_out, rows), rows);
    };

    const int64_t Dh = m.Hc.D;
    for (int i = 0; i < (int)ds.n_layers; ++i) {
        const FfnLayer& y = ds.L[(size_t)i];
        const HcaLayer& Hc = y.Hc;
        const HcaHandles& w = Hc.w;

        /* Attention */
        RAD_ISSUE_N(c, Hc.pre, T, brows(m.h[0], T), RAD_W(Hc.hc.fn), RAD_W(Hc.hc.scale), RAD_W(Hc.hc.base),
                    RAD_W(Hc.gain), brows(m.x, T), brows(m.post, T), brows(m.comb, T));
        if (Hc.need_qx && (need_a8(Hc.q_qa, T) || need_a8(Hc.q_kva, T))) issue_qact(Hc.qop_x, m.qa_x, m.x, T);
        issue_qgemm(Hc.q_qa, m.x, m.qa_x, m.hqa, T);
        RAD_ISSUE_N(c, Hc.qn, T, brows(m.hqa, T), RAD_W(w.qa_norm), brows(m.hqn, T));
        if (Hc.need_qhqn && (need_a8(Hc.q_qb, T) || need_a8(Hc.q_gate, T))) issue_qact(Hc.qop_hqn, m.qa_hqn, m.hqn, T);
        issue_qgemm(Hc.q_qb, m.hqn, m.qa_hqn, m.hq, T);
        RAD_ISSUE_N(c, Hc.qp, T, brows(m.hq, T), pos);
        issue_qgemm(Hc.q_kva, m.x, m.qa_x, m.hkv, T);
        RAD_ISSUE_N(c, Hc.kvn, T, brows(m.hkv, T), RAD_W(w.kv_norm), brows(m.hkv, T));
        RAD_ISSUE_N(c, Hc.rk, T, brows(m.hkv, T), pos);
        RAD_ISSUE_N(c, Hc.store, T, bcol(m.hkv, 0, Dh / 2, T), bcol(m.hkv, Dh / 2, Dh / 2, T),
                    praw(kw->slot_mapping, RAD_I32, T), kv_cache(ds.kv_ds, (int)(m.layers + i)));
        RAD_ISSUE_N(c, Hc.att, T, brows(m.hq, T), kv_cache(ds.kv_ds, (int)(m.layers + i)),
                    bt, su, cu, RAD_W(w.sink), brows(m.hol, T));
        if (Hc.vha_ag && T <= m.hca_vha_dec_max) {
            RAD_ISSUE_N(c, Hc.vha_ag, T * m.hca_heads_loc * Dh, brows(m.hol, T), brows(m.hol_all, m.world * T));
            RAD_ISSUE_N(c, Hc.vha, T, brows(m.hol_all, m.world * T), RAD_W(w.u), RAD_W(w.v), brows(m.hom, T));
        } else if (Hc.vha_proj) {
            RAD_ISSUE_N(c, Hc.vha_proj, T, brows(m.hol, T), RAD_W(w.u), brows(m.hz, T));
            RAD_ISSUE_N(c, Hc.vha_ar, T * Hc.vha_rank * Dh, brows(m.hz, T), RAD_NONE);
            RAD_ISSUE_N(c, Hc.vha_apply, T, brows(m.hol, T), brows(m.hz, T), RAD_W(w.v), brows(m.hom, T));
        } else {
            RAD_ISSUE_N(c, Hc.vha, T, brows(m.hol, T), RAD_W(w.u), RAD_W(w.v), brows(m.hom, T));
        }
        RAD_ISSUE_N(c, Hc.wo, T, brows(m.hom, T), pos, RAD_W(w.og), brows(m.hwo, T));
        issue_qgemm(Hc.q_gate, m.hqn, m.qa_hqn, m.hgt, T);
        RAD_ISSUE_N(c, Hc.gmul, T, brows(m.hgt, T), brows(m.hwo, T), brows(m.hog, T));
        if (Hc.need_qhog && need_a8(Hc.q_out, T)) issue_qact(Hc.qop_hog, m.qa_hog, m.hog, T);
        issue_qgemm(Hc.q_out, m.hog, m.qa_hog, m.y, T);
        if (Hc.out_ar) RAD_ISSUE_N(c, Hc.out_ar, T * m.hidden, brows(m.y, T), RAD_NONE);
        RAD_ISSUE_N(c, Hc.post, T, brows(m.y, T), brows(m.h[0], T), brows(m.post, T), brows(m.comb, T), brows(m.h[1], T));

        /* Routed MXFP4 MoE + Shared Expert */
        const RoutedOps& R = y.R;
        const RoutedHandles& rw_h = R.w;
        const int64_t P = T * m.K;
        const RadOperand ys_opd = R.combine_ar ? bcol(m.ffn_ar, 0, m.hidden, T) : brows(m.ys, T);
        const RadOperand comb_opd = R.combine_ar ? bcol(m.ffn_ar, m.hidden, m.Lt, T) : brows(m.comb2, T);
        RAD_ISSUE_N(c, y.pre, T, brows(m.h[1], T), RAD_W(y.fn), RAD_W(y.scale), RAD_W(y.base),
                    RAD_W(y.gain), brows(m.x, T), brows(m.post, T), brows(m.comb, T));
        if (y.need_qx && (need_a8(R.q_fc1, T) || need_a8(y.q_gu, T))) issue_qact(y.qop_x, m.qa_x, m.x, T);
        RAD_ISSUE_N(c, R.r01, T, brows(m.x, T), RAD_W(rw_h.r0), RAD_W(rw_h.r1),
                    bcol(m.lg, 0, m.E, T), bcol(m.lg, m.E, m.E, T));
        issue_qgemm(R.q_fc1, m.x, m.qa_x, m.xl, T);
        issue_qgemm(y.q_gu, m.x, m.qa_x, m.gu[1], T);
        if (pin::router_fused(T, R.topk_scatter, router_split())) {
            RAD_ISSUE_N(c, R.topk_scatter, 1, brows(m.lg, 1), RAD_W(rw_h.bias), RAD_W(rw_h.scale),
                        brows(m.ids, 1), brows(m.rw, 1), brows(m.sorted, P), brows(m.eoff, m.E + 1), brows(m.ecnt, m.E));
        } else {
            RAD_ISSUE_N(c, R.topk, T, brows(m.lg, T), RAD_W(rw_h.bias), RAD_W(rw_h.scale), brows(m.ids, T), brows(m.rw, T));
            RAD_ISSUE_N(c, R.scatter, T, brows(m.ids, T), brows(m.sorted, P), brows(m.eoff, m.E + 1), brows(m.ecnt, m.E));
        }
        if (rw_h.mx) issue_qact(R.qop_xl, m.qa_xl, m.xl, T);
        if (y.need_qact && need_a8(y.q_down, T) && y.act_q) {
            RAD_ISSUE_N(c, y.act_q, T, brows(m.gu[1], T),
                        brow_slice(m.qa_act[1].q, 0, T, m.qa_act[1].K),
                        brow_slice(m.qa_act[1].s, 0, T, m.qa_act[1].K / 128));
        } else {
            RAD_ISSUE_N(c, y.act, T, brows(m.gu[1], T), brows(m.act[1], T));
            if (y.need_qact && need_a8(y.q_down, T)) issue_qact(y.qop_act, m.qa_act[1], m.act[1], T);
        }
        if (rw_h.mx) {
            const int64_t n0 = (m.E + 1) / 2, n1 = m.E / 2;
            RAD_ISSUE_N(c, R.gu_q, T, brows(m.qa_xl.q, T), brows(m.qa_xl.s, T),
                        RAD_WTAB(rw_h.gu_c0[0], n0), RAD_WTAB(rw_h.gu_s0[0], n0),
                        brows(m.sorted, P), brows(m.eoff, m.E + 1),
                        RAD_WTAB(rw_h.gu_c1[0], n1), RAD_WTAB(rw_h.gu_s1[0], n1), brows(m.egu, P));
        } else {
            RAD_ISSUE_N(c, R.gate, T, brows(m.xl, T), RAD_WTAB(rw_h.ge[0], m.E), brows(m.sorted, P),
                        brows(m.eoff, m.E + 1), bcol(m.egu, 0, m.Fe, P));
            RAD_ISSUE_N(c, R.up, T, brows(m.xl, T), RAD_WTAB(rw_h.ue[0], m.E), brows(m.sorted, P),
                        brows(m.eoff, m.E + 1), bcol(m.egu, m.Fe, m.Fe, P));
        }
        issue_qgemm_op(y.q_down, m.act[1], m.qa_act[1], ys_opd, T);
        if (rw_h.mx) {
            const int64_t n0 = (m.E + 1) / 2, n1 = m.E / 2;
            if (R.act_q) {
                RAD_ISSUE_N(c, R.act_q, P, brows(m.egu, P),
                            brow_slice(m.qa_eact.q, 0, P, m.qa_eact.K),
                            brow_slice(m.qa_eact.s, 0, P, m.qa_eact.K / 128));
            } else {
                RAD_ISSUE_N(c, R.act, P, brows(m.egu, P), brows(m.eact, P));
                issue_qact(R.qop_eact, m.qa_eact, m.eact, P);
            }
            RAD_ISSUE_N(c, R.dn_q, T, brows(m.qa_eact.q, P), brows(m.qa_eact.s, P),
                        RAD_WTAB(rw_h.dn_c0[0], n0), RAD_WTAB(rw_h.dn_s0[0], n0),
                        brows(m.sorted, P), brows(m.eoff, m.E + 1),
                        RAD_WTAB(rw_h.dn_c1[0], n1), RAD_WTAB(rw_h.dn_s1[0], n1), brows(m.edn, P));
        } else {
            RAD_ISSUE_N(c, R.act, P, brows(m.egu, P), brows(m.eact, P));
            RAD_ISSUE_N(c, R.down, T, brows(m.eact, P), RAD_WTAB(rw_h.de[0], m.E), brows(m.sorted, P),
                        brows(m.eoff, m.E + 1), brows(m.edn, P));
        }
        RAD_ISSUE_N(c, R.gather, T, brows(m.edn, P), brows(m.rw, T), brows(m.sorted, P), RAD_NONE, RAD_NONE, comb_opd);
        if (R.ffn_ar) {
            RAD_ISSUE_N(c, R.ffn_ar, T * (m.hidden + m.Lt), brows(m.ffn_ar, T), RAD_NONE);
        } else {
            if (y.down_ar) RAD_ISSUE_N(c, y.down_ar, T * m.hidden, brows(m.ys, T), RAD_NONE);
            if (R.comb_ar) RAD_ISSUE_N(c, R.comb_ar, T * m.Lt, brows(m.comb2, T), RAD_NONE);
        }
        RAD_ISSUE_N(c, R.norm, T, comb_opd, RAD_W(rw_h.norm), brows(m.nl, T));
        if (R.need_qnl && need_a8(R.q_fc2, T)) issue_qact(R.qop_nl, m.qa_nl, m.nl, T);
        issue_qgemm(R.q_fc2, m.nl, m.qa_nl, m.ro, T);
        RAD_ISSUE_N(c, R.add_post, T, brows(m.ro, T), ys_opd, brows(m.h[1], T), brows(m.post, T),
                    brows(m.comb, T), brows(m.h[0], T));
    }

    if (O <= 0) return;
    RAD_ISSUE_N(c, ds.gather, O, brows(m.h[0], T), praw(batch->out_ids, RAD_I32, O), brows(m.hout, O));
    RAD_ISSUE_N(c, ds.head, O, brows(m.hout, O), RAD_W(ds.head_fn), RAD_W(ds.head_scale), RAD_W(ds.head_base), brows(m.hx, O));
    RAD_ISSUE_N(c, ds.norm, O, brows(m.hx, O), RAD_W(ds.out_norm), brows(m.hn, O));
    RAD_ISSUE_N(c, ds.proj, O, brows(m.hn, O), RAD_W(m.lm_head), brows(m.lgt, O));
    RAD_ISSUE_N(c, ds.lgt_cast, O, brows(m.lgt, O), brows(ds.lgt_bf16, O));
    if (m.world > 1) {
        RAD_ISSUE_N(c, ds.topk, O, brows(ds.lgt_bf16, O), brows(ds.cand, O), brows(ds.unary, O), brows(ds.pair, O));
        RAD_ISSUE_N(c, ds.gather_topk, O * ds.top_k * 2, brows(ds.pair, O), brows(ds.gath, O));
        RAD_ISSUE_N(c, ds.merge_topk, O, brows(ds.gath, O), brows(ds.cand, O), brows(ds.unary, O));
    } else {
        RAD_ISSUE_N(c, ds.topk, O, brows(ds.lgt_bf16, O), brows(ds.cand, O), brows(ds.unary, O), RAD_NONE);
    }
    RAD_ISSUE_N(c, ds.sel, S, brows(ds.cand, O), brows(ds.unary, O), RAD_NONE,
                praw(batch->token_ids, RAD_I32, T), RAD_W(ds.mk1), RAD_W(ds.mk2), brows(ds.tokens, S));
}

/* ---- The collectives overlap's step (see ov_declare). A segment is the program between two
 * collectives: 0 the embedding; for layer l, 1 + 3l the previous layer's feed-forward tail (or
 * hc_enter) and the attention up to the VHA rank projection, 2 + 3l the VHA apply to the output
 * projection, 3 + 3l the mHC post and the feed-forward up to its reduction; 1 + 3L the last
 * layer's tail. Collective k follows segment k. */
struct OvPiece {
    RadCtx* c;
    const RadBatch* batch;
    const FfnProgram* m;
    const std::vector<int64_t>* pitch;
    int64_t r0, n;
    const RadKVGroupBatch *kl, *ki, *kw, *ks, *kc;
    int64_t pof(rad_buf h) const { return h < pitch->size() ? (*pitch)[h] : 0; }
    RadOperand b(rad_buf h) const { return rad::arch::brow_slice(h, r0, n, pof(h)); }
    RadOperand bk(rad_buf h, int64_t k) const { return rad::arch::brow_slice(h, r0 * k, n * k, pof(h)); }
    RadOperand col(rad_buf h, int64_t c0, int64_t w) const { return rad::arch::bcol_at(h, r0, pof(h), c0, w, n); }
    RadOperand colk(rad_buf h, int64_t k, int64_t c0, int64_t w) const {
        return rad::arch::bcol_at(h, r0 * k, pof(h), c0, w, n * k);
    }
    RadOperand pi(const int32_t* p) const { return rad::arch::praw(p + r0, RAD_I32, n); }
    RadOperand q8(const QAct& qa) const { return rad::arch::brow_slice(qa.q, r0, n, qa.K); }
    RadOperand s8(const QAct& qa) const { return rad::arch::brow_slice(qa.s, r0, n, qa.K / 128); }
    void qact(rad_op qop, const QAct& qa, rad_buf src) const {
        RAD_ISSUE_N(c, qop, n, b(src), q8(qa), s8(qa));
    }
    /* the serial step's rule (integration round 2-4: ERNIE_A16_MAX, 0 in the row-identical bands), at the piece's row count */
    bool use_a16(const QOp& qo) const { return pin::use_a16(qo.w.fp8, n, a16_max_rows(), qo.op_a16 != 0); }
    bool need_a8(const QOp& qo) const { return qo.w.fp8 && !use_a16(qo); }
    void qgemm_op(const QOp& qo, rad_buf x_bf16, const QAct& qa, RadOperand y_opd) const {
        if (!qo.w.fp8) {
            RAD_ISSUE_N(c, qo.op, n, b(x_bf16), RAD_W(qo.w.w), y_opd);
        } else if (use_a16(qo)) {
            RAD_ISSUE_N(c, qo.op_a16, n, b(x_bf16), RAD_NONE, RAD_W(qo.w.w), RAD_W(qo.w.s), y_opd, RAD_NONE, RAD_NONE);
        } else {
            RAD_ISSUE_N(c, qo.op, n, q8(qa), s8(qa), RAD_W(qo.w.w), RAD_W(qo.w.s), y_opd, RAD_NONE, RAD_NONE);
        }
    }
    void qgemm(const QOp& qo, rad_buf x_bf16, const QAct& qa, rad_buf y_out) const { qgemm_op(qo, x_bf16, qa, b(y_out)); }
};

/* The feed-forward tail of layer l (after its reduction): the step's ops from R.norm (routed) or
 * y.post (dense) on. */
static void ov_ffn_tail(const OvPiece& p, int64_t l) {
    const FfnProgram& m = *p.m;
    RadCtx* c = p.c;
    const FfnLayer& y = m.L[(size_t)l];
    const rad_buf cur = y.ffn_in, nxt = y.ffn_out;
    if (y.routed) {
        const RoutedOps& R = y.R;
        const RadOperand ys_opd = R.combine_ar ? p.col(m.ffn_ar, 0, m.hidden) : p.b(m.ys);
        const RadOperand comb_opd = R.combine_ar ? p.col(m.ffn_ar, m.hidden, m.Lt) : p.b(m.comb2);
        RAD_ISSUE_N(c, R.norm, p.n, comb_opd, RAD_W(R.w.norm), p.b(m.nl));
        if (R.need_qnl && p.need_a8(R.q_fc2)) p.qact(R.qop_nl, m.qa_nl, m.nl);
        p.qgemm(R.q_fc2, m.nl, m.qa_nl, m.ro);
        RAD_ISSUE_N(c, R.add_post, p.n, p.b(m.ro), ys_opd, p.b(cur), p.b(m.post), p.b(m.comb), p.b(nxt));
    } else {
        RAD_ISSUE_N(c, y.post, p.n, p.b(m.y), p.b(cur), p.b(m.post), p.b(m.comb), p.b(nxt));
    }
    /* the drafter's taps (the serial step's, after the layer's feed-forward), on the piece's rows */
    if (m.ds.on) {
        for (int ti = 0; ti < (int)m.ds.n_taps; ++ti)
            if (m.ds.target_layers[(size_t)ti] == l)
                RAD_ISSUE_N(c, m.ds.tap_mean[(size_t)ti], p.n, p.b(nxt), p.col(m.ds.taps, (int64_t)ti * m.hidden, m.hidden));
    }
}

static void ov_segment(const OvPiece& p, int64_t k) {
    const FfnProgram& m = *p.m;
    RadCtx* c = p.c;
    const RadBatch* batch = p.batch;
    const int64_t n = p.n, S = batch->n_seq;
    if (k == 0) {
        RAD_ISSUE_N(c, m.embed, n, p.pi(batch->token_ids), RAD_W(m.tok), p.b(m.emb));
        return;
    }
    const int64_t l = (k - 1) / 3, ph = (k - 1) % 3;
    if (l == m.layers) {   /* 1 + 3L: the last layer's tail */
        ov_ffn_tail(p, l - 1);
        return;
    }
    const FfnLayer& y = m.L[(size_t)l];
    const RadOperand pos = p.pi(batch->positions);
    if (ph == 0) {
        if (l == 0) RAD_ISSUE_N(c, m.enter, n, p.b(m.emb), p.b(m.h[0]));
        else        ov_ffn_tail(p, l - 1);
        if (y.hca) {
            const HcaLayer& Hc = y.Hc;
            const HcaHandles& w = Hc.w;
            const HcaConfig& hh = m.Hc;
            const RadKVGroupBatch *kw = p.kw, *ks = p.ks, *kc = p.kc;
            const RadOperand wbt = rad::arch::praw2(kw->block_table, RAD_I32, S, kw->block_table_pitch);
            const RadOperand sbt = rad::arch::praw2(ks->block_table, RAD_I32, S, ks->block_table_pitch);
            const RadOperand cbt = rad::arch::praw2(kc->block_table, RAD_I32, S, kc->block_table_pitch);
            RAD_ISSUE_N(c, Hc.pre, n, p.b(m.h[0]), RAD_W(Hc.hc.fn), RAD_W(Hc.hc.scale), RAD_W(Hc.hc.base), RAD_W(Hc.gain),
                        p.b(m.x), p.b(m.post), p.b(m.comb));
            if (Hc.need_qx && (p.need_a8(Hc.q_qa) || p.need_a8(Hc.q_kva))) p.qact(Hc.qop_x, m.qa_x, m.x);
            p.qgemm(Hc.q_qa, m.x, m.qa_x, m.hqa);
            RAD_ISSUE_N(c, Hc.qn, n, p.b(m.hqa), RAD_W(w.qa_norm), p.b(m.hqn));
            if (Hc.need_qhqn && (p.need_a8(Hc.q_qb) || p.need_a8(Hc.q_gate))) p.qact(Hc.qop_hqn, m.qa_hqn, m.hqn);
            p.qgemm(Hc.q_qb, m.hqn, m.qa_hqn, m.hq);
            RAD_ISSUE_N(c, Hc.qp, n, p.b(m.hq), pos);
            p.qgemm(Hc.q_kva, m.x, m.qa_x, m.hkv);
            RAD_ISSUE_N(c, Hc.kvn, n, p.b(m.hkv), RAD_W(w.kv_norm), p.b(m.hkv));
            RAD_ISSUE_N(c, Hc.rk, n, p.b(m.hkv), pos);
            RAD_ISSUE_N(c, Hc.store, n, p.col(m.hkv, 0, hh.D / 2), p.col(m.hkv, hh.D / 2, hh.D / 2), p.pi(kw->slot_mapping),
                        rad::arch::kv_cache(m.kvw, (int)l));
            RAD_ISSUE_N(c, Hc.ckv_csc, n, p.b(m.x), RAD_W(w.ckv), RAD_W(w.cgate), p.b(m.hkc), p.b(m.hsc));
            RAD_ISSUE_N(c, Hc.sw, n, p.b(m.hkc), p.b(m.hsc), RAD_W(w.ape), pos, p.pi(ks->slot_mapping),
                        rad::arch::kv_cache(m.kvs, (int)l));
            RAD_ISSUE_N(c, Hc.gs, n, pos, p.b(m.req), sbt, rad::arch::brows(m.hstate_origin, S), p.b(m.hgs));
            RAD_ISSUE_N(c, Hc.cmp, n, rad::arch::kv_cache(m.kvs, (int)l), p.b(m.hgs), pos, p.pi(kc->slot_mapping),
                        RAD_W(w.cnorm), rad::arch::kv_cache(m.kvc, (int)l));
            RAD_ISSUE_N(c, Hc.att, n, p.b(m.hq), rad::arch::kv_cache(m.kvw, (int)l), rad::arch::kv_cache(m.kvc, (int)l), pos,
                        p.b(m.req), wbt, cbt, rad::arch::brows(m.hwin_origin, S), RAD_W(w.sink), p.b(m.hol));
            RAD_ISSUE_N(c, Hc.vha_proj, n, p.b(m.hol), RAD_W(w.u), p.b(m.hz));
            return;
        }
        const AttnLayer& At = y.At;
        const AttnHandles& w = At.w;
        const AttnConfig& a = m.A;
        const RadKVGroupBatch *kl = p.kl, *ki = p.ki;
        const RadOperand lbt = rad::arch::praw2(kl->block_table, RAD_I32, S, kl->block_table_pitch);
        const RadOperand ibt = rad::arch::praw2(ki->block_table, RAD_I32, S, ki->block_table_pitch);
        RAD_ISSUE_N(c, At.pre, n, p.b(m.h[0]), RAD_W(At.hc.fn), RAD_W(At.hc.scale), RAD_W(At.hc.base), RAD_W(At.gain),
                    p.b(m.x), p.b(m.post), p.b(m.comb));
        if (At.need_qx && (p.need_a8(At.q_qa) || p.need_a8(At.q_kva))) p.qact(At.qop_x, m.qa_x, m.x);
        p.qgemm(At.q_qa, m.x, m.qa_x, m.qa);
        RAD_ISSUE_N(c, At.qn, n, p.b(m.qa), RAD_W(w.qa_norm), p.b(m.qn));
        if (At.need_qqn && (p.need_a8(At.q_qb) || p.need_a8(At.q_iq) || p.need_a8(At.q_gate))) p.qact(At.qop_qn, m.qa_qn, m.qn);
        p.qgemm(At.q_qb, m.qn, m.qa_qn, m.q);
        p.qgemm(At.q_kva, m.x, m.qa_x, m.kv);
        RAD_ISSUE_N(c, At.kvn, n, p.col(m.kv, 0, a.lat), RAD_W(w.kv_norm), p.col(m.kv, 0, a.lat));
        RAD_ISSUE_N(c, At.rq, n, p.b(m.q), pos);
        RAD_ISSUE_N(c, At.rk, n, p.b(m.kv), pos);
        const int64_t cw = a.lat + a.rope;
        if (m.lat_fp8)   /* --kv-cache-dtype fp8: the quantised latent row store (every format) takes the whole row */
            RAD_ISSUE_N(c, At.store, n, p.col(m.kv, 0, cw), p.pi(kl->slot_mapping), rad::arch::kv_cache(m.kvl, (int)l));
        else
            RAD_ISSUE_N(c, At.store, n, p.col(m.kv, 0, cw / 2), p.col(m.kv, cw / 2, cw / 2), p.pi(kl->slot_mapping),
                        rad::arch::kv_cache(m.kvl, (int)l));
        RAD_ISSUE_N(c, At.ik, n, p.b(m.x), RAD_W(w.ik), p.b(m.ik));
        RAD_ISSUE_N(c, At.iw, n, p.b(m.x), RAD_W(w.iw), p.b(m.iw));
        p.qgemm(At.q_iq, m.qn, m.qa_qn, m.iq);
        RAD_ISSUE_N(c, At.ikey, n, p.b(m.ik), RAD_W(w.ik_g), RAD_W(w.ik_b), pos, p.pi(ki->slot_mapping),
                    rad::arch::kv_cache(m.kvi, (int)l));
        RAD_ISSUE_N(c, At.iqn, n, p.b(m.iq), pos, p.b(m.qf), p.b(m.qs));
        if (At.topk_tiled) {
            /* the tiled indexer (ernie55_idx_tile): m.sc is one flat scratch and m.topk_state the op's running top-k, both used by lane 0 only and
             * one piece at a time (a piece's segment completes on lane 0 before the next piece's starts), so they are not row buffers */
            RAD_ISSUE_N(c, At.topk_tiled, n, p.b(m.qf), p.b(m.qs), p.b(m.iw), rad::arch::kv_cache(m.kvi, (int)l), ibt, p.b(m.req), pos, lbt,
                        RAD_B(m.sc), p.b(m.aids), p.b(m.sl), RAD_B(m.topk_state),
                        ki->seqused ? rad::arch::praw(ki->seqused, RAD_I32, S) : RAD_NONE);
        } else {
            RAD_ISSUE_N(c, At.logit, n, p.b(m.qf), p.b(m.qs), p.b(m.iw), rad::arch::kv_cache(m.kvi, (int)l), ibt, p.b(m.req), pos,
                        p.b(m.sc));
            RAD_ISSUE_N(c, At.select, n, p.b(m.sc), pos, p.b(m.req), lbt, p.b(m.aids), p.b(m.sl));
        }
        RAD_ISSUE_N(c, At.slice, n, p.b(m.q), p.b(m.qr));
        RAD_ISSUE_N(c, At.absorb, n, p.b(m.q), RAD_W(w.kvb), p.b(m.qabs));
        RAD_ISSUE_N(c, At.attn, n, p.b(m.qabs), p.b(m.qr), rad::arch::kv_cache(m.kvl, (int)l), p.b(m.sl), RAD_W(w.sink),
                    p.b(m.ol));
        RAD_ISSUE_N(c, At.expand, n, p.b(m.ol), RAD_W(w.kvb), p.b(m.ov));
        RAD_ISSUE_N(c, At.vha_proj, n, p.b(m.ov), RAD_W(w.u), p.b(m.oz));
        return;
    }
    if (ph == 1) {
        if (y.hca) {
            const HcaLayer& Hc = y.Hc;
            const HcaHandles& w = Hc.w;
            RAD_ISSUE_N(c, Hc.vha_apply, n, p.b(m.hol), p.b(m.hz), RAD_W(w.v), p.b(m.hom));
            RAD_ISSUE_N(c, Hc.wo, n, p.b(m.hom), pos, RAD_W(w.og), p.b(m.hwo));
            p.qgemm(Hc.q_gate, m.hqn, m.qa_hqn, m.hgt);
            RAD_ISSUE_N(c, Hc.gmul, n, p.b(m.hgt), p.b(m.hwo), p.b(m.hog));
            if (Hc.need_qhog && p.need_a8(Hc.q_out)) p.qact(Hc.qop_hog, m.qa_hog, m.hog);
            p.qgemm(Hc.q_out, m.hog, m.qa_hog, m.y);
            return;
        }
        const AttnLayer& At = y.At;
        const AttnHandles& w = At.w;
        RAD_ISSUE_N(c, At.vha_apply, n, p.b(m.ov), p.b(m.oz), RAD_W(w.v), p.b(m.om));
        p.qgemm(At.q_gate, m.qn, m.qa_qn, m.gt);
        RAD_ISSUE_N(c, At.gmul, n, p.b(m.gt), p.b(m.om), p.b(m.og));
        if (At.need_qog && p.need_a8(At.q_out)) p.qact(At.qop_og, m.qa_og, m.og);
        p.qgemm(At.q_out, m.og, m.qa_og, m.y);
        return;
    }
    /* ph == 2: the attention's mHC post, then the feed-forward up to its reduction */
    const rad_buf cur = y.ffn_in;
    if (y.hca) RAD_ISSUE_N(c, y.Hc.post, n, p.b(m.y), p.b(m.h[0]), p.b(m.post), p.b(m.comb), p.b(m.h[1]));
    else       RAD_ISSUE_N(c, y.At.post, n, p.b(m.y), p.b(m.h[0]), p.b(m.post), p.b(m.comb), p.b(m.h[1]));
    RAD_ISSUE_N(c, y.pre, n, p.b(cur), RAD_W(y.fn), RAD_W(y.scale), RAD_W(y.base), RAD_W(y.gain), p.b(m.x), p.b(m.post),
                p.b(m.comb));
    if (y.routed) {
        const RoutedOps& R = y.R;
        const RoutedHandles& w = R.w;
        const int64_t K = m.K, P = n * K;
        const RadOperand ys_opd = R.combine_ar ? p.col(m.ffn_ar, 0, m.hidden) : p.b(m.ys);
        const RadOperand comb_opd = R.combine_ar ? p.col(m.ffn_ar, m.hidden, m.Lt) : p.b(m.comb2);
        if (y.need_qx && (p.need_a8(R.q_fc1) || p.need_a8(y.q_gu))) p.qact(y.qop_x, m.qa_x, m.x);
        RAD_ISSUE_N(c, R.r01, n, p.b(m.x), RAD_W(w.r0), RAD_W(w.r1), p.col(m.lg, 0, m.E), p.col(m.lg, m.E, m.E));
        p.qgemm(R.q_fc1, m.x, m.qa_x, m.xl);
        p.qgemm(y.q_gu, m.x, m.qa_x, m.gu[y.kind]);
        RAD_ISSUE_N(c, R.topk, n, p.b(m.lg), RAD_W(w.bias), RAD_W(w.scale), p.b(m.ids), p.b(m.rw));
        RAD_ISSUE_N(c, R.scatter, n, p.b(m.ids), p.bk(m.sorted, K), rad::arch::brows(m.eoff, m.E + 1),
                    rad::arch::brows(m.ecnt, m.E));
        if (w.mx) p.qact(R.qop_xl, m.qa_xl, m.xl);
        if (y.need_qact && p.need_a8(y.q_down) && y.act_q) {
            RAD_ISSUE_N(c, y.act_q, n, p.b(m.gu[y.kind]), p.q8(m.qa_act[y.kind]), p.s8(m.qa_act[y.kind]));
        } else {
            RAD_ISSUE_N(c, y.act, n, p.b(m.gu[y.kind]), p.b(m.act[y.kind]));
            if (y.need_qact && p.need_a8(y.q_down)) p.qact(y.qop_act, m.qa_act[y.kind], m.act[y.kind]);
        }
        const RadOperand eq8 = rad::arch::brow_slice(m.qa_eact.q, p.r0 * K, P, m.qa_eact.K);
        const RadOperand es8 = rad::arch::brow_slice(m.qa_eact.s, p.r0 * K, P, m.qa_eact.K / 128);
        if (w.mx) {
            const int64_t n0 = (m.E + 1) / 2, n1 = m.E / 2;
            RAD_ISSUE_N(c, R.gu_q, n, p.q8(m.qa_xl), p.s8(m.qa_xl), RAD_WTAB(w.gu_c0[0], n0), RAD_WTAB(w.gu_s0[0], n0),
                        p.bk(m.sorted, K), rad::arch::brows(m.eoff, m.E + 1), RAD_WTAB(w.gu_c1[0], n1),
                        RAD_WTAB(w.gu_s1[0], n1), p.bk(m.egu, K));
        } else {
            RAD_ISSUE_N(c, R.gate, n, p.b(m.xl), RAD_WTAB(w.ge[0], m.E), p.bk(m.sorted, K), rad::arch::brows(m.eoff, m.E + 1),
                        p.colk(m.egu, K, 0, m.Fe));
            RAD_ISSUE_N(c, R.up, n, p.b(m.xl), RAD_WTAB(w.ue[0], m.E), p.bk(m.sorted, K), rad::arch::brows(m.eoff, m.E + 1),
                        p.colk(m.egu, K, m.Fe, m.Fe));
        }
        p.qgemm_op(y.q_down, m.act[y.kind], m.qa_act[y.kind], ys_opd);
        if (w.mx) {
            const int64_t n0 = (m.E + 1) / 2, n1 = m.E / 2;
            if (R.act_q) {
                RAD_ISSUE_N(c, R.act_q, P, p.bk(m.egu, K), eq8, es8);
            } else {
                RAD_ISSUE_N(c, R.act, P, p.bk(m.egu, K), p.bk(m.eact, K));
                RAD_ISSUE_N(c, R.qop_eact, P, p.bk(m.eact, K), eq8, es8);
            }
            RAD_ISSUE_N(c, R.dn_q, n, eq8, es8, RAD_WTAB(w.dn_c0[0], n0), RAD_WTAB(w.dn_s0[0], n0), p.bk(m.sorted, K),
                        rad::arch::brows(m.eoff, m.E + 1), RAD_WTAB(w.dn_c1[0], n1), RAD_WTAB(w.dn_s1[0], n1),
                        p.bk(m.edn, K));
        } else {
            RAD_ISSUE_N(c, R.act, P, p.bk(m.egu, K), p.bk(m.eact, K));
            RAD_ISSUE_N(c, R.down, n, p.bk(m.eact, K), RAD_WTAB(w.de[0], m.E), p.bk(m.sorted, K),
                        rad::arch::brows(m.eoff, m.E + 1), p.bk(m.edn, K));
        }
        RAD_ISSUE_N(c, R.gather, n, p.bk(m.edn, K), p.b(m.rw), p.bk(m.sorted, K), RAD_NONE, RAD_NONE, comb_opd);
        return;
    }
    if (y.need_qx && p.need_a8(y.q_gu)) p.qact(y.qop_x, m.qa_x, m.x);
    p.qgemm(y.q_gu, m.x, m.qa_x, m.gu[y.kind]);
    if (y.need_qact && p.need_a8(y.q_down) && y.act_q) {
        RAD_ISSUE_N(c, y.act_q, n, p.b(m.gu[y.kind]), p.q8(m.qa_act[y.kind]), p.s8(m.qa_act[y.kind]));
    } else {
        RAD_ISSUE_N(c, y.act, n, p.b(m.gu[y.kind]), p.b(m.act[y.kind]));
        if (y.need_qact && p.need_a8(y.q_down)) p.qact(y.qop_act, m.qa_act[y.kind], m.act[y.kind]);
    }
    p.qgemm(y.q_down, m.act[y.kind], m.qa_act[y.kind], m.y);
}

/* Collective k (after segment k) of one piece, on whatever lane is current (lane 1). */
static void ov_collective(const OvPiece& p, int64_t k) {
    const FfnProgram& m = *p.m;
    RadCtx* c = p.c;
    const int64_t n = p.n;
    if (k == 0) {
        RAD_ISSUE_N(c, m.embed_ar, n * m.hidden, p.b(m.emb), RAD_NONE);
        return;
    }
    const int64_t l = (k - 1) / 3, ph = (k - 1) % 3;
    if (l >= m.layers) return;
    const FfnLayer& y = m.L[(size_t)l];
    if (ph == 0) {
        if (y.hca) RAD_ISSUE_N(c, y.Hc.vha_ar, n * y.Hc.vha_rank * m.Hc.D, p.b(m.hz), RAD_NONE);
        else       RAD_ISSUE_N(c, y.At.vha_ar, n * y.At.vha_rank * m.A.vdim, p.b(m.oz), RAD_NONE);
    } else if (ph == 1) {
        RAD_ISSUE_N(c, y.hca ? y.Hc.out_ar : y.At.out_ar, n * m.hidden, p.b(m.y), RAD_NONE);
    } else if (y.routed) {
        const RoutedOps& R = y.R;
        if (R.ffn_ar) {
            RAD_ISSUE_N(c, R.ffn_ar, n * (m.hidden + m.Lt), p.b(m.ffn_ar), RAD_NONE);
        } else {
            RAD_ISSUE_N(c, y.down_ar, n * m.hidden, p.b(m.ys), RAD_NONE);
            RAD_ISSUE_N(c, R.comb_ar, n * m.Lt, p.b(m.comb2), RAD_NONE);
        }
    } else {
        RAD_ISSUE_N(c, y.down_ar, n * m.hidden, p.b(m.y), RAD_NONE);
    }
}

/* Before the first pipelined step: every concurrent row buffer has a recorded pitch and its own
 * arena bytes (the rad_buf_concurrent promise, checked rather than trusted). */
static bool ov_plan_ok(RadCtx* c, const FfnProgram& m, int slot, std::string* why) {
    struct Span { uintptr_t lo, hi; rad_buf h; };
    std::vector<Span> v;
    const int64_t P = m.T * m.K;
    for (const OvBuf& r : g_ov_bufs[slot]) {
        if (r.kind != RAD_BUF_TRANSIENT) continue;
        if (r.h == m.hout || r.h == m.hx || r.h == m.hn || r.h == m.lgt || r.h == m.ov_all || r.h == m.hol_all) continue;
        if (r.rows != m.T && !(m.moe && r.rows == P)) continue;
        if (r.pitch <= 0 || r.esize <= 0) { *why = "a row buffer without a pitch or element size"; return false; }
        const uintptr_t lo = (uintptr_t)rad_buf_ptr(c, r.h);
        if (!lo) continue;   /* not placed (an unused optional buffer) */
        v.push_back({lo, lo + (uintptr_t)(r.rows * r.pitch * r.esize), r.h});
    }
    std::sort(v.begin(), v.end(), [](const Span& a, const Span& b) { return a.lo < b.lo; });
    for (size_t i = 1; i < v.size(); ++i)
        if (v[i].lo < v[i - 1].hi) {
            *why = "two row buffers share arena bytes (handles " + std::to_string(v[i - 1].h) + " and " +
                   std::to_string(v[i].h) + ")";
            return false;
        }
    return true;
}

/* Runs the step as two pieces if it qualifies; false leaves it to the serial step. */
static bool ov_step(RadCtx* c, const RadBatch* batch) {
    using namespace rad::arch;
    const int slot = rad_slot(c);
    OvState& ov = g_ov[slot];
    const FfnProgram& m = g_ffn[slot];
    const int64_t T = batch->n_tok, S = batch->n_seq;
    if (!ov.on || batch->phase != RAD_PHASE_PREFILL || batch->n_seq_decode != 0 || T < ov.min_rows || T > m.T ||
        batch->score_only)
        return false;
    const int64_t R0 = T / 2, R1 = T - R0;
    const int64_t floor_rows = std::max<int64_t>({8, m.mla_vha_dec_max, m.hca_vha_dec_max});
    if (R0 <= floor_rows || R1 <= floor_rows) return false;
    if (!ov.checked) {
        ov.checked = true;
        std::string why;
        ov.ok = ov_plan_ok(c, m, slot, &why);
        if (!ov.ok && m.rank == 0)
            fprintf(stderr, "radiance: ernie55: collectives overlap off for this program: %s; steps run serially\n",
                    why.c_str());
    }
    if (!ov.ok) return false;
    OvPiece p{c, batch, &m, &g_ov_pitch[slot], 0, 0, nullptr, nullptr, nullptr, nullptr, nullptr};
    if (m.kvl) {
        p.kl = kv_batch(batch, m.kvl);
        p.ki = kv_batch(batch, m.kvi);
        if (!p.kl || !p.ki || !p.kl->slot_mapping || !p.kl->block_table || !p.ki->slot_mapping || !p.ki->block_table) {
            rad_step_fail(c, "ernie55 attention requires the latent and index caches' slot mappings and block tables");
            return true;
        }
    }
    if (m.kvw) {
        p.kw = kv_batch(batch, m.kvw);
        p.ks = kv_batch(batch, m.kvs);
        p.kc = kv_batch(batch, m.kvc);
        if (!p.kw || !p.ks || !p.kc || !p.kw->slot_mapping || !p.kw->block_table || !p.ks->slot_mapping ||
            !p.ks->block_table || !p.kc->slot_mapping || !p.kc->block_table) {
            rad_step_fail(c, "ernie55 HCA attention requires the window, state and compressed caches' slot mappings and block tables");
            return true;
        }
    }
    if (!ov.noted) {
        ov.noted = true;
        if (m.rank == 0)
            fprintf(stderr, "radiance: ernie55: a %lld-row prefill step runs as two pieces (%lld + %lld rows), "
                            "all_reduce on the second lane\n", (long long)T, (long long)R0, (long long)R1);
    }
    RAD_ISSUE(c, m.row_req, praw(batch->cu_seqlens, RAD_I32, S + 1), brows(m.req, T));
    const int64_t r0s[2] = {0, R0}, ns[2] = {R0, R1};
    const int64_t K = 2 + 3 * m.layers;   /* segments 0 .. 1 + 3L */
    for (int64_t k = 0; k < K; ++k) {
        for (int i = 0; i < 2; ++i) {
            p.r0 = r0s[i];
            p.n = ns[i];
            ov_segment(p, k);
            /* Lane 0 goes on behind every collective issued so far: the one this segment's other
             * piece (or the previous segment) needs next. */
            rad_lane_join(c, 1, 0);
            if (k + 1 < K) {
                rad_lane(c, 1);
                rad_lane_join(c, 0, 1);   /* this piece's collective waits for its segment */
                ov_collective(p, k);
                rad_lane(c, 0);
            }
        }
    }
    const int64_t Rn = batch->n_out;
    if (Rn <= 0) return true;
    RAD_ISSUE_N(c, m.gather, Rn, brows(m.h[m.fin], T), praw(batch->out_ids, RAD_I32, Rn), brows(m.hout, Rn));
    RAD_ISSUE_N(c, m.head, Rn, brows(m.hout, Rn), RAD_W(m.head_fn), RAD_W(m.head_scale), RAD_W(m.head_base), brows(m.hx, Rn));
    RAD_ISSUE_N(c, m.norm, Rn, brows(m.hx, Rn), RAD_W(m.out_norm), brows(m.hn, Rn));
    RAD_ISSUE_N(c, m.proj, Rn, brows(m.hn, Rn), RAD_W(m.lm_head), brows(m.lgt, Rn));
    RAD_ISSUE_N(c, m.mmax, Rn, brows(m.lgt, Rn), RAD_W(m.mm_r), RAD_W(m.mm_t), brows(m.lgt, Rn));
    return true;
}

static void step(RadCtx* c, const RadBatch* batch) {
    using namespace rad::arch;
    const FfnProgram& m = g_ffn[rad_slot(c)];
    if (!m.on) return;
    if (batch->draft_pass < 0) { emit_dspark_context(c, m, batch); return; }
    if (batch->draft_pass > 0) { emit_dspark_query(c, m, batch); return; }
    const int64_t T = batch->n_tok, S = batch->n_seq;
    if (T <= 0 || S <= 0) return;
    if (g_ov[rad_slot(c)].on && ov_step(c, batch)) return;
    RAD_ISSUE(c, m.embed, praw(batch->token_ids, RAD_I32, T), RAD_W(m.tok), brows(m.emb, T));
    if (m.embed_ar) RAD_ISSUE_N(c, m.embed_ar, T * m.hidden, brows(m.emb, T), RAD_NONE);
    RAD_ISSUE(c, m.enter, brows(m.emb, T), brows(m.h[0], T));
    const RadKVGroupBatch *kl = nullptr, *ki = nullptr, *kw = nullptr, *ks = nullptr, *kc = nullptr;
    if (m.attn) {
        if (m.kvl) {
            kl = kv_batch(batch, m.kvl);
            ki = kv_batch(batch, m.kvi);
            if (!kl || !ki || !kl->slot_mapping || !kl->block_table || !ki->slot_mapping || !ki->block_table) {
                rad_step_fail(c, "ernie55 attention requires the latent and index caches' slot mappings and block tables");
                return;
            }
        }
        if (m.kvw) {
            kw = kv_batch(batch, m.kvw);
            ks = kv_batch(batch, m.kvs);
            kc = kv_batch(batch, m.kvc);
            if (!kw || !ks || !kc || !kw->slot_mapping || !kw->block_table || !ks->slot_mapping || !ks->block_table || !kc->slot_mapping ||
                !kc->block_table) {
                rad_step_fail(c, "ernie55 HCA attention requires the window, state and compressed caches' slot mappings and block tables");
                return;
            }
        }
        RAD_ISSUE(c, m.row_req, praw(batch->cu_seqlens, RAD_I32, S + 1), brows(m.req, T));
    }
    for (int64_t l = 0; l < m.layers; ++l) {
        const FfnLayer& y = m.L[(size_t)l];
        const rad_buf cur = y.ffn_in, nxt = y.ffn_out;
        const int64_t I = y.width;
        if (m.attn && y.hca) {
            const HcaLayer& Hc = y.Hc;
            const HcaHandles& w = Hc.w;
            const HcaConfig& hh = m.Hc;
            const RadOperand pos = praw(batch->positions, RAD_I32, T);
            const RadOperand wbt = praw2(kw->block_table, RAD_I32, S, kw->block_table_pitch);
            const RadOperand sbt = praw2(ks->block_table, RAD_I32, S, ks->block_table_pitch);
            const RadOperand cbt = praw2(kc->block_table, RAD_I32, S, kc->block_table_pitch);
            RAD_ISSUE(c, Hc.pre, brows(m.h[0], T), RAD_W(Hc.hc.fn), RAD_W(Hc.hc.scale), RAD_W(Hc.hc.base), RAD_W(Hc.gain), brows(m.x, T),
                      brows(m.post, T), brows(m.comb, T));
            auto issue_qact = [&](rad_op qop, const QAct& qa, rad_buf src, int64_t rows) {
                RAD_ISSUE_N(c, qop, rows, brow_slice(src, 0, rows, qa.K),
                            brow_slice(qa.q, 0, rows, qa.K), brow_slice(qa.s, 0, rows, qa.K / 128));
            };
            auto use_a16 = [&](const QOp& qo, int64_t rows) { return pin::use_a16(qo.w.fp8, rows, a16_max_rows(), qo.op_a16 != 0); };
            auto need_a8 = [&](const QOp& qo, int64_t rows) { return qo.w.fp8 && !use_a16(qo, rows); };
            auto issue_qgemm = [&](const QOp& qo, rad_buf x_bf16, const QAct& qa, rad_buf y_out, int64_t rows) {
                if (!qo.w.fp8) {
                    RAD_ISSUE_N(c, qo.op, rows, brows(x_bf16, rows), RAD_W(qo.w.w), brows(y_out, rows));
                } else if (use_a16(qo, rows)) {
                    RAD_ISSUE_N(c, qo.op_a16, rows, brows(x_bf16, rows), RAD_NONE,
                                RAD_W(qo.w.w), RAD_W(qo.w.s), brows(y_out, rows), RAD_NONE, RAD_NONE);
                } else {
                    RAD_ISSUE_N(c, qo.op, rows, brows(qa.q, rows), brows(qa.s, rows),
                                RAD_W(qo.w.w), RAD_W(qo.w.s), brows(y_out, rows), RAD_NONE, RAD_NONE);
                }
            };
            if (m.quant_fwd) {
                if (Hc.need_qx && (need_a8(Hc.q_qa, T) || need_a8(Hc.q_kva, T))) issue_qact(Hc.qop_x, m.qa_x, m.x, T);
                issue_qgemm(Hc.q_qa, m.x, m.qa_x, m.hqa, T);
                RAD_ISSUE(c, Hc.qn, brows(m.hqa, T), RAD_W(w.qa_norm), brows(m.hqn, T));
                if (Hc.need_qhqn && (need_a8(Hc.q_qb, T) || need_a8(Hc.q_gate, T))) issue_qact(Hc.qop_hqn, m.qa_hqn, m.hqn, T);
                issue_qgemm(Hc.q_qb, m.hqn, m.qa_hqn, m.hq, T);
                RAD_ISSUE(c, Hc.qp, brows(m.hq, T), pos);
                issue_qgemm(Hc.q_kva, m.x, m.qa_x, m.hkv, T);
            } else {
                RAD_ISSUE(c, Hc.qa, brows(m.x, T), RAD_W(w.qa), brows(m.hqa, T));
                RAD_ISSUE(c, Hc.qn, brows(m.hqa, T), RAD_W(w.qa_norm), brows(m.hqn, T));
                RAD_ISSUE(c, Hc.qb, brows(m.hqn, T), RAD_W(w.qb), brows(m.hq, T));
                RAD_ISSUE(c, Hc.qp, brows(m.hq, T), pos);
                RAD_ISSUE(c, Hc.kva, brows(m.x, T), RAD_W(w.kv), brows(m.hkv, T));
            }
            RAD_ISSUE(c, Hc.kvn, brows(m.hkv, T), RAD_W(w.kv_norm), brows(m.hkv, T));
            RAD_ISSUE(c, Hc.rk, brows(m.hkv, T), pos);
            RAD_ISSUE(c, Hc.store, bcol(m.hkv, 0, hh.D / 2, T), bcol(m.hkv, hh.D / 2, hh.D / 2, T), praw(kw->slot_mapping, RAD_I32, T),
                      kv_cache(m.kvw, (int)l));
            RAD_ISSUE(c, Hc.ckv_csc, brows(m.x, T), RAD_W(w.ckv), RAD_W(w.cgate), brows(m.hkc, T), brows(m.hsc, T));
            RAD_ISSUE(c, Hc.sw, brows(m.hkc, T), brows(m.hsc, T), RAD_W(w.ape), pos, praw(ks->slot_mapping, RAD_I32, T), kv_cache(m.kvs, (int)l));
            RAD_ISSUE(c, Hc.gs, pos, brows(m.req, T), sbt, brows(m.hstate_origin, S), brows(m.hgs, T));
            RAD_ISSUE(c, Hc.cmp, kv_cache(m.kvs, (int)l), brows(m.hgs, T), pos, praw(kc->slot_mapping, RAD_I32, T), RAD_W(w.cnorm),
                      kv_cache(m.kvc, (int)l));
            RAD_ISSUE(c, Hc.att, brows(m.hq, T), kv_cache(m.kvw, (int)l), kv_cache(m.kvc, (int)l), pos, brows(m.req, T), wbt, cbt,
                      brows(m.hwin_origin, S), RAD_W(w.sink), brows(m.hol, T));
            if (Hc.vha_ag && T <= m.hca_vha_dec_max) {
                RAD_ISSUE_N(c, Hc.vha_ag, T * m.hca_heads_loc * hh.D, brows(m.hol, T), brows(m.hol_all, m.world * T));
                RAD_ISSUE_N(c, Hc.vha, T, brows(m.hol_all, m.world * T), RAD_W(w.u), RAD_W(w.v), brows(m.hom, T));
            } else if (Hc.vha_proj) {
                RAD_ISSUE_N(c, Hc.vha_proj, T, brows(m.hol, T), RAD_W(w.u), brows(m.hz, T));
                RAD_ISSUE_N(c, Hc.vha_ar, T * Hc.vha_rank * hh.D, brows(m.hz, T), RAD_NONE);
                RAD_ISSUE_N(c, Hc.vha_apply, T, brows(m.hol, T), brows(m.hz, T), RAD_W(w.v), brows(m.hom, T));
            } else {
                RAD_ISSUE(c, Hc.vha, brows(m.hol, T), RAD_W(w.u), RAD_W(w.v), brows(m.hom, T));
            }
            RAD_ISSUE(c, Hc.wo, brows(m.hom, T), pos, RAD_W(w.og), brows(m.hwo, T));
            if (m.quant_fwd) {
                issue_qgemm(Hc.q_gate, m.hqn, m.qa_hqn, m.hgt, T);
                RAD_ISSUE(c, Hc.gmul, brows(m.hgt, T), brows(m.hwo, T), brows(m.hog, T));
                if (Hc.need_qhog && need_a8(Hc.q_out, T)) issue_qact(Hc.qop_hog, m.qa_hog, m.hog, T);
                issue_qgemm(Hc.q_out, m.hog, m.qa_hog, m.y, T);
                if (Hc.out_ar) RAD_ISSUE_N(c, Hc.out_ar, T * m.hidden, brows(m.y, T), RAD_NONE);
            } else {
                RAD_ISSUE(c, Hc.gate, brows(m.hqn, T), RAD_W(w.gate), brows(m.hgt, T));
                RAD_ISSUE(c, Hc.gmul, brows(m.hgt, T), brows(m.hwo, T), brows(m.hog, T));
                RAD_ISSUE(c, Hc.out, brows(m.hog, T), RAD_W(w.o), brows(m.y, T));
            }
            RAD_ISSUE(c, Hc.post, brows(m.y, T), brows(m.h[0], T), brows(m.post, T), brows(m.comb, T), brows(m.h[1], T));
        } else if (m.attn) {
            const AttnLayer& At = y.At;
            const AttnHandles& w = At.w;
            const AttnConfig& a = m.A;
            const RadOperand pos = praw(batch->positions, RAD_I32, T);
            const RadOperand lbt = praw2(kl->block_table, RAD_I32, S, kl->block_table_pitch);
            const RadOperand ibt = praw2(ki->block_table, RAD_I32, S, ki->block_table_pitch);
            RAD_ISSUE(c, At.pre, brows(m.h[0], T), RAD_W(At.hc.fn), RAD_W(At.hc.scale), RAD_W(At.hc.base),
                      RAD_W(At.gain), brows(m.x, T), brows(m.post, T), brows(m.comb, T));
            auto issue_qact_m = [&](rad_op qop, const QAct& qa, rad_buf src, int64_t rows) {
                RAD_ISSUE_N(c, qop, rows, brow_slice(src, 0, rows, qa.K),
                            brow_slice(qa.q, 0, rows, qa.K), brow_slice(qa.s, 0, rows, qa.K / 128));
            };
            auto use_a16_m = [&](const QOp& qo, int64_t rows) { return pin::use_a16(qo.w.fp8, rows, a16_max_rows(), qo.op_a16 != 0); };
            auto need_a8_m = [&](const QOp& qo, int64_t rows) { return qo.w.fp8 && !use_a16_m(qo, rows); };
            auto issue_qgemm_m = [&](const QOp& qo, rad_buf x_bf16, const QAct& qa, rad_buf y_out, int64_t rows) {
                if (!qo.w.fp8) {
                    RAD_ISSUE_N(c, qo.op, rows, brows(x_bf16, rows), RAD_W(qo.w.w), brows(y_out, rows));
                } else if (use_a16_m(qo, rows)) {
                    RAD_ISSUE_N(c, qo.op_a16, rows, brows(x_bf16, rows), RAD_NONE,
                                RAD_W(qo.w.w), RAD_W(qo.w.s), brows(y_out, rows), RAD_NONE, RAD_NONE);
                } else {
                    RAD_ISSUE_N(c, qo.op, rows, brows(qa.q, rows), brows(qa.s, rows),
                                RAD_W(qo.w.w), RAD_W(qo.w.s), brows(y_out, rows), RAD_NONE, RAD_NONE);
                }
            };
            if (m.quant_fwd) {
                if (At.need_qx && (need_a8_m(At.q_qa, T) || need_a8_m(At.q_kva, T))) issue_qact_m(At.qop_x, m.qa_x, m.x, T);
                issue_qgemm_m(At.q_qa, m.x, m.qa_x, m.qa, T);
                RAD_ISSUE(c, At.qn, brows(m.qa, T), RAD_W(w.qa_norm), brows(m.qn, T));
                if (At.need_qqn && (need_a8_m(At.q_qb, T) || need_a8_m(At.q_iq, T) || need_a8_m(At.q_gate, T)))
                    issue_qact_m(At.qop_qn, m.qa_qn, m.qn, T);
                issue_qgemm_m(At.q_qb, m.qn, m.qa_qn, m.q, T);
                issue_qgemm_m(At.q_kva, m.x, m.qa_x, m.kv, T);
            } else {
                RAD_ISSUE(c, At.qa, brows(m.x, T), RAD_W(w.qa), brows(m.qa, T));
                RAD_ISSUE(c, At.qn, brows(m.qa, T), RAD_W(w.qa_norm), brows(m.qn, T));
                RAD_ISSUE(c, At.qb, brows(m.qn, T), RAD_W(w.qb), brows(m.q, T));
                RAD_ISSUE(c, At.kva, brows(m.x, T), RAD_W(w.kva), brows(m.kv, T));
            }
            RAD_ISSUE(c, At.kvn, bcol(m.kv, 0, a.lat, T), RAD_W(w.kv_norm), bcol(m.kv, 0, a.lat, T));
            RAD_ISSUE(c, At.rq, brows(m.q, T), pos);
            RAD_ISSUE(c, At.rk, brows(m.kv, T), pos);
            const int64_t cw = a.lat + a.rope;
            if (m.lat_fp8)
                RAD_ISSUE(c, At.store, bcol(m.kv, 0, cw, T), praw(kl->slot_mapping, RAD_I32, T), kv_cache(m.kvl, (int)l));
            else
            RAD_ISSUE(c, At.store, bcol(m.kv, 0, cw / 2, T), bcol(m.kv, cw / 2, cw / 2, T),
                      praw(kl->slot_mapping, RAD_I32, T), kv_cache(m.kvl, (int)l));
            /* the indexer */
            RAD_ISSUE(c, At.ik, brows(m.x, T), RAD_W(w.ik), brows(m.ik, T));
            RAD_ISSUE(c, At.iw, brows(m.x, T), RAD_W(w.iw), brows(m.iw, T));
            if (m.quant_fwd) {
                issue_qgemm_m(At.q_iq, m.qn, m.qa_qn, m.iq, T);
            } else {
                RAD_ISSUE(c, At.iq, brows(m.qn, T), RAD_W(w.iq), brows(m.iq, T));
            }
            RAD_ISSUE(c, At.ikey, brows(m.ik, T), RAD_W(w.ik_g), RAD_W(w.ik_b), pos,
                      praw(ki->slot_mapping, RAD_I32, T), kv_cache(m.kvi, (int)l));
            RAD_ISSUE(c, At.iqn, brows(m.iq, T), pos, brows(m.qf, T), brows(m.qs, T));
            if (At.topk_tiled) {
                RAD_ISSUE(c, At.topk_tiled, brows(m.qf, T), brows(m.qs, T), brows(m.iw, T), kv_cache(m.kvi, (int)l), ibt,
                          brows(m.req, T), pos, lbt, RAD_B(m.sc), brows(m.aids, T), brows(m.sl, T), RAD_B(m.topk_state),
                          /* the index groups seqused: the engine reads the steps host-side context bound from it, which sets the tile count of the device op */
                          ki->seqused ? praw(ki->seqused, RAD_I32, S) : RAD_NONE);
            } else {
            RAD_ISSUE(c, At.logit, brows(m.qf, T), brows(m.qs, T), brows(m.iw, T), kv_cache(m.kvi, (int)l), ibt,
                      brows(m.req, T), pos, brows(m.sc, T));
            RAD_ISSUE(c, At.select, brows(m.sc, T), pos, brows(m.req, T), lbt, brows(m.aids, T), brows(m.sl, T));
            }
            /* the attention */
            RAD_ISSUE(c, At.slice, brows(m.q, T), brows(m.qr, T));
            RAD_ISSUE(c, At.absorb, brows(m.q, T), RAD_W(w.kvb), brows(m.qabs, T));
            RAD_ISSUE(c, At.attn, brows(m.qabs, T), brows(m.qr, T), kv_cache(m.kvl, (int)l), brows(m.sl, T),
                      RAD_W(w.sink), brows(m.ol, T));
            RAD_ISSUE(c, At.expand, brows(m.ol, T), RAD_W(w.kvb), brows(m.ov, T));
            if (At.vha_ag && T <= m.mla_vha_dec_max) {
                RAD_ISSUE_N(c, At.vha_ag, T * m.mla_heads_loc * a.vdim, brows(m.ov, T), brows(m.ov_all, m.world * T));
                RAD_ISSUE_N(c, At.vha, T, brows(m.ov_all, m.world * T), RAD_W(w.u), RAD_W(w.v), brows(m.om, T));
            } else if (At.vha_proj) {
                RAD_ISSUE_N(c, At.vha_proj, T, brows(m.ov, T), RAD_W(w.u), brows(m.oz, T));
                RAD_ISSUE_N(c, At.vha_ar, T * At.vha_rank * a.vdim, brows(m.oz, T), RAD_NONE);
                RAD_ISSUE_N(c, At.vha_apply, T, brows(m.ov, T), brows(m.oz, T), RAD_W(w.v), brows(m.om, T));
            } else {
                RAD_ISSUE(c, At.vha, brows(m.ov, T), RAD_W(w.u), RAD_W(w.v), brows(m.om, T));
            }
            if (m.quant_fwd) {
                issue_qgemm_m(At.q_gate, m.qn, m.qa_qn, m.gt, T);
                RAD_ISSUE(c, At.gmul, brows(m.gt, T), brows(m.om, T), brows(m.og, T));
                if (At.need_qog && need_a8_m(At.q_out, T)) issue_qact_m(At.qop_og, m.qa_og, m.og, T);
                issue_qgemm_m(At.q_out, m.og, m.qa_og, m.y, T);
                if (At.out_ar) RAD_ISSUE_N(c, At.out_ar, T * m.hidden, brows(m.y, T), RAD_NONE);
            } else {
                RAD_ISSUE(c, At.gate, brows(m.qn, T), RAD_W(w.gate), brows(m.gt, T));
                RAD_ISSUE(c, At.gmul, brows(m.gt, T), brows(m.om, T), brows(m.og, T));
                RAD_ISSUE(c, At.out, brows(m.og, T), RAD_W(w.o), brows(m.y, T));
            }
            RAD_ISSUE(c, At.post, brows(m.y, T), brows(m.h[0], T), brows(m.post, T), brows(m.comb, T), brows(m.h[1], T));
        }
        RAD_ISSUE(c, y.pre, brows(cur, T), RAD_W(y.fn), RAD_W(y.scale), RAD_W(y.base),
                  RAD_W(y.gain), brows(m.x, T), brows(m.post, T), brows(m.comb, T));
        auto issue_qact_f = [&](rad_op qop, const QAct& qa, rad_buf src, int64_t rows) {
            RAD_ISSUE_N(c, qop, rows, brow_slice(src, 0, rows, qa.K),
                        brow_slice(qa.q, 0, rows, qa.K), brow_slice(qa.s, 0, rows, qa.K / 128));
        };
        auto use_a16_f = [&](const QOp& qo, int64_t rows) { return pin::use_a16(qo.w.fp8, rows, a16_max_rows(), qo.op_a16 != 0); };
        auto need_a8_f = [&](const QOp& qo, int64_t rows) { return qo.w.fp8 && !use_a16_f(qo, rows); };
        auto issue_qgemm_f_op = [&](const QOp& qo, rad_buf x_bf16, const QAct& qa, RadOperand y_opd, int64_t rows) {
            if (!qo.w.fp8) {
                RAD_ISSUE_N(c, qo.op, rows, brows(x_bf16, rows), RAD_W(qo.w.w), y_opd);
            } else if (use_a16_f(qo, rows)) {
                RAD_ISSUE_N(c, qo.op_a16, rows, brows(x_bf16, rows), RAD_NONE,
                            RAD_W(qo.w.w), RAD_W(qo.w.s), y_opd, RAD_NONE, RAD_NONE);
            } else {
                RAD_ISSUE_N(c, qo.op, rows, brows(qa.q, rows), brows(qa.s, rows),
                            RAD_W(qo.w.w), RAD_W(qo.w.s), y_opd, RAD_NONE, RAD_NONE);
            }
        };
        auto issue_qgemm_f = [&](const QOp& qo, rad_buf x_bf16, const QAct& qa, rad_buf y_out, int64_t rows) {
            issue_qgemm_f_op(qo, x_bf16, qa, brows(y_out, rows), rows);
        };
        if (m.quant_fwd && y.routed) {
            const RoutedOps& R = y.R;
            const RoutedHandles& w = R.w;
            const int64_t P = T * m.K;
            const RadOperand ys_opd = R.combine_ar ? bcol(m.ffn_ar, 0, m.hidden, T) : brows(m.ys, T);
            const RadOperand comb_opd = R.combine_ar ? bcol(m.ffn_ar, m.hidden, m.Lt, T) : brows(m.comb2, T);
            if (y.need_qx && (need_a8_f(R.q_fc1, T) || need_a8_f(y.q_gu, T))) issue_qact_f(y.qop_x, m.qa_x, m.x, T);
            RAD_ISSUE(c, R.r01, brows(m.x, T), RAD_W(w.r0), RAD_W(w.r1),
                      bcol(m.lg, 0, m.E, T), bcol(m.lg, m.E, m.E, T));
            issue_qgemm_f(R.q_fc1, m.x, m.qa_x, m.xl, T);
            issue_qgemm_f(y.q_gu, m.x, m.qa_x, m.gu[y.kind], T);
            if (pin::router_fused(T, R.topk_scatter, router_split())) {
                RAD_ISSUE_N(c, R.topk_scatter, 1, brows(m.lg, 1), RAD_W(w.bias), RAD_W(w.scale),
                            brows(m.ids, 1), brows(m.rw, 1), brows(m.sorted, P),
                            brows(m.eoff, m.E + 1), brows(m.ecnt, m.E));
            } else {
                RAD_ISSUE(c, R.topk, brows(m.lg, T), RAD_W(w.bias), RAD_W(w.scale), brows(m.ids, T),
                          brows(m.rw, T));
                RAD_ISSUE(c, R.scatter, brows(m.ids, T), brows(m.sorted, P), brows(m.eoff, m.E + 1),
                          brows(m.ecnt, m.E));
            }
            if (w.mx) issue_qact_f(R.qop_xl, m.qa_xl, m.xl, T);
            if (y.need_qact && need_a8_f(y.q_down, T) && y.act_q) {
                RAD_ISSUE_N(c, y.act_q, T, brows(m.gu[y.kind], T),
                            brow_slice(m.qa_act[y.kind].q, 0, T, m.qa_act[y.kind].K),
                            brow_slice(m.qa_act[y.kind].s, 0, T, m.qa_act[y.kind].K / 128));
            } else {
                RAD_ISSUE(c, y.act, brows(m.gu[y.kind], T), brows(m.act[y.kind], T));
                if (y.need_qact && need_a8_f(y.q_down, T)) issue_qact_f(y.qop_act, m.qa_act[y.kind], m.act[y.kind], T);
            }
            if (w.mx) {
                const int64_t n0 = (m.E + 1) / 2, n1 = m.E / 2;
                RAD_ISSUE_N(c, R.gu_q, T, brows(m.qa_xl.q, T), brows(m.qa_xl.s, T),
                            RAD_WTAB(w.gu_c0[0], n0), RAD_WTAB(w.gu_s0[0], n0),
                            brows(m.sorted, P), brows(m.eoff, m.E + 1),
                            RAD_WTAB(w.gu_c1[0], n1), RAD_WTAB(w.gu_s1[0], n1),
                            brows(m.egu, P));
            } else {
                RAD_ISSUE_N(c, R.gate, T, brows(m.xl, T), RAD_WTAB(w.ge[0], m.E), brows(m.sorted, P),
                            brows(m.eoff, m.E + 1), bcol(m.egu, 0, m.Fe, P));
                RAD_ISSUE_N(c, R.up, T, brows(m.xl, T), RAD_WTAB(w.ue[0], m.E), brows(m.sorted, P),
                            brows(m.eoff, m.E + 1), bcol(m.egu, m.Fe, m.Fe, P));
            }
            issue_qgemm_f_op(y.q_down, m.act[y.kind], m.qa_act[y.kind], ys_opd, T);
            if (w.mx) {
                const int64_t n0 = (m.E + 1) / 2, n1 = m.E / 2;
                if (R.act_q) {
                    RAD_ISSUE_N(c, R.act_q, P, brows(m.egu, P),
                                brow_slice(m.qa_eact.q, 0, P, m.qa_eact.K),
                                brow_slice(m.qa_eact.s, 0, P, m.qa_eact.K / 128));
                } else {
                    RAD_ISSUE_N(c, R.act, P, brows(m.egu, P), brows(m.eact, P));
                    issue_qact_f(R.qop_eact, m.qa_eact, m.eact, P);
                }
                RAD_ISSUE_N(c, R.dn_q, T, brows(m.qa_eact.q, P), brows(m.qa_eact.s, P),
                            RAD_WTAB(w.dn_c0[0], n0), RAD_WTAB(w.dn_s0[0], n0),
                            brows(m.sorted, P), brows(m.eoff, m.E + 1),
                            RAD_WTAB(w.dn_c1[0], n1), RAD_WTAB(w.dn_s1[0], n1),
                            brows(m.edn, P));
            } else {
                RAD_ISSUE_N(c, R.act, P, brows(m.egu, P), brows(m.eact, P));
                RAD_ISSUE_N(c, R.down, T, brows(m.eact, P), RAD_WTAB(w.de[0], m.E), brows(m.sorted, P),
                            brows(m.eoff, m.E + 1), brows(m.edn, P));
            }
            RAD_ISSUE(c, R.gather, brows(m.edn, P), brows(m.rw, T), brows(m.sorted, P), RAD_NONE,
                      RAD_NONE, comb_opd);
            if (R.ffn_ar) {
                RAD_ISSUE_N(c, R.ffn_ar, T * (m.hidden + m.Lt), brows(m.ffn_ar, T), RAD_NONE);
            } else {
                if (y.down_ar) RAD_ISSUE_N(c, y.down_ar, T * m.hidden, brows(m.ys, T), RAD_NONE);
                if (R.comb_ar) RAD_ISSUE_N(c, R.comb_ar, T * m.Lt, brows(m.comb2, T), RAD_NONE);
            }
            RAD_ISSUE(c, R.norm, comb_opd, RAD_W(w.norm), brows(m.nl, T));
            if (R.need_qnl && need_a8_f(R.q_fc2, T)) issue_qact_f(R.qop_nl, m.qa_nl, m.nl, T);
            issue_qgemm_f(R.q_fc2, m.nl, m.qa_nl, m.ro, T);
            RAD_ISSUE(c, R.add_post, brows(m.ro, T), ys_opd, brows(cur, T), brows(m.post, T),
                      brows(m.comb, T), brows(nxt, T));
            if (m.ds.on) {
                for (int ti = 0; ti < (int)m.ds.n_taps; ++ti)
                    if (m.ds.target_layers[(size_t)ti] == l)
                        RAD_ISSUE_N(c, m.ds.tap_mean[(size_t)ti], T, brows(nxt, T),
                                    bcol(m.ds.taps, (int64_t)ti * m.hidden, m.hidden, T));
            }
            continue;
        } else {
            if (m.quant_fwd) {
                if (y.need_qx && need_a8_f(y.q_gu, T)) issue_qact_f(y.qop_x, m.qa_x, m.x, T);
                issue_qgemm_f(y.q_gu, m.x, m.qa_x, m.gu[y.kind], T);
                if (y.need_qact && need_a8_f(y.q_down, T) && y.act_q) {
                    RAD_ISSUE_N(c, y.act_q, T, brows(m.gu[y.kind], T),
                                brow_slice(m.qa_act[y.kind].q, 0, T, m.qa_act[y.kind].K),
                                brow_slice(m.qa_act[y.kind].s, 0, T, m.qa_act[y.kind].K / 128));
                } else {
                    RAD_ISSUE(c, y.act, brows(m.gu[y.kind], T), brows(m.act[y.kind], T));
                    if (y.need_qact && need_a8_f(y.q_down, T)) issue_qact_f(y.qop_act, m.qa_act[y.kind], m.act[y.kind], T);
                }
                issue_qgemm_f(y.q_down, m.act[y.kind], m.qa_act[y.kind], m.y, T);
                if (y.down_ar) RAD_ISSUE_N(c, y.down_ar, T * m.hidden, brows(m.y, T), RAD_NONE);
            } else {
                RAD_ISSUE(c, y.gate_op, brows(m.x, T), RAD_W(y.gate), bcol(m.gu[y.kind], 0, I, T));
                RAD_ISSUE(c, y.up_op, brows(m.x, T), RAD_W(y.up), bcol(m.gu[y.kind], I, I, T));
                RAD_ISSUE(c, y.act, brows(m.gu[y.kind], T), brows(m.act[y.kind], T));
                RAD_ISSUE(c, y.down_op, brows(m.act[y.kind], T), RAD_W(y.down), brows(y.routed ? m.ys : m.y, T));
            }
            if (y.routed) {
                const RoutedOps& R = y.R;
                const RoutedHandles& w = R.w;
                const int64_t P = T * m.K;
                RAD_ISSUE(c, R.r0, brows(m.x, T), RAD_W(w.r0), bcol(m.lg, 0, m.E, T));
                RAD_ISSUE(c, R.r1, brows(m.x, T), RAD_W(w.r1), bcol(m.lg, m.E, m.E, T));
                RAD_ISSUE(c, R.topk, brows(m.lg, T), RAD_W(w.bias), RAD_W(w.scale), brows(m.ids, T),
                          brows(m.rw, T));
                RAD_ISSUE(c, R.scatter, brows(m.ids, T), brows(m.sorted, P), brows(m.eoff, m.E + 1),
                          brows(m.ecnt, m.E));
                RAD_ISSUE(c, R.fc1, brows(m.x, T), RAD_W(w.in), brows(m.xl, T));
                RAD_ISSUE_N(c, R.gate, T, brows(m.xl, T), RAD_WTAB(w.ge[0], m.E), brows(m.sorted, P),
                            brows(m.eoff, m.E + 1), bcol(m.egu, 0, m.Fe, P));
                RAD_ISSUE_N(c, R.up, T, brows(m.xl, T), RAD_WTAB(w.ue[0], m.E), brows(m.sorted, P),
                            brows(m.eoff, m.E + 1), bcol(m.egu, m.Fe, m.Fe, P));
                RAD_ISSUE_N(c, R.act, P, brows(m.egu, P), brows(m.eact, P));
                RAD_ISSUE_N(c, R.down, T, brows(m.eact, P), RAD_WTAB(w.de[0], m.E), brows(m.sorted, P),
                            brows(m.eoff, m.E + 1), brows(m.edn, P));
                RAD_ISSUE(c, R.gather, brows(m.edn, P), brows(m.rw, T), brows(m.sorted, P), RAD_NONE,
                          RAD_NONE, brows(m.comb2, T));
                RAD_ISSUE(c, R.norm, brows(m.comb2, T), RAD_W(w.norm), brows(m.nl, T));
                RAD_ISSUE(c, R.fc2, brows(m.nl, T), RAD_W(w.out), brows(m.ro, T));
                RAD_ISSUE(c, R.add, brows(m.ro, T), brows(m.ys, T), brows(m.y, T));
            }
        }
        RAD_ISSUE(c, y.post, brows(m.y, T), brows(cur, T), brows(m.post, T), brows(m.comb, T),
                  brows(nxt, T));
        if (m.ds.on) {
            for (int ti = 0; ti < (int)m.ds.n_taps; ++ti)
                if (m.ds.target_layers[(size_t)ti] == l)
                    RAD_ISSUE_N(c, m.ds.tap_mean[(size_t)ti], T, brows(nxt, T),
                                bcol(m.ds.taps, (int64_t)ti * m.hidden, m.hidden, T));
        }
    }
    if (m.logits) {
        const int64_t R = batch->n_out;
        if (R <= 0) return;      /* a chunk that finishes no sequence samples nothing */
        RAD_ISSUE_N(c, m.gather, R, brows(m.h[m.fin], T), praw(batch->out_ids, RAD_I32, R),
                    brows(m.hout, R));
        RAD_ISSUE_N(c, m.head, R, brows(m.hout, R), RAD_W(m.head_fn), RAD_W(m.head_scale),
                    RAD_W(m.head_base), brows(m.hx, R));
        RAD_ISSUE_N(c, m.norm, R, brows(m.hx, R), RAD_W(m.out_norm), brows(m.hn, R));
        if (!batch->score_only) {
            RAD_ISSUE_N(c, m.proj, R, brows(m.hn, R), RAD_W(m.lm_head), brows(m.lgt, R));
            RAD_ISSUE_N(c, m.mmax, R, brows(m.lgt, R), RAD_W(m.mm_r), RAD_W(m.mm_t), brows(m.lgt, R));
        }
        return;
    }
    const RadOperand cu = praw(batch->cu_seqlens, RAD_I32, S + 1);
    RAD_ISSUE(c, m.pool, brows(m.h[m.fin], T), cu, brows(m.mean, S));
    RAD_ISSUE_N(c, m.cast, S, brows(m.mean, S), brows(m.pooled, S));
}

}  /* namespace ernie55 */

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PLUGIN(ernie55, "ernie55moe", "", "0.1.0",
                "ERNIE 5.5 Flash (ernie5_5_moe): architecture identity, validated hybrid "
                "schedule and weights; only the opt-in feed-forward block probe executes")
RAD_ARCH_PROBE(ernie55)
#endif
