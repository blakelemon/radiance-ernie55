/* ernie55_pins.h -- the plugin's row-band rules and env pins (ERNIE_A16_MAX, ERNIE_ROUTER_SPLIT, ERNIE_VHA_DEC_MAX) as plain functions of the
 * variable's text, so a test can hold each to its documented value: unset is the shipped rule, a pin is the pinned choice. ernie55.cpp reads each
 * variable once and calls these. */
#ifndef ERNIE55_PINS_H
#define ERNIE55_PINS_H
#include <algorithm>
#include <cstdint>
#include <cstdlib>

namespace ernie55 {
namespace pin {
constexpr int64_t kA16Max = 8;      /* ceiling: the fp8a16f32 ops are declared for M <= 8 and libr4d refuses more */
/* Row identity (the row-identity checks): unset = 0, every row on the forms prefill uses (fp8a8, the VHA reduce path), so a row's
 * logits do not depend on how many rows share its step. 8 restores the old small-step forms (bf16-activation GEMV, all-gather mix). */
constexpr int64_t kA16Default = 0;
constexpr int64_t kVhaDecMax = 0;
constexpr int kMinDraftDepth = 2;   /* the DSpark block is the anchor plus depth - 1 mask rows; the scheduler refuses a block below 2 */

/* ERNIE_A16_MAX: unset/empty 0 (every row down the fp8a8 path); otherwise clamped to 0..8. */
inline int64_t a16_max(const char* e) {
    if (!e || !*e) return kA16Default;
    return std::min<int64_t>(kA16Max, std::max<int64_t>(0, (int64_t)std::atoll(e)));
}
/* A block-fp8 projection reads its bf16 activation (the fp8a16f32 GEMV) when the step has at most `max` rows and the op was declared. */
inline bool use_a16(bool fp8, int64_t rows, int64_t max, bool has_a16_op) { return fp8 && rows <= max && has_a16_op; }
/* ERNIE_ROUTER_SPLIT: any value but unset, empty or starting with '0' routes a one-row step through the split ops. */
inline bool router_split(const char* e) { return e && *e && *e != '0'; }
inline bool router_fused(int64_t T, bool has_scatter, bool split) { return T == 1 && has_scatter && !split; }
/* ERNIE_VHA_DEC_MAX: unset/empty 0 (the reduce path for every step), otherwise >= 0; the step mixes by all-gather + ernie_vha_mix up to this many rows. */
inline int64_t vha_dec_cap(const char* e) { return e && *e ? std::max<int64_t>(0, (int64_t)std::atoll(e)) : kVhaDecMax; }

/* ---- Row bands tied to speculation (integration round 3) ----
 * Row identity (a row's output does not depend on how many rows share its step) is what makes the multi-token drafter lossless, but its band choices cost
 * about 10% single-stream decode and move decode-row activations to fp8a8 (ERNIE_A16_MAX=0), which the owner has not approved as a general default. So the
 * bands are one declared decision at model build:
 *   speculation on  (the model file carries the DSpark drafter, or a speculative depth > 0 was requested): the row-identical bands
 *   speculation off (neither): the bands before row identity (ERNIE_A16_MAX 8, ERNIE_VHA_DEC_MAX 8, R4D_MHC_DECODE_ROWS 4, declared fp8a8 mvm_max 8)
 * ERNIE_ROW_IDENTITY=0|1 overrides the decision for A/B; an explicit ERNIE_A16_MAX / ERNIE_VHA_DEC_MAX / R4D_MHC_DECODE_ROWS / R4D_F8A8_MVM_MAX wins over both.
 * The four REWRITTEN families (the pair GEMM, wo_a, the D512 union attention and the sparse MLA attention) are row identical in both modes: they have no band. */
constexpr int64_t kA16Prev = 8, kVhaDecPrev = 8;   /* the bands before row identity */
constexpr int kMhcDecodeRowsPrev = 4, kMhcDecodeRowsRowId = 0, kF8a8MvmPrev = 8, kF8a8MvmRowId = 0;
constexpr int kHcaDecodeMaxLog = 4, kMlaCoopMaxLog = 8;   /* libr4d/r4d_ernie_pins.h kHcaDecodeMax / kMlaCoopMax (tests/ernie_pins_test.cpp holds them equal); the same in both modes */

/* The decision: ERNIE_ROW_IDENTITY ("0" off, any other non-empty value on) wins; otherwise on exactly when the model has the drafter or a depth was requested. */
inline bool row_identity_on(const char* env, bool has_drafter, int max_spec) {
    if (env && *env) return *env != '0';
    return has_drafter || max_spec > 0;
}
/* ERNIE_A16_MAX / ERNIE_VHA_DEC_MAX with the mode's default (the 1-argument forms above are the row-identical mode) */
inline int64_t a16_max(const char* e, bool row_identity) {
    if (!e || !*e) return row_identity ? kA16Default : kA16Prev;
    return a16_max(e);
}
inline int64_t vha_dec_cap(const char* e, bool row_identity) { return e && *e ? vha_dec_cap(e) : (row_identity ? kVhaDecMax : kVhaDecPrev); }

/* The fp8a8 matvec bound (R4D_F8A8_MVM_MAX of libr4d) is a DECLARED per-op parameter, "mvm_max", on every fp8a8f32 gemm_nt_q of this model: 0 in the row-identical
 * bands (every step takes the WMMA forms, equal per row), 8 in the previous bands. It travels with the op: nothing is exported, a launch before declare() or another model
 * in the process is not affected, an operator's stale variable cannot silently change this model. The variable remains a DIAGNOSTIC override read at model build (it
 * replaces the declared value, the build logs it) and a row-identical build refuses a nonzero effective bound.
 * The mHC decode form (R4D_MHC_DECODE_ROWS) has no op parameter: it stays a process export under the plugin's mutex (resolve_row_bands), 0 / 4 by the same decision. */
inline int f8a8_mvm_declared(const char* e, bool row_identity, bool* overridden = nullptr) {
    const bool ov = e && *e;
    if (overridden) *overridden = ov;
    if (!ov) return row_identity ? kF8a8MvmRowId : kF8a8MvmPrev;
    return std::min(8, std::max(0, std::atoi(e)));
}
/* A matvec bound above 0 makes a row's fp8a8 output depend on whether the step has more or fewer than `bound` rows: not allowed in the row-identical bands
 * (speculation: the file carries the drafter or a depth was requested, unless ERNIE_ROW_IDENTITY=0 says otherwise). A non-drafter file at depth 0 runs the previous bands
 * with mvm_max 8 legitimately. */
inline bool f8a8_mvm_refused(int effective, bool row_identity) { return row_identity && effective > 0; }
/* The plugin's libr4d process variable that has no op parameter: the mHC decode rows, exported once at model build (under the plugin's mutex). An explicit variable wins. */
struct ProcessPin { const char* name; const char* value; };
inline void export_process_pins(bool row_identity) {
    const ProcessPin pins[] = {{"R4D_MHC_DECODE_ROWS", row_identity ? "0" : "4"}};
    /* an EMPTY value counts as unset (libr4d's r4d_pin_env and the band parsers do): overwrite it, or the log would read a value libr4d does not use */
    for (const ProcessPin& p : pins) { const char* c = std::getenv(p.name); setenv(p.name, p.value, (c && !*c) ? 1 : 0); }
}
inline void export_process_pins() { export_process_pins(true); }
inline int env_int(const char* name, int def) { const char* e = std::getenv(name); return e && *e ? std::atoi(e) : def; }
}  // namespace pin
}  // namespace ernie55
#endif
