/* r4d_ernie_pins.h -- the row-band rules of the ERNIE device ops and their env pins, as plain functions.
 *
 * Several ERNIE kernels change arithmetic with the step's row count (a decode form up to a bound, a wide form above it), so a row's value can
 * depend on how many rows share its step. Each rule has an env pin (diagnostic, read once per process in the launcher) whose unset value is the
 * shipped rule. The rules and the pin parsing live here, free of HIP, so tests/ernie_pins_test.cpp can hold both to their documented values:
 * unset = today's kernel choice, a pin = the pinned choice. The launchers call these and nothing else decides a band. */
#ifndef R4D_ERNIE_PINS_H
#define R4D_ERNIE_PINS_H
#include <algorithm>
#include <cstdlib>

namespace r4d_ernie {
namespace pin {

/* The shipped bounds (the value of each pin when its variable is unset).
 * Row identity (the row-identity checks): a band is 0 where the small-row and the wide kernel sum in different orders (mHC decode rows, and
 * the fp8a8 matvec for ERNIE, which declares mvm_max=0 per op), so every row count takes the form prefill takes. Union attention (4) and sparse MLA (8)
 * keep their bands because their small-row and wide kernels compute the same bits (shared part/merge device functions). The variables still set a
 * band for diagnosis (e.g. MHC_DECODE_ROWS=4 restores the old mHC rule). The pair GEMM and wo_a no longer have bands (one chunked WMMA sum for every
 * row count). */
constexpr int kHcaDecodeMax = 4;        /* R4D_HCA_DECODE_MAX: union attention, split-KV decode kernel up to this many rows */
constexpr int kHcaWoADecodeMax = 16;    /* R4D_HCA_WOA_DECODE_MAX: RETIRED (row-identity): wo_a picks its form by shape only and reads no variable */
constexpr int kMhcDecodeRows = 0;       /* R4D_MHC_DECODE_ROWS: mHC pre-collapse decode form below this many rows */
constexpr int kMlaIdxlSmallMax = 4;     /* R4D_MLA_IDXL_SMALL_MAX: 2-tile indexer logits up to this many rows */
constexpr int kMlaProjRows = 0;         /* R4D_MLA_PROJ_ROWS: 0 the rule, 1 one row a block */
constexpr int kMlaCoopMax = 8;          /* R4D_MLA_COOP_MAX: cooperative sparse attention up to this many rows */
constexpr int kF8a8MvmMax = 8;          /* R4D_F8A8_MVM_MAX: fp8a8 matvec up to this many rows. GENERIC default (other models use this kernel); ERNIE declares mvm_max=0 on its fp8a8f32 ops (the WMMA forms, equal per row, serve every count) */
constexpr int kF8a8MvmCeiling = 8;      /* the matvec is declared for M <= 8: larger values are clamped */
constexpr int kMoeMxSmallPin = 0;       /* R4D_MOE_MX_SMALL_PIN: 0 the rule, 2 expert-major forms for every step */
constexpr long kPairVecMax = 8;         /* R4D_PAIR_VEC_MAX: RETIRED (row-identity): gemm_nt_f32_pair takes one chunked WMMA sum for every row count and reads no variable; the parser stays for old scripts */
constexpr long kPairVecCeiling = 65535; /* the launcher's row limit; this value pins every row count to the vector kernels */

/* An env value: unset or empty gives `def`, otherwise the integer (atoi, as every pin has always read it). */
inline int parse(const char* e, int def) { return e && *e ? std::atoi(e) : def; }
inline int from_env(const char* name, int def) { return parse(std::getenv(name), def); }

/* ---- the band rules ---- */
inline bool hca_decode(int rows, int max) { return rows <= max; }
inline bool woa_decode(int rows, int max) { return rows <= max; }
inline bool mhc_decode(int rows, int bound) { return rows < bound; }
inline bool idxl_small(int rows, int max) { return rows <= max; }
inline int mla_proj_rows(int rows, int pin) { return pin == 1 ? 1 : rows >= 16 ? 16 : rows >= 8 ? 8 : rows >= 4 ? 4 : rows >= 2 ? 2 : 1; }
inline bool mla_coop(int rows, int window, int max) { return rows <= max && window >= 64; }
inline int f8a8_mvm_max(const char* e) { return e && *e ? std::min(kF8a8MvmCeiling, std::atoi(e)) : kF8a8MvmMax; }
inline bool f8a8_mvm(int rows, int max) { return rows <= max; }
/* The op's declared "mvm_max" (>= 0) wins over the generic bound; absent (< 0) the generic bound stands. Clamped to the ceiling. */
inline int f8a8_mvm_bound(long long declared, int generic) { return declared >= 0 ? (int)std::min<long long>(kF8a8MvmCeiling, declared) : generic; }
inline long pair_vec_max(const char* e) { return e && *e ? std::min<long>(std::max<long>(0, std::atol(e)), kPairVecCeiling) : kPairVecMax; }
inline bool pair_vec(long rows, long max) { return rows <= max; }

}  // namespace pin
}  // namespace r4d_ernie
#endif
