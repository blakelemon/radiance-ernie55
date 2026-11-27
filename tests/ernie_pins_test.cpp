/* ernie_pins_test.cpp -- the ERNIE row-band rules and their env pins hold to their documented values.
 *
 * A pin is diagnostic: unset (or empty) it must leave today's kernel choice, set it must give the pinned choice. The rules and the pin parsing are plain
 * functions (libr4d/r4d_ernie_pins.h, arch/ernie55/ernie55_pins.h) that the launchers call, so a band edge cannot move without this test seeing it.
 * Each case lists the shipped bound first (the value an unset variable has), the rows either side of it, then the pinned forms. */
#include "rad_test.h"
#include "r4d_ernie_pins.h"
#include "ernie55_pins.h"
#include <cstdlib>
#include <string>

namespace {
namespace P = r4d_ernie::pin;
namespace A = ernie55::pin;

struct Env {   /* sets a variable for one scope */
    std::string name;
    Env(const char* n, const char* v) : name(n) { if (v) setenv(n, v, 1); else unsetenv(n); }
    ~Env() { unsetenv(name.c_str()); }
};
}  // namespace

TEST(parse_unset_and_empty_give_the_default_a_value_gives_itself) {
    unsetenv("R4D_PIN_TEST");
    CHECK_EQ(P::from_env("R4D_PIN_TEST", 7), 7);
    { Env e("R4D_PIN_TEST", ""); CHECK_EQ(P::from_env("R4D_PIN_TEST", 7), 7); }
    { Env e("R4D_PIN_TEST", "0"); CHECK_EQ(P::from_env("R4D_PIN_TEST", 7), 0); }
    { Env e("R4D_PIN_TEST", "12"); CHECK_EQ(P::from_env("R4D_PIN_TEST", 7), 12); }
}

TEST(hca_union_attention_takes_the_split_kv_decode_kernel_up_to_4_rows_unless_set) {
    /* row identity: the decode and the wide kernel compute the same bits for a row (hca512_part / hca512_merge), so the fast decode kernel stays the rule */
    CHECK_EQ(P::kHcaDecodeMax, 4);
    for (int rows : {1, 2, 3, 4}) CHECK(P::hca_decode(rows, P::kHcaDecodeMax));
    for (int rows : {5, 8, 16, 43, 129}) CHECK(!P::hca_decode(rows, P::kHcaDecodeMax));
    for (int rows : {1, 5, 129}) CHECK(!P::hca_decode(rows, 0));       /* =0: every step on the wide kernel */
    for (int rows : {1, 5, 16}) CHECK(P::hca_decode(rows, 16));         /* =16: every 1..16-row step on the decode kernel */
}

TEST(hca_wo_a_decode_kernel_is_up_to_16_rows_unless_pinned) {
    CHECK_EQ(P::kHcaWoADecodeMax, 16);
    CHECK(P::woa_decode(16, P::kHcaWoADecodeMax));
    CHECK(!P::woa_decode(17, P::kHcaWoADecodeMax));
    CHECK(P::woa_decode(129, 65535));
    CHECK(!P::woa_decode(1, 0));
}

TEST(mhc_pre_collapse_takes_the_many_row_form_for_every_row_count_unless_set) {
    CHECK_EQ(P::kMhcDecodeRows, 0);
    for (int rows : {1, 2, 3, 4, 129}) CHECK(!P::mhc_decode(rows, P::kMhcDecodeRows));
    for (int rows : {1, 2, 3}) CHECK(P::mhc_decode(rows, 4));           /* =4: the old rule */
    for (int rows : {4, 5, 64}) CHECK(!P::mhc_decode(rows, 4));
    for (int rows : {1, 4, 129}) CHECK(!P::mhc_decode(rows, 0));        /* =0: every step on the many-row form */
}

TEST(mla_indexer_logits_two_tile_form_is_up_to_4_rows_unless_pinned) {
    CHECK_EQ(P::kMlaIdxlSmallMax, 4);
    CHECK(P::idxl_small(4, P::kMlaIdxlSmallMax));
    CHECK(!P::idxl_small(5, P::kMlaIdxlSmallMax));
    CHECK(!P::idxl_small(1, 0));
}

TEST(mla_projection_rows_a_block_follow_the_step_unless_pinned_to_one) {
    CHECK_EQ(P::kMlaProjRows, 0);
    const int want[][2] = {{1, 1}, {2, 2}, {3, 2}, {4, 4}, {7, 4}, {8, 8}, {15, 8}, {16, 16}, {100, 16}};
    for (auto& w : want) CHECK_EQ(P::mla_proj_rows(w[0], P::kMlaProjRows), w[1]);
    for (auto& w : want) CHECK_EQ(P::mla_proj_rows(w[0], 1), 1);        /* =1: one row a block for every step */
    for (auto& w : want) CHECK_EQ(P::mla_proj_rows(w[0], 5), w[1]);     /* any other value is the rule */
}

TEST(mla_sparse_attention_takes_the_cooperative_kernel_up_to_8_rows_unless_set) {
    /* row identity: the cooperative and the wave kernel compute the same bits (mla_part / the partition fold), so the cooperative decode kernel stays the rule */
    CHECK_EQ(P::kMlaCoopMax, 8);
    CHECK(P::mla_coop(8, 64, P::kMlaCoopMax));
    CHECK(!P::mla_coop(9, 64, P::kMlaCoopMax));
    CHECK(!P::mla_coop(8, 63, P::kMlaCoopMax));
    CHECK(!P::mla_coop(1, 4096, 0));                                    /* =0: every step on the wave kernel */
}

TEST(fp8a8_generic_default_keeps_the_matvec_to_8_rows_and_ernie_pins_0) {
    CHECK_EQ(P::f8a8_mvm_max(nullptr), 8);                                /* generic default: other models use this kernel */
    CHECK_EQ(P::f8a8_mvm_max(""), 8);
    CHECK(P::f8a8_mvm(1, P::f8a8_mvm_max(nullptr)));
    CHECK_EQ(P::f8a8_mvm_max("0"), 0);
    CHECK_EQ(P::f8a8_mvm_max("4"), 4);
    CHECK_EQ(P::f8a8_mvm_max("64"), 8);                                  /* the matvec is declared for M <= 8: larger values are clamped */
    CHECK(P::f8a8_mvm(8, 8));
    CHECK(!P::f8a8_mvm(9, 8));
    CHECK(!P::f8a8_mvm(1, 0));
}

TEST(moe_mxfp4_small_pin_is_0_the_rule_unless_set) {
    CHECK_EQ(P::kMoeMxSmallPin, 0);
    Env e("R4D_MOE_MX_SMALL_PIN", nullptr);
    CHECK_EQ(P::from_env("R4D_MOE_MX_SMALL_PIN", P::kMoeMxSmallPin), 0);
    Env f("R4D_MOE_MX_SMALL_PIN", "2");
    CHECK_EQ(P::from_env("R4D_MOE_MX_SMALL_PIN", P::kMoeMxSmallPin), 2);
}

TEST(pair_vec_parser_retired_but_kept_for_old_scripts) {   /* gemm_nt_f32_pair reads no variable (row identity) */
    CHECK_EQ(P::pair_vec_max(nullptr), 8);
    CHECK_EQ(P::pair_vec_max(""), 8);
    CHECK_EQ(P::pair_vec_max("65535"), 65535);
    CHECK_EQ(P::pair_vec_max("99999999"), 65535);                       /* clamped to the launcher's row limit */
    CHECK_EQ(P::pair_vec_max("-3"), 0);
    CHECK_EQ(P::pair_vec_max("0"), 0);
    for (long rows : {1L, 2L, 8L}) CHECK(P::pair_vec(rows, P::pair_vec_max(nullptr)));
    for (long rows : {9L, 17L, 43L, 129L, 2048L}) CHECK(!P::pair_vec(rows, P::pair_vec_max(nullptr)));
    for (long rows : {1L, 9L, 129L, 2048L, 65535L}) CHECK(P::pair_vec(rows, P::pair_vec_max("65535")));
}

TEST(plugin_a16_activation_path_is_off_unless_set_up_to_8) {
    CHECK_EQ(A::a16_max(nullptr), 0);                                    /* row identity: fp8a8 for every row */
    CHECK_EQ(A::a16_max(""), 0);
    CHECK(!A::use_a16(true, 1, A::a16_max(nullptr), true));
    CHECK_EQ(A::a16_max("0"), 0);
    CHECK_EQ(A::a16_max("3"), 3);
    CHECK_EQ(A::a16_max("64"), 8);                                       /* clamped: the fp8a16f32 ops are declared for M <= 8 */
    CHECK_EQ(A::a16_max("-5"), 0);
    for (int64_t rows : {1, 4, 8}) CHECK(A::use_a16(true, rows, A::a16_max("8"), true));
    for (int64_t rows : {9, 64}) CHECK(!A::use_a16(true, rows, A::a16_max("8"), true));
    CHECK(!A::use_a16(true, 1, A::a16_max("0"), true));                  /* =0: every row on the fp8a8 path */
    CHECK(!A::use_a16(false, 1, 8, true));                               /* a non-fp8 weight has no a16 form */
    CHECK(!A::use_a16(true, 1, 8, false));                               /* nor does an op declared without one */
}

TEST(plugin_router_one_row_step_is_fused_unless_split_is_pinned) {
    CHECK(!A::router_split(nullptr));
    CHECK(!A::router_split(""));
    CHECK(!A::router_split("0"));
    CHECK(A::router_split("1"));
    CHECK(A::router_split("yes"));
    CHECK(A::router_fused(1, true, A::router_split(nullptr)));
    CHECK(!A::router_fused(2, true, false));
    CHECK(!A::router_fused(1, false, false));
    CHECK(!A::router_fused(1, true, A::router_split("1")));
}

TEST(plugin_vha_takes_the_reduce_path_for_every_step_unless_set) {
    CHECK_EQ(A::vha_dec_cap(nullptr), 0);
    CHECK_EQ(A::vha_dec_cap(""), 0);
    CHECK_EQ(A::vha_dec_cap("0"), 0);
    CHECK_EQ(A::vha_dec_cap("32"), 32);
    CHECK_EQ(A::vha_dec_cap("-1"), 0);
}

RAD_TEST_MAIN()

TEST(the_plugin_declares_its_fp8a8_matvec_bound_per_op_and_sets_no_environment) {
    /* each fp8a8f32 op carries mvm_max: 0 in the row-identical bands (the WMMA forms for every row count), 8 in the previous bands; no process state, no setenv */
    unsetenv("R4D_F8A8_MVM_MAX");
    bool ov = true;
    CHECK_EQ(ernie55::pin::f8a8_mvm_declared(nullptr, true, &ov), 0);
    CHECK(!ov);
    CHECK_EQ(ernie55::pin::f8a8_mvm_declared(nullptr, false, &ov), 8);   /* a non-drafter file at depth 0: the previous bands, legitimately 8 */
    CHECK(!ov);
    CHECK(getenv("R4D_F8A8_MVM_MAX") == nullptr);                        /* the plugin exports nothing */
    CHECK_EQ(ernie55::pin::f8a8_mvm_declared("", true, &ov), 0);          /* empty = unset */
    CHECK(!ov);
    CHECK_EQ(ernie55::pin::f8a8_mvm_declared("8", true, &ov), 8);        /* a diagnostic override is read and reported */
    CHECK(ov);
    CHECK_EQ(ernie55::pin::f8a8_mvm_declared("64", true, &ov), 8);       /* clamped to the matvec's ceiling */
    CHECK_EQ(ernie55::pin::f8a8_mvm_declared("-3", false, &ov), 0);
    CHECK(!ernie55::pin::f8a8_mvm_refused(0, true));                     /* declared 0 in the row-identical bands: fine */
    CHECK(!ernie55::pin::f8a8_mvm_refused(8, false));                    /* the previous bands (non-drafter file, depth 0): 8 is the rule, not refused */
    CHECK(ernie55::pin::f8a8_mvm_refused(8, true));                      /* a nonzero bound in the row-identical bands: refused */
    CHECK(ernie55::pin::f8a8_mvm_refused(1, true));
    /* the launcher: a declared bound wins over the generic one, an absent one (negative) leaves the generic bound; the ceiling clamps */
    CHECK_EQ(P::f8a8_mvm_bound(0, 8), 0);
    CHECK_EQ(P::f8a8_mvm_bound(-1, 8), 8);
    CHECK_EQ(P::f8a8_mvm_bound(-1, 3), 3);
    CHECK_EQ(P::f8a8_mvm_bound(64, 0), 8);
    CHECK_EQ(P::f8a8_mvm_bound(4, 8), 4);
}

TEST(row_bands_follow_speculation_unless_overridden) {
    /* the declared decision: the drafter in the file or a requested depth turns row identity on; ERNIE_ROW_IDENTITY overrides both ways */
    CHECK(!A::row_identity_on(nullptr, false, 0));
    CHECK(A::row_identity_on(nullptr, true, 0));
    CHECK(A::row_identity_on(nullptr, false, 7));
    CHECK(A::row_identity_on("", false, 3));
    CHECK(A::row_identity_on("1", false, 0));
    CHECK(!A::row_identity_on("0", true, 7));
    /* the bands of each mode (an explicit value wins in both) */
    CHECK_EQ(A::a16_max(nullptr, false), 8);
    CHECK_EQ(A::a16_max(nullptr, true), 0);
    CHECK_EQ(A::a16_max("3", false), 3);
    CHECK_EQ(A::a16_max("0", false), 0);
    CHECK_EQ(A::vha_dec_cap(nullptr, false), 8);
    CHECK_EQ(A::vha_dec_cap(nullptr, true), 0);
    CHECK_EQ(A::vha_dec_cap("32", true), 32);
    for (int64_t rows : {1, 4, 8}) CHECK(A::use_a16(true, rows, A::a16_max(nullptr, false), true));   /* previous mode: the bf16 GEMV up to 8 rows, as before */
    for (int64_t rows : {9, 64}) CHECK(!A::use_a16(true, rows, A::a16_max(nullptr, false), true));
    /* the log's copies of the libr4d bands that no mode moves */
    CHECK_EQ(A::kHcaDecodeMaxLog, P::kHcaDecodeMax);
    CHECK_EQ(A::kMlaCoopMaxLog, P::kMlaCoopMax);
    CHECK_EQ(A::kMhcDecodeRowsRowId, P::kMhcDecodeRows);   /* libr4d's own default is the row-identical value */
    CHECK_EQ(A::kF8a8MvmPrev, P::kF8a8MvmMax);             /* and the generic matvec bound is the previous one */
}

TEST(process_pins_name_both_variables_in_both_modes_and_an_explicit_value_wins) {
    const char* names[] = {"R4D_F8A8_MVM_MAX", "R4D_MHC_DECODE_ROWS"};   /* the plugin exports only the second; the first is a declared op parameter and stays untouched */
    for (int mode = 0; mode < 2; ++mode) {
        for (const char* n : names) unsetenv(n);
        A::export_process_pins(mode == 1);
        CHECK_EQ(A::env_int("R4D_F8A8_MVM_MAX", -1), -1);
        CHECK_EQ(A::env_int("R4D_MHC_DECODE_ROWS", -1), mode ? A::kMhcDecodeRowsRowId : A::kMhcDecodeRowsPrev);
    }
    setenv("R4D_MHC_DECODE_ROWS", "4", 1);
    A::export_process_pins(true);
    CHECK_EQ(A::env_int("R4D_MHC_DECODE_ROWS", -1), 4);
    for (const char* n : names) unsetenv(n);
}

TEST(an_empty_band_variable_is_overwritten_by_the_export_like_an_unset_one) {
    for (const char* n : {"R4D_F8A8_MVM_MAX", "R4D_MHC_DECODE_ROWS"}) setenv(n, "", 1);
    A::export_process_pins(false);
    CHECK_EQ(A::env_int("R4D_MHC_DECODE_ROWS", -1), A::kMhcDecodeRowsPrev);   /* what libr4d will use (4), and what the log prints */
    CHECK_EQ(A::env_int("R4D_F8A8_MVM_MAX", -1), -1);                         /* untouched (empty reads as unset) */
    for (const char* n : {"R4D_F8A8_MVM_MAX", "R4D_MHC_DECODE_ROWS"}) unsetenv(n);
}
