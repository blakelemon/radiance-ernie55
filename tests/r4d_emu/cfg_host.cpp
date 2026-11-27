/* cfg_host.cpp -- the tuned-point reader the bf16 GEMM rows call (r4d_gemm_args.h cfg_for). libr4d keeps it in r4d_rows.cpp, which the emulator tests do not link:
 * the axis values the core appends to RadArgs::p (a test passes them to mimic a tune cache), else the axis defaults of tGemmSkinny (wv 1, sk 0, mb 0, npw 1, nt 1). */
#include "rad_plugin.h"
#include "r4d_args.h"

extern "C" void r4d_cfg_gemm(const RadArgs* a, R4dGemmCfg* out) {
    out->wv  = (int)r4d_geti_or(a, "wv", 1);
    out->sk  = (int)r4d_geti_or(a, "sk", 0);
    out->mb  = (int)r4d_geti_or(a, "mb", 0);
    out->npw = (int)r4d_geti_or(a, "npw", 1);
    out->nt  = (int)r4d_geti_or(a, "nt", 1);
}
