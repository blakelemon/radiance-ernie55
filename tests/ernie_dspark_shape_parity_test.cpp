/* ernie_dspark_shape_parity_test.cpp -- libr4d's operand descriptions for the two DSpark ops hold to libref's.
 *
 * `opd_shape` sizes and types every operand a harness draws for an op, so a description that
 * disagrees with the kernel (libr4d once described ernie_hca_dspark_attn's seqused as rank-2 {S, M}
 * where the kernel and libref read a rank-1 [S]) is a bug only a device run would otherwise show.
 * libref is the reference the CPU suites run, libr4d's shapes (host code, compiled into this test)
 * must name the same dtype, rank, extents, fill kind, flags (an ascending cu_seqlens is RAD_OPD_F_IDX_CU) and
 * index bound for each operand at a geometry both accept.
 */
#include "rad_test.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"

#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>

namespace {

RadParam pint(const char* k, int64_t v) {
    RadParam p{};
    p.key = k;
    p.kind = RAD_P_INT;
    p.ival = v;
    return p;
}

RadShapeFn ref_shape(const char* op) {
    static void* lib = nullptr;
    if (!lib) {
        const char* home = std::getenv("RADIANCE_HOME");
        const std::string path = std::string(home ? home : ".") + "/kernels/libref.so";
        lib = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!lib) return nullptr;
    }
    auto count = reinterpret_cast<int (*)()>(dlsym(lib, "rad_kernel_count"));
    auto at = reinterpret_cast<const RadKernelInfo* (*)(int)>(dlsym(lib, "rad_kernel_at"));
    if (!count || !at) return nullptr;
    for (int i = 0; i < count(); ++i) {
        const RadKernelInfo* k = at(i);
        if (k && k->op && std::strcmp(k->op, op) == 0 && k->opd_shape) return k->opd_shape;
    }
    return nullptr;
}

/* Every operand both describe agrees on dtype, rank, extents and a constant fill. */
void parity(const char* op, RadShapeFn r4d, const std::vector<RadParam>& p, int n_operands) {
    RadShapeFn ref = ref_shape(op);
    REQUIRE(ref != nullptr);
    for (int i = 0; i < n_operands; ++i) {
        RadOpdDesc a{}, b{};
        REQUIRE_EQ(ref(p.data(), (int)p.size(), i, &a), RAD_OK);
        REQUIRE_EQ(r4d(p.data(), (int)p.size(), i, &b), RAD_OK);
        CHECK_EQ(b.dtype, a.dtype);
        CHECK_EQ(b.rank, a.rank);
        for (uint32_t d = 0; d < a.rank && d < b.rank; ++d) CHECK_EQ(b.shape[d], a.shape[d]);
        CHECK_EQ(b.idx_const, a.idx_const);
        CHECK_EQ(b.fill, a.fill);
        CHECK_EQ(b.flags, a.flags);
        if (a.fill == (uint32_t)RAD_FILL_INDEX && a.idx_const < 0) CHECK_EQ(b.idx_max, a.idx_max);
    }
}

}  // namespace

TEST(ernie_mhc_stream_mean_shapes_match_libref) {
    const std::vector<RadParam> p = {pint("M", 3), pint("hidden", 16), pint("mult", 4)};
    parity("ernie_mhc_stream_mean", r4d_shape_ernie_mhc_stream_mean, p, 2);
}

TEST(ernie_hca_dspark_attn_shapes_match_libref) {
    /* q, kv, block_table, seqused, cu_seqlens, sink, out */
    for (int64_t M : {1, 3, 7, 56}) {
        const std::vector<RadParam> p = {pint("M", M), pint("heads", 2), pint("head_dim", 8), pint("seqs", 2),
                                         pint("block_size", 4), pint("blocks", 3)};
        parity("ernie_hca_dspark_attn", r4d_shape_ernie_hca_dspark_attn, p, 7);
    }
}

RAD_TEST_MAIN()
