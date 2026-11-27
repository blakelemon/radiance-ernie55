/* checkpoint_test.cpp -- a checkpoint as a source of logical weights (core/format/checkpoint.h).
 *
 * rad-convert and the loader (which serves a trivial checkpoint directly) both go through
 * ckpt_resolve: which tensors make a declared weight, their encoding, and its rows. A wrong answer
 * yields a model that loads but computes wrongly (an expert that is the whole stack, a scale
 * applied to the next piece's rows), so each case builds one real-world layout small and reads it
 * back.
 */
#include "rad_test.h"
#include "gguf_writer.h"
#include "format/checkpoint.h"
#include "format/encoding.h"
#include "rad_plugin.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_set>
#include <vector>

using namespace rad;

namespace {

struct T {
    std::string name, dtype;          /* safetensors spelling: BF16, F32, F8_E4M3 */
    std::vector<int64_t> shape;
    std::vector<uint8_t> bytes;
};

/* A new empty directory under the test's temporary directory, so no checkpoint holds a file
 * another case wrote. */
std::string fresh_dir() {
    std::string t = radtest::tmp_path("ckpt.XXXXXX");
    if (!mkdtemp(t.data())) throw std::runtime_error("cannot make " + t);
    return t;
}

/* One safetensors file holding `ts`, in order. */
void write_shard(const std::string& path, const std::vector<T>& ts) {
    std::string hdr = "{";
    uint64_t off = 0;
    for (size_t i = 0; i < ts.size(); ++i) {
        std::string shape;
        for (size_t k = 0; k < ts[i].shape.size(); ++k)
            shape += (k ? "," : "") + std::to_string(ts[i].shape[k]);
        hdr += (i ? "," : "") + std::string("\"") + ts[i].name + "\":{\"dtype\":\"" + ts[i].dtype +
               "\",\"shape\":[" + shape + "],\"data_offsets\":[" + std::to_string(off) + "," +
               std::to_string(off + ts[i].bytes.size()) + "]}";
        off += ts[i].bytes.size();
    }
    hdr += "}";
    while (hdr.size() % 8) hdr += ' ';
    FILE* f = std::fopen(path.c_str(), "wb");
    const uint64_t n = hdr.size();
    std::fwrite(&n, 8, 1, f);
    std::fwrite(hdr.data(), 1, hdr.size(), f);
    for (const T& t : ts)
        if (!t.bytes.empty()) std::fwrite(t.bytes.data(), 1, t.bytes.size(), f);
    std::fclose(f);
}

/* One model.safetensors and a config.json in a fresh directory. */
void write_file(const std::string& path, const std::string& body) {
    FILE* f = std::fopen(path.c_str(), "wb");
    std::fwrite(body.data(), 1, body.size(), f);
    std::fclose(f);
}

std::string write_checkpoint(const std::vector<T>& ts, const std::string& config) {
    const std::string dir = fresh_dir();
    write_shard(dir + "/model.safetensors", ts);
    write_file(dir + "/config.json", config);
    return dir;
}

/* Element i is tag + (i % 64), exact in bf16, so a value identifies its tensor and position. */
T bf16(const std::string& name, std::vector<int64_t> shape, float tag) {
    T t{ name, "BF16", shape, {} };
    int64_t n = 1;
    for (int64_t s : shape) n *= s;
    t.bytes.resize((size_t)n * 2);
    for (int64_t i = 0; i < n; ++i) rad_store_f32(t.bytes.data(), RAD_BF16, i, tag + (float)(i % 64));
    return t;
}

T cache_scale_f32(const std::string& name, float value) {
    T t{ name, "F32", {1}, std::vector<uint8_t>(4) };
    std::memcpy(t.bytes.data(), &value, sizeof value);
    return t;
}

T fp8(const std::string& name, std::vector<int64_t> shape) {
    T t{ name, "F8_E4M3", shape, {} };
    int64_t n = 1;
    for (int64_t s : shape) n *= s;
    t.bytes.resize((size_t)n);
    for (int64_t i = 0; i < n; ++i) rad_store_f32(t.bytes.data(), RAD_F8E4M3, i, (float)(i % 7) - 3.0f);
    return t;
}

RadNameMap map(const char* declared, int mode, std::vector<const char*> src,
               std::vector<int> index = {}, int concat_dim = 0) {
    RadNameMap m{};
    m.declared = declared;
    m.mode = mode;
    m.n_src = (int)src.size();
    m.concat_dim = concat_dim;
    for (int i = 0; i < 8; ++i) m.src_index[i] = -1;
    for (size_t i = 0; i < src.size(); ++i) m.src[i] = src[i];
    for (size_t i = 0; i < index.size(); ++i) m.src_index[i] = index[i];
    return m;
}

const char* kConfig =
    "{\"model_type\":\"qwen3_5\",\"hidden_size\":256,\"num_hidden_layers\":2,"
    "\"quantization_config\":{\"quant_method\":\"fp8\",\"fmt\":\"e4m3\","
    "\"weight_block_size\":[128,128]}}";


}  /* namespace */
TEST(preview_mxfp4_bytes_shapes_and_decode) {
    const std::string base = "model.layers.1.block_sparse_moe.experts.0.";
    std::vector<T> ts;
    for (const char* proj : {"w1", "w2", "w3"}) {
        const std::string n=base+proj+".weight";
        T c{n,"I8",{2,32},std::vector<uint8_t>(64,0x21)};
        c.bytes[0]=0x98; c.bytes[32]=0xf7;
        T s{n+"_scale_inv","F8_E8M0",{2,2},{127,128,126,129}};
        ts.push_back(c); ts.push_back(s);
    }
    ts.push_back(fp8("dense.weight",{2,2}));
    ts.push_back(T{"dense.weight_scale_inv", "F32", {1,1}, {0,0,128,63}});
    const char* cfg="{\"model_type\":\"ernie4_5_moe\",\"quantization_config\":{\"quant_method\":\"fp8\",\"fmt\":\"e4m3\",\"expert_dtype\":\"mxfp4\",\"scale_fmt\":\"ue8m0\",\"weight_block_size\":[128,128]}}";
    Checkpoint ck; REQUIRE_EQ(ck.open(write_checkpoint(ts,cfg)),RAD_OK);
    Program p;
    for (const char* proj : {"w1","w2","w3"}) {
        std::string n=base+proj+".weight";
        p.name_map.clear(); p.name_map.push_back(map("expert",RAD_MAP_COPY,{n.c_str()}));
        CkptWeight w; std::string why;
        REQUIRE_EQ(ckpt_resolve(ck,p,"expert",false,&w,&why),RAD_OK);
        CHECK_EQ(w.rows,(int64_t)2); CHECK_EQ(w.cols,(int64_t)64);
        REQUIRE_EQ(w.enc.n_planes,2); CHECK_EQ(w.enc.plane[0].dtype,(uint32_t)RAD_FP4E2M1);
        CHECK_EQ(w.enc.plane[1].dtype,(uint32_t)RAD_E8M0); CHECK_EQ(w.enc.plane[1].block[1],(int64_t)32);
        std::vector<uint8_t> c(64),s(4); REQUIRE_EQ(ckpt_plane_rows(w,0,0,2,c.data(),&why),RAD_OK); REQUIRE_EQ(ckpt_plane_rows(w,1,0,2,s.data(),&why),RAD_OK);
        CHECK_EQ(c[0],(uint8_t)0x98); CHECK_EQ(c[32],(uint8_t)0xf7); CHECK_EQ(s[0],(uint8_t)127); CHECK_EQ(s[3],(uint8_t)129);
        std::vector<float> f(128); REQUIRE_EQ(ckpt_rows_f32(w,0,2,f.data(),&why),RAD_OK);
        CHECK_EQ(f[0],0.0f); CHECK_EQ(f[1],-0.5f); CHECK_EQ(f[2],0.5f); CHECK_EQ(f[3],1.0f);
        CHECK_EQ(f[32],1.0f); CHECK_EQ(f[64],3.0f); CHECK_EQ(f[65],-3.0f);
    }
    p.name_map.clear(); p.name_map.push_back(map("dense",RAD_MAP_COPY,{"dense.weight"}));
    CkptWeight w; std::string why; REQUIRE_EQ(ckpt_resolve(ck,p,"dense",false,&w,&why),RAD_OK);
    CHECK_EQ(w.enc.plane[0].dtype,(uint32_t)RAD_F8E4M3); CHECK_EQ(w.rows,(int64_t)2); CHECK_EQ(w.cols,(int64_t)2);
}
TEST(preview_mxfp4_refuses_missing_or_malformed_scale) {
    const char* n="model.layers.1.block_sparse_moe.experts.0.w1.weight";
    const char* cfg="{\"model_type\":\"ernie4_5_moe\",\"quantization_config\":{\"quant_method\":\"fp8\",\"fmt\":\"e4m3\",\"expert_dtype\":\"mxfp4\",\"scale_fmt\":\"ue8m0\",\"weight_block_size\":[128,128]}}";
    T c{n,"I8",{2,32},std::vector<uint8_t>(64,0x21)};
    for (int mode=0; mode<3; ++mode) {
        std::vector<T> ts{c};
        if (mode==1) ts.push_back(T{std::string(n)+"_scale_inv","F8_E8M0",{2,1},{127,127}});
        if (mode==2) ts.push_back(T{std::string(n)+"_scale_inv","U8",{2,2},{127,127,127,127}});
        Checkpoint ck; int rc=ck.open(write_checkpoint(ts,cfg));
        if (rc==RAD_OK) {Program p; p.name_map.push_back(map("expert",RAD_MAP_COPY,{n})); CkptWeight w; std::string why; rc=ckpt_resolve(ck,p,"expert",false,&w,&why);}
        CHECK(rc!=RAD_OK);
    }
}
/* The DSpark drafter's routed experts (model.mtp.L...) ship the trunk's packed-I8 + E8M0 form and
 * must import the same way; a name outside model.layers / model.mtp must not. */
TEST(preview_mxfp4_drafter_mtp_experts_import_like_the_trunks) {
    const char* cfg="{\"model_type\":\"ernie4_5_moe\",\"quantization_config\":{\"quant_method\":\"fp8\",\"fmt\":\"e4m3\",\"expert_dtype\":\"mxfp4\",\"scale_fmt\":\"ue8m0\",\"weight_block_size\":[128,128]}}";
    const std::string trunk="model.layers.1.block_sparse_moe.experts.0.";
    const std::string mtp="model.mtp.0.block_sparse_moe.experts.0.";
    const std::string other="model.mtpx.0.block_sparse_moe.experts.0.";
    std::vector<T> ts;
    for (const std::string& base : {trunk, mtp, other})
        for (const char* proj : {"w1","w2","w3"}) {
            const std::string n=base+proj+".weight";
            T c{n,"I8",{2,32},std::vector<uint8_t>(64,0x21)};
            c.bytes[0]=0x98; c.bytes[32]=0xf7;
            ts.push_back(c);
            ts.push_back(T{n+"_scale_inv","F8_E8M0",{2,2},{127,128,126,129}});
        }
    Checkpoint ck; REQUIRE_EQ(ck.open(write_checkpoint(ts,cfg)),RAD_OK);
    for (const char* proj : {"w1","w2","w3"}) {
        const std::string nt=trunk+proj+".weight", nm=mtp+proj+".weight", no=other+proj+".weight";
        Program p;
        CkptWeight wt, wm; std::string why;
        p.name_map.push_back(map("t",RAD_MAP_COPY,{nt.c_str()}));
        p.name_map.push_back(map("m",RAD_MAP_COPY,{nm.c_str()}));
        REQUIRE_EQ(ckpt_resolve(ck,p,"t",false,&wt,&why),RAD_OK);
        REQUIRE_EQ(ckpt_resolve(ck,p,"m",false,&wm,&why),RAD_OK);
        CHECK_EQ(wm.rows,wt.rows); CHECK_EQ(wm.cols,wt.cols); CHECK_EQ(wm.cols,(int64_t)64);
        REQUIRE_EQ(wm.enc.n_planes,2);
        CHECK_EQ(wm.enc.plane[0].dtype,(uint32_t)RAD_FP4E2M1);
        CHECK_EQ(wm.enc.plane[1].dtype,(uint32_t)RAD_E8M0);
        CHECK_EQ(wm.enc.plane[1].block[1],(int64_t)32);
        std::vector<uint8_t> c(64),sc(4);
        REQUIRE_EQ(ckpt_plane_rows(wm,0,0,2,c.data(),&why),RAD_OK);
        REQUIRE_EQ(ckpt_plane_rows(wm,1,0,2,sc.data(),&why),RAD_OK);
        CHECK_EQ(c[0],(uint8_t)0x98); CHECK_EQ(c[32],(uint8_t)0xf7); CHECK_EQ(sc[0],(uint8_t)127); CHECK_EQ(sc[3],(uint8_t)129);
        const SrcTensor *codes=nullptr,*scale=nullptr;
        CHECK_EQ(ck.import_piece(nm,&codes,&scale,&why),RAD_OK);
        CHECK_EQ(ck.import_piece(no,&codes,&scale,&why),RAD_E_NOTFOUND);
    }
}
RAD_TEST_MAIN()
