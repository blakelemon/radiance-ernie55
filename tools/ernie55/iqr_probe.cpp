/* iqr_probe -- quantise one f32 weight with libquant exactly as rad-convert does (core/format/quantise.h's loop into memory), decode it back with the core's own decoder,
 * and report the bytes of each plane and the relative error. Story 5-0-iqr-v4-target; CPU only, no model plugin, no kernel library, no device.
 *
 *   iqr_probe --in W.f32 --rows N --cols K --name blk.1.attn_output.weight --dec-out OUT.f32 [--home DIR] [--out-planes PREFIX] -- QUANTISER key=value ...
 *
 * W.f32 is little-endian f32 [N, K]. The options are the recipe's (rad-convert --list-quantizers): each one's kind (int, number, string) is read from the quantiser's own option
 * table. Prints one JSON line last: status, plane_bytes, total_bytes, rel_err, seconds, max_abs_err. Build: tools/ernie55/build_iqr_probe.sh. */
#include "rad_plugin.h"
#include "rad_quant.h"
#include "format/encoding.h"
#include "format/quantise.h"
#include "plugin/registry.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::string jesc(const std::string& s) {
    std::string o;
    for (char c : s) { if (c == '"' || c == '\\') { o += '\\'; o += c; } else if ((unsigned char)c < 0x20) o += ' '; else o += c; }
    return o;
}

static int die(const char* m) { std::fprintf(stderr, "iqr_probe: %s\n", m); return 2; }

int main(int argc, char** argv) {
    std::string in, name = "blk.0.weight", dec_out, home = "radiance_home", planes_prefix;
    int64_t rows = 0, cols = 0;
    int i = 1;
    for (; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--") { ++i; break; }
        auto need = [&](const char* flag) -> const char* { return (i + 1 < argc) ? argv[++i] : (std::fprintf(stderr, "iqr_probe: %s needs a value\n", flag), nullptr); };
        if (a == "--in") { const char* v = need("--in"); if (!v) return 2; in = v; }
        else if (a == "--rows") { const char* v = need("--rows"); if (!v) return 2; rows = std::atoll(v); }
        else if (a == "--cols") { const char* v = need("--cols"); if (!v) return 2; cols = std::atoll(v); }
        else if (a == "--name") { const char* v = need("--name"); if (!v) return 2; name = v; }
        else if (a == "--dec-out") { const char* v = need("--dec-out"); if (!v) return 2; dec_out = v; }
        else if (a == "--home") { const char* v = need("--home"); if (!v) return 2; home = v; }
        else if (a == "--out-planes") { const char* v = need("--out-planes"); if (!v) return 2; planes_prefix = v; }
        else return die(("unknown flag " + a).c_str());
    }
    if (in.empty() || rows <= 0 || cols <= 0 || i >= argc) return die("usage: --in F --rows N --cols K [--name N] [--dec-out F] -- QUANTISER key=value ...");
    const std::string qname = argv[i++];

    rad::Registry reg;
    if (reg.load_plugin(home + "/quantizers/libquant.so", 0) < 0) return die("cannot load libquant.so");
    const rad::Registry::QuantRow* qr = reg.quantizer(qname);
    if (!qr || !qr->info) return die(("no quantiser " + qname).c_str());
    const RadQuantizerInfo* q = qr->info;

    std::vector<std::string> keys, vals;
    std::vector<RadParam> p;
    for (; i < argc; ++i) {
        std::string kv = argv[i];
        const size_t eq = kv.find('=');
        if (eq == std::string::npos) return die(("option without '=': " + kv).c_str());
        const std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
        int type = -1;
        for (int j = 0; j < q->n_options; ++j) if (k == q->options[j].key) type = q->options[j].type;
        if (type < 0) return die(("the quantiser has no option " + k).c_str());
        keys.push_back(k);
        vals.push_back(v);
        p.push_back(type == RAD_P_INT ? RAD_INT(nullptr, std::atoll(v.c_str())) : type == RAD_P_F64 ? RAD_F64(nullptr, std::atof(v.c_str())) : RAD_STR(nullptr, nullptr));
    }
    for (size_t j = 0; j < p.size(); ++j) { p[j].key = keys[j].c_str(); if (p[j].kind == RAD_P_STR) p[j].sval = vals[j].c_str(); }

    std::vector<float> w((size_t)(rows * cols));
    {
        FILE* f = std::fopen(in.c_str(), "rb");
        if (!f) return die("cannot open --in");
        const size_t got = std::fread(w.data(), 4, w.size(), f);
        std::fclose(f);
        if (got != w.size()) return die("--in is shorter than rows x cols f32");
    }

    RadQuantWeight qw{};
    qw.name = name.c_str();
    qw.rank = 2;
    qw.shape[0] = rows; qw.shape[1] = cols; qw.rows = rows; qw.cols = cols;
    qw.layer = qw.expert = -1;
    RadEncoding enc{};
    int st = q->encoding(p.data(), (int)p.size(), &qw, &enc);
    if (st != RAD_OK) { std::printf("{\"status\":%d,\"error\":\"the quantiser declined or refused this weight\"}\n", st); return 1; }
    std::vector<std::vector<uint8_t>> planes;
    for (int k = 0; k < enc.n_planes; ++k) {
        const int64_t pb = rad_enc_plane_bytes(&enc.plane[k], rows, cols);
        if (pb < 0) return die("a plane has a negative size for this shape");
        planes.emplace_back((size_t)pb, 0);
    }
    rad::QuantIo io;
    io.read = [&](int64_t r0, int64_t n, float* dst, std::string*) { std::memcpy(dst, w.data() + r0 * cols, (size_t)(n * cols) * sizeof(float)); return (int)RAD_OK; };
    io.write = [&](int k, int64_t off, const void* data, int64_t bytes) {
        if (k < 0 || (size_t)k >= planes.size()) return (int)RAD_E_INVAL;
        std::vector<uint8_t>& pl = planes[(size_t)k];
        if (k < 0 || (size_t)k >= planes.size()) return (int)RAD_E_INVAL;
        if (off < 0 || bytes < 0 || (size_t)(off + bytes) > pl.size()) return (int)RAD_E_INVAL;
        std::memcpy(pl.data() + off, data, (size_t)bytes);
        return (int)RAD_OK;
    };
    std::string why;
    const auto t0 = std::chrono::steady_clock::now();
    st = rad::quantise_weight(*q, p.data(), (int)p.size(), qw, enc, rad::kQuantBandBytes, io, &why);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (st != RAD_OK) { std::printf("{\"status\":%d,\"error\":\"%s\"}\n", st, jesc(why).c_str()); return 1; }

    std::vector<const void*> d;
    for (auto& pl : planes) d.push_back(pl.data());
    rad::EncodedView v;
    rad::enc_view(enc, rows, cols, d.data(), &v);
    std::vector<float> dec((size_t)(rows * cols));
    if (rad::enc_decode_rows(v, 0, rows, dec.data(), &why) != RAD_OK) { std::printf("{\"status\":-1,\"error\":\"decode: %s\"}\n", jesc(why).c_str()); return 1; }
    double num = 0, den = 0, mx = 0;
    for (size_t k = 0; k < w.size(); ++k) { const double e = (double)w[k] - dec[k]; num += e * e; den += (double)w[k] * w[k]; mx = std::max(mx, std::fabs(e)); }
    if (!dec_out.empty()) {
        FILE* f = std::fopen(dec_out.c_str(), "wb");
        if (!f) return die("cannot write --dec-out");
        const size_t wr = std::fwrite(dec.data(), 4, dec.size(), f);
        if (std::fclose(f) != 0 || wr != dec.size()) return die("cannot write --dec-out completely");
    }
    int64_t total = 0;
    std::string pb;
    for (size_t k = 0; k < planes.size(); ++k) {
        total += (int64_t)planes[k].size();
        pb += (k ? "," : "") + std::to_string(planes[k].size());
        if (!planes_prefix.empty()) { FILE* f = std::fopen((planes_prefix + "." + std::to_string(k)).c_str(), "wb"); if (f) { std::fwrite(planes[k].data(), 1, planes[k].size(), f); std::fclose(f); } }
    }
    if (!std::isfinite(num) || !std::isfinite(mx)) return die("the error is not finite");
    std::printf("{\"status\":0,\"n_planes\":%d,\"plane_bytes\":[%s],\"total_bytes\":%lld,\"rel_err\":%.9g,\"max_abs_err\":%.9g,\"seconds\":%.3f}\n", enc.n_planes, pb.c_str(), (long long)total, den > 0 ? std::sqrt(num / den) : 0.0, mx, secs);
    return 0;
}
