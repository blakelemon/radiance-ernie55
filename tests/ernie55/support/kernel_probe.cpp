#include "rad_abi.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// Independent ABI client: deliberately does not include libref implementation headers.
struct Storage {
    std::vector<float> f;
    std::vector<uint16_t> b;
    uint32_t dtype;
    Storage(int size, uint32_t dt): f(size, 1234.5f), b(size, 0x449a), dtype(dt) {}
    void* data() { return dtype == RAD_BF16 ? static_cast<void*>(b.data()) : f.data(); }
    float get(int i) const {
        return dtype == RAD_BF16 ? std::bit_cast<float>(uint32_t(b[i]) << 16) : f[i];
    }
    void put(int i, float v) {
        if (dtype == RAD_BF16) {
            uint32_t bits = std::bit_cast<uint32_t>(v);
            b[i] = uint16_t((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
        } else f[i] = v;
    }
};

RadTensor tensor(Storage& a, int rows, int cols, int pitch) {
    RadTensor t{};
    t.data = a.data(); t.dtype = a.dtype; t.rank = 2;
    t.shape[0] = rows; t.shape[1] = cols; t.stride[0] = pitch; t.stride[1] = 1;
    return t;
}

int main(int argc, char** argv) {
    if (argc != 12) return 20;
    void* handle = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!handle) { std::cerr << dlerror(); return 21; }
    auto count = reinterpret_cast<int(*)()>(dlsym(handle, "rad_kernel_count"));
    auto at = reinterpret_cast<const RadKernelInfo*(*)(int)>(dlsym(handle, "rad_kernel_at"));
    auto scount = reinterpret_cast<int(*)()>(dlsym(handle, "rad_kernel_schema_count"));
    auto sat = reinterpret_cast<const RadOpSchema*(*)(int)>(dlsym(handle, "rad_kernel_schema_at"));
    if (!count || !at || !scount || !sat) return 22;
    const std::string op = argv[2];
    bool registered = false;
    for (int i = 0; i < scount(); i++) if (sat(i) && op == sat(i)->op) registered = true;
    const RadKernelInfo* kernel = nullptr;
    for (int i = 0; i < count(); i++)
        if (at(i) && op == at(i)->op && at(i)->domain == RAD_DOMAIN_HOST) { kernel = at(i); break; }
    if (!registered || !kernel) { std::cout << "{\"missing\":true}\n"; return 3; }
    const uint32_t dtype = std::string(argv[3]) == "bf16" ? RAD_BF16 : RAD_F32;
    const int rows = std::stoi(argv[4]), n = std::stoi(argv[5]);
    const int ip = std::stoi(argv[6]), yp = std::stoi(argv[7]);
    const double beta = std::stod(argv[8]), linear_beta = std::stod(argv[9]);
    const bool bad_shape = std::string(argv[10]) == "bad-shape";
    const bool situ = op == "situ_mul" || op == "silu_mul";
    const int width = situ ? 2*n : n;
    Storage input(rows*ip, dtype), output(rows*yp, dtype);
    for (int r = 0; r < rows; r++) for (int i = 0; i < width; i++) {
        float v; if (!(std::cin >> v)) return 23; input.put(r*ip+i, v);
    }
    auto old_f = input.f; auto old_b = input.b;
    auto out_f = output.f; auto out_b = output.b;
    Storage ranges(4, RAD_F32), coeff(4, RAD_F32);
    if (!situ) {
        for (int i=0;i<4;i++) if (!(std::cin >> ranges.f[i])) return 24;
        for (int i=0;i<4;i++) if (!(std::cin >> coeff.f[i])) return 25;
    }
    RadTensor ts[4] = {tensor(input,rows,width,ip), tensor(output,rows,n,yp)};
    if (!situ) {
        ts[1] = tensor(ranges,1,4,4); ts[1].rank=1; ts[1].shape[0]=4; ts[1].stride[0]=1;
        ts[2] = tensor(coeff,1,4,4); ts[2].rank=1; ts[2].shape[0]=4; ts[2].stride[0]=1;
        ts[3] = tensor(output,rows,n,yp);
        if (bad_shape) ts[1].shape[0]=3;
    } else if (bad_shape) ts[0].shape[1]--;
    RadParam params[] = {RAD_INT("M",rows),RAD_INT("n",n),RAD_STR("dtype",argv[3]),
                         RAD_F64("beta",beta),RAD_F64("linear_beta",linear_beta)};
    RadArgs args{}; args.t=ts; args.n_t=situ?2:4; args.p=params; args.n_p=5; args.world_size=1;
    const int status=kernel->launch(&args,nullptr);
    bool guards=true;
    for (int r=0;r<rows;r++) for(int i=n;i<yp;i++) {
        int p=r*yp+i;
        if (dtype==RAD_BF16 ? output.b[p]!=out_b[p] : output.f[p]!=out_f[p]) guards=false;
    }
    std::cout << std::setprecision(10) << "{\"status\":" << status
              << ",\"input_unchanged\":" << ((old_f==input.f && old_b==input.b)?"true":"false")
              << ",\"guards_ok\":" << (guards?"true":"false") << ",\"values\":[";
    bool first=true;
    for(int r=0;r<rows;r++) for(int i=0;i<n;i++) {
        if(!first) std::cout << ',';
        first=false;
        float value=output.get(r*yp+i);
        if(std::isfinite(value)) std::cout << value; else std::cout << "null";
    }
    std::cout << "]}\n";
    dlclose(handle);
}
