#!/bin/bash
# Builds the card version of the row-identity harness (K1 of the row-identity GPU plan): the same families, probe rows and byte comparison as
# r4d_emu_row_identity, but each launcher runs on the gfx1201 card on device copies of its operands, looked up in libr4d.so by dlsym.
# usage: build_row_identity_device.sh [OUT]   (default ./rad-row-identity; needs ROCm's clang++ and headers, as the build container has)
# run:   rad-row-identity FAMILY prod        FAMILY: pair woa mlaproj hca hcawide hcadec mla mlawide mhc moe f8a8 f8a8env f8a8ctl gemmnt gemmntctl idxl router gather gatherchain
#        ROWID_LIBR4D=/path/libr4d.so overrides the library (default /opt/radiance/share/radiance/kernels/libr4d.so). Exit 0 = every probe row byte-equal to its one-row launch.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
src=$(cd "$here/../.." && pwd)
out=${1:-./rad-row-identity}
CXX=${CXX:-/opt/rocm/llvm/bin/clang++}
"$CXX" -std=c++20 -O2 -DROWID_DEVICE -D__HIP_PLATFORM_AMD__ -I"$src/abi" -I"$here" -I/opt/rocm/include \
    "$here/test_row_identity.cpp" -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib -lamdhip64 -ldl -o "$out"
