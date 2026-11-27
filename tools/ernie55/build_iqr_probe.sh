#!/bin/bash
# Build tools/ernie55/iqr_probe.cpp against the ernie build tree (story 5-0-iqr-v4-target). usage: build_iqr_probe.sh OUT_BINARY [BUILD_DIR]
# The core archive and the quantiser plugin must already exist in BUILD_DIR (default: ~/src/ernie/build-ernie55). Nothing is rebuilt there.
set -euo pipefail
OUT=${1:?usage: build_iqr_probe.sh OUT_BINARY [BUILD_DIR]}
SRC=$(cd "$(dirname "$0")/../.." && pwd)
B=${2:-${ERNIE_BUILD_DIR:-$HOME/src/ernie/build-ernie55}}
[ -f "$B/core/librad_core.a" ] && [ -f "$B/radiance_home/quantizers/libquant.so" ] || { echo "build_iqr_probe: no core archive or libquant.so under $B" >&2; exit 2; }
c++ -std=gnu++20 -O2 -fPIE -DNDEBUG -DRAD_VERSION='"2.2.0"' -DRAD_DEFAULT_HOME='"/usr/local/share/radiance"' \
  -I"$SRC/core" -I"$SRC/abi" -I"$B/rad_include" -I"$SRC/vendor/cpp-httplib" -isystem "$SRC/vendor" -isystem "$SRC/vendor/shim" -isystem "$SRC/vendor/llama/common" -isystem "$SRC/vendor/rapidyaml" \
  -o "$OUT" "$SRC/tools/ernie55/iqr_probe.cpp" \
  -Wl,--push-state,--whole-archive "$B/core/librad_core.a" -Wl,--pop-state -ldl \
  "$B/vendor/librad_vendor_llama.a" "$B/core/librad_json.a" "$B/core/librad_simd.a" "$B/vendor/librad_vendor_yyjson.a" "$B/vendor/librad_vendor_httplib.a" "$B/vendor/librad_vendor_ryml.a" -lgomp -lpthread
