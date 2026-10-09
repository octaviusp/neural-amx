#!/usr/bin/env bash
# Builds the neural-amx AMX Mod X module.
#
#   amxx/build.sh wasm      -> build/amxx/neural_amx_amxx_emscripten_wasm32.wasm (Emscripten side module, SIMD128)
#   amxx/build.sh linux     -> build/amxx/neural_amx_amxx_i386.so    (classic HLDS / ReHLDS, 32-bit, SSE2)
#   amxx/build.sh host      -> build/amxx/neural_amx_amxx_host.so    (this machine; compile check)
#
# Environment:
#   AMXX_SDK   AMX Mod X source root (public/sdk/amxxmodule.cpp)        [required]
#   HLSDK      Half-Life SDK root (common/, engine/, dlls/, public/)    [required]
#   METAMOD    Metamod headers root (metamod/)                          [required]
#   EMSDK_ENV  emsdk_env.sh of the Emscripten that built your server     [wasm]
#   WASM_FLAGS extra flags that must match your server's modules (e.g. "-flto")
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
target="${1:-wasm}"
out="$root/build/amxx"
: "${AMXX_SDK:?set AMXX_SDK to the AMX Mod X source root}" "${HLSDK:?set HLSDK}" "${METAMOD:?set METAMOD}"
mkdir -p "$out/obj-$target"
defs=(-DAMX_NOPROPLIST -DPAWN_CELL_SIZE=32 -DHAVE_STDINT_H -DNDEBUG)
incs=(-I"$root/amxx" -I"$AMXX_SDK/public" -I"$AMXX_SDK/public/sdk" -I"$AMXX_SDK/public/amtl" -I"$AMXX_SDK/public/amtl/amtl"
  -I"$METAMOD/metamod" -I"$HLSDK/common" -I"$HLSDK/dlls" -I"$HLSDK/engine" -I"$HLSDK/public" -I"$HLSDK/pm_shared")
common=(-O3 -fPIC -fvisibility=hidden -fno-strict-aliasing -ffp-contract=off)
cxx=(-std=c++14 -fno-exceptions -fno-rtti -fvisibility-inlines-hidden -w)
linux_defs=(-D__linux__=1 -D_LINUX -DLINUX -DPOSIX)

case "$target" in
  wasm)
    : "${EMSDK_ENV:?set EMSDK_ENV to emsdk_env.sh}"
    # shellcheck disable=SC1090
    source "$EMSDK_ENV" >/dev/null 2>&1
    cc=emcc; cxxc=em++
    arch=(-msimd128 ${WASM_FLAGS:-}); defs+=("${linux_defs[@]}" -D__linux=1)
    link=(-sSIDE_MODULE=1 -Wl,-Bsymbolic)
    file="$out/neural_amx_amxx_emscripten_wasm32.wasm" ;;
  linux)
    cc="${CC:-gcc}"; cxxc="${CXX:-g++}"
    arch=(-m32 -msse2 -mfpmath=sse -march=pentium4); defs+=("${linux_defs[@]}")  # never x87: keeps results bit-exact
    link=(-shared -m32 -static-libgcc)
    file="$out/neural_amx_amxx_i386.so" ;;
  host)
    cc="${CC:-cc}"; cxxc="${CXX:-c++}"
    arch=(-pipe); [[ "$(uname -s)" == Linux ]] && defs+=("${linux_defs[@]}")
    link=(-shared $([[ "$(uname -s)" == Darwin ]] && echo -undefined dynamic_lookup))
    file="$out/neural_amx_amxx_host.so" ;;
  *) echo "usage: $0 wasm|linux|host" >&2; exit 2 ;;
esac

objs=()
for c in "$root"/core/*.c; do
  o="$out/obj-$target/$(basename "$c" .c).o"
  "$cc" -std=c99 "${common[@]}" "${arch[@]}" -c "$c" -o "$o"
  objs+=("$o")
done
for c in "$root/amxx/neural_amx_amxx.cpp" "$AMXX_SDK/public/sdk/amxxmodule.cpp"; do
  o="$out/obj-$target/$(basename "$c" .cpp).o"
  "$cxxc" "${common[@]}" "${cxx[@]}" "${arch[@]}" "${defs[@]}" "${incs[@]}" -c "$c" -o "$o"
  objs+=("$o")
done
"$cxxc" "${objs[@]}" "${common[@]}" "${arch[@]}" "${link[@]}" -o "$file"
ls -la "$file"
