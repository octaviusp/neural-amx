#!/usr/bin/env bash
# Builds the test/bench runner for every backend into build/:
#   runner-native (NEON on arm64), runner-scalar, runner-sse2 (x86_64, runs under Rosetta on Apple Silicon),
#   runner-wasm-simd.js and runner-wasm-scalar.js (Node, need Emscripten: EMSDK_ENV=/path/to/emsdk_env.sh).
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
out="$root/build"
mkdir -p "$out"
src=("$root"/core/*.c "$root/tests/runner.c")
flags=(-std=c99 -O3 -ffp-contract=off -Wall -Wextra -Wno-unused-parameter)
clang "${flags[@]}" "${src[@]}" -lm -o "$out/runner-native"
clang "${flags[@]}" -DNAM_NO_SIMD "${src[@]}" -lm -o "$out/runner-scalar"
if [[ "$(uname -s)" == Darwin ]]; then clang -arch x86_64 "${flags[@]}" "${src[@]}" -o "$out/runner-sse2"; fi
if [[ -n "${EMSDK_ENV:-}" ]]; then
  # shellcheck disable=SC1090
  source "$EMSDK_ENV" >/dev/null 2>&1
  wasm=(-sNODERAWFS=1 -sALLOW_MEMORY_GROWTH=1 -sENVIRONMENT=node -sEXIT_RUNTIME=1)
  emcc "${flags[@]}" -msimd128 "${wasm[@]}" "${src[@]}" -o "$out/runner-wasm-simd.js"
  emcc "${flags[@]}" "${wasm[@]}" "${src[@]}" -o "$out/runner-wasm-scalar.js"
fi
ls "$out"/runner-*
