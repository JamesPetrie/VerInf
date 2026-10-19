#!/bin/bash
# Build verinf_runner against a llama.cpp checkout and its build directory.
#   LLAMA_CPP=~/build/llama.cpp-verinf LLAMA_BUILD=build-cpu tools/llama_runner/build.sh
# The binary lands in $LLAMA_CPP/$LLAMA_BUILD/bin and records the checkout's commit.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
L=${LLAMA_CPP:?set LLAMA_CPP to the llama.cpp checkout}
B=$L/${LLAMA_BUILD:-build}
commit=$(git -C "$L" rev-parse HEAD)
if [ -n "$(git -C "$L" status --porcelain --untracked-files=no)" ]; then
  commit="$commit-dirty"
fi
${CXX:-g++} -O2 -std=c++17 "$here/verinf_runner.cpp" -o "$B/bin/verinf_runner" \
  -I"$L/include" -I"$L/ggml/include" -L"$B/bin" -lllama -lggml -lggml-base \
  -Wl,-rpath,"$B/bin" -DVERINF_LLAMA_COMMIT="\"$commit\""
echo "$B/bin/verinf_runner ($commit)"
