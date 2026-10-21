#!/bin/bash
# Build verinf_runner and verinf_render against a llama.cpp checkout and its build directory.
#   LLAMA_CPP=~/build/llama.cpp-verinf LLAMA_BUILD=build-cpu tools/llama_runner/build.sh
# The binaries (verinf_runner, verinf_render, verinf_optest) land in $LLAMA_CPP/$LLAMA_BUILD/bin and
# record the checkout's commit.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
L=${LLAMA_CPP:?set LLAMA_CPP to the llama.cpp checkout}
B=$L/${LLAMA_BUILD:-build}
if git -C "$L" rev-parse HEAD > /dev/null 2>&1; then
  commit=$(git -C "$L" rev-parse HEAD)
  if [ -n "$(git -C "$L" status --porcelain --untracked-files=no)" ]; then
    commit="$commit-dirty"
  fi
  [ -z "${LLAMA_COMMIT:-}" ] || [ "$commit" = "$LLAMA_COMMIT" ] || { echo "checkout is $commit, not $LLAMA_COMMIT"; exit 1; }
else
  commit=${LLAMA_COMMIT:?not a git checkout: set LLAMA_COMMIT}
fi
${CXX:-g++} -O2 -std=c++17 "$here/verinf_runner.cpp" -o "$B/bin/verinf_runner" \
  -I"$L/include" -I"$L/ggml/include" -L"$B/bin" -lllama -lggml -lggml-base \
  -Wl,-rpath,"$B/bin" -DVERINF_LLAMA_COMMIT="\"$commit\""
echo "$B/bin/verinf_runner ($commit)"
${CXX:-g++} -O2 -std=c++17 "$here/verinf_render.cpp" -o "$B/bin/verinf_render" \
  -I"$L/include" -I"$L/ggml/include" -I"$L/common" -I"$L/vendor" -L"$B/bin" \
  -lllama-common -lllama -lggml -lggml-base -Wl,-rpath,"$B/bin" -DVERINF_LLAMA_COMMIT="\"$commit\""
echo "$B/bin/verinf_render ($commit)"
${CXX:-g++} -O2 -std=c++17 "$here/verinf_optest.cpp" -o "$B/bin/verinf_optest" \
  -I"$L/ggml/include" -L"$B/bin" -lggml -lggml-base -Wl,-rpath,"$B/bin"
echo "$B/bin/verinf_optest"
