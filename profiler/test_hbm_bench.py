"""CPU regression of the actual CUDA chase kernel's indexing.

Compile its unchanged body with CUDA index globals shimmed, then check the
requested population and guard cells on either side of block boundaries.
This needs a C++ compiler, not torch/CUDA, and makes no timing claims.

    python3 -m unittest profiler/test_hbm_bench.py
"""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(shutil.which("c++"), "C++ compiler required for kernel indexing check")
class ChaseIndexingTests(unittest.TestCase):
    def test_requested_walkers_and_guard_cells(self):
        source = (Path(__file__).resolve().parent / "bench/bench_hbm_random.cu").read_text()
        kernel = re.search(r"__global__ void k_chase\(.*?\n}", source, re.S).group(0)
        shim = """#include <cassert>
#include <cstdint>
#include <vector>
#define __global__
struct Dim { uint64_t x; } blockIdx{0}, blockDim{256}, threadIdx{0};
"""
        driver = """
int main() {
    const uint64_t x[8] = {1, 2, 3, 4, 5, 6, 7, 0};
    const uint64_t untouched = UINT64_MAX;
    for (uint64_t walkers : {1, 255, 256, 257, 65536, 65537}) {
        uint64_t blocks = (walkers + 255) / 256;
        std::vector<uint64_t> y(blocks * 256 + 1, untouched);
        for (blockIdx.x = 0; blockIdx.x < blocks; ++blockIdx.x)
            for (threadIdx.x = 0; threadIdx.x < 256; ++threadIdx.x)
                k_chase(x, y.data(), 3, 8, 0, walkers);
        for (uint64_t tid = 0; tid < walkers; ++tid)
            assert(y[tid] < 8); // every requested chain executed
        for (uint64_t tid = walkers; tid < y.size(); ++tid)
            assert(y[tid] == untouched); // no padded thread wrote its sink
    }
}
"""
        with tempfile.TemporaryDirectory() as td:
            cpp, exe = Path(td) / "chase.cpp", Path(td) / "chase"
            cpp.write_text(shim + kernel + driver)
            compiled = subprocess.run(
                ["c++", "-std=c++17", "-O1", str(cpp), "-o", str(exe)],
                capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            result = subprocess.run([str(exe)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
