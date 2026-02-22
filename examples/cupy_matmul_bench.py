"""CuPy matmul benchmark example.

Run:
  python examples/cupy_matmul_bench.py

Notes:
- Requires cupy installed (matching your CUDA version).
- Demonstrates passing a CuPy sync function.
"""

import time

from profgpu import gpu_profile

try:
    import cupy as cp
except Exception as e:
    raise SystemExit(
        "CuPy is not installed (or failed to import). Install cupy for your CUDA version, e.g. cupy-cuda12x.\n"
        f"Original error: {e}"
    ) from e


SYNC = cp.cuda.Stream.null.synchronize


@gpu_profile(interval_s=0.1, sync_fn=SYNC, warmup_s=0.2)
def matmul_bench(n: int = 8192, steps: int = 10):
    a = cp.random.randn(n, n, dtype=cp.float32)
    b = cp.random.randn(n, n, dtype=cp.float32)

    # Warmup
    _ = a @ b
    SYNC()

    t0 = time.perf_counter()
    for _ in range(steps):
        _ = a @ b
    SYNC()
    dt = time.perf_counter() - t0
    print(f"matmul: n={n} steps={steps} wall={dt:.3f}s")


if __name__ == "__main__":
    matmul_bench()
