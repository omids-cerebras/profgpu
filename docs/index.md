# gpu-profile

`gpu-profile` is a small, dependency-light library for **sampling GPU utilization** while your Python code (or an external command) runs.

It is designed for two common workflows:

1. **Library mode** (import in Python)
   - A decorator (`@gpu_profile(...)`) for wrapping a function.
   - A context manager (`with GpuMonitor(...)`) for wrapping an arbitrary block.

2. **CLI mode** (profile a subprocess)
   - `gpu-profile -- python train.py ...` prints a summary at the end.
   - `gpu-profile --json -- ...` emits machine-friendly JSON.

## What it measures

The primary signal is **device-level utilization**:

- `util.gpu` — typically “% of time the GPU was busy” over the driver’s sampling window.
- `util.mem` — memory controller utilization (%), a rough proxy for memory pressure.

Optionally (depending on backend / driver support) it also samples:

- memory used / total
- power draw (W)
- temperature (°C)
- SM and memory clocks

> Important: `util.gpu = 90%` does **not** mean “90% of peak FLOPS.” It usually means the GPU had *some* work running 90% of the time.

## Backends

`gpu-profile` supports:

- **NVML** (recommended): low overhead, robust, fast polling.
  - Install: `pip install gpu-profile[nvml]`
- **nvidia-smi** fallback: works if `nvidia-smi` exists, but calls an external process each sample.

Today the package targets **NVIDIA** GPUs (NVML / nvidia-smi). The public API is backend-agnostic, so additional backends (AMD/Intel) can be added later.

## Quick example

```python
from gpu_profile import gpu_profile

@gpu_profile(interval_s=0.2)
def work():
    # GPU workload here
    ...

work()
```

If you’re using a framework that schedules GPU work asynchronously (e.g. **PyTorch**, **CuPy**), pass a `sync_fn` so the decorator boundaries match “all GPU work launched by the function”:

```python
import torch
from gpu_profile import gpu_profile

@gpu_profile(interval_s=0.1, sync_fn=torch.cuda.synchronize, warmup_s=0.2)
def matmul_bench():
    a = torch.randn(8192, 8192, device="cuda")
    b = torch.randn(8192, 8192, device="cuda")
    for _ in range(10):
        _ = a @ b

matmul_bench()
```

## Where to go next

- **Install & verify**: see [Installation](installation.md)
- **Start using it**: see [Quickstart](quickstart.md)
- **Understand the metrics**: see [Concepts](concepts.md)
- **Concrete walkthroughs**: see [Tutorials](tutorials/pytorch.md)
- **Run the Jupyter notebooks**: see [Notebooks](notebooks.md)
