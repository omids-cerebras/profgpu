# gpu-profile

Programmatic GPU utilization profiling for Python:

- **Decorator**: `@gpu_profile(...)`
- **Context manager**: `with GpuMonitor(...)`
- **CLI**: `gpu-profile -- python train.py`

Primary target: **NVIDIA** GPUs.

Backends:
- **NVML** (recommended, low overhead) via `nvidia-ml-py3`
- **`nvidia-smi`** fallback

> `util.gpu` should be interpreted as “% of time the device was busy” over a sampling window—not “% of peak FLOPS.”

---

## Installation

### Install from GitHub

```bash
pip install git+https://github.com/<you>/gpu-profile.git
```

### Recommended: NVML support

```bash
pip install "gpu-profile[nvml] @ git+https://github.com/<you>/gpu-profile.git"
```

### Development install

```bash
pip install -e .[dev]
pytest
```

---

## Quickstart

### Decorator

```python
from gpu_profile import gpu_profile

@gpu_profile(interval_s=0.2)
def work():
    ...

work()
```

### PyTorch (CUDA async): use `sync_fn`

PyTorch queues GPU work asynchronously. If you want the profiled region to include the queued GPU work launched by the function, pass a synchronization function:

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

### Context manager

```python
from gpu_profile import GpuMonitor

with GpuMonitor(device=0, interval_s=0.2) as mon:
    ...

print(mon.summary.format())
```

### Get structured results back (no printing)

```python
from gpu_profile import gpu_profile

@gpu_profile(report=False, return_profile=True)
def work():
    ...
    return 123

res = work()
print(res.value)
print(res.gpu.util_gpu_mean, res.gpu.util_gpu_p95)
```

---

## CLI

Profile any command:

```bash
gpu-profile --device 0 --interval 0.2 -- python train.py --epochs 3
```

Emit JSON:

```bash
gpu-profile --json -- python train.py
```

Return code: the CLI returns your command’s exit code (safe for CI).

---

## Documentation

This repo includes a docs site in `docs/` (MkDocs) and runnable Jupyter notebooks in `notebooks/`.

- Docs entry: `docs/index.md`
- Notebooks overview: `docs/notebooks.md`

To build the docs locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

---

## Included tutorials & notebooks

- `docs/tutorials/pytorch.md`: practical PyTorch patterns
- `docs/tutorials/cli.md`: CLI recipes
- `docs/tutorials/logging.md`: structured logging and exporting samples

Notebooks (in `notebooks/`):

- `00_Check_GPU_and_Backends.ipynb`
- `01_Decorator_Quickstart.ipynb`
- `02_PyTorch_Training_Loop.ipynb`
- `03_Async_Pitfalls.ipynb`
- `04_CLI_Profile_Command.ipynb`
- `05_Export_Samples_and_Plot.ipynb`
- `06_CuPy_Benchmark.ipynb`

---

## License

MIT. See `LICENSE`.
