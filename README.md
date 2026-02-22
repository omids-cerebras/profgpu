# profgpu

Programmatic GPU utilization profiling for Python:

- **Decorator**: `@gpu_profile(...)`
- **Context manager**: `with GpuMonitor(...)`
- **CLI**: `profgpu -- python train.py`
- **Multi-run**: `@gpu_profile(repeats=5)` or `profile_repeats(fn, repeats=5)`

Primary target: **NVIDIA** GPUs.

Backends:
- **NVML** (recommended, low overhead) via `nvidia-ml-py3`
- **`nvidia-smi`** fallback

> `util.gpu` should be interpreted as "% of time the device was busy" over a sampling window---not "% of peak FLOPS."

---

## Installation

### Install from GitHub

```bash
pip install git+https://github.com/<you>/profgpu.git
```

### Recommended: NVML support

```bash
pip install "profgpu[nvml] @ git+https://github.com/<you>/profgpu.git"
```

### Development install

```bash
pip install -e .[dev]
pytest
```

### Installing PyTorch (for examples & notebooks)

`profgpu` itself does not require PyTorch, but most examples and notebooks
use it. Install the version matching your **CUDA driver**:

| CUDA driver version | pip install command |
|---------------------|--------------------|
| CUDA 12.x | `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| CUDA 11.8 | `pip install torch --index-url https://download.pytorch.org/whl/cu118` |
| CPU only | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |

> **Tip:** Check your CUDA driver with `nvidia-smi` (top right shows
> "CUDA Version: 12.x"). PyTorch `cu124` wheels work with any CUDA
> 12.x driver (12.4, 12.6, etc.) --- you do **not** need an exact match.
> Python >= 3.9 is required for PyTorch 2.x.

### Conda environment (recommended for old host toolchains)

If your system has an outdated GCC or CUDA, use the bundled script to
create a self-contained Conda environment with a modern compiler and
CUDA toolkit:

```bash
bash create_env.sh          # creates 'profgpu' env
conda activate profgpu
pytest
```

### Docker

Build a GPU-ready development image that bypasses all host compiler and
CUDA compatibility issues:

```bash
docker build -t profgpu:latest .
docker run --rm --gpus all profgpu:latest pytest
docker run --rm --gpus all profgpu:latest \
    profgpu --interval 0.1 -- python examples/pytorch_example.py
```

---

## Quickstart

### Decorator

```python
from profgpu import gpu_profile

@gpu_profile(interval_s=0.2)
def work():
    ...

work()
```

### PyTorch (CUDA async): use `sync_fn`

PyTorch queues GPU work asynchronously. If you want the profiled region to include the queued GPU work launched by the function, pass a synchronization function:

```python
import torch
from profgpu import gpu_profile

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
from profgpu import GpuMonitor

with GpuMonitor(device=0, interval_s=0.2) as mon:
    ...

print(mon.summary.format())
```

### Get structured results back (no printing)

```python
from profgpu import gpu_profile

@gpu_profile(report=False, return_profile=True)
def work():
    ...
    return 123

res = work()
print(res.value)
print(res.gpu.util_gpu_mean, res.gpu.util_gpu_p95)
```

### Multi-run benchmarking

Run your function multiple times to get mean, std, min, max across runs:

```python
from profgpu import gpu_profile

@gpu_profile(repeats=5, warmup_runs=1, return_profile=True, report=False)
def bench():
    ...

result = bench()  # MultiRunResult
print(f"util: {result.util_gpu.mean:.1f}% +- {result.util_gpu.std:.1f}%")
print(f"duration: {result.duration.format('s', 3)}")
print(f"energy: {result.energy.format(' J', 1)}")
```

Or without a decorator:

```python
from profgpu import profile_repeats

result = profile_repeats(lambda: my_function(), repeats=5, warmup_runs=1)
print(result.format())
```

---

## CLI

Profile any command:

```bash
profgpu --device 0 --interval 0.2 -- python train.py --epochs 3
```

Emit JSON:

```bash
profgpu --json -- python train.py
```

Multi-run:

```bash
profgpu --repeats 5 --warmup-runs 1 -- python train.py
```

Return code: the CLI returns your command's exit code (safe for CI).

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

- `docs/tutorials/pytorch.md`: practical PyTorch patterns (including multi-run)
- `docs/tutorials/cli.md`: CLI recipes (including `--repeats`)
- `docs/tutorials/logging.md`: structured logging and exporting samples

Notebooks (in `notebooks/`):

- `00_Check_GPU_and_Backends.ipynb`
- `01_Decorator_Quickstart.ipynb`
- `02_PyTorch_Training_Loop.ipynb`
- `03_Async_Pitfalls.ipynb`
- `04_CLI_Profile_Command.ipynb`
- `05_Export_Samples_and_Plot.ipynb`
- `06_CuPy_Benchmark.ipynb`
- `07_Multi_Run_Benchmarking.ipynb`

---

## License

MIT. See `LICENSE`.
