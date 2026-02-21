# Jupyter notebooks

This repo includes a set of **concrete, runnable notebooks** in the `notebooks/` directory.

They are written to be:

- readable as tutorials even if you don’t run them
- runnable on any NVIDIA GPU environment that has PyTorch installed
- robust to “no GPU” environments (they print a message and skip the heavy cells)

## Running the notebooks

1) Create an environment with Jupyter support:

```bash
pip install -e ".[notebooks,nvml]"
```

2) Launch Jupyter:

```bash
jupyter lab
```

3) Open the notebook files in `notebooks/`.

## Notebook overview

- **00_Check_GPU_and_Backends.ipynb**
  - Verifies NVML vs `nvidia-smi` sampling.
  - Shows a quick `GpuMonitor` block.

- **01_Decorator_Quickstart.ipynb**
  - Uses `@gpu_profile` on a small GPU workload.
  - Shows how `sync_fn` changes the measurement.

- **02_PyTorch_Training_Loop.ipynb**
  - Profiles a small training loop on synthetic data.
  - Demonstrates per-epoch summaries.

- **03_Async_Pitfalls.ipynb**
  - Demonstrates the classic “CUDA is async” footgun.
  - Shows how to measure correctly.

- **04_CLI_Profile_Command.ipynb**
  - Runs the `gpu-profile` CLI from a notebook and parses JSON output.

- **05_Export_Samples_and_Plot.ipynb**
  - Collects raw samples (`keep_samples=True`), exports to CSV, and plots `util.gpu` over time.

- **06_CuPy_Benchmark.ipynb**
  - Minimal CuPy example showing `sync_fn=cp.cuda.Stream.null.synchronize`.


## Notes

- The notebooks assume **PyTorch** is installed.
- They do *not* vendor or pin PyTorch versions, because that’s environment-specific.

If you want additional notebooks (e.g., CuPy, Triton, torch.compile), add an issue or PR.
