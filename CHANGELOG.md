# Changelog

## 0.3.0

- **Breaking**: `GpuSummary` fields are all required (no more default `float('nan')`). Fields are grouped logically: metadata, GPU utilization, memory utilization, device memory, power, temperature, clocks, derived.
- **New metrics**: `util_gpu_std`, `util_gpu_min`, `util_gpu_p5`, `util_gpu_p99`, `idle_pct`, `active_pct`, `mem_used_mean_mb`, `mem_util_pct`, `energy_j`, `sm_clock_mean_mhz`, `sm_clock_max_mhz`, `temp_mean_c`.
- **Multi-run profiling**: `gpu_profile(repeats=N, warmup_runs=M)` runs a function multiple times and aggregates cross-run statistics.
- **New types**: `RunStats` (mean/std/min/max/values), `MultiRunResult` (aggregated multi-run result with pre-computed `RunStats` for common metrics).
- **New function**: `profile_repeats(fn, repeats=N, ...)` for multi-run profiling without a decorator.
- **CLI**: added `--repeats` and `--warmup-runs` flags for multi-run profiling from the command line.
- Updated all documentation and tutorials to reflect the new API.
- Added `07_Multi_Run_Benchmarking.ipynb` notebook.

## 0.2.0

- Added extensive documentation in `docs/` (MkDocs).
- Added tutorials (PyTorch, CLI, logging/export).
- Added runnable Jupyter notebooks in `notebooks/`.
- Added additional concrete scripts in `examples/`.

## 0.1.0

- Initial release: decorator, context manager, and CLI.
- NVML backend with `nvidia-ml-py3` optional dependency.
- `nvidia-smi` fallback backend.
