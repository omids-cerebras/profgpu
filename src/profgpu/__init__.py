"""profgpu: programmatic GPU utilization monitoring (decorator + context manager).

This package provides three ways to profile GPU utilization on NVIDIA GPUs:

1. **Decorator** -- ``@gpu_profile(...)`` wraps a function and prints (or
   returns) a :class:`GpuSummary` when the function completes.
2. **Context manager** -- ``with GpuMonitor(...) as mon: ...`` samples in a
   background thread and exposes ``mon.summary`` after the block exits.
3. **CLI** -- ``profgpu -- python train.py`` profiles an arbitrary command.

Backends (from lowest to highest overhead):

* **NVML** -- recommended; calls NVIDIA Management Library via *pynvml* (no
  subprocess spawns).  Install with ``pip install nvidia-ml-py3``.
* **nvidia-smi** -- fallback; spawns ``nvidia-smi`` once per sample.
* **None / Null** -- always returns ``None`` metrics; useful for CPU-only
  testing or graceful degradation.
"""

__version__: str = "0.2.0"

from typing import List

from .monitor import (
    GpuBackendError,
    GpuMonitor,
    GpuSample,
    GpuSummary,
    ProfiledResult,
    gpu_profile,
)

__all__: List[str] = [
    "GpuBackendError",
    "GpuMonitor",
    "GpuSample",
    "GpuSummary",
    "ProfiledResult",
    "gpu_profile",
]
