# API reference

The public API is intentionally small.

```python
from gpu_profile import (
    gpu_profile,
    GpuMonitor,
    GpuSummary,
    GpuSample,
    ProfiledResult,
    GpuBackendError,
)
```

## `gpu_profile(...)(fn)` decorator

Signature (simplified):

```python
@gpu_profile(
    device: int = 0,
    interval_s: float = 0.2,
    backend: str = "auto",   # "auto" | "nvml" | "smi" | "none"
    strict: bool = True,
    sync_fn: Optional[Callable[[], None]] = None,
    warmup_s: float = 0.0,
    report: Union[bool, Callable[[GpuSummary], None]] = True,
    return_profile: bool = False,
)
```

### Parameters

- **device**: GPU index (0-based).
- **interval_s**: sampling period (seconds).
- **backend**:
  - `"auto"`: NVML if available, otherwise `nvidia-smi`
  - `"nvml"`: force NVML backend
  - `"smi"`: force `nvidia-smi` backend
  - `"none"`: disable sampling (useful for dry runs)
- **strict**: when `True`, missing tooling/backends raise an exception.
- **sync_fn**: optional synchronization function to make the measured region include queued async work.
  - PyTorch: `torch.cuda.synchronize`
  - CuPy: `cp.cuda.Stream.null.synchronize`
- **warmup_s**: ignore the first N seconds of samples when computing summary stats.
- **report**:
  - `True`: print a formatted summary
  - `False`: print nothing
  - callable: called with the `GpuSummary`
- **return_profile**:
  - `False`: return the wrapped function’s original return value
  - `True`: return `ProfiledResult(value=..., gpu=GpuSummary(...))`

### Return value

- If `return_profile=False`: returns the wrapped function’s return value.
- If `return_profile=True`: returns a `ProfiledResult`.

## `GpuMonitor` context manager

```python
with GpuMonitor(device=0, interval_s=0.2, sync_fn=...) as mon:
    ...
summary = mon.summary
```

### Parameters

`GpuMonitor` accepts the same core parameters as the decorator:

- `device`, `interval_s`, `backend`, `strict`, `sync_fn`, `warmup_s`

Additionally:

- `keep_samples`: whether to retain full samples in memory.

### Attributes

- `summary: Optional[GpuSummary]` — populated when the monitor exits.
- `samples: List[GpuSample]` — sampled raw metrics (if retained).

## `GpuSummary`

`GpuSummary` is a frozen dataclass that stores the aggregate stats over a monitoring region.

Fields include:

- device/name
- duration
- sample count / interval
- util.gpu mean/p50/p95/max
- util.mem mean
- memory used max / total
- power mean/max
- temperature max
- busy time estimate
- sparkline (mini utilization trace)

Convenience:

- `GpuSummary.format()` returns a human-friendly report string.

## Exceptions

- `GpuBackendError`: raised when a requested backend cannot be initialized (when `strict=True`).

---

If you’re looking for “per-process” GPU accounting, note that the util counters are device-level.
See [Concepts](concepts.md) and [Troubleshooting](troubleshooting.md).
