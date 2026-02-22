# API reference

## Public imports

```python
from profgpu import (
    gpu_profile,        # decorator (single or multi-run)
    profile_repeats,    # multi-run function (non-decorator)
    GpuMonitor,         # context manager
    GpuSummary,         # single-run result
    GpuSample,          # raw sample point
    ProfiledResult,     # single-run wrapper (value + GpuSummary)
    MultiRunResult,     # multi-run wrapper (value + cross-run stats)
    RunStats,           # mean/std/min/max across runs
    GpuBackendError,    # exception
)
```

---

## `gpu_profile(...)(fn)` decorator

```python
@gpu_profile(
    device: int = 0,
    interval_s: float = 0.2,
    backend: str = "auto",        # "auto" | "nvml" | "smi" | "none"
    strict: bool = False,
    sync_fn: Optional[Callable] = None,
    warmup_s: float = 0.0,
    store_samples: bool = False,
    report: Union[bool, Callable] = True,
    return_profile: bool = False,
    repeats: int = 1,             # run the function N times
    warmup_runs: int = 0,         # discard first M runs
)
```

### Parameters

| Parameter | Description |
|---|---|
| `device` | GPU index (0-based). |
| `interval_s` | Sampling period (seconds). |
| `backend` | `"auto"` (NVML then smi), `"nvml"`, `"smi"`, or `"none"`. |
| `strict` | Raise `GpuBackendError` if no backend available. |
| `sync_fn` | Sync function called before/after (e.g. `torch.cuda.synchronize`). |
| `warmup_s` | Ignore the first N seconds of *samples within each run*. |
| `store_samples` | Keep raw `GpuSample` objects. |
| `report` | `True` prints summary, `False` suppresses, callable receives the result. |
| `return_profile` | Return `ProfiledResult` (single) or `MultiRunResult` (multi) instead of raw value. |
| `repeats` | Number of times to run the function (>1 enables multi-run mode). |
| `warmup_runs` | Discard results from the first M runs (still executed). |

### Return value

| Mode | `return_profile=False` | `return_profile=True` |
|---|---|---|
| Single run (`repeats=1`) | Original return value | `ProfiledResult(value, gpu)` |
| Multi-run (`repeats>1`) | Last return value | `MultiRunResult` |

### Multi-run example

```python
@gpu_profile(repeats=5, warmup_runs=1, return_profile=True, report=False)
def train_epoch():
    ...

result = train_epoch()        # MultiRunResult
print(result.util_gpu.mean)   # mean util across 5 runs
print(result.util_gpu.std)    # standard deviation across runs
```

---

## `profile_repeats(fn, ...)` function

For when you don't want a decorator:

```python
from profgpu import profile_repeats

result = profile_repeats(
    lambda: train_epoch(model, loader),
    repeats=5,
    warmup_runs=1,
    interval_s=0.1,
)
print(result.util_gpu.mean, "+-", result.util_gpu.std)
```

Same parameters as `gpu_profile` (minus `return_profile`). Always returns `MultiRunResult`.

---

## `GpuMonitor` context manager

```python
with GpuMonitor(device=0, interval_s=0.2, sync_fn=...) as mon:
    ...
summary = mon.summary
```

### Parameters

Same core parameters as the decorator: `device`, `interval_s`, `backend`, `strict`, `sync_fn`, `warmup_s`, plus:

- `store_samples`: retain raw `GpuSample` objects in memory.
- `max_samples`: cap on stored samples before automatic 2x down-sampling.
- `reservoir_size`: size of the reservoir for approximate percentiles.
- `trace_len`: length of compressed sparkline trace.

### Attributes

- `summary: Optional[GpuSummary]` --- populated after the monitor exits.
- `samples: List[GpuSample]` --- raw metrics (if `store_samples=True`).

---

## `GpuSummary`

Frozen dataclass with all aggregate stats from a single monitoring session.
All fields are required; metrics that were never observed are `float('nan')`.

### Fields

| Group | Fields |
|---|---|
| **Metadata** | `device`, `name`, `duration_s`, `interval_s`, `n_samples` |
| **GPU utilization** | `util_gpu_mean`, `util_gpu_std`, `util_gpu_min`, `util_gpu_max`, `util_gpu_p5`, `util_gpu_p50`, `util_gpu_p95`, `util_gpu_p99` |
| **Idle / Active** | `idle_pct` (% samples < 5%), `active_pct` (% samples >= 50%) |
| **Memory util** | `util_mem_mean` |
| **Device memory** | `mem_used_mean_mb`, `mem_used_max_mb`, `mem_total_mb`, `mem_util_pct` |
| **Power** | `power_mean_w`, `power_max_w`, `energy_j` |
| **Temperature** | `temp_mean_c`, `temp_max_c` |
| **Clocks** | `sm_clock_mean_mhz`, `sm_clock_max_mhz` |
| **Derived** | `busy_time_est_s`, `sparkline`, `notes` |

### Methods

- `format() -> str` --- human-friendly multi-line report.

---

## `MultiRunResult`

Frozen dataclass aggregating multiple `GpuSummary` runs (returned by `gpu_profile(repeats>1)` or `profile_repeats()`).

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `value` | `Any` | Return value of the last run. |
| `runs` | `Tuple[GpuSummary, ...]` | Per-run summaries. |
| `duration` | `RunStats` | Cross-run stats of `duration_s`. |
| `util_gpu` | `RunStats` | Cross-run stats of `util_gpu_mean`. |
| `power` | `RunStats` | Cross-run stats of `power_mean_w`. |
| `energy` | `RunStats` | Cross-run stats of `energy_j`. |
| `peak_memory` | `RunStats` | Cross-run stats of `mem_used_max_mb`. |
| `peak_temp` | `RunStats` | Cross-run stats of `temp_max_c`. |

### Methods

- `from_runs(runs, value) -> MultiRunResult` --- constructor from list of summaries.
- `stats_for(field: str) -> RunStats` --- compute stats for any numeric `GpuSummary` field.
- `format() -> str` --- human-friendly multi-run report.

---

## `RunStats`

Frozen dataclass: descriptive statistics across runs.

| Field | Description |
|---|---|
| `mean` | Arithmetic mean (NaN values excluded). |
| `std` | Sample standard deviation (N-1). |
| `min` | Minimum value. |
| `max` | Maximum value. |
| `values` | Tuple of all per-run values. |

### Methods

- `format(unit, digits) -> str` --- e.g. `"87.3% +- 1.2%  (range 85.6-89.1%)"`.

---

## `ProfiledResult`

Frozen dataclass returned by `@gpu_profile(return_profile=True)` for a single run.

| Field | Description |
|---|---|
| `value` | Original return value of the decorated function. |
| `gpu` | `GpuSummary` captured during the run. |

---

## Exceptions

- `GpuBackendError`: raised when a requested backend cannot be initialized (when `strict=True`) or sampling fails at runtime.

---

If you're looking for "per-process" GPU accounting, note that util counters are device-level.
See [Concepts](concepts.md) and [Troubleshooting](troubleshooting.md).
