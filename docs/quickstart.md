# Quickstart

This page shows the most common usage patterns:

- Decorator: profile a function.
- Context manager: profile a block.
- CLI: profile an external command.
- Structured reporting: capture results in code.
- Multi-run benchmarking: estimate variance across repeated runs.

## 1) Decorator

```python
from profgpu import gpu_profile

@gpu_profile(interval_s=0.2)
def work():
    ...  # GPU work
    return 123

value = work()
```

At the end of the call, a one-page summary is printed to stdout.

### CUDA async frameworks (PyTorch/CuPy)

Many GPU frameworks schedule work asynchronously; the Python function can return *before* the GPU has finished.

If you want "the function call" to include queued GPU work, pass a synchronization function:

```python
import torch
from profgpu import gpu_profile

@gpu_profile(interval_s=0.1, sync_fn=torch.cuda.synchronize, warmup_s=0.2)
def bench():
    a = torch.randn(8192, 8192, device="cuda")
    b = torch.randn(8192, 8192, device="cuda")
    for _ in range(10):
        _ = a @ b

bench()
```

## 2) Context manager

Use the context manager when you want to profile arbitrary blocks of code:

```python
from profgpu import GpuMonitor

with GpuMonitor(device=0, interval_s=0.2) as mon:
    ...

print(mon.summary.format())
```

## 3) Get results programmatically (no printing)

Set `report=False` to disable printing, and `return_profile=True` to get a structured result:

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

The result is a `ProfiledResult(value=..., gpu=GpuSummary(...))`.

## 4) Use a custom report function (logging)

The `report` parameter can also be a callable. It receives a `GpuSummary` (single run) or `MultiRunResult` (multi-run).

```python
import json
from profgpu import gpu_profile

def write_jsonl(summary):
    with open("gpu_stats.jsonl", "a") as f:
        f.write(json.dumps(summary.__dict__) + "\n")

@gpu_profile(report=write_jsonl)
def work():
    ...

work()
```

## 5) CLI usage

Profile an external command:

```bash
profgpu --device 0 --interval 0.2 -- python train.py --epochs 3
```

Emit JSON:

```bash
profgpu --json -- python train.py
```

Multi-run from CLI:

```bash
profgpu --repeats 5 --warmup-runs 1 -- python train.py
```

See [CLI Tutorial](tutorials/cli.md) for patterns like profiling shell pipelines and handling exit codes.

## 6) Multi-run benchmarking

A single run can be noisy. Use `repeats` to run your function multiple times and get cross-run statistics (mean, std, min, max):

### With the decorator

```python
from profgpu import gpu_profile

@gpu_profile(repeats=5, warmup_runs=1, return_profile=True, report=False)
def train_epoch():
    ...

result = train_epoch()  # MultiRunResult
print(f"util.gpu: {result.util_gpu.mean:.1f}% +- {result.util_gpu.std:.1f}%")
print(f"duration: {result.duration.mean:.3f}s +- {result.duration.std:.3f}s")
print(f"energy:   {result.energy.mean:.1f} J")
```

### With `profile_repeats` (non-decorator)

```python
from profgpu import profile_repeats

result = profile_repeats(
    lambda: my_function(arg1, arg2),
    repeats=5,
    warmup_runs=1,
    interval_s=0.1,
)
print(result.format())  # human-friendly summary
```

### Accessing any field across runs

```python
# Pre-computed stats for the most common metrics:
result.duration      # RunStats for duration_s
result.util_gpu      # RunStats for util_gpu_mean
result.power         # RunStats for power_mean_w
result.energy        # RunStats for energy_j
result.peak_memory   # RunStats for mem_used_max_mb
result.peak_temp     # RunStats for temp_max_c

# On-demand stats for any GpuSummary field:
idle = result.stats_for("idle_pct")
print(f"idle: {idle.mean:.1f}% +- {idle.std:.1f}%")
```

## 7) What the numbers mean

The key metric is **device-level** `util.gpu`:

- `util_gpu_mean` --- % of time the GPU was busy (averaged across samples)
- `util_gpu_std` --- standard deviation of per-sample utilization
- `idle_pct` --- fraction of samples where GPU util < 5%
- `active_pct` --- fraction of samples where GPU util >= 50%
- `util_mem_mean` --- memory-controller utilization

The summary includes mean/std/min/max/p5/p50/p95/p99 percentiles, plus:

```
busy_time_est_s = duration_s * (util_gpu_mean / 100)
```

For deeper interpretation, see [Concepts](concepts.md).
