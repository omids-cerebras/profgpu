# Quickstart

This page shows the most common usage patterns:

- Decorator: profile a function.
- Context manager: profile a block.
- CLI: profile an external command.
- Structured reporting: capture results in code.

## 1) Decorator

```python
from gpu_profile import gpu_profile

@gpu_profile(interval_s=0.2)
def work():
    ...  # GPU work
    return 123

value = work()
```

At the end of the call, a one-page summary is printed to stdout.

### CUDA async frameworks (PyTorch/CuPy)

Many GPU frameworks schedule work asynchronously; the Python function can return *before* the GPU has finished.

If you want “the function call” to include queued GPU work, pass a synchronization function:

```python
import torch
from gpu_profile import gpu_profile

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
from gpu_profile import GpuMonitor

with GpuMonitor(device=0, interval_s=0.2) as mon:
    ...

print(mon.summary.format())
```

## 3) Get results programmatically (no printing)

Set `report=False` to disable printing, and `return_profile=True` to get a structured result:

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

## 4) Use a custom report function (logging)

The `report` parameter can also be a callable. It receives a `GpuSummary`.

```python
import json
from gpu_profile import gpu_profile

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
gpu-profile --device 0 --interval 0.2 -- python train.py --epochs 3
```

Emit JSON:

```bash
gpu-profile --json -- python train.py
```

See [CLI Tutorial](tutorials/cli.md) for patterns like profiling shell pipelines and handling exit codes.

## 6) What the numbers mean

The key metric is **device-level** `util.gpu`:

- `util.gpu` ≈ % of time the GPU was busy over a recent sampling window
- `util.mem` ≈ memory-controller utilization

The summary includes mean/p50/p95/max and an estimate of “busy time”:

```
busy_time_est_s = duration_s * (util_gpu_mean / 100)
```

For deeper interpretation, see [Concepts](concepts.md).
