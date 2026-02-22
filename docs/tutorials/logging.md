# Tutorial: Logging & export

Most teams eventually want utilization measurements to be:

- written to a JSONL log for offline analysis
- emitted to structured logs
- sent to a metrics system (Prometheus/StatsD)

`profgpu` is designed to make this easy.

## 1) Write JSONL with the decorator

Pass a callable to `report=`. It receives a `GpuSummary` (single run) or `MultiRunResult` (multi-run).

```python
import json
from profgpu import gpu_profile

def write_jsonl(summary):
    with open("profgpu.jsonl", "a") as f:
        f.write(json.dumps(summary.__dict__) + "\n")

@gpu_profile(report=write_jsonl)
def work():
    ...

work()
```

## 2) Use Python's logging module

```python
import logging
from profgpu import gpu_profile

log = logging.getLogger("gpu")

@gpu_profile(report=False, return_profile=True)
def work():
    ...

res = work()
log.info("gpu_profile", extra={"gpu": res.gpu.__dict__})
```

## 3) Attach the summary to your function result

If you want to propagate the GPU summary through your codebase:

```python
from profgpu import gpu_profile

@gpu_profile(report=False, return_profile=True)
def train():
    ...
    return {"loss": 0.123}

res = train()
metrics = res.value
metrics["gpu_util_mean"] = res.gpu.util_gpu_mean
metrics["gpu_idle_pct"] = res.gpu.idle_pct
metrics["gpu_energy_j"] = res.gpu.energy_j
```

## 4) Emit a compact summary

`GpuSummary.format()` is intentionally human-friendly and short.

If you want a compact one-liner:

```python
from profgpu import gpu_profile

def one_liner(s):
    print(
        f"gpu={s.device} util_mean={s.util_gpu_mean:.1f}% "
        f"p95={s.util_gpu_p95:.1f}% idle={s.idle_pct:.0f}% "
        f"mem_max={s.mem_used_max_mb:.0f}MB energy={s.energy_j:.1f}J"
    )

@gpu_profile(report=one_liner)
def work():
    ...

work()
```

## 5) Log multi-run results

When using `repeats>1`, the `report` callable receives a `MultiRunResult`:

```python
import json
from profgpu import gpu_profile

def log_multi(result):
    record = {
        "n_runs": len(result.runs),
        "util_gpu_mean": result.util_gpu.mean,
        "util_gpu_std": result.util_gpu.std,
        "duration_mean": result.duration.mean,
        "duration_std": result.duration.std,
        "energy_mean": result.energy.mean,
        "peak_memory_mean": result.peak_memory.mean,
    }
    with open("profgpu_multi.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")

@gpu_profile(repeats=5, warmup_runs=1, report=log_multi)
def bench():
    ...

bench()
```

## 6) Keep and export raw samples

If you need a time series (not just aggregates), set `store_samples=True`.

```python
import csv
from profgpu import GpuMonitor

with GpuMonitor(interval_s=0.2, store_samples=True) as mon:
    ...

with open("gpu_samples.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t_s", "util_gpu", "util_mem", "mem_used_mb", "power_w", "temp_c"])
    for s in mon.samples:
        w.writerow([s.t_s, s.util_gpu, s.util_mem, s.mem_used_mb, s.power_w, s.temp_c])
```

This is especially useful for "bursty" workloads where a single mean can hide the pattern.

## 7) Integrate with experiment trackers

Most trackers accept key/value metrics.

### Single run

- MLflow: `mlflow.log_metric("gpu_util_mean", summary.util_gpu_mean)`
- Weights & Biases: `wandb.log({"gpu/util_mean": summary.util_gpu_mean})`

### Multi-run

```python
wandb.log({
    "gpu/util_mean": result.util_gpu.mean,
    "gpu/util_std": result.util_gpu.std,
    "gpu/energy_mean": result.energy.mean,
    "gpu/duration_mean": result.duration.mean,
})
```

Those libraries are intentionally not dependencies of `profgpu`; you can integrate in your own codebase.

---

If you want built-in exporters (Prometheus, StatsD), consider adding a small wrapper module in your org that uses `report=`.
