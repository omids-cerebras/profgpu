# Concepts

This section explains what `profgpu` is (and is **not**) measuring, and why the defaults are the way they are.

## Device-level utilization is a time fraction

Most GPU "utilization" counters reported by NVML / `nvidia-smi` are best interpreted as:

> **The fraction of time the device had at least one active workload in the last sampling window.**

So if you see `util.gpu = 80%`, that often means:

- For ~80% of the last window, there were kernels running (or the graphics engine was active).
- For ~20% of the window, the device was idle.

It does **not** guarantee anything about:

- how close you are to peak FLOPS
- whether kernels are compute-bound vs memory-bound
- whether you are saturating memory bandwidth

To answer "how close am I to the hardware limit?", you typically need kernel-level profiling tools (Nsight Compute/Systems) or hardware counters beyond util%.

## Why sampling interval matters

A utilization counter is a *statistic over time*. If your workload is bursty (short kernels separated by CPU work or I/O), then:

- With a long interval (e.g. 1s), bursts may be averaged out.
- With a short interval (e.g. 50ms), you can see bursts more clearly.

`profgpu` defaults to `interval_s=0.2` (200ms) because it's usually:

- fast enough to show "bursty vs steady" patterns
- slow enough to avoid noticeable overhead in most runs

If you have micro-kernels or extremely bursty patterns, try `interval_s=0.05` or `0.1`.

## The full GpuSummary

Each profiling session produces a `GpuSummary` with 30 fields grouped as:

| Group | What it tells you |
|---|---|
| **GPU utilization** (mean/std/min/max/p5/p50/p95/p99) | How busy the GPU was - and how variable |
| **idle_pct / active_pct** | Quick classification: idle (<5%) vs active (>=50%) |
| **Memory utilization** | Memory-controller busyness |
| **Device memory** (mean/max/total/util_pct) | How much VRAM you're using |
| **Power** (mean/max/energy_j) | Watt draw and total energy consumed |
| **Temperature** (mean/max) | Thermal state |
| **Clocks** (sm_mean/sm_max) | Whether throttling occurred |
| **Derived** (busy_time_est, sparkline) | Quick interpretations |

## The "busy time estimate"

The summary reports:

```
busy_time_est_s = duration_s * mean(util.gpu)/100
```

This is a coarse estimate of *how much of the wall time the GPU was busy*, under the assumption that the utilization counter is representative.

Common interpretations:

- **Busy time is close to duration** --- the GPU is likely the bottleneck (or at least continuously active).
- **Busy time is much smaller than duration** --- likely CPU/I/O bottlenecks, data loading, synchronization gaps, or very small kernels.

## Why multiple runs matter

A single profiling run gives you one sample of GPU behavior. That sample is affected by:

- CUDA context initialization / JIT warmup
- OS scheduling jitter
- Thermal throttling (the GPU heats up during the run)
- Background processes
- Memory allocator behavior

Running the same workload multiple times gives you:

- **Mean**: a more stable estimate of typical behavior
- **Std**: how much run-to-run variation exists
- **Min/Max**: best-case and worst-case across runs

### Using `repeats` and `warmup_runs`

```python
@gpu_profile(repeats=5, warmup_runs=1, return_profile=True)
def bench():
    ...
```

- `warmup_runs=1`: the first run is executed but its results are discarded (handles JIT, cache warmup)
- `repeats=5`: the next 5 runs are measured and aggregated

The result is a `MultiRunResult` with pre-computed `RunStats` for duration, utilization, power, energy, peak memory, and peak temperature. You can also compute stats for any `GpuSummary` field with `result.stats_for("field_name")`.

### When to use multi-run

- **Benchmarking**: always use multiple runs to report mean +- std
- **Comparing configurations**: without variance estimates, a 2% difference might be noise
- **Tracking regressions**: compute confidence intervals from multi-run results
- **Short workloads**: more sensitive to startup overhead, so run-to-run variance is higher

### When single run is fine

- **Long training jobs**: variance within the run dominates run-to-run variance
- **Quick debugging**: just checking "is the GPU being used at all?"

## CUDA async scheduling: why `sync_fn` exists

Frameworks like PyTorch and CuPy generally schedule GPU work asynchronously:

- your Python code queues kernels to the GPU
- the CPU thread continues without waiting

That means a function can return before the GPU finishes the work it launched.

### What `sync_fn` does

`profgpu` can call a synchronization function **before and after** the profiled region:

- before: flush previously queued work so you don't "inherit" earlier activity
- after: wait for queued work so the region includes what you launched

For PyTorch, pass:

```python
sync_fn=torch.cuda.synchronize
```

For CuPy, pass:

```python
import cupy as cp
sync_fn = cp.cuda.Stream.null.synchronize
```

If you don't set `sync_fn`, the measurements are still useful for *long-running loops* (where the GPU is active throughout), but can mislead for short regions.

## Warmup

Many workloads have warmup effects:

- CUDA context init
- kernel autotuning
- first batch caches
- JIT compilation (Triton/torch.compile)

There are two levels of warmup in `profgpu`:

1. **`warmup_s`** (per-run): Ignore early samples within a single run. Handles in-run warmup effects.
2. **`warmup_runs`** (multi-run): Discard entire initial runs. Handles cross-run warmup effects like JIT compilation.

```python
# warmup_s: ignore first 0.5s of samples WITHIN each run
@gpu_profile(warmup_s=0.5)

# warmup_runs: discard entire first run, measure the next 5
@gpu_profile(repeats=5, warmup_runs=1)

# combine both for maximum accuracy
@gpu_profile(repeats=5, warmup_runs=1, warmup_s=0.2)
```

## Multi-process / shared GPUs

NVML and `nvidia-smi` utilization counters are **device-level**. If multiple processes use the same GPU, the counter reflects the combined activity.

If you need per-process attribution, you typically need:

- the GPU scheduler/driver metrics (limited)
- DCGM-based accounting
- or application-level instrumentation

`profgpu` intentionally focuses on "is the device busy?" rather than perfect attribution.

## Interpreting common patterns

### Low util.gpu but slow runtime

Common causes:

- input pipeline/data loader bottleneck
- CPU preprocessing dominates
- frequent synchronization points
- too-small batches
- lots of host-to-device transfers

### High util.gpu but low throughput

Common causes:

- memory-bandwidth bound kernels
- kernel launch overhead dominating (many tiny kernels)
- power/thermal throttling (watch `sm_clock_mean_mhz` vs `sm_clock_max_mhz` and `temp_max_c`)

### High idle_pct

If `idle_pct` is high (many samples below 5% util), the GPU is frequently idle. Look for:

- data loading gaps
- synchronization barriers
- CPU bottlenecks between GPU kernels

### Bursty utilization trace

Often indicates:

- intermittent GPU work (e.g. per-batch) with gaps
- CPU/data pipeline not feeding GPU continuously

### High run-to-run variance

If `result.util_gpu.std` is large across multi-run benchmarks:

- thermal throttling may be kicking in on later runs
- OS scheduling is inconsistent
- memory allocator behavior differs between runs
- consider longer warmup_runs

## What `util.mem` is (and isn't)

`util.mem` is typically a memory-controller utilization percentage. It can hint that you are memory-bound, but it is not a direct measure of achieved GB/s.

To measure achieved bandwidth you typically need more detailed counters.

---

If you want to *debug* a pattern you see in the utilization trace, the next step is often to correlate with:

- CPU utilization
- data loader timings
- kernel-level profiling (Nsight)

The [PyTorch tutorial](tutorials/pytorch.md) includes concrete examples.
