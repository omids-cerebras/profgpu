# Concepts

This section explains what `gpu-profile` is (and is **not**) measuring, and why the defaults are the way they are.

## Device-level utilization is a time fraction

Most GPU “utilization” counters reported by NVML / `nvidia-smi` are best interpreted as:

> **The fraction of time the device had at least one active workload in the last sampling window.**

So if you see `util.gpu = 80%`, that often means:

- For ~80% of the last window, there were kernels running (or the graphics engine was active).
- For ~20% of the window, the device was idle.

It does **not** guarantee anything about:

- how close you are to peak FLOPS
- whether kernels are compute-bound vs memory-bound
- whether you are saturating memory bandwidth

To answer “how close am I to the hardware limit?”, you typically need kernel-level profiling tools (Nsight Compute/Systems) or hardware counters beyond util%.

## Why sampling interval matters

A utilization counter is a *statistic over time*. If your workload is bursty (short kernels separated by CPU work or I/O), then:

- With a long interval (e.g. 1s), bursts may be averaged out.
- With a short interval (e.g. 50ms), you can see bursts more clearly.

`gpu-profile` defaults to `interval_s=0.2` (200ms) because it's usually:

- fast enough to show “bursty vs steady” patterns
- slow enough to avoid noticeable overhead in most runs

If you have micro-kernels or extremely bursty patterns, try `interval_s=0.05` or `0.1`.

## The “busy time estimate”

The summary reports:

```
busy_time_est_s = duration_s * mean(util.gpu)/100
```

This is a coarse estimate of *how much of the wall time the GPU was busy*, under the assumption that the utilization counter is representative.

Common interpretations:

- **Busy time is close to duration** → the GPU is likely the bottleneck (or at least continuously active).
- **Busy time is much smaller than duration** → likely CPU/I/O bottlenecks, data loading, synchronization gaps, or very small kernels.

## CUDA async scheduling: why `sync_fn` exists

Frameworks like PyTorch and CuPy generally schedule GPU work asynchronously:

- your Python code queues kernels to the GPU
- the CPU thread continues without waiting

That means a function can return before the GPU finishes the work it launched.

### What `sync_fn` does

`gpu-profile` can call a synchronization function **before and after** the profiled region:

- before: flush previously queued work so you don’t “inherit” earlier activity
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

If you don’t set `sync_fn`, the measurements are still useful for *long-running loops* (where the GPU is active throughout), but can mislead for short regions.

## Warmup

Many workloads have warmup effects:

- CUDA context init
- kernel autotuning
- first batch caches
- JIT compilation (Triton/torch.compile)

Use `warmup_s` to ignore the first part of the run in the summary stats:

```python
GpuMonitor(warmup_s=1.0)
```

## Multi-process / shared GPUs

NVML and `nvidia-smi` utilization counters are **device-level**. If multiple processes use the same GPU, the counter reflects the combined activity.

If you need per-process attribution, you typically need:

- the GPU scheduler/driver metrics (limited)
- DCGM-based accounting
- or application-level instrumentation

`gpu-profile` intentionally focuses on “is the device busy?” rather than perfect attribution.

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
- power/thermal throttling (watch clocks/power/temperature)

### Bursty utilization trace

Often indicates:

- intermittent GPU work (e.g. per-batch) with gaps
- CPU/data pipeline not feeding GPU continuously

## What `util.mem` is (and isn’t)

`util.mem` is typically a memory-controller utilization percentage. It can hint that you are memory-bound, but it is not a direct measure of achieved GB/s.

To measure achieved bandwidth you typically need more detailed counters.

---

If you want to *debug* a pattern you see in the utilization trace, the next step is often to correlate with:

- CPU utilization
- data loader timings
- kernel-level profiling (Nsight)

The [PyTorch tutorial](tutorials/pytorch.md) includes concrete examples.
