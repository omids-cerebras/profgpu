# FAQ

## Does this measure GPU utilization per process?

No. The utilization counters exposed by NVML/`nvidia-smi` are **device-level** and reflect combined activity.

If multiple processes share the same GPU, the reported utilization includes all of them.

If you need per-process accounting, you typically need:

- DCGM-based tooling (cluster monitoring)
- GPU scheduler accounting (limited)
- application-level instrumentation

## What exactly is `util.gpu`?

A good mental model is:

> **`util.gpu` = "% of time the GPU was busy" over a recent sampling window.**

It is not "% of theoretical peak compute," and it does not directly tell you whether you're compute- or memory-bound.

See [Concepts](concepts.md) for details.

## What are all the metrics in GpuSummary?

`GpuSummary` now has 30 fields grouped into: metadata, GPU utilization (mean/std/min/max/p5/p50/p95/p99), idle/active percentages, memory utilization, device memory, power, temperature, clocks, and derived metrics.

See [API reference](api.md#gpusummary) for the complete field list.

## Why do I sometimes see low utilization even though my code uses the GPU?

Common causes:

- **CUDA async scheduling:** your Python region ends before the GPU finishes.
  Fix: pass `sync_fn=torch.cuda.synchronize` (PyTorch) or equivalent.

- **Region too short:** your work finishes before you collect enough samples.
  Fix: profile a longer window (e.g., an epoch), or reduce `interval_s`.

- **Input pipeline bottleneck:** the GPU is waiting on the CPU/I/O.

## How do I choose `interval_s`?

Rules of thumb:

- `0.2s` is a solid default.
- If your utilization is **bursty** or your kernels are very short, try `0.05-0.1s`.
- Avoid extremely small intervals unless you truly need them; sampling too fast can add overhead and noise.

## What's the difference between `repeats` and `warmup_runs`?

- **`repeats`**: how many measured runs to aggregate (the function is run this many times *after* warmup).
- **`warmup_runs`**: how many initial runs to discard (the function is run but results are thrown away).

Total executions = `warmup_runs + repeats`.

Example: `repeats=5, warmup_runs=1` means 6 total executions, 5 measured.

## When should I use multi-run profiling?

Use multi-run (`repeats>1`) when:

- **Benchmarking**: you need mean +- std to report reliable numbers.
- **Comparing configs**: telling apart 2% differences from noise.
- **Short workloads**: run-to-run variance is higher for short tasks.

Single run is fine for:

- Long training jobs where in-run variance dominates.
- Quick debugging ("is the GPU being used at all?").

## What is `RunStats`?

`RunStats` is a frozen dataclass that holds cross-run statistics: `mean`, `std` (sample, N-1), `min`, `max`, and the raw `values` tuple.

`MultiRunResult` has pre-computed `RunStats` for the most common metrics (duration, util_gpu, power, energy, peak_memory, peak_temp), plus a `stats_for(field)` method for any `GpuSummary` field.

## Do I need NVML?

You can run without NVML if `nvidia-smi` is available, but NVML is recommended because:

- lower overhead (no subprocess per sample)
- more robust and faster polling

Install NVML support via:

```bash
pip install profgpu[nvml]
```

## Will this slow down my training?

Usually the overhead is negligible at typical sampling intervals.

- NVML backend: low overhead
- `nvidia-smi` backend: higher overhead (spawns a process each sample)

If you care about overhead:

- use NVML (`profgpu[nvml]`)
- keep `interval_s` reasonable (e.g., 0.1-1.0)
- profile longer regions (epochs) rather than tiny micro-regions

## Can I export a time series, not just aggregates?

Yes. Use the context manager with `store_samples=True` and write `mon.samples` to CSV/JSON.

See [Logging & export](tutorials/logging.md).

## How does this differ from Nsight Systems / Nsight Compute?

- `profgpu` answers: **"Is the device busy, and how does that change over time?"**
- Nsight tools answer: **"Why is this kernel slow?"** (kernel-level counters, stall reasons, timelines)

They are complementary. `profgpu` is typically a first-pass instrumentation step.

## Does it work for multi-GPU?

Yes. Set the `device` parameter (0-based index):

```python
GpuMonitor(device=1)
```

In distributed training, each process usually has one GPU; run `profgpu` per process and log per-rank summaries.

## Does it support AMD/Intel GPUs?

Not yet. The current backends target NVIDIA (NVML / `nvidia-smi`).

The API is backend-agnostic, so AMD/Intel backends can be added.
