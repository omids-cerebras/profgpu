# Tutorial: PyTorch

This tutorial shows practical patterns for profiling PyTorch workloads.

> Key point: **CUDA is asynchronous**. If you want the profiled region to include all queued GPU work, pass `sync_fn=torch.cuda.synchronize`.

## 1) Minimal matmul benchmark

```python
import torch
from gpu_profile import gpu_profile

# Ensures the region includes queued GPU work.
SYNC = torch.cuda.synchronize

@gpu_profile(interval_s=0.1, sync_fn=SYNC, warmup_s=0.2)
def matmul_bench(n: int = 8192, steps: int = 10):
    a = torch.randn(n, n, device="cuda")
    b = torch.randn(n, n, device="cuda")
    for _ in range(steps):
        _ = a @ b

matmul_bench()
```

### Why the warmup?

The first CUDA op often includes:

- CUDA context initialization
- kernel loading and caching
- cuBLAS autotuning

That can distort the summary for short runs. `warmup_s=0.2` ignores the first 200ms of samples.

## 2) Profile a training loop

Here is a small, self-contained training loop using synthetic data.

```python
import torch
import torch.nn as nn
import torch.optim as optim

from gpu_profile import GpuMonitor

SYNC = torch.cuda.synchronize

class SmallMLP(nn.Module):
    def __init__(self, d_in=1024, d_hidden=2048, d_out=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


def train(epochs: int = 3, batches_per_epoch: int = 100, batch_size: int = 256):
    device = "cuda"
    model = SmallMLP().to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        # Monitor one epoch at a time for a clean breakdown.
        with GpuMonitor(interval_s=0.2, sync_fn=SYNC, warmup_s=0.1) as mon:
            for _ in range(batches_per_epoch):
                x = torch.randn(batch_size, 1024, device=device)
                y = torch.randint(0, 10, (batch_size,), device=device)

                opt.zero_grad(set_to_none=True)
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                opt.step()

        print(f"epoch {epoch}:\n{mon.summary.format()}\n")


if __name__ == "__main__":
    assert torch.cuda.is_available()
    train()
```

### Pattern: epoch-level monitoring

Monitoring per-epoch is often the best first step because:

- you get a stable measurement window
- you can spot warmup vs steady-state
- it avoids the overhead of monitoring every step

## 3) Monitor just a step (micro-regions)

If you want to measure a single batch step, keep the sampling interval small and the region long enough to collect a few samples.

For example:

```python
from gpu_profile import GpuMonitor
import torch

SYNC = torch.cuda.synchronize

with GpuMonitor(interval_s=0.05, sync_fn=SYNC) as mon:
    # run several steps to get enough samples
    for _ in range(20):
        ...

print(mon.summary.format())
```

If you measure *one* extremely fast step, you may see little signal simply because the sampling window is coarser than the operation.

## 4) Split “data loading” vs “compute”

A common question is:

> Is my GPU underutilized because my input pipeline is too slow?

You can measure two regions separately:

```python
from gpu_profile import GpuMonitor
import torch

SYNC = torch.cuda.synchronize

# ... inside your loop

with GpuMonitor(interval_s=0.2, sync_fn=SYNC) as data_mon:
    batch = next(loader)          # CPU + I/O

with GpuMonitor(interval_s=0.2, sync_fn=SYNC) as compute_mon:
    batch = batch.to("cuda", non_blocking=True)
    out = model(batch)
    ...

print("data", data_mon.summary.util_gpu_mean)
print("compute", compute_mon.summary.util_gpu_mean)
```

Be careful: moving data to the GPU can be asynchronous too (non-blocking H2D). If you want “transfer time included,” keep `sync_fn`.

## 5) Multi-GPU

If you have multiple GPUs, set the `device` parameter:

```python
with GpuMonitor(device=1, interval_s=0.2, sync_fn=torch.cuda.synchronize) as mon:
    ...
```

For DDP, each process usually uses one GPU. Run the monitor in each process and write JSON summaries to separate files.

## 6) When `util.gpu` looks wrong

If you see unexpectedly low utilization:

- confirm your region includes GPU work (`sync_fn`!)
- increase region duration
- reduce CPU bottlenecks (data loader, preprocessing)
- look at batch size / kernel launch overhead

For deeper analysis, you’ll typically move to Nsight Systems/Compute.

---

See also:

- [Concepts](../concepts.md)
- [Logging & Export tutorial](logging.md)
- `examples/` and `notebooks/` in the repo
