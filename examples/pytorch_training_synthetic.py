"""Synthetic PyTorch training loop example.

Run:
  python examples/pytorch_training_synthetic.py

Notes:
- Requires a CUDA-capable PyTorch install.
- Profiles each epoch separately so you can see warmup vs steady-state.
"""

import time

import torch
import torch.nn as nn
import torch.optim as optim

from gpu_profile import GpuMonitor


class SmallMLP(nn.Module):
    def __init__(self, d_in: int = 1024, d_hidden: int = 2048, d_out: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


def train(epochs: int = 3, batches_per_epoch: int = 200, batch_size: int = 256):
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available. Install a CUDA-enabled PyTorch build.")

    device = torch.device("cuda")
    model = SmallMLP().to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        t0 = time.perf_counter()
        with GpuMonitor(interval_s=0.2, sync_fn=torch.cuda.synchronize, warmup_s=0.2) as mon:
            for _ in range(batches_per_epoch):
                x = torch.randn(batch_size, 1024, device=device)
                y = torch.randint(0, 10, (batch_size,), device=device)

                opt.zero_grad(set_to_none=True)
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                opt.step()

        dt = time.perf_counter() - t0
        print(f"epoch {epoch} wall={dt:.3f}s")
        print(mon.summary.format())
        print()


if __name__ == "__main__":
    train()
