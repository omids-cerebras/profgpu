import torch
from gpu_profile import gpu_profile


@gpu_profile(interval_s=0.1, sync_fn=torch.cuda.synchronize, warmup_s=0.2)
def matmul_bench():
    a = torch.randn(4096, 4096, device="cuda")
    b = torch.randn(4096, 4096, device="cuda")
    for _ in range(20):
        _ = a @ b


if __name__ == "__main__":
    matmul_bench()
