# Troubleshooting

## “No backend available” / `GpuBackendError`

By default, `gpu-profile` uses `backend="auto"`:

1. try NVML (requires `nvidia-ml-py3`)
2. fall back to `nvidia-smi`

If neither is available, and `strict=True`, you’ll see an error.

### Fix: install NVML support

```bash
pip install gpu-profile[nvml]
```

### Fix: ensure `nvidia-smi` is on PATH

On most Linux systems, `nvidia-smi` ships with NVIDIA drivers. Verify:

```bash
nvidia-smi
```

## NVML errors

### “Driver/library version mismatch”

This typically indicates mismatched driver and user-space libraries (common in containers).

- If `nvidia-smi` fails, NVML will fail too.
- In Docker, ensure you’re using NVIDIA Container Toolkit and that the host driver is correctly mounted.

### Permission errors

NVML usually needs access to `/dev/nvidia*` devices.

- In Docker, run with `--gpus all`.
- On shared clusters, ensure you’re inside an allocated GPU job.

## `util.gpu` is near-zero but you believe you are using the GPU

Common causes:

1. **Your code is CUDA-async and the region is too short.**
   - Solution: pass `sync_fn=torch.cuda.synchronize` (PyTorch) or the equivalent.

2. **Your workload is bursty and your sampling interval is too long.**
   - Solution: use `interval_s=0.05` or `0.1`.

3. **The GPU is actually idle waiting on input.**
   - Solution: profile your input pipeline; measure “data loading” vs “compute” separately.

4. **Another process is using the GPU.**
   - Solution: check `nvidia-smi` process list.

## `util.gpu` is high but throughput is low

Common causes:

- memory-bandwidth bound kernels
- very small kernels (launch overhead)
- power/thermal throttling

Check:

- clocks (SM + memory)
- power draw and temperature

For deeper analysis, use Nsight Systems/Compute.

## WSL notes

WSL2 with NVIDIA GPU support generally provides `nvidia-smi` and NVML, but driver/tooling must be installed correctly.

If `nvidia-smi` works inside WSL, `gpu-profile` should work.

## MIG / multi-instance GPUs

Device indexing and utilization reporting can be different under MIG.

- Always specify the device index you want (`--device` or `device=`).
- If you use MIG partitions, check how your environment exposes them.

---

If you hit an issue not covered here, include:

- `nvidia-smi` output
- whether you installed `gpu-profile[nvml]`
- OS + driver version
- whether you are in Docker/WSL

…and open an issue.
