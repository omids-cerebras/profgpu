# Installation

## Supported platforms

- **Python**: 3.8+
- **GPUs**: NVIDIA (via NVML or `nvidia-smi`)
- **OS**: Linux/WSL/Windows/macOS are all *possible*, but the GPU tooling is driver-dependent.
  - NVML and `nvidia-smi` are typically available on Linux, WSL2 (with NVIDIA drivers), and Windows.
  - On macOS, NVIDIA GPUs/drivers are uncommon.

## Install from GitHub (common for internal tooling)

Once you push this repo to GitHub:

```bash
pip install git+https://github.com/<you>/gpu-profile.git
```

### Recommended: install NVML support

NVML is the lowest overhead backend.

```bash
pip install "gpu-profile[nvml] @ git+https://github.com/<you>/gpu-profile.git"
```

## Install from a wheel (air-gapped / cluster environments)

Build a wheel on a machine that has the repo:

```bash
python -m pip wheel . -w dist
```

Copy the `.whl` to your target environment and install:

```bash
pip install gpu_profile-*.whl
```

If you want NVML support in that target environment, also install:

```bash
pip install nvidia-ml-py3
```

## Local dev install

```bash
pip install -e .[dev]
pytest
```

## Verify installation

### 1) Import the package

```bash
python -c "from gpu_profile import GpuMonitor; print('ok')"
```

### 2) Check the CLI is available

```bash
gpu-profile --help
```

### 3) Quick smoke test (no GPU work required)

On an NVIDIA machine, this should print a summary (values may be low/idle):

```bash
gpu-profile --interval 0.5 -- python -c "import time; time.sleep(2)"
```

## Backend selection

By default, the library uses `backend="auto"`, which means:

1. Try NVML (if `nvidia-ml-py3` is installed and NVML is available)
2. Fall back to `nvidia-smi` (if `nvidia-smi` is on PATH)

You can force a backend:

- `backend="nvml"` — requires NVML
- `backend="smi"` — requires `nvidia-smi`
- `backend="none"` — disables sampling (useful for dry runs)

If you set `strict=True`, missing backend/tooling raises an exception.

## Containers and permissions

- In Docker, you usually need the NVIDIA container runtime (or `--gpus all`).
- NVML may require access to `/dev/nvidia*` devices.
- If `nvidia-smi` works, NVML typically works too.

If you run into issues, see [Troubleshooting](troubleshooting.md).
