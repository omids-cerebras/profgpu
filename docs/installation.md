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
pip install git+https://github.com/<you>/profgpu.git
```

### Recommended: install NVML support

NVML is the lowest overhead backend.

```bash
pip install "profgpu[nvml] @ git+https://github.com/<you>/profgpu.git"
```

## Install from a wheel (air-gapped / cluster environments)

Build a wheel on a machine that has the repo:

```bash
python -m pip wheel . -w dist
```

Copy the `.whl` to your target environment and install:

```bash
pip install profgpu-*.whl
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

## Conda environment (recommended for old host toolchains)

If your system has an outdated GCC or CUDA, the repo includes
`create_env.sh` which creates a self-contained Conda environment with a
modern GCC toolchain and matching CUDA toolkit:

```bash
bash create_env.sh                # creates 'profgpu'
conda activate profgpu
pytest
```

You can customise the environment name, Python version, CUDA version,
and GCC version via environment variables:

```bash
ENV_NAME=myenv PYTHON=3.11 CUDA_VERSION=12.4 GCC_VERSION=13 bash create_env.sh
```

## Docker

The repo includes a `Dockerfile` using an NVIDIA CUDA base image.
This entirely bypasses host compiler and CUDA compatibility issues:

```bash
docker build -t profgpu:latest .
docker run --rm --gpus all profgpu:latest pytest
docker run --rm -it --gpus all profgpu:latest bash
```

> Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/).

## Verify installation

### 1) Import the package

```bash
python -c "from profgpu import GpuMonitor; print('ok')"
```

### 2) Check the CLI is available

```bash
profgpu --help
```

### 3) Quick smoke test (no GPU work required)

On an NVIDIA machine, this should print a summary (values may be low/idle):

```bash
profgpu --interval 0.5 -- python -c "import time; time.sleep(2)"
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
