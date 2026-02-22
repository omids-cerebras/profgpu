# ---------------------------------------------------------------
# Dockerfile – GPU-ready development image for profgpu.
#
# Uses an NVIDIA CUDA base image so the container ships a known-good
# CUDA toolkit, driver-compat shims, and a modern GCC.  This entirely
# bypasses any old host-side compiler or CUDA version problems.
#
# Build:
#   docker build -t profgpu:latest .
#
# Run interactively (GPU-enabled):
#   docker run --rm -it --gpus all profgpu:latest bash
#
# Run a profiled command directly:
#   docker run --rm --gpus all profgpu:latest \
#       profgpu --interval 0.1 -- python examples/pytorch_example.py
#
# Run tests:
#   docker run --rm profgpu:latest pytest
#
# Notes:
#   • The --gpus flag requires the NVIDIA Container Toolkit:
#     https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/
#   • Change the base image tag to match your driver's CUDA compat
#     level (e.g. 11.8.0 for older drivers).
# ---------------------------------------------------------------

# ---------- build args ----------
ARG CUDA_VERSION=12.4.1
ARG UBUNTU_VERSION=22.04
ARG PYTHON_VERSION=3.11

# ---------- base image ----------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

LABEL maintainer="profgpu contributors"
LABEL description="Development image for the profgpu GPU profiler"

# Avoid interactive prompts during apt installs.
ENV DEBIAN_FRONTEND=noninteractive

# ---------- system dependencies ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
        git \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-distutils \
    && rm -rf /var/lib/apt/lists/*

# Make the desired Python the default.
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python${PYTHON_VERSION} 1

# Install pip.
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3

# ---------- project install ----------
WORKDIR /workspace/profgpu

# Copy only dependency metadata first for better layer caching.
COPY pyproject.toml README.md LICENSE ./

# Install runtime + dev + NVML dependencies (no editable yet because
# source isn't copied; we just want pip to cache the heavy deps).
RUN pip install --no-cache-dir ".[dev,nvml]" || true

# Now copy the full source.
COPY . .

# Editable install so local changes take effect immediately if
# you bind-mount the source at runtime.
RUN pip install --no-cache-dir -e ".[dev,nvml]"

# ---------- smoke test (CPU-only; NVML will fall back gracefully) -----
RUN python -c "import profgpu; print(f'profgpu {profgpu.__version__}')"

# ---------- default entrypoint ----------
# Drop into bash so users can run any command.
CMD ["bash"]
