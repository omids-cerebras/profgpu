#!/usr/bin/env bash
# ---------------------------------------------------------------
# create_env.sh – Create a self-contained Conda environment for
#                 profgpu development.
#
# Why Conda?
#   The host system may have an outdated GCC / glibc / CUDA toolkit.
#   Conda ships its own compiler toolchain (gcc_linux-64, gxx_linux-64)
#   and a compatible CUDA toolkit, so builds that need a modern compiler
#   (e.g. PyTorch C++ extensions, pynvml) work out of the box.
#
# Usage:
#   bash create_env.sh            # uses defaults
#   ENV_NAME=myenv PYTHON=3.11 bash create_env.sh
#
# After creation:
#   conda activate profgpu   # (or $ENV_NAME)
#   pip install -e ".[dev,nvml]"
#   pytest
# ---------------------------------------------------------------
set -euo pipefail

# ---------- configurable knobs ----------
ENV_NAME="${ENV_NAME:-profgpu}"
PYTHON="${PYTHON:-3.11}"
CUDA_VERSION="${CUDA_VERSION:-12.4}"        # conda-forge CUDA toolkit version
GCC_VERSION="${GCC_VERSION:-13}"            # conda GCC version (major)
# ----------------------------------------

log() { printf "\n>>> %s\n" "$*"; }

# --- 0. Ensure conda / mamba is available ----------------------------
if command -v mamba &>/dev/null; then
    CONDA=mamba          # mamba is a faster drop-in replacement
elif command -v conda &>/dev/null; then
    CONDA=conda
else
    echo "ERROR: Neither conda nor mamba found on PATH."
    echo "Install Miniforge: https://github.com/conda-forge/miniforge"
    exit 1
fi
log "Using package manager: $CONDA"

# --- 1. Create (or recreate) the environment -------------------------
if $CONDA env list | grep -qw "^${ENV_NAME} "; then
    log "Environment '${ENV_NAME}' already exists – removing first."
    $CONDA env remove -n "${ENV_NAME}" -y
fi

log "Creating Conda environment '${ENV_NAME}' (Python ${PYTHON})"
$CONDA create -n "${ENV_NAME}" -y -c conda-forge \
    "python=${PYTHON}" \
    pip \
    setuptools \
    wheel

# --- 2. Install a modern GCC / G++ from conda-forge ------------------
log "Installing GCC ${GCC_VERSION} toolchain"
$CONDA install -n "${ENV_NAME}" -y -c conda-forge \
    "gcc_linux-64>=${GCC_VERSION},<$((GCC_VERSION+1))" \
    "gxx_linux-64>=${GCC_VERSION},<$((GCC_VERSION+1))" \
    "sysroot_linux-64>=2.17"

# --- 3. Install CUDA toolkit (headers + libraries) -------------------
log "Installing CUDA toolkit ${CUDA_VERSION} from conda-forge"
$CONDA install -n "${ENV_NAME}" -y -c conda-forge \
    "cuda-toolkit=${CUDA_VERSION}" \
    "cuda-nvcc"

# --- 4. Activate & pip-install the project in editable mode ----------
log "Activating '${ENV_NAME}' and installing profgpu[dev,nvml]"

# We need to run inside the env; use `conda run` for non-interactive shells.
# Note: older mamba/conda may not support --no-banner, so we omit it.
$CONDA run -n "${ENV_NAME}" \
    pip install -e ".[dev,nvml]"

# --- 5. Install PyTorch (CUDA wheels) --------------------------------
# PyTorch publishes wheels per CUDA version.  cu124 wheels are
# forward-compatible with any CUDA 12.x driver (12.4, 12.6, etc.).
# There is NO cu126 index — use cu124 for all CUDA 12.x.
log "Installing PyTorch (cu124 wheels — compatible with CUDA 12.x drivers)"
$CONDA run -n "${ENV_NAME}" \
    pip install torch --index-url https://download.pytorch.org/whl/cu124

# --- 6. Verify --------------------------------------------------------
log "Verifying installation"
$CONDA run -n "${ENV_NAME}" python -c "
import profgpu
print(f'profgpu {profgpu.__version__} installed OK')
"
$CONDA run -n "${ENV_NAME}" x86_64-conda-linux-gnu-gcc --version | head -1

log "Done!  Activate with:"
echo "  conda activate ${ENV_NAME}"
