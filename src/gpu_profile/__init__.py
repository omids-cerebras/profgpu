"""gpu-profile: programmatic GPU utilization monitoring (decorator + context manager)."""

__version__ = "0.2.0"


from .monitor import (
    GpuBackendError,
    GpuMonitor,
    GpuSample,
    GpuSummary,
    ProfiledResult,
    gpu_profile,
)

__all__ = [
    "GpuBackendError",
    "GpuMonitor",
    "GpuSample",
    "GpuSummary",
    "ProfiledResult",
    "gpu_profile",
]
