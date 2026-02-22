"""Core GPU monitoring engine.

This module contains:

* **Backends** – ``NvidiaNvmlBackend`` (low-overhead, recommended),
  ``NvidiaSmiBackend`` (fallback), ``NullGpuBackend`` (no-op stub).
* **Data model** – :class:`GpuSample`, :class:`GpuSummary`,
  :class:`ProfiledResult`.
* **Public API** – :class:`GpuMonitor` (context manager) and
  :func:`gpu_profile` (decorator).

Thread safety
-------------
Each :class:`GpuMonitor` runs a single daemon thread that calls
``backend.read()`` at a fixed cadence.  The aggregator is updated only
by that thread; the main thread reads the results **after** the thread
has been joined in :meth:`GpuMonitor.stop`.
"""

from __future__ import annotations

import abc
import contextlib
import functools
import math
import random
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _percentile(sorted_vals: Sequence[float], p: float) -> float:
    """Return the *p*-th percentile using linear interpolation.

    Parameters
    ----------
    sorted_vals:
        Values **sorted in ascending order**.  An empty sequence returns
        ``float('nan')``.
    p:
        Percentile in the range [0, 100].  Values outside that range are
        clamped to the minimum / maximum of *sorted_vals*.

    Returns
    -------
    float
        The interpolated percentile value.
    """
    if not sorted_vals:
        return float("nan")
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 100:
        return float(sorted_vals[-1])

    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    return float(sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f))


def _sparkline(values: Sequence[float], width: int = 40) -> str:
    """Render a mini utilization trace using Unicode block characters.

    The input values are down-sampled (nearest-neighbour) to *width*
    characters.  Values are linearly mapped to one of eight vertical
    block elements (``▁`` … ``█``).

    Parameters
    ----------
    values:
        Numeric sequence (typically GPU utilization %).  Empty input → "".
    width:
        Number of output characters.  Must be > 0.

    Returns
    -------
    str
        A fixed-width string suitable for terminal output.
    """
    blocks = "▁▂▃▄▅▆▇█"
    if not values or width <= 0:
        return ""

    # Sample (with nearest-neighbor) to exactly `width` points.
    if len(values) == width:
        sampled = list(values)
    else:
        sampled = []
        for i in range(width):
            idx = int(i * (len(values) - 1) / max(width - 1, 1))
            sampled.append(values[idx])

    vmin = min(sampled)
    vmax = max(sampled)
    if math.isclose(vmin, vmax):
        return (blocks[-1] if vmax > 0 else blocks[0]) * len(sampled)

    out = []
    for v in sampled:
        t = (v - vmin) / (vmax - vmin)  # [0,1]
        j = round(t * (len(blocks) - 1))
        j = max(0, min(len(blocks) - 1, j))
        out.append(blocks[j])
    return "".join(out)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

#: Keys that every backend **must** include in the dict returned by
#: :meth:`BaseGpuBackend.read`.  Missing / unsupported metrics are ``None``.
METRIC_KEYS: tuple[str, ...] = (
    "util_gpu",  # GPU core utilization (%)
    "util_mem",  # Memory-controller utilization (%)
    "mem_used_mb",  # Allocated device memory (MiB)
    "mem_total_mb",  # Total device memory (MiB)
    "power_w",  # Board power draw (W)
    "temp_c",  # GPU die temperature (°C)
    "sm_clock_mhz",  # Current SM clock frequency (MHz)
    "mem_clock_mhz",  # Current memory clock frequency (MHz)
)


class GpuBackendError(RuntimeError):
    """Raised when a backend cannot be initialised or a sample fails."""


class BaseGpuBackend(abc.ABC):
    """Abstract base for all GPU sampling backends.

    Subclasses must implement :meth:`device_name` and :meth:`read`.
    Override :meth:`close` if the backend holds resources that need cleanup.
    """

    @abc.abstractmethod
    def device_name(self) -> str:
        """Return a human-readable name for the monitored GPU."""

    @abc.abstractmethod
    def read(self) -> Dict[str, Optional[float]]:
        """Sample current GPU metrics.

        Returns a ``dict`` whose keys are :data:`METRIC_KEYS`.  Metrics
        that are unavailable or unsupported should be ``None``.
        """

    def close(self) -> None:  # noqa: B027 – intentionally empty
        """Release any resources held by the backend (optional)."""


# NVML refcount so multiple monitors don't fight over init/shutdown.
_nvml_lock = threading.Lock()
_nvml_refcount = 0
_nvml_mod = None


def _nvml_init(pynvml_mod) -> None:
    global _nvml_refcount, _nvml_mod
    with _nvml_lock:
        if _nvml_refcount == 0:
            pynvml_mod.nvmlInit()
            _nvml_mod = pynvml_mod
        _nvml_refcount += 1


def _nvml_shutdown() -> None:
    global _nvml_refcount, _nvml_mod
    with _nvml_lock:
        if _nvml_refcount <= 0:
            return
        _nvml_refcount -= 1
        if _nvml_refcount == 0 and _nvml_mod is not None:
            try:
                _nvml_mod.nvmlShutdown()
            finally:
                _nvml_mod = None


class NvidiaNvmlBackend(BaseGpuBackend):
    """
    Best option when available: low overhead, avoids spawning processes per sample.

    Requires:
      pip install nvidia-ml-py3
    """

    def __init__(self, device: int = 0):
        try:
            import pynvml  # type: ignore
        except Exception as e:
            raise GpuBackendError(
                "NVML backend requested but pynvml not available. "
                "Install with: pip install nvidia-ml-py3"
            ) from e

        self._pynvml = pynvml
        self._device = int(device)

        _nvml_init(self._pynvml)
        try:
            self._handle = self._pynvml.nvmlDeviceGetHandleByIndex(self._device)
            name = self._pynvml.nvmlDeviceGetName(self._handle)
            self._name = name.decode() if isinstance(name, (bytes, bytearray)) else str(name)
        except Exception:
            _nvml_shutdown()
            raise

    def device_name(self) -> str:
        return self._name

    def _safe(self, fn, default=None):
        try:
            return fn()
        except Exception:
            return default

    def read(self) -> Dict[str, Optional[float]]:
        p = self._pynvml
        h = self._handle

        out: Dict[str, Optional[float]] = {
            "util_gpu": None,
            "util_mem": None,
            "mem_used_mb": None,
            "mem_total_mb": None,
            "power_w": None,
            "temp_c": None,
            "sm_clock_mhz": None,
            "mem_clock_mhz": None,
        }

        util = self._safe(lambda: p.nvmlDeviceGetUtilizationRates(h))
        if util is not None:
            out["util_gpu"] = float(getattr(util, "gpu", 0.0))
            out["util_mem"] = float(getattr(util, "memory", 0.0))

        mem = self._safe(lambda: p.nvmlDeviceGetMemoryInfo(h))
        if mem is not None:
            out["mem_used_mb"] = float(mem.used) / (1024**2)
            out["mem_total_mb"] = float(mem.total) / (1024**2)

        power_mw = self._safe(lambda: p.nvmlDeviceGetPowerUsage(h))
        if power_mw is not None:
            out["power_w"] = float(power_mw) / 1000.0

        temp = self._safe(lambda: p.nvmlDeviceGetTemperature(h, p.NVML_TEMPERATURE_GPU))
        if temp is not None:
            out["temp_c"] = float(temp)

        sm_clock = self._safe(lambda: p.nvmlDeviceGetClockInfo(h, p.NVML_CLOCK_SM))
        if sm_clock is not None:
            out["sm_clock_mhz"] = float(sm_clock)

        mem_clock = self._safe(lambda: p.nvmlDeviceGetClockInfo(h, p.NVML_CLOCK_MEM))
        if mem_clock is not None:
            out["mem_clock_mhz"] = float(mem_clock)

        return out

    def close(self) -> None:
        _nvml_shutdown()


class NvidiaSmiBackend(BaseGpuBackend):
    """Fallback backend that shells out to ``nvidia-smi`` once per sample.

    This works on any system where the ``nvidia-smi`` binary is on ``$PATH``
    but is considerably heavier than :class:`NvidiaNvmlBackend` because it
    spawns a subprocess for every sample.
    """

    def __init__(self, device: int = 0):
        if shutil.which("nvidia-smi") is None:
            raise GpuBackendError("nvidia-smi not found on PATH.")
        self._device = int(device)
        self._name = f"NVIDIA GPU {self._device}"

        try:
            cmd = [
                "nvidia-smi",
                "-i",
                str(self._device),
                "--query-gpu=name",
                "--format=csv,noheader",
            ]
            name = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
            if name:
                self._name = name
        except Exception:
            pass

        self._fields = [
            "utilization.gpu",
            "utilization.memory",
            "memory.used",
            "memory.total",
            "power.draw",
            "temperature.gpu",
            "clocks.sm",
            "clocks.mem",
        ]

    def device_name(self) -> str:
        return self._name

    def read(self) -> Dict[str, Optional[float]]:
        cmd = [
            "nvidia-smi",
            "-i",
            str(self._device),
            f"--query-gpu={','.join(self._fields)}",
            "--format=csv,noheader,nounits",
        ]
        try:
            line = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
        except Exception as e:
            raise GpuBackendError("nvidia-smi query failed") from e

        parts = [p.strip() for p in line.split(",")]

        def parse(i: int) -> Optional[float]:
            if i >= len(parts):
                return None
            s = parts[i]
            if not s or s.upper() == "N/A":
                return None
            try:
                return float(s)
            except ValueError:
                return None

        return {
            "util_gpu": parse(0),
            "util_mem": parse(1),
            "mem_used_mb": parse(2),
            "mem_total_mb": parse(3),
            "power_w": parse(4),
            "temp_c": parse(5),
            "sm_clock_mhz": parse(6),
            "mem_clock_mhz": parse(7),
        }


class NullGpuBackend(BaseGpuBackend):
    """No-op backend that returns ``None`` for every metric.

    Used automatically when ``backend='none'`` is requested, or when
    ``strict=False`` and no real backend could be initialised.  The
    :attr:`reason` attribute records *why* no real backend was used.
    """

    def __init__(self, reason: str):
        self.reason = reason

    def device_name(self) -> str:
        return "N/A"

    def read(self) -> Dict[str, Optional[float]]:
        return {k: None for k in METRIC_KEYS}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GpuSample:
    """A single point-in-time GPU measurement.

    All metric fields are ``Optional[float]`` because not every backend
    can report every metric (or the GPU may not support it).

    Attributes
    ----------
    t_s:
        Elapsed wall-clock seconds since monitoring started.
    util_gpu:
        GPU core utilization (0–100 %).
    util_mem:
        Memory controller utilization (0–100 %).
    mem_used_mb:
        Allocated GPU memory (MiB).
    mem_total_mb:
        Total GPU memory (MiB).
    power_w:
        Board power draw (W).
    temp_c:
        GPU die temperature (°C).
    sm_clock_mhz:
        Current SM clock frequency (MHz).
    mem_clock_mhz:
        Current memory clock frequency (MHz).
    """

    t_s: float
    util_gpu: Optional[float]
    util_mem: Optional[float]
    mem_used_mb: Optional[float]
    mem_total_mb: Optional[float]
    power_w: Optional[float]
    temp_c: Optional[float]
    sm_clock_mhz: Optional[float]
    mem_clock_mhz: Optional[float]


@dataclass(frozen=True)
class GpuSummary:
    """Aggregated statistics produced after a monitoring session.

    This is the primary output of :meth:`GpuMonitor.stop` (or equivalently
    ``mon.summary`` after the context manager exits).  It includes mean,
    percentile, and max aggregates for GPU utilisation, memory, power,
    temperature, and clocks.

    All fields are required.  Metrics that were never observed are
    ``float('nan')``.
    """

    # --- Metadata ---
    device: int
    name: str
    duration_s: float
    interval_s: float
    n_samples: int

    # --- GPU core utilization ---
    util_gpu_mean: float
    util_gpu_std: float
    util_gpu_min: float
    util_gpu_max: float
    util_gpu_p5: float
    util_gpu_p50: float
    util_gpu_p95: float
    util_gpu_p99: float
    idle_pct: float  # % of samples where util_gpu < 5 %
    active_pct: float  # % of samples where util_gpu >= 50 %

    # --- Memory-controller utilization ---
    util_mem_mean: float

    # --- Device memory ---
    mem_used_mean_mb: float
    mem_used_max_mb: float
    mem_total_mb: float
    mem_util_pct: float  # peak mem_used_max as % of mem_total

    # --- Power ---
    power_mean_w: float
    power_max_w: float
    energy_j: float

    # --- Temperature ---
    temp_mean_c: float
    temp_max_c: float

    # --- Clocks ---
    sm_clock_mean_mhz: float
    sm_clock_max_mhz: float

    # --- Derived ---
    busy_time_est_s: float
    sparkline: str
    notes: str

    def format(self) -> str:
        """Return a human-readable multi-line summary string.

        Suitable for printing to the terminal.  Numeric values that are
        ``NaN`` (i.e. the metric was never observed) render as ``n/a``.
        """

        def f(x: float, unit: str = "", digits: int = 1) -> str:
            if x is None or (isinstance(x, float) and math.isnan(x)):
                return "n/a"
            return f"{x:.{digits}f}{unit}"

        lines: List[str] = []
        lines.append(f"[GPU {self.device}] {self.name}")
        lines.append(
            f"  duration: {f(self.duration_s, 's', 3)} | samples: {self.n_samples}"
            f" @ {f(self.interval_s, 's', 3)}"
        )

        # GPU utilization — core stats
        lines.append(
            "  util.gpu: "
            f"mean {f(self.util_gpu_mean, '%', 1)} | std {f(self.util_gpu_std, '%', 1)}"
            f" | min {f(self.util_gpu_min, '%', 1)} | max {f(self.util_gpu_max, '%', 1)}"
        )
        lines.append(
            f"            p5 {f(self.util_gpu_p5, '%', 1)} | p50 {f(self.util_gpu_p50, '%', 1)}"
            f" | p95 {f(self.util_gpu_p95, '%', 1)} | p99 {f(self.util_gpu_p99, '%', 1)}"
        )
        lines.append(
            f"            idle (<5%) {f(self.idle_pct, '%', 1)}"
            f" | active (≥50%) {f(self.active_pct, '%', 1)}"
        )

        # Memory-controller utilization
        lines.append(f"  util.mem: mean {f(self.util_mem_mean, '%', 1)}")

        # Device memory (skip entire line if all memory metrics are NaN)
        has_mem = (
            not math.isnan(self.mem_used_mean_mb)
            or not math.isnan(self.mem_used_max_mb)
            or not math.isnan(self.mem_total_mb)
        )
        if has_mem:
            mem_parts = f"  memory: mean {f(self.mem_used_mean_mb, ' MB', 0)} | max {f(self.mem_used_max_mb, ' MB', 0)}"
            if not math.isnan(self.mem_total_mb):
                mem_parts += f" / total {f(self.mem_total_mb, ' MB', 0)}"
                if not math.isnan(self.mem_util_pct):
                    mem_parts += f" ({f(self.mem_util_pct, '%', 1)} peak)"
            lines.append(mem_parts)

        # Power & energy
        power_line = (
            f"  power: mean {f(self.power_mean_w, ' W', 1)} | max {f(self.power_max_w, ' W', 1)}"
        )
        if not math.isnan(self.energy_j):
            power_line += f" | energy {f(self.energy_j, ' J', 1)}"
        lines.append(power_line)

        # Clocks
        if not math.isnan(self.sm_clock_mean_mhz):
            clock_line = f"  clocks: SM mean {f(self.sm_clock_mean_mhz, ' MHz', 0)}"
            if not math.isnan(self.sm_clock_max_mhz):
                clock_line += f" (max {f(self.sm_clock_max_mhz, ' MHz', 0)})"
            lines.append(clock_line)

        # Temperature
        temp_line = f"  temp: mean {f(self.temp_mean_c, ' °C', 0)}"
        temp_line += f" | max {f(self.temp_max_c, ' °C', 0)}"
        lines.append(temp_line)

        lines.append(f"  busy time (est): {f(self.busy_time_est_s, 's', 3)}")
        if self.sparkline:
            lines.append(f"  util trace: {self.sparkline}")
        if self.notes:
            lines.append(f"  notes: {self.notes}")
        return "\n".join(lines)


@dataclass(frozen=True)
class ProfiledResult:
    """Wrapper returned by ``@gpu_profile(return_profile=True)`` for a single run.

    Attributes
    ----------
    value:
        The original return value of the decorated function.
    gpu:
        The :class:`GpuSummary` captured while the function ran.
    """

    value: Any
    gpu: GpuSummary


@dataclass(frozen=True)
class RunStats:
    """Descriptive statistics computed across multiple profiling runs.

    Use this to assess whether a GPU metric is stable across runs or
    varies significantly (high ``std``).
    """

    mean: float
    std: float
    min: float
    max: float
    values: Tuple[float, ...]

    def format(self, unit: str = "", digits: int = 1) -> str:
        """One-line summary like ``87.3% ± 1.2%  (range 85.6–89.1%)``."""

        def _f(x: float) -> str:
            if math.isnan(x):
                return "n/a"
            return f"{x:.{digits}f}{unit}"

        parts = f"{_f(self.mean)} ± {_f(self.std)}"
        parts += f"  (range {_f(self.min)}-{_f(self.max)})"
        return parts


def _compute_run_stats(values: Sequence[float]) -> RunStats:
    """Build a :class:`RunStats` from a sequence of per-run measurements."""
    clean = [v for v in values if not math.isnan(v)]
    tup = tuple(values)
    if not clean:
        return RunStats(
            mean=float("nan"),
            std=float("nan"),
            min=float("nan"),
            max=float("nan"),
            values=tup,
        )
    n = len(clean)
    mean_val = sum(clean) / n
    if n >= 2:
        variance = sum((x - mean_val) ** 2 for x in clean) / (n - 1)
        std_val = math.sqrt(variance)
    else:
        std_val = float("nan")
    return RunStats(
        mean=mean_val,
        std=std_val,
        min=min(clean),
        max=max(clean),
        values=tup,
    )


@dataclass(frozen=True)
class MultiRunResult:
    """Aggregated result from running a profiled workload multiple times.

    Attributes
    ----------
    value:
        Return value of the **last** run.
    runs:
        Individual :class:`GpuSummary` for each counted run (warmup
        runs excluded).
    duration:
        Cross-run :class:`RunStats` of ``duration_s``.
    util_gpu:
        Cross-run :class:`RunStats` of ``util_gpu_mean``.
    power:
        Cross-run :class:`RunStats` of ``power_mean_w``.
    energy:
        Cross-run :class:`RunStats` of ``energy_j``.
    peak_memory:
        Cross-run :class:`RunStats` of ``mem_used_max_mb``.
    peak_temp:
        Cross-run :class:`RunStats` of ``temp_max_c``.
    """

    value: Any
    runs: Tuple[GpuSummary, ...]

    # Pre-computed cross-run statistics for the most-used metrics
    duration: RunStats
    util_gpu: RunStats
    power: RunStats
    energy: RunStats
    peak_memory: RunStats
    peak_temp: RunStats

    @staticmethod
    def from_runs(runs: Sequence[GpuSummary], value: Any) -> MultiRunResult:
        """Construct from a list of per-run summaries."""
        tup = tuple(runs)
        return MultiRunResult(
            value=value,
            runs=tup,
            duration=_compute_run_stats([r.duration_s for r in tup]),
            util_gpu=_compute_run_stats([r.util_gpu_mean for r in tup]),
            power=_compute_run_stats([r.power_mean_w for r in tup]),
            energy=_compute_run_stats([r.energy_j for r in tup]),
            peak_memory=_compute_run_stats([r.mem_used_max_mb for r in tup]),
            peak_temp=_compute_run_stats([r.temp_max_c for r in tup]),
        )

    def stats_for(self, field: str) -> RunStats:
        """Compute :class:`RunStats` for any numeric :class:`GpuSummary` field."""
        vals = [getattr(r, field) for r in self.runs]
        return _compute_run_stats(vals)

    def format(self) -> str:
        """Return a human-readable multi-run summary string."""
        if not self.runs:
            return "(no runs)"
        first = self.runs[0]
        lines: List[str] = []
        lines.append(f"=== Multi-Run GPU Profile ({len(self.runs)} runs) ===")
        lines.append(f"[GPU {first.device}] {first.name}")
        lines.append(f"  duration:  {self.duration.format('s', 3)}")
        lines.append(f"  util.gpu:  {self.util_gpu.format('%', 1)}")
        lines.append(f"  power:     {self.power.format(' W', 1)}")
        lines.append(f"  energy:    {self.energy.format(' J', 1)}")
        lines.append(f"  peak mem:  {self.peak_memory.format(' MB', 0)}")
        lines.append(f"  peak temp: {self.peak_temp.format(' °C', 0)}")
        lines.append("")
        lines.append("--- Per-Run util.gpu mean ---")
        for i, r in enumerate(self.runs, 1):
            u = r.util_gpu_mean
            u_str = f"{u:.1f}%" if not math.isnan(u) else "n/a"
            lines.append(f"  run {i}: {u_str}  ({r.duration_s:.3f}s)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Streaming aggregation
# ---------------------------------------------------------------------------


class _Aggregator:
    """Online statistics accumulator for GPU samples.

    Designed to run inside the sampling thread.  All operations are O(1)
    amortised per sample:

    * **Means** are kept as running sums + counts.
    * **Maxima** are maintained incrementally.
    * **Percentiles** are approximated via *reservoir sampling* so we
      never need to store the full history.
    * **Sparkline** trace is kept as a compressed ring-buffer
      (pair-wise averaging when the buffer overflows).
    """

    def __init__(self, *, warmup_s: float, reservoir_size: int, trace_len: int) -> None:
        """Initialise the aggregator.

        Parameters
        ----------
        warmup_s:
            Ignore samples whose ``t_s`` is less than this value.
        reservoir_size:
            Maximum number of ``util_gpu`` values to keep for percentile
            estimation (Algorithm R reservoir sampling).
        trace_len:
            Maximum length of the compressed utilisation trace used for
            the sparkline.
        """
        self.warmup_s = float(max(0.0, warmup_s))
        self.reservoir_size = int(max(16, reservoir_size))
        self.trace_len = int(max(40, trace_len))

        self.n_samples = 0

        # --- GPU utilization ---
        self._util_gpu_sum = 0.0
        self._util_gpu_sq_sum = 0.0  # for std deviation
        self._util_gpu_n = 0
        self._idle_count = 0  # samples where util_gpu < 5%
        self._active_count = 0  # samples where util_gpu >= 50%

        # --- Memory-controller utilization ---
        self._util_mem_sum = 0.0
        self._util_mem_n = 0

        # --- Power ---
        self._power_sum = 0.0
        self._power_n = 0

        # --- SM clocks ---
        self._sm_clock_sum = 0.0
        self._sm_clock_n = 0
        self.sm_clock_max_mhz = float("nan")

        # --- Memory usage ---
        self._mem_used_sum = 0.0
        self._mem_used_n = 0

        # --- Temperature ---
        self._temp_sum = 0.0
        self._temp_n = 0

        # --- Max trackers ---
        self.util_gpu_max = float("nan")
        self.util_gpu_min = float("nan")
        self.mem_used_max_mb = float("nan")
        self.mem_total_mb = float("nan")
        self.power_max_w = float("nan")
        self.temp_max_c = float("nan")

        # For percentile approximation (reservoir sample)
        self._util_gpu_seen = 0
        self._util_gpu_reservoir: List[float] = []

        # For sparkline (compressed history)
        self._util_trace: List[float] = []

    def _trace_append(self, v: float) -> None:
        """Append *v* to the sparkline trace, compressing when full."""
        self._util_trace.append(v)
        # Compress by averaging pairs until we fit.
        while len(self._util_trace) > self.trace_len:
            it = iter(self._util_trace)
            new: List[float] = []
            for a in it:
                try:
                    b = next(it)
                    new.append((a + b) / 2.0)
                except StopIteration:
                    new.append(a)
            self._util_trace = new

    def _reservoir_add(self, v: float) -> None:
        """Insert *v* into the reservoir using Algorithm R sampling."""
        self._util_gpu_seen += 1
        if len(self._util_gpu_reservoir) < self.reservoir_size:
            self._util_gpu_reservoir.append(v)
            return
        j = random.randrange(self._util_gpu_seen)
        if j < self.reservoir_size:
            self._util_gpu_reservoir[j] = v

    def add(self, t_s: float, metrics: Dict[str, Optional[float]]) -> None:
        """Ingest a single sample at time *t_s* seconds.

        Samples before :attr:`warmup_s` are silently discarded.
        """
        if t_s < self.warmup_s:
            return

        self.n_samples += 1

        util_gpu = metrics.get("util_gpu")
        if util_gpu is not None:
            v = float(util_gpu)
            self._util_gpu_sum += v
            self._util_gpu_sq_sum += v * v
            self._util_gpu_n += 1
            self.util_gpu_max = v if math.isnan(self.util_gpu_max) else max(self.util_gpu_max, v)
            self.util_gpu_min = v if math.isnan(self.util_gpu_min) else min(self.util_gpu_min, v)
            if v < 5.0:
                self._idle_count += 1
            if v >= 50.0:
                self._active_count += 1
            self._reservoir_add(v)
            self._trace_append(v)

        util_mem = metrics.get("util_mem")
        if util_mem is not None:
            self._util_mem_sum += float(util_mem)
            self._util_mem_n += 1

        mem_used = metrics.get("mem_used_mb")
        if mem_used is not None:
            v = float(mem_used)
            self._mem_used_sum += v
            self._mem_used_n += 1
            self.mem_used_max_mb = (
                v if math.isnan(self.mem_used_max_mb) else max(self.mem_used_max_mb, v)
            )

        mem_total = metrics.get("mem_total_mb")
        if mem_total is not None:
            self.mem_total_mb = float(mem_total)

        power = metrics.get("power_w")
        if power is not None:
            v = float(power)
            self._power_sum += v
            self._power_n += 1
            self.power_max_w = v if math.isnan(self.power_max_w) else max(self.power_max_w, v)

        temp = metrics.get("temp_c")
        if temp is not None:
            v = float(temp)
            self._temp_sum += v
            self._temp_n += 1
            self.temp_max_c = v if math.isnan(self.temp_max_c) else max(self.temp_max_c, v)

        sm_clock = metrics.get("sm_clock_mhz")
        if sm_clock is not None:
            v = float(sm_clock)
            self._sm_clock_sum += v
            self._sm_clock_n += 1
            self.sm_clock_max_mhz = (
                v if math.isnan(self.sm_clock_max_mhz) else max(self.sm_clock_max_mhz, v)
            )

    def util_gpu_mean(self) -> float:
        """Return the arithmetic mean of GPU utilisation, or ``NaN``."""
        return (self._util_gpu_sum / self._util_gpu_n) if self._util_gpu_n else float("nan")

    def util_gpu_std(self) -> float:
        """Return the population standard deviation of GPU utilisation, or ``NaN``."""
        if self._util_gpu_n < 2:
            return float("nan")
        mean = self._util_gpu_sum / self._util_gpu_n
        variance = (self._util_gpu_sq_sum / self._util_gpu_n) - (mean * mean)
        return math.sqrt(max(0.0, variance))

    def idle_pct(self) -> float:
        """Return % of samples where GPU utilisation was < 5%."""
        if self._util_gpu_n == 0:
            return float("nan")
        return 100.0 * self._idle_count / self._util_gpu_n

    def active_pct(self) -> float:
        """Return % of samples where GPU utilisation was >= 50%."""
        if self._util_gpu_n == 0:
            return float("nan")
        return 100.0 * self._active_count / self._util_gpu_n

    def util_mem_mean(self) -> float:
        """Return the arithmetic mean of memory-controller utilisation, or ``NaN``."""
        return (self._util_mem_sum / self._util_mem_n) if self._util_mem_n else float("nan")

    def mem_used_mean_mb(self) -> float:
        """Return the arithmetic mean of memory used (MB), or ``NaN``."""
        return (self._mem_used_sum / self._mem_used_n) if self._mem_used_n else float("nan")

    def power_mean_w(self) -> float:
        """Return the arithmetic mean of power draw (watts), or ``NaN``."""
        return (self._power_sum / self._power_n) if self._power_n else float("nan")

    def sm_clock_mean_mhz(self) -> float:
        """Return the arithmetic mean SM clock frequency (MHz), or ``NaN``."""
        return (self._sm_clock_sum / self._sm_clock_n) if self._sm_clock_n else float("nan")

    def temp_mean_c(self) -> float:
        """Return the arithmetic mean GPU temperature (°C), or ``NaN``."""
        return (self._temp_sum / self._temp_n) if self._temp_n else float("nan")

    def util_gpu_quantiles(self) -> Dict[str, float]:
        """Return approximate p5, p50, p95, p99 of GPU utilisation from the reservoir."""
        if not self._util_gpu_reservoir:
            return {
                "p5": float("nan"),
                "p50": float("nan"),
                "p95": float("nan"),
                "p99": float("nan"),
            }
        vals = sorted(self._util_gpu_reservoir)
        return {
            "p5": _percentile(vals, 5),
            "p50": _percentile(vals, 50),
            "p95": _percentile(vals, 95),
            "p99": _percentile(vals, 99),
        }

    def sparkline(self, width: int = 40) -> str:
        """Return a sparkline string of the compressed utilisation trace."""
        if not self._util_trace:
            return ""
        return _sparkline(self._util_trace, width=width)


# ---------------------------------------------------------------------------
# Public API: monitor + decorator
# ---------------------------------------------------------------------------


class GpuMonitor:
    """Context manager that samples GPU metrics in a background thread.

    Usage::

        with GpuMonitor(device=0, interval_s=0.2) as mon:
            run_training()
        print(mon.summary.format())

    If your GPU framework dispatches work asynchronously (e.g. PyTorch
    CUDA streams), pass a *sync_fn* (e.g. ``torch.cuda.synchronize``)
    so the monitored region matches actual GPU work.

    Parameters
    ----------
    device:
        GPU index (0-based).
    interval_s:
        Sampling interval in seconds.  Lower values give finer
        resolution but slightly higher CPU overhead.
    backend:
        ``"auto"`` tries NVML then nvidia-smi.  ``"nvml"`` or ``"smi"``
        force a specific backend.  ``"none"`` disables real sampling.
    strict:
        If ``True``, raise :class:`GpuBackendError` when no backend can
        be initialised or when a sample fails.
    sync_fn:
        Called on ``__enter__`` and ``__exit__`` to synchronise
        asynchronous GPU work (e.g. ``torch.cuda.synchronize``).
    warmup_s:
        Ignore samples taken before this many seconds have elapsed.
    store_samples:
        If ``True``, keep raw :class:`GpuSample` objects (accessible
        via :attr:`samples`).  Automatically down-samples when
        *max_samples* is exceeded.
    max_samples:
        Cap on stored samples before automatic 2× down-sampling.
    reservoir_size:
        Size of the reservoir used for approximate percentiles.
    trace_len:
        Length of the compressed utilisation trace (sparkline).
    """

    def __init__(
        self,
        device: int = 0,
        interval_s: float = 0.2,
        backend: str = "auto",  # "auto" | "nvml" | "smi" | "none"
        strict: bool = False,
        sync_fn: Optional[Callable[[], None]] = None,
        warmup_s: float = 0.0,
        store_samples: bool = False,
        max_samples: int = 50_000,
        reservoir_size: int = 4096,
        trace_len: int = 200,
    ):
        if interval_s <= 0:
            raise ValueError("interval_s must be > 0")
        if max_samples < 100 and store_samples:
            raise ValueError("max_samples too small; set >= 100 or store_samples=False")

        self.device = int(device)
        self.interval_s = float(interval_s)
        self.backend = backend.lower()
        self.strict = bool(strict)
        self.sync_fn = sync_fn
        self.warmup_s = float(max(0.0, warmup_s))

        self.store_samples = bool(store_samples)
        self.max_samples = int(max_samples)

        self._backend_obj: BaseGpuBackend = self._make_backend()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0: Optional[float] = None
        self._t1: Optional[float] = None

        self._agg = _Aggregator(
            warmup_s=self.warmup_s, reservoir_size=reservoir_size, trace_len=trace_len
        )
        self._samples: List[GpuSample] = []

        self._thread_error: Optional[BaseException] = None
        self._thread_error_msg: str = ""

        self.summary: Optional[GpuSummary] = None

    def _make_backend(self) -> BaseGpuBackend:
        """Instantiate the requested (or auto-detected) sampling backend."""
        errors: List[str] = []

        if self.backend in ("auto", "nvml"):
            try:
                return NvidiaNvmlBackend(self.device)
            except Exception as e:
                errors.append(f"NVML: {e}")
                if self.backend == "nvml" and self.strict:
                    raise

        if self.backend in ("auto", "smi"):
            try:
                return NvidiaSmiBackend(self.device)
            except Exception as e:
                errors.append(f"nvidia-smi: {e}")
                if self.backend == "smi" and self.strict:
                    raise

        if self.backend == "none":
            return NullGpuBackend("Backend disabled (backend='none').")

        if self.backend not in ("auto", "nvml", "smi", "none"):
            raise ValueError("backend must be one of: auto, nvml, smi, none")

        reason = " | ".join(errors) if errors else "No backend selected."
        if self.strict:
            raise GpuBackendError(f"Could not initialize a GPU backend. Details: {reason}")
        return NullGpuBackend(reason)

    def __enter__(self) -> GpuMonitor:
        """Start the background sampling thread."""
        # Optional sync to exclude earlier queued work
        if self.sync_fn is not None:
            with contextlib.suppress(Exception):
                self.sync_fn()

        self._t0 = time.perf_counter()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="GpuMonitor", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Sync (if applicable), stop sampling, and compute the summary."""
        # Optional sync so the block includes queued async GPU work
        if self.sync_fn is not None:
            with contextlib.suppress(Exception):
                self.sync_fn()

        self.stop()
        return None

    def _run(self) -> None:
        """Sampling loop executed inside the daemon thread."""
        assert self._t0 is not None
        t0 = self._t0

        while not self._stop.is_set():
            t = time.perf_counter() - t0
            try:
                m = self._backend_obj.read()
            except BaseException as e:
                self._thread_error = e
                self._thread_error_msg = str(e)
                # Stop sampling; caller decides whether it's fatal.
                self._stop.set()
                break

            # Aggregate for summary (warmup handled inside aggregator)
            self._agg.add(t, m)

            # Optionally store samples for external inspection/plotting.
            if self.store_samples:
                self._samples.append(
                    GpuSample(
                        t_s=t,
                        util_gpu=m.get("util_gpu"),
                        util_mem=m.get("util_mem"),
                        mem_used_mb=m.get("mem_used_mb"),
                        mem_total_mb=m.get("mem_total_mb"),
                        power_w=m.get("power_w"),
                        temp_c=m.get("temp_c"),
                        sm_clock_mhz=m.get("sm_clock_mhz"),
                        mem_clock_mhz=m.get("mem_clock_mhz"),
                    )
                )
                if len(self._samples) > self.max_samples:
                    # Downsample in-place: keep every other sample.
                    self._samples = self._samples[::2]

            self._stop.wait(self.interval_s)

    @property
    def samples(self) -> List[GpuSample]:
        """Raw samples (only populated if store_samples=True)."""
        return list(self._samples)

    def stop(self) -> GpuSummary:
        """Stop sampling and return the computed :class:`GpuSummary`.

        Safe to call multiple times; subsequent calls return the cached
        summary.

        Raises
        ------
        GpuBackendError
            If ``strict=True`` and the sampling thread encountered an error.
        """
        if self._t1 is None:
            self._t1 = time.perf_counter()

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 5 * self.interval_s))

        with contextlib.suppress(Exception):
            self._backend_obj.close()

        self.summary = self._summarize()

        # Surface thread error after producing a best-effort summary.
        if self._thread_error is not None and self.strict:
            raise GpuBackendError(
                f"GPU sampling failed: {self._thread_error_msg}"
            ) from self._thread_error

        return self.summary

    def _summarize(self) -> GpuSummary:
        """Build a :class:`GpuSummary` from the current aggregator state."""
        name = self._backend_obj.device_name()
        duration = max(0.0, (self._t1 or time.perf_counter()) - (self._t0 or 0.0))

        # Effective window after warmup
        effective_duration = max(0.0, duration - self.warmup_s)

        util_mean = self._agg.util_gpu_mean()
        util_q = self._agg.util_gpu_quantiles()
        util_p5 = util_q["p5"]
        util_p50 = util_q["p50"]
        util_p95 = util_q["p95"]
        util_p99 = util_q["p99"]

        busy_est = float("nan")
        if not math.isnan(util_mean):
            busy_est = effective_duration * (util_mean / 100.0)

        # Memory utilization as % of total
        mem_util = float("nan")
        if (
            not math.isnan(self._agg.mem_used_max_mb)
            and not math.isnan(self._agg.mem_total_mb)
            and self._agg.mem_total_mb > 0
        ):
            mem_util = 100.0 * self._agg.mem_used_max_mb / self._agg.mem_total_mb

        # Estimated energy consumption (joules)
        power_mean = self._agg.power_mean_w()
        energy = float("nan")
        if not math.isnan(power_mean):
            energy = power_mean * effective_duration

        notes_parts: List[str] = []
        if isinstance(self._backend_obj, NullGpuBackend):
            notes_parts.append(self._backend_obj.reason)
        if self.warmup_s > 0:
            notes_parts.append(f"warmup ignored: first {self.warmup_s:.2f}s")
        if self._thread_error is not None and not self.strict:
            notes_parts.append(f"sampling stopped early: {self._thread_error_msg}")
        notes = " | ".join([p for p in notes_parts if p])

        return GpuSummary(
            device=self.device,
            name=name,
            duration_s=duration,
            interval_s=self.interval_s,
            n_samples=self._agg.n_samples,
            # GPU utilization
            util_gpu_mean=util_mean,
            util_gpu_std=self._agg.util_gpu_std(),
            util_gpu_min=self._agg.util_gpu_min,
            util_gpu_max=self._agg.util_gpu_max,
            util_gpu_p5=util_p5,
            util_gpu_p50=util_p50,
            util_gpu_p95=util_p95,
            util_gpu_p99=util_p99,
            idle_pct=self._agg.idle_pct(),
            active_pct=self._agg.active_pct(),
            # Memory-controller utilization
            util_mem_mean=self._agg.util_mem_mean(),
            # Device memory
            mem_used_mean_mb=self._agg.mem_used_mean_mb(),
            mem_used_max_mb=self._agg.mem_used_max_mb,
            mem_total_mb=self._agg.mem_total_mb,
            mem_util_pct=mem_util,
            # Power
            power_mean_w=power_mean,
            power_max_w=self._agg.power_max_w,
            energy_j=energy,
            # Temperature
            temp_mean_c=self._agg.temp_mean_c(),
            temp_max_c=self._agg.temp_max_c,
            # Clocks
            sm_clock_mean_mhz=self._agg.sm_clock_mean_mhz(),
            sm_clock_max_mhz=self._agg.sm_clock_max_mhz,
            # Derived
            busy_time_est_s=busy_est,
            sparkline=self._agg.sparkline(width=40),
            notes=notes,
        )


def gpu_profile(
    *,
    device: int = 0,
    interval_s: float = 0.2,
    backend: str = "auto",
    strict: bool = False,
    sync_fn: Optional[Callable[[], None]] = None,
    warmup_s: float = 0.0,
    store_samples: bool = False,
    report: Union[bool, Callable] = True,
    return_profile: bool = False,
    repeats: int = 1,
    warmup_runs: int = 0,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that profiles GPU utilisation while a function runs.

    Usage::

        @gpu_profile(interval_s=0.1)
        def train_epoch():
            ...

    For multi-run benchmarking::

        @gpu_profile(repeats=5, warmup_runs=1, return_profile=True)
        def bench():
            ...
        result = bench()   # returns MultiRunResult

    Parameters
    ----------
    device, interval_s, backend, strict, sync_fn, warmup_s, store_samples:
        Forwarded to :class:`GpuMonitor`.
    report:
        ``True`` (default) prints the summary to *stdout*.  ``False``
        suppresses output.  A callable is invoked with the
        :class:`GpuSummary` (single run) or :class:`MultiRunResult`
        (multi-run).
    return_profile:
        If ``True`` the decorated function returns a
        :class:`ProfiledResult` (single run) or :class:`MultiRunResult`
        (multi-run) instead of the raw return value.
    repeats:
        Number of times to run the function.  When > 1, results are
        aggregated into a :class:`MultiRunResult` with cross-run
        statistics (mean, std, min, max).
    warmup_runs:
        Number of initial runs to discard from the cross-run statistics
        (they are still executed).  Only meaningful when ``repeats > 1``.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be >= 0")

    def _make_monitor() -> GpuMonitor:
        return GpuMonitor(
            device=device,
            interval_s=interval_s,
            backend=backend,
            strict=strict,
            sync_fn=sync_fn,
            warmup_s=warmup_s,
            store_samples=store_samples,
        )

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if repeats == 1:
                return _single_run(fn, args, kwargs)
            return _multi_run(fn, args, kwargs)

        def _single_run(fn, args, kwargs):
            with _make_monitor() as mon:
                value = fn(*args, **kwargs)
            summary = mon.summary
            assert summary is not None
            if report is True:
                print(summary.format())
            elif callable(report):
                report(summary)
            if return_profile:
                return ProfiledResult(value=value, gpu=summary)
            return value

        def _multi_run(fn, args, kwargs):
            all_summaries: List[GpuSummary] = []
            value: Any = None
            for i in range(warmup_runs + repeats):
                with _make_monitor() as mon:
                    value = fn(*args, **kwargs)
                summary = mon.summary
                assert summary is not None
                if i >= warmup_runs:
                    all_summaries.append(summary)

            result = MultiRunResult.from_runs(all_summaries, value)
            if report is True:
                print(result.format())
            elif callable(report):
                report(result)
            if return_profile:
                return result
            return value

        return wrapper

    return decorator


def profile_repeats(
    fn: Callable[..., Any],
    *,
    repeats: int = 5,
    warmup_runs: int = 0,
    device: int = 0,
    interval_s: float = 0.2,
    backend: str = "auto",
    strict: bool = False,
    sync_fn: Optional[Callable[[], None]] = None,
    warmup_s: float = 0.0,
    store_samples: bool = False,
    report: Union[bool, Callable[[MultiRunResult], None]] = True,
) -> MultiRunResult:
    """Profile a callable over multiple runs and return cross-run statistics.

    The callable *fn* is invoked ``warmup_runs + repeats`` times.  The
    first *warmup_runs* executions are discarded; the remaining *repeats*
    are profiled and aggregated into a :class:`MultiRunResult`.

    Parameters
    ----------
    fn:
        No-argument callable to profile.  Use ``functools.partial`` or a
        ``lambda`` to bind arguments.
    repeats:
        Number of measured runs (must be >= 1).
    warmup_runs:
        Number of discarded warm-up runs.
    device, interval_s, backend, strict, sync_fn, warmup_s, store_samples:
        Forwarded to :class:`GpuMonitor` for each run.
    report:
        ``True`` prints the multi-run summary.  A callable receives the
        :class:`MultiRunResult`.

    Returns
    -------
    MultiRunResult
        Cross-run statistics and the individual :class:`GpuSummary`
        objects for every measured run.

    Example
    -------
    ::

        from profgpu import profile_repeats

        result = profile_repeats(
            lambda: train_epoch(model, loader),
            repeats=5,
            warmup_runs=1,
            interval_s=0.1,
        )
        print(result.util_gpu.mean, "±", result.util_gpu.std)
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be >= 0")

    summaries: List[GpuSummary] = []
    value: Any = None

    for i in range(warmup_runs + repeats):
        with GpuMonitor(
            device=device,
            interval_s=interval_s,
            backend=backend,
            strict=strict,
            sync_fn=sync_fn,
            warmup_s=warmup_s,
            store_samples=store_samples,
        ) as mon:
            value = fn()
        summary = mon.summary
        assert summary is not None
        if i >= warmup_runs:
            summaries.append(summary)

    result = MultiRunResult.from_runs(summaries, value)
    if report is True:
        print(result.format())
    elif callable(report):
        report(result)
    return result
