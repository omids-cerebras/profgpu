from __future__ import annotations

import functools
import math
import random
import shutil
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Union


# ---------------------------
# Small utilities
# ---------------------------

def _percentile(sorted_vals: Sequence[float], p: float) -> float:
    """
    Linear-interpolated percentile (like NumPy's default method for many cases).
    Input must be sorted ascending.
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
    """Mini trace using unicode block characters."""
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
        j = int(round(t * (len(blocks) - 1)))
        j = max(0, min(len(blocks) - 1, j))
        out.append(blocks[j])
    return "".join(out)


# ---------------------------
# Backends
# ---------------------------

class GpuBackendError(RuntimeError):
    """Backend initialization or sampling failed."""


class BaseGpuBackend:
    def device_name(self) -> str:
        raise NotImplementedError

    def read(self) -> Dict[str, Optional[float]]:
        """
        Return a dict of metrics. Unsupported metrics should be None.

        Keys:
          - util_gpu (%)
          - util_mem (%)
          - mem_used_mb
          - mem_total_mb
          - power_w
          - temp_c
          - sm_clock_mhz
          - mem_clock_mhz
        """
        raise NotImplementedError

    def close(self) -> None:
        return


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
            out["mem_used_mb"] = float(mem.used) / (1024 ** 2)
            out["mem_total_mb"] = float(mem.total) / (1024 ** 2)

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
    """
    Fallback backend using `nvidia-smi` queries. Heavier than NVML because it spawns a process per sample.
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
    """A backend that always returns None metrics; used when no backend is available and strict=False."""

    def __init__(self, reason: str):
        self.reason = reason

    def device_name(self) -> str:
        return "N/A"

    def read(self) -> Dict[str, Optional[float]]:
        return {
            "util_gpu": None,
            "util_mem": None,
            "mem_used_mb": None,
            "mem_total_mb": None,
            "power_w": None,
            "temp_c": None,
            "sm_clock_mhz": None,
            "mem_clock_mhz": None,
        }


# ---------------------------
# Data model
# ---------------------------

@dataclass(frozen=True)
class GpuSample:
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
    device: int
    name: str
    duration_s: float
    interval_s: float
    n_samples: int

    util_gpu_mean: float
    util_gpu_p50: float
    util_gpu_p95: float
    util_gpu_max: float

    util_mem_mean: float

    mem_used_max_mb: float
    mem_total_mb: float

    power_mean_w: float
    power_max_w: float
    temp_max_c: float

    busy_time_est_s: float
    sparkline: str
    notes: str = ""

    def format(self) -> str:
        def f(x: float, unit: str = "", digits: int = 1) -> str:
            if x is None or (isinstance(x, float) and math.isnan(x)):
                return "n/a"
            return f"{x:.{digits}f}{unit}"

        lines: List[str] = []
        lines.append(f"[GPU {self.device}] {self.name}")
        lines.append(f"  duration: {f(self.duration_s,'s',3)} | samples: {self.n_samples} @ {f(self.interval_s,'s',3)}")
        lines.append(
            "  util.gpu: "
            f"mean {f(self.util_gpu_mean,'%',1)} | p50 {f(self.util_gpu_p50,'%',1)} | "
            f"p95 {f(self.util_gpu_p95,'%',1)} | max {f(self.util_gpu_max,'%',1)}"
        )
        lines.append(f"  util.mem: mean {f(self.util_mem_mean,'%',1)}")
        if not math.isnan(self.mem_total_mb):
            lines.append(f"  memory: max used {f(self.mem_used_max_mb,' MB',0)} / total {f(self.mem_total_mb,' MB',0)}")
        lines.append(f"  power: mean {f(self.power_mean_w,' W',1)} | max {f(self.power_max_w,' W',1)}")
        lines.append(f"  temp: max {f(self.temp_max_c,' °C',0)}")
        lines.append(f"  busy time (est): {f(self.busy_time_est_s,'s',3)}")
        if self.sparkline:
            lines.append(f"  util trace: {self.sparkline}")
        if self.notes:
            lines.append(f"  notes: {self.notes}")
        return "\n".join(lines)


@dataclass(frozen=True)
class ProfiledResult:
    value: Any
    gpu: GpuSummary


# ---------------------------
# Streaming aggregation
# ---------------------------

class _Aggregator:
    def __init__(self, *, warmup_s: float, reservoir_size: int, trace_len: int):
        self.warmup_s = float(max(0.0, warmup_s))
        self.reservoir_size = int(max(16, reservoir_size))
        self.trace_len = int(max(40, trace_len))

        self.n_samples = 0

        # Means (separate counts because values can be missing)
        self._util_gpu_sum = 0.0
        self._util_gpu_n = 0

        self._util_mem_sum = 0.0
        self._util_mem_n = 0

        self._power_sum = 0.0
        self._power_n = 0

        # Maxes
        self.util_gpu_max = float("nan")
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
        self._util_gpu_seen += 1
        if len(self._util_gpu_reservoir) < self.reservoir_size:
            self._util_gpu_reservoir.append(v)
            return
        j = random.randrange(self._util_gpu_seen)
        if j < self.reservoir_size:
            self._util_gpu_reservoir[j] = v

    def add(self, t_s: float, metrics: Dict[str, Optional[float]]) -> None:
        if t_s < self.warmup_s:
            return

        self.n_samples += 1

        util_gpu = metrics.get("util_gpu")
        if util_gpu is not None:
            v = float(util_gpu)
            self._util_gpu_sum += v
            self._util_gpu_n += 1
            self.util_gpu_max = v if math.isnan(self.util_gpu_max) else max(self.util_gpu_max, v)
            self._reservoir_add(v)
            self._trace_append(v)

        util_mem = metrics.get("util_mem")
        if util_mem is not None:
            self._util_mem_sum += float(util_mem)
            self._util_mem_n += 1

        mem_used = metrics.get("mem_used_mb")
        if mem_used is not None:
            v = float(mem_used)
            self.mem_used_max_mb = v if math.isnan(self.mem_used_max_mb) else max(self.mem_used_max_mb, v)

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
            self.temp_max_c = v if math.isnan(self.temp_max_c) else max(self.temp_max_c, v)

    def util_gpu_mean(self) -> float:
        return (self._util_gpu_sum / self._util_gpu_n) if self._util_gpu_n else float("nan")

    def util_mem_mean(self) -> float:
        return (self._util_mem_sum / self._util_mem_n) if self._util_mem_n else float("nan")

    def power_mean_w(self) -> float:
        return (self._power_sum / self._power_n) if self._power_n else float("nan")

    def util_gpu_quantiles(self) -> Dict[str, float]:
        if not self._util_gpu_reservoir:
            return {"p50": float("nan"), "p95": float("nan")}
        vals = sorted(self._util_gpu_reservoir)
        return {"p50": _percentile(vals, 50), "p95": _percentile(vals, 95)}

    def sparkline(self, width: int = 40) -> str:
        if not self._util_trace:
            return ""
        return _sparkline(self._util_trace, width=width)


# ---------------------------
# Public API: monitor + decorator
# ---------------------------

class GpuMonitor:
    """
    Context manager that samples GPU metrics in a background thread.

    If your GPU framework launches work asynchronously (e.g. PyTorch), pass a `sync_fn`
    (e.g. `torch.cuda.synchronize`) to make the monitor boundaries match actual GPU work.
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

        self._agg = _Aggregator(warmup_s=self.warmup_s, reservoir_size=reservoir_size, trace_len=trace_len)
        self._samples: List[GpuSample] = []

        self._thread_error: Optional[BaseException] = None
        self._thread_error_msg: str = ""

        self.summary: Optional[GpuSummary] = None

    def _make_backend(self) -> BaseGpuBackend:
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

    def __enter__(self) -> "GpuMonitor":
        # Optional sync to exclude earlier queued work
        if self.sync_fn is not None:
            try:
                self.sync_fn()
            except Exception:
                pass

        self._t0 = time.perf_counter()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="GpuMonitor", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # Optional sync so the block includes queued async GPU work
        if self.sync_fn is not None:
            try:
                self.sync_fn()
            except Exception:
                pass

        self.stop()
        return False

    def _run(self) -> None:
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
        if self._t1 is None:
            self._t1 = time.perf_counter()

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 5 * self.interval_s))

        try:
            self._backend_obj.close()
        except Exception:
            pass

        self.summary = self._summarize()

        # Surface thread error after producing a best-effort summary.
        if self._thread_error is not None and self.strict:
            raise GpuBackendError(f"GPU sampling failed: {self._thread_error_msg}") from self._thread_error

        return self.summary

    def _summarize(self) -> GpuSummary:
        name = self._backend_obj.device_name()
        duration = max(0.0, (self._t1 or time.perf_counter()) - (self._t0 or 0.0))

        # Effective window after warmup
        effective_duration = max(0.0, duration - self.warmup_s)

        util_mean = self._agg.util_gpu_mean()
        util_q = self._agg.util_gpu_quantiles()
        util_p50 = util_q["p50"]
        util_p95 = util_q["p95"]

        busy_est = float("nan")
        if not math.isnan(util_mean):
            busy_est = effective_duration * (util_mean / 100.0)

        notes_parts: List[str] = []
        if isinstance(self._backend_obj, NullGpuBackend):
            notes_parts.append(getattr(self._backend_obj, "reason", ""))
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
            util_gpu_mean=util_mean,
            util_gpu_p50=util_p50,
            util_gpu_p95=util_p95,
            util_gpu_max=self._agg.util_gpu_max,
            util_mem_mean=self._agg.util_mem_mean(),
            mem_used_max_mb=self._agg.mem_used_max_mb,
            mem_total_mb=self._agg.mem_total_mb,
            power_mean_w=self._agg.power_mean_w(),
            power_max_w=self._agg.power_max_w,
            temp_max_c=self._agg.temp_max_c,
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
    report: Union[bool, Callable[[GpuSummary], None]] = True,
    return_profile: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to sample GPU utilization while a function runs.

    report:
      - True: print summary
      - False: no printing
      - callable: called with GpuSummary

    return_profile:
      - False: return original function value
      - True: return ProfiledResult(value, gpu_summary)
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with GpuMonitor(
                device=device,
                interval_s=interval_s,
                backend=backend,
                strict=strict,
                sync_fn=sync_fn,
                warmup_s=warmup_s,
                store_samples=store_samples,
            ) as mon:
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

        return wrapper

    return decorator
