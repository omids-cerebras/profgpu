"""Comprehensive test suite for profgpu.

All tests use ``backend='none'`` (or monkey-patched test backends) so
they run on any machine without needing an NVIDIA GPU or driver.  The
``NullGpuBackend`` returns ``None`` for every metric, but the entire code
path — threading, aggregation, formatting, and the public API — is still
exercised.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from dataclasses import asdict
from typing import Dict, List, Optional
from unittest import mock

import pytest

from profgpu import (
    GpuBackendError,
    GpuMonitor,
    GpuSample,
    GpuSummary,
    MultiRunResult,
    ProfiledResult,
    gpu_profile,
    profile_repeats,
)
from profgpu.monitor import (
    METRIC_KEYS,
    BaseGpuBackend,
    NullGpuBackend,
    NvidiaSmiBackend,
    _Aggregator,
    _compute_run_stats,
    _percentile,
    _sparkline,
)


def _make_summary(**overrides) -> GpuSummary:
    """Build a GpuSummary with sensible defaults; override any field via kwargs."""
    defaults = dict(
        device=0,
        name="TestGPU",
        duration_s=1.0,
        interval_s=0.1,
        n_samples=10,
        util_gpu_mean=50.0,
        util_gpu_std=10.0,
        util_gpu_min=20.0,
        util_gpu_max=80.0,
        util_gpu_p5=22.0,
        util_gpu_p50=50.0,
        util_gpu_p95=78.0,
        util_gpu_p99=80.0,
        idle_pct=0.0,
        active_pct=60.0,
        util_mem_mean=30.0,
        mem_used_mean_mb=4000.0,
        mem_used_max_mb=6000.0,
        mem_total_mb=8192.0,
        mem_util_pct=73.2,
        power_mean_w=100.0,
        power_max_w=150.0,
        energy_j=100.0,
        temp_mean_c=55.0,
        temp_max_c=60.0,
        sm_clock_mean_mhz=1400.0,
        sm_clock_max_mhz=1600.0,
        busy_time_est_s=0.5,
        sparkline="",
        notes="",
    )
    defaults.update(overrides)
    return GpuSummary(**defaults)


# ===================================================================
# Utilities: helper backends for testing
# ===================================================================


class ConstantBackend(BaseGpuBackend):
    """Backend that returns fixed, deterministic metrics."""

    def __init__(
        self,
        util_gpu: float = 50.0,
        util_mem: float = 30.0,
        mem_used_mb: float = 4096.0,
        mem_total_mb: float = 8192.0,
        power_w: float = 150.0,
        temp_c: float = 65.0,
        sm_clock_mhz: float = 1500.0,
        mem_clock_mhz: float = 900.0,
    ) -> None:
        self._metrics: Dict[str, Optional[float]] = {
            "util_gpu": util_gpu,
            "util_mem": util_mem,
            "mem_used_mb": mem_used_mb,
            "mem_total_mb": mem_total_mb,
            "power_w": power_w,
            "temp_c": temp_c,
            "sm_clock_mhz": sm_clock_mhz,
            "mem_clock_mhz": mem_clock_mhz,
        }

    def device_name(self) -> str:
        return "TestGPU"

    def read(self) -> Dict[str, Optional[float]]:
        return dict(self._metrics)


class FailingBackend(BaseGpuBackend):
    """Backend that raises on the *n*-th call to ``read()``."""

    def __init__(self, fail_after: int = 3) -> None:
        self._fail_after = fail_after
        self._count = 0

    def device_name(self) -> str:
        return "FailGPU"

    def read(self) -> Dict[str, Optional[float]]:
        self._count += 1
        if self._count > self._fail_after:
            raise RuntimeError("simulated hardware fault")
        return {k: 42.0 for k in METRIC_KEYS}


class RampBackend(BaseGpuBackend):
    """Backend whose ``util_gpu`` ramps linearly from 0 to 100 over calls."""

    def __init__(self, total_calls: int = 100) -> None:
        self._total = total_calls
        self._i = 0

    def device_name(self) -> str:
        return "RampGPU"

    def read(self) -> Dict[str, Optional[float]]:
        util = 100.0 * self._i / max(self._total - 1, 1)
        self._i += 1
        return {
            "util_gpu": util,
            "util_mem": util * 0.5,
            "mem_used_mb": 1000.0 + util * 50.0,
            "mem_total_mb": 16384.0,
            "power_w": 50.0 + util * 2.0,
            "temp_c": 30.0 + util * 0.5,
            "sm_clock_mhz": 1200.0,
            "mem_clock_mhz": 800.0,
        }


# ===================================================================
# _percentile
# ===================================================================


class TestPercentile:
    """Tests for the ``_percentile`` helper."""

    def test_empty_returns_nan(self) -> None:
        assert math.isnan(_percentile([], 50))

    def test_single_value(self) -> None:
        assert _percentile([7.0], 50) == 7.0
        assert _percentile([7.0], 0) == 7.0
        assert _percentile([7.0], 100) == 7.0

    def test_two_values_median(self) -> None:
        assert math.isclose(_percentile([10.0, 20.0], 50), 15.0)

    def test_boundary_clamp(self) -> None:
        vals = [1.0, 2.0, 3.0]
        assert _percentile(vals, -10) == 1.0  # clamped to min
        assert _percentile(vals, 200) == 3.0  # clamped to max

    def test_known_percentiles(self) -> None:
        vals = list(range(101))  # 0, 1, 2, ..., 100
        assert math.isclose(_percentile(vals, 25), 25.0)
        assert math.isclose(_percentile(vals, 75), 75.0)
        assert math.isclose(_percentile(vals, 95), 95.0)


# ===================================================================
# _sparkline
# ===================================================================


class TestSparkline:
    """Tests for the ``_sparkline`` helper."""

    def test_empty_returns_empty(self) -> None:
        assert _sparkline([]) == ""

    def test_zero_width_returns_empty(self) -> None:
        assert _sparkline([1, 2, 3], width=0) == ""

    def test_output_length(self) -> None:
        s = _sparkline([10, 20, 30, 40, 50], width=10)
        assert len(s) == 10

    def test_constant_values_all_same_block(self) -> None:
        s = _sparkline([50, 50, 50, 50], width=4)
        # All values identical → every char is the same
        assert len(set(s)) == 1

    def test_ascending_values_end_higher(self) -> None:
        s = _sparkline([0, 25, 50, 75, 100], width=5)
        blocks = "▁▂▃▄▅▆▇█"
        assert s[0] == blocks[0]
        assert s[-1] == blocks[-1]


# ===================================================================
# NullGpuBackend
# ===================================================================


class TestNullGpuBackend:
    """Tests for the no-op fallback backend."""

    def test_device_name(self) -> None:
        assert NullGpuBackend("reason").device_name() == "N/A"

    def test_read_returns_all_none(self) -> None:
        m = NullGpuBackend("no GPU").read()
        for k in METRIC_KEYS:
            assert k in m
            assert m[k] is None

    def test_reason_stored(self) -> None:
        assert NullGpuBackend("testing").reason == "testing"

    def test_close_is_noop(self) -> None:
        NullGpuBackend("x").close()  # should not raise


# ===================================================================
# _Aggregator
# ===================================================================


class TestAggregator:
    """Tests for the streaming statistics accumulator."""

    @staticmethod
    def _make_metrics(**overrides: Optional[float]) -> Dict[str, Optional[float]]:
        base: Dict[str, Optional[float]] = {k: None for k in METRIC_KEYS}
        base.update(overrides)
        return base

    def test_no_samples_returns_nan(self) -> None:
        agg = _Aggregator(warmup_s=0, reservoir_size=64, trace_len=40)
        assert math.isnan(agg.util_gpu_mean())
        assert math.isnan(agg.util_mem_mean())
        assert math.isnan(agg.power_mean_w())
        q = agg.util_gpu_quantiles()
        assert math.isnan(q["p50"])
        assert math.isnan(q["p95"])

    def test_single_sample(self) -> None:
        agg = _Aggregator(warmup_s=0, reservoir_size=64, trace_len=40)
        agg.add(0.1, self._make_metrics(util_gpu=80.0, power_w=200.0, temp_c=70.0))
        assert agg.n_samples == 1
        assert math.isclose(agg.util_gpu_mean(), 80.0)
        assert math.isclose(agg.power_mean_w(), 200.0)
        assert math.isclose(agg.temp_max_c, 70.0)

    def test_warmup_filters_early_samples(self) -> None:
        agg = _Aggregator(warmup_s=1.0, reservoir_size=64, trace_len=40)
        agg.add(0.5, self._make_metrics(util_gpu=99.0))
        assert agg.n_samples == 0

        agg.add(1.5, self._make_metrics(util_gpu=50.0))
        assert agg.n_samples == 1
        assert math.isclose(agg.util_gpu_mean(), 50.0)

    def test_max_tracking(self) -> None:
        agg = _Aggregator(warmup_s=0, reservoir_size=64, trace_len=40)
        for v in [10.0, 50.0, 30.0, 80.0, 20.0]:
            agg.add(
                0.0,
                self._make_metrics(
                    util_gpu=v,
                    mem_used_mb=v * 10,
                    power_w=v,
                    temp_c=v / 2,
                ),
            )
        assert math.isclose(agg.util_gpu_max, 80.0)
        assert math.isclose(agg.mem_used_max_mb, 800.0)
        assert math.isclose(agg.power_max_w, 80.0)
        assert math.isclose(agg.temp_max_c, 40.0)

    def test_mean_calculation(self) -> None:
        agg = _Aggregator(warmup_s=0, reservoir_size=64, trace_len=40)
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        for v in values:
            agg.add(0.0, self._make_metrics(util_gpu=v, util_mem=v * 2, power_w=v))
        expected_mean = sum(values) / len(values)
        assert math.isclose(agg.util_gpu_mean(), expected_mean)
        assert math.isclose(agg.util_mem_mean(), expected_mean * 2)
        assert math.isclose(agg.power_mean_w(), expected_mean)

    def test_mem_total_takes_last_value(self) -> None:
        agg = _Aggregator(warmup_s=0, reservoir_size=64, trace_len=40)
        agg.add(0.0, self._make_metrics(mem_total_mb=8000.0))
        agg.add(1.0, self._make_metrics(mem_total_mb=16000.0))
        assert math.isclose(agg.mem_total_mb, 16000.0)

    def test_reservoir_bounded(self) -> None:
        reservoir_size = 32
        agg = _Aggregator(warmup_s=0, reservoir_size=reservoir_size, trace_len=40)
        for i in range(500):
            agg.add(0.0, self._make_metrics(util_gpu=float(i % 100)))
        assert len(agg._util_gpu_reservoir) <= reservoir_size

    def test_trace_compression(self) -> None:
        trace_len = 40
        agg = _Aggregator(warmup_s=0, reservoir_size=64, trace_len=trace_len)
        for i in range(1000):
            agg.add(0.0, self._make_metrics(util_gpu=float(i % 100)))
        assert len(agg._util_trace) <= trace_len

    def test_sparkline_output(self) -> None:
        agg = _Aggregator(warmup_s=0, reservoir_size=64, trace_len=40)
        for i in range(50):
            agg.add(0.0, self._make_metrics(util_gpu=float(i * 2)))
        s = agg.sparkline(width=20)
        assert isinstance(s, str)
        assert len(s) == 20

    def test_quantiles_approximate(self) -> None:
        agg = _Aggregator(warmup_s=0, reservoir_size=4096, trace_len=40)
        for v in range(101):
            agg.add(0.0, self._make_metrics(util_gpu=float(v)))
        q = agg.util_gpu_quantiles()
        assert abs(q["p50"] - 50.0) < 5.0
        assert abs(q["p95"] - 95.0) < 5.0
        assert abs(q["p5"] - 5.0) < 5.0
        assert abs(q["p99"] - 99.0) < 5.0

    def test_missing_metrics_handled(self) -> None:
        agg = _Aggregator(warmup_s=0, reservoir_size=64, trace_len=40)
        agg.add(0.0, {k: None for k in METRIC_KEYS})
        assert agg.n_samples == 1
        assert math.isnan(agg.util_gpu_mean())

    def test_warmup_s_clamped_to_zero(self) -> None:
        agg = _Aggregator(warmup_s=-5.0, reservoir_size=64, trace_len=40)
        assert agg.warmup_s == 0.0

    def test_reservoir_size_floor(self) -> None:
        agg = _Aggregator(warmup_s=0, reservoir_size=1, trace_len=40)
        assert agg.reservoir_size >= 16

    def test_trace_len_floor(self) -> None:
        agg = _Aggregator(warmup_s=0, reservoir_size=64, trace_len=1)
        assert agg.trace_len >= 40

    # --- Extended metric tests (v0.3+) ---

    def test_util_gpu_min(self) -> None:
        agg = _Aggregator(warmup_s=0, reservoir_size=64, trace_len=40)
        for v in [80.0, 20.0, 60.0, 10.0, 50.0]:
            agg.add(0.0, self._make_metrics(util_gpu=v))
        assert math.isclose(agg.util_gpu_min, 10.0)

    def test_util_gpu_min_nan_when_empty(self) -> None:
        agg = _Aggregator(warmup_s=0, reservoir_size=64, trace_len=40)
        assert math.isnan(agg.util_gpu_min)

    def test_util_gpu_std_constant(self) -> None:
        agg = _Aggregator(warmup_s=0, reservoir_size=64, trace_len=40)
        for _ in range(10):
            agg.add(0.0, self._make_metrics(util_gpu=50.0))
        assert math.isclose(agg.util_gpu_std(), 0.0, abs_tol=1e-9)

    def test_util_gpu_std_variable(self) -> None:
        agg = _Aggregator(warmup_s=0, reservoir_size=64, trace_len=40)
        for v in [0.0, 100.0]:
            agg.add(0.0, self._make_metrics(util_gpu=v))
        assert math.isclose(agg.util_gpu_std(), 50.0, abs_tol=0.1)

    def test_util_gpu_std_nan_single_sample(self) -> None:
        agg = _Aggregator(warmup_s=0, reservoir_size=64, trace_len=40)
        agg.add(0.0, self._make_metrics(util_gpu=50.0))
        assert math.isnan(agg.util_gpu_std())

    def test_idle_pct(self) -> None:
        agg = _Aggregator(warmup_s=0, reservoir_size=64, trace_len=40)
        # 3 idle (<5%), 2 not idle
        for v in [0.0, 2.0, 4.9, 50.0, 100.0]:
            agg.add(0.0, self._make_metrics(util_gpu=v))
        assert math.isclose(agg.idle_pct(), 60.0)

    def test_active_pct(self) -> None:
        agg = _Aggregator(warmup_s=0, reservoir_size=64, trace_len=40)
        # 3 active (>=50%), 2 not active
        for v in [0.0, 10.0, 50.0, 75.0, 100.0]:
            agg.add(0.0, self._make_metrics(util_gpu=v))
        assert math.isclose(agg.active_pct(), 60.0)

    def test_idle_active_nan_when_empty(self) -> None:
        agg = _Aggregator(warmup_s=0, reservoir_size=64, trace_len=40)
        assert math.isnan(agg.idle_pct())
        assert math.isnan(agg.active_pct())

    def test_mem_used_mean_mb(self) -> None:
        agg = _Aggregator(warmup_s=0, reservoir_size=64, trace_len=40)
        for v in [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]:
            agg.add(0.0, self._make_metrics(mem_used_mb=v))
        assert math.isclose(agg.mem_used_mean_mb(), 3000.0)

    def test_mem_used_mean_nan_when_empty(self) -> None:
        agg = _Aggregator(warmup_s=0, reservoir_size=64, trace_len=40)
        assert math.isnan(agg.mem_used_mean_mb())

    def test_sm_clock_mean_mhz(self) -> None:
        agg = _Aggregator(warmup_s=0, reservoir_size=64, trace_len=40)
        for v in [1200.0, 1400.0, 1600.0]:
            agg.add(0.0, self._make_metrics(sm_clock_mhz=v))
        assert math.isclose(agg.sm_clock_mean_mhz(), 1400.0)

    def test_sm_clock_max_mhz(self) -> None:
        agg = _Aggregator(warmup_s=0, reservoir_size=64, trace_len=40)
        for v in [1200.0, 1600.0, 1400.0]:
            agg.add(0.0, self._make_metrics(sm_clock_mhz=v))
        assert math.isclose(agg.sm_clock_max_mhz, 1600.0)

    def test_temp_mean_c(self) -> None:
        agg = _Aggregator(warmup_s=0, reservoir_size=64, trace_len=40)
        for v in [60.0, 70.0, 80.0]:
            agg.add(0.0, self._make_metrics(temp_c=v))
        assert math.isclose(agg.temp_mean_c(), 70.0)

    def test_quantiles_return_all_four_keys(self) -> None:
        agg = _Aggregator(warmup_s=0, reservoir_size=64, trace_len=40)
        agg.add(0.0, self._make_metrics(util_gpu=50.0))
        q = agg.util_gpu_quantiles()
        assert set(q.keys()) == {"p5", "p50", "p95", "p99"}

    def test_quantiles_empty_returns_nan(self) -> None:
        agg = _Aggregator(warmup_s=0, reservoir_size=64, trace_len=40)
        q = agg.util_gpu_quantiles()
        for key in ["p5", "p50", "p95", "p99"]:
            assert math.isnan(q[key])


# ===================================================================
# GpuSample
# ===================================================================


class TestGpuSample:
    """Tests for the GpuSample dataclass."""

    def test_frozen(self) -> None:
        s = GpuSample(
            t_s=0.1,
            util_gpu=50.0,
            util_mem=30.0,
            mem_used_mb=1024.0,
            mem_total_mb=8192.0,
            power_w=100.0,
            temp_c=60.0,
            sm_clock_mhz=1500.0,
            mem_clock_mhz=900.0,
        )
        with pytest.raises(AttributeError):
            s.util_gpu = 99.0  # type: ignore[misc]

    def test_asdict(self) -> None:
        s = GpuSample(
            t_s=0.5,
            util_gpu=75.0,
            util_mem=None,
            mem_used_mb=None,
            mem_total_mb=None,
            power_w=None,
            temp_c=None,
            sm_clock_mhz=None,
            mem_clock_mhz=None,
        )
        d = asdict(s)
        assert d["t_s"] == 0.5
        assert d["util_gpu"] == 75.0
        assert d["util_mem"] is None


# ===================================================================
# GpuSummary
# ===================================================================


class TestGpuSummary:
    """Tests for the GpuSummary dataclass and its ``format()`` method."""

    @pytest.fixture()
    def summary(self) -> GpuSummary:
        return _make_summary(
            duration_s=10.0,
            interval_s=0.2,
            n_samples=50,
            util_gpu_mean=65.0,
            util_gpu_std=15.3,
            util_gpu_min=10.0,
            util_gpu_max=98.0,
            util_gpu_p5=12.0,
            util_gpu_p50=62.0,
            util_gpu_p95=90.0,
            util_gpu_p99=97.0,
            idle_pct=5.0,
            active_pct=80.0,
            power_mean_w=180.0,
            power_max_w=250.0,
            temp_mean_c=68.0,
            temp_max_c=75.0,
            energy_j=1800.0,
            mem_used_mean_mb=4500.0,
            sparkline="▁▃▅▇",
            busy_time_est_s=6.5,
        )

    def test_format_contains_device(self, summary: GpuSummary) -> None:
        text = summary.format()
        assert "[GPU 0]" in text
        assert "TestGPU" in text

    def test_format_contains_stats(self, summary: GpuSummary) -> None:
        text = summary.format()
        assert "65.0%" in text
        assert "98.0%" in text
        assert "180.0 W" in text
        assert "75 °C" in text

    def test_format_contains_sparkline(self, summary: GpuSummary) -> None:
        assert "▁▃▅▇" in summary.format()

    def test_format_contains_extended_metrics(self, summary: GpuSummary) -> None:
        text = summary.format()
        # std and min
        assert "std" in text.lower()
        assert "min" in text.lower()
        # percentiles p5 / p99
        assert "p5" in text
        assert "p99" in text
        # idle / active
        assert "idle" in text.lower()
        assert "active" in text.lower()
        # energy
        assert "energy" in text.lower() or "J" in text
        # clocks
        assert "MHz" in text
        # mean temp
        assert "68" in text  # temp_mean_c

    def test_format_all_nan_graceful(self) -> None:
        """All-NaN fields format gracefully without crashing."""
        nan = float("nan")
        s = _make_summary(
            name="N/A",
            duration_s=0.0,
            n_samples=0,
            util_gpu_mean=nan,
            util_gpu_std=nan,
            util_gpu_min=nan,
            util_gpu_max=nan,
            util_gpu_p5=nan,
            util_gpu_p50=nan,
            util_gpu_p95=nan,
            util_gpu_p99=nan,
            idle_pct=nan,
            active_pct=nan,
            util_mem_mean=nan,
            mem_used_mean_mb=nan,
            mem_used_max_mb=nan,
            mem_total_mb=nan,
            mem_util_pct=nan,
            power_mean_w=nan,
            power_max_w=nan,
            energy_j=nan,
            temp_mean_c=nan,
            temp_max_c=nan,
            sm_clock_mean_mhz=nan,
            sm_clock_max_mhz=nan,
            busy_time_est_s=nan,
            sparkline="",
            notes="",
        )
        text = s.format()
        assert isinstance(text, str)
        assert "[GPU 0]" in text
        assert "n/a" in text

    def test_format_nan_shows_na(self) -> None:
        nan = float("nan")
        s = _make_summary(
            name="N/A",
            duration_s=0.0,
            n_samples=0,
            util_gpu_mean=nan,
            util_gpu_std=nan,
            util_gpu_min=nan,
            util_gpu_max=nan,
            util_gpu_p5=nan,
            util_gpu_p50=nan,
            util_gpu_p95=nan,
            util_gpu_p99=nan,
            idle_pct=nan,
            active_pct=nan,
            util_mem_mean=nan,
            mem_used_mean_mb=nan,
            mem_used_max_mb=nan,
            mem_total_mb=nan,
            mem_util_pct=nan,
            power_mean_w=nan,
            power_max_w=nan,
            energy_j=nan,
            temp_mean_c=nan,
            temp_max_c=nan,
            sm_clock_mean_mhz=nan,
            sm_clock_max_mhz=nan,
            busy_time_est_s=nan,
            sparkline="",
            notes="no GPU",
        )
        text = s.format()
        assert "n/a" in text
        assert "no GPU" in text

    def test_format_with_notes(self) -> None:
        s = _make_summary(
            device=1,
            name="GPU1",
            duration_s=5.0,
            n_samples=50,
            busy_time_est_s=2.5,
            notes="warmup ignored: first 1.00s",
        )
        assert "warmup ignored" in s.format()

    def test_frozen(self) -> None:
        s = _make_summary()
        with pytest.raises(AttributeError):
            s.device = 1  # type: ignore[misc]


# ===================================================================
# ProfiledResult
# ===================================================================


class TestProfiledResult:
    """Tests for the ProfiledResult wrapper."""

    def test_fields(self) -> None:
        s = _make_summary()
        pr = ProfiledResult(value=42, gpu=s)
        assert pr.value == 42
        assert pr.gpu is s

    def test_frozen(self) -> None:
        pr = ProfiledResult(value="hello", gpu=_make_summary())
        with pytest.raises(AttributeError):
            pr.value = "bye"  # type: ignore[misc]


# ===================================================================
# GpuMonitor – core context manager
# ===================================================================


class TestGpuMonitor:
    """Tests for GpuMonitor using the NullGpuBackend."""

    def test_none_backend_no_crash(self) -> None:
        with GpuMonitor(backend="none", strict=False, interval_s=0.01) as mon:
            pass
        assert mon.summary is not None
        assert mon.summary.duration_s >= 0.0

    def test_none_backend_notes_contain_reason(self) -> None:
        with GpuMonitor(backend="none", strict=False, interval_s=0.01) as mon:
            pass
        notes = mon.summary.notes.lower()
        assert "disabled" in notes or "none" in notes

    def test_summary_populated_after_exit(self) -> None:
        with GpuMonitor(backend="none", strict=False, interval_s=0.01) as mon:
            time.sleep(0.05)
        assert mon.summary is not None
        assert mon.summary.n_samples >= 0

    def test_stop_idempotent(self) -> None:
        with GpuMonitor(backend="none", strict=False, interval_s=0.01) as mon:
            pass
        s1 = mon.stop()
        s2 = mon.stop()
        assert s1 is not None
        assert s2 is not None

    def test_interval_validation(self) -> None:
        with pytest.raises(ValueError, match="interval_s"):
            GpuMonitor(backend="none", interval_s=0)
        with pytest.raises(ValueError, match="interval_s"):
            GpuMonitor(backend="none", interval_s=-1)

    def test_max_samples_validation(self) -> None:
        with pytest.raises(ValueError, match="max_samples"):
            GpuMonitor(backend="none", store_samples=True, max_samples=10)

    def test_invalid_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="backend"):
            GpuMonitor(backend="bogus")

    def test_strict_auto_raises_when_no_gpu(self) -> None:
        with mock.patch(
            "profgpu.monitor.NvidiaNvmlBackend.__init__",
            side_effect=GpuBackendError("no nvml"),
        ):
            with mock.patch(
                "profgpu.monitor.NvidiaSmiBackend.__init__",
                side_effect=GpuBackendError("no smi"),
            ):
                with pytest.raises(GpuBackendError):
                    GpuMonitor(backend="auto", strict=True)


class TestGpuMonitorWithConstantBackend:
    """Tests using a patched ConstantBackend for deterministic assertions."""

    def _make_monitor(self, **kwargs) -> GpuMonitor:
        mon = GpuMonitor(backend="none", strict=False, **kwargs)
        mon._backend_obj = ConstantBackend()
        return mon

    def test_summary_stats_converge(self) -> None:
        mon = self._make_monitor(interval_s=0.01)
        with mon:
            time.sleep(0.15)
        s = mon.summary
        assert s is not None
        assert s.n_samples >= 5
        if not math.isnan(s.util_gpu_mean):
            assert abs(s.util_gpu_mean - 50.0) < 1.0

    def test_store_samples(self) -> None:
        mon = self._make_monitor(interval_s=0.01, store_samples=True)
        with mon:
            time.sleep(0.1)
        samples = mon.samples
        assert len(samples) > 0
        for sample in samples:
            assert isinstance(sample, GpuSample)
            assert sample.t_s >= 0.0

    def test_samples_empty_when_disabled(self) -> None:
        mon = self._make_monitor(interval_s=0.01, store_samples=False)
        with mon:
            time.sleep(0.05)
        assert mon.samples == []

    def test_warmup_filters(self) -> None:
        mon = self._make_monitor(interval_s=0.01, warmup_s=0.1)
        with mon:
            time.sleep(0.2)
        assert mon.summary is not None
        assert "warmup" in mon.summary.notes

    def test_device_name_propagated(self) -> None:
        mon = self._make_monitor(interval_s=0.01)
        with mon:
            time.sleep(0.03)
        assert mon.summary.name == "TestGPU"

    def test_extended_summary_fields_populated(self) -> None:
        """Verify all 12 extended metrics are populated (not NaN) with ConstantBackend."""
        mon = self._make_monitor(interval_s=0.01)
        with mon:
            time.sleep(0.15)
        s = mon.summary
        assert s is not None
        assert s.n_samples >= 5
        # ConstantBackend returns util_gpu=50 every sample
        if s.n_samples >= 2:
            assert not math.isnan(s.util_gpu_min)
            assert not math.isnan(s.util_gpu_std)
            assert not math.isnan(s.util_gpu_p5)
            assert not math.isnan(s.util_gpu_p99)
            assert not math.isnan(s.idle_pct)
            assert not math.isnan(s.active_pct)
            assert not math.isnan(s.mem_used_mean_mb)
            assert not math.isnan(s.mem_util_pct)
            assert not math.isnan(s.energy_j)
            assert not math.isnan(s.sm_clock_mean_mhz)
            assert not math.isnan(s.sm_clock_max_mhz)
            assert not math.isnan(s.temp_mean_c)

    def test_extended_summary_values_correct(self) -> None:
        """Verify numeric correctness of extended metrics with ConstantBackend."""
        mon = self._make_monitor(interval_s=0.01)
        with mon:
            time.sleep(0.15)
        s = mon.summary
        assert s is not None
        if s.n_samples >= 5:
            # util_gpu=50 constant → min=50, std≈0, idle=0%, active=100%
            assert math.isclose(s.util_gpu_min, 50.0, abs_tol=0.1)
            assert math.isclose(s.util_gpu_std, 0.0, abs_tol=0.1)
            assert math.isclose(s.idle_pct, 0.0, abs_tol=0.1)
            assert math.isclose(s.active_pct, 100.0, abs_tol=0.1)
            # mem_used=4096, mem_total=8192 → util=50%
            assert math.isclose(s.mem_util_pct, 50.0, abs_tol=0.1)
            # sm_clock constant at 1500
            assert math.isclose(s.sm_clock_mean_mhz, 1500.0, abs_tol=0.1)
            assert math.isclose(s.sm_clock_max_mhz, 1500.0, abs_tol=0.1)
            # temp constant at 65
            assert math.isclose(s.temp_mean_c, 65.0, abs_tol=0.1)
            # energy = power_mean * duration
            assert s.energy_j > 0.0


class TestGpuMonitorWithRampBackend:
    """Tests with a ramping backend to verify aggregation correctness."""

    def _make_monitor(self, **kwargs) -> GpuMonitor:
        mon = GpuMonitor(backend="none", strict=False, **kwargs)
        mon._backend_obj = RampBackend(total_calls=200)
        return mon

    def test_max_near_100(self) -> None:
        mon = self._make_monitor(interval_s=0.005)
        with mon:
            time.sleep(0.5)
        s = mon.summary
        assert s is not None
        if not math.isnan(s.util_gpu_max):
            assert s.util_gpu_max > 30.0  # ramp starts at 0; must reach well above 30

    def test_mean_reasonable(self) -> None:
        mon = self._make_monitor(interval_s=0.005)
        with mon:
            time.sleep(0.5)
        s = mon.summary
        assert s is not None
        if not math.isnan(s.util_gpu_mean):
            assert 20.0 < s.util_gpu_mean < 80.0

    def test_ramp_extended_metrics(self) -> None:
        """Ramping util_gpu from 0→100 should have high std & mixed idle/active."""
        mon = self._make_monitor(interval_s=0.005)
        with mon:
            time.sleep(0.5)
        s = mon.summary
        assert s is not None
        if s.n_samples >= 10:
            # min should be near 0
            assert s.util_gpu_min < 20.0
            # std should be non-trivial (ramp has high variance)
            assert s.util_gpu_std > 10.0
            # Should have some idle samples (ramp starts at 0)
            assert s.idle_pct > 0.0


class TestGpuMonitorWithFailingBackend:
    """Tests that backend failures are handled correctly."""

    def _make_monitor(self, fail_after: int = 3, **kwargs) -> GpuMonitor:
        mon = GpuMonitor(backend="none", strict=False, **kwargs)
        mon._backend_obj = FailingBackend(fail_after=fail_after)
        return mon

    def test_non_strict_produces_summary(self) -> None:
        mon = self._make_monitor(fail_after=3, interval_s=0.01)
        with mon:
            time.sleep(0.15)
        s = mon.summary
        assert s is not None
        assert s.n_samples >= 1
        assert "sampling stopped early" in s.notes

    def test_strict_raises_on_thread_error(self) -> None:
        mon = self._make_monitor(fail_after=2, interval_s=0.01)
        mon.strict = True
        with pytest.raises(GpuBackendError, match="GPU sampling failed"):
            with mon:
                time.sleep(0.15)


class TestGpuMonitorStoreSamplesDownsampling:
    """Tests that sample storage doesn't grow unbounded."""

    def test_downsample_when_exceeding_limit(self) -> None:
        mon = GpuMonitor(
            backend="none",
            strict=False,
            interval_s=0.001,
            store_samples=True,
            max_samples=200,
        )
        mon._backend_obj = ConstantBackend()
        with mon:
            time.sleep(0.5)
        assert len(mon.samples) <= 200


class TestGpuMonitorSyncFn:
    """Tests for the sync_fn parameter."""

    def test_sync_fn_called(self) -> None:
        sync_calls: List[str] = []

        def fake_sync() -> None:
            sync_calls.append("synced")

        with GpuMonitor(
            backend="none",
            strict=False,
            interval_s=0.01,
            sync_fn=fake_sync,
        ):
            pass
        # sync_fn should be called on __enter__ and __exit__
        assert len(sync_calls) >= 2

    def test_sync_fn_exception_ignored(self) -> None:
        def bad_sync() -> None:
            raise RuntimeError("sync failed")

        with GpuMonitor(
            backend="none",
            strict=False,
            interval_s=0.01,
            sync_fn=bad_sync,
        ) as mon:
            pass
        assert mon.summary is not None


# ===================================================================
# gpu_profile decorator
# ===================================================================


class TestGpuProfileDecorator:
    """Tests for the ``@gpu_profile`` decorator."""

    def test_returns_original_value(self) -> None:
        @gpu_profile(backend="none", strict=False, report=False)
        def f(x: int) -> int:
            return x + 1

        assert f(1) == 2

    def test_return_profile(self) -> None:
        @gpu_profile(backend="none", strict=False, report=False, return_profile=True)
        def f(x: int) -> int:
            return x * 2

        result = f(5)
        assert isinstance(result, ProfiledResult)
        assert result.value == 10
        assert isinstance(result.gpu, GpuSummary)

    def test_report_true_prints(self, capsys: pytest.CaptureFixture) -> None:
        @gpu_profile(backend="none", strict=False, report=True, interval_s=0.01)
        def f() -> None:
            pass

        f()
        captured = capsys.readouterr()
        assert "[GPU" in captured.out

    def test_report_false_silent(self, capsys: pytest.CaptureFixture) -> None:
        @gpu_profile(backend="none", strict=False, report=False, interval_s=0.01)
        def f() -> None:
            pass

        f()
        assert capsys.readouterr().out == ""

    def test_report_callable(self) -> None:
        summaries: List[GpuSummary] = []

        @gpu_profile(backend="none", strict=False, report=summaries.append, interval_s=0.01)
        def f() -> str:
            return "hello"

        assert f() == "hello"
        assert len(summaries) == 1
        assert isinstance(summaries[0], GpuSummary)

    def test_preserves_function_metadata(self) -> None:
        @gpu_profile(backend="none", strict=False, report=False)
        def my_function() -> None:
            """My docstring."""

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    def test_with_args_and_kwargs(self) -> None:
        @gpu_profile(backend="none", strict=False, report=False)
        def add(a: int, b: int, offset: int = 0) -> int:
            return a + b + offset

        assert add(1, 2) == 3
        assert add(1, 2, offset=10) == 13

    def test_exception_propagates(self) -> None:
        @gpu_profile(backend="none", strict=False, report=False)
        def boom() -> None:
            raise ValueError("kaboom")

        with pytest.raises(ValueError, match="kaboom"):
            boom()


# ===================================================================
# Backend selection (_make_backend)
# ===================================================================


class TestBackendSelection:
    """Tests for backend auto-detection and explicit selection."""

    def test_explicit_none(self) -> None:
        mon = GpuMonitor(backend="none")
        assert isinstance(mon._backend_obj, NullGpuBackend)

    def test_auto_falls_back_gracefully(self) -> None:
        with mock.patch(
            "profgpu.monitor.NvidiaNvmlBackend.__init__",
            side_effect=GpuBackendError("no nvml"),
        ):
            with mock.patch(
                "profgpu.monitor.NvidiaSmiBackend.__init__",
                side_effect=GpuBackendError("no smi"),
            ):
                mon = GpuMonitor(backend="auto", strict=False)
                assert isinstance(mon._backend_obj, NullGpuBackend)

    def test_explicit_nvml_strict_raises(self) -> None:
        with mock.patch(
            "profgpu.monitor.NvidiaNvmlBackend.__init__",
            side_effect=GpuBackendError("no nvml"),
        ):
            with pytest.raises(GpuBackendError):
                GpuMonitor(backend="nvml", strict=True)

    def test_explicit_smi_strict_raises(self) -> None:
        with mock.patch(
            "profgpu.monitor.NvidiaSmiBackend.__init__",
            side_effect=GpuBackendError("no smi"),
        ):
            with pytest.raises(GpuBackendError):
                GpuMonitor(backend="smi", strict=True)


# ===================================================================
# CLI
# ===================================================================


class TestCli:
    """Tests for the command-line interface."""

    def test_no_command_returns_2(self) -> None:
        from profgpu.cli import main

        assert main([]) == 2

    def test_no_command_with_separator_returns_2(self) -> None:
        from profgpu.cli import main

        assert main(["--"]) == 2

    def test_basic_command(self) -> None:
        from profgpu.cli import main

        ret = main(["--backend", "none", "--", sys.executable, "-c", "pass"])
        assert ret == 0

    def test_json_output(self, capsys: pytest.CaptureFixture) -> None:
        from profgpu.cli import main

        ret = main(
            [
                "--backend",
                "none",
                "--json",
                "--",
                sys.executable,
                "-c",
                "pass",
            ]
        )
        assert ret == 0
        data = json.loads(capsys.readouterr().out)
        assert "device" in data
        assert "n_samples" in data

    def test_nonzero_exit_code_propagated(self) -> None:
        from profgpu.cli import main

        ret = main(
            [
                "--backend",
                "none",
                "--",
                sys.executable,
                "-c",
                "raise SystemExit(42)",
            ]
        )
        assert ret == 42

    def test_warmup_flag(self) -> None:
        from profgpu.cli import main

        ret = main(
            [
                "--backend",
                "none",
                "--warmup",
                "0.1",
                "--",
                sys.executable,
                "-c",
                "pass",
            ]
        )
        assert ret == 0

    def test_device_flag(self, capsys: pytest.CaptureFixture) -> None:
        from profgpu.cli import main

        ret = main(
            [
                "--backend",
                "none",
                "--device",
                "2",
                "--json",
                "--",
                sys.executable,
                "-c",
                "pass",
            ]
        )
        assert ret == 0
        data = json.loads(capsys.readouterr().out)
        assert data["device"] == 2

    def test_interval_flag(self) -> None:
        from profgpu.cli import main

        ret = main(
            [
                "--backend",
                "none",
                "--interval",
                "0.05",
                "--",
                sys.executable,
                "-c",
                "pass",
            ]
        )
        assert ret == 0

    def test_torch_sync_without_torch(self) -> None:
        from profgpu.cli import main

        with mock.patch.dict("sys.modules", {"torch": None}):
            with pytest.raises(SystemExit):
                main(
                    [
                        "--backend",
                        "none",
                        "--torch-sync",
                        "--",
                        sys.executable,
                        "-c",
                        "pass",
                    ]
                )


# ===================================================================
# NvidiaSmiBackend (unit-level, mocked)
# ===================================================================


class TestNvidiaSmiBackendMocked:
    """Tests for NvidiaSmiBackend with mocked subprocess calls."""

    def test_init_fails_without_nvidia_smi(self) -> None:
        with mock.patch("shutil.which", return_value=None):
            with pytest.raises(GpuBackendError, match="nvidia-smi not found"):
                NvidiaSmiBackend(device=0)

    def test_read_parses_csv(self) -> None:
        with mock.patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            with mock.patch("subprocess.check_output", return_value="TestGPU\n"):
                backend = NvidiaSmiBackend(device=0)

        csv_line = "85, 40, 6144, 16384, 250.5, 72, 1800, 1200\n"
        with mock.patch("subprocess.check_output", return_value=csv_line):
            metrics = backend.read()

        assert metrics["util_gpu"] == 85.0
        assert metrics["util_mem"] == 40.0
        assert metrics["mem_used_mb"] == 6144.0
        assert metrics["mem_total_mb"] == 16384.0
        assert math.isclose(metrics["power_w"], 250.5)
        assert metrics["temp_c"] == 72.0
        assert metrics["sm_clock_mhz"] == 1800.0
        assert metrics["mem_clock_mhz"] == 1200.0

    def test_read_handles_na(self) -> None:
        with mock.patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            with mock.patch("subprocess.check_output", return_value="TestGPU\n"):
                backend = NvidiaSmiBackend(device=0)

        csv_line = "N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A\n"
        with mock.patch("subprocess.check_output", return_value=csv_line):
            metrics = backend.read()

        for k in METRIC_KEYS:
            assert metrics[k] is None

    def test_read_failure_raises(self) -> None:
        with mock.patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            with mock.patch("subprocess.check_output", return_value="TestGPU\n"):
                backend = NvidiaSmiBackend(device=0)

        with mock.patch(
            "subprocess.check_output",
            side_effect=subprocess.CalledProcessError(1, "nvidia-smi"),
        ):
            with pytest.raises(GpuBackendError, match="nvidia-smi query failed"):
                backend.read()

    def test_device_name_fallback(self) -> None:
        """If the name query fails, the default name is used."""
        with mock.patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            with mock.patch(
                "subprocess.check_output",
                side_effect=subprocess.CalledProcessError(1, "nvidia-smi"),
            ):
                backend = NvidiaSmiBackend(device=3)
        assert "3" in backend.device_name()


# ===================================================================
# Package exports
# ===================================================================


class TestPackageExports:
    """Verify that the public API is correctly exported."""

    def test_all_exports(self) -> None:
        import profgpu

        for name in [
            "GpuBackendError",
            "GpuMonitor",
            "GpuSample",
            "GpuSummary",
            "MultiRunResult",
            "ProfiledResult",
            "RunStats",
            "gpu_profile",
            "profile_repeats",
        ]:
            assert hasattr(profgpu, name)

    def test_version(self) -> None:
        import profgpu

        assert isinstance(profgpu.__version__, str)
        assert len(profgpu.__version__) > 0

    def test_all_list_matches_exports(self) -> None:
        import profgpu

        for name in profgpu.__all__:
            assert hasattr(profgpu, name), f"__all__ contains {name!r} but it's not exported"


# ===================================================================
# Edge cases and integration
# ===================================================================


class TestEdgeCases:
    """Miscellaneous edge-case tests."""

    def test_very_short_monitoring(self) -> None:
        with GpuMonitor(backend="none", strict=False, interval_s=0.01) as mon:
            pass
        assert mon.summary is not None

    def test_summary_json_serializable(self) -> None:
        with GpuMonitor(backend="none", strict=False, interval_s=0.01) as mon:
            time.sleep(0.03)
        d = mon.summary.__dict__
        text = json.dumps(
            d,
            default=lambda x: None if isinstance(x, float) and math.isnan(x) else x,
        )
        assert isinstance(text, str)

    def test_concurrent_monitors(self) -> None:
        with GpuMonitor(backend="none", strict=False, interval_s=0.01) as mon1:
            with GpuMonitor(backend="none", strict=False, interval_s=0.01) as mon2:
                time.sleep(0.05)
        assert mon1.summary is not None
        assert mon2.summary is not None

    def test_busy_time_estimate(self) -> None:
        mon = GpuMonitor(backend="none", strict=False, interval_s=0.01)
        mon._backend_obj = ConstantBackend(util_gpu=50.0)
        with mon:
            time.sleep(0.1)
        s = mon.summary
        if not math.isnan(s.util_gpu_mean) and not math.isnan(s.busy_time_est_s):
            expected = s.duration_s * (s.util_gpu_mean / 100.0)
            assert abs(s.busy_time_est_s - expected) < 0.15

    def test_format_without_sparkline(self) -> None:
        s = _make_summary(n_samples=0, sparkline="")
        assert "util trace" not in s.format()

    def test_format_without_notes(self) -> None:
        s = _make_summary(notes="")
        assert "notes:" not in s.format()

    def test_format_memory_hidden_when_nan(self) -> None:
        """When mem_total_mb is NaN the memory line should be omitted."""
        nan = float("nan")
        s = _make_summary(
            mem_used_mean_mb=nan,
            mem_used_max_mb=nan,
            mem_total_mb=nan,
            mem_util_pct=nan,
        )
        assert "memory:" not in s.format()


# ===================================================================
# RunStats
# ===================================================================


class TestRunStats:
    """Tests for the RunStats dataclass and _compute_run_stats helper."""

    def test_single_value(self) -> None:
        rs = _compute_run_stats([42.0])
        assert math.isclose(rs.mean, 42.0)
        assert math.isnan(rs.std)  # can't compute std from 1 sample
        assert math.isclose(rs.min, 42.0)
        assert math.isclose(rs.max, 42.0)

    def test_two_values(self) -> None:
        rs = _compute_run_stats([40.0, 60.0])
        assert math.isclose(rs.mean, 50.0)
        assert rs.std > 0
        assert math.isclose(rs.min, 40.0)
        assert math.isclose(rs.max, 60.0)

    def test_constant_values(self) -> None:
        rs = _compute_run_stats([10.0, 10.0, 10.0])
        assert math.isclose(rs.mean, 10.0)
        assert math.isclose(rs.std, 0.0, abs_tol=1e-12)

    def test_empty_returns_nan(self) -> None:
        rs = _compute_run_stats([])
        assert math.isnan(rs.mean)

    def test_all_nan_returns_nan(self) -> None:
        rs = _compute_run_stats([float("nan"), float("nan")])
        assert math.isnan(rs.mean)

    def test_mixed_nan_ignored(self) -> None:
        rs = _compute_run_stats([10.0, float("nan"), 30.0])
        assert math.isclose(rs.mean, 20.0)

    def test_values_tuple_preserved(self) -> None:
        rs = _compute_run_stats([1.0, 2.0, 3.0])
        assert rs.values == (1.0, 2.0, 3.0)

    def test_format_output(self) -> None:
        rs = _compute_run_stats([50.0, 60.0, 70.0])
        text = rs.format("%", 1)
        assert "±" in text
        assert "range" in text

    def test_frozen(self) -> None:
        rs = _compute_run_stats([1.0, 2.0])
        with pytest.raises(AttributeError):
            rs.mean = 99.0  # type: ignore[misc]


# ===================================================================
# MultiRunResult
# ===================================================================


class TestMultiRunResult:
    """Tests for the MultiRunResult dataclass."""

    @staticmethod
    def _make_summaries(n: int = 3) -> List[GpuSummary]:
        return [
            _make_summary(
                duration_s=1.0 + i * 0.1,
                util_gpu_mean=50.0 + i * 5.0,
                power_mean_w=100.0 + i * 10.0,
                energy_j=100.0 + i * 10.0,
                mem_used_max_mb=4000.0 + i * 100.0,
                temp_max_c=60.0 + i * 2.0,
            )
            for i in range(n)
        ]

    def test_from_runs(self) -> None:
        runs = self._make_summaries(3)
        result = MultiRunResult.from_runs(runs, value=42)
        assert result.value == 42
        assert len(result.runs) == 3

    def test_cross_run_stats(self) -> None:
        runs = self._make_summaries(3)
        result = MultiRunResult.from_runs(runs, value=None)
        assert math.isclose(result.util_gpu.mean, 55.0, abs_tol=0.1)
        assert result.util_gpu.std > 0

    def test_stats_for_arbitrary_field(self) -> None:
        runs = self._make_summaries(3)
        result = MultiRunResult.from_runs(runs, value=None)
        rs = result.stats_for("util_gpu_p50")
        assert not math.isnan(rs.mean)

    def test_format(self) -> None:
        runs = self._make_summaries(3)
        result = MultiRunResult.from_runs(runs, value=None)
        text = result.format()
        assert "Multi-Run" in text
        assert "3 runs" in text
        assert "util.gpu" in text
        assert "run 1" in text

    def test_empty_runs(self) -> None:
        result = MultiRunResult.from_runs([], value=None)
        assert result.format() == "(no runs)"

    def test_frozen(self) -> None:
        result = MultiRunResult.from_runs(self._make_summaries(2), value=1)
        with pytest.raises(AttributeError):
            result.value = 99  # type: ignore[misc]


# ===================================================================
# Multi-run decorator and profile_repeats
# ===================================================================


class TestMultiRunDecorator:
    """Tests for @gpu_profile(repeats=N) and profile_repeats()."""

    def test_repeats_returns_multi_run_result(self) -> None:
        @gpu_profile(
            backend="none",
            strict=False,
            report=False,
            return_profile=True,
            repeats=3,
        )
        def f() -> int:
            return 42

        result = f()
        assert isinstance(result, MultiRunResult)
        assert len(result.runs) == 3
        assert result.value == 42

    def test_repeats_without_return_profile(self) -> None:
        @gpu_profile(
            backend="none",
            strict=False,
            report=False,
            repeats=2,
        )
        def f() -> int:
            return 7

        assert f() == 7

    def test_warmup_runs_excluded(self) -> None:
        call_count = 0

        @gpu_profile(
            backend="none",
            strict=False,
            report=False,
            return_profile=True,
            repeats=2,
            warmup_runs=1,
        )
        def f() -> int:
            nonlocal call_count
            call_count += 1
            return call_count

        result = f()
        assert isinstance(result, MultiRunResult)
        assert len(result.runs) == 2  # only 2 counted
        assert call_count == 3  # 1 warmup + 2 measured
        assert result.value == 3

    def test_repeats_report_prints(self, capsys: pytest.CaptureFixture) -> None:
        @gpu_profile(
            backend="none",
            strict=False,
            report=True,
            repeats=2,
            interval_s=0.01,
        )
        def f() -> None:
            pass

        f()
        captured = capsys.readouterr()
        assert "Multi-Run" in captured.out

    def test_single_repeat_returns_profiled_result(self) -> None:
        @gpu_profile(
            backend="none",
            strict=False,
            report=False,
            return_profile=True,
            repeats=1,
        )
        def f() -> int:
            return 5

        result = f()
        assert isinstance(result, ProfiledResult)
        assert result.value == 5

    def test_profile_repeats_function(self) -> None:
        call_count = 0

        def work() -> int:
            nonlocal call_count
            call_count += 1
            return call_count

        result = profile_repeats(
            work,
            repeats=3,
            warmup_runs=1,
            backend="none",
            strict=False,
            report=False,
        )
        assert isinstance(result, MultiRunResult)
        assert len(result.runs) == 3
        assert call_count == 4  # 1 warmup + 3 measured

    def test_profile_repeats_report(self, capsys: pytest.CaptureFixture) -> None:
        result = profile_repeats(
            lambda: None,
            repeats=2,
            backend="none",
            strict=False,
            report=True,
            interval_s=0.01,
        )
        assert isinstance(result, MultiRunResult)
        assert "Multi-Run" in capsys.readouterr().out

    def test_invalid_repeats_raises(self) -> None:
        with pytest.raises(ValueError, match="repeats"):
            gpu_profile(backend="none", repeats=0)

    def test_invalid_warmup_runs_raises(self) -> None:
        with pytest.raises(ValueError, match="warmup_runs"):
            gpu_profile(backend="none", warmup_runs=-1)
