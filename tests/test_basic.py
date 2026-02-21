from __future__ import annotations

from gpu_profile import GpuMonitor, gpu_profile


def test_monitor_none_backend_does_not_crash():
    with GpuMonitor(backend="none", strict=False, interval_s=0.01) as mon:
        # no-op
        pass
    s = mon.summary
    assert s is not None
    assert s.duration_s >= 0.0
    # With backend=none, metrics are missing; the key check is "no crash".


def test_decorator_returns_value():
    @gpu_profile(backend="none", strict=False, report=False)
    def f(x: int) -> int:
        return x + 1

    assert f(1) == 2
