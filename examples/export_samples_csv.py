"""Example: export raw GPU samples to CSV.

Run:
  python examples/export_samples_csv.py

This uses ``store_samples=True`` to retain the time-series of
:class:`~profgpu.GpuSample` objects for later export.
"""

import csv
import time

from profgpu import GpuMonitor


def main() -> None:
    with GpuMonitor(interval_s=0.2, store_samples=True, strict=False) as mon:
        time.sleep(5)

    print(mon.summary.format())

    with open("gpu_samples.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "util_gpu", "util_mem", "mem_used_mb", "power_w", "temp_c"])
        for s in mon.samples:
            w.writerow([s.t_s, s.util_gpu, s.util_mem, s.mem_used_mb, s.power_w, s.temp_c])

    print("Wrote gpu_samples.csv")


if __name__ == "__main__":
    main()
