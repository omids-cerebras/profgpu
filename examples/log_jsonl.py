"""Example: write profgpu summaries to JSONL.

Run:
  python examples/log_jsonl.py

This example is intentionally minimal and uses time.sleep() as the "work".
On a real GPU workload, you will see non-zero utilization.
"""

import json
import time

from profgpu import gpu_profile


def write_jsonl(summary):
    with open("profgpu.jsonl", "a") as f:
        f.write(json.dumps(summary.__dict__) + "\n")


@gpu_profile(report=write_jsonl, interval_s=0.5, strict=False)
def work():
    time.sleep(2)


if __name__ == "__main__":
    work()
    print("Wrote profgpu.jsonl")
