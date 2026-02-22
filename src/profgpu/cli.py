"""Command-line interface for the ``profgpu`` tool.

The CLI wraps an arbitrary shell command, profiles GPU utilisation while
it runs, and prints a summary when the command exits.  Example::

    profgpu --device 0 --interval 0.1 -- python train.py

Use ``--json`` for machine-readable output.  The exit code mirrors the
child process.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Callable, Optional

from .monitor import GpuMonitor


def _maybe_torch_sync(enabled: bool) -> Optional[Callable[[], None]]:
    """Return ``torch.cuda.synchronize`` if *enabled*, else ``None``.

    Raises :class:`SystemExit` when ``--torch-sync`` is requested but
    PyTorch cannot be imported.
    """
    if not enabled:
        return None
    try:
        import torch  # type: ignore[import-untyped]
    except Exception as exc:
        raise SystemExit(
            "Error: --torch-sync requested but torch is not importable in this environment.\n"
            "Install torch or omit --torch-sync."
        ) from exc
    return torch.cuda.synchronize  # type: ignore[no-any-return]


def main(argv: Optional[list[str]] = None) -> int:
    """Entry-point for the ``profgpu`` CLI.

    Parameters
    ----------
    argv:
        Argument list to parse.  Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code: mirrors the profiled command's return code, or
        **2** if no command was supplied.
    """
    parser = argparse.ArgumentParser(
        prog="profgpu",
        description=(
            "Profile GPU utilization while running a command "
            "(or import as a library for decorators / context managers)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", type=int, default=0, help="GPU index")
    parser.add_argument("--interval", type=float, default=0.2, help="Sampling interval (seconds)")
    parser.add_argument(
        "--backend",
        choices=["auto", "nvml", "smi", "none"],
        default="auto",
        help="Sampling backend",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise error if no backend / sampling fails",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=0.0,
        help="Ignore first N seconds in stats",
    )
    parser.add_argument(
        "--torch-sync",
        action="store_true",
        help="Call torch.cuda.synchronize() before/after the run",
    )
    parser.add_argument("--json", action="store_true", help="Output summary as JSON on stdout")
    parser.add_argument(
        "cmd",
        nargs=argparse.REMAINDER,
        help=("Command to run (use `--` before the command). Example: profgpu -- python train.py"),
    )

    args = parser.parse_args(argv)

    # Strip the optional leading "--" separator.
    cmd: list[str] = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]

    if not cmd:
        parser.print_usage(sys.stderr)
        print(
            "\nExample:\n  profgpu --device 0 --interval 0.2 -- python train.py\n",
            file=sys.stderr,
        )
        return 2

    sync_fn = _maybe_torch_sync(args.torch_sync)

    # Run the child command under GPU monitoring.
    with GpuMonitor(
        device=args.device,
        interval_s=args.interval,
        backend=args.backend,
        strict=args.strict,
        sync_fn=sync_fn,
        warmup_s=args.warmup,
    ) as mon:
        proc = subprocess.run(cmd)

    summary = mon.summary
    assert summary is not None, "summary should be set after GpuMonitor.__exit__"

    if args.json:
        print(json.dumps(summary.__dict__, sort_keys=True))
    else:
        print(summary.format())

    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
