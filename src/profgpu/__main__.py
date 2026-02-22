"""Allow running the package as ``python -m profgpu``.

Delegates to :func:`profgpu.cli.main`.
"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
