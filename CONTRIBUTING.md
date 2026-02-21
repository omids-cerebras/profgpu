# Contributing

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Linting

This repo uses `ruff`.

```bash
ruff check .
```

## Documentation

Docs are built with MkDocs.

```bash
pip install -e ".[docs]"
mkdocs serve
```

## Adding examples / notebooks

- Keep examples self-contained.
- Avoid hard-pinning large dependencies.
- In notebooks, guard GPU-only cells with checks like:

```python
import torch
if not torch.cuda.is_available():
    print("CUDA not available")
```

## Design goals

- Minimal runtime dependencies.
- Clear, stable API.
- Helpful defaults (auto backend + reasonable sampling).
- “Pretty” but loggable outputs.

PRs that add AMD/Intel backends are welcome.
