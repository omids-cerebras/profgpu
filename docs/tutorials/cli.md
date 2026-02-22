# Tutorial: CLI

The CLI is useful when you want to profile:

- a training script
- an inference job
- a benchmark binary
- any command you can run in a shell

The CLI wraps your command, samples GPU metrics while it runs, and prints a summary at the end.

## Basic usage

```bash
profgpu -- python train.py --epochs 3
```

Use `--` to separate `profgpu` arguments from your command’s arguments.

## Choosing device and interval

```bash
profgpu --device 0 --interval 0.1 -- python train.py
```

## JSON output

To parse in scripts or logs:

```bash
profgpu --json -- python train.py
```

Example output:

```json
{"device":0,"duration_s":12.34,"util_gpu_mean":87.2, ...}
```

## Backend control

Force NVML:

```bash
profgpu --backend nvml -- python train.py
```

Force `nvidia-smi`:

```bash
profgpu --backend smi -- python train.py
```

Disable sampling (dry run):

```bash
profgpu --backend none -- python train.py
```

## CUDA async and `--torch-sync`

If your command uses PyTorch, you can request synchronization at the boundaries:

```bash
profgpu --torch-sync -- python train.py
```

This is equivalent to passing `sync_fn=torch.cuda.synchronize` in library mode.

## Exit codes

`profgpu` returns your command’s exit code.

That means you can safely use it in CI:

```bash
profgpu -- python -m pytest
```

## Profiling shell pipelines

Wrap the whole pipeline in `bash -lc`:

```bash
profgpu -- bash -lc "python train.py | tee train.log"
```

## Capturing JSON + exit code in bash

```bash
set -euo pipefail

out=$(profgpu --json -- python train.py)
code=$?

echo "$out" >> gpu.jsonl
exit $code
```

If your command fails, you still get the last summary that was printed.

---

See also:

- [Logging & Export](logging.md)
- [Troubleshooting](../troubleshooting.md)
