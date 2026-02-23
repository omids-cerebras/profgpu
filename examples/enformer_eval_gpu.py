#!/usr/bin/env python
"""Run Enformer in eval mode on 100 random sequences and plot GPU utilization.

This script:
1. Downloads pretrained Enformer weights from HuggingFace.
2. Runs inference on 100 random one-hot-encoded DNA sequences (196 608 bp each).
3. Uses profgpu.profile_repeats (10 repeats per sequence) for robust GPU metrics.
4. Produces per-sequence and aggregate GPU utilization figures via seaborn/matplotlib.

Usage:
    conda activate profgpu
    python examples/enformer_eval_gpu.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from enformer_pytorch import SEQUENCE_LENGTH, Enformer, from_pretrained
from torch.utils.flop_counter import FlopCounterMode

from profgpu import MultiRunResult, profile_repeats

# ── seaborn style ─────────────────────────────────────────────────────────
sns.set_theme(
    style="whitegrid",
    font_scale=1.4,
    rc={
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    },
)
PAL = sns.color_palette("deep")


# ── constants ────────────────────────────────────────────────────────────────
NUM_SEQUENCES = 100
NUM_REPEATS = 10  # repeats per sequence for robust profiling
WARMUP_RUNS = 1  # warmup repeats (excluded from stats)
BATCH_SIZE = 1  # Enformer is large; run one at a time
OUTPUT_DIR = Path(__file__).resolve().parent / "enformer_gpu_results"
SEQ_LEN = SEQUENCE_LENGTH  # 196_608
NUM_BASES = 4  # A, C, G, T


# ── GPU peak TFLOPS estimation ───────────────────────────────────────────────

# FP32 CUDA cores per SM, keyed by (compute_capability_major, minor).
# Source: NVIDIA CUDA Programming Guide, Table "Technical Specifications".
_CORES_PER_SM: dict[tuple[int, int], int] = {
    (7, 0): 64,  # V100
    (7, 5): 64,  # T4, RTX 2080
    (8, 0): 64,  # A100
    (8, 6): 128,  # A10G, RTX 3090
    (8, 7): 128,  # Jetson Orin
    (8, 9): 128,  # L4, L40, RTX 4090
    (9, 0): 128,  # H100
}


def _get_max_clock_mhz(device: int = 0) -> float:
    """Return max SM clock in MHz (pynvml → nvidia-smi → fallback)."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        mhz = float(pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM))
        pynvml.nvmlShutdown()
        return mhz
    except Exception:
        pass
    import subprocess

    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=clocks.max.sm",
                "--format=csv,noheader,nounits",
                f"--id={device}",
            ],
            text=True,
        ).strip()
        return float(out)
    except Exception:
        return 1500.0  # conservative fallback


def theoretical_peak_tflops(device: int = 0) -> float:
    """Estimate peak FP32 TFLOPS from CUDA device properties.

    Formula: SMs × cores_per_SM × 2 (FMA) × clock_Hz / 1e12

    Uses pynvml for the max boost clock (falls back to nvidia-smi).
    """
    props = torch.cuda.get_device_properties(device)
    cc = (props.major, props.minor)
    cores = _CORES_PER_SM.get(cc)
    if cores is None:
        cores = 128 if props.major >= 8 else 64

    clock_hz = _get_max_clock_mhz(device) * 1e6
    peak = props.multi_processor_count * cores * 2 * clock_hz / 1e12
    return peak


def empirical_peak_tflops(
    device: int = 0,
    warmup: int = 20,
    iters: int = 100,
    dtype: torch.dtype = torch.float32,
) -> dict[str, float]:
    """Measure empirical peak TFLOPS by running large GEMMs.

    Sweeps matrix sizes (2048, 4096, 8192) with TF32 on and off.
    Returns dict with keys "fp32" (CUDA cores only) and "tf32" (Tensor Cores).

    On Ampere+, NVIDIA's "FP32" datasheet number typically uses TF32 Tensor
    Cores (inputs/outputs are float32 but mantissa is rounded to TF32).
    """
    dev = torch.device(f"cuda:{device}")
    results: dict[str, float] = {"fp32": 0.0, "tf32": 0.0}

    for n in (2048, 4096, 8192):
        a = torch.randn(n, n, device=dev, dtype=dtype)
        b = torch.randn(n, n, device=dev, dtype=dtype)
        flops_per_mm = 2 * n * n * n

        for label, tf32_flag in (("fp32", False), ("tf32", True)):
            torch.backends.cuda.matmul.allow_tf32 = tf32_flag
            for _ in range(warmup):
                torch.mm(a, b)
            torch.cuda.synchronize(dev)
            start = time.perf_counter()
            for _ in range(iters):
                torch.mm(a, b)
            torch.cuda.synchronize(dev)
            elapsed = time.perf_counter() - start
            tflops = flops_per_mm * iters / elapsed / 1e12
            results[label] = max(results[label], tflops)

        del a, b

    torch.backends.cuda.matmul.allow_tf32 = False  # restore conservative default
    torch.cuda.empty_cache()
    return results


def make_random_sequence(device: torch.device) -> torch.Tensor:
    """Return a random one-hot encoded DNA sequence (1, 196608, 4)."""
    indices = torch.randint(0, NUM_BASES, (BATCH_SIZE, SEQ_LEN), device=device)
    return torch.nn.functional.one_hot(indices, NUM_BASES).float()


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(
        f"Config: {NUM_SEQUENCES} sequences x {NUM_REPEATS} repeats "
        f"({WARMUP_RUNS} warmup run{'s' if WARMUP_RUNS != 1 else ''} each)"
    )

    # ── load model ───────────────────────────────────────────────────────
    print("Loading pretrained Enformer weights …")
    model: Enformer = from_pretrained("EleutherAI/enformer-official-rough")
    model = model.to(device).eval()
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── count FLOPs per forward pass ──────────────────────────────────────
    print("Counting FLOPs per forward pass …")
    flop_seq = make_random_sequence(device)
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        _ = model(flop_seq)
    flops_per_pass = fc.get_total_flops()
    print(f"FLOPs per forward pass: {flops_per_pass:,} ({flops_per_pass / 1e12:.2f} TFLOP)")
    del flop_seq

    # ── GPU peak TFLOPS (estimated programmatically) ─────────────────────
    gpu_name = torch.cuda.get_device_name(0)
    precision = os.environ.get("PRECISION", "fp32").lower()
    if precision not in ("fp32", "tf32", "fp16", "bf16"):
        precision = "fp32"
    if precision == "bf16":
        precision = "fp16"  # same Tensor Core peak

    # 1) Theoretical peak from CUDA device properties
    theo_tflops = theoretical_peak_tflops(0)
    print(f"GPU: {gpu_name}")
    props = torch.cuda.get_device_properties(0)
    print(
        f"  SMs: {props.multi_processor_count}  "
        f"Compute: {props.major}.{props.minor}  "
        f"Max SM clock: {_get_max_clock_mhz(0):.0f} MHz"
    )
    print(f"  Theoretical peak FP32: {theo_tflops:.1f} TFLOPS")

    # 2) Empirical peak from large GEMM benchmark
    print("  Running GEMM benchmark (N=2048,4096,8192 x 100 iters) ...")
    emp = empirical_peak_tflops(0)
    print(f"  Empirical peak FP32 (CUDA cores):  {emp['fp32']:.1f} TFLOPS")
    print(f"  Empirical peak TF32 (Tensor Cores): {emp['tf32']:.1f} TFLOPS")

    # Select the appropriate ceiling
    if precision == "fp32":
        # Enformer runs in FP32; check if TF32 is enabled
        tf32_enabled = torch.backends.cuda.matmul.allow_tf32
        if tf32_enabled:
            peak_tflops = emp["tf32"]
            peak_label = "TF32 (empirical)"
        else:
            peak_tflops = emp["fp32"]
            peak_label = "FP32 (empirical)"
    elif precision == "tf32":
        peak_tflops = emp["tf32"]
        peak_label = "TF32 (empirical)"
    else:
        # FP16/BF16: scale from TF32 empirical (rough 2× for half-precision TC)
        peak_tflops = emp["tf32"] * 2.0
        peak_label = "FP16 (estimated from TF32)"

    print(f"  -> Using peak: {peak_tflops:.1f} TFLOPS [{peak_label}]")

    # ── global warm-up pass ──────────────────────────────────────────────
    print("Global warm-up pass …")
    with torch.no_grad():
        _ = model(make_random_sequence(device))
    torch.cuda.synchronize()

    # ── eval loop with profile_repeats per sequence ──────────────────────
    print("\nRunning inference …")

    # Per-sequence aggregated stats (mean over 10 repeats)
    gpu_utils: list[float] = []
    mem_peaks: list[float] = []
    power_draws: list[float] = []
    temps: list[float] = []
    durations: list[float] = []
    energies: list[float] = []

    # Per-sequence FLOPS utilization
    flops_utils: list[float] = []  # achieved TFLOPS / peak TFLOPS × 100
    achieved_tflops_list: list[float] = []

    # Per-sequence std from repeats (shows run-to-run variance)
    gpu_util_stds: list[float] = []
    duration_stds: list[float] = []

    for i in range(NUM_SEQUENCES):
        seq = make_random_sequence(device)

        def run_inference(seq=seq) -> torch.Tensor:
            with torch.no_grad():
                return model(seq)

        result: MultiRunResult = profile_repeats(
            run_inference,
            repeats=NUM_REPEATS,
            warmup_runs=WARMUP_RUNS,
            device=0,
            interval_s=0.05,
            sync_fn=torch.cuda.synchronize,
            report=False,
        )

        gpu_utils.append(result.util_gpu.mean)
        gpu_util_stds.append(result.util_gpu.std)
        mem_peaks.append(result.peak_memory.mean)
        power_draws.append(result.power.mean)
        temps.append(result.peak_temp.mean)
        durations.append(result.duration.mean)
        duration_stds.append(result.duration.std)
        energies.append(result.energy.mean)

        # FLOPS utilization: (FLOPs / time) / peak_FLOPS × 100
        achieved_tflops = (flops_per_pass / result.duration.mean) / 1e12
        flops_util_pct = (achieved_tflops / peak_tflops) * 100.0
        achieved_tflops_list.append(achieved_tflops)
        flops_utils.append(flops_util_pct)

        if (i + 1) % 10 == 0:
            print(
                f"  [{i + 1:3d}/{NUM_SEQUENCES}]  "
                f"gpu={result.util_gpu.mean:.0f}%±{result.util_gpu.std:.0f}  "
                f"FLOPS={flops_util_pct:.1f}%  "
                f"mem={result.peak_memory.mean:.0f}MB  "
                f"power={result.power.mean:.0f}W  "
                f"time={result.duration.mean:.3f}s±{result.duration.std:.3f}"
            )

    # ── aggregate stats ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"Aggregate GPU Statistics  ({NUM_SEQUENCES} sequences x {NUM_REPEATS} repeats)")
    print("=" * 70)
    print(
        f"  FLOPS Util      : {np.mean(flops_utils):.1f}% of peak FP32  "
        f"(min {np.min(flops_utils):.1f}%, max {np.max(flops_utils):.1f}%)"
    )
    print(f"  Achieved TFLOPS : {np.mean(achieved_tflops_list):.2f} / {peak_tflops} TFLOPS")
    print(
        f"  NVML GPU Busy   : {np.mean(gpu_utils):.1f}% ± {np.std(gpu_utils):.1f}%  "
        f"(min {np.min(gpu_utils):.0f}%, max {np.max(gpu_utils):.0f}%)"
    )
    print(f"    repeat std avg: {np.mean(gpu_util_stds):.1f}%")
    print(
        f"  Peak Memory     : {np.mean(mem_peaks):.0f} MB ± {np.std(mem_peaks):.0f} MB  "
        f"(max {np.max(mem_peaks):.0f} MB)"
    )
    print(
        f"  Power Draw      : {np.mean(power_draws):.1f} W ± {np.std(power_draws):.1f} W  "
        f"(max {np.max(power_draws):.0f} W)"
    )
    print(
        f"  Temperature     : {np.mean(temps):.1f}°C ± {np.std(temps):.1f}°C  "
        f"(max {np.max(temps):.0f}°C)"
    )
    print(f"  Energy          : {np.mean(energies):.2f} J ± {np.std(energies):.2f} J / inference")
    print(f"  Inference Time  : {np.mean(durations):.3f}s ± {np.std(durations):.3f}s / sequence")
    print(f"    repeat std avg: {np.mean(duration_stds):.4f}s")
    print(f"  Total Inferences: {NUM_SEQUENCES * NUM_REPEATS}")
    print("=" * 70)

    # ── plotting ─────────────────────────────────────────────────────────
    seq_ids = np.arange(1, NUM_SEQUENCES + 1)
    MEAN_C = sns.color_palette("bright")[3]  # red-ish for mean lines

    # --- Figure 1: FLOPS Utilization per sequence ---
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(seq_ids, flops_utils, color=PAL[9], edgecolor="none", alpha=0.85)
    ax.axhline(
        np.mean(flops_utils),
        color=MEAN_C,
        ls="--",
        lw=2,
        label=f"Mean: {np.mean(flops_utils):.1f}% of {peak_tflops} TFLOPS",
    )
    ax.set_xlabel("Sequence #")
    ax.set_ylabel(f"FLOPS Utilization (% of {peak_tflops} TFLOPS {precision.upper()})")
    ax.set_title(f"Enformer Eval - FLOPS Utilization ({NUM_REPEATS} repeats/seq)")
    ax.set_ylim(0, max(np.max(flops_utils) * 1.15, 50))
    ax.legend(frameon=True)
    sns.despine(left=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "flops_utilization_per_seq.png")
    print(f"\nSaved: {OUTPUT_DIR / 'flops_utilization_per_seq.png'}")

    # --- Figure 2: NVML GPU Busy % per sequence (with repeat error bars) ---
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(
        seq_ids,
        gpu_utils,
        yerr=gpu_util_stds,
        capsize=1.5,
        color=PAL[0],
        edgecolor="none",
        alpha=0.85,
        error_kw={"lw": 0.8},
    )
    ax.axhline(
        np.mean(gpu_utils), color=MEAN_C, ls="--", lw=2, label=f"Mean: {np.mean(gpu_utils):.1f}%"
    )
    ax.set_xlabel("Sequence #")
    ax.set_ylabel("NVML GPU Busy (%)\nmean +/- std over repeats")
    ax.set_title(f"Enformer Eval - NVML GPU Busy ({NUM_REPEATS} repeats/seq)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=True)
    sns.despine(left=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "gpu_busy_per_seq.png")
    print(f"Saved: {OUTPUT_DIR / 'gpu_busy_per_seq.png'}")

    # --- Figure 2: Peak Memory per sequence ---
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(seq_ids, mem_peaks, "o-", ms=4, color=PAL[1], lw=1.2)
    ax.axhline(
        np.mean(mem_peaks), color=MEAN_C, ls="--", lw=2, label=f"Mean: {np.mean(mem_peaks):.0f} MB"
    )
    ax.set_xlabel("Sequence #")
    ax.set_ylabel("Peak GPU Memory (MB)")
    ax.set_title(f"Enformer Eval - Peak Memory ({NUM_REPEATS} repeats/seq)")
    ax.legend(frameon=True)
    sns.despine(left=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "gpu_memory_per_seq.png")
    print(f"Saved: {OUTPUT_DIR / 'gpu_memory_per_seq.png'}")

    # --- Figure 3: Power Draw per sequence ---
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.fill_between(seq_ids, power_draws, alpha=0.3, color=PAL[2])
    ax.plot(seq_ids, power_draws, ".-", ms=4, color=PAL[2], lw=1.2)
    ax.axhline(
        np.mean(power_draws),
        color=MEAN_C,
        ls="--",
        lw=2,
        label=f"Mean: {np.mean(power_draws):.1f} W",
    )
    ax.set_xlabel("Sequence #")
    ax.set_ylabel("Power Draw (W)")
    ax.set_title(f"Enformer Eval - Power Draw ({NUM_REPEATS} repeats/seq)")
    ax.legend(frameon=True)
    sns.despine(left=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "gpu_power_per_seq.png")
    print(f"Saved: {OUTPUT_DIR / 'gpu_power_per_seq.png'}")

    # --- Figure 4: Temperature per sequence ---
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(seq_ids, temps, "s-", ms=4, color=PAL[3], lw=1.2)
    ax.axhline(np.mean(temps), color="grey", ls="--", lw=2, label=f"Mean: {np.mean(temps):.1f}°C")
    ax.set_xlabel("Sequence #")
    ax.set_ylabel("Peak Temperature (°C)")
    ax.set_title(f"Enformer Eval - Temperature ({NUM_REPEATS} repeats/seq)")
    ax.legend(frameon=True)
    sns.despine(left=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "gpu_temperature_per_seq.png")
    print(f"Saved: {OUTPUT_DIR / 'gpu_temperature_per_seq.png'}")

    # --- Figure 6: Combined dashboard ---
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))

    axes[0, 0].bar(seq_ids, flops_utils, color=PAL[9], edgecolor="none", alpha=0.85)
    axes[0, 0].axhline(np.mean(flops_utils), color=MEAN_C, ls="--", lw=1.5)
    axes[0, 0].set_ylabel(f"% of {peak_tflops} TFLOPS")
    axes[0, 0].set_title(f"FLOPS Utilization ({np.mean(flops_utils):.1f}%)")
    axes[0, 0].set_ylim(0, max(np.max(flops_utils) * 1.15, 50))

    axes[0, 1].bar(
        seq_ids,
        gpu_utils,
        yerr=gpu_util_stds,
        capsize=1,
        color=PAL[0],
        edgecolor="none",
        alpha=0.8,
        error_kw={"lw": 0.5},
    )
    axes[0, 1].axhline(np.mean(gpu_utils), color=MEAN_C, ls="--", lw=1.5)
    axes[0, 1].set_ylabel("NVML Busy (%)")
    axes[0, 1].set_title(f"NVML GPU Busy ({np.mean(gpu_utils):.1f}%)")
    axes[0, 1].set_ylim(0, 105)

    axes[0, 2].plot(seq_ids, mem_peaks, "o-", ms=3, color=PAL[1])
    axes[0, 2].axhline(np.mean(mem_peaks), color=MEAN_C, ls="--", lw=1.5)
    axes[0, 2].set_ylabel("Peak Memory (MB)")
    axes[0, 2].set_title(f"Peak Memory ({np.mean(mem_peaks):.0f} MB)")

    axes[1, 0].fill_between(seq_ids, power_draws, alpha=0.25, color=PAL[2])
    axes[1, 0].plot(seq_ids, power_draws, "-", lw=1.2, color=PAL[2])
    axes[1, 0].axhline(np.mean(power_draws), color=MEAN_C, ls="--", lw=1.5)
    axes[1, 0].set_ylabel("Power (W)")
    axes[1, 0].set_xlabel("Sequence #")
    axes[1, 0].set_title(f"Power Draw ({np.mean(power_draws):.0f} W)")

    axes[1, 1].plot(seq_ids, temps, "-", lw=1.2, color=PAL[3])
    axes[1, 1].axhline(np.mean(temps), color="grey", ls="--", lw=1.5)
    axes[1, 1].set_ylabel("Peak Temp (°C)")
    axes[1, 1].set_xlabel("Sequence #")
    axes[1, 1].set_title(f"Temperature ({np.mean(temps):.1f}°C)")

    axes[1, 2].bar(seq_ids, energies, color=PAL[8], edgecolor="none", alpha=0.8)
    axes[1, 2].axhline(np.mean(energies), color=MEAN_C, ls="--", lw=1.5)
    axes[1, 2].set_ylabel("Energy (J)")
    axes[1, 2].set_xlabel("Sequence #")
    axes[1, 2].set_title(f"Energy ({np.mean(energies):.1f} J/inference)")

    for a in axes.flat:
        sns.despine(ax=a, left=True)

    fig.suptitle(
        f"Enformer Eval - GPU Dashboard ({NUM_SEQUENCES} seq x {NUM_REPEATS} repeats)\n"
        f"Model: {flops_per_pass / 1e12:.2f} TFLOP/pass | GPU: {gpu_name} ({peak_tflops} TFLOPS {precision.upper()} peak)",
        fontsize=16,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "gpu_dashboard.png", bbox_inches="tight")
    print(f"Saved: {OUTPUT_DIR / 'gpu_dashboard.png'}")

    # --- Figure 7: Inference time with error bars + histogram ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    ax1.errorbar(
        seq_ids,
        durations,
        yerr=duration_stds,
        fmt="o-",
        ms=3,
        lw=1,
        capsize=2,
        color=PAL[4],
        ecolor="grey",
        elinewidth=0.6,
    )
    ax1.axhline(
        np.mean(durations), color=MEAN_C, ls="--", lw=2, label=f"Mean: {np.mean(durations):.3f}s"
    )
    ax1.set_xlabel("Sequence #")
    ax1.set_ylabel("Inference Time (s)\nmean +/- std")
    ax1.set_title("Per-Sequence Inference Time")
    ax1.legend(frameon=True)
    sns.despine(ax=ax1, left=True)

    sns.histplot(durations, bins=20, color=PAL[4], edgecolor="white", alpha=0.85, kde=True, ax=ax2)
    ax2.axvline(
        np.mean(durations), color=MEAN_C, ls="--", lw=2, label=f"Mean: {np.mean(durations):.3f}s"
    )
    ax2.set_xlabel("Inference Time (s)")
    ax2.set_ylabel("Count")
    ax2.set_title("Inference Time Distribution")
    ax2.legend(frameon=True)
    sns.despine(ax=ax2, left=True)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "inference_time.png")
    print(f"Saved: {OUTPUT_DIR / 'inference_time.png'}")

    # --- Figure 8: FLOPS util vs NVML busy comparison ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    ax1.scatter(gpu_utils, flops_utils, alpha=0.6, s=40, color=PAL[9], edgecolors="w", lw=0.5)
    ax1.set_xlabel("NVML GPU Busy (%)")
    ax1.set_ylabel(f"FLOPS Utilization (% of {peak_tflops} TFLOPS)")
    ax1.set_title("FLOPS Utilization vs NVML Busy")
    ax1.axhline(np.mean(flops_utils), color=MEAN_C, ls="--", lw=1.5, alpha=0.7)
    ax1.axvline(np.mean(gpu_utils), color=PAL[0], ls="--", lw=1.5, alpha=0.7)
    sns.despine(ax=ax1, left=True)

    # Side-by-side bar: mean NVML busy vs mean FLOPS util
    labels = ["NVML GPU Busy", "FLOPS Utilization"]
    vals = [np.mean(gpu_utils), np.mean(flops_utils)]
    colors = [PAL[0], PAL[9]]
    bars = ax2.bar(labels, vals, color=colors, edgecolor="none", alpha=0.85, width=0.5)
    for bar, v in zip(bars, vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{v:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=14,
        )
    ax2.set_ylabel("Utilization (%)")
    ax2.set_title("NVML Busy vs True FLOPS Utilization")
    ax2.set_ylim(0, 105)
    sns.despine(ax=ax2, left=True)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "flops_vs_nvml.png")
    print(f"Saved: {OUTPUT_DIR / 'flops_vs_nvml.png'}")

    plt.close("all")
    print(f"\nAll figures saved in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
