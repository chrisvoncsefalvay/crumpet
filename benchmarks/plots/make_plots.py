"""Generate Tufte-style performance plots from CRUMPET benchmark JSON outputs.

Author: Chris von Csefalvay
Licence: MIT
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks" / "results"
OUT = ROOT / "benchmarks" / "plots"
OUT.mkdir(exist_ok=True)


INK = "#1a1a1a"
MUTED = "#9a9a9a"
SOFT = "#cccccc"
ACCENT = "#bf5700"
ACCENT_LIGHT = "#e8a87c"


def tufte_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "regular",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 200,
    })


def load(name: str) -> dict:
    with (RESULTS / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def headline(ax, title: str, subtitle: str | None = None) -> None:
    """Place title and optional subtitle stacked above the axes, with no collision."""
    ax.text(
        0.0,
        1.14 if subtitle else 1.04,
        title,
        transform=ax.transAxes,
        fontsize=11.5,
        color=INK,
        ha="left",
        va="bottom",
    )
    if subtitle:
        ax.text(
            0.0,
            1.04,
            subtitle,
            transform=ax.transAxes,
            fontsize=9,
            color=MUTED,
            ha="left",
            va="bottom",
        )


def save(fig, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=200)
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.svg")
    plt.close(fig)


def plot_per_kernel_speedup() -> None:
    labels = [
        "fused_shift_partition_3d",
        "fused_unshift_unpartition_3d",
        "compute_attn_mask_3d",
        "fused_swin_attention",
    ]
    values = [2.90, 2.74, 22.4, 7.4]

    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    fig.subplots_adjust(top=0.78)
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=ACCENT, height=0.55, edgecolor="none")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + max(values) * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}x",
            va="center",
            ha="left",
            color=INK,
            fontsize=9.5,
        )

    ax.axvline(1.0, color=MUTED, linewidth=0.7, zorder=0)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.18)
    ax.set_xlabel("speedup over eager reference (x)")
    headline(ax,
             "Per-kernel speedup, fp16, NVIDIA GB10",
             "shift / unshift kernels at D=49; mask kernel at D=98; attention kernel at BTCV stage 0")
    ax.tick_params(left=False)
    ax.spines["left"].set_visible(False)

    save(fig, "per_kernel_speedup")


def plot_e2e_btcv() -> None:
    labels = ["eager", "eager + compile", "CRUMPET", "CRUMPET + compile"]
    ms = [5218, 2449, 3082, 1910]
    is_crumpet = [False, False, True, True]
    ref = ms[0]

    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    fig.subplots_adjust(top=0.80)
    y = np.arange(len(labels))
    colors = [ACCENT if c else SOFT for c in is_crumpet]
    bars = ax.barh(y, ms, color=colors, height=0.55, edgecolor="none")

    for bar, val in zip(bars, ms):
        speedup = ref / val
        text = f"{val:,} ms   {speedup:.2f}x"
        ax.text(
            bar.get_width() + max(ms) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            text,
            va="center",
            ha="left",
            fontsize=9.5,
            color=INK,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, max(ms) * 1.22)
    ax.set_xlabel("mean per inference (ms, lower is better)")
    headline(ax,
             "End-to-end BTCV inference, MONAI Swin UNETR",
             "img0025, 96^3 ROI, sw_batch_size=1, 20 iterations, fp16, NVIDIA GB10")
    ax.tick_params(left=False)
    ax.spines["left"].set_visible(False)

    save(fig, "e2e_btcv")


def plot_mask_scaling() -> None:
    data = load("mask_results.json")
    cases = data["cases"]
    Ds = [c["D"] for c in cases]
    ref = [c["reference"]["mean_ms"] for c in cases]
    kernel = [c["kernel"]["mean_ms"] for c in cases]
    cached = [c["cached"]["mean_ms"] for c in cases]

    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    fig.subplots_adjust(top=0.84, right=0.80)
    x = np.array(Ds, dtype=float)

    ax.plot(x, ref, "o-", color=MUTED, lw=1.4, ms=5)
    ax.plot(x, kernel, "o-", color=ACCENT, lw=1.6, ms=5)
    ax.plot(x, cached, "o--", color=ACCENT_LIGHT, lw=1.2, ms=5)

    ax.set_yscale("log")
    ax.set_xticks(Ds)
    ax.set_xticklabels([str(d) for d in Ds])
    ax.set_xlabel("spatial extent D = H = W")
    ax.set_ylabel("mean construction time (ms, log scale)")
    headline(ax,
             "Attention mask construction time",
             "MONAI-style Python reference vs CRUMPET kernel and cache hit, fp16, NVIDIA GB10")

    for xi, yi, val in zip(x, ref, ref):
        ax.annotate(f"{val:.2f}", (xi, yi), xytext=(0, 8), textcoords="offset points",
                    fontsize=8, color=MUTED, ha="center")
    for xi, yi, val in zip(x, kernel, kernel):
        ax.annotate(f"{val:.2f}", (xi, yi), xytext=(0, 8), textcoords="offset points",
                    fontsize=8, color=ACCENT, ha="center")
    for xi, yi, val in zip(x, cached, cached):
        ax.annotate(f"{val:.4f}", (xi, yi), xytext=(0, 8), textcoords="offset points",
                    fontsize=8, color=ACCENT_LIGHT, ha="center")

    ax.text(x[-1] * 1.04, ref[-1], "MONAI reference", fontsize=9, color=MUTED, va="center", ha="left")
    ax.text(x[-1] * 1.04, kernel[-1], "CRUMPET kernel", fontsize=9, color=ACCENT, va="center", ha="left")
    ax.text(x[-1] * 1.04, cached[-1], "cache hit", fontsize=9, color=ACCENT_LIGHT, va="center", ha="left")

    ax.set_xlim(min(Ds) - 5, max(Ds) * 1.45)

    save(fig, "mask_scaling")


def plot_partition_bandwidth() -> None:
    data = load("profiling_summary.json")
    shifted = data["event_microbenchmarks"]["shifted_cases"]
    Cs = [c["C"] for c in shifted]
    sp_part = [c["partition_speedup"] for c in shifted]
    sp_unp = [c["unpartition_speedup"] for c in shifted]
    bw = [c["estimated_effective_bandwidth_gbps"] for c in shifted]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.4))
    fig.subplots_adjust(top=0.82, wspace=0.32)
    x = np.array(Cs, dtype=float)

    ax1.plot(x, sp_part, "o-", color=ACCENT, lw=1.6, ms=5)
    ax1.plot(x, sp_unp, "s--", color=ACCENT_LIGHT, lw=1.2, ms=5)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(Cs)
    ax1.set_xticklabels([str(c) for c in Cs])
    ax1.set_xlabel("channels (C)")
    ax1.set_ylabel("speedup vs eager (x)")
    headline(ax1,
             "Speedup grows with channels",
             "shifted partition / unpartition, D=49, fp16")

    for xi, yi in zip(x, sp_part):
        ax1.annotate(f"{yi:.2f}", (xi, yi), xytext=(0, 8), textcoords="offset points",
                     fontsize=8, color=ACCENT, ha="center")
    for xi, yi in zip(x, sp_unp):
        ax1.annotate(f"{yi:.2f}", (xi, yi), xytext=(0, -14), textcoords="offset points",
                     fontsize=8, color=ACCENT_LIGHT, ha="center")
    ax1.text(x[-1], sp_part[-1] + 0.6, "shift_partition", fontsize=9, color=ACCENT, ha="right")
    ax1.text(x[-1], sp_unp[-1] - 0.9, "unshift_unpartition", fontsize=9, color=ACCENT_LIGHT, ha="right")
    ax1.set_ylim(0, max(sp_part + sp_unp) * 1.40)

    ax2.plot(x, bw, "o-", color=ACCENT, lw=1.6, ms=5)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(Cs)
    ax2.set_xticklabels([str(c) for c in Cs])
    ax2.set_xlabel("channels (C)")
    ax2.set_ylabel("effective bandwidth (GB/s)")
    headline(ax2,
             "Bandwidth utilisation",
             "estimated effective DRAM bandwidth, NVIDIA GB10")
    for xi, yi in zip(x, bw):
        ax2.annotate(f"{yi:g}", (xi, yi), xytext=(0, 8), textcoords="offset points",
                     fontsize=8, color=ACCENT, ha="center")
    ax2.set_ylim(0, max(bw) * 1.20)

    save(fig, "partition_bandwidth_scaling")


def plot_baseline_decomposition() -> None:
    data = load("baseline_results.json")
    records = data["records"]

    labels = []
    fusable = []
    attn = []
    for r in records:
        labels.append(f"s{r['stage_index']}/b{r['block_index']}  D={r['padded_D']}  ss={r['shift_size'][0]}")
        f = r["roll_partition"]["mean_ms"] + r["window_reverse_roll"]["mean_ms"] + r["compute_mask"]["mean_ms"]
        a = r["attention_probe"]["mean_ms"]
        fusable.append(f)
        attn.append(a)

    fusable = np.array(fusable)
    attn = np.array(attn)
    total = fusable + attn

    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    fig.subplots_adjust(top=0.84, left=0.22)
    y = np.arange(len(labels))

    ax.barh(y, attn, color=SOFT, height=0.55, edgecolor="none")
    ax.barh(y, fusable, left=attn, color=ACCENT, height=0.55, edgecolor="none")

    for yi, a, f, t in zip(y, attn, fusable, total):
        pct = f / t * 100
        ax.text(t + max(total) * 0.01, yi, f"{pct:.0f}% fusable",
                fontsize=8.5, color=INK, va="center")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5, family="monospace")
    ax.invert_yaxis()
    ax.set_xlabel("time per call (ms)")
    headline(ax,
             "Where eager Swin UNETR spends time, by block",
             "grey = window attention probe; orange = fusable mechanics (roll, partition, reverse, mask)")
    ax.tick_params(left=False)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, max(total) * 1.22)

    save(fig, "baseline_decomposition")


def plot_zero_shift_routing() -> None:
    data = load("profiling_summary.json")
    shifted = data["event_microbenchmarks"]["shifted_cases"]
    zero = data["event_microbenchmarks"]["zero_shift_cases_before_wrapper_routing"]

    Cs = [c["C"] for c in shifted]
    sh = [c["partition_speedup"] for c in shifted]
    zs = [c["partition_speedup"] for c in zero]

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    fig.subplots_adjust(top=0.82, right=0.82)
    x = np.array(Cs, dtype=float)

    ax.plot(x, sh, "o-", color=ACCENT, lw=1.6, ms=5)
    ax.plot(x, zs, "s--", color=MUTED, lw=1.2, ms=5)
    ax.axhline(1.0, color=INK, linewidth=0.6, linestyle=":", zorder=0)

    ax.set_xscale("log", base=2)
    ax.set_xticks(Cs)
    ax.set_xticklabels([str(c) for c in Cs])
    ax.set_xlabel("channels (C)")
    ax.set_ylabel("partition speedup vs eager (x)")
    headline(ax,
             "Why zero-shift calls now route to view/permute",
             "raw kernel breaks even on zero-shift inputs; the wrapper picks the cheaper path")

    ax.text(x[-1] * 1.04, sh[-1], "shifted (ss=3)", fontsize=9, color=ACCENT, va="center", ha="left")
    ax.text(x[-1] * 1.04, zs[-1], "zero-shift", fontsize=9, color=MUTED, va="center", ha="left")
    ax.text(x[0], 1.06, "  break-even", fontsize=8, color=INK, va="bottom")

    ax.set_ylim(0, max(sh) * 1.25)
    ax.set_xlim(min(Cs) * 0.85, max(Cs) * 1.6)

    save(fig, "zero_shift_routing")


def main() -> None:
    tufte_style()
    plot_per_kernel_speedup()
    plot_e2e_btcv()
    plot_mask_scaling()
    plot_partition_bandwidth()
    plot_baseline_decomposition()
    plot_zero_shift_routing()
    print(f"wrote plots to {OUT}")


if __name__ == "__main__":
    main()
