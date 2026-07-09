"""Plot training-loss comparison across three MT variants.

Reads loss lines from the three train.log files, plots raw (faint) and
EMA-smoothed curves, and writes the figure to figures/.
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOGS = [
    ("MT baseline", ROOT / "outputs/enc_mt_xen_100k/train.log", "#2a78d6"),
    ("MT + fusion", ROOT / "outputs/enc_mt_fusion_xen_100k/train.log", "#1baf7a"),
    ("MT embed-only", ROOT / "outputs/enc_mt_embed_only_zh_en_100k/train.log", "#eda100"),
]
OUT = ROOT / "figures/loss_compare_mt_baseline_vs_fusion.png"

INK = "#0b0b0b"
INK_MUTED = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e6e6e3"

LINE_RE = re.compile(r"step=(\d+)\s+loss=([\d.]+)")


def load(path: Path):
    steps, losses = [], []
    with open(path) as f:
        for line in f:
            m = LINE_RE.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
    return np.array(steps), np.array(losses)


def ema(y, alpha=0.08):
    out = np.empty_like(y, dtype=float)
    s = y[0]
    for i, v in enumerate(y):
        s = alpha * v + (1 - alpha) * s
        out[i] = s
    return out


def main():
    series = []
    for label, path, color in LOGS:
        s, l = load(path)
        series.append((label, color, s, l))

    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_MUTED,
        "text.color": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "font.size": 11,
    })

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=150)

    for label, color, s, l in series:
        ax.scatter(s, l, s=6, color=color, alpha=0.18, linewidths=0, zorder=2)
        ax.plot(s, ema(l), color=color, lw=2.0, label=label, zorder=3)
        ax.annotate(f"{ema(l)[-1]:.2f}", (s[-1], ema(l)[-1]),
                    textcoords="offset points", xytext=(6, 0),
                    color=color, fontsize=10, va="center")

    ax.set_xlabel("training step", fontsize=11)
    ax.set_ylabel("cross-entropy loss", fontsize=11)
    ax.set_title("Training loss: MT baseline vs. fusion vs. embed-only",
                 fontsize=13, weight="bold", pad=12)

    ax.grid(True, color=GRID, lw=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    leg = ax.legend(frameon=False, loc="upper right", fontsize=11)
    for text in leg.get_texts():
        text.set_color(INK)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    parts = ", ".join(f"{lbl} {len(l)} pts" for lbl, _, _, l in series)
    print(f"wrote {OUT}  ({parts})")


if __name__ == "__main__":
    main()
