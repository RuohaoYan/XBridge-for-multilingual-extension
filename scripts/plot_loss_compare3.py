"""Plot three-way training loss comparison: baseline / fusion / embed-only.

Reads loss lines from train.log files, plots raw (faint) + EMA-smoothed curves.
NOTE: baseline & fusion ran on multi-source x->en; embed-only ran on opus100 zh->en.
Data differs — curves are indicative, not a controlled comparison.
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    ("MT baseline (multi-xen)", ROOT / "outputs/enc_mt_xen_100k/train.log", "#2a78d6"),
    ("MT + fusion (multi-xen)", ROOT / "outputs/enc_mt_fusion_xen_100k/train.log", "#1baf7a"),
    ("embed-only (opus100 zh-en)", ROOT / "outputs/enc_mt_embed_only_zh_en_100k/train.log", "#e34948"),
]
OUT = ROOT / "figures/loss_compare3_baseline_fusion_embedonly.png"

INK = "#0b0b0b"
INK_MUTED = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e6e6e3"
LINE_RE = re.compile(r"step=(\d+)\s+loss=([\d.]+)")


def load(path: Path):
    if not path.is_file():
        return np.array([]), np.array([])
    steps, losses = [], []
    with open(path) as f:
        for line in f:
            m = LINE_RE.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
    return np.array(steps), np.array(losses)


def ema(y, alpha=0.08):
    if len(y) == 0:
        return y
    out = np.empty_like(y, dtype=float)
    s = y[0]
    for i, v in enumerate(y):
        s = alpha * v + (1 - alpha) * s
        out[i] = s
    return out


def main():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_MUTED,
        "text.color": INK, "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "font.size": 11,
    })
    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=150)

    for label, path, color in RUNS:
        s, l = load(path)
        if len(l) == 0:
            print(f"skip (no data): {path}")
            continue
        ax.scatter(s, l, s=6, color=color, alpha=0.16, linewidths=0, zorder=2)
        ax.plot(s, ema(l), color=color, lw=2.0, label=label, zorder=3)
        ax.annotate(f"{ema(l)[-1]:.2f}", (s[-1], ema(l)[-1]),
                    textcoords="offset points", xytext=(6, 0),
                    color=color, fontsize=10, va="center")
        print(f"{label}: {len(l)} pts, step {s[0]}->{s[-1]}, last EMA {ema(l)[-1]:.3f}")

    ax.set_xlabel("training step", fontsize=11)
    ax.set_ylabel("cross-entropy loss", fontsize=11)
    ax.set_title("Training loss: baseline vs. fusion vs. embed-only",
                 fontsize=13, weight="bold", pad=12)
    ax.grid(True, color=GRID, lw=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    leg = ax.legend(frameon=False, loc="upper right", fontsize=10)
    for text in leg.get_texts():
        text.set_color(INK)

    fig.text(0.99, 0.01,
             "Note: baseline/fusion on multi-source x→en; embed-only on opus100 zh→en (different data).",
             ha="right", va="bottom", fontsize=8, color=INK_MUTED, style="italic")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
