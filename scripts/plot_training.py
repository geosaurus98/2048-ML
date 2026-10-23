"""Plot a training CSV: score, max tile, loss and epsilon per episode.

  uv run python scripts/plot_training.py runs/dqn_0.csv   # writes runs/dqn_0.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"
SERIES_LIGHT = "#86b6ef"


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    def col(name):
        return np.array([float(r[name]) if r[name] != "" else np.nan for r in rows])

    return {k: col(k) for k in ("episode", "score", "max_tile", "mean_loss", "epsilon")}


def rolling(x, window):
    out = np.full_like(x, np.nan)
    for i in range(len(x)):
        seg = x[max(0, i - window + 1): i + 1]
        seg = seg[~np.isnan(seg)]
        if len(seg):
            out[i] = seg.mean()
    return out


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=TEXT, fontsize=11)
    ax.set_ylabel(ylabel, color=TEXT_2)
    ax.set_xlabel("episode", color=TEXT_2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=TEXT_2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def plot(csv_path, out_path, window=50):
    d = load(csv_path)
    ep = d["episode"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), facecolor=SURFACE)
    fig.suptitle(f"DQN training — {Path(csv_path).name}", color=TEXT, x=0.01, ha="left")

    ax = axes[0, 0]
    ax.scatter(ep, d["score"], s=8, color=SERIES_LIGHT, linewidths=0, label="episode")
    ax.plot(ep, rolling(d["score"], window), color=SERIES, linewidth=2, label=f"{window}-episode mean")
    style(ax, "Score", "score")
    ax.legend(frameon=False, labelcolor=TEXT_2)

    ax = axes[0, 1]
    log_tile = np.log2(np.maximum(d["max_tile"], 1))
    ax.scatter(ep, log_tile, s=8, color=SERIES_LIGHT, linewidths=0, label="episode")
    ax.plot(ep, rolling(log_tile, window), color=SERIES, linewidth=2, label=f"{window}-episode mean")
    ticks = np.arange(max(1, int(np.nanmin(log_tile))), int(np.nanmax(log_tile)) + 1)
    ax.set_yticks(ticks, [str(1 << int(t)) for t in ticks])
    style(ax, "Max tile", "tile (log scale)")
    ax.legend(frameon=False, labelcolor=TEXT_2)

    ax = axes[1, 0]
    ax.plot(ep, rolling(d["mean_loss"], window), color=SERIES, linewidth=2)
    style(ax, f"Huber loss ({window}-episode mean)", "loss")

    ax = axes[1, 1]
    ax.plot(ep, d["epsilon"], color=SERIES, linewidth=2)
    ax.set_ylim(0, 1.05)
    style(ax, "Exploration ε", "ε")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv")
    parser.add_argument("--out", default=None, help="default: CSV path with .png")
    parser.add_argument("--window", type=int, default=50)
    args = parser.parse_args(argv)
    out = args.out or str(Path(args.csv).with_suffix(".png"))
    print(plot(args.csv, out, args.window))


if __name__ == "__main__":
    main()
