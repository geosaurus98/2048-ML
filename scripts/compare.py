"""Compare agents on the same seeded games.

Every agent plays games seeded seed, seed+1, ..., seed+n-1: game i starts from the
same board and draws spawns from the same RNG stream for every agent (spawns
still diverge once the agents' boards differ).

  uv run python scripts/compare.py --games 200
writes runs/compare.csv (one row per agent), runs/compare_games.csv (one row per
agent and game) and runs/compare_max_tile.png.

Learned agents load the most recently written checkpoint in runs/ unless
--dqn-checkpoint / --ntuple-checkpoint is given; an agent without a checkpoint
is skipped.
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from game2048.agents import RandomAgent  # noqa: E402
from game2048.evaluate import EvalResult, evaluate  # noqa: E402
from game2048.expectimax import ExpectimaxAgent  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"

REACH = (512, 1024, 2048, 4096, 8192)
SUMMARY_FIELDS = ["agent", "checkpoint", "games", "mean_score", "mean_score_ci95", "median_score",
                  "min_score", "max_score"] + [f"pct_{t}" for t in REACH] + ["mean_moves", "games_per_sec"]


def latest(run_dir, pattern):
    paths = sorted(Path(run_dir).glob(pattern), key=lambda p: p.stat().st_mtime)
    return paths[-1] if paths else None


def evaluate_seeded(agent, n_games, seed):
    """One evaluate() call per game so game i always starts from seed + i."""
    parts = [evaluate(agent, n_games=1, seed=seed + i) for i in range(n_games)]
    return EvalResult(
        n_games,
        np.concatenate([p.scores for p in parts]),
        np.concatenate([p.max_tiles for p in parts]),
        np.concatenate([p.moves for p in parts]),
        sum(p.seconds for p in parts),
    )


def build_agents(args):
    agents = [("Random", lambda: RandomAgent(), ""),
              ("Expectimax d2", lambda: ExpectimaxAgent(depth=2), "")]
    dqn = args.dqn_checkpoint or latest(args.run_dir, "dqn_*/latest.pt")
    if dqn:
        from game2048.dqn import DQNAgent
        agents.append(("DQN", lambda: DQNAgent(dqn), str(dqn)))
    else:
        print("no DQN checkpoint found; skipping DQN", file=sys.stderr)
    nt = args.ntuple_checkpoint or latest(args.run_dir, "ntuple_*/weights.npy")
    if nt:
        from game2048.ntuple import NTupleAgent
        agents.append(("N-tuple TD(0)", lambda: NTupleAgent(nt), str(nt)))
    else:
        print("no n-tuple checkpoint found; skipping N-tuple", file=sys.stderr)
    return agents


def summary_row(name, ckpt, r):
    sd = float(np.std(r.scores, ddof=1)) if r.n_games > 1 else 0.0
    row = {
        "agent": name,
        "checkpoint": ckpt,
        "games": r.n_games,
        "mean_score": f"{r.mean_score:.1f}",
        "mean_score_ci95": f"{1.96 * sd / np.sqrt(r.n_games):.1f}",
        "median_score": f"{r.median_score:.1f}",
        "min_score": int(r.scores.min()),
        "max_score": int(r.scores.max()),
        "mean_moves": f"{r.moves.mean():.1f}",
        "games_per_sec": f"{r.games_per_sec:.1f}",
    }
    for t in REACH:
        row[f"pct_{t}"] = f"{r.reach_pct(t):.1f}"
    return row


def plot_max_tiles(results, out_path):
    """Small multiples: one panel per agent, % of games by final max tile, shared axes."""
    all_tiles = sorted({t for _, r in results for t in r.max_tile_counts})
    tiles = [1 << e for e in range(int(np.log2(all_tiles[0])), int(np.log2(all_tiles[-1])) + 1)]
    x = np.arange(len(tiles))
    fig, axes = plt.subplots(1, len(results), figsize=(3.6 * len(results), 3.8),
                             sharey=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, (name, r) in zip(axes, results):
        pct = np.array([100.0 * r.max_tile_counts.get(t, 0) / r.n_games for t in tiles])
        ax.bar(x, pct, width=0.7, color=SERIES, edgecolor=SURFACE, linewidth=2)
        for xi, p in zip(x, pct):
            if p >= 5:
                ax.text(xi, p + 1.5, f"{p:.0f}%", ha="center", va="bottom", fontsize=8, color=TEXT_2)
        ax.set_facecolor(SURFACE)
        ax.set_title(f"{name}\nmean score {r.mean_score:,.0f}", loc="left", fontsize=10, color=TEXT)
        ax.set_xticks(x, [str(t) for t in tiles], rotation=45, ha="right", fontsize=8, color=TEXT_2)
        ax.tick_params(colors=TEXT_2)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.set_xlabel("max tile", color=TEXT_2)
    axes[0].set_ylabel("% of games", color=TEXT_2)
    axes[0].set_ylim(0, 105)
    n = results[0][1].n_games
    fig.suptitle(f"Final max tile per agent ({n} seeded games each)", x=0.01, ha="left", color=TEXT)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-dir", default="runs")
    parser.add_argument("--out-dir", default=None, help="default: --run-dir")
    parser.add_argument("--dqn-checkpoint", default=None)
    parser.add_argument("--ntuple-checkpoint", default=None)
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir or args.run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results, rows, game_rows = [], [], []
    for name, make, ckpt in build_agents(args):
        print(f"evaluating {name} ...", flush=True)
        r = evaluate_seeded(make(), args.games, args.seed)
        results.append((name, r))
        rows.append(summary_row(name, ckpt, r))
        for i in range(r.n_games):
            game_rows.append({"agent": name, "game_seed": args.seed + i, "score": int(r.scores[i]),
                              "max_tile": int(r.max_tiles[i]), "moves": int(r.moves[i])})

    with open(out_dir / "compare.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "compare_games.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["agent", "game_seed", "score", "max_tile", "moves"])
        w.writeheader()
        w.writerows(game_rows)
    plot_max_tiles(results, out_dir / "compare_max_tile.png")

    print()
    header = f"{'agent':<15}{'mean':>10}{'±95%':>8}{'median':>10}" + "".join(f"{t:>7}" for t in REACH) + f"{'games/s':>9}"
    print(header)
    for row in rows:
        print(f"{row['agent']:<15}{float(row['mean_score']):>10,.0f}{float(row['mean_score_ci95']):>8,.0f}"
              f"{float(row['median_score']):>10,.0f}" + "".join(f"{row[f'pct_{t}']:>6}%" for t in REACH)
              + f"{float(row['games_per_sec']):>9.1f}")
    print(f"\n(% columns: share of games reaching that tile)\nwrote {out_dir / 'compare.csv'}, "
          f"{out_dir / 'compare_games.csv'}, {out_dir / 'compare_max_tile.png'}")


if __name__ == "__main__":
    main()
