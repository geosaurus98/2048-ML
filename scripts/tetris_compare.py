"""Compare Tetris agents on the same seeded games, on two boards.

  small     10x10, games capped at 100,000 pieces: games end, so this ranks agents
  standard  10x20, capped at 10,000 pieces (4,000 lines): good agents survive
            the cap, so this mostly measures who survives

(Gabillon et al. 2013 report Tetris results on the same pair of boards.) Every
agent plays games seeded seed, seed+1, ..., seed+n-1: game i draws the same
piece sequence for every agent.

  uv run python scripts/tetris_compare.py --games 100
writes runs/tetris_compare.csv (one row per agent and board),
runs/tetris_compare_games.csv (one row per agent, board and game) and
runs/tetris_compare_lines.png.

Learned agents load the most recently written checkpoint in runs/ unless
--cem-checkpoint / --dqn-checkpoint is given; an agent without a checkpoint
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

from tetris.agents import RandomAgent  # noqa: E402
from tetris.evaluate import REACH_LINES, EvalResult, evaluate  # noqa: E402
from tetris.features import LinearAgent  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"

BOARDS = {"small": (10, 100_000), "standard": (20, 10_000)}  # name: (height, max pieces)
SUMMARY_FIELDS = (["agent", "checkpoint", "board", "height", "max_pieces", "games", "mean_lines",
                   "mean_lines_ci95", "median_lines", "min_lines", "max_lines", "survival_pct"]
                  + [f"pct_{n}" for n in REACH_LINES] + ["mean_pieces", "pieces_per_sec"])


def latest(run_dir, pattern):
    paths = sorted(Path(run_dir).glob(pattern), key=lambda p: p.stat().st_mtime)
    return paths[-1] if paths else None


def evaluate_seeded(agent, n_games, seed, max_pieces, height):
    """One evaluate() call per game so game i always starts from seed + i."""
    parts = [evaluate(agent, n_games=1, seed=seed + i, max_pieces=max_pieces, height=height)
             for i in range(n_games)]
    return EvalResult(
        n_games,
        np.concatenate([p.lines for p in parts]),
        np.concatenate([p.pieces for p in parts]),
        np.concatenate([p.topped_out for p in parts]),
        sum(p.seconds for p in parts),
        max_pieces,
    )


def build_agents(args):
    agents = [("Random", lambda: RandomAgent(), ""),
              ("Heuristic", lambda: LinearAgent(), "")]
    cem = args.cem_checkpoint or latest(args.run_dir, "tetris_cem_*/state.json")
    if cem:
        from tetris.cem import load_weights
        agents.append(("CEM", lambda: LinearAgent(load_weights(cem)), str(cem)))
    else:
        print("no CEM checkpoint found; skipping CEM", file=sys.stderr)
    dqn = args.dqn_checkpoint or latest(args.run_dir, "tetris_dqn_*/latest.pt")
    if dqn:
        from tetris.dqn import DQNAgent
        agents.append(("DQN", lambda: DQNAgent(dqn), str(dqn)))
    else:
        print("no DQN checkpoint found; skipping DQN", file=sys.stderr)
    return agents


def ci95(r):
    sd = float(np.std(r.lines, ddof=1)) if r.n_games > 1 else 0.0
    return 1.96 * sd / np.sqrt(r.n_games)


def summary_row(name, ckpt, board, r):
    height, max_pieces = BOARDS[board]
    row = {
        "agent": name,
        "checkpoint": ckpt,
        "board": board,
        "height": height,
        "max_pieces": max_pieces,
        "games": r.n_games,
        "mean_lines": f"{r.mean_lines:.1f}",
        "mean_lines_ci95": f"{ci95(r):.1f}",
        "median_lines": f"{r.median_lines:.1f}",
        "min_lines": int(r.lines.min()),
        "max_lines": int(r.lines.max()),
        "survival_pct": f"{r.survival_pct:.1f}",
        "mean_pieces": f"{r.pieces.mean():.1f}",
        "pieces_per_sec": f"{r.pieces_per_sec:.0f}",
    }
    for n in REACH_LINES:
        row[f"pct_{n}"] = f"{r.reach_pct(n):.1f}"
    return row


def plot_lines(results, out_path):
    """One panel per board: mean lines per agent with 95% CI, agents top to bottom."""
    boards = list(dict.fromkeys(b for _, b, _ in results))
    fig, axes = plt.subplots(1, len(boards), figsize=(5.4 * len(boards), 0.6 * len(results) / len(boards) + 2.2),
                             facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, board in zip(axes, boards):
        rows = [(name, r) for name, b, r in results if b == board]
        names = [n for n, _ in rows][::-1]
        means = np.array([r.mean_lines for _, r in rows])[::-1]
        errs = np.array([ci95(r) for _, r in rows])[::-1]
        y = np.arange(len(rows))
        ax.barh(y, means, height=0.6, color=SERIES, edgecolor=SURFACE, linewidth=2,
                xerr=errs, error_kw={"ecolor": TEXT_2, "elinewidth": 1, "capsize": 3})
        right = float((means + errs).max()) or 1.0
        for yi, (name, r) in zip(y, rows[::-1]):
            label = f"{r.mean_lines:,.0f}"
            if board == "standard":
                label += f"  ({r.survival_pct:.0f}% survive)"
            ax.text(r.mean_lines + ci95(r) + right * 0.02, yi, label, va="center", fontsize=8, color=TEXT_2)
        ax.set_xlim(0, right * 1.45)
        ax.set_yticks(y, names, fontsize=9, color=TEXT)
        height, max_pieces = BOARDS[board]
        sub = "full games" if board == "small" else f"capped at {max_pieces:,} pieces"
        ax.set_title(f"10x{height} board ({sub})", loc="left", fontsize=10, color=TEXT)
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=TEXT_2)
        ax.grid(True, axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.set_xlabel("mean lines cleared (95% CI)", color=TEXT_2)
    n = results[0][2].n_games
    fig.suptitle(f"Tetris: lines cleared per agent ({n} seeded games each)", x=0.01, ha="left", color=TEXT)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--boards", nargs="+", choices=list(BOARDS), default=list(BOARDS))
    parser.add_argument("--run-dir", default="runs")
    parser.add_argument("--out-dir", default=None, help="default: --run-dir")
    parser.add_argument("--cem-checkpoint", default=None)
    parser.add_argument("--dqn-checkpoint", default=None)
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir or args.run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results, rows, game_rows = [], [], []
    for name, make, ckpt in build_agents(args):
        agent = make()
        for board in args.boards:
            height, max_pieces = BOARDS[board]
            print(f"evaluating {name} on 10x{height} ...", flush=True)
            r = evaluate_seeded(agent, args.games, args.seed, max_pieces, height)
            results.append((name, board, r))
            rows.append(summary_row(name, ckpt, board, r))
            for i in range(r.n_games):
                game_rows.append({"agent": name, "board": board, "game_seed": args.seed + i,
                                  "lines": int(r.lines[i]), "pieces": int(r.pieces[i]),
                                  "topped_out": int(r.topped_out[i])})

    with open(out_dir / "tetris_compare.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "tetris_compare_games.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["agent", "board", "game_seed", "lines", "pieces", "topped_out"])
        w.writeheader()
        w.writerows(game_rows)
    plot_lines(results, out_dir / "tetris_compare_lines.png")

    print()
    header = (f"{'agent':<11}{'board':<10}{'mean':>9}{'±95%':>8}{'median':>9}{'survive':>9}"
              + "".join(f"{n:>7}" for n in REACH_LINES) + f"{'pieces/s':>10}")
    print(header)
    for row in rows:
        print(f"{row['agent']:<11}{row['board']:<10}{float(row['mean_lines']):>9,.0f}"
              f"{float(row['mean_lines_ci95']):>8,.0f}{float(row['median_lines']):>9,.0f}"
              f"{row['survival_pct']:>8}%" + "".join(f"{row[f'pct_{n}']:>6}%" for n in REACH_LINES)
              + f"{float(row['pieces_per_sec']):>10,.0f}")
    print(f"\n(% columns: share of games reaching that many lines)\nwrote {out_dir / 'tetris_compare.csv'}, "
          f"{out_dir / 'tetris_compare_games.csv'}, {out_dir / 'tetris_compare_lines.png'}")


if __name__ == "__main__":
    main()
