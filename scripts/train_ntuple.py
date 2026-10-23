"""Train the n-tuple network with afterstate TD(0).

Example (10-minute smoke test):
  uv run python scripts/train_ntuple.py --seed 0 --max-minutes 10
Resume the latest checkpoint for that seed:
  uv run python scripts/train_ntuple.py --seed 0 --games 200000 --resume auto
"""

import argparse

from game2048.ntuple import train


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--games", type=int, default=None, help="total games (including resumed ones)")
    parser.add_argument("--max-minutes", type=float, default=None,
                        help="wall-clock budget, checked between 1,000-game blocks")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="TD step size, split over the 32 features (ignored on resume)")
    parser.add_argument("--checkpoint-every", type=int, default=10_000, help="games between checkpoints")
    parser.add_argument("--run-dir", default="runs")
    parser.add_argument("--resume", default=None, help="checkpoint dir/weights.npy, or 'auto'")
    parser.add_argument("--overwrite", action="store_true", help="start fresh even if the CSV exists")
    args = parser.parse_args(argv)
    if args.games is None and args.max_minutes is None:
        parser.error("set --games and/or --max-minutes")
    train(seed=args.seed, games=args.games, max_minutes=args.max_minutes, run_dir=args.run_dir,
          alpha=args.alpha, checkpoint_every=args.checkpoint_every, resume=args.resume,
          overwrite=args.overwrite)


if __name__ == "__main__":
    main()
