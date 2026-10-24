"""Train the Tetris linear policy with the noisy cross-entropy method.

Example (10-minute run):
  uv run python scripts/tetris_train_cem.py --seed 0 --max-minutes 10
Resume the latest checkpoint for that seed:
  uv run python scripts/tetris_train_cem.py --seed 0 --iterations 100 --resume auto
"""

import argparse
from dataclasses import fields

from tetris.cem import CEMConfig, train


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=None, help="total iterations (including resumed ones)")
    parser.add_argument("--max-minutes", type=float, default=None,
                        help="wall-clock budget, checked between iterations")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="iterations between checkpoints")
    parser.add_argument("--run-dir", default="runs")
    parser.add_argument("--resume", default=None, help="checkpoint dir/state.json, or 'auto'")
    parser.add_argument("--overwrite", action="store_true", help="start fresh even if the CSV exists")
    hp = parser.add_argument_group("hyperparameters (ignored on resume: the checkpoint's config is used)")
    for f in fields(CEMConfig):
        hp.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    args = parser.parse_args(argv)
    if args.iterations is None and args.max_minutes is None:
        parser.error("set --iterations and/or --max-minutes")

    config = CEMConfig(**{f.name: getattr(args, f.name) for f in fields(CEMConfig)})
    train(seed=args.seed, iterations=args.iterations, max_minutes=args.max_minutes, run_dir=args.run_dir,
          config=config, checkpoint_every=args.checkpoint_every, resume=args.resume, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
