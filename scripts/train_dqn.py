"""Train the DQN agent.

Example (5-minute smoke test):
  uv run python scripts/train_dqn.py --seed 0 --max-minutes 5 --checkpoint-every 50
Resume the latest checkpoint for that seed:
  uv run python scripts/train_dqn.py --seed 0 --episodes 20000 --resume auto
"""

import argparse
from dataclasses import fields

from game2048.dqn import DQNConfig, train


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=None, help="total episodes (including resumed ones)")
    parser.add_argument("--max-minutes", type=float, default=None, help="wall-clock budget for this invocation")
    parser.add_argument("--checkpoint-every", type=int, default=100, help="episodes between checkpoints")
    parser.add_argument("--run-dir", default="runs")
    parser.add_argument("--resume", default=None, help="checkpoint path, or 'auto' for runs/dqn_<seed>/latest.pt")
    parser.add_argument("--overwrite", action="store_true", help="start fresh even if the CSV exists")
    parser.add_argument("--device", default="auto", help="auto (cuda > mps > cpu), cpu, cuda, ...")
    hp = parser.add_argument_group("hyperparameters (ignored on resume: the checkpoint's config is used)")
    for f in fields(DQNConfig):
        hp.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    args = parser.parse_args(argv)
    if args.episodes is None and args.max_minutes is None:
        parser.error("set --episodes and/or --max-minutes")

    config = DQNConfig(**{f.name: getattr(args, f.name) for f in fields(DQNConfig)})
    train(seed=args.seed, episodes=args.episodes, max_minutes=args.max_minutes,
          run_dir=args.run_dir, checkpoint_every=args.checkpoint_every, resume=args.resume,
          overwrite=args.overwrite, device=args.device, config=config)


if __name__ == "__main__":
    main()
