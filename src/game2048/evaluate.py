"""Evaluate an agent over full games.

Usage: uv run python -m game2048.evaluate --agent random --games 200 --seed 0
"""

import argparse
import time
from dataclasses import dataclass, field

import numpy as np

from game2048.engine import legal_moves, max_tile, new_game, seed_rng, step

REACH_TILES = (512, 1024, 2048, 4096)


@dataclass
class EvalResult:
    n_games: int
    scores: np.ndarray
    max_tiles: np.ndarray
    moves: np.ndarray
    seconds: float
    max_tile_counts: dict = field(init=False)

    def __post_init__(self):
        tiles, counts = np.unique(self.max_tiles, return_counts=True)
        self.max_tile_counts = {int(t): int(c) for t, c in zip(tiles, counts)}

    @property
    def mean_score(self):
        return float(np.mean(self.scores))

    @property
    def median_score(self):
        return float(np.median(self.scores))

    @property
    def games_per_sec(self):
        return self.n_games / self.seconds if self.seconds > 0 else float("inf")

    def reach_pct(self, tile):
        """Percentage of games whose max tile is at least `tile`."""
        return 100.0 * float(np.mean(self.max_tiles >= tile))

    def summary(self):
        lines = [
            f"Games         {self.n_games:>10d}",
            f"Mean score    {self.mean_score:>10.1f}",
            f"Median score  {self.median_score:>10.1f}",
            f"Min / max     {int(self.scores.min()):>5d} / {int(self.scores.max())}",
            f"Mean moves    {float(np.mean(self.moves)):>10.1f}",
            f"Games/sec     {self.games_per_sec:>10.1f}",
            "",
            f"{'Max tile':>8}  {'Games':>6}  {'%':>6}",
        ]
        for tile, count in sorted(self.max_tile_counts.items()):
            lines.append(f"{tile:>8d}  {count:>6d}  {100.0 * count / self.n_games:>6.1f}")
        lines += ["", f"{'Reached':>8}  {'%':>6}"]
        for tile in REACH_TILES:
            lines.append(f"{tile:>8d}  {self.reach_pct(tile):>6.1f}")
        return "\n".join(lines)


def _warmup():
    """Trigger Numba compilation so it is not counted in games/sec."""
    board = np.zeros((4, 4), dtype=np.uint8)
    board[0, 0] = 1
    legal_moves(board)
    step(board, 1)


def play_game(agent):
    """Play one full game. Returns (score, max tile value, number of moves)."""
    board = new_game()
    score = moves = 0
    done = not legal_moves(board).any()
    while not done:
        action = agent.act(board)
        board, reward, moved, done = step(board, action)
        if not moved:
            raise ValueError(f"agent chose illegal action {action}")
        score += reward
        moves += 1
    return score, max_tile(board), moves


def evaluate(agent, n_games=100, seed=0):
    """Play `n_games` full games and return an EvalResult.

    Seeds the engine RNG (tile spawns) and, if the agent has a `seed` method,
    the agent too, so results are reproducible for a fixed seed.
    """
    _warmup()
    seed_rng(seed)
    if hasattr(agent, "seed"):
        agent.seed(seed)
    scores = np.zeros(n_games, dtype=np.int64)
    tiles = np.zeros(n_games, dtype=np.int64)
    moves = np.zeros(n_games, dtype=np.int64)
    start = time.perf_counter()
    for i in range(n_games):
        scores[i], tiles[i], moves[i] = play_game(agent)
    seconds = time.perf_counter() - start
    return EvalResult(n_games, scores, tiles, moves, seconds)


def main(argv=None):
    from game2048.agents import AGENTS

    parser = argparse.ArgumentParser(description="Evaluate a 2048 agent.")
    parser.add_argument("--agent", choices=sorted(AGENTS), default="random")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    agent = AGENTS[args.agent]()
    result = evaluate(agent, n_games=args.games, seed=args.seed)
    print(f"Agent: {args.agent}  seed: {args.seed}\n")
    print(result.summary())


if __name__ == "__main__":
    main()
