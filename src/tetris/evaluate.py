"""Evaluate a Tetris agent over full games.

Usage: uv run python -m tetris.evaluate --agent heuristic --games 100 --seed 0

Good agents rarely top out, so each game is capped at --max-pieces placements
(10,000 by default, which allows at most 4,000 lines). A game that reaches the
cap counts as survived; its lines are a lower bound on what it would have scored.
"""

import argparse
import time
from dataclasses import dataclass

import numpy as np

from tetris.engine import HEIGHT, is_legal, new_board, random_piece, seed_rng, step

REACH_LINES = (10, 100, 1000)
MAX_PIECES = 10_000


@dataclass
class EvalResult:
    n_games: int
    lines: np.ndarray
    pieces: np.ndarray
    topped_out: np.ndarray
    seconds: float
    max_pieces: int

    @property
    def mean_lines(self):
        return float(np.mean(self.lines))

    @property
    def median_lines(self):
        return float(np.median(self.lines))

    @property
    def games_per_sec(self):
        return self.n_games / self.seconds if self.seconds > 0 else float("inf")

    @property
    def pieces_per_sec(self):
        return float(self.pieces.sum()) / self.seconds if self.seconds > 0 else float("inf")

    @property
    def survival_pct(self):
        """Percentage of games that reached the piece cap without topping out."""
        return 100.0 * float(np.mean(~self.topped_out))

    def reach_pct(self, lines):
        """Percentage of games that cleared at least `lines` lines."""
        return 100.0 * float(np.mean(self.lines >= lines))

    def summary(self):
        lines = [
            f"Games          {self.n_games:>10d}",
            f"Mean lines     {self.mean_lines:>10.1f}",
            f"Median lines   {self.median_lines:>10.1f}",
            f"Min / max      {int(self.lines.min()):>5d} / {int(self.lines.max())}",
            f"Mean pieces    {float(np.mean(self.pieces)):>10.1f}",
            f"Survived cap   {self.survival_pct:>9.1f}%  ({self.max_pieces:,} pieces)",
            f"Games/sec      {self.games_per_sec:>10.2f}",
            f"Pieces/sec     {self.pieces_per_sec:>10.0f}",
            "",
            f"{'Lines':>8}  {'reached':>7}",
        ]
        for n in REACH_LINES:
            lines.append(f"{n:>8d}  {self.reach_pct(n):>6.1f}%")
        return "\n".join(lines)


def _warmup(height):
    """Trigger Numba compilation so it is not counted in games/sec."""
    seed_rng(0)
    step(new_board(height), random_piece(), 0)


def play_game(agent, max_pieces=MAX_PIECES, height=HEIGHT):
    """Play one game. Returns (lines, pieces placed, topped out)."""
    board = new_board(height)
    piece = random_piece()
    lines = 0
    for placed in range(max_pieces):
        action = agent.act(board, piece)
        if not is_legal(piece, action):
            raise ValueError(f"agent chose illegal action {action} for piece {piece}")
        board, n, piece, done = step(board, piece, action)
        if done:
            return lines, placed, True
        lines += n
    return lines, max_pieces, False


def evaluate(agent, n_games=100, seed=0, max_pieces=MAX_PIECES, height=HEIGHT):
    """Play `n_games` games and return an EvalResult.

    Seeds the engine RNG (piece draws) and, if the agent has a `seed` method,
    the agent too, so results are reproducible for a fixed seed.
    """
    _warmup(height)
    seed_rng(seed)
    if hasattr(agent, "seed"):
        agent.seed(seed)
    lines = np.zeros(n_games, dtype=np.int64)
    pieces = np.zeros(n_games, dtype=np.int64)
    topped = np.zeros(n_games, dtype=np.bool_)
    start = time.perf_counter()
    for i in range(n_games):
        lines[i], pieces[i], topped[i] = play_game(agent, max_pieces, height)
    seconds = time.perf_counter() - start
    return EvalResult(n_games, lines, pieces, topped, seconds, max_pieces)


def main(argv=None):
    from tetris.agents import AGENTS

    parser = argparse.ArgumentParser(description="Evaluate a Tetris agent.")
    parser.add_argument("--agent", choices=sorted(AGENTS), default="random")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-pieces", type=int, default=MAX_PIECES, help="cap on placements per game")
    parser.add_argument("--height", type=int, default=HEIGHT, help="board height (width is always 10)")
    parser.add_argument("--checkpoint", default=None, help="model checkpoint (for learned agents)")
    args = parser.parse_args(argv)

    kwargs = {"checkpoint": args.checkpoint} if args.checkpoint else {}
    agent = AGENTS[args.agent](**kwargs)
    result = evaluate(agent, n_games=args.games, seed=args.seed, max_pieces=args.max_pieces,
                      height=args.height)
    source = f"  checkpoint: {args.checkpoint}" if args.checkpoint else ""
    print(f"Agent: {args.agent}  board: 10x{args.height}  seed: {args.seed}{source}\n")
    print(result.summary())


if __name__ == "__main__":
    main()
