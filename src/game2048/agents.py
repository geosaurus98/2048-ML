"""Agents. Every agent exposes `act(board) -> action` and only returns legal moves."""

import numpy as np

from game2048.engine import legal_moves


class RandomAgent:
    """Picks uniformly among the legal moves."""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def seed(self, seed):
        """Reset the agent's RNG. Called by `evaluate` so runs are reproducible."""
        self.rng = np.random.default_rng(seed)

    def act(self, board):
        legal = np.flatnonzero(legal_moves(board))
        return int(self.rng.choice(legal))


AGENTS = {
    "random": RandomAgent,
}
