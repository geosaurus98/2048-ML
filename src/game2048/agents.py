"""Agents. Every agent exposes `act(board) -> action` and only returns legal moves."""

from functools import partial

import numpy as np

from game2048.engine import legal_moves
from game2048.expectimax import ExpectimaxAgent


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


def _dqn_agent(checkpoint=None, **kwargs):
    if checkpoint is None:
        raise ValueError("the dqn agent needs --checkpoint")
    from game2048.dqn import DQNAgent  # lazy: keeps torch out of non-DQN imports

    return DQNAgent(checkpoint, **kwargs)


def _ntuple_agent(checkpoint=None):
    if checkpoint is None:
        raise ValueError("the ntuple agent needs --checkpoint")
    from game2048.ntuple import NTupleAgent

    return NTupleAgent(checkpoint)


AGENTS = {
    "random": RandomAgent,
    "expectimax": ExpectimaxAgent,  # depth 2
    "expectimax-d3": partial(ExpectimaxAgent, depth=3),
    "dqn": _dqn_agent,
    "ntuple": _ntuple_agent,
}
