"""Agents. Every agent exposes `act(board, piece) -> action` and only returns legal actions."""

import numpy as np

from tetris.engine import legal_actions
from tetris.features import LinearAgent


class RandomAgent:
    """Picks uniformly among the legal placements."""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def seed(self, seed):
        """Reset the agent's RNG. Called by `evaluate` so runs are reproducible."""
        self.rng = np.random.default_rng(seed)

    def act(self, board, piece):
        legal = np.flatnonzero(legal_actions(piece))
        return int(self.rng.choice(legal))


def _cem_agent(checkpoint=None):
    if checkpoint is None:
        raise ValueError("the cem agent needs --checkpoint")
    from tetris.cem import load_weights

    return LinearAgent(load_weights(checkpoint))


def _dqn_agent(checkpoint=None, **kwargs):
    if checkpoint is None:
        raise ValueError("the dqn agent needs --checkpoint")
    from tetris.dqn import DQNAgent  # lazy: keeps torch out of non-DQN imports

    return DQNAgent(checkpoint, **kwargs)


AGENTS = {
    "random": RandomAgent,
    "heuristic": LinearAgent,  # Dellacherie's hand-tuned weights
    "cem": _cem_agent,
    "dqn": _dqn_agent,
}
