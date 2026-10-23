"""Gymnasium environment for 2048.

Observation: (16, 4, 4) float32 one-hot of tile exponents (engine.to_onehot).
Actions: 0 = up, 1 = right, 2 = down, 3 = left.
Reward: sum of merged tile values for the move.
info["action_mask"]: bool array of the legal actions. Illegal actions raise
ValueError, so agents must mask them.

Tile spawns use Numba's global RNG (engine.seed_rng). Each reset reseeds it from
the env's own np_random, so `reset(seed=s)` fully determines the episode. That
RNG is shared, so interleaving steps of several envs in one process makes their
spawns depend on each other.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from game2048.engine import legal_moves, max_tile, new_game, render, seed_rng, step, to_onehot


class Game2048Env(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, render_mode=None):
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        self.render_mode = render_mode
        self.observation_space = spaces.Box(0.0, 1.0, (16, 4, 4), np.float32)
        self.action_space = spaces.Discrete(4)
        self.board = None
        self.score = 0

    def _info(self):
        return {
            "action_mask": legal_moves(self.board),
            "score": self.score,
            "max_tile": max_tile(self.board),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        seed_rng(int(self.np_random.integers(2**31 - 1)))
        self.board = new_game()
        self.score = 0
        return to_onehot(self.board), self._info()

    def step(self, action):
        if self.board is None:
            raise RuntimeError("call reset() before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r}")
        board, reward, moved, done = step(self.board, int(action))
        if not moved:
            raise ValueError(f"illegal action {int(action)}: board unchanged")
        self.board = board
        self.score += int(reward)
        return to_onehot(self.board), float(reward), bool(done), False, self._info()

    def render(self):
        if self.render_mode == "ansi":
            return render(self.board)
        return None
