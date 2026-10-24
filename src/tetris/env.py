"""Gymnasium environment for placement Tetris.

Observation: dict with "board" ((H, 10) uint8, 1 = filled) and "piece" (0..6 = IOTSZJL).
Actions: rotation * 10 + column (engine docstring), 40 in total.
Reward: lines cleared by the placement.
info["action_mask"]: bool array of the legal actions. Illegal actions raise
ValueError, so agents must mask them. Topping out terminates the episode;
reaching `max_pieces` (if set) truncates it.

Piece draws use Numba's global RNG (engine.seed_rng). Each reset reseeds it from
the env's own np_random, so `reset(seed=s)` fully determines the episode. That
RNG is shared, so interleaving steps of several envs in one process makes their
pieces depend on each other.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tetris.engine import (
    HEIGHT, N_ACTIONS, N_PIECES, WIDTH, is_legal, legal_actions, new_board, random_piece,
    render, seed_rng, step,
)


class TetrisEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, height=HEIGHT, max_pieces=None, render_mode=None):
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        self.render_mode = render_mode
        self.height = height
        self.max_pieces = max_pieces
        self.observation_space = spaces.Dict({
            "board": spaces.Box(0, 1, (height, WIDTH), np.uint8),
            "piece": spaces.Discrete(N_PIECES),
        })
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.board = None
        self.piece = None
        self.lines = 0
        self.pieces = 0
        self.done = False

    def _obs(self):
        return {"board": self.board.copy(), "piece": self.piece}

    def _info(self):
        return {"action_mask": legal_actions(self.piece), "lines": self.lines, "pieces": self.pieces}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        seed_rng(int(self.np_random.integers(2**31 - 1)))
        self.board = new_board(self.height)
        self.piece = int(random_piece())
        self.lines = 0
        self.pieces = 0
        self.done = False
        return self._obs(), self._info()

    def step(self, action):
        if self.board is None:
            raise RuntimeError("call reset() before step()")
        if self.done:
            raise RuntimeError("the game topped out; call reset()")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r}")
        if not is_legal(self.piece, int(action)):
            raise ValueError(f"illegal action {int(action)} for piece {self.piece}")
        board, lines, piece, done = step(self.board, self.piece, int(action))
        self.board, self.piece, self.done = board, int(piece), bool(done)
        if not done:
            self.lines += int(lines)
            self.pieces += 1
        truncated = not done and self.max_pieces is not None and self.pieces >= self.max_pieces
        return self._obs(), float(lines), bool(done), truncated, self._info()

    def render(self):
        if self.render_mode == "ansi":
            return render(self.board, self.piece)
        return None
