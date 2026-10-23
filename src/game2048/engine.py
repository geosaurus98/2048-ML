"""Fast headless 2048 engine.

Board: (4, 4) uint8 array of tile exponents (0 = empty, 1 = 2, 2 = 4, ...).
Actions: 0 = up, 1 = right, 2 = down, 3 = left.

Moves use precomputed 65536-entry row lookup tables. A row is packed into 16 bits,
4 bits per cell, with cell 0 in the lowest nibble. "Left" moves tiles toward cell 0.

Limitation: exponents are capped at 15 (tile 32768); two 15s do not merge.
"""

import numpy as np
from numba import njit

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTIONS = (UP, RIGHT, DOWN, LEFT)
MAX_EXP = 15


def _build_tables():
    left = np.zeros(65536, dtype=np.uint16)
    right = np.zeros(65536, dtype=np.uint16)
    score_left = np.zeros(65536, dtype=np.int32)
    score_right = np.zeros(65536, dtype=np.int32)

    def slide_left(cells):
        tiles = [c for c in cells if c != 0]
        out, score, i = [], 0, 0
        while i < len(tiles):
            if i + 1 < len(tiles) and tiles[i] == tiles[i + 1] and tiles[i] < MAX_EXP:
                merged = tiles[i] + 1
                out.append(merged)
                score += 1 << merged
                i += 2
            else:
                out.append(tiles[i])
                i += 1
        return out + [0] * (4 - len(out)), score

    def pack(cells):
        return cells[0] | (cells[1] << 4) | (cells[2] << 8) | (cells[3] << 12)

    for idx in range(65536):
        cells = [(idx >> (4 * i)) & 0xF for i in range(4)]
        out, s = slide_left(cells)
        left[idx], score_left[idx] = pack(out), s
        out_r, s_r = slide_left(cells[::-1])
        right[idx], score_right[idx] = pack(out_r[::-1]), s_r
    return left, right, score_left, score_right


LUT_LEFT, LUT_RIGHT, LUT_SCORE_LEFT, LUT_SCORE_RIGHT = _build_tables()


@njit(cache=True)
def seed_rng(seed):
    """Seed Numba's internal RNG (separate from NumPy's global RNG)."""
    np.random.seed(seed)


@njit(cache=True)
def apply_move(board, action):
    """Slide and merge without spawning. Returns (afterstate, reward, moved).

    The input board is never modified. `afterstate` is what TD learning evaluates.
    """
    out = np.zeros((4, 4), dtype=np.uint8)
    reward = 0
    vertical = action == 0 or action == 2
    toward_start = action == 0 or action == 3
    for k in range(4):
        idx = 0
        for i in range(4):
            v = board[i, k] if vertical else board[k, i]
            idx |= int(v) << (4 * i)
        if toward_start:
            new = LUT_LEFT[idx]
            reward += LUT_SCORE_LEFT[idx]
        else:
            new = LUT_RIGHT[idx]
            reward += LUT_SCORE_RIGHT[idx]
        for i in range(4):
            v = np.uint8((new >> (4 * i)) & 0xF)
            if vertical:
                out[i, k] = v
            else:
                out[k, i] = v
    moved = False
    for i in range(4):
        for j in range(4):
            if out[i, j] != board[i, j]:
                moved = True
    return out, reward, moved


@njit(cache=True)
def spawn_tile(board):
    """Spawn a 2 (p=0.9) or 4 (p=0.1) on a random empty cell, in place.

    Returns False if the board is full.
    """
    n_empty = 0
    for i in range(4):
        for j in range(4):
            if board[i, j] == 0:
                n_empty += 1
    if n_empty == 0:
        return False
    target = np.random.randint(0, n_empty)
    value = 1 if np.random.random() < 0.9 else 2
    c = 0
    for i in range(4):
        for j in range(4):
            if board[i, j] == 0:
                if c == target:
                    board[i, j] = value
                    return True
                c += 1
    return False


@njit(cache=True)
def legal_moves(board):
    """Boolean array of length 4: which actions change the board."""
    mask = np.zeros(4, dtype=np.bool_)
    for a in range(4):
        _, _, moved = apply_move(board, a)
        mask[a] = moved
    return mask


@njit(cache=True)
def is_game_over(board):
    for a in range(4):
        _, _, moved = apply_move(board, a)
        if moved:
            return False
    return True


@njit(cache=True)
def new_game():
    board = np.zeros((4, 4), dtype=np.uint8)
    spawn_tile(board)
    spawn_tile(board)
    return board


@njit(cache=True)
def step(board, action):
    """Full environment step. Returns (new_board, reward, moved, done).

    If the move is illegal (board unchanged), nothing spawns, reward is 0 and
    `moved` is False. `done` is True when no legal moves remain afterwards.
    """
    after, reward, moved = apply_move(board, action)
    if not moved:
        return board.copy(), 0, False, is_game_over(board)
    spawn_tile(after)
    return after, reward, True, is_game_over(after)


def max_tile(board):
    """Largest tile value (not exponent)."""
    m = int(np.max(board))
    return 0 if m == 0 else 1 << m


def to_onehot(board):
    """(4, 4) exponents -> (16, 4, 4) float32 one-hot tensor for DQN input."""
    return (board[None, :, :] == np.arange(16, dtype=np.uint8)[:, None, None]).astype(np.float32)


def render(board):
    rows = []
    for r in board:
        rows.append(" ".join(f"{(1 << int(v)) if v else 0:5d}" for v in r))
    return "\n".join(rows)
