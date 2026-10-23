"""Expectimax agent with a hand-crafted heuristic, searched in Numba.

Search tree for depth d (d player moves ahead):
  max node (root) -> chance node -> max node -> ... -> afterstate of move d -> heuristic

Chance nodes average over empty cells, spawning a 2 (p=0.9) or 4 (p=0.1). The
first chance node uses every empty cell; deeper ones use at most `max_samples`
evenly spaced empty cells, so the search is deterministic.
"""

from dataclasses import astuple, dataclass

import numpy as np
from numba import njit

from game2048.engine import apply_move, legal_moves


@dataclass
class HeuristicWeights:
    """Weights of the leaf heuristic. All terms are computed on tile exponents.

    empty:        number of empty cells.
    monotonicity: minus the gradient violations, with one left/right direction shared
                  by all rows and one up/down direction shared by all columns.
    smoothness:   minus sum of |difference| between adjacent non-empty tiles.
    corner:       max exponent if the max tile sits in a corner, else 0.
    game_over:    value of a position where no move is legal.
    """

    empty: float = 2.7
    monotonicity: float = 1.0
    smoothness: float = 0.1
    corner: float = 1.0
    game_over: float = -1000.0

    def to_array(self):
        return np.array(astuple(self), dtype=np.float64)


@njit(cache=True)
def heuristic(board, w):
    empty = 0
    smooth = 0.0
    top = 0
    for i in range(4):
        for j in range(4):
            v = np.int64(board[i, j])
            if v == 0:
                empty += 1
            if v > top:
                top = v
    inc_r = dec_r = inc_c = dec_c = 0.0
    for k in range(4):
        for i in range(3):
            a, b = np.int64(board[k, i]), np.int64(board[k, i + 1])
            if a > b:
                dec_r += a - b
            else:
                inc_r += b - a
            if a != 0 and b != 0:
                smooth -= abs(a - b)
            a, b = np.int64(board[i, k]), np.int64(board[i + 1, k])
            if a > b:
                dec_c += a - b
            else:
                inc_c += b - a
            if a != 0 and b != 0:
                smooth -= abs(a - b)
    mono = -(min(inc_r, dec_r) + min(inc_c, dec_c))
    corner = 0.0
    if (board[0, 0] == top or board[0, 3] == top
            or board[3, 0] == top or board[3, 3] == top):
        corner = float(top)
    return w[0] * empty + w[1] * mono + w[2] * smooth + w[3] * corner


@njit
def _chance_value(board, depth, w, max_samples, sample_all):
    """Expected value of an afterstate with `depth` player moves still to search.

    Averages over spawns; under each spawn takes the best legal move. Max and
    chance levels share one function. Recursive functions are not cached
    (cache=True): loading them from Numba's on-disk cache segfaults.
    """
    rows = np.empty(16, dtype=np.int64)
    cols = np.empty(16, dtype=np.int64)
    n = 0
    for i in range(4):
        for j in range(4):
            if board[i, j] == 0:
                rows[n] = i
                cols[n] = j
                n += 1
    m = n if sample_all or n <= max_samples else max_samples
    total = 0.0
    for k in range(m):
        idx = k if m == n else (k * n) // m
        i, j = rows[idx], cols[idx]
        for tile, p in ((1, 0.9), (2, 0.1)):
            board[i, j] = tile
            best = -np.inf
            for a in range(4):
                after, _, moved = apply_move(board, a)
                if not moved:
                    continue
                if depth == 1:
                    v = heuristic(after, w)
                else:
                    v = _chance_value(after, depth - 1, w, max_samples, False)
                if v > best:
                    best = v
            if best == -np.inf:
                best = w[4]
            total += p * best
        board[i, j] = 0
    return total / m


@njit
def search(board, depth, w, max_samples):
    """Return (best action, its value). Action is -1 if no move is legal."""
    best_a, best_v = -1, -np.inf
    for a in range(4):
        after, _, moved = apply_move(board, a)
        if not moved:
            continue
        if depth == 1:
            v = heuristic(after, w)
        else:
            v = _chance_value(after, depth - 1, w, max_samples, True)
        if best_a == -1 or v > best_v:
            best_a, best_v = a, v
    return best_a, best_v


class ExpectimaxAgent:
    def __init__(self, depth=2, weights=None, max_samples=6):
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.depth = depth
        self.weights = weights if weights is not None else HeuristicWeights()
        self.max_samples = max_samples
        self._w = self.weights.to_array()
        # Compile now so JIT time is not counted against games/sec.
        warm = np.zeros((4, 4), dtype=np.uint8)
        warm[0, 0] = 1
        search(warm, 2, self._w, self.max_samples)

    def act(self, board):
        action, _ = search(board, self.depth, self._w, self.max_samples)
        if action < 0:
            raise ValueError("no legal moves")
        return int(action)
