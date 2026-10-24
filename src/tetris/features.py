"""Placement features and the linear greedy policy (heuristic and CEM agents).

Eight features of a placement; all but the first two are read off the
afterstate board (after line clears):
  0 landing height      height of the piece's centre where it landed
  1 eroded cells        lines cleared * the piece's own cells those lines removed
  2 row transitions     filled/empty changes along each row (walls count as filled)
  3 column transitions  changes down each column (the floor counts as filled)
  4 holes               empty cells with a filled cell somewhere above them
  5 cumulative wells    each well of depth d adds 1 + 2 + ... + d
  6 hole depth          filled cells above each hole, summed over holes
  7 rows with holes     rows containing at least one hole
0-5 are Pierre Dellacherie's features (as reported by Fahey 2003); 6-7 come from
Thiery & Scherrer 2009.

The linear policy picks the legal placement maximising weights . features.
Placements that top out are only chosen when every placement does.
"""

import numpy as np
from numba import njit

from tetris.engine import N_ROTATIONS, PIECE_H, PIECE_W, WIDTH, place

N_FEATURES = 8
FEATURE_NAMES = ("landing_height", "eroded_cells", "row_transitions", "col_transitions",
                 "holes", "cum_wells", "hole_depth", "rows_with_holes")
# Dellacherie's hand-tuned weights; the two extra features are unused.
DELLACHERIE_WEIGHTS = np.array([-1.0, 1.0, -1.0, -1.0, -4.0, -1.0, 0.0, 0.0])


@njit(cache=True)
def board_features(board, out):
    """Fill out[2:8] with the afterstate features of `board`."""
    H, W = board.shape
    row_t = 0
    for r in range(H):
        prev = 1
        for c in range(W):
            v = board[r, c]
            if v != prev:
                row_t += 1
            prev = v
        if prev == 0:
            row_t += 1
    col_t = holes = wells = hole_depth = 0
    row_has_hole = np.zeros(H, dtype=np.bool_)
    for c in range(W):
        above = 0
        depth = 0
        for r in range(H):
            v = board[r, c]
            if r > 0 and v != board[r - 1, c]:
                col_t += 1
            if v:
                above += 1
                depth = 0
            else:
                if above > 0:
                    holes += 1
                    hole_depth += above
                    row_has_hole[r] = True
                left = c == 0 or board[r, c - 1] != 0
                right = c == W - 1 or board[r, c + 1] != 0
                if left and right:
                    depth += 1
                    wells += depth
                else:
                    depth = 0
        if board[H - 1, c] == 0:
            col_t += 1  # bottom cell against the filled floor
    out[2] = row_t
    out[3] = col_t
    out[4] = holes
    out[5] = wells
    out[6] = hole_depth
    out[7] = row_has_hole.sum()


@njit(cache=True)
def placement_features(after, piece, rot, row, eroded, out):
    """Fill `out` (length N_FEATURES) for a placement returned by engine.place."""
    out[0] = after.shape[0] - row - (PIECE_H[piece, rot] - 1) / 2.0
    out[1] = eroded
    board_features(after, out)


@njit(cache=True)
def best_action(board, piece, weights):
    """Greedy placement under a linear evaluation. Returns (action, afterstate, lines, ok).

    ok is False only when every placement tops out; the action is then the first
    legal one and the afterstate an unchanged copy of the board.
    """
    f = np.zeros(N_FEATURES)
    best_a, best_lines, best_v = -1, 0, -np.inf
    best_after = board
    for rot in range(N_ROTATIONS[piece]):
        for col in range(WIDTH - PIECE_W[piece, rot] + 1):
            a = rot * WIDTH + col
            after, lines, row, eroded, ok = place(board, piece, a)
            if not ok:
                continue
            placement_features(after, piece, rot, row, eroded, f)
            v = 0.0
            for k in range(N_FEATURES):
                v += weights[k] * f[k]
            if best_a == -1 or v > best_v:
                best_a, best_lines, best_v, best_after = a, lines, v, after
    if best_a == -1:
        return 0, board.copy(), 0, False
    return best_a, best_after, best_lines, True


@njit(cache=True)
def play_sequence(weights, pieces, height):
    """Play the linear policy on a fixed piece sequence. Returns (lines, pieces placed, topped out).

    The game stops when the sequence runs out, which caps its length.
    """
    board = np.zeros((height, WIDTH), dtype=np.uint8)
    lines = 0
    for i in range(pieces.shape[0]):
        _, after, n, ok = best_action(board, pieces[i], weights)
        if not ok:
            return lines, i, True
        lines += n
        board = after
    return lines, pieces.shape[0], False


class LinearAgent:
    """Greedy placement by weights . features (the heuristic and CEM agents)."""

    def __init__(self, weights=DELLACHERIE_WEIGHTS):
        self.weights = np.asarray(weights, dtype=np.float64)
        if self.weights.shape != (N_FEATURES,):
            raise ValueError(f"expected {N_FEATURES} weights, got shape {self.weights.shape}")
        best_action(np.zeros((4, WIDTH), dtype=np.uint8), 0, self.weights)  # compile before timing

    def act(self, board, piece):
        a, _, _, _ = best_action(board, piece, self.weights)
        return int(a)
