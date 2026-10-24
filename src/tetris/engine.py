"""Fast headless Tetris engine: "placement" Tetris, as in the RL literature.

Board: (H, W) uint8 array, row 0 at the top, 1 = filled. W is fixed at 10; the
standard height is 20, but any height works (CEM trains on shorter wells).
Pieces: 0..6 = I, O, T, S, Z, J, L, drawn uniformly at random (no bag, no preview).
Actions: rotation * W + column, where column is the leftmost column of the
rotated piece, which is hard-dropped straight down from above the board.
An action is legal if the piece has that rotation and it fits between the walls,
so every piece always has legal actions. A legal placement that leaves part of
the piece above row 0 "tops out" and ends the game.
Reward: lines cleared by the placement.
"""

import numpy as np
from numba import njit

WIDTH = 10
HEIGHT = 20
N_PIECES = 7
N_ROT = 4
N_ACTIONS = N_ROT * WIDTH
PIECE_NAMES = "IOTSZJL"

_BASE_SHAPES = (
    ((0, 0), (0, 1), (0, 2), (0, 3)),  # I
    ((0, 0), (0, 1), (1, 0), (1, 1)),  # O
    ((0, 0), (0, 1), (0, 2), (1, 1)),  # T
    ((0, 1), (0, 2), (1, 0), (1, 1)),  # S
    ((0, 0), (0, 1), (1, 1), (1, 2)),  # Z
    ((0, 0), (1, 0), (1, 1), (1, 2)),  # J
    ((0, 2), (1, 0), (1, 1), (1, 2)),  # L
)


def _build_tables():
    """Distinct rotations of each piece as (row, col) cells in their bounding box."""
    cells = np.zeros((N_PIECES, N_ROT, 4, 2), dtype=np.int64)
    n_rot = np.zeros(N_PIECES, dtype=np.int64)
    widths = np.zeros((N_PIECES, N_ROT), dtype=np.int64)
    heights = np.zeros((N_PIECES, N_ROT), dtype=np.int64)
    bottoms = np.full((N_PIECES, N_ROT, 4), -1, dtype=np.int64)  # lowest row per piece column
    for p, shape in enumerate(_BASE_SHAPES):
        seen = []
        cur = list(shape)
        for _ in range(4):
            r0 = min(r for r, _ in cur)
            c0 = min(c for _, c in cur)
            norm = sorted((r - r0, c - c0) for r, c in cur)
            if norm not in seen:
                seen.append(norm)
            cur = [(c, -r) for r, c in cur]  # rotate 90 degrees clockwise
        n_rot[p] = len(seen)
        for k, norm in enumerate(seen):
            cells[p, k] = norm
            widths[p, k] = max(c for _, c in norm) + 1
            heights[p, k] = max(r for r, _ in norm) + 1
            for r, c in norm:
                bottoms[p, k, c] = max(bottoms[p, k, c], r)
    return cells, n_rot, widths, heights, bottoms


CELLS, N_ROTATIONS, PIECE_W, PIECE_H, BOTTOM = _build_tables()


@njit(cache=True)
def seed_rng(seed):
    """Seed Numba's internal RNG (separate from NumPy's global RNG)."""
    np.random.seed(seed)


@njit(cache=True)
def random_piece():
    return np.random.randint(0, N_PIECES)


@njit(cache=True)
def new_board(height=HEIGHT):
    return np.zeros((height, WIDTH), dtype=np.uint8)


@njit(cache=True)
def legal_actions(piece):
    """Boolean array of length N_ACTIONS: which placements exist for this piece."""
    mask = np.zeros(N_ACTIONS, dtype=np.bool_)
    for rot in range(N_ROTATIONS[piece]):
        for col in range(WIDTH - PIECE_W[piece, rot] + 1):
            mask[rot * WIDTH + col] = True
    return mask


@njit(cache=True)
def is_legal(piece, action):
    if action < 0 or action >= N_ACTIONS:
        return False
    rot, col = action // WIDTH, action % WIDTH
    return rot < N_ROTATIONS[piece] and col + PIECE_W[piece, rot] <= WIDTH


@njit(cache=True)
def drop_row(board, piece, rot, col):
    """Top row of the piece's bounding box after a hard drop; negative means it tops out."""
    H = board.shape[0]
    row = H
    for j in range(PIECE_W[piece, rot]):
        top = H
        for r in range(H):
            if board[r, col + j]:
                top = r
                break
        cand = top - 1 - BOTTOM[piece, rot, j]
        if cand < row:
            row = cand
    return row


@njit(cache=True)
def place(board, piece, action):
    """Drop a piece and clear full lines. Returns (after, lines, row, eroded, ok).

    `row` is the landing row of the piece's top edge and `eroded` the number of the
    piece's own cells removed by line clears (both used by the features). If the
    placement tops out, ok is False and `after` is an unchanged copy of the board.
    The input board is never modified; `action` must be legal.
    """
    H = board.shape[0]
    rot, col = action // WIDTH, action % WIDTH
    row = drop_row(board, piece, rot, col)
    if row < 0:
        return board.copy(), 0, row, 0, False
    out = board.copy()
    for k in range(4):
        out[row + CELLS[piece, rot, k, 0], col + CELLS[piece, rot, k, 1]] = 1
    full = np.zeros(H, dtype=np.bool_)
    lines = 0
    for r in range(row, row + PIECE_H[piece, rot]):
        f = True
        for c in range(WIDTH):
            if out[r, c] == 0:
                f = False
                break
        if f:
            full[r] = True
            lines += 1
    eroded = 0
    if lines > 0:
        for k in range(4):
            if full[row + CELLS[piece, rot, k, 0]]:
                eroded += 1
        eroded *= lines
        dst = H - 1
        for r in range(H - 1, -1, -1):
            if not full[r]:
                if dst != r:
                    for c in range(WIDTH):
                        out[dst, c] = out[r, c]
                dst -= 1
        for r in range(dst + 1):
            for c in range(WIDTH):
                out[r, c] = 0
    return out, lines, row, eroded, True


@njit(cache=True)
def step(board, piece, action):
    """Full environment step. Returns (new_board, lines, next_piece, done).

    `done` is True when the placement tops out (the board is then unchanged).
    The next piece is drawn either way, so RNG use does not depend on the outcome.
    """
    after, lines, _, _, ok = place(board, piece, action)
    return after, lines, random_piece(), not ok


def render(board, piece=None):
    rows = ["|" + "".join("#" if v else "." for v in r) + "|" for r in board]
    rows.append("+" + "-" * board.shape[1] + "+")
    if piece is not None:
        rows.insert(0, f"piece: {PIECE_NAMES[piece]}")
    return "\n".join(rows)
