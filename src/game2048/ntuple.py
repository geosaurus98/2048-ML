"""N-tuple network trained with afterstate TD(0).

Four 6-tuples (two long shapes, two 2x3 rectangles; Yeh et al. 2016), each shared
across the 8 board symmetries, so V(board) sums 32 weight lookups. Cells are
numbered r*4 + c; a tuple's index packs its 6 exponents into 24 bits.

Policy: among legal moves, argmax of r + V(afterstate). TD(0) on afterstates:
  V(after) += alpha * (r_next + V(after_next) - V(after)), target 0 at game end,
with each of the 32 features stepping alpha / 32.

Games run entirely inside Numba (`train_games`). Spawns use Numba's global RNG
(engine.seed_rng); its state cannot be saved, so a resumed run reseeds with
seed + games_done and is reproducible but not bit-identical to an uninterrupted run.
"""

import csv
import json
import os
import time
from pathlib import Path

import numpy as np
from numba import njit

from game2048.engine import apply_move, new_game, seed_rng, spawn_tile

TUPLES = np.array([
    (0, 1, 2, 3, 4, 5),     # long: row 0 + first two cells of row 1
    (4, 5, 6, 7, 8, 9),     # long: row 1 + first two cells of row 2
    (0, 1, 2, 4, 5, 6),     # 2x3 rectangle, rows 0-1
    (4, 5, 6, 8, 9, 10),    # 2x3 rectangle, rows 1-2
], dtype=np.int64)
N_TUPLES, TUPLE_LEN = TUPLES.shape
N_SYM = 8
N_FEATURES = N_TUPLES * N_SYM
TABLE_SIZE = 16 ** TUPLE_LEN


def _symmetry_grids():
    """The 8 images of the cell-id grid under rotations and reflection."""
    grid = np.arange(16).reshape(4, 4)
    grids = []
    for g in (grid, np.fliplr(grid)):
        for k in range(4):
            grids.append(np.rot90(g, k))
    return grids


def _build_sym_cells():
    sym = np.zeros((N_TUPLES, N_SYM, TUPLE_LEN), dtype=np.int64)
    for s, g in enumerate(_symmetry_grids()):
        flat = g.ravel()
        for t in range(N_TUPLES):
            sym[t, s] = flat[TUPLES[t]]
    return sym


SYM_CELLS = _build_sym_cells()


def new_weights():
    """Zero weight table: float32, (4, 16**6) = 268 MB."""
    return np.zeros((N_TUPLES, TABLE_SIZE), dtype=np.float32)


@njit(cache=True)
def tuple_index(board, cells):
    idx = 0
    for k in range(cells.shape[0]):
        c = cells[k]
        idx |= np.int64(board[c // 4, c % 4]) << (4 * k)
    return idx


@njit(cache=True)
def value(board, W):
    v = 0.0
    for t in range(N_TUPLES):
        for s in range(N_SYM):
            v += W[t, tuple_index(board, SYM_CELLS[t, s])]
    return v


@njit(cache=True)
def update(board, W, step):
    """Add `step` to each of the board's 32 feature weights."""
    for t in range(N_TUPLES):
        for s in range(N_SYM):
            W[t, tuple_index(board, SYM_CELLS[t, s])] += step


@njit(cache=True)
def greedy(board, W):
    """Best legal move by r + V(afterstate). Returns (action, reward, afterstate, r + V).

    Action is -1 (and afterstate the unchanged board) if no move is legal.
    """
    best_a, best_r, best_v = -1, 0, -np.inf
    best_after = board
    for a in range(4):
        after, r, moved = apply_move(board, a)
        if not moved:
            continue
        v = r + value(after, W)
        if best_a == -1 or v > best_v:
            best_a, best_r, best_v, best_after = a, r, v, after
    return best_a, best_r, best_after, best_v


@njit(cache=True)
def td_step(prev_after, W, target, alpha):
    """Move V(prev_after) toward `target`; each feature steps alpha / N_FEATURES."""
    delta = target - value(prev_after, W)
    update(prev_after, W, np.float32(alpha / N_FEATURES * delta))
    return delta


@njit(cache=True)
def play_and_learn(W, alpha):
    """Play one full game greedily, learning online. Returns (score, max exponent, moves)."""
    board = new_game()
    prev_after = board
    has_prev = False
    score = 0
    moves = 0
    while True:
        a, r, after, v = greedy(board, W)
        if a < 0:
            if has_prev:
                td_step(prev_after, W, 0.0, alpha)  # terminal: no future reward
            break
        if has_prev:
            # target r + V(after); greedy already computed it as v
            td_step(prev_after, W, v, alpha)
        score += r
        moves += 1
        prev_after = after
        has_prev = True
        board = after.copy()
        spawn_tile(board)
    return score, board.max(), moves


@njit(cache=True)
def train_games(W, n_games, alpha):
    scores = np.zeros(n_games, dtype=np.int64)
    max_exps = np.zeros(n_games, dtype=np.int64)
    moves = np.zeros(n_games, dtype=np.int64)
    for g in range(n_games):
        scores[g], max_exps[g], moves[g] = play_and_learn(W, alpha)
    return scores, max_exps, moves


# ---------------------------------------------------------------- checkpoints

def save_checkpoint(ckpt_dir, W, meta):
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tmp = ckpt_dir / "weights.tmp.npy"
    np.save(tmp, W)
    os.replace(tmp, ckpt_dir / "weights.npy")
    tmp_meta = ckpt_dir / "meta.tmp.json"
    tmp_meta.write_text(json.dumps(meta, indent=2))
    os.replace(tmp_meta, ckpt_dir / "meta.json")


def load_checkpoint(path):
    """`path` is weights.npy or its directory. Returns (W, meta)."""
    path = Path(path)
    if path.is_dir():
        path = path / "weights.npy"
    W = np.load(path)
    if W.shape != (N_TUPLES, TABLE_SIZE) or W.dtype != np.float32:
        raise ValueError(f"unexpected weight table {W.shape} {W.dtype}")
    meta_path = path.with_name("meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return W, meta


# ---------------------------------------------------------------- training

REACH_EXPS = (9, 10, 11, 12, 13)  # 512 .. 8192
CSV_FIELDS = ["games", "total_moves", "mean_score", "max_score",
              "pct_512", "pct_1024", "pct_2048", "pct_4096", "pct_8192",
              "games_per_sec", "elapsed_s"]


def train(seed=0, games=None, max_minutes=None, run_dir="runs", alpha=0.1, block=1000,
          checkpoint_every=10_000, resume=None, overwrite=False, log=print):
    """Train until `games` total games or `max_minutes` (checked between blocks).

    Logs one CSV row per `block` games to runs/ntuple_<seed>.csv and saves
    runs/ntuple_<seed>/weights.npy + meta.json every `checkpoint_every` games
    and on exit. `resume` is a checkpoint path or "auto". Returns (W, meta).
    """
    if games is None and max_minutes is None:
        raise ValueError("set games and/or max_minutes")
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / f"ntuple_{seed}"
    csv_path = run_dir / f"ntuple_{seed}.csv"

    if resume == "auto":
        resume = ckpt_dir if (ckpt_dir / "weights.npy").exists() else None
    if resume is not None:
        W, meta = load_checkpoint(resume)
        alpha = meta.get("alpha", alpha)
        log(f"resumed from {resume} at game {meta['games']}")
    else:
        if csv_path.exists() and not overwrite:
            raise FileExistsError(f"{csv_path} exists; pass resume or overwrite")
        W = new_weights()
        meta = {"seed": seed, "alpha": alpha, "games": 0, "total_moves": 0, "train_seconds": 0.0}
    seed_rng(seed + meta["games"])

    run_dir.mkdir(parents=True, exist_ok=True)
    new_csv = resume is None
    csv_file = open(csv_path, "w" if new_csv else "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if new_csv:
        writer.writeheader()

    start = time.perf_counter()
    last_ckpt = meta["games"]
    try:
        while games is None or meta["games"] < games:
            if max_minutes is not None and time.perf_counter() - start >= 60 * max_minutes:
                break
            n = block if games is None else min(block, games - meta["games"])
            t0 = time.perf_counter()
            scores, max_exps, moves = train_games(W, n, alpha)
            dt = time.perf_counter() - t0
            meta["games"] += n
            meta["total_moves"] += int(moves.sum())
            meta["train_seconds"] += dt
            row = {
                "games": meta["games"],
                "total_moves": meta["total_moves"],
                "mean_score": f"{scores.mean():.1f}",
                "max_score": int(scores.max()),
                "games_per_sec": f"{n / dt:.1f}",
                "elapsed_s": f"{time.perf_counter() - start:.1f}",
            }
            for e in REACH_EXPS:
                row[f"pct_{1 << e}"] = f"{100.0 * np.mean(max_exps >= e):.1f}"
            writer.writerow(row)
            csv_file.flush()
            log(f"games {meta['games']:>7d}  mean {row['mean_score']:>9}  "
                f"2048 {row['pct_2048']:>5}%  4096 {row['pct_4096']:>5}%  {row['games_per_sec']} games/s")
            if meta["games"] - last_ckpt >= checkpoint_every:
                save_checkpoint(ckpt_dir, W, meta)
                last_ckpt = meta["games"]
    finally:
        csv_file.close()
        save_checkpoint(ckpt_dir, W, meta)
    log(f"stopped at game {meta['games']}; checkpoint {ckpt_dir / 'weights.npy'}")
    return W, meta


class NTupleAgent:
    """Greedy policy: argmax over legal moves of r + V(afterstate)."""

    def __init__(self, checkpoint):
        self.W, self.meta = load_checkpoint(checkpoint)
        warm = np.zeros((4, 4), dtype=np.uint8)
        warm[0, 0] = 1
        greedy(warm, self.W)  # compile before evaluate starts its clock

    def act(self, board):
        a, _, _, _ = greedy(board, self.W)
        if a < 0:
            raise ValueError("no legal moves")
        return int(a)
