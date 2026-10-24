"""Noisy cross-entropy method over the linear policy's 8 feature weights.

Szita & Lorincz 2006 ("Learning Tetris using the noisy cross-entropy method"),
with the features of Thiery & Scherrer 2009 (features.py). Each iteration:
  1. sample `population` weight vectors from N(mean, diag(std^2));
  2. play each on the same `games_per_eval` piece sequences (common random
     numbers: every candidate faces identical games, so selection compares
     weights rather than luck);
  3. keep the top `elite_frac` by mean lines and refit mean and std to them,
     adding noise to the variance, max(noise - iteration / noise_decay, min_noise),
     so the distribution does not collapse before it has found good weights.
     Szita & Lorincz let the noise reach 0; the floor keeps a converged run
     searching locally instead of resampling one point.
The policy is invariant to scaling the weights, so only their direction matters.

Fitness is measured on a `height` x 10 board (10 by default): games there end
after a few thousand pieces, while on 10x20 good weights outlast any practical
cap and selection would have nothing to compare. Games are still capped at
`max_pieces`. Candidates play in parallel (Numba prange).

All randomness of iteration t comes from SeedSequence([seed, t]), so a run is
reproducible and a resumed run is bit-identical to an uninterrupted one.
"""

import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numba
import numpy as np
from numba import njit, prange

from tetris.engine import N_PIECES
from tetris.features import FEATURE_NAMES, N_FEATURES, play_sequence


@dataclass
class CEMConfig:
    population: int = 100
    elite_frac: float = 0.1
    games_per_eval: int = 4
    height: int = 10
    max_pieces: int = 50_000
    init_std: float = 10.0
    noise: float = 4.0  # variance added to std^2 at iteration 0
    noise_decay: float = 10.0  # iterations per unit of noise removed
    min_noise: float = 0.5  # learned weights end up around 5-50 in magnitude

    @property
    def n_elite(self):
        return max(1, int(round(self.population * self.elite_frac)))


@njit(parallel=True, cache=True)
def play_population(weights, pieces, height):
    """Play every candidate (row of `weights`) on every sequence (row of `pieces`).

    Returns (lines, placed), both (n_candidates, n_games).
    """
    n, g = weights.shape[0], pieces.shape[0]
    lines = np.zeros((n, g), dtype=np.int64)
    placed = np.zeros((n, g), dtype=np.int64)
    for k in prange(n * g):
        i, j = k // g, k % g
        lines[i, j], placed[i, j], _ = play_sequence(weights[i], pieces[j], height)
    return lines, placed


def iteration_rng(seed, iteration):
    return np.random.default_rng(np.random.SeedSequence([seed, iteration]))


def cem_iteration(mean, std, iteration, seed, config):
    """One CEM update. Returns (new_mean, new_std, stats)."""
    c = config
    rng = iteration_rng(seed, iteration)
    candidates = mean + std * rng.standard_normal((c.population, N_FEATURES))
    pieces = rng.integers(0, N_PIECES, size=(c.games_per_eval, c.max_pieces))
    with numba.parallel_chunksize(1):  # games vary a lot in length; hand them out one at a time
        lines, placed = play_population(candidates, pieces, c.height)
    fitness = lines.mean(axis=1)
    elite = np.argsort(-fitness, kind="stable")[:c.n_elite]
    new_mean = candidates[elite].mean(axis=0)
    noise = max(c.noise - iteration / c.noise_decay, c.min_noise)
    new_std = np.sqrt(candidates[elite].var(axis=0) + noise)
    stats = {
        "mean_fitness": float(fitness.mean()),
        "elite_fitness": float(fitness[elite].mean()),
        "best_fitness": float(fitness[elite[0]]),
        "best_weights": candidates[elite[0]].tolist(),
        "pieces": int(placed.sum()),
        "noise": noise,
    }
    return new_mean, new_std, stats


# ---------------------------------------------------------------- checkpoints

def save_state(ckpt_dir, state):
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tmp = ckpt_dir / "state.tmp.json"
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, ckpt_dir / "state.json")


def load_state(path):
    """`path` is state.json or its directory."""
    path = Path(path)
    if path.is_dir():
        path = path / "state.json"
    return json.loads(path.read_text())


def load_weights(path):
    """The policy weights of a checkpoint: the mean of the sampling distribution."""
    w = np.array(load_state(path)["mean"], dtype=np.float64)
    if w.shape != (N_FEATURES,):
        raise ValueError(f"unexpected weight vector of shape {w.shape}")
    return w


# ---------------------------------------------------------------- training

CSV_FIELDS = (["iteration", "total_pieces", "mean_fitness", "elite_fitness", "best_fitness", "noise",
               "pieces_per_sec", "elapsed_s"] + [f"w_{name}" for name in FEATURE_NAMES])


def train(seed=0, iterations=None, max_minutes=None, run_dir="runs", config=None, checkpoint_every=1,
          resume=None, overwrite=False, log=print):
    """Run CEM until `iterations` total iterations or `max_minutes` (checked between iterations).

    Logs one CSV row per iteration to runs/tetris_cem_<seed>.csv (fitness and the
    new mean weights) and saves runs/tetris_cem_<seed>/state.json every
    `checkpoint_every` iterations and on exit. `resume` is a checkpoint path or
    "auto". Returns the final state dict.
    """
    if iterations is None and max_minutes is None:
        raise ValueError("set iterations and/or max_minutes")
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / f"tetris_cem_{seed}"
    csv_path = run_dir / f"tetris_cem_{seed}.csv"

    if resume == "auto":
        resume = ckpt_dir if (ckpt_dir / "state.json").exists() else None
    if resume is not None:
        state = load_state(resume)
        config = CEMConfig(**state["config"])
        log(f"resumed from {resume} at iteration {state['iteration']}")
    else:
        if csv_path.exists() and not overwrite:
            raise FileExistsError(f"{csv_path} exists; pass resume or overwrite")
        config = config or CEMConfig()
        state = {"seed": seed, "config": asdict(config), "iteration": 0, "total_pieces": 0,
                 "train_seconds": 0.0, "mean": [0.0] * N_FEATURES,
                 "std": [config.init_std] * N_FEATURES, "best_fitness": None, "best_weights": None}
    mean, std = np.array(state["mean"]), np.array(state["std"])

    run_dir.mkdir(parents=True, exist_ok=True)
    new_csv = resume is None
    csv_file = open(csv_path, "w" if new_csv else "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if new_csv:
        writer.writeheader()

    start = time.perf_counter()
    try:
        while iterations is None or state["iteration"] < iterations:
            if max_minutes is not None and time.perf_counter() - start >= 60 * max_minutes:
                break
            t0 = time.perf_counter()
            mean, std, stats = cem_iteration(mean, std, state["iteration"], seed, config)
            dt = time.perf_counter() - t0
            state["iteration"] += 1
            state["total_pieces"] += stats["pieces"]
            state["train_seconds"] += dt
            state["mean"], state["std"] = mean.tolist(), std.tolist()
            state["best_fitness"], state["best_weights"] = stats["best_fitness"], stats["best_weights"]
            row = {
                "iteration": state["iteration"],
                "total_pieces": state["total_pieces"],
                "mean_fitness": f"{stats['mean_fitness']:.1f}",
                "elite_fitness": f"{stats['elite_fitness']:.1f}",
                "best_fitness": f"{stats['best_fitness']:.1f}",
                "noise": f"{stats['noise']:.2f}",
                "pieces_per_sec": f"{stats['pieces'] / dt:.0f}",
                "elapsed_s": f"{time.perf_counter() - start:.1f}",
            }
            for name, w in zip(FEATURE_NAMES, mean):
                row[f"w_{name}"] = f"{w:.4f}"
            writer.writerow(row)
            csv_file.flush()
            log(f"iter {state['iteration']:>4d}  mean {row['mean_fitness']:>8}  elite {row['elite_fitness']:>8}  "
                f"best {row['best_fitness']:>8} lines  {row['pieces_per_sec']} pieces/s")
            if state["iteration"] % checkpoint_every == 0:
                save_state(ckpt_dir, state)
    finally:
        csv_file.close()
        save_state(ckpt_dir, state)
    log(f"stopped at iteration {state['iteration']}; checkpoint {ckpt_dir / 'state.json'}")
    return state
