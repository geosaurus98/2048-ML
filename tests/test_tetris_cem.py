import csv

import numpy as np
import pytest

from tetris.agents import AGENTS
from tetris.cem import CEMConfig, cem_iteration, load_weights, play_population, train
from tetris.engine import legal_actions, new_board
from tetris.features import DELLACHERIE_WEIGHTS, N_FEATURES, play_sequence

QUIET = {"log": lambda *_: None}
TINY = CEMConfig(population=8, elite_frac=0.25, games_per_eval=2, height=8, max_pieces=300)


def test_play_population_matches_play_sequence():
    rng = np.random.default_rng(0)
    W = np.vstack([DELLACHERIE_WEIGHTS, rng.standard_normal((3, N_FEATURES))])
    pieces = rng.integers(0, 7, size=(2, 400))
    lines, placed = play_population(W, pieces, 10)
    for i in range(len(W)):
        for j in range(len(pieces)):
            l, p, _ = play_sequence(W[i], pieces[j], 10)
            assert (lines[i, j], placed[i, j]) == (l, p)


def test_iteration_refits_to_the_elite():
    mean, std = np.zeros(N_FEATURES), np.full(N_FEATURES, 10.0)
    m1, s1, stats = cem_iteration(mean, std, 0, 0, TINY)
    m2, s2, _ = cem_iteration(mean, std, 0, 0, TINY)
    assert np.array_equal(m1, m2) and np.array_equal(s1, s2)  # deterministic in (seed, iteration)
    assert stats["elite_fitness"] >= stats["mean_fitness"]
    assert stats["best_fitness"] >= stats["elite_fitness"]
    assert np.all(s1 ** 2 >= TINY.noise - 1e-9)  # noise added at iteration 0


def test_noise_floor():
    mean, std = np.zeros(N_FEATURES), np.zeros(N_FEATURES)  # every candidate identical
    _, s, stats = cem_iteration(mean, std, 10_000, 0, TINY)
    assert stats["noise"] == TINY.min_noise
    assert np.allclose(s ** 2, TINY.min_noise)


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def without_timing(rows):
    return [{k: v for k, v in r.items() if k not in ("pieces_per_sec", "elapsed_s")} for r in rows]


def test_seeded_training_is_reproducible(tmp_path):
    a = train(seed=3, iterations=3, run_dir=tmp_path / "a", config=TINY, **QUIET)
    b = train(seed=3, iterations=3, run_dir=tmp_path / "b", config=TINY, **QUIET)
    assert a["mean"] == b["mean"] and a["std"] == b["std"]
    rows = read_rows(tmp_path / "a/tetris_cem_3.csv")
    assert without_timing(rows) == without_timing(read_rows(tmp_path / "b/tetris_cem_3.csv"))
    assert [int(r["iteration"]) for r in rows] == [1, 2, 3]
    assert a["mean"] != [0.0] * N_FEATURES


def test_resume_is_bit_identical(tmp_path):
    straight = train(seed=1, iterations=4, run_dir=tmp_path / "a", config=TINY, **QUIET)
    train(seed=1, iterations=2, run_dir=tmp_path / "b", config=TINY, **QUIET)
    resumed = train(seed=1, iterations=4, run_dir=tmp_path / "b", resume="auto", **QUIET)
    assert resumed["mean"] == straight["mean"] and resumed["iteration"] == 4
    assert [int(r["iteration"]) for r in read_rows(tmp_path / "b/tetris_cem_1.csv")] == [1, 2, 3, 4]


def test_fresh_run_refuses_to_clobber_csv(tmp_path):
    train(seed=0, iterations=1, run_dir=tmp_path, config=TINY, **QUIET)
    with pytest.raises(FileExistsError):
        train(seed=0, iterations=1, run_dir=tmp_path, config=TINY, **QUIET)


def test_cem_agent_loads_the_mean_weights(tmp_path):
    state = train(seed=0, iterations=2, run_dir=tmp_path, config=TINY, **QUIET)
    ckpt = tmp_path / "tetris_cem_0"
    assert np.array_equal(load_weights(ckpt), state["mean"])
    assert np.array_equal(load_weights(ckpt / "state.json"), state["mean"])
    agent = AGENTS["cem"](checkpoint=str(ckpt))
    assert legal_actions(2)[agent.act(new_board(), 2)]
    with pytest.raises(ValueError):
        AGENTS["cem"]()
