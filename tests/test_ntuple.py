import csv

import numpy as np
import pytest

from game2048.engine import legal_moves
from game2048.evaluate import evaluate
from game2048.ntuple import (
    N_FEATURES, SYM_CELLS, TUPLES, NTupleAgent, greedy, new_weights, save_checkpoint,
    td_step, train, tuple_index, value,
)

from test_evaluate import CheckingAgent

QUIET = {"log": lambda *_: None}


def board_transforms(board):
    """All 8 rotations/reflections of a board."""
    out = []
    for b in (board, np.fliplr(board)):
        for k in range(4):
            out.append(np.ascontiguousarray(np.rot90(b, k)))
    return out


def random_board(rng, high=12):
    return rng.integers(0, high, size=(4, 4)).astype(np.uint8)


@pytest.fixture(scope="module")
def random_weights():
    rng = np.random.default_rng(0)
    return rng.standard_normal((len(TUPLES), 16 ** 6), dtype=np.float32)


def test_tuple_index_matches_reference():
    rng = np.random.default_rng(1)
    for _ in range(50):
        b = random_board(rng, 16)
        for cells in TUPLES:
            expected = sum(int(b.flat[c]) * 16 ** k for k, c in enumerate(cells))
            assert tuple_index(b, cells) == expected


def test_sym_cells_are_the_8_images_of_each_tuple():
    rng = np.random.default_rng(2)
    b = random_board(rng, 16)
    for t in range(len(TUPLES)):
        assert list(SYM_CELLS[t, 0]) == list(TUPLES[t])
        # 8 distinct ordered variants (the central 2x3 rectangle covers only 4 cell sets,
        # each read in two orders, so compare ordered tuples rather than sets).
        assert len({tuple(v) for v in SYM_CELLS[t].tolist()}) == 8
        # Reading the base tuple on a transformed board == reading some variant on the original.
        variant_idx = {tuple_index(b, SYM_CELLS[t, s]) for s in range(8)}
        transformed_idx = {tuple_index(g, TUPLES[t]) for g in board_transforms(b)}
        assert variant_idx == transformed_idx


def test_value_is_symmetric(random_weights):
    rng = np.random.default_rng(3)
    for _ in range(20):
        b = random_board(rng)
        vals = [value(g, random_weights) for g in board_transforms(b)]
        assert np.allclose(vals, vals[0], rtol=1e-5, atol=1e-3)


def asymmetric_board():
    b = np.arange(16, dtype=np.uint8).reshape(4, 4) % 13
    indices = {(t, tuple_index(b, SYM_CELLS[t, s])) for t in range(len(TUPLES)) for s in range(8)}
    assert len(indices) == N_FEATURES  # every feature hits its own weight
    return b


def test_td_step_moves_value_by_alpha_delta():
    W = new_weights()
    b = asymmetric_board()
    W[:, :] = 0
    td_step(b, W, 0.0, 0.1)  # no-op at V = target = 0
    assert value(b, W) == 0.0
    delta = td_step(b, W, 100.0, 0.1)
    assert delta == pytest.approx(100.0)
    assert value(b, W) == pytest.approx(10.0, rel=1e-5)
    delta = td_step(b, W, 50.0, 0.5)
    assert delta == pytest.approx(40.0, rel=1e-5)
    assert value(b, W) == pytest.approx(30.0, rel=1e-5)


def test_terminal_target_pulls_toward_zero():
    W = new_weights()
    b = asymmetric_board()
    td_step(b, W, 200.0, 1.0)
    assert value(b, W) == pytest.approx(200.0, rel=1e-5)
    td_step(b, W, 0.0, 0.25)  # terminal update: target 0
    assert value(b, W) == pytest.approx(150.0, rel=1e-5)


def corner_board():
    b = np.zeros((4, 4), dtype=np.uint8)
    b[0, 0] = 3  # only RIGHT (1) and DOWN (2) are legal
    return b


def test_greedy_never_picks_illegal_even_if_rigged():
    W = new_weights()
    b = corner_board()
    # An illegal move leaves the board unchanged; make that "afterstate" look great.
    td_step(b, W, 1e6, 1.0)
    a, r, after, _ = greedy(b, W)
    assert legal_moves(b)[a]
    assert not np.array_equal(after, b)


def test_greedy_maximizes_reward_plus_value():
    W = new_weights()
    b = np.zeros((4, 4), dtype=np.uint8)
    b[0, :2] = [1, 1]  # LEFT/RIGHT merge (+4); UP is illegal, DOWN legal without reward
    a, r, _, v = greedy(b, W)
    assert a in (1, 3) and r == 4 and v == 4.0


def test_greedy_returns_minus_one_on_dead_board():
    b = np.array([[1, 2, 1, 2], [2, 1, 2, 1], [1, 2, 1, 2], [2, 1, 2, 1]], dtype=np.uint8)
    a, _, _, _ = greedy(b, new_weights())
    assert a == -1


def test_agent_plays_legal_moves_and_raises_when_dead(tmp_path, random_weights):
    save_checkpoint(tmp_path, random_weights, {"games": 0})
    inner = NTupleAgent(tmp_path / "weights.npy")
    agent = CheckingAgent(inner)
    agent.seed = lambda seed: None
    evaluate(agent, n_games=3, seed=0)
    assert agent.n_actions > 0 and agent.n_illegal == 0
    dead = np.array([[1, 2, 1, 2], [2, 1, 2, 1], [1, 2, 1, 2], [2, 1, 2, 1]], dtype=np.uint8)
    with pytest.raises(ValueError):
        inner.act(dead)


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def without_timing(rows):
    return [{k: v for k, v in r.items() if k not in ("games_per_sec", "elapsed_s")} for r in rows]


def test_seeded_training_is_reproducible(tmp_path):
    W1, _ = train(seed=5, games=20, block=10, run_dir=tmp_path / "a", **QUIET)
    W2, _ = train(seed=5, games=20, block=10, run_dir=tmp_path / "b", **QUIET)
    assert np.array_equal(W1, W2)
    assert np.abs(W1).sum() > 0  # it learned something
    rows = read_rows(tmp_path / "a/ntuple_5.csv")
    assert without_timing(rows) == without_timing(read_rows(tmp_path / "b/ntuple_5.csv"))
    assert [int(r["games"]) for r in rows] == [10, 20]


def test_checkpoint_and_resume(tmp_path):
    W, meta = train(seed=1, games=20, block=10, run_dir=tmp_path, **QUIET)
    assert (tmp_path / "ntuple_1/weights.npy").exists()
    W2, meta2 = train(seed=1, games=40, block=10, run_dir=tmp_path, resume="auto", **QUIET)
    assert meta2["games"] == 40 and meta2["total_moves"] > meta["total_moves"]
    assert [int(r["games"]) for r in read_rows(tmp_path / "ntuple_1.csv")] == [10, 20, 30, 40]


def test_fresh_run_refuses_to_clobber_csv(tmp_path):
    train(seed=0, games=2, block=2, run_dir=tmp_path, **QUIET)
    with pytest.raises(FileExistsError):
        train(seed=0, games=2, block=2, run_dir=tmp_path, **QUIET)
