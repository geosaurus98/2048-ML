import numpy as np
import pytest

from game2048.engine import (
    DOWN, LEFT, RIGHT, UP,
    apply_move, is_game_over, legal_moves, max_tile, new_game,
    seed_rng, spawn_tile, step, to_onehot,
)


def row_board(values):
    """Place a row of tile VALUES (2, 4, ...) into row 0 of an otherwise empty board."""
    b = np.zeros((4, 4), dtype=np.uint8)
    for j, v in enumerate(values):
        b[0, j] = 0 if v == 0 else int(np.log2(v))
    return b


def row_values(board, r=0):
    return [0 if v == 0 else 1 << int(v) for v in board[r]]


@pytest.mark.parametrize(
    "row, expected, reward",
    [
        ([2, 2, 2, 2], [4, 4, 0, 0], 8),
        ([2, 2, 4, 0], [4, 4, 0, 0], 4),
        ([4, 2, 2, 0], [4, 4, 0, 0], 4),
        ([2, 2, 2, 0], [4, 2, 0, 0], 4),
        ([2, 0, 2, 0], [4, 0, 0, 0], 4),
        ([0, 0, 0, 2], [2, 0, 0, 0], 0),
        ([2, 4, 2, 4], [2, 4, 2, 4], 0),
        ([4, 4, 8, 0], [8, 8, 0, 0], 8),  # no chain merging in one move
        ([2, 2, 4, 4], [4, 8, 0, 0], 12),
    ],
)
def test_move_left(row, expected, reward):
    after, r, _ = apply_move(row_board(row), LEFT)
    assert row_values(after) == expected
    assert r == reward


@pytest.mark.parametrize(
    "row, expected",
    [
        ([2, 2, 2, 0], [0, 0, 2, 4]),
        ([2, 2, 2, 2], [0, 0, 4, 4]),
        ([0, 2, 2, 4], [0, 0, 4, 4]),
        ([2, 0, 0, 0], [0, 0, 0, 2]),
    ],
)
def test_move_right(row, expected):
    after, _, _ = apply_move(row_board(row), RIGHT)
    assert row_values(after) == expected


def test_move_up_and_down():
    b = np.zeros((4, 4), dtype=np.uint8)
    b[0, 0], b[1, 0], b[3, 0] = 1, 1, 2  # column 0: 2, 2, _, 4
    up, r_up, _ = apply_move(b, UP)
    assert [1 << int(v) if v else 0 for v in up[:, 0]] == [4, 4, 0, 0]
    assert r_up == 4
    down, r_down, _ = apply_move(b, DOWN)
    assert [1 << int(v) if v else 0 for v in down[:, 0]] == [0, 0, 4, 4]
    assert r_down == 4


def test_noop_not_moved():
    b = row_board([2, 4, 8, 16])
    _, r, moved = apply_move(b, LEFT)
    assert not moved and r == 0


def test_apply_move_does_not_mutate_input():
    b = row_board([2, 2, 4, 0])
    before = b.copy()
    apply_move(b, LEFT)
    assert np.array_equal(b, before)


def test_step_illegal_move_spawns_nothing():
    b = row_board([2, 4, 8, 16])
    new, r, moved, _ = step(b, LEFT)
    assert not moved and r == 0
    assert np.array_equal(new, b)


def test_step_spawns_exactly_one_tile():
    seed_rng(0)
    b = row_board([2, 2, 0, 0])
    new, _, moved, _ = step(b, LEFT)
    assert moved
    assert np.count_nonzero(new) == 2  # merged tile + 1 spawn


def test_new_game_has_two_tiles():
    seed_rng(1)
    b = new_game()
    assert np.count_nonzero(b) == 2
    assert set(np.unique(b)) <= {0, 1, 2}


def test_spawn_distribution():
    seed_rng(123)
    twos = fours = 0
    for _ in range(20000):
        b = np.zeros((4, 4), dtype=np.uint8)
        spawn_tile(b)
        v = int(b.max())
        twos += v == 1
        fours += v == 2
    assert abs(fours / 20000 - 0.1) < 0.01


def test_spawn_on_full_board_returns_false():
    b = np.ones((4, 4), dtype=np.uint8)
    assert not spawn_tile(b)


def test_game_over_detection():
    # Checkerboard of 2/4: no empty cells, no equal neighbours.
    b = np.array([[1, 2, 1, 2], [2, 1, 2, 1], [1, 2, 1, 2], [2, 1, 2, 1]], dtype=np.uint8)
    assert is_game_over(b)
    assert not legal_moves(b).any()
    b[3, 3] = 2  # now two equal neighbours (row 3: 2,1,2,2)
    assert not is_game_over(b)


def test_legal_moves_mask():
    b = np.zeros((4, 4), dtype=np.uint8)
    b[0, 0] = 1  # top-left corner tile
    assert list(legal_moves(b)) == [False, True, True, False]


def test_max_exponent_does_not_overflow():
    b = np.zeros((4, 4), dtype=np.uint8)
    b[0, 0] = b[0, 1] = 15
    after, _, _ = apply_move(b, LEFT)
    assert list(after[0, :2]) == [15, 15]  # capped: two 32768s do not merge


def test_max_tile_and_onehot():
    b = np.zeros((4, 4), dtype=np.uint8)
    b[1, 2] = 11
    assert max_tile(b) == 2048
    oh = to_onehot(b)
    assert oh.shape == (16, 4, 4)
    assert oh[11, 1, 2] == 1.0 and oh[0].sum() == 15
    assert np.all(oh.sum(axis=0) == 1)


def test_random_playout_invariants():
    seed_rng(7)
    rng = np.random.default_rng(7)
    for _ in range(20):
        b = new_game()
        done = False
        while not done:
            legal = np.flatnonzero(legal_moves(b))
            b, _, moved, done = step(b, int(rng.choice(legal)))
            assert moved
            assert int(b.max()) <= 15
        assert is_game_over(b)
