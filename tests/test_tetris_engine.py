import numpy as np
import pytest

from tetris.engine import (
    CELLS, HEIGHT, N_ACTIONS, N_ROTATIONS, PIECE_H, PIECE_W, WIDTH,
    is_legal, legal_actions, new_board, place, random_piece, render, seed_rng, step,
)

I, O, T, S, Z, J, L = range(7)
VERTICAL_I = 1 * WIDTH  # rotation 1 of I, column 0


def piece_cells(piece, rot):
    return {tuple(c) for c in CELLS[piece, rot].tolist()}


def test_rotation_counts_and_shapes():
    assert list(N_ROTATIONS) == [2, 1, 4, 2, 2, 4, 4]
    for p in range(7):
        rots = [frozenset(piece_cells(p, k)) for k in range(N_ROTATIONS[p])]
        assert len(set(rots)) == len(rots)  # distinct
        for k, cells in enumerate(rots):
            assert len(cells) == 4
            assert min(r for r, _ in cells) == 0 and min(c for _, c in cells) == 0
            assert PIECE_W[p, k] == max(c for _, c in cells) + 1
            assert PIECE_H[p, k] == max(r for r, _ in cells) + 1


@pytest.mark.parametrize("piece, n_legal", [(I, 17), (O, 9), (T, 34), (S, 17), (Z, 17), (J, 34), (L, 34)])
def test_legal_action_counts(piece, n_legal):
    assert legal_actions(piece).sum() == n_legal


def test_is_legal_matches_mask():
    for p in range(7):
        mask = legal_actions(p)
        for a in range(-2, N_ACTIONS + 2):
            assert is_legal(p, a) == (0 <= a < N_ACTIONS and mask[a])


def test_drop_on_empty_board_lands_on_floor():
    after, lines, row, eroded, ok = place(new_board(), I, 3)  # flat I at columns 3-6
    assert ok and lines == 0 and eroded == 0 and row == HEIGHT - 1
    assert list(after[-1]) == [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
    assert after.sum() == 4


def test_piece_rests_on_overhang():
    b = new_board()
    b[HEIGHT - 2, :2] = 1  # a ledge with an empty row beneath it
    after, _, row, _, ok = place(b, O, 0)
    assert ok and row == HEIGHT - 4
    assert after[HEIGHT - 4:HEIGHT - 2, :2].all() and not after[HEIGHT - 1].any()


def test_single_line_clear():
    b = new_board()
    b[-1, :6] = 1
    after, lines, _, eroded, ok = place(b, I, 6)
    assert ok and lines == 1 and eroded == 4
    assert after.sum() == 0


def test_tetris_clears_four_lines():
    b = new_board()
    b[-4:, :9] = 1
    after, lines, _, eroded, ok = place(b, I, VERTICAL_I + 9)
    assert ok and lines == 4 and eroded == 16
    assert after.sum() == 0


def test_rows_above_a_clear_shift_down():
    b = new_board()
    b[-1, :9] = 1
    b[-2, 0] = 1
    after, lines, _, eroded, ok = place(b, I, VERTICAL_I + 9)
    assert ok and lines == 1 and eroded == 1
    expected = new_board()
    expected[-1, [0, 9]] = 1
    expected[-3:-1, 9] = 1
    assert np.array_equal(after, expected)


def test_top_out_leaves_board_unchanged():
    b = new_board()
    b[1:, 0] = 1
    after, lines, row, _, ok = place(b, I, VERTICAL_I)
    assert not ok and lines == 0 and row < 0
    assert np.array_equal(after, b)
    _, _, _, done = step(b, I, VERTICAL_I)
    assert done


def test_place_does_not_mutate_input():
    b = new_board()
    b[-1, :6] = 1
    before = b.copy()
    place(b, I, 6)
    assert np.array_equal(b, before)


def test_other_heights():
    b = new_board(6)
    assert b.shape == (6, WIDTH)
    after, _, row, _, ok = place(b, I, VERTICAL_I)
    assert ok and row == 2
    assert not place(after, I, VERTICAL_I)[4]  # a second vertical I sticks out of the top


def test_piece_draws_are_uniform_and_seeded():
    seed_rng(5)
    a = [random_piece() for _ in range(70_000)]
    seed_rng(5)
    assert a[:100] == [random_piece() for _ in range(100)]
    counts = np.bincount(a, minlength=7)
    assert np.all(np.abs(counts / len(a) - 1 / 7) < 0.01)


def test_random_playout_invariants():
    seed_rng(3)
    rng = np.random.default_rng(3)
    for _ in range(30):
        b, piece, done = new_board(), random_piece(), False
        while not done:
            action = int(rng.choice(np.flatnonzero(legal_actions(piece))))
            nb, lines, nxt, done = step(b, piece, action)
            if done:
                assert np.array_equal(nb, b)
            else:
                assert nb.sum() == b.sum() + 4 - WIDTH * lines
                assert not nb.all(axis=1).any()  # no full rows survive
            assert set(np.unique(nb)) <= {0, 1}
            b, piece = nb, nxt


def test_render():
    out = render(new_board(), T).splitlines()
    assert out[0] == "piece: T"
    assert len(out) == HEIGHT + 2
