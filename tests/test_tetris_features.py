import numpy as np
import pytest

from tetris.engine import HEIGHT, WIDTH, legal_actions, new_board, place
from tetris.features import (
    DELLACHERIE_WEIGHTS, N_FEATURES, LinearAgent, best_action, board_features,
    placement_features, play_sequence,
)

I, O, T = 0, 1, 2
VERTICAL_I = WIDTH


def feats(board):
    out = np.zeros(N_FEATURES)
    board_features(board, out)
    return dict(zip(("row_t", "col_t", "holes", "wells", "hole_depth", "rows_with_holes"), out[2:]))


def test_empty_board():
    f = feats(new_board())
    # every empty row: wall|empty and empty|wall; every column: bottom cell against the floor
    assert f == {"row_t": 2 * HEIGHT, "col_t": WIDTH, "holes": 0, "wells": 0,
                 "hole_depth": 0, "rows_with_holes": 0}


def test_row_transitions_count_walls():
    b = new_board()
    b[-1, 5] = 1  # bottom row: wall 1 | 0 0 0 0 0 1 0 0 0 0 | 1 wall -> 4 changes
    assert feats(b)["row_t"] == 2 * (HEIGHT - 1) + 4


def test_holes_and_hole_depth():
    b = new_board()
    b[HEIGHT - 4:HEIGHT - 2, 0] = 1  # two filled cells over two empty ones
    f = feats(b)
    assert f["holes"] == 2 and f["hole_depth"] == 4 and f["rows_with_holes"] == 2


def test_column_transitions():
    b = new_board()
    b[HEIGHT - 2, 0] = 1  # column 0 top to bottom: ... 0 1 0 | floor
    assert feats(b)["col_t"] == WIDTH + 2


def test_cumulative_wells():
    b = new_board()
    b[HEIGHT - 3:, :WIDTH - 1] = 1  # a 3-deep well against the right wall
    assert feats(b)["wells"] == 1 + 2 + 3


def test_landing_height_and_eroded_cells():
    out = np.zeros(N_FEATURES)
    after, _, row, eroded, _ = place(new_board(), I, 0)
    placement_features(after, I, 0, row, eroded, out)
    assert out[0] == 1.0 and out[1] == 0
    after, _, row, eroded, _ = place(new_board(), I, VERTICAL_I)
    placement_features(after, I, 1, row, eroded, out)
    assert out[0] == 2.5  # occupies heights 1..4


def test_heuristic_takes_the_tetris():
    b = new_board()
    b[-4:, :WIDTH - 1] = 1
    a, after, lines, ok = best_action(b, I, DELLACHERIE_WEIGHTS)
    assert ok and a == VERTICAL_I + WIDTH - 1 and lines == 4
    assert after.sum() == 0


def test_best_action_avoids_top_out_when_possible():
    b = new_board(6)
    b[2:, :WIDTH - 4] = 1  # only the right side has room
    for p in range(7):
        a, _, _, ok = best_action(b, p, -DELLACHERIE_WEIGHTS)  # even with inverted weights
        assert ok and legal_actions(p)[a] and place(b, p, a)[4]


def test_best_action_reports_forced_top_out():
    b = new_board(4)
    b[1:, :] = 1
    b[1:, 0] = 0  # one-wide shaft: no full rows, but nothing fits except a vertical I
    a, after, lines, ok = best_action(b, O, DELLACHERIE_WEIGHTS)
    assert not ok and legal_actions(O)[a] and np.array_equal(after, b)


def test_play_sequence_is_deterministic_and_capped():
    pieces = np.random.default_rng(0).integers(0, 7, size=500)
    a = play_sequence(DELLACHERIE_WEIGHTS, pieces, HEIGHT)
    assert a == play_sequence(DELLACHERIE_WEIGHTS, pieces, HEIGHT)
    lines, placed, topped = a
    assert placed == 500 and not topped and lines > 150


def test_linear_agent_validates_weights():
    with pytest.raises(ValueError):
        LinearAgent(np.ones(3))
    assert LinearAgent().act(new_board(), T) in np.flatnonzero(legal_actions(T))
