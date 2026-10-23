import numpy as np
import pytest

from game2048.engine import legal_moves, new_game, seed_rng, step
from game2048.evaluate import evaluate
from game2048.expectimax import ExpectimaxAgent, HeuristicWeights, heuristic

from test_evaluate import CheckingAgent


class SeedlessChecker(CheckingAgent):
    def seed(self, seed):
        pass


def test_never_returns_illegal_move_in_full_games():
    agent = SeedlessChecker(ExpectimaxAgent(depth=1))
    evaluate(agent, n_games=3, seed=0)
    assert agent.n_actions > 0
    assert agent.n_illegal == 0


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_never_returns_illegal_move_on_random_positions(depth):
    agent = ExpectimaxAgent(depth=depth)
    seed_rng(5)
    rng = np.random.default_rng(5)
    checked = 0
    for _ in range(5):
        b = new_game()
        done = False
        while not done:
            legal = legal_moves(b)
            if checked % 7 == 0:
                assert legal[agent.act(b)]
            checked += 1
            b, _, _, done = step(b, int(rng.choice(np.flatnonzero(legal))))
    assert checked > 0


def test_single_legal_move_is_chosen():
    # Rows 0-2 form a full checkerboard and row 3 is empty: only DOWN (2) is legal.
    b = np.array([[1, 2, 1, 2], [2, 1, 2, 1], [1, 2, 1, 2], [0, 0, 0, 0]], dtype=np.uint8)
    assert list(legal_moves(b)) == [False, False, True, False]
    for depth in (1, 2, 3):
        assert ExpectimaxAgent(depth=depth).act(b) == 2


def test_no_legal_moves_raises():
    b = np.array([[1, 2, 1, 2], [2, 1, 2, 1], [1, 2, 1, 2], [2, 1, 2, 1]], dtype=np.uint8)
    with pytest.raises(ValueError):
        ExpectimaxAgent(depth=2).act(b)


def test_act_is_deterministic():
    seed_rng(3)
    b = new_game()
    for _ in range(10):
        b, _, _, _ = step(b, int(np.flatnonzero(legal_moves(b))[0]))
    agent = ExpectimaxAgent(depth=2)
    assert len({agent.act(b) for _ in range(5)}) == 1


def test_heuristic_prefers_monotone_and_empty():
    w = HeuristicWeights().to_array()
    mono = np.zeros((4, 4), dtype=np.uint8)
    mono[0] = [4, 3, 2, 1]
    shuffled = np.zeros((4, 4), dtype=np.uint8)
    shuffled[0] = [3, 1, 4, 2]
    assert heuristic(mono, w) > heuristic(shuffled, w)
    fuller = mono.copy()
    fuller[1, 1] = 1
    assert heuristic(mono, w) > heuristic(fuller, w)


def test_heuristic_handles_decreasing_pairs():
    # Regression: uint8 subtraction must not wrap around.
    w = HeuristicWeights(empty=0, monotonicity=0, smoothness=1, corner=0).to_array()
    b = np.zeros((4, 4), dtype=np.uint8)
    b[0, :2] = [5, 1]
    assert heuristic(b, w) == -4.0


def test_weights_are_configurable():
    w = HeuristicWeights(empty=1.0, monotonicity=0.0, smoothness=0.0, corner=0.0)
    b = np.zeros((4, 4), dtype=np.uint8)
    b[0, 0] = 1
    assert heuristic(b, w.to_array()) == 15.0
    assert ExpectimaxAgent(depth=1, weights=w).weights is w


def test_invalid_depth():
    with pytest.raises(ValueError):
        ExpectimaxAgent(depth=0)
