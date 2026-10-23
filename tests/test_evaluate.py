import numpy as np
import pytest

from game2048.agents import RandomAgent
from game2048.engine import legal_moves
from game2048.evaluate import evaluate, main


class CheckingAgent:
    """Wraps an agent and records whether every chosen action was legal."""

    def __init__(self, inner):
        self.inner = inner
        self.n_actions = 0
        self.n_illegal = 0

    def seed(self, seed):
        self.inner.seed(seed)

    def act(self, board):
        action = self.inner.act(board)
        self.n_actions += 1
        if not legal_moves(board)[action]:
            self.n_illegal += 1
        return action


class AlwaysUpAgent:
    def act(self, board):
        return 0


def test_random_agent_only_picks_legal_moves():
    agent = CheckingAgent(RandomAgent())
    evaluate(agent, n_games=20, seed=3)
    assert agent.n_actions > 0
    assert agent.n_illegal == 0


def test_random_agent_covers_all_legal_moves():
    board = np.zeros((4, 4), dtype=np.uint8)
    board[1, 1] = 1  # all four moves legal
    agent = RandomAgent(seed=0)
    assert {agent.act(board) for _ in range(200)} == {0, 1, 2, 3}


def test_evaluate_is_reproducible():
    a = evaluate(RandomAgent(), n_games=30, seed=42)
    b = evaluate(RandomAgent(), n_games=30, seed=42)
    assert np.array_equal(a.scores, b.scores)
    assert np.array_equal(a.max_tiles, b.max_tiles)
    assert np.array_equal(a.moves, b.moves)


def test_different_seeds_differ():
    a = evaluate(RandomAgent(), n_games=30, seed=1)
    b = evaluate(RandomAgent(), n_games=30, seed=2)
    assert not np.array_equal(a.scores, b.scores)


def test_result_statistics_consistent():
    r = evaluate(RandomAgent(), n_games=25, seed=0)
    assert r.n_games == 25
    assert sum(r.max_tile_counts.values()) == 25
    assert r.mean_score == pytest.approx(r.scores.mean())
    assert r.median_score == pytest.approx(np.median(r.scores))
    assert r.games_per_sec > 0
    assert r.reach_pct(2) == 100.0
    assert r.reach_pct(512) >= r.reach_pct(1024) >= r.reach_pct(2048)


def test_illegal_action_raises():
    # Up eventually becomes illegal; evaluate must refuse rather than silently no-op.
    with pytest.raises(ValueError, match="illegal"):
        evaluate(AlwaysUpAgent(), n_games=1, seed=0)


def test_cli_prints_summary(capsys):
    main(["--agent", "random", "--games", "5", "--seed", "0"])
    out = capsys.readouterr().out
    for label in ("Mean score", "Median score", "Games/sec", "Max tile", "Reached", "2048"):
        assert label in out
