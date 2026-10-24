import numpy as np
import pytest

from tetris.agents import RandomAgent
from tetris.engine import legal_actions, new_board
from tetris.evaluate import evaluate, main
from tetris.features import LinearAgent


class CheckingAgent:
    """Wraps a Tetris agent and records whether every chosen action was legal."""

    def __init__(self, inner):
        self.inner = inner
        self.n_actions = 0
        self.n_illegal = 0

    def seed(self, seed):
        if hasattr(self.inner, "seed"):
            self.inner.seed(seed)

    def act(self, board, piece):
        action = self.inner.act(board, piece)
        self.n_actions += 1
        if not legal_actions(piece)[action]:
            self.n_illegal += 1
        return action


class AlwaysLastColumnAgent:
    def act(self, board, piece):
        return 9  # rotation 0 at column 9: no piece is one cell wide


def test_random_agent_only_picks_legal_actions():
    agent = CheckingAgent(RandomAgent())
    evaluate(agent, n_games=20, seed=3)
    assert agent.n_actions > 0 and agent.n_illegal == 0


def test_random_agent_covers_all_legal_actions():
    agent = RandomAgent(seed=0)
    chosen = {agent.act(new_board(), 2) for _ in range(2000)}
    assert chosen == set(np.flatnonzero(legal_actions(2)).tolist())


def test_evaluate_is_reproducible():
    a = evaluate(RandomAgent(), n_games=30, seed=42)
    b = evaluate(RandomAgent(), n_games=30, seed=42)
    assert np.array_equal(a.lines, b.lines) and np.array_equal(a.pieces, b.pieces)


def test_different_seeds_differ():
    a = evaluate(RandomAgent(), n_games=30, seed=1)
    b = evaluate(RandomAgent(), n_games=30, seed=2)
    assert not np.array_equal(a.pieces, b.pieces)


def test_piece_cap_counts_as_survival():
    r = evaluate(LinearAgent(), n_games=3, seed=0, max_pieces=200)
    assert np.all(r.pieces == 200) and not r.topped_out.any()
    assert r.survival_pct == 100.0 and r.mean_lines > 50


def test_random_agent_tops_out():
    r = evaluate(RandomAgent(), n_games=25, seed=0)
    assert r.topped_out.all() and r.survival_pct == 0.0
    assert r.reach_pct(0) == 100.0 and r.reach_pct(10) >= r.reach_pct(100)
    assert r.games_per_sec > 0 and r.pieces_per_sec > 0


def test_smaller_board_ends_games_sooner():
    tall = evaluate(RandomAgent(), n_games=30, seed=0)
    short = evaluate(RandomAgent(), n_games=30, seed=0, height=8)
    assert short.pieces.mean() < tall.pieces.mean()


def test_illegal_action_raises():
    with pytest.raises(ValueError, match="illegal"):
        evaluate(AlwaysLastColumnAgent(), n_games=1, seed=0)


def test_cli_prints_summary(capsys):
    main(["--agent", "heuristic", "--games", "2", "--seed", "0", "--max-pieces", "100", "--height", "12"])
    out = capsys.readouterr().out
    for label in ("10x12", "Mean lines", "Median lines", "Survived cap", "Pieces/sec", "reached"):
        assert label in out
