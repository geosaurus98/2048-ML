import csv
import importlib.util
from pathlib import Path

import numpy as np

from game2048.dqn import DQNConfig, train
from game2048.ntuple import new_weights, save_checkpoint

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare.py"


def load_compare():
    spec = importlib.util.spec_from_file_location("compare", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compare_writes_outputs_for_all_agents(tmp_path):
    tiny = DQNConfig(buffer_size=500, learning_starts=32, batch_size=16, conv_channels=8, hidden=16)
    train(seed=0, episodes=1, run_dir=tmp_path, device="cpu", config=tiny, log=lambda *_: None)
    save_checkpoint(tmp_path / "ntuple_0", new_weights(), {"games": 0})
    load_compare().main(["--games", "3", "--run-dir", str(tmp_path)])

    with open(tmp_path / "compare.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["agent"] for r in rows] == ["Random", "Expectimax d2", "DQN", "N-tuple TD(0)"]
    assert all(int(r["games"]) == 3 for r in rows)
    with open(tmp_path / "compare_games.csv", newline="") as f:
        games = list(csv.DictReader(f))
    assert len(games) == 12
    assert (tmp_path / "compare_max_tile.png").stat().st_size > 0


def test_seeded_games_are_paired():
    compare = load_compare()
    from game2048.agents import RandomAgent

    a = compare.evaluate_seeded(RandomAgent(), 5, seed=10)
    b = compare.evaluate_seeded(RandomAgent(), 5, seed=10)
    assert np.array_equal(a.scores, b.scores)
    # Game i depends only on seed + i, not on how long earlier games ran.
    c = compare.evaluate_seeded(RandomAgent(), 3, seed=12)
    assert np.array_equal(a.scores[2:], c.scores)
