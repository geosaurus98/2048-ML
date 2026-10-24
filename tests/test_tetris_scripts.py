import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from tetris.agents import RandomAgent
from tetris.cem import CEMConfig
from tetris.cem import train as train_cem
from tetris.dqn import DQNConfig
from tetris.dqn import train as train_dqn
from tetris.evaluate import evaluate
from tetris.features import LinearAgent

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
QUIET = {"log": lambda *_: None}


def load(name):
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_checkpoints(run_dir):
    train_cem(seed=0, iterations=1, run_dir=run_dir,
              config=CEMConfig(population=4, games_per_eval=1, height=6, max_pieces=50), **QUIET)
    train_dqn(seed=0, episodes=1, run_dir=run_dir, device="cpu",
              config=DQNConfig(buffer_size=100, learning_starts=8, batch_size=8, hidden=8, max_pieces=30), **QUIET)


def test_compare_writes_outputs_for_all_agents_and_boards(tmp_path):
    make_checkpoints(tmp_path)
    load("tetris_compare").main(["--games", "2", "--run-dir", str(tmp_path)])
    with open(tmp_path / "tetris_compare.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["agent"], r["board"]) for r in rows] == [
        (a, b) for a in ("Random", "Heuristic", "CEM", "DQN") for b in ("small", "standard")]
    assert all(int(r["games"]) == 2 for r in rows)
    with open(tmp_path / "tetris_compare_games.csv", newline="") as f:
        assert len(list(csv.DictReader(f))) == 16
    assert (tmp_path / "tetris_compare_lines.png").stat().st_size > 0


def test_seeded_games_are_paired():
    compare = load("tetris_compare")
    a = compare.evaluate_seeded(RandomAgent(), 5, seed=10, max_pieces=1000, height=20)
    b = compare.evaluate_seeded(RandomAgent(), 3, seed=12, max_pieces=1000, height=20)
    assert np.array_equal(a.pieces[2:], b.pieces)  # game i depends only on seed + i


def test_record_replays_exactly_and_matches_evaluate():
    rec_mod = load("tetris_record_games")
    for agent, height in ((RandomAgent(), 20), (LinearAgent(), 6)):
        rec = rec_mod.record_game(agent, 4, max_pieces=300, height=height)
        assert len(rec["pieces"]) == len(rec["actions"]) > 0
        assert rec_mod.replay(rec, height) == (rec["final"], rec["lines"], rec["topped_out"])
        r = evaluate(agent, n_games=1, seed=4, max_pieces=300, height=height)  # same game as evaluate()
        assert rec["lines"] == int(r.lines[0]) and rec["topped_out"] == bool(r.topped_out[0])
        assert len(rec["pieces"]) == int(r.pieces[0]) + rec["topped_out"]


def test_record_cli_builds_viewer(tmp_path):
    make_checkpoints(tmp_path)
    rec_mod = load("tetris_record_games")
    rec_mod.main(["--games", "2", "--max-pieces", "40", "--run-dir", str(tmp_path)])
    data = json.loads((tmp_path / "tetris_replays.json").read_text())
    assert [a["name"] for a in data["agents"]] == ["Random", "Heuristic", "CEM", "DQN"]
    assert data["height"] == 20 and len(data["shapes"]) == 7 and len(data["shapes"][2]) == 4
    page = (tmp_path / "tetris_viewer.html").read_text(encoding="utf-8")
    assert rec_mod.PLACEHOLDER not in page and "const DATA = {" in page


def test_viewer_template_has_data_placeholder():
    rec_mod = load("tetris_record_games")
    assert rec_mod.TEMPLATE.read_text(encoding="utf-8").count(rec_mod.PLACEHOLDER) == 1
    assert len(rec_mod.ACTION_ALPHABET) >= 40
