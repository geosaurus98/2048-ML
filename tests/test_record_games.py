import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from game2048.agents import RandomAgent
from game2048.evaluate import evaluate

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_record():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("record_games", SCRIPTS / "record_games.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_record_replays_exactly_and_matches_evaluate():
    rec_mod = load_record()
    rec = rec_mod.record_game(RandomAgent(), 4)
    assert len(rec["actions"]) == len(rec["spawns"]) > 0
    assert rec_mod.replay(rec) == (rec["final"], rec["score"])
    # Same game as evaluate() with the same seed.
    r = evaluate(RandomAgent(), n_games=1, seed=4)
    assert int(r.scores[0]) == rec["score"]
    assert len(rec["actions"]) == int(r.moves[0])


def test_viewer_template_has_data_placeholder():
    rec_mod = load_record()
    html = rec_mod.TEMPLATE.read_text(encoding="utf-8")
    assert html.count(rec_mod.PLACEHOLDER) == 1
    page = html.replace(rec_mod.PLACEHOLDER, json.dumps({"seeds": [0], "agents": []}))
    assert "const DATA = {" in page
    assert np.all([c in rec_mod.SPAWN_ALPHABET for c in "AZaf"])
