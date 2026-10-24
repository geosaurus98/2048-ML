"""Record seeded games from each agent and build the replay viewer page.

Game i is seeded seed + i for every agent, exactly as in scripts/compare.py.
Each move is stored as its action plus the tile that spawned after it, so the
viewer replays the real games without running the agents in the browser.

  uv run python scripts/record_games.py --games 10
writes runs/replays.json and runs/game_viewer.html.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from game2048.engine import apply_move, legal_moves, new_game, seed_rng, spawn_tile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare import build_agents  # noqa: E402

TEMPLATE = Path(__file__).resolve().parent / "viewer_template.html"
PLACEHOLDER = "/*__REPLAY_DATA__*/null"
# One character per spawn: index = cell * 2 + (exponent - 1), cell in 0..15, exponent 1 or 2.
SPAWN_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef"


def record_game(agent, game_seed):
    """Play one game seeded like evaluate(); return the compact record."""
    seed_rng(game_seed)
    if hasattr(agent, "seed"):
        agent.seed(game_seed)
    board = new_game()
    start = board.ravel().tolist()
    actions, spawns = [], []
    score = 0
    while legal_moves(board).any():
        a = agent.act(board)
        after, reward, moved = apply_move(board, a)
        if not moved:
            raise ValueError(f"agent chose illegal action {a}")
        nxt = after.copy()
        spawn_tile(nxt)  # same RNG draws as engine.step
        cell = int(np.flatnonzero(nxt.ravel() != after.ravel())[0])
        spawns.append(SPAWN_ALPHABET[cell * 2 + int(nxt.flat[cell]) - 1])
        actions.append(str(a))
        score += int(reward)
        board = nxt
    return {
        "seed": game_seed,
        "start": start,
        "actions": "".join(actions),
        "spawns": "".join(spawns),
        "score": score,
        "final": board.ravel().tolist(),
    }


def replay(record):
    """Rebuild the final board and score from a record (used to verify it)."""
    board = np.array(record["start"], dtype=np.uint8).reshape(4, 4)
    score = 0
    for a, s in zip(record["actions"], record["spawns"]):
        board, reward, moved = apply_move(board, int(a))
        assert moved
        score += int(reward)
        idx = SPAWN_ALPHABET.index(s)
        board.flat[idx // 2] = idx % 2 + 1
    return board.ravel().tolist(), score


def compare_means(run_dir):
    path = Path(run_dir) / "compare.csv"
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {r["agent"]: {"games": int(r["games"]), "mean": float(r["mean_score"]),
                             "pct2048": float(r["pct_2048"])} for r in csv.DictReader(f)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-dir", default="runs")
    parser.add_argument("--dqn-checkpoint", default=None)
    parser.add_argument("--ntuple-checkpoint", default=None)
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    means = compare_means(run_dir)
    agents = []
    for name, make, ckpt in build_agents(args):
        print(f"recording {name} ...", flush=True)
        agent = make()
        games = []
        for i in range(args.games):
            rec = record_game(agent, args.seed + i)
            assert replay(rec) == (rec["final"], rec["score"]), "record does not replay"
            games.append(rec)
        agents.append({"name": name, "checkpoint": ckpt, "summary": means.get(name), "games": games})

    data = {"seeds": [args.seed + i for i in range(args.games)], "agents": agents}
    (run_dir / "replays.json").write_text(json.dumps(data, separators=(",", ":")))
    html = TEMPLATE.read_text(encoding="utf-8")
    if PLACEHOLDER not in html:
        raise RuntimeError("viewer template is missing the data placeholder")
    out = run_dir / "game_viewer.html"
    out.write_text(html.replace(PLACEHOLDER, json.dumps(data, separators=(",", ":"))), encoding="utf-8")
    moves = sum(len(g["actions"]) for a in agents for g in a["games"])
    print(f"wrote {run_dir / 'replays.json'} and {out} ({moves:,} moves, {out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
