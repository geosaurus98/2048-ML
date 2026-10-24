"""Record seeded Tetris games from each agent and build the replay viewer page.

Game i is seeded seed + i for every agent, exactly as in scripts/tetris_compare.py,
so every agent is dealt the same pieces. Each placement is stored as its piece
and action; the viewer replays them with the piece shapes embedded from
tetris.engine and checks every game against its recorded final board on load.

  uv run python scripts/tetris_record_games.py --games 10
writes runs/tetris_replays.json and runs/tetris_viewer.html. Use --height 10 to
record on the small board, where good agents also top out eventually, and
--out-dir to keep several versions, e.g.
  uv run python scripts/tetris_record_games.py --max-pieces 100000 --out-dir runs/compare_100k
"""

import argparse
import csv
import json
import string
import sys
from pathlib import Path

from tetris.engine import (
    CELLS, HEIGHT, N_ROTATIONS, PIECE_NAMES, is_legal, new_board, place, random_piece, seed_rng, step,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tetris_compare import build_agents  # noqa: E402

TEMPLATE = Path(__file__).resolve().parent / "tetris_viewer_template.html"
PLACEHOLDER = "/*__REPLAY_DATA__*/null"
ACTION_ALPHABET = string.digits + string.ascii_uppercase + string.ascii_lowercase  # one char per action (< 62)
MAX_PIECES = 5_000


def record_game(agent, game_seed, max_pieces=MAX_PIECES, height=HEIGHT):
    """Play one game seeded like evaluate(); return the compact record.

    If the game tops out, the final (failed) placement is included in the record.
    """
    seed_rng(game_seed)
    if hasattr(agent, "seed"):
        agent.seed(game_seed)
    board = new_board(height)
    piece = random_piece()
    pieces, actions = [], []
    lines = 0
    topped = False
    for _ in range(max_pieces):
        a = agent.act(board, piece)
        if not is_legal(piece, a):
            raise ValueError(f"agent chose illegal action {a} for piece {piece}")
        pieces.append(str(piece))
        actions.append(ACTION_ALPHABET[a])
        board, n, piece, done = step(board, piece, a)
        if done:
            topped = True
            break
        lines += int(n)
    return {
        "seed": game_seed,
        "pieces": "".join(pieces),
        "actions": "".join(actions),
        "lines": lines,
        "topped_out": topped,
        "final": board.ravel().tolist(),
    }


def replay(record, height=HEIGHT):
    """Rebuild (final board, lines, topped out) from a record (used to verify it)."""
    board = new_board(height)
    lines = 0
    n_moves = len(record["pieces"])
    for t, (p, a) in enumerate(zip(record["pieces"], record["actions"])):
        board, n, _, _, ok = place(board, int(p), ACTION_ALPHABET.index(a))
        if not ok:
            assert t == n_moves - 1, "top-out before the end of the record"
            return board.ravel().tolist(), lines, True
        lines += int(n)
    return board.ravel().tolist(), lines, False


def compare_summaries(run_dir, height):
    path = Path(run_dir) / "tetris_compare.csv"
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {r["agent"]: {"games": int(r["games"]), "mean": float(r["mean_lines"]),
                             "survival": float(r["survival_pct"]), "maxPieces": int(r["max_pieces"])}
                for r in csv.DictReader(f) if int(r["height"]) == height}


def shapes():
    """Piece cells per rotation, as the viewer's place() expects them."""
    return [[CELLS[p, k].tolist() for k in range(N_ROTATIONS[p])] for p in range(len(PIECE_NAMES))]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=HEIGHT, help="board height (width is always 10)")
    parser.add_argument("--max-pieces", type=int, default=MAX_PIECES, help="cap on placements per game")
    parser.add_argument("--run-dir", default="runs", help="where checkpoints are found")
    parser.add_argument("--out-dir", default=None,
                        help="where the replays and viewer go (default: --run-dir); a tetris_compare.csv "
                             "there is used for the per-agent summaries")
    parser.add_argument("--cem-checkpoint", default=None)
    parser.add_argument("--dqn-checkpoint", default=None)
    args = parser.parse_args(argv)
    run_dir = Path(args.out_dir or args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    summaries = compare_summaries(run_dir, args.height) or compare_summaries(args.run_dir, args.height)
    agents = []
    for name, make, ckpt in build_agents(args):
        print(f"recording {name} ...", flush=True)
        agent = make()
        games = []
        for i in range(args.games):
            rec = record_game(agent, args.seed + i, args.max_pieces, args.height)
            assert replay(rec, args.height) == (rec["final"], rec["lines"], rec["topped_out"]), \
                "record does not replay"
            games.append(rec)
        agents.append({"name": name, "checkpoint": ckpt, "summary": summaries.get(name), "games": games})

    data = {"height": args.height, "width": 10, "maxPieces": args.max_pieces, "pieceNames": PIECE_NAMES,
            "actionAlphabet": ACTION_ALPHABET, "shapes": shapes(),
            "seeds": [args.seed + i for i in range(args.games)], "agents": agents}
    payload = json.dumps(data, separators=(",", ":"))
    (run_dir / "tetris_replays.json").write_text(payload)
    html = TEMPLATE.read_text(encoding="utf-8")
    if PLACEHOLDER not in html:
        raise RuntimeError("viewer template is missing the data placeholder")
    out = run_dir / "tetris_viewer.html"
    out.write_text(html.replace(PLACEHOLDER, payload), encoding="utf-8")
    moves = sum(len(g["pieces"]) for a in agents for g in a["games"])
    print(f"wrote {run_dir / 'tetris_replays.json'} and {out} ({moves:,} placements, "
          f"{out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
