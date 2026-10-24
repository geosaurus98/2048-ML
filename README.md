# 2048-ML

Agents that learn to play 2048 and Tetris, with fast Numba engines, Gymnasium
environments, seeded evaluation and side-by-side replay viewers.

## Setup

Needs [uv](https://docs.astral.sh/uv/) and git; uv fetches Python 3.11+ itself.

```sh
git clone https://github.com/geosaurus98/2048-ML.git
cd 2048-ML
uv sync --extra dev
uv run pytest
```

On Windows with the repo on a network drive, keep the virtualenv on a local disk
(`$env:UV_PROJECT_ENVIRONMENT = "$env:USERPROFILE\.venvs\2048-ML"`): Windows will
not load NumPy's and PyTorch's DLLs from a UNC path.

## 2048

Agents: random, expectimax (no training), Double DQN, and an n-tuple network
trained with afterstate TD(0).

```sh
uv run python scripts/train_ntuple.py --seed 0 --max-minutes 10
uv run python scripts/train_dqn.py --seed 0 --max-minutes 5
uv run python -m game2048.evaluate --agent ntuple --checkpoint runs/ntuple_0/weights.npy
uv run python scripts/compare.py --games 200      # runs/compare.csv, compare_max_tile.png
uv run python scripts/record_games.py --games 10  # runs/game_viewer.html
```

## Tetris

Placement Tetris as in the RL literature: 10 columns, 7 pieces drawn uniformly
at random, and each action places the current piece (rotation and column) with
a hard drop. Score is lines cleared.

Agents:

- **random**
- **heuristic**: Dellacherie's hand-tuned weights on 6 board features
- **CEM**: the noisy cross-entropy method learns weights for 8 features
- **DQN**: an MLP learns the value of the board after each placement

```sh
uv run python scripts/tetris_train_cem.py --seed 0 --max-minutes 5
uv run python scripts/tetris_train_dqn.py --seed 0 --max-minutes 10
uv run python -m tetris.evaluate --agent cem --checkpoint runs/tetris_cem_0 --height 10
uv run python scripts/tetris_compare.py --games 100      # runs/tetris_compare.csv, tetris_compare_lines.png
uv run python scripts/tetris_record_games.py --games 10  # runs/tetris_viewer.html
```

Good Tetris agents almost never lose on the standard 10x20 board, so games are
capped (`--max-pieces`). The comparison therefore uses two boards. On 10x10,
games end and the agents can be ranked. On 10x20, the result is mostly who
survives the cap. CEM trains on 10x10 for the same reason. Add `--resume auto`
to either training script to continue from the last checkpoint.
