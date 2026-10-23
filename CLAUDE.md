# 2048 ML project
- Python 3.11+, managed with uv. NumPy + Numba for the engine, PyTorch for DQN.
- Board: (4,4) uint8 array of tile exponents (0=empty, 1=2, 2=4, ...).
- Actions: 0=up, 1=right, 2=down, 3=left.
- src/game2048/engine.py is verified and tested. Do NOT change its behavior
  without asking me first. Add new tests if you extend it.
- Reward = sum of merged tile values. Illegal moves (board unchanged) must be
  masked, never sampled.
- Every agent exposes `act(board) -> action` (greedy, no exploration) and is
  evaluated by src/game2048/evaluate.py (100+ games; report mean score,
  max-tile distribution as % reaching 512/1024/2048/4096, and games/sec).
- Training scripts must be seedable (--seed), log metrics to CSV in runs/,
  and save checkpoints regularly.
- Run pytest after every change. Commit after each milestone with a clear message.
- Add gymnasium, torch and matplotlib to dependencies if missing.
