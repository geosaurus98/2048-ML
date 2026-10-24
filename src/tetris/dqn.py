"""Afterstate DQN for placement Tetris.

Placement Tetris has up to 34 legal actions whose outcomes are deterministic
until the next piece is drawn, so instead of Q(s, a) the network learns the
value V of the afterstate (the board after the drop and line clears) and acts by
  argmax over legal placements of  r + gamma * V(afterstate),
where a placement that tops out is worth 0. V sees 22 features of the board
(Bertsekas & Tsitsiklis 1996, plus rows with holes): the 10 column heights, the
9 height differences between neighbours, the max height, holes and rows with
holes, normalised by the board size.

Training reward r = 1 + lines cleared for every placement that does not top
out, so the agent is paid for surviving and for clearing lines. Each stored
transition is (afterstate x, then the greedy next placement's reward r', its
afterstate x' and whether it tops out); the target is
  y = r' + gamma * V_target(x'), or 0 if that placement tops out.
The next placement is chosen by the online network when it is played
(Double DQN, with the choice made at collection time). Training episodes are
truncated at `max_pieces`; truncation does not cut the bootstrap.

The training loop lives in `train()`; scripts/tetris_train_dqn.py is its CLI.
"""

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numba import njit

from tetris.engine import HEIGHT, N_ROTATIONS, PIECE_W, WIDTH, place
from tetris.env import TetrisEnv

N_INPUTS = 2 * WIDTH + 2


@dataclass
class DQNConfig:
    # gamma 0.99 / lr 1e-3 learned faster but swung between ~20 and ~1,000 lines from
    # one checkpoint to the next; this pair improved steadily in a 4-minute sweep.
    gamma: float = 0.95
    lr: float = 2.5e-4
    batch_size: int = 512
    buffer_size: int = 100_000
    learning_starts: int = 2_000
    train_every: int = 1
    target_update: int = 500  # gradient steps between hard target copies
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_steps: int = 30_000
    grad_clip: float = 10.0
    hidden: int = 64
    height: int = HEIGHT  # training board height
    max_pieces: int = 5_000  # training episodes are truncated here


def pick_device(device="auto"):
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_reward(lines):
    """Survival bonus plus lines cleared, for placements that do not top out."""
    return 1.0 + lines


@njit(cache=True)
def value_inputs(board, out):
    """Fill `out` (length N_INPUTS) with the network inputs for an afterstate."""
    H, W = board.shape
    heights = np.zeros(W)
    holes = 0
    row_has_hole = np.zeros(H, dtype=np.bool_)
    for c in range(W):
        top = H
        for r in range(H):
            if board[r, c]:
                if top == H:
                    top = r
            elif top < H:
                holes += 1
                row_has_hole[r] = True
        heights[c] = H - top
    for c in range(W):
        out[c] = heights[c] / H
    for c in range(W - 1):
        out[W + c] = abs(heights[c] - heights[c + 1]) / H
    out[2 * W - 1] = heights.max() / H
    out[2 * W] = holes / W
    out[2 * W + 1] = row_has_hole.sum() / H


@njit(cache=True)
def candidates(board, piece):
    """Every legal placement of `piece`: (actions, inputs, lines, ok).

    inputs[i] are the afterstate features; rows for placements that top out
    (ok[i] False) are left at zero.
    """
    n = 0
    for rot in range(N_ROTATIONS[piece]):
        n += WIDTH - PIECE_W[piece, rot] + 1
    actions = np.zeros(n, dtype=np.int64)
    inputs = np.zeros((n, N_INPUTS), dtype=np.float32)
    lines = np.zeros(n, dtype=np.int64)
    ok = np.zeros(n, dtype=np.bool_)
    i = 0
    for rot in range(N_ROTATIONS[piece]):
        for col in range(WIDTH - PIECE_W[piece, rot] + 1):
            a = rot * WIDTH + col
            after, n_lines, _, _, good = place(board, piece, a)
            actions[i], lines[i], ok[i] = a, n_lines, good
            if good:
                value_inputs(after, inputs[i])
            i += 1
    return actions, inputs, lines, ok


class VNet(nn.Module):
    """MLP from afterstate features to a scalar value."""

    def __init__(self, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(N_INPUTS, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x).squeeze(-1)


def placement_values(net, inputs, lines, ok, gamma, device):
    """r + gamma * V(afterstate) for each candidate; 0 for placements that top out."""
    with torch.no_grad():
        v = net(torch.as_tensor(inputs, device=device)).cpu().numpy()
    return np.where(ok, train_reward(lines) + gamma * v, 0.0)


class ReplayBuffer:
    """Uniform circular buffer of afterstate transitions."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.x = np.zeros((capacity, N_INPUTS), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_x = np.zeros((capacity, N_INPUTS), dtype=np.float32)
        self.terminal = np.zeros(capacity, dtype=np.bool_)
        self.pos = 0
        self.size = 0

    def __len__(self):
        return self.size

    def add(self, x, reward, next_x, terminal):
        i = self.pos
        self.x[i] = x
        self.rewards[i] = reward
        self.next_x[i] = next_x
        self.terminal[i] = terminal
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        idx = rng.integers(0, self.size, size=batch_size)
        return self.x[idx], self.rewards[idx], self.next_x[idx], self.terminal[idx]

    _FIELDS = ("x", "rewards", "next_x", "terminal")

    def state_dict(self):
        n = self.size
        state = {k: getattr(self, k)[:n].copy() for k in self._FIELDS}
        state.update(capacity=self.capacity, pos=self.pos, size=self.size)
        return state

    def load_state_dict(self, state):
        if state["capacity"] != self.capacity:
            raise ValueError("replay capacity mismatch")
        n = state["size"]
        for k in self._FIELDS:
            getattr(self, k)[:n] = state[k]
        self.pos, self.size = state["pos"], n


class DQNTrainer:
    def __init__(self, config=None, device="auto", seed=0):
        self.config = config or DQNConfig()
        c = self.config
        self.device = pick_device(device)
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.online = VNet(c.hidden).to(self.device)
        self.target = VNet(c.hidden).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=c.lr)
        self.buffer = ReplayBuffer(c.buffer_size)
        self.env_steps = 0
        self.grad_steps = 0
        self.episode = 0

    @property
    def epsilon(self):
        c = self.config
        frac = min(1.0, self.env_steps / c.eps_decay_steps)
        return c.eps_start + frac * (c.eps_end - c.eps_start)

    def values(self, inputs, lines, ok):
        return placement_values(self.online, inputs, lines, ok, self.config.gamma, self.device)

    def select(self, q):
        """Epsilon-greedy candidate index: explore uniformly over all legal placements."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(q)))
        return int(np.argmax(q))

    def observe(self, x, reward, next_x, terminal):
        """Store a transition and train when due. Returns (loss, mean V) or None."""
        self.buffer.add(x, reward, next_x, terminal)
        c = self.config
        if len(self.buffer) >= c.learning_starts and self.env_steps % c.train_every == 0:
            return self.train_step()
        return None

    def train_step(self):
        c = self.config
        dev = self.device
        x, r, nx, term = (torch.as_tensor(a, device=dev) for a in self.buffer.sample(c.batch_size, self.rng))
        with torch.no_grad():
            y = torch.where(term, torch.zeros_like(r), r + c.gamma * self.target(nx))
        v = self.online(x)
        loss = F.smooth_l1_loss(v, y)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), c.grad_clip)
        self.optimizer.step()
        self.grad_steps += 1
        if self.grad_steps % c.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach()), float(v.detach().mean())

    def state_dict(self):
        return {
            "config": asdict(self.config),
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "buffer": self.buffer.state_dict(),
            "env_steps": self.env_steps,
            "grad_steps": self.grad_steps,
            "episode": self.episode,
            "np_rng": self.rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
        }

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save(self.state_dict(), tmp)
        tmp.replace(path)

    @classmethod
    def load(cls, path, device="auto"):
        state = torch.load(path, map_location="cpu", weights_only=False)
        trainer = cls(DQNConfig(**state["config"]), device=device)
        trainer.online.load_state_dict(state["online"])
        trainer.target.load_state_dict(state["target"])
        trainer.optimizer.load_state_dict(state["optimizer"])
        trainer.buffer.load_state_dict(state["buffer"])
        trainer.env_steps = state["env_steps"]
        trainer.grad_steps = state["grad_steps"]
        trainer.episode = state["episode"]
        trainer.rng.bit_generator.state = state["np_rng"]
        torch.set_rng_state(state["torch_rng"])
        return trainer


class DQNAgent:
    """Greedy policy from a checkpoint: argmax of r + gamma * V(afterstate). For evaluate.py."""

    def __init__(self, checkpoint, device="auto"):
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.config = DQNConfig(**state["config"])
        self.device = pick_device(device)
        self.net = VNet(self.config.hidden).to(self.device)
        self.net.load_state_dict(state["online"])
        self.net.eval()

    def act(self, board, piece):
        actions, inputs, lines, ok = candidates(board, piece)
        q = placement_values(self.net, inputs, lines, ok, self.config.gamma, self.device)
        return int(actions[np.argmax(q)])


CSV_FIELDS = ["episode", "env_steps", "grad_steps", "lines", "pieces", "topped_out",
              "mean_loss", "epsilon", "mean_v", "elapsed_s"]


def train(seed=0, episodes=None, max_minutes=None, run_dir="runs", checkpoint_every=100,
          resume=None, overwrite=False, device="auto", config=None, log=print):
    """Train until `episodes` total episodes or `max_minutes` of wall time.

    Writes runs/tetris_dqn_<seed>.csv (one row per episode) and checkpoints to
    runs/tetris_dqn_<seed>/ckpt_ep<N>.pt every `checkpoint_every` episodes plus
    latest.pt on exit. `resume` is a checkpoint path or "auto" (latest.pt).
    Returns the trainer.
    """
    if episodes is None and max_minutes is None:
        raise ValueError("set episodes and/or max_minutes")
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / f"tetris_dqn_{seed}"
    csv_path = run_dir / f"tetris_dqn_{seed}.csv"
    latest = ckpt_dir / "latest.pt"

    if resume == "auto":
        resume = latest if latest.exists() else None
    if resume is not None:
        trainer = DQNTrainer.load(resume, device=device)
        env_seed = seed + trainer.episode  # Numba's RNG state can't be saved; reseed deterministically
        log(f"resumed from {resume} at episode {trainer.episode}, env step {trainer.env_steps}")
    else:
        if csv_path.exists() and not overwrite:
            raise FileExistsError(f"{csv_path} exists; pass resume or overwrite")
        trainer = DQNTrainer(config, device=device, seed=seed)
        env_seed = seed
    c = trainer.config
    log(f"device: {trainer.device}")

    run_dir.mkdir(parents=True, exist_ok=True)
    new_csv = resume is None
    csv_file = open(csv_path, "w" if new_csv else "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if new_csv:
        writer.writeheader()

    env = TetrisEnv(height=c.height, max_pieces=c.max_pieces)
    start = time.perf_counter()
    first = True
    try:
        while episodes is None or trainer.episode < episodes:
            if max_minutes is not None and time.perf_counter() - start >= 60 * max_minutes:
                break
            _, info = env.reset(seed=env_seed if first else None)
            first = False
            losses, vs = [], []
            pending = None  # afterstate features of the previous placement
            terminated = truncated = False
            while not (terminated or truncated):
                actions, inputs, lines, ok = candidates(env.board, env.piece)
                q = trainer.values(inputs, lines, ok)
                if pending is not None:
                    g = int(np.argmax(q))
                    reward = train_reward(lines[g]) if ok[g] else 0.0
                    out = trainer.observe(pending, reward, inputs[g], not ok[g])
                    if out is not None:
                        losses.append(out[0])
                        vs.append(out[1])
                i = trainer.select(q)
                _, _, terminated, truncated, info = env.step(int(actions[i]))
                trainer.env_steps += 1
                pending = inputs[i] if ok[i] else None
            trainer.episode += 1
            writer.writerow({
                "episode": trainer.episode,
                "env_steps": trainer.env_steps,
                "grad_steps": trainer.grad_steps,
                "lines": info["lines"],
                "pieces": info["pieces"],
                "topped_out": int(terminated),
                "mean_loss": f"{np.mean(losses):.5f}" if losses else "",
                "epsilon": f"{trainer.epsilon:.4f}",
                "mean_v": f"{np.mean(vs):.4f}" if vs else "",
                "elapsed_s": f"{time.perf_counter() - start:.1f}",
            })
            csv_file.flush()
            if trainer.episode % checkpoint_every == 0:
                trainer.save(ckpt_dir / f"ckpt_ep{trainer.episode}.pt")
                trainer.save(latest)
                log(f"episode {trainer.episode:>6d}  lines {info['lines']:>6d}  pieces {info['pieces']:>6d}  "
                    f"eps {trainer.epsilon:.3f}  steps {trainer.env_steps}")
    finally:
        trainer.save(latest)
        csv_file.close()
    log(f"stopped at episode {trainer.episode}, env step {trainer.env_steps}, "
        f"grad step {trainer.grad_steps}; checkpoint {latest}")
    return trainer
