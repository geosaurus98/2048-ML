"""Double DQN for 2048 with invalid-action masking.

Masking is applied in three places so illegal actions are never taken or read:
action selection (explore and exploit), the bootstrap argmax on the next state,
and DQNAgent.act. Rewards are scaled as log2(1 + merge_reward).

The training loop lives in `train()`; scripts/train_dqn.py is its CLI.
"""

import csv
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from game2048.engine import legal_moves, max_tile
from game2048.env import Game2048Env


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 2.5e-4
    batch_size: int = 256
    buffer_size: int = 200_000
    learning_starts: int = 5_000
    train_every: int = 4
    target_update: int = 2_000  # gradient steps between hard target copies
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_steps: int = 300_000
    grad_clip: float = 10.0
    conv_channels: int = 128
    hidden: int = 256


def pick_device(device="auto"):
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def scale_reward(reward):
    return math.log2(1.0 + reward)


class QNet(nn.Module):
    """Small CNN on the 16x4x4 one-hot board. Takes uint8 exponent boards (B, 4, 4)."""

    def __init__(self, conv_channels=128, hidden=256):
        super().__init__()
        self.conv1 = nn.Conv2d(16, conv_channels, kernel_size=2)
        self.conv2 = nn.Conv2d(conv_channels, conv_channels, kernel_size=2)
        self.fc1 = nn.Linear(conv_channels * 2 * 2, hidden)
        self.fc2 = nn.Linear(hidden, 4)

    def forward(self, boards):
        x = F.one_hot(boards.long(), 16).permute(0, 3, 1, 2).float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc1(x.flatten(1)))
        return self.fc2(x)


def masked_argmax(q, mask):
    """Argmax over legal actions only. Rows with no legal action return 0."""
    return q.masked_fill(~mask, -torch.inf).argmax(dim=1)


@torch.no_grad()
def double_dqn_targets(online, target, rewards, next_boards, dones, next_masks, gamma):
    """y = r + gamma * Q_target(s', argmax_{legal a} Q_online(s', a)); no bootstrap at terminal s'."""
    next_actions = masked_argmax(online(next_boards), next_masks)
    next_q = target(next_boards).gather(1, next_actions[:, None]).squeeze(1)
    bootstrap = dones | ~next_masks.any(dim=1)
    next_q = torch.where(bootstrap, torch.zeros_like(next_q), next_q)
    return rewards + gamma * next_q


class ReplayBuffer:
    """Uniform circular buffer of compact uint8 boards."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.boards = np.zeros((capacity, 4, 4), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_boards = np.zeros((capacity, 4, 4), dtype=np.uint8)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.next_masks = np.zeros((capacity, 4), dtype=np.bool_)
        self.pos = 0
        self.size = 0

    def __len__(self):
        return self.size

    def add(self, board, action, reward, next_board, done, next_mask):
        i = self.pos
        self.boards[i] = board
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_boards[i] = next_board
        self.dones[i] = done
        self.next_masks[i] = next_mask
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        idx = rng.integers(0, self.size, size=batch_size)
        return (self.boards[idx], self.actions[idx], self.rewards[idx],
                self.next_boards[idx], self.dones[idx], self.next_masks[idx])

    _FIELDS = ("boards", "actions", "rewards", "next_boards", "dones", "next_masks")

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
        self.online = QNet(c.conv_channels, c.hidden).to(self.device)
        self.target = QNet(c.conv_channels, c.hidden).to(self.device)
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

    @torch.no_grad()
    def greedy_action(self, board, mask):
        q = self.online(torch.as_tensor(board[None], device=self.device))
        return int(masked_argmax(q, torch.as_tensor(mask[None], device=self.device))[0])

    def select_action(self, board, mask):
        """Masked epsilon-greedy: explore uniformly over legal actions only."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(np.flatnonzero(mask)))
        return self.greedy_action(board, mask)

    def observe(self, board, action, reward, next_board, done, next_mask):
        """Store a transition (raw merge reward) and train when due. Returns (loss, mean_q) or None."""
        self.buffer.add(board, action, scale_reward(reward), next_board, done, next_mask)
        self.env_steps += 1
        c = self.config
        if len(self.buffer) >= c.learning_starts and self.env_steps % c.train_every == 0:
            return self.train_step()
        return None

    def train_step(self):
        c = self.config
        dev = self.device
        b, a, r, nb, d, nm = self.buffer.sample(c.batch_size, self.rng)
        b, nb = torch.as_tensor(b, device=dev), torch.as_tensor(nb, device=dev)
        a, r = torch.as_tensor(a, device=dev), torch.as_tensor(r, device=dev)
        d, nm = torch.as_tensor(d, device=dev), torch.as_tensor(nm, device=dev)
        y = double_dqn_targets(self.online, self.target, r, nb, d, nm, c.gamma)
        q = self.online(b).gather(1, a[:, None]).squeeze(1)
        loss = F.smooth_l1_loss(q, y)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), c.grad_clip)
        self.optimizer.step()
        self.grad_steps += 1
        if self.grad_steps % c.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach()), float(q.detach().mean())

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
    """Greedy, masked policy from a checkpoint. For evaluate.py."""

    def __init__(self, checkpoint, device="auto"):
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = DQNConfig(**state["config"])
        self.device = pick_device(device)
        self.net = QNet(config.conv_channels, config.hidden).to(self.device)
        self.net.load_state_dict(state["online"])
        self.net.eval()

    @torch.no_grad()
    def act(self, board):
        mask = legal_moves(board)
        if not mask.any():
            raise ValueError("no legal moves")
        q = self.net(torch.as_tensor(board[None], device=self.device))
        return int(masked_argmax(q, torch.as_tensor(mask[None], device=self.device))[0])


CSV_FIELDS = ["episode", "env_steps", "grad_steps", "score", "max_tile", "moves",
              "mean_loss", "epsilon", "mean_q", "elapsed_s"]


def train(seed=0, episodes=None, max_minutes=None, run_dir="runs", checkpoint_every=100,
          resume=None, overwrite=False, device="auto", config=None, log=print):
    """Train until `episodes` total episodes or `max_minutes` of wall time.

    Writes runs/dqn_<seed>.csv (one row per episode) and checkpoints to
    runs/dqn_<seed>/ckpt_ep<N>.pt every `checkpoint_every` episodes plus
    latest.pt on exit. `resume` is a checkpoint path or "auto" (latest.pt).
    Returns the trainer.
    """
    if episodes is None and max_minutes is None:
        raise ValueError("set episodes and/or max_minutes")
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / f"dqn_{seed}"
    csv_path = run_dir / f"dqn_{seed}.csv"
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
    log(f"device: {trainer.device}")

    run_dir.mkdir(parents=True, exist_ok=True)
    new_csv = resume is None
    csv_file = open(csv_path, "w" if new_csv else "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if new_csv:
        writer.writeheader()

    env = Game2048Env()
    start = time.perf_counter()
    first = True
    try:
        while episodes is None or trainer.episode < episodes:
            if max_minutes is not None and time.perf_counter() - start >= 60 * max_minutes:
                break
            obs, info = env.reset(seed=env_seed if first else None)
            first = False
            board, mask = env.board, info["action_mask"]
            losses, qs, done, moves = [], [], False, 0
            while not done:
                action = trainer.select_action(board, mask)
                _, reward, done, _, info = env.step(action)
                next_board, next_mask = env.board, info["action_mask"]
                out = trainer.observe(board, action, reward, next_board, done, next_mask)
                if out is not None:
                    losses.append(out[0])
                    qs.append(out[1])
                board, mask = next_board, next_mask
                moves += 1
            trainer.episode += 1
            writer.writerow({
                "episode": trainer.episode,
                "env_steps": trainer.env_steps,
                "grad_steps": trainer.grad_steps,
                "score": info["score"],
                "max_tile": max_tile(board),
                "moves": moves,
                "mean_loss": f"{np.mean(losses):.5f}" if losses else "",
                "epsilon": f"{trainer.epsilon:.4f}",
                "mean_q": f"{np.mean(qs):.4f}" if qs else "",
                "elapsed_s": f"{time.perf_counter() - start:.1f}",
            })
            csv_file.flush()
            if trainer.episode % checkpoint_every == 0:
                trainer.save(ckpt_dir / f"ckpt_ep{trainer.episode}.pt")
                trainer.save(latest)
    finally:
        trainer.save(latest)
        csv_file.close()
    log(f"stopped at episode {trainer.episode}, env step {trainer.env_steps}, "
        f"grad step {trainer.grad_steps}; checkpoint {latest}")
    return trainer
