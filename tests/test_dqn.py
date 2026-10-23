import csv
import math

import numpy as np
import pytest
import torch

from game2048.dqn import (
    DQNAgent, DQNConfig, DQNTrainer, QNet, ReplayBuffer,
    double_dqn_targets, masked_argmax, scale_reward, train,
)
from game2048.engine import legal_moves
from game2048.evaluate import evaluate, main as evaluate_main

from test_evaluate import CheckingAgent

TINY = DQNConfig(buffer_size=2_000, learning_starts=64, batch_size=32, train_every=2,
                 target_update=10, eps_decay_steps=500, conv_channels=16, hidden=32)


class FixedQ(torch.nn.Module):
    """Returns the same Q-values for every board."""

    def __init__(self, q):
        super().__init__()
        self.q = torch.tensor(q, dtype=torch.float32)

    def forward(self, boards):
        return self.q.expand(boards.shape[0], 4).clone()


def corner_board():
    b = np.zeros((4, 4), dtype=np.uint8)
    b[0, 0] = 1  # only RIGHT (1) and DOWN (2) are legal
    return b


def test_qnet_shapes():
    net = QNet(conv_channels=8, hidden=16)
    out = net(torch.zeros((5, 4, 4), dtype=torch.uint8))
    assert out.shape == (5, 4)


def test_masked_argmax_ignores_illegal():
    q = torch.tensor([[10.0, 1.0, 2.0, 10.0]])
    mask = torch.tensor([[False, True, True, False]])
    assert masked_argmax(q, mask).item() == 2


def test_greedy_never_picks_illegal_even_if_preferred():
    trainer = DQNTrainer(TINY, device="cpu")
    trainer.online = FixedQ([100.0, 0.0, 1.0, 100.0])  # prefers illegal UP/LEFT
    b = corner_board()
    assert trainer.greedy_action(b, legal_moves(b)) == 2


def test_exploration_only_picks_legal():
    trainer = DQNTrainer(TINY, device="cpu", seed=0)
    assert trainer.epsilon == 1.0
    b = corner_board()
    picks = {trainer.select_action(b, legal_moves(b)) for _ in range(200)}
    assert picks == {1, 2}


def test_epsilon_schedule():
    trainer = DQNTrainer(TINY, device="cpu")
    trainer.env_steps = TINY.eps_decay_steps // 2
    assert trainer.epsilon == pytest.approx((TINY.eps_start + TINY.eps_end) / 2)
    trainer.env_steps = 10 * TINY.eps_decay_steps
    assert trainer.epsilon == pytest.approx(TINY.eps_end)


def test_reward_scaling():
    assert scale_reward(0) == 0.0
    assert scale_reward(4) == pytest.approx(math.log2(5))


def test_bootstrap_ignores_illegal_next_actions():
    online = FixedQ([100.0, 1.0, 5.0, 100.0])
    target = FixedQ([-50.0, 3.0, 7.0, -50.0])
    nb = torch.zeros((1, 4, 4), dtype=torch.uint8)
    mask = torch.tensor([[False, True, True, False]])
    y = double_dqn_targets(online, target, torch.tensor([2.0]), nb, torch.tensor([False]), mask, 0.5)
    # online picks a*=2 among legal {1, 2}; target evaluates it: 2 + 0.5 * 7
    assert y.item() == pytest.approx(5.5)


def test_bootstrap_zero_on_terminal():
    online = FixedQ([1.0, 2.0, 3.0, 4.0])
    target = FixedQ([9.0, 9.0, 9.0, 9.0])
    nb = torch.zeros((2, 4, 4), dtype=torch.uint8)
    masks = torch.tensor([[False] * 4, [True] * 4])
    dones = torch.tensor([True, True])
    y = double_dqn_targets(online, target, torch.tensor([1.5, 2.5]), nb, dones, masks, 0.99)
    assert torch.equal(y, torch.tensor([1.5, 2.5]))
    assert torch.isfinite(y).all()


def test_replay_buffer_wraparound_and_sample():
    buf = ReplayBuffer(5)
    for i in range(8):
        b = np.full((4, 4), i, dtype=np.uint8)
        buf.add(b, i % 4, float(i), b, i == 7, np.array([True, False, True, False]))
    assert len(buf) == 5
    assert sorted(buf.rewards.tolist()) == [3.0, 4.0, 5.0, 6.0, 7.0]
    b, a, r, nb, d, nm = buf.sample(16, np.random.default_rng(0))
    assert b.shape == (16, 4, 4) and b.dtype == np.uint8
    assert a.dtype == np.int64 and r.dtype == np.float32
    assert d.dtype == np.bool_ and nm.shape == (16, 4)


def test_replay_state_roundtrip():
    buf = ReplayBuffer(10)
    for i in range(4):
        b = np.full((4, 4), i, dtype=np.uint8)
        buf.add(b, 1, float(i), b, False, np.ones(4, dtype=bool))
    other = ReplayBuffer(10)
    other.load_state_dict(buf.state_dict())
    assert len(other) == 4 and other.pos == buf.pos
    assert np.array_equal(other.boards[:4], buf.boards[:4])


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_seeded_training_is_reproducible(tmp_path):
    t1 = train(seed=3, episodes=4, run_dir=tmp_path / "a", device="cpu", config=TINY, log=lambda *_: None)
    t2 = train(seed=3, episodes=4, run_dir=tmp_path / "b", device="cpu", config=TINY, log=lambda *_: None)
    rows1, rows2 = read_csv(tmp_path / "a/dqn_3.csv"), read_csv(tmp_path / "b/dqn_3.csv")
    assert t1.grad_steps > 0
    strip = lambda rows: [{k: v for k, v in r.items() if k != "elapsed_s"} for r in rows]  # noqa: E731
    assert strip(rows1) == strip(rows2)
    for p1, p2 in zip(t1.online.parameters(), t2.online.parameters()):
        assert torch.equal(p1, p2)


def test_csv_columns_and_checkpoints(tmp_path):
    train(seed=0, episodes=4, run_dir=tmp_path, checkpoint_every=2, device="cpu", config=TINY,
          log=lambda *_: None)
    rows = read_csv(tmp_path / "dqn_0.csv")
    assert [int(r["episode"]) for r in rows] == [1, 2, 3, 4]
    assert set(rows[0]) >= {"episode", "score", "max_tile", "mean_loss", "epsilon"}
    ckpts = sorted(p.name for p in (tmp_path / "dqn_0").iterdir())
    assert ckpts == ["ckpt_ep2.pt", "ckpt_ep4.pt", "latest.pt"]


def test_fresh_run_refuses_to_clobber_csv(tmp_path):
    train(seed=0, episodes=1, run_dir=tmp_path, device="cpu", config=TINY, log=lambda *_: None)
    with pytest.raises(FileExistsError):
        train(seed=0, episodes=1, run_dir=tmp_path, device="cpu", config=TINY, log=lambda *_: None)


def test_checkpoint_roundtrip_and_resume(tmp_path):
    t = train(seed=1, episodes=3, run_dir=tmp_path, device="cpu", config=TINY, log=lambda *_: None)
    loaded = DQNTrainer.load(tmp_path / "dqn_1/latest.pt", device="cpu")
    assert (loaded.episode, loaded.env_steps, loaded.grad_steps) == (t.episode, t.env_steps, t.grad_steps)
    assert len(loaded.buffer) == len(t.buffer)
    for p1, p2 in zip(t.online.parameters(), loaded.online.parameters()):
        assert torch.equal(p1, p2)

    resumed = train(seed=1, episodes=5, run_dir=tmp_path, resume="auto", device="cpu",
                    log=lambda *_: None)
    assert resumed.episode == 5 and resumed.env_steps > t.env_steps
    rows = read_csv(tmp_path / "dqn_1.csv")
    assert [int(r["episode"]) for r in rows] == [1, 2, 3, 4, 5]


def test_dqn_agent_plays_legal_moves(tmp_path):
    train(seed=0, episodes=2, run_dir=tmp_path, device="cpu", config=TINY, log=lambda *_: None)
    agent = CheckingAgent(DQNAgent(tmp_path / "dqn_0/latest.pt", device="cpu"))
    agent.seed = lambda seed: None
    evaluate(agent, n_games=2, seed=0)
    assert agent.n_actions > 0 and agent.n_illegal == 0


def test_evaluate_cli_with_checkpoint(tmp_path, capsys):
    train(seed=0, episodes=1, run_dir=tmp_path, device="cpu", config=TINY, log=lambda *_: None)
    evaluate_main(["--agent", "dqn", "--checkpoint", str(tmp_path / "dqn_0/latest.pt"), "--games", "2"])
    assert "Mean score" in capsys.readouterr().out
