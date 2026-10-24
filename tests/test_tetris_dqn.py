import csv

import numpy as np
import pytest
import torch

from tetris.agents import AGENTS
from tetris.dqn import (
    N_INPUTS, DQNConfig, DQNTrainer, ReplayBuffer, VNet, candidates, placement_values, train,
    train_reward, value_inputs,
)
from tetris.engine import HEIGHT, WIDTH, legal_actions, new_board, place
from tetris.evaluate import evaluate

from test_tetris_evaluate import CheckingAgent

QUIET = {"log": lambda *_: None}
TINY = DQNConfig(buffer_size=500, learning_starts=8, batch_size=16, hidden=8, height=8, max_pieces=50)


def inputs_of(board):
    out = np.zeros(N_INPUTS, dtype=np.float32)
    value_inputs(board, out)
    return out


def test_value_inputs():
    assert not inputs_of(new_board()).any()
    b = new_board()
    b[HEIGHT - 3, 0] = 1  # column 0: height 3 with two holes beneath
    b[HEIGHT - 1, 1] = 1  # column 1: height 1
    x = inputs_of(b) * np.r_[[HEIGHT] * (2 * WIDTH), WIDTH, HEIGHT]
    assert np.allclose(x[:WIDTH], [3, 1] + [0] * 8)
    assert np.allclose(x[WIDTH:2 * WIDTH - 1], [2, 1] + [0] * 7)
    assert np.allclose(x[2 * WIDTH - 1:], [3, 2, 2])  # max height, holes, rows with holes


@pytest.mark.parametrize("piece", range(7))
def test_candidates_cover_every_legal_placement(piece):
    b = new_board(6)
    b[2:, :6] = 1
    actions, inputs, lines, ok = candidates(b, piece)
    assert sorted(actions.tolist()) == np.flatnonzero(legal_actions(piece)).tolist()
    for a, x, n, good in zip(actions, inputs, lines, ok):
        after, n2, _, _, good2 = place(b, piece, a)
        assert (n, good) == (n2, good2)
        assert np.array_equal(x, inputs_of(after) if good else np.zeros(N_INPUTS, np.float32))
    assert ok.any()
    if piece == 0:
        assert not ok.all()  # a vertical I on the ledge sticks out of the top


def test_topped_out_placements_are_worth_zero():
    net = VNet(8)
    with torch.no_grad():
        net.out.bias.fill_(1000.0)
    inputs = np.zeros((3, N_INPUTS), dtype=np.float32)
    q = placement_values(net, inputs, np.array([0, 2, 0]), np.array([True, True, False]), 0.5, "cpu")
    assert q[2] == 0.0
    assert q[1] - q[0] == pytest.approx(train_reward(2) - train_reward(0))


def test_replay_buffer_round_trip():
    buf = ReplayBuffer(4)
    for i in range(6):
        buf.add(np.full(N_INPUTS, i, np.float32), float(i), np.full(N_INPUTS, -i, np.float32), i % 2 == 0)
    assert len(buf) == 4
    copy = ReplayBuffer(4)
    copy.load_state_dict(buf.state_dict())
    for k in ReplayBuffer._FIELDS:
        assert np.array_equal(getattr(copy, k), getattr(buf, k))
    x, r, nx, term = buf.sample(8, np.random.default_rng(0))
    assert np.array_equal(x[:, 0], r) and np.array_equal(nx[:, 0], -r)
    assert np.array_equal(term, r % 2 == 0)


def test_train_step_moves_v_toward_target():
    trainer = DQNTrainer(DQNConfig(buffer_size=64, batch_size=32, learning_starts=1, lr=1e-2, hidden=16),
                         device="cpu")
    x = np.full(N_INPUTS, 0.5, np.float32)
    for _ in range(64):
        trainer.buffer.add(x, 5.0, x, True)  # terminal next placement: target is 0
    for _ in range(300):
        trainer.train_step()
    with torch.no_grad():
        assert abs(float(trainer.online(torch.as_tensor(x[None])))) < 0.2


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_train_writes_csv_and_checkpoints_and_resumes(tmp_path):
    trainer = train(seed=2, episodes=4, run_dir=tmp_path, checkpoint_every=2, device="cpu",
                    config=TINY, **QUIET)
    assert trainer.episode == 4 and trainer.grad_steps > 0
    ckpt_dir = tmp_path / "tetris_dqn_2"
    assert {p.name for p in ckpt_dir.iterdir()} >= {"latest.pt", "ckpt_ep2.pt", "ckpt_ep4.pt"}
    resumed = train(seed=2, episodes=6, run_dir=tmp_path, resume="auto", device="cpu", **QUIET)
    assert resumed.episode == 6 and resumed.env_steps > trainer.env_steps
    rows = read_rows(tmp_path / "tetris_dqn_2.csv")
    assert [int(r["episode"]) for r in rows] == list(range(1, 7))
    assert all(int(r["pieces"]) <= TINY.max_pieces for r in rows)


def test_seeded_training_is_reproducible(tmp_path):
    a = train(seed=4, episodes=3, run_dir=tmp_path / "a", device="cpu", config=TINY, **QUIET)
    b = train(seed=4, episodes=3, run_dir=tmp_path / "b", device="cpu", config=TINY, **QUIET)
    assert a.env_steps == b.env_steps
    for k, v in a.online.state_dict().items():
        assert torch.equal(v, b.online.state_dict()[k])


def test_fresh_run_refuses_to_clobber_csv(tmp_path):
    train(seed=0, episodes=1, run_dir=tmp_path, device="cpu", config=TINY, **QUIET)
    with pytest.raises(FileExistsError):
        train(seed=0, episodes=1, run_dir=tmp_path, device="cpu", config=TINY, **QUIET)


def test_agent_plays_legal_actions(tmp_path):
    train(seed=0, episodes=2, run_dir=tmp_path, device="cpu", config=TINY, **QUIET)
    agent = CheckingAgent(AGENTS["dqn"](checkpoint=str(tmp_path / "tetris_dqn_0/latest.pt"), device="cpu"))
    evaluate(agent, n_games=2, seed=0, max_pieces=100)
    assert agent.n_actions > 0 and agent.n_illegal == 0
    with pytest.raises(ValueError):
        AGENTS["dqn"]()
