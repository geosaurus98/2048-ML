import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from tetris.engine import HEIGHT, N_ACTIONS, legal_actions
from tetris.env import TetrisEnv

VERTICAL_I = 10


def masked_sampling_env(**kwargs):
    """Env whose action_space.sample() only draws legal actions (see test_env.py)."""
    env = TetrisEnv(**kwargs)
    space = env.action_space
    unmasked_sample = space.sample

    def sample(mask=None, **kw):
        if mask is None and env.piece is not None:
            mask = legal_actions(env.piece).astype(np.int8)
        return unmasked_sample(mask=mask, **kw)

    space.sample = sample
    return env


def test_check_env_passes():
    check_env(masked_sampling_env(), skip_render_check=True)


def test_spaces():
    env = TetrisEnv()
    assert env.action_space.n == N_ACTIONS
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert obs["board"].shape == (HEIGHT, 10) and not obs["board"].any()
    assert np.array_equal(info["action_mask"], legal_actions(obs["piece"]))


def rollout(seed, n_steps=300, **kwargs):
    """Play from reset(seed) taking the lowest-numbered legal action each time."""
    env = TetrisEnv(**kwargs)
    obs, info = env.reset(seed=seed)
    trace = [obs["piece"]]
    for _ in range(n_steps):
        action = int(np.flatnonzero(info["action_mask"])[0])
        obs, reward, terminated, truncated, info = env.step(action)
        trace.append((action, reward, terminated, truncated, obs["piece"], obs["board"].tobytes()))
        if terminated or truncated:
            break
    return trace


def test_same_seed_same_rollout():
    assert rollout(7) == rollout(7)


def test_different_seed_different_rollout():
    assert rollout(7) != rollout(8)


def test_reset_seed_independent_of_prior_engine_rng_use():
    a = rollout(3)
    rollout(99)
    assert a == rollout(3)


def test_reward_is_lines_cleared():
    env = TetrisEnv()
    env.reset(seed=0)
    env.piece = 0  # I
    env.board[-4:, :9] = 1
    _, reward, terminated, _, info = env.step(VERTICAL_I + 9)
    assert reward == 4.0 and not terminated
    assert info["lines"] == 4 and info["pieces"] == 1
    assert not env.board.any()


def test_top_out_terminates_and_blocks_further_steps():
    env = TetrisEnv(height=6)
    env.reset(seed=0)
    env.piece = 0
    env.step(VERTICAL_I)
    env.piece = 0
    _, reward, terminated, truncated, info = env.step(VERTICAL_I)
    assert terminated and not truncated and reward == 0.0
    assert info["pieces"] == 1
    with pytest.raises(RuntimeError):
        env.step(0)


def test_truncates_at_max_pieces():
    trace = rollout(0, n_steps=100, max_pieces=5, height=40)
    assert len(trace) == 6
    assert trace[-1][3] and not trace[-1][2]


def test_illegal_action_raises():
    env = TetrisEnv()
    env.reset(seed=0)
    env.piece = 1  # O: one rotation, columns 0-8
    before = env.board.copy()
    for action in (9, 10):
        with pytest.raises(ValueError, match="illegal"):
            env.step(action)
    assert np.array_equal(env.board, before)


@pytest.mark.parametrize("action", [-1, N_ACTIONS, 1.5])
def test_out_of_range_action_raises(action):
    env = TetrisEnv()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(action)


def test_ansi_render():
    env = TetrisEnv(render_mode="ansi")
    env.reset(seed=0)
    assert len(env.render().splitlines()) == HEIGHT + 2
