import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from game2048.engine import legal_moves, to_onehot
from game2048.env import Game2048Env


def masked_sampling_env():
    """Env whose action_space.sample() only draws legal actions.

    check_env steps with action_space.sample(); the env itself rejects illegal
    actions, so the sampler must mask like any agent would.
    """
    env = Game2048Env()
    space = env.action_space
    unmasked_sample = space.sample

    def sample(mask=None, **kwargs):
        if mask is None and env.board is not None and legal_moves(env.board).any():
            mask = legal_moves(env.board).astype(np.int8)
        return unmasked_sample(mask=mask, **kwargs)

    space.sample = sample
    return env


def test_check_env_passes():
    check_env(masked_sampling_env(), skip_render_check=True)


def test_spaces():
    env = Game2048Env()
    assert env.observation_space.shape == (16, 4, 4)
    assert env.observation_space.dtype == np.float32
    assert env.action_space.n == 4
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert np.array_equal(obs, to_onehot(env.board))


def rollout(seed, n_steps=200):
    """Play from reset(seed) with a fixed action preference order (first legal of 3, 0, 1, 2)."""
    env = Game2048Env()
    obs, info = env.reset(seed=seed)
    trace = [obs]
    for _ in range(n_steps):
        mask = info["action_mask"]
        action = next(a for a in (3, 0, 1, 2) if mask[a])
        obs, reward, terminated, truncated, info = env.step(action)
        trace.append((action, reward, terminated, obs))
        if terminated:
            break
    return trace


def assert_same_trace(a, b):
    assert len(a) == len(b)
    assert np.array_equal(a[0], b[0])
    for (a1, r1, t1, o1), (a2, r2, t2, o2) in zip(a[1:], b[1:]):
        assert (a1, r1, t1) == (a2, r2, t2)
        assert np.array_equal(o1, o2)


def test_same_seed_same_rollout():
    assert_same_trace(rollout(7), rollout(7))


def test_different_seed_different_rollout():
    a, b = rollout(7), rollout(8)
    assert not all(np.array_equal(x[-1], y[-1]) for x, y in zip(a[1:], b[1:]))


def test_reset_seed_independent_of_prior_engine_rng_use():
    a = rollout(3)
    rollout(99)  # consume Numba's RNG in between
    assert_same_trace(a, rollout(3))


def test_action_mask_matches_legal_moves():
    env = Game2048Env()
    _, info = env.reset(seed=1)
    rng = np.random.default_rng(1)
    terminated = False
    while not terminated:
        mask = info["action_mask"]
        assert mask.dtype == np.bool_ and mask.shape == (4,)
        assert np.array_equal(mask, legal_moves(env.board))
        _, _, terminated, truncated, info = env.step(int(rng.choice(np.flatnonzero(mask))))
        assert not truncated
    assert not info["action_mask"].any()


def test_reward_is_merge_reward():
    env = Game2048Env()
    env.reset(seed=0)
    env.board = np.zeros((4, 4), dtype=np.uint8)
    env.board[0, :4] = [1, 1, 2, 2]  # 2 2 4 4 -> left gives 4 8: reward 12
    _, reward, _, _, info = env.step(3)
    assert reward == 12.0
    assert info["score"] == 12


def test_illegal_action_raises():
    env = Game2048Env()
    env.reset(seed=0)
    env.board = np.zeros((4, 4), dtype=np.uint8)
    env.board[0, 0] = 1  # top-left: up and left are illegal
    before = env.board.copy()
    with pytest.raises(ValueError, match="illegal"):
        env.step(0)
    assert np.array_equal(env.board, before)


@pytest.mark.parametrize("action", [-1, 4, 1.5])
def test_out_of_range_action_raises(action):
    env = Game2048Env()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(action)


def test_terminates_exactly_when_no_legal_moves():
    env = Game2048Env()
    _, info = env.reset(seed=4)
    rng = np.random.default_rng(4)
    terminated, steps = False, 0
    while not terminated:
        action = int(rng.choice(np.flatnonzero(info["action_mask"])))
        _, _, terminated, _, info = env.step(action)
        assert terminated == (not legal_moves(env.board).any())
        steps += 1
    assert steps > 10
    with pytest.raises(ValueError):
        env.step(0)


def test_ansi_render():
    env = Game2048Env(render_mode="ansi")
    env.reset(seed=0)
    assert len(env.render().splitlines()) == 4
