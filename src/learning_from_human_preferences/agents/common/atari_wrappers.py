import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from PIL import Image


# =========================================================
# No-op reset
# =========================================================

class NoopResetEnv(gym.Wrapper):
    """
    Sample initial states by taking random number of NOOP actions.
    """

    def __init__(self, env, noop_max=30):

        super().__init__(env)

        self.noop_max = noop_max
        self.override_num_noops = None

        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):

        obs, info = self.env.reset(**kwargs)

        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)

        for _ in range(noops):

            obs, _, terminated, truncated, info = self.env.step(0)

            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)

        return obs, info


# =========================================================
# Fire reset
# =========================================================

class FireResetEnv(gym.Wrapper):

    def __init__(self, env):

        super().__init__(env)

        assert env.unwrapped.get_action_meanings()[1] == "FIRE"

    def reset(self, **kwargs):

        obs, info = self.env.reset(**kwargs)

        obs, _, terminated, truncated, info = self.env.step(1)

        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)

        obs, _, terminated, truncated, info = self.env.step(2)

        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)

        return obs, info


# =========================================================
# Episodic life
# =========================================================

class EpisodicLifeEnv(gym.Wrapper):

    def __init__(self, env):

        super().__init__(env)

        self.lives = 0
        self.was_real_done = True

    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(action)

        self.was_real_done = terminated

        lives = self.env.unwrapped.ale.lives()

        if lives < self.lives and lives > 0:
            terminated = True

        self.lives = lives

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):

        if self.was_real_done:

            obs, info = self.env.reset(**kwargs)

        else:

            obs, _, _, _, info = self.env.step(0)

        self.lives = self.env.unwrapped.ale.lives()

        return obs, info


# =========================================================
# Frame skipping + max pooling
# =========================================================

class MaxAndSkipEnv(gym.Wrapper):

    def __init__(self, env, skip=4):

        super().__init__(env)

        self._skip = skip
        self._obs_buffer = deque(maxlen=2)

    def step(self, action):

        total_reward = 0.0
        terminated = False
        truncated = False

        for _ in range(self._skip):

            obs, reward, terminated, truncated, info = self.env.step(action)

            self._obs_buffer.append(obs)

            total_reward += reward

            if terminated or truncated:
                break

        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):

        self._obs_buffer.clear()

        obs, info = self.env.reset(**kwargs)

        self._obs_buffer.append(obs)

        return obs, info


# =========================================================
# Reward clipping
# =========================================================

class ClipRewardEnv(gym.RewardWrapper):

    def reward(self, reward):

        return np.sign(reward)


# =========================================================
# Warp frame (grayscale + resize)
# =========================================================

class WarpFrame(gym.ObservationWrapper):

    def __init__(self, env):

        super().__init__(env)

        self.res = 84

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.res, self.res, 1),
            dtype=np.uint8,
        )

    def observation(self, obs):

        frame = np.dot(
            obs.astype("float32"),
            np.array([0.299, 0.587, 0.114], dtype="float32"),
        )

        frame = np.array(
            Image.fromarray(frame).resize(
                (self.res, self.res),
                resample=Image.BILINEAR,
            ),
            dtype=np.uint8,
        )

        return frame.reshape(self.res, self.res, 1)


# =========================================================
# Frame stack
# =========================================================

class FrameStack(gym.Wrapper):

    def __init__(self, env, k):

        super().__init__(env)

        self.k = k

        self.frames = deque([], maxlen=k)

        shp = env.observation_space.shape

        assert shp[2] == 1

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(shp[0], shp[1], k),
            dtype=np.uint8,
        )

    def reset(self, **kwargs):

        obs, info = self.env.reset(**kwargs)

        for _ in range(self.k):
            self.frames.append(obs)

        return self._get_obs(), info

    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(action)

        self.frames.append(obs)

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):

        return np.concatenate(self.frames, axis=2)


# =========================================================
# DeepMind Atari wrapper
# =========================================================

def wrap_deepmind(env, episode_life=True, clip_rewards=True):

    assert "NoFrameskip" in env.spec.id

    if episode_life:
        env = EpisodicLifeEnv(env)

    env = NoopResetEnv(env, noop_max=30)

    env = MaxAndSkipEnv(env, skip=4)

    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)

    env = WarpFrame(env)

    if clip_rewards:
        env = ClipRewardEnv(env)

    return env