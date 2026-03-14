import os
import numpy as np
import gymnasium as gym
import random
import tempfile
import time
import zipfile
import pickle

from collections import deque


# =========================================================
# zip same length sequences
# =========================================================

def zipsame(*seqs):

    L = len(seqs[0])

    assert all(len(seq) == L for seq in seqs[1:])

    return zip(*seqs)


# =========================================================
# unpack helper
# =========================================================

def unpack(seq, sizes):

    seq = list(seq)

    it = iter(seq)

    assert sum(1 if s is None else s for s in sizes) == len(seq)

    for size in sizes:

        if size is None:

            yield next(it)

        else:

            out = []

            for _ in range(size):

                out.append(next(it))

            yield out


# =========================================================
# EzPickle
# =========================================================

class EzPickle:

    def __init__(self, *args, **kwargs):

        self._ezpickle_args = args

        self._ezpickle_kwargs = kwargs

    def __getstate__(self):

        return {
            "_ezpickle_args": self._ezpickle_args,
            "_ezpickle_kwargs": self._ezpickle_kwargs
        }

    def __setstate__(self, state):

        obj = type(self)(*state["_ezpickle_args"], **state["_ezpickle_kwargs"])

        self.__dict__.update(obj.__dict__)


# =========================================================
# Random seeds
# =========================================================

def set_global_seeds(seed):

    np.random.seed(seed)

    random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# =========================================================
# Pretty ETA
# =========================================================

def pretty_eta(seconds_left):

    minutes = seconds_left // 60
    seconds_left %= 60

    hours = minutes // 60
    minutes %= 60

    days = hours // 24
    hours %= 24

    def fmt(cnt, name):

        return f"{cnt} {name}" + ("s" if cnt > 1 else "")

    if days > 0:

        msg = fmt(days, "day")

        if hours > 0:
            msg += " and " + fmt(hours, "hour")

        return msg

    if hours > 0:

        msg = fmt(hours, "hour")

        if minutes > 0:
            msg += " and " + fmt(minutes, "minute")

        return msg

    if minutes > 0:

        return fmt(minutes, "minute")

    return "less than a minute"


# =========================================================
# Running average
# =========================================================

class RunningAvg:

    def __init__(self, gamma, init_value=None):

        self._value = init_value

        self._gamma = gamma

    def update(self, new_val):

        if self._value is None:

            self._value = new_val

        else:

            self._value = self._gamma * self._value + (1 - self._gamma) * new_val

    def __float__(self):

        return self._value


# =========================================================
# SimpleMonitor
# =========================================================

class SimpleMonitor(gym.Wrapper):

    def __init__(self, env):

        super().__init__(env)

        self._current_reward = None
        self._num_steps = None

        self._time_offset = None
        self._total_steps = None

        self._episode_rewards = []
        self._episode_lengths = []
        self._episode_end_times = []

    def reset(self, **kwargs):

        obs, info = self.env.reset(**kwargs)

        if self._time_offset is None:

            self._time_offset = time.time()

            if self._episode_end_times:

                self._time_offset -= self._episode_end_times[-1]

        if self._total_steps is None:

            self._total_steps = sum(self._episode_lengths)

        if self._current_reward is not None:

            self._episode_rewards.append(self._current_reward)

            self._episode_lengths.append(self._num_steps)

            self._episode_end_times.append(time.time() - self._time_offset)

        self._current_reward = 0

        self._num_steps = 0

        return obs, info

    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(action)

        done = terminated or truncated

        self._current_reward += reward

        self._num_steps += 1

        self._total_steps += 1

        info["steps"] = self._total_steps

        info["rewards"] = self._episode_rewards

        return obs, reward, terminated, truncated, info


# =========================================================
# argparse helper
# =========================================================

def boolean_flag(parser, name, default=False, help=None):

    dest = name.replace("-", "_")

    parser.add_argument(f"--{name}", action="store_true", default=default, dest=dest, help=help)

    parser.add_argument(f"--no-{name}", action="store_false", dest=dest)


# =========================================================
# Wrapper lookup
# =========================================================

def get_wrapper_by_name(env, classname):

    current = env

    while True:

        if current.__class__.__name__ == classname:

            return current

        elif isinstance(current, gym.Wrapper):

            current = current.env

        else:

            raise ValueError(f"Couldn't find wrapper named {classname}")


# =========================================================
# Safe pickle dump
# =========================================================

def relatively_safe_pickle_dump(obj, path, compression=False):

    temp_storage = path + ".relatively_safe"

    if compression:

        with tempfile.NamedTemporaryFile() as tmp:

            pickle.dump(obj, tmp)

            with zipfile.ZipFile(temp_storage, "w", compression=zipfile.ZIP_DEFLATED) as zf:

                zf.write(tmp.name, "data")

    else:

        with open(temp_storage, "wb") as f:

            pickle.dump(obj, f)

    os.rename(temp_storage, path)


# =========================================================
# Pickle load
# =========================================================

def pickle_load(path, compression=False):

    if compression:

        with zipfile.ZipFile(path, "r", compression=zipfile.ZIP_DEFLATED) as zf:

            with zf.open("data") as f:

                return pickle.load(f)

    else:

        with open(path, "rb") as f:

            return pickle.load(f)