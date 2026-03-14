import queue
import random
import socket
import time
from multiprocessing import Process

import gymnasium as gym
import numpy as np
import pyglet
from scipy.ndimage import zoom

from agents.common.atari_wrappers import wrap_deepmind


# ==========================================================
# Running statistics (Welford algorithm)
# ==========================================================

class RunningStat:

    def __init__(self, shape=()):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):

        x = np.asarray(x)
        assert x.shape == self._M.shape

        self._n += 1

        if self._n == 1:
            self._M[...] = x
            return

        oldM = self._M.copy()

        self._M[...] = oldM + (x - oldM) / self._n
        self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):

        if self._n >= 2:
            return self._S / (self._n - 1)

        return np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


# ==========================================================
# Simple image viewer
# ==========================================================

class Im:

    def __init__(self, display=None):

        self.window = None
        self.display = display
        self.isopen = False

    def imshow(self, arr):

        if self.window is None:

            height, width = arr.shape

            self.window = pyglet.window.Window(
                width=width,
                height=height,
                display=self.display
            )

            self.width = width
            self.height = height
            self.isopen = True

        assert arr.shape == (self.height, self.width)

        image = pyglet.image.ImageData(
            self.width,
            self.height,
            "L",
            arr.tobytes(),
            pitch=-self.width
        )

        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        image.blit(0, 0)

        self.window.flip()

    def close(self):

        if self.isopen:
            self.window.close()
            self.isopen = False

    def __del__(self):
        self.close()


# ==========================================================
# Video renderer
# ==========================================================

class VideoRenderer:

    play_through_mode = 0
    restart_on_get_mode = 1

    def __init__(self, vid_queue, mode, zoom_factor=1, playback_speed=1):

        assert mode in (
            VideoRenderer.play_through_mode,
            VideoRenderer.restart_on_get_mode
        )

        self.mode = mode
        self.vid_queue = vid_queue
        self.zoom_factor = zoom_factor
        self.playback_speed = playback_speed

        self.proc = Process(target=self.render)
        self.proc.start()

    def stop(self):
        self.proc.terminate()

    def render(self):

        viewer = Im()

        frames = self.vid_queue.get(block=True)

        t = 0

        while True:

            width = frames[t].shape[1]

            fraction_played = t / len(frames)
            x = int(fraction_played * width)

            frames[t][-1][x] = 128

            zoomed_frame = zoom(frames[t], self.zoom_factor)

            viewer.imshow(zoomed_frame)

            if self.mode == VideoRenderer.play_through_mode:

                t += self.playback_speed

                if t >= len(frames):

                    frames = self.get_queue_most_recent()
                    t = 0

                else:

                    time.sleep(1 / 60)

            else:

                try:
                    frames = self.vid_queue.get(block=False)
                    t = 0

                except queue.Empty:

                    t = (t + self.playback_speed) % len(frames)

                    time.sleep(1 / 60)

    def get_queue_most_recent(self):

        item = self.vid_queue.get(block=True)

        while True:
            try:
                item = self.vid_queue.get(block=True, timeout=0.1)
            except queue.Empty:
                break

        return item


# ==========================================================
# Port utilities
# ==========================================================

def get_port_range(start_port, n_ports, random_stagger=False):

    if random_stagger:
        start_port += random.randint(0, 20) * n_ports

    while True:

        ports = []

        for i in range(n_ports):

            port = start_port + i

            try:

                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(("127.0.0.1", port))

                ports.append(port)

            except socket.error as e:

                if e.errno in (98, 48):
                    print(f"Warning: port {port} already in use")
                    break

                raise

            finally:
                s.close()

        if len(ports) == n_ports:
            return ports

        start_port = port + 1


# ==========================================================
# Memory profiling
# ==========================================================

def profile_memory(log_path, pid):

    import memory_profiler

    def profile():

        with open(log_path, "w") as f:

            memory_profiler.memory_usage(
                pid,
                stream=f,
                timeout=99999,
                interval=1
            )

    p = Process(target=profile, daemon=True)
    p.start()

    return p


# ==========================================================
# Batch iterator
# ==========================================================

def batch_iter(data, batch_size, shuffle=False):

    idxs = list(range(len(data)))

    if shuffle:
        np.random.shuffle(idxs)

    start = 0

    while start < len(data):

        end = min(start + batch_size, len(data))

        batch_idxs = idxs[start:end]

        yield [data[i] for i in batch_idxs]

        start += batch_size


# ==========================================================
# Environment creation
# ==========================================================

def make_env(env_id, seed=0):

    moving_dot_envs = {
        "MovingDot-v0",
        "MovingDotNoFrameskip-v0",
        "MovingDotDiscreteNoFrameskip-v0",
        "MovingDotDiscrete-v0",
        "MovingDotContinuous-v0",
        "MovingDotContinuousNoFrameskip-v0",
    }

    if env_id in moving_dot_envs:

        import gym_moving_dot

        env = gym.make(env_id)

        env.reset(seed=seed)

        return env

    env = gym.make(env_id)

    env.reset(seed=seed)

    if env_id == "EnduroNoFrameskip-v4":

        from enduro_wrapper import EnduroWrapper

        env = EnduroWrapper(env)

    return wrap_deepmind(env)