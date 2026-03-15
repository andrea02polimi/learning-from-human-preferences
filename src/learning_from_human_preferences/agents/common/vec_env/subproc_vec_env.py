import numpy as np
from multiprocessing import Process, Pipe

from learning_from_human_preferences.agents.common.vec_env import VecEnv


# ==========================================================
# Worker
# ==========================================================

def worker(remote, env_fn_wrapper):

    env = env_fn_wrapper.x()

    while True:

        cmd, data = remote.recv()

        # --------------------------------------------------
        # STEP
        # --------------------------------------------------

        if cmd == "step":

            result = env.step(data)

            # Gymnasium API
            if len(result) == 5:
                ob, reward, terminated, truncated, info = result
                done = terminated or truncated

            # Gym legacy API
            else:
                ob, reward, done, info = result

            if done:

                reset_result = env.reset()

                if isinstance(reset_result, tuple):
                    ob = reset_result[0]
                else:
                    ob = reset_result

            remote.send((ob, reward, done, info))

        # --------------------------------------------------
        # RESET
        # --------------------------------------------------

        elif cmd == "reset":

            reset_result = env.reset()

            if isinstance(reset_result, tuple):
                ob = reset_result[0]
            else:
                ob = reset_result

            remote.send(ob)

        # --------------------------------------------------
        # CLOSE
        # --------------------------------------------------

        elif cmd == "close":

            remote.close()
            break

        # --------------------------------------------------
        # GET SPACES
        # --------------------------------------------------

        elif cmd == "get_spaces":

            remote.send((env.action_space, env.observation_space))

        # --------------------------------------------------
        # GET ACTION MEANINGS
        # --------------------------------------------------

        elif cmd == "get_action_meanings":

            if hasattr(env.unwrapped, "get_action_meanings"):
                remote.send(env.unwrapped.get_action_meanings())
            else:
                remote.send(None)

        else:

            raise NotImplementedError


# ==========================================================
# Cloudpickle wrapper
# ==========================================================

class CloudpickleWrapper:

    def __init__(self, x):
        self.x = x

    def __getstate__(self):

        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):

        import pickle
        self.x = pickle.loads(ob)


# ==========================================================
# SubprocVecEnv
# ==========================================================

class SubprocVecEnv(VecEnv):

    def __init__(self, env_id, env_fns):

        nenvs = len(env_fns)

        self.remotes, self.work_remotes = zip(
            *[Pipe() for _ in range(nenvs)]
        )

        self.ps = [
            Process(
                target=worker,
                args=(work_remote, CloudpickleWrapper(env_fn)),
                daemon=True,
            )
            for work_remote, env_fn in zip(self.work_remotes, env_fns)
        ]

        for p in self.ps:
            p.start()

        # --------------------------------------------------
        # Get spaces
        # --------------------------------------------------

        self.remotes[0].send(("get_spaces", None))
        self.action_space, self.observation_space = self.remotes[0].recv()

        # --------------------------------------------------
        # Action meanings (Atari)
        # --------------------------------------------------

        self.remotes[0].send(("get_action_meanings", None))
        self.action_meanings = self.remotes[0].recv()

        self.env_id = env_id

    # ======================================================
    # STEP
    # ======================================================

    def step(self, actions):

        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))

        results = [remote.recv() for remote in self.remotes]

        obs, rews, dones, infos = zip(*results)

        return (
            np.stack(obs),
            np.stack(rews),
            np.stack(dones),
            infos,
        )

    # ======================================================
    # RESET
    # ======================================================

    def reset(self):

        for remote in self.remotes:
            remote.send(("reset", None))

        obs = [remote.recv() for remote in self.remotes]

        return np.stack(obs)

    # ======================================================
    # CLOSE
    # ======================================================

    def close(self):

        for remote in self.remotes:
            remote.send(("close", None))

        for p in self.ps:
            p.join()

    # ======================================================
    # NUM ENVS
    # ======================================================

    @property
    def num_envs(self):

        return len(self.remotes)