import numpy as np


class DummyVecEnv(object):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.remotes = [None] * self.num_envs

        env = self.envs[0]
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        # aggiungi questo
        self.env_id = env.spec.id

    def reset(self):
        obs = []
        for env in self.envs:
            result = env.reset()
            if isinstance(result, tuple):  # Gymnasium
                o, _ = result
            else:  # Gym legacy
                o = result
            obs.append(o)
        return np.stack(obs)

    def step(self, actions):
        results = [env.step(a) for env, a in zip(self.envs, actions)]

        obs, rewards, dones, infos = [], [], [], []

        for r in results:
            if len(r) == 5:  # Gymnasium
                o, rew, terminated, truncated, info = r
                done = terminated or truncated
            else:  # Gym legacy
                o, rew, done, info = r

            obs.append(o)
            rewards.append(rew)
            dones.append(done)
            infos.append(info)

        return (
            np.stack(obs),
            np.array(rewards),
            np.array(dones),
            infos,
        )

    def close(self):
        for env in self.envs:
            env.close()