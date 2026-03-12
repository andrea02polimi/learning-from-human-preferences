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
        obs = [env.reset() for env in self.envs]
        return np.stack(obs)

    def step(self, actions):
        obs, rewards, dones, infos = [], [], [], []
        for env, action in zip(self.envs, actions):
            ob, r, done, info = env.step(action)
            if done:
                ob = env.reset()
            obs.append(ob)
            rewards.append(r)
            dones.append(done)
            infos.append(info)

        return (
            np.stack(obs),
            np.array(rewards),
            np.array(dones),
            infos
        )

    def close(self):
        for env in self.envs:
            env.close()