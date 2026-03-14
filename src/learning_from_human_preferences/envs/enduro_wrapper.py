"""
Environment wrapper for Enduro.

- Removes the speedometer region from observations
- Terminates episode when weather begins changing (~3000 steps)
"""

import gymnasium as gym


class EnduroWrapper(gym.Wrapper):

    def __init__(self, env):

        super().__init__(env)

        assert "EnduroNoFrameskip" in env.spec.id

        self._steps = None

    def reset(self, **kwargs):

        observation, info = self.env.reset(**kwargs)

        self._steps = 0

        return observation, info

    def step(self, action):

        observation, reward, terminated, truncated, info = self.env.step(action)

        # Blank out speedometer
        observation[160:] = 0

        self._steps += 1

        # Stop when weather starts changing
        if self._steps == 3000:
            terminated = True

        return observation, reward, terminated, truncated, info