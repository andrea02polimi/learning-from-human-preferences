"""
PyTorch refactor of A2C used in
Deep RL from Human Preferences.

rollout → segment generation → reward predictor → A2C update
"""

import os
import os.path as osp
import queue

import numpy as np
import torch
import torch.nn.functional as F

from agents.common import set_global_seeds, explained_variance
from agents.a2c.utils import discount_with_dones
from learning_from_human_preferences.preferences.pref_db import Segment


# =========================================================
# Model
# =========================================================

class Model:

    def __init__(
        self,
        policy,
        ob_space,
        ac_space,
        nstack,
        lr_scheduler,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        alpha=0.99,
        epsilon=1e-5,
    ):

        self.policy = policy(ob_space, ac_space, nstack)

        self.lr_scheduler = lr_scheduler
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.RMSprop(
            self.policy.parameters(),
            lr=lr_scheduler.value(),
            alpha=alpha,
            eps=epsilon,
        )

        self.initial_state = None

    # -----------------------------------------------------

    def step(self, obs, states, dones):

        obs = torch.tensor(obs).float()

        logits, values = self.policy(obs)

        probs = torch.softmax(logits, dim=1)
        actions = torch.multinomial(probs, 1).squeeze(1)

        return (
            actions.detach().numpy(),
            values.detach().numpy(),
            states,
        )

    # -----------------------------------------------------

    def value(self, obs, states, dones):

        obs = torch.tensor(obs).float()
        _, values = self.policy(obs)

        return values.detach().numpy()

    # -----------------------------------------------------

    def train(self, obs, states, rewards, masks, actions, values):

        obs = torch.tensor(obs).float()
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards).float()
        values = torch.tensor(values).float()

        advantages = rewards - values

        logits, value_pred = self.policy(obs)

        neglogpac = F.cross_entropy(logits, actions, reduction="none")
        policy_loss = torch.mean(advantages * neglogpac)

        value_loss = F.mse_loss(value_pred.squeeze(), rewards)

        probs = torch.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()

        loss = policy_loss - self.ent_coef * entropy + self.vf_coef * value_loss

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(),
            self.max_grad_norm,
        )

        self.optimizer.step()

        # learning rate schedule
        for _ in range(len(obs)):
            current_lr = self.lr_scheduler.value()

        for pg in self.optimizer.param_groups:
            pg["lr"] = current_lr

        return (
            policy_loss.item(),
            value_loss.item(),
            entropy.item(),
            current_lr,
        )

    # -----------------------------------------------------

    def save(self, path, step):

        save_path = f"{path}_{step}.pt"

        torch.save(self.policy.state_dict(), save_path)

        print("Saved policy checkpoint to", save_path)


# =========================================================
# Runner
# =========================================================

class Runner:

    def __init__(
        self,
        env,
        model,
        nsteps,
        nstack,
        gamma,
        gen_segments,
        seg_pipe,
        reward_predictor,
        episode_vid_queue,
    ):

        self.env = env
        self.model = model

        nh, nw, nc = env.observation_space.shape
        nenv = env.num_envs

        self.obs = np.zeros((nenv, nh, nw, nc * nstack), dtype=np.uint8)

        obs = env.reset()
        self.update_obs(obs)

        self.gamma = gamma
        self.nsteps = nsteps

        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

        self.gen_segments = gen_segments
        self.seg_pipe = seg_pipe
        self.reward_predictor = reward_predictor

        self.segment = Segment()

        self.episode_vid_queue = episode_vid_queue

    # -----------------------------------------------------

    def update_obs(self, obs):

        self.obs = np.roll(self.obs, shift=-1, axis=3)
        self.obs[:, :, :, -1] = obs[:, :, :, 0]

    # -----------------------------------------------------

    def update_segment_buffer(self, mb_obs, mb_rewards, mb_dones):

        e0_obs = mb_obs[0]
        e0_rew = mb_rewards[0]
        e0_dones = mb_dones[0]

        for step in range(self.nsteps):

            self.segment.append(
                np.copy(e0_obs[step]),
                np.copy(e0_rew[step]),
            )

            if len(self.segment) == 25 or e0_dones[step]:

                while len(self.segment) < 25:
                    self.segment.append(e0_obs[step], 0)

                self.segment.finalize()

                try:
                    self.seg_pipe.put(self.segment, block=False)
                except queue.Full:
                    pass

                self.segment = Segment()

    # -----------------------------------------------------

    def run(self):

        nenvs = self.env.num_envs

        mb_obs, mb_rewards = [], []
        mb_actions, mb_values, mb_dones = [], [], []

        mb_states = self.states

        for _ in range(self.nsteps):

            actions, values, states = self.model.step(
                self.obs,
                self.states,
                self.dones,
            )

            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)

            obs, rewards, dones, _ = self.env.step(actions)

            self.states = states
            self.dones = dones

            for n, done in enumerate(dones):
                if done:
                    self.obs[n] *= 0

            self.update_obs(obs)
            mb_rewards.append(rewards)

        mb_dones.append(self.dones)

        mb_obs = np.asarray(mb_obs).swapaxes(1, 0)
        mb_rewards = np.asarray(mb_rewards).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions).swapaxes(1, 0)
        mb_values = np.asarray(mb_values).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones).swapaxes(1, 0)

        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]

        if self.gen_segments:
            self.update_segment_buffer(mb_obs, mb_rewards, mb_dones)

        if self.reward_predictor:

            mb_obs_all = mb_obs.reshape(
                nenvs * self.nsteps,
                84,
                84,
                4,
            )

            rewards_all = self.reward_predictor.reward(mb_obs_all)

            mb_rewards = rewards_all.reshape(nenvs, self.nsteps)

        last_values = self.model.value(
            self.obs,
            self.states,
            self.dones,
        ).tolist()

        for n, (rewards, dones, value) in enumerate(
            zip(mb_rewards, mb_dones, last_values)
        ):

            rewards = rewards.tolist()
            dones = dones.tolist()

            if dones[-1] == 0:

                rewards = discount_with_dones(
                    rewards + [value],
                    dones + [0],
                    self.gamma,
                )[:-1]

            else:

                rewards = discount_with_dones(
                    rewards,
                    dones,
                    self.gamma,
                )

            mb_rewards[n] = rewards

        mb_obs = mb_obs.reshape((-1,) + mb_obs.shape[2:])

        return (
            mb_obs,
            mb_states,
            mb_rewards.flatten(),
            mb_masks.flatten(),
            mb_actions.flatten(),
            mb_values.flatten(),
        )


# =========================================================
# Training loop
# =========================================================

def learn(
    policy,
    env,
    seg_pipe,
    start_policy_training_pipe,
    episode_vid_queue,
    reward_predictor,
    ckpt_save_dir,
    gen_segments,
    seed=0,
    total_timesteps=int(80e6),
    nsteps=5,
    nstack=4,
    gamma=0.99,
    lr_scheduler=None,
):

    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space

    model = Model(
        policy,
        ob_space,
        ac_space,
        nstack,
        lr_scheduler,
    )

    # ----------------------------------------------------
    # Phase 1: collect segments
    # ----------------------------------------------------

    runner = Runner(
        env,
        model,
        nsteps,
        nstack,
        gamma,
        gen_segments,
        seg_pipe,
        None,
        episode_vid_queue,
    )

    print("Collecting initial segments")

    while True:

        runner.run()

        if start_policy_training_pipe.get():
            break

    # ----------------------------------------------------
    # Phase 2: RL with learned reward
    # ----------------------------------------------------

    runner = Runner(
        env,
        model,
        nsteps,
        nstack,
        gamma,
        gen_segments,
        seg_pipe,
        reward_predictor,
        episode_vid_queue,
    )

    nbatch = nenvs * nsteps
    nupdates = total_timesteps // nbatch

    for update in range(1, nupdates + 1):

        obs, states, rewards, masks, actions, values = runner.run()

        policy_loss, value_loss, entropy, lr = model.train(
            obs,
            states,
            rewards,
            masks,
            actions,
            values,
        )

        if update % 100 == 0:

            ev = explained_variance(values, rewards)

            print(
                f"update {update} | "
                f"policy_loss {policy_loss:.3f} | "
                f"value_loss {value_loss:.3f} | "
                f"entropy {entropy:.3f}"
            )

        if update % 1000 == 0:

            model.save(
                osp.join(ckpt_save_dir, "policy"),
                update,
            )

    model.save(
        osp.join(ckpt_save_dir, "policy"),
        update,
    )