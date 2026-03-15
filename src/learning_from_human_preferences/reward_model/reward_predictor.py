import logging
import os
import os.path as osp
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from learning_from_human_preferences.envs.utils import RunningStat, batch_iter


class RewardPredictorNetwork(nn.Module):
    """
    Predict rewards for trajectory segments and infer preferences.
    """

    def __init__(self, core_network):
        super().__init__()
        self.core_network = core_network

    def forward(self, s1, s2):

        B, T = s1.shape[0], s1.shape[1]

        s1_unrolled = s1.reshape(-1, 84, 84, 4)
        s2_unrolled = s2.reshape(-1, 84, 84, 4)

        r1 = self.core_network(s1_unrolled)
        r2 = self.core_network(s2_unrolled)

        r1 = r1.reshape(B, T)
        r2 = r2.reshape(B, T)

        rs1 = r1.sum(dim=1)
        rs2 = r2.sum(dim=1)

        logits = torch.stack([rs1, rs2], dim=1)

        pred = torch.softmax(logits, dim=1)

        return r1, r2, rs1, rs2, pred, logits


class RewardPredictorEnsemble:
    """
    Ensemble of reward predictors.

    Args:
        core_network: callable with no arguments that returns a nn.Module
                      (e.g. AtariRewardNetwork or
                       functools.partial(AtariRewardNetwork, dropout_prob=0.3))
        lr:           Adam learning rate
        n_preds:      ensemble size (paper uses 3)
        log_dir:      experiment log directory for TensorBoard & checkpoints;
                      pass None to disable both (inference-only mode)
        device:       torch device string
    """

    def __init__(
        self,
        core_network,
        lr=1e-4,
        n_preds=1,
        log_dir=None,
        device="cpu",
    ):

        self.device = device
        self.n_preds = n_preds

        self.models = []
        self.optimizers = []

        for _ in range(n_preds):
            model = RewardPredictorNetwork(core_network()).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            self.models.append(model)
            self.optimizers.append(optimizer)

        if log_dir is not None:
            self.writer_train = SummaryWriter(
                osp.join(log_dir, "reward_predictor", "train")
            )
            self.writer_test = SummaryWriter(
                osp.join(log_dir, "reward_predictor", "test")
            )
            self.checkpoint_dir = osp.join(log_dir, "reward_predictor_checkpoints")
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        else:
            self.writer_train = None
            self.writer_test = None
            self.checkpoint_dir = None

        self.n_steps = 0

        self.r_norm = RunningStat(shape=n_preds)

    # ----------------------------------------------------------

    def save(self):

        if self.checkpoint_dir is None:
            raise RuntimeError("Cannot save: log_dir was not set")

        path = osp.join(self.checkpoint_dir, f"reward_predictor_{self.n_steps}.pt")

        state = {
            "step": self.n_steps,
            "models": [m.state_dict() for m in self.models],
        }

        torch.save(state, path)

        print(f"Saved reward predictor checkpoint to {path}")

    # ----------------------------------------------------------

    def load(self, path):

        state = torch.load(path, weights_only=True)

        for m, s in zip(self.models, state["models"]):
            m.load_state_dict(s)

        self.n_steps = state["step"]

        print(f"Loaded reward predictor checkpoint from {path}")

    # ----------------------------------------------------------

    @staticmethod
    def latest_checkpoint(ckpt_dir):
        """Return path of the latest checkpoint in ckpt_dir, or None."""
        if not osp.exists(ckpt_dir):
            return None
        files = [
            f for f in os.listdir(ckpt_dir)
            if f.startswith("reward_predictor_") and f.endswith(".pt")
        ]
        if not files:
            return None
        files.sort(key=lambda f: int(f[len("reward_predictor_"):-len(".pt")]))
        return osp.join(ckpt_dir, files[-1])

    # ----------------------------------------------------------

    def raw_rewards(self, obs):

        obs = torch.tensor(obs).float().to(self.device)

        rs = []

        for model in self.models:

            model.eval()

            with torch.no_grad():

                r = model.core_network(obs)

                r = r.cpu().numpy()

                rs.append(r)

        rs = np.array(rs)

        return rs

    # ----------------------------------------------------------

    def reward(self, obs):

        ensemble_rs = self.raw_rewards(obs)

        n_preds, n_steps = ensemble_rs.shape

        ensemble_rs = ensemble_rs.T

        for step_reward in ensemble_rs:
            self.r_norm.push(step_reward)

        ensemble_rs -= self.r_norm.mean
        ensemble_rs /= (self.r_norm.std + 1e-12)
        ensemble_rs *= 0.05

        ensemble_rs = ensemble_rs.T

        rs = np.mean(ensemble_rs, axis=0)

        return rs

    # ----------------------------------------------------------

    def preferences(self, s1s, s2s):

        s1s = torch.tensor(s1s).float().to(self.device)
        s2s = torch.tensor(s2s).float().to(self.device)

        preds = []

        for model in self.models:

            model.eval()

            with torch.no_grad():

                _, _, _, _, pred, _ = model(s1s, s2s)

                preds.append(pred.cpu().numpy())

        return preds

    # ----------------------------------------------------------

    def train(self, prefs_train, prefs_val, val_interval):

        print(
            "Training/testing with %d/%d preferences"
            % (len(prefs_train), len(prefs_val))
        )

        start_time = time.time()

        for _, batch in enumerate(batch_iter(prefs_train.prefs, 32, shuffle=True)):

            self.train_step(batch, prefs_train)

            self.n_steps += 1

            if self.n_steps % val_interval == 0:
                self.val_step(prefs_val)

        rate = self.n_steps / (time.time() - start_time)

        self.writer_train.add_scalar(
            "reward_predictor_training_steps_per_second", rate, self.n_steps
        )

    # ----------------------------------------------------------

    def train_step(self, batch, prefs_train):

        s1s = [prefs_train.segments[k1] for k1, k2, _ in batch]
        s2s = [prefs_train.segments[k2] for k1, k2, _ in batch]
        prefs = [pref for _, _, pref in batch]

        s1s = torch.tensor(s1s).float().to(self.device)
        s2s = torch.tensor(s2s).float().to(self.device)
        prefs = torch.tensor(prefs).float().to(self.device)

        for model, optimizer in zip(self.models, self.optimizers):

            model.train()

            optimizer.zero_grad()

            _, _, _, _, pred, logits = model(s1s, s2s)

            targets = prefs.argmax(dim=1)

            loss = nn.CrossEntropyLoss(reduction="sum")(logits, targets)

            loss.backward()

            optimizer.step()

            acc = (pred.argmax(dim=1) == targets).float().mean()

            if self.writer_train is not None:
                self.writer_train.add_scalar(
                    "reward_predictor_loss", loss.item(), self.n_steps
                )
                self.writer_train.add_scalar(
                    "reward_predictor_accuracy", acc.item(), self.n_steps
                )

    # ----------------------------------------------------------

    def val_step(self, prefs_val):

        batch_size = min(32, len(prefs_val.prefs))

        idxs = np.random.choice(len(prefs_val.prefs), batch_size, replace=False)

        batch = [prefs_val.prefs[i] for i in idxs]

        s1s = [prefs_val.segments[k1] for k1, k2, _ in batch]
        s2s = [prefs_val.segments[k2] for k1, k2, _ in batch]
        prefs = [pref for _, _, pref in batch]

        s1s = torch.tensor(s1s).float().to(self.device)
        s2s = torch.tensor(s2s).float().to(self.device)
        prefs = torch.tensor(prefs).float().to(self.device)

        for model in self.models:

            model.eval()

            with torch.no_grad():

                _, _, _, _, pred, logits = model(s1s, s2s)

                targets = prefs.argmax(dim=1)

                loss = nn.CrossEntropyLoss(reduction="sum")(logits, targets)

                acc = (pred.argmax(dim=1) == targets).float().mean()

                if self.writer_test is not None:
                    self.writer_test.add_scalar(
                        "reward_predictor_val_loss", loss.item(), self.n_steps
                    )
                    self.writer_test.add_scalar(
                        "reward_predictor_val_accuracy", acc.item(), self.n_steps
                    )