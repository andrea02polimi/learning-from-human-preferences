#!/usr/bin/env python3
"""
Main entry point — Deep RL from Human Preferences (Christiano et al., 2017).

Four modes
──────────
gather_initial_prefs
    Run A2C with env rewards and collect labeled segment pairs.
    Saves train.pkl.gz / val.pkl.gz to --log_dir.

pretrain_reward_predictor
    Load saved preferences from --load_prefs_dir and train the reward
    predictor for --n_initial_epochs epochs. No policy training.

train_policy_with_original_rewards
    Baseline: train A2C using environment rewards only.

train_policy_with_preferences
    Full paper pipeline:
      1. Collect initial preferences (random policy + human / synthetic labels)
      2. Pretrain reward predictor
      3. A2C trains with reward predictor while preferences and reward
         predictor both keep updating.
"""

import functools
import logging
import os
import os.path as osp
import signal
import sys
import time
from multiprocessing import Process, Queue

import multiprocessing as mp

import gymnasium as gym
import ale_py

from learning_from_human_preferences.agents.a2c.a2c import learn
from learning_from_human_preferences.agents.a2c.policies import CnnPolicy, MlpPolicy
from learning_from_human_preferences.agents.common import set_global_seeds
from learning_from_human_preferences.agents.common.vec_env.subproc_vec_env import SubprocVecEnv
from learning_from_human_preferences.envs.utils import VideoRenderer, make_env
from learning_from_human_preferences.preferences.pref_db import PrefDB, PrefBuffer
from learning_from_human_preferences.preferences.pref_interface import PrefInterface
from learning_from_human_preferences.reward_model.reward_predictor import RewardPredictorEnsemble
from learning_from_human_preferences.reward_model.reward_predictor_core_network import (
    AtariRewardNetwork,
    MovingDotRewardNetwork,
)
from learning_from_human_preferences.training.params import parse_args, PREFS_VAL_FRACTION


# ============================================================
# Helpers
# ============================================================

def _select_policy(env_id):
    return MlpPolicy if "MovingDot" in env_id else CnnPolicy


def _select_reward_network(env_id, dropout):
    if "MovingDot" in env_id:
        return MovingDotRewardNetwork
    return functools.partial(AtariRewardNetwork, dropout_prob=dropout)


def _make_pref_dbs(max_prefs):
    n_train = int(max_prefs * (1 - PREFS_VAL_FRACTION))
    n_val   = int(max_prefs * PREFS_VAL_FRACTION)
    return PrefDB(maxlen=n_train), PrefDB(maxlen=n_val)


def _save_prefs(log_dir, db_train, db_val):
    train_path = osp.join(log_dir, "train.pkl.gz")
    db_train.save(train_path)
    print(f"Saved training preferences → '{train_path}'")

    val_path = osp.join(log_dir, "val.pkl.gz")
    db_val.save(val_path)
    print(f"Saved validation preferences → '{val_path}'")


# ============================================================
# Worker: A2C training  (non-daemon — may spawn SubprocVecEnv)
# ============================================================

def _training_worker(
    policy_fn,
    env_id,
    n_envs,
    seed,
    seg_pipe,
    start_flag,
    episode_vid_queue,
    ckpt_dir,
    gen_segments,
    rew_pred_ckpt_dir,      # None → use env rewards
    reward_network_fn,      # callable() → nn.Module  (or None)
    a2c_params,
    log_dir,
):
    """Runs A2C in a separate process.  Non-daemon so it can spawn its own
    SubprocVecEnv workers without hitting Python's daemon restriction."""

    import gymnasium as gym, ale_py
    gym.register_envs(ale_py)

    def _thunk(rank):
        def f(): return make_env(env_id, seed + rank)
        return f

    set_global_seeds(seed)
    env = SubprocVecEnv(env_id, [_thunk(i) for i in range(n_envs)])

    reward_predictor = None
    if rew_pred_ckpt_dir is not None and reward_network_fn is not None:
        reward_predictor = RewardPredictorEnsemble(
            core_network=reward_network_fn,
            log_dir=None,   # inference-only; no TensorBoard / checkpoint writes
        )

    learn(
        policy=policy_fn,
        env=env,
        seg_pipe=seg_pipe,
        start_policy_training_pipe=start_flag,
        episode_vid_queue=episode_vid_queue,
        reward_predictor=reward_predictor,
        ckpt_save_dir=ckpt_dir,
        gen_segments=gen_segments,
        log_dir=log_dir,
        **a2c_params,
    )


# ============================================================
# Worker: preference interface  (daemon)
# ============================================================

def _pref_interface_worker(synthetic_prefs, max_segs, pi_log_dir,
                           seg_pipe, pref_pipe):
    """Asks for human (or synthetic) labels and forwards them to pref_pipe."""
    import gymnasium as gym, ale_py
    gym.register_envs(ale_py)
    sys.stdin = os.fdopen(0)

    pi = PrefInterface(
        synthetic_prefs=synthetic_prefs,
        max_segs=max_segs,
        log_dir=pi_log_dir,
    )
    pi.run(seg_pipe, pref_pipe)


# ============================================================
# Main
# ============================================================

def main():
    mp.set_start_method("spawn", force=True)
    gym.register_envs(ale_py)

    general_params, a2c_params, pref_params, rew_pred_params = parse_args()

    if general_params["debug"]:
        logging.getLogger().setLevel(logging.DEBUG)

    log_dir  = general_params["log_dir"]
    mode     = general_params["mode"]
    env_id   = a2c_params["env_id"]
    dropout  = rew_pred_params["dropout"]

    ckpt_dir         = osp.join(log_dir, "policy_checkpoints")
    rew_pred_ckpt_dir = osp.join(log_dir, "reward_predictor_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(rew_pred_ckpt_dir, exist_ok=True)

    reward_network_fn = _select_reward_network(env_id, dropout)

    # Optional episode renderer (GUI subprocess)
    if general_params["render_episodes"]:
        episode_vid_queue = Queue()
        episode_renderer = VideoRenderer(
            episode_vid_queue,
            mode=VideoRenderer.play_through_mode,
            zoom_factor=2, playback_speed=2,
        )
    else:
        episode_vid_queue = None
        episode_renderer  = None

    # ──────────────────────────────────────────────────────
    if mode == "gather_initial_prefs":
        _gather_initial_prefs(
            general_params, a2c_params, pref_params,
            env_id, ckpt_dir, episode_vid_queue, reward_network_fn, log_dir,
        )

    elif mode == "pretrain_reward_predictor":
        _pretrain_reward_predictor(
            general_params, rew_pred_params, reward_network_fn,
            rew_pred_ckpt_dir, log_dir,
        )

    elif mode == "train_policy_with_original_rewards":
        _train_policy_original_rewards(
            a2c_params, ckpt_dir, env_id, episode_vid_queue,
        )

    elif mode == "train_policy_with_preferences":
        _train_policy_with_preferences(
            general_params, a2c_params, pref_params, rew_pred_params,
            env_id, ckpt_dir, rew_pred_ckpt_dir,
            episode_vid_queue, reward_network_fn, log_dir,
        )

    else:
        raise RuntimeError(f"Unknown mode: {mode}")

    if episode_renderer:
        episode_renderer.stop()


# ============================================================
# Mode: gather_initial_prefs
# ============================================================

def _gather_initial_prefs(
    general_params, a2c_params, pref_params,
    env_id, ckpt_dir, episode_vid_queue, reward_network_fn, log_dir,
):
    """
    Train a policy with env rewards while sending segments to the preference
    interface.  Stop once n_initial_prefs labeled pairs have been collected.
    """
    seg_pipe  = Queue(maxsize=1)
    pref_pipe = Queue(maxsize=1)
    start_flag = Queue(maxsize=1)

    seed   = a2c_params["seed"]
    n_envs = a2c_params["n_envs"]
    extra  = {k: v for k, v in a2c_params.items() if k not in ("env_id", "n_envs")}

    trainer = Process(
        target=_training_worker,
        args=(
            _select_policy(env_id), env_id, n_envs, seed,
            seg_pipe, start_flag, episode_vid_queue,
            ckpt_dir,
            True,          # gen_segments
            None, None,    # no reward predictor
            extra,
            log_dir,
        ),
    )
    trainer.start()

    pi_proc = Process(
        target=_pref_interface_worker,
        args=(
            pref_params["synthetic_prefs"],
            pref_params["max_segs"],
            osp.join(log_dir, "pref_interface"),
            seg_pipe, pref_pipe,
        ),
    )
    pi_proc.start()

    db_train, db_val = _make_pref_dbs(general_params["max_prefs"])
    buf = PrefBuffer(db_train=db_train, db_val=db_val,
                     log_dir=osp.join(log_dir, "pref_buffer"))
    buf.start_recv_thread(pref_pipe)

    def _shutdown(*_):
        trainer.terminate()
        pi_proc.terminate()
        buf.stop_recv_thread()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)

    buf.wait_until_len(general_params["n_initial_prefs"])

    _save_prefs(log_dir, *buf.get_dbs())

    trainer.terminate()
    pi_proc.terminate()
    buf.stop_recv_thread()


# ============================================================
# Mode: pretrain_reward_predictor
# ============================================================

def _pretrain_reward_predictor(
    general_params, rew_pred_params, reward_network_fn,
    rew_pred_ckpt_dir, log_dir,
):
    """
    Load saved preference databases and pretrain the reward predictor.
    No policy training happens here.
    """
    prefs_dir = general_params["prefs_dir"]
    if prefs_dir is None:
        raise RuntimeError(
            "--load_prefs_dir is required for pretrain_reward_predictor"
        )

    db_train = PrefDB.load(osp.join(prefs_dir, "train.pkl.gz"))
    db_val   = PrefDB.load(osp.join(prefs_dir, "val.pkl.gz"))
    print(f"Loaded {len(db_train)} train / {len(db_val)} val preferences")

    rp = RewardPredictorEnsemble(
        core_network=reward_network_fn,
        lr=rew_pred_params["lr"],
        log_dir=log_dir,
    )

    if rew_pred_params.get("load_ckpt_dir"):
        ckpt = RewardPredictorEnsemble.latest_checkpoint(rew_pred_params["load_ckpt_dir"])
        if ckpt:
            rp.load(ckpt)

    n_epochs     = rew_pred_params["n_initial_epochs"]
    val_interval = rew_pred_params["val_interval"]
    ckpt_interval = rew_pred_params["ckpt_interval"]

    print(f"Pretraining reward predictor for {n_epochs} epochs…")
    for epoch in range(n_epochs):
        rp.train(db_train, db_val, val_interval)
        if (epoch + 1) % ckpt_interval == 0:
            rp.save()

    rp.save()
    print("Pretraining complete.")


# ============================================================
# Mode: train_policy_with_original_rewards
# ============================================================

def _train_policy_original_rewards(a2c_params, ckpt_dir, env_id, episode_vid_queue):
    """Baseline: A2C with environment rewards, no reward predictor."""

    seg_pipe   = Queue(maxsize=1)
    start_flag = Queue(maxsize=1)

    seed   = a2c_params["seed"]
    n_envs = a2c_params["n_envs"]
    extra  = {k: v for k, v in a2c_params.items() if k not in ("env_id", "n_envs")}

    trainer = Process(
        target=_training_worker,
        args=(
            _select_policy(env_id), env_id, n_envs, seed,
            seg_pipe, start_flag, episode_vid_queue,
            ckpt_dir,
            False,         # gen_segments=False → skip Phase 1
            None, None,
            extra,
            log_dir,
        ),
    )
    trainer.start()

    def _shutdown(*_):
        trainer.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    trainer.join()


# ============================================================
# Mode: train_policy_with_preferences
# ============================================================

def _train_policy_with_preferences(
    general_params, a2c_params, pref_params, rew_pred_params,
    env_id, ckpt_dir, rew_pred_ckpt_dir,
    episode_vid_queue, reward_network_fn, log_dir,
):
    """
    Full Christiano et al. pipeline:

      Phase 1  Collect initial preferences while running A2C with env rewards.
      Phase 2  Pretrain reward predictor on the initial preferences.
      Phase 3  Signal trainer to switch to reward-predictor rewards; continue
               collecting preferences and updating the reward predictor.
    """
    seg_pipe   = Queue(maxsize=1)
    pref_pipe  = Queue(maxsize=1)
    start_flag = Queue(maxsize=1)   # signals trainer: switch to Phase 2

    seed   = a2c_params["seed"]
    n_envs = a2c_params["n_envs"]
    extra  = {k: v for k, v in a2c_params.items() if k not in ("env_id", "n_envs")}

    # Trainer: starts in Phase 1 (generates segments, env rewards).
    # After start_flag it loads the reward predictor checkpoint and trains with it.
    trainer = Process(
        target=_training_worker,
        args=(
            _select_policy(env_id), env_id, n_envs, seed,
            seg_pipe, start_flag, episode_vid_queue,
            ckpt_dir,
            True,                  # gen_segments
            rew_pred_ckpt_dir,     # reward predictor will be loaded from here
            reward_network_fn,
            extra,
            log_dir,
        ),
    )
    trainer.start()

    # Preference interface subprocess
    pi_proc = Process(
        target=_pref_interface_worker,
        args=(
            pref_params["synthetic_prefs"],
            pref_params["max_segs"],
            osp.join(log_dir, "pref_interface"),
            seg_pipe, pref_pipe,
        ),
    )
    pi_proc.start()

    # Preference buffer: async thread in main process
    db_train, db_val = _make_pref_dbs(general_params["max_prefs"])
    buf = PrefBuffer(db_train=db_train, db_val=db_val,
                     log_dir=osp.join(log_dir, "pref_buffer"))
    buf.start_recv_thread(pref_pipe)

    def _shutdown(*_):
        trainer.terminate()
        pi_proc.terminate()
        buf.stop_recv_thread()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)

    # ── Phase 1: wait for initial preferences ──────────────────────────────
    print(f"Phase 1: collecting {general_params['n_initial_prefs']} "
          "initial preferences…")
    buf.wait_until_len(general_params["n_initial_prefs"])
    print("Initial preferences collected.")

    # ── Phase 2: pretrain reward predictor ─────────────────────────────────
    rp = RewardPredictorEnsemble(
        core_network=reward_network_fn,
        lr=rew_pred_params["lr"],
        log_dir=log_dir,
    )

    if rew_pred_params.get("load_ckpt_dir"):
        ckpt = RewardPredictorEnsemble.latest_checkpoint(
            rew_pred_params["load_ckpt_dir"]
        )
        if ckpt:
            rp.load(ckpt)
            print(f"Loaded reward predictor checkpoint from {ckpt}")
    else:
        n_epochs     = rew_pred_params["n_initial_epochs"]
        val_interval = rew_pred_params["val_interval"]
        ckpt_interval = rew_pred_params["ckpt_interval"]

        print(f"Pretraining reward predictor for {n_epochs} epochs…")
        for epoch in range(n_epochs):
            t_db, v_db = buf.get_dbs()
            rp.train(t_db, v_db, val_interval)
            if (epoch + 1) % ckpt_interval == 0:
                rp.save()
        rp.save()
        print("Reward predictor pretraining done.")

    # ── Phase 3: signal trainer + continuous RP training ───────────────────
    start_flag.put(True)   # trainer exits Phase 1 and loads the checkpoint
    print("Phase 3: A2C training with reward predictor…")

    val_interval  = rew_pred_params["val_interval"]
    ckpt_interval = rew_pred_params["ckpt_interval"]
    epoch = 0

    while trainer.is_alive():
        t_db, v_db = buf.get_dbs()
        if len(t_db) > 0 and len(v_db) > 0:
            rp.train(t_db, v_db, val_interval)
            epoch += 1
            if epoch % ckpt_interval == 0:
                rp.save()
        else:
            time.sleep(1.0)

    rp.save()
    pi_proc.terminate()
    buf.stop_recv_thread()
    print("Training complete.")


if __name__ == "__main__":
    main()
