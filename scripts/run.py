#!/usr/bin/env python3

import os
import os.path as osp
import sys
import logging
import multiprocessing as mp
import signal

from multiprocessing import Process, Queue

import cloudpickle

from learning_from_human_preferences.agents.a2c.a2c import learn
from learning_from_human_preferences.agents.a2c.policies import CnnPolicy, MlpPolicy
from learning_from_human_preferences.agents.common import set_global_seeds
from learning_from_human_preferences.agents.common.vec_env.subproc_vec_env import SubprocVecEnv

from learning_from_human_preferences.training.params import (
    parse_args,
    PREFS_VAL_FRACTION,
)

from learning_from_human_preferences.preferences.pref_db import (
    PrefDB,
    PrefBuffer,
)

from learning_from_human_preferences.preferences.pref_interface import (
    PrefInterface,
)

from learning_from_human_preferences.reward_model.reward_predictor import (
    RewardPredictorEnsemble,
)

from learning_from_human_preferences.reward_model.reward_predictor_core_network import (
    AtariRewardNetwork,
    MovingDotRewardNetwork,
)

from learning_from_human_preferences.envs.utils import (
    VideoRenderer,
    make_env,
)

mp.set_start_method("spawn", force=True)

import gymnasium as gym
import ale_py

gym.register_envs(ale_py)


# ==========================================================
# Training worker
# ==========================================================

def training_worker(
    policy_fn,
    env_id,
    n_envs,
    seed,
    seg_pipe,
    start_policy_training_pipe,
    episode_vid_queue,
    ckpt_dir,
    gen_segments,
    a2c_params,
):

    def wrap_make_env(rank):
        def thunk():
            return make_env(env_id, seed + rank)
        return thunk

    set_global_seeds(seed)

    env = SubprocVecEnv(
        env_id,
        [wrap_make_env(i) for i in range(n_envs)]
    )

    learn(
        policy=policy_fn,
        env=env,
        seg_pipe=seg_pipe,
        start_policy_training_pipe=start_policy_training_pipe,
        episode_vid_queue=episode_vid_queue,
        reward_predictor=None,
        ckpt_save_dir=ckpt_dir,
        gen_segments=gen_segments,
        **a2c_params,
    )


# ==========================================================
# Main
# ==========================================================

def main():

    general_params, a2c_params, pref_interface_params, rew_pred_params = parse_args()

    seg_pipe = Queue(maxsize=1)
    pref_pipe = Queue(maxsize=1)
    start_flag = Queue(maxsize=1)

    if general_params["render_episodes"]:
        episode_vid_queue, episode_renderer = start_episode_renderer()
    else:
        episode_vid_queue = None
        episode_renderer = None

    env_id = a2c_params["env_id"]

    if "MovingDot" in env_id:
        policy_fn = MlpPolicy
    else:
        policy_fn = CnnPolicy

    seed = a2c_params["seed"]
    n_envs = a2c_params["n_envs"]

    del a2c_params["env_id"]
    del a2c_params["n_envs"]

    ckpt_dir = osp.join(general_params["log_dir"], "policy_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    trainer = Process(
        target=training_worker,
        args=(
            policy_fn,
            env_id,
            n_envs,
            seed,
            seg_pipe,
            start_flag,
            episode_vid_queue,
            ckpt_dir,
            True,
            a2c_params,
        ),
    )

    trainer.start()

    pi = PrefInterface(
        synthetic_prefs=pref_interface_params["synthetic_prefs"],
        max_segs=pref_interface_params["max_segs"],
        log_dir=osp.join(general_params["log_dir"], "pref_interface"),
    )

    pref_db_train = PrefDB(maxlen=int(general_params["max_prefs"] * (1 - PREFS_VAL_FRACTION)))
    pref_db_val = PrefDB(maxlen=int(general_params["max_prefs"] * PREFS_VAL_FRACTION))

    pref_buffer = PrefBuffer(
        db_train=pref_db_train,
        db_val=pref_db_val,
    )

    pref_buffer.start_recv_thread(pref_pipe)

    def shutdown(*_):
        trainer.terminate()
        pref_buffer.stop_recv_thread()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    print("Starting preference GUI")

    # GUI nel main thread
    pi.run(seg_pipe, pref_pipe)

    pref_buffer.wait_until_len(general_params["n_initial_prefs"])

    train_db, val_db = pref_buffer.get_dbs()

    save_prefs(general_params["log_dir"], train_db, val_db)

    trainer.terminate()

    if episode_renderer:
        episode_renderer.stop()


# ==========================================================
# Utils
# ==========================================================

def save_prefs(log_dir, pref_db_train, pref_db_val):

    train_path = osp.join(log_dir, "train.pkl.gz")
    pref_db_train.save(train_path)

    val_path = osp.join(log_dir, "val.pkl.gz")
    pref_db_val.save(val_path)

    print("Preferences saved")


def start_episode_renderer():

    episode_vid_queue = Queue()

    renderer = VideoRenderer(
        episode_vid_queue,
        mode=VideoRenderer.play_through_mode,
        zoom_factor=2,
        playback_speed=2,
    )

    return episode_vid_queue, renderer


if __name__ == "__main__":
    main()