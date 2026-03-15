#!/usr/bin/env python3

import os
import os.path as osp
import sys
import logging
import multiprocessing as mp
import time

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
    get_port_range,
    make_env,
)

mp.set_start_method("spawn", force=True)

import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

# ==========================================================
# Worker process (necessario per spawn)
# ==========================================================

def policy_training_worker(
    policy_fn,
    env_id,
    n_envs,
    seed,
    seg_pipe,
    start_policy_training_pipe,
    episode_vid_queue,
    make_reward_predictor,
    cluster_dict,
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

    reward_predictor = (
        make_reward_predictor("a2c", cluster_dict)
        if make_reward_predictor
        else None
    )

    learn(
        policy=policy_fn,
        env=env,
        seg_pipe=seg_pipe,
        start_policy_training_pipe=start_policy_training_pipe,
        episode_vid_queue=episode_vid_queue,
        reward_predictor=reward_predictor,
        ckpt_save_dir=ckpt_dir,
        gen_segments=gen_segments,
        **a2c_params,
    )


# ==========================================================
# Main
# ==========================================================

def main():

    general_params, a2c_params, pref_interface_params, rew_pred_params = parse_args()

    if general_params["debug"]:
        logging.getLogger().setLevel(logging.DEBUG)

    run(
        general_params,
        a2c_params,
        pref_interface_params,
        rew_pred_params,
    )


# ==========================================================
# Experiment runner
# ==========================================================

def run(
    general_params,
    a2c_params,
    pref_interface_params,
    rew_pred_params,
):

    seg_pipe = Queue(maxsize=1)
    pref_pipe = Queue(maxsize=1)
    start_policy_training_flag = Queue(maxsize=1)

    if general_params["render_episodes"]:
        episode_vid_queue, episode_renderer = start_episode_renderer()
    else:
        episode_vid_queue = None
        episode_renderer = None

    env_id = a2c_params["env_id"]

    if "MovingDot" in env_id:
        reward_predictor_network = MovingDotRewardNetwork

    elif "Pong" in env_id or "Enduro" in env_id or env_id.startswith("ALE/"):
        reward_predictor_network = AtariRewardNetwork

    else:
        raise RuntimeError(f"Unknown reward predictor network for {env_id}")

    def make_reward_predictor(name, cluster_dict):

        return RewardPredictorEnsemble(
            cluster_job_name=name,
            cluster_dict=cluster_dict,
            log_dir=general_params["log_dir"],
            batchnorm=rew_pred_params["batchnorm"],
            dropout=rew_pred_params["dropout"],
            lr=rew_pred_params["lr"],
            core_network=reward_predictor_network,
        )

    save_make_reward_predictor(general_params["log_dir"], make_reward_predictor)

    mode = general_params["mode"]

    if mode == "gather_initial_prefs":

        env, a2c_proc = start_policy_training(
            cluster_dict=None,
            make_reward_predictor=None,
            gen_segments=True,
            start_policy_training_pipe=start_policy_training_flag,
            seg_pipe=seg_pipe,
            episode_vid_queue=episode_vid_queue,
            log_dir=general_params["log_dir"],
            a2c_params=a2c_params,
        )

        pi, pi_proc = start_pref_interface(
            seg_pipe=seg_pipe,
            pref_pipe=pref_pipe,
            log_dir=general_params["log_dir"],
            **pref_interface_params,
        )

        n_train = int(general_params["max_prefs"] * (1 - PREFS_VAL_FRACTION))
        n_val = int(general_params["max_prefs"] * PREFS_VAL_FRACTION)

        pref_db_train = PrefDB(maxlen=n_train)
        pref_db_val = PrefDB(maxlen=n_val)

        pref_buffer = PrefBuffer(
            db_train=pref_db_train,
            db_val=pref_db_val,
        )

        pref_buffer.start_recv_thread(pref_pipe)
        pref_buffer.wait_until_len(general_params["n_initial_prefs"])

        pref_db_train, pref_db_val = pref_buffer.get_dbs()

        save_prefs(general_params["log_dir"], pref_db_train, pref_db_val)

        pi_proc.terminate()
        pi.stop_renderer()

        a2c_proc.terminate()

        pref_buffer.stop_recv_thread()

        env.close()

    elif mode == "train_policy_with_original_rewards":

        env, a2c_proc = start_policy_training(
            cluster_dict=None,
            make_reward_predictor=None,
            gen_segments=False,
            start_policy_training_pipe=start_policy_training_flag,
            seg_pipe=seg_pipe,
            episode_vid_queue=episode_vid_queue,
            log_dir=general_params["log_dir"],
            a2c_params=a2c_params,
        )

        start_policy_training_flag.put(True)

        a2c_proc.join()

        env.close()

    else:
        raise RuntimeError(f"Unknown mode: {mode}")

    if episode_renderer:
        episode_renderer.stop()


# ==========================================================
# Utilities
# ==========================================================

def save_prefs(log_dir, pref_db_train, pref_db_val):

    train_path = osp.join(log_dir, "train.pkl.gz")
    pref_db_train.save(train_path)

    print(f"Saved training preferences to '{train_path}'")

    val_path = osp.join(log_dir, "val.pkl.gz")
    pref_db_val.save(val_path)

    print(f"Saved validation preferences to '{val_path}'")


def save_make_reward_predictor(log_dir, make_reward_predictor):

    save_dir = osp.join(log_dir, "reward_predictor_checkpoints")

    os.makedirs(save_dir, exist_ok=True)

    with open(osp.join(save_dir, "make_reward_predictor.pkl"), "wb") as f:
        f.write(cloudpickle.dumps(make_reward_predictor))


# ==========================================================
# Environment creation
# ==========================================================

def make_envs(env_id, n_envs, seed):

    def wrap_make_env(rank):

        def thunk():
            return make_env(env_id, seed + rank)

        return thunk

    set_global_seeds(seed)

    env = SubprocVecEnv(
        env_id,
        [wrap_make_env(i) for i in range(n_envs)]
    )

    return env


# ==========================================================
# Policy training
# ==========================================================

def start_policy_training(
    cluster_dict,
    make_reward_predictor,
    gen_segments,
    start_policy_training_pipe,
    seg_pipe,
    episode_vid_queue,
    log_dir,
    a2c_params,
):

    env_id = a2c_params["env_id"]
    n_envs = a2c_params["n_envs"]
    seed = a2c_params["seed"]

    if "MovingDot" in env_id:
        policy_fn = MlpPolicy
    else:
        policy_fn = CnnPolicy

    env = None

    del a2c_params["env_id"]
    del a2c_params["n_envs"]

    ckpt_dir = osp.join(log_dir, "policy_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    proc = Process(
        target=policy_training_worker,
        args=(
            policy_fn,
            env_id,
            n_envs,
            seed,
            seg_pipe,
            start_policy_training_pipe,
            episode_vid_queue,
            make_reward_predictor,
            cluster_dict,
            ckpt_dir,
            gen_segments,
            a2c_params,
        ),
    )

    proc.start()

    return env, proc

def pref_interface_worker(seg_pipe, pref_pipe, max_segs, synthetic_prefs, log_dir):

    sys.stdin = os.fdopen(0)

    prefs_log_dir = osp.join(log_dir, "pref_interface")

    pi = PrefInterface(
        synthetic_prefs=synthetic_prefs,
        max_segs=max_segs,
        log_dir=prefs_log_dir,
    )

    pi.run(
        segment_pipe=seg_pipe,
        preference_pipe=pref_pipe,
    )


# ==========================================================
# Preference interface
# ==========================================================

def start_pref_interface(seg_pipe, pref_pipe, max_segs, synthetic_prefs, log_dir):

    proc = Process(
        target=pref_interface_worker,
        args=(seg_pipe, pref_pipe, max_segs, synthetic_prefs, log_dir),
    )

    proc.start()

    return None, proc


# ==========================================================
# Episode renderer
# ==========================================================

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