#!/usr/bin/env python3

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import os
import os.path as osp
import sys
import logging
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



import gymnasium as gym
import ale_py

gym.register_envs(ale_py)


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

    # ======================================================
    # Gather preferences
    # ======================================================

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

    # ======================================================
    # Pretrain reward predictor
    # ======================================================

    elif mode == "pretrain_reward_predictor":

        cluster_dict = create_cluster_dict(["ps", "train"])

        ps_proc = start_parameter_server(
            cluster_dict,
            make_reward_predictor,
        )

        rpt_proc = start_reward_predictor_training(
            cluster_dict=cluster_dict,
            make_reward_predictor=make_reward_predictor,
            just_pretrain=True,
            pref_pipe=pref_pipe,
            start_policy_training_pipe=start_policy_training_flag,
            max_prefs=general_params["max_prefs"],
            prefs_dir=general_params["prefs_dir"],
            load_ckpt_dir=None,
            n_initial_prefs=general_params["n_initial_prefs"],
            n_initial_epochs=rew_pred_params["n_initial_epochs"],
            val_interval=rew_pred_params["val_interval"],
            ckpt_interval=rew_pred_params["ckpt_interval"],
            log_dir=general_params["log_dir"],
        )

        rpt_proc.join()
        ps_proc.terminate()

    # ======================================================
    # Train policy with environment rewards
    # ======================================================

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

    # ======================================================
    # RL with preferences
    # ======================================================

    elif mode == "train_policy_with_preferences":

        cluster_dict = create_cluster_dict(["ps", "a2c", "train"])

        ps_proc = start_parameter_server(
            cluster_dict,
            make_reward_predictor,
        )

        env, a2c_proc = start_policy_training(
            cluster_dict=cluster_dict,
            make_reward_predictor=make_reward_predictor,
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

        rpt_proc = start_reward_predictor_training(
            cluster_dict=cluster_dict,
            make_reward_predictor=make_reward_predictor,
            just_pretrain=False,
            pref_pipe=pref_pipe,
            start_policy_training_pipe=start_policy_training_flag,
            max_prefs=general_params["max_prefs"],
            prefs_dir=general_params["prefs_dir"],
            load_ckpt_dir=rew_pred_params["load_ckpt_dir"],
            n_initial_prefs=general_params["n_initial_prefs"],
            n_initial_epochs=rew_pred_params["n_initial_epochs"],
            val_interval=rew_pred_params["val_interval"],
            ckpt_interval=rew_pred_params["ckpt_interval"],
            log_dir=general_params["log_dir"],
        )

        a2c_proc.join()

        rpt_proc.terminate()
        pi_proc.terminate()

        pi.stop_renderer()

        ps_proc.terminate()

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
# Cluster utilities
# ==========================================================

def create_cluster_dict(jobs):

    ports = get_port_range(
        start_port=2200,
        n_ports=len(jobs) + 1,
        random_stagger=True,
    )

    cluster_dict = {}

    for part, port in zip(jobs, ports):
        cluster_dict[part] = [f"localhost:{port}"]

    return cluster_dict


def start_parameter_server(cluster_dict, make_reward_predictor):

    def f():
        make_reward_predictor("ps", cluster_dict)
        while True:
            time.sleep(1)

    proc = Process(target=f, daemon=True)
    proc.start()

    return proc


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

    if "MovingDot" in env_id:
        policy_fn = MlpPolicy
    else:
        policy_fn = CnnPolicy

    env = make_envs(
        a2c_params["env_id"],
        a2c_params["n_envs"],
        a2c_params["seed"],
    )

    del a2c_params["env_id"]
    del a2c_params["n_envs"]

    ckpt_dir = osp.join(log_dir, "policy_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    def f():

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

    proc = Process(target=f, daemon=True)
    proc.start()

    return env, proc


# ==========================================================
# Preference interface
# ==========================================================

def start_pref_interface(seg_pipe, pref_pipe, max_segs, synthetic_prefs, log_dir):

    prefs_log_dir = osp.join(log_dir, "pref_interface")

    pi = PrefInterface(
        synthetic_prefs=synthetic_prefs,
        max_segs=max_segs,
        log_dir=prefs_log_dir,
    )

    def f():
        sys.stdin = os.fdopen(0)
        pi.run(segment_pipe=seg_pipe, preference_pipe=pref_pipe)

    proc = Process(target=f, daemon=True)
    proc.start()

    return pi, proc


# ==========================================================
# Reward predictor training
# ==========================================================

def start_reward_predictor_training(
    cluster_dict,
    make_reward_predictor,
    just_pretrain,
    pref_pipe,
    start_policy_training_pipe,
    max_prefs,
    prefs_dir,
    load_ckpt_dir,
    n_initial_prefs,
    n_initial_epochs,
    val_interval,
    ckpt_interval,
    log_dir,
):

    def f():

        rew_pred = make_reward_predictor("train", cluster_dict)

        rew_pred.init_network(load_ckpt_dir)

        if prefs_dir is not None:

            train_path = osp.join(prefs_dir, "train.pkl.gz")
            pref_db_train = PrefDB.load(train_path)

            val_path = osp.join(prefs_dir, "val.pkl.gz")
            pref_db_val = PrefDB.load(val_path)

        else:

            n_train = int(max_prefs * (1 - PREFS_VAL_FRACTION))
            n_val = int(max_prefs * PREFS_VAL_FRACTION)

            pref_db_train = PrefDB(maxlen=n_train)
            pref_db_val = PrefDB(maxlen=n_val)

        pref_buffer = PrefBuffer(
            db_train=pref_db_train,
            db_val=pref_db_val,
        )

        pref_buffer.start_recv_thread(pref_pipe)

        if prefs_dir is None:
            pref_buffer.wait_until_len(n_initial_prefs)

        save_prefs(log_dir, pref_db_train, pref_db_val)

        if not load_ckpt_dir:

            for i in range(n_initial_epochs):

                pref_db_train, pref_db_val = pref_buffer.get_dbs()

                rew_pred.train(pref_db_train, pref_db_val, val_interval)

                if i and i % ckpt_interval == 0:
                    rew_pred.save()

            rew_pred.save()

        if just_pretrain:
            return

        start_policy_training_pipe.put(True)

        i = 0

        while True:

            pref_db_train, pref_db_val = pref_buffer.get_dbs()

            save_prefs(log_dir, pref_db_train, pref_db_val)

            rew_pred.train(pref_db_train, pref_db_val, val_interval)

            if i and i % ckpt_interval == 0:
                rew_pred.save()

            i += 1

    proc = Process(target=f, daemon=True)
    proc.start()

    return proc


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