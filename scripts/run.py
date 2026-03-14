#!/usr/bin/env python3

import os
import os.path as osp
import sys
import logging
import multiprocessing as mp

from multiprocessing import Process, Queue

import cloudpickle

from agents.a2c.a2c import learn
from agents.a2c.policies import CnnPolicy, MlpPolicy
from agents.common import set_global_seeds
from agents.common.vec_env.dummy_vec_env import DummyVecEnv

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

mp.set_start_method("fork", force=True)


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

    # ------------------------------------------------------
    # Episode renderer
    # ------------------------------------------------------

    if general_params["render_episodes"]:
        episode_vid_queue, episode_renderer = start_episode_renderer()
    else:
        episode_vid_queue = None
        episode_renderer = None

    # ------------------------------------------------------
    # Reward predictor network selection
    # ------------------------------------------------------

    env_id = a2c_params["env_id"]

    if env_id in {
        "MovingDot-v0",
        "MovingDotNoFrameskip-v0",
        "MovingDotDiscreteNoFrameskip-v0",
    }:
        reward_predictor_network = MovingDotRewardNetwork

    elif env_id in {
        "PongNoFrameskip-v4",
        "EnduroNoFrameskip-v4",
    }:
        reward_predictor_network = AtariRewardNetwork

    else:
        raise RuntimeError(f"Unknown reward predictor network for {env_id}")

    # ------------------------------------------------------

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

    # ======================================================
    # Modes
    # ======================================================

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

        if a2c_proc:
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

        if a2c_proc:
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


# ==========================================================
# Environment creation
# ==========================================================

def make_envs(env_id, n_envs, seed):

    def wrap_make_env(env_id, rank):

        def thunk():
            return make_env(env_id, seed + rank)

        return thunk

    set_global_seeds(seed)

    env = DummyVecEnv(
        [wrap_make_env(env_id, i) for i in range(n_envs)]
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

    if env_id in {
        "MovingDot-v0",
        "MovingDotNoFrameskip-v0",
        "MovingDotDiscreteNoFrameskip-v0",
    }:
        policy_fn = MlpPolicy

    elif env_id in {
        "PongNoFrameskip-v4",
        "EnduroNoFrameskip-v4",
    }:
        policy_fn = CnnPolicy

    else:
        raise RuntimeError(f"Unknown policy network for {env_id}")

    env = make_envs(
        a2c_params["env_id"],
        a2c_params["n_envs"],
        a2c_params["seed"],
    )

    del a2c_params["env_id"]
    del a2c_params["n_envs"]

    ckpt_dir = osp.join(log_dir, "policy_checkpoints")

    os.makedirs(ckpt_dir, exist_ok=True)

    reward_predictor = (
        make_reward_predictor("a2c2", cluster_dict)
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

    return env, None


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

    proc = Process(target=f)

    proc.start()

    return pi, proc


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