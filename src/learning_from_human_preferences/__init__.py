"""
learning_from_human_preferences
================================
PyTorch implementation of Deep RL from Human Preferences (Christiano et al., 2017).

Public API
----------
Policy training:
    from learning_from_human_preferences.agents.a2c.a2c import learn
    from learning_from_human_preferences.agents.a2c.policies import CnnPolicy, MlpPolicy

Reward predictor:
    from learning_from_human_preferences.reward_model.reward_predictor import RewardPredictorEnsemble
    from learning_from_human_preferences.reward_model.reward_predictor_core_network import (
        AtariRewardNetwork, MovingDotRewardNetwork
    )

Preference database:
    from learning_from_human_preferences.preferences.pref_db import PrefDB, PrefBuffer, Segment

Preference interface:
    from learning_from_human_preferences.preferences.pref_interface import PrefInterface

Environment utilities:
    from learning_from_human_preferences.envs.utils import make_env, RunningStat
"""

__version__ = "0.1.0"
