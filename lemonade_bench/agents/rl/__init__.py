# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
Reinforcement Learning module for LemonadeBench.

Provides Gymnasium-compatible wrappers and utilities for training traditional
RL agents (PPO, DQN, SAC, etc.) on the Lemonade Stand environment.

Quick Start:
    >>> from lemonade_bench.agents.rl import LemonadeGymEnv
    >>> env = LemonadeGymEnv(seed=42)
    >>> obs, info = env.reset()
    >>> action = env.action_space.sample()
    >>> obs, reward, terminated, truncated, info = env.step(action)

With Stable Baselines3:
    >>> from stable_baselines3 import PPO
    >>> from lemonade_bench.agents.rl import LemonadeGymEnv
    >>> env = LemonadeGymEnv()
    >>> model = PPO("MlpPolicy", env, verbose=1)
    >>> model.learn(total_timesteps=100_000)

Vectorized Training:
    >>> from gymnasium.vector import SyncVectorEnv
    >>> from lemonade_bench.agents.rl import make_env
    >>> envs = SyncVectorEnv([lambda i=i: make_env(seed=i) for i in range(8)])
"""

from .gym_wrapper import LemonadeGymEnv, make_env
from .spaces import (
    OBSERVATION_DIM,
    ACTION_DIM,
    ACTION_BOUNDS,
    MIXED_ACTION_CONTINUOUS_DIM,
    MIXED_ACTION_LOCATION_DIM,
    MIXED_ACTION_UPGRADE_DIM,
    encode_observation,
    decode_action,
    decode_mixed_action,
    encode_action,
    get_observation_labels,
    get_action_labels,
)
from .evaluate import (
    EvalResult,
    evaluate_rl_model,
    evaluate_random_agent,
    evaluate_rule_based_agent,
    evaluate_constant_agent,
    compare_agents,
)

__all__ = [
    # Main environment
    "LemonadeGymEnv",
    "make_env",
    # Space utilities - flat action space
    "OBSERVATION_DIM",
    "ACTION_DIM",
    "ACTION_BOUNDS",
    "encode_observation",
    "decode_action",
    "encode_action",
    "get_observation_labels",
    "get_action_labels",
    # Space utilities - mixed action space
    "MIXED_ACTION_CONTINUOUS_DIM",
    "MIXED_ACTION_LOCATION_DIM",
    "MIXED_ACTION_UPGRADE_DIM",
    "decode_mixed_action",
    # Evaluation
    "EvalResult",
    "evaluate_rl_model",
    "evaluate_random_agent",
    "evaluate_rule_based_agent",
    "evaluate_constant_agent",
    "compare_agents",
]
