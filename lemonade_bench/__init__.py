# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
LemonadeBench - A Lemonade Stand Tycoon Environment for OpenEnv.

A classic lemonade stand simulation where RL agents learn to:
- Set optimal prices based on weather conditions
- Manage inventory and supplies efficiently
- Build customer reputation over time
- Maximize profit over a simulated summer season

Quick Start:
    # Run an evaluation with the CLI
    lemonade run --model claude-sonnet-4-20250514 --seed 42
    
    # Or use the Python API
    from lemonade_bench.agents import LLMAgent
    from lemonade_bench.agents.providers import AnthropicProvider
    from lemonade_bench.server.lemonade_environment import LemonadeEnvironment
    
    provider = AnthropicProvider(model="claude-sonnet-4-20250514")
    agent = LLMAgent(provider)
    env = LemonadeEnvironment(seed=42)
    result = agent.run_episode(env)
"""

from .client import LemonadeEnv
from .models import LemonadeAction, LemonadeObservation, GameConfig, Weather
from .db import SupabaseLogger, SupabaseReader

__all__ = [
    # Core models
    "LemonadeAction",
    "LemonadeObservation",
    "LemonadeEnv",
    "GameConfig",
    "Weather",
    # Database
    "SupabaseLogger",
    "SupabaseReader",
]

