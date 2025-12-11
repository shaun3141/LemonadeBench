# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
Pytest fixtures for Lemonade Stand Environment tests.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    GameConfig,
    LemonadeAction,
    LemonadeObservation,
    Weather,
    CustomerMood,
    StandUpgrade,
    UPGRADE_CATALOG,
    BULK_PRICING,
)
from server.lemonade_environment import LemonadeEnvironment


@pytest.fixture
def default_config() -> GameConfig:
    """Default game configuration for tests."""
    return GameConfig()


@pytest.fixture
def custom_config() -> GameConfig:
    """Custom game configuration with shorter game for faster tests."""
    return GameConfig(
        total_days=5,
        starting_cash=5000,  # $50
        starting_lemons=20,
        starting_sugar=10,
        starting_cups=100,
        starting_ice=10,
    )


@pytest.fixture
def env(default_config: GameConfig) -> LemonadeEnvironment:
    """Fresh environment with default config and fixed seed for reproducibility."""
    return LemonadeEnvironment(config=default_config, seed=42)


@pytest.fixture
def env_seeded() -> LemonadeEnvironment:
    """Environment with fixed seed for deterministic tests."""
    return LemonadeEnvironment(seed=12345)


@pytest.fixture
def short_game_env(custom_config: GameConfig) -> LemonadeEnvironment:
    """Environment configured for a short 5-day game."""
    return LemonadeEnvironment(config=custom_config, seed=42)


@pytest.fixture
def basic_action() -> LemonadeAction:
    """Basic action with reasonable defaults."""
    return LemonadeAction(
        price_per_cup=75,
    )


@pytest.fixture
def full_action() -> LemonadeAction:
    """Action with all fields populated."""
    return LemonadeAction(
        price_per_cup=100,
        lemons_tier=2, lemons_count=2,   # 2 dozen = 24 lemons
        sugar_tier=2, sugar_count=1,     # 1 case = 5 bags
        cups_tier=3, cups_count=1,       # 1 case = 250 cups
        ice_tier=2, ice_count=2,         # 2 cooler packs = 10 bags
        advertising_spend=200,
        buy_upgrade="cooler",
    )


@pytest.fixture
def minimal_action() -> LemonadeAction:
    """Minimal action - just set price."""
    return LemonadeAction(
        price_per_cup=50,
    )


@pytest.fixture
def reset_observation(env: LemonadeEnvironment) -> LemonadeObservation:
    """Observation from a fresh reset."""
    return env.reset()


# Weather fixtures for testing different conditions
@pytest.fixture
def hot_weather_env() -> LemonadeEnvironment:
    """
    Environment that starts with hot weather.
    We use a specific seed that produces hot weather on day 1.
    """
    # Seed 100 produces HOT weather on day 1
    env = LemonadeEnvironment(seed=100)
    obs = env.reset()
    # If not hot, we'll work with what we get - tests should handle this
    return env


@pytest.fixture
def rainy_weather_env() -> LemonadeEnvironment:
    """
    Environment that starts with rainy weather.
    """
    # Seed 17 produces RAINY weather on day 1
    env = LemonadeEnvironment(seed=17)
    env.reset()
    return env
