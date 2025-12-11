# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
Action and observation space definitions for RL training.

Provides utilities to convert between structured LemonadeAction/LemonadeObservation
and flat numpy arrays suitable for RL algorithms.
"""

import numpy as np

from ...models import (
    LemonadeAction,
    LemonadeObservation,
    Weather,
    Location,
    StandUpgrade,
    LOCATION_CATALOG,
    UPGRADE_CATALOG,
    BULK_PRICING,
)


# Weather encoding order (for one-hot)
WEATHER_ORDER = [Weather.SUNNY, Weather.HOT, Weather.CLOUDY, Weather.RAINY, Weather.STORMY]

# Location encoding order (for one-hot)  
LOCATION_ORDER = [Location.PARK, Location.DOWNTOWN, Location.MALL, Location.POOL]

# Upgrade encoding order (for binary flags)
UPGRADE_ORDER = [StandUpgrade.COOLER]


# =============================================================================
# Observation Space Configuration
# =============================================================================

# Total observation dimensions:
# - day_progress: 1
# - weather (one-hot): 5
# - weather_forecast (one-hot): 5
# - temperature_norm: 1
# - cash_norm: 1
# - lemons_norm: 1
# - sugar_norm: 1
# - cups_norm: 1
# - ice_norm: 1
# - lemons_expiring_norm: 1
# - ice_expiring_norm: 1
# - reputation: 1
# - customer_satisfaction: 1
# - days_remaining_norm: 1
# - location (one-hot): 4
# - owned_upgrades (binary): 1
# Total: 27

OBSERVATION_DIM = 27


def encode_observation(obs: LemonadeObservation) -> np.ndarray:
    """
    Convert a LemonadeObservation to a flat numpy array.
    
    All values are normalized to approximately [0, 1] range for training stability.
    
    Args:
        obs: The observation from the environment
        
    Returns:
        Flat numpy array of shape (OBSERVATION_DIM,)
    """
    features = []
    
    # Day progress [0, 1]
    # Assume 14-day season, normalize day to [0, 1]
    day_progress = (obs.day - 1) / 14.0
    features.append(np.clip(day_progress, 0, 1))
    
    # Weather one-hot encoding (5 dims)
    weather_onehot = np.zeros(len(WEATHER_ORDER), dtype=np.float32)
    try:
        weather = Weather(obs.weather)
        idx = WEATHER_ORDER.index(weather)
        weather_onehot[idx] = 1.0
    except (ValueError, IndexError):
        weather_onehot[0] = 1.0  # Default to sunny
    features.extend(weather_onehot)
    
    # Weather forecast one-hot encoding (5 dims)
    forecast_onehot = np.zeros(len(WEATHER_ORDER), dtype=np.float32)
    try:
        forecast = Weather(obs.weather_forecast)
        idx = WEATHER_ORDER.index(forecast)
        forecast_onehot[idx] = 1.0
    except (ValueError, IndexError):
        forecast_onehot[0] = 1.0  # Default to sunny
    features.extend(forecast_onehot)
    
    # Temperature normalized [0, 1] (assume range 50-105F)
    temp_norm = (obs.temperature - 50) / 55.0
    features.append(np.clip(temp_norm, 0, 1))
    
    # Cash normalized [0, 1] (assume max $100 = 10000 cents)
    cash_norm = obs.cash / 10000.0
    features.append(np.clip(cash_norm, 0, 1))
    
    # Inventory normalized [0, 1]
    # Lemons (assume max 200)
    features.append(np.clip(obs.lemons / 200.0, 0, 1))
    # Sugar bags (assume max 50)
    features.append(np.clip(obs.sugar_bags / 50.0, 0, 1))
    # Cups (assume max 500)
    features.append(np.clip(obs.cups_available / 500.0, 0, 1))
    # Ice bags (assume max 50)
    features.append(np.clip(obs.ice_bags / 50.0, 0, 1))
    
    # Expiring inventory normalized
    features.append(np.clip(obs.lemons_expiring_tomorrow / 50.0, 0, 1))
    features.append(np.clip(obs.ice_expiring_tomorrow / 20.0, 0, 1))
    
    # Reputation [0, 1] - already in range
    features.append(np.clip(obs.reputation, 0, 1))
    
    # Customer satisfaction [0, 1] - already in range
    features.append(np.clip(obs.customer_satisfaction, 0, 1))
    
    # Days remaining normalized [0, 1]
    features.append(np.clip(obs.days_remaining / 14.0, 0, 1))
    
    # Location one-hot encoding (4 dims)
    location_onehot = np.zeros(len(LOCATION_ORDER), dtype=np.float32)
    try:
        location = Location(obs.current_location)
        idx = LOCATION_ORDER.index(location)
        location_onehot[idx] = 1.0
    except (ValueError, IndexError):
        location_onehot[0] = 1.0  # Default to park
    features.extend(location_onehot)
    
    # Owned upgrades binary flags (1 dim for now - just cooler)
    has_cooler = 1.0 if StandUpgrade.COOLER.value in obs.owned_upgrades else 0.0
    features.append(has_cooler)
    
    return np.array(features, dtype=np.float32)


def get_observation_labels() -> list[str]:
    """Get human-readable labels for each observation dimension."""
    labels = ["day_progress"]
    labels.extend([f"weather_{w.value}" for w in WEATHER_ORDER])
    labels.extend([f"forecast_{w.value}" for w in WEATHER_ORDER])
    labels.extend([
        "temperature_norm",
        "cash_norm", 
        "lemons_norm",
        "sugar_norm",
        "cups_norm",
        "ice_norm",
        "lemons_expiring_norm",
        "ice_expiring_norm",
        "reputation",
        "customer_satisfaction",
        "days_remaining_norm",
    ])
    labels.extend([f"location_{loc.value}" for loc in LOCATION_ORDER])
    labels.extend([f"has_{u.value}" for u in UPGRADE_ORDER])
    return labels


# =============================================================================
# Action Space Configuration
# =============================================================================

# Flat Action Space (default, compatible with all algorithms):
# - price: 1 (continuous [0, 1] -> [25, 200] cents)
# - lemons_qty: 1 (continuous [0, 1] -> [0, 50] units, converted to tier+count)
# - sugar_qty: 1 (continuous [0, 1] -> [0, 20] bags, converted to tier+count)
# - cups_qty: 1 (continuous [0, 1] -> [0, 100] cups, converted to tier+count)
# - ice_qty: 1 (continuous [0, 1] -> [0, 30] bags, converted to tier+count)
# - advertising: 1 (continuous [0, 1] -> [0, 500] cents)
# - location: 1 (continuous [0, 1] -> discrete location choice)
# - buy_upgrade: 1 (continuous [0, 1] -> threshold for buying upgrade)
# Total: 8
# Note: Supply quantities are auto-converted to optimal tier+count via quantity_to_tier_count()

ACTION_DIM = 8

# Mixed Action Space (more natural representation):
# Continuous actions:
#   - price: 1 (continuous [0, 1] -> [25, 200] cents)
#   - lemons_qty: 1 (continuous [0, 1] -> [0, 50] units)
#   - sugar_qty: 1 (continuous [0, 1] -> [0, 20] bags)
#   - cups_qty: 1 (continuous [0, 1] -> [0, 100] cups)
#   - ice_qty: 1 (continuous [0, 1] -> [0, 30] bags)
#   - advertising: 1 (continuous [0, 1] -> [0, 500] cents)
# Discrete actions:
#   - location: 4 options (park, downtown, mall, pool)
#   - buy_upgrade: 2 options (no, yes)
# Note: Quantities auto-converted to optimal tier+count

MIXED_ACTION_CONTINUOUS_DIM = 6
MIXED_ACTION_LOCATION_DIM = 4
MIXED_ACTION_UPGRADE_DIM = 2

# Action bounds for mapping
ACTION_BOUNDS = {
    "price_min": 25,
    "price_max": 200,
    "lemons_max": 50,
    "sugar_max": 20,
    "cups_max": 100,
    "ice_max": 30,
    "advertising_max": 500,
}


def quantity_to_tier_count(supply_type: str, target_qty: int) -> tuple[int, int]:
    """
    Convert a target quantity to the best tier+count combination.
    
    Uses the smallest tier that achieves at least the target quantity.
    
    Args:
        supply_type: One of "lemons", "sugar", "cups", "ice"
        target_qty: Target quantity to purchase
        
    Returns:
        Tuple of (tier, count) where tier is 1-3
    """
    if target_qty <= 0:
        return (1, 0)
    
    pricing = BULK_PRICING.get(supply_type)
    if not pricing:
        return (1, 0)
    
    # Find the tier that best matches the target quantity
    # Prefer tier 1 for small quantities, higher tiers for larger
    for tier_idx, tier in enumerate(pricing.tiers):
        tier_num = tier_idx + 1
        tier_qty = tier.quantity
        
        # If this tier's quantity is larger than or equal to target, use it
        if tier_qty >= target_qty:
            count = 1  # Buy one of this tier
            return (tier_num, count)
        
        # Calculate how many of this tier we need
        count_needed = (target_qty + tier_qty - 1) // tier_qty  # Ceiling division
        
        # If this tier with count would overshoot by less than next tier, use it
        if tier_idx == len(pricing.tiers) - 1:
            # Last tier, use it
            return (tier_num, count_needed)
    
    return (1, target_qty)  # Fallback


def tier_count_to_quantity(supply_type: str, tier: int, count: int) -> int:
    """Convert tier+count to actual quantity."""
    pricing = BULK_PRICING.get(supply_type)
    if not pricing or count <= 0:
        return 0
    tier_idx = min(tier - 1, len(pricing.tiers) - 1)
    return pricing.tiers[tier_idx].quantity * count


def decode_action(action: np.ndarray, current_upgrades: list[str] = None) -> LemonadeAction:
    """
    Convert a flat numpy array action to a LemonadeAction.
    
    Args:
        action: Numpy array of shape (ACTION_DIM,) with values in [0, 1]
        current_upgrades: List of currently owned upgrade IDs
        
    Returns:
        LemonadeAction instance
    """
    current_upgrades = current_upgrades or []
    
    # Clip action values to [0, 1]
    action = np.clip(action, 0, 1)
    
    # Price: map [0, 1] to [25, 200] cents
    price = int(action[0] * (ACTION_BOUNDS["price_max"] - ACTION_BOUNDS["price_min"]) 
                + ACTION_BOUNDS["price_min"])
    
    # Purchases: map [0, 1] to target quantities, then convert to tier+count
    target_lemons = int(action[1] * ACTION_BOUNDS["lemons_max"])
    target_sugar = int(action[2] * ACTION_BOUNDS["sugar_max"])
    target_cups = int(action[3] * ACTION_BOUNDS["cups_max"])
    target_ice = int(action[4] * ACTION_BOUNDS["ice_max"])
    
    lemons_tier, lemons_count = quantity_to_tier_count("lemons", target_lemons)
    sugar_tier, sugar_count = quantity_to_tier_count("sugar", target_sugar)
    cups_tier, cups_count = quantity_to_tier_count("cups", target_cups)
    ice_tier, ice_count = quantity_to_tier_count("ice", target_ice)
    
    # Advertising: map [0, 1] to [0, 500] cents
    advertising = int(action[5] * ACTION_BOUNDS["advertising_max"])
    
    # Location: use continuous value to select from 4 locations
    # [0, 0.25) -> park, [0.25, 0.5) -> downtown, [0.5, 0.75) -> mall, [0.75, 1] -> pool
    location_idx = min(int(action[6] * len(LOCATION_ORDER)), len(LOCATION_ORDER) - 1)
    location = LOCATION_ORDER[location_idx].value
    
    # Upgrade: threshold at 0.5 to decide whether to buy
    # Only buy if not already owned
    buy_upgrade = None
    if action[7] > 0.5:
        # Try to buy cooler if not owned
        if StandUpgrade.COOLER.value not in current_upgrades:
            buy_upgrade = StandUpgrade.COOLER.value
    
    return LemonadeAction(
        price_per_cup=price,
        lemons_tier=lemons_tier,
        lemons_count=lemons_count,
        sugar_tier=sugar_tier,
        sugar_count=sugar_count,
        cups_tier=cups_tier,
        cups_count=cups_count,
        ice_tier=ice_tier,
        ice_count=ice_count,
        advertising_spend=advertising,
        location=location,
        buy_upgrade=buy_upgrade,
    )


def decode_mixed_action(
    continuous_action: np.ndarray,
    location_action: int,
    upgrade_action: int,
    current_upgrades: list[str] = None,
) -> LemonadeAction:
    """
    Convert a mixed (continuous + discrete) action to a LemonadeAction.
    
    This is the natural representation where location and upgrade are discrete choices,
    while price, quantities, and advertising are continuous.
    
    Args:
        continuous_action: Numpy array of shape (6,) with values in [0, 1]
            [price, lemons_qty, sugar_qty, cups_qty, ice_qty, advertising]
        location_action: Integer index 0-3 for location choice
        upgrade_action: Integer 0 (no) or 1 (yes) for buying upgrade
        current_upgrades: List of currently owned upgrade IDs
        
    Returns:
        LemonadeAction instance
    """
    current_upgrades = current_upgrades or []
    
    # Clip continuous action values to [0, 1]
    continuous_action = np.clip(continuous_action, 0, 1)
    
    # Price: map [0, 1] to [25, 200] cents
    price = int(continuous_action[0] * (ACTION_BOUNDS["price_max"] - ACTION_BOUNDS["price_min"]) 
                + ACTION_BOUNDS["price_min"])
    
    # Purchases: map [0, 1] to target quantities, then convert to tier+count
    target_lemons = int(continuous_action[1] * ACTION_BOUNDS["lemons_max"])
    target_sugar = int(continuous_action[2] * ACTION_BOUNDS["sugar_max"])
    target_cups = int(continuous_action[3] * ACTION_BOUNDS["cups_max"])
    target_ice = int(continuous_action[4] * ACTION_BOUNDS["ice_max"])
    
    lemons_tier, lemons_count = quantity_to_tier_count("lemons", target_lemons)
    sugar_tier, sugar_count = quantity_to_tier_count("sugar", target_sugar)
    cups_tier, cups_count = quantity_to_tier_count("cups", target_cups)
    ice_tier, ice_count = quantity_to_tier_count("ice", target_ice)
    
    # Advertising: map [0, 1] to [0, 500] cents
    advertising = int(continuous_action[5] * ACTION_BOUNDS["advertising_max"])
    
    # Location: direct index selection
    location_idx = min(max(int(location_action), 0), len(LOCATION_ORDER) - 1)
    location = LOCATION_ORDER[location_idx].value
    
    # Upgrade: discrete yes/no decision
    buy_upgrade = None
    if upgrade_action == 1:
        # Try to buy cooler if not owned
        if StandUpgrade.COOLER.value not in current_upgrades:
            buy_upgrade = StandUpgrade.COOLER.value
    
    return LemonadeAction(
        price_per_cup=price,
        lemons_tier=lemons_tier,
        lemons_count=lemons_count,
        sugar_tier=sugar_tier,
        sugar_count=sugar_count,
        cups_tier=cups_tier,
        cups_count=cups_count,
        ice_tier=ice_tier,
        ice_count=ice_count,
        advertising_spend=advertising,
        location=location,
        buy_upgrade=buy_upgrade,
    )


def encode_action(action: LemonadeAction) -> np.ndarray:
    """
    Convert a LemonadeAction to a flat numpy array.
    
    Useful for imitation learning or behavior cloning from expert demonstrations.
    
    Args:
        action: LemonadeAction instance
        
    Returns:
        Numpy array of shape (ACTION_DIM,) with values in [0, 1]
    """
    encoded = np.zeros(ACTION_DIM, dtype=np.float32)
    
    # Price
    encoded[0] = (action.price_per_cup - ACTION_BOUNDS["price_min"]) / \
                 (ACTION_BOUNDS["price_max"] - ACTION_BOUNDS["price_min"])
    
    # Purchases - convert tier+count back to effective quantity
    qty_lemons = tier_count_to_quantity("lemons", action.lemons_tier, action.lemons_count)
    qty_sugar = tier_count_to_quantity("sugar", action.sugar_tier, action.sugar_count)
    qty_cups = tier_count_to_quantity("cups", action.cups_tier, action.cups_count)
    qty_ice = tier_count_to_quantity("ice", action.ice_tier, action.ice_count)
    
    encoded[1] = qty_lemons / ACTION_BOUNDS["lemons_max"]
    encoded[2] = qty_sugar / ACTION_BOUNDS["sugar_max"]
    encoded[3] = qty_cups / ACTION_BOUNDS["cups_max"]
    encoded[4] = qty_ice / ACTION_BOUNDS["ice_max"]
    
    # Advertising
    encoded[5] = action.advertising_spend / ACTION_BOUNDS["advertising_max"]
    
    # Location
    try:
        location = Location(action.location) if action.location else Location.PARK
        encoded[6] = LOCATION_ORDER.index(location) / (len(LOCATION_ORDER) - 1)
    except (ValueError, IndexError):
        encoded[6] = 0.0
    
    # Upgrade
    encoded[7] = 1.0 if action.buy_upgrade else 0.0
    
    return np.clip(encoded, 0, 1)


def get_action_labels() -> list[str]:
    """Get human-readable labels for each action dimension."""
    return [
        "price",
        "lemons_qty",
        "sugar_qty", 
        "cups_qty",
        "ice_qty",
        "advertising",
        "location",
        "buy_upgrade",
    ]
