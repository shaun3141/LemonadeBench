# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
Behavior Injection for RFT Warmup.

Implements trajectory augmentation techniques from recent research to prepare
LLMs for more effective reinforcement learning.

Key technique: Behavior Injection (Cen et al., 2025)
- Enriches SFT data with exploratory and exploitative behaviors
- Makes models more receptive to RL fine-tuning
- Significantly increases performance gains from RFT

Reference: https://arxiv.org/abs/2505.18917
"""

import copy
import random
from dataclasses import dataclass
from typing import Any

from .data_collector import Trajectory, TrajectoryTurn


@dataclass
class AugmentationConfig:
    """Configuration for behavior injection augmentation."""
    
    # Exploration augmentation
    enable_exploration: bool = True
    exploration_ratio: float = 0.3  # Fraction of trajectories to augment with exploration
    price_noise_range: tuple[int, int] = (-25, 50)  # Cents to add/subtract from price
    inventory_noise_range: tuple[int, int] = (-5, 10)  # Items to add/subtract
    
    # Exploitation augmentation  
    enable_exploitation: bool = True
    exploitation_ratio: float = 0.2  # Fraction to augment with optimal-like behaviors
    
    # Reasoning augmentation
    enable_reasoning_scaffolds: bool = True
    reasoning_templates: list[str] | None = None


# Default reasoning scaffolds to inject during training
# These help the model internalize step-by-step decision making
DEFAULT_REASONING_SCAFFOLDS = [
    "Let me think through this step by step.",
    "First, I'll analyze the weather and forecast.",
    "I need to calculate the expected demand.",
    "Let me check my current inventory levels.",
    "Based on these factors, I'll set my strategy.",
]


def augment_action_exploration(
    action: dict[str, Any],
    config: AugmentationConfig,
) -> dict[str, Any]:
    """
    Add exploratory noise to an action.
    
    Exploration helps the model learn from varied strategies,
    not just the expert's exact choices.
    """
    augmented = copy.deepcopy(action)
    
    # Perturb price (within reasonable bounds)
    price_noise = random.randint(*config.price_noise_range)
    new_price = max(25, min(300, augmented["price_per_cup"] + price_noise))
    augmented["price_per_cup"] = new_price
    
    # Perturb inventory purchase counts (tier-based)
    for count_key in ["lemons_count", "sugar_count", "cups_count", "ice_count"]:
        if augmented.get(count_key, 0) > 0 or random.random() < 0.3:
            noise = random.randint(*config.inventory_noise_range)
            # Noise at count level (small adjustments)
            count_noise = noise // 5 if noise > 0 else noise // 5
            new_val = max(0, augmented.get(count_key, 0) + count_noise)
            augmented[count_key] = new_val
    
    return augmented


def augment_action_exploitation(
    action: dict[str, Any],
    weather: str,
    temperature: int,
) -> dict[str, Any]:
    """
    Augment action with optimal-like exploitative behavior.
    
    Injects knowledge of good strategies for known conditions,
    helping the model learn to exploit favorable situations.
    """
    augmented = copy.deepcopy(action)
    
    weather_lower = weather.lower()
    
    # Optimal pricing by weather (based on game mechanics)
    optimal_prices = {
        "hot": random.randint(90, 125),      # High demand, premium prices
        "sunny": random.randint(75, 100),    # Good demand
        "cloudy": random.randint(50, 75),    # Moderate demand
        "rainy": random.randint(35, 55),     # Low demand, discount
        "stormy": random.randint(25, 40),    # Very low demand
    }
    
    if weather_lower in optimal_prices:
        augmented["price_per_cup"] = optimal_prices[weather_lower]
    
    # Temperature adjustment
    if temperature > 90:
        augmented["price_per_cup"] = min(150, augmented["price_per_cup"] + 15)
    elif temperature < 60:
        augmented["price_per_cup"] = max(25, augmented["price_per_cup"] - 10)
    
    # Optimal inventory for weather (tier-based)
    if weather_lower in ["hot", "sunny"]:
        # Stock up for high demand - use tier 2 (dozen lemons, cooler pack ice)
        augmented["lemons_tier"] = 2
        augmented["lemons_count"] = max(augmented.get("lemons_count", 0), 1)
        augmented["ice_tier"] = 2
        augmented["ice_count"] = max(augmented.get("ice_count", 0), 1)
    elif weather_lower in ["rainy", "stormy"]:
        # Minimize inventory to avoid spoilage
        augmented["lemons_tier"] = 1
        augmented["lemons_count"] = min(augmented.get("lemons_count", 0), 3)
        augmented["ice_count"] = 0
    
    return augmented


def augment_reasoning(
    reasoning: str,
    config: AugmentationConfig,
) -> str:
    """
    Augment reasoning with scaffolding templates.
    
    Prior Prompt Engineering (Taveekitworachai et al., 2025) shows that
    reasoning scaffolds during training help models internalize behaviors.
    """
    templates = config.reasoning_templates or DEFAULT_REASONING_SCAFFOLDS
    
    # Randomly select a scaffold to prepend
    scaffold = random.choice(templates)
    
    # Combine scaffold with original reasoning
    if reasoning and not reasoning.startswith(scaffold):
        return f"{scaffold} {reasoning}"
    
    return reasoning


def augment_trajectory(
    trajectory: Trajectory,
    config: AugmentationConfig,
) -> Trajectory:
    """
    Apply behavior injection augmentation to a trajectory.
    
    Creates a modified copy with exploratory or exploitative behaviors
    to prepare the model for RL fine-tuning.
    """
    augmented_turns = []
    
    # Decide augmentation type for this trajectory
    rand = random.random()
    augment_type = "none"
    
    if rand < config.exploration_ratio:
        augment_type = "exploration"
    elif rand < config.exploration_ratio + config.exploitation_ratio:
        augment_type = "exploitation"
    
    for turn in trajectory.turns:
        new_turn = TrajectoryTurn(
            day=turn.day,
            observation_text=turn.observation_text,
            action=copy.deepcopy(turn.action),
            reasoning=turn.reasoning,
            cups_sold=turn.cups_sold,
            daily_profit=turn.daily_profit,
            daily_revenue=turn.daily_revenue,
            daily_costs=turn.daily_costs,
        )
        
        # Apply action augmentation
        if augment_type == "exploration":
            new_turn.action = augment_action_exploration(turn.action, config)
        elif augment_type == "exploitation":
            # Extract weather info from observation text
            weather = _extract_weather(turn.observation_text)
            temp = _extract_temperature(turn.observation_text)
            new_turn.action = augment_action_exploitation(turn.action, weather, temp)
        
        # Apply reasoning scaffolding
        if config.enable_reasoning_scaffolds and random.random() < 0.5:
            new_turn.reasoning = augment_reasoning(turn.reasoning, config)
        
        augmented_turns.append(new_turn)
    
    # Create augmented trajectory
    return Trajectory(
        episode_id=f"{trajectory.episode_id}_aug_{augment_type}",
        seed=trajectory.seed,
        total_profit=trajectory.total_profit,  # Keep original profit for weighting
        total_cups_sold=trajectory.total_cups_sold,
        final_cash=trajectory.final_cash,
        final_reputation=trajectory.final_reputation,
        turn_count=trajectory.turn_count,
        turns=augmented_turns,
        source_model=f"{trajectory.source_model}_augmented",
        source_provider=trajectory.source_provider,
        collected_at=trajectory.collected_at,
    )


def _extract_weather(observation_text: str) -> str:
    """Extract weather from observation text."""
    weather_types = ["hot", "sunny", "cloudy", "rainy", "stormy"]
    text_lower = observation_text.lower()
    
    for weather in weather_types:
        if weather in text_lower:
            return weather
    
    return "sunny"  # Default


def _extract_temperature(observation_text: str) -> int:
    """Extract temperature from observation text."""
    import re
    
    # Look for temperature pattern like "85°F" or "(85°F)"
    match = re.search(r'(\d+)°F', observation_text)
    if match:
        return int(match.group(1))
    
    return 75  # Default


def create_augmented_dataset(
    trajectories: list[Trajectory],
    config: AugmentationConfig | None = None,
    augmentation_factor: float = 1.5,
) -> list[Trajectory]:
    """
    Create an augmented dataset with behavior injection.
    
    Args:
        trajectories: Original trajectories from expert model
        config: Augmentation configuration
        augmentation_factor: Multiplier for dataset size (1.5 = 50% more data)
        
    Returns:
        List containing original + augmented trajectories
    """
    if config is None:
        config = AugmentationConfig()
    
    # Start with original trajectories
    augmented = list(trajectories)
    
    # Calculate how many augmented samples to add
    num_to_add = int(len(trajectories) * (augmentation_factor - 1.0))
    
    # Randomly sample and augment
    for _ in range(num_to_add):
        source = random.choice(trajectories)
        aug_traj = augment_trajectory(source, config)
        augmented.append(aug_traj)
    
    # Shuffle to mix original and augmented
    random.shuffle(augmented)
    
    return augmented

