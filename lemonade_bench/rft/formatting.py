# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
Data formatting utilities for RFT training.

Converts trajectory data into formats suitable for different
fine-tuning objectives (SFT, GRPO, DPO).

Includes importance-weighted SFT (iw-SFT) which weights samples by
episode profit, optimizing a tighter bound on the RL objective.

Reference: "Supervised Fine Tuning on Curated Data is Reinforcement Learning"
https://arxiv.org/abs/2507.12856
"""

import json
import math
from pathlib import Path
from typing import Any

from .data_collector import Trajectory, TrajectoryTurn


# System prompt for the fine-tuned model
LEMONADE_SYSTEM_PROMPT = """You are an expert AI agent running a lemonade stand business. Your goal is to maximize profit over a 14-day summer season.

You make decisions based on weather conditions, inventory levels, and market dynamics. You use the take_action tool to submit your daily business decisions.

Key strategies:
- Price higher on hot/sunny days when demand is high
- Stock up on inventory before good weather
- Avoid over-buying perishables (lemons expire in 3 days, ice melts daily)
- Build reputation by serving customers well (don't run out of supplies)"""


def format_action_as_tool_call(action: dict[str, Any], reasoning: str) -> str:
    """Format an action as a tool call JSON for training."""
    tool_input = {
        "reasoning": reasoning,
        "price_per_cup": action["price_per_cup"],
        "lemons_tier": action.get("lemons_tier", 1),
        "lemons_count": action.get("lemons_count", 0),
        "sugar_tier": action.get("sugar_tier", 1),
        "sugar_count": action.get("sugar_count", 0),
        "cups_tier": action.get("cups_tier", 1),
        "cups_count": action.get("cups_count", 0),
        "ice_tier": action.get("ice_tier", 1),
        "ice_count": action.get("ice_count", 0),
        "advertising_spend": action["advertising_spend"],
    }
    
    if action.get("buy_upgrade"):
        tool_input["buy_upgrade"] = action["buy_upgrade"]
    if action.get("location"):
        tool_input["location"] = action["location"]
    
    return json.dumps(tool_input, indent=2)


def trajectory_to_sft_samples(
    trajectory: Trajectory,
    include_system: bool = True,
) -> list[dict[str, Any]]:
    """
    Convert a trajectory to SFT (Supervised Fine-Tuning) samples.
    
    Each turn becomes a separate training sample with the observation
    as input and the action (with reasoning) as output.
    
    Args:
        trajectory: Trajectory to convert
        include_system: Whether to include system prompt
        
    Returns:
        List of training samples in chat format
    """
    samples = []
    
    for i, turn in enumerate(trajectory.turns):
        messages = []
        
        # Add system message
        if include_system:
            messages.append({
                "role": "system",
                "content": LEMONADE_SYSTEM_PROMPT,
            })
        
        # Add user message (observation)
        messages.append({
            "role": "user",
            "content": turn.observation_text,
        })
        
        # Add assistant message (tool call with reasoning)
        # Format as the model should output it
        assistant_content = f"""I'll analyze the current situation and make my decision.

{turn.reasoning}

<tool_call>
{{"name": "take_action", "arguments": {format_action_as_tool_call(turn.action, turn.reasoning)}}}
</tool_call>"""
        
        messages.append({
            "role": "assistant",
            "content": assistant_content,
        })
        
        samples.append({
            "id": f"{trajectory.episode_id}_turn_{i}",
            "messages": messages,
            "metadata": {
                "episode_id": trajectory.episode_id,
                "day": turn.day,
                "cups_sold": turn.cups_sold,
                "daily_profit": turn.daily_profit,
                "total_profit": trajectory.total_profit,
            },
        })
    
    return samples


def trajectory_to_conversation(
    trajectory: Trajectory,
    include_system: bool = True,
) -> dict[str, Any]:
    """
    Convert a trajectory to a single multi-turn conversation.
    
    The entire episode becomes one training sample with alternating
    user (observations) and assistant (actions) messages.
    
    Args:
        trajectory: Trajectory to convert
        include_system: Whether to include system prompt
        
    Returns:
        Single training sample with full conversation
    """
    messages = []
    
    # Add system message
    if include_system:
        messages.append({
            "role": "system",
            "content": LEMONADE_SYSTEM_PROMPT,
        })
    
    for turn in trajectory.turns:
        # User message (observation)
        messages.append({
            "role": "user",
            "content": turn.observation_text,
        })
        
        # Assistant message (action)
        assistant_content = f"""{turn.reasoning}

<tool_call>
{{"name": "take_action", "arguments": {format_action_as_tool_call(turn.action, turn.reasoning)}}}
</tool_call>"""
        
        messages.append({
            "role": "assistant",
            "content": assistant_content,
        })
    
    return {
        "id": trajectory.episode_id,
        "messages": messages,
        "metadata": {
            "total_profit": trajectory.total_profit,
            "total_cups_sold": trajectory.total_cups_sold,
            "turn_count": trajectory.turn_count,
            "seed": trajectory.seed,
        },
    }


def compute_importance_weights(
    trajectories: list[Trajectory],
    temperature: float = 1.0,
    normalize: bool = True,
) -> dict[str, float]:
    """
    Compute importance weights for trajectories based on profit.
    
    Implements importance-weighted SFT (iw-SFT) from:
    "Supervised Fine Tuning on Curated Data is Reinforcement Learning"
    https://arxiv.org/abs/2507.12856
    
    Higher profit trajectories get higher weights, making the SFT
    objective a tighter bound on the RL objective.
    
    Args:
        trajectories: List of trajectories
        temperature: Softmax temperature (lower = more extreme weights)
        normalize: Whether to normalize weights to sum to len(trajectories)
        
    Returns:
        Dict mapping episode_id to importance weight
    """
    if not trajectories:
        return {}
    
    # Get profits
    profits = [t.total_profit for t in trajectories]
    
    # Normalize profits to reasonable range for softmax
    min_profit = min(profits)
    max_profit = max(profits)
    profit_range = max_profit - min_profit
    
    if profit_range < 1e-8:
        # All profits are the same - uniform weights
        return {t.episode_id: 1.0 for t in trajectories}
    
    # Compute softmax weights
    normalized_profits = [(p - min_profit) / profit_range for p in profits]
    exp_profits = [math.exp(p / temperature) for p in normalized_profits]
    sum_exp = sum(exp_profits)
    
    weights = {}
    for i, traj in enumerate(trajectories):
        weight = exp_profits[i] / sum_exp
        if normalize:
            # Scale so weights sum to N (like uniform weighting on average)
            weight *= len(trajectories)
        weights[traj.episode_id] = weight
    
    return weights


def create_sft_dataset(
    trajectories: list[Trajectory],
    output_path: str | Path,
    format_type: str = "individual_turns",
    val_split: float = 0.1,
    use_importance_weighting: bool = True,
    importance_temperature: float = 1.0,
) -> tuple[Path, Path]:
    """
    Create SFT dataset files from trajectories.
    
    Supports importance-weighted SFT (iw-SFT) which weights samples
    by episode profit for better RL alignment.
    
    Args:
        trajectories: List of trajectories to convert
        output_path: Base path for output files
        format_type: "individual_turns" or "full_conversation"
        val_split: Fraction of data to use for validation
        use_importance_weighting: Enable iw-SFT profit-based weighting
        importance_temperature: Softmax temperature for weights (lower = more extreme)
        
    Returns:
        Tuple of (train_path, val_path)
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Compute importance weights if enabled
    importance_weights = {}
    if use_importance_weighting:
        importance_weights = compute_importance_weights(
            trajectories,
            temperature=importance_temperature,
        )
        print(f"[iw-SFT] Computed importance weights for {len(trajectories)} trajectories")
        
        # Log weight distribution
        weights = list(importance_weights.values())
        print(f"  Weight range: {min(weights):.3f} - {max(weights):.3f}")
        print(f"  Mean weight: {sum(weights)/len(weights):.3f}")
    
    all_samples = []
    
    for traj in trajectories:
        # Get importance weight for this trajectory
        weight = importance_weights.get(traj.episode_id, 1.0)
        
        if format_type == "individual_turns":
            samples = trajectory_to_sft_samples(traj)
            # Add weight to each sample
            for sample in samples:
                sample["weight"] = weight
                sample["metadata"]["importance_weight"] = weight
            all_samples.extend(samples)
        else:
            sample = trajectory_to_conversation(traj)
            sample["weight"] = weight
            sample["metadata"]["importance_weight"] = weight
            all_samples.append(sample)
    
    # Sort by total profit (descending) to prioritize good examples
    all_samples.sort(
        key=lambda x: x.get("metadata", {}).get("total_profit", 0),
        reverse=True,
    )
    
    # Split into train/val
    split_idx = int(len(all_samples) * (1 - val_split))
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    # Save as JSONL (standard format for fine-tuning)
    train_path = output_path / "train.jsonl"
    val_path = output_path / "val.jsonl"
    
    with open(train_path, "w") as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + "\n")
    
    with open(val_path, "w") as f:
        for sample in val_samples:
            f.write(json.dumps(sample) + "\n")
    
    print(f"Created {len(train_samples)} training samples -> {train_path}")
    print(f"Created {len(val_samples)} validation samples -> {val_path}")
    
    if use_importance_weighting:
        # Save weight statistics
        stats_path = output_path / "weight_stats.json"
        weights = [s["weight"] for s in train_samples]
        with open(stats_path, "w") as f:
            json.dump({
                "num_samples": len(train_samples),
                "min_weight": min(weights),
                "max_weight": max(weights),
                "mean_weight": sum(weights) / len(weights),
                "temperature": importance_temperature,
            }, f, indent=2)
        print(f"Saved weight statistics -> {stats_path}")
    
    return train_path, val_path


def create_grpo_dataset(
    trajectories: list[Trajectory],
    output_path: str | Path,
    num_groups: int = 4,
) -> Path:
    """
    Create GRPO (Group Relative Policy Optimization) dataset.
    
    Groups trajectories by similar initial conditions and uses
    relative performance within groups for preference learning.
    
    Args:
        trajectories: List of trajectories to convert
        output_path: Output file path
        num_groups: Number of trajectories per group
        
    Returns:
        Path to output file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Group trajectories by seed (same initial conditions)
    seed_groups: dict[int, list[Trajectory]] = {}
    for traj in trajectories:
        seed = traj.seed % 100  # Group by base seed
        if seed not in seed_groups:
            seed_groups[seed] = []
        seed_groups[seed].append(traj)
    
    grpo_samples = []
    
    for seed, group in seed_groups.items():
        if len(group) < 2:
            continue
        
        # Sort by profit
        group.sort(key=lambda t: t.total_profit, reverse=True)
        
        # Create preference pairs
        for i, better_traj in enumerate(group[:-1]):
            for worse_traj in group[i + 1:]:
                # For each turn, create a preference sample
                for turn_idx in range(min(len(better_traj.turns), len(worse_traj.turns))):
                    better_turn = better_traj.turns[turn_idx]
                    worse_turn = worse_traj.turns[turn_idx]
                    
                    sample = {
                        "prompt": better_turn.observation_text,
                        "chosen": format_action_as_tool_call(
                            better_turn.action, better_turn.reasoning
                        ),
                        "rejected": format_action_as_tool_call(
                            worse_turn.action, worse_turn.reasoning
                        ),
                        "metadata": {
                            "chosen_profit": better_traj.total_profit,
                            "rejected_profit": worse_traj.total_profit,
                            "seed": seed,
                            "turn": turn_idx,
                        },
                    }
                    grpo_samples.append(sample)
    
    with open(output_path, "w") as f:
        for sample in grpo_samples:
            f.write(json.dumps(sample) + "\n")
    
    print(f"Created {len(grpo_samples)} GRPO samples -> {output_path}")
    
    return output_path

