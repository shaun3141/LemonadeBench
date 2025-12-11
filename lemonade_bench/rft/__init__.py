"""
Reinforcement Fine-Tuning (RFT) module for LemonadeBench.

This module provides tools and utilities for reinforcement fine-tuning
of language models using the Lemonade Stand environment.

Key components:
- config: Configuration dataclasses for training hyperparameters
- data_collector: Trajectory collection from LLM agents
- formatting: Convert trajectories to training formats (SFT, GRPO, DPO)
- train: LoRA fine-tuning with Unsloth or PEFT
- collect: CLI for data collection

Usage:
    # Collect training data
    python -m lemonade_bench.rft.collect --provider anthropic --episodes 100
    
    # Train with LoRA
    python -m lemonade_bench.rft.train --train-data ./rft_trajectories/sft/train.jsonl
"""

from .config import (
    RFTConfig,
    ModelConfig,
    LoRAConfig,
    TrainingConfig,
    DataConfig,
    DEFAULT_CONFIG,
)
from .data_collector import (
    Trajectory,
    TrajectoryTurn,
    collect_trajectories,
    load_trajectories,
    episode_to_trajectory,
)
from .formatting import (
    create_sft_dataset,
    create_grpo_dataset,
    trajectory_to_sft_samples,
    trajectory_to_conversation,
    compute_importance_weights,
)
from .augmentation import (
    AugmentationConfig,
    create_augmented_dataset,
    augment_trajectory,
    augment_action_exploration,
    augment_action_exploitation,
)
from .grpo_trainer import (
    GRPOConfig,
    GRPOTrainer,
    TrajectoryWithReward,
    compute_group_advantages,
    normalize_rewards,
)
from .online_grpo import (
    OnlineGRPOConfig,
    phase1_warmup_sft,
    phase2_online_grpo,
    phase3_evaluation,
)

__all__ = [
    # Config
    "RFTConfig",
    "ModelConfig",
    "LoRAConfig",
    "TrainingConfig",
    "DataConfig",
    "DEFAULT_CONFIG",
    # Data collection
    "Trajectory",
    "TrajectoryTurn",
    "collect_trajectories",
    "load_trajectories",
    "episode_to_trajectory",
    # Formatting
    "create_sft_dataset",
    "create_grpo_dataset",
    "trajectory_to_sft_samples",
    "trajectory_to_conversation",
    "compute_importance_weights",
    # Augmentation (Behavior Injection)
    "AugmentationConfig",
    "create_augmented_dataset",
    "augment_trajectory",
    "augment_action_exploration",
    "augment_action_exploitation",
    # GRPO Training
    "GRPOConfig",
    "GRPOTrainer",
    "TrajectoryWithReward",
    "compute_group_advantages",
    "normalize_rewards",
    # Online GRPO Pipeline
    "OnlineGRPOConfig",
    "phase1_warmup_sft",
    "phase2_online_grpo",
    "phase3_evaluation",
]

