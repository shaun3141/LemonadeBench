# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
GRPO (Group Relative Policy Optimization) trainer for LemonadeBench.

Implements GRPO training using environment profit as the reward signal.
Groups trajectories by seed and computes relative advantages within groups.

Reference:
    GRPO: Group Relative Policy Optimization
    https://arxiv.org/abs/2402.03300
"""

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .data_collector import Trajectory, TrajectoryTurn


@dataclass
class GRPOConfig:
    """Configuration for GRPO training.
    
    Key parameters (from Tinker research):
    - batch_size (num_seeds): Number of unique environment seeds
    - group_size: Number of rollouts per seed for GRPO grouping
    - LR should scale as LR ∝ √batch_size when changing batch size
    - KL divergence should stay below 0.01 for stable training
    - num_substeps: Multiple gradient updates per batch (start with 2-4)
    
    See: https://tinker-docs.thinkingmachines.ai/rl/rl-hyperparams
    See: https://tinker-docs.thinkingmachines.ai/lora-primer
    """
    
    # Group parameters
    group_size: int = 8  # Number of rollouts per seed/prompt group
    num_seeds: int = 8   # Number of different seeds per iteration (batch_size)
    
    # Training parameters
    # LR ≈ 5e-4 for Qwen ~30B with LoRA (see sl-hyperparams)
    # Scale as LR ∝ √batch_size if changing num_seeds
    learning_rate: float = 2e-4
    kl_coef: float = 0.1  # KL divergence penalty coefficient
    clip_range: float = 0.2  # PPO-style clipping (optional)
    max_grad_norm: float = 1.0
    
    # Multiple updates per sampling iteration (like PPO/GRPO)
    # Higher values = more sample efficiency but risk off-policy issues
    num_substeps: int = 2  # Recommended: 2-4 per Tinker research
    
    # KL divergence threshold for training stability
    # Training is stable when KL < 0.01; higher indicates instability
    max_kl_divergence: float = 0.01
    
    # Reward normalization
    normalize_rewards: bool = True
    reward_baseline: str = "group_mean"  # "group_mean", "group_min", "none"
    
    # Advantage computation
    advantage_normalization: bool = True
    
    # Batch parameters
    mini_batch_size: int = 4
    gradient_accumulation_steps: int = 2


@dataclass
class TrajectoryWithReward:
    """Trajectory with computed reward and log probabilities."""
    trajectory: Trajectory
    reward: float  # Normalized reward (usually profit)
    seed: int
    log_probs: list[float] = field(default_factory=list)  # Log prob per action
    ref_log_probs: list[float] = field(default_factory=list)  # Reference model log probs
    advantage: float = 0.0


def compute_group_advantages(
    trajectories: list[TrajectoryWithReward],
    baseline: str = "group_mean",
    normalize: bool = True,
) -> list[TrajectoryWithReward]:
    """
    Compute advantages for trajectories within groups.
    
    Groups trajectories by seed and computes relative advantages
    based on the reward difference from the group baseline.
    
    Args:
        trajectories: List of trajectories with rewards
        baseline: How to compute baseline ("group_mean", "group_min", "none")
        normalize: Whether to normalize advantages
        
    Returns:
        Trajectories with computed advantages
    """
    # Group by seed
    groups: dict[int, list[TrajectoryWithReward]] = {}
    for traj in trajectories:
        if traj.seed not in groups:
            groups[traj.seed] = []
        groups[traj.seed].append(traj)
    
    # Compute advantages within each group
    all_advantages = []
    
    for seed, group in groups.items():
        rewards = [t.reward for t in group]
        
        if baseline == "group_mean":
            baseline_value = sum(rewards) / len(rewards)
        elif baseline == "group_min":
            baseline_value = min(rewards)
        else:
            baseline_value = 0.0
        
        for traj in group:
            traj.advantage = traj.reward - baseline_value
            all_advantages.append(traj.advantage)
    
    # Normalize advantages across all groups
    if normalize and len(all_advantages) > 1:
        mean_adv = sum(all_advantages) / len(all_advantages)
        std_adv = math.sqrt(sum((a - mean_adv) ** 2 for a in all_advantages) / len(all_advantages))
        std_adv = max(std_adv, 1e-8)  # Prevent division by zero
        
        for traj in trajectories:
            traj.advantage = (traj.advantage - mean_adv) / std_adv
    
    return trajectories


def normalize_rewards(
    trajectories: list[TrajectoryWithReward],
) -> list[TrajectoryWithReward]:
    """
    Normalize rewards across all trajectories.
    
    Converts raw profit to normalized reward in [-1, 1] range.
    """
    rewards = [t.reward for t in trajectories]
    
    if len(rewards) <= 1:
        return trajectories
    
    min_reward = min(rewards)
    max_reward = max(rewards)
    reward_range = max_reward - min_reward
    
    if reward_range < 1e-8:
        # All rewards are the same
        for traj in trajectories:
            traj.reward = 0.0
    else:
        for traj in trajectories:
            # Normalize to [-1, 1]
            traj.reward = 2 * (traj.reward - min_reward) / reward_range - 1
    
    return trajectories


class GRPODataset(Dataset):
    """Dataset for GRPO training samples."""
    
    def __init__(
        self,
        trajectories: list[TrajectoryWithReward],
        tokenizer,
        max_length: int = 2048,
    ):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Flatten trajectories into individual turn samples
        for traj in trajectories:
            for i, turn in enumerate(traj.trajectory.turns):
                self.samples.append({
                    "observation": turn.observation_text,
                    "action": turn.action,
                    "reasoning": turn.reasoning,
                    "advantage": traj.advantage,
                    "log_prob": traj.log_probs[i] if i < len(traj.log_probs) else 0.0,
                    "ref_log_prob": traj.ref_log_probs[i] if i < len(traj.ref_log_probs) else 0.0,
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class GRPOTrainer:
    """
    GRPO trainer for fine-tuning language models on LemonadeBench.
    
    Uses environment profit as reward signal and computes relative
    advantages within groups of trajectories from the same seed.
    """
    
    def __init__(
        self,
        model,
        ref_model,  # Reference model for KL divergence
        tokenizer,
        config: GRPOConfig,
        optimizer=None,
    ):
        """
        Initialize GRPO trainer.
        
        Args:
            model: Model to train (with LoRA adapters)
            ref_model: Frozen reference model for KL computation
            tokenizer: Tokenizer for the model
            config: GRPO training configuration
            optimizer: Optional optimizer (created if None)
        """
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = config
        
        if optimizer is None:
            # Only optimize LoRA parameters
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=config.learning_rate,
            )
        else:
            self.optimizer = optimizer
        
        self.device = next(model.parameters()).device
    
    def compute_log_probs(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probabilities for the given labels.
        
        Args:
            model: Model to use for computation
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels (shifted)
            
        Returns:
            Log probabilities per token
        """
        with torch.no_grad() if model == self.ref_model else torch.enable_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits
        
        # Shift for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Compute log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs for actual labels
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)
        
        # Mask padding
        mask = (shift_labels != self.tokenizer.pad_token_id).float()
        token_log_probs = token_log_probs * mask
        
        # Sum over sequence
        return token_log_probs.sum(dim=-1)
    
    def grpo_loss(
        self,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute GRPO loss.
        
        GRPO loss = -E[advantage * log_prob] + kl_coef * KL(policy || ref)
        
        Args:
            log_probs: Log probabilities from current policy
            ref_log_probs: Log probabilities from reference policy
            advantages: Computed advantages
            
        Returns:
            Tuple of (loss tensor, metrics dict)
        """
        # Policy gradient loss (weighted by advantage)
        pg_loss = -(advantages * log_probs).mean()
        
        # KL divergence: D_KL[π_sampler || π_θ]
        # Per Tinker: training is stable when KL < 0.01
        kl_div = (ref_log_probs - log_probs).mean()  # Note: order matters for KL direction
        
        # Total loss
        loss = pg_loss + self.config.kl_coef * kl_div
        
        # Check KL stability threshold
        kl_warning = kl_div.item() > self.config.max_kl_divergence
        
        return loss, {
            "pg_loss": pg_loss.item(),
            "kl_div": kl_div.item(),
            "kl_warning": kl_warning,
            "total_loss": loss.item(),
        }
    
    def train_step(
        self,
        trajectories: list[TrajectoryWithReward],
    ) -> dict[str, float]:
        """
        Perform a single GRPO training step.
        
        Args:
            trajectories: Batch of trajectories with computed advantages
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_pg_loss = 0.0
        total_kl_div = 0.0
        num_samples = 0
        
        # Process trajectories in mini-batches
        for i in range(0, len(trajectories), self.config.mini_batch_size):
            batch = trajectories[i:i + self.config.mini_batch_size]
            
            batch_log_probs = []
            batch_ref_log_probs = []
            batch_advantages = []
            
            for traj in batch:
                # Get precomputed values
                if traj.log_probs:
                    batch_log_probs.append(sum(traj.log_probs))
                    batch_ref_log_probs.append(sum(traj.ref_log_probs))
                    batch_advantages.append(traj.advantage)
            
            if not batch_log_probs:
                continue
            
            # Convert to tensors
            log_probs = torch.tensor(batch_log_probs, device=self.device)
            ref_log_probs = torch.tensor(batch_ref_log_probs, device=self.device)
            advantages = torch.tensor(batch_advantages, device=self.device)
            
            # Compute loss
            loss, metrics = self.grpo_loss(log_probs, ref_log_probs, advantages)
            
            # Scale loss for gradient accumulation
            scaled_loss = loss / self.config.gradient_accumulation_steps
            scaled_loss.backward()
            
            total_loss += metrics["total_loss"]
            total_pg_loss += metrics["pg_loss"]
            total_kl_div += metrics["kl_div"]
            num_samples += len(batch)
        
        # Gradient step
        if num_samples > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return {
            "loss": total_loss / max(num_samples, 1),
            "pg_loss": total_pg_loss / max(num_samples, 1),
            "kl_div": total_kl_div / max(num_samples, 1),
            "num_samples": num_samples,
        }
    
    def prepare_trajectories(
        self,
        trajectories: list[Trajectory],
    ) -> list[TrajectoryWithReward]:
        """
        Prepare trajectories for training by computing rewards and advantages.
        
        Args:
            trajectories: Raw trajectories from environment
            
        Returns:
            Trajectories with rewards, log probs, and advantages
        """
        # Convert to TrajectoryWithReward
        prepared = []
        for traj in trajectories:
            prepared.append(TrajectoryWithReward(
                trajectory=traj,
                reward=float(traj.total_profit),  # Use profit as reward
                seed=traj.seed,
            ))
        
        # Normalize rewards
        if self.config.normalize_rewards:
            prepared = normalize_rewards(prepared)
        
        # Compute advantages
        prepared = compute_group_advantages(
            prepared,
            baseline=self.config.reward_baseline,
            normalize=self.config.advantage_normalization,
        )
        
        return prepared

