# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
Configuration for Reinforcement Fine-Tuning (RFT).

Defines hyperparameters and settings for LoRA fine-tuning
of language models on LemonadeBench trajectories.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class ModelConfig:
    """Configuration for the base model."""
    
    # Model identifier on HuggingFace
    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    
    # Maximum sequence length for training
    max_seq_length: int = 8192
    
    # Load in 4-bit quantization (QLoRA) for memory efficiency
    load_in_4bit: bool = True
    
    # Use bfloat16 for training (requires Ampere+ GPU)
    dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"
    
    # Trust remote code (required for Qwen models)
    trust_remote_code: bool = True


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation) layers.
    
    Key insights from research:
    - RL requires very low capacity - small ranks (8-16) work as well as larger ranks
    - SL on large datasets may need higher ranks (32-64)
    - Apply LoRA to ALL weight matrices, especially MLP/MoE layers
    - Learning rate multiplier is ~10x vs full fine-tuning (not 20-100x)
    - LR is independent of LoRA rank
    
    See: https://tinker-docs.thinkingmachines.ai/lora-primer
    See: https://tinker-docs.thinkingmachines.ai/supervised-learning/sl-hyperparams
    """
    
    # LoRA rank - for RL, r=16 is sufficient; for large SL datasets, use r=32-64
    r: int = 16
    
    # LoRA alpha - scaling factor, typically set to 2*r
    lora_alpha: int = 32
    
    # Dropout for LoRA layers
    lora_dropout: float = 0.05
    
    # Target modules for LoRA adaptation
    # For Qwen3 MoE, we need to target both attention and expert layers
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj",      # MLP/Expert layers
    ])
    
    # Whether to use RSLoRA (Rank-Stabilized LoRA)
    use_rslora: bool = True
    
    # Use gradient checkpointing to save memory
    use_gradient_checkpointing: bool = True
    
    # Gradient checkpointing method
    gradient_checkpointing_method: Literal["unsloth", "standard"] = "unsloth"


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters.
    
    Learning Rate Formula (from Tinker research):
        LR = lr_base × M_LoRA × (2000 / H_m)^P_m
        
        Where:
        - lr_base = 5e-5
        - M_LoRA = 10 (LoRA multiplier, 1 for full fine-tuning)
        - H_m = hidden size of model
        - P_m = 0.0775 for Qwen, 0.781 for Llama
        
        For Qwen3-30B-A3B (H≈4096): LR ≈ 5e-4
        
    Training Steps: Aim for at least 100 steps, ideally 1000+ for best results.
    Batch Size: Smaller batch sizes often give better performance for LLM fine-tuning.
    
    See: https://tinker-docs.thinkingmachines.ai/supervised-learning/sl-hyperparams
    """
    
    # Output directory for checkpoints and final model
    output_dir: str = "./rft_output"
    
    # Number of training epochs
    num_train_epochs: int = 3
    
    # Batch size per device (smaller is often better for LLM fine-tuning)
    per_device_train_batch_size: int = 1
    
    # Gradient accumulation steps (effective batch = batch_size * grad_accum)
    # Effective batch of 8 is small, which is recommended per Tinker research
    gradient_accumulation_steps: int = 8
    
    # Learning rate: LR = 5e-5 × 10 × (2000/4096)^0.0775 ≈ 5e-4 for Qwen ~30B
    learning_rate: float = 5e-4
    
    # Weight decay
    weight_decay: float = 0.01
    
    # Warmup ratio (fraction of total steps for warmup)
    warmup_ratio: float = 0.03
    
    # Learning rate scheduler
    lr_scheduler_type: str = "cosine"
    
    # Maximum gradient norm for clipping
    max_grad_norm: float = 1.0
    
    # Logging steps
    logging_steps: int = 10
    
    # Save checkpoint every N steps
    save_steps: int = 100
    
    # Maximum checkpoints to keep
    save_total_limit: int = 3
    
    # Use 8-bit AdamW optimizer
    optim: str = "adamw_8bit"
    
    # Random seed
    seed: int = 42
    
    # Use Flash Attention 2 if available
    use_flash_attention: bool = True


@dataclass
class DataConfig:
    """Configuration for trajectory data collection and formatting."""
    
    # Number of episodes to collect for training
    num_episodes: int = 1000
    
    # Directory to store collected trajectories
    trajectories_dir: str = "./rft_trajectories"
    
    # Minimum profit threshold to include episode (filter bad examples)
    min_profit_threshold: float | None = None
    
    # Maximum profit threshold (for curriculum learning)
    max_profit_threshold: float | None = None
    
    # Include only successful episodes (positive profit)
    successful_only: bool = False
    
    # Format for training data
    # - "sft": Supervised fine-tuning (imitation learning)
    # - "grpo": Group Relative Policy Optimization
    # - "dpo": Direct Preference Optimization (requires pairs)
    training_format: Literal["sft", "grpo", "dpo"] = "sft"
    
    # Number of random seeds to use for diversity
    num_seeds: int = 100
    
    # Train/validation split ratio
    val_split: float = 0.1


@dataclass
class RFTConfig:
    """Complete configuration for Reinforcement Fine-Tuning."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Experiment name for tracking
    experiment_name: str = "lemonade_rft"
    
    # Whether to push to HuggingFace Hub
    push_to_hub: bool = False
    hub_model_id: str | None = None
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "RFTConfig":
        """Load configuration from a YAML file."""
        import yaml
        
        with open(path) as f:
            data = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**data.get("model", {})),
            lora=LoRAConfig(**data.get("lora", {})),
            training=TrainingConfig(**data.get("training", {})),
            data=DataConfig(**data.get("data", {})),
            experiment_name=data.get("experiment_name", "lemonade_rft"),
            push_to_hub=data.get("push_to_hub", False),
            hub_model_id=data.get("hub_model_id"),
        )
    
    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        import yaml
        from dataclasses import asdict
        
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)


# Default configuration for quick start
DEFAULT_CONFIG = RFTConfig()

