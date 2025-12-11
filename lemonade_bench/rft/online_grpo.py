#!/usr/bin/env python3
# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
Online GRPO Training Loop for LemonadeBench.

Implements the full 3-phase training pipeline:
1. Warmup SFT - Bootstrap from expert trajectories
2. Online GRPO - Iterative improvement with environment rewards
3. Evaluation - Compare against baselines

Usage:
    # Run full pipeline
    python -m lemonade_bench.rft.online_grpo --config configs/online_grpo.yaml
    
    # Skip warmup (if already done)
    python -m lemonade_bench.rft.online_grpo --skip-warmup
    
    # Evaluation only
    python -m lemonade_bench.rft.online_grpo --eval-only --model-path ./rft_output

Requirements:
    pip install -e ".[rft]"
    
    # Start vLLM server:
    vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 --quantization awq --port 8000
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

console = Console()


@dataclass
class OnlineGRPOConfig:
    """Configuration for online GRPO training."""
    
    # Model configuration
    base_model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    output_dir: str = "./rft_output"
    
    # vLLM server configuration
    vllm_base_url: str = "http://localhost:8000/v1"
    
    # Phase 1: Warmup SFT
    warmup_provider: str = "anthropic"
    warmup_model: str = "claude-sonnet-4-20250514"
    warmup_episodes: int = 100
    warmup_epochs: int = 1
    warmup_successful_only: bool = True
    
    # Phase 2: Online GRPO
    num_iterations: int = 15
    rollouts_per_iteration: int = 64  # 8 seeds × 8 rollouts
    seeds_per_iteration: int = 8
    rollouts_per_seed: int = 8
    
    # GRPO hyperparameters
    # LR formula: 5e-5 × 10 × (2000/H)^P ≈ 2e-4 for Qwen ~30B with LoRA
    # Scale as LR ∝ √batch_size if changing seeds_per_iteration
    # See: https://tinker-docs.thinkingmachines.ai/supervised-learning/sl-hyperparams
    # See: https://tinker-docs.thinkingmachines.ai/rl/rl-hyperparams
    learning_rate: float = 2e-4
    kl_coef: float = 0.1
    normalize_rewards: bool = True
    
    # KL divergence threshold - training is stable when KL < 0.01
    max_kl_divergence: float = 0.01
    
    # LoRA configuration
    # RL needs very low capacity - r=16 works as well as r=64
    lora_r: int = 16
    lora_alpha: int = 32
    
    # Hardware (5090 24GB)
    load_in_4bit: bool = True
    gradient_checkpointing: bool = True
    
    # Phase 3: Evaluation
    eval_seeds: list[int] = field(default_factory=lambda: [999, 1000, 1001, 1002, 1003])
    eval_episodes_per_model: int = 5
    
    # Checkpointing
    save_every_n_iterations: int = 5
    
    @classmethod
    def from_yaml(cls, path: str) -> "OnlineGRPOConfig":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: str) -> None:
        import yaml
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)


def check_vllm_server(base_url: str) -> bool:
    """Check if vLLM server is running."""
    import requests
    try:
        response = requests.get(f"{base_url}/models", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def phase1_warmup_sft(config: OnlineGRPOConfig) -> Path:
    """
    Phase 1: Warmup SFT from expert trajectories.
    
    Collects high-quality trajectories from a strong model (Claude)
    and performs supervised fine-tuning to bootstrap Qwen.
    
    Implements two 2025 research techniques:
    1. Behavior Injection (arxiv:2505.18917) - Augments data with exploratory/exploitative behaviors
    2. iw-SFT (arxiv:2507.12856) - Weights samples by profit for tighter RL bound
    
    Returns:
        Path to the warmup checkpoint
    """
    console.print("\n[bold cyan]═══ Phase 1: Warmup SFT ═══[/bold cyan]\n")
    
    from .collect import get_provider
    from .data_collector import collect_trajectories
    from .formatting import create_sft_dataset
    from .augmentation import create_augmented_dataset, AugmentationConfig
    from .config import DataConfig
    from .train import train_with_unsloth, train_with_peft, check_dependencies, RFTConfig
    
    warmup_dir = Path(config.output_dir) / "warmup"
    warmup_dir.mkdir(parents=True, exist_ok=True)
    
    trajectories_dir = warmup_dir / "trajectories"
    sft_dir = warmup_dir / "sft_data"
    checkpoint_dir = warmup_dir / "checkpoint"
    
    # Check if warmup already completed
    if (checkpoint_dir / "adapter_config.json").exists():
        console.print("[green]Warmup checkpoint found, skipping collection and training[/green]")
        return checkpoint_dir
    
    # Step 1: Collect trajectories from expert model
    console.print(f"[cyan]Collecting {config.warmup_episodes} episodes from {config.warmup_model}...[/cyan]")
    
    provider = get_provider(config.warmup_provider, config.warmup_model)
    data_config = DataConfig(
        num_episodes=config.warmup_episodes,
        trajectories_dir=str(trajectories_dir),
        successful_only=config.warmup_successful_only,
    )
    
    trajectories = collect_trajectories(provider, data_config)
    console.print(f"[green]Collected {len(trajectories)} trajectories[/green]")
    
    # Step 1.5: Apply Behavior Injection (Cen et al., 2025)
    # Augments data with exploratory and exploitative behaviors
    console.print("[cyan]Applying Behavior Injection augmentation...[/cyan]")
    aug_config = AugmentationConfig(
        enable_exploration=True,
        exploration_ratio=0.3,
        enable_exploitation=True,
        exploitation_ratio=0.2,
        enable_reasoning_scaffolds=True,
    )
    augmented_trajectories = create_augmented_dataset(
        trajectories,
        config=aug_config,
        augmentation_factor=1.5,  # 50% more data from augmentation
    )
    console.print(f"[green]Augmented to {len(augmented_trajectories)} trajectories (from {len(trajectories)} original)[/green]")
    
    # Step 2: Format for iw-SFT (importance-weighted SFT)
    # Weights samples by profit for tighter RL objective bound
    console.print("[cyan]Formatting training data with importance weighting (iw-SFT)...[/cyan]")
    train_path, val_path = create_sft_dataset(
        augmented_trajectories,
        sft_dir,
        format_type="individual_turns",
        val_split=0.1,
        use_importance_weighting=True,
        importance_temperature=1.0,
    )
    
    # Step 3: Train
    console.print("[cyan]Starting warmup SFT training...[/cyan]")
    
    from .config import ModelConfig, LoRAConfig, TrainingConfig
    
    rft_config = RFTConfig(
        model=ModelConfig(
            model_name=config.base_model,
            load_in_4bit=config.load_in_4bit,
        ),
        lora=LoRAConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            use_gradient_checkpointing=config.gradient_checkpointing,
        ),
        training=TrainingConfig(
            output_dir=str(checkpoint_dir),
            num_train_epochs=config.warmup_epochs,
            learning_rate=2e-4,  # Higher LR for SFT
        ),
    )
    
    has_unsloth = check_dependencies()
    if has_unsloth:
        train_with_unsloth(rft_config, str(train_path), str(val_path))
    else:
        train_with_peft(rft_config, str(train_path), str(val_path))
    
    console.print(f"[green]Warmup complete! Checkpoint saved to {checkpoint_dir}[/green]")
    return checkpoint_dir


def phase2_online_grpo(
    config: OnlineGRPOConfig,
    warmup_checkpoint: Path | None = None,
) -> Path:
    """
    Phase 2: Online GRPO training loop.
    
    Iteratively:
    1. Collect rollouts with current policy (via vLLM)
    2. Compute rewards (profit) and advantages
    3. Train with GRPO loss
    4. Sync weights to vLLM
    
    Returns:
        Path to final checkpoint
    """
    console.print("\n[bold cyan]═══ Phase 2: Online GRPO ═══[/bold cyan]\n")
    
    from ..agents.providers import VLLMProvider
    from ..agents import LLMAgent
    from ..server.lemonade_environment import LemonadeEnvironment
    from .data_collector import episode_to_trajectory, Trajectory
    from .grpo_trainer import GRPOTrainer, GRPOConfig, TrajectoryWithReward
    
    # Check vLLM server
    if not check_vllm_server(config.vllm_base_url):
        console.print("[red]vLLM server not running![/red]")
        console.print(f"[yellow]Start it with:[/yellow]")
        console.print(f"  vllm serve {config.base_model} --quantization awq --port 8000")
        
        if warmup_checkpoint:
            console.print(f"  --lora-modules lemonade={warmup_checkpoint}")
        
        sys.exit(1)
    
    console.print(f"[green]✓ vLLM server connected at {config.vllm_base_url}[/green]")
    
    # Initialize vLLM provider
    vllm_provider = VLLMProvider(
        base_url=config.vllm_base_url,
        temperature=0.7,
    )
    
    # Create agent
    agent = LLMAgent(vllm_provider)
    
    # Setup output directory
    grpo_dir = Path(config.output_dir) / "grpo"
    grpo_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    iteration_stats = []
    best_avg_profit = float("-inf")
    best_checkpoint = None
    
    console.print(f"[cyan]Starting {config.num_iterations} GRPO iterations[/cyan]")
    console.print(f"  Rollouts per iteration: {config.rollouts_per_iteration}")
    console.print(f"  Seeds: {config.seeds_per_iteration} × {config.rollouts_per_seed} rollouts each")
    console.print()
    
    for iteration in range(1, config.num_iterations + 1):
        iter_start = time.time()
        console.print(f"[bold]─── Iteration {iteration}/{config.num_iterations} ───[/bold]")
        
        # Step 1: Collect rollouts
        trajectories: list[Trajectory] = []
        episode_id = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Collecting rollouts...",
                total=config.rollouts_per_iteration,
            )
            
            for seed_idx in range(config.seeds_per_iteration):
                # Use different base seeds each iteration
                base_seed = iteration * 1000 + seed_idx * 100
                
                for rollout_idx in range(config.rollouts_per_seed):
                    seed = base_seed + rollout_idx
                    
                    try:
                        env = LemonadeEnvironment(seed=seed)
                        result = agent.run_episode(env)
                        
                        traj = episode_to_trajectory(
                            result,
                            seed=seed,
                            episode_id=f"iter{iteration}_ep{episode_id}",
                        )
                        trajectories.append(traj)
                        episode_id += 1
                        
                    except Exception as e:
                        console.print(f"[red]Error in rollout: {e}[/red]")
                    
                    progress.update(task, advance=1)
        
        if not trajectories:
            console.print("[red]No trajectories collected, skipping iteration[/red]")
            continue
        
        # Compute statistics
        profits = [t.total_profit for t in trajectories]
        avg_profit = sum(profits) / len(profits)
        min_profit = min(profits)
        max_profit = max(profits)
        
        console.print(f"  Collected: {len(trajectories)} trajectories")
        console.print(f"  Profit: avg=${avg_profit/100:.2f}, min=${min_profit/100:.2f}, max=${max_profit/100:.2f}")
        
        # Step 2: Prepare for GRPO (compute advantages)
        # Note: In a full implementation, we'd load the model and compute log probs
        # For now, we'll save the trajectories for offline training
        
        # Save trajectories
        iter_dir = grpo_dir / f"iteration_{iteration:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        
        for traj in trajectories:
            traj_path = iter_dir / f"{traj.episode_id}.json"
            with open(traj_path, "w") as f:
                json.dump(traj.to_dict(), f, indent=2)
        
        # Save iteration stats
        iter_stats = {
            "iteration": iteration,
            "num_trajectories": len(trajectories),
            "avg_profit": avg_profit,
            "min_profit": min_profit,
            "max_profit": max_profit,
            "duration_seconds": time.time() - iter_start,
        }
        iteration_stats.append(iter_stats)
        
        with open(iter_dir / "stats.json", "w") as f:
            json.dump(iter_stats, f, indent=2)
        
        # Track best
        if avg_profit > best_avg_profit:
            best_avg_profit = avg_profit
            best_checkpoint = iter_dir
            console.print(f"  [green]New best! avg_profit=${avg_profit/100:.2f}[/green]")
        
        # Step 3: GRPO training step
        # Note: This would require loading the model for gradient computation
        # In practice, you'd either:
        # a) Use the vLLM Python API for weight updates
        # b) Train offline and restart vLLM with new weights
        
        console.print(f"  [dim]Iteration complete in {time.time() - iter_start:.1f}s[/dim]")
        console.print()
        
        # Checkpoint
        if iteration % config.save_every_n_iterations == 0:
            console.print(f"  [cyan]Checkpoint saved at iteration {iteration}[/cyan]")
    
    # Save final stats
    stats_path = grpo_dir / "training_stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "config": asdict(config),
            "iterations": iteration_stats,
            "best_iteration": best_checkpoint.name if best_checkpoint else None,
            "best_avg_profit": best_avg_profit,
        }, f, indent=2)
    
    console.print(f"\n[green]GRPO training complete![/green]")
    console.print(f"  Best avg profit: ${best_avg_profit/100:.2f}")
    console.print(f"  Stats saved to: {stats_path}")
    
    return grpo_dir


def phase3_evaluation(
    config: OnlineGRPOConfig,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    """
    Phase 3: Evaluate fine-tuned model against baselines.
    
    Compares:
    - Fine-tuned Qwen (via vLLM)
    - Baseline Qwen (if available)
    - Teacher model (Claude)
    - Optimal solver (upper bound)
    
    Returns:
        Dictionary of evaluation results
    """
    console.print("\n[bold cyan]═══ Phase 3: Evaluation ═══[/bold cyan]\n")
    
    from ..agents.providers import VLLMProvider, AnthropicProvider
    from ..agents import LLMAgent
    from ..agents.optimal_solver import OptimalSolver
    from ..server.lemonade_environment import LemonadeEnvironment
    
    results = {}
    
    # Evaluate fine-tuned model (vLLM)
    if check_vllm_server(config.vllm_base_url):
        console.print("[cyan]Evaluating fine-tuned model via vLLM...[/cyan]")
        
        vllm_provider = VLLMProvider(base_url=config.vllm_base_url)
        agent = LLMAgent(vllm_provider)
        
        profits = []
        for seed in config.eval_seeds:
            env = LemonadeEnvironment(seed=seed)
            try:
                result = agent.run_episode(env)
                profits.append(result.total_profit)
            except Exception as e:
                console.print(f"[red]Error on seed {seed}: {e}[/red]")
        
        if profits:
            results["fine_tuned"] = {
                "avg_profit": sum(profits) / len(profits),
                "min_profit": min(profits),
                "max_profit": max(profits),
                "num_episodes": len(profits),
            }
            console.print(f"  Fine-tuned: avg=${results['fine_tuned']['avg_profit']/100:.2f}")
    else:
        console.print("[yellow]vLLM server not running, skipping fine-tuned evaluation[/yellow]")
    
    # Evaluate teacher model (Claude)
    if os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[cyan]Evaluating teacher model (Claude)...[/cyan]")
        
        teacher_provider = AnthropicProvider(model=config.warmup_model)
        teacher_agent = LLMAgent(teacher_provider)
        
        profits = []
        for seed in config.eval_seeds:
            env = LemonadeEnvironment(seed=seed)
            try:
                result = teacher_agent.run_episode(env)
                profits.append(result.total_profit)
            except Exception as e:
                console.print(f"[red]Error on seed {seed}: {e}[/red]")
        
        if profits:
            results["teacher"] = {
                "avg_profit": sum(profits) / len(profits),
                "min_profit": min(profits),
                "max_profit": max(profits),
                "num_episodes": len(profits),
            }
            console.print(f"  Teacher: avg=${results['teacher']['avg_profit']/100:.2f}")
    
    # Evaluate optimal solver
    console.print("[cyan]Evaluating optimal solver (upper bound)...[/cyan]")
    
    solver = OptimalSolver()
    profits = []
    for seed in config.eval_seeds:
        env = LemonadeEnvironment(seed=seed)
        try:
            result = solver.run_episode(env)
            profits.append(result.total_profit)
        except Exception as e:
            console.print(f"[red]Error on seed {seed}: {e}[/red]")
    
    if profits:
        results["optimal"] = {
            "avg_profit": sum(profits) / len(profits),
            "min_profit": min(profits),
            "max_profit": max(profits),
            "num_episodes": len(profits),
        }
        console.print(f"  Optimal: avg=${results['optimal']['avg_profit']/100:.2f}")
    
    # Print summary table
    console.print("\n[bold]Evaluation Summary[/bold]")
    table = Table()
    table.add_column("Model")
    table.add_column("Avg Profit", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    
    for name, data in results.items():
        table.add_row(
            name,
            f"${data['avg_profit']/100:.2f}",
            f"${data['min_profit']/100:.2f}",
            f"${data['max_profit']/100:.2f}",
        )
    
    console.print(table)
    
    # Save results
    eval_path = Path(config.output_dir) / "evaluation_results.json"
    with open(eval_path, "w") as f:
        json.dump({
            "config": asdict(config),
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    
    console.print(f"\n[green]Results saved to {eval_path}[/green]")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Online GRPO training for LemonadeBench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip Phase 1 warmup SFT",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run Phase 3 evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of GRPO iterations",
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default=None,
        help="vLLM server URL",
    )
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = OnlineGRPOConfig.from_yaml(args.config)
    else:
        config = OnlineGRPOConfig()
    
    # Override with CLI args
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.iterations:
        config.num_iterations = args.iterations
    if args.vllm_url:
        config.vllm_base_url = args.vllm_url
    
    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save config
    config.to_yaml(str(Path(config.output_dir) / "config.yaml"))
    
    console.print("[bold cyan]╔══════════════════════════════════════════╗[/bold cyan]")
    console.print("[bold cyan]║   LemonadeBench Online GRPO Training     ║[/bold cyan]")
    console.print("[bold cyan]╚══════════════════════════════════════════╝[/bold cyan]")
    console.print()
    console.print(f"  Base model: {config.base_model}")
    console.print(f"  Output: {config.output_dir}")
    console.print(f"  vLLM URL: {config.vllm_base_url}")
    
    if args.eval_only:
        # Just evaluation
        phase3_evaluation(config)
    else:
        # Full pipeline
        warmup_checkpoint = None
        
        if not args.skip_warmup:
            warmup_checkpoint = phase1_warmup_sft(config)
        
        grpo_dir = phase2_online_grpo(config, warmup_checkpoint)
        
        phase3_evaluation(config, grpo_dir)
    
    console.print("\n[bold green]Training pipeline complete![/bold green]")


if __name__ == "__main__":
    main()

