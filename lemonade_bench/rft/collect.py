#!/usr/bin/env python3
# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
Data collection CLI for RFT.

Collects trajectory data from LemonadeBench using LLM agents
and formats it for fine-tuning.

Usage:
    # Collect with Anthropic
    python -m lemonade_bench.rft.collect --provider anthropic --episodes 100

    # Collect with OpenAI
    python -m lemonade_bench.rft.collect --provider openai --model gpt-4o

    # Collect and format for SFT
    python -m lemonade_bench.rft.collect --provider anthropic --format sft
"""

import argparse
import os
import sys
from pathlib import Path

from rich.console import Console

from .config import DataConfig, RFTConfig
from .data_collector import collect_trajectories, load_trajectories
from .formatting import create_sft_dataset, create_grpo_dataset

console = Console()


def get_provider(provider_name: str, model: str | None = None):
    """Get LLM provider instance."""
    
    if provider_name == "anthropic":
        from ..agents.providers import AnthropicProvider
        return AnthropicProvider(model=model or "claude-sonnet-4-20250514")
    
    elif provider_name == "openai":
        from ..agents.providers import OpenAIProvider
        return OpenAIProvider(model=model or "gpt-4o")
    
    elif provider_name == "google":
        from ..agents.providers import GoogleProvider
        return GoogleProvider(model=model or "gemini-2.0-flash")
    
    else:
        console.print(f"[red]Unknown provider: {provider_name}[/red]")
        console.print("[yellow]Available: anthropic, openai, google[/yellow]")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Collect trajectory data for RFT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        required=True,
        choices=["anthropic", "openai", "google"],
        help="LLM provider to use for data collection",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model to use (default: provider's best model)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to collect (default: 100)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./rft_trajectories",
        help="Output directory for trajectories (default: ./rft_trajectories)",
    )
    parser.add_argument(
        "--successful-only",
        action="store_true",
        help="Only keep episodes with positive profit",
    )
    parser.add_argument(
        "--min-profit",
        type=float,
        default=None,
        help="Minimum profit threshold to include episode",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["sft", "grpo", "both"],
        default="sft",
        help="Format to create (default: sft)",
    )
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Skip collection, only format existing trajectories",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        config = RFTConfig.from_yaml(args.config)
        data_config = config.data
    else:
        data_config = DataConfig(
            num_episodes=args.episodes,
            trajectories_dir=args.output_dir,
            successful_only=args.successful_only,
            min_profit_threshold=args.min_profit,
        )
    
    # Override with CLI args
    if args.episodes != 100:
        data_config.num_episodes = args.episodes
    if args.output_dir != "./rft_trajectories":
        data_config.trajectories_dir = args.output_dir
    if args.successful_only:
        data_config.successful_only = True
    if args.min_profit is not None:
        data_config.min_profit_threshold = args.min_profit
    
    output_dir = Path(data_config.trajectories_dir)
    
    if args.format_only:
        # Just format existing data
        console.print(f"[cyan]Loading trajectories from {output_dir}[/cyan]")
        trajectories = load_trajectories(output_dir)
        console.print(f"[green]Loaded {len(trajectories)} trajectories[/green]")
    else:
        # Collect new data
        console.print("\n[bold cyan]═══ RFT Data Collection ═══[/bold cyan]")
        console.print(f"  Provider: {args.provider}")
        console.print(f"  Model: {args.model or 'default'}")
        console.print(f"  Episodes: {data_config.num_episodes}")
        console.print(f"  Output: {output_dir}")
        console.print()
        
        # Get provider
        provider = get_provider(args.provider, args.model)
        
        # Collect trajectories
        trajectories = collect_trajectories(provider, data_config)
    
    if len(trajectories) == 0:
        console.print("[red]No trajectories collected![/red]")
        sys.exit(1)
    
    # Format data
    console.print("\n[cyan]Formatting training data...[/cyan]")
    
    if args.format in ["sft", "both"]:
        sft_dir = output_dir / "sft"
        create_sft_dataset(
            trajectories,
            sft_dir,
            format_type="individual_turns",
            val_split=data_config.val_split,
        )
    
    if args.format in ["grpo", "both"]:
        grpo_path = output_dir / "grpo" / "train.jsonl"
        create_grpo_dataset(trajectories, grpo_path)
    
    # Summary
    console.print("\n[bold green]═══ Collection Complete ═══[/bold green]")
    console.print(f"  Trajectories: {len(trajectories)}")
    
    profits = [t.total_profit for t in trajectories]
    avg_profit = sum(profits) / len(profits)
    console.print(f"  Average Profit: ${avg_profit / 100:.2f}")
    console.print(f"  Min Profit: ${min(profits) / 100:.2f}")
    console.print(f"  Max Profit: ${max(profits) / 100:.2f}")
    console.print()
    console.print(f"[green]Data saved to: {output_dir}[/green]")
    console.print()
    console.print("[cyan]Next step - train with:[/cyan]")
    console.print(f"  python -m lemonade_bench.rft.train --train-data {output_dir}/sft/train.jsonl")


if __name__ == "__main__":
    main()

