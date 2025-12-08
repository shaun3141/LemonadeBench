#!/usr/bin/env python3
# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
LLM-powered agent for LemonadeBench.

This is a thin wrapper around the modular agent framework. For full
functionality, use the CLI harness:

    # Single run
    lemonade run --model claude-sonnet-4-20250514 --seed 42

    # Batch runs
    lemonade batch config.yaml --parallel 4

Direct usage:
    # Set your API key
    export ANTHROPIC_API_KEY="your-key-here"

    # Run a single episode
    uv run python examples/llm_agent.py

    # Run with a specific seed
    uv run python examples/llm_agent.py --seed 42

    # Run with a different model
    uv run python examples/llm_agent.py --model claude-sonnet-4-20250514

    # Use OpenAI instead
    uv run python examples/llm_agent.py --provider openai --model gpt-4o

    # Use OpenRouter to access 400+ models from different providers
    export OPENROUTER_API_KEY="your-key-here"
    uv run python examples/llm_agent.py --provider openrouter --model openai/gpt-4o
    uv run python examples/llm_agent.py --provider openrouter --model anthropic/claude-3.5-sonnet
    uv run python examples/llm_agent.py --provider openrouter --model meta-llama/llama-3.3-70b-instruct
"""

import argparse
import sys
from pathlib import Path

# Add the parent directory to the path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file if present
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / "lemonade_bench" / ".env")

from rich.console import Console
from rich.table import Table

from lemonade_bench.agents import LLMAgent
from lemonade_bench.agents.providers import AnthropicProvider
from lemonade_bench.server.lemonade_environment import LemonadeEnvironment
from lemonade_bench.harness.runner import RunLogger, VerboseCallback


def get_provider(provider_name: str, model: str):
    """Get the appropriate provider instance."""
    if provider_name == "anthropic":
        return AnthropicProvider(model=model)
    elif provider_name == "openai":
        from lemonade_bench.agents.providers.openai import OpenAIProvider
        return OpenAIProvider(model=model)
    elif provider_name == "openrouter":
        from lemonade_bench.agents.providers.openrouter import OpenRouterProvider
        return OpenRouterProvider(model=model)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")


def run_episode(
    model: str = "claude-sonnet-4-20250514",
    provider: str = "anthropic",
    seed: int | None = None,
    verbose: bool = True,
    log_runs: bool = True,
    use_supabase: bool = True,
) -> float:
    """
    Run a single episode with the LLM agent.

    Args:
        model: Model name to use
        provider: Provider name (anthropic, openai)
        seed: Random seed for reproducibility
        verbose: Print progress to console
        log_runs: Save run logs to disk
        use_supabase: Save run logs to Supabase cloud database

    Returns:
        Total profit for the episode
    """
    console = Console()
    
    # Initialize provider and agent
    llm_provider = get_provider(provider, model)
    agent = LLMAgent(llm_provider)
    
    # Initialize environment
    env = LemonadeEnvironment(seed=seed)

    # Initialize logger
    logger = None
    if log_runs:
        logger = RunLogger(use_supabase=use_supabase)
        logger.save_config({
            "model": f"{provider}/{model}",
            "seed": seed,
            "timestamp": logger.timestamp,
        })

    if verbose:
        console.print()
        console.print("[bold blue]🍋 LemonadeBench - LLM Agent[/bold blue]")
        console.print(f"[dim]Model: {provider}/{model}[/dim]")
        if seed:
            console.print(f"[dim]Seed: {seed}[/dim]")
        if logger:
            console.print(f"[dim]Logging to: {logger.run_dir}[/dim]")
        console.print()

    # Set up callbacks
    callbacks = []
    if verbose:
        callbacks.append(VerboseCallback(console))
    
    # Run episode
    result = agent.run_episode(env, callbacks=callbacks)

    # Save results
    if logger:
        logger.save_episode_result(result)

    # Print summary
    if verbose:
        console.print()
        console.print("[bold]═" * 50 + "[/bold]")
        console.print("[bold green]🏆 GAME OVER![/bold green]")
        console.print("[bold]═" * 50 + "[/bold]")
        
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Total Profit", f"${result.total_profit / 100:.2f}")
        table.add_row("Cups Sold", str(result.total_cups_sold))
        table.add_row("Final Cash", f"${result.final_cash / 100:.2f}")
        table.add_row("Reputation", f"{result.final_reputation:.2f}")
        table.add_row("Turns", str(result.turn_count))
        
        if result.total_input_tokens > 0:
            table.add_row("Input Tokens", f"{result.total_input_tokens:,}")
            table.add_row("Output Tokens", f"{result.total_output_tokens:,}")
            table.add_row("Est. Cost", f"${result.estimated_cost_usd:.4f}")
        
        console.print(table)
        
        if logger:
            console.print(f"\n[dim]Run saved to: {logger.run_dir}[/dim]")

    return result.total_profit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LemonadeBench with an LLM agent",
        epilog="For batch runs and more options, use: lemonade --help"
    )
    parser.add_argument(
        "--model", type=str, default="claude-sonnet-4-20250514",
        help="Model to use (default: claude-sonnet-4-20250514)"
    )
    parser.add_argument(
        "--provider", type=str, default="anthropic",
        choices=["anthropic", "openai", "openrouter"],
        help="LLM provider (default: anthropic). Use openrouter for 400+ models."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--no-logs", action="store_true",
        help="Disable local run logging"
    )
    parser.add_argument(
        "--no-supabase", action="store_true",
        help="Disable Supabase cloud logging"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Reduce output verbosity"
    )
    args = parser.parse_args()

    run_episode(
        model=args.model,
        provider=args.provider,
        seed=args.seed,
        verbose=not args.quiet,
        log_runs=not args.no_logs,
        use_supabase=not args.no_supabase,
    )
