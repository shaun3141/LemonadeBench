#!/usr/bin/env python3
# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
CLI entry point for LemonadeBench evaluation harness.

Provides commands for running evaluations, comparing models, and viewing results.

Usage:
    # Single run with defaults
    lemonade run

    # Run with specific model and seed
    lemonade run --model anthropic/claude-sonnet-4-20250514 --seed 42

    # Run with Plan-Act architecture and aggressive goal framing
    lemonade run --model gpt-4o --architecture plan_act --goal-framing aggressive

    # Run with math encouragement scaffolding
    lemonade run --model claude-sonnet-4 --math-prompt --tools calculator

    # Batch runs from config file
    lemonade batch config.yaml --parallel 4

    # Compare model results
    lemonade compare --models "claude-*,gpt-*"

    # View leaderboard
    lemonade leaderboard --top 10
"""

import sys
from pathlib import Path
from typing import Optional, Literal

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

# Valid options for CLI arguments
VALID_ARCHITECTURES = ["react", "plan_act", "act_reflect", "full"]
VALID_GOAL_FRAMINGS = ["baseline", "aggressive", "conservative", "competitive", "survival", "growth"]
VALID_TOOLS = ["calculator", "code_interpreter"]

app = typer.Typer(
    name="lemonade",
    help="LemonadeBench - Evaluate LLM agents on the Lemonade Stand benchmark",
    add_completion=False,
)
console = Console()


def parse_model_spec(model: str) -> tuple[str, str]:
    """
    Parse a model specification into (provider, model_name).
    
    Formats:
        - "anthropic/claude-sonnet-4" -> ("anthropic", "claude-sonnet-4")
        - "claude-sonnet-4" -> ("anthropic", "claude-sonnet-4")  # inferred
        - "gpt-4o" -> ("openai", "gpt-4o")  # inferred
    """
    if "/" in model:
        provider, model_name = model.split("/", 1)
        return provider, model_name
    
    # Infer provider from model name
    model_lower = model.lower()
    if "claude" in model_lower:
        return "anthropic", model
    elif "gpt" in model_lower or "o1" in model_lower:
        return "openai", model
    elif "gemini" in model_lower:
        return "google", model
    else:
        # Default to anthropic
        return "anthropic", model


def get_provider(provider_name: str, model_name: str):
    """Get the appropriate provider instance."""
    if provider_name == "anthropic":
        from ..agents.providers import AnthropicProvider
        return AnthropicProvider(model=model_name)
    elif provider_name == "openai":
        from ..agents.providers.openai import OpenAIProvider
        return OpenAIProvider(model=model_name)
    elif provider_name == "openrouter":
        from ..agents.providers.openrouter import OpenRouterProvider
        return OpenRouterProvider(model=model_name)
    else:
        raise ValueError(f"Unknown provider: {provider_name}. Supported: anthropic, openai, openrouter")


@app.command()
def run(
    model: str = typer.Option(
        "claude-sonnet-4-20250514",
        "--model", "-m",
        help="Model to use (e.g., 'anthropic/claude-sonnet-4' or 'gpt-4o')"
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed", "-s",
        help="Random seed for reproducibility"
    ),
    architecture: str = typer.Option(
        "react",
        "--architecture", "-a",
        help=f"Agent architecture: {', '.join(VALID_ARCHITECTURES)}"
    ),
    goal_framing: str = typer.Option(
        "baseline",
        "--goal-framing", "-g",
        help=f"Goal framing condition: {', '.join(VALID_GOAL_FRAMINGS)}"
    ),
    math_prompt: bool = typer.Option(
        False,
        "--math-prompt",
        help="Enable math encouragement scaffolding"
    ),
    tools: Optional[str] = typer.Option(
        None,
        "--tools", "-t",
        help=f"Comma-separated list of tools: {', '.join(VALID_TOOLS)}"
    ),
    output_dir: str = typer.Option(
        "runs",
        "--output", "-o",
        help="Directory to save run logs"
    ),
    no_local: bool = typer.Option(
        False,
        "--no-local",
        help="Disable local file logging"
    ),
    no_supabase: bool = typer.Option(
        False,
        "--no-supabase",
        help="Disable Supabase cloud logging"
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet", "-q",
        help="Reduce output verbosity"
    ),
    show_metrics: bool = typer.Option(
        False,
        "--metrics",
        help="Show diagnostic metrics after run"
    ),
):
    """
    Run a single evaluation episode.
    
    Examples:
        lemonade run --model claude-sonnet-4-20250514 --seed 42
        lemonade run --model gpt-4o --architecture plan_act --goal-framing aggressive
        lemonade run --model claude-sonnet-4 --math-prompt --tools calculator
    """
    from ..agents.architectures import create_agent
    from ..server.lemonade_environment import LemonadeEnvironment
    from .runner import RunLogger, RunConfig
    from .metrics import compute_diagnostic_metrics, format_metrics_summary
    
    # Validate options
    if architecture not in VALID_ARCHITECTURES:
        console.print(f"[red]Invalid architecture: {architecture}. Must be one of: {', '.join(VALID_ARCHITECTURES)}[/red]")
        raise typer.Exit(1)
    
    if goal_framing not in VALID_GOAL_FRAMINGS:
        console.print(f"[red]Invalid goal framing: {goal_framing}. Must be one of: {', '.join(VALID_GOAL_FRAMINGS)}[/red]")
        raise typer.Exit(1)
    
    # Parse tools
    tool_list = []
    if tools:
        tool_list = [t.strip() for t in tools.split(",")]
        for tool in tool_list:
            if tool not in VALID_TOOLS:
                console.print(f"[red]Invalid tool: {tool}. Must be one of: {', '.join(VALID_TOOLS)}[/red]")
                raise typer.Exit(1)
    
    # Parse model specification
    provider_name, model_name = parse_model_spec(model)
    
    if not quiet:
        console.print(f"\n[bold blue]🍋 LemonadeBench[/bold blue]")
        console.print(f"[dim]Model: {provider_name}/{model_name}[/dim]")
        console.print(f"[dim]Architecture: {architecture}[/dim]")
        if goal_framing != "baseline":
            console.print(f"[dim]Goal Framing: {goal_framing}[/dim]")
        if math_prompt:
            console.print(f"[dim]Math Prompt: enabled[/dim]")
        if tool_list:
            console.print(f"[dim]Tools: {', '.join(tool_list)}[/dim]")
        if seed:
            console.print(f"[dim]Seed: {seed}[/dim]")
        console.print()
    
    # Initialize provider and create agent
    try:
        provider = get_provider(provider_name, model_name)
    except ValueError as e:
        console.print(f"[red]Error: {escape(str(e))}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Failed to initialize provider: {escape(str(e))}[/red]")
        raise typer.Exit(1)
    
    # Create agent using architecture factory
    agent = create_agent(
        provider=provider,
        architecture=architecture,
        goal_framing=goal_framing,
        tools=tool_list if tool_list else None,
        math_prompt=math_prompt,
    )
    env = LemonadeEnvironment(seed=seed)
    
    # Create run config for logging
    run_config = RunConfig(
        provider=provider_name,
        model=model_name,
        seed=seed,
        tools=tool_list,
        architecture=architecture,
        goal_framing=goal_framing,
        math_prompt=math_prompt,
    )
    
    # Initialize logger
    logger = None
    if not no_local:
        logger = RunLogger(
            base_dir=output_dir,
            use_supabase=not no_supabase,
        )
        logger.save_config({
            "model": f"{provider_name}/{model_name}",
            "seed": seed,
            "timestamp": logger.timestamp,
            "architecture": architecture,
            "goal_framing": goal_framing,
            "math_prompt": math_prompt,
            "tools": tool_list,
        })
    
    # Create verbose callback if not quiet
    callbacks = []
    if not quiet:
        from .runner import VerboseCallback
        callbacks.append(VerboseCallback(console))
    
    # Run episode
    result = agent.run_episode(env, callbacks=callbacks)
    
    # Compute metrics
    metrics = compute_diagnostic_metrics(result)
    
    # Save results
    if logger:
        logger.save_episode_result(result, run_config=run_config)
        if not quiet:
            console.print(f"\n[dim]Run saved to: {logger.run_dir}[/dim]")
    
    # Print summary
    if not quiet:
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
        
        # Add key diagnostic metrics
        table.add_row("", "")  # Separator
        table.add_row("Spoilage Rate", f"{metrics.spoilage_rate:.1%}")
        table.add_row("Stockout Rate", f"{metrics.stockout_rate:.1%}")
        table.add_row("Weather Adapt.", f"{metrics.weather_adaptation_score:.2f}")
        
        console.print(table)
        
        # Show full metrics if requested
        if show_metrics:
            console.print()
            console.print(format_metrics_summary(metrics))
    
    return result


@app.command()
def batch(
    config_file: Path = typer.Argument(
        ...,
        help="YAML configuration file for batch runs"
    ),
    parallel: int = typer.Option(
        1,
        "--parallel", "-p",
        help="Number of parallel runs"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be run without executing"
    ),
):
    """
    Run batch evaluations from a config file.
    
    Supports both direct model lists and experimental matrix configurations.
    
    Examples:
        lemonade batch config.yaml --parallel 4
        lemonade batch experiment.yaml --dry-run
    """
    from .config import load_config
    from .runner import Runner
    
    if not config_file.exists():
        console.print(f"[red]Config file not found: {config_file}[/red]")
        raise typer.Exit(1)
    
    config = load_config(config_file)
    
    console.print(f"\n[bold blue]🍋 LemonadeBench Batch Run[/bold blue]")
    console.print(f"[dim]Config: {config_file}[/dim]")
    console.print(f"[dim]Name: {config.name}[/dim]")
    
    # Show experiment matrix info if applicable
    if config.experiment:
        console.print(f"[dim]Mode: Experimental Matrix[/dim]")
        console.print(f"[dim]  Models: {len(config.experiment.models)}[/dim]")
        console.print(f"[dim]  Seeds: {len(config.experiment.seeds)}[/dim]")
        console.print(f"[dim]  Goal Framings: {', '.join(config.experiment.goal_framings)}[/dim]")
        console.print(f"[dim]  Architectures: {', '.join(config.experiment.architectures)}[/dim]")
        console.print(f"[dim]  Scaffoldings: {', '.join(config.experiment.scaffoldings)}[/dim]")
    
    # Calculate total runs (show both raw and deduplicated counts)
    total_runs = config.get_total_runs()
    unique_runs = config.get_unique_runs()
    if unique_runs < total_runs:
        console.print(f"[dim]Total runs: {unique_runs} unique ({total_runs - unique_runs} duplicates removed from overlapping experiments)[/dim]")
    else:
        console.print(f"[dim]Total runs: {total_runs}[/dim]")
    
    if dry_run:
        console.print("\n[yellow]Dry run - showing planned runs:[/yellow]")
        for model in config.models:
            seeds = model.seeds or [None]
            for seed in seeds:
                desc = f"  • {model.provider}/{model.name}"
                parts = []
                # Show goal framing (always for experiments, or if non-default)
                if config.experiment is not None:
                    parts.append(f"goal={model.goal_framing}")
                elif model.goal_framing != "baseline":
                    parts.append(f"goal={model.goal_framing}")
                # Show non-default architecture
                if model.architecture != "react":
                    parts.append(f"arch={model.architecture}")
                # Show math prompt flag
                if model.math_prompt:
                    parts.append("math_prompt")
                # Show tools
                if model.tools:
                    parts.append(f"tools={','.join(model.tools)}")
                # Add parts to description (use escaped brackets for Rich)
                if parts:
                    desc = desc + " \\[" + ", ".join(parts) + "]"
                desc = desc + f" (seed={seed})"
                console.print(desc)
        return
    
    # Run batch
    runner = Runner(
        output_dir=config.logging.dir if config.logging else "runs",
        use_supabase=config.logging.supabase if config.logging else True,
        parallel=parallel,
    )
    
    results = runner.run_batch(config)
    
    # Print summary
    console.print("\n[bold green]Batch complete![/bold green]")
    console.print(f"Completed {len(results)} runs")
    
    # Aggregate some statistics
    if results:
        profits = [r[0].total_profit for r in results]
        avg_profit = sum(profits) / len(profits)
        console.print(f"[dim]Average profit: ${avg_profit / 100:.2f}[/dim]")


@app.command()
def compare(
    models: str = typer.Option(
        None,
        "--models", "-m",
        help="Comma-separated model patterns to compare (e.g., 'claude-*,gpt-*')"
    ),
    metric: str = typer.Option(
        "profit",
        "--metric",
        help="Metric to compare: profit, cups, reputation, cost"
    ),
    limit: int = typer.Option(
        20,
        "--limit", "-n",
        help="Number of runs to include"
    ),
):
    """
    Compare results across models.
    
    Example:
        lemonade compare --models "claude-*,gpt-*" --metric profit
    """
    from .reports import generate_comparison
    
    console.print(f"\n[bold blue]🍋 LemonadeBench Comparison[/bold blue]")
    
    try:
        comparison = generate_comparison(
            model_patterns=models.split(",") if models else None,
            metric=metric,
            limit=limit,
        )
        console.print(comparison)
    except Exception as e:
        console.print(f"[red]Error generating comparison: {escape(str(e))}[/red]")
        raise typer.Exit(1)


@app.command()
def leaderboard(
    top: int = typer.Option(
        10,
        "--top", "-n",
        help="Number of entries to show"
    ),
    metric: str = typer.Option(
        "profit",
        "--metric",
        help="Metric to rank by: profit, cups, reputation"
    ),
):
    """
    Show the model leaderboard.
    
    Example:
        lemonade leaderboard --top 10 --metric profit
    """
    from .reports import generate_leaderboard
    
    console.print(f"\n[bold blue]🍋 LemonadeBench Leaderboard[/bold blue]")
    console.print(f"[dim]Ranked by: {metric}[/dim]\n")
    
    try:
        leaderboard_table = generate_leaderboard(top=top, metric=metric)
        console.print(leaderboard_table)
    except Exception as e:
        console.print(f"[red]Error generating leaderboard: {escape(str(e))}[/red]")
        console.print("[dim]Make sure Supabase is configured and has run data.[/dim]")
        raise typer.Exit(1)


@app.command()
def validate(
    config_file: Path = typer.Argument(
        ...,
        help="YAML configuration file to validate"
    ),
):
    """
    Validate a batch configuration file.
    
    Checks that all model names are valid on OpenRouter and support
    function calling. Use this before running batch experiments.
    
    Example:
        lemonade validate examples/configs/paper_methodology.yaml
    """
    from .config import load_config
    from ..agents.providers.model_registry import ModelRegistry
    
    if not config_file.exists():
        console.print(f"[red]Config file not found: {config_file}[/red]")
        raise typer.Exit(1)
    
    console.print(f"\n[bold blue]🍋 LemonadeBench Config Validation[/bold blue]")
    console.print(f"[dim]Config: {config_file}[/dim]\n")
    
    # Load config
    config = load_config(config_file)
    
    # Get unique OpenRouter models
    openrouter_models = list(set(
        model.name for model in config.models 
        if model.provider == "openrouter"
    ))
    
    if not openrouter_models:
        console.print("[yellow]No OpenRouter models found in config[/yellow]")
        console.print("[dim]Validation skipped (only OpenRouter models are validated via API)[/dim]")
        return
    
    console.print(f"[dim]Found {len(openrouter_models)} unique OpenRouter models to validate...[/dim]\n")
    
    # Validate models
    registry = ModelRegistry()
    valid, failed = registry.validate_batch(openrouter_models, require_tools=True)
    
    # Show results
    if valid:
        console.print(f"[green]✓ Valid models ({len(valid)}):[/green]")
        for model_id in sorted(valid):
            info = registry.get_model(model_id)
            tools_status = "✓ tools" if info and info.supports_tools else "✗ no tools"
            ctx = f"{info.context_length:,}" if info else "?"
            console.print(f"  [green]✓[/green] {model_id} ({ctx} ctx, {tools_status})")
    
    if failed:
        console.print(f"\n[red]✗ Invalid models ({len(failed)}):[/red]")
        for result in failed:
            console.print(f"  [red]✗[/red] [bold]{result.model_id}[/bold]")
            if result.error_message:
                first_line = result.error_message.split('\n')[0]
                console.print(f"    {first_line}")
            if result.suggestions:
                console.print(f"    [dim]Did you mean: {', '.join(result.suggestions[:3])}[/dim]")
        
        console.print()
        console.print(f"[red bold]Validation FAILED[/red bold]")
        console.print("[dim]Fix the invalid model names and re-run validation.[/dim]")
        raise typer.Exit(1)
    
    console.print()
    console.print(f"[green bold]✓ All {len(valid)} models validated successfully![/green bold]")


@app.command()
def models():
    """
    List available models, providers, and configuration options.
    """
    console.print(f"\n[bold blue]🍋 LemonadeBench Configuration Options[/bold blue]\n")
    
    # Providers table
    table = Table(title="Supported Providers")
    table.add_column("Provider", style="cyan")
    table.add_column("Example Models", style="white")
    table.add_column("Env Variable", style="dim")
    
    table.add_row(
        "anthropic",
        "claude-sonnet-4-20250514, claude-3-5-haiku-20241022",
        "ANTHROPIC_API_KEY"
    )
    table.add_row(
        "openai",
        "gpt-4o, gpt-4o-mini, o1",
        "OPENAI_API_KEY"
    )
    table.add_row(
        "openrouter",
        "anthropic/claude-sonnet-4, openai/gpt-4o, deepseek/deepseek-r1",
        "OPENROUTER_API_KEY"
    )
    
    console.print(table)
    
    # Architectures table
    console.print()
    arch_table = Table(title="Agent Architectures")
    arch_table.add_column("Architecture", style="cyan")
    arch_table.add_column("Description", style="white")
    
    arch_table.add_row("react", "Observe → Decide → Act (baseline)")
    arch_table.add_row("plan_act", "Observe → Plan → Decide → Act")
    arch_table.add_row("act_reflect", "Observe → Decide → Act → Reflect")
    arch_table.add_row("full", "Observe → Reflect → Plan → Decide → Act")
    
    console.print(arch_table)
    
    # Goal framings table
    console.print()
    goal_table = Table(title="Goal Framing Conditions")
    goal_table.add_column("Condition", style="cyan")
    goal_table.add_column("Description", style="white")
    
    goal_table.add_row("baseline", "No additional framing (default)")
    goal_table.add_row("aggressive", "Risk-taking, maximize returns")
    goal_table.add_row("conservative", "Loss aversion, protect capital")
    goal_table.add_row("competitive", "Tournament framing, beat competitors")
    goal_table.add_row("survival", "Capital preservation priority")
    goal_table.add_row("growth", "Long-term learning and reputation focus")
    
    console.print(goal_table)
    
    # Tools table
    console.print()
    tools_table = Table(title="Cognitive Scaffolding Tools")
    tools_table.add_column("Tool", style="cyan")
    tools_table.add_column("Description", style="white")
    
    tools_table.add_row("calculator", "Basic arithmetic for business calculations")
    tools_table.add_row("code_interpreter", "Python execution for complex analysis")
    tools_table.add_row("--math-prompt", "Encourage step-by-step calculations (flag)")
    
    console.print(tools_table)
    
    console.print("\n[dim]Example: lemonade run --model gpt-4o --architecture plan_act --goal-framing competitive --tools calculator[/dim]")


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
