#!/usr/bin/env python3
# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
Run the Optimal Solver on paper methodology seeds.

This script computes near-optimal action sequences for all seeds used in
the paper methodology experiments and saves results as JSON files.

Usage:
    # Run on all paper methodology seeds
    uv run python examples/run_optimal_solver.py

    # Run on specific seeds
    uv run python examples/run_optimal_solver.py --seeds 1 42 100

    # Run on a single seed with verbose output
    uv run python examples/run_optimal_solver.py --seeds 42 --verbose
"""

import argparse
import json
import sys
from pathlib import Path

# Add the project root to the path for local development
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from lemonade_bench.agents import OptimalSolver, solve_seed


# Seeds from paper_methodology.yaml
# Main experiment seeds
PAPER_SEEDS = [1, 42, 100, 7, 2025]
# Additional seeds from ablation studies
ABLATION_SEEDS = [1, 2, 3, 42, 100, 7, 2025, 123, 456, 789]
# All unique seeds
ALL_SEEDS = sorted(set(PAPER_SEEDS + ABLATION_SEEDS))


def run_solver(
    seeds: list[int],
    output_dir: Path,
    beam_width: int = 50,
    verbose: bool = False,
    console: Console = None,
) -> dict[int, dict]:
    """
    Run the optimal solver on specified seeds.
    
    Args:
        seeds: List of seeds to solve
        output_dir: Directory to save JSON results
        beam_width: Number of trajectories to keep in beam search
        verbose: Print detailed output per day
        console: Rich console for output
        
    Returns:
        Dictionary mapping seed -> results
    """
    console = console or Console()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        disable=verbose,
    ) as progress:
        task = progress.add_task("Solving seeds...", total=len(seeds))
        
        for seed in seeds:
            progress.update(task, description=f"Solving seed {seed} (beam_width={beam_width})...")
            
            solver = OptimalSolver(seed=seed, beam_width=beam_width)
            result = solver.solve(verbose=verbose)
            json_result = solver.to_json(result)
            
            # Save to file
            output_file = output_dir / f"seed_{seed}.json"
            with open(output_file, "w") as f:
                json.dump(json_result, f, indent=2)
            
            all_results[seed] = json_result
            
            if verbose:
                console.print(f"\n[bold cyan]Seed {seed}[/bold cyan]")
                console.print(f"Total Profit: [green]${result.total_profit / 100:.2f}[/green]")
                console.print(f"Final Cash: ${result.final_cash / 100:.2f}")
                console.print(f"Cups Sold: {result.total_cups_sold}")
                console.print()
                
                # Day-by-day breakdown
                table = Table(title=f"Day-by-Day Results (Seed {seed})")
                table.add_column("Day", style="cyan", justify="right")
                table.add_column("Weather", style="yellow")
                table.add_column("Location", style="magenta")
                table.add_column("Price", justify="right")
                table.add_column("Sold", justify="right")
                table.add_column("Revenue", justify="right", style="green")
                table.add_column("Profit", justify="right", style="bold green")
                
                for day in result.days:
                    loc = day.action.get("location") or "stayed"
                    table.add_row(
                        str(day.day),
                        day.weather,
                        loc,
                        f"${day.action['price_per_cup']/100:.2f}",
                        str(day.cups_sold),
                        f"${day.revenue/100:.2f}",
                        f"${day.profit/100:.2f}",
                    )
                
                console.print(table)
                console.print()
            
            progress.advance(task)
    
    return all_results


def print_summary(results: dict[int, dict], console: Console):
    """Print a summary table of all results."""
    console.print()
    table = Table(title="Optimal Solver Results Summary")
    table.add_column("Seed", style="cyan", justify="right")
    table.add_column("Total Profit", justify="right", style="green")
    table.add_column("Final Cash", justify="right")
    table.add_column("Cups Sold", justify="right")
    
    profits = []
    for seed in sorted(results.keys()):
        r = results[seed]
        profit = r["total_profit"]
        profits.append(profit)
        table.add_row(
            str(seed),
            f"${profit / 100:.2f}",
            f"${r['final_cash'] / 100:.2f}",
            str(r["total_cups_sold"]),
        )
    
    console.print(table)
    
    # Statistics
    if profits:
        avg_profit = sum(profits) / len(profits)
        min_profit = min(profits)
        max_profit = max(profits)
        
        console.print()
        console.print("[bold]Statistics:[/bold]")
        console.print(f"  Average Profit: [green]${avg_profit / 100:.2f}[/green]")
        console.print(f"  Min Profit: ${min_profit / 100:.2f}")
        console.print(f"  Max Profit: ${max_profit / 100:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the Optimal Solver on LemonadeBench seeds",
        epilog="Results are saved to runs/optimal/ as JSON files.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=PAPER_SEEDS,
        help=f"Seeds to solve (default: paper seeds {PAPER_SEEDS})",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=f"Run on all seeds including ablation: {ALL_SEEDS}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/optimal"),
        help="Directory to save JSON results (default: runs/optimal)",
    )
    parser.add_argument(
        "--beam-width", "-b",
        type=int,
        default=50,
        help="Beam width for search (higher = better but slower, default: 50)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed day-by-day output",
    )
    args = parser.parse_args()
    
    console = Console()
    
    seeds = ALL_SEEDS if args.all else args.seeds
    
    console.print(f"[bold]LemonadeBench Optimal Solver (Beam Search)[/bold]")
    console.print(f"Seeds: {seeds}")
    console.print(f"Beam width: {args.beam_width}")
    console.print(f"Output: {args.output_dir}")
    console.print()
    
    results = run_solver(
        seeds=seeds,
        output_dir=args.output_dir,
        beam_width=args.beam_width,
        verbose=args.verbose,
        console=console,
    )
    
    print_summary(results, console)
    
    console.print()
    console.print(f"[dim]Results saved to {args.output_dir}/[/dim]")


if __name__ == "__main__":
    main()
