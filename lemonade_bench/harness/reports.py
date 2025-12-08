# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
Reporting utilities for LemonadeBench.

Generates comparison tables, leaderboards, and exports.
"""

import fnmatch
from typing import Any

from rich.table import Table
from rich.console import Console


def generate_leaderboard(
    top: int = 10,
    metric: str = "profit",
) -> Table:
    """
    Generate a leaderboard table from Supabase data.
    
    Args:
        top: Number of entries to show
        metric: Metric to rank by (profit, cups, reputation)
        
    Returns:
        Rich Table with leaderboard
    """
    from ..db import SupabaseReader
    
    reader = SupabaseReader()
    
    # Get best runs per model
    runs = reader.get_best_runs_per_model()
    
    # Sort by metric
    metric_map = {
        "profit": "total_profit",
        "cups": "total_cups_sold",
        "reputation": "final_reputation",
    }
    sort_key = metric_map.get(metric, "total_profit")
    
    runs = sorted(runs, key=lambda r: r.get(sort_key, 0), reverse=True)[:top]
    
    # Build table
    table = Table(title=f"🏆 Leaderboard (by {metric})")
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Model", style="cyan")
    table.add_column("Provider", style="dim")
    table.add_column("Profit", style="green", justify="right")
    table.add_column("Cups", justify="right")
    table.add_column("Rep.", justify="right")
    table.add_column("Runs", justify="right", style="dim")
    
    for i, run in enumerate(runs, 1):
        rank_str = f"#{i}"
        if i == 1:
            rank_str = "🥇"
        elif i == 2:
            rank_str = "🥈"
        elif i == 3:
            rank_str = "🥉"
        
        table.add_row(
            rank_str,
            run.get("model_name", "unknown"),
            run.get("provider", "unknown"),
            f"${run.get('total_profit', 0) / 100:.2f}",
            str(run.get("total_cups_sold", 0)),
            f"{run.get('final_reputation', 0):.2f}",
            str(run.get("run_count", 1)),
        )
    
    return table


def generate_comparison(
    model_patterns: list[str] | None = None,
    metric: str = "profit",
    limit: int = 20,
) -> Table:
    """
    Generate a comparison table for models.
    
    Args:
        model_patterns: List of glob patterns to filter models
        metric: Primary metric for comparison
        limit: Max runs to include
        
    Returns:
        Rich Table with comparison data
    """
    from ..db import SupabaseReader
    
    reader = SupabaseReader()
    runs = reader.get_all_runs(limit=limit * 2)  # Get extra for filtering
    
    # Filter by patterns if provided
    if model_patterns:
        filtered = []
        for run in runs:
            model_name = run.get("models", {}).get("name", "") if run.get("models") else ""
            for pattern in model_patterns:
                if fnmatch.fnmatch(model_name.lower(), pattern.lower()):
                    filtered.append(run)
                    break
        runs = filtered
    
    runs = runs[:limit]
    
    # Build table
    table = Table(title="📊 Model Comparison")
    table.add_column("Model", style="cyan")
    table.add_column("Seed", style="dim", justify="right")
    table.add_column("Profit", style="green", justify="right")
    table.add_column("Cups", justify="right")
    table.add_column("Cash", justify="right")
    table.add_column("Rep.", justify="right")
    table.add_column("Turns", justify="right", style="dim")
    table.add_column("Date", style="dim")
    
    for run in runs:
        model_info = run.get("models", {}) if run.get("models") else {}
        model_name = model_info.get("name", "unknown") if model_info else "unknown"
        
        completed = run.get("completed_at", "")
        if completed:
            # Format date nicely
            date_str = completed[:10] if len(completed) >= 10 else completed
        else:
            date_str = "-"
        
        table.add_row(
            model_name,
            str(run.get("seed", "-") or "-"),
            f"${run.get('total_profit', 0) / 100:.2f}",
            str(run.get("total_cups_sold", 0)),
            f"${run.get('final_cash', 0) / 100:.2f}",
            f"{run.get('final_reputation', 0):.2f}",
            str(run.get("turn_count", 0)),
            date_str,
        )
    
    return table


def export_results_csv(
    output_path: str,
    model_patterns: list[str] | None = None,
    limit: int = 1000,
) -> int:
    """
    Export results to CSV file.
    
    Args:
        output_path: Path for output CSV
        model_patterns: Optional patterns to filter models
        limit: Max results to export
        
    Returns:
        Number of rows exported
    """
    import csv
    from ..db import SupabaseReader
    
    reader = SupabaseReader()
    runs = reader.get_all_runs(limit=limit)
    
    # Filter if patterns provided
    if model_patterns:
        filtered = []
        for run in runs:
            model_name = run.get("models", {}).get("name", "") if run.get("models") else ""
            for pattern in model_patterns:
                if fnmatch.fnmatch(model_name.lower(), pattern.lower()):
                    filtered.append(run)
                    break
        runs = filtered
    
    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            "run_id",
            "model_name",
            "provider",
            "seed",
            "total_profit",
            "total_cups_sold",
            "final_cash",
            "final_reputation",
            "turn_count",
            "started_at",
            "completed_at",
        ])
        
        # Data
        for run in runs:
            model_info = run.get("models", {}) if run.get("models") else {}
            writer.writerow([
                run.get("id", ""),
                model_info.get("name", "") if model_info else "",
                model_info.get("provider", "") if model_info else "",
                run.get("seed", ""),
                run.get("total_profit", 0),
                run.get("total_cups_sold", 0),
                run.get("final_cash", 0),
                run.get("final_reputation", 0),
                run.get("turn_count", 0),
                run.get("started_at", ""),
                run.get("completed_at", ""),
            ])
    
    return len(runs)


def calculate_statistics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Calculate statistics for a list of runs.
    
    Args:
        runs: List of run dictionaries
        
    Returns:
        Dictionary with statistics
    """
    if not runs:
        return {}
    
    profits = [r.get("total_profit", 0) for r in runs]
    cups = [r.get("total_cups_sold", 0) for r in runs]
    reps = [r.get("final_reputation", 0) for r in runs]
    
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0
    
    def std(lst):
        if len(lst) < 2:
            return 0
        mean = avg(lst)
        variance = sum((x - mean) ** 2 for x in lst) / len(lst)
        return variance ** 0.5
    
    return {
        "count": len(runs),
        "profit_avg": avg(profits),
        "profit_std": std(profits),
        "profit_min": min(profits),
        "profit_max": max(profits),
        "cups_avg": avg(cups),
        "cups_std": std(cups),
        "reputation_avg": avg(reps),
    }

