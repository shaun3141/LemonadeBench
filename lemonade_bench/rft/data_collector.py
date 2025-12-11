# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
Trajectory data collection for Reinforcement Fine-Tuning.

Collects episodes from LemonadeBench using existing LLM agents
and formats them for fine-tuning.
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from ..agents.base import EpisodeResult
from ..agents.llm_agent import LLMAgent, format_observation
from ..agents.providers.base import LLMProvider
from ..client import LemonadeEnvironment
from .config import DataConfig


console = Console()


@dataclass
class TrajectoryTurn:
    """A single turn in a trajectory."""
    
    day: int
    observation_text: str
    action: dict[str, Any]
    reasoning: str
    cups_sold: int
    daily_profit: int
    daily_revenue: int
    daily_costs: int
    

@dataclass
class Trajectory:
    """A complete episode trajectory for training."""
    
    episode_id: str
    seed: int
    total_profit: int
    total_cups_sold: int
    final_cash: int
    final_reputation: float
    turn_count: int
    turns: list[TrajectoryTurn]
    
    # Source model info
    source_model: str
    source_provider: str
    
    # Metadata
    collected_at: str
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "episode_id": self.episode_id,
            "seed": self.seed,
            "total_profit": self.total_profit,
            "total_cups_sold": self.total_cups_sold,
            "final_cash": self.final_cash,
            "final_reputation": self.final_reputation,
            "turn_count": self.turn_count,
            "turns": [asdict(t) for t in self.turns],
            "source_model": self.source_model,
            "source_provider": self.source_provider,
            "collected_at": self.collected_at,
        }


def episode_to_trajectory(
    result: EpisodeResult,
    seed: int,
    episode_id: str,
) -> Trajectory:
    """
    Convert an EpisodeResult to a Trajectory for training.
    
    Args:
        result: Episode result from running an agent
        seed: Random seed used for the episode
        episode_id: Unique identifier for this trajectory
        
    Returns:
        Trajectory object ready for training data formatting
    """
    turns = []
    for turn in result.turns:
        # Format observation as text (what the model saw)
        obs_text = format_observation(turn.observation, is_initial=(turn.day == 1))
        
        # Convert action to dict
        action_dict = {
            "price_per_cup": turn.action.price_per_cup,
            "lemons_tier": turn.action.lemons_tier,
            "lemons_count": turn.action.lemons_count,
            "sugar_tier": turn.action.sugar_tier,
            "sugar_count": turn.action.sugar_count,
            "cups_tier": turn.action.cups_tier,
            "cups_count": turn.action.cups_count,
            "ice_tier": turn.action.ice_tier,
            "ice_count": turn.action.ice_count,
            "advertising_spend": turn.action.advertising_spend,
            "buy_upgrade": turn.action.buy_upgrade,
            "location": turn.action.location,
        }
        
        turns.append(TrajectoryTurn(
            day=turn.day,
            observation_text=obs_text,
            action=action_dict,
            reasoning=turn.reasoning,
            cups_sold=turn.cups_sold,
            daily_profit=turn.daily_profit,
            daily_revenue=turn.daily_revenue,
            daily_costs=turn.daily_costs,
        ))
    
    return Trajectory(
        episode_id=episode_id,
        seed=seed,
        total_profit=result.total_profit,
        total_cups_sold=result.total_cups_sold,
        final_cash=result.final_cash,
        final_reputation=result.final_reputation,
        turn_count=result.turn_count,
        turns=turns,
        source_model=result.model_name or "unknown",
        source_provider=result.provider or "unknown",
        collected_at=datetime.now().isoformat(),
    )


def collect_trajectories(
    provider: LLMProvider,
    config: DataConfig,
    show_progress: bool = True,
) -> list[Trajectory]:
    """
    Collect trajectory data by running an LLM agent.
    
    Args:
        provider: LLM provider to use for generating actions
        config: Data collection configuration
        show_progress: Whether to show progress bar
        
    Returns:
        List of collected trajectories
    """
    trajectories: list[Trajectory] = []
    
    # Create output directory
    output_dir = Path(config.trajectories_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create agent
    agent = LLMAgent(provider)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        disable=not show_progress,
    ) as progress:
        task = progress.add_task(
            f"[cyan]Collecting {config.num_episodes} trajectories...",
            total=config.num_episodes,
        )
        
        for i in range(config.num_episodes):
            # Use different seeds for diversity
            seed = (i % config.num_seeds) * 1000 + i
            
            try:
                # Create environment with seed
                env = LemonadeEnvironment(seed=seed)
                
                # Run episode
                result = agent.run_episode(env)
                
                # Apply filters
                if config.successful_only and result.total_profit <= 0:
                    progress.update(task, advance=1)
                    continue
                
                if config.min_profit_threshold is not None:
                    if result.total_profit < config.min_profit_threshold:
                        progress.update(task, advance=1)
                        continue
                
                if config.max_profit_threshold is not None:
                    if result.total_profit > config.max_profit_threshold:
                        progress.update(task, advance=1)
                        continue
                
                # Convert to trajectory
                episode_id = f"ep_{i:06d}_seed_{seed}"
                trajectory = episode_to_trajectory(result, seed, episode_id)
                trajectories.append(trajectory)
                
                # Save individual trajectory
                traj_path = output_dir / f"{episode_id}.json"
                with open(traj_path, "w") as f:
                    json.dump(trajectory.to_dict(), f, indent=2)
                
            except Exception as e:
                console.print(f"[red]Error collecting episode {i}: {e}[/red]")
            
            progress.update(task, advance=1)
    
    console.print(f"[green]Collected {len(trajectories)} trajectories[/green]")
    
    # Save manifest
    manifest = {
        "num_trajectories": len(trajectories),
        "config": asdict(config),
        "collected_at": datetime.now().isoformat(),
        "episodes": [t.episode_id for t in trajectories],
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    return trajectories


def load_trajectories(trajectories_dir: str | Path) -> list[Trajectory]:
    """
    Load previously collected trajectories from disk.
    
    Args:
        trajectories_dir: Directory containing trajectory JSON files
        
    Returns:
        List of Trajectory objects
    """
    trajectories_dir = Path(trajectories_dir)
    trajectories = []
    
    for traj_path in sorted(trajectories_dir.glob("ep_*.json")):
        with open(traj_path) as f:
            data = json.load(f)
        
        turns = [
            TrajectoryTurn(**t) for t in data["turns"]
        ]
        
        trajectory = Trajectory(
            episode_id=data["episode_id"],
            seed=data["seed"],
            total_profit=data["total_profit"],
            total_cups_sold=data["total_cups_sold"],
            final_cash=data["final_cash"],
            final_reputation=data["final_reputation"],
            turn_count=data["turn_count"],
            turns=turns,
            source_model=data["source_model"],
            source_provider=data["source_provider"],
            collected_at=data["collected_at"],
        )
        trajectories.append(trajectory)
    
    return trajectories

