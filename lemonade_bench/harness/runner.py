# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
Run orchestration for LemonadeBench.

Handles single runs, batch execution, logging, and progress tracking.
Supports multiple agent architectures, goal-framing conditions, and
cognitive scaffolding options.
"""

import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal

from rich.console import Console
from rich.markup import escape
from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn

from ..agents.base import EpisodeResult, TurnResult, AgentCallback, LemonadeAgent
from ..models import LemonadeObservation
from .metrics import compute_diagnostic_metrics, DiagnosticMetrics


# Type aliases matching config types
ArchitectureType = Literal["react", "plan_act", "act_reflect", "full"]
GoalFramingType = Literal["baseline", "aggressive", "conservative", "competitive", "survival", "growth"]


@dataclass
class RunConfig:
    """Configuration for a single run."""
    provider: str
    model: str
    seed: int | None = None
    tags: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)  # Optional tools: ["calculator", "code_interpreter"]
    architecture: ArchitectureType = "react"  # Agent architecture
    goal_framing: GoalFramingType = "baseline"  # Goal framing condition
    math_prompt: bool = False  # Enable math encouragement scaffolding


def observation_to_dict(obs: LemonadeObservation) -> dict:
    """Convert observation to a JSON-serializable dict."""
    return {
        "day": obs.day,
        "weather": obs.weather,
        "temperature": obs.temperature,
        "weather_forecast": obs.weather_forecast,
        "cash": obs.cash,
        "daily_revenue": obs.daily_revenue,
        "daily_costs": obs.daily_costs,
        "daily_profit": obs.daily_profit,
        "cups_sold": obs.cups_sold,
        "customers_served": obs.customers_served,
        "customers_turned_away": obs.customers_turned_away,
        "lemons": obs.lemons,
        "sugar_bags": obs.sugar_bags,
        "cups_available": obs.cups_available,
        "ice_bags": obs.ice_bags,
        "lemons_spoiled": obs.lemons_spoiled,
        "ice_melted": obs.ice_melted,
        "reputation": obs.reputation,
        "total_profit": obs.total_profit,
        "done": obs.done,
    }


def action_to_dict(action) -> dict:
    """Convert action to a JSON-serializable dict."""
    return {
        "price_per_cup": action.price_per_cup,
        "buy_lemons": action.buy_lemons,
        "buy_sugar": action.buy_sugar,
        "buy_cups": action.buy_cups,
        "buy_ice": action.buy_ice,
        "advertising_spend": action.advertising_spend,
        "buy_upgrade": action.buy_upgrade,
        "location": action.location,
    }


class VerboseCallback:
    """Callback that prints verbose progress to console."""
    
    def __init__(self, console: Console | None = None):
        self.console = console or Console()
    
    def on_episode_start(self, observation: LemonadeObservation) -> None:
        self.console.print(f"[bold]═" * 50 + "[/bold]")
        self.console.print(f"[bold blue]🍋 Starting Episode[/bold blue]")
        self.console.print(f"[bold]═" * 50 + "[/bold]")
        self.console.print(f"Starting cash: ${observation.cash / 100:.2f}")
        self.console.print(f"Season length: {observation.days_remaining + 1} days")
        self.console.print()
    
    def on_turn_start(self, day: int, observation: LemonadeObservation) -> None:
        weather_emoji = {
            "hot": "🔥",
            "sunny": "☀️",
            "cloudy": "☁️",
            "rainy": "🌧️",
            "stormy": "⛈️",
        }.get(observation.weather.lower(), "🌡️")
        
        self.console.print(f"[cyan]📅 Day {day}:[/cyan] {weather_emoji} {observation.weather.upper()} ({observation.temperature}°F)")
        self.console.print(f"   [dim]Forecast: {observation.weather_forecast}[/dim]")
        self.console.print(f"   [dim]Inventory: {observation.lemons} lemons, {observation.sugar_bags:.1f} sugar, {observation.cups_available} cups, {observation.ice_bags} ice[/dim]")
    
    def on_turn_end(self, turn: TurnResult) -> None:
        from rich.markup import escape
        
        emoji = "📈" if turn.daily_profit > 0 else "📉"
        
        # Escape reasoning to prevent Rich markup interpretation
        safe_reasoning = escape(turn.reasoning) if turn.reasoning else ""
        self.console.print(f"   [dim]Strategy: {safe_reasoning}[/dim]")
        
        # Build action string
        action_parts = [f"${turn.action.price_per_cup/100:.2f}/cup"]
        if turn.action.buy_lemons > 0:
            action_parts.append(f"buy {turn.action.buy_lemons} lemons")
        if turn.action.buy_sugar > 0:
            action_parts.append(f"buy {turn.action.buy_sugar} sugar")
        if turn.action.buy_cups > 0:
            action_parts.append(f"buy {turn.action.buy_cups} cups")
        if turn.action.buy_ice > 0:
            action_parts.append(f"buy {turn.action.buy_ice} ice")
        self.console.print(f"   [dim]Action: {', '.join(action_parts)}[/dim]")
        
        self.console.print(f"   {emoji} Sold {turn.cups_sold} cups, profit: ${turn.daily_profit/100:.2f}")
        
        if turn.customers_turned_away > 0:
            self.console.print(f"   [yellow]⚠️  Turned away {turn.customers_turned_away} customers![/yellow]")
        
        self.console.print()
    
    def on_episode_end(self, result: EpisodeResult) -> None:
        pass  # Summary handled by CLI


class RunLogger:
    """Logs agent runs to disk and optionally to Supabase."""
    
    def __init__(
        self,
        base_dir: str = "runs",
        use_supabase: bool = True,
    ):
        # Include short UUID to prevent collisions when parallel workers start in the same second
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S") + f"_{uuid.uuid4().hex[:6]}"
        self.run_dir = Path(base_dir) / self.timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.turns_file = self.run_dir / "turns.jsonl"
        self.turns: list[dict] = []
        
        # Initialize Supabase logger if enabled
        self.supabase_logger = None
        self.supabase_run_id: str | None = None
        
        if use_supabase:
            if os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY"):
                from ..db import SupabaseLogger
                self.supabase_logger = SupabaseLogger()
    
    def save_config(self, config: dict[str, Any]):
        """Save run configuration."""
        config_file = self.run_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        
        # Create run in Supabase
        if self.supabase_logger:
            model_spec = config.get("model", "unknown")
            if "/" in model_spec:
                provider, model_name = model_spec.split("/", 1)
            else:
                model_name = model_spec
                provider = "anthropic" if "claude" in model_name.lower() else "unknown"
            
            # Determine scaffolding from config
            scaffolding = "none"
            tools = config.get("tools", [])
            if "code_interpreter" in tools:
                scaffolding = "code_interpreter"
            elif "calculator" in tools:
                scaffolding = "calculator"
            elif config.get("math_prompt"):
                scaffolding = "math_prompt"
            
            self.supabase_run_id = self.supabase_logger.create_run(
                model_name=model_name,
                provider=provider,
                seed=config.get("seed"),
                goal_framing=config.get("goal_framing", "baseline"),
                architecture=config.get("architecture", "react"),
                scaffolding=scaffolding,
            )
    
    def log_turn(
        self,
        day: int,
        observation: dict,
        action: dict,
        reasoning: str,
        result: dict,
        is_error: bool = False,
        error_messages: list[str] | None = None,
    ):
        """Log a single turn."""
        turn_data = {
            "day": day,
            "observation": observation,
            "action": action,
            "reasoning": reasoning,
            "result": result,
            "is_error": is_error,
        }
        if error_messages:
            turn_data["error_messages"] = error_messages
        
        self.turns.append(turn_data)
        
        # Append to JSONL file
        with open(self.turns_file, "a") as f:
            f.write(json.dumps(turn_data) + "\n")
        
        # Log to Supabase
        if self.supabase_logger and self.supabase_run_id:
            self.supabase_logger.log_turn(
                run_id=self.supabase_run_id,
                day=day,
                observation=observation,
                action=action,
                reasoning=reasoning,
                result=result,
                is_error=is_error,
                error_messages=error_messages,
            )
    
    def save_episode_result(self, result: EpisodeResult, run_config: RunConfig | None = None):
        """Save complete episode result with diagnostic metrics."""
        # Save successful turns
        for turn in result.turns:
            self.log_turn(
                day=turn.day,
                observation=observation_to_dict(turn.observation),
                action=action_to_dict(turn.action),
                reasoning=turn.reasoning,
                result={
                    "cups_sold": turn.cups_sold,
                    "daily_revenue": turn.daily_revenue,
                    "daily_profit": turn.daily_profit,
                    "customers_turned_away": turn.customers_turned_away,
                },
                is_error=turn.is_error,
                error_messages=turn.error_messages if turn.error_messages else None,
            )
        
        # Save error turns (invalid action attempts)
        for error_turn in result.error_turns:
            self.log_turn(
                day=error_turn.day,
                observation=observation_to_dict(error_turn.observation),
                action=action_to_dict(error_turn.action),
                reasoning=error_turn.reasoning,
                result={
                    "cups_sold": 0,
                    "daily_revenue": 0,
                    "daily_profit": 0,
                    "customers_turned_away": 0,
                },
                is_error=True,
                error_messages=error_turn.error_messages if error_turn.error_messages else None,
            )
        
        # Compute diagnostic metrics
        metrics = compute_diagnostic_metrics(result)
        
        # Save summary with metrics
        summary = {
            "total_profit": result.total_profit,
            "total_cups_sold": result.total_cups_sold,
            "final_cash": result.final_cash,
            "final_reputation": result.final_reputation,
            "turns": result.turn_count,
            "error_count": result.error_count,
            "model_name": result.model_name,
            "provider": result.provider,
            "total_input_tokens": result.total_input_tokens,
            "total_output_tokens": result.total_output_tokens,
            "estimated_cost_usd": result.estimated_cost_usd,
            # Diagnostic metrics
            "diagnostic_metrics": metrics.to_dict(),
        }
        
        # Add config info if available
        if run_config:
            summary["config"] = {
                "architecture": run_config.architecture,
                "goal_framing": run_config.goal_framing,
                "math_prompt": run_config.math_prompt,
                "tools": run_config.tools,
            }
        elif result.config:
            summary["config"] = result.config
        
        summary_file = self.run_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        # Save metrics separately for easy analysis
        metrics_file = self.run_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)
        
        # Save conversation
        if result.conversation_history:
            self._save_conversation(result.conversation_history)
        
        # Complete run in Supabase
        if self.supabase_logger and self.supabase_run_id:
            self.supabase_logger.complete_run(
                run_id=self.supabase_run_id,
                total_profit=result.total_profit,
                total_cups_sold=result.total_cups_sold,
                final_cash=result.final_cash,
                final_reputation=result.final_reputation,
                turn_count=result.turn_count,
                error_count=result.error_count,
            )
        
        return metrics
    
    def _save_conversation(self, messages: list[dict]):
        """Save the conversation history."""
        conv_file = self.run_dir / "conversation.json"
        
        # Make messages serializable
        serializable = []
        for msg in messages:
            serializable_msg = {"role": msg["role"]}
            if isinstance(msg["content"], str):
                serializable_msg["content"] = msg["content"]
            else:
                serializable_msg["content"] = []
                for block in msg["content"]:
                    if hasattr(block, "model_dump"):
                        serializable_msg["content"].append(block.model_dump())
                    elif isinstance(block, dict):
                        serializable_msg["content"].append(block)
                    elif hasattr(block, "__dict__"):
                        serializable_msg["content"].append(vars(block))
                    else:
                        serializable_msg["content"].append(str(block))
            serializable.append(serializable_msg)
        
        with open(conv_file, "w") as f:
            json.dump(serializable, f, indent=2)


class Runner:
    """
    Orchestrates batch runs with parallel execution and progress tracking.
    
    Supports multiple agent architectures, goal-framing conditions,
    and cognitive scaffolding options for systematic experiments.
    """
    
    def __init__(
        self,
        output_dir: str = "runs",
        use_supabase: bool = True,
        parallel: int = 1,
        skip_existing: bool = True,
    ):
        self.output_dir = output_dir
        self.use_supabase = use_supabase
        self.parallel = parallel
        self.skip_existing = skip_existing
        
        # Initialize Supabase logger for existence checks
        self._supabase_logger = None
        if use_supabase and skip_existing:
            if os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY"):
                from ..db import SupabaseLogger
                self._supabase_logger = SupabaseLogger()
    
    def _get_scaffolding(self, config: RunConfig) -> str:
        """Determine scaffolding type from config."""
        if "code_interpreter" in config.tools:
            return "code_interpreter"
        elif "calculator" in config.tools:
            return "calculator"
        elif config.math_prompt:
            return "math_prompt"
        return "none"
    
    def _run_exists(self, config: RunConfig, completed_only: bool = True) -> bool:
        """Check if a run with this configuration already exists in the database."""
        if not self._supabase_logger:
            return False
        
        return self._supabase_logger.run_exists(
            model_name=config.model,
            provider=config.provider,
            seed=config.seed,
            goal_framing=config.goal_framing,
            architecture=config.architecture,
            scaffolding=self._get_scaffolding(config),
            completed_only=completed_only,
        )
    
    def _create_provider(self, config: RunConfig):
        """Create an LLM provider from config."""
        if config.provider == "anthropic":
            from ..agents.providers import AnthropicProvider
            return AnthropicProvider(model=config.model)
        elif config.provider == "openai":
            from ..agents.providers.openai import OpenAIProvider
            return OpenAIProvider(model=config.model)
        elif config.provider == "openrouter":
            from ..agents.providers.openrouter import OpenRouterProvider
            return OpenRouterProvider(model=config.model)
        else:
            raise ValueError(f"Unknown provider: {config.provider}. Supported: anthropic, openai, openrouter")
    
    def _create_agent(self, config: RunConfig) -> LemonadeAgent:
        """
        Create an agent from config using the architecture factory.
        
        Uses the new architecture system to create agents with the specified
        architecture, goal framing, and scaffolding options.
        """
        from ..agents.architectures import create_agent
        
        provider = self._create_provider(config)
        
        return create_agent(
            provider=provider,
            architecture=config.architecture,
            goal_framing=config.goal_framing,
            tools=config.tools if config.tools else None,
            math_prompt=config.math_prompt,
        )
    
    def run_single(
        self,
        config: RunConfig,
        callbacks: list[AgentCallback] | None = None,
    ) -> tuple[EpisodeResult, DiagnosticMetrics]:
        """
        Run a single evaluation.
        
        Args:
            config: Run configuration
            callbacks: Optional callbacks for monitoring
            
        Returns:
            Tuple of (EpisodeResult, DiagnosticMetrics)
        """
        from ..server.lemonade_environment import LemonadeEnvironment
        
        agent = self._create_agent(config)
        env = LemonadeEnvironment(seed=config.seed)
        
        # Initialize logger
        logger = RunLogger(
            base_dir=self.output_dir,
            use_supabase=self.use_supabase,
        )
        logger.save_config({
            "model": f"{config.provider}/{config.model}",
            "seed": config.seed,
            "timestamp": logger.timestamp,
            "tags": config.tags,
            "tools": config.tools,
            "architecture": config.architecture,
            "goal_framing": config.goal_framing,
            "math_prompt": config.math_prompt,
        })
        
        # Run episode
        result = agent.run_episode(env, callbacks=callbacks)
        
        # Save results with metrics
        metrics = logger.save_episode_result(result, run_config=config)
        
        return result, metrics
    
    def run_batch(self, config) -> list[tuple[EpisodeResult, DiagnosticMetrics]]:
        """
        Run batch evaluations from a HarnessConfig.
        
        Args:
            config: HarnessConfig with model specifications
            
        Returns:
            List of (EpisodeResult, DiagnosticMetrics) tuples for each run
        """
        from .config import HarnessConfig
        
        # Build list of run configs
        all_run_configs: list[RunConfig] = []
        for model in config.models:
            seeds = model.seeds or [None]
            for seed in seeds:
                all_run_configs.append(RunConfig(
                    provider=model.provider,
                    model=model.name,
                    seed=seed,
                    tags=model.tags or [],
                    tools=model.tools or [],
                    architecture=model.architecture,
                    goal_framing=model.goal_framing,
                    math_prompt=model.math_prompt,
                ))
        
        # Deduplicate configs within the batch first (before DB check)
        # This handles overlapping experimental matrices (e.g., architecture + scaffolding ablations)
        seen_configs: set[tuple] = set()
        deduplicated_configs: list[RunConfig] = []
        duplicate_count = 0
        
        for rc in all_run_configs:
            # Create a hashable key for this config
            config_key = (
                rc.provider,
                rc.model,
                rc.seed,
                rc.goal_framing,
                rc.architecture,
                self._get_scaffolding(rc),  # Normalize tools/math_prompt to scaffolding type
            )
            
            if config_key in seen_configs:
                duplicate_count += 1
            else:
                seen_configs.add(config_key)
                deduplicated_configs.append(rc)
        
        console = Console()
        if duplicate_count > 0:
            console.print(f"[dim]Deduplicated {duplicate_count} overlapping configs from experimental matrix[/dim]")
        
        # Filter out existing runs if skip_existing is enabled
        run_configs: list[RunConfig] = []
        skipped_count = 0
        
        if self.skip_existing and self._supabase_logger:
            console.print("[dim]Checking for existing runs...[/dim]")
            for rc in deduplicated_configs:
                # Check for ANY run (including in-progress) to avoid duplicates with parallel execution
                if self._run_exists(rc, completed_only=False):
                    skipped_count += 1
                else:
                    run_configs.append(rc)
            
            if skipped_count > 0:
                console.print(f"[yellow]Skipping {skipped_count} existing runs, {len(run_configs)} remaining[/yellow]")
            else:
                console.print(f"[green]No existing runs found, running all {len(run_configs)}[/green]")
        else:
            run_configs = deduplicated_configs
        
        if not run_configs:
            console.print("[green]All runs already exist in database. Nothing to do![/green]")
            return []
        
        results: list[tuple[EpisodeResult, DiagnosticMetrics]] = []
        
        completed_count = 0
        
        if self.parallel <= 1:
            # Sequential execution with progress
            for run_config in run_configs:
                desc = f"{run_config.provider}/{run_config.model}"
                if run_config.architecture != "react":
                    desc += f" [{run_config.architecture}]"
                if run_config.goal_framing != "baseline":
                    desc += f" ({run_config.goal_framing})"
                
                console.print(f"[cyan]Starting:[/cyan] {desc} (seed={run_config.seed})")
                
                result, metrics = self.run_single(run_config)
                results.append((result, metrics))
                completed_count += 1
                
                profit_color = "green" if result.total_profit >= 0 else "red"
                console.print(f"  [{profit_color}]✓ Profit: ${result.total_profit/100:.2f}[/{profit_color}] ({completed_count}/{len(run_configs)})")
        else:
            # Parallel execution
            console.print(f"[dim]Running {len(run_configs)} evaluations with {self.parallel} workers...[/dim]")
            
            with ThreadPoolExecutor(max_workers=self.parallel) as executor:
                futures = {
                    executor.submit(self.run_single, rc): rc
                    for rc in run_configs
                }
                
                for future in as_completed(futures):
                    run_config = futures[future]
                    desc = f"{run_config.model}"
                    if run_config.goal_framing != "baseline":
                        desc += f" ({run_config.goal_framing})"
                    
                    try:
                        result, metrics = future.result()
                        results.append((result, metrics))
                        completed_count += 1
                        
                        profit_color = "green" if result.total_profit >= 0 else "red"
                        console.print(f"[{profit_color}]✓[/{profit_color}] {desc} seed={run_config.seed}: ${result.total_profit/100:.2f} ({completed_count}/{len(run_configs)})")
                    except Exception as e:
                        completed_count += 1
                        console.print(f"[red]✗ {desc} seed={run_config.seed}: {e}[/red] ({completed_count}/{len(run_configs)})")
        
        return results
