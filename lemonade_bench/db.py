# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
Database helper module for storing benchmark runs in Supabase.

This module provides functions to store and retrieve benchmark data including:
- Models being benchmarked
- Run summaries (profit, cups sold, reputation, etc.)
- Turn-by-turn action/observation data

Usage:
    from lemonade_bench.db import SupabaseLogger

    # Initialize with service role key (for writes)
    logger = SupabaseLogger()

    # Log a complete run
    run_id = logger.create_run(model_name="claude-sonnet-4", provider="anthropic", seed=42)
    logger.log_turn(run_id, day=1, observation={...}, action={...}, reasoning="...", result={...})
    logger.complete_run(run_id, total_profit=5000, total_cups_sold=100, ...)
"""

import os
from datetime import datetime
from typing import Any

from supabase import create_client, Client


class SupabaseLogger:
    """Logger that saves benchmark runs to Supabase."""

    def __init__(self, url: str | None = None, key: str | None = None):
        """
        Initialize Supabase client.

        Args:
            url: Supabase project URL. Defaults to SUPABASE_URL env var.
            key: Supabase service role key. Defaults to SUPABASE_SERVICE_KEY env var.
        """
        self.url = url or os.environ.get("SUPABASE_URL")
        self.key = key or os.environ.get("SUPABASE_SERVICE_KEY")

        if not self.url or not self.key:
            raise ValueError(
                "Supabase credentials not found. Set SUPABASE_URL and SUPABASE_SERVICE_KEY "
                "environment variables or pass them directly."
            )

        self.client: Client = create_client(self.url, self.key)

    def get_or_create_model(self, name: str, provider: str) -> str:
        """
        Get existing model by name or create a new one.

        Args:
            name: Model name (e.g., "claude-sonnet-4-20250514")
            provider: Provider name (e.g., "anthropic", "openai")

        Returns:
            Model UUID
        """
        # Try to find existing model
        result = self.client.table("models").select("id").eq("name", name).execute()

        if result.data:
            return result.data[0]["id"]

        # Try to create new model, handle race condition
        try:
            result = self.client.table("models").insert({
                "name": name,
                "provider": provider,
            }).execute()
            return result.data[0]["id"]
        except Exception as e:
            # If duplicate key error, another process created it - fetch it
            if "duplicate key" in str(e) or "23505" in str(e):
                result = self.client.table("models").select("id").eq("name", name).execute()
                if result.data:
                    return result.data[0]["id"]
            raise

    def create_run(
        self,
        model_name: str,
        provider: str,
        seed: int | None = None,
        goal_framing: str = "baseline",
        architecture: str = "react",
        scaffolding: str = "none",
    ) -> str:
        """
        Create a new run record.

        Args:
            model_name: Name of the model being benchmarked
            provider: Provider of the model
            seed: Random seed used for the run
            goal_framing: Goal framing condition (baseline, aggressive, conservative, 
                         competitive, survival, growth)
            architecture: Agent architecture (react, plan_act, act_reflect, full)
            scaffolding: Cognitive scaffolding (none, calculator, math_prompt, code_interpreter)

        Returns:
            Run UUID
        """
        model_id = self.get_or_create_model(model_name, provider)

        result = self.client.table("runs").insert({
            "model_id": model_id,
            "seed": seed,
            "goal_framing": goal_framing,
            "architecture": architecture,
            "scaffolding": scaffolding,
            "total_profit": 0,
            "total_cups_sold": 0,
            "final_cash": 0,
            "final_reputation": 0.0,
            "turn_count": 0,
            "started_at": datetime.utcnow().isoformat(),
        }).execute()

        return result.data[0]["id"]

    def complete_run(
        self,
        run_id: str,
        total_profit: int,
        total_cups_sold: int,
        final_cash: int,
        final_reputation: float,
        turn_count: int,
        error_count: int = 0,
    ) -> None:
        """
        Update a run with final results.

        Args:
            run_id: UUID of the run to update
            total_profit: Total profit in cents
            total_cups_sold: Total cups sold
            final_cash: Final cash in cents
            final_reputation: Final reputation (0.0 to 1.0)
            turn_count: Number of turns played
            error_count: Total number of invalid action attempts
        """
        self.client.table("runs").update({
            "total_profit": total_profit,
            "total_cups_sold": total_cups_sold,
            "final_cash": final_cash,
            "final_reputation": final_reputation,
            "turn_count": turn_count,
            "error_count": error_count,
            "completed_at": datetime.utcnow().isoformat(),
        }).eq("id", run_id).execute()

    def log_turn(
        self,
        run_id: str,
        day: int,
        observation: dict[str, Any],
        action: dict[str, Any],
        reasoning: str | None,
        result: dict[str, Any],
        is_error: bool = False,
        error_messages: list[str] | None = None,
    ) -> str:
        """
        Log a single turn for a run.

        Args:
            run_id: UUID of the parent run
            day: Day number (1-14)
            observation: Observation data at start of turn
            action: Action taken by the agent
            reasoning: Agent's reasoning (if available)
            result: Result of the action
            is_error: True if this was an invalid action attempt
            error_messages: List of validation error messages if is_error=True

        Returns:
            Turn UUID
        """
        turn_data = {
            "run_id": run_id,
            "day": day,
            "observation": observation,
            "action": action,
            "reasoning": reasoning,
            "result": result,
            "is_error": is_error,
        }
        
        if error_messages:
            turn_data["error_messages"] = error_messages
        
        result_data = self.client.table("turns").insert(turn_data).execute()

        return result_data.data[0]["id"]

    def log_turns_batch(
        self,
        run_id: str,
        turns: list[dict[str, Any]],
    ) -> None:
        """
        Log multiple turns at once (more efficient).

        Args:
            run_id: UUID of the parent run
            turns: List of turn data dicts with keys: day, observation, action, reasoning, result
                   Optional keys: is_error, error_messages
        """
        rows = [
            {
                "run_id": run_id,
                "day": turn["day"],
                "observation": turn["observation"],
                "action": turn["action"],
                "reasoning": turn.get("reasoning"),
                "result": turn["result"],
                "is_error": turn.get("is_error", False),
                **({"error_messages": turn["error_messages"]} if turn.get("error_messages") else {}),
            }
            for turn in turns
        ]

        self.client.table("turns").insert(rows).execute()


    def run_exists(
        self,
        model_name: str,
        provider: str,
        seed: int | None = None,
        goal_framing: str = "baseline",
        architecture: str = "react",
        scaffolding: str = "none",
        completed_only: bool = True,
    ) -> bool:
        """
        Check if a run with the given configuration already exists.

        Args:
            model_name: Name of the model
            provider: Provider name
            seed: Random seed used
            goal_framing: Goal framing condition
            architecture: Agent architecture
            scaffolding: Cognitive scaffolding
            completed_only: If True, only count completed runs; if False, count any run

        Returns:
            True if a matching run exists
        """
        # First get the model_id
        model_result = self.client.table("models").select("id").eq("name", model_name).execute()
        if not model_result.data:
            return False
        
        model_id = model_result.data[0]["id"]
        
        # Check for existing run with matching config
        query = self.client.table("runs").select("id").eq("model_id", model_id)
        
        if seed is not None:
            query = query.eq("seed", seed)
        else:
            query = query.is_("seed", "null")
        
        query = query.eq("goal_framing", goal_framing)
        query = query.eq("architecture", architecture)
        query = query.eq("scaffolding", scaffolding)
        
        if completed_only:
            query = query.not_.is_("completed_at", "null")  # Only count completed runs
        
        result = query.execute()
        return len(result.data) > 0


class SupabaseReader:
    """Reader for fetching benchmark data from Supabase (uses anon key)."""

    def __init__(self, url: str | None = None, key: str | None = None):
        """
        Initialize Supabase client for reading.

        Args:
            url: Supabase project URL. Defaults to SUPABASE_URL env var.
            key: Supabase anon key. Defaults to SUPABASE_ANON_KEY env var.
        """
        self.url = url or os.environ.get("SUPABASE_URL")
        self.key = key or os.environ.get("SUPABASE_ANON_KEY")

        if not self.url or not self.key:
            raise ValueError(
                "Supabase credentials not found. Set SUPABASE_URL and SUPABASE_ANON_KEY "
                "environment variables or pass them directly."
            )

        self.client: Client = create_client(self.url, self.key)

    def get_best_runs_per_model(self) -> list[dict[str, Any]]:
        """
        Get the best run for each model.

        Returns:
            List of run records with model info, ordered by profit descending
        """
        result = self.client.table("best_runs_per_model").select("*").order(
            "total_profit", desc=True
        ).execute()

        return result.data

    def get_all_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        """
        Get all runs with model info.

        Args:
            limit: Maximum number of runs to return

        Returns:
            List of run records with model info
        """
        result = self.client.table("runs").select(
            "*, models(name, provider)"
        ).order("completed_at", desc=True).limit(limit).execute()

        return result.data

    def get_run_turns(self, run_id: str) -> list[dict[str, Any]]:
        """
        Get all turns for a specific run.

        Args:
            run_id: UUID of the run

        Returns:
            List of turn records ordered by day
        """
        result = self.client.table("turns").select("*").eq(
            "run_id", run_id
        ).order("day").execute()

        return result.data

    def get_models(self) -> list[dict[str, Any]]:
        """
        Get all models.

        Returns:
            List of model records
        """
        result = self.client.table("models").select("*").order("name").execute()

        return result.data
