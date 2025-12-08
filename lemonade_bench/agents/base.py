# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
Base agent classes for LemonadeBench.

Provides the abstract base class that all agents must implement, along with
data structures for tracking episode results and callbacks for monitoring.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from ..models import LemonadeAction, LemonadeObservation


@dataclass(kw_only=True)
class TurnResult:
    """Result of a single turn/day in the game."""
    day: int
    observation: LemonadeObservation
    action: LemonadeAction
    reasoning: str
    cups_sold: int
    daily_profit: int
    daily_revenue: int
    daily_costs: int
    customers_served: int
    customers_turned_away: int
    
    # Error tracking
    is_error: bool = False  # True if this was an invalid action attempt
    error_messages: list[str] = field(default_factory=list)  # Validation errors if is_error=True
    retry_count: int = 0  # Number of retries before success (0 = first attempt succeeded)


@dataclass(kw_only=True)
class EpisodeResult:
    """Complete result of running one episode (full game)."""
    total_profit: int
    total_cups_sold: int
    final_cash: int
    final_reputation: float
    turn_count: int
    turns: list[TurnResult] = field(default_factory=list)
    
    # Metadata
    model_name: str | None = None
    provider: str | None = None
    seed: int | None = None
    
    # Cost tracking (for LLM agents)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    
    # Error tracking
    error_count: int = 0  # Total number of invalid action attempts
    error_turns: list[TurnResult] = field(default_factory=list)  # All error attempts (for logging)
    
    # Raw data for logging
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)


class AgentCallback(Protocol):
    """Protocol for callbacks that monitor agent execution."""
    
    def on_turn_start(self, day: int, observation: LemonadeObservation) -> None:
        """Called at the start of each turn."""
        ...
    
    def on_turn_end(self, turn: TurnResult) -> None:
        """Called after each turn completes."""
        ...
    
    def on_episode_start(self, observation: LemonadeObservation) -> None:
        """Called when an episode starts."""
        ...
    
    def on_episode_end(self, result: EpisodeResult) -> None:
        """Called when an episode ends."""
        ...


class SimpleCallback:
    """Simple callback implementation that can be subclassed or used with lambdas."""
    
    def __init__(
        self,
        on_turn_start: Callable[[int, LemonadeObservation], None] | None = None,
        on_turn_end: Callable[[TurnResult], None] | None = None,
        on_episode_start: Callable[[LemonadeObservation], None] | None = None,
        on_episode_end: Callable[[EpisodeResult], None] | None = None,
    ):
        self._on_turn_start = on_turn_start
        self._on_turn_end = on_turn_end
        self._on_episode_start = on_episode_start
        self._on_episode_end = on_episode_end
    
    def on_turn_start(self, day: int, observation: LemonadeObservation) -> None:
        if self._on_turn_start:
            self._on_turn_start(day, observation)
    
    def on_turn_end(self, turn: TurnResult) -> None:
        if self._on_turn_end:
            self._on_turn_end(turn)
    
    def on_episode_start(self, observation: LemonadeObservation) -> None:
        if self._on_episode_start:
            self._on_episode_start(observation)
    
    def on_episode_end(self, result: EpisodeResult) -> None:
        if self._on_episode_end:
            self._on_episode_end(result)


class LemonadeAgent(ABC):
    """
    Abstract base class for Lemonade Stand agents.
    
    All agents must implement the `decide` method which takes an observation
    and returns an action with reasoning.
    
    The `run_episode` method provides a standard episode execution loop that
    calls `decide` for each turn and tracks results.
    """
    
    # Maximum retries per day for invalid actions
    MAX_RETRIES_PER_DAY = 3
    
    @abstractmethod
    def decide(self, observation: LemonadeObservation) -> tuple[LemonadeAction, str]:
        """
        Decide what action to take given the current observation.
        
        Args:
            observation: Current game state
            
        Returns:
            Tuple of (action, reasoning) where reasoning is a string explaining
            the decision (for logging/analysis)
        """
        pass
    
    def reset(self) -> None:
        """
        Reset agent state for a new episode.
        
        Override this in subclasses that maintain state (e.g., conversation history).
        """
        pass
    
    def run_episode(
        self,
        env,
        callbacks: list[AgentCallback] | None = None,
    ) -> EpisodeResult:
        """
        Run a complete episode using this agent.
        
        Handles action validation errors by allowing the agent to retry
        up to MAX_RETRIES_PER_DAY times per day. Error attempts are logged
        but do not consume game days.
        
        Args:
            env: LemonadeEnvironment instance
            callbacks: Optional list of callbacks for monitoring
            
        Returns:
            EpisodeResult with complete episode data including error tracking
        """
        callbacks = callbacks or []
        
        # Reset agent state
        self.reset()
        
        # Reset environment and get initial observation
        obs = env.reset()
        
        # Notify callbacks
        for cb in callbacks:
            cb.on_episode_start(obs)
        
        turns: list[TurnResult] = []
        error_turns: list[TurnResult] = []
        total_cups_sold = 0
        total_errors = 0
        
        while not obs.done:
            # Notify callbacks of turn start
            for cb in callbacks:
                cb.on_turn_start(obs.day, obs)
            
            # Retry loop for this day
            retries = 0
            pre_obs = obs
            
            while retries <= self.MAX_RETRIES_PER_DAY:
                # Get action from agent
                action, reasoning = self.decide(obs)
                
                # Execute action
                result_obs = env.step(action)
                
                # Check if action was valid
                if not result_obs.is_error_response:
                    # Valid action - proceed to next day
                    obs = result_obs
                    break
                
                # Invalid action - log error and retry
                error_turn = TurnResult(
                    day=pre_obs.day,
                    observation=pre_obs,
                    action=action,
                    reasoning=reasoning,
                    cups_sold=0,
                    daily_profit=0,
                    daily_revenue=0,
                    daily_costs=0,
                    customers_served=0,
                    customers_turned_away=0,
                    is_error=True,
                    error_messages=result_obs.action_errors,
                    retry_count=retries,
                )
                error_turns.append(error_turn)
                total_errors += 1
                
                # Feed error back to agent for retry
                obs = result_obs
                retries += 1
            
            # If max retries exceeded, force a minimal safe action
            if retries > self.MAX_RETRIES_PER_DAY:
                # Use minimal action that should always succeed
                safe_action = LemonadeAction(price_per_cup=50)
                obs = env.step(safe_action)
                action = safe_action
                reasoning = "[FALLBACK] Max retries exceeded, using safe default action"
            
            total_cups_sold += obs.cups_sold
            
            # Create turn result for successful action
            turn = TurnResult(
                day=pre_obs.day,
                observation=pre_obs,
                action=action,
                reasoning=reasoning,
                cups_sold=obs.cups_sold,
                daily_profit=obs.daily_profit,
                daily_revenue=obs.daily_revenue,
                daily_costs=obs.daily_costs,
                customers_served=obs.customers_served,
                customers_turned_away=obs.customers_turned_away,
                retry_count=retries,
            )
            turns.append(turn)
            
            # Notify callbacks of turn end
            for cb in callbacks:
                cb.on_turn_end(turn)
        
        # Build episode result
        result = EpisodeResult(
            total_profit=obs.total_profit,
            total_cups_sold=total_cups_sold,
            final_cash=obs.cash,
            final_reputation=obs.reputation,
            turn_count=len(turns),
            turns=turns,
            error_count=total_errors,
            error_turns=error_turns,
        )
        
        # Notify callbacks of episode end
        for cb in callbacks:
            cb.on_episode_end(result)
        
        return result
