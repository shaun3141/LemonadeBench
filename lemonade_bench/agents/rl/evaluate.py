# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
Evaluation utilities for RL agents on LemonadeBench.

Provides functions to evaluate trained RL models and compare them against
various baselines (random, rule-based, optimal solver).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from ...server.lemonade_environment import LemonadeEnvironment
from ...models import LemonadeAction, Weather, BULK_PRICING
from .gym_wrapper import LemonadeGymEnv
from .spaces import decode_action


def _qty_to_tier_count(supply_type: str, qty: int) -> tuple[int, int]:
    """Helper to convert quantity to tier+count."""
    if qty <= 0:
        return (1, 0)
    pricing = BULK_PRICING.get(supply_type)
    if not pricing:
        return (1, 0)
    for i, tier in enumerate(pricing.tiers):
        if tier.quantity >= qty:
            return (i + 1, 1)
        if i == len(pricing.tiers) - 1:
            return (i + 1, (qty + tier.quantity - 1) // tier.quantity)
    return (1, qty)


@dataclass
class EvalResult:
    """Result of evaluating an agent over multiple episodes."""
    agent_name: str
    n_episodes: int
    mean_profit: float
    std_profit: float
    min_profit: float
    max_profit: float
    mean_cups_sold: float
    mean_reputation: float
    profits: list[float] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "agent_name": self.agent_name,
            "n_episodes": self.n_episodes,
            "mean_profit": self.mean_profit,
            "std_profit": self.std_profit,
            "min_profit": self.min_profit,
            "max_profit": self.max_profit,
            "mean_cups_sold": self.mean_cups_sold,
            "mean_reputation": self.mean_reputation,
            "profits": self.profits,
        }
    
    def __str__(self) -> str:
        return (
            f"{self.agent_name}:\n"
            f"  Profit: ${self.mean_profit/100:.2f} (+/- ${self.std_profit/100:.2f})\n"
            f"  Range: ${self.min_profit/100:.2f} to ${self.max_profit/100:.2f}\n"
            f"  Cups Sold: {self.mean_cups_sold:.1f} avg\n"
            f"  Reputation: {self.mean_reputation:.3f} avg"
        )


def evaluate_rl_model(
    model,
    n_episodes: int = 100,
    seed: int = 0,
    deterministic: bool = True,
    verbose: bool = False,
) -> EvalResult:
    """
    Evaluate a trained RL model.
    
    Args:
        model: Trained Stable Baselines3 model
        n_episodes: Number of evaluation episodes
        seed: Starting seed for reproducibility
        deterministic: Use deterministic actions
        verbose: Print progress
        
    Returns:
        EvalResult with aggregate statistics
    """
    profits = []
    cups_sold = []
    reputations = []
    
    for i in range(n_episodes):
        env = LemonadeGymEnv(seed=seed + i)
        obs, info = env.reset()
        
        episode_cups = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_cups += info.get("cups_sold", 0)
            
            if terminated or truncated:
                break
        
        profits.append(info["total_profit"])
        cups_sold.append(episode_cups)
        reputations.append(info["reputation"])
        env.close()
        
        if verbose and (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{n_episodes} episodes")
    
    mean_profit = np.mean(profits)
    std_profit = np.std(profits)
    
    return EvalResult(
        agent_name="RL Agent",
        n_episodes=n_episodes,
        mean_profit=mean_profit,
        std_profit=std_profit,
        min_profit=min(profits),
        max_profit=max(profits),
        mean_cups_sold=np.mean(cups_sold),
        mean_reputation=np.mean(reputations),
        profits=profits,
    )


def evaluate_random_agent(
    n_episodes: int = 100,
    seed: int = 0,
    verbose: bool = False,
) -> EvalResult:
    """
    Evaluate a random action agent.
    
    Args:
        n_episodes: Number of evaluation episodes
        seed: Starting seed for reproducibility
        verbose: Print progress
        
    Returns:
        EvalResult with aggregate statistics
    """
    profits = []
    cups_sold = []
    reputations = []
    
    rng = np.random.default_rng(seed)
    
    for i in range(n_episodes):
        env = LemonadeGymEnv(seed=seed + i)
        obs, info = env.reset()
        
        episode_cups = 0
        
        while True:
            # Random action in [0, 1] for each dimension
            action = rng.random(size=8).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_cups += info.get("cups_sold", 0)
            
            if terminated or truncated:
                break
        
        profits.append(info["total_profit"])
        cups_sold.append(episode_cups)
        reputations.append(info["reputation"])
        env.close()
        
        if verbose and (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{n_episodes} episodes")
    
    mean_profit = np.mean(profits)
    std_profit = np.std(profits)
    
    return EvalResult(
        agent_name="Random Agent",
        n_episodes=n_episodes,
        mean_profit=mean_profit,
        std_profit=std_profit,
        min_profit=min(profits),
        max_profit=max(profits),
        mean_cups_sold=np.mean(cups_sold),
        mean_reputation=np.mean(reputations),
        profits=profits,
    )


def evaluate_rule_based_agent(
    n_episodes: int = 100,
    seed: int = 0,
    verbose: bool = False,
) -> EvalResult:
    """
    Evaluate a simple rule-based agent that adjusts prices based on weather.
    
    This mirrors the strategy from examples/simple_agent.py.
    
    Args:
        n_episodes: Number of evaluation episodes
        seed: Starting seed for reproducibility
        verbose: Print progress
        
    Returns:
        EvalResult with aggregate statistics
    """
    profits = []
    cups_sold = []
    reputations = []
    
    for i in range(n_episodes):
        env = LemonadeEnvironment(seed=seed + i)
        obs = env.reset()
        
        episode_cups = 0
        
        while not obs.done:
            # Weather-based pricing strategy
            weather = obs.weather
            
            if weather == "hot":
                price = 125
            elif weather == "sunny":
                price = 100
            elif weather == "cloudy":
                price = 75
            else:  # rainy or stormy
                price = 50
            
            # Simple inventory management - convert to tier+count
            target_lemons = 15 if obs.lemons < 10 else 0
            target_sugar = 5 if obs.sugar_bags < 5 else 0
            target_cups = 50 if obs.cups_available < 30 else 0
            target_ice = 10 if obs.ice_bags < 5 and weather in ["hot", "sunny"] else 0
            
            lt, lc = _qty_to_tier_count("lemons", target_lemons)
            st, sc = _qty_to_tier_count("sugar", target_sugar)
            ct, cc = _qty_to_tier_count("cups", target_cups)
            it, ic = _qty_to_tier_count("ice", target_ice)
            
            # Advertising on good weather
            advertising = 100 if weather in ["hot", "sunny"] else 0
            
            action = LemonadeAction(
                price_per_cup=price,
                lemons_tier=lt, lemons_count=lc,
                sugar_tier=st, sugar_count=sc,
                cups_tier=ct, cups_count=cc,
                ice_tier=it, ice_count=ic,
                advertising_spend=advertising,
            )
            
            obs = env.step(action)
            episode_cups += obs.cups_sold
        
        profits.append(obs.total_profit)
        cups_sold.append(episode_cups)
        reputations.append(obs.reputation)
        
        if verbose and (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{n_episodes} episodes")
    
    mean_profit = np.mean(profits)
    std_profit = np.std(profits)
    
    return EvalResult(
        agent_name="Rule-Based Agent",
        n_episodes=n_episodes,
        mean_profit=mean_profit,
        std_profit=std_profit,
        min_profit=min(profits),
        max_profit=max(profits),
        mean_cups_sold=np.mean(cups_sold),
        mean_reputation=np.mean(reputations),
        profits=profits,
    )


def evaluate_constant_agent(
    price: int = 75,
    n_episodes: int = 100,
    seed: int = 0,
    verbose: bool = False,
) -> EvalResult:
    """
    Evaluate an agent that uses a constant price strategy.
    
    Args:
        price: Fixed price per cup in cents
        n_episodes: Number of evaluation episodes
        seed: Starting seed for reproducibility
        verbose: Print progress
        
    Returns:
        EvalResult with aggregate statistics
    """
    profits = []
    cups_sold = []
    reputations = []
    
    for i in range(n_episodes):
        env = LemonadeEnvironment(seed=seed + i)
        obs = env.reset()
        
        episode_cups = 0
        
        while not obs.done:
            # Fixed price, simple restocking - convert to tier+count
            target_lemons = 12 if obs.lemons < 10 else 0
            target_sugar = 5 if obs.sugar_bags < 5 else 0
            target_cups = 50 if obs.cups_available < 30 else 0
            
            lt, lc = _qty_to_tier_count("lemons", target_lemons)
            st, sc = _qty_to_tier_count("sugar", target_sugar)
            ct, cc = _qty_to_tier_count("cups", target_cups)
            
            action = LemonadeAction(
                price_per_cup=price,
                lemons_tier=lt, lemons_count=lc,
                sugar_tier=st, sugar_count=sc,
                cups_tier=ct, cups_count=cc,
            )
            
            obs = env.step(action)
            episode_cups += obs.cups_sold
        
        profits.append(obs.total_profit)
        cups_sold.append(episode_cups)
        reputations.append(obs.reputation)
        
        if verbose and (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{n_episodes} episodes")
    
    mean_profit = np.mean(profits)
    std_profit = np.std(profits)
    
    return EvalResult(
        agent_name=f"Constant ${price/100:.2f} Agent",
        n_episodes=n_episodes,
        mean_profit=mean_profit,
        std_profit=std_profit,
        min_profit=min(profits),
        max_profit=max(profits),
        mean_cups_sold=np.mean(cups_sold),
        mean_reputation=np.mean(reputations),
        profits=profits,
    )


def compare_agents(
    rl_model=None,
    n_episodes: int = 100,
    seed: int = 0,
    include_baselines: bool = True,
    verbose: bool = True,
    save_path: Optional[str] = None,
) -> list[EvalResult]:
    """
    Compare an RL agent against various baselines.
    
    Args:
        rl_model: Trained RL model (optional)
        n_episodes: Number of evaluation episodes per agent
        seed: Starting seed for reproducibility
        include_baselines: Include baseline agents in comparison
        verbose: Print progress and results
        save_path: Optional path to save results as JSON
        
    Returns:
        List of EvalResult for each agent
    """
    results = []
    
    if verbose:
        print("=" * 60)
        print("Agent Comparison on LemonadeBench")
        print("=" * 60)
        print(f"Episodes per agent: {n_episodes}")
        print()
    
    # Evaluate RL model if provided
    if rl_model is not None:
        if verbose:
            print("Evaluating RL Agent...")
        result = evaluate_rl_model(rl_model, n_episodes, seed, verbose=verbose)
        results.append(result)
    
    if include_baselines:
        # Random agent
        if verbose:
            print("Evaluating Random Agent...")
        result = evaluate_random_agent(n_episodes, seed, verbose=verbose)
        results.append(result)
        
        # Rule-based agent
        if verbose:
            print("Evaluating Rule-Based Agent...")
        result = evaluate_rule_based_agent(n_episodes, seed, verbose=verbose)
        results.append(result)
        
        # Constant price agents at different price points
        for price in [50, 75, 100]:
            if verbose:
                print(f"Evaluating Constant ${price/100:.2f} Agent...")
            result = evaluate_constant_agent(price, n_episodes, seed, verbose=verbose)
            results.append(result)
    
    # Sort by mean profit (descending)
    results.sort(key=lambda r: r.mean_profit, reverse=True)
    
    # Print results
    if verbose:
        print()
        print("=" * 60)
        print("Results (sorted by mean profit)")
        print("=" * 60)
        print()
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result}")
            print()
    
    # Save results if path provided
    if save_path:
        with open(save_path, "w") as f:
            json.dump(
                {
                    "n_episodes": n_episodes,
                    "seed": seed,
                    "results": [r.to_dict() for r in results],
                },
                f,
                indent=2,
            )
        if verbose:
            print(f"Results saved to: {save_path}")
    
    return results

