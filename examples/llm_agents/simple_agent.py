#!/usr/bin/env python3
# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
Simple rule-based agent for LemonadeBench.

This demonstrates how to interact with the environment and provides
a baseline strategy that adjusts prices based on weather.
"""

import sys
from pathlib import Path

# Add the parent directory to the path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

from lemonade_bench import LemonadeAction, LemonadeObservation
from lemonade_bench.models import quantity_to_tier_count
from lemonade_bench.server.lemonade_environment import LemonadeEnvironment


def weather_based_strategy(obs: LemonadeObservation) -> LemonadeAction:
    """
    A simple rule-based strategy that adjusts prices based on weather.
    
    Strategy:
    - Hot/Sunny days: Higher prices (people want cold drinks)
    - Cloudy days: Moderate prices
    - Rainy/Stormy: Low prices to attract the few customers
    - Always maintain some inventory buffer
    
    Note: Cups are made on-demand, so we just need to ensure we have enough
    supplies to meet expected demand.
    """
    weather = obs.weather
    
    # Price strategy based on weather
    if weather == "hot":
        price = 125  # $1.25 - premium pricing
    elif weather == "sunny":
        price = 100  # $1.00
    elif weather == "cloudy":
        price = 75   # $0.75
    elif weather == "rainy":
        price = 50   # $0.50 - discount to attract customers
    else:  # stormy
        price = 50
    
    # Inventory management - buy enough supplies to meet expected demand
    target_lemons = 15 if obs.lemons < 10 else 0
    target_sugar = 5 if obs.sugar_bags < 5 else 0
    target_cups = 50 if obs.cups_available < 30 else 0
    
    # Convert quantities to optimal tier+count (bulk discounts auto-applied)
    lt, lc = quantity_to_tier_count("lemons", target_lemons)
    st, sc = quantity_to_tier_count("sugar", target_sugar)
    ct, cc = quantity_to_tier_count("cups", target_cups)
    
    # Advertising on good weather days
    advertising = 100 if weather in ["hot", "sunny"] else 0
    
    return LemonadeAction(
        price_per_cup=price,
        lemons_tier=lt, lemons_count=lc,
        sugar_tier=st, sugar_count=sc,
        cups_tier=ct, cups_count=cc,
        advertising_spend=advertising,
    )


def run_episode(seed: int = None, verbose: bool = True) -> float:
    """Run a single episode with the weather-based strategy."""
    env = LemonadeEnvironment(seed=seed)
    obs = env.reset()
    
    if verbose:
        print("=" * 60)
        print("🍋 LemonadeBench - Weather-Based Agent")
        print("=" * 60)
        print(f"Starting cash: ${obs.cash / 100:.2f}")
        print(f"Season length: {obs.days_remaining + 1} days")
        print()
    
    total_revenue = 0
    total_cups_sold = 0
    
    while not obs.done:
        if verbose:
            print(f"📅 Day {obs.day}: {obs.weather.upper()} ({obs.temperature}°F)")
            print(f"   Forecast: {obs.weather_forecast}")
            print(f"   Inventory: {obs.lemons} lemons, {obs.sugar_bags} sugar, {obs.cups_available} cups")
        
        action = weather_based_strategy(obs)
        
        if verbose:
            print(f"   Action: ${action.price_per_cup/100:.2f}/cup")
        
        obs = env.step(action)
        
        total_revenue += obs.daily_revenue
        total_cups_sold += obs.cups_sold
        
        if verbose:
            emoji = "📈" if obs.daily_profit > 0 else "📉"
            print(f"   {emoji} Sold {obs.cups_sold} cups, profit: ${obs.daily_profit/100:.2f}")
            if obs.customers_turned_away > 0:
                print(f"   ⚠️  Turned away {obs.customers_turned_away} customers (not enough supplies)!")
            print(f"   💰 Cash: ${obs.cash/100:.2f} | Reputation: {obs.reputation:.2f}")
            print()
    
    if verbose:
        print("=" * 60)
        print("🏆 GAME OVER!")
        print("=" * 60)
        print(f"Total Profit: ${obs.total_profit / 100:.2f}")
        print(f"Total Revenue: ${total_revenue / 100:.2f}")
        print(f"Total Cups Sold: {total_cups_sold}")
        print(f"Final Reputation: {obs.reputation:.2f}")
        print(f"Final Cash: ${obs.cash / 100:.2f}")
    
    return obs.total_profit


def benchmark(n_episodes: int = 100):
    """Run multiple episodes to benchmark the strategy."""
    print(f"Running {n_episodes} episodes...")
    
    profits = []
    for i in range(n_episodes):
        profit = run_episode(seed=i, verbose=False)
        profits.append(profit)
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{n_episodes} episodes")
    
    avg_profit = sum(profits) / len(profits)
    min_profit = min(profits)
    max_profit = max(profits)
    
    print()
    print("=" * 60)
    print("📊 Benchmark Results")
    print("=" * 60)
    print(f"Episodes: {n_episodes}")
    print(f"Average Profit: ${avg_profit / 100:.2f}")
    print(f"Min Profit: ${min_profit / 100:.2f}")
    print(f"Max Profit: ${max_profit / 100:.2f}")
    print(f"Std Dev: ${(sum((p - avg_profit)**2 for p in profits) / len(profits))**0.5 / 100:.2f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LemonadeBench with a simple agent")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark mode")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes for benchmark")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for single episode")
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark(args.episodes)
    else:
        run_episode(seed=args.seed)

