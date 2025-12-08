#!/usr/bin/env python3
# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
Evaluate trained PPO model on paper methodology seeds and register as baseline.

This script evaluates the trained PPO agent on the exact seeds used in the
LemonadeBench paper experiments, enabling direct comparison with LLM agents.

Paper methodology seeds:
- Goal Framing study: [1, 42, 100, 7, 2025]
- Architecture/Scaffolding ablation: [1, 2, 3, 42, 100, 7, 2025, 123, 456, 789]

Usage:
    python examples/evaluate_ppo_baseline.py
    python examples/evaluate_ppo_baseline.py --model-path examples/rl_models/lemonade_ppo
"""

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np

# Add parent directory for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError:
    print("Error: stable-baselines3 is required.")
    print("Install with: pip install stable-baselines3")
    sys.exit(1)

from lemonade_bench.agents.rl import LemonadeGymEnv


# Paper methodology seeds
GOAL_FRAMING_SEEDS = [1, 42, 100, 7, 2025]
ABLATION_SEEDS = [1, 2, 3, 42, 100, 7, 2025, 123, 456, 789]
ALL_UNIQUE_SEEDS = sorted(set(GOAL_FRAMING_SEEDS + ABLATION_SEEDS))


@dataclass
class SeedResult:
    """Result for a single seed evaluation."""
    seed: int
    profit: float  # in cents
    cups_sold: int
    reputation: float
    daily_profits: list[float]
    
    def to_dict(self):
        return asdict(self)


@dataclass 
class PPOBaselineResult:
    """Complete PPO baseline evaluation result."""
    model_path: str
    total_seeds: int
    mean_profit: float
    std_profit: float
    min_profit: float
    max_profit: float
    mean_cups_sold: float
    mean_reputation: float
    seed_results: list[SeedResult]
    
    def to_dict(self):
        return {
            "model_path": self.model_path,
            "total_seeds": self.total_seeds,
            "mean_profit": self.mean_profit,
            "mean_profit_dollars": self.mean_profit / 100,
            "std_profit": self.std_profit,
            "std_profit_dollars": self.std_profit / 100,
            "min_profit": self.min_profit,
            "min_profit_dollars": self.min_profit / 100,
            "max_profit": self.max_profit,
            "max_profit_dollars": self.max_profit / 100,
            "mean_cups_sold": self.mean_cups_sold,
            "mean_reputation": self.mean_reputation,
            "seed_results": [r.to_dict() for r in self.seed_results],
        }


def evaluate_on_seed(
    model,
    seed: int,
    vec_normalize: Optional[VecNormalize] = None,
    verbose: bool = False,
) -> SeedResult:
    """
    Evaluate the PPO model on a specific seed.
    
    Uses randomize_seed=False to ensure deterministic evaluation on the exact seed.
    """
    # Create environment with fixed seed (no randomization for eval)
    env = LemonadeGymEnv(seed=seed, randomize_seed=False)
    
    # If we have VecNormalize stats, wrap appropriately
    if vec_normalize is not None:
        vec_env = DummyVecEnv([lambda: LemonadeGymEnv(seed=seed, randomize_seed=False)])
        eval_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, training=False)
        eval_env.obs_rms = vec_normalize.obs_rms
        eval_env.ret_rms = vec_normalize.ret_rms
        
        obs = eval_env.reset()
        daily_profits = []
        total_cups = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            
            daily_profits.append(info[0].get("daily_profit", 0))
            total_cups += info[0].get("cups_sold", 0)
            
            if done[0]:
                break
        
        result = SeedResult(
            seed=seed,
            profit=info[0]["total_profit"],
            cups_sold=total_cups,
            reputation=info[0]["reputation"],
            daily_profits=daily_profits,
        )
        eval_env.close()
    else:
        # Raw environment evaluation
        obs, info = env.reset()
        daily_profits = []
        total_cups = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            daily_profits.append(info.get("daily_profit", 0))
            total_cups += info.get("cups_sold", 0)
            
            if terminated or truncated:
                break
        
        result = SeedResult(
            seed=seed,
            profit=info["total_profit"],
            cups_sold=total_cups,
            reputation=info["reputation"],
            daily_profits=daily_profits,
        )
        env.close()
    
    if verbose:
        print(f"  Seed {seed}: ${result.profit/100:.2f} profit, {result.cups_sold} cups")
    
    return result


def evaluate_ppo_baseline(
    model_path: str,
    seeds: list[int] = None,
    verbose: bool = True,
) -> PPOBaselineResult:
    """
    Evaluate the trained PPO model on paper methodology seeds.
    
    Args:
        model_path: Path to saved PPO model (without .zip extension)
        seeds: List of seeds to evaluate on (default: all paper seeds)
        verbose: Print progress
        
    Returns:
        PPOBaselineResult with complete evaluation data
    """
    if seeds is None:
        seeds = ALL_UNIQUE_SEEDS
    
    # Load model
    model_file = Path(model_path)
    if not model_file.with_suffix(".zip").exists():
        raise FileNotFoundError(f"Model not found: {model_file}.zip")
    
    if verbose:
        print("=" * 70)
        print("PPO Baseline Evaluation on Paper Methodology Seeds")
        print("=" * 70)
        print(f"Model: {model_path}")
        print(f"Seeds: {seeds}")
        print()
    
    model = PPO.load(model_path)
    
    # Try to load VecNormalize stats
    vecnorm_path = Path(f"{model_path}_vecnormalize.pkl")
    vec_normalize = None
    if vecnorm_path.exists():
        if verbose:
            print(f"Loading VecNormalize stats from: {vecnorm_path}")
        dummy_env = DummyVecEnv([lambda: LemonadeGymEnv(seed=0, randomize_seed=False)])
        vec_normalize = VecNormalize.load(str(vecnorm_path), dummy_env)
        vec_normalize.training = False
        vec_normalize.norm_reward = False
    
    # Evaluate on each seed
    if verbose:
        print(f"\nEvaluating on {len(seeds)} seeds...")
    
    seed_results = []
    for seed in seeds:
        result = evaluate_on_seed(model, seed, vec_normalize, verbose=verbose)
        seed_results.append(result)
    
    # Compute aggregate statistics
    profits = [r.profit for r in seed_results]
    cups = [r.cups_sold for r in seed_results]
    reps = [r.reputation for r in seed_results]
    
    baseline_result = PPOBaselineResult(
        model_path=str(model_path),
        total_seeds=len(seeds),
        mean_profit=np.mean(profits),
        std_profit=np.std(profits),
        min_profit=min(profits),
        max_profit=max(profits),
        mean_cups_sold=np.mean(cups),
        mean_reputation=np.mean(reps),
        seed_results=seed_results,
    )
    
    if verbose:
        print()
        print("=" * 70)
        print("PPO Baseline Results")
        print("=" * 70)
        print(f"Seeds Evaluated: {baseline_result.total_seeds}")
        print(f"Mean Profit: ${baseline_result.mean_profit/100:.2f} (+/- ${baseline_result.std_profit/100:.2f})")
        print(f"Min Profit: ${baseline_result.min_profit/100:.2f}")
        print(f"Max Profit: ${baseline_result.max_profit/100:.2f}")
        print(f"Mean Cups Sold: {baseline_result.mean_cups_sold:.1f}")
        print(f"Mean Reputation: {baseline_result.mean_reputation:.3f}")
    
    return baseline_result


def save_baseline(result: PPOBaselineResult, output_path: str):
    """Save baseline results to JSON."""
    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"\nBaseline saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PPO model on paper methodology seeds"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="examples/rl_models/lemonade_ppo",
        help="Path to trained PPO model (default: examples/rl_models/lemonade_ppo)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="examples/rl_models/ppo_baseline_results.json",
        help="Output path for baseline results JSON",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="all",
        choices=["all", "goal_framing", "ablation"],
        help="Which seed set to evaluate on (default: all)",
    )
    
    args = parser.parse_args()
    
    # Select seeds
    if args.seeds == "goal_framing":
        seeds = GOAL_FRAMING_SEEDS
    elif args.seeds == "ablation":
        seeds = ABLATION_SEEDS
    else:
        seeds = ALL_UNIQUE_SEEDS
    
    # Evaluate
    result = evaluate_ppo_baseline(
        model_path=args.model_path,
        seeds=seeds,
        verbose=True,
    )
    
    # Save results
    save_baseline(result, args.output)
    
    # Print comparison format for paper
    print()
    print("=" * 70)
    print("Paper Comparison Format")
    print("=" * 70)
    print(f"PPO Baseline: ${result.mean_profit/100:.2f} ± ${result.std_profit/100:.2f}")
    print(f"  (n={result.total_seeds} seeds, range: ${result.min_profit/100:.2f} - ${result.max_profit/100:.2f})")


if __name__ == "__main__":
    main()

