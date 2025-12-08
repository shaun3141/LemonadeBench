# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
Optimal Solver for LemonadeBench using Beam Search.

This solver finds near-optimal strategies by:
1. Generating candidate actions at each step (discretized action space)
2. Simulating each candidate through the actual LemonadeEnvironment
3. Keeping top K trajectories by cumulative profit (beam search)
4. Returning the best complete trajectory found

The key insight is that the game is fully deterministic per seed, so we can
use search-based optimization rather than heuristics.
"""

import copy
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from itertools import product

from ..models import (
    LemonadeAction,
    LemonadeObservation,
    GameConfig,
    Weather,
    Location,
    LOCATION_CATALOG,
    StandUpgrade,
    UPGRADE_CATALOG,
    calculate_bulk_cost,
)
from ..server.lemonade_environment import LemonadeEnvironment


@dataclass
class DayResult:
    """Result of a single day's simulation."""
    day: int
    weather: str
    temperature: int
    action: dict
    cups_sold: int
    revenue: int
    costs: int
    profit: int
    cash_after: int
    inventory_after: dict
    reasoning: str = ""


@dataclass
class SolverResult:
    """Complete result from the optimal solver."""
    seed: int
    total_profit: int
    final_cash: int
    total_cups_sold: int
    days: list[DayResult] = field(default_factory=list)


@dataclass
class Trajectory:
    """A trajectory through the game (sequence of states and actions)."""
    env: LemonadeEnvironment
    obs: LemonadeObservation
    actions: list[LemonadeAction] = field(default_factory=list)
    day_results: list[DayResult] = field(default_factory=list)
    cumulative_profit: int = 0
    
    def clone(self) -> "Trajectory":
        """Create a deep copy of this trajectory."""
        new_env = LemonadeEnvironment(seed=self.env._seed)
        new_env.reset()
        
        # Replay all actions to get to the same state
        new_obs = new_env.reset()
        for action in self.actions:
            new_obs = new_env.step(action)
        
        return Trajectory(
            env=new_env,
            obs=new_obs,
            actions=list(self.actions),
            day_results=list(self.day_results),
            cumulative_profit=self.cumulative_profit,
        )


class OptimalSolver:
    """
    Near-optimal solver using beam search over the action space.
    
    For each day:
    1. Generate all candidate actions (discretized)
    2. For each beam trajectory, try each candidate action
    3. Keep top K trajectories by cumulative profit
    4. Continue to next day
    
    This is mathematically sound because we're doing exhaustive search
    over a discretized action space with pruning.
    """
    
    # Action space discretization
    PRICES = [50, 75, 100, 125, 150, 175, 200]  # cents (skip very low prices)
    LEMON_PURCHASES = [0, 12, 24]  # units (12+ gets 10% off)
    SUGAR_PURCHASES = [0, 5]  # bags (5+ gets 10% off)
    CUP_PURCHASES = [0, 50, 100]  # cups (50+ gets 10% off)
    ICE_PURCHASES = [0, 5, 10]  # bags (5+ gets 10% off)
    
    def __init__(
        self,
        seed: int,
        beam_width: int = 50,
        config: Optional[GameConfig] = None,
    ):
        """
        Initialize the beam search solver.
        
        Args:
            seed: Random seed for the game
            beam_width: Number of trajectories to keep at each step (higher = more thorough but slower)
            config: Optional game configuration
        """
        self.seed = seed
        self.beam_width = beam_width
        self.config = config or GameConfig()
    
    def generate_candidate_actions(
        self,
        obs: LemonadeObservation,
    ) -> list[LemonadeAction]:
        """
        Generate all candidate actions for the current state.
        
        Filters out clearly invalid actions (can't afford, etc.)
        """
        candidates = []
        cash = obs.cash
        current_location = Location(obs.current_location)
        has_cooler = "cooler" in obs.owned_upgrades
        
        # Determine which locations to consider
        locations_to_try = [None]  # None = stay at current location
        
        # Consider switching to pool on very hot days
        weather = Weather(obs.weather)
        if weather == Weather.HOT:
            pool_cost = LOCATION_CATALOG[Location.POOL].permit_cost
            if current_location != Location.POOL and cash >= pool_cost + 200:
                locations_to_try.append(Location.POOL.value)
        
        # Consider returning to park (free)
        if current_location != Location.PARK:
            locations_to_try.append(Location.PARK.value)
        
        # Determine upgrade options
        upgrades_to_try = [None]
        # Skip cooler - rarely worth it in a 14-day game
        
        # Generate all combinations
        for price in self.PRICES:
            for lemons in self.LEMON_PURCHASES:
                for sugar in self.SUGAR_PURCHASES:
                    for cups in self.CUP_PURCHASES:
                        for ice in self.ICE_PURCHASES:
                            for location in locations_to_try:
                                for upgrade in upgrades_to_try:
                                    # Calculate total cost
                                    total_cost = 0
                                    
                                    # Location cost
                                    if location:
                                        loc = Location(location)
                                        if loc != current_location:
                                            total_cost += LOCATION_CATALOG[loc].permit_cost
                                    
                                    # Purchase costs
                                    total_cost += calculate_bulk_cost("lemons", lemons)
                                    total_cost += calculate_bulk_cost("sugar", sugar)
                                    total_cost += calculate_bulk_cost("cups", cups)
                                    total_cost += calculate_bulk_cost("ice", ice)
                                    
                                    # Upgrade cost
                                    if upgrade:
                                        up = StandUpgrade(upgrade)
                                        if up not in [StandUpgrade(u) for u in obs.owned_upgrades]:
                                            total_cost += UPGRADE_CATALOG[up].cost
                                    
                                    # Skip if we can't afford it
                                    if total_cost > cash:
                                        continue
                                    
                                    # Create action
                                    action = LemonadeAction(
                                        price_per_cup=price,
                                        buy_lemons=lemons,
                                        buy_sugar=sugar,
                                        buy_cups=cups,
                                        buy_ice=ice,
                                        advertising_spend=0,
                                        buy_upgrade=upgrade,
                                        location=location,
                                    )
                                    candidates.append(action)
        
        # If no candidates (shouldn't happen), add a minimal action
        if not candidates:
            candidates.append(LemonadeAction(price_per_cup=50))
        
        return candidates
    
    def solve(self, verbose: bool = False) -> SolverResult:
        """
        Run beam search to find the optimal strategy.
        
        Returns:
            SolverResult with the best trajectory found
        """
        # Initialize with a single trajectory
        initial_env = LemonadeEnvironment(config=self.config, seed=self.seed)
        initial_obs = initial_env.reset()
        
        beams = [Trajectory(
            env=initial_env,
            obs=initial_obs,
            actions=[],
            day_results=[],
            cumulative_profit=0,
        )]
        
        day = 1
        while not beams[0].obs.done:
            if verbose:
                print(f"Day {day}: Exploring {len(beams)} beams...")
            
            # Generate all next-step candidates from all beams
            next_candidates = []
            
            for beam in beams:
                # Generate candidate actions for this beam's state
                candidates = self.generate_candidate_actions(beam.obs)
                
                if verbose:
                    print(f"  Beam (profit=${beam.cumulative_profit/100:.2f}): {len(candidates)} candidate actions")
                
                for action in candidates:
                    # Clone the trajectory and apply this action
                    new_traj = beam.clone()
                    
                    # Store pre-action state
                    pre_obs = new_traj.obs
                    weather = pre_obs.weather
                    temperature = pre_obs.temperature
                    
                    # Apply action
                    new_obs = new_traj.env.step(action)
                    
                    # Skip if action was invalid
                    if new_obs.is_error_response:
                        continue
                    
                    # Record the result
                    day_result = DayResult(
                        day=pre_obs.day,
                        weather=weather,
                        temperature=temperature,
                        action={
                            "price_per_cup": action.price_per_cup,
                            "buy_lemons": action.buy_lemons,
                            "buy_sugar": action.buy_sugar,
                            "buy_cups": action.buy_cups,
                            "buy_ice": action.buy_ice,
                            "advertising_spend": action.advertising_spend,
                            "buy_upgrade": action.buy_upgrade,
                            "location": action.location,
                        },
                        cups_sold=new_obs.cups_sold,
                        revenue=new_obs.daily_revenue,
                        costs=new_obs.daily_costs,
                        profit=new_obs.daily_profit,
                        cash_after=new_obs.cash,
                        inventory_after={
                            "lemons": new_obs.lemons,
                            "sugar": new_obs.sugar_bags,
                            "cups": new_obs.cups_available,
                            "ice": new_obs.ice_bags,
                        },
                    )
                    
                    new_traj.actions.append(action)
                    new_traj.day_results.append(day_result)
                    new_traj.obs = new_obs
                    new_traj.cumulative_profit += new_obs.daily_profit
                    
                    next_candidates.append(new_traj)
            
            # Keep top K trajectories by cumulative profit
            next_candidates.sort(key=lambda t: t.cumulative_profit, reverse=True)
            beams = next_candidates[:self.beam_width]
            
            if verbose:
                print(f"  Best beam: profit=${beams[0].cumulative_profit/100:.2f}, "
                      f"cash=${beams[0].obs.cash/100:.2f}")
            
            day += 1
        
        # Return the best trajectory
        best = beams[0]
        
        return SolverResult(
            seed=self.seed,
            total_profit=best.obs.total_profit,
            final_cash=best.obs.cash,
            total_cups_sold=sum(d.cups_sold for d in best.day_results),
            days=best.day_results,
        )
    
    def to_json(self, result: Optional[SolverResult] = None) -> dict:
        """
        Return results as JSON-serializable dict.
        
        Args:
            result: Pre-computed SolverResult. If None, solve() will be called.
        """
        if result is None:
            result = self.solve()
        return {
            "seed": result.seed,
            "total_profit": result.total_profit,
            "final_cash": result.final_cash,
            "total_cups_sold": result.total_cups_sold,
            "days": [asdict(d) for d in result.days],
        }


def solve_seed(
    seed: int,
    beam_width: int = 50,
    output_dir: Optional[Path] = None,
    verbose: bool = False,
) -> SolverResult:
    """
    Convenience function to solve a single seed and optionally save results.
    
    Args:
        seed: Random seed to solve
        beam_width: Number of trajectories to keep (higher = better but slower)
        output_dir: Optional directory to save JSON results
        verbose: Print progress during search
        
    Returns:
        SolverResult with complete solution
    """
    solver = OptimalSolver(seed=seed, beam_width=beam_width)
    result = solver.solve(verbose=verbose)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"seed_{seed}.json"
        with open(output_file, "w") as f:
            json.dump(solver.to_json(result), f, indent=2)
    
    return result
