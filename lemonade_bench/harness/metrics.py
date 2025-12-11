# Copyright (c) 2024 LemonadeBench Contributors
# BSD-3-Clause License

"""
Diagnostic metrics computation for LemonadeBench.

Computes the diagnostic metrics described in the paper methodology (Section 4.1):
- Spoilage Rate
- Stockout Rate
- Weather Adaptation Score
- Price Volatility
- Recovery Score
- Location Efficiency

These metrics help understand *how* agents achieve (or fail to achieve) profit.
"""

import statistics
from dataclasses import dataclass, field
from typing import Any

from ..agents.base import EpisodeResult, TurnResult
from ..models import BULK_PRICING


def _get_purchased_quantity(action, supply_type: str) -> int:
    """Calculate the actual quantity purchased from tier+count."""
    tier = getattr(action, f"{supply_type}_tier", 1)
    count = getattr(action, f"{supply_type}_count", 0)
    if count <= 0:
        return 0
    pricing = BULK_PRICING.get(supply_type)
    if not pricing:
        return 0
    tier_idx = min(tier - 1, len(pricing.tiers) - 1)
    return pricing.tiers[tier_idx].quantity * count


# Weather-optimal price ranges (cents) based on game mechanics
# These represent the "sweet spot" prices for each weather condition
OPTIMAL_PRICE_RANGES = {
    "hot": (100, 150),      # High demand, premium prices
    "sunny": (75, 125),     # Good demand, standard prices
    "cloudy": (50, 100),    # Moderate demand
    "rainy": (40, 75),      # Low demand, lower prices
    "stormy": (25, 50),     # Very low demand, discount prices
}


@dataclass
class DiagnosticMetrics:
    """
    Diagnostic metrics for an episode.
    
    These metrics provide insight into agent behavior beyond raw profit.
    """
    # Primary metric
    total_profit: int  # cents
    
    # Inventory management
    spoilage_rate: float  # (lemons_spoiled + ice_melted) / total_perishables
    total_lemons_spoiled: int
    total_ice_melted: int
    total_perishables_purchased: int
    
    # Demand management
    stockout_rate: float  # customers_turned_away / total_potential_customers
    total_customers_served: int
    total_customers_turned_away: int
    
    # Weather responsiveness
    weather_adaptation_score: float  # Correlation between price and optimal price
    
    # Pricing behavior
    price_volatility: float  # Standard deviation of daily prices
    mean_price: float
    min_price: int
    max_price: int
    
    # Resilience
    recovery_score: float  # Average profit on days following a loss
    loss_days: int
    recovery_days: int
    
    # Location strategy
    location_efficiency: float  # Revenue per permit dollar spent
    
    # Error handling (action validation)
    error_count: int  # Total invalid action attempts
    error_rate: float  # error_count / total_action_attempts
    
    # Fields with defaults must come last
    turn_count: int = 0
    total_permit_cost: int = 0
    prices_by_weather: dict[str, list[int]] = field(default_factory=dict)
    locations_visited: list[str] = field(default_factory=list)
    location_revenue: dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to a dictionary for serialization."""
        return {
            "total_profit": self.total_profit,
            "spoilage_rate": self.spoilage_rate,
            "total_lemons_spoiled": self.total_lemons_spoiled,
            "total_ice_melted": self.total_ice_melted,
            "total_perishables_purchased": self.total_perishables_purchased,
            "stockout_rate": self.stockout_rate,
            "total_customers_served": self.total_customers_served,
            "total_customers_turned_away": self.total_customers_turned_away,
            "weather_adaptation_score": self.weather_adaptation_score,
            "price_volatility": self.price_volatility,
            "mean_price": self.mean_price,
            "min_price": self.min_price,
            "max_price": self.max_price,
            "recovery_score": self.recovery_score,
            "loss_days": self.loss_days,
            "recovery_days": self.recovery_days,
            "location_efficiency": self.location_efficiency,
            "locations_visited": self.locations_visited,
            "total_permit_cost": self.total_permit_cost,
            "turn_count": self.turn_count,
            "error_count": self.error_count,
            "error_rate": self.error_rate,
        }


def _compute_weather_adaptation(prices_by_weather: dict[str, list[int]]) -> float:
    """
    Compute weather adaptation score.
    
    Measures how well prices align with weather-optimal prices.
    Score of 1.0 means prices are always in the optimal range.
    Score of 0.0 means prices are never in the optimal range.
    
    Args:
        prices_by_weather: Dict mapping weather conditions to list of prices charged
        
    Returns:
        Float score between 0 and 1
    """
    if not prices_by_weather:
        return 0.0
    
    total_prices = 0
    optimal_prices = 0
    
    for weather, prices in prices_by_weather.items():
        weather_lower = weather.lower()
        if weather_lower not in OPTIMAL_PRICE_RANGES:
            continue
        
        opt_min, opt_max = OPTIMAL_PRICE_RANGES[weather_lower]
        
        for price in prices:
            total_prices += 1
            if opt_min <= price <= opt_max:
                optimal_prices += 1
            elif price < opt_min:
                # Partial credit for being close
                distance = (opt_min - price) / opt_min
                optimal_prices += max(0, 1 - distance)
            else:  # price > opt_max
                distance = (price - opt_max) / opt_max
                optimal_prices += max(0, 1 - distance)
    
    if total_prices == 0:
        return 0.0
    
    return optimal_prices / total_prices


def _compute_recovery_score(turns: list[TurnResult]) -> tuple[float, int, int]:
    """
    Compute recovery score - average profit on days following a loss.
    
    Args:
        turns: List of turn results
        
    Returns:
        Tuple of (recovery_score, loss_days, recovery_days)
    """
    loss_days = 0
    recovery_profits = []
    
    for i, turn in enumerate(turns):
        if turn.daily_profit < 0:
            loss_days += 1
            # Check if there's a next day
            if i + 1 < len(turns):
                recovery_profits.append(turns[i + 1].daily_profit)
    
    if not recovery_profits:
        return 0.0, loss_days, 0
    
    return statistics.mean(recovery_profits), loss_days, len(recovery_profits)


# Location permit costs (cents)
LOCATION_PERMIT_COSTS = {
    "park": 0,
    "downtown": 1000,  # $10
    "mall": 1500,      # $15
    "pool": 250,       # $2.50
}


def compute_diagnostic_metrics(result: EpisodeResult) -> DiagnosticMetrics:
    """
    Compute all diagnostic metrics from an episode result.
    
    Args:
        result: Complete episode result with all turn data
        
    Returns:
        DiagnosticMetrics instance with computed values
    """
    turns = result.turns
    
    # Compute error rate
    error_count = result.error_count
    total_action_attempts = len(turns) + error_count
    error_rate = error_count / total_action_attempts if total_action_attempts > 0 else 0.0
    
    if not turns:
        return DiagnosticMetrics(
            total_profit=result.total_profit,
            spoilage_rate=0.0,
            total_lemons_spoiled=0,
            total_ice_melted=0,
            total_perishables_purchased=0,
            stockout_rate=0.0,
            total_customers_served=0,
            total_customers_turned_away=0,
            weather_adaptation_score=0.0,
            price_volatility=0.0,
            mean_price=0.0,
            min_price=0,
            max_price=0,
            recovery_score=0.0,
            loss_days=0,
            recovery_days=0,
            location_efficiency=0.0,
            error_count=error_count,
            error_rate=error_rate,
            turn_count=0,
        )
    
    # Aggregate values across turns
    total_lemons_spoiled = 0
    total_ice_melted = 0
    total_lemons_purchased = 0
    total_ice_purchased = 0
    total_customers_served = 0
    total_customers_turned_away = 0
    
    prices: list[int] = []
    prices_by_weather: dict[str, list[int]] = {}
    locations_visited: set[str] = set()
    location_revenue: dict[str, int] = {}
    total_permit_cost = 0
    
    previous_location: str | None = None
    
    for turn in turns:
        obs = turn.observation
        action = turn.action
        
        # Inventory metrics
        total_lemons_spoiled += obs.lemons_spoiled
        total_ice_melted += obs.ice_melted
        total_lemons_purchased += _get_purchased_quantity(action, "lemons")
        total_ice_purchased += _get_purchased_quantity(action, "ice")
        
        # Demand metrics
        total_customers_served += turn.customers_served
        total_customers_turned_away += turn.customers_turned_away
        
        # Pricing metrics
        prices.append(action.price_per_cup)
        weather = obs.weather.lower()
        if weather not in prices_by_weather:
            prices_by_weather[weather] = []
        prices_by_weather[weather].append(action.price_per_cup)
        
        # Location metrics
        current_location = obs.current_location.lower()
        locations_visited.add(current_location)
        
        # Track revenue by location
        if current_location not in location_revenue:
            location_revenue[current_location] = 0
        location_revenue[current_location] += turn.daily_revenue
        
        # Track permit costs when changing location
        if action.location and action.location.lower() != current_location:
            new_location = action.location.lower()
            if new_location in LOCATION_PERMIT_COSTS:
                total_permit_cost += LOCATION_PERMIT_COSTS[new_location]
    
    # Compute spoilage rate
    total_perishables = total_lemons_purchased + total_ice_purchased
    spoilage_rate = 0.0
    if total_perishables > 0:
        spoilage_rate = (total_lemons_spoiled + total_ice_melted) / total_perishables
    
    # Compute stockout rate
    total_potential_customers = total_customers_served + total_customers_turned_away
    stockout_rate = 0.0
    if total_potential_customers > 0:
        stockout_rate = total_customers_turned_away / total_potential_customers
    
    # Compute weather adaptation
    weather_adaptation = _compute_weather_adaptation(prices_by_weather)
    
    # Compute price volatility
    price_volatility = 0.0
    mean_price = 0.0
    if len(prices) > 1:
        price_volatility = statistics.stdev(prices)
        mean_price = statistics.mean(prices)
    elif prices:
        mean_price = prices[0]
    
    # Compute recovery score
    recovery_score, loss_days, recovery_days = _compute_recovery_score(turns)
    
    # Compute location efficiency
    total_revenue = sum(location_revenue.values())
    location_efficiency = 0.0
    if total_permit_cost > 0:
        location_efficiency = total_revenue / total_permit_cost
    elif total_revenue > 0:
        # If no permits paid (only used park), efficiency is infinite
        # We cap at a high value to indicate good efficiency
        location_efficiency = float('inf')
    
    return DiagnosticMetrics(
        total_profit=result.total_profit,
        spoilage_rate=spoilage_rate,
        total_lemons_spoiled=total_lemons_spoiled,
        total_ice_melted=total_ice_melted,
        total_perishables_purchased=total_perishables,
        stockout_rate=stockout_rate,
        total_customers_served=total_customers_served,
        total_customers_turned_away=total_customers_turned_away,
        weather_adaptation_score=weather_adaptation,
        prices_by_weather=prices_by_weather,
        price_volatility=price_volatility,
        mean_price=mean_price,
        min_price=min(prices) if prices else 0,
        max_price=max(prices) if prices else 0,
        recovery_score=recovery_score,
        loss_days=loss_days,
        recovery_days=recovery_days,
        location_efficiency=location_efficiency if location_efficiency != float('inf') else -1,  # -1 indicates infinite (no permits)
        error_count=error_count,
        error_rate=error_rate,
        locations_visited=list(locations_visited),
        total_permit_cost=total_permit_cost,
        location_revenue=location_revenue,
        turn_count=len(turns),
    )


def format_metrics_summary(metrics: DiagnosticMetrics) -> str:
    """
    Format metrics as a human-readable summary.
    
    Args:
        metrics: Computed diagnostic metrics
        
    Returns:
        Formatted string summary
    """
    lines = [
        "=" * 50,
        "DIAGNOSTIC METRICS",
        "=" * 50,
        "",
        "INVENTORY MANAGEMENT",
        f"  Spoilage Rate: {metrics.spoilage_rate:.1%}",
        f"    Lemons Spoiled: {metrics.total_lemons_spoiled}",
        f"    Ice Melted: {metrics.total_ice_melted}",
        f"    Total Perishables: {metrics.total_perishables_purchased}",
        "",
        "DEMAND MANAGEMENT",
        f"  Stockout Rate: {metrics.stockout_rate:.1%}",
        f"    Customers Served: {metrics.total_customers_served}",
        f"    Customers Turned Away: {metrics.total_customers_turned_away}",
        "",
        "WEATHER RESPONSIVENESS",
        f"  Weather Adaptation Score: {metrics.weather_adaptation_score:.2f}",
        "",
        "PRICING BEHAVIOR",
        f"  Mean Price: ${metrics.mean_price / 100:.2f}",
        f"  Price Range: ${metrics.min_price / 100:.2f} - ${metrics.max_price / 100:.2f}",
        f"  Price Volatility (σ): {metrics.price_volatility:.1f}¢",
        "",
        "RESILIENCE",
        f"  Recovery Score: ${metrics.recovery_score / 100:.2f}",
        f"  Loss Days: {metrics.loss_days}",
        f"  Recovery Days: {metrics.recovery_days}",
        "",
        "LOCATION STRATEGY",
        f"  Locations Visited: {', '.join(metrics.locations_visited) if metrics.locations_visited else 'None'}",
        f"  Total Permit Cost: ${metrics.total_permit_cost / 100:.2f}",
        f"  Location Efficiency: {'∞ (no permits)' if metrics.location_efficiency < 0 else f'{metrics.location_efficiency:.1f}'}",
        "",
        "ACTION VALIDATION",
        f"  Error Rate: {metrics.error_rate:.1%}",
        f"  Invalid Actions: {metrics.error_count}",
        "",
        "=" * 50,
    ]
    
    return "\n".join(lines)
