# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
Lemonade Stand Environment Implementation.

A simulation of running a lemonade stand where the agent must:
- Set prices strategically based on weather
- Manage inventory efficiently
- Build reputation over time
- Maximize profit over a summer season
"""

import random
from typing import Optional
from uuid import uuid4

from openenv_core.env_server.interfaces import Environment
from openenv_core.env_server.types import State

# Support both in-repo and standalone imports
try:
    from models import LemonadeAction, LemonadeObservation, GameConfig, Weather, MarketHints, StandUpgrade, UPGRADE_CATALOG, BULK_PRICING, calculate_bulk_cost, Location, LOCATION_CATALOG
except ImportError:
    from ..models import LemonadeAction, LemonadeObservation, GameConfig, Weather, MarketHints, StandUpgrade, UPGRADE_CATALOG, BULK_PRICING, calculate_bulk_cost, Location, LOCATION_CATALOG


class LemonadeEnvironment(Environment):
    """
    A lemonade stand tycoon simulation environment.
    
    The agent runs a lemonade stand for a simulated summer season,
    making daily decisions about pricing and inventory.
    Success is measured by total profit at the end of the season.
    
    Game Mechanics:
    - Weather affects customer demand (hot/sunny = more customers)
    - Price affects conversion rate (lower price = more sales)
    - Reputation builds over time based on customer satisfaction
    - Cups are made on-demand - you serve customers up to your supply capacity
    - Running out of supplies means lost sales (customers turned away)
    
    Example:
        >>> env = LemonadeEnvironment()
        >>> obs = env.reset()
        >>> print(f"Day {obs.day}: {obs.weather}, {obs.temperature}°F")
        >>>
        >>> action = LemonadeAction(price_per_cup=75, buy_lemons=10)
        >>> obs = env.step(action)
        >>> print(f"Sold {obs.cups_sold} cups, profit: ${obs.daily_profit/100:.2f}")
    """
    
    def __init__(self, config: Optional[GameConfig] = None, seed: Optional[int] = None):
        """
        Initialize the lemonade stand environment.
        
        Args:
            config: Game configuration (uses defaults if None)
            seed: Random seed for reproducibility
        """
        self.config = config or GameConfig()
        self._seed = seed
        self._rng = random.Random(seed)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_game_state()
    
    def _reset_game_state(self):
        """Reset all game state variables."""
        self.day = 1
        self.cash = self.config.starting_cash
        self.sugar_bags: float = float(self.config.starting_sugar)  # Track as float for precision
        self.cups = self.config.starting_cups
        self.reputation = 0.5  # Start with neutral reputation
        self.total_profit = 0
        
        # Current location - start at configured location (no permit fee on day 1)
        try:
            self.current_location = Location(self.config.starting_location)
        except ValueError:
            self.current_location = Location.PARK  # Default fallback
        
        # Stand upgrades owned by the player
        self.owned_upgrades: set[StandUpgrade] = set()
        
        # Track perishable inventory with expiration
        # Each entry is (quantity, days_until_expiration)
        # Lemons: shelf life of 3 days
        self.lemon_batches: list[tuple[int, int]] = [(self.config.starting_lemons, self.config.lemon_shelf_life)]
        # Ice: melts based on cooler ownership (50% with cooler, 100% without)
        self.ice_batches: list[tuple[int, int]] = [(self.config.starting_ice, self.config.ice_shelf_life)]
        
        # Track spoilage for reporting
        self.lemons_spoiled_today = 0
        self.ice_melted_today = 0
        
        # Generate all daily randomness upfront for deterministic runs
        # This ensures the same seed produces identical results regardless of agent actions
        self._generate_daily_schedules()
    
    @property
    def lemons(self) -> int:
        """Total lemons across all batches."""
        return sum(qty for qty, _ in self.lemon_batches)
    
    @property
    def ice_bags(self) -> int:
        """Total ice bags across all batches."""
        return sum(qty for qty, _ in self.ice_batches)
    
    def _get_lemons_expiring_tomorrow(self) -> int:
        """Get count of lemons that will expire after tomorrow."""
        return sum(qty for qty, days in self.lemon_batches if days <= 1)
    
    def _process_expiration(self):
        """Process end-of-day expiration. Called at start of each day."""
        # Age all batches and remove expired ones
        self.lemons_spoiled_today = 0
        self.ice_melted_today = 0
        
        # Process lemons (expire after 3 days)
        new_lemon_batches = []
        for qty, days in self.lemon_batches:
            if days <= 1:
                self.lemons_spoiled_today += qty
            else:
                new_lemon_batches.append((qty, days - 1))
        self.lemon_batches = new_lemon_batches
        
        # Process ice based on cooler ownership
        if StandUpgrade.COOLER in self.owned_upgrades:
            # Cooler preserves 50% of ice per day
            new_ice_batches = []
            for qty, _ in self.ice_batches:
                melted = qty - (qty // 2)  # Melt 50% (round up melted amount)
                remaining = qty // 2  # Keep 50% (round down preserved amount)
                self.ice_melted_today += melted
                if remaining > 0:
                    new_ice_batches.append((remaining, 1))  # Reset shelf life
            self.ice_batches = new_ice_batches
        else:
            # No cooler - all ice melts overnight
            for qty, _ in self.ice_batches:
                self.ice_melted_today += qty
            self.ice_batches = []
    
    def _consume_lemons(self, amount: float) -> float:
        """Consume lemons from oldest batches first (FIFO). Returns actual consumed."""
        remaining = amount
        new_batches = []
        
        for qty, days in self.lemon_batches:
            if remaining <= 0:
                new_batches.append((qty, days))
            elif qty <= remaining:
                remaining -= qty
            else:
                new_batches.append((qty - remaining, days))
                remaining = 0
        
        self.lemon_batches = new_batches
        return amount - remaining
    
    def _consume_ice(self, amount: float) -> float:
        """Consume ice from oldest batches first (FIFO). Returns actual consumed."""
        remaining = amount
        new_batches = []
        
        for qty, days in self.ice_batches:
            if remaining <= 0:
                new_batches.append((qty, days))
            elif qty <= remaining:
                remaining -= qty
            else:
                new_batches.append((qty - remaining, days))
                remaining = 0
        
        self.ice_batches = new_batches
        return amount - remaining
    
    def _generate_daily_schedules(self):
        """
        Generate all daily randomness upfront for the entire season.
        
        This ensures 100% deterministic runs when using the same seed.
        All random elements (weather, foot traffic, conversion) are pre-computed
        so the same seed always produces identical game states regardless of
        what actions the agent takes.
        """
        weather_weights = {
            Weather.SUNNY: 0.35,
            Weather.HOT: 0.20,
            Weather.CLOUDY: 0.25,
            Weather.RAINY: 0.15,
            Weather.STORMY: 0.05,
        }
        
        self._weather_schedule = []
        self._foot_traffic_modifiers = []  # Pre-computed ±10% modifiers
        self._conversion_modifiers = []    # Pre-computed ±5% modifiers
        
        for _ in range(self.config.total_days + 1):  # +1 for forecast
            # Generate weather
            weather = self._rng.choices(
                list(weather_weights.keys()),
                weights=list(weather_weights.values())
            )[0]
            
            # Temperature based on weather
            if weather == Weather.HOT:
                temp = self._rng.randint(90, 105)
            elif weather == Weather.SUNNY:
                temp = self._rng.randint(75, 90)
            elif weather == Weather.CLOUDY:
                temp = self._rng.randint(65, 80)
            elif weather == Weather.RAINY:
                temp = self._rng.randint(55, 70)
            else:  # STORMY
                temp = self._rng.randint(50, 65)
            
            self._weather_schedule.append((weather, temp))
            
            # Generate foot traffic modifier (±10% randomness)
            foot_traffic_mod = self._rng.uniform(0.9, 1.1)
            self._foot_traffic_modifiers.append(foot_traffic_mod)
            
            # Generate conversion modifier (±5% randomness)
            conversion_mod = self._rng.uniform(0.95, 1.05)
            self._conversion_modifiers.append(conversion_mod)
    
    def _get_weather_multiplier(self, weather: Weather, temperature: int, location: Optional[Location] = None) -> float:
        """
        Get customer demand multiplier based on weather, blended with location's weather exposure.
        
        Args:
            weather: Current weather condition
            temperature: Current temperature in Fahrenheit
            location: Optional location to get weather exposure from (uses current_location if None)
        
        Returns:
            Weather multiplier adjusted for location's weather exposure
        """
        base_multipliers = {
            Weather.HOT: 1.8,
            Weather.SUNNY: 1.3,
            Weather.CLOUDY: 0.9,
            Weather.RAINY: 0.4,
            Weather.STORMY: 0.1,
        }
        
        multiplier = base_multipliers[weather]
        
        # Temperature bonus/penalty
        if temperature > 85:
            multiplier *= 1.0 + (temperature - 85) * 0.02
        elif temperature < 60:
            multiplier *= 0.5
        
        # Apply location's weather exposure
        # weather_exposure: 0.0 = indoor (no effect), 1.0 = full effect, >1.0 = amplified
        loc = location or self.current_location
        location_info = LOCATION_CATALOG.get(loc)
        if location_info:
            weather_exposure = location_info.weather_exposure
            # Blend between 1.0 (no weather effect) and the multiplier based on exposure
            # At exposure=0, multiplier becomes 1.0 (neutral)
            # At exposure=1, multiplier stays as-is
            # At exposure>1, multiplier is amplified (good weather is better, bad is worse)
            multiplier = 1.0 + (multiplier - 1.0) * weather_exposure
        
        return multiplier
    
    def _calculate_foot_traffic(
        self,
        weather: Weather,
        temperature: int,
        advertising: int,
        location: Optional[Location] = None
    ) -> int:
        """
        Calculate how many people stop by the stand (before price consideration).
        
        Foot traffic is affected by:
        - Location's foot traffic multiplier
        - Weather (blended with location's weather exposure)
        - Reputation
        - Advertising
        - Pre-computed daily randomness modifier (±10%)
        
        Returns the number of potential customers who visit the stand.
        """
        loc = location or self.current_location
        location_info = LOCATION_CATALOG.get(loc)
        
        # Base foot traffic with location multiplier
        traffic = float(self.config.base_customers)
        if location_info:
            traffic *= location_info.foot_traffic_multiplier
        
        # Weather effect (blended with location's weather exposure)
        traffic *= self._get_weather_multiplier(weather, temperature, loc)
        
        # Reputation effect (0.5x to 1.5x based on reputation)
        traffic *= (0.5 + self.reputation)
        
        # Advertising effect (diminishing returns)
        if advertising > 0:
            ad_bonus = min(0.5, (advertising / 100) ** 0.5 * 0.3)
            traffic *= (1 + ad_bonus)
        
        # Apply pre-computed daily randomness (±10%) - deterministic for same seed
        day_index = min(self.day - 1, len(self._foot_traffic_modifiers) - 1)
        traffic *= self._foot_traffic_modifiers[day_index]
        
        return max(0, int(traffic))
    
    def _calculate_conversion_rate(
        self,
        price: int,
        weather: Weather,
        has_ice: bool = True,
        location: Optional[Location] = None
    ) -> float:
        """
        Calculate what percentage of foot traffic will buy (0.0 to 1.0).
        
        Conversion is affected by:
        - Price (using location's price sensitivity)
        - Ice availability on hot days
        - Pre-computed daily randomness modifier (±5%)
        
        Returns a conversion rate between 0.0 and 1.0.
        """
        loc = location or self.current_location
        location_info = LOCATION_CATALOG.get(loc)
        is_hot_weather = weather in [Weather.HOT, Weather.SUNNY]
        
        # Determine price sensitivity for this location
        if location_info:
            if is_hot_weather and location_info.hot_weather_price_sensitivity is not None:
                price_sensitivity = location_info.hot_weather_price_sensitivity
            else:
                price_sensitivity = location_info.price_sensitivity
        else:
            price_sensitivity = self.config.price_sensitivity
        
        # Base conversion rate at optimal price ($0.50) is 95%
        conversion = 0.95
        
        if price > 50:  # Above $0.50
            # Gentler price curve using power function for smoother falloff
            price_delta = (price - 50) / 100  # Normalize: 0 at $0.50, 1.5 at $2.00
            price_factor = 1.0 - (price_delta ** 0.7) * price_sensitivity * 50
            price_factor = max(0.1, price_factor)  # At least 10% conversion
            conversion *= price_factor
        
        # Very high prices kill conversion
        if price > self.config.max_price_tolerance:
            conversion *= 0.05
        
        # Ice bonus/penalty on hot days
        if is_hot_weather:
            if has_ice:
                # Ice bonus: +20% conversion on hot days (multiplicative)
                conversion *= (1 + self.config.ice_demand_bonus)
            else:
                # No ice penalty: -20% conversion on hot days
                conversion *= 0.8
        
        # Apply pre-computed daily randomness (±5%) - deterministic for same seed
        day_index = min(self.day - 1, len(self._conversion_modifiers) - 1)
        conversion *= self._conversion_modifiers[day_index]
        
        # Clamp to valid range
        return max(0.0, min(1.0, conversion))
    
    def _calculate_demand(
        self,
        price: int,
        weather: Weather,
        temperature: int,
        advertising: int,
        has_ice: bool = True,
        location: Optional[Location] = None
    ) -> int:
        """
        Calculate how many customers want to buy lemonade.
        
        This is a two-stage model:
        1. Foot traffic: People who stop by the stand
        2. Conversion rate: Percentage of visitors who buy
        
        Returns: foot_traffic * conversion_rate
        """
        foot_traffic = self._calculate_foot_traffic(
            weather, temperature, advertising, location
        )
        conversion_rate = self._calculate_conversion_rate(
            price, weather, has_ice, location
        )
        
        return max(0, int(foot_traffic * conversion_rate))
    
    def _calculate_satisfaction(
        self,
        price: int,
        cups_sold: int,
        customers_turned_away: int,
        weather: Weather
    ) -> float:
        """Calculate customer satisfaction for the day."""
        if cups_sold == 0:
            return 0.5  # Neutral if no sales
        
        # Base satisfaction from price (lower = happier)
        price_satisfaction = max(0, 1 - (price - 50) / 150)
        
        # Penalty for turning customers away
        if customers_turned_away > 0:
            turnaway_penalty = customers_turned_away / (cups_sold + customers_turned_away)
            price_satisfaction *= (1 - turnaway_penalty * 0.5)
        
        # Weather bonus (customers appreciate lemonade more when it's hot)
        if weather in [Weather.HOT, Weather.SUNNY]:
            price_satisfaction *= 1.1
        
        return min(1.0, max(0.0, price_satisfaction))
    
    def _calculate_market_hints(
        self,
        weather: Weather,
        temperature: int
    ) -> dict:
        """
        Calculate market intelligence to help players make decisions.
        
        This surfaces the hidden game mechanics so players can learn strategically.
        Uses a two-stage model:
        1. Foot traffic: People who stop by (affected by location, weather, reputation, ads)
        2. Conversion rate: % who buy at each price point (affected by price, ice)
        """
        # Get location info for current location
        location_info = LOCATION_CATALOG.get(self.current_location)
        foot_traffic_multiplier = location_info.foot_traffic_multiplier if location_info else 1.0
        
        # Determine price sensitivity - use hot weather override if applicable
        is_hot_weather = weather in [Weather.HOT, Weather.SUNNY]
        if location_info:
            if is_hot_weather and location_info.hot_weather_price_sensitivity is not None:
                price_sensitivity = location_info.hot_weather_price_sensitivity
            else:
                price_sensitivity = location_info.price_sensitivity
        else:
            price_sensitivity = self.config.price_sensitivity
        
        # Get weather multiplier (blended with location's weather exposure)
        weather_multiplier = self._get_weather_multiplier(weather, temperature, self.current_location)
        
        # Check if we have ice (affects conversion on hot days)
        has_ice = self.ice_bags > 0
        
        # Calculate base foot traffic (before randomness)
        # Foot traffic = base_customers × location × weather × reputation
        base_foot_traffic = (
            self.config.base_customers * 
            foot_traffic_multiplier * 
            weather_multiplier * 
            (0.5 + self.reputation)
        )
        
        # Foot traffic range (accounting for ±10% randomness)
        foot_traffic_low = int(base_foot_traffic * 0.9)
        foot_traffic_high = int(base_foot_traffic * 1.1)
        
        # Calculate conversion curve (price -> conversion rate as decimal)
        # Base conversion at optimal price ($0.50) is 95%
        price_points = [25, 50, 75, 100, 125, 150, 175, 200]
        conversion_curve = {}
        
        for price in price_points:
            # Base conversion at optimal price
            conversion = 0.95
            
            if price > 50:
                # Gentler price curve using power function for smoother falloff
                price_delta = (price - 50) / 100
                price_factor = 1.0 - (price_delta ** 0.7) * price_sensitivity * 50
                price_factor = max(0.1, price_factor)
                conversion *= price_factor
            
            if price > self.config.max_price_tolerance:
                conversion *= 0.05
            
            # Clamp to valid range
            conversion = max(0.0, min(1.0, conversion))
            conversion_curve[price] = round(conversion, 2)
        
        # Ice conversion bonus on hot days (shown separately so agents can plan)
        ice_conversion_bonus = self.config.ice_demand_bonus if is_hot_weather else 0.0
        
        # Calculate expected sales and revenue curves using foot traffic × conversion
        # Use midpoint of foot traffic for these estimates
        avg_foot_traffic = (foot_traffic_low + foot_traffic_high) // 2
        
        # Apply ice bonus/penalty to conversion for the curves
        price_demand_curve = {}
        revenue_curve = {}
        for price, base_conversion in conversion_curve.items():
            # Apply ice modifier for actual expected demand
            if is_hot_weather:
                if has_ice:
                    adjusted_conversion = min(1.0, base_conversion * (1 + ice_conversion_bonus))
                else:
                    adjusted_conversion = base_conversion * 0.8
            else:
                adjusted_conversion = base_conversion
            
            expected_sales = int(avg_foot_traffic * adjusted_conversion)
            price_demand_curve[price] = expected_sales
            revenue_curve[price] = expected_sales * price
        
        # Find optimal revenue price (highest revenue, not just highest demand)
        optimal_revenue_price = max(revenue_curve, key=revenue_curve.get)
        
        # Calculate max cups producible from inventory
        max_from_lemons = int(self.lemons / self.config.lemons_per_cup)
        max_from_sugar = int(self.sugar_bags / self.config.sugar_per_cup)
        max_from_cups = self.cups
        max_from_ice = int(self.ice_bags / self.config.ice_per_cup) if self.ice_bags > 0 else 0
        
        # Ice is optional but provides bonus - calculate with and without
        max_cups_with_ice = min(max_from_lemons, max_from_sugar, max_from_cups, max_from_ice) if max_from_ice > 0 else 0
        max_cups_without_ice = min(max_from_lemons, max_from_sugar, max_from_cups)
        max_cups_producible = max_cups_without_ice  # Can always make without ice
        
        # Determine limiting resource (for iced lemonade)
        if max_cups_with_ice > 0:
            limits = [
                (max_from_lemons, "lemons"),
                (max_from_sugar, "sugar"),
                (max_from_cups, "cups"),
                (max_from_ice, "ice"),
            ]
            limiting_resource = min(limits, key=lambda x: x[0])[1]
        else:
            limits = [
                (max_from_lemons, "lemons"),
                (max_from_sugar, "sugar"),
                (max_from_cups, "cups"),
            ]
            limiting_resource = min(limits, key=lambda x: x[0])[1]
        
        # Calculate ingredient cost per cup (with ice)
        ingredient_cost_per_cup = int(
            self.config.lemon_cost * self.config.lemons_per_cup +
            self.config.sugar_cost * self.config.sugar_per_cup +
            self.config.ice_cost * self.config.ice_per_cup +
            self.config.cup_cost
        )
        
        # Break-even price is just the ingredient cost
        break_even_price = ingredient_cost_per_cup
        
        # Suggested production: aim for expected sales at optimal price
        optimal_demand = price_demand_curve.get(self.config.optimal_price, foot_traffic_high)
        suggested_production = min(optimal_demand, max_cups_producible)
        
        return {
            # Foot traffic (people who stop by)
            "foot_traffic_low": foot_traffic_low,
            "foot_traffic_high": foot_traffic_high,
            "weather_traffic_multiplier": round(weather_multiplier, 2),
            
            # Conversion rates (% who buy at each price)
            "conversion_curve": conversion_curve,
            "ice_conversion_bonus": round(ice_conversion_bonus, 2),
            
            # Price guidance
            "optimal_price": self.config.optimal_price,
            "price_demand_curve": price_demand_curve,
            "revenue_curve": revenue_curve,
            "optimal_revenue_price": optimal_revenue_price,
            "max_cups_producible": max_cups_producible,
            "max_cups_with_ice": max_cups_with_ice,
            "limiting_resource": limiting_resource,
            "ingredient_cost_per_cup": ingredient_cost_per_cup,
            "break_even_price": break_even_price,
            "suggested_production": suggested_production,
            "has_ice": has_ice,
            "ice_bonus_active": has_ice and is_hot_weather,
            # Additional helper info
            "weather_label": self._get_weather_description(weather),
            "recipe_info": {
                "lemons_per_cup": self.config.lemons_per_cup,
                "sugar_per_cup": self.config.sugar_per_cup,
                "ice_per_cup": self.config.ice_per_cup,
                "cups_from_one_lemon": int(1 / self.config.lemons_per_cup),
                "cups_from_one_sugar_bag": int(1 / self.config.sugar_per_cup),
                "cups_from_one_ice_bag": int(1 / self.config.ice_per_cup),
            },
            "supply_costs": {
                "lemon": self.config.lemon_cost,
                "sugar_bag": self.config.sugar_cost,
                "cup": self.config.cup_cost,
                "ice_bag": self.config.ice_cost,
            },
            "bulk_pricing": {
                supply_type: {
                    "unit_name": pricing.unit_name,
                    "unit_name_plural": pricing.unit_name_plural,
                    "base_price": pricing.base_price,
                    "tiers": [
                        {
                            "name": tier.name,
                            "quantity": tier.quantity,
                            "discount_percent": tier.discount_percent,
                            "total_price": int(tier.quantity * pricing.base_price * (1 - tier.discount_percent)),
                            "price_per_unit": int(pricing.base_price * (1 - tier.discount_percent)),
                        }
                        for tier in pricing.tiers
                    ],
                }
                for supply_type, pricing in BULK_PRICING.items()
            },
            "expiration_info": {
                "lemons_expiring_tomorrow": self._get_lemons_expiring_tomorrow(),
                "ice_melt_rate": 0.5 if StandUpgrade.COOLER in self.owned_upgrades else 1.0,
                "has_cooler": StandUpgrade.COOLER in self.owned_upgrades,
                "lemon_shelf_life": self.config.lemon_shelf_life,
            },
            "location_info": {
                "current_location": self.current_location.value,
                "foot_traffic_multiplier": foot_traffic_multiplier,
                "price_sensitivity": price_sensitivity,
                "hot_weather_price_sensitivity": location_info.hot_weather_price_sensitivity if location_info else None,
                "using_hot_weather_sensitivity": is_hot_weather and location_info and location_info.hot_weather_price_sensitivity is not None,
                "weather_exposure": location_info.weather_exposure if location_info else 1.0,
                "current_permit_cost": location_info.permit_cost if location_info else self.config.location_permit_cost,
            }
        }
    
    def _get_weather_description(self, weather: Weather) -> str:
        """Get a human-readable description of weather impact."""
        descriptions = {
            Weather.HOT: "🔥 Perfect lemonade weather! Expect HIGH demand.",
            Weather.SUNNY: "☀️ Great day for sales. Demand is ABOVE AVERAGE.",
            Weather.CLOUDY: "☁️ Okay conditions. Demand is AVERAGE.",
            Weather.RAINY: "🌧️ People staying indoors. Demand is LOW.",
            Weather.STORMY: "⛈️ Stay inside! Demand is VERY LOW.",
        }
        return descriptions.get(weather, "Unknown weather")
    
    def _get_upgrade_catalog_for_observation(self) -> list[dict]:
        """Get the upgrade catalog as a list of dicts for the observation."""
        return [
            {
                "id": upgrade.value,
                "name": info.name,
                "description": info.description,
                "cost": info.cost,
                "effect": info.effect,
                "owned": upgrade in self.owned_upgrades,
            }
            for upgrade, info in UPGRADE_CATALOG.items()
        ]
    
    def _get_location_catalog_for_observation(self) -> list[dict]:
        """Get the location catalog as a list of dicts for the observation."""
        return [
            {
                "id": location.value,
                "name": info.name,
                "description": info.description,
                "foot_traffic_multiplier": info.foot_traffic_multiplier,
                "price_sensitivity": info.price_sensitivity,
                "hot_weather_price_sensitivity": info.hot_weather_price_sensitivity,
                "weather_exposure": info.weather_exposure,
                "permit_cost": info.permit_cost,
                "is_current": location == self.current_location,
            }
            for location, info in LOCATION_CATALOG.items()
        ]
    
    def _get_ice_expiring_tomorrow(self) -> int:
        """Get count of ice that will melt tomorrow, accounting for cooler."""
        if StandUpgrade.COOLER in self.owned_upgrades:
            # With cooler, 50% of ice melts per day
            return sum(qty - (qty // 2) for qty, _ in self.ice_batches)
        else:
            # Without cooler, all ice melts
            return self.ice_bags
    
    def reset(self) -> LemonadeObservation:
        """
        Reset the environment for a new game.
        
        Returns:
            Initial observation with starting state
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._rng = random.Random(self._seed)
        self._reset_game_state()
        
        weather, temp = self._weather_schedule[0]
        forecast_weather, _ = self._weather_schedule[1]
        
        # Calculate market hints for the starting state
        market_hints = self._calculate_market_hints(weather, temp)
        
        return LemonadeObservation(
            day=self.day,
            weather=weather.value,
            temperature=temp,
            weather_forecast=forecast_weather.value,
            cash=self.cash,
            daily_revenue=0,
            daily_costs=0,
            daily_profit=0,
            cups_sold=0,
            cups_wasted=0,
            customers_served=0,
            customers_turned_away=0,
            lemons=self.lemons,
            sugar_bags=self.sugar_bags,
            cups_available=self.cups,
            ice_bags=self.ice_bags,
            lemons_expiring_tomorrow=self._get_lemons_expiring_tomorrow(),
            ice_expiring_tomorrow=self._get_ice_expiring_tomorrow(),
            lemons_spoiled=0,
            ice_melted=0,
            ice_used=0,
            customer_satisfaction=0.5,
            reputation=self.reputation,
            days_remaining=self.config.total_days - self.day,
            total_profit=self.total_profit,
            owned_upgrades=[u.value for u in self.owned_upgrades],
            upgrade_catalog=self._get_upgrade_catalog_for_observation(),
            current_location=self.current_location.value,
            location_catalog=self._get_location_catalog_for_observation(),
            done=False,
            reward=0.0,
            market_hints=market_hints,
        )
    
    def validate_action(self, action: LemonadeAction) -> list[str]:
        """
        Validate an action and return a list of errors (empty if valid).
        
        Checks that the agent can afford all purchases in the action.
        Purchases are validated in order: location, lemons, sugar, cups, ice, upgrade, advertising.
        
        Args:
            action: The action to validate
            
        Returns:
            List of error messages (empty if action is valid)
        """
        errors = []
        remaining_cash = self.cash
        
        # Check location change cost
        if action.location:
            try:
                new_location = Location(action.location)
                if new_location != self.current_location:
                    new_location_info = LOCATION_CATALOG.get(new_location)
                    permit_cost = new_location_info.permit_cost if new_location_info else self.config.location_permit_cost
                    if permit_cost > remaining_cash:
                        errors.append(f"Cannot afford location change to {action.location}: ${permit_cost/100:.2f} needed, ${remaining_cash/100:.2f} available")
                    else:
                        remaining_cash -= permit_cost
            except ValueError:
                errors.append(f"Invalid location: {action.location}")
        
        # Check purchase costs
        if action.buy_lemons > 0:
            cost = calculate_bulk_cost("lemons", action.buy_lemons)
            if cost > remaining_cash:
                errors.append(f"Cannot afford {action.buy_lemons} lemons: ${cost/100:.2f} needed, ${remaining_cash/100:.2f} available")
            else:
                remaining_cash -= cost
        
        if action.buy_sugar > 0:
            cost = calculate_bulk_cost("sugar", action.buy_sugar)
            if cost > remaining_cash:
                errors.append(f"Cannot afford {action.buy_sugar} sugar bags: ${cost/100:.2f} needed, ${remaining_cash/100:.2f} available")
            else:
                remaining_cash -= cost
        
        if action.buy_cups > 0:
            cost = calculate_bulk_cost("cups", action.buy_cups)
            if cost > remaining_cash:
                errors.append(f"Cannot afford {action.buy_cups} cups: ${cost/100:.2f} needed, ${remaining_cash/100:.2f} available")
            else:
                remaining_cash -= cost
        
        if action.buy_ice > 0:
            cost = calculate_bulk_cost("ice", action.buy_ice)
            if cost > remaining_cash:
                errors.append(f"Cannot afford {action.buy_ice} ice bags: ${cost/100:.2f} needed, ${remaining_cash/100:.2f} available")
            else:
                remaining_cash -= cost
        
        # Check upgrade cost
        if action.buy_upgrade:
            try:
                upgrade = StandUpgrade(action.buy_upgrade)
                if upgrade in self.owned_upgrades:
                    errors.append(f"Already own upgrade: {action.buy_upgrade}")
                elif upgrade in UPGRADE_CATALOG:
                    upgrade_info = UPGRADE_CATALOG[upgrade]
                    if upgrade_info.cost > remaining_cash:
                        errors.append(f"Cannot afford upgrade '{action.buy_upgrade}': ${upgrade_info.cost/100:.2f} needed, ${remaining_cash/100:.2f} available")
                    else:
                        remaining_cash -= upgrade_info.cost
                else:
                    errors.append(f"Unknown upgrade: {action.buy_upgrade}")
            except ValueError:
                errors.append(f"Invalid upgrade: {action.buy_upgrade}")
        
        # Check advertising spend
        if action.advertising_spend > remaining_cash:
            errors.append(f"Cannot afford advertising spend: ${action.advertising_spend/100:.2f} needed, ${remaining_cash/100:.2f} available")
        
        return errors
    
    def _create_error_observation(self, errors: list[str]) -> LemonadeObservation:
        """
        Create an error observation that doesn't advance the day.
        
        Args:
            errors: List of validation error messages
            
        Returns:
            Observation with is_error_response=True and action_errors populated
        """
        # Get current day's weather for context
        weather, temperature = self._weather_schedule[self.day - 1]
        forecast_weather, _ = self._weather_schedule[min(self.day, len(self._weather_schedule) - 1)]
        
        # Calculate market hints for current day
        market_hints = self._calculate_market_hints(weather, temperature)
        
        return LemonadeObservation(
            day=self.day,
            weather=weather.value,
            temperature=temperature,
            weather_forecast=forecast_weather.value,
            cash=self.cash,
            daily_revenue=0,
            daily_costs=0,
            daily_profit=0,
            cups_sold=0,
            cups_wasted=0,
            customers_served=0,
            customers_turned_away=0,
            lemons=self.lemons,
            sugar_bags=self.sugar_bags,
            cups_available=self.cups,
            ice_bags=self.ice_bags,
            lemons_expiring_tomorrow=self._get_lemons_expiring_tomorrow(),
            ice_expiring_tomorrow=self._get_ice_expiring_tomorrow(),
            lemons_spoiled=0,
            ice_melted=0,
            ice_used=0,
            customer_satisfaction=0.0,
            reputation=self.reputation,
            days_remaining=self.config.total_days - self.day,
            total_profit=self.total_profit,
            owned_upgrades=[u.value for u in self.owned_upgrades],
            upgrade_catalog=self._get_upgrade_catalog_for_observation(),
            current_location=self.current_location.value,
            location_catalog=self._get_location_catalog_for_observation(),
            done=False,
            reward=0.0,
            market_hints=market_hints,
            action_errors=errors,
            is_error_response=True,
        )
    
    def step(self, action: LemonadeAction) -> LemonadeObservation:
        """
        Execute one day of lemonade stand operations.
        
        Args:
            action: The agent's decisions for the day
            
        Returns:
            Observation with results of the day's operations.
            If the action is invalid (e.g., can't afford purchases), returns
            an error observation with is_error_response=True and the day is not advanced.
        """
        # Validate action first
        errors = self.validate_action(action)
        if errors:
            return self._create_error_observation(errors)
        
        self._state.step_count += 1
        
        # Get today's weather
        weather, temperature = self._weather_schedule[self.day - 1]
        
        # 1. Process location change (costs permit fee if switching)
        location_cost = 0
        if action.location:
            try:
                new_location = Location(action.location)
                if new_location != self.current_location:
                    # Charge permit fee for the NEW location
                    new_location_info = LOCATION_CATALOG.get(new_location)
                    permit_cost = new_location_info.permit_cost if new_location_info else self.config.location_permit_cost
                    if permit_cost <= self.cash:
                        self.current_location = new_location
                        self.cash -= permit_cost
                        location_cost = permit_cost
            except ValueError:
                pass  # Invalid location name, stay at current location
        
        # 2. Process supply purchases (morning) - with bulk discounts!
        purchase_cost = 0
        
        if action.buy_lemons > 0:
            cost = calculate_bulk_cost("lemons", action.buy_lemons)
            if cost <= self.cash:
                # Add new batch with full shelf life
                self.lemon_batches.append((action.buy_lemons, self.config.lemon_shelf_life))
                self.cash -= cost
                purchase_cost += cost
        
        if action.buy_sugar > 0:
            cost = calculate_bulk_cost("sugar", action.buy_sugar)
            if cost <= self.cash:
                self.sugar_bags += action.buy_sugar
                self.cash -= cost
                purchase_cost += cost
        
        if action.buy_cups > 0:
            cost = calculate_bulk_cost("cups", action.buy_cups)
            if cost <= self.cash:
                self.cups += action.buy_cups
                self.cash -= cost
                purchase_cost += cost
        
        if action.buy_ice > 0:
            cost = calculate_bulk_cost("ice", action.buy_ice)
            if cost <= self.cash:
                # Add new batch with full shelf life (1 day for ice)
                self.ice_batches.append((action.buy_ice, self.config.ice_shelf_life))
                self.cash -= cost
                purchase_cost += cost
        
        # Process upgrade purchase
        if action.buy_upgrade:
            try:
                upgrade = StandUpgrade(action.buy_upgrade)
                if upgrade not in self.owned_upgrades and upgrade in UPGRADE_CATALOG:
                    upgrade_info = UPGRADE_CATALOG[upgrade]
                    if upgrade_info.cost <= self.cash:
                        self.owned_upgrades.add(upgrade)
                        self.cash -= upgrade_info.cost
                        purchase_cost += upgrade_info.cost
            except ValueError:
                pass  # Invalid upgrade name, ignore
        
        ad_cost = min(action.advertising_spend, self.cash)
        self.cash -= ad_cost
        
        # 3. Calculate customer demand (uses current location's modifiers)
        has_ice = self.ice_bags > 0
        customer_demand = self._calculate_demand(
            action.price_per_cup,
            weather,
            temperature,
            ad_cost,
            has_ice=has_ice
        )
        
        # 4. Calculate max cups we can make from available supplies
        max_from_lemons = int(self.lemons / self.config.lemons_per_cup)
        max_from_sugar = int(self.sugar_bags / self.config.sugar_per_cup)
        max_from_cups = self.cups
        max_cups_possible = min(max_from_lemons, max_from_sugar, max_from_cups)
        
        # Make lemonade on-demand (only make what we can sell, up to supply limit)
        cups_to_make = min(customer_demand, max_cups_possible)
        cups_sold = cups_to_make  # We sell everything we make (on-demand)
        cups_wasted = 0  # No waste in on-demand model
        customers_turned_away = max(0, customer_demand - max_cups_possible)
        
        # Ice is optional - if we don't have enough, we make lemonade without ice
        # but we still use what ice we have
        ice_to_use = min(cups_to_make * self.config.ice_per_cup, self.ice_bags)
        
        # Consume supplies (using FIFO for perishables)
        self._consume_lemons(cups_to_make * self.config.lemons_per_cup)
        self.sugar_bags -= cups_to_make * self.config.sugar_per_cup
        self.cups -= cups_to_make
        ice_consumed = self._consume_ice(ice_to_use)
        
        # 5. Calculate financials
        revenue = cups_sold * action.price_per_cup
        total_costs = location_cost + purchase_cost + ad_cost
        daily_profit = revenue - total_costs
        
        self.cash += revenue
        self.total_profit += daily_profit
        
        # 6. Update reputation
        satisfaction = self._calculate_satisfaction(
            action.price_per_cup,
            cups_sold,
            customers_turned_away,
            weather
        )
        
        # Reputation slowly moves toward current satisfaction
        self.reputation = self.reputation * 0.8 + satisfaction * 0.2
        self.reputation = min(1.0, max(0.0, self.reputation))
        
        # 7. Process overnight expiration (items spoil/melt at end of business day)
        # This happens AFTER sales, so starting inventory and purchases are usable today
        self._process_expiration()
        lemons_spoiled = self.lemons_spoiled_today
        ice_melted = self.ice_melted_today
        
        # 8. Advance to next day
        self.day += 1
        game_over = self.day > self.config.total_days
        
        # Get weather for the NEW day (the day we're about to plan for)
        if not game_over:
            new_day_weather, new_day_temp = self._weather_schedule[self.day - 1]  # Current day (1-indexed)
            forecast_weather, _ = self._weather_schedule[self.day]  # Tomorrow's forecast
            # Calculate market hints for TODAY (the day player is about to plan)
            market_hints = self._calculate_market_hints(new_day_weather, new_day_temp)
        else:
            # Game over - show final day's weather (what we just played)
            new_day_weather = weather
            new_day_temp = temperature
            forecast_weather = Weather.SUNNY  # Doesn't matter
            market_hints = None
        
        # Calculate reward (normalized daily profit)
        # Positive reward for profit, negative for loss
        reward = daily_profit / 100.0  # Convert cents to dollars for nicer numbers
        
        # Bonus reward at end of game based on total profit
        if game_over:
            reward += self.total_profit / 1000.0  # Bonus for overall performance
        
        return LemonadeObservation(
            day=self.day if not game_over else self.config.total_days,
            weather=new_day_weather.value if not game_over else weather.value,
            temperature=new_day_temp if not game_over else temperature,
            weather_forecast=forecast_weather.value,
            cash=self.cash,
            daily_revenue=revenue,
            daily_costs=total_costs,
            daily_profit=daily_profit,
            cups_sold=cups_sold,
            cups_wasted=cups_wasted,
            customers_served=cups_sold,
            customers_turned_away=customers_turned_away,
            lemons=self.lemons,
            sugar_bags=self.sugar_bags,
            cups_available=self.cups,
            ice_bags=self.ice_bags,
            lemons_expiring_tomorrow=self._get_lemons_expiring_tomorrow(),
            ice_expiring_tomorrow=self._get_ice_expiring_tomorrow(),
            lemons_spoiled=lemons_spoiled,
            ice_melted=ice_melted,
            ice_used=int(ice_consumed),
            customer_satisfaction=satisfaction,
            reputation=self.reputation,
            days_remaining=max(0, self.config.total_days - self.day),
            total_profit=self.total_profit,
            owned_upgrades=[u.value for u in self.owned_upgrades],
            upgrade_catalog=self._get_upgrade_catalog_for_observation(),
            current_location=self.current_location.value,
            location_catalog=self._get_location_catalog_for_observation(),
            done=game_over,
            reward=reward,
            market_hints=market_hints,
            metadata={
                "weather_multiplier": self._get_weather_multiplier(weather, temperature),
                "customer_demand": customer_demand,
                "had_ice": has_ice,
                "has_cooler": StandUpgrade.COOLER in self.owned_upgrades,
                "location": self.current_location.value,
                "location_cost": location_cost,
            },
        )
    
    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state
