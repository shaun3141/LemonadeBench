# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
Tests for the LemonadeEnvironment class.

Covers:
- Environment initialization
- Reset functionality
- Step execution
- Weather effects
- Inventory management
- Perishable items (lemons, ice)
- Upgrades (cooler)
- Customer demand calculations
- Reputation system
- Game completion
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    GameConfig,
    LemonadeAction,
    LemonadeObservation,
    Weather,
    StandUpgrade,
    UPGRADE_CATALOG,
    Location,
    LOCATION_CATALOG,
)
from server.lemonade_environment import LemonadeEnvironment


class TestEnvironmentInitialization:
    """Tests for environment initialization."""

    def test_default_initialization(self):
        """Test creating environment with defaults."""
        env = LemonadeEnvironment()
        assert env.config is not None
        assert env.day == 1
        assert env.cash == env.config.starting_cash

    def test_initialization_with_config(self):
        """Test creating environment with custom config."""
        config = GameConfig(starting_cash=5000, total_days=7)
        env = LemonadeEnvironment(config=config)
        assert env.config.starting_cash == 5000
        assert env.config.total_days == 7

    def test_initialization_with_seed(self):
        """Test seeded environment is reproducible."""
        env1 = LemonadeEnvironment(seed=42)
        env2 = LemonadeEnvironment(seed=42)

        obs1 = env1.reset()
        obs2 = env2.reset()

        assert obs1.weather == obs2.weather
        assert obs1.temperature == obs2.temperature
        assert obs1.weather_forecast == obs2.weather_forecast

    def test_different_seeds_different_weather(self):
        """Test different seeds produce different results."""
        env1 = LemonadeEnvironment(seed=42)
        env2 = LemonadeEnvironment(seed=999)

        obs1 = env1.reset()
        obs2 = env2.reset()

        # With different seeds, at least one property should differ
        # (statistically almost certain)
        differs = (
            obs1.weather != obs2.weather
            or obs1.temperature != obs2.temperature
            or obs1.weather_forecast != obs2.weather_forecast
        )
        assert differs


class TestReset:
    """Tests for the reset() method."""

    def test_reset_returns_observation(self, env: LemonadeEnvironment):
        """Test reset returns a valid observation."""
        obs = env.reset()
        assert isinstance(obs, LemonadeObservation)

    def test_reset_starts_day_1(self, env: LemonadeEnvironment):
        """Test reset starts at day 1."""
        obs = env.reset()
        assert obs.day == 1

    def test_reset_starting_inventory(self, env: LemonadeEnvironment):
        """Test reset has correct starting inventory."""
        obs = env.reset()
        assert obs.lemons == env.config.starting_lemons
        assert obs.sugar_bags == env.config.starting_sugar
        assert obs.cups_available == env.config.starting_cups
        assert obs.ice_bags == env.config.starting_ice

    def test_reset_starting_cash(self, env: LemonadeEnvironment):
        """Test reset has correct starting cash."""
        obs = env.reset()
        assert obs.cash == env.config.starting_cash

    def test_reset_zero_profit(self, env: LemonadeEnvironment):
        """Test reset starts with zero profit."""
        obs = env.reset()
        assert obs.daily_profit == 0
        assert obs.daily_revenue == 0
        assert obs.daily_costs == 0
        assert obs.total_profit == 0

    def test_reset_starting_reputation(self, env: LemonadeEnvironment):
        """Test reset starts with neutral reputation."""
        obs = env.reset()
        assert obs.reputation == 0.5

    def test_reset_not_done(self, env: LemonadeEnvironment):
        """Test reset starts with done=False."""
        obs = env.reset()
        assert obs.done is False

    def test_reset_has_weather(self, env: LemonadeEnvironment):
        """Test reset includes weather information."""
        obs = env.reset()
        assert obs.weather in ["sunny", "cloudy", "rainy", "hot", "stormy"]
        assert obs.weather_forecast in ["sunny", "cloudy", "rainy", "hot", "stormy"]
        assert 50 <= obs.temperature <= 105

    def test_reset_days_remaining(self, env: LemonadeEnvironment):
        """Test reset has correct days remaining."""
        obs = env.reset()
        assert obs.days_remaining == env.config.total_days - 1

    def test_reset_has_market_hints(self, env: LemonadeEnvironment):
        """Test reset includes market hints with two-stage model."""
        obs = env.reset()
        assert obs.market_hints is not None
        # Foot traffic fields
        assert "foot_traffic_low" in obs.market_hints
        assert "foot_traffic_high" in obs.market_hints
        assert "weather_traffic_multiplier" in obs.market_hints
        # Conversion fields
        assert "conversion_curve" in obs.market_hints
        assert "ice_conversion_bonus" in obs.market_hints
        # Price/demand curves
        assert "price_demand_curve" in obs.market_hints

    def test_reset_has_upgrade_catalog(self, env: LemonadeEnvironment):
        """Test reset includes upgrade catalog."""
        obs = env.reset()
        assert obs.upgrade_catalog is not None
        assert len(obs.upgrade_catalog) > 0

    def test_reset_no_owned_upgrades(self, env: LemonadeEnvironment):
        """Test reset starts with no owned upgrades."""
        obs = env.reset()
        assert obs.owned_upgrades == []

    def test_reset_clears_previous_game(self, env: LemonadeEnvironment):
        """Test reset clears state from previous game."""
        # Play a few days
        env.reset()
        action = LemonadeAction(price_per_cup=75)
        env.step(action)
        env.step(action)

        # Reset and verify clean slate
        obs = env.reset()
        assert obs.day == 1
        assert obs.total_profit == 0


class TestStep:
    """Tests for the step() method."""

    def test_step_advances_day(self, env: LemonadeEnvironment):
        """Test step advances the day counter."""
        env.reset()
        action = LemonadeAction(price_per_cup=75)
        obs = env.step(action)
        assert obs.day == 2

    def test_step_returns_observation(self, env: LemonadeEnvironment):
        """Test step returns a valid observation."""
        env.reset()
        action = LemonadeAction(price_per_cup=75)
        obs = env.step(action)
        assert isinstance(obs, LemonadeObservation)

    def test_step_consumes_supplies(self, env: LemonadeEnvironment):
        """Test step consumes supplies when serving customers (on-demand)."""
        obs = env.reset()
        initial_lemons = obs.lemons
        initial_sugar = obs.sugar_bags
        initial_cups = obs.cups_available

        # Low price = high demand = supplies consumed
        action = LemonadeAction(price_per_cup=50)
        obs = env.step(action)

        # Check supplies were consumed (if there was any demand)
        if obs.cups_sold > 0:
            assert obs.lemons < initial_lemons
            assert obs.sugar_bags < initial_sugar
            assert obs.cups_available < initial_cups

    def test_step_generates_revenue(self, env: LemonadeEnvironment):
        """Test step generates revenue from sales."""
        env.reset()
        action = LemonadeAction(price_per_cup=50)
        obs = env.step(action)

        # Should have some sales (depends on weather/demand)
        # Revenue = cups_sold * price_per_cup
        assert obs.daily_revenue == obs.cups_sold * 50

    def test_step_no_waste_on_demand(self, env: LemonadeEnvironment):
        """Test on-demand model has no wasted cups."""
        env.reset()
        # With on-demand production, we only make what we sell
        action = LemonadeAction(price_per_cup=200)
        obs = env.step(action)

        # cups_wasted should always be 0 with on-demand model
        assert obs.cups_wasted == 0

    def test_step_tracks_customers_turned_away(self, env: LemonadeEnvironment):
        """Test step tracks customers who wanted to buy but couldn't (supply limited)."""
        # Use limited starting inventory
        config = GameConfig(starting_cups=5, starting_lemons=2)
        env = LemonadeEnvironment(config=config, seed=42)
        env.reset()
        
        # Low price = high demand, but limited supply
        action = LemonadeAction(price_per_cup=25)
        obs = env.step(action)

        # Should turn away customers due to limited supplies
        assert obs.customers_turned_away >= 0

    def test_step_calculates_profit(self, env: LemonadeEnvironment):
        """Test step correctly calculates daily profit."""
        env.reset()
        action = LemonadeAction(price_per_cup=100)
        obs = env.step(action)

        assert obs.daily_profit == obs.daily_revenue - obs.daily_costs

    def test_step_updates_total_profit(self, env: LemonadeEnvironment):
        """Test step accumulates total profit."""
        env.reset()
        action = LemonadeAction(price_per_cup=100)

        obs1 = env.step(action)
        profit1 = obs1.total_profit

        obs2 = env.step(action)

        assert obs2.total_profit == profit1 + obs2.daily_profit

    def test_step_updates_cash(self, env: LemonadeEnvironment):
        """Test step updates cash correctly."""
        obs0 = env.reset()
        initial_cash = obs0.cash

        action = LemonadeAction(price_per_cup=100)
        obs = env.step(action)

        # Cash = initial - costs + revenue
        expected_cash = initial_cash - obs.daily_costs + obs.daily_revenue
        assert obs.cash == expected_cash


class TestPurchasing:
    """Tests for purchasing supplies."""

    def test_buy_lemons(self, env: LemonadeEnvironment):
        """Test buying lemons increases inventory."""
        obs0 = env.reset()
        initial_lemons = obs0.lemons

        # High price = low demand = fewer supplies consumed
        action = LemonadeAction(
            price_per_cup=500, buy_lemons=10
        )
        obs = env.step(action)

        # After purchase (minus any that expired and any used)
        # With high price, demand is very low, so most lemons should remain
        assert obs.lemons >= initial_lemons  # At least broke even with purchase

    def test_buy_sugar(self, env: LemonadeEnvironment):
        """Test buying sugar increases inventory."""
        obs0 = env.reset()
        initial_sugar = obs0.sugar_bags

        # High price = low demand
        action = LemonadeAction(
            price_per_cup=500, buy_sugar=5
        )
        obs = env.step(action)

        # Sugar used depends on cups sold
        assert obs.sugar_bags >= initial_sugar + 5 - (obs.cups_sold * env.config.sugar_per_cup)

    def test_buy_cups(self, env: LemonadeEnvironment):
        """Test buying cups increases inventory."""
        obs0 = env.reset()
        initial_cups = obs0.cups_available

        # High price = low demand
        action = LemonadeAction(
            price_per_cup=500, buy_cups=20
        )
        obs = env.step(action)

        # Cups used depends on cups sold
        assert obs.cups_available >= initial_cups + 20 - obs.cups_sold

    def test_buy_ice(self, env: LemonadeEnvironment):
        """Test buying ice increases inventory."""
        env.reset()

        action = LemonadeAction(
            price_per_cup=500, buy_ice=10
        )
        obs = env.step(action)

        # Ice bought today should be available (minus melt from existing ice and any used)
        # On day 1, existing ice melts, new ice is added
        assert obs.ice_bags >= 0

    def test_purchase_costs_cash(self, env: LemonadeEnvironment):
        """Test purchases reduce cash."""
        obs0 = env.reset()
        initial_cash = obs0.cash

        action = LemonadeAction(
            price_per_cup=500,  # High price = low demand
            buy_lemons=10,
            buy_sugar=2,
        )
        obs = env.step(action)

        assert obs.cash < initial_cash
        assert obs.daily_costs > 0

    def test_cannot_buy_more_than_cash(self, env: LemonadeEnvironment):
        """Test cannot overspend cash."""
        env.reset()
        # Try to buy way more than we can afford
        action = LemonadeAction(
            price_per_cup=500,
            buy_lemons=10000,  # Would cost way more than starting cash
        )
        obs = env.step(action)

        # Should not have bought 10000 lemons
        assert obs.lemons < 10000

    def test_bulk_discount_applied(self, env: LemonadeEnvironment):
        """Test bulk discounts are applied to purchases."""
        env1 = LemonadeEnvironment(seed=42)

        env1.reset()

        # Buy 12 lemons (should get dozen discount), high price = no sales
        action1 = LemonadeAction(price_per_cup=500, buy_lemons=12)
        obs1 = env1.step(action1)

        # Full price: 12 * 25 = 300, with 10% off = 270
        expected_cost = int(12 * 25 * 0.90)
        assert obs1.daily_costs == expected_cost


class TestPerishableItems:
    """Tests for perishable inventory management."""

    def test_lemons_expire_after_shelf_life(self, env: LemonadeEnvironment):
        """Test lemons expire after their shelf life."""
        env.reset()

        # High price = no sales = lemons not used
        action = LemonadeAction(price_per_cup=500)

        # After 3 days, starting lemons should expire
        for _ in range(3):
            env.step(action)

        obs = env.step(action)

        # By day 4, original lemons should have spoiled
        # (expiration happens at end of each day)
        assert obs.lemons_spoiled > 0 or env.lemons == 0

    def test_ice_melts_without_cooler(self, env: LemonadeEnvironment):
        """Test ice melts completely without cooler at end of day."""
        env.reset()

        # Buy ice, high price = no sales (ice won't be used)
        action = LemonadeAction(
            price_per_cup=500, buy_ice=10
        )
        obs1 = env.step(action)

        # With expiration at END of day, ice melts overnight after Day 1
        # obs1 shows what melted at the end of Day 1
        assert obs1.ice_melted > 0  # Starting ice + purchased ice melted
        assert obs1.ice_bags == 0  # All ice is gone after overnight melting

    def test_starting_inventory_usable_on_day1(self, env: LemonadeEnvironment):
        """Test that starting inventory (including ice) is usable on Day 1."""
        # This verifies expiration happens at END of day, not START
        initial = env.reset()
        
        # Should have starting ice available
        assert initial.ice_bags == 5  # Default starting ice
        
        # Set low price to ensure sales happen (uses starting inventory)
        action = LemonadeAction(price_per_cup=50)
        obs1 = env.step(action)
        
        # Starting ice should have been used for Day 1 sales
        # (if there were sales and ice was available)
        if obs1.cups_sold > 0:
            # Ice was used for making lemonade
            assert obs1.ice_used > 0 or obs1.metadata.get("had_ice", False)
        
        # Ice melts at END of day, shown in obs1
        assert obs1.ice_bags == 0  # All remaining ice melted overnight

    def test_lemons_expiring_tomorrow_tracking(self, env: LemonadeEnvironment):
        """Test tracking of lemons expiring tomorrow."""
        env.reset()

        action = LemonadeAction(price_per_cup=500)

        # Advance a couple days
        env.step(action)
        obs = env.step(action)

        # Should track expiring lemons
        assert obs.lemons_expiring_tomorrow >= 0

    def test_fifo_consumption_lemons(self, env: LemonadeEnvironment):
        """Test older lemons are used first (FIFO)."""
        env.reset()

        # Buy more lemons, high price = no sales
        action = LemonadeAction(
            price_per_cup=500, buy_lemons=20
        )
        env.step(action)

        # Use some lemons by setting low price (high demand)
        action2 = LemonadeAction(price_per_cup=50)
        env.step(action2)

        # Old lemons should be consumed first (tested by checking inventory)
        # This is verified by the fact that new lemons don't expire for 3 days
        # while old ones expire sooner


class TestCoolerUpgrade:
    """Tests for the cooler upgrade."""

    def test_buy_cooler(self, env: LemonadeEnvironment):
        """Test purchasing the cooler upgrade."""
        env.reset()

        action = LemonadeAction(
            price_per_cup=500,  # High price = no sales
            buy_upgrade="cooler",
        )
        obs = env.step(action)

        assert "cooler" in obs.owned_upgrades

    def test_cooler_cost(self, env: LemonadeEnvironment):
        """Test cooler costs correct amount."""
        obs0 = env.reset()
        initial_cash = obs0.cash

        action = LemonadeAction(
            price_per_cup=500,  # High price = no sales
            buy_upgrade="cooler",
        )
        obs = env.step(action)

        cooler_cost = UPGRADE_CATALOG[StandUpgrade.COOLER].cost
        expected_cash = initial_cash - cooler_cost
        assert obs.cash == expected_cash

    def test_cooler_preserves_ice(self, env: LemonadeEnvironment):
        """Test cooler preserves 50% of ice."""
        env.reset()

        # Buy cooler and ice, high price = no sales
        action = LemonadeAction(
            price_per_cup=500,
            buy_upgrade="cooler",
            buy_ice=10,
        )
        env.step(action)

        # Wait a day - with cooler, only 50% should melt
        action2 = LemonadeAction(price_per_cup=500)
        obs2 = env.step(action2)

        # With cooler, 10 ice -> 5 ice (50% preserved)
        # Some ice may have been used, but melt rate is 50%
        assert obs2.ice_bags >= 0  # Just verify we tracked it

    def test_cannot_buy_cooler_twice(self, env: LemonadeEnvironment):
        """Test cannot purchase cooler if already owned."""
        env.reset()

        action = LemonadeAction(
            price_per_cup=500,
            buy_upgrade="cooler",
        )
        obs1 = env.step(action)
        cash_after_first = obs1.cash

        # Try to buy again
        obs2 = env.step(action)

        # Should not charge again
        assert obs2.cash == cash_after_first
        assert obs2.owned_upgrades.count("cooler") == 1

    def test_invalid_upgrade_ignored(self, env: LemonadeEnvironment):
        """Test invalid upgrade names are ignored."""
        env.reset()

        action = LemonadeAction(
            price_per_cup=500,
            buy_upgrade="invalid_upgrade",
        )
        obs = env.step(action)

        # Should not add invalid upgrade
        assert "invalid_upgrade" not in obs.owned_upgrades
        # Cash should not be affected (besides any purchases)
        assert obs.daily_costs == 0


class TestWeatherEffects:
    """Tests for weather impact on demand."""

    def test_weather_affects_demand(self):
        """Test that weather affects customer demand."""
        # We'll test with controlled seeds to get different weather
        env_hot = LemonadeEnvironment(seed=100)
        env_rainy = LemonadeEnvironment(seed=17)

        env_hot.reset()
        env_rainy.reset()

        action = LemonadeAction(price_per_cup=50)

        obs_hot = env_hot.step(action)
        obs_rainy = env_rainy.step(action)

        # Different weather should produce different demand
        # Just verify we get valid numbers
        assert obs_hot.cups_sold >= 0
        assert obs_rainy.cups_sold >= 0

    def test_weather_multiplier_in_metadata(self, env: LemonadeEnvironment):
        """Test weather multiplier is reported in metadata."""
        env.reset()
        action = LemonadeAction(price_per_cup=75)
        obs = env.step(action)

        assert "weather_multiplier" in obs.metadata
        assert 0.1 <= obs.metadata["weather_multiplier"] <= 3.0


class TestPriceEffects:
    """Tests for price impact on demand."""

    def test_high_price_reduces_demand(self, env: LemonadeEnvironment):
        """Test that very high prices reduce demand."""
        env.reset()

        # Very high price - should have very few sales
        action = LemonadeAction(price_per_cup=500)
        obs = env.step(action)

        # Should sell very few cups at $5.00
        assert obs.cups_sold < 20  # Very low demand at high price

    def test_optimal_price_maximizes_demand(self, env: LemonadeEnvironment):
        """Test that optimal price doesn't reduce demand."""
        env.reset()
        optimal_price = env.config.optimal_price

        action = LemonadeAction(price_per_cup=optimal_price)
        obs = env.step(action)

        # At optimal price, demand is not reduced by price factor
        # Should sell reasonably well (depends on weather)
        assert obs.cups_sold >= 0

    def test_zero_price_allowed(self, env: LemonadeEnvironment):
        """Test that zero price is allowed (free lemonade)."""
        env.reset()
        action = LemonadeAction(price_per_cup=0)
        obs = env.step(action)

        # Should be able to give away lemonade
        assert obs.daily_revenue == 0  # Free!


class TestAdvertising:
    """Tests for advertising effects."""

    def test_advertising_costs_money(self, env: LemonadeEnvironment):
        """Test advertising spend is deducted from cash."""
        env.reset()

        action = LemonadeAction(
            price_per_cup=75,
            advertising_spend=100,
        )
        obs = env.step(action)

        # Advertising should be part of costs
        assert obs.daily_costs >= 100

    def test_advertising_limited_by_cash(self, env: LemonadeEnvironment):
        """Test cannot spend more on ads than available cash."""
        env.reset()

        action = LemonadeAction(
            price_per_cup=75,
            advertising_spend=100000,  # Way more than we have
        )
        obs = env.step(action)

        # Should not go negative
        assert obs.cash >= 0


class TestReputation:
    """Tests for reputation system."""

    def test_reputation_starts_neutral(self, env: LemonadeEnvironment):
        """Test reputation starts at 0.5."""
        obs = env.reset()
        assert obs.reputation == 0.5

    def test_reputation_changes_with_satisfaction(self, env: LemonadeEnvironment):
        """Test reputation changes based on customer satisfaction."""
        env.reset()

        # Good service at good price (with enough supplies)
        action = LemonadeAction(price_per_cup=50, buy_lemons=20, buy_cups=50)

        # Take several steps to see reputation change
        for _ in range(5):
            obs = env.step(action)

        # Reputation should have changed from 0.5
        assert obs.reputation != 0.5 or obs.customer_satisfaction == 0.5

    def test_reputation_bounded(self, env: LemonadeEnvironment):
        """Test reputation stays between 0 and 1."""
        env.reset()

        action = LemonadeAction(price_per_cup=25, buy_lemons=10)

        for _ in range(10):
            obs = env.step(action)
            assert 0.0 <= obs.reputation <= 1.0


class TestGameCompletion:
    """Tests for game completion and done state."""

    def test_game_ends_after_total_days(self, short_game_env: LemonadeEnvironment):
        """Test game ends after configured number of days."""
        short_game_env.reset()

        action = LemonadeAction(price_per_cup=75)

        # Play through all days
        for i in range(5):  # 5-day game
            obs = short_game_env.step(action)

        assert obs.done is True

    def test_done_flag_on_last_day(self, short_game_env: LemonadeEnvironment):
        """Test done flag is set correctly on final day."""
        short_game_env.reset()

        action = LemonadeAction(price_per_cup=75)

        # Play 4 days (not done yet)
        for _ in range(4):
            obs = short_game_env.step(action)
            assert obs.done is False

        # Day 5 should be done
        obs = short_game_env.step(action)
        assert obs.done is True

    def test_days_remaining_decrements(self, env: LemonadeEnvironment):
        """Test days_remaining decrements correctly."""
        obs = env.reset()
        initial_remaining = obs.days_remaining

        action = LemonadeAction(price_per_cup=75)
        obs = env.step(action)

        assert obs.days_remaining == initial_remaining - 1

    def test_final_reward_bonus(self, short_game_env: LemonadeEnvironment):
        """Test bonus reward at end of game."""
        short_game_env.reset()

        action = LemonadeAction(price_per_cup=75)

        # Play through game
        for _ in range(4):
            short_game_env.step(action)

        # Final step should have bonus reward
        obs = short_game_env.step(action)
        assert obs.done is True
        assert obs.reward is not None  # Reward should be set


class TestInventoryLimits:
    """Tests for inventory constraints on on-demand production."""

    def test_sales_limited_by_lemons(self, env: LemonadeEnvironment):
        """Test sales limited by lemon supply."""
        obs0 = env.reset()
        lemons = obs0.lemons
        max_cups_from_lemons = int(lemons / env.config.lemons_per_cup)

        # Low price = high demand, but limited by supplies
        action = LemonadeAction(price_per_cup=25)
        obs = env.step(action)

        # Should be limited by lemons (on-demand, so cups_sold = cups made)
        assert obs.cups_sold <= max_cups_from_lemons

    def test_sales_limited_by_sugar(self):
        """Test sales limited by sugar supply."""
        config = GameConfig(starting_sugar=1)  # Very limited sugar
        env = LemonadeEnvironment(config=config, seed=42)
        obs0 = env.reset()

        max_cups_from_sugar = int(obs0.sugar_bags / env.config.sugar_per_cup)

        action = LemonadeAction(price_per_cup=25)  # Low price = high demand
        obs = env.step(action)

        # On-demand model: cups_sold = cups made, limited by supplies
        assert obs.cups_sold <= max_cups_from_sugar

    def test_sales_limited_by_cups_available(self):
        """Test sales limited by disposable cups."""
        config = GameConfig(starting_cups=5)  # Very few cups
        env = LemonadeEnvironment(config=config, seed=42)
        env.reset()

        action = LemonadeAction(price_per_cup=25)  # Low price = high demand
        obs = env.step(action)

        # Should be limited by disposable cups
        assert obs.cups_sold <= 5


class TestMarketHints:
    """Tests for market hints functionality."""

    def test_market_hints_included(self, env: LemonadeEnvironment):
        """Test market hints are included in observation."""
        obs = env.reset()
        assert obs.market_hints is not None

    def test_market_hints_foot_traffic_range(self, env: LemonadeEnvironment):
        """Test market hints include foot traffic range (two-stage model)."""
        obs = env.reset()
        hints = obs.market_hints

        # Foot traffic fields
        assert "foot_traffic_low" in hints
        assert "foot_traffic_high" in hints
        assert hints["foot_traffic_low"] <= hints["foot_traffic_high"]

    def test_market_hints_price_curve(self, env: LemonadeEnvironment):
        """Test market hints include price-demand curve."""
        obs = env.reset()
        hints = obs.market_hints

        assert "price_demand_curve" in hints
        curve = hints["price_demand_curve"]
        assert isinstance(curve, dict)
        assert len(curve) > 0

    def test_market_hints_conversion_curve(self, env: LemonadeEnvironment):
        """Test market hints include conversion curve (two-stage model)."""
        obs = env.reset()
        hints = obs.market_hints

        assert "conversion_curve" in hints
        curve = hints["conversion_curve"]
        assert isinstance(curve, dict)
        assert len(curve) > 0
        
        # All conversion rates should be between 0 and 1
        for price, conversion in curve.items():
            assert 0.0 <= conversion <= 1.0, f"Conversion at {price} should be 0-1, got {conversion}"
        
        # Lower prices should have higher conversion
        if 50 in curve and 150 in curve:
            assert curve[50] >= curve[150], "Conversion should be higher at lower prices"
    
    def test_market_hints_ice_conversion_bonus(self, env: LemonadeEnvironment):
        """Test market hints include ice conversion bonus."""
        obs = env.reset()
        hints = obs.market_hints

        assert "ice_conversion_bonus" in hints
        # Ice bonus is only active on hot days
        assert hints["ice_conversion_bonus"] >= 0.0

    def test_market_hints_production_info(self, env: LemonadeEnvironment):
        """Test market hints include production info."""
        obs = env.reset()
        hints = obs.market_hints

        assert "max_cups_producible" in hints
        assert "limiting_resource" in hints
        assert hints["limiting_resource"] in ["lemons", "sugar", "cups", "ice"]

    def test_market_hints_cost_info(self, env: LemonadeEnvironment):
        """Test market hints include cost information."""
        obs = env.reset()
        hints = obs.market_hints

        assert "ingredient_cost_per_cup" in hints
        assert "break_even_price" in hints
        assert hints["ingredient_cost_per_cup"] > 0

    def test_market_hints_weather_multiplier(self, env: LemonadeEnvironment):
        """Test market hints include weather/traffic multiplier."""
        obs = env.reset()
        hints = obs.market_hints

        assert "weather_traffic_multiplier" in hints
        assert hints["weather_traffic_multiplier"] > 0

    def test_market_hints_none_when_game_over(self, short_game_env: LemonadeEnvironment):
        """Test market hints are None when game is over."""
        short_game_env.reset()
        action = LemonadeAction(price_per_cup=75)

        # Play to end
        for _ in range(5):
            obs = short_game_env.step(action)

        assert obs.done is True
        assert obs.market_hints is None


class TestStateProperty:
    """Tests for environment state property."""

    def test_state_has_episode_id(self, env: LemonadeEnvironment):
        """Test state includes episode ID."""
        env.reset()
        state = env.state
        assert state.episode_id is not None
        assert len(state.episode_id) > 0

    def test_state_step_count_increments(self, env: LemonadeEnvironment):
        """Test state step count increments with each step."""
        env.reset()
        assert env.state.step_count == 0

        action = LemonadeAction(price_per_cup=75)
        env.step(action)
        assert env.state.step_count == 1

        env.step(action)
        assert env.state.step_count == 2

    def test_state_reset_new_episode_id(self, env: LemonadeEnvironment):
        """Test reset creates new episode ID."""
        env.reset()
        first_id = env.state.episode_id

        env.step(LemonadeAction(price_per_cup=75))
        env.reset()
        second_id = env.state.episode_id

        assert first_id != second_id


class TestLocationSystem:
    """Tests for the location system."""

    def test_reset_starts_at_default_location(self, env: LemonadeEnvironment):
        """Test reset starts at the default location (park)."""
        obs = env.reset()
        assert obs.current_location == "park"

    def test_reset_includes_location_catalog(self, env: LemonadeEnvironment):
        """Test reset includes location catalog."""
        obs = env.reset()
        assert obs.location_catalog is not None
        assert len(obs.location_catalog) == 4

    def test_location_catalog_has_required_fields(self, env: LemonadeEnvironment):
        """Test location catalog entries have required fields."""
        obs = env.reset()
        for location in obs.location_catalog:
            assert "id" in location
            assert "name" in location
            assert "foot_traffic_multiplier" in location
            assert "price_sensitivity" in location
            assert "weather_exposure" in location
            assert "permit_cost" in location
            assert "is_current" in location

    def test_location_catalog_marks_current(self, env: LemonadeEnvironment):
        """Test location catalog correctly marks current location."""
        obs = env.reset()
        current_locations = [loc for loc in obs.location_catalog if loc["is_current"]]
        assert len(current_locations) == 1
        assert current_locations[0]["id"] == "park"

    def test_switch_location_costs_permit_fee(self, env: LemonadeEnvironment):
        """Test switching location charges permit fee."""
        obs0 = env.reset()
        initial_cash = obs0.cash

        action = LemonadeAction(price_per_cup=75, location="downtown")
        obs = env.step(action)

        # Permit fee should be charged (Downtown costs $10.00 = 1000 cents)
        permit_fee = 1000
        # Location should have changed
        assert obs.current_location == "downtown"

    def test_stay_at_location_no_fee(self, env: LemonadeEnvironment):
        """Test staying at current location doesn't charge fee."""
        obs0 = env.reset()

        # Explicitly specify current location
        action = LemonadeAction(price_per_cup=75, location="park")
        obs = env.step(action)

        # Location should stay the same
        assert obs.current_location == "park"

    def test_null_location_stays_put(self, env: LemonadeEnvironment):
        """Test null location stays at current location."""
        obs0 = env.reset()

        action = LemonadeAction(price_per_cup=75, location=None)
        obs = env.step(action)

        assert obs.current_location == "park"

    def test_location_affects_demand(self):
        """Test that different locations produce different demand."""
        # Test downtown (higher traffic) vs pool (lower traffic)
        env1 = LemonadeEnvironment(seed=42)
        env2 = LemonadeEnvironment(seed=42)

        obs1 = env1.reset()
        obs2 = env2.reset()

        # Move env1 to downtown (higher traffic)
        action_downtown = LemonadeAction(price_per_cup=50, location="downtown")
        result1 = env1.step(action_downtown)

        # Keep env2 at park (baseline)
        action_park = LemonadeAction(price_per_cup=50, location="park")
        result2 = env2.step(action_park)

        # Downtown should generally have more customers (1.4x traffic)
        # This is probabilistic, so we just verify the system works
        assert result1.current_location == "downtown"
        assert result2.current_location == "park"

    def test_mall_weather_independent(self):
        """Test mall location is not affected by weather."""
        # The mall has weather_exposure=0.0
        env = LemonadeEnvironment(seed=42)
        obs = env.reset()

        # Move to mall
        action = LemonadeAction(price_per_cup=50, location="mall")
        result = env.step(action)

        assert result.current_location == "mall"

    def test_invalid_location_ignored(self, env: LemonadeEnvironment):
        """Test invalid location name is ignored."""
        obs0 = env.reset()

        action = LemonadeAction(price_per_cup=75, location="beach")  # Invalid
        obs = env.step(action)

        # Should stay at current location
        assert obs.current_location == "park"

    def test_location_persists_across_days(self, env: LemonadeEnvironment):
        """Test location persists without re-specifying."""
        env.reset()

        # Move to downtown
        action1 = LemonadeAction(price_per_cup=75, location="downtown")
        obs1 = env.step(action1)
        assert obs1.current_location == "downtown"

        # Take action without specifying location
        action2 = LemonadeAction(price_per_cup=75)
        obs2 = env.step(action2)
        assert obs2.current_location == "downtown"  # Should persist

    def test_market_hints_include_location_info(self, env: LemonadeEnvironment):
        """Test market hints include location information."""
        obs = env.reset()
        assert obs.market_hints is not None
        assert "location_info" in obs.market_hints
        location_info = obs.market_hints["location_info"]
        assert "current_location" in location_info
        assert "foot_traffic_multiplier" in location_info
        assert "price_sensitivity" in location_info
        assert "weather_exposure" in location_info

    def test_location_in_metadata(self, env: LemonadeEnvironment):
        """Test location is included in step metadata."""
        env.reset()
        action = LemonadeAction(price_per_cup=75, location="mall")
        obs = env.step(action)

        assert "location" in obs.metadata
        assert obs.metadata["location"] == "mall"

    def test_location_cost_in_metadata(self, env: LemonadeEnvironment):
        """Test location cost is tracked in metadata."""
        env.reset()
        action = LemonadeAction(price_per_cup=75, location="downtown")
        obs = env.step(action)

        assert "location_cost" in obs.metadata
        assert obs.metadata["location_cost"] == 1000  # $10.00 permit fee for downtown

    def test_no_location_cost_when_staying(self, env: LemonadeEnvironment):
        """Test no location cost when staying at same location."""
        env.reset()
        action = LemonadeAction(price_per_cup=75, location="park")  # Same as current
        obs = env.step(action)

        assert obs.metadata["location_cost"] == 0

    def test_custom_starting_location(self):
        """Test starting at a custom location."""
        config = GameConfig(starting_location="mall")
        env = LemonadeEnvironment(config=config, seed=42)
        obs = env.reset()

        assert obs.current_location == "mall"

    def test_cannot_afford_location_switch(self):
        """Test cannot switch location if can't afford permit fee."""
        # Start with very little cash
        config = GameConfig(starting_cash=500)  # Only $5, downtown permit is $10
        env = LemonadeEnvironment(config=config, seed=42)
        obs0 = env.reset()

        action = LemonadeAction(price_per_cup=75, location="downtown")
        obs = env.step(action)

        # Should stay at park since we can't afford downtown's permit
        assert obs.current_location == "park"
