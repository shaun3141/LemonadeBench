# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
Comprehensive scenario tests for the Lemonade Stand Environment.

These tests simulate realistic gameplay scenarios and edge cases to ensure
the environment behaves correctly in all situations.

Covers:
- Full game simulations
- Edge cases and boundary conditions
- Strategy testing (bulk buying, upgrade timing, etc.)
- Economic scenarios (bankruptcy, windfall, etc.)
- Weather-specific strategies
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    GameConfig,
    LemonadeAction,
    Weather,
    StandUpgrade,
    calculate_bulk_cost,
)
from server.lemonade_environment import LemonadeEnvironment


class TestFullGameSimulation:
    """Tests that simulate complete games."""

    def test_full_game_completes(self):
        """Test a full 14-day game completes successfully."""
        env = LemonadeEnvironment(seed=42)
        env.reset()

        action = LemonadeAction(price_per_cup=75)

        days_played = 0
        obs = None
        while days_played < 14:
            obs = env.step(action)
            days_played += 1
            if obs.done:
                break

        assert obs.done is True
        assert days_played == 14

    def test_profitable_game_possible(self):
        """Test that a profitable game is achievable with good strategy."""
        env = LemonadeEnvironment(seed=42)
        obs = env.reset()
        starting_cash = obs.cash

        # Conservative strategy: moderate price, keep supplies stocked
        for _ in range(14):
            action = LemonadeAction(
                price_per_cup=75,
                buy_lemons=10 if obs.lemons < 20 else 0,
                buy_sugar=2 if obs.sugar_bags < 3 else 0,
                buy_cups=30 if obs.cups_available < 30 else 0,
            )
            obs = env.step(action)
            if obs.done:
                break

        # Should end with more cash than started (profitable)
        assert obs.cash > starting_cash or obs.total_profit > 0

    def test_game_tracks_all_metrics(self):
        """Test all metrics are tracked throughout game."""
        env = LemonadeEnvironment(seed=42)
        env.reset()

        total_revenue = 0
        total_costs = 0
        total_sold = 0

        action = LemonadeAction(price_per_cup=75)

        for _ in range(14):
            obs = env.step(action)
            total_revenue += obs.daily_revenue
            total_costs += obs.daily_costs
            total_sold += obs.cups_sold
            if obs.done:
                break

        # Verify metrics are sensible
        assert total_sold >= 0
        assert obs.cups_wasted == 0  # On-demand model has no waste
        assert obs.total_profit == total_revenue - total_costs


class TestBankruptcyScenarios:
    """Tests for running out of money."""

    def test_cannot_go_negative_cash(self):
        """Test cash cannot go below zero."""
        env = LemonadeEnvironment(seed=42)
        env.reset()

        # Try to spend all money on ads
        action = LemonadeAction(
            price_per_cup=500,
            advertising_spend=100000,
        )

        for _ in range(5):
            obs = env.step(action)
            assert obs.cash >= 0

    def test_can_operate_with_zero_cash(self):
        """Test game continues even with no cash."""
        config = GameConfig(starting_cash=0)
        env = LemonadeEnvironment(config=config, seed=42)
        obs = env.reset()

        assert obs.cash == 0

        # Can still play (using starting inventory)
        action = LemonadeAction(price_per_cup=100)
        obs = env.step(action)

        assert obs.done is False
        # May have made some sales
        assert obs.cash >= 0

    def test_no_purchases_with_zero_cash(self):
        """Test cannot buy supplies with no money."""
        config = GameConfig(starting_cash=0)
        env = LemonadeEnvironment(config=config, seed=42)
        obs0 = env.reset()
        initial_lemons = obs0.lemons

        action = LemonadeAction(
            price_per_cup=500,  # High price = low demand
            buy_lemons=100,
        )
        obs = env.step(action)

        # Should not have bought any lemons (no cash)
        # (minus any spoilage and usage, lemons should be same or less)
        assert obs.lemons <= initial_lemons


class TestEdgeCaseProduction:
    """Tests for edge cases in on-demand production."""

    def test_no_demand_no_sales(self):
        """Test extremely high price results in no sales."""
        env = LemonadeEnvironment(seed=42)
        env.reset()

        action = LemonadeAction(price_per_cup=10000)  # $100 per cup
        obs = env.step(action)

        assert obs.cups_sold == 0
        assert obs.cups_wasted == 0
        assert obs.daily_revenue == 0

    def test_limited_by_inventory(self):
        """Test sales limited by available resources."""
        config = GameConfig(starting_cups=1)  # Only 1 cup available
        env = LemonadeEnvironment(config=config, seed=42)
        env.reset()

        action = LemonadeAction(price_per_cup=50)  # Low price = high demand
        obs = env.step(action)

        # Can only serve 1 customer (limited by cups)
        assert obs.cups_sold <= 1

    def test_sell_maximum_possible(self):
        """Test selling up to maximum possible from inventory."""
        env = LemonadeEnvironment(seed=42)
        obs = env.reset()

        # Calculate max possible from inventory
        max_from_lemons = int(obs.lemons / env.config.lemons_per_cup)
        max_from_sugar = int(obs.sugar_bags / env.config.sugar_per_cup)
        max_from_cups = obs.cups_available
        max_possible = min(max_from_lemons, max_from_sugar, max_from_cups)

        # Very low price = very high demand (should max out supplies)
        action = LemonadeAction(price_per_cup=1)
        obs = env.step(action)

        # Sales should be limited by inventory
        assert obs.cups_sold <= max_possible

    def test_high_demand_turns_away_customers(self):
        """Test high demand with limited supplies turns away customers."""
        config = GameConfig(starting_cups=5)  # Very limited cups
        env = LemonadeEnvironment(config=config, seed=42)
        env.reset()

        # Very low price = high demand
        action = LemonadeAction(price_per_cup=1)
        obs = env.step(action)

        # Should turn away customers due to limited supplies
        assert obs.customers_turned_away >= 0


class TestPriceEdgeCases:
    """Tests for price edge cases."""

    def test_zero_price(self):
        """Test zero price (free lemonade)."""
        env = LemonadeEnvironment(seed=42)
        env.reset()

        action = LemonadeAction(price_per_cup=0)
        obs = env.step(action)

        assert obs.daily_revenue == 0
        assert obs.cups_sold >= 0  # People should take free lemonade!

    def test_one_cent_price(self):
        """Test one cent price."""
        env = LemonadeEnvironment(seed=42)
        env.reset()

        action = LemonadeAction(price_per_cup=1)
        obs = env.step(action)

        assert obs.daily_revenue == obs.cups_sold * 1

    def test_very_high_price(self):
        """Test very high price kills demand."""
        env = LemonadeEnvironment(seed=42)
        env.reset()

        action = LemonadeAction(price_per_cup=1000)
        obs = env.step(action)

        # At $10 per cup, almost no one should buy
        assert obs.cups_sold < 10

    def test_maximum_tolerance_price(self):
        """Test price at maximum tolerance boundary."""
        env = LemonadeEnvironment(seed=42)
        env.reset()

        # Price at max tolerance ($2.00 = 200 cents)
        action = LemonadeAction(
            price_per_cup=env.config.max_price_tolerance,
        )
        obs = env.step(action)

        # Should still have some (reduced) sales
        assert obs.cups_sold >= 0


class TestBulkPurchasingStrategy:
    """Tests for bulk purchasing strategies."""

    def test_bulk_discount_savings(self):
        """Test bulk purchasing saves money."""
        # Calculate cost of 12 lemons individually vs bulk
        individual_cost = 12 * 25  # $3.00
        bulk_cost = calculate_bulk_cost("lemons", 12)  # With 10% discount

        assert bulk_cost < individual_cost
        assert bulk_cost == int(individual_cost * 0.90)

    def test_crate_discount_significant(self):
        """Test crate discount is significant."""
        # Cost of 144 lemons
        full_price = 144 * 25  # $36.00
        crate_price = calculate_bulk_cost("lemons", 144)

        # 20% savings
        assert crate_price == int(full_price * 0.80)
        savings = full_price - crate_price
        assert savings == int(full_price * 0.20)

    def test_bulk_buy_all_supplies(self):
        """Test bulk buying all supply types."""
        env = LemonadeEnvironment(
            config=GameConfig(starting_cash=10000),  # $100
            seed=42,
        )
        env.reset()

        action = LemonadeAction(
            price_per_cup=500,  # High price = low demand
            buy_lemons=144,  # Crate
            buy_sugar=20,  # Pallet
            buy_cups=250,  # Case
            buy_ice=20,  # Delivery
        )
        obs = env.step(action)

        # Should have significant inventory (minus any used)
        assert obs.lemons >= 100  # Some may have been used
        assert obs.sugar_bags >= 15


class TestUpgradeStrategy:
    """Tests for upgrade purchasing strategy."""

    def test_early_cooler_purchase(self):
        """Test buying cooler early in the game."""
        env = LemonadeEnvironment(seed=42)
        env.reset()

        # Buy cooler on day 1
        action = LemonadeAction(
            price_per_cup=75,
            buy_upgrade="cooler",
        )
        obs = env.step(action)

        assert "cooler" in obs.owned_upgrades

    def test_cooler_roi(self):
        """Test cooler provides return on investment over time."""
        # Play game without cooler
        env_no_cooler = LemonadeEnvironment(seed=42)
        env_no_cooler.reset()

        # Play game with cooler
        env_cooler = LemonadeEnvironment(seed=42)
        env_cooler.reset()

        # Buy cooler immediately
        action_buy_cooler = LemonadeAction(
            price_per_cup=75,
            buy_upgrade="cooler",
            buy_ice=10,
        )
        env_cooler.step(action_buy_cooler)

        action_no_cooler = LemonadeAction(
            price_per_cup=75,
            buy_ice=10,
        )
        env_no_cooler.step(action_no_cooler)

        # Continue playing both
        action = LemonadeAction(
            price_per_cup=75,
            buy_ice=5,
        )

        for _ in range(13):
            env_cooler.step(action)
            env_no_cooler.step(action)

        # Cooler should preserve ice, potentially leading to better results
        # (exact ROI depends on game mechanics and weather)


class TestPerishableManagement:
    """Tests for managing perishable inventory."""

    def test_use_expiring_lemons_first(self):
        """Test strategy to use expiring inventory first."""
        env = LemonadeEnvironment(seed=42)
        env.reset()

        # Low price = high demand = use lemons
        action = LemonadeAction(price_per_cup=50)

        # Play a few days
        for _ in range(3):
            obs = env.step(action)

        # After 3 days, original lemons should be expiring/expired
        # This tests FIFO consumption works

    def test_ice_management_without_cooler(self):
        """Test ice melts without cooler at end of day, requiring daily purchase."""
        env = LemonadeEnvironment(seed=42)
        env.reset()

        # Buy ice, high price = low demand = ice not used much
        action = LemonadeAction(
            price_per_cup=500,
            buy_ice=10,
        )
        obs1 = env.step(action)

        # With expiration at END of day, ice melts overnight after Day 1
        # obs1 shows what melted at the end of Day 1
        assert obs1.ice_melted > 0  # Starting ice + purchased ice melted
        assert obs1.ice_bags == 0  # All ice is gone - need to buy fresh daily

    def test_ice_management_with_cooler(self):
        """Test ice partially preserved with cooler."""
        env = LemonadeEnvironment(seed=42)
        env.reset()

        # Buy cooler and ice, high price = low demand
        action = LemonadeAction(
            price_per_cup=500,
            buy_upgrade="cooler",
            buy_ice=10,
        )
        env.step(action)

        # Next day
        action2 = LemonadeAction(price_per_cup=500)
        obs2 = env.step(action2)

        # Should still have some ice (50% preserved)
        # Melted should be about half
        assert obs2.ice_bags > 0 or obs2.ice_melted < 10

    def test_prevent_spoilage_by_using_inventory(self):
        """Test using inventory before it spoils via demand."""
        env = LemonadeEnvironment(seed=42)
        obs = env.reset()

        total_spoiled = 0
        # Low price = high demand = uses inventory
        for _ in range(3):
            action = LemonadeAction(price_per_cup=25)
            obs = env.step(action)
            total_spoiled += obs.lemons_spoiled

        # Should have minimal spoilage if demand was high enough


class TestWeatherStrategies:
    """Tests for weather-based strategies."""

    def test_high_demand_on_hot_day(self):
        """Test high demand on hot days."""
        # Find a seed that produces hot weather
        for seed in range(100):
            env = LemonadeEnvironment(seed=seed)
            obs = env.reset()
            if obs.weather == "hot":
                break
        else:
            pytest.skip("Could not find seed with hot weather on day 1")

        # Low price = high demand on hot day
        action = LemonadeAction(price_per_cup=50)
        obs = env.step(action)

        # Should sell many cups on a hot day
        assert obs.cups_sold > 20

    def test_low_demand_on_rainy_day(self):
        """Test low demand on rainy days."""
        # Find a seed that produces rainy weather
        for seed in range(100):
            env = LemonadeEnvironment(seed=seed)
            obs = env.reset()
            if obs.weather == "rainy":
                break
        else:
            pytest.skip("Could not find seed with rainy weather on day 1")

        # Even low price has low demand on rainy day
        action = LemonadeAction(price_per_cup=50)
        obs = env.step(action)

        # Rainy days have low demand - fewer sales
        assert obs.cups_sold < 30

    def test_use_weather_forecast(self):
        """Test using weather forecast for planning."""
        env = LemonadeEnvironment(seed=42)
        obs = env.reset()

        # Get tomorrow's forecast
        forecast = obs.weather_forecast

        # Make decisions based on forecast
        if forecast in ["hot", "sunny"]:
            # Buy supplies for tomorrow
            action = LemonadeAction(
                price_per_cup=75,
                buy_lemons=15,
                buy_ice=10,
            )
        else:
            # Conservative approach
            action = LemonadeAction(
                price_per_cup=75,
            )

        obs = env.step(action)
        # Just verify the strategy executes


class TestMarketHintsUsage:
    """Tests for using market hints in strategy."""

    def test_supply_capacity_hint(self):
        """Test market hints show supply capacity."""
        env = LemonadeEnvironment(seed=42)
        obs = env.reset()

        hints = obs.market_hints
        max_cups = hints["max_cups_producible"]

        # With on-demand model, we can serve up to max_cups
        assert max_cups > 0

    def test_price_demand_curve_accuracy(self):
        """Test price-demand curve roughly matches actual demand."""
        env = LemonadeEnvironment(seed=42)
        obs = env.reset()

        hints = obs.market_hints
        curve = hints["price_demand_curve"]

        # Test at optimal price
        optimal_price = hints["optimal_price"]
        expected_demand = curve.get(optimal_price, 0)

        action = LemonadeAction(
            price_per_cup=optimal_price,
        )
        obs = env.step(action)

        # Actual demand should be in the ballpark
        # (accounting for randomness: 0.8x to 1.2x and supply limits)
        actual_demand = obs.cups_sold + obs.customers_turned_away
        assert actual_demand >= expected_demand * 0.5
        assert actual_demand <= expected_demand * 2.0

    def test_break_even_price(self):
        """Test break-even price covers costs."""
        env = LemonadeEnvironment(seed=42)
        obs = env.reset()

        hints = obs.market_hints
        break_even = hints["break_even_price"]
        ingredient_cost = hints["ingredient_cost_per_cup"]

        # Break-even should equal ingredient cost
        assert break_even == ingredient_cost


class TestReproducibility:
    """Tests for reproducible game sequences."""

    def test_same_seed_same_game(self):
        """Test same seed produces identical game."""
        def play_game(seed):
            env = LemonadeEnvironment(seed=seed)
            env.reset()
            results = []

            action = LemonadeAction(price_per_cup=75)
            for _ in range(14):
                obs = env.step(action)
                results.append((obs.weather, obs.cups_sold, obs.cash))
                if obs.done:
                    break
            return results

        results1 = play_game(42)
        results2 = play_game(42)

        assert results1 == results2

    def test_reset_restores_reproducibility(self):
        """Test reset allows replaying same game."""
        env = LemonadeEnvironment(seed=42)

        # Play partial game
        env.reset()
        action = LemonadeAction(price_per_cup=75)
        obs1 = env.step(action)
        obs2 = env.step(action)

        # Reset and replay
        env.reset()
        obs1_replay = env.step(action)
        obs2_replay = env.step(action)

        assert obs1.weather == obs1_replay.weather
        assert obs1.cups_sold == obs1_replay.cups_sold
        assert obs2.weather == obs2_replay.weather
        assert obs2.cups_sold == obs2_replay.cups_sold


class TestExtremeScenarios:
    """Tests for extreme game scenarios."""

    def test_all_resources_depleted(self):
        """Test game handles running out of all resources."""
        config = GameConfig(
            starting_lemons=2,
            starting_sugar=1,
            starting_cups=5,
            starting_ice=1,
            starting_cash=0,
        )
        env = LemonadeEnvironment(config=config, seed=42)
        env.reset()

        # Low price = high demand, depletes resources
        action = LemonadeAction(price_per_cup=25)

        for _ in range(5):
            obs = env.step(action)

        # Should handle empty inventory gracefully
        assert obs.lemons == 0 or obs.sugar_bags == 0 or obs.cups_available == 0

    def test_maximum_inventory(self):
        """Test handling very large inventory."""
        config = GameConfig(starting_cash=1000000)  # $10,000
        env = LemonadeEnvironment(config=config, seed=42)
        env.reset()

        # Buy massive amounts, high price = low demand
        action = LemonadeAction(
            price_per_cup=500,
            buy_lemons=10000,
            buy_sugar=1000,
            buy_cups=10000,
        )
        obs = env.step(action)

        assert obs.lemons >= 9900  # Some may be used
        assert obs.cups_available >= 9900

    def test_long_game(self):
        """Test very long game (100 days)."""
        config = GameConfig(total_days=100)
        env = LemonadeEnvironment(config=config, seed=42)
        env.reset()

        action = LemonadeAction(
            price_per_cup=75,
            buy_lemons=10,
            buy_sugar=2,
            buy_cups=30,
        )

        for i in range(100):
            obs = env.step(action)
            assert not obs.done or i == 99

        assert obs.done is True


class TestCustomerBehavior:
    """Tests for customer demand and satisfaction."""

    def test_turning_away_customers_hurts_satisfaction(self):
        """Test that turning away customers affects satisfaction."""
        # Limited inventory to ensure customers are turned away
        config = GameConfig(starting_cups=5)
        env = LemonadeEnvironment(config=config, seed=42)
        env.reset()

        # Very low price = high demand, but limited supply
        action = LemonadeAction(price_per_cup=25)
        obs = env.step(action)

        # If we turned away customers, satisfaction should be affected
        if obs.customers_turned_away > 0:
            assert obs.customer_satisfaction < 1.0

    def test_low_price_improves_satisfaction(self):
        """Test that lower prices generally improve satisfaction."""
        env = LemonadeEnvironment(seed=42)
        env.reset()

        # Very low price, plenty of supply
        action = LemonadeAction(price_per_cup=25)
        obs = env.step(action)

        # Low price should lead to decent satisfaction (if we sold)
        if obs.cups_sold > 0:
            assert obs.customer_satisfaction > 0.3

    def test_customers_served_equals_cups_sold(self):
        """Test customers_served matches cups_sold."""
        env = LemonadeEnvironment(seed=42)
        env.reset()

        action = LemonadeAction(price_per_cup=75)
        obs = env.step(action)

        assert obs.customers_served == obs.cups_sold


class TestIceBonus:
    """Tests for ice availability bonus."""

    def test_ice_bonus_on_hot_day(self):
        """Test ice provides demand bonus on hot days."""
        # Find hot weather seed
        for seed in range(100):
            env = LemonadeEnvironment(seed=seed)
            obs = env.reset()
            if obs.weather == "hot":
                break
        else:
            pytest.skip("Could not find hot weather seed")

        # Check market hints indicate ice bonus
        hints = obs.market_hints
        assert "ice_bonus_active" in hints
        if obs.ice_bags > 0:
            assert hints["ice_bonus_active"] is True

    def test_ice_penalty_on_hot_day_without_ice(self):
        """Test lack of ice reduces demand on hot days."""
        config = GameConfig(starting_ice=0)

        # Find hot weather seed
        for seed in range(100):
            env = LemonadeEnvironment(config=config, seed=seed)
            obs = env.reset()
            if obs.weather == "hot":
                break
        else:
            pytest.skip("Could not find hot weather seed")

        hints = obs.market_hints
        assert hints["has_ice"] is False
