# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""Test ice consumption during lemonade production."""

import pytest
from ..server.lemonade_environment import LemonadeEnvironment
from ..models import LemonadeAction, GameConfig


class TestIceConsumption:
    """Tests for ice consumption during lemonade production."""

    def test_starting_ice_is_available(self):
        """Test that starting ice is properly initialized."""
        env = LemonadeEnvironment(seed=42)
        obs = env.reset()
        
        # Default starting ice is 5
        assert obs.ice_bags == 5, f"Expected 5 starting ice, got {obs.ice_bags}"
        assert env.ice_batches == [(5, 1)], f"Expected ice_batches=[(5, 1)], got {env.ice_batches}"

    def test_ice_consumed_when_making_lemonade(self):
        """Test that ice is consumed when cups are sold."""
        env = LemonadeEnvironment(seed=42)
        obs = env.reset()
        
        print(f"\n=== Initial state ===")
        print(f"ice_bags: {obs.ice_bags}")
        print(f"ice_batches: {env.ice_batches}")
        print(f"weather: {obs.weather}, temp: {obs.temperature}")
        
        # Take action - set reasonable price to ensure sales
        action = LemonadeAction(
            price_per_cup=75,
            buy_lemons=0,
            buy_sugar=0,
            buy_cups=0,
            buy_ice=0,
            advertising_spend=0,
        )
        
        result = env.step(action)
        
        print(f"\n=== After Day 1 ===")
        print(f"cups_sold: {result.cups_sold}")
        print(f"ice_used: {result.ice_used}")
        print(f"ice_melted: {result.ice_melted}")
        print(f"ice_bags remaining: {result.ice_bags}")
        print(f"ice_batches: {env.ice_batches}")
        
        # If cups were sold, ice should have been used
        if result.cups_sold > 0:
            # Calculate expected ice usage
            ice_per_cup = env.config.ice_per_cup  # 0.2
            expected_ice_used = min(result.cups_sold * ice_per_cup, 5)  # capped at starting ice
            
            print(f"\n=== Expected vs Actual ===")
            print(f"cups_sold: {result.cups_sold}")
            print(f"ice_per_cup: {ice_per_cup}")
            print(f"expected_ice_used: {expected_ice_used}")
            print(f"actual_ice_used: {result.ice_used}")
            
            assert result.ice_used > 0, (
                f"Ice should have been used! "
                f"cups_sold={result.cups_sold}, ice_used={result.ice_used}, "
                f"ice_melted={result.ice_melted}"
            )
            
            # Ice that was used shouldn't melt
            # If all 5 ice was used, ice_melted should be 0
            if result.ice_used >= 5:
                assert result.ice_melted == 0, (
                    f"If all ice was used, none should melt! "
                    f"ice_used={result.ice_used}, ice_melted={result.ice_melted}"
                )

    def test_ice_consumption_calculation(self):
        """Test the ice consumption calculation directly."""
        env = LemonadeEnvironment(seed=42)
        env.reset()
        
        # Verify initial state
        assert env.ice_bags == 5
        assert env.ice_batches == [(5, 1)]
        
        # Test _consume_ice directly
        consumed = env._consume_ice(3.0)
        
        print(f"\n=== Direct _consume_ice test ===")
        print(f"Requested: 3.0, Consumed: {consumed}")
        print(f"Remaining ice_batches: {env.ice_batches}")
        print(f"Remaining ice_bags: {env.ice_bags}")
        
        assert consumed == 3.0, f"Expected to consume 3.0, got {consumed}"
        assert env.ice_bags == 2, f"Expected 2 ice remaining, got {env.ice_bags}"

    def test_ice_fully_consumed(self):
        """Test consuming all available ice."""
        env = LemonadeEnvironment(seed=42)
        env.reset()
        
        # Consume all 5 ice
        consumed = env._consume_ice(5.0)
        
        print(f"\n=== Consume all ice test ===")
        print(f"Requested: 5.0, Consumed: {consumed}")
        print(f"Remaining ice_batches: {env.ice_batches}")
        print(f"Remaining ice_bags: {env.ice_bags}")
        
        assert consumed == 5.0, f"Expected to consume 5.0, got {consumed}"
        assert env.ice_bags == 0, f"Expected 0 ice remaining, got {env.ice_bags}"
        assert env.ice_batches == [], f"Expected empty ice_batches, got {env.ice_batches}"

    def test_ice_to_use_calculation(self):
        """Test the ice_to_use calculation logic."""
        env = LemonadeEnvironment(seed=42)
        env.reset()
        
        cups_to_make = 37
        ice_per_cup = env.config.ice_per_cup  # 0.2
        ice_available = env.ice_bags  # 5
        
        ice_to_use = min(cups_to_make * ice_per_cup, ice_available)
        
        print(f"\n=== ice_to_use calculation ===")
        print(f"cups_to_make: {cups_to_make}")
        print(f"ice_per_cup: {ice_per_cup}")
        print(f"cups_to_make * ice_per_cup: {cups_to_make * ice_per_cup}")
        print(f"ice_available: {ice_available}")
        print(f"ice_to_use: {ice_to_use}")
        
        assert ice_to_use == 5.0, f"Expected ice_to_use=5.0, got {ice_to_use}"


class TestAPISerializaton:
    """Test that ice_used is properly serialized in API responses."""
    
    def test_observation_serialization_includes_ice_used(self):
        """Test that asdict includes ice_used field."""
        from dataclasses import asdict
        
        env = LemonadeEnvironment(seed=42)
        obs = env.reset()
        
        # Take an action
        action = LemonadeAction(
            price_per_cup=75,
            buy_lemons=0,
            buy_sugar=0,
            buy_cups=0,
            buy_ice=0,
            advertising_spend=0,
        )
        
        result = env.step(action)
        
        # Serialize like the API does
        obs_dict = asdict(result)
        reward = obs_dict.pop("reward", None)
        done = obs_dict.pop("done", False)
        
        print(f"\n=== Serialized observation ===")
        print(f"ice_used in obs_dict: {'ice_used' in obs_dict}")
        print(f"ice_used value: {obs_dict.get('ice_used')}")
        print(f"ice_melted value: {obs_dict.get('ice_melted')}")
        print(f"cups_sold value: {obs_dict.get('cups_sold')}")
        
        # Verify ice_used is in the serialized dict
        assert 'ice_used' in obs_dict, "ice_used field missing from serialized observation!"
        
        # If cups were sold, ice_used should be > 0
        if obs_dict['cups_sold'] > 0:
            assert obs_dict['ice_used'] > 0, (
                f"ice_used should be > 0 when cups were sold! "
                f"cups_sold={obs_dict['cups_sold']}, ice_used={obs_dict['ice_used']}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

