# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
Lemonade Stand Environment HTTP Client.

This module provides the client for connecting to a Lemonade Stand Environment
server over HTTP.
"""

from typing import Dict

from openenv_core.client_types import StepResult
from openenv_core.env_server.types import State
from openenv_core.http_env_client import HTTPEnvClient

from .models import LemonadeAction, LemonadeObservation


class LemonadeEnv(HTTPEnvClient[LemonadeAction, LemonadeObservation]):
    """
    HTTP client for the Lemonade Stand Environment.
    
    This client connects to a LemonadeEnvironment HTTP server and provides
    methods to interact with it: reset(), step(), and state access.
    
    Example:
        >>> # Connect to a running server
        >>> client = LemonadeEnv(base_url="http://localhost:8000")
        >>> result = client.reset()
        >>> print(f"Day {result.observation.day}: {result.observation.weather}")
        >>>
        >>> # Make decisions for the day (cups made on-demand based on customer demand)
        >>> action = LemonadeAction(
        ...     price_per_cup=75,  # $0.75
        ...     lemons_tier=2, lemons_count=1,  # Buy 1 dozen lemons
        ... )
        >>> result = client.step(action)
        >>> print(f"Sold {result.observation.cups_sold} cups!")
        >>> print(f"Profit: ${result.observation.daily_profit / 100:.2f}")
    
    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = LemonadeEnv.from_docker_image("lemonade-bench:latest")
        >>> result = client.reset()
        >>> # Play the game...
        >>> client.close()
    """
    
    def _step_payload(self, action: LemonadeAction) -> Dict:
        """
        Convert LemonadeAction to JSON payload for step request.
        
        Args:
            action: LemonadeAction instance
            
        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "price_per_cup": action.price_per_cup,
            "lemons_tier": action.lemons_tier,
            "lemons_count": action.lemons_count,
            "sugar_tier": action.sugar_tier,
            "sugar_count": action.sugar_count,
            "cups_tier": action.cups_tier,
            "cups_count": action.cups_count,
            "ice_tier": action.ice_tier,
            "ice_count": action.ice_count,
            "advertising_spend": action.advertising_spend,
            "buy_upgrade": action.buy_upgrade,
            "location": action.location,
        }
    
    def _parse_result(self, payload: Dict) -> StepResult[LemonadeObservation]:
        """
        Parse server response into StepResult[LemonadeObservation].
        
        Args:
            payload: JSON response from server
            
        Returns:
            StepResult with LemonadeObservation
        """
        obs_data = payload.get("observation", {})
        
        observation = LemonadeObservation(
            day=obs_data.get("day", 1),
            weather=obs_data.get("weather", "sunny"),
            temperature=obs_data.get("temperature", 75),
            weather_forecast=obs_data.get("weather_forecast", "sunny"),
            cash=obs_data.get("cash", 0),
            daily_revenue=obs_data.get("daily_revenue", 0),
            daily_costs=obs_data.get("daily_costs", 0),
            daily_profit=obs_data.get("daily_profit", 0),
            cups_sold=obs_data.get("cups_sold", 0),
            cups_wasted=obs_data.get("cups_wasted", 0),
            customers_served=obs_data.get("customers_served", 0),
            customers_turned_away=obs_data.get("customers_turned_away", 0),
            lemons=obs_data.get("lemons", 0),
            sugar_bags=obs_data.get("sugar_bags", 0),
            cups_available=obs_data.get("cups_available", 0),
            ice_bags=obs_data.get("ice_bags", 0),
            lemons_expiring_tomorrow=obs_data.get("lemons_expiring_tomorrow", 0),
            ice_expiring_tomorrow=obs_data.get("ice_expiring_tomorrow", 0),
            lemons_spoiled=obs_data.get("lemons_spoiled", 0),
            ice_melted=obs_data.get("ice_melted", 0),
            customer_satisfaction=obs_data.get("customer_satisfaction", 0.5),
            reputation=obs_data.get("reputation", 0.5),
            days_remaining=obs_data.get("days_remaining", 0),
            total_profit=obs_data.get("total_profit", 0),
            owned_upgrades=obs_data.get("owned_upgrades", []),
            upgrade_catalog=obs_data.get("upgrade_catalog"),
            current_location=obs_data.get("current_location", "park"),
            location_catalog=obs_data.get("location_catalog"),
            market_hints=obs_data.get("market_hints"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )
    
    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.
        
        Args:
            payload: JSON response from /state endpoint
            
        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
