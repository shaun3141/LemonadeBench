# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
Gymnasium-compatible wrapper for the Lemonade Stand Environment.

Enables training with standard RL libraries like Stable Baselines3, RLlib, and CleanRL
by providing a standard Gymnasium interface with flat action/observation spaces.
"""

from typing import Any, Optional, SupportsFloat

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    raise ImportError(
        "gymnasium is required for RL training. "
        "Install with: pip install gymnasium"
    )

from ...server.lemonade_environment import LemonadeEnvironment
from ...models import GameConfig
from .spaces import (
    OBSERVATION_DIM,
    ACTION_DIM,
    MIXED_ACTION_CONTINUOUS_DIM,
    MIXED_ACTION_LOCATION_DIM,
    MIXED_ACTION_UPGRADE_DIM,
    encode_observation,
    decode_action,
    decode_mixed_action,
)


class LemonadeGymEnv(gym.Env):
    """
    Gymnasium-compatible wrapper for the Lemonade Stand Environment.
    
    This wrapper converts the structured LemonadeAction/LemonadeObservation
    to flat numpy arrays suitable for RL algorithms.
    
    Action Space (Box):
        8-dimensional continuous action in [0, 1]:
        - [0] price: maps to [25, 200] cents
        - [1] lemons_qty: maps to [0, 50] units (converted to tier+count)
        - [2] sugar_qty: maps to [0, 20] bags (converted to tier+count)
        - [3] cups_qty: maps to [0, 100] cups (converted to tier+count)
        - [4] ice_qty: maps to [0, 30] bags (converted to tier+count)
        - [5] advertising: maps to [0, 500] cents
        - [6] location: selects from 4 locations
        - [7] buy_upgrade: threshold for buying upgrade
    
    Observation Space (Box):
        27-dimensional continuous observation in [0, 1]:
        - Day progress, weather (one-hot), forecast (one-hot), temperature
        - Cash, inventory levels, expiring items
        - Reputation, satisfaction, days remaining
        - Location (one-hot), owned upgrades
    
    Reward Shaping (optional, controlled by reward_shaping parameter):
        In addition to the base daily profit reward, shaped rewards can include:
        - Inventory efficiency bonus (minimize waste from spoilage)
        - Reputation growth reward (encourage building customer loyalty)
        - Customer satisfaction bonus (reward keeping customers happy)
        - Sales efficiency bonus (reward selling more of potential demand)
    
    Example:
        >>> from lemonade_bench.agents.rl import LemonadeGymEnv
        >>> env = LemonadeGymEnv(seed=42)
        >>> obs, info = env.reset()
        >>> action = env.action_space.sample()
        >>> obs, reward, terminated, truncated, info = env.step(action)
    
    With Stable Baselines3:
        >>> from stable_baselines3 import PPO
        >>> from lemonade_bench.agents.rl import LemonadeGymEnv
        >>> env = LemonadeGymEnv()
        >>> model = PPO("MlpPolicy", env, verbose=1)
        >>> model.learn(total_timesteps=100_000)
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        config: Optional[GameConfig] = None,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        max_invalid_actions: int = 3,
        reward_shaping: bool = True,
        reward_shaping_scale: float = 0.1,
        use_mixed_actions: bool = False,
        randomize_seed: bool = True,
    ):
        """
        Initialize the Gymnasium wrapper.
        
        Args:
            config: Game configuration (uses defaults if None)
            seed: Random seed for reproducibility (used to seed the RNG for randomize_seed)
            render_mode: Render mode ("human" for text output, None for no rendering)
            max_invalid_actions: Max consecutive invalid actions before forcing a valid one
            reward_shaping: Enable dense reward shaping (default: True)
            reward_shaping_scale: Scale factor for shaped rewards (default: 0.1)
            use_mixed_actions: Use mixed discrete/continuous action space (default: False)
                If True, action space is a Dict with:
                - "continuous": Box(6,) for price, quantities, advertising
                - "location": Discrete(4) for location choice
                - "upgrade": Discrete(2) for upgrade decision
                If False, uses flat Box(8,) action space (compatible with all algorithms)
            randomize_seed: If True, generate a new random seed each episode for better
                generalization. If False, use the same seed every episode (for debugging
                or reproducing specific scenarios). Default: True for training.
        """
        super().__init__()
        
        self.config = config or GameConfig()
        self._seed = seed
        self._base_seed = seed  # Store original seed for RNG
        self.render_mode = render_mode
        self.max_invalid_actions = max_invalid_actions
        self.reward_shaping = reward_shaping
        self.reward_shaping_scale = reward_shaping_scale
        self.use_mixed_actions = use_mixed_actions
        self.randomize_seed = randomize_seed
        
        # RNG for generating episode seeds (seeded by base_seed for reproducibility)
        import random
        self._seed_rng = random.Random(seed)
        self._episode_count = 0
        
        # Create the underlying environment
        self._env = LemonadeEnvironment(config=self.config, seed=seed)
        
        # Track previous state for reward shaping
        self._prev_reputation = 0.5
        self._prev_cash = self.config.starting_cash
        
        # Define action space
        if use_mixed_actions:
            # Mixed action space: Dict with continuous + discrete components
            # More natural representation but requires algorithms that support Dict spaces
            self.action_space = spaces.Dict({
                "continuous": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(MIXED_ACTION_CONTINUOUS_DIM,),
                    dtype=np.float32,
                ),
                "location": spaces.Discrete(MIXED_ACTION_LOCATION_DIM),
                "upgrade": spaces.Discrete(MIXED_ACTION_UPGRADE_DIM),
            })
        else:
            # Flat action space: 8-dimensional continuous in [0, 1]
            # Compatible with all standard RL algorithms
            self.action_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(ACTION_DIM,),
                dtype=np.float32,
            )
        
        # Define observation space: 27-dimensional continuous in [0, 1]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBSERVATION_DIM,),
            dtype=np.float32,
        )
        
        # Track current observation for owned_upgrades reference
        self._current_obs = None
        self._consecutive_invalid = 0
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the environment and return initial observation.
        
        Args:
            seed: Optional seed to reset with (overrides constructor seed and randomize_seed)
            options: Additional options. Supports:
                - "seed": Override seed for this episode
                - "randomize": Override randomize_seed setting for this episode
            
        Returns:
            Tuple of (observation, info dict)
        """
        self._episode_count += 1
        
        # Determine the seed for this episode
        if seed is not None:
            # Explicit seed passed - use it
            episode_seed = seed
        elif options and "seed" in options:
            # Seed in options
            episode_seed = options["seed"]
        elif self.randomize_seed:
            # Generate a new random seed for this episode
            # This ensures the agent sees diverse weather patterns during training
            episode_seed = self._seed_rng.randint(0, 2**31 - 1)
        else:
            # Use the fixed seed (same game every episode)
            episode_seed = self._seed
        
        # Create new environment with this episode's seed
        self._env = LemonadeEnvironment(config=self.config, seed=episode_seed)
        
        super().reset(seed=seed)
        
        # Reset the underlying environment
        obs = self._env.reset()
        self._current_obs = obs
        self._consecutive_invalid = 0
        
        # Reset reward shaping state
        self._prev_reputation = obs.reputation
        self._prev_cash = obs.cash
        
        # Encode observation
        encoded_obs = encode_observation(obs)
        
        # Build info dict with raw observation data
        info = self._build_info(obs)
        info["episode_seed"] = episode_seed  # Include seed in info for debugging
        info["episode_number"] = self._episode_count
        
        if self.render_mode == "human":
            self._render_obs(obs, is_reset=True)
        
        return encoded_obs, info
    
    def step(
        self,
        action,
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Either:
                - 8-dimensional numpy array in [0, 1] (flat action space)
                - Dict with "continuous", "location", "upgrade" keys (mixed action space)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Get current owned upgrades for action decoding
        owned_upgrades = self._current_obs.owned_upgrades if self._current_obs else []
        
        # Decode action to LemonadeAction based on action space type
        if self.use_mixed_actions:
            # Mixed action space: Dict with continuous + discrete components
            lemonade_action = decode_mixed_action(
                action["continuous"],
                action["location"],
                action["upgrade"],
                owned_upgrades,
            )
        else:
            # Flat action space: 8-dimensional continuous
            lemonade_action = decode_action(action, owned_upgrades)
        
        # Execute step
        obs = self._env.step(lemonade_action)
        self._current_obs = obs
        
        # Handle error responses (invalid actions)
        if obs.is_error_response:
            self._consecutive_invalid += 1
            
            # If too many consecutive invalid actions, force a minimal valid action
            if self._consecutive_invalid >= self.max_invalid_actions:
                from ...models import LemonadeAction as LA
                # Minimal action: just set price, no purchases
                safe_action = LA(price_per_cup=75)
                obs = self._env.step(safe_action)
                self._current_obs = obs
                self._consecutive_invalid = 0
                
                # Continue with normal processing below
                lemonade_action = safe_action
            else:
                # Invalid action - return same state with small negative reward
                encoded_obs = encode_observation(obs)
                info = self._build_info(obs)
                info["action_errors"] = obs.action_errors
                info["action"] = lemonade_action
                info["consecutive_invalid"] = self._consecutive_invalid
                
                if self.render_mode == "human":
                    self._render_error(obs, lemonade_action)
                
                # Small penalty for invalid actions to discourage them
                return encoded_obs, -0.1, False, False, info
        else:
            self._consecutive_invalid = 0
        
        # Encode observation
        encoded_obs = encode_observation(obs)
        
        # Get base reward (already computed by environment)
        base_reward = obs.reward if obs.reward is not None else 0.0
        
        # Calculate shaped rewards if enabled
        shaped_reward = 0.0
        reward_components = {}
        
        if self.reward_shaping:
            shaped_reward, reward_components = self._calculate_shaped_rewards(obs)
        
        # Total reward
        reward = base_reward + shaped_reward
        
        # Update state tracking for next step
        self._prev_reputation = obs.reputation
        self._prev_cash = obs.cash
        
        # Check termination
        terminated = obs.done
        truncated = False  # We don't use truncation
        
        # Build info dict
        info = self._build_info(obs)
        info["action"] = lemonade_action
        info["base_reward"] = base_reward
        info["shaped_reward"] = shaped_reward
        info["reward_components"] = reward_components
        
        if self.render_mode == "human":
            self._render_obs(obs, action=lemonade_action)
        
        return encoded_obs, reward, terminated, truncated, info
    
    def _calculate_shaped_rewards(self, obs) -> tuple[float, dict[str, float]]:
        """
        Calculate dense shaped rewards for better learning signals.
        
        These rewards provide intermediate feedback beyond just daily profit,
        helping the agent learn good behaviors faster.
        
        Returns:
            Tuple of (total_shaped_reward, component_dict)
        """
        components = {}
        scale = self.reward_shaping_scale
        
        # 1. Reputation growth reward
        # Encourage building and maintaining reputation
        reputation_delta = obs.reputation - self._prev_reputation
        reputation_bonus = reputation_delta * 5.0 * scale  # Scale by 5 since delta is small
        components["reputation_growth"] = reputation_bonus
        
        # 2. Customer satisfaction bonus
        # Reward keeping customers happy (satisfaction above 0.5)
        if obs.customer_satisfaction > 0.5:
            satisfaction_bonus = (obs.customer_satisfaction - 0.5) * scale
        else:
            satisfaction_bonus = (obs.customer_satisfaction - 0.5) * scale * 0.5  # Smaller penalty
        components["satisfaction"] = satisfaction_bonus
        
        # 3. Inventory efficiency bonus
        # Penalize spoilage (wasted inventory)
        spoilage_penalty = 0.0
        if obs.lemons_spoiled > 0:
            spoilage_penalty -= (obs.lemons_spoiled / 20.0) * scale  # Normalize by typical batch
        if obs.ice_melted > 0:
            spoilage_penalty -= (obs.ice_melted / 10.0) * scale * 0.5  # Ice melting is normal
        components["inventory_efficiency"] = spoilage_penalty
        
        # 4. Sales efficiency bonus
        # Reward meeting demand (not turning away customers)
        if obs.cups_sold + obs.customers_turned_away > 0:
            fill_rate = obs.cups_sold / (obs.cups_sold + obs.customers_turned_away)
            # Bonus for high fill rate, small penalty for low
            if fill_rate > 0.8:
                sales_bonus = (fill_rate - 0.8) * scale
            elif fill_rate < 0.5:
                sales_bonus = (fill_rate - 0.5) * scale * 0.3  # Small penalty
            else:
                sales_bonus = 0.0
            components["sales_efficiency"] = sales_bonus
        else:
            components["sales_efficiency"] = 0.0
        
        # 5. Cash flow bonus
        # Small reward for positive cash flow (surviving)
        cash_delta = obs.cash - self._prev_cash
        if cash_delta > 0:
            cash_bonus = min(cash_delta / 1000.0, 0.5) * scale  # Cap the bonus
        else:
            cash_bonus = max(cash_delta / 2000.0, -0.25) * scale  # Smaller penalty
        components["cash_flow"] = cash_bonus
        
        # Sum all components
        total_shaped_reward = sum(components.values())
        
        return total_shaped_reward, components
    
    def _build_info(self, obs) -> dict[str, Any]:
        """Build info dict with raw observation data."""
        return {
            "day": obs.day,
            "weather": obs.weather,
            "temperature": obs.temperature,
            "cash": obs.cash,
            "daily_profit": obs.daily_profit,
            "daily_revenue": obs.daily_revenue,
            "daily_costs": obs.daily_costs,
            "cups_sold": obs.cups_sold,
            "customers_served": obs.customers_served,
            "customers_turned_away": obs.customers_turned_away,
            "total_profit": obs.total_profit,
            "reputation": obs.reputation,
            "lemons": obs.lemons,
            "sugar_bags": obs.sugar_bags,
            "cups_available": obs.cups_available,
            "ice_bags": obs.ice_bags,
            "owned_upgrades": obs.owned_upgrades,
            "current_location": obs.current_location,
            "done": obs.done,
        }
    
    def _render_obs(self, obs, is_reset: bool = False, action=None):
        """Render observation to console."""
        if is_reset:
            print("=" * 60)
            print("LemonadeBench - RL Training")
            print("=" * 60)
            print(f"Day 1: {obs.weather.upper()} ({obs.temperature}F)")
            print(f"Starting cash: ${obs.cash / 100:.2f}")
            print()
        else:
            print(f"Day {obs.day}: {obs.weather.upper()} ({obs.temperature}F)")
            if action:
                print(f"  Action: ${action.price_per_cup/100:.2f}/cup, "
                      f"buy: T{action.lemons_tier}x{action.lemons_count}L/T{action.sugar_tier}x{action.sugar_count}S/"
                      f"T{action.cups_tier}x{action.cups_count}C/T{action.ice_tier}x{action.ice_count}I")
            print(f"  Sold: {obs.cups_sold} cups, Profit: ${obs.daily_profit/100:.2f}")
            print(f"  Cash: ${obs.cash/100:.2f}, Reputation: {obs.reputation:.2f}")
            
            if obs.done:
                print()
                print("=" * 60)
                print(f"GAME OVER - Total Profit: ${obs.total_profit/100:.2f}")
                print("=" * 60)
    
    def _render_error(self, obs, action):
        """Render error response to console."""
        print(f"Day {obs.day}: INVALID ACTION")
        print(f"  Errors: {obs.action_errors}")
        if action:
            print(f"  Attempted: ${action.price_per_cup/100:.2f}/cup")
    
    def render(self):
        """Render is handled automatically when render_mode='human'."""
        pass
    
    def close(self):
        """Clean up resources."""
        pass


def make_env(
    seed: int = 0,
    config: Optional[GameConfig] = None,
    render_mode: Optional[str] = None,
    max_invalid_actions: int = 3,
    reward_shaping: bool = True,
    reward_shaping_scale: float = 0.1,
    use_mixed_actions: bool = False,
    randomize_seed: bool = True,
) -> LemonadeGymEnv:
    """
    Factory function for creating LemonadeGymEnv instances.
    
    Useful for creating vectorized environments:
    
        from gymnasium.vector import SyncVectorEnv
        envs = SyncVectorEnv([lambda i=i: make_env(seed=i) for i in range(8)])
    
    Args:
        seed: Random seed (seeds the RNG for episode seed generation)
        config: Game configuration
        render_mode: Render mode
        max_invalid_actions: Max consecutive invalid actions before forcing valid one
        reward_shaping: Enable dense reward shaping (default: True)
        reward_shaping_scale: Scale factor for shaped rewards (default: 0.1)
        use_mixed_actions: Use mixed discrete/continuous action space (default: False)
        randomize_seed: If True, each episode gets a new random seed for diverse
            weather patterns. If False, same seed every episode. (default: True)
        
    Returns:
        LemonadeGymEnv instance
    """
    return LemonadeGymEnv(
        seed=seed,
        config=config,
        render_mode=render_mode,
        max_invalid_actions=max_invalid_actions,
        reward_shaping=reward_shaping,
        reward_shaping_scale=reward_shaping_scale,
        use_mixed_actions=use_mixed_actions,
        randomize_seed=randomize_seed,
    )
