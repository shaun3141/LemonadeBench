# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
LLM-powered agent for LemonadeBench.

This agent uses LLM providers with tool_use for structured action outputs,
maintains conversation history for multi-turn reasoning, and supports
any provider implementing the LLMProvider interface.

Example:
    from lemonade_bench.agents import LLMAgent
    from lemonade_bench.agents.providers import AnthropicProvider
    
    provider = AnthropicProvider(model="claude-sonnet-4-20250514")
    agent = LLMAgent(provider)
    
    # Use with environment
    env = LemonadeEnvironment(seed=42)
    result = agent.run_episode(env)
    print(f"Total profit: ${result.total_profit / 100:.2f}")
"""

from typing import Any

from ..models import LemonadeAction, LemonadeObservation
from .base import LemonadeAgent, EpisodeResult, TurnResult, AgentCallback
from .providers.base import LLMProvider, ToolResponse, LEMONADE_ACTION_TOOL
from .tools import get_tool, Tool


# Default system prompt explaining the game and objective
DEFAULT_SYSTEM_PROMPT = """You are an AI agent running a lemonade stand business. Your goal is to maximize profit over a 14-day summer season.

## Game Mechanics

**Weather Effects on Demand:**
- HOT: Very high demand, customers willing to pay premium prices
- SUNNY: High demand, good day for sales
- CLOUDY: Moderate demand
- RAINY: Low demand, few customers venture out
- STORMY: Very low demand

**Pricing Strategy:**
- Higher prices = fewer customers but more profit per sale
- Lower prices = more customers but less profit per sale
- The "sweet spot" is usually around $0.50-$1.00 depending on weather

**Inventory:**
- Lemons: $0.25 each, expire after 3 days (buy in dozens for 10% discount, crates of 144 for 20% off)
- Sugar bags: $0.50 each, don't expire
- Cups: $0.05 each (packs of 10), don't expire
- Ice: $0.25/bag, melts overnight unless you have a cooler

**Recipe per cup:**
- 0.25 lemons (4 cups per lemon)
- 0.1 sugar bags (10 cups per bag)
- 0.2 ice bags (5 cups per bag) - optional but boosts hot day sales

**Upgrades:**
- Cooler ($2.50): Ice only melts 50% per day instead of 100%

**Locations:**
- Park: High traffic, standard prices (free)
- Downtown: Premium prices accepted, partial weather shelter ($10 permit)
- Mall: Indoor, weather-proof, but lower traffic ($15 permit)
- Pool: Amazing on hot days, terrible otherwise ($2.50 permit)

## Strategy Tips
1. Watch the weather forecast to plan inventory
2. Don't overbuy perishables (lemons, ice)
3. Higher prices on hot/sunny days, lower on bad weather
4. Build reputation by serving customers well (don't run out of supplies!)
5. Advertising helps on good weather days

Make decisions based on the current state and your previous results. Learn from what worked and what didn't."""


def format_observation(obs: LemonadeObservation, is_initial: bool = False) -> str:
    """Format observation into a prompt for the LLM."""
    
    # Handle error responses - show errors and ask for retry
    if obs.is_error_response:
        error_list = "\n".join(f"- {error}" for error in obs.action_errors)
        return f"""# ACTION ERROR - Please Retry

Your previous action was invalid and could not be executed:

{error_list}

## Current State
- Cash Available: ${obs.cash / 100:.2f}
- Day: {obs.day}
- Weather: {obs.weather.upper()} ({obs.temperature}°F)

## Current Inventory
- Lemons: {obs.lemons}
- Sugar Bags: {obs.sugar_bags:.1f}
- Cups: {obs.cups_available}
- Ice Bags: {obs.ice_bags}

Please submit a corrected action that you can afford with your current cash balance."""

    if is_initial:
        header = f"# Day {obs.day} - START OF GAME\n\n"
    else:
        header = f"# Day {obs.day} Results\n\n"

    # Current conditions
    conditions = f"""## Current Conditions
- Weather: {obs.weather.upper()} ({obs.temperature}°F)
- Tomorrow's Forecast: {obs.weather_forecast}
- Days Remaining: {obs.days_remaining}
"""

    # Financial state
    finances = f"""## Finances
- Cash: ${obs.cash / 100:.2f}
- Total Profit So Far: ${obs.total_profit / 100:.2f}
"""

    # Yesterday's results (if not initial)
    if not is_initial:
        results = f"""## Yesterday's Results
- Cups Sold: {obs.cups_sold}
- Revenue: ${obs.daily_revenue / 100:.2f}
- Costs: ${obs.daily_costs / 100:.2f}
- Daily Profit: ${obs.daily_profit / 100:.2f}
- Customers Served: {obs.customers_served}
- Customers Turned Away: {obs.customers_turned_away}
"""
        if obs.lemons_spoiled > 0:
            results += f"- Lemons Spoiled: {obs.lemons_spoiled}\n"
        if obs.ice_melted > 0:
            results += f"- Ice Melted: {obs.ice_melted}\n"
    else:
        results = ""

    # Current inventory
    inventory = f"""## Current Inventory
- Lemons: {obs.lemons} (expiring tomorrow: {obs.lemons_expiring_tomorrow})
- Sugar Bags: {obs.sugar_bags:.1f}
- Cups: {obs.cups_available}
- Ice Bags: {obs.ice_bags}
"""

    # Upgrades and location
    status = f"""## Stand Status
- Location: {obs.current_location}
- Owned Upgrades: {', '.join(obs.owned_upgrades) if obs.owned_upgrades else 'None'}
- Reputation: {obs.reputation:.2f}
- Customer Satisfaction: {obs.customer_satisfaction:.2f}
"""

    # Call to action
    action_prompt = "\n## Your Turn\nDecide your actions for today. Use the take_action tool to submit your decisions."

    return header + conditions + finances + results + inventory + status + action_prompt


def parse_tool_input(tool_input: dict[str, Any]) -> tuple[LemonadeAction, str]:
    """
    Parse tool input into a LemonadeAction.

    Args:
        tool_input: Dictionary of tool arguments

    Returns:
        Tuple of (LemonadeAction, reasoning_string)
    """
    reasoning = tool_input.get("reasoning", "")

    # Handle null values for optional fields
    buy_upgrade = tool_input.get("buy_upgrade")
    if buy_upgrade == "null" or buy_upgrade is None:
        buy_upgrade = None

    location = tool_input.get("location")
    if location == "null" or location is None:
        location = None

    action = LemonadeAction(
        price_per_cup=tool_input["price_per_cup"],
        buy_lemons=tool_input.get("buy_lemons", 0),
        buy_sugar=tool_input.get("buy_sugar", 0),
        buy_cups=tool_input.get("buy_cups", 0),
        buy_ice=tool_input.get("buy_ice", 0),
        advertising_spend=tool_input.get("advertising_spend", 0),
        buy_upgrade=buy_upgrade,
        location=location,
    )
    return action, reasoning


def format_action_result(obs: LemonadeObservation) -> str:
    """Format the result of an action for the tool result message."""
    result_content = f"Action executed. Sold {obs.cups_sold} cups, daily profit: ${obs.daily_profit/100:.2f}"
    if obs.customers_turned_away > 0:
        result_content += f", {obs.customers_turned_away} customers turned away"
    if obs.lemons_spoiled > 0:
        result_content += f", {obs.lemons_spoiled} lemons spoiled"
    if obs.ice_melted > 0:
        result_content += f", {obs.ice_melted} ice melted"
    return result_content


class LLMAgent(LemonadeAgent):
    """
    LLM-powered agent for the Lemonade Stand game.
    
    Uses a provider-agnostic interface to work with any LLM that supports
    tool/function calling. Maintains conversation history for multi-turn
    reasoning.
    
    Attributes:
        provider: LLMProvider instance for generating responses
        system_prompt: System prompt for the LLM
        messages: Conversation history
        total_input_tokens: Cumulative input tokens used
        total_output_tokens: Cumulative output tokens used
        total_cost: Estimated total cost in USD
        tools: List of optional tools available to the agent
    """
    
    # Maximum number of tool calls per turn (to prevent infinite loops)
    MAX_TOOL_CALLS_PER_TURN = 10
    
    def __init__(
        self,
        provider: LLMProvider,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        tools: list[str] | None = None,
    ):
        """
        Initialize the LLM agent.
        
        Args:
            provider: LLMProvider instance (e.g., AnthropicProvider)
            system_prompt: Custom system prompt (defaults to game instructions)
            tools: List of optional tool names to enable (e.g., ["calculator"])
        """
        self.provider = provider
        self.system_prompt = system_prompt
        self.messages: list[dict[str, Any]] = []
        
        # Token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        
        # Track if this is the first turn
        self._is_first_turn = True
        self._pending_tool_response: ToolResponse | None = None
        
        # Initialize optional tools
        self._tool_instances: dict[str, Tool] = {}
        if tools:
            for tool_name in tools:
                tool = get_tool(tool_name)
                if tool:
                    self._tool_instances[tool.name] = tool
        
        # Build tool definitions list (always include action tool)
        self._tool_definitions = [LEMONADE_ACTION_TOOL]
        for tool in self._tool_instances.values():
            self._tool_definitions.append(tool.definition)
        
        # Update system prompt if tools are available
        if self._tool_instances:
            tool_names = list(self._tool_instances.keys())
            tools_section = f"\n\n## Available Tools\nYou have access to the following optional tools: {', '.join(tool_names)}.\nUse them when helpful for calculations or analysis before making your final action."
            self.system_prompt = self.system_prompt + tools_section
    
    def reset(self) -> None:
        """Reset agent state for a new episode."""
        self.messages = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self._is_first_turn = True
        self._pending_tool_response = None
        self._tool_calls_this_turn = 0
    
    def decide(self, observation: LemonadeObservation) -> tuple[LemonadeAction, str]:
        """
        Decide what action to take given the current observation.
        
        Handles multi-step tool use where the agent may call optional tools
        (like calculator) before submitting the final action.
        
        Args:
            observation: Current game state
            
        Returns:
            Tuple of (action, reasoning)
        """
        # Reset tool call counter for this turn
        self._tool_calls_this_turn = 0
        
        # Build the message to send
        if self._is_first_turn:
            # First turn - just the observation
            obs_prompt = format_observation(observation, is_initial=True)
            self.messages.append({"role": "user", "content": obs_prompt})
            self._is_first_turn = False
        else:
            # Subsequent turns - need to add tool result from previous turn
            # This is handled in run_episode after env.step()
            pass
        
        # Loop to handle optional tool calls before the action
        while True:
            self._tool_calls_this_turn += 1
            
            # Safety check to prevent infinite loops
            if self._tool_calls_this_turn > self.MAX_TOOL_CALLS_PER_TURN:
                raise RuntimeError(
                    f"Agent exceeded maximum tool calls per turn ({self.MAX_TOOL_CALLS_PER_TURN})"
                )
            
            # Get response from LLM with all available tools
            response = self.provider.generate_with_tools(
                messages=self.messages,
                system_prompt=self.system_prompt,
                tools=self._tool_definitions,
                required_tool=None,  # Let model choose
            )
            
            # Track tokens
            if response.usage:
                self.total_input_tokens += response.usage.input_tokens
                self.total_output_tokens += response.usage.output_tokens
                self.total_cost += self.provider.estimate_cost(response.usage)
            
            # Add assistant message to history
            self.messages.append(self.provider.format_assistant_message(response))
            
            # Check if this is the final action or an optional tool call
            if response.tool_name == "take_action":
                # This is the final action
                self._pending_tool_response = response
                action, reasoning = parse_tool_input(response.tool_input)
                return action, reasoning
            
            # Handle optional tool call
            tool = self._tool_instances.get(response.tool_name)
            if tool is None:
                raise ValueError(f"Unknown tool called: {response.tool_name}")
            
            # Execute the tool
            result = tool.execute(**response.tool_input)
            
            # Format result for the model
            if result.success:
                result_text = result.result
            else:
                result_text = f"Error: {result.error}"
            
            # Add tool result to conversation
            tool_result_msg = self.provider.format_tool_result(
                tool_use_id=response.tool_use_id,
                result=result_text,
            )
            self.messages.append(tool_result_msg)
    
    def _add_tool_result(self, result_text: str, next_observation: LemonadeObservation | None = None) -> None:
        """
        Add a tool result to the conversation history.
        
        Called after env.step() to include the result in the conversation.
        
        Args:
            result_text: Result of the action execution
            next_observation: Next observation (if game continues)
        """
        if self._pending_tool_response is None:
            return
        
        # Build user message with tool result
        tool_result_msg = self.provider.format_tool_result(
            tool_use_id=self._pending_tool_response.tool_use_id,
            result=result_text,
        )
        
        # If game continues, append next observation
        if next_observation is not None:
            next_obs_prompt = format_observation(next_observation, is_initial=False)
            # Append text to the tool result message content
            if isinstance(tool_result_msg["content"], list):
                tool_result_msg["content"].append({
                    "type": "text",
                    "text": next_obs_prompt,
                })
            else:
                # If content is a string, convert to list format
                tool_result_msg["content"] = [
                    {"type": "text", "text": tool_result_msg["content"]},
                    {"type": "text", "text": next_obs_prompt},
                ]
        
        self.messages.append(tool_result_msg)
        self._pending_tool_response = None
    
    def run_episode(
        self,
        env,
        callbacks: list[AgentCallback] | None = None,
    ) -> EpisodeResult:
        """
        Run a complete episode using this agent.
        
        Overrides base class to handle LLM-specific conversation flow.
        Handles action validation errors by allowing retries and feeding
        error feedback back to the LLM.
        
        Args:
            env: LemonadeEnvironment instance
            callbacks: Optional list of callbacks for monitoring
            
        Returns:
            EpisodeResult with complete episode data including token usage
        """
        callbacks = callbacks or []
        
        # Reset agent state
        self.reset()
        
        # Reset environment and get initial observation
        obs = env.reset()
        
        # Notify callbacks
        for cb in callbacks:
            cb.on_episode_start(obs)
        
        turns: list[TurnResult] = []
        error_turns: list[TurnResult] = []
        total_cups_sold = 0
        total_errors = 0
        
        while not obs.done:
            # Notify callbacks of turn start
            for cb in callbacks:
                cb.on_turn_start(obs.day, obs)
            
            # Retry loop for this day
            retries = 0
            pre_obs = obs
            
            while retries <= self.MAX_RETRIES_PER_DAY:
                # Get action from agent
                action, reasoning = self.decide(obs)
                
                # Execute action
                result_obs = env.step(action)
                
                # Check if action was valid
                if not result_obs.is_error_response:
                    # Valid action - add result to conversation and proceed
                    obs = result_obs
                    result_text = format_action_result(obs)
                    next_obs = obs if not obs.done else None
                    self._add_tool_result(result_text, next_obs)
                    break
                
                # Invalid action - log error
                error_turn = TurnResult(
                    day=pre_obs.day,
                    observation=pre_obs,
                    action=action,
                    reasoning=reasoning,
                    cups_sold=0,
                    daily_profit=0,
                    daily_revenue=0,
                    daily_costs=0,
                    customers_served=0,
                    customers_turned_away=0,
                    is_error=True,
                    error_messages=result_obs.action_errors,
                    retry_count=retries,
                )
                error_turns.append(error_turn)
                total_errors += 1
                
                # Feed error back to LLM as tool result with error observation
                error_text = "ACTION ERROR: " + "; ".join(result_obs.action_errors)
                error_obs_prompt = format_observation(result_obs, is_initial=False)
                self._add_tool_result(error_text + "\n\n" + error_obs_prompt, None)
                
                # Update obs so next decide() call sees the error state
                obs = result_obs
                retries += 1
            
            # If max retries exceeded, force a minimal safe action
            if retries > self.MAX_RETRIES_PER_DAY:
                safe_action = LemonadeAction(price_per_cup=50)
                obs = env.step(safe_action)
                action = safe_action
                reasoning = "[FALLBACK] Max retries exceeded, using safe default action"
                
                # Add fallback result to conversation
                result_text = format_action_result(obs)
                next_obs = obs if not obs.done else None
                self._add_tool_result(result_text, next_obs)
            
            total_cups_sold += obs.cups_sold
            
            # Create turn result for successful action
            turn = TurnResult(
                day=pre_obs.day,
                observation=pre_obs,
                action=action,
                reasoning=reasoning,
                cups_sold=obs.cups_sold,
                daily_profit=obs.daily_profit,
                daily_revenue=obs.daily_revenue,
                daily_costs=obs.daily_costs,
                customers_served=obs.customers_served,
                customers_turned_away=obs.customers_turned_away,
                retry_count=retries,
            )
            turns.append(turn)
            
            # Notify callbacks of turn end
            for cb in callbacks:
                cb.on_turn_end(turn)
        
        # Build episode result with LLM-specific data
        result = EpisodeResult(
            total_profit=obs.total_profit,
            total_cups_sold=total_cups_sold,
            final_cash=obs.cash,
            final_reputation=obs.reputation,
            turn_count=len(turns),
            turns=turns,
            model_name=self.provider.model_name,
            provider=self.provider.provider_name,
            total_input_tokens=self.total_input_tokens,
            total_output_tokens=self.total_output_tokens,
            estimated_cost_usd=self.total_cost,
            conversation_history=self.messages.copy(),
            error_count=total_errors,
            error_turns=error_turns,
        )
        
        # Notify callbacks of episode end
        for cb in callbacks:
            cb.on_episode_end(result)
        
        return result
