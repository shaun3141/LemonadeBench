# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
React agent architecture - the baseline observe -> decide -> act loop.

This is the standard ReAct-style agent where the model receives an observation
and immediately outputs an action. No explicit planning or reflection phases.
"""

from typing import Any

from ...models import LemonadeAction, LemonadeObservation, quantity_to_tier_count
from ..base import LemonadeAgent, EpisodeResult, TurnResult, AgentCallback
from ..providers.base import LLMProvider, ToolResponse, LEMONADE_ACTION_TOOL
from ..tools import get_tool, Tool
from ..prompts import build_system_prompt, GoalFramingType


def format_observation(obs: LemonadeObservation, is_initial: bool = False) -> str:
    """
    Format observation into a clear markdown prompt for the LLM.
    
    Uses structured markdown tables for efficient, scannable information.
    Handles error responses with clear feedback for the model.
    """
    # Handle error responses - show errors and ask for retry
    if obs.is_error_response:
        error_list = "\n".join(f"- {error}" for error in obs.action_errors)
        return f"""# ⚠️ ACTION ERROR - Please Retry

Your previous action was invalid:

{error_list}

## Current State
| Cash | Lemons | Sugar | Cups | Ice |
|------|--------|-------|------|-----|
| ${obs.cash / 100:.2f} | {obs.lemons:.0f} | {obs.sugar_bags:.1f} | {obs.cups_available} | {obs.ice_bags:.0f} |

**Please submit a corrected action within your budget of ${obs.cash / 100:.2f}.**
"""
    
    # Header with weather info
    forecast = obs.weather_forecast.lower() if obs.weather_forecast else "unknown"
    header = f"# Day {obs.day} | {obs.weather.upper()} {obs.temperature}°F → {forecast} tomorrow\n\n"
    
    # Yesterday's results (if not initial)
    if not is_initial:
        results = "## Yesterday's Results\n"
        if obs.cups_sold > 0:
            price_per_cup = obs.daily_revenue / obs.cups_sold / 100
            results += f"- Sold: {obs.cups_sold} cups @ ${price_per_cup:.2f} = ${obs.daily_revenue / 100:.2f} revenue\n"
        else:
            results += f"- Sold: 0 cups\n"
        results += f"- Costs: ${obs.daily_costs / 100:.2f}\n"
        results += f"- **Profit: ${obs.daily_profit / 100:+.2f}**\n"
        results += f"- Customers served: {obs.customers_served}"
        if obs.customers_turned_away > 0:
            results += f" | Turned away: {obs.customers_turned_away}"
        results += "\n"
        
        # Spoilage/waste
        if obs.lemons_spoiled > 0:
            results += f"- Lemons spoiled overnight: {obs.lemons_spoiled:.1f}\n"
        if obs.ice_melted > 0:
            results += f"- Ice melted overnight: {obs.ice_melted:.1f} bags\n"
        results += "\n"
    else:
        results = ""
    
    # Current state table
    state = "## Current State\n"
    state += "| Cash | Total Profit | Reputation |\n"
    state += "|------|--------------|------------|\n"
    state += f"| ${obs.cash / 100:.2f} | ${obs.total_profit / 100:.2f} | {obs.reputation:.2f} |\n\n"
    
    # Inventory table
    inventory = "## Inventory\n"
    inventory += "| Item | Quantity | Expiring Tomorrow |\n"
    inventory += "|------|----------|-------------------|\n"
    
    # Lemons with expiry warning
    lemon_expiry = f"{obs.lemons_expiring_tomorrow} (will spoil!)" if obs.lemons_expiring_tomorrow > 0 else "-"
    inventory += f"| Lemons | {obs.lemons:.0f} | {lemon_expiry} |\n"
    inventory += f"| Sugar | {obs.sugar_bags:.1f} bags | - |\n"
    inventory += f"| Cups | {obs.cups_available} | - |\n"
    
    # Ice with expiry info
    ice_expiry = f"{obs.ice_expiring_tomorrow:.0f} bags" if hasattr(obs, 'ice_expiring_tomorrow') and obs.ice_expiring_tomorrow > 0 else "-"
    inventory += f"| Ice | {obs.ice_bags:.0f} bags | {ice_expiry} |\n\n"
    
    # Status
    upgrades = ', '.join(obs.owned_upgrades) if obs.owned_upgrades else 'None'
    status = "## Status\n"
    status += f"- Location: {obs.current_location}\n"
    status += f"- Upgrades: {upgrades}\n"
    status += f"- Days remaining: {obs.days_remaining}\n"
    
    return header + results + state + inventory + status


def parse_tool_input(tool_input: dict[str, Any]) -> tuple[LemonadeAction, str]:
    """Parse tool input into a LemonadeAction."""
    reasoning = tool_input.get("reasoning", "")

    buy_upgrade = tool_input.get("buy_upgrade")
    if buy_upgrade == "null" or buy_upgrade is None:
        buy_upgrade = None

    location = tool_input.get("location")
    if location == "null" or location is None:
        location = None

    # Convert quantities to optimal tier+count (auto-selects best bulk tier)
    lt, lc = quantity_to_tier_count("lemons", tool_input.get("buy_lemons", 0))
    st, sc = quantity_to_tier_count("sugar", tool_input.get("buy_sugar", 0))
    ct, cc = quantity_to_tier_count("cups", tool_input.get("buy_cups", 0))
    it, ic = quantity_to_tier_count("ice", tool_input.get("buy_ice", 0))
    
    action = LemonadeAction(
        price_per_cup=tool_input["price_per_cup"],
        lemons_tier=lt, lemons_count=lc,
        sugar_tier=st, sugar_count=sc,
        cups_tier=ct, cups_count=cc,
        ice_tier=it, ice_count=ic,
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


class ReactAgent(LemonadeAgent):
    """
    React agent - basic observe -> decide -> act loop.
    
    This is the baseline architecture where the model receives an observation
    and immediately outputs an action without explicit planning or reflection.
    
    Attributes:
        provider: LLM provider instance
        system_prompt: Complete system prompt with goal framing
        goal_framing: Goal framing condition applied
        messages: Conversation history
    """
    
    MAX_TOOL_CALLS_PER_TURN = 10
    
    def __init__(
        self,
        provider: LLMProvider,
        goal_framing: GoalFramingType = "baseline",
        tools: list[str] | None = None,
        math_prompt: bool = False,
    ):
        """
        Initialize the React agent.
        
        Args:
            provider: LLM provider instance
            goal_framing: Goal framing condition to apply
            tools: List of optional tool names to enable
            math_prompt: Whether to enable math encouragement prompt
        """
        self.provider = provider
        self.goal_framing = goal_framing
        self.math_prompt = math_prompt
        self.messages: list[dict[str, Any]] = []
        
        # Token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        
        # State tracking
        self._is_first_turn = True
        self._pending_tool_response: ToolResponse | None = None
        
        # Initialize optional tools
        self._tool_instances: dict[str, Tool] = {}
        tool_names: list[str] = []
        if tools:
            for tool_name in tools:
                tool = get_tool(tool_name)
                if tool:
                    self._tool_instances[tool.name] = tool
                    tool_names.append(tool.name)
        
        # Build tool definitions
        self._tool_definitions = [LEMONADE_ACTION_TOOL]
        for tool in self._tool_instances.values():
            self._tool_definitions.append(tool.definition)
        
        # Build system prompt using STATIC_HEADER (no custom base_prompt)
        self.system_prompt = build_system_prompt(
            goal_framing=goal_framing,
            math_prompt=math_prompt,
            tools_available=tool_names if tool_names else None,
        )
    
    def reset(self) -> None:
        """Reset agent state for a new episode."""
        self.messages = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self._is_first_turn = True
        self._pending_tool_response = None
        self._tool_calls_this_turn = 0
    
    def _generate_response(self, tools: list[dict] | None = None) -> ToolResponse:
        """
        Generate a response from the LLM.
        
        Args:
            tools: Tool definitions to provide (defaults to all available)
            
        Returns:
            ToolResponse from the provider
        """
        if tools is None:
            tools = self._tool_definitions
        
        response = self.provider.generate_with_tools(
            messages=self.messages,
            system_prompt=self.system_prompt,
            tools=tools,
            required_tool=None,
        )
        
        if response.usage:
            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens
            self.total_cost += self.provider.estimate_cost(response.usage)
        
        return response
    
    def _handle_optional_tool_call(self, response: ToolResponse) -> str:
        """
        Handle a call to an optional tool (not take_action).
        
        Args:
            response: The tool response to handle
            
        Returns:
            Result text from the tool execution
        """
        tool = self._tool_instances.get(response.tool_name)
        if tool is None:
            raise ValueError(f"Unknown tool called: {response.tool_name}")
        
        result = tool.execute(**response.tool_input)
        
        if result.success:
            return result.result
        else:
            return f"Error: {result.error}"
    
    def decide(self, observation: LemonadeObservation) -> tuple[LemonadeAction, str]:
        """
        Decide what action to take given the current observation.
        
        Args:
            observation: Current game state
            
        Returns:
            Tuple of (action, reasoning)
        """
        self._tool_calls_this_turn = 0
        
        if self._is_first_turn:
            obs_prompt = format_observation(observation, is_initial=True)
            self.messages.append({"role": "user", "content": obs_prompt})
            self._is_first_turn = False
        
        while True:
            self._tool_calls_this_turn += 1
            
            if self._tool_calls_this_turn > self.MAX_TOOL_CALLS_PER_TURN:
                raise RuntimeError(
                    f"Agent exceeded maximum tool calls per turn ({self.MAX_TOOL_CALLS_PER_TURN})"
                )
            
            response = self._generate_response()
            self.messages.append(self.provider.format_assistant_message(response))
            
            if response.tool_name == "take_action":
                self._pending_tool_response = response
                action, reasoning = parse_tool_input(response.tool_input)
                return action, reasoning
            
            result_text = self._handle_optional_tool_call(response)
            tool_result_msg = self.provider.format_tool_result(
                tool_use_id=response.tool_use_id,
                result=result_text,
            )
            self.messages.append(tool_result_msg)
    
    def _add_tool_result(self, result_text: str, next_observation: LemonadeObservation | None = None) -> None:
        """Add a tool result to the conversation history."""
        if self._pending_tool_response is None:
            return
        
        tool_result_msg = self.provider.format_tool_result(
            tool_use_id=self._pending_tool_response.tool_use_id,
            result=result_text,
        )
        
        if next_observation is not None:
            next_obs_prompt = format_observation(next_observation, is_initial=False)
            if isinstance(tool_result_msg["content"], list):
                tool_result_msg["content"].append({
                    "type": "text",
                    "text": next_obs_prompt,
                })
            else:
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
        """Run a complete episode using this agent."""
        callbacks = callbacks or []
        
        self.reset()
        obs = env.reset()
        
        for cb in callbacks:
            cb.on_episode_start(obs)
        
        turns: list[TurnResult] = []
        total_cups_sold = 0
        
        while not obs.done:
            for cb in callbacks:
                cb.on_turn_start(obs.day, obs)
            
            action, reasoning = self.decide(obs)
            pre_obs = obs
            obs = env.step(action)
            
            total_cups_sold += obs.cups_sold
            
            result_text = format_action_result(obs)
            next_obs = obs if not obs.done else None
            self._add_tool_result(result_text, next_obs)
            
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
            )
            turns.append(turn)
            
            for cb in callbacks:
                cb.on_turn_end(turn)
        
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
            config={
                "architecture": "react",
                "goal_framing": self.goal_framing,
                "math_prompt": self.math_prompt,
                "tools": list(self._tool_instances.keys()),
            },
        )
        
        for cb in callbacks:
            cb.on_episode_end(result)
        
        return result
