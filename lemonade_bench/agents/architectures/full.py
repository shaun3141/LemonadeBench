# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
Full agent architecture - combines planning and reflection phases.

This architecture includes both explicit planning before actions and
reflection after results. Tests whether the benefits of planning and
reflection compound or whether the additional compute is wasteful.
"""

from typing import Any

from ...models import LemonadeAction, LemonadeObservation
from ..base import EpisodeResult, TurnResult, AgentCallback
from ..prompts import PLANNING_PROMPT, REFLECTION_PROMPT, PLAN_UPDATE_PROMPT, GoalFramingType
from .react import ReactAgent, format_observation, format_action_result, parse_tool_input


class FullAgent(ReactAgent):
    """
    Full agent - combines planning and reflection in each turn.
    
    Each turn follows the pattern:
    1. Observe current state
    2. Reflect on previous results (if not first turn)
    3. Generate/update multi-day plan
    4. Execute action based on plan
    
    This tests whether planning and reflection benefits compound,
    requiring 3x the API calls of the basic React agent.
    
    Architecture:
        Observe -> **Reflect** (if not first) -> **Plan** -> Decide -> Act
    """
    
    def __init__(
        self,
        provider,
        goal_framing: GoalFramingType = "baseline",
        tools: list[str] | None = None,
        math_prompt: bool = False,
    ):
        """Initialize the Full agent."""
        super().__init__(
            provider=provider,
            goal_framing=goal_framing,
            tools=tools,
            math_prompt=math_prompt,
        )
        self._current_plan: str | None = None
        self._last_reflection: str | None = None
    
    def reset(self) -> None:
        """Reset agent state for a new episode."""
        super().reset()
        self._current_plan = None
        self._last_reflection = None
    
    def _generate_reflection(self, result_text: str, observation: LemonadeObservation) -> str:
        """
        Generate a reflection on the previous day's results.
        
        Args:
            result_text: Summary of yesterday's results  
            observation: Current game state
            
        Returns:
            The generated reflection text
        """
        # Include plan context in reflection
        plan_context = ""
        if self._current_plan:
            plan_context = f"\n\n## Your Previous Plan\n{self._current_plan}\n"
        
        reflection_prompt = f"""## Yesterday's Outcome
{result_text}
{plan_context}
{REFLECTION_PROMPT}

Consider whether your plan is still valid or needs adjustment."""
        
        self.messages.append({"role": "user", "content": reflection_prompt})
        
        response = self.provider.generate_with_tools(
            messages=self.messages,
            system_prompt=self.system_prompt,
            tools=[],
            required_tool=None,
        )
        
        if response.usage:
            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens
            self.total_cost += self.provider.estimate_cost(response.usage)
        
        reflection_text = response.text_content or ""
        self.messages.append({"role": "assistant", "content": reflection_text})
        
        return reflection_text
    
    def _generate_plan(self, observation: LemonadeObservation, is_update: bool = False) -> str:
        """
        Generate or update a strategic plan.
        
        Args:
            observation: Current game state
            is_update: Whether this is updating an existing plan
            
        Returns:
            The generated plan text
        """
        obs_text = format_observation(observation, is_initial=self._is_first_turn)
        
        if is_update and self._current_plan:
            # Use update prompt that references previous plan
            plan_prompt = f"""{obs_text}

{PLAN_UPDATE_PROMPT.format(previous_plan=self._current_plan)}"""
        else:
            # Initial plan
            plan_prompt = f"""{obs_text}

{PLANNING_PROMPT}"""
        
        self.messages.append({"role": "user", "content": plan_prompt})
        
        response = self.provider.generate_with_tools(
            messages=self.messages,
            system_prompt=self.system_prompt,
            tools=[],
            required_tool=None,
        )
        
        if response.usage:
            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens
            self.total_cost += self.provider.estimate_cost(response.usage)
        
        plan_text = response.text_content or ""
        self.messages.append({"role": "assistant", "content": plan_text})
        
        return plan_text
    
    def decide(self, observation: LemonadeObservation) -> tuple[LemonadeAction, str]:
        """
        Decide what action to take with full planning and reflection.
        
        Args:
            observation: Current game state
            
        Returns:
            Tuple of (action, reasoning)
        """
        self._tool_calls_this_turn = 0
        
        # Phase 1: Planning (will include observation)
        is_update = self._current_plan is not None
        self._current_plan = self._generate_plan(observation, is_update=is_update)
        
        if self._is_first_turn:
            self._is_first_turn = False
        
        # Phase 2: Execute action based on plan
        action_prompt = """Based on your plan, now take today's action.

Use the take_action tool to submit your decisions."""
        
        self.messages.append({"role": "user", "content": action_prompt})
        
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
                reasoning = f"[Plan: {self._current_plan[:100]}...] {reasoning}" if self._current_plan else reasoning
                return action, reasoning
            
            result_text = self._handle_optional_tool_call(response)
            tool_result_msg = self.provider.format_tool_result(
                tool_use_id=response.tool_use_id,
                result=result_text,
            )
            self.messages.append(tool_result_msg)
    
    def _add_tool_result_with_reflection(
        self,
        result_text: str,
        next_observation: LemonadeObservation | None,
        game_done: bool,
    ) -> None:
        """
        Add tool result and generate reflection if game continues.
        
        Args:
            result_text: Result of action execution
            next_observation: Next observation (if game continues)
            game_done: Whether the game is over
        """
        if self._pending_tool_response is None:
            return
        
        # Add tool result
        tool_result_msg = self.provider.format_tool_result(
            tool_use_id=self._pending_tool_response.tool_use_id,
            result=result_text,
        )
        self.messages.append(tool_result_msg)
        self._pending_tool_response = None
        
        # Generate reflection if game continues (before next planning phase)
        if not game_done and next_observation is not None:
            self._last_reflection = self._generate_reflection(result_text, next_observation)
    
    def run_episode(
        self,
        env,
        callbacks: list[AgentCallback] | None = None,
    ) -> EpisodeResult:
        """Run a complete episode using the Full architecture."""
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
            
            # Add result and reflect
            result_text = format_action_result(obs)
            self._add_tool_result_with_reflection(
                result_text,
                obs if not obs.done else None,
                obs.done,
            )
            
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
                "architecture": "full",
                "goal_framing": self.goal_framing,
                "math_prompt": self.math_prompt,
                "tools": list(self._tool_instances.keys()),
            },
        )
        
        for cb in callbacks:
            cb.on_episode_end(result)
        
        return result

