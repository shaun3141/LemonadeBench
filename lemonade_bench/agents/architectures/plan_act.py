# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
Plan-Act agent architecture - generates a multi-day plan before each action.

This architecture adds an explicit planning phase before the action phase,
testing whether generating a strategic plan improves long-horizon reasoning.
"""

from typing import Any

from ...models import LemonadeAction, LemonadeObservation
from ..base import EpisodeResult, TurnResult, AgentCallback
from ..prompts import PLANNING_PROMPT, GoalFramingType
from .react import ReactAgent, format_observation, format_action_result, parse_tool_input


class PlanActAgent(ReactAgent):
    """
    Plan-Act agent - generates a plan before each action.
    
    Before deciding on an action, the agent is prompted to generate an explicit
    multi-day plan considering weather forecasts, inventory levels, cash position,
    and reputation. This tests whether explicit planning improves long-horizon
    reasoning.
    
    Architecture:
        Observe -> **Plan** -> Decide -> Act
    """
    
    def __init__(
        self,
        provider,
        goal_framing: GoalFramingType = "baseline",
        tools: list[str] | None = None,
        math_prompt: bool = False,
    ):
        """Initialize the Plan-Act agent."""
        super().__init__(
            provider=provider,
            goal_framing=goal_framing,
            tools=tools,
            math_prompt=math_prompt,
        )
        self._current_plan: str | None = None
    
    def reset(self) -> None:
        """Reset agent state for a new episode."""
        super().reset()
        self._current_plan = None
    
    def _generate_plan(self, observation: LemonadeObservation) -> str:
        """
        Generate a strategic plan for the next few days.
        
        Args:
            observation: Current game state
            
        Returns:
            The generated plan text
        """
        # Build planning prompt
        obs_text = format_observation(observation, is_initial=self._is_first_turn)
        
        plan_prompt = f"""{obs_text}

{PLANNING_PROMPT}"""
        
        # Add planning request to conversation
        self.messages.append({"role": "user", "content": plan_prompt})
        
        # Generate plan (no tools needed for planning)
        response = self.provider.generate_with_tools(
            messages=self.messages,
            system_prompt=self.system_prompt,
            tools=[],  # No tools for planning phase
            required_tool=None,
        )
        
        # Track tokens
        if response.usage:
            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens
            self.total_cost += self.provider.estimate_cost(response.usage)
        
        # Extract plan text from response
        plan_text = response.text_content or ""
        
        # Add plan to conversation
        self.messages.append({"role": "assistant", "content": plan_text})
        
        return plan_text
    
    def decide(self, observation: LemonadeObservation) -> tuple[LemonadeAction, str]:
        """
        Decide what action to take, with explicit planning phase.
        
        Args:
            observation: Current game state
            
        Returns:
            Tuple of (action, reasoning)
        """
        self._tool_calls_this_turn = 0
        
        # Phase 1: Generate plan
        self._current_plan = self._generate_plan(observation)
        
        # Mark that we've processed the first turn observation (in the plan prompt)
        if self._is_first_turn:
            self._is_first_turn = False
        
        # Phase 2: Execute plan - prompt for action
        action_prompt = f"""Based on your plan above, now take today's action.

Use the take_action tool to submit your decisions for today."""
        
        self.messages.append({"role": "user", "content": action_prompt})
        
        # Generate action with tools available
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
                # Include plan reference in reasoning
                if reasoning and self._current_plan:
                    reasoning = f"[Following plan] {reasoning}"
                return action, reasoning
            
            # Handle optional tool calls
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
        
        # Don't add next observation here - it will be added in the planning phase
        self.messages.append(tool_result_msg)
        self._pending_tool_response = None
    
    def run_episode(
        self,
        env,
        callbacks: list[AgentCallback] | None = None,
    ) -> EpisodeResult:
        """Run a complete episode using the Plan-Act architecture."""
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
            self._add_tool_result(result_text, None)  # Don't add next obs here
            
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
                "architecture": "plan_act",
                "goal_framing": self.goal_framing,
                "math_prompt": self.math_prompt,
                "tools": list(self._tool_instances.keys()),
            },
        )
        
        for cb in callbacks:
            cb.on_episode_end(result)
        
        return result

