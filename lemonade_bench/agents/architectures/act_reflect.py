# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
Act-Reflect agent architecture - reflects on results after each action.

This architecture adds an explicit reflection phase after each action,
inspired by Reflexion. Tests whether retrospective analysis improves
learning within an episode.
"""

from typing import Any

from ...models import LemonadeAction, LemonadeObservation
from ..base import EpisodeResult, TurnResult, AgentCallback
from ..prompts import REFLECTION_PROMPT, GoalFramingType
from .react import ReactAgent, format_observation, format_action_result, parse_tool_input


class ActReflectAgent(ReactAgent):
    """
    Act-Reflect agent - reflects on results after each action.
    
    After each action, the agent receives results and is prompted to reflect
    on what worked, what didn't, and what to do differently. This tests whether
    retrospective analysis improves learning within an episode.
    
    Architecture:
        Observe -> Decide -> Act -> **Reflect**
    """
    
    def __init__(
        self,
        provider,
        goal_framing: GoalFramingType = "baseline",
        tools: list[str] | None = None,
        math_prompt: bool = False,
    ):
        """Initialize the Act-Reflect agent."""
        super().__init__(
            provider=provider,
            goal_framing=goal_framing,
            tools=tools,
            math_prompt=math_prompt,
        )
        self._last_reflection: str | None = None
        self._pending_reflection: bool = False
    
    def reset(self) -> None:
        """Reset agent state for a new episode."""
        super().reset()
        self._last_reflection = None
        self._pending_reflection = False
    
    def _generate_reflection(self, result_text: str, observation: LemonadeObservation) -> str:
        """
        Generate a reflection on the previous day's results.
        
        Args:
            result_text: Summary of yesterday's results
            observation: Current game state after action
            
        Returns:
            The generated reflection text
        """
        # Build reflection prompt with context
        obs_summary = f"""## Yesterday's Outcome
{result_text}

## Current State After Yesterday
- Cash: ${observation.cash / 100:.2f}
- Total Profit: ${observation.total_profit / 100:.2f}
- Reputation: {observation.reputation:.2f}

{REFLECTION_PROMPT}"""
        
        self.messages.append({"role": "user", "content": obs_summary})
        
        # Generate reflection (no tools needed)
        response = self.provider.generate_with_tools(
            messages=self.messages,
            system_prompt=self.system_prompt,
            tools=[],  # No tools for reflection
            required_tool=None,
        )
        
        # Track tokens
        if response.usage:
            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens
            self.total_cost += self.provider.estimate_cost(response.usage)
        
        reflection_text = response.text_content or ""
        
        # Add reflection to conversation
        self.messages.append({"role": "assistant", "content": reflection_text})
        
        return reflection_text
    
    def decide(self, observation: LemonadeObservation) -> tuple[LemonadeAction, str]:
        """
        Decide what action to take.
        
        Args:
            observation: Current game state
            
        Returns:
            Tuple of (action, reasoning)
        """
        self._tool_calls_this_turn = 0
        
        if self._is_first_turn:
            # First turn - just observe and act, no reflection yet
            obs_prompt = format_observation(observation, is_initial=True)
            self.messages.append({"role": "user", "content": obs_prompt})
            self._is_first_turn = False
        else:
            # Subsequent turns - observation was added by _add_tool_result
            # Now prompt for action based on reflection insights
            action_prompt = """Based on your reflection above and today's conditions, decide your actions.

Use the take_action tool to submit your decisions for today."""
            self.messages.append({"role": "user", "content": action_prompt})
        
        # Generate action
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
            
            # Handle optional tool calls
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
        Add tool result and generate reflection.
        
        Args:
            result_text: Result of action execution
            next_observation: Next observation (if game continues)
            game_done: Whether the game is over
        """
        if self._pending_tool_response is None:
            return
        
        # First add the tool result
        tool_result_msg = self.provider.format_tool_result(
            tool_use_id=self._pending_tool_response.tool_use_id,
            result=result_text,
        )
        self.messages.append(tool_result_msg)
        self._pending_tool_response = None
        
        # Generate reflection if game continues
        if not game_done and next_observation is not None:
            self._last_reflection = self._generate_reflection(result_text, next_observation)
            
            # Add next observation for the next decide() call
            obs_prompt = format_observation(next_observation, is_initial=False)
            self.messages.append({"role": "user", "content": obs_prompt})
    
    def run_episode(
        self,
        env,
        callbacks: list[AgentCallback] | None = None,
    ) -> EpisodeResult:
        """Run a complete episode using the Act-Reflect architecture."""
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
            
            # Add result and generate reflection (if not done)
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
                "architecture": "act_reflect",
                "goal_framing": self.goal_framing,
                "math_prompt": self.math_prompt,
                "tools": list(self._tool_instances.keys()),
            },
        )
        
        for cb in callbacks:
            cb.on_episode_end(result)
        
        return result

