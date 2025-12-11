# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
Base LLM provider interface.

Defines the abstract interface that all LLM providers must implement,
enabling the LLM agent to work with different APIs interchangeably.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class TokenUsage:
    """Token usage statistics from an LLM call."""
    input_tokens: int
    output_tokens: int
    
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class ToolResponse:
    """Response from an LLM that includes a tool call."""
    tool_name: str
    tool_input: dict[str, Any]
    tool_use_id: str
    raw_response: Any  # Provider-specific response object
    usage: TokenUsage | None = None
    text_content: str | None = None  # Text content from the response (for planning/reflection)


# Standard tool definition for the lemonade action
# LLM specifies quantities; system auto-selects optimal bulk tier
LEMONADE_ACTION_TOOL = {
    "name": "take_action",
    "description": "Submit daily decisions: set price, buy supplies, optionally advertise/upgrade/relocate.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of your strategy (1-2 sentences)"
            },
            "price_per_cup": {
                "type": "integer",
                "description": "Price in cents (e.g., 75 = $0.75)"
            },
            "buy_lemons": {
                "type": "integer",
                "description": "Number of lemons to buy (bulk discounts auto-applied)",
                "default": 0
            },
            "buy_sugar": {
                "type": "integer",
                "description": "Number of sugar bags to buy",
                "default": 0
            },
            "buy_cups": {
                "type": "integer",
                "description": "Number of cups to buy",
                "default": 0
            },
            "buy_ice": {
                "type": "integer",
                "description": "Number of ice bags to buy",
                "default": 0
            },
            "advertising_spend": {
                "type": "integer",
                "description": "Amount in cents",
                "default": 0
            },
            "buy_upgrade": {
                "type": "string",
                "enum": ["cooler", None],
                "default": None
            },
            "location": {
                "type": "string",
                "enum": ["park", "downtown", "mall", "pool", None],
                "default": None
            }
        },
        "required": ["reasoning", "price_per_cup"]
    }
}


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    Each provider implementation handles the specifics of its API while
    presenting a unified interface for the LLM agent.
    """
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name/identifier."""
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'anthropic', 'openai')."""
        pass
    
    @abstractmethod
    def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tools: list[dict[str, Any]],
        required_tool: str | None = None,
    ) -> ToolResponse:
        """
        Generate a response with tool calls.
        
        Args:
            messages: Conversation history
            system_prompt: System prompt for the model
            tools: List of tool definitions
            required_tool: If set, force the model to use this specific tool
            
        Returns:
            ToolResponse containing the tool call details
        """
        pass
    
    def generate_with_tool(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tool: dict[str, Any] = LEMONADE_ACTION_TOOL,
    ) -> ToolResponse:
        """
        Generate a response with a single required tool call.
        
        Convenience method for backwards compatibility.
        
        Args:
            messages: Conversation history
            system_prompt: System prompt for the model
            tool: Tool definition (defaults to LEMONADE_ACTION_TOOL)
            
        Returns:
            ToolResponse containing the tool call details
        """
        return self.generate_with_tools(
            messages=messages,
            system_prompt=system_prompt,
            tools=[tool],
            required_tool=tool["name"],
        )
    
    @abstractmethod
    def format_tool_result(
        self,
        tool_use_id: str,
        result: str,
    ) -> dict[str, Any]:
        """
        Format a tool result for inclusion in the conversation.
        
        Args:
            tool_use_id: ID of the tool use to respond to
            result: String result of the tool execution
            
        Returns:
            Message dict in provider-specific format
        """
        pass
    
    @abstractmethod
    def format_assistant_message(self, response: ToolResponse) -> dict[str, Any]:
        """
        Format the assistant's tool call response as a message.
        
        Args:
            response: The ToolResponse from generate_with_tool
            
        Returns:
            Message dict to add to conversation history
        """
        pass
    
    def estimate_cost(self, usage: TokenUsage) -> float:
        """
        Estimate the cost in USD for the given token usage.
        
        Override this in subclasses with actual pricing.
        Default returns 0.0.
        """
        return 0.0
