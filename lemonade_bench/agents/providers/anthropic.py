# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
Anthropic Claude provider for LemonadeBench.

Implements the LLMProvider interface for Anthropic's Claude models,
using the tool_use feature for structured action outputs.
"""

import os
import time
from typing import Any

from anthropic import Anthropic, RateLimitError, APIError, APIConnectionError

from .base import LLMProvider, ToolResponse, TokenUsage


# Retry configuration
MAX_RETRIES = 5
BASE_DELAY = 1.0  # seconds
MAX_DELAY = 60.0  # seconds


# Pricing per 1M tokens (as of late 2024, update as needed)
ANTHROPIC_PRICING = {
    # Claude 3.5 Sonnet
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    # Claude 3.5 Haiku
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    # Claude 3 Opus
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    # Default fallback
    "default": {"input": 3.00, "output": 15.00},
}


class AnthropicProvider(LLMProvider):
    """
    Anthropic Claude provider implementation.
    
    Uses Claude's native tool_use feature for structured outputs.
    
    Example:
        provider = AnthropicProvider(model="claude-sonnet-4-20250514")
        response = provider.generate_with_tool(messages, system_prompt)
        print(response.tool_input)  # {'price_per_cup': 75, ...}
    """
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        max_tokens: int = 1024,
    ):
        """
        Initialize the Anthropic provider.
        
        Args:
            model: Model identifier (e.g., "claude-sonnet-4-20250514")
            api_key: API key (defaults to ANTHROPIC_API_KEY env var)
            max_tokens: Maximum tokens in response
        """
        self._model = model
        self._max_tokens = max_tokens
        
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment "
                "variable or pass api_key parameter."
            )
        
        self._client = Anthropic(api_key=api_key)
    
    @property
    def model_name(self) -> str:
        return self._model
    
    @property
    def provider_name(self) -> str:
        return "anthropic"
    
    def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tools: list[dict[str, Any]],
        required_tool: str | None = None,
    ) -> ToolResponse:
        """
        Generate a response with tool calls using Claude.
        
        Args:
            messages: Conversation history in Anthropic format
            system_prompt: System prompt for Claude
            tools: List of tool definitions (can be empty for text-only responses)
            required_tool: If set, force the model to use this specific tool
            
        Returns:
            ToolResponse with the tool call details
        """
        # Handle case where no tools are provided (planning/reflection phases)
        if not tools:
            return self._generate_text_response(messages, system_prompt)
        
        # Set tool_choice based on whether a specific tool is required
        if required_tool:
            tool_choice = {"type": "tool", "name": required_tool}
        else:
            tool_choice = {"type": "any"}  # Model must use one of the tools
        
        # Retry with exponential backoff for rate limits
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    system=system_prompt,
                    tools=tools,
                    tool_choice=tool_choice,
                    messages=messages,
                )
                break
            except RateLimitError as e:
                last_error = e
                delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                time.sleep(delay)
            except APIConnectionError as e:
                last_error = e
                delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                time.sleep(delay)
            except APIError as e:
                # Check if it's a retryable error (5xx status codes)
                if hasattr(e, 'status_code') and e.status_code >= 500:
                    last_error = e
                    delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                    time.sleep(delay)
                else:
                    raise  # Non-retryable API error
        else:
            raise last_error  # All retries failed
        
        # Extract tool use and text content from response
        tool_use_block = None
        text_content = ""
        for block in response.content:
            if block.type == "tool_use":
                tool_use_block = block
            elif block.type == "text":
                text_content += block.text
        
        if tool_use_block is None:
            raise ValueError(
                f"No tool_use block found in Claude response. "
                f"Text content: {text_content[:200]}..."
            )
        
        # Extract usage
        usage = None
        if response.usage:
            usage = TokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
        
        return ToolResponse(
            tool_name=tool_use_block.name,
            tool_input=tool_use_block.input,
            tool_use_id=tool_use_block.id,
            raw_response=response,
            usage=usage,
            text_content=text_content if text_content else None,
        )
    
    def _generate_text_response(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
    ) -> ToolResponse:
        """
        Generate a text-only response (no tools) for planning/reflection phases.
        
        Args:
            messages: Conversation history in Anthropic format
            system_prompt: System prompt for Claude
            
        Returns:
            ToolResponse with text_content set
        """
        # Retry with exponential backoff for rate limits
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    system=system_prompt,
                    messages=messages,
                )
                break
            except RateLimitError as e:
                last_error = e
                delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                time.sleep(delay)
            except APIConnectionError as e:
                last_error = e
                delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                time.sleep(delay)
            except APIError as e:
                if hasattr(e, 'status_code') and e.status_code >= 500:
                    last_error = e
                    delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                    time.sleep(delay)
                else:
                    raise
        else:
            raise last_error
        
        # Extract text content
        text_content = ""
        for block in response.content:
            if block.type == "text":
                text_content += block.text
        
        # Extract usage
        usage = None
        if response.usage:
            usage = TokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
        
        return ToolResponse(
            tool_name="",  # No tool for text responses
            tool_input={},
            tool_use_id="",
            raw_response=response,
            usage=usage,
            text_content=text_content,
        )
    
    def format_tool_result(
        self,
        tool_use_id: str,
        result: str,
    ) -> dict[str, Any]:
        """
        Format a tool result for Claude's conversation format.
        
        Args:
            tool_use_id: ID of the tool use
            result: Result string
            
        Returns:
            User message with tool_result content block
        """
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result,
                }
            ]
        }
    
    def format_assistant_message(self, response: ToolResponse) -> dict[str, Any]:
        """
        Format Claude's response as an assistant message.
        
        Args:
            response: The ToolResponse
            
        Returns:
            Assistant message dict
        """
        return {
            "role": "assistant",
            "content": response.raw_response.content,
        }
    
    def estimate_cost(self, usage: TokenUsage) -> float:
        """
        Estimate cost in USD based on token usage.
        
        Args:
            usage: Token usage from the response
            
        Returns:
            Estimated cost in USD
        """
        pricing = ANTHROPIC_PRICING.get(self._model, ANTHROPIC_PRICING["default"])
        
        input_cost = (usage.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (usage.output_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost
