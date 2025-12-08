# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
OpenAI provider for LemonadeBench.

Implements the LLMProvider interface for OpenAI's GPT models,
using function calling for structured action outputs.
"""

import os
from typing import Any

from openai import OpenAI

from .base import LLMProvider, ToolResponse, TokenUsage


# Pricing per 1M tokens (as of late 2024, update as needed)
OPENAI_PRICING = {
    # GPT-4o
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-11-20": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.60},
    # o1 models
    "o1-preview": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 3.00, "output": 12.00},
    # GPT-4 Turbo
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4-turbo-preview": {"input": 10.00, "output": 30.00},
    # Default fallback
    "default": {"input": 2.50, "output": 10.00},
}


def convert_tool_to_openai_format(tool: dict[str, Any]) -> dict[str, Any]:
    """
    Convert our tool format to OpenAI's function format.
    
    Args:
        tool: Tool definition in Anthropic-like format
        
    Returns:
        Tool definition in OpenAI format
    """
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"],
        }
    }


class OpenAIProvider(LLMProvider):
    """
    OpenAI GPT provider implementation.
    
    Uses OpenAI's function calling feature for structured outputs.
    
    Example:
        provider = OpenAIProvider(model="gpt-4o")
        response = provider.generate_with_tool(messages, system_prompt)
        print(response.tool_input)  # {'price_per_cup': 75, ...}
    """
    
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        max_tokens: int = 1024,
    ):
        """
        Initialize the OpenAI provider.
        
        Args:
            model: Model identifier (e.g., "gpt-4o", "gpt-4o-mini")
            api_key: API key (defaults to OPENAI_API_KEY env var)
            max_tokens: Maximum tokens in response
        """
        self._model = model
        self._max_tokens = max_tokens
        
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment "
                "variable or pass api_key parameter."
            )
        
        self._client = OpenAI(api_key=api_key)
    
    @property
    def model_name(self) -> str:
        return self._model
    
    @property
    def provider_name(self) -> str:
        return "openai"
    
    def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tools: list[dict[str, Any]],
        required_tool: str | None = None,
    ) -> ToolResponse:
        """
        Generate a response with tool calls using OpenAI.
        
        Args:
            messages: Conversation history (will be converted to OpenAI format)
            system_prompt: System prompt for the model
            tools: List of tool definitions
            required_tool: If set, force the model to use this specific tool
            
        Returns:
            ToolResponse with the tool call details
        """
        import json
        
        # Convert messages to OpenAI format
        openai_messages = self._convert_messages(messages, system_prompt)
        
        # Convert tools to OpenAI format
        openai_tools = [convert_tool_to_openai_format(tool) for tool in tools]
        
        # Set tool_choice based on whether a specific tool is required
        if required_tool:
            tool_choice = {"type": "function", "function": {"name": required_tool}}
        else:
            tool_choice = "required"  # Must use one of the tools
        
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=openai_messages,
            tools=openai_tools,
            tool_choice=tool_choice,
        )
        
        # Extract tool call
        message = response.choices[0].message
        if not message.tool_calls:
            raise ValueError("No tool call found in OpenAI response")
        
        tool_call = message.tool_calls[0]
        
        # Parse arguments
        tool_input = json.loads(tool_call.function.arguments)
        
        # Extract usage
        usage = None
        if response.usage:
            usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
        
        return ToolResponse(
            tool_name=tool_call.function.name,
            tool_input=tool_input,
            tool_use_id=tool_call.id,
            raw_response=response,
            usage=usage,
        )
    
    def _convert_messages(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
    ) -> list[dict[str, Any]]:
        """
        Convert messages from Anthropic format to OpenAI format.
        
        Args:
            messages: Messages in Anthropic-like format
            system_prompt: System prompt to prepend
            
        Returns:
            Messages in OpenAI format
        """
        openai_messages = [{"role": "system", "content": system_prompt}]
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if isinstance(content, str):
                openai_messages.append({"role": role, "content": content})
            elif isinstance(content, list):
                # Handle multi-part content
                text_parts = []
                tool_calls = []
                tool_results = []
                
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_result":
                            tool_results.append({
                                "tool_call_id": block.get("tool_use_id"),
                                "content": block.get("content", ""),
                            })
                        elif block.get("type") == "tool_use":
                            tool_calls.append({
                                "id": block.get("id"),
                                "type": "function",
                                "function": {
                                    "name": block.get("name"),
                                    "arguments": str(block.get("input", {})),
                                }
                            })
                    elif hasattr(block, "type"):
                        # Handle Anthropic response objects
                        if block.type == "tool_use":
                            import json
                            tool_calls.append({
                                "id": block.id,
                                "type": "function",
                                "function": {
                                    "name": block.name,
                                    "arguments": json.dumps(block.input),
                                }
                            })
                        elif block.type == "text":
                            text_parts.append(block.text)
                
                if role == "assistant" and tool_calls:
                    openai_messages.append({
                        "role": "assistant",
                        "content": " ".join(text_parts) if text_parts else None,
                        "tool_calls": tool_calls,
                    })
                elif role == "user" and tool_results:
                    for tr in tool_results:
                        openai_messages.append({
                            "role": "tool",
                            "tool_call_id": tr["tool_call_id"],
                            "content": tr["content"],
                        })
                    # Add any text content as user message
                    if text_parts:
                        openai_messages.append({
                            "role": "user",
                            "content": " ".join(text_parts),
                        })
                else:
                    openai_messages.append({
                        "role": role,
                        "content": " ".join(text_parts) if text_parts else "",
                    })
        
        return openai_messages
    
    def format_tool_result(
        self,
        tool_use_id: str,
        result: str,
    ) -> dict[str, Any]:
        """
        Format a tool result for OpenAI's conversation format.
        
        Note: OpenAI uses a separate "tool" role for tool results.
        We return in our internal format and convert in _convert_messages.
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
        Format OpenAI's response as an assistant message.
        
        Stores in a format that can be converted back for OpenAI.
        """
        import json
        return {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": response.tool_use_id,
                    "name": response.tool_name,
                    "input": response.tool_input,
                }
            ]
        }
    
    def estimate_cost(self, usage: TokenUsage) -> float:
        """
        Estimate cost in USD based on token usage.
        """
        pricing = OPENAI_PRICING.get(self._model, OPENAI_PRICING["default"])
        
        input_cost = (usage.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (usage.output_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost
