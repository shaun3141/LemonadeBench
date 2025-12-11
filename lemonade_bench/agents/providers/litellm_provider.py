# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
LiteLLM provider for LemonadeBench.

LiteLLM provides a unified interface for calling 100+ LLMs with automatic
handling of provider-specific features like Gemini 3's thought signatures.

This provider is especially useful for:
- Models requiring special handling (Gemini 3 Pro, etc.)
- Unified interface across all providers
- Automatic retry and fallback logic
- Cost tracking across providers

Best Practices Applied (from LiteLLM docs):
1. Use built-in num_retries and timeout instead of custom retry loops
2. Append full response.choices[0].message for thought signature preservation
3. Use default temperature=1.0 for Gemini 3 models
4. Add metadata/tags for tracking
5. Use LiteLLM's cost tracking when available
"""

import json
import os
from typing import Any

import litellm
from litellm import completion
from litellm.exceptions import RateLimitError, APIConnectionError, APIError

from .base import LLMProvider, ToolResponse, TokenUsage
from .context_manager import ContextManager


# Configuration
DEFAULT_TIMEOUT = 120  # seconds - generous for reasoning models
DEFAULT_NUM_RETRIES = 5  # LiteLLM handles exponential backoff automatically


# Model name mapping for convenience
# LiteLLM uses format: provider/model-name
# For OpenRouter routing: openrouter/provider/model-name
MODEL_ALIASES = {
    # Google Gemini (direct API)
    "gemini-3-pro": "gemini/gemini-3-pro-preview",
    "gemini-3-pro-preview": "gemini/gemini-3-pro-preview",
    "gemini-2.5-pro": "gemini/gemini-2.5-pro",
    "gemini-2.5-flash": "gemini/gemini-2.5-flash",
    "gemini-2.0-flash": "gemini/gemini-2.0-flash",
    "gemini-pro": "gemini/gemini-pro",
    
    # Google Gemini via OpenRouter (uses OPENROUTER_API_KEY)
    "openrouter-gemini-3-pro": "openrouter/google/gemini-3-pro-preview",
    "openrouter-gemini-2.5-pro": "openrouter/google/gemini-2.5-pro",
    
    # OpenAI
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-4-turbo": "openai/gpt-4-turbo",
    "o1": "openai/o1",
    "o3-mini": "openai/o3-mini",
    
    # Anthropic
    "claude-3.5-sonnet": "anthropic/claude-3-5-sonnet-latest",
    "claude-3.5-haiku": "anthropic/claude-3-5-haiku-latest",
    "claude-sonnet-4": "anthropic/claude-sonnet-4-latest",
    "claude-opus-4": "anthropic/claude-opus-4-latest",
    
    # DeepSeek
    "deepseek-chat": "deepseek/deepseek-chat",
    "deepseek-r1": "deepseek/deepseek-reasoner",
}


# Pricing per 1M tokens (fallback if LiteLLM cost tracking unavailable)
LITELLM_PRICING = {
    "gemini/gemini-3-pro-preview": {"input": 1.50, "output": 12.00},
    "gemini/gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini/gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini/gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "openrouter/google/gemini-3-pro-preview": {"input": 1.50, "output": 12.00},
    "openrouter/google/gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "openai/gpt-4o": {"input": 2.50, "output": 10.00},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "anthropic/claude-3-5-sonnet-latest": {"input": 3.00, "output": 15.00},
    "anthropic/claude-3-5-haiku-latest": {"input": 0.80, "output": 4.00},
    "deepseek/deepseek-chat": {"input": 0.14, "output": 0.28},
    "default": {"input": 1.00, "output": 3.00},
}


def convert_tool_to_openai_format(tool: dict[str, Any]) -> dict[str, Any]:
    """
    Convert our tool format to OpenAI's function format (used by LiteLLM).
    """
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"],
        }
    }


class LiteLLMProvider(LLMProvider):
    """
    LiteLLM provider implementation.
    
    LiteLLM provides a unified interface for calling 100+ LLMs including:
    - Google Gemini (with thought signature support for Gemini 3)
    - OpenAI GPT models
    - Anthropic Claude models
    - DeepSeek models
    - And many more
    
    Key benefits:
    - Automatic handling of model-specific features (thought signatures, etc.)
    - Unified tool/function calling interface
    - Built-in retry logic with exponential backoff
    - Cost tracking across providers
    
    Example:
        provider = LiteLLMProvider(model="gemini/gemini-3-pro-preview")
        response = provider.generate_with_tool(messages, system_prompt)
        print(response.tool_input)  # {'price_per_cup': 75, ...}
    """
    
    def __init__(
        self,
        model: str = "gemini/gemini-3-pro-preview",
        api_key: str | None = None,
        max_tokens: int = 1024,
        reasoning_effort: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        num_retries: int = DEFAULT_NUM_RETRIES,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Initialize the LiteLLM provider.
        
        Args:
            model: Model identifier (e.g., "gemini/gemini-3-pro-preview")
                  Can also use short aliases like "gemini-3-pro"
            api_key: API key (will use appropriate env var if not provided)
            max_tokens: Maximum tokens in response
            reasoning_effort: For Gemini 3 models, set to "low"/"medium"/"high" 
                            to enable thought signatures
            timeout: Request timeout in seconds (default: 120)
            num_retries: Number of retries for transient errors (default: 5)
            metadata: Optional metadata/tags for tracking (e.g., {"tags": ["production"]})
        """
        # Resolve model aliases
        self._model = MODEL_ALIASES.get(model, model)
        self._max_tokens = max_tokens
        self._reasoning_effort = reasoning_effort
        self._timeout = timeout
        self._num_retries = num_retries
        self._metadata = metadata or {}
        
        # Set API key based on model provider
        if api_key:
            self._set_api_key(api_key)
        else:
            self._configure_api_keys()
        
        # Context manager for automatic truncation
        self._context_manager = ContextManager(
            model_id=self._model,
            max_completion_tokens=max_tokens,
        )
        
        # Store last response message for thought signature preservation
        # BEST PRACTICE: Always append response.choices[0].message to preserve thought signatures
        self._last_assistant_message: Any = None
    
    def _set_api_key(self, api_key: str) -> None:
        """Set API key for the appropriate provider."""
        if self._model.startswith("openrouter/"):
            os.environ["OPENROUTER_API_KEY"] = api_key
        elif self._model.startswith("gemini/"):
            os.environ["GEMINI_API_KEY"] = api_key
        elif self._model.startswith("openai/"):
            os.environ["OPENAI_API_KEY"] = api_key
        elif self._model.startswith("anthropic/"):
            os.environ["ANTHROPIC_API_KEY"] = api_key
        elif self._model.startswith("deepseek/"):
            os.environ["DEEPSEEK_API_KEY"] = api_key
    
    def _configure_api_keys(self) -> None:
        """Verify appropriate API keys are configured."""
        if self._model.startswith("openrouter/"):
            # OpenRouter routing - uses OPENROUTER_API_KEY
            if not os.environ.get("OPENROUTER_API_KEY"):
                raise ValueError(
                    "OPENROUTER_API_KEY environment variable required for OpenRouter models. "
                    "Get a key at https://openrouter.ai/keys"
                )
        elif self._model.startswith("gemini/"):
            if not os.environ.get("GEMINI_API_KEY"):
                raise ValueError(
                    "GEMINI_API_KEY environment variable required for Gemini models. "
                    "Get a key at https://aistudio.google.com/apikey"
                )
        elif self._model.startswith("openai/"):
            if not os.environ.get("OPENAI_API_KEY"):
                raise ValueError(
                    "OPENAI_API_KEY environment variable required for OpenAI models."
                )
        elif self._model.startswith("anthropic/"):
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise ValueError(
                    "ANTHROPIC_API_KEY environment variable required for Anthropic models."
                )
    
    @property
    def model_name(self) -> str:
        return self._model
    
    @property
    def provider_name(self) -> str:
        return "litellm"
    
    def _is_gemini_3(self) -> bool:
        """Check if this is a Gemini 3 model requiring special handling."""
        return "gemini-3" in self._model
    
    def _build_request_kwargs(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
    ) -> dict[str, Any]:
        """
        Build the request kwargs for LiteLLM completion.
        
        Applies best practices:
        - Use num_retries for built-in retry with exponential backoff
        - Use timeout for request limits
        - Use default temperature=1.0 for Gemini 3 (lowering causes issues)
        - Add reasoning_effort for Gemini 3 thought signatures
        - Include metadata for tracking
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "timeout": self._timeout,
            "num_retries": self._num_retries,
        }
        
        # Add tools if provided
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
        
        # Gemini 3 specific settings
        if self._is_gemini_3():
            # BEST PRACTICE: Use default temperature=1.0 for Gemini 3
            # Lowering temperature leads to suboptimal performance
            kwargs["temperature"] = 1.0
            
            # Enable thought signatures with reasoning_effort
            if self._reasoning_effort:
                kwargs["reasoning_effort"] = self._reasoning_effort
        
        # Add metadata for tracking if provided
        if self._metadata:
            kwargs["metadata"] = self._metadata
        
        return kwargs
    
    def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tools: list[dict[str, Any]],
        required_tool: str | None = None,
    ) -> ToolResponse:
        """
        Generate a response with tool calls using LiteLLM.
        
        LiteLLM automatically handles thought signatures for Gemini 3 models
        when we append the full response message to conversation history.
        
        Args:
            messages: Conversation history
            system_prompt: System prompt for the model
            tools: List of tool definitions (can be empty for text-only)
            required_tool: If set, force the model to use this specific tool
            
        Returns:
            ToolResponse with the tool call details
        """
        # Auto-truncate if approaching context limit
        if self._context_manager.should_truncate(messages):
            messages = self._context_manager.truncate_if_needed(
                messages,
                keep_first=1,
                keep_last=6,
            )
        
        # Convert messages to LiteLLM format (OpenAI-compatible)
        litellm_messages = self._convert_messages(messages, system_prompt)
        
        # Handle case where no tools are provided (planning/reflection phases)
        if not tools:
            return self._generate_text_response(litellm_messages)
        
        # Convert tools to OpenAI format
        openai_tools = [convert_tool_to_openai_format(tool) for tool in tools]
        
        # Set tool_choice based on whether a specific tool is required
        if required_tool:
            tool_choice = {"type": "function", "function": {"name": required_tool}}
        else:
            tool_choice = "required"
        
        # Build request with best practices applied
        request_kwargs = self._build_request_kwargs(
            messages=litellm_messages,
            tools=openai_tools,
            tool_choice=tool_choice,
        )
        
        # Make the call - LiteLLM handles retries internally
        response = completion(**request_kwargs)
        
        # Extract tool call
        message = response.choices[0].message
        text_content = message.content
        
        if not message.tool_calls:
            raise ValueError(
                f"No tool call found in LiteLLM response for model {self._model}. "
                f"Response content: {text_content[:200] if text_content else 'None'}... "
                "The model may not support function calling or the prompt needs adjustment."
            )
        
        tool_call = message.tool_calls[0]
        
        # Parse arguments
        args_str = tool_call.function.arguments
        try:
            tool_input = json.loads(args_str)
        except json.JSONDecodeError:
            import ast
            import re
            try:
                converted = args_str.replace(': null', ': None')
                converted = converted.replace(':null', ':None')
                converted = re.sub(r'\bnull\b', 'None', converted)
                converted = re.sub(r'\btrue\b', 'True', converted)
                converted = re.sub(r'\bfalse\b', 'False', converted)
                tool_input = ast.literal_eval(converted)
            except (ValueError, SyntaxError) as e:
                raise ValueError(
                    f"Failed to parse tool arguments for model {self._model}: "
                    f"{args_str[:200]}... Error: {e}"
                )
        
        # Extract usage
        usage = None
        if response.usage:
            usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
        
        # BEST PRACTICE: Store the FULL message object for thought signature preservation
        # This is critical for Gemini 3 - thought signatures are in provider_specific_fields
        self._last_assistant_message = message
        
        return ToolResponse(
            tool_name=tool_call.function.name,
            tool_input=tool_input,
            tool_use_id=tool_call.id,
            raw_response=response,
            usage=usage,
            text_content=text_content,
        )
    
    def _generate_text_response(self, litellm_messages: list[dict[str, Any]]) -> ToolResponse:
        """Generate a text-only response for planning/reflection phases."""
        # Build request with best practices
        request_kwargs = self._build_request_kwargs(messages=litellm_messages)
        
        # Make the call - LiteLLM handles retries internally
        response = completion(**request_kwargs)
        
        message = response.choices[0].message
        text_content = message.content or ""
        
        usage = None
        if response.usage:
            usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
        
        # Store for thought signature preservation
        self._last_assistant_message = message
        
        return ToolResponse(
            tool_name="",
            tool_input={},
            tool_use_id="",
            raw_response=response,
            usage=usage,
            text_content=text_content,
        )
    
    def _convert_messages(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
    ) -> list[dict[str, Any]]:
        """
        Convert messages to LiteLLM format (OpenAI-compatible).
        
        BEST PRACTICE: For Gemini 3 models, we preserve the full message objects
        to maintain thought signatures across turns. LiteLLM automatically
        handles missing thought signatures when transitioning between models.
        """
        litellm_messages = [{"role": "system", "content": system_prompt}]
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            # BEST PRACTICE: If content is a LiteLLM message object, preserve it
            # This maintains thought signatures for Gemini 3
            if hasattr(content, "model_dump"):
                litellm_messages.append(content.model_dump())
                continue
            
            # Handle ChatCompletionMessage objects directly
            if hasattr(content, "role") and hasattr(content, "content"):
                msg_dict = {"role": content.role, "content": content.content}
                if hasattr(content, "tool_calls") and content.tool_calls:
                    msg_dict["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        }
                        for tc in content.tool_calls
                    ]
                # Preserve provider_specific_fields for thought signatures
                if hasattr(content, "provider_specific_fields") and content.provider_specific_fields:
                    msg_dict["provider_specific_fields"] = content.provider_specific_fields
                litellm_messages.append(msg_dict)
                continue
            
            if isinstance(content, str):
                litellm_messages.append({"role": role, "content": content})
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
                                    "arguments": json.dumps(block.get("input", {})),
                                }
                            })
                    elif hasattr(block, "type"):
                        if block.type == "tool_use":
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
                    litellm_messages.append({
                        "role": "assistant",
                        "content": " ".join(text_parts) if text_parts else None,
                        "tool_calls": tool_calls,
                    })
                elif role == "user" and tool_results:
                    for tr in tool_results:
                        litellm_messages.append({
                            "role": "tool",
                            "tool_call_id": tr["tool_call_id"],
                            "content": tr["content"],
                        })
                    if text_parts:
                        litellm_messages.append({
                            "role": "user",
                            "content": " ".join(text_parts),
                        })
                else:
                    litellm_messages.append({
                        "role": role,
                        "content": " ".join(text_parts) if text_parts else "",
                    })
        
        return litellm_messages
    
    def format_tool_result(
        self,
        tool_use_id: str,
        result: str,
    ) -> dict[str, Any]:
        """Format a tool result for inclusion in the conversation."""
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
        Format the assistant's tool call response as a message.
        
        BEST PRACTICE: For thought signature preservation, we store the full
        message object. When appending to conversation history, use:
        
            messages.append(response.choices[0].message)
        
        NOT:
            messages.append({"role": "assistant", "tool_calls": [...]})
        """
        # Use the raw LiteLLM message to preserve thought signatures
        if self._last_assistant_message is not None:
            return {
                "role": "assistant",
                "content": self._last_assistant_message,
            }
        
        # Fallback to standard format (loses thought signatures)
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
        
        Uses our local pricing table. LiteLLM also provides cost tracking
        via litellm.completion_cost() but may require additional setup.
        """
        pricing = LITELLM_PRICING.get(self._model, LITELLM_PRICING["default"])
        
        input_cost = (usage.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (usage.output_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost
    
    @staticmethod
    def list_supported_models() -> list[str]:
        """Return a list of commonly used models with LiteLLM."""
        return [
            # Google Gemini via OpenRouter
            "openrouter/google/gemini-3-pro-preview",
            "openrouter/google/gemini-2.5-pro",
            
            # Google Gemini (direct API - requires GEMINI_API_KEY)
            "gemini/gemini-3-pro-preview",
            "gemini/gemini-2.5-pro",
            "gemini/gemini-2.5-flash",
            "gemini/gemini-2.0-flash",
            
            # OpenAI
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "openai/o1",
            "openai/o3-mini",
            
            # Anthropic
            "anthropic/claude-3-5-sonnet-latest",
            "anthropic/claude-3-5-haiku-latest",
            "anthropic/claude-sonnet-4-latest",
            
            # DeepSeek
            "deepseek/deepseek-chat",
            "deepseek/deepseek-reasoner",
        ]
