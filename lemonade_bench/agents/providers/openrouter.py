# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
OpenRouter provider for LemonadeBench.

Implements the LLMProvider interface for OpenRouter's API,
which provides access to 400+ models from different providers
(OpenAI, Anthropic, Google, Meta, Mistral, etc.) through a unified API.

OpenRouter is OpenAI SDK compatible, so we use the OpenAI client
with a custom base_url.
"""

import os
import time
from typing import Any

from openai import OpenAI, RateLimitError, APIError, APIConnectionError

from .base import LLMProvider, ToolResponse, TokenUsage
from .context_manager import ContextManager


# Retry configuration
MAX_RETRIES = 5
BASE_DELAY = 1.0  # seconds
MAX_DELAY = 60.0  # seconds


# OpenRouter API base URL
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Pricing per 1M tokens for popular models (as of December 2025)
# These are approximate - OpenRouter may have different pricing
# You can check actual pricing at https://openrouter.ai/models
OPENROUTER_PRICING = {
    # ==========================================================================
    # OpenAI models
    # ==========================================================================
    "openai/gpt-4o": {"input": 2.50, "output": 10.00},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "openai/gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "openai/o1": {"input": 15.00, "output": 60.00},
    "openai/o1-mini": {"input": 3.00, "output": 12.00},
    "openai/o1-preview": {"input": 15.00, "output": 60.00},
    "openai/o3-mini": {"input": 1.10, "output": 4.40},
    "openai/o3-mini-high": {"input": 1.10, "output": 4.40},
    "openai/gpt-4.5-preview": {"input": 75.00, "output": 150.00},
    "openai/gpt-5.1": {"input": 1.25, "output": 10.00},  # Dec 2025 flagship
    
    # ==========================================================================
    # Anthropic models - Top tier for agentic tasks
    # ==========================================================================
    "anthropic/claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
    "anthropic/claude-3.5-sonnet:beta": {"input": 3.00, "output": 15.00},
    "anthropic/claude-3.5-haiku": {"input": 0.80, "output": 4.00},
    "anthropic/claude-3-opus": {"input": 15.00, "output": 75.00},
    "anthropic/claude-sonnet-4": {"input": 3.00, "output": 15.00},
    "anthropic/claude-opus-4": {"input": 15.00, "output": 75.00},
    "anthropic/claude-opus-4.5": {"input": 15.00, "output": 75.00},  # Latest Dec 2025
    
    # ==========================================================================
    # Google Gemini models - Strong multimodal & reasoning
    # ==========================================================================
    "google/gemini-2.5-pro-preview": {"input": 1.25, "output": 10.00},
    "google/gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "google/gemini-2.5-flash-preview": {"input": 0.30, "output": 2.50},
    "google/gemini-3-pro": {"input": 1.50, "output": 12.00},  # Dec 2025 flagship
    "google/gemini-2.0-flash-exp:free": {"input": 0.0, "output": 0.0},
    "google/gemini-pro-1.5": {"input": 1.25, "output": 5.00},
    "google/gemini-flash-1.5": {"input": 0.075, "output": 0.30},
    "google/gemini-flash-1.5-8b": {"input": 0.0375, "output": 0.15},
    
    # ==========================================================================
    # DeepSeek models - Excellent value for reasoning (Dec 2025 top picks)
    # ==========================================================================
    "deepseek/deepseek-chat": {"input": 0.14, "output": 0.28},
    "deepseek/deepseek-r1": {"input": 0.55, "output": 2.19},
    "deepseek/deepseek-r1-distill-llama-70b": {"input": 0.40, "output": 0.40},
    "deepseek/deepseek-r1-distill-qwen-32b": {"input": 0.20, "output": 0.20},
    "deepseek/deepseek-v3": {"input": 0.14, "output": 0.28},
    
    # ==========================================================================
    # Meta Llama models - Strong open source
    # ==========================================================================
    "meta-llama/llama-3.3-70b-instruct": {"input": 0.40, "output": 0.40},
    "meta-llama/llama-3.1-405b-instruct": {"input": 2.00, "output": 2.00},
    "meta-llama/llama-3.1-70b-instruct": {"input": 0.40, "output": 0.40},
    "meta-llama/llama-3.1-8b-instruct": {"input": 0.06, "output": 0.06},
    "meta-llama/llama-4-scout": {"input": 0.50, "output": 0.50},  # Dec 2025
    "meta-llama/llama-4-maverick": {"input": 0.80, "output": 0.80},  # Dec 2025
    
    # ==========================================================================
    # Qwen models - Strong multilingual & coding
    # ==========================================================================
    "qwen/qwen-2.5-72b-instruct": {"input": 0.35, "output": 0.40},
    "qwen/qwen-2.5-coder-32b-instruct": {"input": 0.20, "output": 0.20},
    "qwen/qwen3-235b-a22b": {"input": 0.50, "output": 0.50},  # Dec 2025 flagship
    "qwen/qwen3-32b": {"input": 0.20, "output": 0.20},
    "qwen/qwq-32b": {"input": 0.20, "output": 0.20},  # Reasoning model
    
    # ==========================================================================
    # Mistral models
    # ==========================================================================
    "mistralai/mistral-large": {"input": 2.00, "output": 6.00},
    "mistralai/mistral-medium": {"input": 2.70, "output": 8.10},
    "mistralai/mistral-small": {"input": 0.20, "output": 0.60},
    "mistralai/mistral-small-3": {"input": 0.10, "output": 0.30},  # Jan 2025
    "mistralai/mixtral-8x7b-instruct": {"input": 0.24, "output": 0.24},
    "mistralai/codestral-latest": {"input": 0.30, "output": 0.90},
    
    # ==========================================================================
    # Cohere models
    # ==========================================================================
    "cohere/command-r-plus": {"input": 2.50, "output": 10.00},
    "cohere/command-r": {"input": 0.15, "output": 0.60},
    "cohere/command-a": {"input": 2.50, "output": 10.00},  # Dec 2025
    
    # ==========================================================================
    # Other notable models
    # ==========================================================================
    "nvidia/llama-3.1-nemotron-70b-instruct": {"input": 0.40, "output": 0.40},
    "x-ai/grok-2": {"input": 5.00, "output": 10.00},
    "x-ai/grok-3": {"input": 3.00, "output": 15.00},  # Dec 2025
    
    # Default fallback (conservative estimate)
    "default": {"input": 1.00, "output": 3.00},
}

# Common model aliases for convenience
# These allow using short names like "deepseek-r1" instead of "deepseek/deepseek-r1"
MODEL_ALIASES = {
    # OpenAI
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-4.5": "openai/gpt-4.5-preview",
    "o1": "openai/o1",
    "o1-mini": "openai/o1-mini",
    "o3-mini": "openai/o3-mini",
    "o3-mini-high": "openai/o3-mini-high",
    "gpt-5.1": "openai/gpt-5.1",
    
    # Anthropic
    "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
    "claude-3.5-haiku": "anthropic/claude-3.5-haiku",
    "claude-3-opus": "anthropic/claude-3-opus",
    "claude-sonnet-4": "anthropic/claude-sonnet-4",
    "claude-opus-4": "anthropic/claude-opus-4",
    "claude-opus-4.5": "anthropic/claude-opus-4.5",
    
    # Google Gemini
    "gemini-pro": "google/gemini-pro-1.5",
    "gemini-flash": "google/gemini-flash-1.5",
    "gemini-2.5-pro": "google/gemini-2.5-pro-preview",
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    "gemini-2.0-flash": "google/gemini-2.0-flash-exp:free",
    "gemini-3-pro": "google/gemini-3-pro",
    
    # DeepSeek - Best value reasoning models
    "deepseek-chat": "deepseek/deepseek-chat",
    "deepseek-r1": "deepseek/deepseek-r1",
    "deepseek-v3": "deepseek/deepseek-v3",
    "deepseek-r1-70b": "deepseek/deepseek-r1-distill-llama-70b",
    "deepseek-r1-32b": "deepseek/deepseek-r1-distill-qwen-32b",
    
    # Meta Llama
    "llama-3.3-70b": "meta-llama/llama-3.3-70b-instruct",
    "llama-3.1-405b": "meta-llama/llama-3.1-405b-instruct",
    "llama-3.1-70b": "meta-llama/llama-3.1-70b-instruct",
    "llama-4-scout": "meta-llama/llama-4-scout",
    "llama-4-maverick": "meta-llama/llama-4-maverick",
    
    # Qwen
    "qwen-2.5-72b": "qwen/qwen-2.5-72b-instruct",
    "qwen-coder": "qwen/qwen-2.5-coder-32b-instruct",
    "qwen3-235b": "qwen/qwen3-235b-a22b",
    "qwq-32b": "qwen/qwq-32b",
    
    # Mistral
    "mistral-large": "mistralai/mistral-large",
    "mistral-small": "mistralai/mistral-small-3",
    "codestral": "mistralai/codestral-latest",
    
    # X.AI
    "grok-2": "x-ai/grok-2",
    "grok-3": "x-ai/grok-3",
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


class OpenRouterProvider(LLMProvider):
    """
    OpenRouter provider implementation.
    
    OpenRouter provides access to 400+ AI models through a single API.
    It uses an OpenAI-compatible interface, making it easy to switch
    between models from different providers.
    
    Models are specified in the format "provider/model-name", for example:
    - "openai/gpt-4o"
    - "anthropic/claude-3.5-sonnet"
    - "google/gemini-pro-1.5"
    - "meta-llama/llama-3.3-70b-instruct"
    
    You can also use short aliases like "gpt-4o" or "claude-3.5-sonnet".
    
    Example:
        provider = OpenRouterProvider(model="anthropic/claude-3.5-sonnet")
        response = provider.generate_with_tool(messages, system_prompt)
        print(response.tool_input)  # {'price_per_cup': 75, ...}
    
    Browse available models: https://openrouter.ai/models
    """
    
    def __init__(
        self,
        model: str = "openai/gpt-4o",
        api_key: str | None = None,
        max_tokens: int = 1024,
        site_url: str | None = None,
        site_name: str | None = None,
    ):
        """
        Initialize the OpenRouter provider.
        
        Args:
            model: Model identifier (e.g., "openai/gpt-4o", "anthropic/claude-3.5-sonnet")
                  Can also use short aliases like "gpt-4o" or "claude-3.5-sonnet"
            api_key: API key (defaults to OPENROUTER_API_KEY env var)
            max_tokens: Maximum tokens in response
            site_url: Optional URL for your site (helps OpenRouter track usage)
            site_name: Optional name for your site/app
        """
        # Resolve model aliases
        self._model = MODEL_ALIASES.get(model, model)
        self._max_tokens = max_tokens
        self._site_url = site_url
        self._site_name = site_name or "LemonadeBench"
        
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY environment "
                "variable or pass api_key parameter. Get a key at https://openrouter.ai/keys"
            )
        
        # Use OpenAI SDK with OpenRouter base URL
        self._client = OpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": self._site_url or "https://github.com/Shaun3141/LemonadeBench",
                "X-Title": self._site_name,
            }
        )
        
        # Context manager for automatic truncation
        self._context_manager = ContextManager(
            model_id=self._model,
            max_completion_tokens=max_tokens,
        )
    
    @property
    def model_name(self) -> str:
        return self._model
    
    @property
    def provider_name(self) -> str:
        return "openrouter"
    
    def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tools: list[dict[str, Any]],
        required_tool: str | None = None,
    ) -> ToolResponse:
        """
        Generate a response with tool calls using OpenRouter.
        
        Args:
            messages: Conversation history (will be converted to OpenAI format)
            system_prompt: System prompt for the model
            tools: List of tool definitions (can be empty for text-only responses)
            required_tool: If set, force the model to use this specific tool
            
        Returns:
            ToolResponse with the tool call details (or text_content for text-only)
        """
        import json
        
        # Auto-truncate if approaching context limit (90%)
        if self._context_manager.should_truncate(messages):
            messages = self._context_manager.truncate_if_needed(
                messages,
                keep_first=1,  # Keep initial context
                keep_last=6,   # Keep recent turns
            )
        
        # Convert messages to OpenAI format
        openai_messages = self._convert_messages(messages, system_prompt)
        
        # Handle case where no tools are provided (planning/reflection phases)
        if not tools:
            return self._generate_text_response(openai_messages)
        
        # Convert tools to OpenAI format
        openai_tools = [convert_tool_to_openai_format(tool) for tool in tools]
        
        # Set tool_choice based on whether a specific tool is required
        if required_tool:
            tool_choice = {"type": "function", "function": {"name": required_tool}}
        else:
            tool_choice = "required"  # Must use one of the tools
        
        # Retry with exponential backoff for rate limits
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    messages=openai_messages,
                    tools=openai_tools,
                    tool_choice=tool_choice,
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
        
        # Extract tool call
        message = response.choices[0].message
        text_content = message.content  # May contain reasoning text
        
        if not message.tool_calls:
            raise ValueError(
                f"No tool call found in OpenRouter response for model {self._model}. "
                f"Response content: {text_content[:200] if text_content else 'None'}... "
                "The model may not support function calling or the prompt needs adjustment."
            )
        
        tool_call = message.tool_calls[0]
        
        # Parse arguments - try JSON first, then Python literal as fallback
        # Some models return Python-style dicts with single quotes, or mix JSON/Python
        args_str = tool_call.function.arguments
        try:
            tool_input = json.loads(args_str)
        except json.JSONDecodeError:
            import ast
            import re
            try:
                # Convert JSON null/true/false to Python None/True/False
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
        
        return ToolResponse(
            tool_name=tool_call.function.name,
            tool_input=tool_input,
            tool_use_id=tool_call.id,
            raw_response=response,
            usage=usage,
            text_content=text_content,
        )
    
    def _generate_text_response(self, openai_messages: list[dict[str, Any]]) -> ToolResponse:
        """
        Generate a text-only response (no tools) for planning/reflection phases.
        
        Args:
            openai_messages: Messages in OpenAI format
            
        Returns:
            ToolResponse with text_content set
        """
        # Retry with exponential backoff for rate limits
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    messages=openai_messages,
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
        
        message = response.choices[0].message
        text_content = message.content or ""
        
        # Extract usage
        usage = None
        if response.usage:
            usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
        
        return ToolResponse(
            tool_name="",  # No tool for text responses
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
        Format a tool result for OpenRouter's conversation format.
        
        Note: OpenRouter uses OpenAI format, which has a separate "tool" role.
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
        Format OpenRouter's response as an assistant message.
        
        Stores in a format that can be converted back for OpenRouter.
        """
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
        
        Note: OpenRouter pricing varies by model and may change.
        Check https://openrouter.ai/models for current pricing.
        """
        pricing = OPENROUTER_PRICING.get(self._model, OPENROUTER_PRICING["default"])
        
        input_cost = (usage.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (usage.output_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost
    
    @staticmethod
    def list_popular_models() -> list[str]:
        """
        Return a list of popular models available on OpenRouter.
        
        For the full list, visit https://openrouter.ai/models
        """
        return [
            # === Top Tier for Agentic Reasoning (Dec 2025) ===
            "anthropic/claude-opus-4.5",      # Best for long autonomous tasks
            "anthropic/claude-sonnet-4",       # Best all-around balance
            "google/gemini-3-pro",             # Top multimodal reasoning
            "deepseek/deepseek-r1",            # Best value reasoning
            "openai/o3-mini",                  # Strong reasoning, cost-effective
            
            # === Premium Models ===
            "anthropic/claude-opus-4",
            "openai/o1",
            "openai/gpt-4o",
            "google/gemini-2.5-pro-preview",
            "x-ai/grok-3",
            
            # === Cost-Effective Reasoning ===
            "deepseek/deepseek-chat",          # Very cheap, solid
            "deepseek/deepseek-r1-distill-llama-70b",
            "deepseek/deepseek-v3",
            "qwen/qwq-32b",                    # Reasoning specialist
            
            # === Strong Open Source ===
            "meta-llama/llama-3.3-70b-instruct",
            "meta-llama/llama-3.1-405b-instruct",
            "qwen/qwen-2.5-72b-instruct",
            "qwen/qwen3-235b-a22b",
            
            # === Fast & Efficient ===
            "anthropic/claude-3.5-haiku",
            "openai/gpt-4o-mini",
            "google/gemini-2.5-flash",
            "mistralai/mistral-small-3",
            
            # === Free Tier ===
            "google/gemini-2.0-flash-exp:free",
        ]
    
    @staticmethod
    def list_best_for_agentic() -> list[str]:
        """
        Return the best models for agentic reasoning tasks (Dec 2025).
        
        These models excel at:
        - Tool/function calling
        - Multi-step reasoning
        - Long-horizon autonomous tasks
        - Code generation and debugging
        """
        return [
            # Tier 1: Best overall agentic performance
            "anthropic/claude-opus-4.5",       # 80.9% SWE-Bench, long autonomy
            "anthropic/claude-sonnet-4",       # Best balance of quality/cost
            "google/gemini-3-pro",             # Top multimodal, MathArena leader
            
            # Tier 2: Excellent reasoning
            "deepseek/deepseek-r1",            # 97.3% MATH-500, MIT licensed
            "openai/o3-mini",                  # 87.5% ARC-AGI
            "openai/o1",                       # Strong reasoning
            
            # Tier 3: Cost-effective options
            "deepseek/deepseek-chat",          # $0.14/M input, very capable
            "qwen/qwq-32b",                    # Reasoning specialist
            "meta-llama/llama-3.3-70b-instruct",  # Open source leader
        ]
