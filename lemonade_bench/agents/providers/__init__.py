# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
LLM Provider adapters for LemonadeBench.

This module provides a unified interface for interacting with different LLM
providers (Anthropic, OpenAI, OpenRouter, etc.) for the LLM-powered agent.
"""

from .base import LLMProvider, ToolResponse, TokenUsage
from .anthropic import AnthropicProvider

# OpenAI is optional - only import if the package is available
try:
    from .openai import OpenAIProvider
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    OpenAIProvider = None  # type: ignore

# OpenRouter uses the OpenAI SDK, so it's also optional
try:
    from .openrouter import OpenRouterProvider
    _OPENROUTER_AVAILABLE = True
except ImportError:
    _OPENROUTER_AVAILABLE = False
    OpenRouterProvider = None  # type: ignore

__all__ = [
    "LLMProvider",
    "ToolResponse",
    "TokenUsage",
    "AnthropicProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
]
