# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
LLM Provider adapters for LemonadeBench.

This module provides a unified interface for interacting with different LLM
providers (Anthropic, OpenAI, OpenRouter, LiteLLM, etc.) for the LLM-powered agent.
"""

from .base import LLMProvider, ToolResponse, TokenUsage
from .anthropic import AnthropicProvider
from .model_registry import ModelRegistry, ModelInfo, ValidationResult, validate_models_or_raise

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

# LiteLLM is optional - provides unified interface with thought signature support
try:
    from .litellm_provider import LiteLLMProvider
    _LITELLM_AVAILABLE = True
except ImportError:
    _LITELLM_AVAILABLE = False
    LiteLLMProvider = None  # type: ignore

# vLLM provider for local inference with fine-tuned models
try:
    from .vllm_provider import VLLMProvider
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False
    VLLMProvider = None  # type: ignore

__all__ = [
    "LLMProvider",
    "ToolResponse",
    "TokenUsage",
    "AnthropicProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
    "LiteLLMProvider",
    "VLLMProvider",
    "ModelRegistry",
    "ModelInfo",
    "ValidationResult",
    "validate_models_or_raise",
]
