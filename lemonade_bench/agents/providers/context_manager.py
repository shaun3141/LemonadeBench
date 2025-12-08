# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
Context window management for LLM providers.

Fetches model context limits from OpenRouter and provides utilities
for tracking token usage and truncating conversation history when needed.
"""

import os
import json
import time
from pathlib import Path
from typing import Any

import requests


# Cache file for model limits
CACHE_FILE = Path(__file__).parent / ".model_limits_cache.json"
CACHE_TTL = 86400  # 24 hours


def fetch_openrouter_models() -> dict[str, dict]:
    """
    Fetch all model information from OpenRouter API.
    
    Returns:
        Dict mapping model_id to model info (context_length, pricing, etc.)
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return {}
    
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        response.raise_for_status()
        
        models = response.json().get("data", [])
        return {m["id"]: m for m in models}
    except Exception as e:
        print(f"Warning: Failed to fetch OpenRouter models: {e}")
        return {}


def get_model_limits(force_refresh: bool = False) -> dict[str, dict]:
    """
    Get model limits, using cache if available and fresh.
    
    Args:
        force_refresh: Force fetching from API even if cache exists
        
    Returns:
        Dict mapping model_id to model info
    """
    # Check cache
    if not force_refresh and CACHE_FILE.exists():
        try:
            with open(CACHE_FILE) as f:
                cache = json.load(f)
            
            # Check if cache is fresh
            if time.time() - cache.get("timestamp", 0) < CACHE_TTL:
                return cache.get("models", {})
        except Exception:
            pass
    
    # Fetch fresh data
    models = fetch_openrouter_models()
    
    if models:
        # Save to cache
        try:
            with open(CACHE_FILE, "w") as f:
                json.dump({"timestamp": time.time(), "models": models}, f)
        except Exception:
            pass
    
    return models


def get_context_limit(model_id: str) -> int:
    """
    Get the context limit for a specific model.
    
    Args:
        model_id: OpenRouter model ID (e.g., "openai/gpt-4o-mini")
        
    Returns:
        Context limit in tokens, or default of 128000 if not found
    """
    models = get_model_limits()
    
    # Try exact match
    if model_id in models:
        return models[model_id].get("context_length", 128000)
    
    # Try without provider prefix
    if "/" in model_id:
        base_name = model_id.split("/", 1)[1]
        for mid, info in models.items():
            if mid.endswith(base_name):
                return info.get("context_length", 128000)
    
    # Default fallback
    return 128000


def estimate_tokens(text: str) -> int:
    """
    Rough estimate of tokens in text.
    Uses ~4 chars per token heuristic for English text.
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    return len(text) // 4


def estimate_message_tokens(messages: list[dict[str, Any]]) -> int:
    """
    Estimate total tokens in a message list.
    
    Args:
        messages: List of message dicts
        
    Returns:
        Estimated token count
    """
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        total += estimate_tokens(block.get("text", ""))
                    elif block.get("type") == "tool_result":
                        total += estimate_tokens(block.get("content", ""))
                    elif block.get("type") == "tool_use":
                        total += estimate_tokens(str(block.get("input", {})))
        # Add overhead for role, etc.
        total += 10
    return total


def truncate_conversation(
    messages: list[dict[str, Any]],
    max_tokens: int,
    keep_first: int = 1,
    keep_last: int = 4,
) -> list[dict[str, Any]]:
    """
    Truncate conversation history to fit within token limit.
    
    Preserves the first N messages (usually system context) and 
    last M messages (recent context), removing middle messages.
    
    Args:
        messages: Full message history
        max_tokens: Maximum allowed tokens
        keep_first: Number of messages to always keep at start
        keep_last: Number of messages to always keep at end
        
    Returns:
        Truncated message list
    """
    if not messages:
        return messages
    
    current_tokens = estimate_message_tokens(messages)
    
    if current_tokens <= max_tokens:
        return messages
    
    # Always keep first and last messages
    if len(messages) <= keep_first + keep_last:
        return messages
    
    first_messages = messages[:keep_first]
    last_messages = messages[-keep_last:]
    middle_messages = messages[keep_first:-keep_last]
    
    # Calculate tokens for preserved messages
    preserved_tokens = (
        estimate_message_tokens(first_messages) +
        estimate_message_tokens(last_messages)
    )
    
    # Add truncation marker overhead
    marker_tokens = 50
    available_for_middle = max_tokens - preserved_tokens - marker_tokens
    
    if available_for_middle <= 0:
        # Can't fit any middle messages
        truncation_marker = {
            "role": "user",
            "content": f"[... {len(middle_messages)} earlier messages truncated to fit context limit ...]"
        }
        return first_messages + [truncation_marker] + last_messages
    
    # Keep as many middle messages as possible (from the end, more recent)
    kept_middle = []
    middle_tokens = 0
    
    for msg in reversed(middle_messages):
        msg_tokens = estimate_message_tokens([msg])
        if middle_tokens + msg_tokens <= available_for_middle:
            kept_middle.insert(0, msg)
            middle_tokens += msg_tokens
        else:
            break
    
    # Calculate how many were removed
    removed_count = len(middle_messages) - len(kept_middle)
    
    if removed_count > 0:
        truncation_marker = {
            "role": "user",
            "content": f"[... {removed_count} earlier messages truncated to fit context limit ...]"
        }
        return first_messages + [truncation_marker] + kept_middle + last_messages
    
    return first_messages + kept_middle + last_messages


class ContextManager:
    """
    Manages context window for a specific model.
    
    Tracks token usage and automatically truncates conversation
    when approaching context limit.
    """
    
    TRUNCATION_THRESHOLD = 0.70  # Truncate at 70% of context limit (more aggressive)
    
    def __init__(self, model_id: str, max_completion_tokens: int = 1024):
        """
        Initialize context manager.
        
        Args:
            model_id: OpenRouter model ID
            max_completion_tokens: Tokens reserved for completion
        """
        self.model_id = model_id
        self.context_limit = get_context_limit(model_id)
        self.max_completion_tokens = max_completion_tokens
        
        # Effective limit is context - completion tokens
        self.effective_limit = self.context_limit - max_completion_tokens
        self.truncation_threshold = int(self.effective_limit * self.TRUNCATION_THRESHOLD)
    
    def should_truncate(self, messages: list[dict[str, Any]]) -> bool:
        """Check if messages should be truncated."""
        return estimate_message_tokens(messages) > self.truncation_threshold
    
    def truncate_if_needed(
        self,
        messages: list[dict[str, Any]],
        keep_first: int = 1,
        keep_last: int = 4,
    ) -> list[dict[str, Any]]:
        """
        Truncate messages if they exceed the threshold.
        
        Args:
            messages: Message history
            keep_first: Messages to preserve at start
            keep_last: Messages to preserve at end
            
        Returns:
            Possibly truncated messages
        """
        if not self.should_truncate(messages):
            return messages
        
        return truncate_conversation(
            messages,
            self.truncation_threshold,
            keep_first=keep_first,
            keep_last=keep_last,
        )
    
    def get_usage_stats(self, messages: list[dict[str, Any]]) -> dict:
        """Get current context usage statistics."""
        current = estimate_message_tokens(messages)
        return {
            "current_tokens": current,
            "context_limit": self.context_limit,
            "effective_limit": self.effective_limit,
            "truncation_threshold": self.truncation_threshold,
            "usage_percent": current / self.effective_limit * 100,
            "should_truncate": current > self.truncation_threshold,
        }


# Script to refresh cache
if __name__ == "__main__":
    print("Fetching model limits from OpenRouter...")
    models = get_model_limits(force_refresh=True)
    print(f"Cached {len(models)} models")
    
    # Show some popular models
    popular = [
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "anthropic/claude-sonnet-4",
        "anthropic/claude-3.5-sonnet",
        "google/gemini-2.0-flash-exp:free",
        "deepseek/deepseek-r1",
    ]
    
    print("\nPopular model context limits:")
    for mid in popular:
        limit = get_context_limit(mid)
        print(f"  {mid}: {limit:,} tokens")

