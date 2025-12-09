# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
Model registry for OpenRouter model validation.

Fetches model information from OpenRouter API and provides utilities for:
- Validating model names exist
- Checking function calling support
- Suggesting similar models for invalid names
- Caching model data for performance
"""

import json
import os
import time
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import requests


# Cache configuration
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_FILE = CACHE_DIR / "openrouter_models.json"
CACHE_TTL = 3600  # 1 hour (more frequent than context_manager for accuracy)


@dataclass
class ModelInfo:
    """Information about a model from OpenRouter."""
    id: str
    name: str
    context_length: int
    supports_tools: bool
    supports_tool_choice: bool
    supported_parameters: list[str]
    pricing_prompt: float  # per token
    pricing_completion: float  # per token
    description: str = ""
    
    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> "ModelInfo":
        """Create ModelInfo from OpenRouter API response."""
        params = data.get("supported_parameters", [])
        pricing = data.get("pricing", {})
        
        return cls(
            id=data["id"],
            name=data.get("name", data["id"]),
            context_length=data.get("context_length", 128000),
            supports_tools="tools" in params,
            supports_tool_choice="tool_choice" in params,
            supported_parameters=params,
            pricing_prompt=float(pricing.get("prompt", "0")),
            pricing_completion=float(pricing.get("completion", "0")),
            description=data.get("description", "")[:200],  # Truncate for storage
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for caching."""
        return {
            "id": self.id,
            "name": self.name,
            "context_length": self.context_length,
            "supports_tools": self.supports_tools,
            "supports_tool_choice": self.supports_tool_choice,
            "supported_parameters": self.supported_parameters,
            "pricing_prompt": self.pricing_prompt,
            "pricing_completion": self.pricing_completion,
            "description": self.description,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelInfo":
        """Create from cached dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            context_length=data["context_length"],
            supports_tools=data["supports_tools"],
            supports_tool_choice=data["supports_tool_choice"],
            supported_parameters=data["supported_parameters"],
            pricing_prompt=data["pricing_prompt"],
            pricing_completion=data["pricing_completion"],
            description=data.get("description", ""),
        )


@dataclass
class ValidationResult:
    """Result of model validation."""
    is_valid: bool
    model_id: str
    error_message: str | None = None
    suggestions: list[str] | None = None
    model_info: ModelInfo | None = None
    
    def raise_if_invalid(self):
        """Raise ValueError if validation failed."""
        if not self.is_valid:
            msg = self.error_message or f"Invalid model: {self.model_id}"
            if self.suggestions:
                msg += f"\n\nDid you mean one of these?\n  - " + "\n  - ".join(self.suggestions)
            raise ValueError(msg)


class ModelRegistry:
    """
    Registry of available OpenRouter models.
    
    Provides model validation, capability checking, and suggestions.
    Automatically fetches and caches model data from OpenRouter API.
    
    Example:
        registry = ModelRegistry()
        
        # Validate a model
        result = registry.validate("openai/gpt-4o")
        result.raise_if_invalid()
        
        # Check function calling support
        if registry.supports_function_calling("openai/gpt-4o"):
            print("Function calling supported!")
        
        # Get suggestions for invalid model
        suggestions = registry.get_suggestions("openai/gpt4o")  # typo
        print(f"Did you mean: {suggestions}")
    """
    
    _instance: "ModelRegistry | None" = None
    _models: dict[str, ModelInfo] | None = None
    _last_fetch: float = 0
    
    def __new__(cls):
        """Singleton pattern for efficient caching."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the registry (loads from cache if available)."""
        if self._models is None:
            self._load_models()
    
    def _load_models(self, force_refresh: bool = False) -> None:
        """Load models from cache or fetch from API."""
        # Ensure cache directory exists
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Check cache
        if not force_refresh and CACHE_FILE.exists():
            try:
                with open(CACHE_FILE) as f:
                    cache = json.load(f)
                
                if time.time() - cache.get("timestamp", 0) < CACHE_TTL:
                    self._models = {
                        mid: ModelInfo.from_dict(mdata)
                        for mid, mdata in cache.get("models", {}).items()
                    }
                    self._last_fetch = cache.get("timestamp", 0)
                    return
            except Exception:
                pass  # Cache corrupted, fetch fresh
        
        # Fetch from API
        self._fetch_from_api()
    
    def _fetch_from_api(self) -> None:
        """Fetch model list from OpenRouter API."""
        api_key = os.environ.get("OPENROUTER_API_KEY")
        
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        try:
            response = requests.get(
                "https://openrouter.ai/api/v1/models",
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            
            data = response.json()
            models_list = data.get("data", [])
            
            self._models = {
                m["id"]: ModelInfo.from_api_response(m)
                for m in models_list
            }
            self._last_fetch = time.time()
            
            # Save to cache
            self._save_cache()
            
        except Exception as e:
            # If API fails and we have cached data, use it
            if self._models is None:
                self._models = {}
            print(f"Warning: Failed to fetch OpenRouter models: {e}")
    
    def _save_cache(self) -> None:
        """Save current models to cache file."""
        if not self._models:
            return
        
        try:
            cache = {
                "timestamp": self._last_fetch,
                "models": {
                    mid: minfo.to_dict()
                    for mid, minfo in self._models.items()
                }
            }
            with open(CACHE_FILE, "w") as f:
                json.dump(cache, f)
        except Exception:
            pass  # Non-critical
    
    def refresh(self) -> int:
        """
        Force refresh of model list from API.
        
        Returns:
            Number of models fetched
        """
        self._fetch_from_api()
        return len(self._models) if self._models else 0
    
    @property
    def model_count(self) -> int:
        """Number of models in registry."""
        return len(self._models) if self._models else 0
    
    @property
    def all_model_ids(self) -> list[str]:
        """List of all available model IDs."""
        return list(self._models.keys()) if self._models else []
    
    def get_model(self, model_id: str) -> ModelInfo | None:
        """
        Get model information by ID.
        
        Args:
            model_id: OpenRouter model ID (e.g., "openai/gpt-4o")
            
        Returns:
            ModelInfo if found, None otherwise
        """
        if not self._models:
            return None
        return self._models.get(model_id)
    
    def exists(self, model_id: str) -> bool:
        """Check if a model ID exists."""
        return model_id in (self._models or {})
    
    def supports_function_calling(self, model_id: str) -> bool:
        """
        Check if a model supports function/tool calling.
        
        Args:
            model_id: OpenRouter model ID
            
        Returns:
            True if model supports tools, False otherwise
        """
        model = self.get_model(model_id)
        if not model:
            return False
        return model.supports_tools
    
    def get_suggestions(self, model_id: str, n: int = 5) -> list[str]:
        """
        Get similar model suggestions for an invalid ID.
        
        Uses fuzzy matching to find similar model names.
        
        Args:
            model_id: Invalid model ID to find suggestions for
            n: Maximum number of suggestions
            
        Returns:
            List of similar model IDs
        """
        if not self._models:
            return []
        
        all_ids = list(self._models.keys())
        
        # Try fuzzy matching on full ID
        suggestions = get_close_matches(model_id, all_ids, n=n, cutoff=0.4)
        
        # If no matches, try matching on model name part only
        if not suggestions and "/" in model_id:
            provider, name = model_id.split("/", 1)
            # Filter to same provider
            provider_models = [m for m in all_ids if m.startswith(provider + "/")]
            if provider_models:
                model_names = [m.split("/", 1)[1] for m in provider_models]
                name_matches = get_close_matches(name, model_names, n=n, cutoff=0.3)
                suggestions = [f"{provider}/{nm}" for nm in name_matches]
        
        # Also show models from same provider if still no matches
        if not suggestions and "/" in model_id:
            provider = model_id.split("/")[0]
            suggestions = [m for m in all_ids if m.startswith(provider + "/")][:n]
        
        return suggestions
    
    def validate(self, model_id: str, require_tools: bool = True) -> ValidationResult:
        """
        Validate a model ID.
        
        Args:
            model_id: Model ID to validate
            require_tools: If True, also check for function calling support
            
        Returns:
            ValidationResult with status and details
        """
        # Check if model exists
        if not self.exists(model_id):
            suggestions = self.get_suggestions(model_id)
            return ValidationResult(
                is_valid=False,
                model_id=model_id,
                error_message=f"Model '{model_id}' not found on OpenRouter",
                suggestions=suggestions,
            )
        
        model_info = self.get_model(model_id)
        
        # Check for function calling support if required
        if require_tools and not model_info.supports_tools:
            # Find similar models that DO support tools
            provider = model_id.split("/")[0] if "/" in model_id else ""
            tool_models = [
                mid for mid, m in (self._models or {}).items()
                if m.supports_tools and mid.startswith(provider)
            ][:5]
            
            return ValidationResult(
                is_valid=False,
                model_id=model_id,
                error_message=(
                    f"Model '{model_id}' does not support function calling.\n"
                    f"Supported parameters: {model_info.supported_parameters}"
                ),
                suggestions=tool_models,
                model_info=model_info,
            )
        
        return ValidationResult(
            is_valid=True,
            model_id=model_id,
            model_info=model_info,
        )
    
    def validate_batch(
        self, 
        model_ids: list[str], 
        require_tools: bool = True
    ) -> tuple[list[str], list[ValidationResult]]:
        """
        Validate multiple models at once.
        
        Args:
            model_ids: List of model IDs to validate
            require_tools: If True, also check for function calling support
            
        Returns:
            Tuple of (valid_ids, failed_results)
        """
        valid = []
        failed = []
        
        for model_id in model_ids:
            result = self.validate(model_id, require_tools=require_tools)
            if result.is_valid:
                valid.append(model_id)
            else:
                failed.append(result)
        
        return valid, failed


def validate_models_or_raise(
    model_ids: list[str],
    require_tools: bool = True,
) -> None:
    """
    Validate a list of models and raise if any are invalid.
    
    Convenience function for pre-flight validation.
    
    Args:
        model_ids: List of model IDs to validate
        require_tools: If True, also check for function calling support
        
    Raises:
        ValueError: If any models are invalid, with helpful error message
    """
    registry = ModelRegistry()
    valid, failed = registry.validate_batch(model_ids, require_tools=require_tools)
    
    if failed:
        lines = [
            f"Found {len(failed)} invalid model(s):",
            "",
        ]
        
        for result in failed:
            lines.append(f"  ✗ {result.model_id}")
            if result.error_message:
                lines.append(f"    {result.error_message.split(chr(10))[0]}")
            if result.suggestions:
                lines.append(f"    Suggestions: {', '.join(result.suggestions[:3])}")
            lines.append("")
        
        raise ValueError("\n".join(lines))


# CLI for testing
if __name__ == "__main__":
    import sys
    
    print("Fetching model registry from OpenRouter...")
    registry = ModelRegistry()
    registry.refresh()
    print(f"Loaded {registry.model_count} models\n")
    
    # Test validation
    test_models = [
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "anthropic/claude-3.5-sonnet",
        "openai/gpt-5.1-chat",  # May not exist
        "google/gemini-3-pro",  # May not exist
        "mistralai/mistral-small-3",  # May not exist
        "x-ai/grok-2",  # May not exist
    ]
    
    print("=== Model Validation ===\n")
    for model_id in test_models:
        result = registry.validate(model_id)
        status = "✓" if result.is_valid else "✗"
        print(f"{status} {model_id}")
        if not result.is_valid:
            print(f"  Error: {result.error_message}")
            if result.suggestions:
                print(f"  Suggestions: {result.suggestions}")
        elif result.model_info:
            print(f"  Context: {result.model_info.context_length:,} tokens")
            print(f"  Tools: {result.model_info.supports_tools}")
        print()

