# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
Base tool interface for LemonadeBench.

Defines the abstract interface that all tools must implement, enabling
consistent tool registration and execution across different LLM providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TypedDict


class ToolDefinition(TypedDict):
    """Standard tool definition format (compatible with both Anthropic and OpenAI)."""
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolResult:
    """Result from executing a tool."""
    success: bool
    result: str
    error: str | None = None


class Tool(ABC):
    """
    Abstract base class for tools.
    
    Tools provide additional capabilities to LLM agents during evaluation.
    Each tool must define its schema and implement execution logic.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the tool."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what the tool does."""
        pass
    
    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """
        Return the tool definition in standard format.
        
        Returns:
            ToolDefinition dict compatible with LLM provider APIs
        """
        pass
    
    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute the tool with the given arguments.
        
        Args:
            **kwargs: Tool-specific arguments
            
        Returns:
            ToolResult with success status and output
        """
        pass


# Registry of available tools
_TOOL_REGISTRY: dict[str, type[Tool]] = {}


def register_tool(tool_class: type[Tool]) -> type[Tool]:
    """
    Decorator to register a tool class.
    
    Example:
        @register_tool
        class MyTool(Tool):
            ...
    """
    # Create a temporary instance to get the name
    # We'll use a simple approach that doesn't require instantiation
    tool_name = tool_class.__name__.lower().replace("tool", "")
    _TOOL_REGISTRY[tool_name] = tool_class
    return tool_class


# Tool name aliases (alternative names that map to registered tools)
_TOOL_ALIASES: dict[str, str] = {
    "code_interpreter": "codeinterpreter",  # Maps config name to registry name
    "run_python": "codeinterpreter",
}


def get_tool(name: str) -> Tool | None:
    """
    Get a tool instance by name.
    
    Args:
        name: Tool name (e.g., "calculator", "code_interpreter")
        
    Returns:
        Tool instance or None if not found
    """
    name_lower = name.lower()
    
    # Check aliases first
    if name_lower in _TOOL_ALIASES:
        name_lower = _TOOL_ALIASES[name_lower]
    
    tool_class = _TOOL_REGISTRY.get(name_lower)
    if tool_class:
        return tool_class()
    return None


def AVAILABLE_TOOLS() -> list[str]:
    """Return list of registered tool names (including common aliases)."""
    tools = list(_TOOL_REGISTRY.keys())
    # Add common aliases that users might expect
    tools.extend(["code_interpreter"])
    return tools
