# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
Tools module for LemonadeBench agents.

Provides optional tools that can be made available to LLM agents during evaluation.
Tools are configured per-model in the batch config file.

Available tools:
- calculator: Basic arithmetic calculations
- code_interpreter (run_python): Sandboxed Python execution for complex analysis
"""

from .base import Tool, ToolDefinition, ToolResult, AVAILABLE_TOOLS, get_tool
from .calculator import CalculatorTool
from .code_interpreter import CodeInterpreterTool

__all__ = [
    "Tool",
    "ToolDefinition",
    "ToolResult",
    "AVAILABLE_TOOLS",
    "get_tool",
    "CalculatorTool",
    "CodeInterpreterTool",
]
