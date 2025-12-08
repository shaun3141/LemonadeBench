# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
LemonadeBench Agent Framework.

This module provides the agent abstraction for building agents that play the
lemonade stand game. Includes:
- Base agent class with episode execution
- LLM-powered agent with multi-provider support
- Provider adapters for different LLM APIs
- Optional tools for agent assistance
- Multiple agent architectures (React, Plan-Act, Act-Reflect, Full)
"""

from .base import LemonadeAgent, EpisodeResult, TurnResult, AgentCallback
from .llm_agent import LLMAgent
from .tools import Tool, ToolResult, AVAILABLE_TOOLS, get_tool, CalculatorTool
from .prompts import (
    GOAL_FRAMING_PROMPTS,
    MATH_ENCOURAGEMENT_PROMPT,
    get_goal_framing_prompt,
    build_system_prompt,
)
from .architectures import (
    create_agent,
    get_architecture_class,
    ARCHITECTURES,
    ReactAgent,
    PlanActAgent,
    ActReflectAgent,
    FullAgent,
)
from .optimal_solver import OptimalSolver, SolverResult, DayResult, solve_seed

__all__ = [
    # Base classes
    "LemonadeAgent",
    "EpisodeResult",
    "TurnResult",
    "AgentCallback",
    # Legacy LLMAgent (React architecture)
    "LLMAgent",
    # Architecture agents
    "ReactAgent",
    "PlanActAgent",
    "ActReflectAgent",
    "FullAgent",
    "create_agent",
    "get_architecture_class",
    "ARCHITECTURES",
    # Tools
    "Tool",
    "ToolResult",
    "AVAILABLE_TOOLS",
    "get_tool",
    "CalculatorTool",
    # Prompts
    "GOAL_FRAMING_PROMPTS",
    "MATH_ENCOURAGEMENT_PROMPT",
    "get_goal_framing_prompt",
    "build_system_prompt",
    # Optimal Solver
    "OptimalSolver",
    "SolverResult",
    "DayResult",
    "solve_seed",
]
