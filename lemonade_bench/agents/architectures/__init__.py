# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
Agent architecture implementations for LemonadeBench.

This module provides different agent loop architectures as described in the paper:
- React: Basic observe -> decide -> act loop (baseline)
- Plan-Act: Generates multi-day plan before each action
- Act-Reflect: Reflects on results after each action  
- Full: Combines planning and reflection

Example:
    from lemonade_bench.agents.architectures import create_agent
    
    agent = create_agent(
        provider=provider,
        architecture="plan_act",
        goal_framing="aggressive",
        tools=["calculator"],
    )
"""

from typing import Literal

from ..providers.base import LLMProvider
from ..base import LemonadeAgent
from .react import ReactAgent
from .plan_act import PlanActAgent
from .act_reflect import ActReflectAgent
from .full import FullAgent

ArchitectureType = Literal["react", "plan_act", "act_reflect", "full"]

# Registry of architecture implementations
ARCHITECTURES: dict[ArchitectureType, type[LemonadeAgent]] = {
    "react": ReactAgent,
    "plan_act": PlanActAgent,
    "act_reflect": ActReflectAgent,
    "full": FullAgent,
}


def get_architecture_class(architecture: ArchitectureType) -> type[LemonadeAgent]:
    """
    Get the agent class for a given architecture type.
    
    Args:
        architecture: Architecture type name
        
    Returns:
        Agent class implementing that architecture
        
    Raises:
        ValueError: If architecture is not recognized
    """
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Valid options: {list(ARCHITECTURES.keys())}"
        )
    return ARCHITECTURES[architecture]


def create_agent(
    provider: LLMProvider,
    architecture: ArchitectureType = "react",
    goal_framing: str = "baseline",
    tools: list[str] | None = None,
    math_prompt: bool = False,
) -> LemonadeAgent:
    """
    Factory function to create an agent with the specified architecture and configuration.
    
    Args:
        provider: LLM provider instance
        architecture: Agent architecture type
        goal_framing: Goal framing condition
        tools: List of tool names to enable
        math_prompt: Whether to enable math encouragement prompt
        
    Returns:
        Configured agent instance
        
    Example:
        agent = create_agent(
            provider=AnthropicProvider(model="claude-sonnet-4-20250514"),
            architecture="plan_act",
            goal_framing="competitive",
            tools=["calculator"],
        )
    """
    agent_cls = get_architecture_class(architecture)
    
    return agent_cls(
        provider=provider,
        goal_framing=goal_framing,
        tools=tools,
        math_prompt=math_prompt,
    )


__all__ = [
    "ReactAgent",
    "PlanActAgent", 
    "ActReflectAgent",
    "FullAgent",
    "create_agent",
    "get_architecture_class",
    "ARCHITECTURES",
    "ArchitectureType",
]

