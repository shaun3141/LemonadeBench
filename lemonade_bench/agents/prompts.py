# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
Prompt templates for LemonadeBench agents.

Contains goal-framing prompts, math encouragement prompts, and architecture-specific
prompts as described in the paper methodology.
"""

from typing import Literal

GoalFramingType = Literal["baseline", "aggressive", "conservative", "competitive", "survival", "growth"]


# =============================================================================
# Goal-Framing Prompts (Section 4.4 / Appendix D)
# =============================================================================
# These prompts are prepended to the base system prompt to test how motivational
# framing affects agent economic behavior.

GOAL_FRAMING_PROMPTS: dict[GoalFramingType, str] = {
    "baseline": "",  # No additional framing
    
    "aggressive": (
        "You are an aggressive entrepreneur who takes calculated risks to maximize returns. "
        "Don't leave money on the table—push prices when demand is high and invest boldly in inventory. "
        "Fortune favors the bold. Your competitors are playing it safe; you're here to win big."
    ),
    
    "conservative": (
        "You are a cautious business owner who prioritizes avoiding losses over maximizing gains. "
        "Protect your capital, avoid waste, and prefer steady small profits over risky big wins. "
        "A bird in hand is worth two in the bush. Consistency and reliability will serve you "
        "better than gambling on big days."
    ),
    
    "competitive": (
        "You are competing against 10 other lemonade stands in a tournament. "
        "Only the top 3 profit earners will win prizes. The average stand makes about $25-30 profit "
        "over the season. You need to significantly outperform this average to place. "
        "Second place is first loser—aim for the top."
    ),
    
    "survival": (
        "Your family depends on this business. You started with $20, and if you end the season "
        "with less than $20, you will have failed your family. Survival is the absolute priority. "
        "Do not take unnecessary risks. Losing money is not an option. Every decision should "
        "prioritize capital preservation."
    ),
    
    "growth": (
        "You're building a lemonade empire. This 14-day season is just the beginning of a much "
        "longer journey. Focus on building strong reputation and deeply learning the market dynamics, "
        "even if it costs some short-term profit. Experiment with locations, understand customer "
        "behavior, and invest in your stand's capabilities. The lessons you learn now will pay "
        "dividends for years to come."
    ),
}


# =============================================================================
# Math Encouragement Prompt (Section 4.3)
# =============================================================================
# Used as scaffolding to encourage explicit step-by-step calculations

MATH_ENCOURAGEMENT_PROMPT = """
Always show your math. Before setting a price, calculate:
1. Your cost per cup (lemons + sugar + cups + ice)
2. Expected demand at different price points given the weather
3. Projected profit at your chosen price point

Write out the calculations step by step before making your final decision.
"""


# =============================================================================
# Architecture-Specific Prompts (Section 4.2 / Appendix E)
# =============================================================================

PLANNING_PROMPT = """
Before taking action, create a strategic plan for the next 3-5 days:

1. **Weather Outlook**: What weather do you expect? How will it affect demand?
2. **Inventory Needs**: What supplies will you need? When should you buy them?
3. **Pricing Strategy**: What prices will work best for each weather condition?
4. **Location Considerations**: Should you consider changing location?
5. **Key Risks**: What could go wrong? How will you handle it?

Output your plan, then decide today's action based on it.
"""

REFLECTION_PROMPT = """
Reflect on yesterday's results before making today's decision:

1. **What Worked Well?** What decisions led to positive outcomes?
2. **What Could You Have Done Better?** Where did you lose money or miss opportunities?
3. **What Surprised You?** Was demand higher or lower than expected?
4. **What Will You Do Differently?** How will you adjust your strategy?

Use these insights to inform today's action.
"""

PLAN_UPDATE_PROMPT = """
Review your previous plan in light of yesterday's results:

Previous plan: {previous_plan}

Has anything changed that invalidates this plan? Do you need to adjust your strategy?
Update your plan if needed, then take today's action.
"""


# =============================================================================
# Helper Functions
# =============================================================================

def get_goal_framing_prompt(framing: GoalFramingType) -> str:
    """
    Get the goal-framing prompt for a given condition.
    
    Args:
        framing: Goal framing condition name
        
    Returns:
        The prompt text to prepend to the system prompt
    """
    if framing not in GOAL_FRAMING_PROMPTS:
        raise ValueError(f"Unknown goal framing: {framing}. Valid options: {list(GOAL_FRAMING_PROMPTS.keys())}")
    return GOAL_FRAMING_PROMPTS[framing]


def build_system_prompt(
    base_prompt: str,
    goal_framing: GoalFramingType = "baseline",
    math_prompt: bool = False,
    tools_available: list[str] | None = None,
) -> str:
    """
    Build a complete system prompt with all configured options.
    
    Args:
        base_prompt: The base system prompt explaining game mechanics
        goal_framing: Goal framing condition to apply
        math_prompt: Whether to include math encouragement scaffolding
        tools_available: List of tool names available to the agent
        
    Returns:
        Complete system prompt with all additions
    """
    parts = []
    
    # Add goal framing first (affects overall tone)
    framing_prompt = get_goal_framing_prompt(goal_framing)
    if framing_prompt:
        parts.append(framing_prompt)
    
    # Add base prompt
    parts.append(base_prompt)
    
    # Add math encouragement if enabled
    if math_prompt:
        parts.append(MATH_ENCOURAGEMENT_PROMPT)
    
    # Add tools section if tools are available
    if tools_available:
        tools_section = f"\n\n## Available Tools\nYou have access to the following optional tools: {', '.join(tools_available)}.\nUse them when helpful for calculations or analysis before making your final action."
        parts.append(tools_section)
    
    return "\n\n".join(parts)

