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
# Static Header (System Prompt) - Comprehensive game reference
# =============================================================================
# This is the main system prompt that explains all game mechanics.
# Sent once per episode as context, with goal framing injected.

STATIC_HEADER = """# Lemonade Stand Challenge

You run a lemonade stand for 14 days. **Goal: Maximize total profit.**

## Starting Conditions
- Cash: $20.00
- Inventory: 10 lemons, 5 sugar bags, 50 cups, 5 ice bags
- Location: Park (free)
- Reputation: 0.50 (neutral)

## Weather & Demand

| Weather | Multiplier | Typical Price |
|---------|------------|---------------|
| HOT     | 1.8x       | $0.75-1.25    |
| SUNNY   | 1.3x       | $0.60-1.00    |
| CLOUDY  | 0.9x       | $0.50-0.80    |
| RAINY   | 0.4x       | $0.40-0.60    |
| STORMY  | 0.1x       | $0.25-0.50    |

Temperature bonus: >85°F adds +2%/degree. <60°F halves demand.

## Supplies & Recipe

| Supply | Base Cost | Yield | Expires |
|--------|-----------|-------|---------|
| Lemon | $0.25 | 4 cups | 3 days |
| Sugar | $0.50/bag | 10 cups | Never |
| Cups | $0.05 | 1 cup | Never |
| Ice | $0.25/bag | 5 cups | Overnight* |

*Ice melts 100% overnight. With cooler upgrade: only 50% melts.

**Cost per cup**: ~$0.18 (lemon + sugar + ice + cup)

## Bulk Discounts (Auto-Applied)

Just specify the quantity you want - the system automatically applies the best bulk discount.

| Supply | 10% off threshold | 20% off threshold |
|--------|-------------------|-------------------|
| Lemons | 12+ (Dozen) | 144+ (Crate) |
| Sugar | 5+ (Case) | 20+ (Pallet) |
| Cups | 50+ (Sleeve) | 250+ (Case) |
| Ice | 5+ (Cooler Pack) | 20+ (Delivery) |

**Example**: `buy_lemons=24` → System buys 2 Dozens at 10% off = $5.40

## Locations

| Location | Permit | Traffic | Weather Exposure | Price Sensitivity |
|----------|--------|---------|------------------|-------------------|
| Park | Free | 1.0x | 1.0x (full) | Normal |
| Downtown | $10 | 1.2x | 0.7x (sheltered) | Lower (premium OK) |
| Mall | $15 | 0.7x | 0.0x (indoor) | Normal |
| Pool | $2.50 | 0.8x | 1.5x (amplified) | Lower on hot days |

## Upgrades
- **Cooler** ($2.50): Ice melts 50% per night instead of 100%

## Demand Formula
`Customers = Foot_Traffic × Conversion_Rate`

- **Foot Traffic**: base(50) × location × weather × reputation(0.5-1.5) × ads
- **Conversion**: 95% at $0.50, drops with higher prices
- **Ice Bonus**: +20% conversion on HOT/SUNNY if you have ice; -20% without

## Reputation
- Starts at 0.50, moves 20% toward daily satisfaction each day
- Lower prices = happier customers
- Turning away customers hurts satisfaction
"""


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
    base_prompt: str | None = None,
    goal_framing: GoalFramingType = "baseline",
    math_prompt: bool = False,
    tools_available: list[str] | None = None,
) -> str:
    """
    Build a complete system prompt with all configured options.
    
    Args:
        base_prompt: Optional custom base prompt. If None, uses STATIC_HEADER.
        goal_framing: Goal framing condition to apply
        math_prompt: Whether to include math encouragement scaffolding
        tools_available: List of tool names available to the agent
        
    Returns:
        Complete system prompt with all additions
    """
    # Use STATIC_HEADER if no custom base prompt provided
    prompt = base_prompt if base_prompt is not None else STATIC_HEADER
    
    # Add goal framing at the end
    framing_prompt = get_goal_framing_prompt(goal_framing)
    if framing_prompt:
        prompt = prompt + "\n\n## Your Approach\n" + framing_prompt
    
    # Add math encouragement if enabled
    if math_prompt:
        prompt = prompt + "\n" + MATH_ENCOURAGEMENT_PROMPT
    
    # Add tools section if tools are available
    if tools_available:
        tools_section = f"\n\n## Available Tools\nYou have access to the following optional tools: {', '.join(tools_available)}.\nUse them when helpful for calculations or analysis before making your final action."
        prompt = prompt + tools_section
    
    return prompt


def build_static_header(
    goal_framing: GoalFramingType = "baseline",
    math_prompt: bool = False,
    tools_available: list[str] | None = None,
) -> str:
    """
    Build the static header system prompt with goal framing.
    
    This is a convenience wrapper that always uses STATIC_HEADER.
    
    Args:
        goal_framing: Goal framing condition to apply
        math_prompt: Whether to include math encouragement scaffolding
        tools_available: List of tool names available to the agent
        
    Returns:
        Complete system prompt with static header and goal framing
    """
    return build_system_prompt(
        base_prompt=None,  # Use STATIC_HEADER
        goal_framing=goal_framing,
        math_prompt=math_prompt,
        tools_available=tools_available,
    )

