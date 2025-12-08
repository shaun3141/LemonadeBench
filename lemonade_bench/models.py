# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
Data models for the Lemonade Stand Environment.

A classic lemonade stand tycoon simulation where agents must:
- Set daily lemonade prices
- Decide how many cups to prepare
- Manage inventory and supplies
- Respond to weather conditions and customer demand
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from openenv_core.env_server.types import Action, Observation


class Weather(str, Enum):
    """Weather conditions that affect lemonade sales."""
    SUNNY = "sunny"
    CLOUDY = "cloudy"
    RAINY = "rainy"
    HOT = "hot"  # Extra hot day - high demand!
    STORMY = "stormy"  # Very few customers


class CustomerMood(str, Enum):
    """General customer sentiment in the area."""
    HAPPY = "happy"
    NEUTRAL = "neutral"
    GRUMPY = "grumpy"


class StandUpgrade(str, Enum):
    """Upgrades that can be purchased for the lemonade stand."""
    COOLER = "cooler"  # Ice melts 50% per day instead of 100%
    # Future upgrades can be added here:
    # UMBRELLA = "umbrella"  # Protects from rain, boosts rainy day sales
    # FANCY_CUPS = "fancy_cups"  # Premium cups increase customer satisfaction
    # JUICER = "juicer"  # More efficient lemon usage
    # NEON_SIGN = "neon_sign"  # Permanent advertising boost


class Location(str, Enum):
    """Locations where the lemonade stand can be set up."""
    PARK = "park"  # Neighborhood park - baseline location
    DOWNTOWN = "downtown"  # Business district - higher prices, partial shelter
    MALL = "mall"  # Shopping mall - indoor, weather-independent
    POOL = "pool"  # Community pool - weather-dependent hot spot


@dataclass(kw_only=True)
class LocationInfo:
    """Information about a stand location."""
    id: Location
    name: str
    description: str
    foot_traffic_multiplier: float  # Multiplier on base_customers (1.0 = normal)
    price_sensitivity: float  # How much demand drops per cent above optimal (lower = tolerates higher prices)
    hot_weather_price_sensitivity: Optional[float] = None  # Override sensitivity on hot/sunny days (for Pool)
    weather_exposure: float  # 0.0 = indoor (no weather effect), 1.0 = full weather effect, >1.0 = amplified
    permit_cost: int  # Cost in cents to switch to this location


# The catalog of all available locations
# Balance notes - designed for STRATEGIC DIVERSITY:
# - Park: High-volume play with moderate prices - most foot traffic, standard pricing
# - Downtown: Premium balanced - good prices accepted, consistent with partial shelter
# - Mall: Premium niche - LOW volume but can charge high prices, zero weather risk
# - Pool: Weather betting - amazing on hot days (traffic + premium pricing), terrible otherwise
LOCATION_CATALOG: Dict[Location, LocationInfo] = {
    Location.PARK: LocationInfo(
        id=Location.PARK,
        name="Neighborhood Park",
        description="Popular park with lots of families. High traffic, standard pricing.",
        foot_traffic_multiplier=1.2,  # High volume location
        price_sensitivity=0.018,  # Moderate - some room to raise prices
        weather_exposure=1.0,  # Full weather impact
        permit_cost=0,  # Free - home base location
    ),
    Location.DOWNTOWN: LocationInfo(
        id=Location.DOWNTOWN,
        name="Downtown",
        description="Business district with office workers. Premium prices accepted, partial shelter.",
        foot_traffic_multiplier=1.0,  # Standard traffic
        price_sensitivity=0.012,  # Tolerates higher prices - office workers have money
        weather_exposure=0.7,  # Good shelter from buildings
        permit_cost=1000,  # $10.00 - premium location
    ),
    Location.MALL: LocationInfo(
        id=Location.MALL,
        name="Shopping Mall",
        description="Indoor mall - low traffic but shoppers pay premium. Weather-proof.",
        foot_traffic_multiplier=0.7,  # LOW volume - trade traffic for price tolerance
        price_sensitivity=0.008,  # Best price tolerance - mall shoppers expect high prices
        weather_exposure=0.0,  # Indoor - no weather effect
        permit_cost=1500,  # $15.00 - most expensive, pay for consistency
    ),
    Location.POOL: LocationInfo(
        id=Location.POOL,
        name="Community Pool",
        description="Hot day goldmine! Premium prices on hot days, dead otherwise.",
        foot_traffic_multiplier=0.9,  # Slightly below average base
        price_sensitivity=0.020,  # Budget-conscious normally...
        hot_weather_price_sensitivity=0.010,  # ...but on hot days, people PAY for cold drinks!
        weather_exposure=1.8,  # Highly amplified - best on hot days, worst on bad weather
        permit_cost=250,  # $2.50 - cheap community location
    ),
}


@dataclass(kw_only=True)
class UpgradeInfo:
    """Information about a stand upgrade."""
    id: StandUpgrade
    name: str
    description: str
    cost: int  # cents
    effect: str  # Human-readable effect description


# The catalog of all available upgrades
UPGRADE_CATALOG: Dict[StandUpgrade, UpgradeInfo] = {
    StandUpgrade.COOLER: UpgradeInfo(
        id=StandUpgrade.COOLER,
        name="Ice Cooler",
        description="A portable cooler to keep your ice from melting overnight.",
        cost=250,  # $2.50
        effect="Ice melts 50% per day instead of 100%",
    ),
}


@dataclass(kw_only=True)
class BulkTier:
    """A bulk purchasing tier with quantity and discount."""
    name: str  # Display name (e.g., "Dozen", "Crate")
    quantity: int  # Number of units in this tier
    discount_percent: float  # Discount as decimal (0.10 = 10% off)


@dataclass(kw_only=True)
class BulkPricing:
    """Bulk pricing tiers for a supply type."""
    unit_name: str  # Name for single unit (e.g., "Lemon", "Bag")
    unit_name_plural: str  # Plural name (e.g., "Lemons", "Bags")
    base_price: int  # Price per unit in cents (before discounts)
    tiers: List[BulkTier]  # Available bulk tiers (ordered small to large)


# Bulk pricing configuration for all supplies
BULK_PRICING: Dict[str, BulkPricing] = {
    "lemons": BulkPricing(
        unit_name="Lemon",
        unit_name_plural="Lemons",
        base_price=25,  # $0.25 per lemon
        tiers=[
            BulkTier(name="Single", quantity=1, discount_percent=0.0),
            BulkTier(name="Dozen", quantity=12, discount_percent=0.10),  # 10% off
            BulkTier(name="Crate", quantity=144, discount_percent=0.20),  # 20% off
        ],
    ),
    "sugar": BulkPricing(
        unit_name="Bag",
        unit_name_plural="Bags",
        base_price=50,  # $0.50 per bag
        tiers=[
            BulkTier(name="Single", quantity=1, discount_percent=0.0),
            BulkTier(name="Case", quantity=5, discount_percent=0.10),  # 10% off
            BulkTier(name="Pallet", quantity=20, discount_percent=0.20),  # 20% off
        ],
    ),
    "cups": BulkPricing(
        unit_name="Pack",
        unit_name_plural="Packs",
        base_price=50,  # $0.50 per 10-pack (5 cents per cup)
        tiers=[
            BulkTier(name="Pack", quantity=10, discount_percent=0.0),  # 10 cups
            BulkTier(name="Sleeve", quantity=50, discount_percent=0.10),  # 50 cups, 10% off
            BulkTier(name="Case", quantity=250, discount_percent=0.20),  # 250 cups, 20% off
        ],
    ),
    "ice": BulkPricing(
        unit_name="Bag",
        unit_name_plural="Bags",
        base_price=25,  # $0.25 per bag
        tiers=[
            BulkTier(name="Single", quantity=1, discount_percent=0.0),
            BulkTier(name="Cooler", quantity=5, discount_percent=0.10),  # 10% off
            BulkTier(name="Delivery", quantity=20, discount_percent=0.20),  # 20% off
        ],
    ),
}


def calculate_bulk_cost(supply_type: str, quantity: int) -> int:
    """
    Calculate the cost for purchasing a quantity of supplies with bulk discounts.
    
    Applies the best available discount based on quantity thresholds.
    For example, buying 15 lemons would get the dozen rate (10% off) for all 15.
    
    Args:
        supply_type: One of "lemons", "sugar", "cups", "ice"
        quantity: Number of units to purchase
        
    Returns:
        Total cost in cents
    """
    if quantity <= 0:
        return 0
    
    pricing = BULK_PRICING.get(supply_type)
    if not pricing:
        return 0
    
    # Find the best discount tier that applies to this quantity
    # Tiers are ordered small to large, so we find the largest tier <= quantity
    best_discount = 0.0
    for tier in pricing.tiers:
        if quantity >= tier.quantity:
            best_discount = tier.discount_percent
    
    # Calculate discounted price
    base_total = quantity * pricing.base_price
    discount_amount = int(base_total * best_discount)
    return base_total - discount_amount


@dataclass(kw_only=True)
class LemonadeAction(Action):
    """
    Action for the Lemonade Stand environment.
    
    The agent decides daily operations:
    - price_per_cup: How much to charge (in cents, e.g., 50 = $0.50)
    - buy_lemons: How many lemons to purchase (optional) - PERISHABLE: expires after 3 days
    - buy_sugar: How many bags of sugar to purchase (optional) - Non-perishable
    - buy_cups: How many disposable cups to purchase (optional) - Non-perishable
    - buy_ice: How many bags of ice to purchase (optional) - PERISHABLE: melts based on cooler
    - advertising_spend: How much to spend on signs/advertising (optional)
    - buy_upgrade: Stand upgrade to purchase (optional)
    - location: Where to set up the stand (optional) - costs $10 permit fee to switch
    
    Note: Cups are made on-demand based on customer traffic. You will automatically
    serve as many customers as your supplies allow.
    """
    price_per_cup: int  # cents
    buy_lemons: int = 0
    buy_sugar: int = 0
    buy_cups: int = 0
    buy_ice: int = 0
    advertising_spend: int = 0  # cents
    buy_upgrade: Optional[str] = None  # StandUpgrade value to purchase
    location: Optional[str] = None  # Location value to set up at (None = stay at current location)


@dataclass(kw_only=True)
class LemonadeObservation(Observation):
    """
    Observation from the Lemonade Stand environment.
    
    Contains the current game state after an action is taken.
    """
    # Current day info
    day: int
    weather: str
    temperature: int  # Fahrenheit
    weather_forecast: str  # Tomorrow's predicted weather
    
    # Financial state
    cash: int  # cents
    daily_revenue: int  # cents from today's sales
    daily_costs: int  # cents spent on supplies/advertising
    daily_profit: int  # revenue - costs
    
    # Sales metrics
    cups_sold: int
    cups_wasted: int  # unsold lemonade
    customers_served: int
    customers_turned_away: int  # wanted to buy but couldn't
    
    # Inventory (current stock)
    lemons: int
    sugar_bags: float  # Fractional - tracks partial bags
    cups_available: int
    ice_bags: int = 0
    
    # Expiration tracking (items that will expire tomorrow if not used)
    lemons_expiring_tomorrow: int = 0  # Lemons that will expire after next day
    ice_expiring_tomorrow: int = 0  # Ice expires daily (all ice expires tomorrow)
    
    # Spoilage from previous day
    lemons_spoiled: int = 0  # Lemons that expired and were thrown away
    ice_melted: int = 0  # Ice that melted and was lost
    
    # Resource consumption (for lemonade production)
    ice_used: int = 0  # Ice bags consumed for making lemonade
    
    # Market info
    customer_satisfaction: float  # 0.0 to 1.0
    reputation: float  # 0.0 to 1.0, affects customer flow
    
    # Game state
    days_remaining: int
    total_profit: int  # cumulative profit over all days
    
    # Stand upgrades
    owned_upgrades: List[str] = field(default_factory=list)  # List of StandUpgrade values owned
    upgrade_catalog: Optional[List[Dict[str, Any]]] = None  # Available upgrades for purchase
    
    # Location
    current_location: str = "park"  # Current Location value
    location_catalog: Optional[List[Dict[str, Any]]] = None  # Available locations with their properties
    
    # Market intelligence (helps players understand the game mechanics)
    market_hints: Optional[Dict[str, Any]] = None
    
    # Error feedback (when action validation fails)
    action_errors: List[str] = field(default_factory=list)  # List of validation error messages
    is_error_response: bool = False  # True if this is an error response (day not advanced)


@dataclass(kw_only=True)
class MarketHints:
    """
    Market intelligence to help players make informed decisions.
    
    Uses a two-stage model:
    1. Foot traffic: People who stop by (affected by location, weather, reputation, ads)
    2. Conversion rate: % who buy at each price point (affected by price, ice on hot days)
    
    Expected sales = foot_traffic × conversion_rate
    """
    # Foot traffic forecast (people who stop by the stand)
    foot_traffic_low: int  # Conservative estimate (with -10% randomness)
    foot_traffic_high: int  # Optimistic estimate (with +10% randomness)
    weather_traffic_multiplier: float  # How weather affects foot traffic (0.1-1.8)
    
    # Conversion rates (% who buy at each price point)
    conversion_curve: Dict[int, float]  # Map of price -> conversion rate (0.0-1.0)
    ice_conversion_bonus: float  # Bonus conversion % on hot days when you have ice
    
    # Price guidance
    optimal_price: int  # Price that maximizes demand ($0.50 = 50 cents)
    price_demand_curve: Dict[int, int]  # Map of price -> expected customers
    revenue_curve: Dict[int, int]  # Map of price -> expected revenue (price * customers)
    optimal_revenue_price: int  # Price that maximizes revenue (often higher than optimal_price)
    
    # Inventory insights
    max_cups_producible: int  # Max cups you can make with current inventory
    limiting_resource: str  # Which resource limits production ("lemons", "sugar", "cups")
    ingredient_cost_per_cup: int  # Cost to make one cup (in cents)
    
    # Strategy hints
    break_even_price: int  # Minimum price to cover ingredient costs
    suggested_production: int  # Recommended cups to make


@dataclass(kw_only=True)
class GameConfig:
    """Configuration for a lemonade stand game session."""
    total_days: int = 14  # Two weeks of summer
    starting_cash: int = 2000  # $20.00 in cents
    starting_lemons: int = 10
    starting_sugar: int = 5
    starting_cups: int = 50
    starting_ice: int = 5  # Starting ice bags (usable on Day 1, melts overnight after)
    starting_location: str = "park"  # Default starting location
    
    # Supply costs (in cents)
    lemon_cost: int = 25  # per lemon
    sugar_cost: int = 50  # per bag
    cup_cost: int = 5  # per cup
    ice_cost: int = 25  # per bag of ice
    
    # Location switching cost (in cents) - charged when changing locations
    location_permit_cost: int = 1000  # $10.00
    
    # Recipe (per cup of lemonade)
    lemons_per_cup: float = 0.25  # 4 cups per lemon
    sugar_per_cup: float = 0.1  # 10 cups per bag
    ice_per_cup: float = 0.2  # 5 cups per bag of ice
    
    # Expiration settings
    lemon_shelf_life: int = 3  # Days before lemons go bad
    ice_shelf_life: int = 1  # Ice melts after 1 day (must buy fresh daily)
    
    # Customer behavior
    base_customers: int = 50  # Average customers per day
    price_sensitivity: float = 0.02  # How much demand drops per cent increase
    max_price_tolerance: int = 200  # $2.00 - above this, almost no sales
    optimal_price: int = 50  # The "sweet spot" price in cents ($0.50)
    
    # Ice bonus - customers prefer iced lemonade on hot days
    ice_demand_bonus: float = 0.2  # 20% more demand when ice is available on hot days
