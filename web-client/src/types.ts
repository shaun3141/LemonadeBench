// LemonadeBench Types

// Stand upgrade types
export type StandUpgradeId = 'cooler';

// Location types
export type LocationId = 'park' | 'downtown' | 'mall' | 'pool';

export interface LocationInfo {
  id: LocationId;
  name: string;
  description: string;
  foot_traffic_multiplier: number;  // Multiplier on base customers (1.0 = normal)
  price_sensitivity: number;  // How much demand drops per cent above optimal
  hot_weather_price_sensitivity?: number | null;  // Override sensitivity on hot/sunny days (Pool special)
  weather_exposure: number;  // 0.0 = indoor, 1.0 = full weather effect, >1.0 = amplified
  permit_cost: number;  // Cost in cents to switch to this location
  is_current: boolean;  // Whether this is the current location
}

// Bulk purchasing types
export type SupplyType = 'lemons' | 'sugar' | 'cups' | 'ice';

export interface BulkTier {
  name: string;  // e.g., "Single", "Dozen", "Crate"
  quantity: number;  // Number of units in this tier
  discount_percent: number;  // Discount as decimal (0.10 = 10% off)
  total_price: number;  // Price for this tier in cents
  price_per_unit: number;  // Discounted price per unit in cents
}

export interface BulkPricing {
  unit_name: string;  // e.g., "Lemon", "Bag"
  unit_name_plural: string;  // e.g., "Lemons", "Bags"
  base_price: number;  // Base price per unit in cents
  tiers: BulkTier[];  // Available bulk tiers
}

export interface UpgradeInfo {
  id: StandUpgradeId;
  name: string;
  description: string;
  cost: number;  // cents
  effect: string;
  owned: boolean;
}

export interface MarketHints {
  // Foot traffic forecast (people who stop by the stand)
  foot_traffic_low: number;  // Conservative estimate (with -10% randomness)
  foot_traffic_high: number;  // Optimistic estimate (with +10% randomness)
  weather_traffic_multiplier: number;  // How weather affects foot traffic (0.1-1.8)
  
  // Conversion rates (% who buy at each price point)
  conversion_curve: Record<number, number>;  // price -> conversion rate (0.0-1.0)
  ice_conversion_bonus: number;  // Bonus conversion % on hot days when you have ice
  
  // Price guidance
  optimal_price: number;
  price_demand_curve: Record<number, number>;  // price -> expected customers
  revenue_curve: Record<number, number>;  // price -> expected revenue (price * customers)
  optimal_revenue_price: number;  // price point that maximizes revenue (often higher than optimal_price)
  
  // Inventory insights
  max_cups_producible: number;
  max_cups_with_ice: number;
  limiting_resource: 'lemons' | 'sugar' | 'cups' | 'ice';
  ingredient_cost_per_cup: number;
  
  // Strategy hints
  break_even_price: number;
  suggested_production: number;
  has_ice: boolean;
  ice_bonus_active: boolean;
  
  // Helper info
  weather_label: string;
  recipe_info: {
    lemons_per_cup: number;
    sugar_per_cup: number;
    ice_per_cup: number;
    cups_from_one_lemon: number;
    cups_from_one_sugar_bag: number;
    cups_from_one_ice_bag: number;
  };
  supply_costs: {
    lemon: number;
    sugar_bag: number;
    cup: number;
    ice_bag: number;
  };
  expiration_info: {
    lemons_expiring_tomorrow: number;
    ice_melt_rate: number;  // 1.0 = all melts, 0.5 = half melts (with cooler)
    has_cooler: boolean;
    lemon_shelf_life: number;
  };
  bulk_pricing: Record<SupplyType, BulkPricing>;
  location_info: {
    current_location: LocationId;
    foot_traffic_multiplier: number;
    price_sensitivity: number;
    hot_weather_price_sensitivity: number | null;  // Overrides price_sensitivity on hot/sunny days
    using_hot_weather_sensitivity: boolean;  // True if hot weather sensitivity is active today
    weather_exposure: number;
    current_permit_cost: number;
  };
}

export interface LemonadeObservation {
  day: number;
  weather: 'sunny' | 'hot' | 'cloudy' | 'rainy' | 'stormy';
  temperature: number;
  weather_forecast: string;
  cash: number;
  daily_revenue: number;
  daily_costs: number;
  daily_profit: number;
  cups_sold: number;
  cups_wasted: number;
  customers_served: number;
  customers_turned_away: number;
  // Inventory
  lemons: number;
  sugar_bags: number;
  cups_available: number;
  ice_bags: number;
  // Expiration tracking
  lemons_expiring_tomorrow: number;
  ice_expiring_tomorrow: number;
  lemons_spoiled: number;
  ice_melted: number;
  ice_used: number;  // Ice bags consumed for making lemonade
  // Other stats
  customer_satisfaction: number;
  reputation: number;
  days_remaining: number;
  total_profit: number;
  // Stand upgrades
  owned_upgrades: StandUpgradeId[];
  upgrade_catalog?: UpgradeInfo[];
  // Location
  current_location: LocationId;
  location_catalog?: LocationInfo[];
  // Game state
  done?: boolean;
  reward?: number;
  metadata?: Record<string, unknown>;
  market_hints?: MarketHints;
}

export interface LemonadeAction {
  price_per_cup: number;
  buy_lemons: number;
  buy_sugar: number;
  buy_cups: number;
  buy_ice: number;
  advertising_spend: number;
  buy_upgrade?: StandUpgradeId;  // Optional upgrade to purchase
  location?: LocationId;  // Optional location to set up at (costs permit fee to switch)
}

export interface GameState {
  observation: LemonadeObservation;
  reward: number;
  done: boolean;
  seed?: number;  // The seed used for this game (returned on reset)
}

export interface GameHistory {
  day: number;
  action: LemonadeAction;
  result: GameState;
  // Weather conditions when the day was played
  weather: 'sunny' | 'hot' | 'cloudy' | 'rainy' | 'stormy';
  temperature: number;
  location: LocationId;
}

// Experimental factor types
export type GoalFraming = 'baseline' | 'aggressive' | 'conservative' | 'competitive' | 'survival' | 'growth';
export type Architecture = 'react' | 'plan_act' | 'act_reflect' | 'full';
export type Scaffolding = 'none' | 'calculator' | 'math_prompt' | 'code_interpreter';

// Leaderboard types (from Supabase)
export interface LeaderboardRun {
  run_id: string;
  model_id: string;
  model_name: string;
  provider: string;
  seed: number | null;
  // Experimental factors
  goal_framing: GoalFraming;
  architecture: Architecture;
  scaffolding: Scaffolding;
  // Performance metrics
  total_profit: number;
  total_cups_sold: number;
  final_cash: number;
  final_reputation: number;
  turn_count: number;
  error_count?: number;
  started_at: string;
  completed_at: string | null;
}

export interface RunTurn {
  id: string;
  run_id: string;
  day: number;
  observation: Record<string, unknown>;
  action: Record<string, unknown>;
  reasoning: string | null;
  result: Record<string, unknown>;
  created_at: string;
}

