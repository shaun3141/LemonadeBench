import type { LemonadeObservation } from '../types';

interface ModelObservationProps {
  observation: LemonadeObservation;
}

export function ModelObservation({ observation }: ModelObservationProps) {
  const hints = observation.market_hints;
  
  // Format price curves for display (only key price points)
  const formatPriceCurve = (curve: Record<number, number> | undefined) => {
    if (!curve) return '{}';
    const keyPrices = [50, 75, 100, 125, 150, 175, 200];
    const entries = keyPrices
      .filter(p => curve[p] !== undefined)
      .map(p => `${p}: ${curve[p]}`);
    return `{ ${entries.join(', ')} }`;
  };
  
  // Format location catalog concisely
  const formatLocationCatalog = () => {
    if (!observation.location_catalog) return '[]';
    return JSON.stringify(observation.location_catalog.map(loc => ({
      id: loc.id,
      name: loc.name,
      foot_traffic: loc.foot_traffic_multiplier,
      price_sensitivity: loc.price_sensitivity,
      hot_weather_sensitivity: loc.hot_weather_price_sensitivity,
      weather_exposure: loc.weather_exposure,
      permit_cost: loc.permit_cost,
      is_current: loc.is_current,
    })), null, 2);
  };
  
  // Format upgrade catalog concisely
  const formatUpgradeCatalog = () => {
    if (!observation.upgrade_catalog) return '[]';
    return JSON.stringify(observation.upgrade_catalog.map(u => ({
      id: u.id,
      name: u.name,
      cost: u.cost,
      effect: u.effect,
      owned: u.owned,
    })), null, 2);
  };
  
  // Format bulk pricing tiers
  const formatBulkPricing = () => {
    if (!hints?.bulk_pricing) return '{}';
    const simplified: Record<string, { base_price: number; tiers: Array<{ name: string; qty: number; price: number; discount: string }> }> = {};
    for (const [supply, pricing] of Object.entries(hints.bulk_pricing)) {
      simplified[supply] = {
        base_price: pricing.base_price,
        tiers: pricing.tiers.map(t => ({
          name: t.name,
          qty: t.quantity,
          price: t.total_price,
          discount: t.discount_percent > 0 ? `${Math.round(t.discount_percent * 100)}% off` : 'none',
        })),
      };
    }
    return JSON.stringify(simplified, null, 2);
  };

  return (
    <div className="font-mono text-sm bg-zinc-900 text-green-400 p-4 rounded-lg overflow-auto max-h-[600px]">
      <pre className="whitespace-pre-wrap">
{`{
  // ═══════════════════════════════════════════════════════════════
  // CURRENT DAY INFO
  // ═══════════════════════════════════════════════════════════════
  "day": ${observation.day},
  "weather": "${observation.weather}",
  "temperature": ${observation.temperature},
  "weather_forecast": "${observation.weather_forecast}",
  
  // ═══════════════════════════════════════════════════════════════
  // FINANCIAL STATE (all values in cents)
  // ═══════════════════════════════════════════════════════════════
  "cash": ${observation.cash},
  "daily_revenue": ${observation.daily_revenue},
  "daily_costs": ${observation.daily_costs},
  "daily_profit": ${observation.daily_profit},
  "total_profit": ${observation.total_profit},
  
  // ═══════════════════════════════════════════════════════════════
  // YESTERDAY'S SALES METRICS
  // ═══════════════════════════════════════════════════════════════
  "cups_sold": ${observation.cups_sold},
  "cups_wasted": ${observation.cups_wasted},
  "customers_served": ${observation.customers_served},
  "customers_turned_away": ${observation.customers_turned_away},
  
  // ═══════════════════════════════════════════════════════════════
  // CURRENT INVENTORY
  // ═══════════════════════════════════════════════════════════════
  "lemons": ${observation.lemons},
  "sugar_bags": ${Number(observation.sugar_bags).toFixed(1)},
  "cups_available": ${observation.cups_available},
  "ice_bags": ${observation.ice_bags ?? 0},
  
  // ═══════════════════════════════════════════════════════════════
  // EXPIRATION & SPOILAGE
  // ═══════════════════════════════════════════════════════════════
  "lemons_expiring_tomorrow": ${observation.lemons_expiring_tomorrow ?? 0},
  "ice_expiring_tomorrow": ${observation.ice_expiring_tomorrow ?? 0},
  "lemons_spoiled": ${observation.lemons_spoiled ?? 0},
  "ice_melted": ${observation.ice_melted ?? 0},
  "ice_used": ${observation.ice_used ?? 0},  // ice consumed for making lemonade
  
  // ═══════════════════════════════════════════════════════════════
  // LOCATION & STAND
  // ═══════════════════════════════════════════════════════════════
  "current_location": "${observation.current_location}",
  "owned_upgrades": ${JSON.stringify(observation.owned_upgrades ?? [])},
  
  // ═══════════════════════════════════════════════════════════════
  // REPUTATION (affects customer traffic, 0.0 to 1.0)
  // ═══════════════════════════════════════════════════════════════
  "customer_satisfaction": ${observation.customer_satisfaction.toFixed(3)},
  "reputation": ${observation.reputation.toFixed(3)},
  
  // ═══════════════════════════════════════════════════════════════
  // GAME STATE
  // ═══════════════════════════════════════════════════════════════
  "days_remaining": ${observation.days_remaining},
  "done": ${observation.done ?? false},
  "reward": ${observation.reward?.toFixed(2) ?? 0},
  
  // ═══════════════════════════════════════════════════════════════
  // MARKET INTELLIGENCE (key data for decision-making)
  // ═══════════════════════════════════════════════════════════════
  "market_hints": {
    // Foot traffic: people who stop by (before price effects)
    "foot_traffic_low": ${hints?.foot_traffic_low ?? 0},
    "foot_traffic_high": ${hints?.foot_traffic_high ?? 0},
    "weather_traffic_multiplier": ${hints?.weather_traffic_multiplier ?? 1.0},
    "weather_label": "${hints?.weather_label ?? 'Unknown'}",
    
    // Conversion: % who buy at each price (0.0-1.0)
    // Expected sales = foot_traffic × conversion_rate
    "conversion_curve": ${formatPriceCurve(hints?.conversion_curve)},
    "ice_conversion_bonus": ${hints?.ice_conversion_bonus ?? 0},
    
    // Price strategy - price (cents) -> expected sales at that price
    "price_demand_curve": ${formatPriceCurve(hints?.price_demand_curve)},
    // Price strategy - price (cents) -> expected revenue (cents) at that price
    "revenue_curve": ${formatPriceCurve(hints?.revenue_curve)},
    "optimal_revenue_price": ${hints?.optimal_revenue_price ?? 50},
    
    // Supply capacity
    "max_cups_producible": ${hints?.max_cups_producible ?? 0},
    "max_cups_with_ice": ${hints?.max_cups_with_ice ?? 0},
    "limiting_resource": "${hints?.limiting_resource ?? 'unknown'}",
    "ingredient_cost_per_cup": ${hints?.ingredient_cost_per_cup ?? 0},
    "break_even_price": ${hints?.break_even_price ?? 0},
    
    // Ice status (ice gives +20% conversion on hot/sunny days)
    // NOTE: All ice melts overnight! Buy fresh daily or get a cooler (preserves 50%)
    "has_ice": ${hints?.has_ice ?? false},
    "ice_bonus_active": ${hints?.ice_bonus_active ?? false},
    "ice_melts_overnight": true,  // ice bags reset to 0 each day (or 50% with cooler)
    
    // Recipe reference (how many cups each supply makes)
    "cups_from_one_lemon": ${hints?.recipe_info?.cups_from_one_lemon ?? 4},
    "cups_from_one_sugar_bag": ${hints?.recipe_info?.cups_from_one_sugar_bag ?? 10},
    "cups_from_one_ice_bag": ${hints?.recipe_info?.cups_from_one_ice_bag ?? 5},
    
    // Current location stats
    "location_foot_traffic": ${hints?.location_info?.foot_traffic_multiplier ?? 1.0},
    "location_price_sensitivity": ${hints?.location_info?.price_sensitivity ?? 0.02},
    "location_weather_exposure": ${hints?.location_info?.weather_exposure ?? 1.0}
  },
  
  // ═══════════════════════════════════════════════════════════════
  // BULK PRICING (buy in bulk for discounts)
  // Prices in cents. Buy >= tier quantity to unlock discount.
  // ═══════════════════════════════════════════════════════════════
  "bulk_pricing": ${formatBulkPricing()},
  
  // ═══════════════════════════════════════════════════════════════
  // AVAILABLE LOCATIONS (can switch for permit_cost)
  // foot_traffic: multiplier on base customers
  // price_sensitivity: demand drop per cent above $0.50 (lower = tolerates higher prices)
  // weather_exposure: 0=indoor(no weather effect), 1=normal, >1=amplified
  // ═══════════════════════════════════════════════════════════════
  "location_catalog": ${formatLocationCatalog()},
  
  // ═══════════════════════════════════════════════════════════════
  // AVAILABLE UPGRADES
  // ═══════════════════════════════════════════════════════════════
  "upgrade_catalog": ${formatUpgradeCatalog()}
}`}
      </pre>
    </div>
  );
}

