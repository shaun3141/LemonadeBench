import { TreePine, Building2, ShoppingBag, Waves, Users, CloudSun, DollarSign, Ticket } from 'lucide-react';
import type { LocationInfo } from '../types';

/**
 * Icon components for each location
 */
export const LOCATION_ICONS: Record<string, typeof TreePine> = {
  park: TreePine,
  downtown: Building2,
  mall: ShoppingBag,
  pool: Waves,
};

/**
 * Stat icons for location descriptions
 */
export const LOCATION_STAT_ICONS = {
  traffic: Users,
  weather: CloudSun,
  price: DollarSign,
  permit: Ticket,
};

/**
 * Color classes for each location
 */
export const LOCATION_COLORS: Record<string, string> = {
  park: 'text-green-600',
  downtown: 'text-blue-600',
  mall: 'text-purple-600',
  pool: 'text-cyan-600',
};

/**
 * Background color classes for each location
 */
export const LOCATION_BG_COLORS: Record<string, string> = {
  park: 'bg-green-100 dark:bg-green-900/30',
  downtown: 'bg-blue-100 dark:bg-blue-900/30',
  mall: 'bg-purple-100 dark:bg-purple-900/30',
  pool: 'bg-cyan-100 dark:bg-cyan-900/30',
};

/**
 * Get the icon component for a location
 */
export function getLocationIcon(locationId: string) {
  return LOCATION_ICONS[locationId] || TreePine;
}

/**
 * Get location color class
 */
export function getLocationColor(locationId: string): string {
  return LOCATION_COLORS[locationId] || 'text-gray-600';
}

/**
 * Calculate location curves for market analysis
 */
export function calculateLocationCurves(
  location: LocationInfo,
  baseWeatherMultiplier: number,
  weather: string,
  baseCustomers: number,
  reputation: number,
  hasIce: boolean
) {
  // Adjust weather multiplier for location's exposure
  const weatherMultiplier = 1.0 + (baseWeatherMultiplier - 1.0) * location.weather_exposure;

  // Determine price sensitivity (use hot weather sensitivity if applicable)
  const isHotWeather = weather === 'hot' || weather === 'sunny';
  const priceSensitivity =
    isHotWeather && location.hot_weather_price_sensitivity != null
      ? location.hot_weather_price_sensitivity
      : location.price_sensitivity;

  // Stage 1: Foot traffic (people who stop by) - NOT affected by price or ice
  const footTraffic = Math.floor(
    baseCustomers * location.foot_traffic_multiplier * weatherMultiplier * (0.5 + reputation)
  );

  // Stage 2: Conversion curve (% who buy at each price point)
  const pricePoints = [50, 75, 100, 125, 150, 175, 200];
  const conversionCurve: Record<number, number> = {};
  const priceDemandCurve: Record<number, number> = {};
  const revenueCurve: Record<number, number> = {};

  // Ice bonus/penalty on conversion (only on hot days)
  const iceConversionBonus = isHotWeather ? 0.2 : 0;

  for (const price of pricePoints) {
    // Base conversion at optimal price ($0.50) is 95%
    let conversion = 0.95;

    if (price > 50) {
      const priceDelta = (price - 50) / 100;
      const priceFactor = 1.0 - Math.pow(priceDelta, 0.7) * priceSensitivity * 50;
      conversion *= Math.max(0.1, priceFactor);
    }

    // Very high prices kill conversion
    if (price > 200) {
      conversion *= 0.05;
    }

    // Clamp to valid range
    conversion = Math.max(0, Math.min(1, conversion));
    conversionCurve[price] = conversion;

    // Apply ice modifier for demand calculation
    let adjustedConversion = conversion;
    if (isHotWeather) {
      if (hasIce) {
        adjustedConversion = Math.min(1.0, conversion * (1 + iceConversionBonus));
      } else {
        adjustedConversion = conversion * 0.8;
      }
    }

    const demand = Math.floor(footTraffic * adjustedConversion);
    priceDemandCurve[price] = demand;
    revenueCurve[price] = demand * price;
  }

  // Find optimal prices
  const optimalRevenuePrice = Object.entries(revenueCurve).reduce(
    (best, [price, rev]) => (rev > best.rev ? { price: Number(price), rev } : best),
    { price: 50, rev: 0 }
  ).price;

  return {
    weatherMultiplier,
    footTraffic,
    conversionCurve,
    iceConversionBonus: isHotWeather ? iceConversionBonus : 0,
    priceDemandCurve,
    revenueCurve,
    optimalRevenuePrice,
    usingHotWeatherSensitivity: isHotWeather && location.hot_weather_price_sensitivity != null,
    priceSensitivity,
  };
}

/**
 * Generate location description blurb based on stats
 */
export function getLocationBlurb(location: LocationInfo) {
  const trafficDesc =
    location.foot_traffic_multiplier > 1
      ? `${Math.round((location.foot_traffic_multiplier - 1) * 100)}% more foot traffic`
      : location.foot_traffic_multiplier < 1
        ? `${Math.round((1 - location.foot_traffic_multiplier) * 100)}% less foot traffic`
        : 'Standard foot traffic';

  const weatherDesc =
    location.weather_exposure === 0
      ? 'Indoor — weather has no effect'
      : location.weather_exposure < 1
        ? `Partially sheltered (${Math.round((1 - location.weather_exposure) * 100)}% weather protection)`
        : location.weather_exposure > 1
          ? `Weather amplified ${location.weather_exposure}x — great on hot days, risky on bad days`
          : 'Full weather exposure';

  // Show premium pricing capability
  let priceDesc: string;
  if (
    location.hot_weather_price_sensitivity != null &&
    location.hot_weather_price_sensitivity < location.price_sensitivity
  ) {
    priceDesc = 'Premium prices on hot days!';
  } else if (location.price_sensitivity <= 0.01) {
    priceDesc = 'Best for premium prices';
  } else if (location.price_sensitivity <= 0.014) {
    priceDesc = 'Customers accept higher prices';
  } else if (location.price_sensitivity >= 0.02) {
    priceDesc = 'Budget-conscious customers';
  } else {
    priceDesc = 'Standard price sensitivity';
  }

  const costDesc =
    location.permit_cost === 0
      ? 'Free (home base)'
      : `$${(location.permit_cost / 100).toFixed(2)} permit to move here`;

  return { trafficDesc, weatherDesc, priceDesc, costDesc };
}
