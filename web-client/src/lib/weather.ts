import {
  Sun,
  Cloud,
  CloudRain,
  CloudLightning,
  Thermometer,
} from 'lucide-react';

export type WeatherType = 'sunny' | 'hot' | 'cloudy' | 'rainy' | 'stormy';

/**
 * Get the Lucide icon component for a weather type
 */
export function getWeatherIconComponent(weather: string) {
  switch (weather?.toLowerCase()) {
    case 'hot':
      return Thermometer;
    case 'sunny':
      return Sun;
    case 'cloudy':
      return Cloud;
    case 'rainy':
      return CloudRain;
    case 'stormy':
      return CloudLightning;
    default:
      return Sun;
  }
}

/**
 * Get weather icon color class
 */
export function getWeatherIconColor(weather: string): string {
  switch (weather?.toLowerCase()) {
    case 'hot':
      return 'text-red-500';
    case 'sunny':
      return 'text-yellow-500';
    case 'cloudy':
      return 'text-gray-500';
    case 'rainy':
      return 'text-blue-500';
    case 'stormy':
      return 'text-purple-500';
    default:
      return 'text-yellow-500';
  }
}

/**
 * Get emoji for weather type
 */
export function getWeatherEmoji(weather: string): string {
  switch (weather?.toLowerCase()) {
    case 'hot':
      return '🔥';
    case 'sunny':
      return '☀️';
    case 'cloudy':
      return '☁️';
    case 'rainy':
      return '🌧️';
    case 'stormy':
      return '⛈️';
    default:
      return '☀️';
  }
}

/**
 * Get human-readable label for weather
 */
export function getWeatherLabel(weather: string): string {
  switch (weather?.toLowerCase()) {
    case 'hot':
      return 'Hot';
    case 'sunny':
      return 'Sunny';
    case 'cloudy':
      return 'Cloudy';
    case 'rainy':
      return 'Rainy';
    case 'stormy':
      return 'Stormy';
    default:
      return weather || 'Unknown';
  }
}

/**
 * Get base weather multiplier for demand (before location exposure)
 */
export function getBaseWeatherMultiplier(weather: string, temperature: number): number {
  const baseMultipliers: Record<string, number> = {
    hot: 1.8,
    sunny: 1.3,
    cloudy: 0.9,
    rainy: 0.4,
    stormy: 0.1,
  };

  let multiplier = baseMultipliers[weather] || 1.0;

  // Temperature bonus/penalty
  if (temperature > 85) {
    multiplier *= 1.0 + (temperature - 85) * 0.02;
  } else if (temperature < 60) {
    multiplier *= 0.5;
  }

  return multiplier;
}

/**
 * Get demand level info based on weather multiplier
 */
export function getDemandLevel(multiplier: number): {
  label: string;
  color: string;
  emoji: string;
} {
  if (multiplier >= 1.5) return { label: 'VERY HIGH', color: 'bg-green-500', emoji: '🔥' };
  if (multiplier >= 1.1) return { label: 'HIGH', color: 'bg-lime-500', emoji: '☀️' };
  if (multiplier >= 0.8) return { label: 'AVERAGE', color: 'bg-yellow-500', emoji: '☁️' };
  if (multiplier >= 0.3) return { label: 'LOW', color: 'bg-orange-500', emoji: '🌧️' };
  return { label: 'VERY LOW', color: 'bg-red-500', emoji: '⛈️' };
}

/**
 * Check if weather is considered "hot" for ice bonus purposes
 */
export function isHotWeather(weather: string): boolean {
  return weather === 'hot' || weather === 'sunny';
}

