import { Sun, Cloud, CloudRain, CloudLightning, Thermometer } from 'lucide-react';

interface WeatherIconProps {
  weather: string;
  size?: number;
  className?: string;
}

export function WeatherIcon({ weather, size = 24, className = '' }: WeatherIconProps) {
  const iconProps = { size, className };
  
  switch (weather) {
    case 'hot':
      return <Thermometer {...iconProps} className={`text-orange-500 ${className}`} />;
    case 'sunny':
      return <Sun {...iconProps} className={`text-yellow-500 ${className}`} />;
    case 'cloudy':
      return <Cloud {...iconProps} className={`text-gray-400 ${className}`} />;
    case 'rainy':
      return <CloudRain {...iconProps} className={`text-blue-400 ${className}`} />;
    case 'stormy':
      return <CloudLightning {...iconProps} className={`text-purple-500 ${className}`} />;
    default:
      return <Sun {...iconProps} />;
  }
}

// Re-export weather utilities from lib/weather.ts for convenience
export { getWeatherEmoji, getWeatherLabel } from '@/lib/weather';

