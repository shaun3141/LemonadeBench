import { Badge } from '@/components/ui/badge';
import {
  ChevronRight,
  ChevronDown,
  Cloud,
  Sun,
  CloudRain,
  CloudLightning,
  Thermometer,
  DollarSign,
  ShoppingCart,
  Users,
  Brain,
  Package,
  Snowflake,
  Megaphone,
  MapPin,
  TrendingUp,
  CupSoda,
} from 'lucide-react';
import type { RunTurn } from '@/types';
import { formatCents } from '@/lib/format';

// Weather icon helper
function getWeatherIcon(weather: string) {
  switch (weather?.toLowerCase()) {
    case 'hot':
      return <Thermometer className="h-4 w-4 text-red-500" />;
    case 'sunny':
      return <Sun className="h-4 w-4 text-yellow-500" />;
    case 'cloudy':
      return <Cloud className="h-4 w-4 text-gray-500" />;
    case 'rainy':
      return <CloudRain className="h-4 w-4 text-blue-500" />;
    case 'stormy':
      return <CloudLightning className="h-4 w-4 text-purple-500" />;
    default:
      return <Sun className="h-4 w-4 text-yellow-500" />;
  }
}

interface TurnCardProps {
  turn: RunTurn;
  isExpanded: boolean;
  onToggle: () => void;
}

export function TurnCard({ turn, isExpanded, onToggle }: TurnCardProps) {
  const observation = turn.observation as {
    weather?: string;
    temperature?: number;
    weather_forecast?: string;
    cash?: number;
    lemons?: number;
    sugar_bags?: number;
    cups_available?: number;
    ice_bags?: number;
    reputation?: number;
    current_location?: string;
    owned_upgrades?: string[];
    lemons_expiring_tomorrow?: number;
  };

  const action = turn.action as {
    price_per_cup?: number;
    buy_lemons?: number;
    buy_sugar?: number;
    buy_cups?: number;
    buy_ice?: number;
    advertising_spend?: number;
    buy_upgrade?: string;
    location?: string;
  };

  const result = turn.result as {
    cups_sold?: number;
    daily_revenue?: number;
    daily_profit?: number;
    customers_turned_away?: number;
  };

  return (
    <div className="border rounded-lg overflow-hidden">
      {/* Collapsed header - always visible */}
      <div
        className={`p-3 flex items-center justify-between cursor-pointer hover:bg-muted/50 transition-colors ${
          isExpanded ? 'bg-muted/30 border-b' : ''
        }`}
        onClick={onToggle}
      >
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-yellow-100 dark:bg-yellow-900/30 flex items-center justify-center font-bold text-yellow-700 dark:text-yellow-300">
            {turn.day}
          </div>
          <div>
            <div className="flex items-center gap-2">
              {getWeatherIcon(observation.weather || '')}
              <span className="font-medium capitalize">{observation.weather || 'Unknown'} Day</span>
              <span className="text-xs text-muted-foreground">{observation.temperature}°F</span>
            </div>
            <div className="text-xs text-muted-foreground flex items-center gap-2 mt-0.5">
              <span>Price: {formatCents(action.price_per_cup || 0)}</span>
              <span>•</span>
              <span>Sold: {result.cups_sold ?? 0} cups</span>
              {(result.customers_turned_away ?? 0) > 0 && (
                <>
                  <span>•</span>
                  <span className="text-red-500">{result.customers_turned_away} turned away</span>
                </>
              )}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant={(result.daily_profit ?? 0) >= 0 ? 'default' : 'destructive'}>
            {(result.daily_profit ?? 0) >= 0 ? '+' : ''}
            {formatCents(result.daily_profit ?? 0)}
          </Badge>
          {isExpanded ? (
            <ChevronDown className="h-5 w-5 text-muted-foreground" />
          ) : (
            <ChevronRight className="h-5 w-5 text-muted-foreground" />
          )}
        </div>
      </div>

      {/* Expanded details */}
      {isExpanded && (
        <div className="p-4 space-y-4 bg-muted/10">
          {/* Reasoning */}
          {turn.reasoning && (
            <div className="p-3 bg-purple-50 dark:bg-purple-950/30 rounded-lg">
              <div className="flex items-center gap-2 mb-2 text-purple-700 dark:text-purple-300">
                <Brain className="h-4 w-4" />
                <span className="font-semibold text-sm">Model Reasoning</span>
              </div>
              <p className="text-sm text-muted-foreground italic">"{turn.reasoning}"</p>
            </div>
          )}

          <div className="grid md:grid-cols-3 gap-4">
            {/* Observation - State at start of turn */}
            <div className="p-3 bg-blue-50 dark:bg-blue-950/30 rounded-lg">
              <div className="flex items-center gap-2 mb-2 text-blue-700 dark:text-blue-300">
                <Package className="h-4 w-4" />
                <span className="font-semibold text-sm">State (Start of Day)</span>
              </div>
              <div className="space-y-1.5 text-xs">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Cash:</span>
                  <span className="font-medium">{formatCents(observation.cash || 0)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Weather:</span>
                  <span className="font-medium capitalize flex items-center gap-1">
                    {getWeatherIcon(observation.weather || '')}
                    {observation.weather} ({observation.temperature}°F)
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Forecast:</span>
                  <span className="font-medium capitalize">{observation.weather_forecast}</span>
                </div>
                <div className="border-t pt-1.5 mt-1.5">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Lemons:</span>
                    <span className="font-medium">
                      {observation.lemons}
                      {(observation.lemons_expiring_tomorrow ?? 0) > 0 && (
                        <span className="text-amber-500 ml-1">
                          ({observation.lemons_expiring_tomorrow} expiring)
                        </span>
                      )}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Sugar:</span>
                    <span className="font-medium">
                      {typeof observation.sugar_bags === 'number'
                        ? observation.sugar_bags.toFixed(1)
                        : observation.sugar_bags}{' '}
                      bags
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Cups:</span>
                    <span className="font-medium">{observation.cups_available}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Ice:</span>
                    <span className="font-medium">{observation.ice_bags} bags</span>
                  </div>
                </div>
                <div className="border-t pt-1.5 mt-1.5">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Location:</span>
                    <span className="font-medium capitalize">
                      {observation.current_location || 'park'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Reputation:</span>
                    <span className="font-medium">
                      {((observation.reputation || 0) * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Action taken */}
            <div className="p-3 bg-amber-50 dark:bg-amber-950/30 rounded-lg">
              <div className="flex items-center gap-2 mb-2 text-amber-700 dark:text-amber-300">
                <ShoppingCart className="h-4 w-4" />
                <span className="font-semibold text-sm">Action Taken</span>
              </div>
              <div className="space-y-1.5 text-xs">
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground flex items-center gap-1">
                    <DollarSign className="h-3 w-3" /> Price:
                  </span>
                  <span className="font-bold text-amber-700 dark:text-amber-300">
                    {formatCents(action.price_per_cup || 0)}/cup
                  </span>
                </div>
                {(action.buy_lemons ?? 0) > 0 && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Buy Lemons:</span>
                    <span className="font-medium">{action.buy_lemons}</span>
                  </div>
                )}
                {(action.buy_sugar ?? 0) > 0 && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Buy Sugar:</span>
                    <span className="font-medium">{action.buy_sugar} bags</span>
                  </div>
                )}
                {(action.buy_cups ?? 0) > 0 && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Buy Cups:</span>
                    <span className="font-medium">{action.buy_cups}</span>
                  </div>
                )}
                {(action.buy_ice ?? 0) > 0 && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground flex items-center gap-1">
                      <Snowflake className="h-3 w-3" /> Buy Ice:
                    </span>
                    <span className="font-medium">{action.buy_ice} bags</span>
                  </div>
                )}
                {(action.advertising_spend ?? 0) > 0 && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground flex items-center gap-1">
                      <Megaphone className="h-3 w-3" /> Ads:
                    </span>
                    <span className="font-medium">
                      {formatCents(action.advertising_spend || 0)}
                    </span>
                  </div>
                )}
                {action.buy_upgrade && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Upgrade:</span>
                    <span className="font-medium capitalize">{action.buy_upgrade}</span>
                  </div>
                )}
                {action.location && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground flex items-center gap-1">
                      <MapPin className="h-3 w-3" /> Move to:
                    </span>
                    <span className="font-medium capitalize">{action.location}</span>
                  </div>
                )}
                {!action.buy_lemons &&
                  !action.buy_sugar &&
                  !action.buy_cups &&
                  !action.buy_ice &&
                  !action.advertising_spend &&
                  !action.buy_upgrade &&
                  !action.location && (
                    <div className="text-muted-foreground italic">No purchases this turn</div>
                  )}
              </div>
            </div>

            {/* Result */}
            <div className="p-3 bg-green-50 dark:bg-green-950/30 rounded-lg">
              <div className="flex items-center gap-2 mb-2 text-green-700 dark:text-green-300">
                <TrendingUp className="h-4 w-4" />
                <span className="font-semibold text-sm">Outcome</span>
              </div>
              <div className="space-y-1.5 text-xs">
                <div className="flex justify-between">
                  <span className="text-muted-foreground flex items-center gap-1">
                    <CupSoda className="h-3 w-3" /> Cups Sold:
                  </span>
                  <span className="font-medium">{result.cups_sold ?? 0}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Revenue:</span>
                  <span className="font-medium text-green-600">
                    {formatCents(result.daily_revenue || 0)}
                  </span>
                </div>
                <div className="flex justify-between border-t pt-1.5 mt-1.5">
                  <span className="text-muted-foreground font-medium">Daily Profit:</span>
                  <span
                    className={`font-bold ${(result.daily_profit ?? 0) >= 0 ? 'text-green-600' : 'text-red-500'}`}
                  >
                    {(result.daily_profit ?? 0) >= 0 ? '+' : ''}
                    {formatCents(result.daily_profit ?? 0)}
                  </span>
                </div>
                {(result.customers_turned_away ?? 0) > 0 && (
                  <div className="flex justify-between text-red-500 pt-1">
                    <span className="flex items-center gap-1">
                      <Users className="h-3 w-3" /> Turned Away:
                    </span>
                    <span className="font-medium">{result.customers_turned_away}</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

