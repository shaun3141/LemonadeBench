import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import {
  TrendingUp,
  TrendingDown,
  CupSoda,
  Users,
  PartyPopper,
  AlertTriangle,
  Cloud,
  Sun,
  CloudRain,
  CloudLightning,
  Thermometer,
  Sparkles,
  Snowflake,
  Megaphone,
  MapPin,
  Trash2,
  Target,
  ShoppingCart,
  Star,
  Award,
  Citrus,
  ArrowUpCircle,
  ArrowDownCircle,
  UserX,
  UserCheck,
  Coins,
} from 'lucide-react';
import type { GameHistory } from '@/types';
import { formatCents } from '@/lib/format';

interface DayEndStatsModalProps {
  dayEndStats: GameHistory | null;
  onClose: () => void;
}

// Weather icon helper
function getWeatherIcon(weather: string) {
  switch (weather) {
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

export function DayEndStatsModal({ dayEndStats, onClose }: DayEndStatsModalProps) {
  if (!dayEndStats) return null;

  const obs = dayEndStats.result.observation;
  const action = dayEndStats.action;
  const metadata = obs.metadata || {};

  // Extract typed values from metadata
  const hadIce = Boolean(metadata.had_ice);
  const hasCooler = Boolean(metadata.has_cooler);

  // Calculate ingredients used (based on cups sold)
  const lemonsUsed = Math.ceil(obs.cups_sold * 0.25); // 4 cups per lemon
  const sugarUsed = (obs.cups_sold * 0.1).toFixed(1); // 10 cups per bag
  const cupsUsed = obs.cups_sold;
  const iceUsed = obs.ice_used ?? 0; // Actual ice consumed (from server)

  // Calculate margin per cup
  const ingredientCostPerCup = 25 * 0.25 + 50 * 0.1 + 5 + (hadIce ? 25 * 0.2 : 0); // in cents
  const marginPerCup = action.price_per_cup - ingredientCostPerCup;

  // Get weather multiplier from metadata
  const weatherMultiplier = (metadata.weather_multiplier as number) || 1.0;
  const customerDemand =
    (metadata.customer_demand as number) || obs.customers_served + obs.customers_turned_away;

  // Location info (use the location from when the day was played)
  const playedLocation = dayEndStats.location;
  const locationInfo = obs.location_catalog?.find((l) => l.id === playedLocation);

  // Get the weather for this day (captured before the action was taken)
  const todayWeather = dayEndStats.weather;

  return (
    <Dialog open={dayEndStats !== null} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="sm:max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-xl">
            {obs.daily_profit >= 0 ? (
              <>
                <PartyPopper className="h-6 w-6 text-green-500" />
                Day {dayEndStats.day} Complete!
              </>
            ) : (
              <>
                <AlertTriangle className="h-6 w-6 text-amber-500" />
                Day {dayEndStats.day} Results
              </>
            )}
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-4">
          {/* Profit/Loss Hero */}
          <div
            className={`text-center p-4 rounded-lg ${
              obs.daily_profit >= 0
                ? 'bg-green-50 dark:bg-green-950/30'
                : 'bg-red-50 dark:bg-red-950/30'
            }`}
          >
            <div className="flex items-center justify-center gap-2 mb-1">
              {obs.daily_profit >= 0 ? (
                <TrendingUp className="h-5 w-5 text-green-600" />
              ) : (
                <TrendingDown className="h-5 w-5 text-red-600" />
              )}
              <span className="text-sm text-muted-foreground">Daily Profit</span>
            </div>
            <div
              className={`text-3xl font-bold ${obs.daily_profit >= 0 ? 'text-green-600' : 'text-red-600'}`}
            >
              {obs.daily_profit >= 0 ? '+' : ''}
              {formatCents(obs.daily_profit)}
            </div>
            <div className="text-xs text-muted-foreground mt-1 flex items-center justify-center gap-2">
              <span className="flex items-center gap-1">
                <ArrowUpCircle className="h-3 w-3 text-green-500" />
                {formatCents(obs.daily_revenue)}
              </span>
              −
              <span className="flex items-center gap-1">
                <ArrowDownCircle className="h-3 w-3 text-red-500" />
                {formatCents(obs.daily_costs)}
              </span>
            </div>
          </div>

          {/* Customer Traffic Analysis */}
          <div className="p-4 bg-blue-50 dark:bg-blue-950/30 rounded-lg">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Users className="h-4 w-4 text-blue-600" />
                <span className="font-semibold text-sm">Customer Traffic Analysis</span>
              </div>
              <div className="flex items-center gap-1 text-xs">
                {getWeatherIcon(todayWeather)}
                <span className="capitalize">{todayWeather}</span>
                <span className="text-muted-foreground">• {dayEndStats.temperature}°F</span>
                <span className="text-muted-foreground">
                  @ {locationInfo?.name || playedLocation}
                </span>
              </div>
            </div>

            <div className="space-y-3">
              {/* Traffic breakdown bar */}
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-muted-foreground">
                    Total Interest: {customerDemand} customers
                  </span>
                  <span className="text-muted-foreground">
                    {obs.customers_turned_away > 0 && (
                      <span className="text-red-500">
                        {obs.customers_turned_away} couldn't buy!
                      </span>
                    )}
                  </span>
                </div>
                <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden flex">
                  <div
                    className="bg-green-500 h-full"
                    style={{
                      width: `${customerDemand > 0 ? (obs.customers_served / customerDemand) * 100 : 0}%`,
                    }}
                    title={`Served: ${obs.customers_served}`}
                  />
                  <div
                    className="bg-red-400 h-full"
                    style={{
                      width: `${customerDemand > 0 ? (obs.customers_turned_away / customerDemand) * 100 : 0}%`,
                    }}
                    title={`Turned away: ${obs.customers_turned_away}`}
                  />
                </div>
                <div className="flex justify-between text-xs mt-1">
                  <span className="text-green-600 flex items-center gap-1">
                    <UserCheck className="h-3 w-3" />
                    {obs.customers_served} served
                  </span>
                  {obs.customers_turned_away > 0 && (
                    <span className="text-red-500 flex items-center gap-1">
                      <UserX className="h-3 w-3" />
                      {obs.customers_turned_away} turned away
                    </span>
                  )}
                </div>
              </div>

              {/* Traffic multipliers */}
              <div className="grid grid-cols-3 gap-2 text-xs">
                <div className="p-2 bg-white/50 dark:bg-black/20 rounded">
                  <div className="flex items-center gap-1 text-muted-foreground mb-1">
                    {getWeatherIcon(todayWeather)}
                    Weather
                  </div>
                  <div
                    className={`font-bold ${weatherMultiplier >= 1 ? 'text-green-600' : 'text-red-500'}`}
                  >
                    {weatherMultiplier >= 1 ? '+' : ''}
                    {((weatherMultiplier - 1) * 100).toFixed(0)}%
                  </div>
                </div>
                <div className="p-2 bg-white/50 dark:bg-black/20 rounded">
                  <div className="flex items-center gap-1 text-muted-foreground mb-1">
                    <MapPin className="h-3 w-3" />
                    Location
                  </div>
                  <div
                    className={`font-bold ${(locationInfo?.foot_traffic_multiplier || 1) >= 1 ? 'text-green-600' : 'text-red-500'}`}
                  >
                    {(locationInfo?.foot_traffic_multiplier || 1) >= 1 ? '+' : ''}
                    {(((locationInfo?.foot_traffic_multiplier || 1) - 1) * 100).toFixed(0)}%
                  </div>
                </div>
                <div className="p-2 bg-white/50 dark:bg-black/20 rounded">
                  <div className="flex items-center gap-1 text-muted-foreground mb-1">
                    <Megaphone className="h-3 w-3" />
                    Ads
                  </div>
                  <div className="font-bold text-purple-600">
                    {formatCents(action.advertising_spend)}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Price Sensitivity Analysis */}
          <div className="p-4 bg-purple-50 dark:bg-purple-950/30 rounded-lg">
            <div className="flex items-center gap-2 mb-3">
              <Target className="h-4 w-4 text-purple-600" />
              <span className="font-semibold text-sm">Price Analysis</span>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">Your price:</span>
                  <span className="font-bold">{formatCents(action.price_per_cup)}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">Optimal price:</span>
                  <span className="font-medium text-green-600">$0.50</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">Ingredient cost:</span>
                  <span className="font-medium">{formatCents(ingredientCostPerCup)}/cup</span>
                </div>
              </div>
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">Margin/cup:</span>
                  <span
                    className={`font-bold ${marginPerCup >= 0 ? 'text-green-600' : 'text-red-500'}`}
                  >
                    {formatCents(marginPerCup)}
                  </span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">Price sensitivity:</span>
                  <span className="font-medium">
                    {((locationInfo?.price_sensitivity || 0.02) * 100).toFixed(1)}%/¢
                  </span>
                </div>
                {action.price_per_cup > 50 && (
                  <div className="text-xs text-amber-600">
                    ⚠️ Price above optimal reduces demand
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Ingredients Used */}
          <div className="p-4 bg-amber-50 dark:bg-amber-950/30 rounded-lg">
            <div className="flex items-center gap-2 mb-3">
              <ShoppingCart className="h-4 w-4 text-amber-600" />
              <span className="font-semibold text-sm">Ingredients Used Today</span>
            </div>

            <div className="grid grid-cols-4 gap-2">
              <div className="text-center p-2 bg-yellow-100 dark:bg-yellow-900/30 rounded">
                <Citrus className="h-4 w-4 mx-auto text-yellow-600 mb-1" />
                <div className="font-bold text-sm">{lemonsUsed}</div>
                <div className="text-[10px] text-muted-foreground">Lemons</div>
              </div>
              <div className="text-center p-2 bg-pink-100 dark:bg-pink-900/30 rounded">
                <Sparkles className="h-4 w-4 mx-auto text-pink-500 mb-1" />
                <div className="font-bold text-sm">{sugarUsed}</div>
                <div className="text-[10px] text-muted-foreground">Sugar bags</div>
              </div>
              <div className="text-center p-2 bg-blue-100 dark:bg-blue-900/30 rounded">
                <CupSoda className="h-4 w-4 mx-auto text-blue-500 mb-1" />
                <div className="font-bold text-sm">{cupsUsed}</div>
                <div className="text-[10px] text-muted-foreground">Cups</div>
              </div>
              <div
                className={`text-center p-2 rounded ${hadIce ? 'bg-cyan-100 dark:bg-cyan-900/30' : 'bg-gray-100 dark:bg-gray-800/30'}`}
              >
                <Snowflake
                  className={`h-4 w-4 mx-auto mb-1 ${hadIce ? 'text-cyan-500' : 'text-gray-400'}`}
                />
                <div className="font-bold text-sm">{iceUsed || '—'}</div>
                <div className="text-[10px] text-muted-foreground">Ice bags</div>
              </div>
            </div>

            {hadIce && (todayWeather === 'hot' || todayWeather === 'sunny') && (
              <div className="mt-2 text-xs text-cyan-600 flex items-center gap-1">
                <Snowflake className="h-3 w-3" />
                Ice bonus active! +20% customer demand
              </div>
            )}
            {!hadIce && (todayWeather === 'hot' || todayWeather === 'sunny') && (
              <div className="mt-2 text-xs text-amber-600 flex items-center gap-1">
                <AlertTriangle className="h-3 w-3" />
                No ice on a {todayWeather} day! -20% demand penalty
              </div>
            )}
          </div>

          {/* Spoilage Report (only show if there was spoilage) */}
          {(obs.lemons_spoiled > 0 || obs.ice_melted > 0) && (
            <div className="p-4 bg-red-50 dark:bg-red-950/30 rounded-lg">
              <div className="flex items-center gap-2 mb-3">
                <Trash2 className="h-4 w-4 text-red-600" />
                <span className="font-semibold text-sm">Spoilage Report</span>
              </div>

              <div className="grid grid-cols-2 gap-2">
                {obs.lemons_spoiled > 0 && (
                  <div className="flex items-center gap-2 p-2 bg-red-100 dark:bg-red-900/30 rounded">
                    <Citrus className="h-4 w-4 text-red-500" />
                    <div>
                      <div className="font-bold text-sm text-red-600">
                        {obs.lemons_spoiled} expired
                      </div>
                      <div className="text-[10px] text-muted-foreground">
                        Lost ${(obs.lemons_spoiled * 0.25).toFixed(2)} value
                      </div>
                    </div>
                  </div>
                )}
                {obs.ice_melted > 0 && (
                  <div className="flex items-center gap-2 p-2 bg-cyan-100 dark:bg-cyan-900/30 rounded">
                    <Snowflake className="h-4 w-4 text-cyan-500" />
                    <div>
                      <div className="font-bold text-sm text-cyan-600">{obs.ice_melted} melted</div>
                      <div className="text-[10px] text-muted-foreground">
                        {hasCooler ? 'Cooler saved 50%' : 'Buy a cooler!'}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Reputation & Satisfaction */}
          <div className="p-4 bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-950/30 dark:to-orange-950/30 rounded-lg">
            <div className="flex items-center gap-2 mb-3">
              <Star className="h-4 w-4 text-yellow-500" />
              <span className="font-semibold text-sm">Reputation</span>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <div className="text-xs text-muted-foreground mb-1">Customer Satisfaction</div>
                <div className="flex items-center gap-2">
                  <div className="flex-1 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-yellow-400 to-green-500 transition-all"
                      style={{ width: `${obs.customer_satisfaction * 100}%` }}
                    />
                  </div>
                  <span className="text-xs font-bold">
                    {(obs.customer_satisfaction * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
              <div>
                <div className="text-xs text-muted-foreground mb-1">Stand Reputation</div>
                <div className="flex items-center gap-2">
                  <div className="flex-1 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-orange-400 to-yellow-400 transition-all"
                      style={{ width: `${obs.reputation * 100}%` }}
                    />
                  </div>
                  <span className="text-xs font-bold">{(obs.reputation * 100).toFixed(0)}%</span>
                </div>
              </div>
            </div>
            <div className="text-xs text-muted-foreground mt-2">
              Higher reputation = more customers! Builds from good service & fair prices.
            </div>
          </div>

          {/* Season Progress */}
          <div className="p-4 border-t">
            <div className="flex justify-between items-center mb-3">
              <div className="flex items-center gap-2">
                <Award className="h-4 w-4 text-amber-500" />
                <span className="text-sm font-medium">Season Progress</span>
              </div>
              <span className="text-xs text-muted-foreground">
                {obs.days_remaining > 0 ? `${obs.days_remaining} days remaining` : 'Season Complete!'}
              </span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm font-medium flex items-center gap-1.5">
                <Coins className="h-4 w-4 text-amber-500" />
                Total Profit:
              </span>
              <span
                className={`text-xl font-bold flex items-center gap-1 ${obs.total_profit >= 0 ? 'text-green-600' : 'text-red-600'}`}
              >
                {obs.total_profit >= 0 ? (
                  <TrendingUp className="h-5 w-5" />
                ) : (
                  <TrendingDown className="h-5 w-5" />
                )}
                {obs.total_profit >= 0 ? '+' : ''}
                {formatCents(obs.total_profit)}
              </span>
            </div>

            {/* Progress bar */}
            <div className="mt-2">
              <div className="flex justify-between text-xs text-muted-foreground mb-1">
                <span>Day 1</span>
                <span>Day 14</span>
              </div>
              <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-yellow-400 to-orange-500 transition-all"
                  style={{ width: `${((14 - obs.days_remaining) / 14) * 100}%` }}
                />
              </div>
            </div>
          </div>
        </div>

        <DialogFooter>
          <Button onClick={onClose} className="w-full">
            {obs.done ? 'View Final Results' : 'Continue to Next Day'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
