import { useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Users, AlertCircle, ChefHat, DollarSign, Flame, MapPin, Footprints, TrendingUp, Calculator, Snowflake, BookOpen } from 'lucide-react';
import type { LemonadeObservation } from '../types';
import { formatCents } from '@/lib/format';
import { getBaseWeatherMultiplier, getDemandLevel } from '@/lib/weather';
import { calculateLocationCurves, getLocationIcon, LOCATION_COLORS } from '@/lib/locations';

interface MarketInsightsProps {
  observation: LemonadeObservation;
  selectedPrice: number;
}

export function MarketInsights({ observation, selectedPrice }: MarketInsightsProps) {
  const hints = observation.market_hints;
  const locations = observation.location_catalog || [];

  if (!hints) {
    return (
      <Card className="bg-muted/30">
        <CardContent className="p-4 text-center text-muted-foreground">
          Market data loading...
        </CardContent>
      </Card>
    );
  }

  const trafficMultiplier = hints.weather_traffic_multiplier;
  const demandLevel = getDemandLevel(trafficMultiplier);

  // Calculate base weather multiplier (before location exposure)
  const baseWeatherMultiplier = getBaseWeatherMultiplier(
    observation.weather,
    observation.temperature
  );

  // Calculate curves for all locations
  const locationCurves = useMemo(() => {
    const curves: Record<string, ReturnType<typeof calculateLocationCurves>> = {};
    for (const loc of locations) {
      curves[loc.id] = calculateLocationCurves(
        loc,
        baseWeatherMultiplier,
        observation.weather,
        50, // base_customers from config
        observation.reputation,
        (observation.ice_bags || 0) > 0
      );
    }
    return curves;
  }, [locations, baseWeatherMultiplier, observation.weather, observation.reputation, observation.ice_bags]);

  // Current location curves (for profit projection)
  const currentCurves = locationCurves[observation.current_location];
  const priceCurve = currentCurves?.priceDemandCurve || hints.price_demand_curve;
  const priceKeys = Object.keys(priceCurve)
    .map(Number)
    .sort((a, b) => a - b);

  // Find expected demand at selected price (from price_demand_curve)
  let expectedDemand = hints.foot_traffic_high;
  for (const pricePoint of priceKeys) {
    if (selectedPrice <= pricePoint) {
      expectedDemand = priceCurve[pricePoint];
      break;
    }
    expectedDemand = priceCurve[pricePoint];
  }

  // Calculate projected profit (on-demand model: we make only what we can sell)
  const expectedSales = Math.min(expectedDemand, hints.max_cups_producible);
  const expectedRevenue = expectedSales * selectedPrice;
  const ingredientCosts = expectedSales * hints.ingredient_cost_per_cup; // Only pay for cups made
  const projectedProfit = expectedRevenue - ingredientCosts;

  // Check if supplies might limit sales
  const supplyWarning =
    expectedDemand > hints.max_cups_producible
      ? `You may turn away ~${expectedDemand - hints.max_cups_producible} customers due to low supplies!`
      : null;

  // Get conversion rate at selected price point
  const conversionCurve = hints.conversion_curve || {};
  const pricePointsForConversion = Object.keys(conversionCurve)
    .map(Number)
    .sort((a, b) => a - b);
  let conversionAtSelectedPrice = 0.95; // default
  for (const pricePoint of pricePointsForConversion) {
    if (selectedPrice <= pricePoint) {
      conversionAtSelectedPrice = conversionCurve[pricePoint] || 0.95;
      break;
    }
    conversionAtSelectedPrice = conversionCurve[pricePoint] || 0.95;
  }

  // Calculate expected sales from foot traffic × conversion
  const avgFootTraffic = Math.round((hints.foot_traffic_low + hints.foot_traffic_high) / 2);

  return (
    <div className="space-y-3">
      {/* Weather & Foot Traffic Forecast */}
      <Card className="border-2 border-blue-200 dark:border-blue-800 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-950/50 dark:to-indigo-950/50">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-base">
            <Footprints className="h-4 w-4 text-blue-600" />
            Today's Market Forecast
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {/* Traffic Level Badge */}
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Foot Traffic:</span>
            <Badge className={`${demandLevel.color} text-white`}>
              {demandLevel.emoji} {demandLevel.label}
            </Badge>
          </div>

          {/* Weather Impact */}
          <div className="text-sm p-2 bg-white/50 dark:bg-black/20 rounded-lg">
            <p className="font-medium">{hints.weather_label}</p>
            <p className="text-xs text-muted-foreground mt-1">
              Weather multiplier: {trafficMultiplier}x base traffic
            </p>
          </div>

          {/* Foot Traffic Range */}
          <div className="flex justify-between items-center text-sm">
            <span className="text-muted-foreground">People stopping by:</span>
            <span className="font-bold text-blue-700 dark:text-blue-300">
              {hints.foot_traffic_low} - {hints.foot_traffic_high}
            </span>
          </div>

          {/* Conversion at selected price */}
          <div className="text-sm p-2 bg-white/50 dark:bg-black/20 rounded-lg border-l-4 border-amber-400">
            <div className="flex justify-between items-center">
              <span className="text-muted-foreground">At {formatCents(selectedPrice)}:</span>
              <span className="font-bold text-amber-700 dark:text-amber-300">
                {Math.round(conversionAtSelectedPrice * 100)}% will buy
              </span>
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              ~{avgFootTraffic} visitors × {Math.round(conversionAtSelectedPrice * 100)}% ={' '}
              <strong>~{Math.round(avgFootTraffic * conversionAtSelectedPrice)} sales</strong>
            </p>
          </div>

          {/* Ice bonus indicator */}
          {hints.ice_conversion_bonus > 0 && (
            <div className="text-xs p-2 bg-cyan-50 dark:bg-cyan-950/30 rounded-lg flex items-center gap-2">
              <Snowflake className="h-4 w-4 text-cyan-500" />
              <span className="text-cyan-700 dark:text-cyan-300">
                Ice bonus active! +{Math.round(hints.ice_conversion_bonus * 100)}% conversion on
                this hot day
              </span>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Location Price Comparison - Tabbed */}
      <Card className="border-amber-200 dark:border-amber-800">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-base">
            <MapPin className="h-4 w-4 text-amber-600" />
            Price Strategy by Location
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <Tabs defaultValue={observation.current_location} className="w-full">
            <TabsList className="w-full grid grid-cols-4 h-auto p-1">
              {locations.map((loc) => {
                const Icon = getLocationIcon(loc.id);
                const isCurrent = loc.id === observation.current_location;
                const curves = locationCurves[loc.id];
                const hasHotBonus = curves?.usingHotWeatherSensitivity;

                return (
                  <TabsTrigger
                    key={loc.id}
                    value={loc.id}
                    className="flex flex-col items-center gap-0.5 py-1.5 px-1 text-[10px] data-[state=active]:bg-amber-100 dark:data-[state=active]:bg-amber-900/50"
                  >
                    <Icon className={`h-4 w-4 ${LOCATION_COLORS[loc.id]}`} />
                    <span className="truncate">{loc.name.split(' ')[0]}</span>
                    {isCurrent && <span className="text-[8px] text-amber-600">📍</span>}
                    {hasHotBonus && <span className="text-[8px]">🔥</span>}
                  </TabsTrigger>
                );
              })}
            </TabsList>

            {locations.map((loc) => {
              const curves = locationCurves[loc.id];
              if (!curves) return null;

              const locPriceCurve = curves.priceDemandCurve;
              const locRevenueCurve = curves.revenueCurve;
              const isCurrent = loc.id === observation.current_location;

              return (
                <TabsContent key={loc.id} value={loc.id} className="mt-2">
                  {/* Hot Weather Alert for this location */}
                  {curves.usingHotWeatherSensitivity && (
                    <div className="flex items-center gap-2 p-2 mb-2 bg-orange-100 dark:bg-orange-900/30 rounded-lg text-xs">
                      <Flame className="h-4 w-4 text-orange-500" />
                      <span className="text-orange-700 dark:text-orange-300">
                        <strong>Hot day bonus!</strong> Premium prices work well here today.
                      </span>
                    </div>
                  )}

                  {/* Location stats summary - Two-stage model */}
                  <div className="grid grid-cols-2 gap-2 mb-2 text-[10px] text-muted-foreground">
                    <div className="text-center p-1.5 bg-blue-50 dark:bg-blue-950/30 rounded">
                      <div className="font-medium text-blue-700 dark:text-blue-300">
                        Foot Traffic
                      </div>
                      <div className="text-lg font-bold text-blue-600">{curves.footTraffic}</div>
                      <div className="text-[9px]">
                        ({loc.foot_traffic_multiplier >= 1 ? '+' : ''}
                        {Math.round((loc.foot_traffic_multiplier - 1) * 100)}% ×{' '}
                        {curves.weatherMultiplier.toFixed(1)}x weather)
                      </div>
                    </div>
                    <div className="text-center p-1.5 bg-amber-50 dark:bg-amber-950/30 rounded">
                      <div className="font-medium text-amber-700 dark:text-amber-300">
                        Conversion
                      </div>
                      <div className="text-lg font-bold text-amber-600">
                        {Math.round((curves.conversionCurve[selectedPrice] || 0.95) * 100)}%
                      </div>
                      <div className="text-[9px]">at {formatCents(selectedPrice)}</div>
                    </div>
                  </div>

                  {/* Price/Conversion/Sales/Revenue Table */}
                  <div className="text-xs">
                    <div className="grid grid-cols-5 gap-1 mb-1 font-medium text-muted-foreground">
                      <div>Price</div>
                      <div>Conv.</div>
                      <div>Sales</div>
                      <div>Revenue</div>
                      <div></div>
                    </div>
                    {[50, 75, 100, 125, 150, 175, 200].map((price) => {
                      const conversion = curves.conversionCurve[price] || 0;
                      const demand = locPriceCurve[price] || 0;
                      const revenue = locRevenueCurve[price] || 0;
                      const isOptimalRevenue = price === curves.optimalRevenuePrice;
                      const isOptimalDemand = price === 50; // $0.50 is always max conversion
                      const isSelected = selectedPrice >= price - 12 && selectedPrice <= price + 12;

                      return (
                        <div
                          key={price}
                          className={`grid grid-cols-5 gap-1 py-1 px-1.5 rounded ${
                            isOptimalRevenue
                              ? 'bg-emerald-100 dark:bg-emerald-900/50 font-bold text-emerald-700 dark:text-emerald-300 ring-1 ring-emerald-400'
                              : isOptimalDemand && !isOptimalRevenue
                                ? 'bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300'
                                : isSelected
                                  ? 'bg-gray-100 dark:bg-gray-800'
                                  : ''
                          }`}
                        >
                          <div>{formatCents(price)}</div>
                          <div>{Math.round(conversion * 100)}%</div>
                          <div>{demand}</div>
                          <div>{formatCents(revenue)}</div>
                          <div className="text-right">
                            {isOptimalRevenue && '💰'}
                            {isOptimalDemand && !isOptimalRevenue && '👥'}
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  {/* Strategy tip for this location */}
                  <div className="text-xs mt-2 pt-2 border-t">
                    {curves.optimalRevenuePrice > 50 ? (
                      <div className="p-2 bg-emerald-50 dark:bg-emerald-950/30 rounded-lg text-emerald-700 dark:text-emerald-300">
                        <strong>💰 Best:</strong> Charge{' '}
                        <span className="font-bold">{formatCents(curves.optimalRevenuePrice)}</span>{' '}
                        for{' '}
                        <span className="font-bold">
                          {formatCents(locRevenueCurve[curves.optimalRevenuePrice] || 0)}
                        </span>{' '}
                        revenue
                        {!isCurrent && loc.permit_cost > 0 && (
                          <span className="text-muted-foreground">
                            {' '}
                            ({formatCents(loc.permit_cost)} permit)
                          </span>
                        )}
                      </div>
                    ) : (
                      <div className="p-2 bg-blue-50 dark:bg-blue-950/30 rounded-lg text-blue-700 dark:text-blue-300">
                        <strong>👥 Best:</strong> Volume pricing at{' '}
                        <span className="font-bold">{formatCents(50)}</span> for{' '}
                        <span className="font-bold">{formatCents(locRevenueCurve[50] || 0)}</span>{' '}
                        revenue
                        {!isCurrent && loc.permit_cost > 0 && (
                          <span className="text-muted-foreground">
                            {' '}
                            ({formatCents(loc.permit_cost)} permit)
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                </TabsContent>
              );
            })}
          </Tabs>

          {/* Legend */}
          <div className="flex items-center justify-center gap-4 text-[10px] text-muted-foreground pt-1 border-t">
            <span className="flex items-center gap-0.5"><DollarSign className="h-3 w-3 text-green-500" /> Best Revenue</span>
            <span className="flex items-center gap-0.5"><Users className="h-3 w-3 text-blue-500" /> Max Customers</span>
            <span className="flex items-center gap-0.5"><MapPin className="h-3 w-3 text-amber-500" /> Current</span>
            <span className="flex items-center gap-0.5"><Flame className="h-3 w-3 text-orange-500" /> Hot Day</span>
          </div>
        </CardContent>
      </Card>

      {/* Supply Capacity */}
      <Card className="border-purple-200 dark:border-purple-800">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-base">
            <ChefHat className="h-4 w-4 text-purple-600" />
            Supply Capacity
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          <div className="text-xs text-muted-foreground bg-green-50 dark:bg-green-950/30 p-2 rounded mb-2 flex items-start gap-2">
            <TrendingUp className="h-4 w-4 text-green-500 flex-shrink-0 mt-0.5" />
            <span><strong>On-demand production:</strong> You automatically make cups as customers
            arrive (no waste!)</span>
          </div>

          <div className="flex justify-between">
            <span className="text-muted-foreground">Max you can serve:</span>
            <span className="font-bold">{hints.max_cups_producible} cups</span>
          </div>

          <div className="flex justify-between">
            <span className="text-muted-foreground">Limiting supply:</span>
            <Badge variant="outline" className="capitalize">
              {hints.limiting_resource}
            </Badge>
          </div>

          <div className="flex justify-between">
            <span className="text-muted-foreground">Cost per cup:</span>
            <span>{formatCents(hints.ingredient_cost_per_cup)}</span>
          </div>

          <div className="flex justify-between">
            <span className="text-muted-foreground">Break-even price:</span>
            <span className="text-amber-600 font-medium">{formatCents(hints.break_even_price)}</span>
          </div>

          {/* Recipe reminder */}
          <div className="text-xs text-muted-foreground bg-purple-50 dark:bg-purple-950/30 p-2 rounded mt-2 flex items-start gap-2">
            <BookOpen className="h-4 w-4 text-purple-500 flex-shrink-0 mt-0.5" />
            <span><strong>Recipe:</strong> 1 lemon → {hints.recipe_info.cups_from_one_lemon} cups | 1
            sugar bag → {hints.recipe_info.cups_from_one_sugar_bag} cups</span>
          </div>
        </CardContent>
      </Card>

      {/* Profit Projection */}
      <Card
        className={`border-2 ${projectedProfit >= 0 ? 'border-green-300 dark:border-green-700 bg-green-50/50 dark:bg-green-950/20' : 'border-red-300 dark:border-red-700 bg-red-50/50 dark:bg-red-950/20'}`}
      >
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-base">
            <Calculator className="h-4 w-4" />
            Profit Projection
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          <div className="grid grid-cols-2 gap-2">
            <div>
              <p className="text-xs text-muted-foreground">Expected Sales</p>
              <p className="font-medium">{expectedSales} cups</p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">At {formatCents(selectedPrice)}/cup</p>
              <p className="font-medium text-green-600">{formatCents(expectedRevenue)}</p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Ingredient Cost</p>
              <p className="font-medium text-red-500">-{formatCents(ingredientCosts)}</p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Est. Profit</p>
              <p
                className={`font-bold text-lg ${projectedProfit >= 0 ? 'text-green-600' : 'text-red-500'}`}
              >
                {projectedProfit >= 0 ? '+' : ''}
                {formatCents(projectedProfit)}
              </p>
            </div>
          </div>

          {supplyWarning && (
            <div className="flex items-start gap-2 p-2 bg-amber-100 dark:bg-amber-900/30 rounded text-xs text-amber-800 dark:text-amber-200">
              <AlertCircle className="h-4 w-4 flex-shrink-0 mt-0.5" />
              <span>{supplyWarning}</span>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
