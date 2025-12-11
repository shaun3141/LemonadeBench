import { useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { Package, MapPin, Store, DollarSign, Play, RotateCcw, AlertTriangle, Tag, Timer, BarChart3, History } from 'lucide-react';
import type { LemonadeAction, LemonadeObservation, StandUpgradeId, SupplyType, LocationId, GameHistory } from '../types';
import { CUPS_PER_LEMON, CUPS_PER_SUGAR_BAG } from '@/lib/constants';
import { formatCents } from '@/lib/format';
import {
  RecipeDialog,
  SupplyColumn,
  LocationPicker,
  UpgradeShop,
  PricingControls,
  type TierPurchases,
} from './game';
import { MarketInsights } from './MarketInsights';

interface ActionControlsProps {
  observation: LemonadeObservation;
  history: GameHistory[];
  onSubmit: (action: LemonadeAction) => void;
  onReset: () => void;
  disabled?: boolean;
  selectedPrice: number;
  onPriceChange: (price: number) => void;
}

interface SupplyPurchases {
  lemons: TierPurchases;
  sugar: TierPurchases;
  cups: TierPurchases;
  ice: TierPurchases;
}

export function ActionControls({
  observation,
  history,
  onSubmit,
  onReset,
  disabled,
  selectedPrice,
  onPriceChange,
}: ActionControlsProps) {
  // Track tier purchases for each supply type
  const [purchases, setPurchases] = useState<SupplyPurchases>({
    lemons: {},
    sugar: {},
    cups: {},
    ice: {},
  });
  const [advertising, setAdvertising] = useState(0);
  const [selectedUpgrade, setSelectedUpgrade] = useState<StandUpgradeId | null>(null);
  const [selectedLocation, setSelectedLocation] = useState<LocationId | null>(null);

  // Get bulk pricing from market hints
  const bulkPricing = observation.market_hints?.bulk_pricing;

  // Check if player has cooler
  const hasCooler = observation.owned_upgrades?.includes('cooler') ?? false;

  // Calculate total quantities and costs for each supply type
  const calculateSupplyTotals = useMemo(() => {
    if (!bulkPricing) {
      return {
        lemons: { qty: 0, cost: 0, savings: 0 },
        sugar: { qty: 0, cost: 0, savings: 0 },
        cups: { qty: 0, cost: 0, savings: 0 },
        ice: { qty: 0, cost: 0, savings: 0 },
      };
    }

    const result: Record<SupplyType, { qty: number; cost: number; savings: number }> = {
      lemons: { qty: 0, cost: 0, savings: 0 },
      sugar: { qty: 0, cost: 0, savings: 0 },
      cups: { qty: 0, cost: 0, savings: 0 },
      ice: { qty: 0, cost: 0, savings: 0 },
    };

    for (const supplyType of ['lemons', 'sugar', 'cups', 'ice'] as SupplyType[]) {
      const pricing = bulkPricing[supplyType];
      const tierPurchases = purchases[supplyType];

      let totalQty = 0;
      let totalCost = 0;
      let fullPriceCost = 0;

      pricing.tiers.forEach((tier, index) => {
        const count = tierPurchases[index] || 0;
        if (count > 0) {
          totalQty += tier.quantity * count;
          totalCost += tier.total_price * count;
          fullPriceCost += tier.quantity * pricing.base_price * count;
        }
      });

      result[supplyType] = {
        qty: totalQty,
        cost: totalCost,
        savings: fullPriceCost - totalCost,
      };
    }

    return result;
  }, [bulkPricing, purchases]);

  const buyLemons = calculateSupplyTotals.lemons.qty;
  const buySugar = calculateSupplyTotals.sugar.qty;
  const buyCups = calculateSupplyTotals.cups.qty;
  const buyIce = calculateSupplyTotals.ice.qty;

  // Get current inventory for each supply
  const currentInventory = {
    lemons: observation.lemons,
    sugar: observation.sugar_bags,
    cups: observation.cups_available,
    ice: observation.ice_bags || 0,
  };

  // Calculate max cups we can make with current + purchased inventory
  const totalLemons = observation.lemons + buyLemons;
  const totalSugar = observation.sugar_bags + buySugar;
  const totalCups = observation.cups_available + buyCups;

  // These are used for capacity calculations if needed
  const _maxCupsFromLemons = Math.floor(totalLemons * CUPS_PER_LEMON);
  const _maxCupsFromSugar = Math.floor(totalSugar * CUPS_PER_SUGAR_BAG);
  const _maxCupsFromCups = totalCups;
  void (_maxCupsFromLemons + _maxCupsFromSugar + _maxCupsFromCups); // Silence unused warnings

  // Calculate total costs
  const supplyCost =
    calculateSupplyTotals.lemons.cost +
    calculateSupplyTotals.sugar.cost +
    calculateSupplyTotals.cups.cost +
    calculateSupplyTotals.ice.cost;
  const totalSavings =
    calculateSupplyTotals.lemons.savings +
    calculateSupplyTotals.sugar.savings +
    calculateSupplyTotals.cups.savings +
    calculateSupplyTotals.ice.savings;
  const upgradeCost = selectedUpgrade
    ? observation.upgrade_catalog?.find((u) => u.id === selectedUpgrade)?.cost ?? 0
    : 0;
  // Location permit cost only charged when switching locations
  const isChangingLocation =
    selectedLocation !== null && selectedLocation !== observation.current_location;
  const locationCost = isChangingLocation
    ? observation.location_catalog?.find((l) => l.id === selectedLocation)?.permit_cost ?? 0
    : 0;
  const totalCost = supplyCost + advertising + upgradeCost + locationCost;
  const canAfford = totalCost <= observation.cash;

  // Check for expiring items
  const lemonsExpiring = observation.lemons_expiring_tomorrow || 0;
  const hasExpiringLemons = lemonsExpiring > 0;

  const handleSubmit = () => {
    onSubmit({
      price_per_cup: selectedPrice,
      buy_lemons: buyLemons,
      buy_sugar: buySugar,
      buy_cups: buyCups,
      buy_ice: buyIce,
      advertising_spend: advertising,
      buy_upgrade: selectedUpgrade ?? undefined,
      location: selectedLocation ?? undefined,
    });
    // Reset buy quantities after submit
    setPurchases({ lemons: {}, sugar: {}, cups: {}, ice: {} });
    setAdvertising(0);
    setSelectedUpgrade(null);
    setSelectedLocation(null);
  };

  const isGameOver = observation.done;

  // Update tier purchase count
  const updateTierCount = (supplyType: SupplyType, tierIndex: number, delta: number) => {
    setPurchases((prev) => ({
      ...prev,
      [supplyType]: {
        ...prev[supplyType],
        [tierIndex]: Math.max(0, (prev[supplyType][tierIndex] || 0) + delta),
      },
    }));
  };

  return (
    <div className="space-y-4">
      {/* Header with Title and Action Buttons */}
      <div className="flex items-center justify-between gap-3">
        <h2 className="font-display text-xl text-[#5D4037] flex items-center gap-2">🍋 Your Turn!</h2>
        <div className="flex gap-2">
          {isGameOver ? (
            <Button onClick={onReset} variant="retro-pink" size="retro-sm">
              <RotateCcw className="h-4 w-4 mr-1.5" />
              Play Again!
            </Button>
          ) : (
            <>
              <Button
                onClick={handleSubmit}
                disabled={disabled || !canAfford}
                variant="retro-green"
                size="retro-sm"
              >
                <Play className="h-4 w-4 mr-1.5" />
                Start Day {observation.day}!
              </Button>
              <Button onClick={onReset} variant="retro-outline" size="retro-sm">
                <RotateCcw className="h-4 w-4" />
              </Button>
            </>
          )}
        </div>
      </div>

      {/* Tabbed Decisions - Retro Style */}
      <Tabs defaultValue="supplies" className="w-full">
        <TabsList className="w-full grid grid-cols-6 bg-gradient-to-r from-[#FFFDE7] to-[#FFF9C4] border-2 border-[#8B4513] rounded-xl p-1">
          <TabsTrigger value="supplies" className="gap-1 font-display data-[state=active]:bg-[#FFE135] data-[state=active]:text-[#5D4037] rounded-lg text-[#8B4513] text-xs px-1">
            <Package className="h-3.5 w-3.5" />
            Stock
          </TabsTrigger>
          <TabsTrigger value="location" className="gap-1 font-display data-[state=active]:bg-[#FFE135] data-[state=active]:text-[#5D4037] rounded-lg text-[#8B4513] text-xs px-1">
            <MapPin className="h-3.5 w-3.5" />
            Area
          </TabsTrigger>
          <TabsTrigger value="upgrades" className="gap-1 font-display data-[state=active]:bg-[#FFE135] data-[state=active]:text-[#5D4037] rounded-lg text-[#8B4513] text-xs px-1">
            <Store className="h-3.5 w-3.5" />
            Shop
          </TabsTrigger>
          <TabsTrigger value="pricing" className="gap-1 font-display data-[state=active]:bg-[#FFE135] data-[state=active]:text-[#5D4037] rounded-lg text-[#8B4513] text-xs px-1">
            <DollarSign className="h-3.5 w-3.5" />
            Price
          </TabsTrigger>
          <TabsTrigger value="intel" className="gap-1 font-display data-[state=active]:bg-[#FFE135] data-[state=active]:text-[#5D4037] rounded-lg text-[#8B4513] text-xs px-1">
            <BarChart3 className="h-3.5 w-3.5" />
            Intel
          </TabsTrigger>
          <TabsTrigger value="history" className="gap-1 font-display data-[state=active]:bg-[#FFE135] data-[state=active]:text-[#5D4037] rounded-lg text-[#8B4513] text-xs px-1">
            <History className="h-3.5 w-3.5" />
            Log
          </TabsTrigger>
        </TabsList>

        {/* Supplies Tab */}
        <TabsContent value="supplies">
          <Card variant="retro">
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle variant="retro" className="flex items-center gap-2 text-base">
                  <Package className="h-4 w-4 text-[#FF6B35]" />
                  Bulk Supplies
                  {totalSavings > 0 && (
                    <Badge variant="retro-green">
                      <Tag className="h-3 w-3 mr-1" />
                      Save {formatCents(totalSavings)}
                    </Badge>
                  )}
                </CardTitle>
                <RecipeDialog disabled={disabled || isGameOver} />
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              {/* Expiration Warning */}
              {hasExpiringLemons && (
                <div className="flex items-start gap-2 p-2 bg-gradient-to-r from-[#FFF3E0] to-[#FFE0B2] rounded-xl text-xs border-2 border-[#FF9800]">
                  <Timer className="h-4 w-4 text-[#E65100] flex-shrink-0 mt-0.5" />
                  <div className="text-[#E65100] font-semibold">
                    <div className="flex items-center gap-1">
                      <AlertTriangle className="h-3 w-3" />
                      {lemonsExpiring} lemons expiring tomorrow - use them!
                    </div>
                  </div>
                </div>
              )}

              {/* 4-Column Supply Grid */}
              {bulkPricing && (
                <div className="grid grid-cols-4 gap-2">
                  {(['lemons', 'sugar', 'cups', 'ice'] as SupplyType[]).map((supplyType) => (
                    <SupplyColumn
                      key={supplyType}
                      supplyType={supplyType}
                      pricing={bulkPricing[supplyType]}
                      tierPurchases={purchases[supplyType]}
                      currentStock={currentInventory[supplyType]}
                      totals={calculateSupplyTotals[supplyType]}
                      hasCooler={hasCooler}
                      hasExpiringLemons={supplyType === 'lemons' && hasExpiringLemons}
                      lemonsExpiring={lemonsExpiring}
                      disabled={disabled}
                      isGameOver={isGameOver}
                      onUpdateTierCount={(tierIndex, delta) =>
                        updateTierCount(supplyType, tierIndex, delta)
                      }
                    />
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Location Tab */}
        <TabsContent value="location">
          <Card variant="retro">
            <CardHeader className="pb-2">
              <CardTitle variant="retro" className="flex items-center gap-2 text-base">
                <MapPin className="h-4 w-4 text-[#FF6B35]" />
                Location
                {isChangingLocation && (
                  <Badge variant="retro">
                    Moving: {formatCents(locationCost)} fee
                  </Badge>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <LocationPicker
                locations={observation.location_catalog || []}
                currentLocation={observation.current_location}
                selectedLocation={selectedLocation}
                onSelectLocation={setSelectedLocation}
                disabled={disabled}
                isGameOver={isGameOver}
              />
            </CardContent>
          </Card>
        </TabsContent>

        {/* Upgrades Tab */}
        <TabsContent value="upgrades">
          <Card variant="retro">
            <CardHeader className="pb-2">
              <CardTitle variant="retro" className="flex items-center gap-2 text-base">
                <Store className="h-4 w-4 text-[#FF6B35]" />
                Stand Upgrades
              </CardTitle>
            </CardHeader>
            <CardContent>
              <UpgradeShop
                upgrades={observation.upgrade_catalog || []}
                selectedUpgrade={selectedUpgrade}
                cash={observation.cash}
                onSelectUpgrade={setSelectedUpgrade}
                disabled={disabled}
                isGameOver={isGameOver}
              />
            </CardContent>
          </Card>
        </TabsContent>

        {/* Pricing Tab */}
        <TabsContent value="pricing">
          <Card variant="retro">
            <CardContent className="p-4">
              <PricingControls
                selectedPrice={selectedPrice}
                onPriceChange={onPriceChange}
                advertising={advertising}
                onAdvertisingChange={setAdvertising}
                disabled={disabled}
                isGameOver={isGameOver}
              />
            </CardContent>
          </Card>
        </TabsContent>

        {/* Market Intel Tab */}
        <TabsContent value="intel">
          <Card variant="retro">
            <CardHeader className="pb-2">
              <CardTitle variant="retro" className="flex items-center gap-2 text-base">
                <BarChart3 className="h-4 w-4 text-[#1976D2]" />
                Market Intel
              </CardTitle>
            </CardHeader>
            <CardContent>
              <MarketInsights observation={observation} selectedPrice={selectedPrice} />
            </CardContent>
          </Card>
        </TabsContent>

        {/* History Tab */}
        <TabsContent value="history">
          <Card variant="retro">
            <CardHeader className="pb-2">
              <CardTitle variant="retro" className="flex items-center gap-2 text-base">
                <History className="h-4 w-4 text-[#8B4513]" />
                Game Log
                {history.length > 0 && (
                  <Badge variant="retro" className="text-xs">
                    {history.length} days
                  </Badge>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent>
              {history.length === 0 ? (
                <p className="text-center text-[#5D4037]/70 py-6 font-display text-sm">
                  No history yet. Start playing to see your decisions!
                </p>
              ) : (
                <div className="space-y-2 max-h-[300px] overflow-y-auto">
                  {[...history].reverse().map((entry, idx) => (
                    <div
                      key={history.length - idx - 1}
                      className="p-2.5 bg-gradient-to-r from-[#FFF9C4] to-[#FFECB3] rounded-xl text-sm border-2 border-[#FFA000]"
                    >
                      <div className="flex justify-between items-center mb-1.5">
                        <span className="font-display text-[#5D4037] text-sm">Day {entry.day}</span>
                        <Badge
                          variant={
                            entry.result.observation.daily_profit >= 0
                              ? 'retro-green'
                              : 'retro-pink'
                          }
                          className="text-xs"
                        >
                          {entry.result.observation.daily_profit >= 0 ? '+' : ''}
                          {formatCents(entry.result.observation.daily_profit)}
                        </Badge>
                      </div>
                      <div className="grid grid-cols-2 gap-1.5 text-xs text-[#5D4037]/80">
                        <span>💰 {formatCents(entry.action.price_per_cup)}</span>
                        <span>🥤 {entry.result.observation.cups_sold} sold</span>
                        <span>😊 {entry.result.observation.customers_served} served</span>
                        <span>😢 {entry.result.observation.customers_turned_away} left</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {!canAfford && !isGameOver && (
        <Card variant="retro-pink" className="py-2">
          <CardContent className="p-2 text-center">
            <p className="font-display text-sm text-[#880E4F]">💸 Not enough cash!</p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
