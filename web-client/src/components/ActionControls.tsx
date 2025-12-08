import { useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { Package, MapPin, Store, DollarSign, Play, RotateCcw, AlertTriangle, Tag, Timer } from 'lucide-react';
import type { LemonadeAction, LemonadeObservation, StandUpgradeId, SupplyType, LocationId } from '../types';
import { CUPS_PER_LEMON, CUPS_PER_SUGAR_BAG } from '@/lib/constants';
import { formatCents } from '@/lib/format';
import {
  OrderSummary,
  RecipeDialog,
  SupplyColumn,
  LocationPicker,
  UpgradeShop,
  PricingControls,
  type TierPurchases,
} from './game';

interface ActionControlsProps {
  observation: LemonadeObservation;
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

  const maxCupsFromLemons = Math.floor(totalLemons * CUPS_PER_LEMON);
  const maxCupsFromSugar = Math.floor(totalSugar * CUPS_PER_SUGAR_BAG);
  const maxCupsFromCups = totalCups;
  const maxPossibleCups = Math.min(maxCupsFromLemons, maxCupsFromSugar, maxCupsFromCups);

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
      {/* Order Summary Bar */}
      <OrderSummary
        cash={observation.cash}
        maxPossibleCups={maxPossibleCups}
        hasPendingPurchases={buyLemons > 0 || buySugar > 0 || buyCups > 0}
        totalCost={totalCost}
        totalSavings={totalSavings}
        canAfford={canAfford}
      />

      {/* Tabbed Decisions - Retro Style */}
      <Tabs defaultValue="supplies" className="w-full">
        <TabsList className="w-full grid grid-cols-4 bg-gradient-to-r from-[#FFFDE7] to-[#FFF9C4] border-2 border-[#8B4513] rounded-xl p-1">
          <TabsTrigger value="supplies" className="gap-1.5 font-display data-[state=active]:bg-[#FFE135] data-[state=active]:text-[#5D4037] rounded-lg text-[#8B4513]">
            <Package className="h-4 w-4" />
            Stock
          </TabsTrigger>
          <TabsTrigger value="location" className="gap-1.5 font-display data-[state=active]:bg-[#FFE135] data-[state=active]:text-[#5D4037] rounded-lg text-[#8B4513]">
            <MapPin className="h-4 w-4" />
            Area
          </TabsTrigger>
          <TabsTrigger value="upgrades" className="gap-1.5 font-display data-[state=active]:bg-[#FFE135] data-[state=active]:text-[#5D4037] rounded-lg text-[#8B4513]">
            <Store className="h-4 w-4" />
            Shop
          </TabsTrigger>
          <TabsTrigger value="pricing" className="gap-1.5 font-display data-[state=active]:bg-[#FFE135] data-[state=active]:text-[#5D4037] rounded-lg text-[#8B4513]">
            <DollarSign className="h-4 w-4" />
            Price
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
      </Tabs>

      {/* Action Buttons - Retro Style */}
      <div className="flex gap-2">
        {isGameOver ? (
          <Button onClick={onReset} variant="retro-pink" size="retro-lg" className="flex-1">
            <RotateCcw className="h-5 w-5 mr-2" />
            Play Again!
          </Button>
        ) : (
          <>
            <Button
              onClick={handleSubmit}
              disabled={disabled || !canAfford}
              variant="retro-green"
              size="retro-lg"
              className="flex-1"
            >
              <Play className="h-5 w-5 mr-2" />
              Start Day {observation.day}!
            </Button>
            <Button onClick={onReset} variant="retro-outline" size="retro-default">
              <RotateCcw className="h-5 w-5" />
            </Button>
          </>
        )}
      </div>

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
