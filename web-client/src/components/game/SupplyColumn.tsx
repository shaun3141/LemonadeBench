import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Plus, Minus, ShoppingCart, Coffee, Citrus, Sparkles, CupSoda, Snowflake, Timer } from 'lucide-react';
import type { SupplyType, BulkTier, BulkPricing } from '@/types';
import { formatCents } from '@/lib/format';
import { CUPS_PER_LEMON, CUPS_PER_SUGAR_BAG, CUPS_PER_ICE_BAG } from '@/lib/constants';

// Track how many of each tier the player wants to buy
export interface TierPurchases {
  [tierIndex: number]: number;
}

// Icons and colors for each supply type
export const SUPPLY_CONFIG: Record<
  SupplyType,
  {
    icon: typeof Citrus;
    bgColor: string;
    bgColorLight: string;
    borderColor: string;
    textColor: string;
    darkBg: string;
    label: string;
    unitLabel: string;
  }
> = {
  lemons: {
    icon: Citrus,
    bgColor: 'bg-yellow-100',
    bgColorLight: 'bg-yellow-50',
    borderColor: 'border-yellow-400 dark:border-yellow-600',
    textColor: 'text-yellow-600',
    darkBg: 'dark:bg-yellow-950/50',
    label: 'Lemons',
    unitLabel: 'lemons',
  },
  sugar: {
    icon: Sparkles,
    bgColor: 'bg-pink-100',
    bgColorLight: 'bg-pink-50',
    borderColor: 'border-pink-400 dark:border-pink-600',
    textColor: 'text-pink-500',
    darkBg: 'dark:bg-pink-950/50',
    label: 'Sugar',
    unitLabel: 'bags',
  },
  cups: {
    icon: CupSoda,
    bgColor: 'bg-blue-100',
    bgColorLight: 'bg-blue-50',
    borderColor: 'border-blue-400 dark:border-blue-600',
    textColor: 'text-blue-500',
    darkBg: 'dark:bg-blue-950/50',
    label: 'Cups',
    unitLabel: 'cups',
  },
  ice: {
    icon: Snowflake,
    bgColor: 'bg-cyan-100',
    bgColorLight: 'bg-cyan-50',
    borderColor: 'border-cyan-400 dark:border-cyan-600',
    textColor: 'text-cyan-500',
    darkBg: 'dark:bg-cyan-950/50',
    label: 'Ice',
    unitLabel: 'bags',
  },
};

// Calculate cups that can be made from each supply type
export function getCupsFromSupply(supplyType: SupplyType, quantity: number): number {
  switch (supplyType) {
    case 'lemons':
      return Math.floor(quantity * CUPS_PER_LEMON);
    case 'sugar':
      return Math.floor(quantity * CUPS_PER_SUGAR_BAG);
    case 'cups':
      return quantity;
    case 'ice':
      return Math.floor(quantity * CUPS_PER_ICE_BAG);
  }
}

interface SupplyColumnProps {
  supplyType: SupplyType;
  pricing: BulkPricing;
  tierPurchases: TierPurchases;
  currentStock: number;
  totals: { qty: number; cost: number; savings: number };
  hasCooler: boolean;
  hasExpiringLemons: boolean;
  lemonsExpiring: number;
  disabled?: boolean;
  isGameOver?: boolean;
  onUpdateTierCount: (tierIndex: number, delta: number) => void;
}

export function SupplyColumn({
  supplyType,
  pricing,
  tierPurchases,
  currentStock,
  totals,
  hasCooler,
  hasExpiringLemons,
  lemonsExpiring,
  disabled,
  isGameOver,
  onUpdateTierCount,
}: SupplyColumnProps) {
  const config = SUPPLY_CONFIG[supplyType];
  const Icon = config.icon;
  const totalAfterPurchase = currentStock + totals.qty;
  const cupsFromCurrent = getCupsFromSupply(supplyType, currentStock);
  const cupsAfterPurchase = getCupsFromSupply(supplyType, totalAfterPurchase);

  const renderTierRow = (tier: BulkTier, tierIndex: number) => {
    const count = tierPurchases[tierIndex] || 0;
    const hasDiscount = tier.discount_percent > 0;
    const discountPercent = Math.round(tier.discount_percent * 100);

    return (
      <div
        key={tierIndex}
        className={`relative flex items-center justify-between p-2 rounded-lg border ${count > 0 ? 'border-green-400 bg-green-50 dark:bg-green-950/30' : 'border-gray-200 dark:border-gray-700 bg-white dark:bg-zinc-800'} transition-all`}
      >
        {/* Left side: tier info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-sm">{tier.name}</span>
            {hasDiscount && (
              <Badge className="bg-green-500 text-white text-[10px] px-1.5 py-0 h-4">
                {discountPercent}% OFF
              </Badge>
            )}
          </div>
          <div className="text-xs text-muted-foreground">
            {tier.quantity} × {formatCents(tier.total_price / tier.quantity)} ={' '}
            <span className="font-medium">{formatCents(tier.total_price)}</span>
          </div>
        </div>

        {/* Right side: +/- controls */}
        <div className="flex items-center gap-1 ml-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => onUpdateTierCount(tierIndex, -1)}
            disabled={disabled || isGameOver || count === 0}
            className="h-8 w-8 p-0 rounded-md"
          >
            <Minus className="h-4 w-4" />
          </Button>
          <div className="w-10 h-8 flex items-center justify-center bg-gray-100 dark:bg-zinc-700 rounded-md font-mono font-bold text-sm">
            {count}
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => onUpdateTierCount(tierIndex, 1)}
            disabled={disabled || isGameOver}
            className="h-8 w-8 p-0 rounded-md border-green-500 bg-green-100 hover:bg-green-200 dark:bg-green-900 dark:hover:bg-green-800"
          >
            <Plus className="h-4 w-4" />
          </Button>
        </div>
      </div>
    );
  };

  return (
    <div className={`flex flex-col rounded-xl border-2 ${config.borderColor} overflow-hidden`}>
      {/* Column Header */}
      <div className={`${config.bgColor} ${config.darkBg} p-3`}>
        <div className="flex items-center justify-center gap-2 mb-2">
          <Icon className={`h-5 w-5 ${config.textColor}`} />
          <span className="font-bold uppercase tracking-wide text-sm">{config.label}</span>
          {supplyType === 'lemons' && hasExpiringLemons && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Timer className="h-3 w-3 text-red-500 cursor-help" />
                </TooltipTrigger>
                <TooltipContent>
                  <p>{lemonsExpiring} expiring tomorrow</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
        </div>

        {/* Inventory Stats */}
        <div className="space-y-1 text-center">
          {/* Current Stock */}
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">In Stock:</span>
            <span className="font-bold">
              {Math.floor(currentStock)} {config.unitLabel}
            </span>
          </div>

          {/* Buying */}
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground flex items-center gap-1">
              <ShoppingCart className="h-3 w-3" /> Buying:
            </span>
            <span
              className={`font-bold ${totals.qty > 0 ? 'text-green-600' : 'text-muted-foreground'}`}
            >
              {totals.qty > 0 ? `+${totals.qty}` : '0'}
            </span>
          </div>

          {/* Cups producible */}
          <div className="flex items-center justify-between text-sm pt-1 border-t border-current/20">
            <span className="text-muted-foreground flex items-center gap-1">
              <Coffee className="h-3 w-3" /> Makes:
            </span>
            <span className="font-bold">
              {cupsFromCurrent}
              {totals.qty > 0 && <span className="text-green-600"> → {cupsAfterPurchase}</span>}
              <span className="text-muted-foreground font-normal ml-1">cups</span>
            </span>
          </div>

          {/* Ice melting warning */}
          {supplyType === 'ice' && (
            <div className="text-[10px] text-cyan-700 dark:text-cyan-300 pt-1 border-t border-current/20">
              {hasCooler
                ? '🧊 ~50% melts overnight (cooler helps!)'
                : '🧊 All ice melts overnight!'}
            </div>
          )}
        </div>
      </div>

      {/* Tier Rows */}
      <div className={`flex-1 p-2 space-y-2 ${config.bgColorLight} dark:bg-zinc-900/50`}>
        {pricing.tiers.map((tier, index) => renderTierRow(tier, index))}
      </div>

      {/* Column Footer - Cost */}
      {totals.qty > 0 && (
        <div className={`px-3 py-2 ${config.bgColor} ${config.darkBg} border-t-2 ${config.borderColor}`}>
          <div className="flex justify-between items-center text-sm">
            <span className="font-medium">Cost:</span>
            <span className="font-bold">
              {formatCents(totals.cost)}
              {totals.savings > 0 && (
                <span className="text-green-600 text-xs ml-1">
                  (-{formatCents(totals.savings)})
                </span>
              )}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
