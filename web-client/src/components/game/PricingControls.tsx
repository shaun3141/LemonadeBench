import { Slider } from '@/components/ui/slider';
import { DollarSign, Megaphone, Target } from 'lucide-react';
import { formatCents } from '@/lib/format';
import {
  MIN_PRICE,
  MAX_PRICE,
  PRICE_STEP,
  MIN_ADVERTISING,
  MAX_ADVERTISING,
  ADVERTISING_STEP,
} from '@/lib/constants';

interface PricingControlsProps {
  selectedPrice: number;
  onPriceChange: (price: number) => void;
  advertising: number;
  onAdvertisingChange: (amount: number) => void;
  disabled?: boolean;
  isGameOver?: boolean;
}

export function PricingControls({
  selectedPrice,
  onPriceChange,
  advertising,
  onAdvertisingChange,
  disabled,
  isGameOver,
}: PricingControlsProps) {
  return (
    <div className="space-y-4">
      {/* Pricing */}
      <div className="space-y-3">
        <div className="flex items-center gap-1.5">
          <DollarSign className="h-4 w-4" />
          <span className="text-sm font-medium">Price per Cup</span>
        </div>
        <div className="text-center">
          <span className="text-3xl font-bold">{formatCents(selectedPrice)}</span>
        </div>
        <Slider
          value={[selectedPrice]}
          onValueChange={([v]) => onPriceChange(v)}
          min={MIN_PRICE}
          max={MAX_PRICE}
          step={PRICE_STEP}
          disabled={disabled || isGameOver}
          className="w-full"
        />
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>{formatCents(MIN_PRICE)}</span>
          <span className="text-green-600 flex items-center gap-1">
            <Target className="h-3 w-3" />
            $0.50
          </span>
          <span>{formatCents(MAX_PRICE)}</span>
        </div>
      </div>

      <div className="border-t pt-4">
        {/* Advertising */}
        <div className="space-y-3">
          <div className="flex items-center gap-1.5">
            <Megaphone className="h-4 w-4" />
            <span className="text-sm font-medium">Advertising</span>
          </div>
          <div className="text-center">
            <span className="text-3xl font-bold">{formatCents(advertising)}</span>
          </div>
          <Slider
            value={[advertising]}
            onValueChange={([v]) => onAdvertisingChange(v)}
            min={MIN_ADVERTISING}
            max={MAX_ADVERTISING}
            step={ADVERTISING_STEP}
            disabled={disabled || isGameOver}
            className="w-full"
          />
          <p className="text-xs text-muted-foreground text-center">
            Boost daily demand with advertising
          </p>
        </div>
      </div>
    </div>
  );
}
