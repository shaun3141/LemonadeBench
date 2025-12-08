import { Card, CardContent } from '@/components/ui/card';
import { formatCents } from '@/lib/format';

interface OrderSummaryProps {
  cash: number;
  maxPossibleCups: number;
  hasPendingPurchases: boolean;
  totalCost: number;
  totalSavings: number;
  canAfford: boolean;
}

export function OrderSummary({
  cash,
  maxPossibleCups,
  hasPendingPurchases,
  totalCost,
  totalSavings,
  canAfford,
}: OrderSummaryProps) {
  return (
    <Card className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-950/30 dark:to-emerald-950/30 border-green-200 dark:border-green-800">
      <CardContent className="p-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="text-sm">
              <span className="text-muted-foreground">Cash: </span>
              <span className="font-bold text-green-600">{formatCents(cash)}</span>
            </div>
            <div className="text-sm">
              <span className="text-muted-foreground">Max Cups: </span>
              <span className="font-bold">{maxPossibleCups}</span>
              {hasPendingPurchases && (
                <span className="text-green-600 text-xs ml-1">(after purchase)</span>
              )}
            </div>
          </div>
          {totalCost > 0 && (
            <div className="text-sm">
              <span className="text-muted-foreground">Order: </span>
              <span className={`font-bold ${canAfford ? 'text-blue-600' : 'text-red-500'}`}>
                {formatCents(totalCost)}
              </span>
              {totalSavings > 0 && (
                <span className="text-green-600 text-xs ml-1">
                  (saved {formatCents(totalSavings)})
                </span>
              )}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

