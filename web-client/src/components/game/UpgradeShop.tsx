import { Button } from '@/components/ui/button';
import { Store, ThermometerSnowflake, Check } from 'lucide-react';
import type { StandUpgradeId, UpgradeInfo } from '@/types';
import { formatCents } from '@/lib/format';

interface UpgradeShopProps {
  upgrades: UpgradeInfo[];
  selectedUpgrade: StandUpgradeId | null;
  cash: number;
  onSelectUpgrade: (upgradeId: StandUpgradeId | null) => void;
  disabled?: boolean;
  isGameOver?: boolean;
}

export function UpgradeShop({
  upgrades,
  selectedUpgrade,
  cash,
  onSelectUpgrade,
  disabled,
  isGameOver,
}: UpgradeShopProps) {
  if (!upgrades || upgrades.length === 0) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        <Store className="h-8 w-8 mx-auto mb-2 opacity-50" />
        <p>No upgrades available yet</p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {upgrades.map((upgrade) => {
        const isOwned = upgrade.owned;
        const isSelected = selectedUpgrade === upgrade.id;
        const canBuy = !isOwned && upgrade.cost <= cash;

        return (
          <div
            key={upgrade.id}
            className={`p-3 rounded-lg border-2 transition-all ${
              isOwned
                ? 'bg-green-100 dark:bg-green-950/30 border-green-400 dark:border-green-600'
                : isSelected
                  ? 'bg-purple-100 dark:bg-purple-950/30 border-purple-400 dark:border-purple-600 ring-2 ring-purple-400'
                  : 'bg-gray-100 dark:bg-gray-800/50 border-gray-300 dark:border-gray-600 hover:border-purple-400'
            }`}
          >
            <div className="flex items-start justify-between gap-2">
              <div className="flex items-center gap-2">
                {upgrade.id === 'cooler' && (
                  <ThermometerSnowflake className="h-5 w-5 text-cyan-500" />
                )}
                <div>
                  <div className="font-bold text-sm">{upgrade.name}</div>
                  <div className="text-xs text-muted-foreground">{upgrade.description}</div>
                  <div className="text-xs text-purple-600 dark:text-purple-400 mt-1">
                    {upgrade.effect}
                  </div>
                </div>
              </div>
              <div className="flex flex-col items-end gap-1">
                {isOwned ? (
                  <span className="flex items-center gap-1 text-green-600 text-xs font-bold">
                    <Check className="h-3 w-3" /> Owned
                  </span>
                ) : (
                  <>
                    <span
                      className={`text-sm font-bold ${canBuy ? 'text-purple-600' : 'text-gray-400'}`}
                    >
                      {formatCents(upgrade.cost)}
                    </span>
                    <Button
                      variant={isSelected ? 'default' : 'outline'}
                      size="sm"
                      className="h-7 text-xs"
                      disabled={disabled || isGameOver || !canBuy}
                      onClick={() =>
                        onSelectUpgrade(isSelected ? null : (upgrade.id as StandUpgradeId))
                      }
                    >
                      {isSelected ? 'Selected' : 'Buy'}
                    </Button>
                  </>
                )}
              </div>
            </div>
          </div>
        );
      })}
      {selectedUpgrade && (
        <div className="text-xs text-purple-600 dark:text-purple-400 text-center pt-1">
          Upgrade will be purchased when you start the day
        </div>
      )}
    </div>
  );
}

