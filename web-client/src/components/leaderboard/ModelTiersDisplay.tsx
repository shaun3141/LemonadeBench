import { Bot } from 'lucide-react';
import { MODEL_TIERS } from './constants';

interface ModelTiersDisplayProps {
  selectedTier: string | null;
  selectedModel: string | null;
  onSelectTier: (tier: string | null) => void;
  onSelectModel: (modelId: string | null) => void;
}

export function ModelTiersDisplay({
  selectedTier,
  selectedModel,
  onSelectTier,
  onSelectModel,
}: ModelTiersDisplayProps) {
  // Find which tier the selected model belongs to (if any)
  const modelTierKey = selectedModel 
    ? Object.entries(MODEL_TIERS).find(([, tier]) => 
        tier.models.some(m => m.id === selectedModel)
      )?.[0] || null
    : null;
  
  // Show models for either the selected tier or the tier containing the selected model
  const expandedTierKey = selectedTier || modelTierKey;
  const expandedTier = expandedTierKey ? MODEL_TIERS[expandedTierKey as keyof typeof MODEL_TIERS] : null;

  return (
    <div className="space-y-3">
      {/* Tier badges - horizontal row */}
      <div className="flex flex-wrap gap-2">
        {Object.entries(MODEL_TIERS).map(([key, tier]) => {
          const isSelected = selectedTier === key || modelTierKey === key;
          return (
            <button
              key={key}
              onClick={() => onSelectTier(selectedTier === key ? null : key)}
              className={`px-3 py-1.5 rounded-full border-2 font-semibold text-sm transition-all ${
                isSelected
                  ? `${tier.bgColor} ${tier.borderColor} ${tier.textColor} ring-2 ring-[#FF6B35] shadow-[3px_3px_0_#5D4037]`
                  : `${tier.bgColor} ${tier.borderColor} ${tier.textColor} hover:shadow-[2px_2px_0_#5D4037]`
              }`}
            >
              {tier.name}
            </button>
          );
        })}
      </div>
      
      {/* Expanded model list - only shows when a tier is selected */}
      {expandedTier && (
        <div className="flex flex-wrap gap-2 pl-1 pt-1 border-l-4 border-[#FF6B35]/30">
          {expandedTier.models.map((model) => {
            const isModelSelected = selectedModel === model.id;
            return (
              <button
                key={model.id}
                onClick={() => onSelectModel(isModelSelected ? null : model.id)}
                className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg border-2 text-xs font-medium transition-all ${
                  isModelSelected
                    ? 'bg-[#FF6B35]/20 border-[#FF6B35] text-[#5D4037] ring-2 ring-[#FF6B35] shadow-[2px_2px_0_#5D4037]'
                    : 'bg-white border-[#D7CCC8] text-[#5D4037] hover:border-[#8B4513] hover:shadow-[2px_2px_0_#5D4037]'
                }`}
              >
                <Bot className="h-3 w-3 text-[#8B4513]" />
                {model.name}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

