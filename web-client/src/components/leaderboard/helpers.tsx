import { Badge } from '@/components/ui/badge';
import { Brain } from 'lucide-react';
import type { GoalFraming, Architecture, Scaffolding } from '@/types';
import { GOAL_FRAMINGS, ARCHITECTURES, SCAFFOLDINGS, MODEL_TIERS } from './constants';

export function getProviderBadgeColor(provider: string): string {
  switch (provider.toLowerCase()) {
    case 'anthropic':
      return 'bg-gradient-to-b from-[#FFE0B2] to-[#FFCC80] text-[#E65100] border-2 border-[#E65100]';
    case 'openai':
      return 'bg-gradient-to-b from-[#E8F5E9] to-[#C8E6C9] text-[#2E7D32] border-2 border-[#2E7D32]';
    case 'google':
      return 'bg-gradient-to-b from-[#E3F2FD] to-[#BBDEFB] text-[#1565C0] border-2 border-[#1565C0]';
    case 'deepseek':
      return 'bg-gradient-to-b from-[#E0F7FA] to-[#B2EBF2] text-[#00838F] border-2 border-[#00838F]';
    case 'meta':
      return 'bg-gradient-to-b from-[#E8EAF6] to-[#C5CAE9] text-[#283593] border-2 border-[#283593]';
    case 'mistral':
      return 'bg-gradient-to-b from-[#EDE7F6] to-[#D1C4E9] text-[#4527A0] border-2 border-[#4527A0]';
    case 'qwen':
      return 'bg-gradient-to-b from-[#FCE4EC] to-[#F8BBD9] text-[#AD1457] border-2 border-[#AD1457]';
    case 'xai':
      return 'bg-gradient-to-b from-[#ECEFF1] to-[#CFD8DC] text-[#37474F] border-2 border-[#37474F]';
    default:
      return 'bg-gradient-to-b from-[#FFFDE7] to-[#FFF9C4] text-[#5D4037] border-2 border-[#5D4037]';
  }
}

export function getRankDisplay(rank: number) {
  if (rank === 1) return { icon: '🥇', color: 'text-[#FFD700]' };
  if (rank === 2) return { icon: '🥈', color: 'text-[#C0C0C0]' };
  if (rank === 3) return { icon: '🥉', color: 'text-[#CD7F32]' };
  return { icon: `#${rank}`, color: 'text-[#8B4513]' };
}

export function getGoalFramingBadge(framing: GoalFraming) {
  const config = GOAL_FRAMINGS.find((g) => g.id === framing);
  if (!config) return null;
  const Icon = config.icon;
  return (
    <Badge variant="retro" className="text-xs gap-1">
      <Icon className="h-3 w-3" />
      {config.name}
    </Badge>
  );
}

export function getArchitectureBadge(arch: Architecture) {
  const config = ARCHITECTURES.find((a) => a.id === arch);
  if (!config) return null;
  return (
    <Badge variant="retro-blue" className="text-xs gap-1">
      <Brain className="h-3 w-3" />
      {config.name}
    </Badge>
  );
}

export function getScaffoldingBadge(scaff: Scaffolding) {
  if (scaff === 'none') return null;
  const config = SCAFFOLDINGS.find((s) => s.id === scaff);
  if (!config) return null;
  const Icon = config.icon;
  return (
    <Badge variant="retro-green" className="text-xs gap-1">
      <Icon className="h-3 w-3" />
      {config.name}
    </Badge>
  );
}

// Get all models from MODEL_TIERS as a flat array
export function getAllModels() {
  const models: Array<{ id: string; name: string; provider: string }> = [];
  Object.values(MODEL_TIERS).forEach(tier => {
    models.push(...tier.models);
  });
  return models;
}

// Check if a run's model_name matches a model ID
export function runMatchesModel(runModelName: string, modelId: string): boolean {
  const allModels = getAllModels();
  const model = allModels.find(m => m.id === modelId);
  if (!model) return false;
  
  // Check exact match with display name (case insensitive)
  if (runModelName.toLowerCase() === model.name.toLowerCase()) return true;
  
  // Check if the run model name contains the key part of the model id
  const modelIdPart = modelId.split('/').pop()?.toLowerCase() || '';
  if (runModelName.toLowerCase() === modelIdPart) return true;
  
  // Check if model name is contained in run model name (for variations)
  if (runModelName.toLowerCase().includes(model.name.toLowerCase())) return true;
  
  return false;
}

// Check if a run matches any model in a tier
export function runMatchesTier(runModelName: string, tierKey: string): boolean {
  const tier = MODEL_TIERS[tierKey as keyof typeof MODEL_TIERS];
  if (!tier) return false;
  return tier.models.some(model => runMatchesModel(runModelName, model.id));
}

// Get display name for a model ID
export function getModelDisplayName(modelId: string): string {
  const allModels = getAllModels();
  const model = allModels.find(m => m.id === modelId);
  return model?.name || modelId;
}

// Get display name for selected tier
export function getSelectedTierName(selectedTier: string | null): string | null {
  if (!selectedTier) return null;
  const tier = MODEL_TIERS[selectedTier as keyof typeof MODEL_TIERS];
  return tier?.name || selectedTier;
}

// Get display name for selected goal
export function getSelectedGoalName(selectedGoal: string | null): string | null {
  if (!selectedGoal) return null;
  const goal = GOAL_FRAMINGS.find(g => g.id === selectedGoal);
  return goal?.name || selectedGoal;
}

