import { GOAL_FRAMINGS } from './constants';

interface GoalFramingsDisplayProps {
  selectedGoal: string | null;
  onSelectGoal: (goal: string | null) => void;
}

export function GoalFramingsDisplay({
  selectedGoal,
  onSelectGoal,
}: GoalFramingsDisplayProps) {
  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {GOAL_FRAMINGS.map((framing) => {
        const Icon = framing.icon;
        const isSelected = selectedGoal === framing.id;
        return (
          <div
            key={framing.id}
            className={`${framing.bgColor} ${framing.borderColor} border-4 rounded-2xl p-4 shadow-[4px_4px_0_rgba(0,0,0,0.15)] cursor-pointer transition-all ${
              isSelected 
                ? 'ring-4 ring-[#FF6B35] shadow-[6px_6px_0_rgba(0,0,0,0.25)]' 
                : 'hover:shadow-[6px_6px_0_rgba(0,0,0,0.2)]'
            }`}
            onClick={() => onSelectGoal(isSelected ? null : framing.id)}
          >
            <div className="flex items-start gap-3">
              <div className={`p-2 rounded-xl ${framing.iconBg} border-2 ${framing.borderColor}`}>
                <Icon className={`h-4 w-4 ${framing.iconColor}`} />
              </div>
              <div>
                <h4 className="font-display text-[#5D4037]">{framing.name}</h4>
                <p className="text-sm text-[#5D4037]/70">{framing.description}</p>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

