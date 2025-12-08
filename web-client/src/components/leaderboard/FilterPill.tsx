import { X } from 'lucide-react';

interface FilterPillProps {
  label: string;
  onClear: () => void;
  variant?: 'default' | 'goal' | 'model';
}

const bgColors = {
  default: 'bg-gradient-to-b from-[#FFF9C4] to-[#FFECB3] border-[#FFA000]',
  goal: 'bg-gradient-to-b from-[#E3F2FD] to-[#BBDEFB] border-[#1976D2]',
  model: 'bg-gradient-to-b from-[#E8F5E9] to-[#C8E6C9] border-[#388E3C]',
};

export function FilterPill({ label, onClear, variant = 'default' }: FilterPillProps) {
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold border-2 ${bgColors[variant]} text-[#5D4037]`}>
      {label}
      <button 
        onClick={(e) => { e.stopPropagation(); onClear(); }}
        className="hover:bg-black/10 rounded-full p-0.5 transition-colors"
      >
        <X className="h-3 w-3" />
      </button>
    </span>
  );
}

