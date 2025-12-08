import { BarChart3, Citrus } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';

interface EmptyChartStateProps {
  title?: string;
  message?: string;
}

export function EmptyChartState({ 
  title = 'No Data Yet',
  message = 'Run benchmarks to see results here.'
}: EmptyChartStateProps) {
  return (
    <Card variant="retro" className="h-[300px]">
      <CardContent className="h-full flex flex-col items-center justify-center text-center p-8">
        <div className="relative mb-4">
          <div className="bg-gradient-to-b from-[#FFF9C4] to-[#FFECB3] p-4 rounded-2xl border-4 border-[#FFA000] shadow-[4px_4px_0_#FF8F00]">
            <BarChart3 className="h-10 w-10 text-[#8B4513]" />
          </div>
          <Citrus className="absolute -bottom-1 -right-1 h-6 w-6 text-[#FF6B35]" />
        </div>
        <h3 className="font-display text-lg text-[#5D4037] mb-2">{title}</h3>
        <p className="text-sm text-[#5D4037]/70 max-w-xs">{message}</p>
      </CardContent>
    </Card>
  );
}

