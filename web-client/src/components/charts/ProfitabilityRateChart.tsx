import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { EmptyChartState } from './EmptyChartState';
import type { LeaderboardRun } from '@/types';

// Color palette matching the retro theme
const CONDITION_COLORS: Record<string, string> = {
  baseline: '#8B4513',
  aggressive: '#C62828',
  conservative: '#1976D2',
  competitive: '#7B1FA2',
  survival: '#E65100',
  growth: '#388E3C',
};

// Display names for conditions
const CONDITION_LABELS: Record<string, string> = {
  baseline: 'Baseline',
  aggressive: 'Aggressive',
  conservative: 'Conservative',
  competitive: 'Competitive',
  survival: 'Survival',
  growth: 'Growth',
};

interface ProfitabilityData {
  condition: string;
  label: string;
  profitableRuns: number;
  totalRuns: number;
  profitabilityRate: number;
  meanProfitWhenProfitable: number;
  meanLossWhenUnprofitable: number;
}

interface ProfitabilityRateChartProps {
  runs: LeaderboardRun[];
  title: string;
  loading?: boolean;
}

// Custom tooltip component
function CustomTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: ProfitabilityData }> }) {
  if (!active || !payload || !payload.length) return null;
  
  const data = payload[0].payload;
  const unprofitableRuns = data.totalRuns - data.profitableRuns;
  
  return (
    <div className="bg-white border-4 border-[#8B4513] rounded-xl p-3 shadow-[4px_4px_0_#5D4037]">
      <p className="font-display text-[#5D4037] mb-2">{data.label}</p>
      <div className="space-y-1 text-sm">
        <p className="text-[#5D4037]">
          <span className="font-semibold">Success Rate:</span>{' '}
          <span className={data.profitabilityRate >= 50 ? 'text-[#388E3C] font-bold' : 'text-[#C62828] font-bold'}>
            {data.profitabilityRate.toFixed(1)}%
          </span>
        </p>
        <p className="text-[#388E3C]">
          <span className="font-semibold">Profitable:</span> {data.profitableRuns} runs
          {data.profitableRuns > 0 && (
            <span className="text-[#5D4037]/70"> (avg +${(data.meanProfitWhenProfitable / 100).toFixed(2)})</span>
          )}
        </p>
        <p className="text-[#C62828]">
          <span className="font-semibold">Unprofitable:</span> {unprofitableRuns} runs
          {unprofitableRuns > 0 && (
            <span className="text-[#5D4037]/70"> (avg ${(data.meanLossWhenUnprofitable / 100).toFixed(2)})</span>
          )}
        </p>
      </div>
      <p className="text-xs text-[#5D4037]/50 mt-2">
        {data.totalRuns} total runs
      </p>
    </div>
  );
}

export function ProfitabilityRateChart({ 
  runs, 
  title, 
  loading = false,
}: ProfitabilityRateChartProps) {
  // Group runs by goal_framing and calculate profitability stats
  const grouped: Record<string, LeaderboardRun[]> = {};
  
  for (const run of runs) {
    const key = run.goal_framing || 'baseline';
    if (!grouped[key]) grouped[key] = [];
    grouped[key].push(run);
  }
  
  const chartData: ProfitabilityData[] = Object.entries(grouped).map(([condition, conditionRuns]) => {
    const profitableRuns = conditionRuns.filter(r => r.total_profit > 0);
    const unprofitableRuns = conditionRuns.filter(r => r.total_profit <= 0);
    
    const meanProfitWhenProfitable = profitableRuns.length > 0
      ? profitableRuns.reduce((sum, r) => sum + r.total_profit, 0) / profitableRuns.length
      : 0;
    
    const meanLossWhenUnprofitable = unprofitableRuns.length > 0
      ? unprofitableRuns.reduce((sum, r) => sum + r.total_profit, 0) / unprofitableRuns.length
      : 0;
    
    return {
      condition,
      label: CONDITION_LABELS[condition] || condition,
      profitableRuns: profitableRuns.length,
      totalRuns: conditionRuns.length,
      profitabilityRate: (profitableRuns.length / conditionRuns.length) * 100,
      meanProfitWhenProfitable,
      meanLossWhenUnprofitable,
    };
  });
  
  // Sort by profitability rate descending
  chartData.sort((a, b) => b.profitabilityRate - a.profitabilityRate);

  if (loading) {
    return (
      <Card variant="retro" className="h-[350px]">
        <CardHeader className="pb-2">
          <CardTitle variant="retro" className="text-[#5D4037]">{title}</CardTitle>
        </CardHeader>
        <CardContent className="h-[280px] flex items-center justify-center">
          <div className="animate-pulse text-[#8B4513]">Loading results...</div>
        </CardContent>
      </Card>
    );
  }

  if (!runs || runs.length === 0) {
    return <EmptyChartState title="No Results" message="Complete some benchmark runs to see profitability rates." />;
  }

  return (
    <Card variant="retro" className="h-[350px]">
      <CardHeader className="pb-2">
        <CardTitle variant="retro" className="text-[#5D4037]">{title}</CardTitle>
      </CardHeader>
      <CardContent className="h-[280px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 5, right: 30, left: 80, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#D7CCC8" />
            <XAxis 
              type="number" 
              domain={[0, 100]}
              tickFormatter={(value) => `${value}%`}
              stroke="#8B4513"
              fontSize={12}
            />
            <YAxis 
              type="category" 
              dataKey="label" 
              stroke="#8B4513"
              fontSize={12}
              width={75}
            />
            <Tooltip content={<CustomTooltip />} />
            
            {/* 50% reference line */}
            <ReferenceLine 
              x={50} 
              stroke="#8B4513" 
              strokeDasharray="4 4" 
              strokeOpacity={0.5}
              label={{ 
                value: '50%', 
                position: 'top',
                fill: '#8B4513',
                fontSize: 10,
                opacity: 0.7,
              }}
            />
            
            <Bar 
              dataKey="profitabilityRate" 
              radius={[0, 8, 8, 0]}
              stroke="#5D4037"
              strokeWidth={2}
            >
              {chartData.map((entry) => (
                <Cell 
                  key={entry.condition} 
                  fill={CONDITION_COLORS[entry.condition] || '#8B4513'} 
                  fillOpacity={entry.profitabilityRate >= 50 ? 0.9 : 0.5}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

