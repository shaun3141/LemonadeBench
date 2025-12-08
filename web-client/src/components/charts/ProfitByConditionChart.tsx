import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ErrorBar,
  Cell,
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { EmptyChartState } from './EmptyChartState';
import type { AggregatedResult } from '@/api';

// Color palette matching the retro theme
const CONDITION_COLORS: Record<string, string> = {
  // Goal framings
  baseline: '#8B4513',
  aggressive: '#C62828',
  conservative: '#1976D2',
  competitive: '#7B1FA2',
  survival: '#E65100',
  growth: '#388E3C',
  // Architectures
  react: '#FF6B35',
  plan_act: '#1976D2',
  act_reflect: '#7B1FA2',
  full: '#388E3C',
  // Scaffoldings
  none: '#8B4513',
  calculator: '#1976D2',
  math_prompt: '#E65100',
  code_interpreter: '#388E3C',
};

// Display names for conditions
const CONDITION_LABELS: Record<string, string> = {
  // Goal framings
  baseline: 'Baseline',
  aggressive: 'Aggressive',
  conservative: 'Conservative',
  competitive: 'Competitive',
  survival: 'Survival',
  growth: 'Growth',
  // Architectures
  react: 'ReAct',
  plan_act: 'Plan-Act',
  act_reflect: 'Act-Reflect',
  full: 'Full',
  // Scaffoldings
  none: 'None',
  calculator: 'Calculator',
  math_prompt: 'Math Prompt',
  code_interpreter: 'Code Interpreter',
};

interface ProfitByConditionChartProps {
  data: AggregatedResult[];
  title: string;
  loading?: boolean;
}

// Custom tooltip component
function CustomTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: AggregatedResult }> }) {
  if (!active || !payload || !payload.length) return null;
  
  const data = payload[0].payload;
  const label = CONDITION_LABELS[data.condition] || data.condition;
  
  return (
    <div className="bg-white border-4 border-[#8B4513] rounded-xl p-3 shadow-[4px_4px_0_#5D4037]">
      <p className="font-display text-[#5D4037] mb-1">{label}</p>
      <p className="text-sm text-[#5D4037]">
        <span className="font-semibold">Mean Profit:</span>{' '}
        <span className={data.meanProfit >= 0 ? 'text-[#388E3C]' : 'text-[#C62828]'}>
          {data.meanProfit >= 0 ? '+' : ''}${(data.meanProfit / 100).toFixed(2)}
        </span>
      </p>
      <p className="text-sm text-[#5D4037]/70">
        <span className="font-semibold">Std Dev:</span> ${(data.stdDev / 100).toFixed(2)}
      </p>
      <p className="text-sm text-[#5D4037]/70">
        <span className="font-semibold">Runs:</span> {data.runCount}
      </p>
      <p className="text-xs text-[#5D4037]/50 mt-1">
        Range: ${(data.minProfit / 100).toFixed(2)} to ${(data.maxProfit / 100).toFixed(2)}
      </p>
    </div>
  );
}

export function ProfitByConditionChart({ 
  data, 
  title, 
  loading = false,
}: ProfitByConditionChartProps) {
  // Sort by mean profit descending
  const sortedData = [...data].sort((a, b) => b.meanProfit - a.meanProfit);
  
  // Transform data for chart (convert cents to dollars)
  const chartData = sortedData.map(d => ({
    ...d,
    displayProfit: d.meanProfit / 100,
    displayStdDev: d.stdDev / 100,
    label: CONDITION_LABELS[d.condition] || d.condition,
  }));

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

  if (!data || data.length === 0) {
    return <EmptyChartState title="No Results" message="Complete some benchmark runs to see aggregated results by condition." />;
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
              tickFormatter={(value) => `$${value.toFixed(0)}`}
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
            <Bar 
              dataKey="displayProfit" 
              radius={[0, 8, 8, 0]}
              stroke="#5D4037"
              strokeWidth={2}
            >
              {chartData.map((entry) => (
                <Cell 
                  key={entry.condition} 
                  fill={CONDITION_COLORS[entry.condition] || '#8B4513'} 
                />
              ))}
              <ErrorBar 
                dataKey="displayStdDev" 
                width={4} 
                strokeWidth={2} 
                stroke="#5D4037"
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
