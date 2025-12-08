import {
  ComposedChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
  Scatter,
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

interface BoxPlotData {
  condition: string;
  label: string;
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
  // For the box rendering
  boxBottom: number;
  boxHeight: number;
  // For whiskers
  lowerWhisker: number;
  upperWhisker: number;
  runCount: number;
}

function calculateQuartiles(values: number[]): { min: number; q1: number; median: number; q3: number; max: number } {
  if (values.length === 0) return { min: 0, q1: 0, median: 0, q3: 0, max: 0 };
  
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;
  
  const min = sorted[0];
  const max = sorted[n - 1];
  
  // Median
  const median = n % 2 === 0 
    ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2 
    : sorted[Math.floor(n / 2)];
  
  // Q1 (25th percentile)
  const q1Index = (n - 1) * 0.25;
  const q1Lower = Math.floor(q1Index);
  const q1Upper = Math.ceil(q1Index);
  const q1 = q1Lower === q1Upper 
    ? sorted[q1Lower] 
    : sorted[q1Lower] * (q1Upper - q1Index) + sorted[q1Upper] * (q1Index - q1Lower);
  
  // Q3 (75th percentile)
  const q3Index = (n - 1) * 0.75;
  const q3Lower = Math.floor(q3Index);
  const q3Upper = Math.ceil(q3Index);
  const q3 = q3Lower === q3Upper 
    ? sorted[q3Lower] 
    : sorted[q3Lower] * (q3Upper - q3Index) + sorted[q3Upper] * (q3Index - q3Lower);
  
  return { min, q1, median, q3, max };
}

interface DistributionBoxPlotProps {
  runs: LeaderboardRun[];
  title: string;
  loading?: boolean;
}

// Custom tooltip component
function CustomTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: BoxPlotData }> }) {
  if (!active || !payload || !payload.length) return null;
  
  const data = payload[0].payload;
  
  return (
    <div className="bg-white border-4 border-[#8B4513] rounded-xl p-3 shadow-[4px_4px_0_#5D4037]">
      <p className="font-display text-[#5D4037] mb-2">{data.label}</p>
      <div className="space-y-1 text-sm text-[#5D4037]">
        <p><span className="font-semibold">Max:</span> ${(data.max / 100).toFixed(2)}</p>
        <p><span className="font-semibold">Q3 (75%):</span> ${(data.q3 / 100).toFixed(2)}</p>
        <p><span className="font-semibold">Median:</span> <span className="text-[#FF6B35] font-bold">${(data.median / 100).toFixed(2)}</span></p>
        <p><span className="font-semibold">Q1 (25%):</span> ${(data.q1 / 100).toFixed(2)}</p>
        <p><span className="font-semibold">Min:</span> ${(data.min / 100).toFixed(2)}</p>
      </div>
      <p className="text-xs text-[#5D4037]/50 mt-2">
        {data.runCount} runs
      </p>
    </div>
  );
}

// Custom shape for box plot box (IQR box)
function BoxShape(props: { x?: number; y?: number; width?: number; height?: number; payload?: BoxPlotData }) {
  const { x, y, width, payload } = props;
  if (!payload || x === undefined || y === undefined || width === undefined) return null;
  
  const color = CONDITION_COLORS[payload.condition] || '#8B4513';
  
  // Calculate positions (values are already in display units - dollars)
  // The bar is positioned from boxBottom with boxHeight
  // We need to draw whiskers and median line
  
  return (
    <g>
      {/* IQR Box */}
      <rect
        x={x}
        y={y}
        width={width}
        height={Math.abs(props.height || 0)}
        fill={color}
        fillOpacity={0.6}
        stroke="#5D4037"
        strokeWidth={2}
        rx={4}
      />
    </g>
  );
}

export function DistributionBoxPlot({ 
  runs, 
  title, 
  loading = false,
}: DistributionBoxPlotProps) {
  // Group runs by goal_framing and calculate quartiles
  const grouped: Record<string, number[]> = {};
  
  for (const run of runs) {
    const key = run.goal_framing || 'baseline';
    if (!grouped[key]) grouped[key] = [];
    grouped[key].push(run.total_profit);
  }
  
  const chartData: BoxPlotData[] = Object.entries(grouped).map(([condition, profits]) => {
    const stats = calculateQuartiles(profits);
    return {
      condition,
      label: CONDITION_LABELS[condition] || condition,
      ...stats,
      // Convert to dollars for display
      boxBottom: stats.q1 / 100,
      boxHeight: (stats.q3 - stats.q1) / 100,
      lowerWhisker: (stats.q1 - stats.min) / 100,
      upperWhisker: (stats.max - stats.q3) / 100,
      runCount: profits.length,
    };
  });
  
  // Sort by median descending
  chartData.sort((a, b) => b.median - a.median);

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
    return <EmptyChartState title="No Results" message="Complete some benchmark runs to see the distribution." />;
  }

  // For the box plot, we'll use a horizontal bar chart approach
  // with custom rendering for whiskers and median lines
  const displayData = chartData.map(d => ({
    ...d,
    // For bar chart: start from Q1 (boxBottom), height is IQR
    displayQ1: d.q1 / 100,
    displayQ3: d.q3 / 100,
    displayMedian: d.median / 100,
    displayMin: d.min / 100,
    displayMax: d.max / 100,
    // Bar will go from Q1 to Q3
    iqrStart: d.q1 / 100,
    iqr: (d.q3 - d.q1) / 100,
  }));

  // Calculate domain for X axis
  const allValues = displayData.flatMap(d => [d.displayMin, d.displayMax]);
  const minValue = Math.min(...allValues);
  const maxValue = Math.max(...allValues);
  const padding = (maxValue - minValue) * 0.1;

  return (
    <Card variant="retro" className="h-[350px]">
      <CardHeader className="pb-2">
        <CardTitle variant="retro" className="text-[#5D4037]">{title}</CardTitle>
      </CardHeader>
      <CardContent className="h-[280px]">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={displayData}
            layout="vertical"
            margin={{ top: 5, right: 30, left: 80, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#D7CCC8" />
            <XAxis 
              type="number" 
              domain={[minValue - padding, maxValue + padding]}
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
            
            {/* Whisker lines - draw as reference lines per category */}
            {displayData.map((entry, index) => {
              const yPosition = index;
              const color = CONDITION_COLORS[entry.condition] || '#8B4513';
              return (
                <ReferenceLine
                  key={`whisker-${entry.condition}`}
                  segment={[
                    { x: entry.displayMin, y: entry.label },
                    { x: entry.displayMax, y: entry.label }
                  ]}
                  stroke={color}
                  strokeWidth={2}
                  strokeDasharray="4 2"
                />
              );
            })}
            
            {/* IQR Box - using stacked bars */}
            {/* First invisible bar to offset to Q1 */}
            <Bar 
              dataKey="iqrStart" 
              stackId="box"
              fill="transparent"
              stroke="transparent"
            />
            {/* Visible IQR bar */}
            <Bar 
              dataKey="iqr" 
              stackId="box"
              radius={[0, 4, 4, 0]}
              stroke="#5D4037"
              strokeWidth={2}
            >
              {displayData.map((entry) => (
                <Cell 
                  key={entry.condition} 
                  fill={CONDITION_COLORS[entry.condition] || '#8B4513'} 
                  fillOpacity={0.7}
                />
              ))}
            </Bar>
            
            {/* Median markers */}
            <Scatter
              dataKey="displayMedian"
              fill="#FFD700"
              stroke="#5D4037"
              strokeWidth={2}
              shape={(props: { cx?: number; cy?: number; payload?: typeof displayData[0] }) => {
                const { cx, cy } = props;
                if (cx === undefined || cy === undefined) return null;
                return (
                  <g>
                    <line
                      x1={cx}
                      y1={cy - 12}
                      x2={cx}
                      y2={cy + 12}
                      stroke="#5D4037"
                      strokeWidth={3}
                    />
                    <line
                      x1={cx}
                      y1={cy - 10}
                      x2={cx}
                      y2={cy + 10}
                      stroke="#FFD700"
                      strokeWidth={2}
                    />
                  </g>
                );
              }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

