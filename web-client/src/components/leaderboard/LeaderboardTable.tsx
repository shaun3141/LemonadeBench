import { Link } from 'react-router-dom';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import {
  Citrus,
  CupSoda,
  Star,
  Calendar,
  ChevronRight,
  Play,
  TrendingUp,
  TrendingDown,
  Bot,
} from 'lucide-react';
import type { LeaderboardRun } from '@/types';
import { formatCents, formatDate } from '@/lib/format';
import {
  getProviderBadgeColor,
  getRankDisplay,
  getGoalFramingBadge,
  getArchitectureBadge,
  getScaffoldingBadge,
} from './helpers';

interface LeaderboardTableProps {
  runs: LeaderboardRun[];
  onSelectRun: (run: LeaderboardRun) => void;
  showExperimentFactors?: boolean;
}

export function LeaderboardTable({
  runs,
  onSelectRun,
  showExperimentFactors = false,
}: LeaderboardTableProps) {
  if (runs.length === 0) {
    return (
      <Card variant="retro">
        <CardContent className="p-8 text-center">
          <Citrus className="h-12 w-12 mx-auto text-[#FF6B35] mb-4" />
          <h3 className="font-display text-lg text-[#5D4037] mb-2">No Runs Yet</h3>
          <p className="text-[#5D4037]/70 mb-4">Results will appear here once benchmarks are run.</p>
          <Link to="/">
            <Button variant="retro-green" className="gap-2">
              <Play className="h-4 w-4" />
              Try It Yourself
            </Button>
          </Link>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-3">
      {runs.map((run, index) => {
        const rank = getRankDisplay(index + 1);
        return (
          <div
            key={run.run_id}
            className={`flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-4 p-3 sm:p-4 rounded-2xl transition-all border-4 cursor-pointer ${
              index < 3
                ? 'bg-gradient-to-r from-[#FFF9C4] to-[#FFECB3] border-[#FFA000] shadow-[4px_4px_0_#FF8F00]'
                : 'bg-gradient-to-r from-white to-[#FFFDE7] border-[#D7CCC8] hover:border-[#8B4513] hover:shadow-[4px_4px_0_#5D4037]'
            }`}
            onClick={() => onSelectRun(run)}
          >
            <div className="flex items-center justify-between sm:contents">
              <div className="flex items-center gap-3">
                <div
                  className={`w-8 text-center font-display text-lg ${rank.color}`}
                >
                  {rank.icon}
                </div>
                <div>
                  <div className="font-semibold text-[#5D4037] flex items-center gap-2 flex-wrap">
                    <Bot className="h-4 w-4 text-[#8B4513]" />
                    {run.model_name}
                    <span
                      className={`text-xs px-2 py-0.5 rounded-full font-semibold ${getProviderBadgeColor(run.provider)}`}
                    >
                      {run.provider}
                    </span>
                    {showExperimentFactors &&
                      run.goal_framing &&
                      run.goal_framing !== 'baseline' &&
                      getGoalFramingBadge(run.goal_framing)}
                    {showExperimentFactors &&
                      run.architecture &&
                      run.architecture !== 'react' &&
                      getArchitectureBadge(run.architecture)}
                    {showExperimentFactors && run.scaffolding && getScaffoldingBadge(run.scaffolding)}
                  </div>
                  <div className="text-sm text-[#8B4513]/70 flex items-center gap-3 mt-1 flex-wrap">
                    <span className="flex items-center gap-1">
                      <CupSoda className="h-3 w-3" />
                      {run.total_cups_sold} cups
                    </span>
                    <span className="flex items-center gap-1">
                      <Star className="h-3 w-3" />
                      {(run.final_reputation * 100).toFixed(0)}% rep
                    </span>
                    <span className="flex items-center gap-1">
                      <Calendar className="h-3 w-3" />
                      {formatDate(run.completed_at)}
                    </span>
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="text-right">
                  <div
                    className={`font-pixel text-lg flex items-center gap-1 justify-end ${
                      run.total_profit >= 0 ? 'stat-retro-positive' : 'stat-retro-negative'
                    }`}
                  >
                    {run.total_profit >= 0 ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
                    {run.total_profit >= 0 ? '+' : ''}
                    {formatCents(run.total_profit)}
                  </div>
                  <div className="text-xs text-[#8B4513]/70 font-semibold">profit</div>
                </div>
                <ChevronRight className="h-5 w-5 text-[#8B4513]" />
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

