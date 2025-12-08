import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { PageLayout } from '@/components/layout';
import {
  Play,
  Trophy,
  Github,
  ExternalLink,
  Bot,
  Sparkles,
  Loader2,
} from 'lucide-react';
import { formatCents } from '@/lib/format';
import { getBestRunsPerModel } from '@/api';
import type { LeaderboardRun } from '@/types';

interface LeaderboardEntry {
  rank: number;
  model: string;
  profit: number;
  badge: 'gold' | 'silver' | 'bronze' | null;
}

function getRetroBadgeVariant(badge: string | null): "retro-gold" | "retro-silver" | "retro-bronze" | "retro-blue" | "retro" {
  switch (badge) {
    case 'gold':
      return 'retro-gold';
    case 'silver':
      return 'retro-silver';
    case 'bronze':
      return 'retro-bronze';
    default:
      return 'retro';
  }
}

function transformToLeaderboardEntries(runs: LeaderboardRun[]): LeaderboardEntry[] {
  return runs
    .sort((a, b) => b.total_profit - a.total_profit)
    .slice(0, 10)
    .map((run, index) => ({
      rank: index + 1,
      model: run.model_name,
      profit: run.total_profit,
      badge: index === 0 ? 'gold' : index === 1 ? 'silver' : index === 2 ? 'bronze' : null,
    }));
}

export function LandingPage() {
  const [leaderboardData, setLeaderboardData] = useState<LeaderboardEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchLeaderboard() {
      setLoading(true);
      setError(null);
      const runs = await getBestRunsPerModel();
      setLeaderboardData(transformToLeaderboardEntries(runs));
      setLoading(false);
    }

    fetchLeaderboard().catch((err) => {
      console.error('Failed to fetch leaderboard:', err);
      setError(err instanceof Error ? err.message : 'Failed to load leaderboard');
      setLoading(false);
    });
  }, []);

  return (
    <PageLayout
      headerSubtitle="LLM Decision-Making Benchmark"
      footerTagline="A decision-making benchmark for LLMs"
    >
      {/* Hero Section */}
      <section className="container mx-auto px-4 py-8 sm:py-16 text-center">
        <div className="max-w-3xl mx-auto">
          <div className="mb-6 sm:mb-8 relative">
            <img
              src="/LemonadeBench Logo.png"
              alt="LemonadeBench"
              className="h-24 sm:h-36 w-auto mx-auto mb-4 sm:mb-6 drop-shadow-[4px_4px_0_rgba(139,69,19,0.5)] animate-retro-pop"
            />
            {/* Decorative sparkles - hidden on mobile for cleaner look */}
            <Sparkles className="hidden sm:block absolute top-0 right-1/4 h-6 w-6 text-yellow-400 animate-retro-sparkle" />
            <Sparkles className="hidden sm:block absolute bottom-4 left-1/4 h-5 w-5 text-pink-400 animate-retro-sparkle" style={{ animationDelay: '0.5s' }} />
          </div>

          <h2 className="font-display text-2xl sm:text-4xl md:text-5xl mb-3 sm:mb-4 text-[#5D4037] drop-shadow-[3px_3px_0_#FFD700]">
            Can Your LLM Run a Lemonade Stand?
          </h2>

          <p className="text-base sm:text-lg text-[#5D4037]/80 mb-6 sm:mb-8 max-w-2xl mx-auto font-medium px-2">
            A fun, multi-factor decision-making benchmark for evaluating LLM reasoning. Manage
            pricing, inventory, weather, and customer satisfaction over a 14-day summer season.
          </p>

          <div className="flex flex-col sm:flex-row gap-3 sm:gap-4 justify-center mb-8 sm:mb-12">
            <Link to="/game">
              <Button variant="retro-green" size="retro-lg" className="gap-2 text-base sm:text-lg w-full sm:w-auto">
                <Play className="h-5 w-5 sm:h-6 sm:w-6" />
                Play Now!
              </Button>
            </Link>
            <a
              href="https://github.com/Shaun3141/LemonadeBench"
              target="_blank"
              rel="noopener noreferrer"
            >
              <Button variant="retro-outline" size="retro-lg" className="gap-2 text-base sm:text-lg w-full sm:w-auto">
                <Github className="h-5 w-5 sm:h-6 sm:w-6" />
                GitHub
              </Button>
            </a>
          </div>

          {/* Quick Stats - Retro Style */}
          <div className="grid grid-cols-3 gap-2 sm:gap-4 max-w-lg mx-auto">
            <Card variant="retro-yellow" className="py-3 sm:py-4">
              <CardContent className="p-0 text-center">
                <div className="font-pixel text-lg sm:text-xl text-[#FF6B35]">14</div>
                <div className="text-[10px] sm:text-xs font-display text-[#8B4513] uppercase tracking-wide mt-1">Day Season</div>
              </CardContent>
            </Card>
            <Card variant="retro-pink" className="py-3 sm:py-4">
              <CardContent className="p-0 text-center">
                <div className="font-pixel text-lg sm:text-xl text-[#C71585]">5+</div>
                <div className="text-[10px] sm:text-xs font-display text-[#880E4F] uppercase tracking-wide mt-1">Decisions</div>
              </CardContent>
            </Card>
            <Card variant="retro-green" className="py-3 sm:py-4">
              <CardContent className="p-0 text-center">
                <div className="font-pixel text-lg sm:text-xl text-[#228B22]">$50</div>
                <div className="text-[10px] sm:text-xs font-display text-[#1B5E20] uppercase tracking-wide mt-1">Target</div>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Leaderboard Section */}
      <section className="container mx-auto px-4 py-8 sm:py-12">
        <Card variant="retro" className="max-w-3xl mx-auto">
          <CardHeader className="px-3 sm:px-6">
            <div className="flex items-center justify-between gap-2">
              <CardTitle variant="retro" className="flex items-center gap-1.5 sm:gap-2 text-base sm:text-lg">
                <Trophy className="h-5 w-5 sm:h-6 sm:w-6 text-[#FFD700] drop-shadow-[1px_1px_0_#8B4513]" />
                <span>High Scores!</span>
              </CardTitle>
              <Link to="/leaderboard">
                <Button variant="retro" size="retro-sm" className="gap-1 sm:gap-2 text-xs sm:text-sm px-2 sm:px-3">
                  <span className="hidden sm:inline">Full</span> Scores
                </Button>
              </Link>
            </div>
            <p className="text-xs sm:text-sm text-[#5D4037]/70">
              Total profit over 14-day season. Beat the bots!
            </p>
          </CardHeader>
          <CardContent className="px-3 sm:px-6">
            {loading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-8 w-8 animate-spin text-[#FF6B35]" />
                <span className="ml-2 text-[#5D4037]">Loading scores...</span>
              </div>
            ) : error ? (
              <div className="text-center py-8">
                <p className="text-[#D32F2F]">Failed to load scores</p>
                <p className="text-xs text-[#5D4037]/70 mt-1">{error}</p>
              </div>
            ) : leaderboardData.length === 0 ? (
              <div className="text-center py-8">
                <p className="text-[#5D4037]">No scores yet!</p>
                <p className="text-xs text-[#5D4037]/70 mt-1">Be the first to submit a model run.</p>
              </div>
            ) : (
              <div className="space-y-2">
                {leaderboardData.map((entry) => (
                  <div
                    key={entry.rank}
                    className={`flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-4 p-2.5 sm:p-3 rounded-xl transition-all border-2 ${
                      entry.rank <= 3
                        ? 'bg-gradient-to-r from-[#FFF9C4] to-[#FFECB3] border-[#FFA000] shadow-[2px_2px_0_#FF6F00]'
                        : 'bg-white/80 border-[#D7CCC8] hover:border-[#8B4513] hover:shadow-[2px_2px_0_#5D4037]'
                    }`}
                  >
                    {/* Mobile: Row with rank, model, and profit */}
                    <div className="flex items-center justify-between sm:contents">
                      <div className="flex items-center gap-2 sm:gap-0">
                        <div className={`w-7 sm:w-8 text-center font-display text-base sm:text-lg ${entry.rank <= 3 ? 'text-[#FF6B35]' : 'text-[#8B4513]'}`}>
                          {entry.rank <= 3 ? ['🥇', '🥈', '🥉'][entry.rank - 1] : `#${entry.rank}`}
                        </div>

                        {/* Model info - inline on mobile, separate flex on desktop */}
                        <div className="flex items-center gap-1.5 sm:gap-2 sm:flex-1 sm:ml-4">
                          <Bot className="h-4 w-4 sm:h-5 sm:w-5 text-[#8B4513]" />
                          <span className="font-semibold text-[#5D4037] text-sm sm:text-base">{entry.model}</span>
                          {entry.badge && (
                            <Badge variant={getRetroBadgeVariant(entry.badge)} className="hidden sm:inline-flex">
                              {entry.badge === 'gold' && '1st'}
                              {entry.badge === 'silver' && '2nd'}
                              {entry.badge === 'bronze' && '3rd'}
                            </Badge>
                          )}
                        </div>
                      </div>

                      {/* Profit - always visible */}
                      <div
                        className={`font-pixel text-xs sm:text-sm ${
                          entry.profit >= 0 ? 'stat-retro-positive' : 'stat-retro-negative'
                        }`}
                      >
                        {entry.profit >= 0 ? '+' : ''}
                        {formatCents(entry.profit)}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}

            <Card variant="retro-yellow" className="mt-4 sm:mt-6 py-3 sm:py-4">
              <CardContent className="p-3 sm:p-4">
                <div className="flex items-start gap-2 sm:gap-3">
                  <div className="text-xl sm:text-2xl">🚀</div>
                  <div>
                    <strong className="font-display text-sm sm:text-base text-[#FF6B35]">
                      Submit Your Model!
                    </strong>
                    <p className="text-[#5D4037]/80 mt-1 text-xs sm:text-sm">
                      Check out the{' '}
                      <a
                        href="https://github.com/Shaun3141/LemonadeBench"
                        className="underline font-semibold text-[#1976D2] hover:text-[#0D47A1]"
                      >
                        GitHub repo
                      </a>{' '}
                      for benchmark instructions!
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </CardContent>
        </Card>
      </section>

      {/* OpenEnv Section */}
      <section className="container mx-auto px-4 py-8 sm:py-12">
        <Card variant="retro-purple" className="max-w-3xl mx-auto">
          <CardContent className="p-4 sm:p-6">
            <div className="flex items-start gap-3 sm:gap-4">
              <div className="bg-[#9B59B6]/20 p-2 sm:p-3 rounded-xl border-2 border-[#7B1FA2] shrink-0">
                <ExternalLink className="h-5 w-5 sm:h-6 sm:w-6 text-[#7B1FA2]" />
              </div>
              <div className="flex-1 min-w-0">
                <h3 className="font-display text-base sm:text-lg text-[#4A148C] mb-1.5 sm:mb-2">Built with OpenEnv</h3>
                <p className="text-[#4A148C]/70 mb-3 sm:mb-4 text-sm sm:text-base">
                  LemonadeBench is built on the OpenEnv framework, making it easy to integrate with
                  any RL framework that supports Gymnasium-style APIs.
                </p>
                <a
                  href="https://github.com/meta-pytorch/OpenEnv"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Button variant="retro-pink" size="retro-sm" className="gap-1.5 sm:gap-2 text-xs sm:text-sm">
                    <Github className="h-3.5 w-3.5 sm:h-4 sm:w-4" />
                    View OpenEnv
                  </Button>
                </a>
              </div>
            </div>
          </CardContent>
        </Card>
      </section>
    </PageLayout>
  );
}

export default LandingPage;
