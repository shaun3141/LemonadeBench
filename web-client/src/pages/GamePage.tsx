import { useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent } from '@/components/ui/card';
import { Citrus, WifiOff, Dices } from 'lucide-react';

import { HumanObservation } from '@/components/HumanObservation';
import { ActionControls } from '@/components/ActionControls';
import { DayEndStatsModal } from '@/components/game';
import { useGame, useSeed, useServerConnection } from '@/hooks';
import type { GameHistory } from '@/types';

export function GamePage() {
  const { seed, initializeSeed, getInitialSeed } = useSeed();
  const { isConnected, connectionError } = useServerConnection();
  const {
    observation,
    history,
    isLoading,
    error,
    handleReset: gameReset,
    handleAction,
  } = useGame({
    onSeedReady: initializeSeed,
  });

  const [dayEndStats, setDayEndStats] = useState<GameHistory | null>(null);

  // Action form state (lifted up for MarketInsights)
  const [selectedPrice, setSelectedPrice] = useState(75);

  // Initialize game with seed
  const handleReset = useCallback(
    async (newSeed?: number) => {
      const seedToUse = newSeed ?? getInitialSeed();
      await gameReset(seedToUse);
    },
    [gameReset, getInitialSeed]
  );

  // Handle action and show day end stats
  const onAction = useCallback(
    async (action: Parameters<typeof handleAction>[0]) => {
      const historyEntry = await handleAction(action);
      if (historyEntry) {
        setDayEndStats(historyEntry);
      }
    },
    [handleAction]
  );

  // Auto-start game when connected
  useEffect(() => {
    if (isConnected && !observation && !isLoading) {
      handleReset();
    }
  }, [isConnected, observation, isLoading, handleReset]);

  return (
    <div className="min-h-screen bg-gradient-to-b from-[#87CEEB] via-[#FFE4B5] to-[#98FB98] dark:from-zinc-950 dark:via-zinc-900 dark:to-zinc-950">
      {/* Retro Game Header */}
      <header className="border-b-4 border-[#8B4513] bg-gradient-to-r from-[#FFE135] via-[#FFD700] to-[#FFA500] sticky top-0 z-50 shadow-[0_4px_0_#5D4037]">
        <div className="container mx-auto px-4 py-3">
          <div className="flex items-center justify-between">
            <Link to="/" className="flex items-center gap-3 hover:scale-105 transition-transform">
              <div className="bg-white p-2 rounded-xl border-3 border-[#8B4513] shadow-[3px_3px_0_#5D4037]">
                <Citrus className="h-7 w-7 text-[#FF6B35]" />
              </div>
              <div>
                <h1 className="font-display text-2xl text-[#5D4037] drop-shadow-[2px_2px_0_rgba(255,255,255,0.5)]">
                  LemonadeBench
                </h1>
                <p className="text-xs font-semibold text-[#8B4513]/80">Human Baseline Collection</p>
              </div>
            </Link>

            <div className="flex items-center gap-4">
              {/* Seed Display */}
              {seed !== null && (
                <Badge variant="retro-pixel" className="gap-1.5">
                  <Dices className="h-3 w-3" />
                  #{seed}
                </Badge>
              )}

              {/* Connection Status - only show when disconnected */}
              {!isConnected && (
                <Badge variant="retro-pink" className="gap-1">
                  <WifiOff className="h-3 w-3" />
                  Offline
                </Badge>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-6">
        {(error || connectionError) && (
          <Card variant="retro-pink" className="mb-4">
            <CardContent className="p-4 text-[#880E4F] font-semibold">
              ⚠️ {error || connectionError}
            </CardContent>
          </Card>
        )}

        {!isConnected ? (
          <Card variant="retro" className="max-w-md mx-auto mt-12">
            <CardContent className="p-8 text-center">
              <WifiOff className="h-16 w-16 mx-auto text-[#8B4513] mb-4" />
              <h2 className="font-display text-2xl text-[#5D4037] mb-2">Server Offline!</h2>
              <p className="text-[#5D4037]/80 mb-4">
                Start the LemonadeBench server to play:
              </p>
              <pre className="bg-[#333] text-[#7FFF00] font-pixel text-xs p-4 rounded-xl text-left overflow-x-auto border-2 border-[#8B4513]">
                cd LemonadeBench{'\n'}
                uv run server
              </pre>
            </CardContent>
          </Card>
        ) : observation ? (
          <div className="grid lg:grid-cols-[1fr,420px] gap-6">
            {/* Observation Panel */}
            <div>
              <HumanObservation observation={observation} />
            </div>

            {/* Action Panel */}
            <div className="lg:sticky lg:top-20 lg:self-start">
              <ActionControls
                observation={observation}
                history={history}
                onSubmit={onAction}
                onReset={handleReset}
                disabled={isLoading}
                selectedPrice={selectedPrice}
                onPriceChange={setSelectedPrice}
              />
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center py-12">
            <div className="animate-retro-sparkle text-6xl mb-4">🍋</div>
            <p className="font-display text-[#5D4037]">Loading game...</p>
          </div>
        )}
      </main>

      {/* Day End Stats Modal */}
      <DayEndStatsModal dayEndStats={dayEndStats} onClose={() => setDayEndStats(null)} />

      {/* Retro Footer */}
      <footer className="border-t-4 border-[#8B4513] mt-auto py-4 bg-gradient-to-r from-[#FFF9C4] via-[#FFECB3] to-[#FFF9C4]">
        <div className="container mx-auto px-4 text-center text-sm">
          <span className="font-display text-[#5D4037]">🍋 LemonadeBench</span>
          <span className="text-[#FFB300] mx-2">★</span>
          <span className="text-[#8B4513]">Built with </span>
          <a
            href="https://github.com/meta-pytorch/OpenEnv"
            className="font-semibold text-[#1976D2] hover:text-[#0D47A1]"
          >
            OpenEnv
          </a>
          <span className="text-[#FFB300] mx-2">★</span>
          <span className="text-[#8B4513]">Collecting human baselines!</span>
        </div>
      </footer>
    </div>
  );
}

export default GamePage;
