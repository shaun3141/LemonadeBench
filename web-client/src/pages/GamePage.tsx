import { useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog';
import { Citrus, Monitor, User, WifiOff, History, BarChart3, Dices, Eye } from 'lucide-react';

import { ModelObservation } from '@/components/ModelObservation';
import { HumanObservation } from '@/components/HumanObservation';
import { ActionControls } from '@/components/ActionControls';
import { MarketInsights } from '@/components/MarketInsights';
import { DayEndStatsModal } from '@/components/game';
import { useGame, useSeed, useServerConnection } from '@/hooks';
import type { GameHistory } from '@/types';
import { formatCents } from '@/lib/format';

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

  const [isModelView, setIsModelView] = useState(false);
  const [showMarketIntel, setShowMarketIntel] = useState(false);
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

              {/* View Toggle */}
              <div className="flex items-center gap-2 bg-white/80 rounded-xl px-3 py-1.5 border-2 border-[#8B4513]">
                <User className="h-4 w-4 text-[#5D4037]" />
                <Switch checked={isModelView} onCheckedChange={setIsModelView} id="view-mode" />
                <Monitor className="h-4 w-4 text-[#5D4037]" />
                <Label htmlFor="view-mode" className="text-sm cursor-pointer font-semibold text-[#5D4037]">
                  {isModelView ? 'Model' : 'Human'}
                </Label>
              </div>
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
              <Tabs defaultValue="observation" className="w-full">
                <TabsList className="mb-4 bg-gradient-to-r from-[#FFFDE7] to-[#FFF9C4] border-2 border-[#8B4513] rounded-xl p-1">
                  <TabsTrigger value="observation" className="gap-1.5 font-display data-[state=active]:bg-[#FFE135] data-[state=active]:text-[#5D4037] rounded-lg">
                    <Eye className="h-4 w-4" />
                    {isModelView ? 'Model View' : 'Game State'}
                  </TabsTrigger>
                  <TabsTrigger value="history" className="gap-1.5 font-display data-[state=active]:bg-[#FFE135] data-[state=active]:text-[#5D4037] rounded-lg">
                    <History className="h-4 w-4" />
                    History ({history.length})
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="observation">
                  {isModelView ? (
                    <ModelObservation observation={observation} />
                  ) : (
                    <HumanObservation observation={observation} />
                  )}
                </TabsContent>

                <TabsContent value="history">
                  <Card variant="retro">
                    <CardContent className="p-4">
                      {history.length === 0 ? (
                        <p className="text-center text-[#5D4037]/70 py-8 font-display">
                          No history yet. Start playing to see your decisions!
                        </p>
                      ) : (
                        <div className="space-y-3 max-h-[500px] overflow-y-auto">
                          {[...history].reverse().map((entry, idx) => (
                            <div
                              key={history.length - idx - 1}
                              className="p-3 bg-gradient-to-r from-[#FFF9C4] to-[#FFECB3] rounded-xl text-sm border-2 border-[#FFA000]"
                            >
                              <div className="flex justify-between items-center mb-2">
                                <span className="font-display text-[#5D4037]">Day {entry.day}</span>
                                <Badge
                                  variant={
                                    entry.result.observation.daily_profit >= 0
                                      ? 'retro-green'
                                      : 'retro-pink'
                                  }
                                >
                                  {entry.result.observation.daily_profit >= 0 ? '+' : ''}
                                  {formatCents(entry.result.observation.daily_profit)}
                                </Badge>
                              </div>
                              <div className="grid grid-cols-2 gap-2 text-xs text-[#5D4037]/80">
                                <span>💰 {formatCents(entry.action.price_per_cup)}</span>
                                <span>🥤 {entry.result.observation.cups_sold} sold</span>
                                <span>😊 {entry.result.observation.customers_served} served</span>
                                <span>😢 {entry.result.observation.customers_turned_away} left</span>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </TabsContent>
              </Tabs>
            </div>

            {/* Action Panel */}
            <div className="lg:sticky lg:top-20 lg:self-start">
              <div className="flex items-center justify-between mb-4">
                <h2 className="font-display text-xl text-[#5D4037] flex items-center gap-2">🍋 Your Turn!</h2>
                <Button
                  variant="retro-blue"
                  size="retro-sm"
                  onClick={() => setShowMarketIntel(true)}
                  className="gap-2"
                >
                  <BarChart3 className="h-4 w-4" />
                  Intel
                </Button>
              </div>
              <ActionControls
                observation={observation}
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

      {/* Market Intelligence Modal */}
      <Dialog open={showMarketIntel} onOpenChange={setShowMarketIntel}>
        <DialogContent className="sm:max-w-lg max-h-[85vh] overflow-y-auto bg-gradient-to-b from-[#FFFDE7] to-[#FFF9C4] border-4 border-[#5D4037] rounded-2xl shadow-[6px_6px_0_#3E2723]">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 font-display text-xl text-[#5D4037]">
              <BarChart3 className="h-5 w-5 text-[#1976D2]" />
              Market Intel
            </DialogTitle>
          </DialogHeader>
          {observation && (
            <MarketInsights observation={observation} selectedPrice={selectedPrice} />
          )}
          <DialogFooter>
            <Button variant="retro" onClick={() => setShowMarketIntel(false)} className="w-full">
              Got It!
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

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
