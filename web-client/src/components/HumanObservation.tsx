import { Card, CardContent } from '@/components/ui/card';
import {
  DollarSign,
  Users,
  TrendingUp,
  TrendingDown,
  Star,
  AlertTriangle,
  Target,
  Citrus,
} from 'lucide-react';
import type { LemonadeObservation } from '../types';
import { WeatherIcon, getWeatherEmoji, getWeatherLabel } from './WeatherIcon';
import { formatCents, getReputationStars } from '@/lib/format';

interface HumanObservationProps {
  observation: LemonadeObservation;
}

export function HumanObservation({ observation }: HumanObservationProps) {
  const isFirstDay = observation.day === 1 && observation.daily_revenue === 0;
  const totalDays = observation.day + observation.days_remaining;

  return (
    <div className="space-y-4">
      {/* Combined Status Banner */}
      <Card variant="retro-hero" className={isFirstDay ? 'animate-retro-pop' : ''}>
        <CardContent className="p-4">
          <div className="flex items-stretch gap-4">
            {/* Left: Welcome/Status */}
            <div className="flex items-center gap-3 flex-1">
              <div className="bg-white/20 p-3 rounded-xl border-2 border-white/30">
                <Citrus className="h-10 w-10 text-yellow-300 drop-shadow-[2px_2px_0_rgba(0,0,0,0.3)]" />
              </div>
              <div>
                <h1 className="font-display text-2xl flex items-center gap-2 drop-shadow-[2px_2px_0_rgba(0,0,0,0.3)]">
                  {isFirstDay ? 'Welcome to LemonadeBench!' : `Day ${observation.day} of ${totalDays}`}
                </h1>
                <p className="text-white/90 text-sm flex items-center gap-1">
                  <Target className="h-3 w-3" />
                  {isFirstDay ? (
                    <>Goal: <strong>Maximize profit</strong> over {totalDays} days of summer!</>
                  ) : (
                    <>{observation.days_remaining} days remaining</>
                  )}
                </p>
              </div>
            </div>

            {/* Right: Status Sub-cards */}
            <div className="flex gap-2">
              {/* Weather Mini-card */}
              <div className="bg-white/20 backdrop-blur rounded-xl p-3 border-2 border-white/30 min-w-[100px]">
                <div className="flex items-center gap-1.5 mb-1">
                  <WeatherIcon weather={observation.weather} size={20} />
                  <span className="text-xs font-semibold text-white/90">{observation.temperature}°F</span>
                </div>
                <p className="text-[10px] text-white/70 leading-tight">
                  {getWeatherLabel(observation.weather)}
                </p>
                <p className="text-[10px] text-white/60 leading-tight">
                  → {getWeatherEmoji(observation.weather_forecast)} {getWeatherLabel(observation.weather_forecast)}
                </p>
              </div>

              {/* Money Mini-card */}
              <div className="bg-white/20 backdrop-blur rounded-xl p-3 border-2 border-white/30 min-w-[100px]">
                <div className="flex items-center gap-1.5 mb-1">
                  <DollarSign className="h-4 w-4 text-[#7FFF00]" />
                  <span className="text-xs font-semibold text-white/90">Money</span>
                </div>
                <p className="font-pixel text-sm text-[#7FFF00] drop-shadow-[1px_1px_0_rgba(0,0,0,0.5)]">
                  {formatCents(observation.cash)}
                </p>
                <p className={`text-[10px] ${observation.total_profit >= 0 ? 'text-[#7FFF00]' : 'text-[#FF6B6B]'}`}>
                  Profit: {formatCents(observation.total_profit)}
                </p>
              </div>

              {/* Rating Mini-card */}
              <div className="bg-white/20 backdrop-blur rounded-xl p-3 border-2 border-white/30 min-w-[90px]">
                <div className="flex items-center gap-1.5 mb-1">
                  <Star className="h-4 w-4 text-[#FFD700]" />
                  <span className="text-xs font-semibold text-white/90">Rating</span>
                </div>
                <p className="text-sm leading-none">{getReputationStars(observation.reputation)}</p>
                <p className="text-[10px] text-white/70 mt-1">
                  {(observation.reputation * 100).toFixed(0)}% happy
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Yesterday's Sales - Only show if not first day */}
      {!isFirstDay && (
        <Card variant="retro">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-3">
              <Users className="h-5 w-5 text-[#5D4037]" />
              <h3 className="font-display text-[#5D4037]">Yesterday's Results</h3>
            </div>
            <div className="grid grid-cols-6 gap-4 text-sm">
              <div className="bg-[#E8F5E9] rounded-lg p-2 border border-[#388E3C]">
                <p className="text-xs text-[#1B5E20]/70">💵 Revenue</p>
                <p className="font-semibold text-[#2ECC71]">
                  {formatCents(observation.daily_revenue)}
                </p>
              </div>
              <div className="bg-[#FFEBEE] rounded-lg p-2 border border-[#E57373]">
                <p className="text-xs text-[#C62828]/70">💸 Costs</p>
                <p className="font-semibold text-[#FF4757]">{formatCents(observation.daily_costs)}</p>
              </div>
              <div className={`rounded-lg p-2 border ${observation.daily_profit >= 0 ? 'bg-[#E8F5E9] border-[#388E3C]' : 'bg-[#FFEBEE] border-[#E57373]'}`}>
                <p className="text-xs text-[#5D4037]/70">📊 Profit</p>
                <p
                  className={`font-semibold flex items-center gap-1 ${observation.daily_profit >= 0 ? 'text-[#2ECC71]' : 'text-[#FF4757]'}`}
                >
                  {observation.daily_profit >= 0 ? (
                    <TrendingUp className="h-3 w-3" />
                  ) : (
                    <TrendingDown className="h-3 w-3" />
                  )}
                  {formatCents(observation.daily_profit)}
                </p>
              </div>
              <div className="bg-[#E3F2FD] rounded-lg p-2 border border-[#1976D2]">
                <p className="text-xs text-[#0D47A1]/70">🥤 Sold</p>
                <p className="font-semibold text-[#1976D2]">{observation.cups_sold}</p>
              </div>
              {observation.cups_wasted > 0 && (
                <div className="bg-[#FFF3E0] rounded-lg p-2 border border-[#FF9800]">
                  <p className="text-xs text-[#E65100]/70">🗑️ Wasted</p>
                  <p className="font-semibold text-[#FF9800]">{observation.cups_wasted}</p>
                </div>
              )}
              {observation.customers_turned_away > 0 && (
                <div className="bg-[#FFF3E0] rounded-lg p-2 border border-[#FF9800]">
                  <p className="text-xs text-[#E65100]/70 flex items-center gap-1">
                    <AlertTriangle className="h-3 w-3" />
                    Lost
                  </p>
                  <p className="font-semibold text-[#FF9800]">{observation.customers_turned_away}</p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Game Over */}
      {observation.done && (
        <Card variant="retro-hero" className="animate-retro-pop">
          <CardContent className="p-6 text-center">
            <h2 className="font-display text-4xl mb-2 drop-shadow-[3px_3px_0_rgba(0,0,0,0.3)]">🎉 Season Complete! 🎉</h2>
            <p className="font-pixel text-xl stat-retro-money">
              Final Profit: {formatCents(observation.total_profit)}
            </p>
            <div className="mt-4 flex justify-center gap-2">
              {observation.total_profit >= 5000 && <span className="text-3xl">🏆</span>}
              {observation.total_profit >= 3000 && <span className="text-3xl">⭐</span>}
              {observation.total_profit >= 1000 && <span className="text-3xl">🌟</span>}
              {observation.total_profit < 0 && <span className="text-3xl">😅</span>}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
