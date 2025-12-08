import { Card, CardContent } from '@/components/ui/card';
import {
  DollarSign,
  Users,
  TrendingUp,
  TrendingDown,
  Calendar,
  Star,
  AlertTriangle,
  Wallet,
  PiggyBank,
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
      {/* Goal Banner - Only show on first day */}
      {isFirstDay && (
        <Card variant="retro-hero" className="animate-retro-pop">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="bg-white/20 p-3 rounded-xl border-2 border-white/30">
                <Citrus className="h-10 w-10 text-yellow-300 drop-shadow-[2px_2px_0_rgba(0,0,0,0.3)]" />
              </div>
              <div>
                <h1 className="font-display text-2xl flex items-center gap-2 drop-shadow-[2px_2px_0_rgba(0,0,0,0.3)]">
                  Welcome to LemonadeBench!
                </h1>
                <p className="text-white/90 text-sm flex items-center gap-1">
                  <Target className="h-3 w-3" />
                  Goal: <strong>Maximize profit</strong> over {totalDays} days of summer!
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Top Row: Day/Weather, Financials, Reputation - 3 columns */}
      <div className="grid grid-cols-3 gap-4">
        {/* Day and Weather */}
        <Card variant="retro-yellow">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <Calendar className="h-5 w-5 text-[#FF6B35]" />
              <h2 className="font-display text-lg text-[#5D4037]">
                Day {observation.day}/{observation.day + observation.days_remaining}
              </h2>
            </div>
            <div className="flex items-center gap-2">
              <WeatherIcon weather={observation.weather} size={28} />
              <div>
                <p className="text-sm font-semibold text-[#5D4037]">
                  {getWeatherEmoji(observation.weather)} {getWeatherLabel(observation.weather)} · {observation.temperature}°F
                </p>
                <p className="text-xs text-[#8B4513]/70">
                  Tomorrow: {getWeatherEmoji(observation.weather_forecast)} {getWeatherLabel(observation.weather_forecast)}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Financial Summary */}
        <Card variant="retro-green">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <DollarSign className="h-5 w-5 text-[#228B22]" />
              <h3 className="font-display text-sm text-[#1B5E20]">Money</h3>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <p className="text-xs text-[#1B5E20]/70 flex items-center gap-1">
                  <Wallet className="h-3 w-3" /> Cash
                </p>
                <p className="font-pixel text-sm stat-retro-money">{formatCents(observation.cash)}</p>
              </div>
              <div>
                <p className="text-xs text-[#1B5E20]/70 flex items-center gap-1">
                  <PiggyBank className="h-3 w-3" /> Profit
                </p>
                <p className={`font-pixel text-sm ${observation.total_profit >= 0 ? 'stat-retro-positive' : 'stat-retro-negative'}`}>
                  {formatCents(observation.total_profit)}
                </p>
              </div>
            </div>
            {!isFirstDay && (
              <div className="mt-2 pt-2 border-t border-[#388E3C] text-xs">
                <span className="text-[#1B5E20]/70">Yesterday: </span>
                <span className={`font-semibold ${observation.daily_profit >= 0 ? 'text-[#2ECC71]' : 'text-[#FF4757]'}`}>
                  {observation.daily_profit >= 0 ? '+' : ''}
                  {formatCents(observation.daily_profit)}
                </span>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Reputation */}
        <Card variant="retro-blue">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <Star className="h-5 w-5 text-[#FFD700]" />
              <h3 className="font-display text-sm text-[#0D47A1]">Rating</h3>
            </div>
            <p className="text-xl">{getReputationStars(observation.reputation)}</p>
            <div className="mt-2">
              <div className="w-full bg-[#0D47A1]/20 rounded-full h-2 border border-[#1976D2]">
                <div
                  className="bg-gradient-to-r from-[#FFD700] to-[#FFA500] h-full rounded-full transition-all duration-500"
                  style={{ width: `${observation.reputation * 100}%` }}
                />
              </div>
              <p className="text-xs text-[#0D47A1]/70 mt-1">
                {(observation.reputation * 100).toFixed(0)}% happy customers
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

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
