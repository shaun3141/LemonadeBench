import { useState, useCallback } from 'react';
import { resetGame, stepGame } from '@/api';
import type { LemonadeObservation, LemonadeAction, GameHistory, LocationId } from '@/types';

interface UseGameOptions {
  onSeedReady?: (seed: number) => void;
}

/**
 * Hook for managing game state and actions
 */
export function useGame(options: UseGameOptions = {}) {
  const [observation, setObservation] = useState<LemonadeObservation | null>(null);
  const [history, setHistory] = useState<GameHistory[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleReset = useCallback(
    async (seed?: number) => {
      setIsLoading(true);
      setError(null);
      try {
        const result = await resetGame(seed);

        // Notify about the seed
        const actualSeed = result.seed ?? seed;
        if (actualSeed !== undefined && options.onSeedReady) {
          options.onSeedReady(actualSeed);
        }

        setObservation(result.observation);
        setHistory([]);

        return result;
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to reset game';
        setError(message);
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    [options]
  );

  const handleAction = useCallback(
    async (action: LemonadeAction) => {
      if (!observation) return null;

      setIsLoading(true);
      setError(null);
      try {
        // Capture current state before action
        const dayWeather = observation.weather;
        const dayTemp = observation.temperature;
        const dayLocation: LocationId = (action.location || observation.current_location) as LocationId;

        const result = await stepGame(action);

        // Create history entry with weather/location context
        const historyEntry: GameHistory = {
          day: observation.day,
          action,
          result,
          weather: dayWeather,
          temperature: dayTemp,
          location: dayLocation,
        };

        // Add to history
        setHistory((prev) => [...prev, historyEntry]);
        setObservation(result.observation);

        return historyEntry;
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to take action';
        setError(message);
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    [observation]
  );

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    // State
    observation,
    history,
    isLoading,
    error,

    // Actions
    handleReset,
    handleAction,
    clearError,

    // Derived state
    isGameOver: observation?.done ?? false,
    currentDay: observation?.day ?? 0,
    daysRemaining: observation?.days_remaining ?? 0,
  };
}

