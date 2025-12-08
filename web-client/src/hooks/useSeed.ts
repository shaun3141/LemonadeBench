import { useState, useCallback } from 'react';

/**
 * Generate a random seed between 10000 and 99999
 */
function generateRandomSeed(): number {
  return Math.floor(Math.random() * 90000) + 10000;
}

/**
 * Get seed from URL query params or generate a new one
 */
function getSeedFromUrl(): number {
  const params = new URLSearchParams(window.location.search);
  const seedParam = params.get('seed');
  if (seedParam) {
    const parsed = parseInt(seedParam, 10);
    if (!isNaN(parsed) && parsed >= 0) {
      return parsed;
    }
  }
  return generateRandomSeed();
}

/**
 * Update URL with seed (without reloading)
 */
function updateUrlWithSeed(seed: number) {
  const url = new URL(window.location.href);
  url.searchParams.set('seed', seed.toString());
  window.history.replaceState({}, '', url.toString());
}

/**
 * Hook for managing game seed via URL
 */
export function useSeed() {
  const [seed, setSeedState] = useState<number | null>(null);

  const initializeSeed = useCallback((serverSeed?: number): number => {
    const urlSeed = getSeedFromUrl();
    const actualSeed = serverSeed ?? urlSeed;
    setSeedState(actualSeed);
    updateUrlWithSeed(actualSeed);
    return actualSeed;
  }, []);

  const setSeed = useCallback((newSeed: number) => {
    setSeedState(newSeed);
    updateUrlWithSeed(newSeed);
  }, []);

  const getInitialSeed = useCallback(() => {
    return getSeedFromUrl();
  }, []);

  return {
    seed,
    setSeed,
    initializeSeed,
    getInitialSeed,
    generateRandomSeed,
  };
}

