// API client for LemonadeBench server

import type { LemonadeAction, GameState, LeaderboardRun, RunTurn, GoalFraming, Architecture, Scaffolding } from './types';

const API_BASE = '/api';

// Supabase configuration from environment variables
const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL;
const SUPABASE_ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY;

// Filter options for leaderboard queries
export interface LeaderboardFilters {
  goal_framing?: GoalFraming;
  architecture?: Architecture;
  scaffolding?: Scaffolding;
}

// Game API
export async function resetGame(seed?: number): Promise<GameState> {
  const response = await fetch(`${API_BASE}/reset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(seed !== undefined ? { seed } : {}),
  });
  
  if (!response.ok) {
    throw new Error(`Failed to reset game: ${response.statusText}`);
  }
  
  return response.json();
}

export async function stepGame(action: LemonadeAction): Promise<GameState> {
  const response = await fetch(`${API_BASE}/step`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(action),
  });
  
  if (!response.ok) {
    throw new Error(`Failed to step game: ${response.statusText}`);
  }
  
  return response.json();
}

export async function getHealth(): Promise<{ status: string }> {
  const response = await fetch(`${API_BASE}/health`);
  return response.json();
}

// Supabase API - Leaderboard
async function supabaseQuery<T>(endpoint: string): Promise<T> {
  const response = await fetch(`${SUPABASE_URL}/rest/v1/${endpoint}`, {
    headers: {
      'apikey': SUPABASE_ANON_KEY,
      'Authorization': `Bearer ${SUPABASE_ANON_KEY}`,
    },
  });
  
  if (!response.ok) {
    throw new Error(`Supabase query failed: ${response.statusText}`);
  }
  
  return response.json();
}

export async function getBestRunsPerModel(filters?: LeaderboardFilters): Promise<LeaderboardRun[]> {
  // Filter out in-progress runs (completed_at is not null)
  let query = 'best_runs_per_model?completed_at=not.is.null&order=total_profit.desc';
  
  if (filters?.goal_framing) {
    query += `&goal_framing=eq.${filters.goal_framing}`;
  }
  if (filters?.architecture) {
    query += `&architecture=eq.${filters.architecture}`;
  }
  if (filters?.scaffolding) {
    query += `&scaffolding=eq.${filters.scaffolding}`;
  }
  
  return supabaseQuery<LeaderboardRun[]>(query);
}

export async function getAllRuns(limit?: number, filters?: LeaderboardFilters): Promise<LeaderboardRun[]> {
  // Join with models table to get model info
  // Filter out in-progress runs (completed_at is not null)
  let query = `runs?select=*,models(name,provider)&completed_at=not.is.null&order=completed_at.desc`;
  
  if (limit) {
    query += `&limit=${limit}`;
  }
  
  if (filters?.goal_framing) {
    query += `&goal_framing=eq.${filters.goal_framing}`;
  }
  if (filters?.architecture) {
    query += `&architecture=eq.${filters.architecture}`;
  }
  if (filters?.scaffolding) {
    query += `&scaffolding=eq.${filters.scaffolding}`;
  }
  
  const runs = await supabaseQuery<Array<{
    id: string;
    model_id: string;
    seed: number | null;
    goal_framing: GoalFraming;
    architecture: Architecture;
    scaffolding: Scaffolding;
    total_profit: number;
    total_cups_sold: number;
    final_cash: number;
    final_reputation: number;
    turn_count: number;
    error_count: number;
    started_at: string;
    completed_at: string | null;
    models: { name: string; provider: string };
  }>>(query);
  
  // Transform to LeaderboardRun format
  return runs.map(run => ({
    run_id: run.id,
    model_id: run.model_id,
    model_name: run.models.name,
    provider: run.models.provider,
    seed: run.seed,
    goal_framing: run.goal_framing || 'baseline',
    architecture: run.architecture || 'react',
    scaffolding: run.scaffolding || 'none',
    total_profit: run.total_profit,
    total_cups_sold: run.total_cups_sold,
    final_cash: run.final_cash,
    final_reputation: run.final_reputation,
    turn_count: run.turn_count,
    error_count: run.error_count,
    started_at: run.started_at,
    completed_at: run.completed_at,
  }));
}

export async function getRunTurns(runId: string): Promise<RunTurn[]> {
  return supabaseQuery<RunTurn[]>(`turns?run_id=eq.${runId}&order=day.asc`);
}

// Aggregated results for charts
export interface AggregatedResult {
  condition: string;
  meanProfit: number;
  stdDev: number;
  runCount: number;
  minProfit: number;
  maxProfit: number;
}

// Helper to calculate statistics from an array of profit values
export function calculateStats(profits: number[]): { mean: number; stdDev: number; min: number; max: number } {
  if (profits.length === 0) return { mean: 0, stdDev: 0, min: 0, max: 0 };
  
  const mean = profits.reduce((a, b) => a + b, 0) / profits.length;
  const squaredDiffs = profits.map(p => Math.pow(p - mean, 2));
  const variance = squaredDiffs.reduce((a, b) => a + b, 0) / profits.length;
  const stdDev = Math.sqrt(variance);
  
  return {
    mean,
    stdDev,
    min: Math.min(...profits),
    max: Math.max(...profits),
  };
}

// Generic function to aggregate runs by a field and calculate statistics
// Can be used on pre-filtered runs from components
type AggregationKey = 'goal_framing' | 'architecture' | 'scaffolding' | 'model_name';

export function aggregateRunsByField(
  runs: LeaderboardRun[],
  fieldKey: AggregationKey,
  defaultValue = 'unknown'
): AggregatedResult[] {
  const grouped: Record<string, number[]> = {};
  
  for (const run of runs) {
    const key = String(run[fieldKey] || defaultValue);
    if (!grouped[key]) grouped[key] = [];
    grouped[key].push(run.total_profit);
  }
  
  return Object.entries(grouped).map(([condition, profits]) => {
    const stats = calculateStats(profits);
    return {
      condition,
      meanProfit: stats.mean,
      stdDev: stats.stdDev,
      runCount: profits.length,
      minProfit: stats.min,
      maxProfit: stats.max,
    };
  });
}

// Get aggregated results by goal framing
// Only includes runs from the goal framing study (architecture=react, scaffolding=none)
export async function getResultsByGoalFraming(): Promise<AggregatedResult[]> {
  const runs = await getAllRuns();
  const studyRuns = runs.filter(run => 
    run.architecture === 'react' && run.scaffolding === 'none'
  );
  return aggregateRunsByField(studyRuns, 'goal_framing', 'baseline');
}

// Get aggregated results by architecture
// Only includes runs from the architecture ablation study (goal_framing=baseline, scaffolding=none)
export async function getResultsByArchitecture(): Promise<AggregatedResult[]> {
  const runs = await getAllRuns();
  const studyRuns = runs.filter(run => 
    run.goal_framing === 'baseline' && run.scaffolding === 'none'
  );
  return aggregateRunsByField(studyRuns, 'architecture', 'react');
}

// Get aggregated results by scaffolding
// Only includes runs from the scaffolding ablation study (goal_framing=baseline, architecture=react)
export async function getResultsByScaffolding(): Promise<AggregatedResult[]> {
  const runs = await getAllRuns();
  const studyRuns = runs.filter(run => 
    run.goal_framing === 'baseline' && run.architecture === 'react'
  );
  return aggregateRunsByField(studyRuns, 'scaffolding', 'none');
}

// Get aggregated results by model
export async function getResultsByModel(): Promise<AggregatedResult[]> {
  const runs = await getAllRuns();
  return aggregateRunsByField(runs, 'model_name')
    .sort((a, b) => b.meanProfit - a.meanProfit);
}

