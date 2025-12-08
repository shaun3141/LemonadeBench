-- LemonadeBench Database Schema
-- Stores benchmark runs for AI model evaluation

-- Models table: tracks different AI models being benchmarked
CREATE TABLE models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    provider TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for looking up models by name
CREATE INDEX idx_models_name ON models(name);

-- Runs table: stores summary of each benchmark run
CREATE TABLE runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    seed INTEGER,
    total_profit INTEGER NOT NULL, -- cents
    total_cups_sold INTEGER NOT NULL,
    final_cash INTEGER NOT NULL, -- cents
    final_reputation FLOAT NOT NULL,
    turn_count INTEGER NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

-- Index for finding best runs per model
CREATE INDEX idx_runs_model_profit ON runs(model_id, total_profit DESC);

-- Index for sorting by date
CREATE INDEX idx_runs_completed_at ON runs(completed_at DESC);

-- Turns table: stores turn-by-turn data for each run
CREATE TABLE turns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    day INTEGER NOT NULL,
    observation JSONB NOT NULL,
    action JSONB NOT NULL,
    reasoning TEXT,
    result JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for fetching turns by run
CREATE INDEX idx_turns_run_day ON turns(run_id, day);

-- Enable Row Level Security
ALTER TABLE models ENABLE ROW LEVEL SECURITY;
ALTER TABLE runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE turns ENABLE ROW LEVEL SECURITY;

-- RLS Policies: Allow public read access (anon key can read)
CREATE POLICY "Allow public read access on models"
    ON models FOR SELECT
    USING (true);

CREATE POLICY "Allow public read access on runs"
    ON runs FOR SELECT
    USING (true);

CREATE POLICY "Allow public read access on turns"
    ON turns FOR SELECT
    USING (true);

-- RLS Policies: Allow service role to insert/update
CREATE POLICY "Allow service role insert on models"
    ON models FOR INSERT
    WITH CHECK (true);

CREATE POLICY "Allow service role insert on runs"
    ON runs FOR INSERT
    WITH CHECK (true);

CREATE POLICY "Allow service role update on runs"
    ON runs FOR UPDATE
    USING (true);

CREATE POLICY "Allow service role insert on turns"
    ON turns FOR INSERT
    WITH CHECK (true);

-- View: Best run per model (for leaderboard)
CREATE VIEW best_runs_per_model AS
SELECT DISTINCT ON (r.model_id)
    r.id AS run_id,
    r.model_id,
    m.name AS model_name,
    m.provider,
    r.seed,
    r.total_profit,
    r.total_cups_sold,
    r.final_cash,
    r.final_reputation,
    r.turn_count,
    r.started_at,
    r.completed_at
FROM runs r
JOIN models m ON r.model_id = m.id
ORDER BY r.model_id, r.total_profit DESC;

-- Grant access to the view
GRANT SELECT ON best_runs_per_model TO anon, authenticated;

