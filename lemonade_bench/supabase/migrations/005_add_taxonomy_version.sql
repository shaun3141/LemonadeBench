-- Add taxonomy version tracking to runs
-- This allows us to track changes to prompts, context, pricing, and other benchmark parameters

-- Add taxonomy_version column to runs table
-- Version 1: Original benchmark (before December 2025 fixes)
-- Version 2: Fixed pricing display, improved context formatting
ALTER TABLE runs ADD COLUMN IF NOT EXISTS taxonomy_version INTEGER DEFAULT 1;

-- Backfill all existing runs as version 1
UPDATE runs SET taxonomy_version = 1 WHERE taxonomy_version IS NULL;

-- Make the column NOT NULL after backfill
ALTER TABLE runs ALTER COLUMN taxonomy_version SET NOT NULL;

-- Add a check constraint for valid versions
ALTER TABLE runs ADD CONSTRAINT chk_taxonomy_version 
    CHECK (taxonomy_version >= 1);

-- Add index for filtering by version
CREATE INDEX IF NOT EXISTS idx_runs_taxonomy_version ON runs(taxonomy_version);

-- Add composite index for version + model queries
CREATE INDEX IF NOT EXISTS idx_runs_version_model ON runs(taxonomy_version, model_id, total_profit DESC);

-- Comment for documentation
COMMENT ON COLUMN runs.taxonomy_version IS 'Benchmark taxonomy version. v1=original, v2=fixed pricing/context (Dec 2025)';

-- Update the best_runs_per_model view to include taxonomy_version
DROP VIEW IF EXISTS best_runs_per_model;
CREATE VIEW best_runs_per_model AS
SELECT DISTINCT ON (r.model_id)
    r.id AS run_id,
    r.model_id,
    m.name AS model_name,
    m.provider,
    r.seed,
    r.goal_framing,
    r.architecture,
    r.scaffolding,
    r.taxonomy_version,
    r.total_profit,
    r.total_cups_sold,
    r.final_cash,
    r.final_reputation,
    r.turn_count,
    r.error_count,
    r.started_at,
    r.completed_at
FROM runs r
JOIN models m ON r.model_id = m.id
WHERE r.completed_at IS NOT NULL
ORDER BY r.model_id, r.total_profit DESC;

-- Grant access to the view
GRANT SELECT ON best_runs_per_model TO anon, authenticated;

-- Update runs_with_model view to include taxonomy_version
DROP VIEW IF EXISTS runs_with_model;
CREATE VIEW runs_with_model AS
SELECT 
    r.id AS run_id,
    r.model_id,
    m.name AS model_name,
    m.provider,
    r.seed,
    r.goal_framing,
    r.architecture,
    r.scaffolding,
    r.taxonomy_version,
    r.total_profit,
    r.total_cups_sold,
    r.final_cash,
    r.final_reputation,
    r.turn_count,
    r.error_count,
    r.started_at,
    r.completed_at
FROM runs r
JOIN models m ON r.model_id = m.id
WHERE r.completed_at IS NOT NULL;

GRANT SELECT ON runs_with_model TO anon, authenticated;

-- Create a view for best runs per model per version (useful for comparing across versions)
CREATE VIEW best_runs_per_model_version AS
SELECT DISTINCT ON (r.model_id, r.taxonomy_version)
    r.id AS run_id,
    r.model_id,
    m.name AS model_name,
    m.provider,
    r.seed,
    r.goal_framing,
    r.architecture,
    r.scaffolding,
    r.taxonomy_version,
    r.total_profit,
    r.total_cups_sold,
    r.final_cash,
    r.final_reputation,
    r.turn_count,
    r.error_count,
    r.started_at,
    r.completed_at
FROM runs r
JOIN models m ON r.model_id = m.id
WHERE r.completed_at IS NOT NULL
ORDER BY r.model_id, r.taxonomy_version, r.total_profit DESC;

GRANT SELECT ON best_runs_per_model_version TO anon, authenticated;
