-- Add error_message column to runs table for storing failure reasons
-- This captures exceptions/errors that cause a run to fail before completion

-- Add the column (nullable since most runs succeed)
ALTER TABLE runs ADD COLUMN IF NOT EXISTS error_message TEXT;

-- Add index for querying failed runs
CREATE INDEX IF NOT EXISTS idx_runs_error_message ON runs(error_message) WHERE error_message IS NOT NULL;

-- Add comment
COMMENT ON COLUMN runs.error_message IS 'Error message if run failed with an exception (NULL for successful runs)';

-- Update the views to include error_message
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
    r.error_message,
    r.started_at,
    r.completed_at
FROM runs r
JOIN models m ON r.model_id = m.id;

GRANT SELECT ON runs_with_model TO anon, authenticated;
