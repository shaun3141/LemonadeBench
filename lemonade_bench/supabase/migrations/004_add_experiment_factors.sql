-- Add experimental factor columns to runs table
-- These track which experimental conditions each run belongs to

-- Goal framing condition (Section 4.4 of the paper)
-- Values: baseline, aggressive, conservative, competitive, survival, growth
ALTER TABLE runs ADD COLUMN IF NOT EXISTS goal_framing TEXT DEFAULT 'baseline';

-- Agent architecture (Section 4.2 of the paper)
-- Values: react, plan_act, act_reflect, full
ALTER TABLE runs ADD COLUMN IF NOT EXISTS architecture TEXT DEFAULT 'react';

-- Cognitive scaffolding (Section 4.3 of the paper)
-- Values: none, calculator, math_prompt, code_interpreter
ALTER TABLE runs ADD COLUMN IF NOT EXISTS scaffolding TEXT DEFAULT 'none';

-- Add check constraints to ensure valid values
ALTER TABLE runs ADD CONSTRAINT chk_goal_framing 
    CHECK (goal_framing IN ('baseline', 'aggressive', 'conservative', 'competitive', 'survival', 'growth'));

ALTER TABLE runs ADD CONSTRAINT chk_architecture 
    CHECK (architecture IN ('react', 'plan_act', 'act_reflect', 'full'));

ALTER TABLE runs ADD CONSTRAINT chk_scaffolding 
    CHECK (scaffolding IN ('none', 'calculator', 'math_prompt', 'code_interpreter'));

-- Add indexes for filtering by experimental conditions
CREATE INDEX IF NOT EXISTS idx_runs_goal_framing ON runs(goal_framing);
CREATE INDEX IF NOT EXISTS idx_runs_architecture ON runs(architecture);
CREATE INDEX IF NOT EXISTS idx_runs_scaffolding ON runs(scaffolding);

-- Composite index for common query patterns
CREATE INDEX IF NOT EXISTS idx_runs_experiment_factors ON runs(goal_framing, architecture, scaffolding);

-- Comments for documentation
COMMENT ON COLUMN runs.goal_framing IS 'Goal framing prompt condition: baseline, aggressive, conservative, competitive, survival, growth';
COMMENT ON COLUMN runs.architecture IS 'Agent architecture: react, plan_act, act_reflect, full';
COMMENT ON COLUMN runs.scaffolding IS 'Cognitive scaffolding: none, calculator, math_prompt, code_interpreter';

-- Update the best_runs_per_model view to include experiment factors
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

-- Create a new view for filtering by experiment type
-- This helps with leaderboard queries by experiment
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

