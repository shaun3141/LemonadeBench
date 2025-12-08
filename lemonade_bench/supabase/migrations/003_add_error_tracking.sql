-- Add error tracking columns to turns and runs tables
-- This supports the action validation error feedback system

-- Add error tracking columns to turns table
ALTER TABLE turns ADD COLUMN IF NOT EXISTS is_error BOOLEAN DEFAULT FALSE;
ALTER TABLE turns ADD COLUMN IF NOT EXISTS error_messages TEXT[];

-- Add error count to runs for quick aggregation
ALTER TABLE runs ADD COLUMN IF NOT EXISTS error_count INTEGER DEFAULT 0;

-- Create index for finding error turns
CREATE INDEX IF NOT EXISTS idx_turns_is_error ON turns(run_id, is_error) WHERE is_error = TRUE;

-- Comment on new columns
COMMENT ON COLUMN turns.is_error IS 'True if this turn was an invalid action attempt';
COMMENT ON COLUMN turns.error_messages IS 'Array of validation error messages if is_error is true';
COMMENT ON COLUMN runs.error_count IS 'Total number of invalid action attempts in this run';

