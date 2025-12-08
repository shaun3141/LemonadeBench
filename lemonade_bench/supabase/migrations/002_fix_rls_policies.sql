-- Fix RLS Policies: Restrict INSERT/UPDATE to service_role only
-- The service_role bypasses RLS, so setting WITH CHECK (false) 
-- prevents anon/authenticated roles from inserting while allowing service_role

-- Drop existing permissive INSERT/UPDATE policies
DROP POLICY IF EXISTS "Allow service role insert on models" ON models;
DROP POLICY IF EXISTS "Allow service role insert on runs" ON runs;
DROP POLICY IF EXISTS "Allow service role update on runs" ON runs;
DROP POLICY IF EXISTS "Allow service role insert on turns" ON turns;

-- Create restrictive policies that deny INSERT/UPDATE for anon/authenticated
-- Service role bypasses RLS entirely, so it can still write
CREATE POLICY "Deny public insert on models"
    ON models FOR INSERT
    WITH CHECK (false);

CREATE POLICY "Deny public insert on runs"
    ON runs FOR INSERT
    WITH CHECK (false);

CREATE POLICY "Deny public update on runs"
    ON runs FOR UPDATE
    USING (false);

CREATE POLICY "Deny public insert on turns"
    ON turns FOR INSERT
    WITH CHECK (false);

-- Note: DELETE is not allowed by default since there's no policy for it

