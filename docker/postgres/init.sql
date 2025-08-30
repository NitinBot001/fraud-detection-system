-- Initialize database with extensions and optimizations

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Optimize PostgreSQL settings for fraud detection workload
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;

-- Create custom functions for fraud detection
CREATE OR REPLACE FUNCTION calculate_phone_similarity(phone1 TEXT, phone2 TEXT)
RETURNS FLOAT AS $$
BEGIN
    RETURN similarity(phone1, phone2);
END;
$$ LANGUAGE plpgsql;

-- Create custom aggregates for risk calculation
CREATE OR REPLACE FUNCTION risk_score_avg_state(state NUMERIC[], value NUMERIC, weight NUMERIC)
RETURNS NUMERIC[] AS $$
BEGIN
    IF state IS NULL THEN
        RETURN ARRAY[value * weight, weight];
    ELSE
        RETURN ARRAY[state[1] + (value * weight), state[2] + weight];
    END IF;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION risk_score_avg_final(state NUMERIC[])
RETURNS NUMERIC AS $$
BEGIN
    IF state IS NULL OR state[2] = 0 THEN
        RETURN 0;
    ELSE
        RETURN state[1] / state[2];
    END IF;
END;
$$ LANGUAGE plpgsql;

DROP AGGREGATE IF EXISTS weighted_avg(NUMERIC, NUMERIC);
CREATE AGGREGATE weighted_avg(NUMERIC, NUMERIC) (
    SFUNC = risk_score_avg_state,
    STYPE = NUMERIC[],
    FINALFUNC = risk_score_avg_final,
    INITCOND = '{0,0}'
);