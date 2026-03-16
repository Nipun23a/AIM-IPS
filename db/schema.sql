-- ============================================================
-- AIM-IPS  |  PostgreSQL Attack Events Schema
-- ============================================================
-- Run once:
--   psql -U postgres -c "CREATE DATABASE aimips;"
--   psql -U postgres -d aimips -f db/schema.sql
-- ============================================================

-- ── Main events table ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS attack_events (
    id              BIGSERIAL           PRIMARY KEY,

    -- Request identity
    request_id      VARCHAR(36)         NOT NULL,
    ip              VARCHAR(45)         NOT NULL,       -- supports IPv6
    method          VARCHAR(10)         NOT NULL DEFAULT 'GET',
    path            TEXT                NOT NULL DEFAULT '/',

    -- Timing (epoch seconds from pipeline + derived timestamptz column)
    timestamp       DOUBLE PRECISION    NOT NULL,
    event_time      TIMESTAMPTZ         GENERATED ALWAYS AS (
                        to_timestamp(timestamp)
                    ) STORED,

    -- Pipeline decision
    final_score     REAL                NOT NULL DEFAULT 0,
    action          VARCHAR(20)         NOT NULL DEFAULT 'ALLOW',
    block_reason    TEXT                DEFAULT '',
    short_circuited BOOLEAN             NOT NULL DEFAULT FALSE,
    network_score   REAL                NOT NULL DEFAULT 0,
    latency_ms      REAL,
    best_label      VARCHAR(100)        DEFAULT 'unknown',

    -- Per-layer scores (flat columns for fast aggregation)
    regex_conf      REAL                DEFAULT 0,
    app_lgbm_score  REAL                DEFAULT 0,
    net_lgbm_score  REAL                DEFAULT 0,
    deep_anomaly    REAL                DEFAULT 0,

    -- Full layer detail (flexible JSON)
    layer_scores    JSONB               NOT NULL DEFAULT '[]',

    created_at      TIMESTAMPTZ         NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_request_id UNIQUE (request_id)
);

-- ── Indexes for dashboard queries ─────────────────────────────
CREATE INDEX IF NOT EXISTS idx_ae_ip          ON attack_events (ip);
CREATE INDEX IF NOT EXISTS idx_ae_action      ON attack_events (action);
CREATE INDEX IF NOT EXISTS idx_ae_event_time  ON attack_events (event_time DESC);
CREATE INDEX IF NOT EXISTS idx_ae_timestamp   ON attack_events (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ae_best_label  ON attack_events (best_label);
-- Composite: filter by action within a time window (used by stats API)
CREATE INDEX IF NOT EXISTS idx_ae_action_time ON attack_events (action, event_time DESC);

-- ── Utility views ─────────────────────────────────────────────

-- Last 24 hours (dashboard default window)
CREATE OR REPLACE VIEW v_events_24h AS
SELECT *
FROM   attack_events
WHERE  event_time >= NOW() - INTERVAL '24 hours';

-- RPM buckets — last 30 minutes (30 x 1-minute slots)
CREATE OR REPLACE VIEW v_rpm_30min AS
SELECT
    FLOOR((EXTRACT(EPOCH FROM NOW()) - timestamp) / 60)::INT  AS mins_ago,
    COUNT(*)                                                   AS total,
    COUNT(*) FILTER (WHERE action = 'BLOCK')                  AS blocked
FROM   attack_events
WHERE  timestamp >= EXTRACT(EPOCH FROM NOW()) - 1800
GROUP  BY 1
ORDER  BY 1;

-- Top threat IPs (last 24 h)
CREATE OR REPLACE VIEW v_top_ips AS
SELECT
    ip,
    COUNT(*)                                              AS total_requests,
    COUNT(*) FILTER (WHERE action = 'BLOCK')             AS blocked,
    MAX(final_score)                                      AS max_score,
    MAX(timestamp)  FILTER (WHERE action = 'BLOCK')      AS last_attack_ts
FROM   v_events_24h
GROUP  BY ip
ORDER  BY blocked DESC, max_score DESC;

-- Attack type breakdown (last 24 h)
CREATE OR REPLACE VIEW v_attack_types AS
SELECT
    best_label,
    COUNT(*) AS cnt
FROM   v_events_24h
WHERE  best_label NOT IN ('clean', 'norm', 'normal', 'unknown', '')
GROUP  BY best_label
ORDER  BY cnt DESC;

-- Detection layer breakdown (unnest JSONB, last 24 h)
CREATE OR REPLACE VIEW v_layer_counts AS
SELECT
    ls->>'layer'  AS layer,
    COUNT(*)      AS cnt
FROM   v_events_24h,
       jsonb_array_elements(layer_scores) AS ls
WHERE  (ls->>'triggered')::boolean = TRUE
GROUP  BY 1
ORDER  BY cnt DESC;
