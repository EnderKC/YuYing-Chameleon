PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS bot_personality_memories (
  id INTEGER PRIMARY KEY AUTOINCREMENT,

  tier TEXT NOT NULL,
  memory_type TEXT NOT NULL,
  scope_type TEXT NOT NULL,
  scope_id TEXT NOT NULL DEFAULT '',
  memory_key TEXT NOT NULL,

  memory_date TEXT NOT NULL,
  window_start_ts INTEGER NOT NULL,
  window_end_ts INTEGER NOT NULL,

  title TEXT NOT NULL,
  content TEXT NOT NULL,
  action_hint TEXT NULL,

  confidence REAL NOT NULL DEFAULT 0.5,
  importance REAL NOT NULL DEFAULT 0.5,

  emotion_label TEXT NULL,
  emotion_valence REAL NULL,
  decay_weight REAL NULL,
  decay_half_life_hours REAL NULL,

  evidence_json TEXT NULL,

  run_id TEXT NOT NULL,
  model TEXT NULL,
  prompt_version TEXT NULL,

  created_at_ts INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER)),
  updated_at_ts INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER)),
  deleted_at_ts INTEGER NULL,

  CONSTRAINT uq_bot_personality_key UNIQUE (
    tier, scope_type, scope_id, memory_type, memory_key
  )
);

CREATE INDEX IF NOT EXISTS idx_bpm_lookup
  ON bot_personality_memories (tier, scope_type, scope_id, memory_type, updated_at_ts);

CREATE INDEX IF NOT EXISTS idx_bpm_tier_window_end
  ON bot_personality_memories (tier, window_end_ts);

CREATE INDEX IF NOT EXISTS idx_bpm_tier_type_updated
  ON bot_personality_memories (tier, memory_type, updated_at_ts);
