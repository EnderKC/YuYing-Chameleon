-- 0001_init.sql: 初始化全部核心表（SQLite）

PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS user_profile (
  qq_id TEXT PRIMARY KEY,
  effective_count INTEGER NOT NULL DEFAULT 0,
  next_memory_at INTEGER NOT NULL DEFAULT 50,
  last_memory_msg_id INTEGER NULL,
  pending_memory INTEGER NOT NULL DEFAULT 0,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS raw_messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  qq_id TEXT NOT NULL,
  scene_type TEXT NOT NULL,
  scene_id TEXT NOT NULL,
  timestamp INTEGER NOT NULL,
  msg_type TEXT NOT NULL,
  content TEXT NOT NULL,
  raw_ref TEXT NULL,
  reply_to_msg_id INTEGER NULL,
  mentioned_bot INTEGER NOT NULL DEFAULT 0,
  is_effective INTEGER NOT NULL DEFAULT 0,
  is_bot INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_raw_qq_ts ON raw_messages(qq_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_raw_scene_ts ON raw_messages(scene_type, scene_id, timestamp);

CREATE TABLE IF NOT EXISTS summaries (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  scene_type TEXT NOT NULL,
  scene_id TEXT NOT NULL,
  window_start_ts INTEGER NOT NULL,
  window_end_ts INTEGER NOT NULL,
  summary_text TEXT NOT NULL,
  topic_state_json TEXT NULL
);

CREATE INDEX IF NOT EXISTS idx_sum_scene_end ON summaries(scene_type, scene_id, window_end_ts);

CREATE TABLE IF NOT EXISTS memories (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  qq_id TEXT NOT NULL,
  tier TEXT NOT NULL,
  type TEXT NOT NULL,
  content TEXT NOT NULL,
  confidence REAL NOT NULL,
  status TEXT NOT NULL,
  visibility TEXT NOT NULL,
  scope_scene_id TEXT NULL,
  ttl_days INTEGER NULL,
  source_memory_ids TEXT NULL,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_mem_qq_tier ON memories(qq_id, tier, updated_at);
CREATE INDEX IF NOT EXISTS idx_mem_qq_type ON memories(qq_id, type, updated_at);

CREATE TABLE IF NOT EXISTS memory_evidence (
  memory_id INTEGER NOT NULL,
  msg_id INTEGER NOT NULL,
  PRIMARY KEY (memory_id, msg_id)
);

CREATE TABLE IF NOT EXISTS media_cache (
  media_key TEXT PRIMARY KEY,
  media_type TEXT NOT NULL,
  caption TEXT NOT NULL,
  tags TEXT NULL,
  ocr_text TEXT NULL,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS bot_rate_limit (
  scene_type TEXT NOT NULL,
  scene_id TEXT NOT NULL,
  last_sent_ts INTEGER NOT NULL DEFAULT 0,
  cooldown_until_ts INTEGER NOT NULL DEFAULT 0,
  recent_bot_msg_count INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (scene_type, scene_id)
);

CREATE TABLE IF NOT EXISTS stickers (
  sticker_id TEXT PRIMARY KEY,
  pack TEXT NOT NULL,
  file_path TEXT NOT NULL,
  file_sha256 TEXT NOT NULL,
  phash TEXT NOT NULL,
  ocr_text TEXT NULL,
  fingerprint TEXT NOT NULL,
  name TEXT NULL,
  tags TEXT NULL,
  intents TEXT NULL,
  style TEXT NULL,
  is_enabled INTEGER NOT NULL DEFAULT 1,
  is_banned INTEGER NOT NULL DEFAULT 0,
  ban_reason TEXT NULL,
  source_scene_id TEXT NULL,
  source_qq_id TEXT NULL,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_stk_fp ON stickers(fingerprint);
CREATE INDEX IF NOT EXISTS idx_stk_enabled ON stickers(is_enabled);

CREATE TABLE IF NOT EXISTS sticker_candidates (
  candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
  fingerprint TEXT NOT NULL,
  phash TEXT NOT NULL,
  ocr_text TEXT NULL,
  sha256_sample TEXT NOT NULL,
  sample_file_path TEXT NOT NULL,
  scene_id TEXT NOT NULL,
  first_seen_ts INTEGER NOT NULL,
  last_seen_ts INTEGER NOT NULL,
  seen_count INTEGER NOT NULL DEFAULT 1,
  status TEXT NOT NULL,
  source_qq_ids TEXT NULL
);

CREATE INDEX IF NOT EXISTS idx_sc_fp ON sticker_candidates(fingerprint);
CREATE INDEX IF NOT EXISTS idx_sc_status ON sticker_candidates(status);

CREATE TABLE IF NOT EXISTS sticker_usage (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  sticker_id TEXT NOT NULL,
  scene_type TEXT NOT NULL,
  scene_id TEXT NOT NULL,
  qq_id TEXT NULL,
  used_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS index_jobs (
  job_id INTEGER PRIMARY KEY AUTOINCREMENT,
  item_type TEXT NOT NULL,
  ref_id TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  status TEXT NOT NULL,
  retry_count INTEGER NOT NULL DEFAULT 0,
  next_retry_ts INTEGER NOT NULL DEFAULT 0,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON index_jobs(status, next_retry_ts);

