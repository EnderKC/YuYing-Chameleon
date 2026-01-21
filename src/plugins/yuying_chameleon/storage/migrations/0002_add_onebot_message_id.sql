-- 0002_add_onebot_message_id.sql: raw_messages 增加 onebot_message_id（平台 message_id）

ALTER TABLE raw_messages ADD COLUMN onebot_message_id INTEGER NULL;

