"""简易 SQLite migrations 运行器（不依赖 Alembic）。

设计目标：
- 满足“必须提供 migrations”的交付要求；
- 使用 SQL 文件，适配 SQLite；
- 可重复运行：已执行过的 migration 会被跳过。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

from nonebot import logger
from sqlalchemy import text

from .sqlalchemy_engine import engine


def _migrations_dir() -> Path:
    return Path(__file__).resolve().parent / "migrations"


def _iter_migration_files() -> Iterable[Path]:
    d = _migrations_dir()
    if not d.exists() or not d.is_dir():
        return []
    files = [p for p in d.iterdir() if p.is_file() and p.name.endswith(".sql")]
    files.sort(key=lambda p: p.name)
    return files


def _split_sql(sql: str) -> list[str]:
    """将一个 .sql 文件内容拆成可执行语句列表（简单分号拆分，适用于本项目的迁移文件）。"""

    # 移除整行注释，避免把注释和 SQL 拼到同一个 statement 里导致驱动报错
    cleaned_lines: list[str] = []
    for line in (sql or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("--"):
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)

    parts: list[str] = []
    for raw in cleaned.split(";"):
        stmt = raw.strip()
        if stmt:
            parts.append(stmt)
    return parts


async def run_migrations() -> None:
    """执行 migrations（在插件启动时调用）。"""

    files = list(_iter_migration_files())
    if not files:
        logger.warning("未发现 migrations 文件，将回退为 create_all。")
        return

    async with engine.begin() as conn:
        await conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                  version TEXT PRIMARY KEY,
                  applied_at INTEGER NOT NULL
                )
                """
            )
        )

        existing = await conn.execute(text("SELECT version FROM schema_migrations"))
        applied = {str(row[0]) for row in existing.fetchall()}

        for f in files:
            version = f.name
            if version in applied:
                continue
            sql = f.read_text(encoding="utf-8").strip()
            if not sql:
                continue
            logger.info(f"Applying migration: {version}")
            for stmt in _split_sql(sql):
                await conn.exec_driver_sql(stmt)
            await conn.execute(
                text("INSERT INTO schema_migrations(version, applied_at) VALUES (:v, :ts)"),
                {"v": version, "ts": int(time.time())},
            )
