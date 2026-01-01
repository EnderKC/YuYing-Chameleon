"""Memory 的数据访问层（DAO）。"""

from __future__ import annotations

import time
from typing import List, Optional

from sqlalchemy import insert
from sqlalchemy import select, update

from ..models import Memory, MemoryEvidence
from ..sqlalchemy_engine import get_session

class MemoryRepository:
    """记忆仓储。"""

    @staticmethod
    async def add(memory: Memory) -> Memory:
        """新增记忆。"""

        async with get_session() as session:
            session.add(memory)
            await session.commit()
            await session.refresh(memory)
            return memory

    @staticmethod
    async def get_by_id(memory_id: int) -> Optional[Memory]:
        """按 id 获取记忆。"""

        async with get_session() as session:
            result = await session.execute(select(Memory).where(Memory.id == memory_id))
            return result.scalar_one_or_none()

    @staticmethod
    async def get_by_qq_id(qq_id: str, tier: Optional[str] = None) -> List[Memory]:
        """按用户查询记忆（可按层级过滤）。"""

        async with get_session() as session:
            stmt = select(Memory).where(Memory.qq_id == qq_id)
            if tier:
                stmt = stmt.where(Memory.tier == tier)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    @staticmethod
    async def get_active_memories(qq_id: str) -> List[Memory]:
        """获取用户的 active 层记忆。"""
        return await MemoryRepository.get_by_qq_id(qq_id, tier="active")

    @staticmethod
    async def get_core_memories(qq_id: str) -> List[Memory]:
        """获取用户的 core 层记忆。"""
        return await MemoryRepository.get_by_qq_id(qq_id, tier="core")

    @staticmethod
    async def list_active_for_user(qq_id: str) -> List[Memory]:
        """获取用户的 active 记忆（按更新时间倒序）。"""

        async with get_session() as session:
            stmt = (
                select(Memory)
                .where(Memory.qq_id == qq_id, Memory.tier == "active")
                .order_by(Memory.updated_at.desc())
                .limit(200)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    @staticmethod
    async def update_tier(memory_id: int, tier: str) -> None:
        """更新记忆层级。"""

        async with get_session() as session:
            stmt = (
                update(Memory)
                .where(Memory.id == memory_id)
                .values(tier=tier, updated_at=int(time.time()))
            )
            await session.execute(stmt)
            await session.commit()

    @staticmethod
    async def update_fields(memory_id: int, **fields) -> None:
        """更新记忆的部分字段。"""

        if not fields:
            return
        fields["updated_at"] = int(time.time())
        async with get_session() as session:
            stmt = update(Memory).where(Memory.id == memory_id).values(**fields)
            await session.execute(stmt)
            await session.commit()

    @staticmethod
    async def get_expired_active(current_ts: int) -> List[Memory]:
        """获取已过期的 active 记忆。"""

        async with get_session() as session:
            stmt = select(Memory).where(
                Memory.tier == "active",
                Memory.ttl_days.is_not(None),
                (Memory.created_at + Memory.ttl_days * 86400) < current_ts
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    @staticmethod
    async def add_evidence(memory_id: int, msg_ids: List[int]) -> None:
        """为某条记忆关联证据消息。"""

        cleaned: List[int] = []
        for x in msg_ids:
            try:
                v = int(x)
            except Exception:
                continue
            if v > 0 and v not in cleaned:
                cleaned.append(v)
        if not cleaned:
            return

        async with get_session() as session:
            values = [{"memory_id": int(memory_id), "msg_id": int(msg_id)} for msg_id in cleaned]
            # SQLite 复合主键冲突时忽略，避免抛异常中断主流程
            stmt = insert(MemoryEvidence.__table__).values(values).prefix_with("OR IGNORE")
            await session.execute(stmt)
            await session.commit()

    @staticmethod
    async def archive_memory(memory_id: int) -> None:
        """将记忆归档：tier=archive 且 status=archived。"""

        async with get_session() as session:
            stmt = (
                update(Memory)
                .where(Memory.id == memory_id)
                .values(tier="archive", status="archived", updated_at=int(time.time()))
            )
            await session.execute(stmt)
            await session.commit()

    @staticmethod
    async def list_recent_archive(qq_id: str, days: int) -> List[Memory]:
        """获取最近 N 天的 archive 记忆。"""

        min_ts = int(time.time()) - max(0, int(days)) * 86400
        async with get_session() as session:
            stmt = (
                select(Memory)
                .where(Memory.qq_id == qq_id, Memory.tier == "archive", Memory.updated_at >= min_ts)
                .order_by(Memory.updated_at.desc())
                .limit(200)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    @staticmethod
    async def replace_core_memories(qq_id: str, core_memories: List[Memory]) -> None:
        """覆盖更新用户的 core 记忆集合。"""

        async with get_session() as session:
            old = await session.execute(select(Memory).where(Memory.qq_id == qq_id, Memory.tier == "core"))
            for item in old.scalars().all():
                maybe_coro = session.delete(item)
                if hasattr(maybe_coro, "__await__"):
                    await maybe_coro
            for mem in core_memories:
                session.add(mem)
            await session.commit()
