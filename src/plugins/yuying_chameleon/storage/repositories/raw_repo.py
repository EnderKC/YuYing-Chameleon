"""RawMessage 的数据访问层（DAO）。"""

from __future__ import annotations

import time
from typing import List, Optional

from sqlalchemy import select, update
from sqlalchemy import func

from ..models import RawMessage
from ..sqlalchemy_engine import get_session


class RawRepository:
    """原始消息仓储。"""

    @staticmethod
    async def add(message: RawMessage) -> RawMessage:
        """新增一条原始消息记录。

        参数：
            message: 待写入的 ORM 对象。

        返回：
            RawMessage: 写入并刷新后的对象（包含自增 id）。
        """

        async with get_session() as session:
            session.add(message)
            await session.commit()
            await session.refresh(message)
            return message

    @staticmethod
    async def get_by_id(msg_id: int) -> Optional[RawMessage]:
        """按消息 id 获取原始消息。"""

        async with get_session() as session:
            result = await session.execute(
                select(RawMessage).where(RawMessage.id == msg_id)
            )
            return result.scalar_one_or_none()

    @staticmethod
    async def update_content(msg_id: int, content: str) -> None:
        """更新某条消息的内容（用于图片说明等异步补全）。"""

        async with get_session() as session:
            stmt = (
                update(RawMessage)
                .where(RawMessage.id == msg_id)
                .values(content=content)
            )
            await session.execute(stmt)
            await session.commit()

    @staticmethod
    async def list_with_image_marker(media_key: str, limit: int = 50) -> List[RawMessage]:
        """查找包含指定图片短标识的消息（用于补全图片说明）。"""

        token = f"[image:{media_key}]"
        async with get_session() as session:
            stmt = (
                select(RawMessage)
                .where(RawMessage.content.like(f"%{token}%"))
                .order_by(RawMessage.id.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    @staticmethod
    async def get_recent_by_scene(
        scene_type: str,
        scene_id: str,
        limit: int = 2,
    ) -> List[RawMessage]:
        """获取某个场景最近的若干条消息（按时间倒序）。"""

        async with get_session() as session:
            stmt = (
                select(RawMessage)
                .where(RawMessage.scene_type == scene_type, RawMessage.scene_id == scene_id)
                .order_by(RawMessage.timestamp.desc(), RawMessage.id.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    @staticmethod
    async def get_recent_by_scene_with_roles(
        scene_type: str,
        scene_id: str,
        limit: int = 20,
    ) -> List[RawMessage]:
        """获取场景最近消息（用于 Hybrid Query 组装）。"""

        return await RawRepository.get_recent_by_scene(scene_type, scene_id, limit=limit)

    @staticmethod
    async def get_messages_by_scene_time_range(
        scene_type: str,
        scene_id: str,
        start_ts: int,
        end_ts: int,
        limit: int = 200,
    ) -> List[RawMessage]:
        """按时间范围获取某个场景的消息（时间正序）。"""

        async with get_session() as session:
            stmt = (
                select(RawMessage)
                .where(
                    RawMessage.scene_type == scene_type,
                    RawMessage.scene_id == scene_id,
                    RawMessage.timestamp >= start_ts,
                    RawMessage.timestamp <= end_ts,
                )
                .order_by(RawMessage.timestamp.asc(), RawMessage.id.asc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    @staticmethod
    async def get_user_messages_since(
        qq_id: str,
        after_msg_id: int,
        overlap: int,
        limit: int,
    ) -> List[RawMessage]:
        """获取用户在某条消息之后的消息列表，并包含 overlap 条重叠窗口。"""

        effective_after = max(0, int(after_msg_id) - max(0, int(overlap)))
        async with get_session() as session:
            stmt = (
                select(RawMessage)
                .where(RawMessage.qq_id == qq_id, RawMessage.id > effective_after)
                .order_by(RawMessage.id.asc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    @staticmethod
    async def count_scene_messages_since(scene_type: str, scene_id: str, since_ts: int) -> int:
        """统计某个场景在指定时间戳之后的消息数量。"""

        async with get_session() as session:
            stmt = (
                select(func.count())
                .select_from(RawMessage)
                .where(
                    RawMessage.scene_type == scene_type,
                    RawMessage.scene_id == scene_id,
                    RawMessage.timestamp >= since_ts,
                )
            )
            result = await session.execute(stmt)
            return int(result.scalar_one() or 0)

    @staticmethod
    async def count_scene_messages_after_id(
        scene_type: str,
        scene_id: str,
        after_id: int,
    ) -> int:
        """统计某个场景在某条 raw_messages.id 之后的消息数量（不含该条）。"""

        after_id = int(after_id or 0)
        async with get_session() as session:
            stmt = (
                select(func.count())
                .select_from(RawMessage)
                .where(
                    RawMessage.scene_type == scene_type,
                    RawMessage.scene_id == scene_id,
                    RawMessage.id > after_id,
                )
            )
            result = await session.execute(stmt)
            return int(result.scalar_one() or 0)
