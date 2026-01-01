"""Sticker 的数据访问层（DAO）。"""

from __future__ import annotations

import time
from typing import List, Optional

from sqlalchemy import or_, select, update

from ..models import Sticker
from ..sqlalchemy_engine import get_session

class StickerRepository:
    """表情包仓储。"""

    @staticmethod
    async def get_by_id(sticker_id: str) -> Optional[Sticker]:
        """按 sticker_id 获取表情包。"""

        async with get_session() as session:
            result = await session.execute(select(Sticker).where(Sticker.sticker_id == sticker_id))
            return result.scalar_one_or_none()

    @staticmethod
    async def get_by_fingerprint(fingerprint: str) -> Optional[Sticker]:
        """按 fingerprint 获取表情包。"""

        async with get_session() as session:
            result = await session.execute(select(Sticker).where(Sticker.fingerprint == fingerprint))
            return result.scalar_one_or_none()

    @staticmethod
    async def add(sticker: Sticker) -> Sticker:
        """新增表情包记录。"""

        async with get_session() as session:
            session.add(sticker)
            await session.commit()
            await session.refresh(sticker)
            return sticker

    @staticmethod
    async def update_status(sticker_id: str, is_enabled: bool, is_banned: bool, ban_reason: Optional[str] = None) -> None:
        """更新表情包启用/封禁状态。"""

        async with get_session() as session:
            stmt = (
                update(Sticker)
                .where(Sticker.sticker_id == sticker_id)
                .values(is_enabled=is_enabled, is_banned=is_banned, ban_reason=ban_reason)
            )
            await session.execute(stmt)
            await session.commit()

    @staticmethod
    async def list_enabled_by_intent(intent: str, limit: int = 50) -> List[Sticker]:
        """按意图筛选可用表情包（intents 为逗号分隔字符串）。"""

        like = f"%{intent}%"
        async with get_session() as session:
            stmt = (
                select(Sticker)
                .where(
                    Sticker.is_enabled.is_(True),
                    Sticker.is_banned.is_(False),
                    Sticker.intents.is_not(None),
                    Sticker.intents.like(like),
                )
                .limit(limit)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    @staticmethod
    async def list_pending_tagging(limit: int = 20) -> List[Sticker]:
        """获取需要进行标签/违规判定的表情包（auto 包且缺少标签/意图）。"""

        async with get_session() as session:
            stmt = (
                select(Sticker)
                .where(
                    Sticker.pack == "auto",
                    Sticker.is_enabled.is_(True),
                    Sticker.is_banned.is_(False),
                    or_(Sticker.tags.is_(None), Sticker.tags == ""),
                )
                .limit(limit)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    @staticmethod
    async def update_metadata(
        sticker_id: str,
        *,
        tags: Optional[str],
        intents: Optional[str],
        style: Optional[str],
        is_enabled: bool,
        is_banned: bool,
        ban_reason: Optional[str],
    ) -> None:
        """更新表情包元信息（标签/意图/风格/封禁状态）。"""

        async with get_session() as session:
            stmt = (
                update(Sticker)
                .where(Sticker.sticker_id == sticker_id)
                .values(
                    tags=tags,
                    intents=intents,
                    style=style,
                    is_enabled=is_enabled,
                    is_banned=is_banned,
                    ban_reason=ban_reason,
                    updated_at=int(time.time()),
                )
            )
            await session.execute(stmt)
            await session.commit()
