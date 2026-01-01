"""StickerUsage 的数据访问层（DAO）。"""

from __future__ import annotations

import time
from typing import Optional

from sqlalchemy import desc, select

from ..models import StickerUsage
from ..sqlalchemy_engine import get_session


class StickerUsageRepository:
    """表情包使用记录仓储。"""

    @staticmethod
    async def add_usage(
        sticker_id: str,
        scene_type: str,
        scene_id: str,
        qq_id: Optional[str] = None,
    ) -> StickerUsage:
        """新增一条使用记录。"""

        usage = StickerUsage(
            sticker_id=sticker_id,
            scene_type=scene_type,
            scene_id=scene_id,
            qq_id=qq_id,
            used_at=int(time.time()),
        )
        async with get_session() as session:
            session.add(usage)
            await session.commit()
            await session.refresh(usage)
            return usage

    @staticmethod
    async def get_last_used_ts(scene_type: str, scene_id: str, sticker_id: str) -> int:
        """获取某个表情包在场景内最近一次使用时间戳（秒）。"""

        async with get_session() as session:
            stmt = (
                select(StickerUsage.used_at)
                .where(
                    StickerUsage.scene_type == scene_type,
                    StickerUsage.scene_id == scene_id,
                    StickerUsage.sticker_id == sticker_id,
                )
                .order_by(desc(StickerUsage.used_at))
                .limit(1)
            )
            result = await session.execute(stmt)
            ts = result.scalar_one_or_none()
            return int(ts or 0)
