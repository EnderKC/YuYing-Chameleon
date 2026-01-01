"""MediaCache 的数据访问层（DAO）。"""

from __future__ import annotations

import time
from typing import Optional

from sqlalchemy import select, update

from ..models import MediaCache
from ..sqlalchemy_engine import get_session

class MediaCacheRepository:
    """媒体缓存仓储。"""

    @staticmethod
    async def get(media_key: str) -> Optional[MediaCache]:
        """按 media_key 获取缓存记录。"""

        async with get_session() as session:
            result = await session.execute(select(MediaCache).where(MediaCache.media_key == media_key))
            return result.scalar_one_or_none()

    @staticmethod
    async def add(media_cache: MediaCache) -> MediaCache:
        """新增缓存记录。"""

        async with get_session() as session:
            session.add(media_cache)
            await session.commit()
            await session.refresh(media_cache)
            return media_cache

    @staticmethod
    async def upsert(
        media_key: str,
        *,
        media_type: str,
        caption: str,
        tags: Optional[str] = None,
        ocr_text: Optional[str] = None,
    ) -> MediaCache:
        """插入或更新 media_cache。"""

        existing = await MediaCacheRepository.get(media_key)
        now_ts = int(time.time())
        if existing:
            async with get_session() as session:
                stmt = (
                    update(MediaCache)
                    .where(MediaCache.media_key == media_key)
                    .values(
                        media_type=media_type,
                        caption=caption,
                        tags=tags,
                        ocr_text=ocr_text,
                        updated_at=now_ts,
                    )
                )
                await session.execute(stmt)
                await session.commit()
            return (await MediaCacheRepository.get(media_key)) or existing

        record = MediaCache(
            media_key=media_key,
            media_type=media_type,
            caption=caption,
            tags=tags,
            ocr_text=ocr_text,
            created_at=now_ts,
            updated_at=now_ts,
        )
        return await MediaCacheRepository.add(record)
