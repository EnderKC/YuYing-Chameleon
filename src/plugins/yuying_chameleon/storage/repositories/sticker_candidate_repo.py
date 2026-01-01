"""StickerCandidate 的数据访问层（DAO）。"""

from __future__ import annotations

import time
import json
from typing import Optional

from sqlalchemy import select, update

from ..models import StickerCandidate
from ..sqlalchemy_engine import get_session

class StickerCandidateRepository:
    """表情包候选仓储。"""

    @staticmethod
    async def get_by_fingerprint(fingerprint: str) -> Optional[StickerCandidate]:
        """按 fingerprint 获取候选记录。"""

        async with get_session() as session:
            result = await session.execute(select(StickerCandidate).where(StickerCandidate.fingerprint == fingerprint))
            return result.scalar_one_or_none()

    @staticmethod
    async def add(candidate: StickerCandidate) -> StickerCandidate:
        """新增候选记录。"""

        async with get_session() as session:
            session.add(candidate)
            await session.commit()
            await session.refresh(candidate)
            return candidate

    @staticmethod
    async def increment_seen_count(candidate_id: int) -> StickerCandidate:
        """seen_count +1，并刷新 last_seen_ts。"""

        async with get_session() as session:
            stmt = (
                update(StickerCandidate)
                .where(StickerCandidate.candidate_id == candidate_id)
                .values(seen_count=StickerCandidate.seen_count + 1, last_seen_ts=int(time.time()))
                .returning(StickerCandidate)
            )
            result = await session.execute(stmt)
            candidate = result.scalar_one()
            await session.commit()
            return candidate

    @staticmethod
    async def update_status(candidate_id: int, status: str) -> None:
        """更新候选状态（pending/promoted/ignored）。"""

        async with get_session() as session:
            stmt = (
                update(StickerCandidate)
                .where(StickerCandidate.candidate_id == candidate_id)
                .values(status=status)
            )
            await session.execute(stmt)
            await session.commit()

    @staticmethod
    async def append_source_qq_id(candidate_id: int, qq_id: str) -> None:
        """将来源 qq_id 追加到 source_qq_ids（JSON 数组字符串）。"""

        async with get_session() as session:
            result = await session.execute(
                select(StickerCandidate).where(StickerCandidate.candidate_id == candidate_id)
            )
            candidate = result.scalar_one_or_none()
            if not candidate:
                return

            raw = candidate.source_qq_ids or "[]"
            try:
                ids = json.loads(raw)
                if not isinstance(ids, list):
                    ids = []
            except Exception:
                ids = []
            if qq_id not in ids:
                ids.append(qq_id)

            stmt = (
                update(StickerCandidate)
                .where(StickerCandidate.candidate_id == candidate_id)
                .values(source_qq_ids=json.dumps(ids, ensure_ascii=False))
            )
            await session.execute(stmt)
            await session.commit()
