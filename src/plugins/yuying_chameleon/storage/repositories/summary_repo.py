"""Summary 的数据访问层（DAO）。"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import select

from ..models import Summary
from ..sqlalchemy_engine import get_session

class SummaryRepository:
    """摘要仓储。"""

    @staticmethod
    async def add(summary: Summary) -> Summary:
        """新增摘要。"""

        async with get_session() as session:
            session.add(summary)
            await session.commit()
            await session.refresh(summary)
            return summary

    @staticmethod
    async def get_latest(scene_type: str, scene_id: str) -> Optional[Summary]:
        """获取指定场景最新的一条摘要。"""

        async with get_session() as session:
            stmt = (
                select(Summary)
                .where(Summary.scene_type == scene_type, Summary.scene_id == scene_id)
                .order_by(Summary.window_end_ts.desc())
                .limit(1)
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    @staticmethod
    async def get_by_id(summary_id: int) -> Optional[Summary]:
        """按 id 获取摘要记录。"""

        async with get_session() as session:
            result = await session.execute(select(Summary).where(Summary.id == summary_id))
            return result.scalar_one_or_none()
