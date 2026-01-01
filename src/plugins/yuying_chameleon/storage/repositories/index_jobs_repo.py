"""IndexJob 的数据访问层（DAO）。"""

from __future__ import annotations

import time
from typing import List, Optional

from sqlalchemy import or_, select, update

from ..models import IndexJob
from ..sqlalchemy_engine import get_session

class IndexJobRepository:
    """索引任务仓储。"""

    @staticmethod
    async def add(job: IndexJob) -> IndexJob:
        """新增索引任务。"""

        async with get_session() as session:
            session.add(job)
            await session.commit()
            await session.refresh(job)
            return job

    @staticmethod
    async def get_pending_jobs(limit: int = 10, item_type: Optional[str] = None) -> List[IndexJob]:
        """获取待处理/可重试的任务。"""

        async with get_session() as session:
            current_ts = int(time.time())
            stmt = (
                select(IndexJob)
                .where(
                    or_(IndexJob.status == "pending", IndexJob.status == "failed"),
                    IndexJob.next_retry_ts <= current_ts
                )
            )
            if item_type:
                stmt = stmt.where(IndexJob.item_type == item_type)
            stmt = stmt.order_by(IndexJob.created_at.asc()).limit(limit)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    @staticmethod
    async def get_pending_jobs_for_types(item_types: List[str], limit: int = 10) -> List[IndexJob]:
        """获取指定类型集合的待处理/可重试任务。"""

        if not item_types:
            return []
        async with get_session() as session:
            current_ts = int(time.time())
            stmt = (
                select(IndexJob)
                .where(
                    or_(IndexJob.status == "pending", IndexJob.status == "failed"),
                    IndexJob.next_retry_ts <= current_ts,
                    IndexJob.item_type.in_(item_types),
                )
                .order_by(IndexJob.created_at.asc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    @staticmethod
    async def mark_processing(job_id: int) -> None:
        """将任务标记为 processing。"""

        async with get_session() as session:
            stmt = (
                update(IndexJob)
                .where(IndexJob.job_id == job_id)
                .values(status="processing", updated_at=int(time.time()))
            )
            await session.execute(stmt)
            await session.commit()

    @staticmethod
    async def update_status(job_id: int, status: str, next_retry_ts: int = 0) -> None:
        """更新任务状态，并在失败时递增 retry_count。"""

        async with get_session() as session:
            stmt = (
                update(IndexJob)
                .where(IndexJob.job_id == job_id)
                .values(status=status, next_retry_ts=next_retry_ts, updated_at=int(time.time()))
            )
            if status == "failed":
                stmt = stmt.values(retry_count=IndexJob.retry_count + 1)
            
            await session.execute(stmt)
            await session.commit()

    @staticmethod
    def compute_backoff_ts(retry_count: int) -> int:
        """计算失败后的下一次重试时间戳（指数退避，最大 1 小时）。"""

        now_ts = int(time.time())
        step = min(3600, 5 * (2 ** max(0, int(retry_count))))
        return now_ts + step
