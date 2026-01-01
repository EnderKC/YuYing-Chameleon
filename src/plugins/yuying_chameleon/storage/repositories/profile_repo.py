"""UserProfile 的数据访问层（DAO）。"""

from __future__ import annotations

import time
from typing import Optional

from sqlalchemy import select, update

from ..models import UserProfile
from ..sqlalchemy_engine import get_session

from ...config import plugin_config

class ProfileRepository:
    """用户档案仓储。"""

    @staticmethod
    async def get_or_create(qq_id: str) -> UserProfile:
        """获取或创建用户档案。"""

        async with get_session() as session:
            result = await session.execute(select(UserProfile).where(UserProfile.qq_id == qq_id))
            profile = result.scalar_one_or_none()
            if not profile:
                profile = UserProfile(
                    qq_id=qq_id,
                    next_memory_at=plugin_config.yuying_memory_effective_count_threshold,
                )
                session.add(profile)
                await session.commit()
                await session.refresh(profile)
            return profile

    @staticmethod
    async def increment_effective_count(qq_id: str) -> UserProfile:
        """有效发言计数 +1（同时确保用户档案存在）。"""

        async with get_session() as session:
            result = await session.execute(select(UserProfile).where(UserProfile.qq_id == qq_id))
            profile = result.scalar_one_or_none()

            if not profile:
                profile = UserProfile(
                    qq_id=qq_id,
                    effective_count=0,
                    next_memory_at=plugin_config.yuying_memory_effective_count_threshold,
                )
                session.add(profile)
                await session.flush()

            profile.effective_count += 1
            profile.updated_at = int(time.time())
            await session.commit()
            await session.refresh(profile)
            return profile

    @staticmethod
    async def update_memory_status(qq_id: str, pending: bool) -> None:
        """更新“待抽取记忆”标记。"""

        # 确保用户存在（避免更新 0 行导致状态未落库）
        await ProfileRepository.get_or_create(qq_id)

        async with get_session() as session:
            stmt = (
                update(UserProfile)
                .where(UserProfile.qq_id == qq_id)
                .values(pending_memory=pending, updated_at=int(time.time()))
            )
            await session.execute(stmt)
            await session.commit()

    @staticmethod
    async def bump_next_memory_checkpoint(qq_id: str) -> None:
        """更新下一次“记忆抽取检查点”。"""

        profile = await ProfileRepository.get_or_create(qq_id)
        threshold = plugin_config.yuying_memory_effective_count_threshold
        next_at = max(profile.effective_count + threshold, profile.next_memory_at)

        async with get_session() as session:
            stmt = (
                update(UserProfile)
                .where(UserProfile.qq_id == qq_id)
                .values(next_memory_at=next_at, updated_at=int(time.time()))
            )
            await session.execute(stmt)
            await session.commit()

    @staticmethod
    async def update_last_memory_msg_id(qq_id: str, last_msg_id: int) -> None:
        """更新用户的 last_memory_msg_id。"""

        await ProfileRepository.get_or_create(qq_id)
        async with get_session() as session:
            stmt = (
                update(UserProfile)
                .where(UserProfile.qq_id == qq_id)
                .values(last_memory_msg_id=int(last_msg_id), updated_at=int(time.time()))
            )
            await session.execute(stmt)
            await session.commit()
