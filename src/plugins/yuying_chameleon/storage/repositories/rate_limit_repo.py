"""BotRateLimit 的数据访问层（DAO）。"""

from __future__ import annotations

import time

from sqlalchemy import select, update

from ..models import BotRateLimit
from ..sqlalchemy_engine import get_session


class RateLimitRepository:
    """机器人频率状态仓储（按场景维度）。"""

    @staticmethod
    async def get_or_create(scene_type: str, scene_id: str) -> BotRateLimit:
        """获取或创建场景频率状态。"""

        async with get_session() as session:
            result = await session.execute(
                select(BotRateLimit).where(
                    BotRateLimit.scene_type == scene_type,
                    BotRateLimit.scene_id == scene_id,
                )
            )
            state = result.scalar_one_or_none()
            if state:
                return state

            state = BotRateLimit(scene_type=scene_type, scene_id=scene_id)
            session.add(state)
            await session.commit()
            await session.refresh(state)
            return state

    @staticmethod
    async def mark_sent(scene_type: str, scene_id: str, cooldown_seconds: int) -> None:
        """记录一次机器人已发送消息，并刷新冷却截止时间。"""

        now_ts = int(time.time())
        cooldown_until_ts = now_ts + max(0, int(cooldown_seconds))

        # 先确保存在（避免更新 0 行）
        await RateLimitRepository.get_or_create(scene_type, scene_id)

        async with get_session() as session:
            stmt = (
                update(BotRateLimit)
                .where(
                    BotRateLimit.scene_type == scene_type,
                    BotRateLimit.scene_id == scene_id,
                )
                .values(
                    last_sent_ts=now_ts,
                    cooldown_until_ts=cooldown_until_ts,
                    recent_bot_msg_count=BotRateLimit.recent_bot_msg_count + 1,
                )
            )
            await session.execute(stmt)
            await session.commit()
