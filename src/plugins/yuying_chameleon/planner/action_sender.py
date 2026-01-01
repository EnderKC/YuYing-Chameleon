"""动作发送器：按人类节奏分段发送，并落库机器人消息。"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Dict, List, Optional

from nonebot.adapters.onebot.v11 import Bot
from nonebot import logger

from ..config import plugin_config
from ..policy.gatekeeper import Gatekeeper
from ..stickers.selector import StickerSelector
from ..stickers.sender import StickerSender
from ..storage.models import RawMessage
from ..storage.models import IndexJob
from ..storage.repositories.index_jobs_repo import IndexJobRepository
from ..storage.repositories.raw_repo import RawRepository
from ..storage.repositories.sticker_usage_repo import StickerUsageRepository
from ..storage.db_writer import db_writer
from ..storage.write_jobs import AsyncCallableJob


class ActionSender:
    """将动作列表发送到聊天窗口。"""

    @staticmethod
    async def send_actions(
        *,
        bot: Bot,
        matcher,
        scene_type: str,
        scene_id: str,
        actions: List[Dict[str, Any]],
    ) -> None:
        """按规则发送 actions（最多 4 条，300~900ms 随机间隔）。"""

        max_count = int(plugin_config.yuying_action_max_count)
        actions = actions[:max_count]

        for idx, action in enumerate(actions):
            if idx > 0:
                await ActionSender._sleep_human_delay()

            t = action.get("type")
            if t == "text":
                content = str(action.get("content") or "").strip()
                if not content:
                    continue
                await matcher.send(content)
                await ActionSender._record_bot_message(
                    bot_id=str(bot.self_id),
                    scene_type=scene_type,
                    scene_id=scene_id,
                    msg_type="text",
                    content=content,
                )
                await Gatekeeper.mark_sent(scene_type, scene_id)
            elif t == "sticker":
                intent = str(action.get("intent") or "neutral").strip()
                sticker = await StickerSelector.select_sticker(intent, scene_type, scene_id)
                if not sticker:
                    continue
                await matcher.send(StickerSender.create_message(sticker))
                await db_writer.submit(
                    AsyncCallableJob(
                        StickerUsageRepository.add_usage,
                        args=(sticker.sticker_id, scene_type, scene_id),
                        kwargs={"qq_id": str(bot.self_id)},
                    ),
                    priority=5,
                )
                await ActionSender._record_bot_message(
                    bot_id=str(bot.self_id),
                    scene_type=scene_type,
                    scene_id=scene_id,
                    msg_type="sticker",
                    content=f"[sticker:{sticker.sticker_id}]",
                )
                await Gatekeeper.mark_sent(scene_type, scene_id)
            else:
                logger.debug(f"未知动作类型，将忽略：{t}")

    @staticmethod
    async def _sleep_human_delay() -> None:
        """随机等待一小段时间，模拟真人节奏。"""

        min_ms = int(plugin_config.yuying_action_min_delay_ms)
        max_ms = int(plugin_config.yuying_action_max_delay_ms)
        if max_ms < min_ms:
            max_ms = min_ms
        await asyncio.sleep(random.randint(min_ms, max_ms) / 1000.0)

    @staticmethod
    async def _record_bot_message(
        *,
        bot_id: str,
        scene_type: str,
        scene_id: str,
        msg_type: str,
        content: str,
    ) -> None:
        """将机器人输出写入 raw_messages。

        说明：
        - 仅用于“对话事实”留痕与调试，不参与向量检索；
        - 向量检索主要服务于理解用户的隐含意图与上下文，索引机器人回复容易导致“复读/自我召回”。
        """

        try:
            msg = RawMessage(
                qq_id=bot_id,
                scene_type=scene_type,
                scene_id=scene_id,
                timestamp=int(time.time()),
                msg_type=msg_type,
                content=content,
                raw_ref=None,
                reply_to_msg_id=None,
                mentioned_bot=False,
                is_effective=False,
                is_bot=True,
            )
            await RawRepository.add(msg)
        except Exception as exc:
            logger.debug(f"写入机器人消息失败，将降级继续：{exc}")
