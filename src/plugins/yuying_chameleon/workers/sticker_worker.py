"""表情包后台任务：自动打标签与违规判断（使用低成本模型）。"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, Optional

from nonebot import logger

from ..llm.client import cheap_llm
from ..storage.models import IndexJob
from ..storage.db_writer import db_writer
from ..storage.repositories.index_jobs_repo import IndexJobRepository
from ..storage.repositories.sticker_repo import StickerRepository
from ..storage.write_jobs import AddIndexJobJob, AsyncCallableJob


class StickerWorker:
    """表情包元数据生成工作循环。"""

    def __init__(self) -> None:
        """初始化表情包后台工作器。"""
        self._running = False

    async def run(self) -> None:
        """启动表情包后台循环。"""

        if self._running:
            return
        self._running = True

        logger.info("StickerWorker 已启动。")
        while True:
            try:
                jobs = await IndexJobRepository.get_pending_jobs(limit=10, item_type="sticker_tag")
                if not jobs:
                    await asyncio.sleep(10)
                    continue

                for job in jobs:
                    await self._process_job(job)
            except Exception as exc:
                logger.error(f"StickerWorker 循环异常：{exc}")
                await asyncio.sleep(10)

    async def _process_job(self, job: IndexJob) -> None:
        """为一个表情包生成 tags/intents/style 与违规判定。"""

        await db_writer.submit_and_wait(
            AsyncCallableJob(IndexJobRepository.mark_processing, args=(job.job_id,)),
            priority=5,
        )
        try:
            payload = json.loads(job.payload_json) if job.payload_json else {}
            sticker_id = str(payload.get("sticker_id") or job.ref_id)
            intent_hint = payload.get("intent_hint")
            ocr_text = payload.get("ocr_text")

            sticker = await StickerRepository.get_by_id(sticker_id)
            if not sticker:
                await db_writer.submit_and_wait(
                    AsyncCallableJob(IndexJobRepository.update_status, args=(job.job_id, "dead")),
                    priority=5,
                )
                return

            prompt = "\n".join(
                [
                    "你要为一个群聊表情包生成标签与意图，并判断是否违规。",
                    "输出必须是严格 JSON，对象结构：",
                    '{"tags":["..."],"intents":["agree"],"style":"可选","is_banned":false,"ban_reason":""}',
                    "",
                    "intents 枚举：agree/tease/shock/sorry/thanks/awkward/think/urge/neutral",
                    "要求：tags 最多 6 个，每个 <=6 字；intents 1~3 个。",
                    "注意：如果不确定，请保守输出 neutral，并将 tags 保持通用。",
                    "",
                    f"sticker_id: {sticker_id}",
                    f"intent_hint: {(intent_hint or '').strip() or '（无）'}",
                    f"ocr_text: {(ocr_text or sticker.ocr_text or '').strip() or '（无）'}",
                ]
            )

            content = await cheap_llm.chat_completion(
                [
                    {"role": "system", "content": "你是表情包标签器，只能输出 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            if not content:
                await db_writer.submit_and_wait(
                    AsyncCallableJob(IndexJobRepository.update_status, args=(job.job_id, "failed")),
                    priority=5,
                )
                return

            data = self._extract_first_json_object(content)
            if not isinstance(data, dict):
                await db_writer.submit_and_wait(
                    AsyncCallableJob(IndexJobRepository.update_status, args=(job.job_id, "failed")),
                    priority=5,
                )
                return

            tags = data.get("tags") if isinstance(data.get("tags"), list) else []
            intents = data.get("intents") if isinstance(data.get("intents"), list) else []
            style = data.get("style") if isinstance(data.get("style"), str) else None
            is_banned = bool(data.get("is_banned", False))
            ban_reason = data.get("ban_reason") if isinstance(data.get("ban_reason"), str) else None

            tags_text = ",".join([str(t).strip() for t in tags if str(t).strip()][:6]) or None
            intents_text = ",".join([str(i).strip() for i in intents if str(i).strip()][:3]) or "neutral"

            await db_writer.submit_and_wait(
                AsyncCallableJob(
                    StickerRepository.update_metadata,
                    args=(sticker_id,),
                    kwargs={
                        "tags": tags_text,
                        "intents": intents_text,
                        "style": style,
                        "is_enabled": not is_banned,
                        "is_banned": is_banned,
                        "ban_reason": ban_reason,
                    },
                ),
                priority=5,
            )

            # 写入索引任务（由 index_worker 写入向量库）
            await db_writer.submit(
                AddIndexJobJob(
                    IndexJob(
                        item_type="sticker",
                        ref_id=sticker_id,
                        payload_json=json.dumps({"sticker_id": sticker_id}, ensure_ascii=False),
                        status="pending",
                    )
                ),
                priority=5,
            )

            await db_writer.submit_and_wait(
                AsyncCallableJob(IndexJobRepository.update_status, args=(job.job_id, "done")),
                priority=5,
            )
        except Exception as exc:
            logger.error(f"StickerWorker 处理任务失败 job_id={job.job_id}：{exc}")
            await db_writer.submit_and_wait(
                AsyncCallableJob(IndexJobRepository.update_status, args=(job.job_id, "failed")),
                priority=5,
            )

        # sticker 向量索引：在打标完成后写入 index_jobs(item_type=sticker)

    @staticmethod
    def _extract_first_json_object(text: str) -> Optional[object]:
        """从文本中提取第一个 JSON 对象或数组。"""

        s = text.strip()
        m = re.search(r"(\{.*\}|\[.*\])", s, flags=re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(1))
        except Exception:
            return None


sticker_worker = StickerWorker()
