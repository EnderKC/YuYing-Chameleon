"""表情包后台任务：自动打标签与违规判断（使用低成本模型）。"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path  # 新增：读取文件路径
from typing import Any, Dict, Optional

from nonebot import logger

from ..config import plugin_config
from ..llm.client import get_task_llm
from ..llm.vision import VisionHelper  # 新增：生成 data URL
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

                max_conc = int(getattr(plugin_config, "yuying_sticker_worker_max_concurrency", 1) or 1)
                max_conc = max(1, min(16, max_conc))

                if max_conc <= 1 or len(jobs) <= 1:
                    for job in jobs:
                        await self._process_job(job)
                else:
                    sem = asyncio.Semaphore(max_conc)

                    async def _run_one(j: IndexJob) -> None:
                        async with sem:
                            await self._process_job(j)

                    await asyncio.gather(*(_run_one(j) for j in jobs))
            except Exception as exc:
                logger.error(f"StickerWorker 循环异常：{exc}")
                await asyncio.sleep(10)

    async def _process_job(self, job: IndexJob) -> None:
        """为一个表情包生成 OCR文字 + tags/intents/style + 违规判定（一次 LLM 调用完成）。"""

        await db_writer.submit_and_wait(
            AsyncCallableJob(IndexJobRepository.mark_processing, args=(job.job_id,)),
            priority=5,
        )
        try:
            payload = json.loads(job.payload_json) if job.payload_json else {}
            sticker_id = str(payload.get("sticker_id") or job.ref_id)
            intent_hint = payload.get("intent_hint")

            sticker = await StickerRepository.get_by_id(sticker_id)
            if not sticker:
                await db_writer.submit_and_wait(
                    AsyncCallableJob(IndexJobRepository.update_status, args=(job.job_id, "dead")),
                    priority=5,
                )
                return

            # ==================== 准备图片 data URL ====================
            try:
                p = Path(sticker.file_path)
                image_url = VisionHelper._to_data_url(p.read_bytes(), p.suffix)
            except Exception as exc:
                logger.error(f"读取表情包图片失败 sticker_id={sticker_id}: {exc}")
                await db_writer.submit_and_wait(
                    AsyncCallableJob(IndexJobRepository.update_status, args=(job.job_id, "failed")),
                    priority=5,
                )
                return

            # ==================== 构建 prompt（合并 OCR + 打标签） ====================
            prompt_text = "\n".join(
                [
                    "你要分析一个群聊表情包图片，完成以下任务：",
                    "1. 识别图片中的所有文字（OCR），输出纯文本",
                    "2. 为表情包生成标签（tags）和意图（intents）",
                    "3. 判断是否违规（涉政、色情、暴力等）",
                    "",
                    "输出必须是严格 JSON，对象结构：",
                    '{',
                    '  "ocr_text": "图片中的文字（如果没有文字则为空字符串）",',
                    '  "tags": ["标签1", "标签2"],',
                    '  "intents": ["agree"],',
                    '  "style": "可选风格描述",',
                    '  "is_banned": false,',
                    '  "ban_reason": ""',
                    '}',
                    "",
                    "intents 枚举：agree/tease/shock/sorry/thanks/awkward/think/urge/neutral",
                    "要求：",
                    "- ocr_text: 原样输出图片中的文字，不解释、不翻译",
                    "- tags: 最多 6 个，每个 <=6 字，描述表情包的主题和情感",
                    "- intents: 1~3 个，表示表情包适用的对话意图",
                    "- style: 可选，描述表情包的风格（如\"手绘\"、\"真人\"等）",
                    "- is_banned: 仅当明确违规时为 true",
                    "",
                    "注意：如果不确定 intents，请保守输出 neutral，并将 tags 保持通用。",
                    "",
                    f"sticker_id: {sticker_id}",
                    f"intent_hint: {(intent_hint or '').strip() or '（无）'}",
                ]
            )

            # ==================== 构建包含图片的 messages ====================
            messages = [
                {"role": "system", "content": "你是表情包分析器，只能输出 JSON。"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ]

            llm = get_task_llm("sticker_tagging")
            content = await llm.chat_completion(messages, temperature=0.2)
            if not content:
                await db_writer.submit_and_wait(
                    AsyncCallableJob(IndexJobRepository.update_status, args=(job.job_id, "failed")),
                    priority=5,
                )
                return

            data = self._extract_first_json_object(content)
            if not isinstance(data, dict):
                logger.warning(f"StickerWorker 无法解析 JSON: {content[:200]}")
                await db_writer.submit_and_wait(
                    AsyncCallableJob(IndexJobRepository.update_status, args=(job.job_id, "failed")),
                    priority=5,
                )
                return

            # ==================== 解析 LLM 输出 ====================
            raw_ocr_text = data.get("ocr_text")
            ocr_text = raw_ocr_text if isinstance(raw_ocr_text, str) else None

            raw_tags = data.get("tags")
            tags = raw_tags if isinstance(raw_tags, list) else []

            raw_intents = data.get("intents")
            intents = raw_intents if isinstance(raw_intents, list) else []
            style = data.get("style") if isinstance(data.get("style"), str) else None
            is_banned = bool(data.get("is_banned", False))
            ban_reason = data.get("ban_reason") if isinstance(data.get("ban_reason"), str) else None

            tags_text = ",".join([str(t).strip() for t in tags if str(t).strip()][:6]) or None
            intents_text = ",".join([str(i).strip() for i in intents if str(i).strip()][:3]) or "neutral"

            # ==================== 更新数据库（包括 ocr_text） ====================
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
                        "ocr_text": ocr_text,  # 新增：更新 OCR 文字
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
