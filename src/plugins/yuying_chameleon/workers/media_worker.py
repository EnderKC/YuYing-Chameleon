"""媒体后台任务：图片说明与轻量文字识别，写入 media_cache。"""

from __future__ import annotations

import asyncio
import json
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

from nonebot import logger

from ..llm.vision import VisionHelper
from ..config import plugin_config
from ..storage.models import IndexJob
from ..storage.repositories.index_jobs_repo import IndexJobRepository
from ..storage.repositories.media_cache_repo import MediaCacheRepository
from ..storage.repositories.raw_repo import RawRepository
from ..storage.write_jobs import AddIndexJobJob, AsyncCallableJob
from ..storage.db_writer import db_writer
from ..paths import assets_dir


class MediaWorker:
    """媒体任务工作循环。"""

    def __init__(self) -> None:
        """初始化媒体后台工作器。"""
        self._running = False

    async def run(self) -> None:
        """启动媒体任务消费循环。"""

        if self._running:
            return
        self._running = True

        logger.info("MediaWorker 已启动。")
        while True:
            try:
                jobs = await IndexJobRepository.get_pending_jobs(limit=10, item_type="ocr")
                if not jobs:
                    await asyncio.sleep(5)
                    continue
                for job in jobs:
                    await self._process_job(job)
            except Exception as exc:
                logger.error(f"MediaWorker 循环异常：{exc}")
                await asyncio.sleep(5)

    async def _process_job(self, job: IndexJob) -> None:
        """处理单个媒体预处理任务。"""

        await db_writer.submit_and_wait(
            AsyncCallableJob(IndexJobRepository.mark_processing, args=(job.job_id,)),
            priority=5,
        )
        try:
            payload = json.loads(job.payload_json) if job.payload_json else {}
            media_key = str(payload.get("media_key") or job.ref_id)
            url = payload.get("url")
            file_path = payload.get("file_path")

            local_path = self._maybe_local_file(file_path)

            # 先尝试用 url（通常更小、更兼容）；失败再回退本地文件
            caption = await VisionHelper.caption_image(local_path, url=url)
            if not caption and url and not local_path:
                local_path = await self._download_to_tmp(media_key, url=url)
                caption = await VisionHelper.caption_image(local_path, url=None)

            ocr_text = ""
            if bool(getattr(plugin_config, "yuying_media_enable_ocr", False)):
                ocr_text = await VisionHelper.ocr_image(local_path, url=url)
                if not ocr_text and url and not local_path:
                    local_path = await self._download_to_tmp(media_key, url=url)
                    ocr_text = await VisionHelper.ocr_image(local_path, url=None)

            if not caption:
                caption = "图片"

            await db_writer.submit_and_wait(
                AsyncCallableJob(
                    MediaCacheRepository.upsert,
                    args=(media_key,),
                    kwargs={
                        "media_type": "image",
                        "caption": caption,
                        "tags": None,
                        "ocr_text": ocr_text or None,
                    },
                ),
                priority=5,
            )

            # 尽力回填历史消息内容，让后续“回忆上一张图”时可直接看到说明
            try:
                short = caption[:20] + ("…" if len(caption) > 20 else "")
                rows = await RawRepository.list_with_image_marker(media_key, limit=50)
                for r in rows:
                    if f"[image:{media_key}:" in (r.content or ""):
                        continue
                    new_content = (r.content or "").replace(f"[image:{media_key}]", f"[image:{media_key}:{short}]")
                    if new_content != (r.content or ""):
                        await db_writer.submit_and_wait(
                            AsyncCallableJob(RawRepository.update_content, args=(r.id, new_content)),
                            priority=5,
                        )
                        # 追加 msg_chunk 索引任务，确保向量库中的片段同步更新
                        await db_writer.submit(
                            AddIndexJobJob(
                                IndexJob(
                                    item_type="msg_chunk",
                                    ref_id=str(r.id),
                                    payload_json="{}",
                                    status="pending",
                                )
                            ),
                            priority=5,
                        )
            except Exception as exc:
                logger.debug(f"回填图片说明到历史消息失败：{exc}")

            await db_writer.submit_and_wait(
                AsyncCallableJob(IndexJobRepository.update_status, args=(job.job_id, "done")),
                priority=5,
            )
        except Exception as exc:
            next_ts = IndexJobRepository.compute_backoff_ts(job.retry_count + 1)
            logger.warning(f"媒体任务失败，将重试 job_id={job.job_id}：{exc}")
            await db_writer.submit_and_wait(
                AsyncCallableJob(
                    IndexJobRepository.update_status,
                    args=(job.job_id, "failed"),
                    kwargs={"next_retry_ts": next_ts},
                ),
                priority=5,
            )

    @staticmethod
    def _maybe_local_file(file_path: Optional[str]) -> Optional[str]:
        """尽力返回可用的本地文件路径（不主动下载）。"""

        if not file_path:
            return None
        p = Path(file_path)
        if p.exists() and p.is_file():
            return str(p)
        return None

    async def _download_to_tmp(self, media_key: str, *, url: str) -> str:
        """下载 url 到临时目录并返回本地路径。"""

        tmp_dir = self._tmp_dir()
        tmp_dir.mkdir(parents=True, exist_ok=True)
        dst = tmp_dir / f"{media_key}.img"

        def _download() -> None:
            urllib.request.urlretrieve(url, dst)  # nosec - 运行时受配置控制

        await asyncio.to_thread(_download)
        return str(dst)

    @staticmethod
    def _tmp_dir() -> Path:
        """获取媒体临时目录（assets/media/tmp）。"""

        return assets_dir() / "media" / "tmp"


media_worker = MediaWorker()
