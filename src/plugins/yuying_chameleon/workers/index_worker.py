"""索引后台任务：从 index_jobs 拉取并写入向量库（Qdrant）。"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Dict, Optional

import httpx
from nonebot import logger
from qdrant_client.http.exceptions import UnexpectedResponse

from ..storage.models import IndexJob
from ..storage.repositories.index_jobs_repo import IndexJobRepository
from ..storage.repositories.memory_repo import MemoryRepository
from ..storage.repositories.raw_repo import RawRepository
from ..storage.repositories.sticker_repo import StickerRepository
from ..storage.repositories.summary_repo import SummaryRepository
from ..storage.db_writer import db_writer
from ..storage.write_jobs import AsyncCallableJob
from ..vector.embedder import embedder
from ..vector.qdrant_client import qdrant_manager


class IndexWorker:
    """索引工作循环：将 SQLite 的事实数据同步到向量库。"""

    @staticmethod
    def _make_point_id(kind: str, unique_key: str) -> str:
        """生成 Qdrant 合法的点 ID（UUID 字符串）。

        说明：
        - Qdrant 仅支持 `uint` 或 `UUID` 作为点 ID；
        - 为避免不同类型/不同 collection 之间的潜在冲突，这里统一使用 UUIDv5（稳定、可复现）。
        """

        seed = f"{kind}:{unique_key}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))

    @staticmethod
    def _split_csv(s: Optional[str]) -> list[str]:
        """将逗号分隔的字符串拆分为列表

        这个方法的作用:
        - 将 "tag1, tag2, tag3" 拆分为 ["tag1", "tag2", "tag3"]
        - 自动去除每个元素的首尾空格
        - 过滤掉空字符串

        为什么需要这个方法?
        - 数据库中的 tags 和 intents 字段是逗号分隔的字符串
        - Qdrant payload 需要数组格式以支持高效过滤和匹配
        - 统一的分割逻辑，避免重复代码

        Args:
            s: 逗号分隔的字符串
                - 示例: "可爱, 猫咪, 搞笑"
                - 可以是 None 或空字符串

        Returns:
            list[str]: 拆分后的列表
                - 示例: ["可爱", "猫咪", "搞笑"]
                - 空输入返回空列表

        Example:
            >>> IndexWorker._split_csv("tag1, tag2,  tag3  ")
            ['tag1', 'tag2', 'tag3']
            >>> IndexWorker._split_csv("")
            []
            >>> IndexWorker._split_csv(None)
            []
        """
        raw = (s or "").strip()
        if not raw:
            return []
        # split(",") 按逗号分割
        # p.strip() 去除每个元素的空格
        # if p.strip() 过滤掉空字符串
        return [p.strip() for p in raw.split(",") if p.strip()]

    def __init__(self) -> None:
        """初始化索引工作器。"""
        self._running = False

    async def run(self) -> None:
        """启动索引任务消费循环。"""

        if self._running:
            return
        self._running = True

        logger.info("IndexWorker 已启动。")
        while True:
            try:
                jobs = await IndexJobRepository.get_pending_jobs_for_types(
                    ["msg_chunk", "summary", "memory", "sticker"],
                    limit=10,
                )
                if not jobs:
                    await asyncio.sleep(3)
                    continue

                for job in jobs:
                    await self._process_job(job)
            except Exception as exc:
                logger.error(f"IndexWorker 循环异常：{exc}")
                await asyncio.sleep(3)

    async def _process_job(self, job: IndexJob) -> None:
        """处理单个索引任务。

        对于表情包（sticker）类型:
        - 优先使用图片 + 文本的多模态 embedding
        - 失败时降级为纯文本 embedding
        - 保证向量化任务不会因图片问题而完全失败
        """

        await db_writer.submit_and_wait(
            AsyncCallableJob(IndexJobRepository.mark_processing, args=(job.job_id,)),
            priority=5,
        )
        try:
            # 获取索引数据（可能包含 image_path）
            collection_name, point_id, text, payload, image_path = await self._build_payload(job)

            # 对表情包使用多模态 embedding（图片 + 文本）
            if payload.get("kind") == "sticker" and image_path:
                try:
                    # 尝试使用图片向量化
                    vector = await embedder.get_embedding_multimodal(
                        text=text, image_path=image_path
                    )
                    logger.debug(
                        f"表情包图片向量化成功: sticker_id={payload.get('sticker_id')}, "
                        f"path={image_path}"
                    )
                except Exception as e:
                    # 图片向量化失败，降级为文本 embedding
                    logger.warning(
                        f"表情包图片向量化失败，降级为文本 embedding: "
                        f"sticker_id={payload.get('sticker_id')}, error={e}"
                    )
                    vector = await embedder.get_embedding(text)
            else:
                # 非表情包或没有图片路径，使用纯文本 embedding
                vector = await embedder.get_embedding(text)

            # 写入向量库
            await qdrant_manager.upsert_text_point(
                collection_name=collection_name,
                point_id=point_id,
                vector=vector,
                payload=payload,
            )

            # 标记任务完成
            await db_writer.submit_and_wait(
                AsyncCallableJob(IndexJobRepository.update_status, args=(job.job_id, "done")),
                priority=5,
            )
        except UnexpectedResponse as exc:
            status = getattr(exc, "status_code", None)
            if status in {400, 401, 403, 404}:
                logger.error(f"索引任务永久失败（不再重试）job_id={job.job_id}：{exc}")
                await db_writer.submit_and_wait(
                    AsyncCallableJob(IndexJobRepository.update_status, args=(job.job_id, "dead")),
                    priority=5,
                )
                return
            next_ts = IndexJobRepository.compute_backoff_ts(job.retry_count + 1)
            logger.warning(f"索引任务失败，将重试 job_id={job.job_id}：{exc}")
            await db_writer.submit_and_wait(
                AsyncCallableJob(
                    IndexJobRepository.update_status,
                    args=(job.job_id, "failed"),
                    kwargs={"next_retry_ts": next_ts},
                ),
                priority=5,
            )
        except httpx.HTTPStatusError as exc:
            # 400/401/403/404 通常是配置或参数问题，重试没有意义，直接标记为 dead，避免日志刷屏
            status = getattr(exc.response, "status_code", None)
            if status in {400, 401, 403, 404}:
                logger.error(f"索引任务永久失败（不再重试）job_id={job.job_id}：{exc}")
                await db_writer.submit_and_wait(
                    AsyncCallableJob(IndexJobRepository.update_status, args=(job.job_id, "dead")),
                    priority=5,
                )
                return
            next_ts = IndexJobRepository.compute_backoff_ts(job.retry_count + 1)
            logger.warning(f"索引任务失败，将重试 job_id={job.job_id}：{exc}")
            await db_writer.submit_and_wait(
                AsyncCallableJob(
                    IndexJobRepository.update_status,
                    args=(job.job_id, "failed"),
                    kwargs={"next_retry_ts": next_ts},
                ),
                priority=5,
            )
        except Exception as exc:
            next_ts = IndexJobRepository.compute_backoff_ts(job.retry_count + 1)
            logger.warning(f"索引任务失败，将重试 job_id={job.job_id}：{exc}")
            await db_writer.submit_and_wait(
                AsyncCallableJob(
                    IndexJobRepository.update_status,
                    args=(job.job_id, "failed"),
                    kwargs={"next_retry_ts": next_ts},
                ),
                priority=5,
            )

    async def _build_payload(
        self, job: IndexJob
    ) -> tuple[str, str, str, Dict[str, Any], Optional[str]]:
        """根据任务类型构建向量库写入内容。

        Returns:
            tuple: 包含 5 个元素
                - collection_name: 向量库集合名称
                - point_id: 向量点 ID（UUID 字符串）
                - text: 用于 embedding 的文本
                - payload: 附加元数据
                - image_path: 图片路径（仅表情包类型，其他为 None）

        为什么 image_path 是可选的?
        - 只有表情包需要图片向量化
        - 其他类型（消息、摘要、记忆）都是纯文本
        - 返回 None 让调用方知道使用纯文本 embedding
        """

        payload_json = {}
        try:
            payload_json = json.loads(job.payload_json) if job.payload_json else {}
        except Exception:
            payload_json = {}

        if job.item_type == "msg_chunk":
            msg = await RawRepository.get_by_id(int(job.ref_id))
            if not msg:
                raise RuntimeError("原始消息不存在")
            text = msg.content
            payload = {
                "kind": "msg_chunk",
                "text": text[:500],
                "scene_type": msg.scene_type,
                "scene_id": msg.scene_id,
                "qq_id": msg.qq_id,
                "is_bot": bool(getattr(msg, "is_bot", False)),
                "msg_id": msg.id,
                "timestamp": msg.timestamp,
            }
            return "rag_items", self._make_point_id("msg", str(msg.id)), text, payload, None

        if job.item_type == "summary":
            summary = await SummaryRepository.get_by_id(int(payload_json.get("summary_id", job.ref_id)))
            if not summary:
                raise RuntimeError("摘要不存在")
            text = summary.summary_text
            payload = {
                "kind": "summary",
                "text": text[:500],
                "scene_type": summary.scene_type,
                "scene_id": summary.scene_id,
                "summary_id": summary.id,
                "window_end_ts": summary.window_end_ts,
            }
            return "rag_items", self._make_point_id("sum", str(summary.id)), text, payload, None

        if job.item_type == "memory":
            memory = await MemoryRepository.get_by_id(int(payload_json.get("memory_id", job.ref_id)))
            if not memory:
                raise RuntimeError("记忆不存在")
            text = memory.content
            payload = {
                "kind": "memory",
                "qq_id": memory.qq_id,
                "tier": memory.tier,
                "type": memory.type,
                "visibility": memory.visibility,
                "scope_scene_id": memory.scope_scene_id,
                "memory_id": memory.id,
            }
            return "memories", self._make_point_id("mem", str(memory.id)), text, payload, None

        if job.item_type == "sticker":
            sticker = await StickerRepository.get_by_id(job.ref_id)
            if not sticker:
                raise RuntimeError("表情包不存在")

            # 拼接文本用于 text-only embedding 的降级场景
            # 以及作为多模态 embedding 的辅助信息
            parts = [sticker.name or "", sticker.tags or "", sticker.intents or "", sticker.ocr_text or ""]
            text = " ".join([p for p in parts if p]).strip() or "表情包"

            # 构建 payload：添加结构化的 tags_list 和 intents_list
            # 保留原始的逗号分隔字符串以兼容现有代码
            payload = {
                "kind": "sticker",
                "sticker_id": sticker.sticker_id,
                "pack": sticker.pack,
                # 原始字符串格式（兼容性）
                "tags": sticker.tags or "",
                "intents": sticker.intents or "",
                # 结构化数组格式（用于高效过滤和rerank）
                "tags_list": self._split_csv(sticker.tags),
                "intents_list": self._split_csv(sticker.intents),
                "is_enabled": sticker.is_enabled,
                "is_banned": sticker.is_banned,
            }

            # 返回图片路径用于多模态 embedding
            # file_path 是表情包图片的本地路径
            return (
                "stickers",
                self._make_point_id("stk", str(sticker.sticker_id)),
                text,
                payload,
                sticker.file_path,  # 关键：返回图片路径
            )

        raise RuntimeError(f"未知任务类型：{job.item_type}")


index_worker = IndexWorker()
