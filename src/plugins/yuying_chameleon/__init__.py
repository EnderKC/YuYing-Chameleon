"""YuYing-Chameleon 插件入口。

职责：
- 初始化数据库/向量库/后台任务/定时任务
- 处理每条消息：解析 -> 归一化 -> 落库 -> 摘要/记忆/检索 -> 动作规划 -> 分段发送
"""

from __future__ import annotations

import asyncio
import json
import re
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from nonebot import get_driver, on_message
from nonebot.adapters.onebot.v11 import Bot, Event
from nonebot.plugin import PluginMetadata
from nonebot import logger

from .config import Config, plugin_config
from .llm.mcp_manager import mcp_manager
from .memory.memory_manager import MemoryManager
from .planner.action_planner import ActionPlanner
from .planner.action_sender import ActionSender
from .policy.gatekeeper import Gatekeeper
from .retrieval.retriever import Retriever
from .scheduler.jobs import init_scheduler
from .storage.db_writer import db_writer
from .storage.models import IndexJob, RawMessage
from .storage.repositories.index_jobs_repo import IndexJobRepository
from .storage.repositories.media_cache_repo import MediaCacheRepository
from .storage.repositories.profile_repo import ProfileRepository
from .storage.repositories.raw_repo import RawRepository
from .storage.sqlalchemy_engine import engine
from .stickers.selector import StickerSelector
from .stickers.stealer import StickerStealer
from .stickers.registry import StickerRegistry
from .summary.summary_manager import SummaryManager
from .vector.qdrant_client import qdrant_manager
from .workers.index_worker import index_worker
from .workers.media_worker import media_worker
from .workers.sticker_worker import sticker_worker
from .adapters.lagrange_parser import LagrangeParser
from .normalize.normalizer import Normalizer, NormalizedMessage
from .llm.vision import VisionHelper
from .paths import assets_dir
from .storage.write_jobs import AddIndexJobJob
from .storage.write_jobs import AsyncCallableJob
from .tools.adaptive_debouncer import AdaptiveDebouncer

__plugin_meta__ = PluginMetadata(
    name="YuYing-Chameleon",
    description="Context-aware QQ Bot with Memory & Stickers",
    usage="Auto-active",
    config=Config,
)

driver = get_driver()


def _format_recent_dialogue_line(
    *,
    current_qq_id: str,
    text: str,
    sender_qq_id: str,
    is_bot: bool,
    timestamp: Optional[int] = None,
) -> str:
    t = (text or "").strip().replace("\n", " ")
    if len(t) > 120:
        t = t[:120] + "…"
    when = ""
    if timestamp is not None:
        try:
            dt = datetime.fromtimestamp(int(timestamp))
            when = f"（{dt.strftime('%Y-%m-%d %H:%M')}）"
        except Exception:
            when = ""
    if is_bot:
        who = "你"
    elif sender_qq_id == current_qq_id:
        who = f"我（{current_qq_id}）"
    else:
        who = f"群友({sender_qq_id})"
    return f"{when}{who}：{t}" if when else f"{who}：{t}"


async def _build_recent_dialogue(
    *,
    current_qq_id: str,
    scene_type: str,
    scene_id: str,
    current_raw_msg_id: int,
    max_lines: int = 30,
) -> list[str]:
    """构建最近对话片段（短、按时间顺序），用于直接注入 LLM prompt。"""

    lines: list[str] = []

    try:
        # 取比 max_lines 稍多的窗口，避免过滤 current 后不足
        recent = await RawRepository.get_recent_by_scene(scene_type, scene_id, limit=max(30, max_lines + 5))
        # get_recent_by_scene 是倒序，反转成时间顺序
        recent = list(reversed(recent))
        # 只取当前消息之前的若干条，避免重复“用户消息”
        prev = [m for m in recent if int(getattr(m, "id", 0)) < int(current_raw_msg_id)]
        prev = prev[-max_lines:]
        for m in prev:
            lines.append(
                _format_recent_dialogue_line(
                    current_qq_id=current_qq_id,
                    text=str(m.content or ""),
                    sender_qq_id=str(m.qq_id),
                    is_bot=bool(getattr(m, "is_bot", False)),
                    timestamp=getattr(m, "timestamp", None),
                )
            )
    except Exception:
        pass

    # 去重（保持顺序）
    uniq: list[str] = []
    for s in lines:
        if s and s not in uniq:
            uniq.append(s)
    return uniq[:max_lines]


async def _collect_image_inputs(
    *,
    bot: Bot,
    image_ref_map: dict[str, str],
) -> list[dict[str, str]]:
    """收集当前消息中的图片输入（url + media_key + caption 可选）。"""

    items: list[dict[str, str]] = []
    for raw_ref, media_key in list((image_ref_map or {}).items())[:2]:
        try:
            url: Optional[str] = None
            if isinstance(raw_ref, str) and (raw_ref.startswith("http://") or raw_ref.startswith("https://")):
                url = raw_ref
            else:
                info = await bot.get_image(file=raw_ref)
                url = info.get("url") if isinstance(info, dict) else None

            caption = ""
            try:
                cached = await MediaCacheRepository.get(media_key)
                if cached and cached.caption:
                    caption = str(cached.caption).strip()
            except Exception:
                caption = ""

            if url:
                items.append(
                    {
                        "url": str(url),
                        "media_key": str(media_key),
                        "caption": caption,
                    }
                )
        except Exception:
            continue

    return items


async def _get_reply_message_info(
    *,
    bot: Bot,
    reply_to_msg_id: Optional[int],
    timeout_seconds: float = 1.8,
) -> Optional[Dict[str, Any]]:
    """获取被引用（回复）消息的信息：sender_id + content（一次 get_msg 复用）。

    说明:
        - reply_to_msg_id 来自 OneBot 的 reply.message_id（平台消息 id）
        - 与 raw_messages.id（数据库自增 id）不是同一个 id 空间
        - 使用 OneBot API `get_msg` 查询被回复消息详情
        - 尽最大努力提取 sender.user_id 与 message 文本（失败降级，不影响主流程）

    Args:
        bot: NoneBot Bot实例，用于调用 OneBot API
        reply_to_msg_id: 被回复的消息ID，来自 event.reply.message_id
        timeout_seconds: API 调用超时时间（秒），避免拖慢消息处理

    Returns:
        Optional[Dict[str, Any]]:
            - reply_to_msg_id 为空: 返回 None
            - 否则返回结构化字典：
              {
                  "sender_id": str,      # 发送者 QQ 号
                  "content": str,        # 消息内容（归一化后的文本）
                  "failed": bool,        # 是否获取失败
                  "reason": str,         # 失败原因（failed=True 时有效）
              }

    失败原因分类:
        - invalid_message_id: 消息 ID 格式错误
        - timeout: API 调用超时
        - not_found_or_deleted: 消息不存在或已删除
        - permission_denied: 权限不足
        - api_error: 其他 API 错误
        - malformed_response: 返回数据格式异常

    Example:
        >>> # 群聊中 A 回复了 bot 的消息
        >>> info = await _get_reply_message_info(bot=bot, reply_to_msg_id=12345)
        >>> if info and not info["failed"]:
        >>>     print(f"回复了 {info['sender_id']} 的消息: {info['content']}")
    """

    # ==================== 步骤1: 参数校验 ====================

    if not reply_to_msg_id:
        return None

    # ==================== 步骤2: 类型转换 ====================

    try:
        mid = int(reply_to_msg_id)
    except Exception:
        return {
            "sender_id": "",
            "content": "",
            "failed": True,
            "reason": "invalid_message_id",
        }

    # ==================== 步骤3: 定义内部辅助函数 ====================

    def _extract_content_from_message(message_obj: Any) -> str:
        """将 get_msg 返回的 message 字段归一化为短文本。

        处理规则:
            - 字符串（CQ 码）: 直接返回
            - 数组段（标准 OneBot v11）: 逐段解析为短标记
            - 其他: 强制字符串化
        """

        if message_obj is None:
            return ""

        # 部分实现直接返回 CQ 字符串
        if isinstance(message_obj, str):
            return message_obj

        parts: list[str] = []

        # 标准 OneBot v11: message 为数组段
        if isinstance(message_obj, list):
            for seg in message_obj:
                if not isinstance(seg, dict):
                    parts.append(str(seg))
                    continue

                seg_type = str(seg.get("type") or "").strip()
                seg_data = seg.get("data") if isinstance(seg.get("data"), dict) else {}

                if seg_type == "text":
                    parts.append(str(seg_data.get("text") or ""))
                elif seg_type == "image":
                    seg_ref = seg_data.get("file") or seg_data.get("file_id") or seg_data.get("url") or "unknown"
                    parts.append(f"[image:{seg_ref}]")
                elif seg_type == "face":
                    parts.append(f"[face:{seg_data.get('id')}]")
                elif seg_type == "at":
                    parts.append(f"@{seg_data.get('qq')}")
                else:
                    # 其他段类型不展开，避免引入过多噪声
                    if seg_type:
                        parts.append(f"[{seg_type}]")

            return "".join(parts)

        # 兜底：未知结构直接字符串化
        return str(message_obj)

    def _classify_failure_reason(exc: Exception) -> str:
        """将异常归类为稳定的失败原因字符串，便于 prompt 明确告知 LLM。"""

        error_msg = (str(exc) or "").strip().lower()
        if not error_msg:
            return "api_error"

        # 常见：消息不存在/已撤回/找不到
        if any(keyword in error_msg for keyword in ["not found", "不存在", "找不到", "no such", "404"]):
            return "not_found_or_deleted"

        # 常见：权限不足
        if any(keyword in error_msg for keyword in ["permission", "权限", "forbidden", "unauthorized"]):
            return "permission_denied"

        # 其他：统一归为 api_error
        return "api_error"

    # ==================== 步骤4: 调用 OneBot API 查询被回复的消息 ====================

    try:
        data = await asyncio.wait_for(
            bot.call_api("get_msg", message_id=mid),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        return {
            "sender_id": "",
            "content": "",
            "timestamp": None,
            "failed": True,
            "reason": "timeout",
        }
    except Exception as exc:
        return {
            "sender_id": "",
            "content": "",
            "timestamp": None,
            "failed": True,
            "reason": _classify_failure_reason(exc),
        }

    # ==================== 步骤5: 提取发送者 ID 和消息内容 ====================

    sender_id = ""
    content = ""
    msg_ts: Optional[int] = None

    if isinstance(data, dict):
        # 提取消息时间（Unix 秒）
        t = data.get("time")
        if t is not None:
            try:
                msg_ts = int(t)
            except Exception:
                msg_ts = None

        # 提取发送者 ID
        sender = data.get("sender")
        if isinstance(sender, dict):
            uid = sender.get("user_id")
            if uid is not None:
                sender_id = str(uid)

        # 提取消息内容
        content = _extract_content_from_message(data.get("message"))

    # 归一化内容：去除换行、前后空格
    content = (content or "").replace("\n", " ").strip()

    # ==================== 步骤6: 返回结构化结果 ====================

    # 即使缺字段也标记 failed，避免"静默当成无引用"
    if not sender_id and not content:
        return {
            "sender_id": "",
            "content": "",
            "timestamp": msg_ts,
            "failed": True,
            "reason": "malformed_response",
        }

    return {
        "sender_id": sender_id,
        "content": content,
        "timestamp": msg_ts,
        "failed": False,
        "reason": "",
    }


async def _download_and_steal(
    *,
    url: str,
    media_key: str,
    scene_id: str,
    source_qq_id: str,
    intent_hint: str,
) -> None:
    """下载图片并尝试按表情包候选流程处理。"""

    try:
        tmp_dir = assets_dir() / "media" / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        dst = tmp_dir / f"{media_key}.img"

        def _download() -> None:
            """下载图片到临时路径。"""
            urllib.request.urlretrieve(url, str(dst))  # nosec - 运行时受配置控制

        await asyncio.to_thread(_download)
        await StickerStealer.process_image(
            scene_id=scene_id,
            source_qq_id=source_qq_id,
            file_path=str(dst),
            intent_hint=intent_hint,
        )
    except Exception as exc:
        logger.debug(f"下载并偷取表情包失败：{exc}")

async def _try_caption_and_ocr(
    *,
    url: Optional[str],
    file_path: Optional[str],
    max_wait_seconds: float = 6.0,
) -> Tuple[str, str]:
    """尽力对图片做说明与轻量 OCR（用于当前轮即时理解）。

    说明：
    - 优先使用 url（对火山方舟等网关更兼容），否则回退读取本地文件并以 data URL 发送；
    - 不保证成功，失败返回空字符串。
    """

    try:
        enable_ocr = bool(getattr(plugin_config, "yuying_media_enable_ocr", False))
        if enable_ocr:
            caption, ocr_text = await asyncio.wait_for(
                VisionHelper.caption_and_ocr_image(file_path, url=url),
                timeout=max_wait_seconds,
            )
        else:
            caption = await asyncio.wait_for(
                VisionHelper.caption_image(file_path, url=url),
                timeout=max_wait_seconds,
            )
            ocr_text = ""
        logger.debug(f"图片说明/OCR 结果：{caption} / {ocr_text}")
        return (caption or "").strip(), (ocr_text or "").strip()
    except Exception as exc:
        logger.debug(f"图片说明/OCR 失败：{exc}")
        return "", ""


@driver.on_startup
async def startup() -> None:
    """NoneBot 启动时执行：初始化依赖并启动后台任务。"""

    # 1) 初始化数据库表（直接 create_all）
    try:
        async with engine.begin() as conn:
            from .storage.models import Base

            await conn.run_sync(Base.metadata.create_all)
    except Exception as exc:
        logger.error(f"初始化数据库失败：{exc}")
        raise

    # 2) 初始化向量库（失败允许降级）
    await qdrant_manager.init_collections()

    # 3) 启动写入队列（非关键写入串行化）
    asyncio.create_task(db_writer.run_forever())

    # 4) 启动后台任务
    asyncio.create_task(index_worker.run())
    asyncio.create_task(media_worker.run())
    asyncio.create_task(sticker_worker.run())

    # 5) 启动扫描本地表情包（默认/自动）
    try:
        await StickerRegistry.scan_local_stickers(str(assets_dir() / "stickers"))
    except Exception as exc:
        logger.warning(f"扫描本地表情包失败，将继续启动：{exc}")

    # 6) 初始化定时任务
    init_scheduler()

    # 7) 初始化 MCP 管理器（如果启用）
    await mcp_manager.on_startup()


@driver.on_shutdown
async def shutdown() -> None:
    """NoneBot 关闭时执行：清理资源。"""

    # 关闭 MCP 管理器（关闭所有 MCP server 连接）
    await mcp_manager.on_shutdown()

    # 清理自适应防抖任务
    global _adaptive_debouncer
    if _adaptive_debouncer is not None:
        await _adaptive_debouncer.shutdown()


_adaptive_debouncer: Optional[AdaptiveDebouncer] = None


def get_adaptive_debouncer() -> AdaptiveDebouncer:
    """获取或创建全局 AdaptiveDebouncer 实例（懒加载）。

    Returns:
        AdaptiveDebouncer: 全局 debouncer 实例
    """
    global _adaptive_debouncer
    if _adaptive_debouncer is None:
        _adaptive_debouncer = AdaptiveDebouncer(
            joiner=str(
                getattr(plugin_config, "yuying_adaptive_debounce_joiner", "auto")
                or "auto"
            ),
            ttl_seconds=float(
                getattr(plugin_config, "yuying_adaptive_debounce_ttl_seconds", 60.0)
                or 60.0
            ),
            max_hold_seconds=float(
                getattr(
                    plugin_config, "yuying_adaptive_debounce_max_hold_seconds", 15.0
                )
                or 15.0
            ),
            max_parts=int(
                getattr(plugin_config, "yuying_adaptive_debounce_max_parts", 12) or 12
            ),
            max_plain_len=int(
                getattr(plugin_config, "yuying_adaptive_debounce_max_plain_len", 300)
                or 300
            ),
            w1=float(
                getattr(plugin_config, "yuying_adaptive_debounce_w1", 0.6) or 0.6
            ),
            w2=float(
                getattr(plugin_config, "yuying_adaptive_debounce_w2", -0.025) or -0.025
            ),
            w3=float(
                getattr(plugin_config, "yuying_adaptive_debounce_w3", -2.5) or -2.5
            ),
            bias=float(
                getattr(plugin_config, "yuying_adaptive_debounce_bias", 1.5) or 1.5
            ),
            min_wait=float(
                getattr(plugin_config, "yuying_adaptive_debounce_min_wait", 0.5) or 0.5
            ),
            max_wait=float(
                getattr(plugin_config, "yuying_adaptive_debounce_max_wait", 5.0) or 5.0
            ),
        )
    return _adaptive_debouncer


matcher = on_message(priority=10, block=False)


async def _process_normalized(bot: Bot, normalized: NormalizedMessage) -> None:
    """处理单条（可能已合并的）归一化消息。

    说明：
        - 此函数包含从"写入 raw_messages"开始到最后的所有处理逻辑
        - 会在防抖合并后或直接调用（防抖未启用时）
        - 原本在 handle_message 中的主链路逻辑已迁移到此处

    Args:
        bot: NoneBot Bot 实例
        normalized: 归一化后的消息（可能已经是多段合并的结果）
    """

    # 1) 写入 raw_messages（高优先级直写）
    raw_msg = RawMessage(
        qq_id=normalized.qq_id,
        scene_type=normalized.scene_type,
        scene_id=normalized.scene_id,
        timestamp=normalized.timestamp,
        msg_type=normalized.msg_type,
        content=normalized.content,
        raw_ref=normalized.raw_ref,
        reply_to_msg_id=normalized.reply_to_msg_id,
        mentioned_bot=normalized.mentioned_bot,
        is_effective=normalized.is_effective,
        is_bot=False,
    )
    raw_msg = await RawRepository.add(raw_msg)

    # 2) 图片预处理任务（图片描述/OCR），写入 index_jobs(item_type=ocr) 由 media_worker 处理
    for raw_ref, media_key in (normalized.image_ref_map or {}).items():
        try:
            if raw_ref in {"unknown", "None", ""}:
                continue
            cached = await MediaCacheRepository.get(media_key)
            if cached and cached.caption:
                continue

            url: Optional[str] = None
            if isinstance(raw_ref, str) and (raw_ref.startswith("http://") or raw_ref.startswith("https://")):
                url = raw_ref
            else:
                try:
                    info = await bot.get_image(file=raw_ref)
                    url = info.get("url") if isinstance(info, dict) else None
                except Exception:
                    url = None

            payload = {"media_key": media_key, "raw_ref": raw_ref, "url": url, "file_path": None}
            await db_writer.submit(
                AddIndexJobJob(
                    IndexJob(
                        item_type="ocr",
                        ref_id=media_key,
                        payload_json=json.dumps(payload, ensure_ascii=False),
                        status="pending",
                    )
                ),
                priority=5,
            )

            # 表情包偷取：仅群聊尝试
            if normalized.scene_type == "group" and url:
                asyncio.create_task(
                    _download_and_steal(
                        url=url,
                        media_key=media_key,
                        scene_id=normalized.scene_id,
                        source_qq_id=normalized.qq_id,
                        intent_hint=StickerSelector.infer_intent(normalized.content),
                    )
                )
        except Exception as exc:
            logger.debug(f"创建图片预处理任务失败，将继续：{exc}")

    # 3) 为检索增强写入索引任务（消息片段）
    try:
        await db_writer.submit(
            AddIndexJobJob(
                IndexJob(
                    item_type="msg_chunk",
                    ref_id=str(raw_msg.id),
                    payload_json="{}",
                    status="pending",
                )
            ),
            priority=5,
        )
    except Exception as exc:
        logger.warning(f"创建消息索引任务失败，将降级继续：{exc}")

    # 4) 推进摘要窗口（无论是否有效发言都计入窗口）
    try:
        await SummaryManager.on_message(
            normalized.scene_type,
            normalized.scene_id,
            raw_msg.id,
            raw_msg.timestamp,
        )
    except Exception as exc:
        logger.warning(f"摘要窗口推进失败，将降级继续：{exc}")

    # 5) 有效发言计数与记忆触发点
    if normalized.is_effective:
        await db_writer.submit_and_wait(
            AsyncCallableJob(ProfileRepository.increment_effective_count, args=(normalized.qq_id,)),
            priority=5,
        )
        await MemoryManager.mark_pending_if_needed(normalized.qq_id)
        MemoryManager.schedule_idle_extract(normalized.qq_id, normalized.scene_type, normalized.scene_id)

    # 6) 回复策略 - 计算directed_to_bot

    # ==================== 获取被引用消息信息（复用 API 调用） ====================

    # reply_to_message: 被引用消息信息（用于提示词显式注入）
    reply_to_message: Optional[Dict[str, Any]] = None

    # replied_to_bot: 是否回复了bot的消息
    # - 仅在群聊中检查(私聊中不需要,私聊本身就是对bot说的)
    # - 仅在未@bot时检查(已@bot则必回,无需额外判断)
    # - 需要reply_to_msg_id存在
    replied_to_bot = False
    if (
        normalized.scene_type == "group"
        and not normalized.mentioned_bot
        and normalized.reply_to_msg_id
    ):
        # 一次 get_msg 同时获取 sender_id + content，复用结果：
        # - 用 sender_id 判断 replied_to_bot（用于 directed_to_bot 计算）
        # - content 留作后续 prompt 注入（避免重复 API 调用）
        reply_info = await _get_reply_message_info(
            bot=bot,
            reply_to_msg_id=normalized.reply_to_msg_id,
        )
        reply_to_message = reply_info

        if reply_info and not bool(reply_info.get("failed")):
            # 判断回复的是否是 bot 的消息
            replied_to_bot = str(reply_info.get("sender_id") or "") == str(getattr(bot, "self_id", ""))

    # ==================== 计算directed_to_bot ====================

    # directed_to_bot: 消息是否直接对bot说的
    # - 私聊: 必然是对bot说的
    # - @bot: 明确指向bot
    # - 回复bot消息: 即使没@,也应视为对bot说的
    # 逻辑: (私聊) OR (@bot) OR (回复bot消息)
    directed_to_bot = (
        (normalized.scene_type == "private")  # 私聊
        or bool(normalized.mentioned_bot)  # @bot
        or bool(replied_to_bot)  # 回复bot消息
    )

    # ==================== 门控决策 ====================

    # 提前获取图片信息（用于心流模式判断和后续规划，避免重复调用）
    image_inputs = await _collect_image_inputs(bot=bot, image_ref_map=normalized.image_ref_map or {})

    should_reply = await Gatekeeper.should_reply(
        normalized.scene_type,
        normalized.scene_id,
        normalized.content,
        mentioned_bot=normalized.mentioned_bot,
        directed_to_bot=directed_to_bot,
        image_inputs=image_inputs,  # 传入图片信息用于心流模式
        raw_msg_id=raw_msg.id,  # 传入消息ID用于精确排除上下文
    )
    # Gatekeeper 统一决策(内部已处理 directed_to_bot 情况)
    if not should_reply:
        return

    # 6.5) 若本轮确定要回复，且存在引用但尚未获取引用详情，则补一次 get_msg 用于 prompt 注入
    if normalized.reply_to_msg_id and reply_to_message is None:
        reply_to_message = await _get_reply_message_info(
            bot=bot,
            reply_to_msg_id=normalized.reply_to_msg_id,
        )

    # 6.6) 若当前消息包含图片，且当前轮需要回复，则尝试同步补全图片说明（减少"看不懂图片"的情况）
    if normalized.image_ref_map and directed_to_bot:
        new_content = normalized.content
        changed = False
        for raw_ref, media_key in list((normalized.image_ref_map or {}).items())[:2]:
            cached = await MediaCacheRepository.get(media_key)
            if cached and cached.caption:
                caption = cached.caption.strip()
                if caption and f"[image:{media_key}:" not in new_content:
                    short = caption[:20] + ("…" if len(caption) > 20 else "")
                    new_content = new_content.replace(f"[image:{media_key}]", f"[image:{media_key}:{short}]")
                    changed = True
                continue

            url: Optional[str] = None
            if isinstance(raw_ref, str) and (raw_ref.startswith("http://") or raw_ref.startswith("https://")):
                url = raw_ref
            else:
                try:
                    info = await bot.get_image(file=raw_ref)
                    url = info.get("url") if isinstance(info, dict) else None
                except Exception:
                    url = None

            caption, ocr_text = await _try_caption_and_ocr(url=url, file_path=None)
            if caption:
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
                short = caption[:20] + ("…" if len(caption) > 20 else "")
                if f"[image:{media_key}:" not in new_content:
                    new_content = new_content.replace(f"[image:{media_key}]", f"[image:{media_key}:{short}]")
                    changed = True

        if changed and new_content != normalized.content:
            normalized.content = new_content
            try:
                await db_writer.submit_and_wait(
                    AsyncCallableJob(RawRepository.update_content, args=(raw_msg.id, new_content)),
                    priority=5,
                )
                # 追加一次 msg_chunk 索引任务，用于覆盖写入含图片说明的新内容
                await db_writer.submit(
                    AddIndexJobJob(
                        IndexJob(
                            item_type="msg_chunk",
                            ref_id=str(raw_msg.id),
                            payload_json="{}",
                            status="pending",
                        )
                    ),
                    priority=5,
                )
            except Exception as exc:
                logger.debug(f"补全图片说明后更新消息失败：{exc}")

    # 7) 检索（混合查询 + 记忆 + 检索增强）
    query = await Retriever.build_hybrid_query(
        normalized.qq_id,
        normalized.scene_type,
        normalized.scene_id,
        normalized.content,
    )
    # logger.debug(f"【检索模块】构建的混合查询: {query}")
    context = await Retriever.retrieve(
        normalized.qq_id,
        normalized.scene_type,
        normalized.scene_id,
        query,
    )
    # logger.debug(f"【检索模块】返回的上下文: {context}")
    # 8) 动作规划
    recent_dialogue = await _build_recent_dialogue(
        current_qq_id=normalized.qq_id,
        scene_type=normalized.scene_type,
        scene_id=normalized.scene_id,
        current_raw_msg_id=raw_msg.id,
        max_lines=max(0, int(getattr(plugin_config, "yuying_recent_dialogue_max_lines", 30) or 30)),
    )
    # logger.debug(f"【检索模块】返回的最近对话: {recent_dialogue}")
    # image_inputs 已在门控决策前获取，此处复用避免重复调用
    # logger.debug(f"【检索模块】返回的图片输入: {image_inputs}")

    # 8.5) 构建提示词元信息(META)
    # - 让LLM明确知道消息属性,避免误判是否对bot说的
    prompt_meta: Dict[str, Any] = {
        "scene_type": str(normalized.scene_type),  # 场景类型(group/private)
        "mentioned_bot": bool(normalized.mentioned_bot),  # 是否@了bot
        "replied_to_bot": bool(replied_to_bot),  # 是否回复了bot消息
        "directed_to_bot": bool(directed_to_bot),  # 是否直接对bot说的(综合判断)
    }
    try:
        dt = datetime.fromtimestamp(int(getattr(raw_msg, "timestamp", 0) or 0))
        prompt_meta["message_time"] = dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        pass

    # 9) 动作规划
    actions = await ActionPlanner.plan_actions(
        normalized.content,
        context.get("memories", []),
        context.get("rag_snippets", []),
        recent_dialogue=recent_dialogue,
        reply_to_message=reply_to_message,
        images=image_inputs,
        meta=prompt_meta,
        context_qq_id=normalized.qq_id,
        context_scene_type=normalized.scene_type,
        context_scene_id=normalized.scene_id,
        context_raw_msg_id=raw_msg.id,
    )

    # 10) 分段发送（含表情包）
    await ActionSender.send_actions(
        bot=bot,
        matcher=matcher,
        scene_type=normalized.scene_type,
        scene_id=normalized.scene_id,
        actions=actions,
    )


@matcher.handle()
async def handle_message(bot: Bot, event: Event) -> None:
    """处理单条入站消息。

    说明：
        - 如果启用自适应防抖，消息会被缓冲并拼接
        - 否则，消息会直接进入处理流程
    """

    inbound = LagrangeParser.parse_event(event)
    if not inbound:
        return

    normalized = await Normalizer.normalize(inbound)

    # 检查是否启用自适应防抖
    if bool(getattr(plugin_config, "yuying_adaptive_debounce_enabled", False)):
        mode = str(
            getattr(plugin_config, "yuying_adaptive_debounce_mode", "full") or "full"
        )
        if mode != "full":
            # 当前仅支持 full 模式，其他模式降级为不使用防抖
            await _process_normalized(bot, normalized)
            return

        # 构建防抖 key（按场景+用户维度）
        key = f"{normalized.scene_type}:{normalized.scene_id}:{normalized.qq_id}"
        debouncer = get_adaptive_debouncer()

        # 提交到防抖缓冲区
        await debouncer.submit(
            key=key,
            part=normalized,
            flush_cb=lambda merged: _process_normalized(bot, merged),
        )
        return

    # 防抖未启用，直接处理
    await _process_normalized(bot, normalized)
