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
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from nonebot import get_driver, on_message
from nonebot.adapters.onebot.v11 import Bot, Event
from nonebot.plugin import PluginMetadata
from nonebot import logger

from .config import Config
from .memory.memory_manager import MemoryManager
from .planner.action_planner import ActionPlanner
from .planner.action_sender import ActionSender
from .policy.gatekeeper import Gatekeeper
from .retrieval.retriever import Retriever
from .scheduler.jobs import init_scheduler
from .storage.db_writer import db_writer
from .storage.migrations_runner import run_migrations
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
from .normalize.normalizer import Normalizer
from .llm.vision import VisionHelper
from .paths import assets_dir
from .storage.write_jobs import AddIndexJobJob
from .storage.write_jobs import AsyncCallableJob

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
) -> str:
    t = (text or "").strip().replace("\n", " ")
    if len(t) > 120:
        t = t[:120] + "…"
    if is_bot:
        who = "你"
    elif sender_qq_id == current_qq_id:
        who = "我"
    else:
        who = f"群友({sender_qq_id})"
    return f"{who}：{t}"


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


async def _is_reply_to_bot(*, bot: Bot, reply_to_msg_id: Optional[int]) -> bool:
    """判断当前消息是否回复了机器人消息。

    说明:
        - reply_to_msg_id 来自 OneBot 的 reply.message_id（平台消息 id）
        - 与 raw_messages.id（数据库自增 id）不是同一个 id 空间
        - 通过 OneBot API `get_msg` 查询被回复消息的 sender.user_id 来判断
        - 用于识别"群聊中回复bot消息但没@"的场景

    Args:
        bot: NoneBot Bot实例,用于调用OneBot API
        reply_to_msg_id: 被回复的消息ID,来自event.reply.message_id

    Returns:
        bool: True表示回复的是bot消息,False表示不是或查询失败

    异常处理:
        - reply_to_msg_id为空: 返回False
        - 类型转换失败: 返回False
        - API调用失败: 返回False(降级处理,不影响主流程)
        - sender信息缺失: 返回False

    Example:
        >>> # 群聊中A回复了bot的消息但没@bot
        >>> replied = await _is_reply_to_bot(bot=bot, reply_to_msg_id=12345)
        >>> print(replied)
        # True (这条消息应该被认为是对bot说的)
    """

    # ==================== 步骤1: 参数校验 ====================

    # reply_to_msg_id为空: 没有回复任何消息
    if not reply_to_msg_id:
        return False

    # ==================== 步骤2: 类型转换 ====================

    try:
        # 确保reply_to_msg_id是整数类型
        mid = int(reply_to_msg_id)
    except Exception:
        # 类型转换失败: reply_to_msg_id格式错误
        return False

    # ==================== 步骤3: 调用OneBot API查询被回复的消息 ====================

    try:
        # await bot.call_api("get_msg", message_id=mid): 查询消息详情
        # - API: get_msg
        # - 参数: message_id (平台消息ID)
        # - 返回: 消息详情字典,包含sender/message/time等字段
        # - 注意: 这是平台API调用,可能失败(消息已删除/权限不足等)
        data = await bot.call_api("get_msg", message_id=mid)
    except Exception:
        # API调用失败: 消息可能已删除、权限不足、网络错误等
        # 降级处理: 返回False,不影响主流程
        return False

    # ==================== 步骤4: 提取发送者ID ====================

    sender_id: Optional[str] = None

    # 检查返回数据是否为字典类型
    if isinstance(data, dict):
        # 提取sender字段
        sender = data.get("sender")

        # 检查sender是否为字典类型
        if isinstance(sender, dict):
            # 提取user_id字段
            uid = sender.get("user_id")

            # 确保user_id存在
            if uid is not None:
                # 转换为字符串(统一比较格式)
                sender_id = str(uid)

    # sender_id提取失败: 返回数据格式异常
    if not sender_id:
        return False

    # ==================== 步骤5: 比较发送者与bot的ID ====================

    # getattr(bot, "self_id", ""): 获取bot自己的ID
    # - bot.self_id: bot的QQ号
    # - 默认值"": 防止属性不存在时报错
    # str(...) == str(...): 统一转为字符串比较,避免类型不一致
    # - 返回True: 被回复的消息是bot发送的
    # - 返回False: 被回复的消息是其他用户发送的
    return str(sender_id) == str(getattr(bot, "self_id", ""))


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
            urllib.request.urlretrieve(url, dst)  # nosec - 运行时受配置控制

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

    async def _cap() -> str:
        return await VisionHelper.caption_image(file_path, url=url)

    async def _ocr() -> str:
        return await VisionHelper.ocr_image(file_path, url=url)

    try:
        caption_task = asyncio.create_task(_cap())
        ocr_task = asyncio.create_task(_ocr())
        done, pending = await asyncio.wait(
            {caption_task, ocr_task},
            timeout=max_wait_seconds,
        )
        for t in pending:
            t.cancel()
        caption = caption_task.result() if caption_task in done else ""
        ocr_text = ocr_task.result() if ocr_task in done else ""
        logger.debug(f"图片说明/OCR 结果：{caption} / {ocr_text}")
        return (caption or "").strip(), (ocr_text or "").strip()
    except Exception as exc:
        logger.debug(f"图片说明/OCR 失败：{exc}")
        return "", ""


@driver.on_startup
async def startup() -> None:
    """NoneBot 启动时执行：初始化依赖并启动后台任务。"""

    # 1) 初始化数据库表（优先 migrations，失败再兜底 create_all）
    try:
        try:
            await run_migrations()
        except Exception as exc:
            logger.warning(f"执行 migrations 失败，将回退为 create_all：{exc}")
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


matcher = on_message(priority=10, block=False)


@matcher.handle()
async def handle_message(bot: Bot, event: Event) -> None:
    """处理单条入站消息。"""

    inbound = LagrangeParser.parse_event(event)
    if not inbound:
        return

    normalized = await Normalizer.normalize(inbound)

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

    # ==================== 计算replied_to_bot ====================

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
        # 调用OneBot API查询被回复消息的发送者
        # - 如果发送者是bot自己,则replied_to_bot=True
        # - 失败降级为False,不影响主流程
        replied_to_bot = await _is_reply_to_bot(bot=bot, reply_to_msg_id=normalized.reply_to_msg_id)

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

    # 6.5) 若当前消息包含图片，且当前轮需要回复，则尝试同步补全图片说明（减少"看不懂图片"的情况）
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

    # 9) 动作规划
    actions = await ActionPlanner.plan_actions(
        normalized.content,
        context.get("memories", []),
        context.get("rag_snippets", []),
        recent_dialogue=recent_dialogue,
        images=image_inputs,
        meta=prompt_meta,
    )

    # 10) 分段发送（含表情包）
    await ActionSender.send_actions(
        bot=bot,
        matcher=matcher,
        scene_type=normalized.scene_type,
        scene_id=normalized.scene_id,
        actions=actions,
    )
