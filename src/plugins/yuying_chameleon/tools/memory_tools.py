"""记忆工具：允许 AI 主动写入重要记忆。

设计原理：
- 让 AI 在对话中识别到重要信息时，主动调用工具写入记忆（active 层）
- 与定时批量提取（3 AM）并行，形成双轨记忆系统
- 内置速率限制，防止滥用
- 自动去重（由 MemoryManager 处理）

使用场景：
1. 用户明确表达重要偏好/习惯/目标
2. 用户纠正之前的错误信息
3. 突发重要事件（生日、考试、面试等）
4. 用户主动要求"记住这个"

不适用场景：
- 日常闲聊（由定时提取处理）
- 临时性/一次性信息
- 低确定性推测
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from nonebot import logger

from ..memory.memory_manager import MemoryManager
from .rate_limiter import get_rate_limiter


# 记忆类型的合法值（与数据库枚举一致）
VALID_MEMORY_TYPES = {"fact", "preference", "habit", "goal", "experience"}

# 可见性的合法值（与数据库枚举一致）
VALID_VISIBILITIES = {"global", "scene", "private"}


async def create_memory(
    *,
    content: str,
    type: str = "fact",
    confidence: float = 0.9,
    visibility: str = "global",
    scope_scene_id: Optional[str] = "",
    ttl_days: int = 0,
    # 上下文参数（由工具管理器注入）
    _context_qq_id: str,
    _context_scene_type: str,
    _context_scene_id: str,
    _context_raw_msg_id: int,
) -> str:
    """创建新记忆（AI 主动调用）。

    说明：
    - 本工具允许 AI 在对话中主动记录重要信息
    - 自动去重：如果内容相似度 >= 0.86，会更新现有记忆而非创建新记忆
    - 速率限制：每个会话最多 3-5 条（群聊/私聊不同），每天最多 25-40 条
    - 证据链接：自动关联当前消息作为证据

    参数说明：
        content: 记忆内容（必填）
            - 类型: 字符串，1-500 字符
            - 要求: 清晰、准确、完整的陈述句
            - 示例: "用户是 Python 程序员，主要使用 Django 框架"

        type: 记忆类型（可选，默认 "fact"）
            - fact: 客观事实（如职业、住址）
            - preference: 个人偏好（如喜欢/讨厌什么）
            - habit: 行为习惯（如作息、使用习惯）
            - goal: 目标计划（如学习目标、项目计划）
            - experience: 重要经历（如生日、面试）

        confidence: 置信度（可选，默认 0.9）
            - 类型: 浮点数，0.0-1.0
            - 建议值:
              - 0.95-1.0: 用户明确陈述的事实
              - 0.85-0.95: 从对话中合理推断
              - 0.7-0.85: 不太确定的推测（慎用）
            - 注意: < 0.7 的记忆不建议写入

        visibility: 可见性范围（可选，默认 "global"）
            - global: 全局记忆（所有场景可见）
            - scene: 场景专属记忆（仅当前群聊/私聊可见）
            - private: 仅私聊可见（不会在群聊注入）

        scope_scene_id: 场景 ID（可选）
            - 仅当 visibility="scene" 时需要
            - 默认使用当前场景

        ttl_days: 生存时间（可选，默认 0=使用系统默认 TTL）
            - 类型: 整数，天数
            - 0: 使用系统默认 TTL（memory_active_ttl_days）
            - > 0: 指定 TTL，N 天后自动过期（如"明天要开会"）

    返回值（JSON 字符串）：
        成功: {"ok": true, "message": "记忆已创建", "dedup": false}
        成功（去重）: {"ok": true, "message": "记忆已更新", "dedup": true}
        失败（速率限制）: {"ok": false, "error": "rate_limit_exceeded", "detail": "..."}
        失败（参数错误）: {"ok": false, "error": "invalid_parameter", "detail": "..."}

    使用建议：
    1. 仅在确实重要时调用（不要记录日常闲聊）
    2. 置信度 < 0.8 时谨慎使用
    3. 优先使用 type="goal" 和 type="preference"（优先级高）
    4. content 应该是完整陈述句，避免碎片化
    5. 同一信息不要重复创建（自动去重会合并）

    示例：
        # 用户明确表达偏好
        await create_memory(
            content="用户喜欢吃辣，不喜欢甜食",
            type="preference",
            confidence=0.95,
            visibility="global",
        )

        # 用户提到重要目标
        await create_memory(
            content="用户计划在 3 月份考研",
            type="goal",
            confidence=0.9,
            visibility="global",
        )

        # 临时提醒（指定 TTL）
        await create_memory(
            content="用户明天下午 3 点有会议",
            type="experience",
            confidence=1.0,
            ttl_days=1,
        )
    """

    # ==================== 步骤 1: 参数校验 ====================

    # 校验 content
    if not isinstance(content, str) or not content.strip():
        return json.dumps(
            {
                "ok": False,
                "error": "invalid_parameter",
                "detail": "content 不能为空",
            },
            ensure_ascii=False,
        )

    content = content.strip()
    if len(content) < 1:
        return json.dumps(
            {
                "ok": False,
                "error": "invalid_parameter",
                "detail": "content 长度不能小于 1 字符",
            },
            ensure_ascii=False,
        )

    if len(content) > 500:
        return json.dumps(
            {
                "ok": False,
                "error": "invalid_parameter",
                "detail": f"content 长度不能超过 500 字符（当前 {len(content)}）",
            },
            ensure_ascii=False,
        )

    # MemoryManager 会将 content 截断到 50 字符；这里提前做一致化，避免“写入后被悄悄截断”
    if len(content) > 50:
        content = content[:50]

    # 校验 type
    if type not in VALID_MEMORY_TYPES:
        return json.dumps(
            {
                "ok": False,
                "error": "invalid_parameter",
                "detail": f"type 必须是以下之一: {', '.join(VALID_MEMORY_TYPES)}",
            },
            ensure_ascii=False,
        )

    # 校验 confidence
    try:
        confidence = float(confidence)
        if not (0.0 <= confidence <= 1.0):
            raise ValueError()
    except (ValueError, TypeError):
        return json.dumps(
            {
                "ok": False,
                "error": "invalid_parameter",
                "detail": "confidence 必须是 0.0-1.0 之间的数字",
            },
            ensure_ascii=False,
        )

    # 低置信度拒绝（避免低质量记忆）
    if confidence < 0.7:
        return json.dumps(
            {
                "ok": False,
                "error": "low_confidence",
                "detail": f"置信度 {confidence} 过低（< 0.7），建议不要写入记忆",
            },
            ensure_ascii=False,
        )

    # 校验 visibility
    if visibility not in VALID_VISIBILITIES:
        return json.dumps(
            {
                "ok": False,
                "error": "invalid_parameter",
                "detail": f"visibility 必须是以下之一: {', '.join(VALID_VISIBILITIES)}",
            },
            ensure_ascii=False,
        )

    # 校验 ttl_days
    try:
        ttl_days = int(ttl_days)
        if ttl_days < 0:
            raise ValueError()
    except (ValueError, TypeError):
        return json.dumps(
            {
                "ok": False,
                "error": "invalid_parameter",
                "detail": "ttl_days 必须是 >= 0 的整数",
            },
            ensure_ascii=False,
        )

    # ==================== 步骤 2: 速率限制检查 ====================

    limiter = get_rate_limiter()
    allowed, reason = limiter.check_and_increment(
        qq_id=_context_qq_id,
        scene_type=_context_scene_type,
        dry_run=False,  # 实际扣除额度
    )

    if not allowed:
        # 获取当前状态用于返回详细信息
        status = limiter.get_status(_context_qq_id, _context_scene_type)

        if reason == "session_limit_exceeded":
            detail = (
                f"本会话记忆写入次数已达上限 "
                f"({status['session_count']}/{status['session_limit']})，"
                f"请在下次对话中继续"
            )
        elif reason == "daily_limit_exceeded":
            detail = (
                f"今日记忆写入次数已达上限 "
                f"({status['daily_count']}/{status['daily_limit']})，"
                f"明天再试"
            )
        else:
            detail = f"速率限制：{reason}"

        logger.info(f"记忆工具速率限制：{_context_qq_id} - {detail}")

        return json.dumps(
            {
                "ok": False,
                "error": "rate_limit_exceeded",
                "detail": detail,
            },
            ensure_ascii=False,
        )

    # ==================== 步骤 3: 构建记忆数据 ====================

    # 处理 scope_scene_id
    if visibility == "scene":
        if not (scope_scene_id or "").strip():
            scope_scene_id = _context_scene_id
    else:
        scope_scene_id = None  # global/private 记忆不需要 scope_scene_id

    # 处理 ttl_days
    ttl_days_value = ttl_days if ttl_days > 0 else None

    memory_data: Dict[str, Any] = {
        "type": type,
        "content": content,
        "confidence": confidence,
        "visibility": visibility,
        "scope_scene_id": scope_scene_id,
        "ttl_days": ttl_days_value,
        "evidence_msg_ids": [_context_raw_msg_id],  # 关联当前消息作为证据
    }

    memories_list: List[Dict[str, Any]] = [memory_data]

    # ==================== 步骤 4: 调用记忆管理器写入 ====================

    try:
        # 记录日志（用于调试和审计）
        logger.info(
            f"AI 主动创建记忆：用户={_context_qq_id}, 类型={type}, "
            f"置信度={confidence:.2f}, 内容=\"{content[:50]}...\""
        )

        # 调用现有的 upsert_memories（自动去重）
        await MemoryManager.upsert_memories(
            qq_id=_context_qq_id,
            memories_data=memories_list,
        )

        # 成功返回
        # 注意：无法直接判断是否去重（需要修改 MemoryManager.upsert_memories 返回值）
        # 这里统一返回成功，去重信息在日志中体现
        return json.dumps(
            {
                "ok": True,
                "message": "记忆已写入（如果相似记忆已存在，会自动更新而非重复创建）",
            },
            ensure_ascii=False,
        )

    except Exception as exc:
        logger.error(f"记忆工具写入失败：{exc}", exc_info=True)

        return json.dumps(
            {
                "ok": False,
                "error": "write_failed",
                "detail": f"写入记忆时发生异常：{type(exc).__name__}",
            },
            ensure_ascii=False,
        )
