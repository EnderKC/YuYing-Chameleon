from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

from nonebot import logger
from sqlalchemy import and_, func, or_, select

from src.plugins.yuying_chameleon.llm.client import get_task_llm
from src.plugins.yuying_chameleon.storage.models import (
    BotPersonalityMemory,
    RawMessage,
    Summary,
)
from src.plugins.yuying_chameleon.storage.sqlalchemy_engine import get_session

from src.plugins.yuying_chameleon.personality.prompts import (
    PERSONALITY_CORE_SYSTEM_PROMPT,
    PERSONALITY_CORE_USER_PROMPT_TEMPLATE,
    PERSONALITY_PROMPT_VERSION,
    PERSONALITY_REFLECTION_SYSTEM_PROMPT,
    PERSONALITY_REFLECTION_USER_PROMPT_TEMPLATE,
)


TZ = ZoneInfo("Asia/Shanghai")


def _clamp01(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _extract_json_object(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty llm output")

    # 去掉 ```json ... ```
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I).strip()
        raw = re.sub(r"\s*```$", "", raw).strip()

    # 截取第一段 {...}
    l = raw.find("{")
    r = raw.rfind("}")
    if l != -1 and r != -1 and r > l:
        raw = raw[l : r + 1]

    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("llm json is not object")
    return data


def _yesterday_range_ts(now_ts: int) -> Tuple[int, int, str]:
    # 以 Asia/Shanghai 计算"昨日 00:00~24:00"
    now_local = datetime.fromtimestamp(now_ts, tz=TZ)
    y = now_local.date() - timedelta(days=1)
    start_local = datetime(y.year, y.month, y.day, 0, 0, 0, tzinfo=TZ)
    end_local = start_local + timedelta(days=1)
    start_ts = int(start_local.timestamp())
    end_ts = int(end_local.timestamp())
    return start_ts, end_ts, y.isoformat()


def _today_date_str(now_ts: int) -> str:
    return datetime.fromtimestamp(now_ts, tz=TZ).date().isoformat()


@dataclass(frozen=True)
class ReflectionInputs:
    memory_date: str
    start_ts: int
    end_ts: int
    stats: Dict[str, Any]
    summaries: Dict[str, Any]


class PersonalityReflectionService:
    """每日 04:00 执行：生成 recent 人格记忆，并更新 core 长期原则。"""

    @staticmethod
    async def run_daily_reflection() -> None:
        logger.info("开始执行每日人格反思任务...")

        now_ts = int(time.time())
        run_id = datetime.utcnow().strftime("%Y%m%d%H%M%S")

        # 获取人格反思任务的 LLM 客户端
        reflection_llm = get_task_llm("personality_reflection")
        model = getattr(reflection_llm, "model", None)

        start_ts, end_ts, memory_date = _yesterday_range_ts(now_ts)
        inputs = await PersonalityReflectionService._collect_inputs(
            memory_date=memory_date,
            start_ts=start_ts,
            end_ts=end_ts,
        )

        # 1) recent：昨日反思（3种类型）
        try:
            recent_payload = await PersonalityReflectionService._llm_reflect(inputs)
            await PersonalityReflectionService._upsert_memories(
                payload=recent_payload,
                tier="recent",
                memory_key=memory_date,
                memory_date=memory_date,
                window_start_ts=start_ts,
                window_end_ts=end_ts,
                run_id=run_id,
                model=model,
            )
        except Exception as exc:
            logger.error(f"每日人格反思（recent）失败：{exc}")
            # recent 失败时不更新 core，避免 core 被不完整数据污染
            return

        # 2) recent 清理：仅保留最近 7 天
        try:
            cutoff_ts = end_ts - 7 * 86400
            await PersonalityReflectionService._soft_delete_old_recent(cutoff_ts=cutoff_ts, now_ts=now_ts)
        except Exception as exc:
            logger.warning(f"清理过期 recent 失败（将继续）：{exc}")

        # 3) core：基于最近 7 天 recent 更新长期原则
        try:
            core_start_ts = end_ts - 7 * 86400
            recent_for_core = await PersonalityReflectionService._load_recent_for_core(
                start_ts=core_start_ts,
                end_ts=end_ts,
                top_group_ids=PersonalityReflectionService._top_group_ids(inputs.stats, limit=8),
            )
            core_payload = await PersonalityReflectionService._llm_core_update(
                today_date=_today_date_str(now_ts),
                start_ts=core_start_ts,
                end_ts=end_ts,
                stats=inputs.stats,
                recent_memories=recent_for_core,
            )
            await PersonalityReflectionService._upsert_memories(
                payload=core_payload,
                tier="core",
                memory_key="core",
                memory_date=_today_date_str(now_ts),
                window_start_ts=core_start_ts,
                window_end_ts=end_ts,
                run_id=run_id,
                model=model,
            )
        except Exception as exc:
            logger.error(f"更新 core 人格原则失败（保留旧 core）：{exc}")

        logger.info("每日人格反思任务结束。")

    @staticmethod
    def _top_group_ids(stats: Dict[str, Any], limit: int = 8) -> List[str]:
        items = stats.get("group_activity_top") or []
        out: List[str] = []
        for x in items:
            sid = str(x.get("scene_id") or "").strip()
            if not sid:
                continue
            out.append(sid)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    async def _collect_inputs(*, memory_date: str, start_ts: int, end_ts: int) -> ReflectionInputs:
        async with get_session() as session:
            # 群活跃度：有效用户消息数（排除 bot）
            group_activity_stmt = (
                select(RawMessage.scene_id, func.count(RawMessage.id).label("msg_count"))
                .where(
                    and_(
                        RawMessage.scene_type == "group",
                        RawMessage.timestamp >= start_ts,
                        RawMessage.timestamp < end_ts,
                        RawMessage.is_effective.is_(True),
                        RawMessage.is_bot.is_(False),
                    )
                )
                .group_by(RawMessage.scene_id)
                .order_by(func.count(RawMessage.id).desc())
                .limit(30)
            )
            group_activity_rows = (await session.execute(group_activity_stmt)).all()

            # 关系反思候选：@机器人最多的用户（按群）
            group_mentioned_stmt = (
                select(
                    RawMessage.scene_id,
                    RawMessage.qq_id,
                    func.count(RawMessage.id).label("mention_count"),
                )
                .where(
                    and_(
                        RawMessage.scene_type == "group",
                        RawMessage.timestamp >= start_ts,
                        RawMessage.timestamp < end_ts,
                        RawMessage.is_effective.is_(True),
                        RawMessage.is_bot.is_(False),
                        RawMessage.mentioned_bot.is_(True),
                    )
                )
                .group_by(RawMessage.scene_id, RawMessage.qq_id)
                .order_by(func.count(RawMessage.id).desc())
                .limit(80)
            )
            group_mentioned_rows = (await session.execute(group_mentioned_stmt)).all()

            # 私聊活跃度：对方用户的有效消息数（用 scene_id 聚合更稳定）
            private_activity_stmt = (
                select(RawMessage.scene_id, func.count(RawMessage.id).label("msg_count"))
                .where(
                    and_(
                        RawMessage.scene_type == "private",
                        RawMessage.timestamp >= start_ts,
                        RawMessage.timestamp < end_ts,
                        RawMessage.is_effective.is_(True),
                        RawMessage.is_bot.is_(False),
                    )
                )
                .group_by(RawMessage.scene_id)
                .order_by(func.count(RawMessage.id).desc())
                .limit(30)
            )
            private_activity_rows = (await session.execute(private_activity_stmt)).all()

            # 摘要：与昨日窗口有交集的摘要都算（避免窗口边界漏掉）
            summary_stmt = (
                select(
                    Summary.id,
                    Summary.scene_type,
                    Summary.scene_id,
                    Summary.window_start_ts,
                    Summary.window_end_ts,
                    Summary.summary_text,
                    Summary.topic_state_json,
                )
                .where(
                    and_(
                        Summary.window_end_ts > start_ts,
                        Summary.window_start_ts < end_ts,
                    )
                )
                .order_by(Summary.scene_type, Summary.scene_id, Summary.window_end_ts)
            )
            summary_rows = (await session.execute(summary_stmt)).all()

        group_activity_top = [
            {"scene_id": str(scene_id), "msg_count": int(msg_count)}
            for scene_id, msg_count in group_activity_rows
        ]
        group_mentioned_top = [
            {"scene_id": str(scene_id), "qq_id": str(qq_id), "mention_count": int(mention_count)}
            for scene_id, qq_id, mention_count in group_mentioned_rows
        ]
        private_activity_top = [
            {"scene_id": str(scene_id), "msg_count": int(msg_count)}
            for scene_id, msg_count in private_activity_rows
        ]

        stats: Dict[str, Any] = {
            "group_activity_top": group_activity_top,
            "group_mentioned_top": group_mentioned_top,
            "private_activity_top": private_activity_top,
        }

        # summaries 按 scene 聚合，并做长度控制（避免 LLM 输出失败）
        top_groups = {x["scene_id"] for x in group_activity_top[:10]}
        top_privates = {x["scene_id"] for x in private_activity_top[:10]}

        summaries_by_scene: Dict[str, List[Dict[str, Any]]] = {}
        for row in summary_rows:
            sid = str(row.scene_id)
            stype = str(row.scene_type)
            if stype == "group" and sid not in top_groups:
                continue
            if stype == "private" and sid not in top_privates:
                continue
            key = f"{stype}:{sid}"
            summaries_by_scene.setdefault(key, []).append(
                {
                    "id": int(row.id),
                    "scene_type": stype,
                    "scene_id": sid,
                    "window_start_ts": int(row.window_start_ts),
                    "window_end_ts": int(row.window_end_ts),
                    "summary_text": (row.summary_text or "").strip(),
                    "topic_state_json": (row.topic_state_json or "").strip(),
                }
            )
        for k in list(summaries_by_scene.keys()):
            summaries_by_scene[k] = summaries_by_scene[k][-8:]  # 每个 scene 最多 8 条

        return ReflectionInputs(
            memory_date=memory_date,
            start_ts=start_ts,
            end_ts=end_ts,
            stats=stats,
            summaries={"by_scene": summaries_by_scene},
        )

    @staticmethod
    async def _llm_reflect(inputs: ReflectionInputs) -> Dict[str, Any]:
        user_prompt = PERSONALITY_REFLECTION_USER_PROMPT_TEMPLATE.format(
            memory_date=inputs.memory_date,
            start_ts=inputs.start_ts,
            end_ts=inputs.end_ts,
            stats_json=_safe_json_dumps(inputs.stats),
            summaries_json=_safe_json_dumps(inputs.summaries),
        )
        messages = [
            {"role": "system", "content": PERSONALITY_REFLECTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        # 使用人格反思任务的 LLM 客户端
        reflection_llm = get_task_llm("personality_reflection")
        text = await reflection_llm.chat_completion(messages, temperature=0.2, return_result=False)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("personality_reflection LLM returned empty text")
        data = _extract_json_object(text)
        return data

    @staticmethod
    async def _llm_core_update(
        *,
        today_date: str,
        start_ts: int,
        end_ts: int,
        stats: Dict[str, Any],
        recent_memories: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        user_prompt = PERSONALITY_CORE_USER_PROMPT_TEMPLATE.format(
            today_date=today_date,
            start_ts=start_ts,
            end_ts=end_ts,
            stats_json=_safe_json_dumps(stats),
            recent_memories_json=_safe_json_dumps(list(recent_memories)),
        )
        messages = [
            {"role": "system", "content": PERSONALITY_CORE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        # 使用人格核心原则更新任务的 LLM 客户端
        core_llm = get_task_llm("personality_core")
        text = await core_llm.chat_completion(messages, temperature=0.2, return_result=False)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("personality_core LLM returned empty text")
        data = _extract_json_object(text)
        return data

    @staticmethod
    async def _load_recent_for_core(*, start_ts: int, end_ts: int, top_group_ids: List[str]) -> List[Dict[str, Any]]:
        # core 更新只用最近 7 天 recent（global + top groups）
        want_scopes = [("global", "")]
        for gid in top_group_ids:
            want_scopes.append(("group", str(gid)))

        async with get_session() as session:
            rows: List[BotPersonalityMemory] = []
            for scope_type, scope_id in want_scopes:
                stmt = (
                    select(BotPersonalityMemory)
                    .where(
                        and_(
                            BotPersonalityMemory.tier == "recent",
                            BotPersonalityMemory.deleted_at_ts.is_(None),
                            BotPersonalityMemory.scope_type == scope_type,
                            BotPersonalityMemory.scope_id == scope_id,
                            BotPersonalityMemory.window_end_ts >= start_ts,
                            BotPersonalityMemory.window_end_ts <= end_ts,
                        )
                    )
                    .order_by(BotPersonalityMemory.window_end_ts.desc(), BotPersonalityMemory.importance.desc())
                    .limit(60)
                )
                part = (await session.execute(stmt)).scalars().all()
                rows.extend(part)

        out: List[Dict[str, Any]] = []
        for r in rows:
            evidence = {}
            if r.evidence_json:
                try:
                    evidence = json.loads(r.evidence_json)
                except Exception:
                    evidence = {"raw": r.evidence_json[:800]}
            out.append(
                {
                    "tier": r.tier,
                    "memory_type": r.memory_type,
                    "scope_type": r.scope_type,
                    "scope_id": r.scope_id,
                    "memory_date": r.memory_date,
                    "title": r.title,
                    "content": r.content,
                    "action_hint": r.action_hint or "",
                    "confidence": float(r.confidence),
                    "importance": float(r.importance),
                    "emotion_label": r.emotion_label,
                    "emotion_valence": r.emotion_valence,
                    "decay_weight": r.decay_weight,
                    "decay_half_life_hours": r.decay_half_life_hours,
                    "evidence": evidence,
                }
            )
        return out

    @staticmethod
    async def _soft_delete_old_recent(*, cutoff_ts: int, now_ts: int) -> None:
        async with get_session() as session:
            stmt = (
                select(BotPersonalityMemory)
                .where(
                    and_(
                        BotPersonalityMemory.tier == "recent",
                        BotPersonalityMemory.deleted_at_ts.is_(None),
                        BotPersonalityMemory.window_end_ts < cutoff_ts,
                    )
                )
                .limit(5000)
            )
            rows = (await session.execute(stmt)).scalars().all()
            for r in rows:
                r.deleted_at_ts = now_ts
            await session.commit()

    @staticmethod
    async def _upsert_memories(
        *,
        payload: Dict[str, Any],
        tier: str,
        memory_key: str,
        memory_date: str,
        window_start_ts: int,
        window_end_ts: int,
        run_id: str,
        model: Optional[str],
    ) -> None:
        memories = payload.get("memories")
        if not isinstance(memories, list):
            raise ValueError("payload.memories must be list")

        async with get_session() as session:
            for m in memories:
                if not isinstance(m, dict):
                    continue

                memory_type = str(m.get("memory_type") or "").strip()
                scope_type = str(m.get("scope_type") or "").strip()
                scope_id = str(m.get("scope_id") or "").strip()

                if memory_type not in {"group_activity", "relationship", "emotion_state"}:
                    continue
                if scope_type not in {"global", "group"}:
                    continue
                if scope_type == "global":
                    scope_id = ""  # 强制 global scope_id 为空字符串，保证唯一约束幂等

                title = str(m.get("title") or "").strip()[:200]
                content = str(m.get("content") or "").strip()
                action_hint = str(m.get("action_hint") or "").strip()

                if not title or not content:
                    continue

                confidence = _clamp01(m.get("confidence", 0.5))
                importance = _clamp01(m.get("importance", 0.5))

                evidence_obj = m.get("evidence") if isinstance(m.get("evidence"), dict) else {}
                evidence_json = _safe_json_dumps(evidence_obj) if evidence_obj else None

                emotion_label = m.get("emotion_label")
                emotion_valence = m.get("emotion_valence")
                decay_weight = m.get("decay_weight")
                decay_half_life_hours = m.get("decay_half_life_hours")

                if memory_type != "emotion_state":
                    emotion_label = None
                    emotion_valence = None
                    decay_weight = None
                    decay_half_life_hours = None
                else:
                    # 兜底：缺省半衰期默认为 24
                    try:
                        if decay_half_life_hours is None:
                            decay_half_life_hours = 24.0
                    except Exception:
                        decay_half_life_hours = 24.0

                find_stmt = select(BotPersonalityMemory).where(
                    and_(
                        BotPersonalityMemory.tier == tier,
                        BotPersonalityMemory.scope_type == scope_type,
                        BotPersonalityMemory.scope_id == scope_id,
                        BotPersonalityMemory.memory_type == memory_type,
                        BotPersonalityMemory.memory_key == memory_key,
                    )
                )
                existing = (await session.execute(find_stmt)).scalars().first()

                if existing is None:
                    row = BotPersonalityMemory(
                        tier=tier,
                        memory_type=memory_type,
                        scope_type=scope_type,
                        scope_id=scope_id,
                        memory_key=memory_key,
                        memory_date=memory_date,
                        window_start_ts=int(window_start_ts),
                        window_end_ts=int(window_end_ts),
                        title=title,
                        content=content,
                        action_hint=action_hint,
                        confidence=float(confidence),
                        importance=float(importance),
                        emotion_label=str(emotion_label).strip() if emotion_label is not None else None,
                        emotion_valence=float(emotion_valence) if emotion_valence is not None else None,
                        decay_weight=float(decay_weight) if decay_weight is not None else None,
                        decay_half_life_hours=float(decay_half_life_hours) if decay_half_life_hours is not None else None,
                        evidence_json=evidence_json,
                        run_id=run_id,
                        model=model,
                        prompt_version=PERSONALITY_PROMPT_VERSION,
                        deleted_at_ts=None,
                    )
                    session.add(row)
                else:
                    existing.memory_date = memory_date
                    existing.window_start_ts = int(window_start_ts)
                    existing.window_end_ts = int(window_end_ts)
                    existing.title = title
                    existing.content = content
                    existing.action_hint = action_hint
                    existing.confidence = float(confidence)
                    existing.importance = float(importance)
                    existing.evidence_json = evidence_json
                    existing.run_id = run_id
                    existing.model = model
                    existing.prompt_version = PERSONALITY_PROMPT_VERSION
                    existing.deleted_at_ts = None

                    existing.emotion_label = str(emotion_label).strip() if emotion_label is not None else None
                    existing.emotion_valence = float(emotion_valence) if emotion_valence is not None else None
                    existing.decay_weight = float(decay_weight) if decay_weight is not None else None
                    existing.decay_half_life_hours = float(decay_half_life_hours) if decay_half_life_hours is not None else None

            await session.commit()
