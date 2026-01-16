from __future__ import annotations

import json
import math
import time
from typing import List

from sqlalchemy import and_, select

from src.plugins.yuying_chameleon.storage.models import BotPersonalityMemory
from src.plugins.yuying_chameleon.storage.sqlalchemy_engine import get_session


def _effective_emotion_weight(row: BotPersonalityMemory, now_ts: int) -> float:
    if row.decay_weight is None or row.decay_half_life_hours is None:
        return 0.0
    try:
        half_life = float(row.decay_half_life_hours)
        if half_life <= 0:
            return 0.0
        age_hours = max(0.0, (now_ts - int(row.window_end_ts)) / 3600.0)
        return float(row.decay_weight) * math.exp(-age_hours * math.log(2.0) / half_life)
    except Exception:
        return 0.0


class PersonalityRetriever:
    @staticmethod
    async def render_system_injection(*, context_scene_type: str, context_scene_id: str) -> str:
        now_ts = int(time.time())
        blocks: List[str] = []

        # 全局人格记忆
        blocks.append(await PersonalityRetriever._render_scope(scope_type="global", scope_id="", now_ts=now_ts))

        # 群人格记忆（仅群聊注入）
        if str(context_scene_type or "").strip() == "group" and str(context_scene_id or "").strip():
            blocks.append(
                await PersonalityRetriever._render_scope(
                    scope_type="group",
                    scope_id=str(context_scene_id).strip(),
                    now_ts=now_ts,
                )
            )

        joined = "\n\n".join([b for b in blocks if b.strip()]).strip()
        if not joined:
            return ""
        return "【人格记忆（用于沟通方式与语气调节）】\n" + joined

    @staticmethod
    async def _render_scope(*, scope_type: str, scope_id: str, now_ts: int) -> str:
        cutoff_ts = now_ts - 7 * 86400
        async with get_session() as session:
            core_stmt = (
                select(BotPersonalityMemory)
                .where(
                    and_(
                        BotPersonalityMemory.tier == "core",
                        BotPersonalityMemory.deleted_at_ts.is_(None),
                        BotPersonalityMemory.scope_type == scope_type,
                        BotPersonalityMemory.scope_id == scope_id,
                    )
                )
                .order_by(BotPersonalityMemory.importance.desc(), BotPersonalityMemory.updated_at_ts.desc())
                .limit(12)
            )
            core_rows = (await session.execute(core_stmt)).scalars().all()

            recent_stmt = (
                select(BotPersonalityMemory)
                .where(
                    and_(
                        BotPersonalityMemory.tier == "recent",
                        BotPersonalityMemory.deleted_at_ts.is_(None),
                        BotPersonalityMemory.scope_type == scope_type,
                        BotPersonalityMemory.scope_id == scope_id,
                        BotPersonalityMemory.window_end_ts >= cutoff_ts,
                    )
                )
                .order_by(BotPersonalityMemory.window_end_ts.desc(), BotPersonalityMemory.importance.desc())
                .limit(10)
            )
            recent_rows = (await session.execute(recent_stmt)).scalars().all()

        lines: List[str] = []
        if scope_type == "group":
            lines.append(f"（群 {scope_id}）")
        else:
            lines.append("（全局）")

        if core_rows:
            lines.append("核心原则：")
            for r in core_rows[:8]:
                lines.append(f"- {r.title}：{(r.content or '').strip()[:220]}")

        if recent_rows:
            lines.append("近期观察(7天内)：")
            for r in recent_rows[:6]:
                d = (r.memory_date or "").strip()
                lines.append(f"- ({d}) {r.title}：{(r.content or '').strip()[:180]}")

        # 语气约束：仅当"不开心"且有效权重大于阈值时输出
        if scope_type == "global":
            emo = None
            for r in recent_rows:
                if r.memory_type == "emotion_state":
                    emo = r
                    break
            if emo is not None:
                val = None
                try:
                    val = float(emo.emotion_valence) if emo.emotion_valence is not None else None
                except Exception:
                    val = None
                eff = _effective_emotion_weight(emo, now_ts)
                if val is not None and val < 0 and eff >= 0.25:
                    lines.append("语气约束：")
                    lines.append("- 当前状态偏不开心：语气更平缓；更多澄清与确认；避免讽刺与玩笑升级。")

        return "\n".join(lines).strip()
