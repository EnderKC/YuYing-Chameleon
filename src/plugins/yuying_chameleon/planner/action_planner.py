"""动作规划：将“用户输入 + 记忆 + RAG”转换为可执行 actions。

说明：
- 本模块负责 prompt 组装、LLM 调用与 actions 校验。
- LLM 不可用时必须降级为可用的文本回复（不阻塞主流程）。
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from nonebot import logger

from ..config import plugin_config
from ..llm.client import main_llm
from ..storage.models import Memory


class ActionPlanner:
    """动作规划器。"""

    _cached_system_prompt: Optional[str] = None

    @staticmethod
    async def plan_actions(
        user_msg: str,
        memories: List[Memory],
        rag_context: List[str],
        recent_dialogue: Optional[List[str]] = None,
        images: Optional[Sequence[Dict[str, str]]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """规划本次回复动作列表。

        参数：
            user_msg: 用户消息（归一化文本）。
            memories: 注入上下文的记忆列表。
            rag_context: RAG 召回片段（字符串列表）。
            recent_dialogue: 最近对话片段（短、按时间顺序）。
            images: 当前消息关联的图片输入（多模态），元素包含 `url`/`media_key`/`caption` 可选字段。
            meta: 提示词元信息（建议包含 scene_type/mentioned_bot/replied_to_bot/directed_to_bot）。
                - 用途: 让LLM明确知道消息是否是对bot说的,避免误判
                - 示例: {"scene_type": "group", "mentioned_bot": False,
                         "replied_to_bot": True, "directed_to_bot": True}

        返回：
            List[Dict[str, Any]]: 动作列表（最少 1 条）。
        """

        prompt = ActionPlanner._build_prompt(
            user_msg, memories, rag_context, meta=meta, recent_dialogue=recent_dialogue
        )
        system_prompt = ActionPlanner._load_system_prompt()
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]

        # 多模态输入：将文本 prompt 与图片 url 合并到同一个 user message 的 content 数组里
        if images:
            content_parts: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
            for item in list(images)[:2]:
                url = (item.get("url") or "").strip()
                if not url:
                    continue
                content_parts.append({"type": "image_url", "image_url": {"url": url}})
            messages.append({"role": "user", "content": content_parts})
        else:
            messages.append({"role": "user", "content": prompt})
        # logger.debug(f"【动作规划模块】构建的提示词: {messages}")
        content = await main_llm.chat_completion(messages, temperature=0.7)

        actions = ActionPlanner._parse_actions(content)
        if actions:
            return actions

        # 降级：无法解析为动作列表时，回退为单条文本回复
        fallback = "我现在有点忙，等我一下。"
        if content and isinstance(content, str) and content.strip():
            fallback = content.strip()
        fallback = fallback[: int(plugin_config.yuying_reply_text_max_chars)]
        return [{"type": "text", "content": fallback}]

    @staticmethod
    def _build_prompt(
        user_msg: str,
        memories: List[Memory],
        rag_context: List[str],
        *,
        meta: Optional[Dict[str, Any]] = None,
        recent_dialogue: Optional[List[str]] = None,
    ) -> str:
        """组装用于大模型的提示词（尽量短、结构化）。

        说明:
            - 在提示词顶部注入META(JSON),明确告知LLM消息属性
            - 添加当前时间信息,让LLM能够根据真实时间回复
            - 根据directed_to_bot标志,添加对应的回复策略提示
            - 结构化组织记忆、RAG、对话历史等上下文信息

        Args:
            user_msg: 用户消息文本
            memories: 记忆列表
            rag_context: RAG检索结果
            meta: 元信息字典,包含scene_type/mentioned_bot/replied_to_bot/directed_to_bot
            recent_dialogue: 最近对话历史

        Returns:
            str: 格式化的提示词文本,包含时间、元信息、上下文等
        """

        # ==================== 步骤1: 构建META(JSON)元信息 ====================

        # 处理meta参数,确保是字典类型
        meta_obj: Dict[str, Any] = dict(meta or {})

        # 序列化为紧凑的JSON字符串
        # - ensure_ascii=False: 支持中文字符,不转义为\uXXXX
        # - separators=(",", ":"): 紧凑格式,无多余空格
        meta_json = json.dumps(meta_obj, ensure_ascii=False, separators=(",", ":"))

        # 提取directed_to_bot标志,用于决定回复策略提示
        directed_to_bot = bool(meta_obj.get("directed_to_bot", False))

        # ==================== 步骤2: 根据directed_to_bot生成策略提示 ====================

        if directed_to_bot:
            # 直接对bot说话: 正常回应,可以详细解答
            mode_hint = (
                "提示：directed_to_bot=true，这条消息主要是对你说的"
                "（私聊/被@/被回复）；按被点名/私聊正常回应。"
            )
        else:
            # 群聊插话: 简短附和,避免过度解读
            mode_hint = (
                "提示：directed_to_bot=false，这条消息不一定在对你说；"
                "把自己当群友插话，可极短附和/吐槽/用表情包/用'？'敷衍，"
                "别把它当成对你提问。"
            )

        # ==================== 步骤2.5: 生成当前时间信息 ====================

        # 获取当前时间
        now = datetime.now()

        # 中文星期映射
        # - weekday(): 返回0-6,0表示周一,6表示周日
        weekday_map = {
            0: "星期一",
            1: "星期二",
            2: "星期三",
            3: "星期四",
            4: "星期五",
            5: "星期六",
            6: "星期日",
        }

        # 格式化时间字符串
        # - 格式: "2025年12月28日 星期六 14:30"
        # - strftime("%Y年%m月%d日"): 年月日
        # - strftime("%H:%M"): 小时:分钟(24小时制)
        current_time = f"{now.strftime('%Y年%m月%d日')} {weekday_map[now.weekday()]} {now.strftime('%H:%M')}"

        # ==================== 步骤3: 格式化上下文信息 ====================

        # 记忆列表: 限制最多20条,格式为 "- (tier/type) content"
        # - tier: active/archive/core (记忆层级)
        # - type: fact/preference/habit/experience (记忆类型)
        mem_lines = [f"- ({m.tier}/{m.type}) {m.content}" for m in memories[:20]]

        # RAG检索结果: 限制最多40条,格式为 "- snippet"
        rag_lines = [f"- {s}" for s in rag_context[:40]]

        # 最近对话历史: 限制最多50条,格式为 "- dialogue_line"
        dialogue_lines = [f"- {s}" for s in (recent_dialogue or [])[:50]]

        # ==================== 步骤4: 组装完整提示词 ====================

        return "\n".join(
            [
                # 元信息区域
                f"## META(JSON)：{meta_json}",
                "",
                # 当前时间区域
                f"## 当前时间：{current_time}",
                "",
                # 用户消息区域
                f"## 用户消息：{user_msg}",
                "",
                # 最近对话区域
                "## 最近对话：",
                *(dialogue_lines or ["- （无）"]),
                "",
                # 记忆区域
                "## 记忆：",
                *(mem_lines or ["- （无）"]),
                "",
                # RAG检索区域
                "## RAG：",
                *(rag_lines or ["- （无）"]),
                "",
                # 策略提示区域
                mode_hint,
                "",
                # 输出格式约束
                "注意：只输出 JSON actions（不要解释）。",
            ]
        )

    @staticmethod
    def _load_system_prompt() -> str:
        """加载 system prompt（带缓存）。"""

        if ActionPlanner._cached_system_prompt:
            return ActionPlanner._cached_system_prompt

        try:
            plugin_dir = Path(__file__).resolve().parents[1]
            path = plugin_dir / "llm" / "system_prompt.md"
            ActionPlanner._cached_system_prompt = path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            logger.warning(f"读取 system_prompt.md 失败，将使用内置默认：{exc}")
            ActionPlanner._cached_system_prompt = "你是 QQ 群聊/私聊机器人，只能输出 JSON actions。"

        return ActionPlanner._cached_system_prompt

    @staticmethod
    def _parse_actions(text: Optional[str]) -> Optional[List[Dict[str, Any]]]:
        """解析并校验大模型输出的动作列表。"""

        if not text:
            return None

        raw = text.strip()
        data = ActionPlanner._extract_first_json_array(raw)
        if data is None:
            return None

        if not isinstance(data, list) or not data:
            return None

        actions: List[Dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            action_type = item.get("type")
            if action_type == "text":
                content = item.get("content")
                if not isinstance(content, str) or not content.strip():
                    continue
                content = content.strip()[: int(plugin_config.yuying_reply_text_max_chars)]
                actions.append({"type": "text", "content": content})
            elif action_type == "sticker":
                intent = item.get("intent")
                if not isinstance(intent, str) or not intent.strip():
                    continue
                actions.append({"type": "sticker", "intent": intent.strip()})

        if not actions:
            return None

        # 约束：一次最多 N 条（符合“人类节奏”策略）
        return actions[: int(plugin_config.yuying_action_max_count)]

    @staticmethod
    def _extract_first_json_array(text: str) -> Optional[object]:
        """从文本中提取第一个 JSON 数组。"""

        m = re.search(r"(\[.*\])", text, flags=re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(1))
        except Exception:
            return None
