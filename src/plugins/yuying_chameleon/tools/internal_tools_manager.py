"""内部工具管理器：管理机器人内部功能工具的注册、schema 生成和执行。

设计原理：
- 内部工具与 MCP 工具并存，共同提供给 LLM
- 内部工具总是可用（无连接/初始化开销）
- 工具名使用 "internal__" 前缀避免与 MCP 工具冲突
- 支持上下文注入（qq_id、scene_type、scene_id、raw_msg_id）

与 MCP 的区别：
- MCP: 外部工具，通过子进程/网络调用，可选启用
- Internal: 内部工具，直接 Python 函数调用，总是可用

工具调用流程：
1. ActionPlanner 合并 internal tools + MCP tools
2. LLM 返回 tool_calls
3. ActionPlanner 根据工具名前缀路由到对应执行器
4. Internal 工具执行时自动注入上下文参数
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from nonebot import logger

from .memory_tools import create_memory


class InternalToolsManager:
    """内部工具管理器（单例）。"""

    def __init__(self):
        # 工具注册表：{public_name: (function, schema)}
        self._registry: Dict[str, tuple[Callable, Dict[str, Any]]] = {}

        # 上下文字段（工具执行时注入）
        self._context_fields = {
            "_context_qq_id",
            "_context_scene_type",
            "_context_scene_id",
            "_context_raw_msg_id",
        }

        # 注册内置工具
        self._register_builtin_tools()

    def _register_builtin_tools(self) -> None:
        """注册内置工具。"""

        # 注册 create_memory 工具
        self.register_tool(
            public_name="internal__create_memory",
            function=create_memory,
            schema={
                "type": "function",
                "function": {
                    "name": "internal__create_memory",
                    "description": (
                        "创建新记忆（AI 主动调用）。\n\n"
                        "**何时使用本工具：**\n"
                        "1. 用户明确表达重要偏好、习惯、目标时\n"
                        "2. 用户纠正之前的错误信息时\n"
                        "3. 突发重要事件（生日、考试、面试等）\n"
                        "4. 用户主动要求\"记住这个\"时\n\n"
                        "**何时不要使用：**\n"
                        "1. 日常闲聊（由系统自动提取）\n"
                        "2. 临时性/一次性信息\n"
                        "3. 低确定性推测（置信度 < 0.8）\n"
                        "4. 已经记录过的信息（自动去重）\n\n"
                        "**速率限制：** 每个会话最多 3-5 条，每天最多 25-40 条\n\n"
                        "**返回值：** JSON 格式 {ok: bool, message/error: str}"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": (
                                    "记忆内容（必填）。要求：\n"
                                    "- 清晰、准确、完整的陈述句\n"
                                    "- 1-500 字符\n"
                                    "- 示例：\"用户是 Python 程序员，主要使用 Django 框架\""
                                ),
                            },
                            "type": {
                                "type": "string",
                                "enum": ["fact", "preference", "habit", "goal", "experience"],
                                "description": (
                                    "记忆类型（可选，默认 fact）：\n"
                                    "- fact: 客观事实（如职业、住址）\n"
                                    "- preference: 个人偏好（如喜欢/讨厌什么）[推荐优先]\n"
                                    "- habit: 行为习惯（如作息、使用习惯）\n"
                                    "- goal: 目标计划（如学习目标、项目计划）[推荐优先]\n"
                                    "- experience: 重要经历（如生日、面试）"
                                ),
                                "default": "fact",
                            },
                            "confidence": {
                                "type": "number",
                                "description": (
                                    "置信度（可选，默认 0.9）。范围 0.0-1.0：\n"
                                    "- 0.95-1.0: 用户明确陈述的事实\n"
                                    "- 0.85-0.95: 从对话中合理推断\n"
                                    "- 0.7-0.85: 不太确定的推测（慎用）\n"
                                    "- < 0.7: 不建议写入（会被拒绝）"
                                ),
                                "default": 0.9,
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                            "visibility": {
                                "type": "string",
                                "enum": ["global", "scene", "private"],
                                "description": (
                                    "可见性范围（可选，默认 global）：\n"
                                    "- global: 全局记忆（所有场景可见）\n"
                                    "- scene: 场景专属记忆（仅当前群聊/私聊可见）\n"
                                    "- private: 仅私聊可见（不会在群聊注入）"
                                ),
                                "default": "global",
                            },
                            "scope_scene_id": {
                                "type": "string",
                                "description": (
                                    "场景 ID（可选）。仅当 visibility=scene 时需要，"
                                    "默认使用当前场景"
                                ),
                                "default": "",
                            },
                            "ttl_days": {
                                "type": "integer",
                                "description": (
                                    "生存时间（可选，默认 0=使用系统默认 TTL）。单位：天\n"
                                    "- 0: 使用系统默认 TTL（memory_active_ttl_days）\n"
                                    "- > 0: 指定 TTL，N 天后自动过期（如\"明天要开会\"）"
                                ),
                                "default": 0,
                                "minimum": 0,
                            },
                        },
                        "required": ["content"],
                        "additionalProperties": False,
                    },
                },
            },
        )

        logger.info(f"内部工具已注册：{len(self._registry)} 个工具")

    def register_tool(
        self,
        public_name: str,
        function: Callable,
        schema: Dict[str, Any],
    ) -> None:
        """注册一个内部工具。

        Args:
            public_name: 工具的公开名称（必须以 "internal__" 开头）
            function: 工具的实现函数（异步）
            schema: OpenAI function calling schema
        """
        if not public_name.startswith("internal__"):
            raise ValueError(f"内部工具名必须以 'internal__' 开头：{public_name}")

        if public_name in self._registry:
            logger.warning(f"内部工具 {public_name} 已存在，将被覆盖")

        self._registry[public_name] = (function, schema)
        logger.debug(f"注册内部工具：{public_name}")

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """获取所有内部工具的 OpenAI tools schema。

        返回格式：
        [
            {
                "type": "function",
                "function": {
                    "name": "internal__create_memory",
                    "description": "...",
                    "parameters": {...}
                }
            },
            ...
        ]
        """
        tools: List[Dict[str, Any]] = []

        for public_name, (_, schema) in self._registry.items():
            tools.append(schema)

        return tools

    async def call_tool(
        self,
        public_name: str,
        arguments: Dict[str, Any],
        *,
        context_qq_id: str,
        context_scene_type: str,
        context_scene_id: str,
        context_raw_msg_id: int,
    ) -> str:
        """执行内部工具调用。

        Args:
            public_name: 工具名（如 "internal__create_memory"）
            arguments: 工具参数（从 LLM tool_calls 提取）
            context_qq_id: 当前用户 QQ 号
            context_scene_type: 当前场景类型（group/private）
            context_scene_id: 当前场景 ID
            context_raw_msg_id: 当前消息 ID（用于关联证据）

        Returns:
            str: 工具执行结果（JSON 字符串）
        """

        # 检查工具是否存在
        if public_name not in self._registry:
            logger.warning(f"调用了不存在的内部工具：{public_name}")
            return json.dumps(
                {
                    "ok": False,
                    "error": "tool_not_found",
                    "detail": f"内部工具 {public_name} 不存在",
                },
                ensure_ascii=False,
            )

        function, _ = self._registry[public_name]

        # 注入上下文参数
        kwargs = dict(arguments)
        kwargs["_context_qq_id"] = context_qq_id
        kwargs["_context_scene_type"] = context_scene_type
        kwargs["_context_scene_id"] = context_scene_id
        kwargs["_context_raw_msg_id"] = context_raw_msg_id

        # 执行工具
        try:
            result = await function(**kwargs)

            # 工具应该返回 JSON 字符串
            if isinstance(result, str):
                return result
            else:
                # 如果返回的不是字符串，尝试序列化
                return json.dumps(result, ensure_ascii=False)

        except TypeError as exc:
            # 参数错误（如缺少必需参数、类型不匹配）
            logger.error(f"内部工具 {public_name} 参数错误：{exc}")
            return json.dumps(
                {
                    "ok": False,
                    "error": "invalid_arguments",
                    "detail": f"工具参数错误：{exc}",
                },
                ensure_ascii=False,
            )

        except Exception as exc:
            # 其他异常
            logger.error(f"内部工具 {public_name} 执行失败：{exc}", exc_info=True)
            return json.dumps(
                {
                    "ok": False,
                    "error": "execution_failed",
                    "detail": f"工具执行失败：{type(exc).__name__}",
                },
                ensure_ascii=False,
            )

    @property
    def enabled(self) -> bool:
        """内部工具是否可用（总是 True）。"""
        return True

    @property
    def tool_count(self) -> int:
        """已注册的内部工具数量。"""
        return len(self._registry)


# 全局单例
internal_tools_manager = InternalToolsManager()
