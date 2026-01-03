"""动作规划：将“用户输入 + 记忆 + RAG”转换为可执行 actions。

说明：
- 本模块负责 prompt 组装、LLM 调用与 actions 校验。
- LLM 不可用时必须降级为可用的文本回复（不阻塞主流程）。
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from nonebot import logger

from ..config import plugin_config
from ..llm.client import ChatCompletionResult, main_llm
from ..llm.mcp_manager import mcp_manager
from ..storage.models import Memory
from ..tools.internal_tools_manager import internal_tools_manager


class ActionPlanner:
    """动作规划器。"""

    _cached_system_prompt: Optional[str] = None

    @staticmethod
    def _safe_exc_str(exc: Exception, max_len: int = 200) -> str:
        """安全地将异常转为字符串，限制长度。"""
        s = f"{type(exc).__name__}: {exc}".replace("\n", " ").strip()
        return s[:max_len]

    @staticmethod
    async def _run_tool_call_loop(
        *,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        first_result: ChatCompletionResult,
        max_tool_calls: int,
        parallel: bool,
        max_parallel: int,
        context_qq_id: str,
        context_scene_type: str,
        context_scene_id: str,
        context_raw_msg_id: int,
    ) -> Optional[str]:
        """执行工具调用循环（MCP + 内部工具），返回最终的文本回复（或 None 表示失败）。

        Args:
            messages: 对话历史（会被修改，追加 assistant/tool messages）
            tools: OpenAI tools 列表（包含 MCP + 内部工具）
            first_result: 第一轮 LLM 返回的结果（已包含 tool_calls）
            max_tool_calls: 单次规划允许的最大 tool call 次数（累计）
            parallel: 是否并发执行同一轮的多个 tool_calls
            max_parallel: 并发执行的最大并行度
            context_qq_id: 当前用户 QQ 号（用于内部工具）
            context_scene_type: 当前场景类型（用于内部工具）
            context_scene_id: 当前场景 ID（用于内部工具）
            context_raw_msg_id: 当前消息 ID（用于内部工具）

        Returns:
            Optional[str]: 最终的文本回复，或 None（表示工具循环失败）
        """
        result = first_result
        tool_call_count = 0
        last_fingerprints: List[str] = []  # 用于检测连续重复调用

        max_tool_calls = max(1, int(max_tool_calls))
        max_parallel = max(1, int(max_parallel))

        def _fingerprint(tc: Any) -> str:
            name = str(getattr(tc, "name", "") or "")
            raw_args = str(getattr(tc, "arguments_json", "") or "")
            if raw_args:
                try:
                    obj = json.loads(raw_args)
                    raw_args = json.dumps(
                        obj,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                except Exception:
                    pass
            return f"{name}:{raw_args}"

        for round_idx in range(max_tool_calls):
            if not result or not result.tool_calls:
                # 本轮无 tool_calls，返回文本 content
                return result.content if result else None

            # 统计本轮 tool_calls 数量
            tool_call_count += len(result.tool_calls)
            if tool_call_count > max_tool_calls:
                logger.warning(
                    f"MCP 工具调用次数超限（limit={max_tool_calls},累计={tool_call_count}），提前终止"
                )
                # 返回最后的 content（可能为 None）
                return result.content

            # 检测重复调用：为本轮所有 tool_calls 生成指纹
            current_fingerprints = [_fingerprint(tc) for tc in result.tool_calls]
            if current_fingerprints == last_fingerprints:
                logger.warning(
                    f"MCP 工具调用循环检测到连续重复（round {round_idx}），提前终止"
                )
                return result.content

            last_fingerprints = current_fingerprints

            # 构建 assistant message（包含 tool_calls）
            # OpenAI API 要求：assistant message 必须包含原始的 tool_calls 结构
            assistant_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": result.content or "",
            }

            # 重建 tool_calls 结构（OpenAI SDK 兼容格式）
            tool_calls_raw = []
            for tc in result.tool_calls:
                tool_calls_raw.append(
                    {
                        "id": tc.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments_json,
                        },
                    }
                )
            assistant_msg["tool_calls"] = tool_calls_raw
            messages.append(assistant_msg)

            # 执行工具调用（串行或并行）
            tool_calls = list(result.tool_calls)
            if parallel and len(tool_calls) > 1 and max_parallel > 1:
                sem = asyncio.Semaphore(max_parallel)

                async def _run_one(tc: Any) -> str:
                    async with sem:
                        return await ActionPlanner._execute_single_tool_call(
                            tc,
                            context_qq_id=context_qq_id,
                            context_scene_type=context_scene_type,
                            context_scene_id=context_scene_id,
                            context_raw_msg_id=context_raw_msg_id,
                        )

                tool_results = await asyncio.gather(*(_run_one(tc) for tc in tool_calls))
            else:
                tool_results = []
                for tc in tool_calls:
                    tool_results.append(
                        await ActionPlanner._execute_single_tool_call(
                            tc,
                            context_qq_id=context_qq_id,
                            context_scene_type=context_scene_type,
                            context_scene_id=context_scene_id,
                            context_raw_msg_id=context_raw_msg_id,
                        )
                    )

            # 将工具结果追加为 tool messages
            for tc, res in zip(tool_calls, tool_results):
                content = res if isinstance(res, str) else json.dumps(
                    {"ok": False, "error": "tool_execution_failed", "detail": "non_string_result"},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.tool_call_id,
                        "content": content,
                    }
                )

            # 继续下一轮 LLM 调用
            try:
                result = await main_llm.chat_completion(
                    messages,
                    temperature=0.7,
                    tools=tools,
                    return_result=True,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(f"MCP 工具循环中 LLM 调用失败：{exc}")
                # fail-open：尽力回退到无 tools 的常规调用，避免工具链路影响主流程
                if getattr(plugin_config, "yuying_mcp_fail_open", True):
                    try:
                        fallback = await main_llm.chat_completion(messages, temperature=0.7)
                        return fallback if isinstance(fallback, str) else None
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        return None
                return None

        # 超过最大轮数，返回最后的 content
        logger.warning(f"MCP 工具调用循环达到最大次数 {max_tool_calls}，返回最后结果")
        return result.content if result else None

    @staticmethod
    async def _execute_single_tool_call(
        tc: Any,
        *,
        context_qq_id: str,
        context_scene_type: str,
        context_scene_id: str,
        context_raw_msg_id: int,
    ) -> str:
        """执行单个工具调用（封装异常处理），根据前缀路由到 MCP 或内部工具。

        Args:
            tc: ToolCall 对象
            context_qq_id: 当前用户 QQ 号
            context_scene_type: 当前场景类型
            context_scene_id: 当前场景 ID
            context_raw_msg_id: 当前消息 ID

        Returns:
            str: 工具返回的 content 字符串
        """
        try:
            # 解析 arguments_json
            args = {}
            try:
                args = json.loads(tc.arguments_json or "{}")
            except Exception:
                args = {}

            # 根据工具名前缀路由
            if tc.name.startswith("internal__"):
                # 内部工具
                content = await internal_tools_manager.call_tool(
                    public_name=tc.name,
                    arguments=args,
                    context_qq_id=context_qq_id,
                    context_scene_type=context_scene_type,
                    context_scene_id=context_scene_id,
                    context_raw_msg_id=context_raw_msg_id,
                )
                return content
            else:
                # MCP 工具
                timeout = float(
                    getattr(plugin_config, "yuying_mcp_tool_timeout", 15.0)
                )
                content = await mcp_manager.call_openai_tool(
                    public_name=tc.name,
                    arguments=args,
                    timeout=timeout,
                )
                return content

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(f"执行工具 {tc.name} 失败：{exc}")
            return json.dumps(
                {
                    "ok": False,
                    "error": "tool_call_exception",
                    "detail": ActionPlanner._safe_exc_str(exc),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )

    @staticmethod
    async def plan_actions(
        user_msg: str,
        memories: List[Memory],
        rag_context: List[str],
        recent_dialogue: Optional[List[str]] = None,
        reply_to_message: Optional[Dict[str, Any]] = None,
        images: Optional[Sequence[Dict[str, str]]] = None,
        meta: Optional[Dict[str, Any]] = None,
        *,
        context_qq_id: str = "",
        context_scene_type: str = "",
        context_scene_id: str = "",
        context_raw_msg_id: int = 0,
    ) -> List[Dict[str, Any]]:
        """规划本次回复动作列表。

        参数：
            user_msg: 用户消息（归一化文本）。
            memories: 注入上下文的记忆列表。
            rag_context: RAG 召回片段（字符串列表）。
            recent_dialogue: 最近对话片段（短、按时间顺序）。
            reply_to_message: 被引用（回复）消息的信息（可选），包含：
                - sender_id: 发送者 QQ 号
                - content: 消息内容
                - failed: 是否获取失败
                - reason: 失败原因（failed=True 时有效）
            images: 当前消息关联的图片输入（多模态），元素包含 `url`/`media_key`/`caption` 可选字段。
            meta: 提示词元信息（建议包含 scene_type/mentioned_bot/replied_to_bot/directed_to_bot）。
                - 用途: 让LLM明确知道消息是否是对bot说的,避免误判
                - 示例: {"scene_type": "group", "mentioned_bot": False,
                         "replied_to_bot": True, "directed_to_bot": True}
            context_qq_id: 当前用户 QQ 号（用于内部工具，如记忆写入）
            context_scene_type: 当前场景类型（用于内部工具）
            context_scene_id: 当前场景 ID（用于内部工具）
            context_raw_msg_id: 当前消息 ID（用于内部工具，关联证据）

        返回：
            List[Dict[str, Any]]: 动作列表（最少 1 条）。
        """

        prompt = ActionPlanner._build_prompt(
            user_msg,
            memories,
            rag_context,
            meta=meta,
            recent_dialogue=recent_dialogue,
            reply_to_message=reply_to_message,
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

        # ==================== 工具调用支持（MCP + 内部工具） ====================
        # 策略：
        # 1. 内部工具总是可用（如 AI 主动写入记忆）
        # 2. MCP 工具可选启用（enable_mcp=true）
        # 3. 合并后一起传给 LLM
        # 4. 执行时根据工具名前缀路由

        tools_list: List[Dict[str, Any]] = []
        content: Optional[str] = None

        # 1) 获取内部工具（总是可用）
        if internal_tools_manager.enabled:
            try:
                internal_tools = internal_tools_manager.get_openai_tools()
                tools_list.extend(internal_tools)
                logger.debug(f"已加载 {len(internal_tools)} 个内部工具")
            except Exception as exc:
                logger.warning(f"加载内部工具失败，将继续（不影响主流程）：{exc}")

        # 2) 获取 MCP 工具（可选）
        enable_mcp = bool(getattr(plugin_config, "yuying_enable_mcp", False))
        if enable_mcp and mcp_manager.enabled:
            # MCP 启用：尽力拉取工具列表
            try:
                mcp_tools = await mcp_manager.get_openai_tools()
                tools_list.extend(mcp_tools)
                logger.debug(f"已加载 {len(mcp_tools)} 个 MCP 工具")
            except Exception as exc:
                if getattr(plugin_config, "yuying_mcp_fail_open", True):
                    logger.warning(f"MCP 拉取工具列表失败（fail-open，继续不用 MCP 工具）：{exc}")
                else:
                    raise

        # 3) 如果没有工具，设为 None（传统路径）
        tools: Optional[List[Dict[str, Any]]] = tools_list or None

        if tools:
            # 工具可用：调用 LLM 并启用 tools
            try:
                result = await main_llm.chat_completion(
                    messages,
                    temperature=0.7,
                    tools=tools,
                    return_result=True,
                )

                if result and result.tool_calls:
                    # LLM 返回了 tool_calls：进入工具调用循环
                    max_tool_calls = int(getattr(plugin_config, "yuying_mcp_max_tool_calls", 6))
                    parallel = bool(getattr(plugin_config, "yuying_mcp_parallel_tools", False))
                    max_parallel = int(getattr(plugin_config, "yuying_mcp_max_parallel_tools", 4))

                    content = await ActionPlanner._run_tool_call_loop(
                        messages=messages,
                        tools=tools,
                        first_result=result,
                        max_tool_calls=max_tool_calls,
                        parallel=parallel,
                        max_parallel=max_parallel,
                        context_qq_id=context_qq_id,
                        context_scene_type=context_scene_type,
                        context_scene_id=context_scene_id,
                        context_raw_msg_id=context_raw_msg_id,
                    )
                else:
                    # 无 tool_calls：直接使用 content
                    content = result.content if result else None

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(f"MCP 模式下 LLM 调用失败：{exc}")
                # fail-open：回退为无 tools 的常规调用
                if getattr(plugin_config, "yuying_mcp_fail_open", True):
                    content = await main_llm.chat_completion(messages, temperature=0.7)
                else:
                    raise
        else:
            # 工具不可用或 MCP 未启用：传统调用（向后兼容）
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
        reply_to_message: Optional[Dict[str, Any]] = None,
    ) -> str:
        """组装用于大模型的提示词（尽量短、结构化）。

        说明:
            - 在提示词顶部注入META(JSON),明确告知LLM消息属性
            - 添加当前时间信息,让LLM能够根据真实时间回复
            - 根据directed_to_bot标志,添加对应的回复策略提示
            - 结构化组织记忆、RAG、对话历史等上下文信息
            - 显式注入被引用消息信息（避免依赖"最近对话"碰巧包含）

        Args:
            user_msg: 用户消息文本
            memories: 记忆列表
            rag_context: RAG检索结果
            meta: 元信息字典,包含scene_type/mentioned_bot/replied_to_bot/directed_to_bot
            recent_dialogue: 最近对话历史
            reply_to_message: 被引用消息的信息（可选），包含 sender_id/content/failed/reason

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

        # ==================== 步骤4: 处理被引用消息（显式注入） ====================

        # 被引用消息：显式注入，避免依赖"最近对话"碰巧包含
        reply_lines: List[str] = []
        if reply_to_message:
            failed = bool(reply_to_message.get("failed"))
            if failed:
                # 获取失败：明确告知 LLM 失败原因
                reason = str(reply_to_message.get("reason") or "unknown")
                # 将失败原因翻译为用户可读的中文提示
                reason_map = {
                    "invalid_message_id": "消息ID格式错误",
                    "timeout": "查询超时",
                    "not_found_or_deleted": "消息不存在或已删除",
                    "permission_denied": "权限不足",
                    "api_error": "API错误",
                    "malformed_response": "返回数据格式异常",
                }
                reason_text = reason_map.get(reason, reason)
                reply_lines = [f"- （获取失败：{reason_text}）"]
            else:
                # 获取成功：注入发送者和内容
                sender_id = str(reply_to_message.get("sender_id") or "").strip()
                content = str(reply_to_message.get("content") or "").replace("\n", " ").strip()

                # 控制引用内容长度，避免 prompt 膨胀（约 200-300 字）
                max_chars = 240
                if len(content) > max_chars:
                    content = content[:max_chars] + "…"

                # 格式化输出
                if sender_id:
                    reply_lines.append(f"- 发送者：{sender_id}")
                reply_lines.append(f"- 内容：{content or '（空）'}")
        else:
            # 无引用消息
            reply_lines = ["- （无）"]

        # ==================== 步骤5: 组装完整提示词 ====================

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
                # 被引用消息区域
                "## 被引用的消息：",
                *reply_lines,
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
