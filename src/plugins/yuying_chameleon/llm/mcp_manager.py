"""MCP（Model Context Protocol）管理器：连接多个 MCP server，并提供工具注册/调用。

实现目标（按你的讨论做的"首期简单 + 向后兼容 + 可扩展"原型）：
1) config.toml 中配置 MCP server 列表（见 config.py 的 MCPServerConfig）
2) 支持多个 MCP server 同时存在（按 server_id 管理）
3) 将 MCP tools 转换为 OpenAI function calling 的 tools 格式（交给 schema_converter）
4) 提供工具调用接口：LLM 返回 tool_calls -> 路由到对应 server -> 执行 -> 返回结果
5) 向后兼容：enable_mcp=false 或缺依赖时，不影响现有功能（fail-open）

重要说明：
- MCP 协议本身是 JSON-RPC + transport（stdio/http/sse 等）；这里优先依赖官方/社区 MCP Python SDK
  来处理协议细节，避免在插件内重复实现 LSP 风格 framing。
- 为了保证"未启用 MCP 时零影响"，本模块不会在 import 阶段强制 import mcp SDK；
  只有在真正需要连接/列工具/调用工具时才尝试导入。
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from nonebot import logger

from ..config import MCPServerConfig, plugin_config
from .schema_converter import convert_mcp_input_schema_to_openai_parameters


class MCPDependencyError(RuntimeError):
    """缺少 MCP Python SDK 依赖（或依赖版本不兼容）。"""


class MCPConfigError(ValueError):
    """MCP 配置错误（例如 server_id 重复、必填字段缺失）。"""


class MCPServerError(RuntimeError):
    """MCP server 运行/通信错误（连接失败、调用失败等）。"""


@dataclass(frozen=True)
class RegisteredTool:
    """工具注册表条目：将 OpenAI tool name 映射到 MCP server + MCP tool name。"""

    public_name: str  # 提供给 LLM 的"工具名"（尽量短；冲突时才加后缀）
    server_id: str
    mcp_name: str
    description: str
    input_schema: Dict[str, Any]


def _json_dumps_safe(obj: Any) -> str:
    """尽力 JSON 序列化（用于回传 tool 结果给 LLM）。"""
    try:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        try:
            return json.dumps({"ok": False, "error": "json_dumps_failed"}, separators=(",", ":"))
        except Exception:
            return '{"ok":false,"error":"json_dumps_failed"}'


def _safe_exc_str(exc: Exception, max_len: int = 200) -> str:
    """安全地将异常转为字符串，限制长度。"""
    s = f"{type(exc).__name__}: {exc}".replace("\n", " ").strip()
    return s[:max_len]


def _truncate_tool_content(content: str, max_chars: int) -> str:
    """截断工具返回内容，避免过长消耗 tokens。"""
    if max_chars <= 0 or len(content) <= max_chars:
        return content

    base = {"truncated": True, "original_len": len(content), "content": ""}
    preview_len = max(0, max_chars - len(json.dumps(base, ensure_ascii=False, separators=(",", ":"))) - 50)
    preview = content[:preview_len]
    base["content"] = preview

    # JSON 转义可能膨胀，做个收缩循环保证不超长
    s = json.dumps(base, ensure_ascii=False, separators=(",", ":"))
    while len(s) > max_chars and preview_len > 0:
        preview_len = max(0, int(preview_len * 0.85))
        base["content"] = content[:preview_len]
        s = json.dumps(base, ensure_ascii=False, separators=(",", ":"))
    return s


def _server_display_name(cfg: MCPServerConfig) -> str:
    name = (cfg.display_name or "").strip()
    return name if name else (cfg.id or "MCP")


def _tool_allowed(cfg: MCPServerConfig, tool_name: str) -> bool:
    """根据 allow_tools/deny_tools 判断某工具是否可用。"""
    name = (tool_name or "").strip()
    if not name:
        return False

    allow = [str(x).strip() for x in (cfg.allow_tools or []) if str(x).strip()]
    deny = [str(x).strip() for x in (cfg.deny_tools or []) if str(x).strip()]

    # allow 非空：只允许 allow 内
    if allow:
        return name in set(allow)

    # allow 为空：deny 生效
    if deny and name in set(deny):
        return False

    return True


class _MCPServerRuntime:
    """单个 MCP server 的运行时状态（懒连接 + 工具缓存）。"""

    def __init__(self, cfg: MCPServerConfig) -> None:
        self.cfg = cfg
        self._lock = asyncio.Lock()

        # 连接相关：使用 AsyncExitStack 统一管理资源释放（stdio 子进程/流/session 等）
        self._exit_stack: Optional[AsyncExitStack] = None
        self._session: Optional[Any] = None

        # 工具缓存（MCP list_tools 的结果）
        self._tools_cache: Optional[List[Dict[str, Any]]] = None

    @staticmethod
    def _import_mcp_sdk() -> Tuple[Any, Any, Any]:
        """延迟导入 MCP SDK，并返回必要符号。

        这里使用"尽力导入 + 明确异常"的策略：
        - 未安装依赖：抛 MCPDependencyError（上层可 fail-open）
        - API 不匹配：同样抛 MCPDependencyError（提示升级/锁版本）
        """
        try:
            # 常见 SDK 形态（参考官方 Python SDK 的典型用法）
            #   from mcp import ClientSession, StdioServerParameters
            #   from mcp.client.stdio import stdio_client
            import mcp  # type: ignore

            ClientSession = getattr(mcp, "ClientSession", None)
            StdioServerParameters = getattr(mcp, "StdioServerParameters", None)
            if ClientSession is None or StdioServerParameters is None:
                raise AttributeError("mcp.ClientSession/StdioServerParameters not found")

            stdio_mod = None
            try:
                # 某些版本：mcp.client.stdio.stdio_client
                from mcp.client.stdio import stdio_client  # type: ignore

                stdio_mod = stdio_client
            except Exception as exc:
                raise AttributeError(f"mcp.client.stdio.stdio_client not found: {exc}") from exc

            return ClientSession, StdioServerParameters, stdio_mod
        except Exception as exc:
            raise MCPDependencyError(
                "MCP 已启用但未找到可用的 MCP Python SDK（建议安装 optional-deps: `pip install .[mcp]`）"
            ) from exc

    async def _connect(self) -> None:
        """确保已连接 MCP server（stdio）。"""
        async with self._lock:
            if self._session is not None:
                return

            if (self.cfg.transport or "").strip().lower() != "stdio":
                raise MCPServerError(f"暂不支持 transport={self.cfg.transport!r}（首期原型仅实现 stdio）")

            cmd = (self.cfg.command or "").strip()
            if not cmd:
                raise MCPConfigError(f"MCP server({self.cfg.id}) transport=stdio 但未配置 command")

            args = [str(x) for x in (self.cfg.args or [])]
            cwd = (self.cfg.cwd or "").strip() or None
            env_overrides = {str(k): str(v) for k, v in (self.cfg.env or {}).items()}
            env: Optional[Dict[str, str]] = None
            if env_overrides:
                # 避免传 {} 导致子进程缺少 PATH 等关键变量（取决于 SDK 是否 merge）
                merged = dict(os.environ)
                merged.update(env_overrides)
                env = merged

            ClientSession, StdioServerParameters, stdio_client = self._import_mcp_sdk()

            # 使用 AsyncExitStack：任意一步失败都能自动回收已创建的资源
            stack = AsyncExitStack()
            try:
                params = StdioServerParameters(command=cmd, args=args, env=env, cwd=cwd)

                # stdio_client(params) 通常会启动子进程并提供 read/write 两个异步流
                # 具体实现细节由 MCP SDK 负责（包括 framing / JSON-RPC）。
                read, write = await stack.enter_async_context(stdio_client(params))

                session = await stack.enter_async_context(ClientSession(read, write))

                # initialize：按 MCP 规范进行握手
                try:
                    await session.initialize()
                except Exception as exc:
                    raise MCPServerError(f"MCP server({self.cfg.id}) initialize 失败: {exc}") from exc

                self._exit_stack = stack
                self._session = session
                self._tools_cache = None  # 连接建立后，工具列表缓存无效，需重拉
                logger.info(f"MCP server 已连接: id={self.cfg.id} cmd={cmd} args={args}")
            except Exception:
                await stack.aclose()
                raise

    async def close(self) -> None:
        async with self._lock:
            if self._exit_stack is not None:
                try:
                    await self._exit_stack.aclose()
                except Exception as exc:
                    logger.warning(f"关闭 MCP server({self.cfg.id}) 失败（忽略继续）：{exc}")
            self._exit_stack = None
            self._session = None
            self._tools_cache = None

    @staticmethod
    def _tool_obj_to_dict(tool: Any) -> Dict[str, Any]:
        """把 MCP SDK 的 tool 对象/字典，规范化成 dict。"""
        if isinstance(tool, dict):
            return dict(tool)
        out: Dict[str, Any] = {}
        for k in ("name", "description", "inputSchema"):
            try:
                v = getattr(tool, k, None)
            except Exception:
                v = None
            if v is not None:
                out[k] = v
        return out

    @staticmethod
    def _tools_from_list_tools_result(res: Any) -> List[Dict[str, Any]]:
        """从 list_tools 响应中抽取 tools 列表（兼容多种 SDK 返回形态）。"""
        if res is None:
            return []
        if isinstance(res, dict):
            tools = res.get("tools")
            if isinstance(tools, list):
                return [dict(x) if isinstance(x, dict) else {} for x in tools]
            return []
        tools_attr = None
        try:
            tools_attr = getattr(res, "tools", None)
        except Exception:
            tools_attr = None
        if isinstance(tools_attr, list):
            out: List[Dict[str, Any]] = []
            for t in tools_attr:
                out.append(_MCPServerRuntime._tool_obj_to_dict(t))
            return out
        return []

    async def list_tools(self) -> List[Dict[str, Any]]:
        """列出 MCP tools，并进行缓存。"""
        await self._connect()
        async with self._lock:
            if self._tools_cache is not None:
                return list(self._tools_cache)

            if self._session is None:
                return []

            try:
                res = await self._session.list_tools()
            except Exception as exc:
                raise MCPServerError(f"MCP server({self.cfg.id}) list_tools 失败: {exc}") from exc

            tools = self._tools_from_list_tools_result(res)
            self._tools_cache = tools
            return list(tools)

    @staticmethod
    def _call_result_to_text(res: Any) -> str:
        """把 MCP call_tool 的返回结果转成可喂给 LLM 的文本（JSON 字符串）。"""
        if res is None:
            return _json_dumps_safe({"ok": False, "error": "empty_result"})

        # 兼容 dict
        if isinstance(res, dict):
            return _json_dumps_safe({"ok": True, "result": res})

        # MCP 常见返回：CallToolResult(content=[TextContent(...), ...])
        content = None
        try:
            content = getattr(res, "content", None)
        except Exception:
            content = None

        parts: List[Any] = []
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    parts.append(item)
                    continue
                # TextContent: .type == "text", .text
                item_type = None
                item_text = None
                try:
                    item_type = getattr(item, "type", None)
                except Exception:
                    item_type = None
                try:
                    item_text = getattr(item, "text", None)
                except Exception:
                    item_text = None
                if item_type == "text" and isinstance(item_text, str):
                    parts.append({"type": "text", "text": item_text})
                else:
                    # 其他类型（图片/资源等）：尽力结构化
                    parts.append({"type": str(item_type or "unknown")})

        # 也可能直接有 text 字段
        if not parts:
            text = None
            try:
                text = getattr(res, "text", None)
            except Exception:
                text = None
            if isinstance(text, str) and text.strip():
                parts = [{"type": "text", "text": text.strip()}]

        return _json_dumps_safe({"ok": True, "content": parts})

    async def call_tool(self, *, name: str, arguments: Dict[str, Any], timeout: float) -> str:
        """调用 MCP tool，并返回字符串内容（作为 OpenAI tool message 的 content）。"""
        await self._connect()
        if self._session is None:
            raise MCPServerError(f"MCP server({self.cfg.id}) session is None")

        async def _do_call() -> Any:
            return await self._session.call_tool(name=name, arguments=arguments)

        try:
            res = await asyncio.wait_for(_do_call(), timeout=timeout)
            return self._call_result_to_text(res)
        except asyncio.TimeoutError as exc:
            raise MCPServerError(f"MCP tool 超时: server={self.cfg.id} tool={name}") from exc
        except Exception as exc:
            raise MCPServerError(f"MCP tool 调用失败: server={self.cfg.id} tool={name}: {exc}") from exc


class MCPManager:
    """MCP 管理器：多 server + 工具注册表 + OpenAI tools 列表构建。"""

    def __init__(self, servers: List[MCPServerConfig]) -> None:
        self._servers_cfg = servers
        self._servers: Dict[str, _MCPServerRuntime] = {}

        # OpenAI tools 缓存 + 注册表缓存（避免每次 plan_actions 都 list_tools）
        self._registry_lock = asyncio.Lock()
        self._tools_cache: Optional[List[Dict[str, Any]]] = None
        self._registry: Dict[str, RegisteredTool] = {}

        # 初始化 server runtime（不连接，保持懒加载）
        for cfg in servers:
            sid = (cfg.id or "").strip()
            if not sid:
                continue
            self._servers[sid] = _MCPServerRuntime(cfg)

    @property
    def enabled(self) -> bool:
        """是否启用 MCP（全局开关 + 至少配置一个 server）。"""
        if not bool(getattr(plugin_config, "yuying_enable_mcp", False)):
            return False
        return bool(self._servers)

    def _get_enabled_servers(self) -> List[_MCPServerRuntime]:
        out: List[_MCPServerRuntime] = []
        for sid, rt in self._servers.items():
            if not rt.cfg.enabled:
                continue
            if not (sid or "").strip():
                continue
            out.append(rt)
        return out

    async def validate_config(self) -> None:
        """快速校验配置（不连接、不拉工具）。"""
        seen: set[str] = set()
        for rt in self._get_enabled_servers():
            sid = (rt.cfg.id or "").strip()
            if not sid:
                raise MCPConfigError("mcp_servers 存在空 id")
            if sid in seen:
                raise MCPConfigError(f"mcp_servers 存在重复 id: {sid}")
            seen.add(sid)

            transport = (rt.cfg.transport or "").strip().lower()
            if transport != "stdio":
                # 首期只实现 stdio：提前提示，但不强制拦截（可 fail-open）
                logger.warning(f"MCP server({sid}) transport={transport!r} 暂不支持（将忽略）")
                continue
            if not (rt.cfg.command or "").strip():
                raise MCPConfigError(f"MCP server({sid}) transport=stdio 但缺少 command")

    async def on_startup(self) -> None:
        """NoneBot startup 钩子：按配置选择"懒加载"或"预热"。"""
        if not self.enabled:
            return

        try:
            await self.validate_config()
        except Exception as exc:
            if getattr(plugin_config, "yuying_mcp_fail_open", True):
                logger.warning(f"MCP 配置校验失败（fail-open，继续不用 MCP）：{exc}")
                return
            raise

        if bool(getattr(plugin_config, "yuying_mcp_lazy_connect", True)):
            # 懒加载：启动时不连接、不拉工具
            logger.info("MCP 懒加载已启用：启动时不连接 server")
            return

        # 预热：尽力连接并拉取工具（失败按 fail-open 处理）
        try:
            await self.get_openai_tools()
            logger.info("MCP 预热完成：已拉取 tools 列表")
        except Exception as exc:
            if getattr(plugin_config, "yuying_mcp_fail_open", True):
                logger.warning(f"MCP 预热失败（fail-open，继续不用 MCP）：{exc}")
                return
            raise

    async def on_shutdown(self) -> None:
        """NoneBot shutdown 钩子：关闭所有连接/子进程。"""
        # 无论是否 enabled，都尽力关闭（防止运行时动态切换导致泄漏）
        for rt in list(self._servers.values()):
            try:
                await rt.close()
            except Exception:
                continue

    async def get_openai_tools(self) -> List[Dict[str, Any]]:
        """构建并缓存 OpenAI tools 列表（用于传给 chat.completions.create(tools=...)）。

        关键点（按你的讨论做的策略）：
        - 工具名尽量短：默认使用 MCP tool 原名
        - 通过 description 前缀标注来源：[ServerName]
        - 只有真正重名冲突时，才对冲突工具加后缀 `__{server_id}`（避免路由混乱）
        - schema：尽力转换 + 降级（详见 schema_converter）
        """
        if not self.enabled:
            return []

        async with self._registry_lock:
            if self._tools_cache is not None:
                return list(self._tools_cache)

            servers = self._get_enabled_servers()
            if not servers:
                self._tools_cache = []
                self._registry = {}
                return []

            # 1) 拉取每个 server 的 tools（这里会触发连接：懒加载模式下首次使用会有额外延迟）
            per_server_tools: List[Tuple[_MCPServerRuntime, List[Dict[str, Any]]]] = []
            for rt in servers:
                # 不支持的 transport：跳过（首期原型）
                if (rt.cfg.transport or "").strip().lower() != "stdio":
                    continue
                try:
                    tools = await rt.list_tools()
                    per_server_tools.append((rt, tools))
                except Exception as exc:
                    if getattr(plugin_config, "yuying_mcp_fail_open", True):
                        logger.warning(f"MCP server({rt.cfg.id}) 列工具失败（将忽略该 server）：{exc}")
                        continue
                    raise

            # 2) 扁平化 + 过滤 allow/deny
            flattened: List[RegisteredTool] = []
            for rt, tools in per_server_tools:
                sid = (rt.cfg.id or "").strip()
                for t in tools:
                    name = str((t.get("name") or "")).strip()
                    if not name:
                        continue
                    if not _tool_allowed(rt.cfg, name):
                        continue
                    desc = str((t.get("description") or "")).strip()
                    schema = t.get("inputSchema")
                    if not isinstance(schema, dict):
                        schema = {}
                    flattened.append(
                        RegisteredTool(
                            public_name=name,  # 临时占位，后面冲突处理后再定最终 public_name
                            server_id=sid,
                            mcp_name=name,
                            description=desc,
                            input_schema=schema,
                        )
                    )

            if not flattened:
                self._tools_cache = []
                self._registry = {}
                return []

            # 3) 冲突处理：统计同名工具出现次数
            name_to_count: Dict[str, int] = {}
            for item in flattened:
                name_to_count[item.public_name] = name_to_count.get(item.public_name, 0) + 1

            # 4) 生成最终 public_name + 注册表 + OpenAI tools
            registry: Dict[str, RegisteredTool] = {}
            openai_tools: List[Dict[str, Any]] = []
            for item in flattened:
                base_name = item.public_name
                sid = item.server_id

                # 只有冲突时才加后缀，避免工具名过长影响可读性
                if name_to_count.get(base_name, 0) > 1:
                    public = f"{base_name}__{sid}"
                else:
                    public = base_name

                # 若仍冲突（极端情况：sid 相同/或重复），再加一个序号
                if public in registry:
                    i = 2
                    while f"{public}_{i}" in registry:
                        i += 1
                    public = f"{public}_{i}"

                server_tag = _server_display_name(self._servers[sid].cfg) if sid in self._servers else sid
                # description 前缀标注来源，提高 LLM 可读性，同时不把 server_id 强塞进 name
                desc_prefix = f"[{server_tag}]"
                full_desc = f"{desc_prefix} {item.description}".strip()

                # schema 转换：尽力转换 + 安全降级；并把约束/降级说明追加到 description
                parameters, note = convert_mcp_input_schema_to_openai_parameters(item.input_schema)
                if note:
                    full_desc = f"{full_desc}\n\nSchema note: {note}"

                registry[public] = RegisteredTool(
                    public_name=public,
                    server_id=item.server_id,
                    mcp_name=item.mcp_name,
                    description=full_desc,
                    input_schema=item.input_schema,
                )

                openai_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": public,
                            "description": full_desc,
                            "parameters": parameters,
                        },
                    }
                )

            self._registry = registry
            self._tools_cache = openai_tools
            logger.info(f"MCP tools 已加载: {len(openai_tools)} 个")
            return list(openai_tools)

    async def call_openai_tool(
        self,
        *,
        public_name: str,
        arguments: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> str:
        """按 OpenAI tool name 调用对应 MCP tool，并返回 tool message content（字符串）。"""
        if not self.enabled:
            return _json_dumps_safe({"ok": False, "error": "mcp_disabled"})

        # 确保 registry 已构建（不要在持锁状态下调用 get_openai_tools：避免死锁）
        if self._tools_cache is None:
            try:
                await self.get_openai_tools()
            except Exception as exc:
                if getattr(plugin_config, "yuying_mcp_fail_open", True):
                    return _json_dumps_safe({"ok": False, "error": f"mcp_tools_unavailable:{exc}"})
                raise

        async with self._registry_lock:
            tool = self._registry.get((public_name or "").strip())

        if tool is None:
            return _json_dumps_safe({"ok": False, "error": f"unknown_tool:{public_name}"})

        rt = self._servers.get(tool.server_id)
        if rt is None:
            return _json_dumps_safe({"ok": False, "error": f"unknown_server:{tool.server_id}"})

        # 参数确保是 dict（LLM 可能生成非 object；做最宽松降级）
        args = arguments if isinstance(arguments, dict) else {}

        t = float(timeout) if timeout is not None else float(getattr(plugin_config, "yuying_mcp_tool_timeout", 15.0))
        try:
            content = await rt.call_tool(name=tool.mcp_name, arguments=args, timeout=t)
            
            # 截断工具结果（避免过长消耗 tokens）
            max_chars = int(getattr(plugin_config, "yuying_mcp_tool_result_max_chars", 2000) or 2000)
            content = _truncate_tool_content(content, max_chars)
            
            return content
        except Exception as exc:
            if getattr(plugin_config, "yuying_mcp_fail_open", True):
                return _json_dumps_safe(
                    {
                        "ok": False,
                        "error": "tool_call_failed",
                        "detail": _safe_exc_str(exc),
                        "tool": public_name,
                        "server": tool.server_id,
                    }
                )
            raise


# 模块级单例：供 planner / plugin 生命周期使用
mcp_manager = MCPManager(getattr(plugin_config, "yuying_mcp_servers", []) or [])
