"""LLM 客户端封装模块 - 统一封装OpenAI协议的LLM调用

这个模块的作用:
1. 提供LLMClient类封装OpenAI协议的异步调用
2. 处理API配置的兼容性问题(URL清理、key回退等)
3. 初始化三个全局LLM客户端实例供全项目使用
4. 提供统一的异常处理和降级策略

OpenAI协议兼容性:
- 支持OpenAI官方API
- 支持火山方舟(Ark)等兼容服务
- 兼容各种OpenAI兼容网关
- 自动处理URL格式差异

四个客户端实例:
1. main_llm: 主对话模型(如GPT-4),用于核心对话生成
2. cheap_llm: 便宜模型(如GPT-3.5),用于简单任务(标签、摘要等)
3. nano_llm: nano模型(如gpt-4o-mini),用于心流模式的前置决策(yes/no判断)
4. vision_llm: 视觉任务客户端,用于图片理解和OCR(直接复用cheap_llm,不提供任何独立的vision配置)

关键概念(新手必读):
- AsyncOpenAI: OpenAI Python SDK的异步客户端
- Chat Completions: OpenAI的对话生成API接口
- temperature: 生成随机性参数(0=确定性,1=高随机性)
- 降级处理: API调用失败时返回None,由上层决定如何处理

使用方式:
```python
from .llm.client import main_llm, cheap_llm

# 主模型生成回复
reply = await main_llm.chat_completion([
    {"role": "system", "content": "你是一个友好的助手"},
    {"role": "user", "content": "你好"}
])

# 便宜模型生成标签
tags = await cheap_llm.chat_completion([...])
```
"""

from __future__ import annotations

import asyncio  # Python异步编程标准库
import os  # 操作系统接口,用于读取环境变量
import random  # 用于退避策略的随机抖动
import time  # 用于记录延迟
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Union, overload  # 类型提示

import openai  # OpenAI官方Python SDK
from nonebot import logger  # NoneBot日志记录器

from ..config import LLMConfig, LLMModelConfig, LLMModelGroupConfig, plugin_config  # 导入插件配置


@dataclass(frozen=True)
class ToolCall:
    """OpenAI tool call 的最小结构（用于 planner 工具循环）。

    注意：
    - OpenAI SDK 在不同版本/不同 provider 下，返回对象形态可能略有差异；
      这里做"最小依赖"的抽取，只保留 tool_call_id/name/arguments_json。
    """

    tool_call_id: str
    name: str
    arguments_json: str


@dataclass(frozen=True)
class ChatCompletionResult:
    """一次 chat completion 的结构化结果（支持 tools/function calling）。"""

    content: Optional[str]
    tool_calls: List[ToolCall]
    raw: Any  # 原始 SDK 响应对象，留给需要更深度调试/兼容处理的场景


class LLMClient:
    """OpenAI协议的异步LLM客户端封装类

    这个类的作用:
    - 统一封装OpenAI SDK的chat completions调用
    - 处理配置初始化(base_url清理、api_key回退等)
    - 提供异常安全的调用接口(失败返回None而不抛异常)
    - 支持OpenAI兼容的各种服务(火山方舟、本地模型等)

    为什么需要封装?
    - 避免在业务代码中重复处理异常
    - 统一配置管理,避免散落各处
    - 提供降级策略,失败时不中断程序
    - 便于未来切换到其他LLM服务

    实例化参数:
    - base_url: API服务地址,如 "https://api.openai.com/v1"
    - api_key: API密钥,用于身份认证
    - model: 模型名称,如 "gpt-4-turbo"
    - timeout: 请求超时时间(秒)
    - default_headers: 透传给OpenAI SDK的默认请求头(可选)

    属性:
    - client: AsyncOpenAI客户端实例
    - model: 使用的模型名称
    """

    def __init__(
        self,
        base_url: Optional[str],
        api_key: Optional[str],
        model: str,
        timeout: float,
        default_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """创建一个LLM客户端实例

        初始化流程:
        1. 清理和标准化base_url(去除多余的endpoint后缀)
        2. 处理api_key回退(配置 → 环境变量 → 占位符)
        3. 创建AsyncOpenAI客户端实例
        4. 保存模型名称供后续调用使用

        Args:
            base_url: API服务的基础URL
                - OpenAI官方: "https://api.openai.com/v1"
                - 火山方舟: "https://ark.cn-beijing.volces.com/api/v3"
                - 本地服务: "http://localhost:8000/v1"
            api_key: API密钥字符串
            model: 模型名称,如 "gpt-4-turbo", "doubao-pro-32k"
            timeout: 请求超时时间(秒),默认从配置读取
            default_headers: OpenAI SDK默认请求头(可选)
                - 作用: 透传给 openai.AsyncOpenAI(default_headers=...)
                - 常见用途: 设置 User-Agent

        Side Effects:
            - 创建openai.AsyncOpenAI客户端实例
            - 如果api_key未配置,会输出警告日志

        注意:
            - 即使api_key未配置也不会抛异常,但实际调用时会返回401错误
            - base_url会自动清理各种endpoint后缀,保证格式统一
        """

        # ==================== 步骤1: 清理和标准化base_url ====================

        # (base_url or "").strip(): 如果base_url是None,转为空字符串,然后去除首尾空格
        # or None: 如果清理后是空字符串,转为None
        base_url = (base_url or "").strip() or None

        if base_url:  # 如果base_url不为空,进行兼容性处理
            # 兼容部分OpenAI兼容网关的URL格式问题
            # 问题: 有些用户会把完整的endpoint URL填到base_url配置里
            # 例如: "https://api.openai.com/v1/chat/completions"
            # 但OpenAI SDK会自动拼接endpoint,导致重复
            # 解决: 检测并移除这些常见的endpoint后缀

            # suffix in base_url: 检查URL中是否包含这些endpoint后缀
            for suffix in ("/chat/completions", "/completions", "/responses", "/embeddings"):
                if suffix in base_url:
                    # .split(suffix, 1): 按suffix分割,最多分割1次
                    # [0]: 取分割后的第一部分(suffix之前的部分)
                    # 例如: "https://api.com/v1/chat/completions" → "https://api.com/v1"
                    base_url = base_url.split(suffix, 1)[0]

            # .rstrip("/"): 移除URL末尾的斜杠,保证格式一致
            # or None: 如果处理后是空字符串,转为None
            base_url = base_url.rstrip("/") or None

        # ==================== 步骤2: 处理api_key的回退逻辑 ====================

        # 尝试使用配置中的api_key,清理首尾空格
        api_key = (api_key or "").strip() or None

        if not api_key:  # 如果配置中没有api_key
            # 回退策略1: 尝试从环境变量OPENAI_API_KEY读取
            # os.getenv("OPENAI_API_KEY"): 读取环境变量
            api_key = (os.getenv("OPENAI_API_KEY") or "").strip() or None

        if not api_key:  # 如果环境变量也没有
            # 回退策略2: 使用占位符"DUMMY"
            # 原因: 避免在插件import阶段(启动时)因缺少key直接崩溃
            # 影响: 真正调用API时仍会返回401错误,由上层降级处理
            # 好处: 保证进程能够正常启动,不因缺少配置就无法运行
            logger.warning("未配置 api_key，将使用占位 key 以保证进程可启动。")
            api_key = "DUMMY"  # 占位符,不是真实的API key

        # ==================== 步骤3: 构建AsyncOpenAI客户端的参数 ====================

        # kwargs: 用于传递给AsyncOpenAI的关键字参数字典
        kwargs = {"timeout": timeout}  # 超时参数必须提供

        if base_url:  # 如果有base_url,加入参数
            kwargs["base_url"] = base_url

        kwargs["api_key"] = api_key  # API密钥参数

        # ==================== 可选: 透传 default_headers ====================
        # default_headers: OpenAI SDK支持的"默认请求头",会附加到每次请求上
        # 常见用途: 设置 User-Agent
        if isinstance(default_headers, dict) and default_headers:
            cleaned: Dict[str, str] = {}
            for k, v in default_headers.items():
                ks = str(k).strip()
                vs = str(v).strip()
                if not ks or not vs:
                    continue
                # 保护敏感header: 不允许覆盖 Authorization / Content-Type
                # 原因: 这些header由SDK自动管理,用户覆盖可能导致鉴权失败或请求异常
                if ks.lower() in {"authorization", "content-type"}:
                    continue
                cleaned[ks] = vs
            if cleaned:
                # 只记录 header 名称（不记录值，避免泄露敏感信息）
                logger.debug(
                    f"LLMClient default_headers enabled: model={model}, keys={sorted(cleaned.keys())}"
                )
                kwargs["default_headers"] = cleaned

        # ==================== 步骤4: 创建客户端实例并保存配置 ====================

        # openai.AsyncOpenAI(**kwargs): 创建异步客户端实例
        # **kwargs: 将字典展开为关键字参数
        # AsyncOpenAI: OpenAI Python SDK提供的异步客户端类
        self.client = openai.AsyncOpenAI(**kwargs)

        # 保存模型名称,供后续调用时使用
        self.model = model

    @overload
    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        return_result: Literal[False] = False,
        raise_on_error: bool = False,
    ) -> Optional[str]: ...

    @overload
    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        return_result: Literal[True],
        raise_on_error: bool = False,
    ) -> Optional[ChatCompletionResult]: ...

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        return_result: bool = False,
        raise_on_error: bool = False,
    ) -> Union[Optional[str], Optional[ChatCompletionResult]]:
        """执行一次Chat Completions请求(对话生成)

        这个方法的作用:
        - 调用OpenAI的chat completions API生成回复
        - 使用异步调用,不阻塞事件循环
        - 捕获所有异常,失败时返回None(降级策略)
        - 只返回生成的文本内容,不返回完整响应对象

        MCP/工具调用支持（向后兼容）：
        - 当 return_result=False（默认）时：仅返回文本 content（Optional[str]）
        - 当 return_result=True 时：返回 ChatCompletionResult，包含 content + tool_calls
          （tool_calls 可能为空列表，用于工具调用循环）
        - tools 参数控制是否启用工具调用，但不影响返回类型（仅由 return_result 决定）

        OpenAI Chat Completions API说明:
        - API名称: POST /v1/chat/completions
        - 输入: 消息列表,每条消息包含role和content
        - 输出: 模型生成的回复内容
        - temperature: 控制生成的随机性和创造性

        Args:
            messages: 消息列表,OpenAI Chat格式
                格式: [
                    {"role": "system", "content": "系统提示词"},
                    {"role": "user", "content": "用户消息"},
                    {"role": "assistant", "content": "助手回复"},
                    ...
                ]
                - role: 角色,可选值:
                  * "system": 系统提示(定义助手行为)
                  * "user": 用户消息
                  * "assistant": 助手的历史回复
                - content: 消息内容文本

            temperature: 生成温度参数(控制随机性)
                - 类型: 浮点数,范围0.0-2.0
                - 默认值: 0.7
                - 0.0: 确定性输出,每次生成几乎相同
                - 0.7: 平衡的随机性,适合对话
                - 1.0-2.0: 高创造性,输出更随机多样

            tools: OpenAI tools 列表(可选)
                - 用于 function calling
                - None: 不使用工具(默认)

            tool_choice: 工具选择策略(可选)
                - "auto": 自动决定是否调用工具
                - "none": 不调用工具
                - {"type": "function", "function": {"name": "xxx"}}: 强制调用指定工具

            return_result: 是否返回结构化结果(可选)
                - False(默认): 只返回 content 文本
                - True: 返回 ChatCompletionResult 对象

        Returns:
            Union[Optional[str], Optional[ChatCompletionResult]]:
                - 成功: 返回模型生成的文本内容(字符串) 或 ChatCompletionResult
                - 失败: 返回None(用于上层降级处理)

        Raises:
            asyncio.CancelledError:
                - 任务被取消时重新抛出,不捕获
                - 允许NoneBot优雅地取消异步任务

        异常处理策略:
        - CancelledError: 重新抛出(保证任务可被取消)
        - 其他异常: 捕获并记录日志,返回None
        - 好处: 不会因为单次API调用失败而中断整个程序

        Example:
            >>> messages = [
            ...     {"role": "system", "content": "你是一个友好的助手"},
            ...     {"role": "user", "content": "你好,介绍一下自己"}
            ... ]
            >>> reply = await client.chat_completion(messages)
            >>> print(reply)  # "你好!我是一个AI助手,很高兴为你服务..."

            >>> # tools/function calling：需要结构化结果
            >>> result = await client.chat_completion(
            ...     messages,
            ...     tools=[{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}],
            ...     return_result=True,
            ... )
            >>> print(result.tool_calls)
        """

        try:
            # 构建请求参数
            req: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            }

            # tools/function calling（可选）：
            # - tools 为空时，保持旧行为，不改变输出
            # - tools 非空时，默认 tool_choice="auto" 让模型自行决定是否调用工具
            if isinstance(tools, list) and tools:
                req["tools"] = tools
                req["tool_choice"] = tool_choice if tool_choice is not None else "auto"

            response = await self.client.chat.completions.create(**req)

            # OpenAI ChatCompletion：取第一个 choice 的 message
            msg = response.choices[0].message

            # content: 可能为 None（当模型只返回 tool_calls 时）
            content = getattr(msg, "content", None)

            if not return_result:
                # 向后兼容：只返回文本 content
                return content

            # return_result=True：尽力抽取 tool_calls（兼容对象/字典两种形态）
            extracted: List[ToolCall] = []
            raw_tool_calls = getattr(msg, "tool_calls", None) or []
            for idx, tc in enumerate(list(raw_tool_calls)):
                # tool_call_id
                tc_id = ""
                try:
                    tc_id = str(getattr(tc, "id", "") or "")
                except Exception:
                    tc_id = ""

                # function.name / function.arguments
                fn = None
                try:
                    fn = getattr(tc, "function", None)
                except Exception:
                    fn = None

                name = ""
                args_json = ""
                try:
                    name = str(getattr(fn, "name", "") or "")
                except Exception:
                    name = ""
                try:
                    args_json = str(getattr(fn, "arguments", "") or "")
                except Exception:
                    args_json = ""

                # 少数 provider 可能返回 dict；做一层兜底
                if not name and isinstance(tc, dict):
                    fn2 = tc.get("function")
                    if isinstance(fn2, dict):
                        name = str(fn2.get("name") or "")
                        args_json = str(fn2.get("arguments") or "")
                    tc_id = str(tc.get("id") or tc_id)

                # 严格要求 name 存在，否则无法路由
                if not name:
                    continue

                # tool_call_id 必须存在才能回传 tool message；没有就生成一个稳定占位
                # 确保单次响应内唯一：加上索引避免同名工具冲突
                if not tc_id:
                    tc_id = f"toolcall_{name}_{idx}"

                extracted.append(ToolCall(tool_call_id=tc_id, name=name, arguments_json=args_json))

            return ChatCompletionResult(content=content, tool_calls=extracted, raw=response)

        except asyncio.CancelledError:
            # asyncio.CancelledError: 异步任务被取消
            # 原因: NoneBot关闭或任务超时被取消
            # 处理: 重新抛出,不捕获,让上层知道任务被取消
            # raise: 重新抛出当前异常
            raise

        except Exception as e:
            # 捕获所有其他异常(网络错误、API错误、超时等)
            # logger.error(): 记录错误日志,f-string格式化
            logger.error(f"LLM 调用失败:{e}")

            if raise_on_error:
                raise

            # 返回None表示调用失败
            return None


class LLMClientPool:
    """LLM 模型组客户端池 - 支持多模型 fallback 和自动重试。

    这个类的作用:
    - 管理一组 LLM 模型，按顺序尝试 (fallback)
    - 智能判断失败类型，决定是否切换下一个模型
    - 实现退避策略 (backoff)，避免快速失败雪崩
    - 提供完整的可观测性 (日志记录每次尝试)

    Fallback 触发条件:
    - 网络错误 (ConnectionError, TimeoutError)
    - HTTP 5xx 错误 (服务端错误)
    - LLM 返回空内容且无 tool_calls (treat_empty_as_failure=True时)
    - 超时错误

    不触发 Fallback (直接失败):
    - HTTP 401/403 (认证错误，换模型也解决不了)
    - HTTP 400 (参数错误，换模型也解决不了)
    - asyncio.CancelledError (任务被取消)

    重试策略:
    - per_model_attempts: 每个模型最多尝试几次
    - total_attempts_cap: 所有模型累计最多尝试几次
    - base_backoff_seconds: 重试前等待基础时间 (带 jitter)

    属性:
    - clients: LLMClient 实例列表
    - policy: 重试策略配置
    """

    def __init__(
        self,
        clients: List[LLMClient],
        *,
        per_model_attempts: int = 1,
        total_attempts_cap: int = 5,
        treat_empty_as_failure: bool = True,
        base_backoff_seconds: float = 0.3,
    ) -> None:
        """创建一个 LLM 模型池实例。

        Args:
            clients: LLMClient 实例列表，按顺序尝试
            per_model_attempts: 每个模型最多尝试次数 (默认1，不重试直接切换)
            total_attempts_cap: 累计最多尝试次数 (默认5，防止模型组太长)
            treat_empty_as_failure: 空响应是否视为失败 (默认True)
            base_backoff_seconds: 退避基础时间秒数 (默认0.3)
        """
        if not clients:
            raise ValueError("LLMClientPool requires at least one client")

        self.clients = clients
        self.policy = {
            "per_model_attempts": max(1, per_model_attempts),
            "total_attempts_cap": max(1, total_attempts_cap),
            "treat_empty_as_failure": treat_empty_as_failure,
            "base_backoff_seconds": max(0.0, base_backoff_seconds),
        }

        # 用于日志展示的模型列表
        self.model_names = [c.model for c in clients]

    def _should_fallback(self, exc: Exception) -> bool:
        """判断异常是否应该触发 fallback。

        Args:
            exc: 捕获的异常对象

        Returns:
            bool: True 表示应该切换下一个模型，False 表示直接失败
        """
        # asyncio.CancelledError: 任务被取消，不 fallback
        if isinstance(exc, asyncio.CancelledError):
            return False

        # OpenAI SDK 的异常类型 (openai.APIStatusError)
        try:
            if isinstance(exc, openai.APIStatusError):
                status_code = getattr(exc, "status_code", None)
                if status_code:
                    # 401/403: 认证错误
                    # 在多供应商场景下，每个供应商有独立的 API key，应该尝试下一个
                    # 注意：如果是配置错误（所有 key 都无效），最多多尝试几次，不会造成严重问题
                    if status_code in {401, 403}:
                        return True  # 允许 fallback 到下一个供应商
                    # 400: 参数错误，换模型也没用
                    if status_code == 400:
                        return False
                    # 429: 触发限流/配额不足，尝试切换下一个模型/供应商
                    if status_code == 429:
                        return True
                    # 5xx: 服务端错误，可能换模型能解决
                    if 500 <= status_code < 600:
                        return True
        except Exception:
            pass

        # 超时错误、连接错误: 触发 fallback
        if isinstance(exc, (asyncio.TimeoutError, ConnectionError, TimeoutError)):
            return True

        # 默认: 其他异常也尝试 fallback (保守策略)
        return True

    async def _wait_with_backoff(self, attempt: int) -> None:
        """执行退避等待 (带随机抖动)。

        Args:
            attempt: 当前尝试次数 (从 0 开始)
        """
        if attempt == 0:
            return  # 第一次尝试不等待

        base = self.policy["base_backoff_seconds"]
        # 指数退避: base * (1.5 ^ attempt) + jitter
        wait_time = base * (1.5**attempt) + random.uniform(0, base * 0.5)
        # 限制最大等待时间 5 秒
        wait_time = min(wait_time, 5.0)
        await asyncio.sleep(wait_time)

    @overload
    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        return_result: Literal[False] = False,
    ) -> Optional[str]: ...

    @overload
    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        return_result: Literal[True],
    ) -> Optional[ChatCompletionResult]: ...

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        return_result: bool = False,
    ) -> Union[Optional[str], Optional[ChatCompletionResult]]:
        """执行 Chat Completion，支持自动 fallback 和重试。

        按顺序尝试池中的模型，失败时自动切换下一个。

        Args:
            messages: OpenAI chat messages 格式
            temperature: 生成温度 (默认 0.7)
            tools: OpenAI tools 列表 (可选)
            tool_choice: 工具选择策略 (可选)
            return_result: 是否返回结构化结果 (默认 False)

        Returns:
            Union[Optional[str], Optional[ChatCompletionResult]]:
                成功返回生成内容或结构化结果，失败返回 None
        """
        total_attempts = 0
        treat_empty_as_failure = self.policy["treat_empty_as_failure"]

        for client_idx, client in enumerate(self.clients):
            model_name = client.model
            model_attempts = 0

            while model_attempts < self.policy["per_model_attempts"]:
                if total_attempts >= self.policy["total_attempts_cap"]:
                    logger.warning(
                        f"LLMClientPool 达到总尝试次数上限 ({self.policy['total_attempts_cap']})，停止重试"
                    )
                    return None

                # 退避等待
                await self._wait_with_backoff(total_attempts)

                # 记录尝试信息
                logger.debug(
                    f"LLMClientPool 尝试 [{total_attempts + 1}/{self.policy['total_attempts_cap']}]: "
                    f"模型={model_name} (第 {model_attempts + 1}/{self.policy['per_model_attempts']} 次)"
                )

                start_time = time.time()
                try:
                    result = await client.chat_completion(
                        messages,
                        temperature,
                        tools=tools,
                        tool_choice=tool_choice,
                        return_result=return_result,
                        raise_on_error=True,
                    )

                    # 成功获取结果
                    latency = time.time() - start_time

                    # 检查空响应
                    is_empty = False
                    if return_result:
                        # ChatCompletionResult
                        if result is not None:
                            content = result.content
                            tool_calls = result.tool_calls
                            is_empty = (not content or not content.strip()) and not tool_calls
                        else:
                            is_empty = True
                    else:
                        # str
                        is_empty = result is None or (isinstance(result, str) and not result.strip())

                    if is_empty and treat_empty_as_failure:
                        logger.warning(
                            f"LLMClientPool: 模型 {model_name} 返回空响应 (耗时 {latency:.2f}s)，切换下一个模型"
                        )
                        break  # 跳出当前模型的重试循环，尝试下一个模型

                    # 成功
                    logger.info(
                        f"LLMClientPool 成功: 模型={model_name} 耗时={latency:.2f}s "
                        f"(总尝试 {total_attempts + 1} 次)"
                    )
                    return result

                except asyncio.CancelledError:
                    # 任务被取消，直接重新抛出
                    raise

                except Exception as exc:
                    latency = time.time() - start_time
                    total_attempts += 1
                    model_attempts += 1

                    should_fallback = self._should_fallback(exc)
                    logger.warning(
                        f"LLMClientPool: 模型 {model_name} 失败 (耗时 {latency:.2f}s): {exc!r} "
                        f"[{'fallback' if should_fallback else 'no fallback'}]"
                    )

                    if not should_fallback:
                        # 不应该 fallback 的错误 (如 401/403/400)，直接返回失败
                        return None

                    # 继续重试当前模型或切换下一个
                    if model_attempts >= self.policy["per_model_attempts"]:
                        break  # 当前模型重试次数用尽，切换下一个

                    # 否则继续重试当前模型 (while 循环会继续)

        # 所有模型都失败
        logger.error(f"LLMClientPool 所有模型均失败: {self.model_names}")
        return None

    @property
    def model(self) -> str:
        """返回第一个模型的名称 (用于向后兼容)。"""
        return self.clients[0].model if self.clients else "unknown"


# ==================== 模块级全局客户端实例 ====================
# 说明: 这三个实例在模块导入时立即创建,供全项目使用
# 好处: 避免重复创建客户端,配置集中管理

# 主模型客户端
# 用途: 核心对话生成,如用户私聊、群聊回复
# 特点: 通常是能力最强的模型(如GPT-4),生成质量高但成本也高
main_llm = LLMClient(
    base_url=plugin_config.yuying_openai_base_url,  # 从配置读取API地址
    api_key=plugin_config.yuying_openai_api_key,  # 从配置读取API密钥
    model=plugin_config.yuying_openai_model,  # 从配置读取模型名称
    timeout=plugin_config.yuying_openai_timeout,  # 从配置读取超时时间
    default_headers=plugin_config.yuying_openai_default_headers,  # OpenAI SDK默认请求头(可选)
)

# 便宜模型客户端
# 用途: 简单任务,如标签生成、内容摘要、记忆提取等
# 特点: 能力稍弱但速度快成本低(如GPT-3.5),适合批量处理
# 节省成本: 简单任务用便宜模型可以大幅降低API消耗
cheap_llm = LLMClient(
    base_url=plugin_config.yuying_cheap_llm_base_url,  # 便宜模型的API地址
    api_key=plugin_config.yuying_cheap_llm_api_key,  # 便宜模型的API密钥
    model=plugin_config.yuying_cheap_llm_model,  # 便宜模型的名称
    timeout=plugin_config.yuying_cheap_llm_timeout,  # 便宜模型的超时时间
    default_headers=plugin_config.yuying_openai_default_headers,  # OpenAI SDK默认请求头(可选)
)

# Nano 模型客户端
# 用途: 心流模式的前置决策,快速判断是否需要回复(严格 yes/no 输出)
# 特点: 超低延迟、超低成本(如gpt-4o-mini),专门用于二分类决策
# 应用场景: 在心流模式下,用于智能判断群聊消息是否需要机器人回复
# 要求: 响应速度快(timeout 默认 5s),输出简洁(只要求 yes/no)
nano_llm = LLMClient(
    base_url=plugin_config.yuying_nano_llm_base_url,  # nano模型的API地址
    api_key=plugin_config.yuying_nano_llm_api_key,  # nano模型的API密钥
    model=plugin_config.yuying_nano_llm_model,  # nano模型的名称(默认gpt-4o-mini)
    timeout=plugin_config.yuying_nano_llm_timeout,  # nano模型的超时时间(默认5.0秒)
    default_headers=plugin_config.yuying_openai_default_headers,  # OpenAI SDK默认请求头(可选)
)

# 视觉任务客户端(用于图片理解和OCR)
# 用途: 图片说明生成、图片OCR(文字识别)、图片标签提取
# 说明: 为简化配置,图片任务完全复用 cheap_llm 的所有配置参数(base_url/api_key/model/timeout)
vision_llm = cheap_llm


# ==================== LLM 模型组与任务路由 ====================
# 说明: 支持新配置 [yuying_chameleon.llm]，提供模型组 fallback 和任务级路由
# 向后兼容: 如果未配置新段，使用上方的旧客户端实例


def _build_model_group_clients(
    group_config: LLMModelGroupConfig,
    *,
    default_base_url: Optional[str] = None,
    default_api_key: Optional[str] = None,
    default_timeout: float = 30.0,
) -> List[LLMClient]:
    """从模型组配置构建 LLMClient 列表。

    配置优先级（从高到低）：
    1. 内联模型配置（LLMModelConfig 对象）
    2. 模型组配置（LLMModelGroupConfig）
    3. 传入的默认值（default_*）

    Args:
        group_config: LLMModelGroupConfig 对象
        default_base_url: 默认 base_url (可选)
        default_api_key: 默认 api_key (可选)
        default_timeout: 默认 timeout (默认30.0)

    Returns:
        List[LLMClient]: LLMClient 实例列表
    """
    clients: List[LLMClient] = []

    # 提取模型组级别的配置（优先级 2）
    group_base_url = group_config.base_url or default_base_url
    group_api_key = group_config.api_key or default_api_key
    group_timeout = group_config.timeout or default_timeout

    # 方式1: 单模型 (model="xxx")
    if group_config.model and isinstance(group_config.model, str):
        model_name = group_config.model.strip()
        if model_name:
            clients.append(
                LLMClient(
                    base_url=group_base_url,
                    api_key=group_api_key,
                    model=model_name,
                    timeout=group_timeout,
                    default_headers=plugin_config.yuying_openai_default_headers,
                )
            )
        return clients

    # 方式2: 模型组 (models=["xxx", {...}])
    if group_config.models and isinstance(group_config.models, list):
        for item in group_config.models:
            if isinstance(item, str):
                # 字符串形式：使用模型组配置
                model_name = item.strip()
                if not model_name:
                    continue
                clients.append(
                    LLMClient(
                        base_url=group_base_url,
                        api_key=group_api_key,
                        model=model_name,
                        timeout=group_timeout,
                        default_headers=plugin_config.yuying_openai_default_headers,
                    )
                )
            elif isinstance(item, LLMModelConfig):
                # 对象形式：内联配置优先，然后是模型组配置
                model_name = item.model.strip() if item.model else ""
                if not model_name:
                    continue
                clients.append(
                    LLMClient(
                        base_url=item.base_url or group_base_url,
                        api_key=item.api_key or group_api_key,
                        model=model_name,
                        timeout=item.timeout or group_timeout,
                        default_headers=plugin_config.yuying_openai_default_headers,
                    )
                )
        return clients

    return clients


def _build_llm_registry() -> Dict[str, Union[LLMClient, LLMClientPool]]:
    """构建 LLM 模型组注册表。

    从 plugin_config.yuying_llm 读取配置，构建模型组字典。
    如果未配置新段，返回空字典 (使用旧客户端 fallback)。

    Returns:
        Dict[str, Union[LLMClient, LLMClientPool]]:
            键为模型组名 (main/cheap/nano/vision)，值为 LLMClient 或 LLMClientPool
    """
    registry: Dict[str, Union[LLMClient, LLMClientPool]] = {}

    llm_config = plugin_config.yuying_llm
    if not isinstance(llm_config, LLMConfig):
        # 未配置新段，返回空注册表
        return registry

    # 读取全局策略
    policy = llm_config.policy
    per_model_attempts = policy.per_model_attempts
    total_attempts_cap = policy.total_attempts_cap
    treat_empty_as_failure = policy.treat_empty_as_failure
    base_backoff_seconds = policy.base_backoff_seconds

    # ==================== main 模型组 ====================
    if llm_config.main:
        clients = _build_model_group_clients(
            llm_config.main,
            default_base_url=plugin_config.yuying_openai_base_url,
            default_api_key=plugin_config.yuying_openai_api_key,
            default_timeout=plugin_config.yuying_openai_timeout,
        )
        if clients:
            if len(clients) == 1:
                registry["main"] = clients[0]
            else:
                registry["main"] = LLMClientPool(
                    clients,
                    per_model_attempts=per_model_attempts,
                    total_attempts_cap=total_attempts_cap,
                    treat_empty_as_failure=treat_empty_as_failure,
                    base_backoff_seconds=base_backoff_seconds,
                )

    # ==================== cheap 模型组 ====================
    if llm_config.cheap:
        clients = _build_model_group_clients(
            llm_config.cheap,
            default_base_url=plugin_config.yuying_cheap_llm_base_url,
            default_api_key=plugin_config.yuying_cheap_llm_api_key,
            default_timeout=plugin_config.yuying_cheap_llm_timeout,
        )
        if clients:
            if len(clients) == 1:
                registry["cheap"] = clients[0]
            else:
                registry["cheap"] = LLMClientPool(
                    clients,
                    per_model_attempts=per_model_attempts,
                    total_attempts_cap=total_attempts_cap,
                    treat_empty_as_failure=treat_empty_as_failure,
                    base_backoff_seconds=base_backoff_seconds,
                )

    # ==================== nano 模型组 ====================
    if llm_config.nano:
        clients = _build_model_group_clients(
            llm_config.nano,
            default_base_url=plugin_config.yuying_nano_llm_base_url,
            default_api_key=plugin_config.yuying_nano_llm_api_key,
            default_timeout=plugin_config.yuying_nano_llm_timeout,
        )
        if clients:
            if len(clients) == 1:
                registry["nano"] = clients[0]
            else:
                registry["nano"] = LLMClientPool(
                    clients,
                    per_model_attempts=per_model_attempts,
                    total_attempts_cap=total_attempts_cap,
                    treat_empty_as_failure=treat_empty_as_failure,
                    base_backoff_seconds=base_backoff_seconds,
                )

    # ==================== vision 模型组 ====================
    if llm_config.vision:
        clients = _build_model_group_clients(
            llm_config.vision,
            default_base_url=plugin_config.yuying_cheap_llm_base_url,
            default_api_key=plugin_config.yuying_cheap_llm_api_key,
            default_timeout=plugin_config.yuying_cheap_llm_timeout,
        )
        if clients:
            if len(clients) == 1:
                registry["vision"] = clients[0]
            else:
                registry["vision"] = LLMClientPool(
                    clients,
                    per_model_attempts=per_model_attempts,
                    total_attempts_cap=total_attempts_cap,
                    treat_empty_as_failure=treat_empty_as_failure,
                    base_backoff_seconds=base_backoff_seconds,
                )

    return registry


# 构建模型组注册表（模块加载时执行一次）
_llm_registry: Dict[str, Union[LLMClient, LLMClientPool]] = {}
try:
    _llm_registry = _build_llm_registry()
    if _llm_registry:
        logger.info(f"LLM 模型组注册表已构建: {list(_llm_registry.keys())}")
except Exception as exc:
    logger.warning(f"构建 LLM 模型组注册表失败 (将使用旧配置): {exc}")


def get_task_llm(
    task_name: str,
) -> Union[LLMClient, LLMClientPool]:
    """根据任务名称获取对应的 LLM 客户端。

    任务路由逻辑:
    1. 如果配置了 [yuying_chameleon.llm]，根据 tasks 配置路由
    2. 任务名映射到模型组名 (如 action_planner -> main)
    3. 从注册表中查找模型组，如果找不到则 fallback 到旧客户端
    4. 如果未配置新段，直接使用旧客户端

    Args:
        task_name: 任务名称，可选值:
            - action_planner: 主对话/动作规划
            - memory_extraction: 记忆提取
            - memory_condenser: 记忆浓缩
            - summary_generation: 摘要生成
            - sticker_tagging: 表情包标签
            - flow_decider: 心流模式决策
            - vision_caption: 图片说明
            - vision_ocr: OCR
            - personality_reflection: 人格反思 (recent)
            - personality_core: 人格核心原则更新 (core)

    Returns:
        Union[LLMClient, LLMClientPool]: 对应的 LLM 客户端或模型池
    """
    # 如果有新配置，使用任务路由
    llm_config = plugin_config.yuying_llm
    if isinstance(llm_config, LLMConfig):
        tasks_config = llm_config.tasks

        # 任务名 -> 模型组名/模型名
        task_to_group = {
            "action_planner": tasks_config.action_planner,
            "memory_extraction": tasks_config.memory_extraction,
            "memory_condenser": tasks_config.memory_condenser,
            "summary_generation": tasks_config.summary_generation,
            "sticker_tagging": tasks_config.sticker_tagging,
            "flow_decider": tasks_config.flow_decider,
            "vision_caption": tasks_config.vision_caption,
            "vision_ocr": tasks_config.vision_ocr,
            "personality_reflection": tasks_config.personality_reflection,
            "personality_core": tasks_config.personality_core,
        }

        group_name = task_to_group.get(task_name, "")
        if group_name:
            # 先从注册表中查找
            if group_name in _llm_registry:
                return _llm_registry[group_name]

    # Fallback: 根据任务类型返回旧客户端
    if task_name in {"action_planner", "memory_condenser", "personality_core"}:
        return main_llm
    elif task_name in {
        "memory_extraction",
        "summary_generation",
        "sticker_tagging",
        "personality_reflection",
    }:
        return cheap_llm
    elif task_name == "flow_decider":
        return nano_llm
    elif task_name in {"vision_caption", "vision_ocr"}:
        return vision_llm
    else:
        # 未知任务，默认使用 main_llm
        logger.warning(f"未知任务名: {task_name}，使用 main_llm")
        return main_llm
