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
from typing import Any, Dict, List, Optional  # 类型提示

import openai  # OpenAI官方Python SDK
from nonebot import logger  # NoneBot日志记录器

from ..config import plugin_config  # 导入插件配置


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
        base_url: str,
        api_key: str,
        model: str,
        timeout: float,
        default_headers: Optional[Dict[str, str]] = None,
    ):
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
                kwargs["default_headers"] = cleaned

        # ==================== 步骤4: 创建客户端实例并保存配置 ====================

        # openai.AsyncOpenAI(**kwargs): 创建异步客户端实例
        # **kwargs: 将字典展开为关键字参数
        # AsyncOpenAI: OpenAI Python SDK提供的异步客户端类
        self.client = openai.AsyncOpenAI(**kwargs)

        # 保存模型名称,供后续调用时使用
        self.model = model

    async def chat_completion(
        self, messages: List[Dict[str, Any]], temperature: float = 0.7
    ) -> Optional[str]:
        """执行一次Chat Completions请求(对话生成)

        这个方法的作用:
        - 调用OpenAI的chat completions API生成回复
        - 使用异步调用,不阻塞事件循环
        - 捕获所有异常,失败时返回None(降级策略)
        - 只返回生成的文本内容,不返回完整响应对象

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

        Returns:
            Optional[str]:
                - 成功: 返回模型生成的文本内容(字符串)
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
        """

        try:
            # await self.client.chat.completions.create(): 调用OpenAI API
            # - self.client: AsyncOpenAI客户端实例
            # - .chat.completions: Chat Completions API接口
            # - .create(): 创建一次completion请求
            # - await: 等待异步调用完成
            response = await self.client.chat.completions.create(
                model=self.model,  # 使用初始化时指定的模型
                messages=messages,  # 传入消息列表
                temperature=temperature  # 传入温度参数
            )

            # 从响应中提取生成的文本内容
            # response.choices: API返回的候选回复列表(通常只有1个)
            # [0]: 取第一个候选
            # .message: 消息对象
            # .content: 消息的文本内容
            return response.choices[0].message.content

        except asyncio.CancelledError:
            # asyncio.CancelledError: 异步任务被取消
            # 原因: NoneBot关闭或任务超时被取消
            # 处理: 重新抛出,不捕获,让上层知道任务被取消
            # raise: 重新抛出当前异常
            raise

        except Exception as e:
            # 捕获所有其他异常(网络错误、API错误、超时等)
            # logger.error(): 记录错误日志,f-string格式化
            # 常见错误:
            # - 网络连接失败
            # - API认证失败(401)
            # - 模型不存在(404)
            # - 请求超时
            # - 配额用尽(429)
            logger.error(f"LLM 调用失败:{e}")

            # 返回None表示调用失败
            # 上层代码应检查返回值是否为None,并实施降级策略
            # 例如: 使用默认回复、跳过本次回复等
            return None


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
