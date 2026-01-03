"""配置管理模块 - 加载和管理YuYing-Chameleon的所有配置项

这个模块的作用:
1. 定义所有配置项的数据结构(Config类)
2. 从多个来源加载配置(环境变量、配置文件、NoneBot配置)
3. 提供配置项的默认值、类型检查和验证
4. 处理配置优先级和配置合并逻辑

配置加载机制(小白必读):
- 配置来源(按优先级从低到高):
  1. 代码中的默认值(Config类中的default参数)
  2. configs/config.toml文件中的[yuying_chameleon]段
  3. NoneBot的全局配置(如.env文件)
  4. 环境变量

- 配置键名规则:
  - 在代码中使用: yuying_xxx (如 yuying_openai_api_key)
  - 在config.toml中使用: xxx (如 openai_api_key)
  - 通过alias机制实现自动映射

配置文件查找顺序:
1. 环境变量YUYING_CONFIG_TOML指定的路径
2. 当前工作目录下的configs/config.toml
3. 从当前文件向上查找父目录中的configs/config.toml

使用方式:
```python
from .config import plugin_config

# 直接使用配置项
api_key = plugin_config.yuying_openai_api_key
database_url = plugin_config.yuying_database_url
```

关键概念:
- Pydantic: Python的数据验证库,提供类型检查和数据转换
- Field: Pydantic的字段定义,可以设置默认值、别名、验证规则
- alias: 字段别名,允许使用短名称在配置文件中设置
"""

import os  # 用于读取环境变量
from pathlib import Path  # 用于处理文件路径
from typing import Any, Dict, List, Optional  # 类型提示

from nonebot import get_driver  # 获取NoneBot驱动器,用于读取全局配置
from nonebot import logger  # NoneBot的日志记录器
from pydantic.v1 import BaseModel, Extra, Field, root_validator  # Pydantic v1的数据模型相关类


class MCPServerConfig(BaseModel):
    """MCP Server 配置项（单个 server）。

    设计目标（首期实现原型）：
    - 支持配置多个 MCP server（并行存在，按 id 区分）。
    - 支持 stdio transport（最常见，便于本地/容器部署）。
    - 兼容扩展（未来可加 http/sse 等 transport，不破坏现有配置）。

    TOML 示例：
      enable_mcp = true
      [[yuying_chameleon.mcp_servers]]
      id = "web"
      transport = "stdio"
      command = "python"
      args = ["-m", "my_mcp_server"]

      [[yuying_chameleon.mcp_servers]]
      id = "fs"
      transport = "stdio"
      command = "node"
      args = ["server.js"]
      env = { "TZ" = "UTC" }
    """

    id: str = Field(default="", alias="id")
    enabled: bool = Field(default=True, alias="enabled")

    # transport：首期建议使用 stdio（由 MCP client SDK 负责协议细节）。
    transport: str = Field(default="stdio", alias="transport")

    # stdio 参数：command/args/cwd/env
    command: str = Field(default="", alias="command")
    args: List[str] = Field(default_factory=list, alias="args")
    cwd: Optional[str] = Field(default=None, alias="cwd")
    env: Dict[str, str] = Field(default_factory=dict, alias="env")

    # 过滤工具（token 成本控制的"简单版本"）：allow 优先生效，其次 deny。
    allow_tools: List[str] = Field(default_factory=list, alias="allow_tools")
    deny_tools: List[str] = Field(default_factory=list, alias="deny_tools")

    # 用于提升 LLM 可读性：显示名会写进工具 description 的 [ServerName] 前缀里
    display_name: str = Field(default="", alias="display_name")

    class Config:
        extra = Extra.ignore
        allow_population_by_field_name = True


class Config(BaseModel):
    """插件配置模型 - 定义所有配置项的数据结构

    这个类使用Pydantic BaseModel,提供:
    - 类型检查: 自动验证配置值的类型是否正确
    - 默认值: 当配置项未设置时使用的默认值
    - 别名: 允许使用短名称设置配置
    - 数据转换: 自动将字符串转为数字、布尔值等

    配置项分类:
    1. API提供方配置: 指定使用OpenAI还是火山方舟等
    2. 机器人基础配置: 超级用户、昵称等
    3. 数据库配置: SQLite连接信息
    4. 向量库配置: Qdrant连接和检索参数
    5. LLM配置: 主模型、便宜模型、图片任务开关信息
    6. Embedding配置: 文本向量化模型配置
    7. 记忆配置: 用户记忆提取和管理参数
    8. 表情包配置: 表情包收集和使用规则
    9. 回复策略配置: 回复频率、冷却时间等
    10. 摘要配置: 对话摘要窗口参数
    """

    # ==================== API提供方配置 ====================
    # 说明: 主LLM、便宜LLM、Embedder可能来自不同的服务商
    # 如果不单独配置,它们会回退使用api_provider的值

    yuying_api_provider: str = Field(default="openai", alias="api_provider")
    # 全局API提供方
    # - 作用: 为所有LLM服务指定默认的提供商
    # - 可选值: "openai"(OpenAI官方或兼容网关), "ark"(火山方舟)
    # - 默认值: "openai"
    # - 影响: 未单独配置时,main/cheap/embedder都使用这个值

    yuying_main_provider: str = Field(default="", alias="main_provider")
    # 主LLM的提供方(可选,覆盖api_provider)
    # - 作用: 单独指定主对话模型的提供商
    # - 默认值: "" (空字符串表示使用api_provider)
    # - 示例: "ark" 表示主模型使用火山方舟

    yuying_cheap_llm_provider: str = Field(default="", alias="cheap_llm_provider")
    # 便宜LLM的提供方(可选,覆盖api_provider)
    # - 作用: 单独指定轻量级任务(标签、摘要)使用的模型提供商
    # - 默认值: "" (使用api_provider)
    # - 用途: 降低成本,用便宜的模型处理简单任务

    yuying_embedder_provider: str = Field(default="", alias="embedder_provider")
    # Embedding模型的提供方(可选,覆盖api_provider)
    # - 作用: 单独指定文本向量化服务的提供商
    # - 默认值: "" (使用api_provider)

    # ==================== 机器人基础配置 ====================

    yuying_superusers: List[str] = Field(default_factory=list, alias="superusers")
    # 超级用户列表
    # - 作用: 拥有最高权限的用户QQ号列表
    # - 类型: 字符串列表,每个元素是QQ号
    # - 默认值: [] (空列表,无超级用户)
    # - 示例: ["12345678", "87654321"]
    # - 权限: 可以使用管理命令、绕过限流等

    yuying_nickname: List[str] = Field(default_factory=lambda: ["YuYing"], alias="nickname")
    # 机器人的昵称列表
    # - 作用: 用户可以用这些名字称呼机器人
    # - 类型: 字符串列表
    # - 默认值: ["YuYing"]
    # - 用途: 消息解析时识别是否在@机器人
    # - 示例: ["小鱼", "鱼鱼", "YuYing"]

    # ==================== 数据库配置 ====================

    yuying_database_url: str = Field(
        default="sqlite+aiosqlite:///data/yuying.db",
        alias="database_url",
    )
    # 数据库连接URL
    # - 作用: 指定SQLite数据库文件的位置
    # - 格式: "sqlite+aiosqlite:///<路径>"
    # - 默认值: "sqlite+aiosqlite:///data/yuying.db" (相对于项目根目录)
    # - 说明: 使用aiosqlite驱动实现异步数据库操作
    # - 存储内容: 消息记录、用户画像、记忆、摘要、表情包等

    yuying_sqlite_busy_timeout_ms: int = Field(default=3000, alias="sqlite_busy_timeout_ms")
    # SQLite忙碌超时时间
    # - 作用: 当数据库被锁定时,等待的最长时间
    # - 单位: 毫秒(ms)
    # - 默认值: 3000 (3秒)
    # - 说明: SQLite是文件数据库,同时只能有一个写操作
    #   当多个任务同时写入时,后来的任务需要等待
    #   这个参数控制等待多久后放弃

    # ==================== 向量库配置 ====================

    yuying_qdrant_host: str = Field(default="localhost", alias="qdrant_host")
    # Qdrant向量数据库主机地址
    # - 作用: 指定Qdrant服务的主机名或IP
    # - 默认值: "localhost" (本地部署)
    # - 示例: "192.168.1.100" 或 "qdrant.example.com"

    yuying_qdrant_port: int = Field(default=6333, alias="qdrant_port")
    # Qdrant服务端口号
    # - 作用: 指定Qdrant服务监听的端口
    # - 默认值: 6333 (Qdrant默认端口)
    # - 类型: 整数

    yuying_qdrant_api_key: Optional[str] = Field(default=None, alias="qdrant_api_key")
    # Qdrant API密钥(可选)
    # - 作用: 访问需要认证的Qdrant服务
    # - 默认值: None (无需认证,适合本地部署)
    # - 使用场景: 云端Qdrant服务或配置了认证的实例

    yuying_qdrant_https: Optional[bool] = Field(default=None, alias="qdrant_https")
    # 是否使用HTTPS连接Qdrant
    # - 作用: 指定连接Qdrant时是否使用加密连接
    # - 默认值: None (自动判断,本地用HTTP,远程用HTTPS)
    # - True: 强制使用HTTPS
    # - False: 强制使用HTTP

    yuying_qdrant_recreate_collections: bool = Field(
        default=False,
        alias="qdrant_recreate_collections",
    )
    # 启动时是否重建Qdrant集合
    # - 作用: 控制启动时是否删除并重建向量库的集合
    # - 默认值: False (保留现有数据)
    # - True: 每次启动都清空向量库(用于开发调试)
    # - 警告: 设为True会丢失所有向量数据!

    yuying_retrieval_topk: int = Field(default=5, alias="retrieval_topk")
    # 向量检索返回的结果数量
    # - 作用: 搜索向量库时返回最相似的前K个结果
    # - 单位: 个(条数)
    # - 默认值: 5
    # - 影响: 值越大,检索到的上下文越丰富,但LLM输入也越长
    # - 建议: 3-10之间

    yuying_retrieval_snippet_max_chars: int = Field(
        default=120,
        alias="retrieval_snippet_max_chars",
    )
    # 检索片段的最大字符数
    # - 作用: 每个检索结果的文本摘要长度上限
    # - 单位: 字符数
    # - 默认值: 120
    # - 说明: 超过此长度的文本会被截断并加上省略号
    # - 目的: 控制输入LLM的上下文长度,节省tokens

    yuying_hybrid_query_recent_messages_limit: int = Field(
        default=30,
        alias="hybrid_query_recent_messages_limit",
    )
    # Hybrid Query 组装时读取的“场景最近消息”数量
    # - 作用: 用于提取【最近用户/最近机器人/最近对方】等上下文
    # - 单位: 条(消息数)
    # - 默认值: 30

    yuying_hybrid_query_recent_user_messages_limit: int = Field(
        default=3,
        alias="hybrid_query_recent_user_messages_limit",
    )
    # Hybrid Query 中“用户自己最近消息”的条数上限
    # - 作用: 理解用户连续提问/话题延续
    # - 单位: 条
    # - 默认值: 3

    yuying_recent_dialogue_max_lines: int = Field(
        default=30,
        alias="recent_dialogue_max_lines",
    )
    # 注入到主对话 prompt 的“最近对话”行数上限
    # - 作用: 给 ActionPlanner 提供短期上下文(按时间顺序)
    # - 单位: 行(消息行)
    # - 默认值: 30

    yuying_vector_size: int = Field(default=2048, alias="vector_size")
    # 向量维度大小
    # - 作用: 文本向量化后的维度数
    # - 默认值: 2048
    # - 说明: 必须与embedder模型的输出维度一致
    # - 示例: text-embedding-3-small默认是1536维,如果用则需改为1536
    # - 影响: Qdrant collection创建时使用此参数

    # ==================== 主模型配置(主对话LLM) ====================

    yuying_openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        alias="openai_base_url",
    )
    # 主LLM的API基础URL
    # - 作用: 指定OpenAI兼容API的服务地址
    # - 默认值: "https://api.openai.com/v1" (OpenAI官方)
    # - 兼容: 可以是任何OpenAI兼容的服务(如本地部署的模型)
    # - 示例: "http://localhost:8000/v1", "https://ark.cn-beijing.volces.com/api/v3"

    yuying_openai_api_key: str = Field(default="", alias="openai_api_key")
    # 主LLM的API密钥
    # - 作用: 身份认证,访问LLM服务
    # - 默认值: "" (空字符串,需要用户配置)
    # - 必填: 是(否则无法调用LLM)
    # - 安全: 不要提交到代码仓库,使用配置文件或环境变量

    yuying_openai_model: str = Field(default="gpt-4-turbo", alias="openai_model")
    # 主LLM的模型名称
    # - 作用: 指定使用哪个具体的模型
    # - 默认值: "gpt-4-turbo"
    # - 示例: "gpt-4", "gpt-3.5-turbo", "doubao-pro-32k"
    # - 影响: 不同模型的能力、速度、成本不同

    yuying_openai_timeout: float = Field(default=30.0, alias="openai_timeout")
    # 主LLM请求超时时间
    # - 作用: API请求等待响应的最长时间
    # - 单位: 秒(s)
    # - 默认值: 30.0 (30秒)
    # - 说明: 超过此时间未响应则放弃请求
    # - 影响: 设置过小可能导致复杂回复超时失败

    yuying_openai_default_headers: Dict[str, str] = Field(
        default_factory=dict,
        alias="openai_default_headers",
    )
    # OpenAI Python SDK 默认请求头(default_headers)
    # - 作用: 透传给 openai.AsyncOpenAI(default_headers=...),为所有通过SDK发出的请求附加HTTP headers
    # - 常见用途: 设置 User-Agent、透传网关需要的自定义header等
    # - 默认值: {} (不额外设置,使用SDK默认行为)
    #
    # 配置示例(TOML):
    #   openai_default_headers = { "User-Agent" = "YuYing-Chameleon/1.0" }
    #
    # 注意:
    # - Authorization和Content-Type会被自动过滤,不允许通过此配置覆盖(由SDK自动管理)

    # ==================== 低成本模型配置(轻量级任务) ====================

    yuying_cheap_llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        alias="cheap_llm_base_url",
    )
    # 便宜LLM的API基础URL
    # - 作用: 用于简单任务(标签生成、内容摘要)的模型服务地址
    # - 默认值: 与主模型相同
    # - 目的: 降低成本,简单任务用便宜的模型

    yuying_cheap_llm_api_key: str = Field(default="", alias="cheap_llm_api_key")
    # 便宜LLM的API密钥
    # - 作用: 访问便宜模型服务的凭证
    # - 默认值: "" (会自动复用openai_api_key)

    yuying_cheap_llm_model: str = Field(default="gpt-3.5-turbo", alias="cheap_llm_model")
    # 便宜LLM的模型名称
    # - 默认值: "gpt-3.5-turbo" (比GPT-4便宜很多)
    # - 用途: 异步任务、标签生成、简单摘要

    yuying_cheap_llm_timeout: float = Field(default=10.0, alias="cheap_llm_timeout")
    # 便宜LLM请求超时时间
    # - 单位: 秒
    # - 默认值: 10.0 (比主模型短,因为任务简单)

    # ==================== Nano 模型配置(心流模式前置决策) ====================

    yuying_nano_llm_provider: str = Field(default="", alias="nano_llm_provider")
    # Nano LLM的提供方(可选,覆盖api_provider)
    # - 作用: 单独指定心流模式使用的微小模型提供商
    # - 默认值: "" (使用api_provider)
    # - 用途: 心流模式的前置决策,判断是否回复

    yuying_nano_llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        alias="nano_llm_base_url",
    )
    # Nano LLM的API基础URL
    # - 作用: 心流模式前置决策模型的服务地址
    # - 默认值: OpenAI官方API地址
    # - 用途: 快速判断是否需要回复当前消息

    yuying_nano_llm_api_key: str = Field(default="", alias="nano_llm_api_key")
    # Nano LLM的API密钥
    # - 作用: 访问nano模型服务的凭证
    # - 默认值: "" (会自动复用openai_api_key)

    yuying_nano_llm_model: str = Field(default="gpt-4o-mini", alias="nano_llm_model")
    # Nano LLM的模型名称
    # - 默认值: "gpt-4o-mini" (快速且便宜)
    # - 用途: 心流模式的前置决策,需要快速响应

    yuying_nano_llm_timeout: float = Field(default=5.0, alias="nano_llm_timeout")
    # Nano LLM请求超时时间
    # - 单位: 秒
    # - 默认值: 5.0 (需要快速响应,避免阻塞主流程)

    # ==================== 向量化配置(Embedding) ====================

    yuying_embedder_base_url: str = Field(
        default="https://api.openai.com/v1",
        alias="embedder_base_url",
    )
    # Embedding服务的API基础URL
    # - 作用: 文本向量化服务地址
    # - 默认值: OpenAI官方API地址

    yuying_embedder_api_key: str = Field(default="", alias="embedder_api_key")
    # Embedding服务的API密钥
    # - 默认值: "" (会自动复用openai_api_key)

    yuying_embedder_model: str = Field(default="text-embedding-3-small", alias="embedder_model")
    # Embedding模型名称
    # - 作用: 将文本转为向量的模型
    # - 默认值: "text-embedding-3-small"
    # - 说明: 必须与vector_size配置的维度匹配

    yuying_embedder_endpoint: str = Field(default="/embeddings", alias="embedder_endpoint")
    # Embedding API的端点路径
    # - 作用: API路径后缀
    # - 默认值: "/embeddings" (OpenAI标准)
    # - 完整URL: base_url + endpoint

    yuying_embedder_timeout: float = Field(default=30.0, alias="embedder_timeout")
    # Embedding请求超时时间
    # - 单位: 秒
    # - 默认值: 30.0

    # ==================== 媒体理解配置(图片处理) ====================
    # 说明:
    # - 图片说明/OCR 完全复用 cheap_llm_* 配置（需支持图片输入的多模态模型）
    # - 若图片任务更易超时/更慢，请调大 cheap_llm_timeout
    # - 本段仅保留是否启用 OCR 的开关

    yuying_media_enable_ocr: bool = Field(default=False, alias="media_enable_ocr")
    # 是否启用OCR(文字识别)
    # - 作用: 从图片中提取文字内容
    # - 默认值: False (不启用,节省成本)
    # - True: 启用,会调用多模态LLM(复用cheap_llm)识别图片中的文字

    # ==================== 记忆配置 ====================

    yuying_memory_effective_count_threshold: int = Field(
        default=50,
        alias="memory_effective_count_threshold",
    )
    # 触发记忆提取的有效发言数阈值
    # - 作用: 用户发送多少条"有效消息"后开始提取记忆
    # - 单位: 条(消息数)
    # - 默认值: 50
    # - 说明: "有效消息"指非纯表情、非纯空格的正常对话
    # - 目的: 积累足够数据后才提取,避免信息不足

    yuying_memory_condense_hour: int = Field(default=3, alias="memory_condense_hour")
    # 记忆浓缩任务的执行时间点
    # - 作用: 每天的哪个小时执行记忆浓缩
    # - 单位: 小时(0-23)
    # - 默认值: 3 (凌晨3点)
    # - 目的: 在低峰期压缩和整理用户记忆

    yuying_memory_core_limit: int = Field(default=20, alias="memory_core_limit")
    # 核心记忆的数量上限
    # - 作用: 每个用户最多保留多少条核心记忆
    # - 单位: 条
    # - 默认值: 20
    # - 说明: 超过此数量会按重要性淘汰旧记忆

    yuying_memory_active_ttl_days: int = Field(default=7, alias="memory_active_ttl_days")
    # 活跃记忆的有效期
    # - 作用: 记忆多少天未被访问视为不活跃
    # - 单位: 天
    # - 默认值: 7
    # - 影响: 不活跃的记忆可能被归档或淘汰

    yuying_memory_idle_seconds: int = Field(default=120, alias="memory_idle_seconds")
    # 空闲等待时间(触发记忆提取)
    # - 作用: 用户停止发言多久后触发记忆提取
    # - 单位: 秒
    # - 默认值: 120 (2分钟)
    # - 目的: 避免对话进行中频繁提取,等对话告一段落

    yuying_memory_overlap_messages: int = Field(default=10, alias="memory_overlap_messages")
    # 记忆提取时的消息重叠数
    # - 作用: 每次提取时与上次重叠的消息条数
    # - 单位: 条
    # - 默认值: 10
    # - 目的: 保证上下文连续性,避免遗漏信息

    yuying_memory_extract_max_messages: int = Field(default=100, alias="memory_extract_max_messages")
    # 单次记忆提取的最大消息数
    # - 作用: 每次从多少条消息中提取记忆
    # - 单位: 条
    # - 默认值: 100
    # - 影响: 值越大,上下文越完整,但LLM成本越高

    yuying_memory_archive_days: int = Field(default=90, alias="memory_archive_days")
    # 记忆归档期限
    # - 作用: 多少天未访问的记忆会被归档
    # - 单位: 天
    # - 默认值: 90 (3个月)
    # - 说明: 归档的记忆不会被删除,但不参与检索

    # ==================== AI 主动记忆写入速率限制配置 ====================

    yuying_memory_session_limit_group: int = Field(default=3, alias="memory_session_limit_group")
    # 群聊每会话记忆写入限额
    # - 作用: AI 在一个群聊会话中最多主动写入多少条记忆
    # - 单位: 条
    # - 默认值: 3
    # - 说明: 会话定义为连续对话，空闲超时后重置

    yuying_memory_session_limit_private: int = Field(default=5, alias="memory_session_limit_private")
    # 私聊每会话记忆写入限额
    # - 作用: AI 在一个私聊会话中最多主动写入多少条记忆
    # - 单位: 条
    # - 默认值: 5
    # - 说明: 私聊通常更深入，允许写入更多记忆

    yuying_memory_daily_limit_group: int = Field(default=25, alias="memory_daily_limit_group")
    # 群聊每日记忆写入限额
    # - 作用: AI 每天最多为单个用户在群聊中写入多少条记忆
    # - 单位: 条/天
    # - 默认值: 25
    # - 说明: 防止单个用户占用过多记忆存储

    yuying_memory_daily_limit_private: int = Field(default=40, alias="memory_daily_limit_private")
    # 私聊每日记忆写入限额
    # - 作用: AI 每天最多为单个用户在私聊中写入多少条记忆
    # - 单位: 条/天
    # - 默认值: 40
    # - 说明: 私聊限额更高，鼓励深度对话

    yuying_memory_session_idle_timeout: float = Field(default=600.0, alias="memory_session_idle_timeout")
    # 记忆写入会话空闲超时
    # - 作用: 多少秒无消息后视为会话结束，重置会话计数
    # - 单位: 秒
    # - 默认值: 600.0 (10 分钟)
    # - 说明: 超时后开启新会话，可再次写入记忆

    # ==================== 表情包配置 ====================

    yuying_sticker_promote_threshold: int = Field(default=3, alias="sticker_promote_threshold")
    # 表情包晋升阈值
    # - 作用: 候选表情被使用多少次后正式收录
    # - 单位: 次
    # - 默认值: 3
    # - 说明: 防止一次性使用的图片被收录为表情包

    yuying_sticker_cooldown_seconds: int = Field(default=60, alias="sticker_cooldown_seconds")
    # 表情包使用冷却时间
    # - 作用: 同一个表情包两次发送之间的最小间隔
    # - 单位: 秒
    # - 默认值: 60 (1分钟)
    # - 目的: 防止刷屏,避免重复使用

    yuying_sticker_meme_score_threshold: int = Field(default=3, alias="sticker_meme_score_threshold")
    # 表情包趣味性评分阈值
    # - 作用: 图片的"梗"评分达到多少才考虑收录
    # - 单位: 分(1-10分制)
    # - 默认值: 3
    # - 说明: 由LLM评估图片是否适合做表情包

    # ==================== 回复策略配置 ====================

    yuying_global_cooldown_seconds: int = Field(default=30, alias="global_cooldown_seconds")
    # 全局回复冷却时间
    # - 作用: 机器人在任何地方回复后,多久内不再回复其他地方
    # - 单位: 秒
    # - 默认值: 30
    # - 目的: 防止机器人过于活跃

    yuying_group_cooldown_seconds: int = Field(default=60, alias="group_cooldown_seconds")
    # 单个群聊的回复冷却时间
    # - 作用: 在同一个群回复后,多久内不再回复该群
    # - 单位: 秒
    # - 默认值: 60 (1分钟)
    # - 目的: 避免在某个群刷屏

    yuying_group_reply_probability: float = Field(default=0.15, alias="group_reply_probability")
    # 群聊主动回复概率
    # - 作用: 未被@时,有多大概率主动回复
    # - 单位: 小数(0.0-1.0)
    # - 默认值: 0.15 (15%)
    # - 说明: 0.0表示从不主动回复,1.0表示必定回复

    yuying_private_reply_probability: float = Field(default=1.0, alias="private_reply_probability")
    # 私聊回复概率
    # - 作用: 私聊消息的回复概率
    # - 默认值: 1.0 (100%,必定回复)

    yuying_spam_window_seconds: int = Field(default=30, alias="spam_window_seconds")
    # 刷屏检测时间窗口
    # - 作用: 在多长时间内检测消息数量
    # - 单位: 秒
    # - 默认值: 30
    # - 配合spam_msg_threshold使用

    yuying_spam_msg_threshold: int = Field(default=12, alias="spam_msg_threshold")
    # 刷屏检测消息数阈值
    # - 作用: 时间窗口内超过多少条消息视为刷屏
    # - 单位: 条
    # - 默认值: 12
    # - 效果: 30秒内超过12条消息时,暂停回复

    yuying_action_max_count: int = Field(default=4, alias="action_max_count")
    # 单次回复的最大动作数
    # - 作用: 一次回复最多包含几个动作(文本/图片/表情)
    # - 单位: 个
    # - 默认值: 4
    # - 目的: 避免单次回复过长

    yuying_action_min_delay_ms: int = Field(default=300, alias="action_min_delay_ms")
    # 动作之间的最小延迟
    # - 作用: 多个动作发送时的最短间隔
    # - 单位: 毫秒(ms)
    # - 默认值: 300
    # - 目的: 模拟真人打字速度,避免被识别为机器人

    yuying_action_max_delay_ms: int = Field(default=900, alias="action_max_delay_ms")
    # 动作之间的最大延迟
    # - 作用: 多个动作发送时的最长间隔
    # - 单位: 毫秒
    # - 默认值: 900
    # - 说明: 实际延迟在min和max之间随机

    yuying_reply_text_max_chars: int = Field(default=40, alias="reply_text_max_chars")
    # 单条回复文本的最大字符数
    # - 作用: 单个文本动作的长度上限
    # - 单位: 字符
    # - 默认值: 40
    # - 目的: 避免单条消息过长,保持简洁

    # ==================== 自适应防抖配置（Adaptive Debounce） ====================

    yuying_adaptive_debounce_enabled: bool = Field(default=False, alias="adaptive_debounce_enabled")
    # 是否启用自适应防抖
    # - 作用: 收集用户的碎片消息，等待话轮结束后再统一处理
    # - 默认值: False (不启用，保持现有行为)
    # - True: 启用，会根据消息特征动态计算等待时间
    # - 优点: 减少碎片消息触发回复，提升语义完整性
    # - 缺点: 增加回复延迟

    yuying_adaptive_debounce_mode: str = Field(default="full", alias="adaptive_debounce_mode")
    # 防抖模式
    # - 作用: 控制防抖的影响范围
    # - 可选值: "full" (全链路防抖，包括写库、摘要、记忆)
    # - 默认值: "full"
    # - 说明: 当前仅支持 full 模式

    yuying_adaptive_debounce_joiner: str = Field(default="auto", alias="adaptive_debounce_joiner")
    # 拼接策略
    # - 作用: 控制如何拼接多段消息
    # - 可选值:
    #   - "auto": 自动识别中英文，智能拼接（推荐）
    #   - "": 直接拼接，不加分隔符
    #   - " ": 以空格拼接
    # - 默认值: "auto"

    yuying_adaptive_debounce_ttl_seconds: float = Field(default=60.0, alias="adaptive_debounce_ttl_seconds")
    # 状态 TTL（防御性清理）
    # - 作用: 防抖状态多久未更新后自动清理
    # - 单位: 秒
    # - 默认值: 60.0 (1分钟)
    # - 目的: 防止异常路径导致的内存泄漏

    yuying_adaptive_debounce_max_hold_seconds: float = Field(default=15.0, alias="adaptive_debounce_max_hold_seconds")
    # 硬截止时间（防止无限延迟）
    # - 作用: 从第一段消息开始，最长等待多久后强制 flush
    # - 单位: 秒
    # - 默认值: 15.0 (15秒)
    # - 目的: 避免用户持续分段发送导致永不触发

    yuying_adaptive_debounce_max_parts: int = Field(default=12, alias="adaptive_debounce_max_parts")
    # 最大拼接段数（硬截止）
    # - 作用: 最多拼接多少段消息后强制 flush
    # - 单位: 段
    # - 默认值: 12
    # - 目的: 防止极端情况下的无限拼接

    yuying_adaptive_debounce_max_plain_len: int = Field(default=300, alias="adaptive_debounce_max_plain_len")
    # 最大纯文本长度（硬截止）
    # - 作用: 纯文本长度（去标记去空白后）超过此值后强制 flush
    # - 单位: 字符数
    # - 默认值: 300
    # - 目的: 避免拼接过长的消息

    yuying_adaptive_debounce_w1: float = Field(default=0.6, alias="adaptive_debounce_w1")
    # 等待时间公式 - 一次项系数 (w1·L)
    # - 作用: 控制等待时间随字数增加的速度
    # - 默认值: 0.6 (正数)
    # - 说明: 字数少时，等待时间随字数增加

    yuying_adaptive_debounce_w2: float = Field(default=-0.025, alias="adaptive_debounce_w2")
    # 等待时间公式 - 二次项系数 (w2·L²)
    # - 作用: 控制抛物线开口向下，字数多时时间衰减
    # - 默认值: -0.025 (负数)
    # - 说明: 字数超过阈值后，加速让等待时间衰减

    yuying_adaptive_debounce_w3: float = Field(default=-2.5, alias="adaptive_debounce_w3")
    # 等待时间公式 - 标点符号系数 (w3·P)
    # - 作用: 检测到结束符（。？！?!~）时的时间调整
    # - 默认值: -2.5 (大负数)
    # - 说明: 检测到结束符时，大幅缩短等待时间

    yuying_adaptive_debounce_bias: float = Field(default=1.5, alias="adaptive_debounce_bias")
    # 等待时间公式 - 基础等待时间（截距 b）
    # - 作用: 所有消息的基础等待时间
    # - 单位: 秒
    # - 默认值: 1.5

    yuying_adaptive_debounce_min_wait: float = Field(default=0.5, alias="adaptive_debounce_min_wait")
    # 最小等待时间（下限）
    # - 作用: 等待时间的最小值
    # - 单位: 秒
    # - 默认值: 0.5 (0.5秒)
    # - 目的: 防止计算出负数或过小的等待时间

    yuying_adaptive_debounce_max_wait: float = Field(default=5.0, alias="adaptive_debounce_max_wait")
    # 最大等待时间（上限）
    # - 作用: 等待时间的最大值
    # - 单位: 秒
    # - 默认值: 5.0 (5秒)
    # - 目的: 防止等待时间过长影响用户体验

    # ==================== 心流模式配置 ====================

    yuying_enable_flow_mode: bool = Field(default=False, alias="enable_flow_mode")
    # 是否启用心流模式
    # - 作用: 使用nano模型作为前置决策,判断是否需要回复
    # - 默认值: False (不启用,使用传统概率策略)
    # - True: 启用,每条消息会先经过nano模型判断
    # - 优点: 更智能的回复决策,能理解上下文
    # - 缺点: 增加模型消耗和处理延迟

    yuying_flow_mode_global_cooldown_seconds: int = Field(
        default=30,
        alias="flow_mode_global_cooldown_seconds",
    )
    # 心流模式下的全局回复冷却时间
    # - 作用: 心流模式启用时,全局回复间隔
    # - 单位: 秒
    # - 默认值: 30
    # - 说明: 仅在enable_flow_mode=True时生效

    yuying_flow_mode_group_cooldown_seconds: int = Field(
        default=60,
        alias="flow_mode_group_cooldown_seconds",
    )
    # 心流模式下的群聊回复冷却时间
    # - 作用: 心流模式启用时,单个群的回复间隔
    # - 单位: 秒
    # - 默认值: 60
    # - 说明: 仅在enable_flow_mode=True时生效

    yuying_flow_mode_group_check_probability: float = Field(
        default=0.8,
        alias="flow_mode_group_check_probability",
    )
    # 心流模式下群聊消息被检测的概率
    # - 作用: 群聊消息有多大概率会被nano模型检测
    # - 单位: 小数(0.0-1.0)
    # - 默认值: 0.8 (80%)
    # - 说明: 仅在enable_flow_mode=True时生效
    # - 目的: 避免每条消息都调用nano模型,节省成本

    yuying_flow_mode_private_check_probability: float = Field(
        default=1.0,
        alias="flow_mode_private_check_probability",
    )
    # 心流模式下私聊消息被检测的概率
    # - 作用: 私聊消息有多大概率会被nano模型检测
    # - 单位: 小数(0.0-1.0)
    # - 默认值: 1.0 (100%)
    # - 说明: 仅在enable_flow_mode=True时生效
    # - 私聊通常都需要检测,所以默认100%

    # ==================== 摘要窗口配置 ====================

    yuying_summary_window_message_count: int = Field(
        default=20,
        alias="summary_window_message_count",
    )
    # 摘要窗口的消息数量
    # - 作用: 积累多少条消息后触发一次摘要
    # - 单位: 条
    # - 默认值: 20
    # - 说明: 对话每20条消息会生成一次摘要

    yuying_summary_window_seconds: int = Field(default=900, alias="summary_window_seconds")
    # 摘要窗口的时间跨度
    # - 作用: 多长时间的对话会被视为一个窗口
    # - 单位: 秒
    # - 默认值: 900 (15分钟)
    # - 说明: 超过15分钟的间隔会开启新窗口

    # ==================== MCP（Model Context Protocol）配置 ====================
    #
    # 说明：
    # - MCP 用于把外部工具（搜索/文件/业务系统等）以统一协议提供给 LLM。
    # - 本插件通过 OpenAI "tools/function calling" 机制对接 MCP tools。
    # - 向后兼容：enable_mcp=false 时，完全不走 MCP 代码路径（零影响）。
    #
    # 首期策略（避免过度设计）：
    # - tools 注入：仅注入 enabled=true 的 server 的工具
    # - 工具过滤：支持 allow_tools/deny_tools
    # - schema：尽力转换 + 安全降级（详见 llm/schema_converter.py）
    # - 工具循环：由 planner/action_planner.py 负责

    yuying_enable_mcp: bool = Field(default=False, alias="enable_mcp")
    # 是否启用 MCP
    # - 默认: False（关闭，保持现有行为）
    # - True: 规划阶段允许 LLM 调用 MCP tools

    yuying_mcp_servers: List[MCPServerConfig] = Field(default_factory=list, alias="mcp_servers")
    # MCP server 列表（支持多个）
    # - 每个元素是 MCPServerConfig
    # - 通过 id 唯一标识

    yuying_mcp_lazy_connect: bool = Field(default=True, alias="mcp_lazy_connect")
    # 是否懒加载连接
    # - True（默认）：启动时不连接 server；第一次需要 tools 时才连接/拉取工具
    # - False：启动时尽力连接并预热工具列表（失败可降级，见 mcp_fail_open）

    yuying_mcp_fail_open: bool = Field(default=True, alias="mcp_fail_open")
    # MCP 失败时是否"放行"（降级继续不用工具）
    # - True（默认）：连接失败/列工具失败/执行失败 -> 记录日志 -> 继续走无工具路径
    # - False：把异常上抛（不推荐，可能影响主流程稳定性）

    yuying_mcp_tool_timeout: float = Field(default=15.0, alias="mcp_tool_timeout")
    # 单次工具调用超时（秒）
    # - 作用：避免外部工具卡死拖垮主流程
    # - 注意：并不会改变 MCP server 自身内部超时逻辑，仅在客户端侧限制等待

    yuying_mcp_max_tool_calls: int = Field(default=6, alias="mcp_max_tool_calls")
    # 单次规划允许的最大 tool call 次数（防止无限循环）
    # - 说明：每轮 LLM 回复可能包含多个 tool_calls；这里限制的是"累计 tool_calls 数"

    yuying_mcp_parallel_tools: bool = Field(default=False, alias="mcp_parallel_tools")
    # 同一轮多个 tool_calls 是否并发执行
    # - 默认 False：串行更安全（避免工具之间隐式依赖导致竞态）
    # - True：并发可降低延迟，但需要工具本身支持并发/无依赖

    yuying_mcp_max_parallel_tools: int = Field(default=4, alias="mcp_max_parallel_tools")
    # 并发执行的最大并行度（仅在 mcp_parallel_tools=true 时生效）

    yuying_mcp_tool_result_max_chars: int = Field(default=2000, alias="mcp_tool_result_max_chars")
    # 工具返回结果的最大字符数
    # - 作用：防止工具返回超长内容（如搜索结果）消耗过多 tokens
    # - 单位：字符数
    # - 默认值：2000
    # - 超过限制会截断并标记 truncated=true

    @root_validator(pre=True)
    def _mcp_key_compat(cls, values: Any) -> Any:
        """兼容早期/简写 MCP 配置键，避免被 Extra.ignore 吞掉。

        允许用户在 [yuying_chameleon] 段使用：
        - lazy_connect / fail_open / tool_timeout / max_tool_calls / parallel_tools / ...
        同时保持当前 mcp_* 前缀键可用（优先级更高）。
        """
        if not isinstance(values, dict):
            return values

        mapping = {
            "lazy_connect": "mcp_lazy_connect",
            "fail_open": "mcp_fail_open",
            "tool_timeout": "mcp_tool_timeout",
            "max_tool_calls": "mcp_max_tool_calls",
            "parallel_tools": "mcp_parallel_tools",
            "max_parallel_tools": "mcp_max_parallel_tools",
            "tool_result_max_chars": "mcp_tool_result_max_chars",
        }
        for src, dst in mapping.items():
            if src in values and dst not in values:
                values[dst] = values.get(src)
        return values

    class Config:
        """Pydantic v1内部配置类

        这个内嵌的Config类用于配置Pydantic模型的行为:
        - extra = Extra.ignore: 忽略未定义的额外字段(不报错)
        - allow_population_by_field_name = True: 允许使用字段原名(yuying_*)设置值
        """

        extra = Extra.ignore  # 忽略配置中多余的未知字段
        allow_population_by_field_name = True  # 允许使用完整字段名和alias同时生效


def _discover_config_toml() -> Optional[Path]:
    """查找config.toml配置文件的位置

    查找策略(按优先级):
    1. 环境变量YUYING_CONFIG_TOML指定的路径(最高优先级)
    2. 当前工作目录下的configs/config.toml
    3. 从当前文件向上遍历父目录,查找configs/config.toml

    Returns:
        Optional[Path]: 找到的配置文件路径,找不到返回None

    Side Effects:
        - 读取环境变量YUYING_CONFIG_TOML
        - 访问文件系统,检查文件是否存在
    """

    # 策略1: 检查环境变量
    # os.getenv("YUYING_CONFIG_TOML"): 获取环境变量的值,不存在返回None
    # or "": 如果环境变量不存在,使用空字符串
    # .strip(): 去除首尾空格
    env_path = (os.getenv("YUYING_CONFIG_TOML") or "").strip()
    if env_path:  # 如果环境变量有值
        path = Path(env_path)  # 转为Path对象
        # .exists(): 检查路径是否存在
        # .is_file(): 检查是否是文件(不是目录)
        if path.exists() and path.is_file():
            return path  # 找到就立即返回

    # 策略2: 检查当前工作目录
    # Path.cwd(): 获取当前工作目录(运行nb run的目录)
    cwd_path = Path.cwd() / "configs" / "config.toml"
    if cwd_path.exists() and cwd_path.is_file():
        return cwd_path

    # 策略3: 向上查找父目录
    # __file__: 当前Python文件(config.py)的路径
    # .resolve(): 转为绝对路径
    here = Path(__file__).resolve()
    # here.parents: 所有父目录的序列
    for parent in here.parents:
        # 在每个父目录下查找configs/config.toml
        candidate = parent / "configs" / "config.toml"
        if candidate.exists() and candidate.is_file():
            return candidate  # 找到就返回

    # 所有策略都失败,返回None
    return None


def load_config() -> Config:
    """加载插件配置并返回Config对象

    配置加载流程(从低到高优先级):
    1. 使用Config类中定义的默认值
    2. 读取configs/config.toml文件(如果存在)
    3. 读取NoneBot的全局配置(driver.config)
    4. 合并并验证所有配置

    配置合并规则:
    - 后加载的配置会覆盖先加载的
    - 空字符串和空列表会被视为"未配置",会被��盖
    - None值会被覆盖
    - 非空值不会被覆盖

    特殊处理:
    - 三路provider(main/cheap/embedder)未配置时回退到api_provider
    - API密钥未配置时自动复用openai_api_key
    - 根据provider自动设置base_url的默认值

    Returns:
        Config: 加载并处理后的配置对象

    Side Effects:
        - 调用get_driver()访问NoneBot驱动器
        - 读取配置文件(如果存在)
        - 输出日志记录配置加载结果
    """

    # 步骤1: 尝试获取NoneBot的驱动器配置
    driver_cfg = None
    try:
        # get_driver(): 获取NoneBot的驱动器对象
        # .config: 驱动器的配置对象,包含所有全局配置
        driver_cfg = get_driver().config
    except Exception:
        # 如果获取失败(比如在测试环境),使用None
        driver_cfg = None

    # 步骤2: 构建原始配置字典
    raw: dict = {}  # 用于存储所有配置项的字典
    if driver_cfg is not None:
        try:
            # .dict(): 将Pydantic模型转为字典
            # raw.update(): 合并字典,新值覆盖旧值
            raw.update(driver_cfg.dict())
        except Exception:
            pass  # 转换失败就跳过

        # 检查是否有yuying_chameleon段配置
        # getattr(obj, name, default): 获取对象的属性,不存在返回default
        section = getattr(driver_cfg, "yuying_chameleon", None)
        if isinstance(section, dict):
            # 如果有专门的yuying_chameleon段,其优先级更高
            raw.update(section)

    # 步骤3: 读取TOML配置文件(如果存在)
    path = _discover_config_toml()  # 查找配置文件
    if path:  # 如果找到了配置文件
        try:
            # 尝试导入TOML解析库
            try:
                import tomllib  # Python 3.11+自带
            except Exception:  # pragma: no cover
                import tomli as tomllib  # type: ignore[no-redef]  # Python 3.10及以下需要安装

            # .read_text(): 读取文件内容为字符串
            # tomllib.loads(): 解析TOML格式字符串为字典
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            # 获取[yuying_chameleon]段
            file_section = data.get("yuying_chameleon")
            if isinstance(file_section, dict):
                # 遍历文件中的每个配置项
                for k, v in file_section.items():
                    # 合并规则:只有当raw中没有配置或值为空时,才使用文件中的值
                    if k not in raw:  # 如果raw中没有这个键
                        raw[k] = v
                        continue
                    existing = raw.get(k)  # 获取raw中的现有值
                    if existing is None:  # 如果是None
                        raw[k] = v
                        continue
                    if isinstance(existing, str) and not existing.strip():  # 如果是空字符串
                        raw[k] = v
                        continue
                    if isinstance(existing, list) and not existing:  # 如果是空列表
                        raw[k] = v
        except Exception as e:
            # 读取文件失败,记录警告但不中断
            logger.warning(f"读取配置文件失败:{path},{e}")

    # 步骤4: 使用Pydantic解析并验证配置
    # Config.parse_obj(): 从字典创建Config对象,自动进行类型检查和转换
    cfg = Config.parse_obj(raw)

    # 步骤5: 处理provider配置
    def _normalize_provider(value: str) -> str:
        """规范化provider字符串

        将不同的provider别名统一为标准名称:
        - "ark", "volc", "volcengine", "doubao" -> "ark"
        - "openai", "oa" -> "openai"

        Args:
            value: 原始provider字符串

        Returns:
            str: 规范化后的provider名称
        """
        v = (value or "").strip().lower()  # 转小写并去空格
        if not v:
            return ""
        # 火山方舟的各种别名
        if v in {"ark", "volc", "volces", "volcengine", "doubao"}:
            return "ark"
        # OpenAI的别名
        if v in {"openai", "oa"}:
            return "openai"
        return v  # 其他值保持原样

    def _apply_provider_default_base_url(provider: str, current: str) -> str:
        """根据provider设置默认的base_url

        只在当前值是默认值或空值时才覆盖,不影响用户自定义的URL

        Args:
            provider: provider类型("ark"或"openai")
            current: 当前的base_url值

        Returns:
            str: 应该使用的base_url
        """
        cur = (current or "").strip()  # 去空格
        if provider == "ark":  # 如果是火山方舟
            ark_default = "https://ark.cn-beijing.volces.com/api/v3"
            # 只在是默认OpenAI地址或空值时替换
            if cur in {"", "https://api.openai.com/v1"}:
                return ark_default
            return cur  # 用户自定义的URL不动
        if provider == "openai":  # 如果是OpenAI
            if cur == "":  # 只在空值时设置默认值
                return "https://api.openai.com/v1"
            return cur
        return cur  # 其他provider保持原样

    # 规范化三路provider
    # 未单独配置时回退使用全局provider
    global_provider = _normalize_provider(cfg.yuying_api_provider) or "openai"
    main_provider = _normalize_provider(cfg.yuying_main_provider) or global_provider
    cheap_provider = _normalize_provider(cfg.yuying_cheap_llm_provider) or global_provider
    embedder_provider = _normalize_provider(cfg.yuying_embedder_provider) or global_provider
    nano_provider = _normalize_provider(cfg.yuying_nano_llm_provider) or global_provider

    # 根据provider自动设置base_url
    cfg.yuying_openai_base_url = _apply_provider_default_base_url(
        main_provider,
        cfg.yuying_openai_base_url,
    )
    cfg.yuying_cheap_llm_base_url = _apply_provider_default_base_url(
        cheap_provider,
        cfg.yuying_cheap_llm_base_url,
    )
    cfg.yuying_embedder_base_url = _apply_provider_default_base_url(
        embedder_provider,
        cfg.yuying_embedder_base_url,
    )
    cfg.yuying_nano_llm_base_url = _apply_provider_default_base_url(
        nano_provider,
        cfg.yuying_nano_llm_base_url,
    )

    # 步骤6: API密钥的自动复用
    # 如果embedder/cheap_llm/nano_llm的API key未配置,自动复用主模型的key
    if not (cfg.yuying_embedder_api_key or "").strip():
        cfg.yuying_embedder_api_key = cfg.yuying_openai_api_key
    if not (cfg.yuying_cheap_llm_api_key or "").strip():
        cfg.yuying_cheap_llm_api_key = cfg.yuying_openai_api_key
    if not (cfg.yuying_nano_llm_api_key or "").strip():
        cfg.yuying_nano_llm_api_key = cfg.yuying_openai_api_key

    # base_url也可以复用
    if not (cfg.yuying_embedder_base_url or "").strip():
        cfg.yuying_embedder_base_url = cfg.yuying_openai_base_url
    if not (cfg.yuying_cheap_llm_base_url or "").strip():
        cfg.yuying_cheap_llm_base_url = cfg.yuying_openai_base_url
    if not (cfg.yuying_nano_llm_base_url or "").strip():
        cfg.yuying_nano_llm_base_url = cfg.yuying_openai_base_url

    # 步骤7: 输出配置加载日志(不泄露敏感信息)
    # bool(xxx): 检查API key是否已配置(非空),但不输出具体内容
    openai_set = bool((cfg.yuying_openai_api_key or "").strip())
    embedder_set = bool((cfg.yuying_embedder_api_key or "").strip())
    cheap_set = bool((cfg.yuying_cheap_llm_api_key or "").strip())
    nano_set = bool((cfg.yuying_nano_llm_api_key or "").strip())

    # logger.info(): 输出INFO级别日志
    logger.info(
        "YuYing-Chameleon 配置加载完成:provider(main/cheap/embedder/nano)={}/{}/{}/{} "
        "openai_api_key={} embedder_api_key={} cheap_llm_api_key={} nano_llm_api_key={} "
        "enable_flow_mode={} config_file={}",
        main_provider,
        cheap_provider,
        embedder_provider,
        nano_provider,
        "已配置" if openai_set else "未配置",  # 只显示是否配置,不显示具体值
        "已配置" if embedder_set else "未配置",
        "已配置" if cheap_set else "未配置",
        "已配置" if nano_set else "未配置",
        cfg.yuying_enable_flow_mode,
        str(path) if path else "未发现/未使用",
    )

    return cfg


# 模块级全局变量: 在模块导入时立即加载配置
# 其他模块可以通过 from .config import plugin_config 来使用配置
plugin_config = load_config()
