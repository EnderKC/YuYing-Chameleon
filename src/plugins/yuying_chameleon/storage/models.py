"""数据库模型定义 - SQLAlchemy ORM模型(SQLite数据库的表结构定义)

这个模块的作用:
1. 定义所有数据库表的结构(表名、字段、类型、索引)
2. 使用SQLAlchemy ORM将Python类映射到数据库表
3. 提供类型安全的数据访问接口

数据库设计说明:
- 使用SQLite作为存储引擎
- 所有时间戳使用Unix时间戳(整数,秒级)
- 使用索引优化常见查询操作
- 字段类型明确,支持类型检查

表分类(共10张表):
1. 用户: UserProfile(用户档案)
2. 消息: RawMessage(原始消息), Summary(对话摘要)
3. 记忆: Memory(用户记忆), MemoryEvidence(记忆证据链接)
4. 媒体: MediaCache(图片等媒体的处理结果缓存)
5. 表情: Sticker(表情包库), StickerCandidate(候选表情), StickerUsage(使用统计)
6. 系统: BotRateLimit(回复限流状态), IndexJob(向量索引任务队列)

关键概念(新手必读):
- ORM(对象关系映射): 用Python类操作数据库,无需手写SQL
- Mapped[类型]: SQLAlchemy 2.0的类型注解,既做类型提示也定义字段
- mapped_column(): 定义字段属性(类型、默认值、约束等)
- Index: 数据库索引,加速查询但占用空间
- 主键(primary_key): 每行记录的唯一标识,不可重复
- 外键(foreign key): 表之间的关联(本项目暂无显式外键约束)
"""

from __future__ import annotations

# typing.Optional - 类型提示,表示可选值(可以是指定类型或None)
from typing import Optional

# SQLAlchemy类型导入
from sqlalchemy import String, Integer, Text, Boolean, Float, Index, UniqueConstraint
# - String: 可变长度字符串(VARCHAR)
# - Integer: 整数类型(INT)
# - Text: 长文本类型(TEXT),无长度限制
# - Boolean: 布尔值(0/1)
# - Float: 浮点数(REAL)
# - Index: 索引定义,用于优化查询性能

# SQLAlchemy ORM核心类
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
# - DeclarativeBase: ORM模型的声明式基类
# - Mapped[T]: SQLAlchemy 2.0的类型注解,表示映射到数据库的字段
# - mapped_column(): 定义数据库列的配置函数

import time  # Python标准库,用于获取Unix时间戳


class Base(DeclarativeBase):
    """ORM 基类 - 所有数据库模型类的父类

    作用:
    - 提供SQLAlchemy的声明式映射基础
    - 所有数据库模型都继承自这个类
    - 自动生成表的元数据信息

    使用方式:
    ```python
    class MyModel(Base):
        __tablename__ = "my_table"
        id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ```
    """

    pass  # 空实现,仅作为基类使用

class UserProfile(Base):
    """用户档案表 - 以QQ号为主键,记录用户的基本统计信息和记忆提取状态

    这个表的作用:
    1. 记录每个用户的有效发言数,用于触发记忆提取
    2. 跟踪记忆提取进度,避免重复处理同一批消息
    3. 标记待处理的记忆任务,供后台worker认领
    4. 动态调整记忆提取阈值,优化性能

    数据增长:
    - 缓慢增长,每个与机器人互动的用户一条记录
    - 预计数据量: 活跃用户数 × 1条

    索引策略:
    - 主键索引: qq_id (自动创建)
    """

    __tablename__ = "user_profile"  # 数据库表名

    # ==================== 主键字段 ====================
    qq_id: Mapped[str] = mapped_column(String, primary_key=True)
    # QQ号 - 用户的唯一标识
    # - 作用: 作为用户档案的主键,关联其他表的用户数据
    # - 类型: 字符串(因为QQ号可能很长,用字符串更安全)
    # - 约束: 主键(primary_key=True),不可为空,不可重复
    # - 示例: "12345678", "1234567890"

    # ==================== 记忆提取相关字段 ====================
    effective_count: Mapped[int] = mapped_column(Integer, default=0)
    # 有效发言数 - 触发记忆提取的计数器
    # - 作用: 统计用户发送的"有效消息"数量
    # - 有效消息: 非纯表情、非纯空格的正常对话内容
    # - 默认值: 0 (新用户从0开始计数)
    # - 更新时机: 每次用户发送有效消息时+1
    # - 触发条件: 达到next_memory_at时触发记忆提取

    next_memory_at: Mapped[int] = mapped_column(Integer, default=50)
    # 下次记忆提取阈值 - 动态调整的目标值
    # - 作用: 当effective_count达到此值时触发记忆提取
    # - 默认值: 50 (首次触发需要50条有效消息)
    # - 动态调整: 提取后会根据提取质量自动调整(如50→100→200)
    # - 目的: 避免频繁提取,在数据积累充分后再处理

    last_memory_msg_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # 上次记忆提取处理到的消息ID - 断点续传标记
    # - 作用: 记录上次记忆提取处理到RawMessage表的哪个ID
    # - 类型: 可空整数(Optional[int])
    # - 默认值: None (新用户尚未提取过记忆)
    # - 用途: 避免重复处理已提取过的消息,实现增量提取
    # - 关联: 指向RawMessage.id

    pending_memory: Mapped[bool] = mapped_column(Boolean, default=False)
    # 待处理记忆标记 - 是否有待提取的记忆任务
    # - 作用: 标记此用户是否有待处理的记忆提取任务
    # - 默认值: False (无待处理任务)
    # - 设为True: 当effective_count达到next_memory_at时
    # - 设回False: 后台worker完成记忆提取后
    # - 目的: 让后台worker知道哪些用户需要处理

    # ==================== 时间戳字段 ====================
    created_at: Mapped[int] = mapped_column(Integer, default=lambda: int(time.time()))
    # 创建时间 - 用户首次与机器人互动的时间
    # - 作用: 记录用户档案的创建时间
    # - 类型: Unix时间戳(整数,秒级)
    # - 默认值: lambda: int(time.time()) - 插入记录时的当前时间
    # - 只设置一次: 创建后不再修改

    updated_at: Mapped[int] = mapped_column(
        Integer,
        default=lambda: int(time.time()),  # 创建时的初始值
        onupdate=lambda: int(time.time()),  # 每次更新时自动更新
    )
    # 更新时间 - 用户档案最后一次修改的时间
    # - 作用: 追踪用户档案的最后修改时间
    # - 类型: Unix时间戳(整数,秒级)
    # - 默认值: 创建时的时间
    # - 自动更新: 每次update操作时自动更新为当前时间(onupdate)
    # - 用途: 数据审计、用户活跃度分析

class RawMessage(Base):
    """原始消息表 - 存储所有收到和发送的消息(包括机器人自己发送的消息)

    这个表的作用:
    1. 完整记录所有消息历史,用于对话上下文恢复
    2. 提供记忆提取的数据源(分析用户对话提取信息)
    3. 作为摘要生成的输入(压缩历史消息)
    4. 支持引用回复(通过reply_to_msg_id建立消息链)
    5. 记录机器人自己的回复(is_bot=True),保持对话完整性

    数据增长:
    - 快速增长,每条消息(用户+机器人)都会记录
    - 预计数据量: 活跃度 × 时间,可能达到数十万至百万级
    - 清理策略: 定期归档或删除3个月前的消息(保留摘要即可)

    索引策略:
    - 主键: id (自增)
    - 复合索引1: (qq_id, timestamp) - 用于查询某用户的历史消息
    - 复合索引2: (scene_type, scene_id, timestamp) - 用于查询某场景(群/私聊)的历史消息

    字段分类:
    - 标识字段: id, qq_id, scene_type, scene_id, timestamp
    - 内容字段: msg_type, content, raw_ref
    - 关系字段: reply_to_msg_id
    - 状态字段: mentioned_bot, is_effective, is_bot
    """

    __tablename__ = "raw_messages"  # 数据库表名

    # ==================== 主键与标识字段 ====================
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 消息ID - 自增主键
    # - 作用: 每条消息的唯一标识
    # - 类型: 自增整数(autoincrement=True)
    # - 用途: 作为消息引用、记忆证据链接的ID
    # - 单调性: 保证时间顺序(ID越大,消息越新)

    qq_id: Mapped[str] = mapped_column(String)
    # 发送者QQ号 - 消息的发送方
    # - 作用: 标识消息是谁发送的
    # - 类型: 字符串(QQ号)
    # - 范围: 用户QQ号 或 机器人自己的QQ号(当is_bot=True时)
    # - 关联: 对应UserProfile.qq_id

    scene_type: Mapped[str] = mapped_column(String)
    # 场景类型 - 消息发生的场景类别
    # - 作用: 区分消息是群聊还是私聊
    # - 类型: 字符串,只有两种值:
    #   * "group" - 群聊消息
    #   * "private" - 私聊消息
    # - 用途: 与scene_id组合定位唯一的对话场景

    scene_id: Mapped[str] = mapped_column(String)
    # 场景标识 - 具体的场景ID
    # - 作用: 标识具体的群号或私聊对象
    # - 类型: 字符串
    # - 含义:
    #   * 当scene_type="group"时,scene_id是群号
    #   * 当scene_type="private"时,scene_id是对方QQ号
    # - 组合键: (scene_type, scene_id)唯一标识一个对话场景

    timestamp: Mapped[int] = mapped_column(Integer)
    # 消息时间戳 - 消息发送的时间
    # - 作用: 记录消息发生的准确时间
    # - 类型: Unix时间戳(整数,秒级)
    # - 来源: 从QQ服务器获取的消息时间
    # - 用途: 排序消息、时间窗口查询、过期数据清理

    # ==================== 消息内容字段 ====================
    msg_type: Mapped[str] = mapped_column(String)
    # 消息类型 - 消息包含的内容类型
    # - 作用: 标记消息的内容构成
    # - 类型: 字符串,可能的值:
    #   * "text" - 纯文本消息
    #   * "image" - 纯图片消息
    #   * "mixed" - 图文混合消息
    #   * 可扩展其他类型
    # - 用途: 快速筛选特定类型的消息

    content: Mapped[str] = mapped_column(Text)
    # 消息内容 - 归一化后的文本内容
    # - 作用: 存储处理后的消息文本
    # - 类型: 长文本(Text,无长度限制)
    # - 处理:
    #   * 纯文本: 直接存储
    #   * 包含图片: 替换为占位符如"[image:xxx]"
    #   * 包含@: 替换为"@某人"
    # - 用途: LLM理解、向量化、全文检索

    raw_ref: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # 原始引用 - 消息中引用的资源(如图片URL)
    # - 作用: 保存消息中的原始资源引用
    # - 类型: 可空长文本(nullable=True)
    # - 内容:
    #   * 图片消息: 存储图片URL或文件路径
    #   * 纯文本消息: None
    #   * 多图消息: JSON格式存储多个URL
    # - 用途: 下载图片、OCR处理、表情包偷取

    # ==================== 关系字段 ====================
    reply_to_msg_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # 引用回复的消息ID - 当前消息回复的是哪条消息
    # - 作用: 建立消息之间的引用关系链
    # - 类型: 可空整数(nullable=True)
    # - 默认值: None (不是回复任何消息)
    # - 关联: 指向同表的另一条消息的id
    # - 用途: 构建对话上下文、理解对话关联

    # ==================== 状态标记字段 ====================
    mentioned_bot: Mapped[bool] = mapped_column(Boolean, default=False)
    # @机器人标记 - 消息是否@了机器人
    # - 作用: 标记用户是否显式@机器人
    # - 类型: 布尔值
    # - 默认值: False (未@)
    # - 识别方式: 解析消息中的CQ码或@名称
    # - 用途: 回复策略(被@时必定回复)

    is_effective: Mapped[bool] = mapped_column(Boolean, default=False)
    # 有效消息标记 - 是否是有效的对话消息
    # - 作用: 区分是否是正常对话内容
    # - 类型: 布尔值
    # - 默认值: False
    # - 有效消息: 包含实质性内容的消息
    # - 无效消息: 纯表情、纯空格、系统提示等
    # - 用途: 统计用户有效发言数,触发记忆提取

    is_bot: Mapped[bool] = mapped_column(Boolean, default=False)
    # 机器人消息标记 - 是否是机器人自己发送的消息
    # - 作用: 区分消息是用户发的还是机器人发的
    # - 类型: 布尔值
    # - 默认值: False (用户消息)
    # - True时: qq_id是机器人的QQ号
    # - 用途: 构建完整对话历史、避免重复处理自己的消息

    # ==================== 索引定义 ====================
    __table_args__ = (
        Index("idx_raw_qq_ts", "qq_id", "timestamp"),
        # 索引1: (qq_id, timestamp) 联合索引
        # - 用途: 查询某用户的历史消息,按时间排序
        # - 查询示例: SELECT * FROM raw_messages WHERE qq_id='123' ORDER BY timestamp DESC

        Index("idx_raw_scene_ts", "scene_type", "scene_id", "timestamp"),
        # 索引2: (scene_type, scene_id, timestamp) 联合索引
        # - 用途: 查询某场景(群聊/私聊)的历史消息,按时间排序
        # - 查询示例: SELECT * FROM raw_messages WHERE scene_type='group' AND scene_id='456' ORDER BY timestamp DESC
        # - 覆盖查询: 索引包含了查询和排序所需的所有字段,性能最优
    )

class Summary(Base):
    """对话摘要表 - 按场景和时间窗口生成的对话摘要

    这个表的作用:
    1. 压缩大量历史消息为简短摘要,节省LLM上下文
    2. 提供远期对话的概览,帮助LLM理解对话背景
    3. 按窗口滚动生成,平衡实时性和完整性
    4. 减少对RawMessage表的频繁查询,提高性能

    滚动窗口策略:
    - 窗口大小: 默认20条消息或15分钟
    - 触发条件: 消息数达到阈值 OR 时间超过阈值
    - 重叠策略: 窗口之间可能有轻微重叠,保证连续性

    数据增长:
    - 中等增长,每个活跃场景每天数条至数十条
    - 预计数据量: 活跃场景数 × 每天对话轮数
    - 清理策略: 保留最近30天,旧摘要可删除

    索引策略:
    - 主键: id (自增)
    - 复合索引: (scene_type, scene_id, window_end_ts) - 查询某场景的最新摘要
    """

    __tablename__ = "summaries"  # 数据库表名

    # ==================== 主键 ====================
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 摘要ID - 自增主键
    # - 作用: 唯一标识每个摘要记录
    # - 类型: 自增整数

    # ==================== 场景标识字段 ====================
    scene_type: Mapped[str] = mapped_column(String)
    # 场景类型 - 摘要所属的场景类别
    # - 作用: 标识这是群聊摘要还是私聊摘要
    # - 类型: 字符串,取值:
    #   * "group" - 群聊摘要
    #   * "private" - 私聊摘要
    # - 说明: 不同场景的摘要独立生成和存储

    scene_id: Mapped[str] = mapped_column(String)
    # 场景标识 - 具体的群号或私聊对象
    # - 作用: 定位到具体的对话场景
    # - 类型: 字符串(群号或QQ号)
    # - 组合键: (scene_type, scene_id)确定唯一场景

    # ==================== 时间窗口字段 ====================
    window_start_ts: Mapped[int] = mapped_column(Integer)
    # 窗口起始时间 - 摘要覆盖的时间范围的起点
    # - 作用: 标记这个摘要包含哪段时间的对话
    # - 类型: Unix时间戳(整数,秒级)
    # - 来源: 窗口内第一条消息的timestamp
    # - 用途: 时间范围查询、摘要完整性验证

    window_end_ts: Mapped[int] = mapped_column(Integer)
    # 窗口结束时间 - 摘要覆盖的时间范围的终点
    # - 作用: 标记摘要包含到哪个时间点
    # - 类型: Unix时间戳(整数,秒级)
    # - 来源: 窗口内最后一条消息的timestamp
    # - 用途: 排序摘要、查找最新摘要、窗口衔接

    # ==================== 摘要内容字段 ====================
    summary_text: Mapped[str] = mapped_column(Text)
    # 摘要文本 - LLM生成的对话摘要内容
    # - 作用: 存储压缩后的对话内容
    # - 类型: 长文本(Text)
    # - 生成方式: 调用LLM API,输入窗口内的所有消息,输出摘要
    # - 内容: 简明扼要地概括对话主题、关键信息、情绪走向
    # - 用途: 提供给LLM作为远期上下文,帮助理解对话历史

    topic_state_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # 话题状态JSON - 结构化的话题跟踪信息(可选扩展字段)
    # - 作用: 记录对话中的话题流转、待办事项等结构化信息
    # - 类型: 可空长文本(存储JSON格式字符串)
    # - 默认值: None (简单场景可不使用)
    # - JSON内容示例:
    #   {
    #     "topics": ["天气", "晚饭吃什么"],
    #     "pending_questions": ["明天去哪玩"],
    #     "mood": "轻松愉快"
    #   }
    # - 用途: 高级对话管理、多轮对话追踪

    # ==================== 索引定义 ====================
    __table_args__ = (
        Index("idx_sum_scene_end", "scene_type", "scene_id", "window_end_ts"),
        # 索引: (scene_type, scene_id, window_end_ts) 联合索引
        # - 用途: 查询某场景的最新N个摘要,按时间倒序
        # - 查询示例:
        #   SELECT * FROM summaries
        #   WHERE scene_type='group' AND scene_id='123'
        #   ORDER BY window_end_ts DESC
        #   LIMIT 5
        # - 性能: 覆盖索引,查询效率高
    )

class BotPersonalityMemory(Base):
    """机器人"人格记忆"表 - 两层架构(recent/core)的自我反思产物

    目标：
    - recent：最近 7 天的观察（每日 04:00 反思生成），用于短期风格/情绪调节
    - core：长期原则（由 recent 浓缩而来），永久保留但允许被"更新"（同一条原则覆盖更新）

    设计原则（关键约束）：
    - 只记录"沟通方式建议/自我调节策略"，避免对用户做人身评价或负面标签
    - 情绪影响必须可衰减：decay_weight + decay_half_life_hours

    索引/唯一性说明：
    - SQLite 的 UNIQUE + NULL 存在"允许重复"的语义坑；
      因此 scope_id 设计为非空字符串：global 场景用空字符串 ""，避免幂等失效。
    """

    __tablename__ = "bot_personality_memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ==================== 维度字段（两层 + 类型 + scope） ====================
    tier: Mapped[str] = mapped_column(String)
    # recent / core

    memory_type: Mapped[str] = mapped_column(String)
    # group_activity / relationship / emotion_state

    scope_type: Mapped[str] = mapped_column(String)
    # global / group

    scope_id: Mapped[str] = mapped_column(String, default="")
    # global: ""（空字符串）
    # group: 群号（scene_id）

    memory_key: Mapped[str] = mapped_column(String)
    # recent: "YYYY-MM-DD"（按天幂等）
    # core: 固定为 "core"（每个 scope+type 只有一条长期原则，按日覆盖更新）

    # ==================== 时间窗口（统一使用 Unix 秒级） ====================
    memory_date: Mapped[str] = mapped_column(String)
    # "YYYY-MM-DD"（core 表示最近一次更新的日期）

    window_start_ts: Mapped[int] = mapped_column(Integer)
    window_end_ts: Mapped[int] = mapped_column(Integer)

    # ==================== 内容字段 ====================
    title: Mapped[str] = mapped_column(String)
    content: Mapped[str] = mapped_column(Text)
    action_hint: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    importance: Mapped[float] = mapped_column(Float, default=0.5)

    # ==================== 情绪字段（仅 emotion_state 使用，其余可为空） ====================
    emotion_label: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    # happy / neutral / unhappy

    emotion_valence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # [-1, 1]

    decay_weight: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # [0, 1]：昨日情绪影响的"初始强度"（运行时再按半衰期计算有效权重）

    decay_half_life_hours: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # 建议默认 24

    # ==================== 证据与审计 ====================
    evidence_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # JSON 字符串：必须包含 summary_ids + stats；允许 top_qq_ids 但只能用于证据，不得输出对人的评判

    run_id: Mapped[str] = mapped_column(String)
    model: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    prompt_version: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    created_at_ts: Mapped[int] = mapped_column(Integer, default=lambda: int(time.time()))
    updated_at_ts: Mapped[int] = mapped_column(
        Integer,
        default=lambda: int(time.time()),
        onupdate=lambda: int(time.time()),
    )
    deleted_at_ts: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "tier",
            "scope_type",
            "scope_id",
            "memory_type",
            "memory_key",
            name="uq_bot_personality_key",
        ),
        Index(
            "idx_bpm_lookup",
            "tier",
            "scope_type",
            "scope_id",
            "memory_type",
            "updated_at_ts",
        ),
        Index("idx_bpm_tier_window_end", "tier", "window_end_ts"),
        Index("idx_bpm_tier_type_updated", "tier", "memory_type", "updated_at_ts"),
    )

class Memory(Base):
    """用户记忆表 - 三层存储架构(active/archive/core)实现长期记忆

    这个表的作用:
    1. 存储从用户对话中提取的结构化信息(事实、偏好、习惯、经历)
    2. 提供个性化上下文给LLM,实现长期对话连贯性
    3. 通过分层管理优化记忆的查询效率和存储成本
    4. 支持记忆的置信度评估和生命周期管理

    三层架构说明:
    - active(活跃层): 最近提取的、经常使用的记忆,查询优先级最高
    - archive(归档层): 较久未用的记忆,降低优先级但保留
    - core(核心层): 经过浓缩的核心记忆,永久保留

    记忆类型(type字段):
    - fact: 客观事实(如"用户是程序员")
    - preference: 主观偏好(如"喜欢吃辣")
    - habit: 行为习惯(如"晚上11点睡觉")
    - experience: 重要经历(如"上周去了北京")

    数据增长:
    - 中等增长,每个用户约20-100条
    - 预计数据量: 活跃用户数 × 平均记忆数
    - 清理策略: 低置信度记忆自动淘汰,超过ttl_days的自动删除

    索引策略:
    - 主键: id (自增)
    - 复合索引1: (qq_id, tier, updated_at) - 按层级查询用户记忆
    - 复合索引2: (qq_id, type, updated_at) - 按类型查询用户记忆
    """

    __tablename__ = "memories"  # 数据库表名

    # ==================== 主键 ====================
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 记忆ID - 自增主键
    # - 作用: 唯一标识每条记忆
    # - 类型: 自增整数
    # - 用途: 作为证据链接、记忆合并时的来源追溯

    # ==================== 用户标识 ====================
    qq_id: Mapped[str] = mapped_column(String)
    # 用户QQ号 - 记忆所属的用户
    # - 作用: 标识这条记忆属于哪个用户
    # - 类型: 字符串(QQ号)
    # - 关联: 对应UserProfile.qq_id

    # ==================== 记忆分层与分类 ====================
    tier: Mapped[str] = mapped_column(String)
    # 记忆层级 - 三层架构的层级标识
    # - 作用: 区分记忆的重要性和查询优先级
    # - 类型: 字符串,取值:
    #   * "active" - 活跃记忆,最近提取或经常使用
    #   * "archive" - 归档记忆,较久未用但保留
    #   * "core" - 核心记忆,经过浓缩,永久保留
    # - 流转: active → archive (长期不用) → core (浓缩合并)
    # - 查询优先级: active > core > archive

    type: Mapped[str] = mapped_column(String)
    # 记忆类型 - 记忆的语义分类
    # - 作用: 标识记忆的内容类别
    # - 类型: 字符串,取值:
    #   * "fact" - 客观事实,如"用户是程序员"、"今年25岁"
    #   * "preference" - 主观偏好,如"喜欢吃辣"、"不喜欢运动"
    #   * "habit" - 行为习惯,如"每天晚上11点睡觉"、"周末爱打游戏"
    #   * "experience" - 重要经历,如"上周去了北京旅游"
    # - 用途: 按类型过滤、分析用户画像

    # ==================== 记忆内容 ====================
    content: Mapped[str] = mapped_column(Text)
    # 记忆内容 - 记忆的具体文本描述
    # - 作用: 存储记忆的主体内容
    # - 类型: 长文本(Text)
    # - 格式: 简洁的陈述句,如"用户是一名Python程序员"
    # - 提取方式: LLM从对话中分析提取
    # - 用途: 提供给LLM作为用户背景信息

    confidence: Mapped[float] = mapped_column(Float)
    # 置信度 - 记忆的可靠程度评分
    # - 作用: 评估这条记忆的准确性和重要性
    # - 类型: 浮点数,范围 0.0-1.0
    # - 示例:
    #   * 0.9: 用户明确表述的信息,高置信度
    #   * 0.6: 从对话推断出的信息,中等置信度
    #   * 0.3: 不确定的猜测,低置信度
    # - 用途: 低置信度记忆可能被淘汰,高置信度优先使用

    # ==================== 记忆状态管理 ====================
    status: Mapped[str] = mapped_column(String)
    # 记忆状态 - 记忆的当前生命周期状态
    # - 作用: 标记记忆是否可用
    # - 类型: 字符串,取值:
    #   * "active" - 正常可用
    #   * "archived" - 已归档,降低优先级
    #   * "deleted" - 已删除,不再使用(软删除)
    # - 注意: 与tier字段不同,status是状态,tier是层级

    visibility: Mapped[str] = mapped_column(String)
    # 可见性 - 记忆的作用域范围
    # - 作用: 控制记忆在哪些场景下可见
    # - 类型: 字符串,取值:
    #   * "global" - 全局可见,所有场景都可用
    #   * "scene" - 场景限定,仅特定场景可见(如某个群)
    #   * "private" - 私有,仅私聊可见
    # - 用途: 避免群聊记忆泄漏到私聊,保护隐私

    scope_scene_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    # 作用域场景ID - 当visibility="scene"时,指定具体场景
    # - 作用: 限定记忆仅在某个特定场景(群)可见
    # - 类型: 可空字符串(场景ID)
    # - 默认值: None (global或private时不需要)
    # - 示例: 群号"123456"表示仅在该群可见
    # - 用途: 实现场景隔离,避免信息串场

    ttl_days: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # 生存时间(TTL) - 记忆的有效期天数
    # - 作用: 设置记忆的自动过期时间
    # - 类型: 可空整数(天数)
    # - 默认值: None (永久保留)
    # - 示例: 7表示7天后自动删除
    # - 用途: 短期记忆(如"今天要开会")自动过期清理

    source_memory_ids: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # 来源记忆ID列表 - 记忆浓缩时的溯源信息
    # - 作用: 记录这条记忆是由哪些旧记忆合并而来
    # - 类型: 可空长文本(存储ID列表,用逗号分隔或JSON格式)
    # - 默认值: None (原始提取的记忆)
    # - 示例: "123,456,789" 表示由记忆123、456、789合并而来
    # - 用途: 记忆溯源、浓缩质量评估、回溯原始证据

    # ==================== 时间戳字段 ====================
    created_at: Mapped[int] = mapped_column(Integer, default=lambda: int(time.time()))
    # 创建时间 - 记忆首次提取的时间
    # - 作用: 记录记忆的创建时间
    # - 类型: Unix时间戳(整数,秒级)
    # - 默认值: 插入记录时的当前时间

    updated_at: Mapped[int] = mapped_column(
        Integer,
        default=lambda: int(time.time()),  # 创建时的初始值
        onupdate=lambda: int(time.time()),  # 每次更新时自动更新
    )
    # 更新时间 - 记忆最后一次修改或访问的时间
    # - 作用: 追踪记忆的最后活跃时间
    # - 类型: Unix时间戳(整数,秒级)
    # - 自动更新: 每次update操作时自动更新
    # - 用途: 判断记忆是否活跃,决定是否归档

    # ==================== 索引定义 ====================
    __table_args__ = (
        Index("idx_mem_qq_tier", "qq_id", "tier", "updated_at"),
        # 索引1: (qq_id, tier, updated_at) 联合索引
        # - 用途: 按层级查询用户记忆,并按更新时间排序
        # - 查询示例:
        #   SELECT * FROM memories
        #   WHERE qq_id='123' AND tier='active'
        #   ORDER BY updated_at DESC
        # - 场景: 获取用户的活跃记忆列表

        Index("idx_mem_qq_type", "qq_id", "type", "updated_at"),
        # 索引2: (qq_id, type, updated_at) 联合索引
        # - 用途: 按类型查询用户记忆,并按更新时间排序
        # - 查询示例:
        #   SELECT * FROM memories
        #   WHERE qq_id='123' AND type='preference'
        #   ORDER BY updated_at DESC
        # - 场景: 分析用户画像,如"获取用户所有偏好"
    )

class MemoryEvidence(Base):
    """记忆证据表 - 建立记忆与原始消息的多对多映射关系

    这个表的作用:
    1. 链接每条记忆与支撑它的原始消息(证据)
    2. 支持记忆的溯源和验证(查看记忆是从哪些对话中提取的)
    3. 实现记忆的可解释性和可审查性
    4. 便于评估记忆提取的质量

    关系说明:
    - 一条记忆可以有多条证据消息(如"用户喜欢运动"可能从多次对话中提取)
    - 一条消息可以支撑多条记忆(如一句话"我是程序员,喜欢跑步"可以提取两条记忆)
    - 因此这是多对多(many-to-many)关系,需要中间表

    数据增长:
    - 与记忆数成正比,每条记忆平均1-5条证据
    - 预计数据量: 记忆数 × 平均证据数
    - 清理策略: 记忆删除时级联删除证据链接

    主键策略:
    - 复合主键: (memory_id, msg_id) - 同一对关系不重复
    """

    __tablename__ = "memory_evidence"  # 数据库表名

    # ==================== 复合主键 ====================
    memory_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # 记忆ID - 指向Memory表的记忆
    # - 作用: 标识哪条记忆
    # - 类型: 整数
    # - 关联: Memory.id
    # - 主键: 与msg_id组成复合主键

    msg_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # 消息ID - 指向RawMessage表的证据消息
    # - 作用: 标识支撑这条记忆的消息
    # - 类型: 整数
    # - 关联: RawMessage.id
    # - 主键: 与memory_id组成复合主键

    # 说明:
    # 这个表没有其他字段,只有两个关联字段
    # 使用复合主键保证同一对(memory_id, msg_id)不会重复
    # 查询示例:
    # - 查找某条记忆的所有证据消息:
    #   SELECT msg_id FROM memory_evidence WHERE memory_id = 123
    # - 查找某条消息支撑了哪些记忆:
    #   SELECT memory_id FROM memory_evidence WHERE msg_id = 456


class MediaCache(Base):
    """媒体缓存表 - 存储图片/表情包的预处理结果(caption、OCR等)

    这个表的作用:
    1. 缓存图片的AI生成描述(caption),避免重复调用视觉模型
    2. 存储OCR提取的图片文字,加速图文理解
    3. 保存图片的标签(tags),用于分类和检索
    4. 作为表情包和消息图片处理的统一缓存层

    处理流程:
    1. 收到图片消息 → 计算图片哈希(media_key)
    2. 查询MediaCache是否已有记录
    3. 如果有 → 直接使用缓存结果
    4. 如果无 → 调用视觉模型处理 → 存入缓存

    数据增长:
    - 与图片数成正比,每张独特的图片一条记录
    - 预计数据量: 独特图片数
    - 清理策略: 定期清理很久未访问的缓存

    索引策略:
    - 主键: media_key (图片哈希,天然唯一)
    """

    __tablename__ = "media_cache"  # 数据库表名

    # ==================== 主键 ====================
    media_key: Mapped[str] = mapped_column(String, primary_key=True)
    # 媒体键 - 图片的唯一标识(通常是哈希值)
    # - 作用: 作为缓存的key,快速查找图片处理结果
    # - 类型: 字符串
    # - 格式: 通常是SHA256哈希值(64位十六进制字符串)
    # - 示例: "a3f2b1c5d6e7f8g9..."
    # - 唯一性: 相同图片的哈希相同,不同图片的哈希不同(概率极低碰撞)

    # ==================== 媒体信息字段 ====================
    media_type: Mapped[str] = mapped_column(String)
    # 媒体类型 - 媒体资源的类别
    # - 作用: 标识媒体的类型
    # - 类型: 字符串,可能的值:
    #   * "image" - 普通图片
    #   * "sticker" - 表情包
    #   * "photo" - 照片
    # - 用途: 区分不同类型媒体的处理策略

    caption: Mapped[str] = mapped_column(Text)
    # 图片描述 - AI生成的图片内容描述
    # - 作用: 用自然语言描述图片内容
    # - 类型: 长文本(Text)
    # - 生成方式: 调用视觉模型(如GPT-4V、Claude Vision)
    # - 示例: "一只橘色的猫坐在沙发上"、"蓝天白云下的海滩"
    # - 用途: 让LLM"看懂"图片,提供对话上下文

    tags: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # 标签 - 图片的关键词标签
    # - 作用: 用简短的关键词描述图片特征
    # - 类型: 可空长文本(通常存储逗号分隔的标签)
    # - 默认值: None
    # - 示例: "猫,宠物,可爱"、"风景,自然,海滩"
    # - 生成方式: LLM提取或视觉模型输出
    # - 用途: 图片分类、检索、表情包匹配

    ocr_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # OCR文字 - 从图片中识别出的文字内容
    # - 作用: 提取图片中的文字信息
    # - 类型: 可空长文本
    # - 默认值: None (图片无文字或OCR未启用)
    # - 提取方式: 调用OCR引擎或视觉模型的文字识别功能
    # - 示例: "今天天气真好"、"重要通知"
    # - 用途: 图文理解、表情包文字匹配、信息提取

    # ==================== 时间戳字段 ====================
    created_at: Mapped[int] = mapped_column(Integer, default=lambda: int(time.time()))
    # 创建时间 - 缓存记录的创建时间
    # - 作用: 记录首次处理该图片的时间
    # - 类型: Unix时间戳(整数,秒级)

    updated_at: Mapped[int] = mapped_column(
        Integer,
        default=lambda: int(time.time()),
        onupdate=lambda: int(time.time()),
    )
    # 更新时间 - 缓存记录的最后更新时间
    # - 作用: 追踪缓存的最后访问或修改时间
    # - 类型: Unix时间戳(整数,秒级)
    # - 用途: LRU缓存淘汰策略(删除最久未访问的缓存)


class BotRateLimit(Base):
    """机器人限流状态表 - 按场景维度控制机器人的发言频率

    这个表的作用:
    1. 防止机器人刷屏,限制回复频率
    2. 记录最后发言时间,实现冷却期控制
    3. 统计最近消息数,检测异常高频场景
    4. 分场景独立限流,避免一个群影响其他群

    限流策略:
    - 全局冷却: 机器人任何地方回复后,30秒内不再回复其他地方
    - 场景冷却: 在某个场景回复后,60秒内不再回复该场景
    - 刷屏检测: 30秒内机器人发送超过N条消息则暂停回复

    数据增长:
    - 缓慢增长,每个活跃场景一条记录
    - 预计数据量: 活跃场景数(群数+私聊数)
    - 清理策略: 定期删除长期不活跃场景的记录

    主键策略:
    - 复合主键: (scene_type, scene_id) - 唯一标识一个场景
    """

    __tablename__ = "bot_rate_limit"  # 数据库表名

    # ==================== 复合主键 - 场景标识 ====================
    scene_type: Mapped[str] = mapped_column(String, primary_key=True)
    # 场景类型 - 限流场景的类别
    # - 作用: 标识是群聊还是私聊
    # - 类型: 字符串,取值:
    #   * "group" - 群聊
    #   * "private" - 私聊
    # - 主键: 与scene_id组成复合主键

    scene_id: Mapped[str] = mapped_column(String, primary_key=True)
    # 场景标识 - 具体的群号或QQ号
    # - 作用: 定位具体的场景
    # - 类型: 字符串(群号或QQ号)
    # - 主键: 与scene_type组成复合主键

    # ==================== 限流状态字段 ====================
    last_sent_ts: Mapped[int] = mapped_column(Integer, default=0)
    # 最后发送时间 - 机器人在该场景最后一次发消息的时间
    # - 作用: 记录最后发言时间,用于计算冷却期
    # - 类型: Unix时间戳(整数,秒级)
    # - 默认值: 0 (尚未发过消息)
    # - 更新时机: 每次机器人发送消息后更新
    # - 用途: 判断冷却期是否结束

    cooldown_until_ts: Mapped[int] = mapped_column(Integer, default=0)
    # 冷却截止时间 - 该场景禁止发言的截止时间
    # - 作用: 设置冷却期的结束时间点
    # - 类型: Unix时间戳(整数,秒级)
    # - 默认值: 0 (无冷却)
    # - 计算方式: last_sent_ts + cooldown_seconds
    # - 用途: 当前时间 < cooldown_until_ts 时禁止回复

    recent_bot_msg_count: Mapped[int] = mapped_column(Integer, default=0)
    # 最近消息计数 - 统计窗口内机器人发送的消息数
    # - 作用: 检测机器人是否发送过多消息
    # - 类型: 整数
    # - 默认值: 0
    # - 计数窗口: 默认30秒
    # - 重置时机: 超过窗口时间后归零
    # - 用途: 达到阈值(如12条)时触发刷屏保护


class Sticker(Base):
    """表情包注册表 - 正式表情包库的权威清单

    这个表的作用:
    1. 存储所有可用的表情包信息(文件路径、指纹、标签等)
    2. 管理表情包的启用/禁用状态
    3. 提供表情包检索和匹配的索引
    4. 记录表情包的来源和统计信息

    表情包生命周期:
    1. 本地表情包 → 启动时扫描导入
    2. 偷取表情包 → 从StickerCandidate晋升而来
    3. 使用中 → 记录使用统计(StickerUsage表)
    4. 禁用/删除 → 设置is_banned或删除记录

    数据增长:
    - 中等增长,取决于本地表情包和偷取速度
    - 预计数据量: 几百至几千张
    - 清理策略: 删除从未使用的、禁用的表情包

    索引策略:
    - 主键: sticker_id (表情包哈希)
    - 索引1: fingerprint - 用于去重和相似度匹配
    - 索引2: is_enabled - 快速筛选可用表情包
    """

    __tablename__ = "stickers"  # 数据库表名

    # ==================== 主键与标识 ====================
    sticker_id: Mapped[str] = mapped_column(String, primary_key=True)
    # 表情包ID - 表情包的唯一标识
    # - 作用: 唯一标识每个表情包
    # - 类型: 字符串(通常是文件哈希)
    # - 生成方式: 文件内容的SHA256哈希
    # - 主键: 保证表情包不重复

    pack: Mapped[str] = mapped_column(String)
    # 表情包分组 - 表情包所属的集合或来源
    # - 作用: 将表情包分类管理
    # - 类型: 字符串
    # - 示例: "本地表情包"、"群123偷取"、"默认集"
    # - 用途: 按分组管理和查询表情包

    file_path: Mapped[str] = mapped_column(Text)
    # 文件路径 - 表情包图片文件的存储路径
    # - 作用: 指向表情包的实际文件位置
    # - 类型: 长文本(路径可能很长)
    # - 示例: "assets/stickers/pack1/cat001.png"
    # - 用途: 发送表情包时读取文件

    # ==================== 指纹与识别 ====================
    file_sha256: Mapped[str] = mapped_column(String)
    # 文件SHA256 - 文件内容的SHA256哈希值
    # - 作用: 精确匹配,用于完全相同的图片去重
    # - 类型: 字符串(64位十六进制)
    # - 特点: 即使文件名不同,内容相同则哈希相同
    # - 用途: 避免重复导入相同表情包

    phash: Mapped[str] = mapped_column(String)
    # 感知哈希(PHash) - 图片的感知哈希值
    # - 作用: 识别相似图片(即使有轻微变化)
    # - 类型: 字符串(感知哈希值)
    # - 特点: 相似图片的phash也相似(汉明距离小)
    # - 用途: 识别表情包的变体(如不同尺寸、不同压缩率)

    ocr_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # OCR文字 - 表情包图片中的文字内容
    # - 作用: 提取表情包上的文字
    # - 类型: 可空长文本
    # - 示例: "无奈"、"我裂开了"、"嘿嘿"
    # - 用途: 根据文字匹配表情包

    fingerprint: Mapped[str] = mapped_column(String)
    # 综合指纹 - phash + 归一化OCR文字的组合
    # - 作用: 唯一但宽容的表情包识别
    # - 类型: 字符串
    # - 生成方式: phash + "|" + normalize(ocr_text)
    # - 特点: 相同意思的表情包指纹相似
    # - 用途: 表情包去重和偷取判断(候选池用这个字段匹配)

    # ==================== 元数据字段 ====================
    name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    # 表情包名称 - 可选的表情包名字
    # - 作用: 给表情包起一个便于识别的名字
    # - 类型: 可空字符串
    # - 示例: "猫猫无奈"、"狗狗开心"
    # - 用途: 管理界面展示、日志输出

    tags: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # 标签 - 表情包的关键词标签
    # - 作用: 描述表情包的视觉特征
    # - 类型: 可空长文本(逗号分隔)
    # - 示例: "猫,宠物,可爱"、"无奈,崩溃,裂开"
    # - 生成方式: LLM分析图片生成
    # - 用途: 辅助表情包检索

    intents: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # 意图/情绪标签 - 表情包适用的对话意图
    # - 作用: 标记表情包适合在什么场景使用
    # - 类型: 可空长文本(逗号分隔)
    # - 示例: "无奈,尴尬"、"开心,兴奋"、"调侃,嘲讽"
    # - 生成方式: LLM分析表情包的表达意图
    # - 用途: 根据对话情绪匹配合适的表情包(核心功能)

    style: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    # 风格 - 表情包的艺术风格
    # - 作用: 标识表情包的视觉风格
    # - 类型: 可空字符串
    # - 示例: "真人照片"、"卡通"、"表情包模板"
    # - 用途: 按风格筛选表情包

    # ==================== 状态管理 ====================
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    # 启用状态 - 表情包是否可用
    # - 作用: 控制表情包是否参与匹配和发送
    # - 类型: 布尔值
    # - 默认值: True (启用)
    # - False: 表情包被禁用,不会被发送
    # - 用途: 临时禁用某些表情包

    is_banned: Mapped[bool] = mapped_column(Boolean, default=False)
    # 封禁状态 - 表情包是否被永久封禁
    # - 作用: 标记不当表情包
    # - 类型: 布尔值
    # - 默认值: False (未封禁)
    # - True: 表情包违规,永久不可用
    # - 用途: 管理不当内容

    ban_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # 封禁原因 - 表情包被封禁的原因说明
    # - 作用: 记录为什么封禁
    # - 类型: 可空长文本
    # - 示例: "内容不当"、"版权问题"
    # - 用途: 审计和管理

    # ==================== 来源信息 ====================
    source_scene_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    # 来源场景ID - 表情包是从哪个场景偷取的
    # - 作用: 记录表情包的来源
    # - 类型: 可空字符串(场景ID)
    # - 默认值: None (本地表情包)
    # - 示例: 群号"123456"
    # - 用途: 溯源、统计

    source_qq_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    # 来源用户QQ - 表情包是谁发的
    # - 作用: 记录最初发送这个表情包的用户
    # - 类型: 可空字符串(QQ号)
    # - 默认值: None (本地表情包)
    # - 用途: 溯源、统计

    # ==================== 时间戳 ====================
    created_at: Mapped[int] = mapped_column(Integer, default=lambda: int(time.time()))
    # 创建时间 - 表情包入库时间
    # - 类型: Unix时间戳(整数,秒级)

    updated_at: Mapped[int] = mapped_column(
        Integer,
        default=lambda: int(time.time()),
        onupdate=lambda: int(time.time()),
    )
    # 更新时间 - 表情包最后修改时间
    # - 类型: Unix时间戳(整数,秒级)

    # ==================== 索引定义 ====================
    __table_args__ = (
        Index("idx_stk_fp", "fingerprint"),
        # 索引1: fingerprint 单列索引
        # - 用途: 快速查找是否已有相同表情包(去重)

        Index("idx_stk_enabled", "is_enabled"),
        # 索引2: is_enabled 单列索引
        # - 用途: 快速筛选所有启用的表情包
        # - 查询示例: SELECT * FROM stickers WHERE is_enabled=True
    )

class StickerCandidate(Base):
    """表情包候选池 - 临时存放潜在表情包,重复出现后晋升

    这个表的作用:
    1. 作为表情包"试用期"的临时存储
    2. 记录每张候选图片的出现次数和首次/最后出现时间
    3. 达到晋升阈值后,将候选图片正式加入Sticker表
    4. 过滤掉一次性使用的图片,只收录真正流行的表情包

    晋升机制:
    - 首次出现: 加入候选池,seen_count=1
    - 再次出现: seen_count+1,更新last_seen_ts
    - 达到阈值(默认3次): 晋升为正式表情包,从候选池删除
    - 长期未晋升: 定期清理(如30天未再见)

    数据增长:
    - 快速流动,候选不断进出
    - 预计数据量: 几十到几百条(未晋升的候选)
    - 清理策略: 晋升后删除,长期未晋升的也删除

    索引策略:
    - 主键: candidate_id (自增)
    - 索引1: fingerprint - 快速查找是否已有相同候选
    - 索引2: status - 按状态筛选候选
    """

    __tablename__ = "sticker_candidates"  # 数据库表名

    # ==================== 主键 ====================
    candidate_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 候选ID - 自增主键
    # - 作用: 唯一标识每个候选记录
    # - 类型: 自增整数

    # ==================== 指纹与识别 ====================
    fingerprint: Mapped[str] = mapped_column(String)
    # 综合指纹 - 候选图片的唯一标识
    # - 作用: 用于匹配是否是同一张表情包
    # - 类型: 字符串
    # - 生成方式: phash + "|" + normalize(ocr_text)
    # - 用途: 检查候选是否已存在于候选池或正式库

    phash: Mapped[str] = mapped_column(String)
    # 感知哈希 - 图片的感知哈希值
    # - 作用: 识别相似图片
    # - 类型: 字符串(感知哈希值)
    # - 用途: 计算fingerprint的一部分

    ocr_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # OCR文字 - 候选图片中的文字
    # - 作用: 提取图片文字
    # - 类型: 可空长文本
    # - 默认值: None (图片无文字)
    # - 用途: 计算fingerprint、辅助匹配

    sha256_sample: Mapped[str] = mapped_column(String)
    # 样本SHA256 - 首次见到的图片文件哈希
    # - 作用: 保存首次出现的原始文件哈希
    # - 类型: 字符串(SHA256哈希)
    # - 用途: 晋升时可以验证是否是同一文件

    sample_file_path: Mapped[str] = mapped_column(Text)
    # 样本文件路径 - 首次见到的图片文件存放位置
    # - 作用: 保存首次出现的图片文件路径
    # - 类型: 长文本(文件路径)
    # - 示例: "assets/media/tmp/candidate_123456.png"
    # - 用途: 晋升时移动文件到正式表情包目录

    # ==================== 场景与来源 ====================
    scene_id: Mapped[str] = mapped_column(String)
    # 场景ID - 首次出现的场景
    # - 作用: 记录候选图片最初在哪个场景出现
    # - 类型: 字符串(场景ID)
    # - 示例: 群号"123456"
    # - 用途: 溯源、统计

    # ==================== 统计字段 ====================
    first_seen_ts: Mapped[int] = mapped_column(Integer)
    # 首次出现时间 - 候选图片第一次被发现的时间
    # - 作用: 记录候选的诞生时间
    # - 类型: Unix时间戳(整数,秒级)
    # - 用途: 计算候选的"年龄",清理太老的候选

    last_seen_ts: Mapped[int] = mapped_column(Integer)
    # 最后出现时间 - 候选图片最近一次出现的时间
    # - 作用: 追踪候选的活跃度
    # - 类型: Unix时间戳(整数,秒级)
    # - 更新时机: 每次再次见到相同候选时更新
    # - 用途: 清理长期未再见的候选(如30天未出现)

    seen_count: Mapped[int] = mapped_column(Integer, default=1)
    # 出现次数 - 候选图片被看到的次数
    # - 作用: 统计候选的流行度
    # - 类型: 整数
    # - 默认值: 1 (首次加入时)
    # - 更新时机: 每次再次见到时+1
    # - 晋升条件: 达到阈值(默认3次)时晋升为正式表情包

    status: Mapped[str] = mapped_column(String)
    # 状态 - 候选的当前状态
    # - 作用: 标记候选的处理状态
    # - 类型: 字符串,可能的值:
    #   * "pending" - 待晋升,还在积累出现次数
    #   * "processing" - 正在处理(晋升中)
    #   * "promoted" - 已晋升(理论上应该被删除了)
    #   * "rejected" - 被拒绝(不适合作为表情包)
    # - 默认值: "pending"

    source_qq_ids: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # 来源用户QQ列表 - 所有发送过这个候选图片的用户
    # - 作用: 记录谁发送过这张图片
    # - 类型: 可空长文本(存储QQ号列表,逗号分隔或JSON格式)
    # - 默认值: None
    # - 示例: "123,456,789"
    # - 更新时机: 每次见到时添加当前发送者的QQ号(去重)
    # - 用途: 溯源、统计谁贡献了表情包

    # ==================== 索引定义 ====================
    __table_args__ = (
        Index("idx_sc_fp", "fingerprint"),
        # 索引1: fingerprint 单列索引
        # - 用途: 快速查找是否已有相同候选
        # - 查询示例: SELECT * FROM sticker_candidates WHERE fingerprint='xxx'

        Index("idx_sc_status", "status"),
        # 索引2: status 单列索引
        # - 用途: 按状态筛选候选
        # - 查询示例: SELECT * FROM sticker_candidates WHERE status='pending'
    )


class StickerUsage(Base):
    """表情包使用记录表 - 记录表情包的每次发送,用于统计和推荐

    这个表的作用:
    1. 记录每次表情包的发送事件(谁、什么时候、在哪里)
    2. 提供使用统计数据(使用频率、热门表情包)
    3. 支持个性化推荐(用户常用表情、场景常用表情)
    4. 用于表情包质量评估(从未使用的表情包可能被清理)

    数据增长:
    - 快速增长,每次发送表情包一条记录
    - 预计数据量: 随机器人活跃度增长,可能达到数万至数十万级
    - 清理策略: 定期删除很久以前的记录(保留最近3个月)

    主键策略:
    - 自增主键: id - 每次使用独立记录
    """

    __tablename__ = "sticker_usage"  # 数据库表名

    # ==================== 主键 ====================
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 使用记录ID - 自增主键
    # - 作用: 唯一标识每次使用记录
    # - 类型: 自增整数

    # ==================== 关联字段 ====================
    sticker_id: Mapped[str] = mapped_column(String)
    # 表情包ID - 使用的是哪个表情包
    # - 作用: 关联到Sticker表
    # - 类型: 字符串
    # - 关联: Sticker.sticker_id
    # - 用途: 统计每个表情包的使用次数

    # ==================== 场景与用户 ====================
    scene_type: Mapped[str] = mapped_column(String)
    # 场景类型 - 表情包在哪类场景使用
    # - 作用: 标识使用场景的类别
    # - 类型: 字符串,取值:
    #   * "group" - 群聊
    #   * "private" - 私聊

    scene_id: Mapped[str] = mapped_column(String)
    # 场景ID - 具体在哪个场景使用
    # - 作用: 定位具体的群或私聊
    # - 类型: 字符串(群号或QQ号)
    # - 用途: 统计某个场景的表情包使用习惯

    qq_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    # 用户QQ号 - 是机器人自己使用还是用户使用
    # - 作用: 标识使用者
    # - 类型: 可空字符串(QQ号)
    # - 默认值: None (通常是机器人发送,记录为None或机器人QQ)
    # - 用途: 区分机器人使用和用户使用(未来扩展)

    # ==================== 时间戳 ====================
    used_at: Mapped[int] = mapped_column(Integer, default=lambda: int(time.time()))
    # 使用时间 - 表情包被发送的时间
    # - 作用: 记录使用时间点
    # - 类型: Unix时间戳(整数,秒级)
    # - 默认值: 当前时间
    # - 用途: 时间序列分析、冷却期判断、数据清理


class IndexJob(Base):
    """索引任务表 - 后台异步处理的向量化和OCR任务队列

    这个表的作用:
    1. 作为向量化任务的队列,避免阻塞主流程
    2. 支持任务的异步处理和失败重试
    3. 提供任务状态追踪和进度监控
    4. 实现数据库作为事实源,向量库作为衍生数据的架构

    任务类型(item_type):
    - "msg_chunk": 消息向量化(RawMessage → Qdrant)
    - "ocr": 图片OCR处理(图片 → MediaCache)
    - "memory": 记忆向量化(Memory → Qdrant)
    - "sticker": 表情包标签生成(Sticker → tags/intents)

    任务生命周期:
    1. 创建任务 → status="pending"
    2. Worker认领 → status="processing"
    3. 处理成功 → status="completed"
    4. 处理失败 → 增加retry_count,设置next_retry_ts,回到"pending"
    5. 重试次数用尽 → status="failed"

    数据增长:
    - 快速增长,每条消息、每张图片都可能创建任务
    - 预计数据量: 随消息量增长,可能达到数十万级
    - 清理策略: 定期删除completed和failed任务(保留最近7天)

    索引策略:
    - 主键: job_id (自增)
    - 复合索引: (status, next_retry_ts) - Worker认领任务
    """

    __tablename__ = "index_jobs"  # 数据库表名

    # ==================== 主键 ====================
    job_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 任务ID - 自增主键
    # - 作用: 唯一标识每个任务
    # - 类型: 自增整数

    # ==================== 任务标识 ====================
    item_type: Mapped[str] = mapped_column(String)
    # 任务类型 - 标识这是什么类型的任务
    # - 作用: 区分不同类型的处理任务
    # - 类型: 字符串,可能的值:
    #   * "msg_chunk" - 消息向量化任务
    #   * "ocr" - 图片OCR处理任务
    #   * "memory" - 记忆向量化任务
    #   * "sticker" - 表情包标签生成任务
    # - 用途: Worker根据类型选择处理器

    ref_id: Mapped[str] = mapped_column(String)
    # 引用ID - 任务关联的数据记录ID
    # - 作用: 指向需要处理的数据记录
    # - 类型: 字符串(ID,转为字符串存储)
    # - 示例:
    #   * item_type="msg_chunk"时,ref_id是RawMessage.id
    #   * item_type="ocr"时,ref_id是MediaCache.media_key
    #   * item_type="memory"时,ref_id是Memory.id
    # - 用途: 定位要处理的数据

    payload_json: Mapped[str] = mapped_column(Text)
    # 任务载荷JSON - 任务的详细数据
    # - 作用: 存储任务执行需要的所有数据
    # - 类型: 长文本(JSON格式字符串)
    # - 内容示例:
    #   {
    #     "message_content": "今天天气真好",
    #     "timestamp": 1234567890,
    #     "scene_id": "123"
    #   }
    # - 用途: Worker读取payload执行任务,避免再次查询数据库

    # ==================== 任务状态 ====================
    status: Mapped[str] = mapped_column(String)
    # 任务状态 - 任务的当前处理状态
    # - 作用: 追踪任务进度
    # - 类型: 字符串,可能的值:
    #   * "pending" - 待处理,等待Worker认领
    #   * "processing" - 处理中,已被Worker认领
    #   * "completed" - 已完成,处理成功
    #   * "failed" - 失败,重试次数用尽
    # - 默认值: "pending"
    # - Worker查询: WHERE status='pending' AND next_retry_ts <= now()

    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    # 重试次数 - 任务已重试的次数
    # - 作用: 记录任务失败后重试了几次
    # - 类型: 整数
    # - 默认值: 0 (首次创建)
    # - 最大重试: 通常设置为3-5次
    # - 用途: 超过最大重试次数则标记为failed

    next_retry_ts: Mapped[int] = mapped_column(Integer, default=0)
    # 下次重试时间 - 任务可以被重新处理的最早时间
    # - 作用: 实现失败后的延迟重试(指数退避)
    # - 类型: Unix时间戳(整数,秒级)
    # - 默认值: 0 (立即可处理)
    # - 计算方式: current_time + (2 ** retry_count) * base_delay
    #   例如: 首次失败等30秒,第二次等60秒,第三次等120秒
    # - 用途: Worker只认领 next_retry_ts <= now() 的任务

    # ==================== 时间戳 ====================
    created_at: Mapped[int] = mapped_column(Integer, default=lambda: int(time.time()))
    # 创建时间 - 任务创建的时间
    # - 作用: 记录任务何时加入队列
    # - 类型: Unix时间戳(整数,秒级)
    # - 用途: 监控任务积压、计算处理延迟

    updated_at: Mapped[int] = mapped_column(
        Integer,
        default=lambda: int(time.time()),
        onupdate=lambda: int(time.time()),
    )
    # 更新时间 - 任务最后修改的时间
    # - 作用: 追踪任务状态变更时间
    # - 类型: Unix时间戳(整数,秒级)
    # - 自动更新: 每次update时更新
    # - 用途: 监控任务处理时长、发现卡住的任务

    # ==================== 索引定义 ====================
    __table_args__ = (
        Index("idx_jobs_status", "status", "next_retry_ts"),
        # 索引: (status, next_retry_ts) 联合索引
        # - 用途: Worker认领任务的核心查询
        # - 查询示例:
        #   SELECT * FROM index_jobs
        #   WHERE status='pending' AND next_retry_ts <= {now}
        #   ORDER BY created_at ASC
        #   LIMIT 10
        # - 性能: 覆盖索引,高效支持任务认领
    )
