"""记忆管理模块 - 用户长期记忆的抽取、存储、选择和管理

这个模块的作用:
1. 实现三层记忆架构(active/archive/core)的管理
2. 从用户对话中智能抽取长期记忆
3. 为对话选择相关的记忆上下文
4. 管理记忆的生命周期(创建、更新、归档、浓缩)
5. 处理记忆去重和冲突检测

记忆系统原理(新手必读):
- 问题: LLM无法记住历史对话,每次对话都是"失忆"
- 解决: 从对话中提取稳定信息(事实、偏好、习惯等)存入数据库
- 流程: 用户发消息 → 积累到阈值 → LLM抽取记忆 → 存入数据库 → 下次对话时注入
- 好处: 实现长期对话连贯性,机器人"记得"用户的信息

三层记忆架构:
1. active(活跃层): 最近抽取的记忆,优先使用
   - 特点: 数量多、更新频繁、时效性强
   - 生存期: 默认30天(yuying_memory_active_ttl_days)
   - 流转: 长期不用→archive,高质量→core
2. archive(归档层): 较久未用的记忆,降低优先级
   - 特点: 数量适中、访问较少、备用存储
   - 生存期: 默认180天
   - 流转: 再次激活→active,浓缩→core,过期→删除
3. core(核心层): 经过浓缩的核心记忆,永久保留
   - 特点: 数量少、高度凝练、最重要
   - 生存期: 永久
   - 来源: 从active/archive浓缩而来

记忆抽取触发机制:
1. 阈值触发: 用户有效发言达到next_memory_at(默认50条)
2. 空闲期抽取: 用户2分钟无新消息时执行抽取(可取消)
3. 手动触发: 管理员命令触发(未实现)

记忆类型(type字段):
- fact: 客观事实(如"用户是程序员")
- preference: 主观偏好(如"喜欢吃辣")
- habit: 行为习惯(如"晚上11点睡觉")
- experience: 重要经历(如"上周去了北京")
- relationship: 人际关系(如"妹妹叫小红")
- goal: 目标计划(如"今年要学Python")
- constraint: 约束条件(如"对海鲜过敏")

可见性控制(visibility字段):
- global: 全局可见,所有场景都可注入
- group_only: 仅群聊可见,且仅在特定群(scope_scene_id)
- private_only: 仅私聊可见

使用方式:
```python
from .memory.memory_manager import MemoryManager

# 1. 用户发消息后检查是否需要标记为待抽取
await MemoryManager.mark_pending_if_needed(qq_id="123456")
# 如果达到阈值,会标记profile.pending_memory=True

# 2. 调度空闲期抽取(2分钟无新消息时触发)
MemoryManager.schedule_idle_extract(
    qq_id="123456",
    scene_type="group",
    scene_id="789"
)
# 2分钟后自动触发抽取,期间有新消息会取消

# 3. 为对话选择相关记忆
memories = await MemoryManager.select_for_context(
    qq_id="123456",
    scene_type="group",
    scene_id="789",
    query="今天天气怎么样"
)
# 返回: [Memory(content="用户住在北京"), ...]

# 4. 手动写入记忆(通常由LLM抽取)
await MemoryManager.upsert_memories(
    qq_id="123456",
    memories_data=[
        {
            "type": "fact",
            "content": "用户是Python程序员",
            "confidence": 0.9,
            "visibility": "global",
            "evidence_msg_ids": [101, 102, 103]
        }
    ]
)
```
"""

from __future__ import annotations

import asyncio  # Python异步编程标准库
import json  # JSON编解码
import re  # 正则表达式
from difflib import SequenceMatcher  # 字符串相似度计算(编辑距离)
from typing import Dict, List, Optional  # 类型提示

from nonebot import logger  # NoneBot日志记录器

# 导入项目模块
from ..config import plugin_config  # 插件配置
from ..llm.client import main_llm  # 主LLM客户端
from ..storage.models import IndexJob  # 索引任务模型
from ..storage.models import Memory  # 记忆模型
from ..storage.repositories.memory_repo import MemoryRepository  # 记忆仓库
from ..storage.repositories.profile_repo import ProfileRepository  # 用户档案仓库
from ..storage.repositories.raw_repo import RawRepository  # 原始消息仓库
from ..storage.repositories.index_jobs_repo import IndexJobRepository  # 索引任务仓库
from ..storage.db_writer import db_writer  # 数据库写入队列
from ..storage.write_jobs import AddIndexJobJob, AsyncCallableJob  # 写入任务


class MemoryManager:
    """记忆管理器 - 用户长期记忆的全生命周期管理

    这个类的作用:
    - 管理记忆的抽取、存储、更新、选择
    - 实现三层记忆架构的流转逻辑
    - 处理记忆的去重、冲突检测、相关性评分
    - 提供空闲期抽取的调度机制

    设计模式:
    - 静态方法类: 所有方法都是静态的,无需实例化
    - 好处: 作为全局工具类使用,避免状态管理

    核心流程:
    1. 用户发消息 → mark_pending_if_needed()检查阈值
    2. 达到阈值 → schedule_idle_extract()调度抽取
    3. 空闲2分钟 → maybe_extract_on_idle()执行抽取
    4. LLM抽取 → _extract_memories_with_llm()调用主模型
    5. 写入数据库 → upsert_memories()去重并存储
    6. 对话时注入 → select_for_context()选择相关记忆

    类属性:
    - _idle_tasks: 空闲期抽取的异步任务字典{qq_id: Task}
    """

    # ==================== 类属性 ====================

    _idle_tasks: Dict[str, asyncio.Task] = {}
    # 空闲期抽取任务字典 - 每个用户一个异步任务
    # - 作用: 存储调度的延迟抽取任务,用于取消和管理
    # - key: 用户QQ号
    # - value: asyncio.Task对象
    # - 用途: 用户有新消息时取消旧任务,避免重复抽取

    @staticmethod
    async def mark_pending_if_needed(qq_id: str) -> None:
        """在满足阈值时将用户标记为"待抽取记忆"状态

        这个方法的作用:
        - 检查用户的有效发言数是否达到抽取阈值
        - 如果达到且未标记,则标记profile.pending_memory=True
        - 标记后会触发空闲期抽取机制

        触发条件:
        - effective_count >= next_memory_at (如50条有效消息)
        - pending_memory == False (避免重复标记)

        Args:
            qq_id: 用户QQ号
                - 类型: 字符串
                - 用途: 查询和更新用户档案

        Side Effects:
            - 更新UserProfile表的pending_memory字段为True
            - 输出日志: "用户 xxx 已标记为待记忆抽取。"

        Example:
            >>> # 用户每次发消息后调用
            >>> await MemoryManager.mark_pending_if_needed("123456")
            # 如果是第50条有效消息,会标记为待抽取
        """

        # ==================== 步骤1: 获取用户档案 ====================

        # await ProfileRepository.get_or_create(qq_id): 查询或创建用户档案
        # - 如果用户不存在,会自动创建默认档案
        # - 返回: UserProfile对象
        profile = await ProfileRepository.get_or_create(qq_id)

        # ==================== 步骤2: 检查是否需要标记 ====================

        # profile.effective_count >= profile.next_memory_at: 达到抽取阈值
        # - effective_count: 有效发言计数器
        # - next_memory_at: 下次抽取的目标值(默认50)
        # not profile.pending_memory: 尚未标记为待抽取
        # - 避免重复标记
        if profile.effective_count >= profile.next_memory_at and not profile.pending_memory:
            # ==================== 步骤3: 标记为待抽取 ====================

            # await db_writer.submit_and_wait(): 提交写入任务并等待完成
            # - submit_and_wait: 阻塞等待任务完成,确保状态更新成功
            # - 参数: 写入任务对象, priority=5(中等优先级)
            await db_writer.submit_and_wait(
                # AsyncCallableJob: 异步可调用任务
                # - 第一个参数: 要调用的异步函数
                # - args: 位置参数元组
                # ProfileRepository.update_memory_status(qq_id, True): 更新pending_memory=True
                AsyncCallableJob(ProfileRepository.update_memory_status, args=(qq_id, True)),
                priority=5,  # 优先级5(中等)
            )

            # ==================== 步骤4: 输出日志 ====================

            # logger.info(): 记录信息级别日志
            logger.info(f"用户 {qq_id} 已标记为待记忆抽取。")

    @staticmethod
    def schedule_idle_extract(qq_id: str, scene_type: str, scene_id: str) -> None:
        """调度空闲期记忆抽取(2分钟无新消息时触发)

        这个方法的作用:
        - 创建一个延迟抽取任务,在N秒后执行
        - 如果用户在延迟期间发新消息,会取消旧任务并重新调度
        - 实现"空闲期才抽取"的策略,避免对话中断

        为什么需要空闲期抽取?
        - 立即抽取: 会阻塞对话,用户体验差
        - 空闲期抽取: 用户停止发消息后再抽取,不影响对话流畅性
        - 可取消: 用户继续发消息会取消抽取,等下次空闲

        工作原理:
        1. 创建异步任务,sleep N秒后执行抽取
        2. 存入_idle_tasks字典
        3. 用户有新消息时,取消旧任务,创建新任务
        4. N秒后无新消息,执行抽取

        Args:
            qq_id: 用户QQ号
                - 类型: 字符串
                - 用途: 标识用户,作为任务字典的key
            scene_type: 场景类型
                - 类型: 字符串
                - 取值: "group" 或 "private"
                - 用途: 传递给抽取任务
            scene_id: 场景标识
                - 类型: 字符串
                - 内容: 群号或对方QQ号
                - 用途: 传递给抽取任务

        Side Effects:
            - 取消该用户的旧抽取任务(如果存在)
            - 创建新的延迟抽取任务
            - 任务存入_idle_tasks字典

        配置项:
            - yuying_memory_idle_seconds: 空闲延迟时间(默认120秒)

        Example:
            >>> # 用户每次发消息后调用
            >>> MemoryManager.schedule_idle_extract(
            ...     qq_id="123456",
            ...     scene_type="group",
            ...     scene_id="789"
            ... )
            # 启动2分钟倒计时,期间有新消息会重置倒计时
        """

        # ==================== 步骤1: 取消旧任务(如果存在) ====================

        # MemoryManager._idle_tasks.get(qq_id): 获取用户的旧抽取任务
        # - 返回: Task对象或None
        old = MemoryManager._idle_tasks.get(qq_id)

        # old: 存在旧任务
        # not old.done(): 旧任务尚未完成(还在sleep中)
        if old and not old.done():
            # old.cancel(): 取消异步任务
            # - 效果: 任务内的sleep会抛出CancelledError
            # - 目的: 避免旧任务触发,因为用户有新消息了
            old.cancel()

        # ==================== 步骤2: 定义延迟抽取任务 ====================

        async def _task() -> None:
            """空闲期抽取任务的实际执行函数

            这个内部函数的作用:
            - 延迟N秒(默认120秒)
            - 延迟结束后执行抽取
            - 被取消时静默退出

            异常处理:
            - CancelledError: 任务被取消(用户有新消息),静默退出
            - 其他异常: 记录错误日志,避免崩溃
            """

            try:
                # ==================== 延迟等待 ====================

                # await asyncio.sleep(): 异步睡眠
                # - 参数: 秒数(从配置读取)
                # int(plugin_config.yuying_memory_idle_seconds): 空闲延迟时间
                # - 默认值: 120秒(2分钟)
                # - 期间如果任务被cancel(),会抛出CancelledError
                await asyncio.sleep(int(plugin_config.yuying_memory_idle_seconds))

                # ==================== 执行抽取 ====================

                # await MemoryManager.maybe_extract_on_idle(): 执行空闲期抽取
                # - 参数: 用户和场景信息
                await MemoryManager.maybe_extract_on_idle(qq_id, scene_type, scene_id)

            except asyncio.CancelledError:
                # 任务被取消(用户有新消息)
                # return: 静默退出,不抛异常
                # 原因: 取消是预期行为,不是错误
                return

            except Exception as exc:
                # 其他异常: 抽取过程中出错
                # logger.error(): 记录错误日志
                logger.error(f"空闲期记忆抽取执行失败:{exc}")

        # ==================== 步骤3: 创建并存储异步任务 ====================

        # asyncio.create_task(_task()): 创建异步任务
        # - 参数: 协程对象
        # - 返回: Task对象
        # - 效果: 任务立即开始运行(进入sleep状态)
        # MemoryManager._idle_tasks[qq_id] = ...: 存入任务字典
        # - 目的: 下次用户发消息时可以取消这个任务
        MemoryManager._idle_tasks[qq_id] = asyncio.create_task(_task())

    @staticmethod
    async def maybe_extract_on_idle(qq_id: str, scene_type: str, scene_id: str) -> None:
        """在空闲时尝试执行记忆抽取(空闲期延迟结束后调用)

        这个方法的作用:
        - 检查用户是否有pending_memory标记
        - 如果有,则执行完整的记忆抽取流程
        - 抽取完成后更新档案状态和检查点

        抽取流程:
        1. 检查pending_memory标记
        2. 查询用户最近的消息(从last_memory_msg_id开始)
        3. 调用LLM抽取记忆
        4. 写入数据库
        5. 更新last_memory_msg_id
        6. 重置pending_memory标记
        7. 前移next_memory_at检查点

        Args:
            qq_id: 用户QQ号
                - 类型: 字符串
            scene_type: 场景类型
                - 类型: 字符串
                - 取值: "group" 或 "private"
                - 用途: 传递给LLM作为上下文
            scene_id: 场景标识
                - 类型: 字符串
                - 用途: 传递给LLM作为上下文

        Side Effects:
            - 调用LLM抽取记忆
            - 写入Memory表
            - 创建IndexJob任务(向量化)
            - 更新UserProfile表(last_memory_msg_id, pending_memory, next_memory_at)

        配置项:
            - yuying_memory_overlap_messages: 重叠消息数(默认5)
            - yuying_memory_extract_max_messages: 最大抽取消息数(默认100)

        Example:
            >>> # 由schedule_idle_extract()的延迟任务调用
            >>> await MemoryManager.maybe_extract_on_idle(
            ...     qq_id="123456",
            ...     scene_type="group",
            ...     scene_id="789"
            ... )
            # 执行完整的记忆抽取流程
        """

        # ==================== 步骤1: 检查是否需要抽取 ====================

        # await ProfileRepository.get_or_create(qq_id): 获取用户档案
        profile = await ProfileRepository.get_or_create(qq_id)

        # not profile.pending_memory: 未标记为待抽取
        # - 可能: 用户发言数还没达到阈值
        # - 或: 已经被其他任务抽取过了
        if not profile.pending_memory:
            return  # 无需抽取,直接返回

        # ==================== 步骤2: 读取抽取配置 ====================

        # int(plugin_config.yuying_memory_overlap_messages): 重叠消息数
        # - 作用: 与上次抽取的消息有overlap条重叠,保证连续性
        # - 默认值: 5条
        # - 例如: 上次抽取到msg_id=100,这次从96开始(100-5+1)
        overlap = int(plugin_config.yuying_memory_overlap_messages)

        # int(plugin_config.yuying_memory_extract_max_messages): 最大抽取消息数
        # - 作用: 限制一次抽取的消息量,避免LLM超限
        # - 默认值: 100条
        limit = int(plugin_config.yuying_memory_extract_max_messages)

        # int(profile.last_memory_msg_id or 0): 上次抽取到的消息ID
        # - 作用: 从这个ID之后开始查询(断点续传)
        # - or 0: 如果是首次抽取,从0开始
        after_id = int(profile.last_memory_msg_id or 0)

        # ==================== 步骤3: 查询用户最近的消息 ====================

        # await RawRepository.get_user_messages_since(): 查询用户消息
        # 参数:
        # - qq_id: 用户QQ号
        # - after_msg_id: 从这个ID之后开始查询
        # - overlap: 向前重叠N条(保证连续性)
        # - limit: 最多查询N条
        # 返回: RawMessage对象列表
        messages = await RawRepository.get_user_messages_since(
            qq_id=qq_id,
            after_msg_id=after_id,
            overlap=overlap,
            limit=limit,
        )

        # ==================== 步骤4: 如果没有新消息,更新状态并返回 ====================

        # not messages: 查询结果为空(没有新消息)
        # - 可能: 上次抽取后用户没发新消息
        # - 或: 用户发的都是无效消息(纯表情、空格等)
        if not messages:
            # 重置pending_memory标记为False
            await db_writer.submit_and_wait(
                AsyncCallableJob(ProfileRepository.update_memory_status, args=(qq_id, False)),
                priority=5,
            )
            # 前移next_memory_at检查点,避免下次重复触发
            # bump_next_memory_checkpoint(): 将next_memory_at增加一定步长(如+50)
            await db_writer.submit_and_wait(
                AsyncCallableJob(ProfileRepository.bump_next_memory_checkpoint, args=(qq_id,)),
                priority=5,
            )
            return  # 无消息可抽取,直接返回

        # ==================== 步骤5: 计算本次抽取的最大消息ID ====================

        # max([m.id for m in messages]): 找出消息列表中最大的ID
        # - 列表推导式: 提取所有消息的id字段
        # - max(): 找出最大值
        # - 用途: 记录本次抽取到哪个消息,作为下次的断点
        max_msg_id = max([m.id for m in messages])

        # ==================== 步骤6: 调用LLM抽取记忆 ====================

        # await MemoryManager._extract_memories_with_llm(): 使用LLM抽取记忆
        # 参数:
        # - qq_id: 用户QQ号
        # - scene_type: 场景类型
        # - scene_id: 场景标识
        # - messages: 消息列表
        # 返回: List[dict],每个dict是一条记忆的结构化数据
        extracted = await MemoryManager._extract_memories_with_llm(
            qq_id=qq_id,
            scene_type=scene_type,
            scene_id=scene_id,
            messages=messages,
        )

        # ==================== 步骤7: 写入记忆到数据库 ====================

        # await MemoryManager.upsert_memories(): 批量写入/更新记忆
        # - 参数: 用户QQ号, 记忆数据列表
        # - 功能: 去重、冲突检测、写入数据库、创建索引任务
        await MemoryManager.upsert_memories(qq_id, extracted)

        # ==================== 步骤8: 更新last_memory_msg_id(记录断点) ====================

        # await db_writer.submit_and_wait(): 更新档案
        # ProfileRepository.update_last_memory_msg_id(qq_id, max_msg_id):
        # - 将last_memory_msg_id更新为本次抽取的最大消息ID
        # - 作用: 下次抽取从这个ID之后开始,避免重复处理
        await db_writer.submit_and_wait(
            AsyncCallableJob(ProfileRepository.update_last_memory_msg_id, args=(qq_id, max_msg_id)),
            priority=5,
        )

        # ==================== 步骤9: 重置"待抽取"标记并前移检查点 ====================

        # 重置pending_memory为False
        # - 作用: 表示本次抽取已完成
        await db_writer.submit_and_wait(
            AsyncCallableJob(ProfileRepository.update_memory_status, args=(qq_id, False)),
            priority=5,
        )

        # 前移next_memory_at检查点
        # - bump_next_memory_checkpoint(): 增加next_memory_at(如50→100)
        # - 作用: 避免重复触发抽取,等用户再积累N条消息后再抽取
        await db_writer.submit_and_wait(
            AsyncCallableJob(ProfileRepository.bump_next_memory_checkpoint, args=(qq_id,)),
            priority=5,
        )

    @staticmethod
    async def select_for_context(
        qq_id: str,
        scene_type: str,
        scene_id: str,
        query: str,
    ) -> List[Memory]:
        """为当前对话选择需要注入上下文的记忆集合

        这个方法的作用:
        - 从数据库查询用户的记忆(core层+active层)
        - 根据可见性过滤记忆(global/group_only/private_only)
        - 按相关性排序active记忆
        - 返回top记忆列表供LLM使用

        选择策略:
        1. 核心记忆(core层): 全部选择,最高优先级
           - 特点: 最重要、高度凝练、永久保留
           - 数量: 默认前N条(yuying_memory_core_limit)
        2. 活跃记忆(active层): 按相关性排序,选择top-5
           - 特点: 最近抽取、时效性强
           - 过滤: 可见性检查(global/group_only/private_only)
           - 排序: 相关性评分(包含关系 + 编辑距离)
           - 数量: 前5条

        相关性评分算法:
        - 包含关系: query包含memory.content或反之 → 1.0分
        - 编辑距离: SequenceMatcher计算相似度 → 0.0~1.0分
        - 阈值: 分数>0才选择

        Args:
            qq_id: 用户QQ号
                - 类型: 字符串
                - 用途: 查询该用户的记忆
            scene_type: 场景类型
                - 类型: 字符串
                - 取值: "group" 或 "private"
                - 用途: 可见性过滤
            scene_id: 场景标识
                - 类型: 字符串
                - 内容: 群号或对方QQ号
                - 用途: 可见性过滤(group_only记忆限定群)
            query: 当前对话的查询文本
                - 类型: 字符串
                - 来源: Hybrid Query
                - 用途: 计算相关性分数

        Returns:
            List[Memory]: 记忆对象列表
                - 顺序: core记忆在前,active记忆在后
                - 数量: core数量 + active数量(最多5条)
                - 示例: [
                    Memory(tier="core", content="用户是程序员"),
                    Memory(tier="active", content="用户喜欢Python"),
                    Memory(tier="active", content="用户住在北京"),
                  ]

        配置项:
            - yuying_memory_core_limit: 核心记忆数量限制(默认5)

        Example:
            >>> memories = await MemoryManager.select_for_context(
            ...     qq_id="123456",
            ...     scene_type="group",
            ...     scene_id="789",
            ...     query="今天天气怎么样"
            ... )
            >>> for m in memories:
            ...     print(f"{m.tier}: {m.content}")
            # core: 用户住在北京
            # active: 用户怕冷
            # active: 用户关心天气
        """

        # ==================== 步骤1: 查询核心记忆(core层) ====================

        # await MemoryRepository.get_core_memories(qq_id): 查询用户的核心记忆
        # - SQL: SELECT * FROM memories WHERE qq_id=? AND tier='core'
        # - 返回: Memory对象列表
        # - 特点: 核心记忆全部选择,不过滤,最高优先级
        core_memories = await MemoryRepository.get_core_memories(qq_id)

        # ==================== 步骤2: 查询活跃记忆(active层) ====================

        # await MemoryRepository.list_active_for_user(qq_id): 查询用户的活跃记忆
        # - SQL: SELECT * FROM memories WHERE qq_id=? AND tier='active'
        # - 返回: Memory对象列表
        active_memories = await MemoryRepository.list_active_for_user(qq_id)

        # ==================== 步骤3: 过滤活跃记忆(可见性检查) ====================

        # 列表推导式: 只保留符合条件的记忆
        # 条件1: m.status == "active" (状态为active,非archived/deleted/conflict)
        # 条件2: MemoryManager._visibility_allows(...) (可见性检查通过)
        active_memories = [
            m for m in active_memories
            if m.status == "active" and MemoryManager._visibility_allows(m.visibility, scene_type, scene_id, m.scope_scene_id)
        ]

        # ==================== 步骤4: 计算相关性分数并排序 ====================

        # 列表推导式: 为每条记忆计算相关性分数
        # (MemoryManager._relevance_score(m.content, query), m):
        # - 返回元组: (相关性分数, Memory对象)
        # - 分数范围: 0.0~1.0
        scored = [(MemoryManager._relevance_score(m.content, query), m) for m in active_memories]

        # scored.sort(): 按相关性分数排序
        # - key=lambda x: x[0]: 按元组的第一个元素(分数)排序
        # - reverse=True: 降序排序(分数高的在前)
        scored.sort(key=lambda x: x[0], reverse=True)

        # ==================== 步骤5: 选择top-5活跃记忆 ====================

        # [m for s, m in scored if s > 0]: 只选择分数>0的记忆
        # - s: 相关性分数
        # - m: Memory对象
        # - if s > 0: 过滤掉完全不相关的记忆
        # [:5]: 取前5条
        selected_active = [m for s, m in scored if s > 0][:5]

        # ==================== 步骤6: 合并核心记忆和活跃记忆 ====================

        # core_memories[: int(plugin_config.yuying_memory_core_limit)]:
        # - 核心记忆取前N条(配置限制)
        # - 默认值: 5条
        # + selected_active: 拼接活跃记忆
        # 返回: 核心记忆在前,活跃记忆在后
        # logger.debug(f"【记忆模块】返回的全部记忆: {core_memories[: int(plugin_config.yuying_memory_core_limit)] + selected_active}")
        return core_memories[: int(plugin_config.yuying_memory_core_limit)] + selected_active

    @staticmethod
    async def upsert_memories(qq_id: str, memories_data: List[dict]) -> None:
        """将抽取到的记忆写入active层(包含去重与简单冲突标记)

        这个方法的作用:
        - 批量写入/更新记忆到数据库
        - 去重: 检查是否已有相似记忆
        - 冲突检测: 标记互相矛盾的记忆
        - 证据链接: 关联支撑记忆的原始消息
        - 索引任务: 创建向量化任务

        处理流程:
        1. 遍历每条记忆数据
        2. 验证和规范化字段
        3. 查找相似记忆(去重)
        4. 如果相似 → 更新置信度,添加证据
        5. 如果不相似 → 冲突检测
        6. 写入数据库(Memory表)
        7. 添加证据链接(MemoryEvidence表)
        8. 创建向量化任务(IndexJob表)

        去重算法:
        - 同type + 内容相似度>=0.86 → 认为是相同记忆
        - 处理: 更新置信度为两者最大值,添加新证据

        冲突检测算法:
        - 同type + 内容相似度<0.45 + 双方置信度>=0.75 → 认为冲突
        - 处理: 标记status="conflict",需人工审核

        Args:
            qq_id: 用户QQ号
                - 类型: 字符串
                - 用途: 记忆所属用户
            memories_data: 记忆数据列表
                - 类型: List[dict]
                - 来源: _extract_memories_with_llm()的返回值
                - 格式: [
                    {
                      "type": "fact",
                      "content": "用户是程序员",
                      "confidence": 0.9,
                      "visibility": "global",
                      "scope_scene_id": None,
                      "ttl_days": None,
                      "evidence_msg_ids": [101, 102]
                    },
                    ...
                  ]

        Side Effects:
            - 写入Memory表(新记忆)
            - 更新Memory表(相似记忆)
            - 写入MemoryEvidence表(证据链接)
            - 创建IndexJob任务(向量化)

        Example:
            >>> await MemoryManager.upsert_memories(
            ...     qq_id="123456",
            ...     memories_data=[
            ...         {
            ...             "type": "fact",
            ...             "content": "用户是Python程序员",
            ...             "confidence": 0.9,
            ...             "visibility": "global",
            ...             "evidence_msg_ids": [101, 102, 103]
            ...         }
            ...     ]
            ... )
            # 写入Memory表,创建IndexJob任务
        """

        # ==================== 步骤1: 查询用户的现有记忆(用于去重) ====================

        # await MemoryRepository.list_active_for_user(qq_id): 查询active层记忆
        # - 返回: Memory对象列表
        # - 用途: 去重和冲突检测时需要对比现有记忆
        existing = await MemoryRepository.list_active_for_user(qq_id)

        # ==================== 步骤2: 遍历每条记忆数据 ====================

        for mem_data in memories_data:
            # ==================== 步骤2.1: 提取和验证content字段 ====================

            # str(mem_data.get("content") or "").strip(): 获取content字段
            # - .get("content"): 安全获取,不存在返回None
            # - or "": None转为空字符串
            # - str(...): 确保是字符串类型
            # - .strip(): 去除首尾空格
            content = str(mem_data.get("content") or "").strip()

            # not content: 内容为空
            # - 跳过: 无效记忆,不写入
            if not content:
                continue

            # ==================== 步骤2.2: 提取和规范化其他字段 ====================

            # str(mem_data.get("type") or "fact").strip(): 记忆类型
            # - 默认值: "fact"
            mem_type = str(mem_data.get("type") or "fact").strip()

            # float(mem_data.get("confidence", 0.7)): 置信度
            # - 默认值: 0.7
            confidence = float(mem_data.get("confidence", 0.7))

            # str(mem_data.get("visibility") or "global").strip(): 可见性
            # - 默认值: "global"
            visibility = str(mem_data.get("visibility") or "global").strip()

            # mem_data.get("scope_scene_id"): 作用域场景ID
            # - 可选字段,可能是None
            scope_scene_id = mem_data.get("scope_scene_id")

            # mem_data.get("ttl_days"): 生存时间(天)
            # - 可选字段,可能是None
            ttl_days = mem_data.get("ttl_days")

            # mem_data.get("evidence_msg_ids") or []: 证据消息ID列表
            # - 默认值: 空列表
            evidence_msg_ids = mem_data.get("evidence_msg_ids") or []

            # ==================== 步骤2.3: 字段规范化处理 ====================

            # 限制content长度为50字符
            # - 原因: 记忆应该简洁,过长的内容不是好记忆
            if len(content) > 50:
                content = content[:50]  # 截断

            # 限制confidence范围为[0.0, 1.0]
            # - max(0.0, ...): 不能小于0
            # - min(1.0, ...): 不能大于1
            confidence = max(0.0, min(1.0, confidence))

            # ==================== 步骤2.4: 查找相似记忆(去重) ====================

            # MemoryManager._find_similar(): 在现有记忆中查找相似项
            # - 参数: 现有记忆列表, 记忆类型, 内容
            # - 返回: 相似的Memory对象或None
            # - 算法: 同type + 相似度>=0.86
            similar = MemoryManager._find_similar(existing, mem_type, content)

            # ==================== 步骤2.5: 如果找到相似记忆,更新而不是新建 ====================

            if similar:  # 找到相似记忆
                # 更新相似记忆的置信度和状态
                await db_writer.submit_and_wait(
                    AsyncCallableJob(
                        MemoryRepository.update_fields,  # 更新字段
                        args=(similar.id,),  # 记忆ID
                        kwargs={
                            # 置信度取两者最大值
                            "confidence": max(similar.confidence, confidence),
                            # 状态重新设为active(如果之前是archived)
                            "status": "active",
                        },
                    ),
                    priority=5,
                )

                # 如果有新证据,添加到证据链接表
                if evidence_msg_ids:
                    await db_writer.submit_and_wait(
                        AsyncCallableJob(
                            MemoryRepository.add_evidence,  # 添加证据
                            args=(
                                similar.id,  # 记忆ID
                                # 将evidence_msg_ids转为整数列表,过滤非数字
                                [int(x) for x in evidence_msg_ids if str(x).isdigit()],
                            ),
                        ),
                        priority=5,
                    )

                # continue: 跳过后续步骤,处理下一条记忆
                continue

            # ==================== 步骤2.6: 没有相似记忆,执行冲突检测 ====================

            # 默认状态为active
            status = "active"

            # MemoryManager._is_conflict(): 检查是否与现有记忆冲突
            # - 参数: 现有记忆列表, 记忆类型, 内容, 置信度
            # - 返回: bool
            # - 算法: 同type + 相似度<0.45 + 双方置信度>=0.75
            if MemoryManager._is_conflict(existing, mem_type, content, confidence):
                # 标记为conflict状态
                # - 需人工审核,不直接注入上下文
                status = "conflict"

            # ==================== 步骤2.7: 创建新记忆对象 ====================

            # Memory(): 创建记忆模型对象
            memory = Memory(
                qq_id=qq_id,  # 用户QQ号
                tier="active",  # 层级: 活跃层
                type=mem_type,  # 类型: fact/preference/habit等
                content=content,  # 内容
                confidence=confidence,  # 置信度
                status=status,  # 状态: active或conflict
                visibility=visibility,  # 可见性: global/group_only/private_only
                # scope_scene_id: 作用域场景ID(group_only时必需)
                scope_scene_id=str(scope_scene_id) if scope_scene_id else None,
                # ttl_days: 生存时间(天),默认从配置读取
                ttl_days=int(ttl_days) if ttl_days is not None else int(plugin_config.yuying_memory_active_ttl_days),
            )

            # ==================== 步骤2.8: 写入数据库 ====================

            # await db_writer.submit_and_wait(): 提交写入任务并等待
            # MemoryRepository.add(memory): 插入记忆到数据库
            # - 返回: 插入后的Memory对象(带id)
            memory = await db_writer.submit_and_wait(
                AsyncCallableJob(MemoryRepository.add, args=(memory,)),
                priority=5,
            )

            # existing.append(memory): 添加到现有记忆列表
            # - 用途: 后续记忆的去重和冲突检测需要对比这条新记忆
            existing.append(memory)

            # ==================== 步骤2.9: 添加证据链接 ====================

            if evidence_msg_ids:  # 如果有证据消息ID
                await db_writer.submit_and_wait(
                    AsyncCallableJob(
                        MemoryRepository.add_evidence,  # 添加证据链接
                        args=(memory.id, [int(x) for x in evidence_msg_ids if str(x).isdigit()]),
                    ),
                    priority=5,
                )

            # ==================== 步骤2.10: 创建向量化任务(索引双写) ====================

            # 为新记忆创建向量化任务
            # - 目的: 将记忆内容向量化,存入Qdrant
            # - 用途: 支持基于向量的记忆检索
            await db_writer.submit(
                AddIndexJobJob(  # 添加索引任务
                    IndexJob(
                        item_type="memory",  # 任务类型: 记忆向量化
                        ref_id=str(memory.id),  # 引用ID: 记忆ID
                        # payload_json: 任务载荷(JSON格式)
                        payload_json=json.dumps({"memory_id": memory.id}, ensure_ascii=False),
                        status="pending",  # 状态: 待处理
                    )
                ),
                priority=5,  # 优先级5
            )

    @staticmethod
    async def _extract_memories_with_llm(
        qq_id: str,
        scene_type: str,
        scene_id: str,
        messages: list,
    ) -> List[dict]:
        """调用主模型从对话窗口中抽取记忆,返回结构化JSON列表

        这个方法的作用:
        - 构建记忆抽取的prompt
        - 调用主LLM模型(temperature=0.2,确定性)
        - 解析LLM返回的JSON
        - 返回记忆数据列表

        prompt设计要点:
        - 明确任务: 抽取"稳定且对未来有用"的长期信息
        - 限制类型: 事实/偏好/风格/关系/目标/约束
        - 限制长度: content<=50字
        - 要求证据: evidence_msg_ids必须引用给出的消息id
        - 严格格式: 输出必须是JSON

        Args:
            qq_id: 用户QQ号
                - 类型: 字符串
                - 用途: 传递给LLM作为上下文
            scene_type: 场景类型
                - 类型: 字符串
                - 用途: 传递给LLM作为上下文
            scene_id: 场景标识
                - 类型: 字符串
                - 用途: 传递给LLM作为上下文
            messages: 消息列表
                - 类型: List[RawMessage]
                - 来源: RawRepository.get_user_messages_since()
                - 用途: 作为LLM分析的原始材料

        Returns:
            List[dict]: 记忆数据列表
                格式: [
                    {
                      "type": "fact",
                      "content": "用户是程序员",
                      "confidence": 0.9,
                      "visibility": "global",
                      "scope_scene_id": None,
                      "ttl_days": 7,
                      "evidence_msg_ids": [101, 102]
                    },
                    ...
                  ]
                - 失败: 返回空列表[]

        Example:
            >>> messages = [
            ...     RawMessage(id=101, content="我是Python程序员"),
            ...     RawMessage(id=102, content="我喜欢写代码"),
            ...     RawMessage(id=103, content="我住在北京"),
            ... ]
            >>> extracted = await MemoryManager._extract_memories_with_llm(
            ...     qq_id="123456",
            ...     scene_type="group",
            ...     scene_id="789",
            ...     messages=messages
            ... )
            >>> print(extracted)
            # [
            #   {"type": "fact", "content": "用户是Python程序员", "confidence": 0.9, ...},
            #   {"type": "preference", "content": "用户喜欢写代码", "confidence": 0.8, ...},
            #   {"type": "fact", "content": "用户住在北京", "confidence": 0.9, ...}
            # ]
        """

        # ==================== 步骤1: 格式化消息列表 ====================

        # "\n".join([...]): 用换行符连接消息列表
        # f"- ({m.id}) {m.content}": 格式化每条消息
        # - ({m.id}): 消息ID,供LLM引用
        # - {m.content}: 消息内容
        # 示例输出:
        #   - (101) 我是Python程序员
        #   - (102) 我喜欢写代码
        #   - (103) 我住在北京
        msg_lines = "\n".join([f"- ({m.id}) {m.content}" for m in messages])

        # ==================== 步骤2: 构建抽取prompt ====================

        # "\n".join([...]): 用换行符连接prompt各部分
        prompt = "\n".join(
            [
                # 任务说明
                "从以下用户消息中抽取'稳定且对未来有用'的长期信息,输出严格 JSON。",
                "要求:",
                # 要求1: 只抽取稳定信息
                "- 只抽取稳定信息:事实/偏好/风格/关系/目标/约束。",
                # 要求2: 内容简洁
                "- content <= 50 字,尽量具体。",
                # 要求3: 置信度范围
                "- confidence 为 0~1。",
                # 要求4: 可见性枚举
                "- visibility 只能是 global/group_only/private_only。",
                # 要求5: 证据引用
                "- evidence_msg_ids 必须引用上面给出的消息 id。",
                "",  # 空行
                # 输出格式示例
                "输出 JSON 格式:",
                '{"memories":[{"type":"fact","content":"...","confidence":0.8,"visibility":"global","scope_scene_id":null,"ttl_days":7,"evidence_msg_ids":[123]}]}',
                "",  # 空行
                # 上下文信息
                "用户 QQ:",
                qq_id,  # 用户QQ号
                f"场景:{scene_type}:{scene_id}",  # 场景信息
                "消息:",  # 消息列表标题
                msg_lines,  # 格式化后的消息列表
            ]
        )

        # ==================== 步骤3: 调用主LLM ====================

        # await main_llm.chat_completion(): 调用主LLM生成回复
        # 参数:
        # - messages: 对话消息列表(OpenAI格式)
        # - temperature=0.2: 低温度,生成确定性输出
        #   * 0.2比默认0.7更低,适合结构化任务
        #   * 确保输出格式稳定,避免随机性
        content = await main_llm.chat_completion(
            [
                # 系统提示
                {"role": "system", "content": "你是记忆抽取器,只能输出 JSON。"},
                # 用户提示
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,  # 低温度
        )

        # ==================== 步骤4: 处理LLM返回值 ====================

        # not content: LLM调用失败或返回空
        if not content:
            return []  # 返回空列表

        # ==================== 步骤5: 解析JSON ====================

        # MemoryManager._extract_first_json_object(content): 从文本中提取JSON
        # - 参数: LLM返回的文本(可能包含非JSON内容)
        # - 返回: 解析后的dict/list或None
        data = MemoryManager._extract_first_json_object(content)

        # 类型检查: 确保是字典
        if not isinstance(data, dict):
            return []  # 格式错误,返回空列表

        # ==================== 步骤6: 提取memories数组 ====================

        # data.get("memories"): 获取memories字段
        memories = data.get("memories")

        # 类型检查: 确保是列表
        if not isinstance(memories, list):
            return []  # 格式错误,返回空列表

        # ==================== 步骤7: 过滤并返回 ====================

        # [m for m in memories if isinstance(m, dict)]: 只保留字典元素
        # - 过滤掉格式错误的元素
        return [m for m in memories if isinstance(m, dict)]

    @staticmethod
    def _extract_first_json_object(text: str) -> Optional[object]:
        """从文本中提取第一个JSON对象或数组

        这个方法的作用:
        - 从LLM返回的文本中提取JSON部分
        - 支持对象{}和数组[]两种格式
        - 处理LLM输出中的多余文本(如解释、注释等)

        为什么需要这个方法?
        - LLM可能输出额外内容: "这是抽取的记忆: {...}"
        - 需要正则匹配提取纯JSON部分
        - 提高解析成功率

        Args:
            text: LLM返回的文本
                - 类型: 字符串
                - 可能包含: JSON + 非JSON内容
                - 示例: "这是抽取结果:\n{\"memories\":[...]}"

        Returns:
            Optional[object]: 解析后的Python对象
                - 成功: dict或list
                - 失败: None

        Example:
            >>> text = "这是结果: {\"memories\":[{\"type\":\"fact\"}]}"
            >>> obj = MemoryManager._extract_first_json_object(text)
            >>> print(obj)
            # {"memories": [{"type": "fact"}]}
        """

        # ==================== 步骤1: 去除首尾空格 ====================
        s = text.strip()

        # ==================== 步骤2: 正则匹配JSON部分 ====================

        # re.search(): 搜索匹配
        # r"(\{.*\}|\[.*\])": 正则表达式
        # - (\{.*\}): 匹配对象 {...}
        # - |: 或
        # - (\[.*\]): 匹配数组 [...]
        # - .*: 匹配任意字符(贪婪模式)
        # flags=re.S: DOTALL模式,让.匹配换行符
        # - 支持多行JSON
        m = re.search(r"(\{.*\}|\[.*\])", s, flags=re.S)

        # not m: 没有匹配到JSON
        if not m:
            return None  # 返回None

        # ==================== 步骤3: 解析JSON ====================

        try:
            # m.group(1): 获取第一个捕获组(JSON字符串)
            # json.loads(): 解析JSON字符串为Python对象
            return json.loads(m.group(1))
        except Exception:
            # 解析失败: JSON格式错误
            return None  # 返回None

    @staticmethod
    def _relevance_score(content: str, query: str) -> float:
        """简单相关性评分:包含关系 + 编辑距离

        这个方法的作用:
        - 计算记忆内容与查询的相关性分数
        - 用于排序active记忆,选择最相关的
        - 算法简单但有效

        评分规则:
        1. 包含关系: content包含query或反之 → 1.0分(完全相关)
        2. 编辑距离: SequenceMatcher计算相似度 → 0.0~1.0分

        为什么这样设计?
        - 包含关系: 如query="北京天气",memory="用户住在北京" → 1.0分(高度相关)
        - 编辑距离: 如query="Python",memory="Python程序员" → ~0.5分(部分相关)

        Args:
            content: 记忆内容
                - 类型: 字符串
                - 示例: "用户是Python程序员"
            query: 查询文本
                - 类型: 字符串
                - 示例: "今天天气怎么样"

        Returns:
            float: 相关性分数
                - 范围: 0.0~1.0
                - 1.0: 完全相关(包含关系)
                - 0.0~1.0: 部分相关(编辑距离)
                - 0.0: 完全不相关

        Example:
            >>> score = MemoryManager._relevance_score("用户住在北京", "北京天气")
            >>> print(score)  # 1.0 (包含关系)

            >>> score = MemoryManager._relevance_score("用户是程序员", "Python编程")
            >>> print(score)  # ~0.3 (编辑距离)
        """

        # ==================== 步骤1: 去除首尾空格 ====================
        c = content.strip()  # 记忆内容
        q = query.strip()  # 查询文本

        # ==================== 步骤2: 空值检查 ====================
        if not c or not q:  # 任一为空
            return 0.0  # 无法计算,返回0分

        # ==================== 步骤3: 检查包含关系 ====================
        # c in q: 记忆内容是查询的子串
        # q in c: 查询是记忆内容的子串
        if c in q or q in c:
            return 1.0  # 包含关系,完全相关

        # ==================== 步骤4: 计算编辑距离相似度 ====================

        # SequenceMatcher(None, c, q): 创建相似度计算器
        # - 第一个参数: None(不使用junk函数)
        # - 第二第三个参数: 要比较的两个字符串
        # .ratio(): 计算相似度比值
        # - 算法: 基于最长公共子序列(LCS)
        # - 返回: 0.0~1.0
        # - 1.0: 完全相同
        # - 0.0: 完全不同
        return SequenceMatcher(None, c, q).ratio()

    @staticmethod
    def _find_similar(existing: List[Memory], mem_type: str, content: str) -> Optional[Memory]:
        """在现有记忆中查找相似项(同type+高相似度)

        这个方法的作用:
        - 去重: 避免存储重复的记忆
        - 查找: 在现有记忆中寻找相似的记忆
        - 返回: 找到的相似记忆对象或None

        相似判定标准:
        - 条件1: 同层级(tier="active")
        - 条件2: 同类型(type相同)
        - 条件3: 高相似度(>=0.86)

        为什么阈值是0.86?
        - 经验值: 0.86可以识别大部分重复记忆
        - 太高(如0.95): 可能漏过重复
        - 太低(如0.7): 可能误判为重复

        Args:
            existing: 现有记忆列表
                - 类型: List[Memory]
                - 来源: MemoryRepository.list_active_for_user()
            mem_type: 记忆类型
                - 类型: 字符串
                - 示例: "fact", "preference"
            content: 记忆内容
                - 类型: 字符串
                - 示例: "用户是Python程序员"

        Returns:
            Optional[Memory]: 相似的记忆对象
                - 成功: 返回Memory对象
                - 失败: 返回None

        Example:
            >>> existing = [
            ...     Memory(type="fact", content="用户是程序员"),
            ...     Memory(type="preference", content="用户喜欢编程"),
            ... ]
            >>> similar = MemoryManager._find_similar(existing, "fact", "用户是Python程序员")
            >>> print(similar.content)  # "用户是程序员" (相似度>=0.86)
        """

        # ==================== 遍历现有记忆 ====================
        for m in existing:
            # ==================== 条件1: 必须是active层 ====================
            if m.tier != "active":
                continue  # 跳过,只在active层查找

            # ==================== 条件2: 必须是同类型 ====================
            if m.type != mem_type:
                continue  # 跳过,不同类型不算相似

            # ==================== 条件3: 计算相似度 ====================

            # SequenceMatcher(None, ..., ...).ratio(): 计算相似度
            # (m.content or "").strip(): 现有记忆的内容
            # content: 新记忆的内容
            # .ratio() >= 0.86: 相似度阈值
            if SequenceMatcher(None, (m.content or "").strip(), content).ratio() >= 0.86:
                return m  # 找到相似记忆,立即返回

        # ==================== 未找到相似记忆 ====================
        return None  # 返回None

    @staticmethod
    def _is_conflict(existing: List[Memory], mem_type: str, content: str, confidence: float) -> bool:
        """简单冲突判定:同type且差异很大、且置信度高

        这个方法的作用:
        - 冲突检测: 识别互相矛盾的记忆
        - 标记: 将冲突记忆标记为status="conflict"
        - 避免: 将矛盾信息注入LLM上下文

        冲突判定标准:
        - 条件1: 新记忆置信度>=0.75(高置信)
        - 条件2: 同层级(tier="active")
        - 条件3: 同类型(type相同)
        - 条件4: 现有记忆置信度>=0.75(双方都高置信)
        - 条件5: 低相似度(<0.45,差异很大)

        为什么需要冲突检测?
        - 例如: "用户喜欢吃辣" vs "用户不喜欢吃辣"
        - 两条记忆矛盾,不能同时注入上下文
        - 标记为conflict,需人工审核

        Args:
            existing: 现有记忆列表
                - 类型: List[Memory]
            mem_type: 新记忆的类型
                - 类型: 字符串
            content: 新记忆的内容
                - 类型: 字符串
            confidence: 新记忆的置信度
                - 类型: float,范围0.0~1.0

        Returns:
            bool: 是否冲突
                - True: 检测到冲突
                - False: 无冲突

        Example:
            >>> existing = [
            ...     Memory(type="preference", content="用户喜欢吃辣", confidence=0.9),
            ... ]
            >>> is_conf = MemoryManager._is_conflict(
            ...     existing, "preference", "用户不喜欢吃辣", 0.8
            ... )
            >>> print(is_conf)  # True (矛盾且双方高置信)
        """

        # ==================== 步骤1: 检查新记忆的置信度 ====================

        # confidence < 0.75: 新记忆置信度不高
        # - 低置信记忆不足以判定冲突
        if confidence < 0.75:
            return False  # 无冲突

        # ==================== 步骤2: 遍历现有记忆 ====================

        for m in existing:
            # ==================== 条件1: 必须是active层 ====================
            if m.tier != "active":
                continue  # 跳过,只检查active层

            # ==================== 条件2: 必须是同类型 ====================
            if m.type != mem_type:
                continue  # 跳过,不同类型不会冲突

            # ==================== 条件3: 现有记忆也必须高置信 ====================
            if m.confidence < 0.75:
                continue  # 跳过,双方都高置信才算冲突

            # ==================== 条件4: 计算相似度 ====================

            # SequenceMatcher(None, ..., ...).ratio(): 计算相似度
            sim = SequenceMatcher(None, (m.content or "").strip(), content).ratio()

            # sim < 0.45: 低相似度(差异很大)
            # - 差异大+双方高置信 → 冲突
            if sim < 0.45:
                return True  # 检测到冲突

        # ==================== 未检测到冲突 ====================
        return False  # 无冲突

    @staticmethod
    def _visibility_allows(visibility: str, scene_type: str, scene_id: str, scope_scene_id: Optional[str]) -> bool:
        """判断记忆可见性是否允许在该场景注入

        这个方法的作用:
        - 可见性控制: 根据visibility和场景判断记忆是否可见
        - 隐私保护: 避免群聊记忆泄漏到私聊,反之亦然
        - 场景隔离: 群聊记忆只在特定群可见

        可见性规则:
        1. global: 全局可见,所有场景都可注入
        2. group_only: 仅群聊可见,且仅在特定群(scope_scene_id匹配)
        3. private_only: 仅私聊可见

        为什么需要可见性控制?
        - 隐私: 私聊的记忆不应该在群聊中暴露
        - 场景: 不同群的记忆应该隔离(避免串场)

        Args:
            visibility: 记忆的可见性标识
                - 类型: 字符串
                - 取值: "global", "group_only", "private_only"
            scene_type: 当前场景类型
                - 类型: 字符串
                - 取值: "group" 或 "private"
            scene_id: 当前场景标识
                - 类型: 字符串
                - 内容: 群号或对方QQ号
            scope_scene_id: 记忆的作用域场景ID
                - 类型: Optional[str]
                - 用途: group_only记忆限定的群号

        Returns:
            bool: 是否允许注入
                - True: 允许注入
                - False: 不允许注入

        Example:
            >>> # 全局记忆在任何场景都可见
            >>> allowed = MemoryManager._visibility_allows(
            ...     "global", "group", "123", None
            ... )
            >>> print(allowed)  # True

            >>> # 群聊限定记忆只在特定群可见
            >>> allowed = MemoryManager._visibility_allows(
            ...     "group_only", "group", "123", "123"
            ... )
            >>> print(allowed)  # True (场景匹配)

            >>> allowed = MemoryManager._visibility_allows(
            ...     "group_only", "group", "456", "123"
            ... )
            >>> print(allowed)  # False (场景不匹配)

            >>> # 私聊限定记忆只在私聊可见
            >>> allowed = MemoryManager._visibility_allows(
            ...     "private_only", "private", "789", None
            ... )
            >>> print(allowed)  # True

            >>> allowed = MemoryManager._visibility_allows(
            ...     "private_only", "group", "123", None
            ... )
            >>> print(allowed)  # False (场景不匹配)
        """

        # ==================== 情况1: global可见性 ====================
        if visibility == "global":
            return True  # 全局可见,所有场景都允许

        # ==================== 情况2: group_only可见性 ====================
        if visibility == "group_only":
            # 条件1: 当前场景必须是群聊
            # 条件2: scope_scene_id必须匹配当前群号
            return scene_type == "group" and scope_scene_id == scene_id

        # ==================== 情况3: private_only可见性 ====================
        if visibility == "private_only":
            # 条件: 当前场景必须是私聊
            return scene_type == "private"

        # ==================== 其他情况: 未知可见性 ====================
        return False  # 默认不允许(安全策略)
