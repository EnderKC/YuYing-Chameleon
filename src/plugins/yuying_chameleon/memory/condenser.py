"""记忆浓缩模块 - 三层记忆架构的自动流转和核心记忆治理

这个模块的作用:
1. 实现记忆的定期浓缩(每日03:00执行)
2. 将过期的active记忆自动归档到archive层
3. 从archive+core记忆中提炼出新的核心记忆
4. 控制核心记忆数量上限,避免无限膨胀
5. 通过LLM智能合并和压缩记忆内容

记忆浓缩原理(新手必读):
- 问题: active记忆不断积累,数据库会越来越大
- 解决: 定期浓缩,将旧记忆归档,提炼核心信息
- 流程: active过期 → archive归档 → LLM浓缩 → core永久保留
- 好处: 保持数据库精简,核心记忆永久保留用户关键信息

三层记忆流转机制:
1. active层(活跃记忆):
   - 来源: 从用户对话中新鲜抽取
   - 生存期: 默认30天(yuying_memory_active_ttl_days)
   - 过期后: 自动归档到archive层
   - 特点: 数量多,时效性强,优先使用

2. archive层(归档记忆):
   - 来源: 从active层过期而来
   - 生存期: 默认180天
   - 过期后: 被删除或浓缩到core层
   - 特点: 降低优先级,作为浓缩的原材料

3. core层(核心记忆):
   - 来源: LLM从archive+旧core中浓缩提炼
   - 生存期: 永久保留
   - 数量限制: 默认5条(yuying_memory_core_limit)
   - 特点: 高度凝练,最重要的信息

浓缩任务触发机制:
- 触发时间: 每日凌晨03:00(由scheduler调度)
- 触发条件: 有active记忆过期
- 处理范围: 所有有过期记忆的用户
- 并发策略: 逐用户串行处理,避免资源竞争

LLM浓缩算法:
1. 输入: archive记忆(最近N天) + 现有core记忆
2. 任务: 生成<=5条核心记忆,只保留稳定信息
3. 优先级: fact > preference > style > relationship
4. 输出: JSON格式的新core记忆列表
5. 去重: 自动合并相似记忆,删除重复信息
6. 溯源: 记录每条core记忆来自哪些旧记忆

核心记忆上限治理:
- 问题: LLM可能生成过多核心记忆
- 限制: 最多5条(可配置)
- 超限处理: 二次调用LLM进行合并压缩
- 降级策略: LLM失败时保留现有core并截断

使用方式:
```python
from .memory.condenser import MemoryCondenser

# 方式1: 定时任务调用(由scheduler自动调用)
await MemoryCondenser.run_daily_condenser()

# 方式2: 手动浓缩某个用户的记忆
await MemoryCondenser.condense_user(qq_id="123456")
```

定时任务配置:
- 执行时间: 每日03:00
- 配置位置: scheduler.py
- 任务名称: "memory_condense"
"""

from __future__ import annotations

import json  # Python标准库,用于JSON编解码
import re  # 正则表达式,用于提取LLM返回的JSON
import time  # 时间戳获取
from typing import List, Optional  # 类型提示

from nonebot import logger  # NoneBot日志记录器

# 导入项目模块
from ..config import plugin_config  # 插件配置
from ..llm.client import main_llm  # 主LLM客户端
from ..storage.models import IndexJob, Memory  # 数据库模型
from ..storage.db_writer import db_writer  # 数据库写入队列
from ..storage.write_jobs import AddIndexJobJob, AsyncCallableJob  # 写入任务
from ..storage.repositories.memory_repo import MemoryRepository  # 记忆仓库


class MemoryCondenser:
    """记忆浓缩器 - 03:00定时执行,实现三层记忆架构的自动流转

    这个类的作用:
    - 管理记忆的定期浓缩任务
    - 实现active → archive → core的流转逻辑
    - 调用LLM进行智能记忆合并和压缩
    - 控制核心记忆数量上限

    设计模式:
    - 静态方法类: 所有方法都是静态的,无需实例化
    - 好处: 作为定时任务入口,避免状态管理

    核心流程:
    1. run_daily_condenser()作为定时任务入口
    2. 查询所有过期的active记忆
    3. 将过期记忆归档到archive层
    4. 收集受影响的用户列表
    5. 对每个用户调用condense_user()
    6. condense_user()调用LLM浓缩记忆
    7. 更新core记忆并创建向量化任务

    配置项:
    - yuying_memory_active_ttl_days: active记忆生存期(默认30天)
    - yuying_memory_archive_days: archive查询窗口(默认180天)
    - yuying_memory_core_limit: 核心记忆数量上限(默认5条)
    """

    @staticmethod
    async def run_daily_condenser() -> None:
        """每日定时任务入口 - 归档过期active,并对受影响用户进行凝练

        这个方法的作用:
        - 作为定时任务的入口点(每日03:00执行)
        - 查询所有过期的active记忆
        - 将过期记忆批量归档到archive层
        - 触发受影响用户的记忆浓缩

        执行流程:
        1. 查询所有过期的active记忆(ttl_days已到期)
        2. 逐条归档(更新tier="archive")
        3. 收集受影响的用户QQ号
        4. 对每个用户执行浓缩(condense_user)
        5. 记录日志并处理异常

        为什么需要归档?
        - active记忆有生存期(默认30天)
        - 过期记忆降低优先级,减少查询开销
        - 归档记忆作为浓缩的原材料
        - 保持active层精简,提高检索效率

        Side Effects:
            - 更新Memory表的tier字段(active → archive)
            - 调用condense_user()浓缩受影响用户的记忆
            - 输出日志信息

        异常处理:
            - 单个用户浓缩失败不影响其他用户
            - 记录错误日志并继续处理

        Example:
            >>> # 由scheduler在每日03:00自动调用
            >>> await MemoryCondenser.run_daily_condenser()
            # 输出: 开始执行每日记忆凝练任务...
            # 输出: 用户 123456 凝练失败:xxx (如果失败)
            # 输出: 每日记忆凝练任务结束。
        """

        # ==================== 步骤1: 输出任务开始日志 ====================

        # logger.info(): 记录信息级别日志
        logger.info("开始执行每日记忆凝练任务...")

        # ==================== 步骤2: 获取当前时间戳 ====================

        # int(time.time()): 获取当前Unix时间戳(秒级)
        # - 用途: 判断记忆是否过期
        # - 过期判断: created_at + ttl_days * 86400 < current_ts
        current_ts = int(time.time())

        # ==================== 步骤3: 查询所有过期的active记忆 ====================

        # await MemoryRepository.get_expired_active(current_ts): 查询过期记忆
        # - 参数: 当前时间戳
        # - SQL逻辑: WHERE tier='active' AND created_at + ttl_days * 86400 < current_ts
        # - 返回: Memory对象列表
        expired_memories = await MemoryRepository.get_expired_active(current_ts)

        # ==================== 步骤4: 归档过期记忆并收集用户 ====================

        # users_to_process: 需要浓缩的用户集合(使用set去重)
        # - 作用: 记录哪些用户有记忆被归档
        # - 类型: set[str] (QQ号集合)
        users_to_process = set()

        # 遍历所有过期记忆
        for mem in expired_memories:
            # ==================== 步骤4.1: 归档单条记忆 ====================

            # await db_writer.submit_and_wait(): 提交写入任务并等待完成
            # AsyncCallableJob: 异步可调用任务
            # MemoryRepository.archive_memory(mem.id): 将记忆归档
            # - 操作: UPDATE memories SET tier='archive' WHERE id=mem.id
            # - 效果: 记忆从active层移到archive层
            await db_writer.submit_and_wait(
                AsyncCallableJob(MemoryRepository.archive_memory, args=(mem.id,)),
                priority=5,  # 优先级5(中等)
            )

            # ==================== 步骤4.2: 记录受影响的用户 ====================

            # users_to_process.add(mem.qq_id): 添加用户QQ号到集合
            # - set自动去重: 同一用户的多条记忆只会触发一次浓缩
            users_to_process.add(mem.qq_id)

        # ==================== 步骤5: 对每个用户执行浓缩 ====================

        # 遍历所有受影响的用户
        for qq_id in users_to_process:
            try:
                # ==================== 步骤5.1: 浓缩单个用户的记忆 ====================

                # await MemoryCondenser.condense_user(qq_id): 浓缩用户记忆
                # - 参数: 用户QQ号
                # - 流程: 查询archive+core → 调用LLM浓缩 → 更新core记忆
                await MemoryCondenser.condense_user(qq_id)

            except Exception as exc:
                # 单个用户浓缩失败不影响其他用户
                # logger.error(): 记录错误日志
                logger.error(f"用户 {qq_id} 凝练失败:{exc}")

        # ==================== 步骤6: 输出任务结束日志 ====================

        logger.info("每日记忆凝练任务结束。")

    @staticmethod
    async def condense_user(qq_id: str) -> None:
        """对单个用户凝练核心记忆(tier=core)

        这个方法的作用:
        - 从archive+core记忆中提炼出新的核心记忆
        - 调用LLM进行智能合并和压缩
        - 控制核心记忆数量不超过上限
        - 为新核心记忆创建向量化任务

        浓缩流程:
        1. 查询用户的archive记忆(最近N天)
        2. 查询用户的现有core记忆
        3. 调用LLM浓缩: archive+core → 新core列表
        4. 如果LLM失败: 降级为保留现有core并截断
        5. 如果新core超限: 二次调用LLM合并压缩
        6. 替换整个core层记忆
        7. 为新core创建向量化任务

        为什么需要浓缩?
        - archive记忆不断积累,需要提炼精华
        - 核心记忆永久保留,必须严格控制数量
        - 浓缩可以合并相似记忆,去除重复信息
        - 保持核心记忆的高质量和高相关性

        Args:
            qq_id: 用户QQ号
                - 类型: 字符串
                - 用途: 标识要浓缩的用户

        Side Effects:
            - 替换Memory表中该用户的所有core记忆
            - 创建IndexJob任务(向量化新core记忆)
            - 输出日志信息

        配置项:
            - yuying_memory_archive_days: archive查询窗口(默认180天)
            - yuying_memory_core_limit: 核心记忆数量上限(默认5条)

        降级策略:
            - LLM失败: 保留现有core并截断到上限
            - 二次合并失败: 直接截断到上限

        Example:
            >>> await MemoryCondenser.condense_user(qq_id="123456")
            # 输出: 已更新用户 123456 的核心记忆:3 条
        """

        # ==================== 步骤1: 读取配置参数 ====================

        # int(plugin_config.yuying_memory_archive_days): archive查询窗口(天数)
        # - 默认值: 180天
        # - 作用: 查询最近N天的archive记忆作为浓缩输入
        # - 原因: 太久远的archive记忆可能已不相关
        archive_days = int(plugin_config.yuying_memory_archive_days)

        # int(plugin_config.yuying_memory_core_limit): 核心记忆数量上限
        # - 默认值: 5条
        # - 作用: 限制核心记忆数量,避免无限膨胀
        # - 原因: 核心记忆会永久注入LLM上下文,过多会占用token
        core_limit = int(plugin_config.yuying_memory_core_limit)

        # ==================== 步骤2: 查询archive和core记忆 ====================

        # await MemoryRepository.list_recent_archive(qq_id, days): 查询最近archive
        # - 参数: 用户QQ号, 查询天数
        # - SQL: WHERE qq_id=? AND tier='archive' AND created_at >= now - days
        # - 返回: Memory对象列表
        archive = await MemoryRepository.list_recent_archive(qq_id, days=archive_days)

        # await MemoryRepository.get_core_memories(qq_id): 查询现有core记忆
        # - SQL: WHERE qq_id=? AND tier='core'
        # - 返回: Memory对象列表
        existing_core = await MemoryRepository.get_core_memories(qq_id)

        # ==================== 步骤3: 检查是否需要浓缩 ====================

        # not archive and not existing_core: 既没有archive也没有core
        # - 原因: 用户没有可浓缩的记忆
        # - 处理: 直接返回,无需浓缩
        if not archive and not existing_core:
            return  # 无内容可浓缩

        # ==================== 步骤4: 调用LLM浓缩记忆 ====================

        # await MemoryCondenser._call_llm_to_condense(): 调用LLM浓缩
        # - 参数: 用户QQ号, archive记忆, 现有core记忆, 数量上限
        # - 返回: List[Memory]或None
        # - 功能: 从archive+core中提炼出新的core记忆列表
        draft = await MemoryCondenser._call_llm_to_condense(qq_id, archive, existing_core, core_limit)

        # ==================== 步骤5: 处理LLM失败的降级情况 ====================

        if not draft:  # LLM浓缩失败(返回None或空列表)
            # 降级策略: 保留现有core并截断到上限
            # existing_core[:core_limit]: 切片,取前N条
            # - 原因: LLM失败时保留现有记忆优于清空
            trimmed = existing_core[:core_limit]

            # 替换core记忆为截断后的列表
            await db_writer.submit_and_wait(
                AsyncCallableJob(MemoryRepository.replace_core_memories, args=(qq_id, trimmed)),
                priority=5,
            )
            return  # 降级处理完成,直接返回

        # ==================== 步骤6: 处理core超限的情况 ====================

        # len(draft) > core_limit: 新core记忆超过数量上限
        if len(draft) > core_limit:
            # 二次调用LLM进行合并压缩
            # await MemoryCondenser._call_llm_to_merge_core(): 合并压缩core
            # - 参数: 用户QQ号, 当前draft列表, 数量上限
            # - 返回: List[Memory]或None
            # - or draft[:core_limit]: 如果二次合并失败,直接截断
            draft = await MemoryCondenser._call_llm_to_merge_core(qq_id, draft, core_limit) or draft[:core_limit]

        # ==================== 步骤7: 替换整个core层记忆 ====================

        # await db_writer.submit_and_wait(): 提交写入任务并等待
        # MemoryRepository.replace_core_memories(qq_id, draft):
        # - 操作: 删除该用户所有旧core记忆,插入新draft列表
        # - 原子性: 在一个事务中完成,避免中间状态
        await db_writer.submit_and_wait(
            AsyncCallableJob(MemoryRepository.replace_core_memories, args=(qq_id, draft)),
            priority=5,
        )

        # ==================== 步骤8: 重新查询持久化后的core记忆 ====================

        # await MemoryRepository.get_core_memories(qq_id): 查询刚插入的core记忆
        # - 目的: 获取数据库自动生成的id字段
        # - 用途: 创建向量化任务需要记忆ID
        persisted = await MemoryRepository.get_core_memories(qq_id)

        # ==================== 步骤9: 为新core记忆创建向量化任务 ====================

        # 遍历每条新core记忆
        for mem in persisted:
            # await db_writer.submit(): 提交写入任务(不等待)
            # AddIndexJobJob: 添加索引任务
            # IndexJob: 索引任务模型
            await db_writer.submit(
                AddIndexJobJob(
                    IndexJob(
                        item_type="memory",  # 任务类型: 记忆向量化
                        ref_id=str(mem.id),  # 引用ID: 记忆ID
                        # payload_json: 任务载荷(JSON格式)
                        # - 内容: {\"memory_id\": 记忆ID}
                        # - ensure_ascii=False: 保留中文字符
                        payload_json=json.dumps({"memory_id": mem.id}, ensure_ascii=False),
                        status="pending",  # 状态: 待处理
                    )
                ),
                priority=5,  # 优先级5(中等)
            )

        # ==================== 步骤10: 输出完成日志 ====================

        # logger.info(): 记录信息级别日志
        # len(persisted): 新core记忆的数量
        logger.info(f"已更新用户 {qq_id} 的核心记忆:{len(persisted)} 条")

    @staticmethod
    async def _call_llm_to_condense(
        qq_id: str,
        archive: List[Memory],
        existing_core: List[Memory],
        core_limit: int,
    ) -> Optional[List[Memory]]:
        """调用主模型从archive+core生成新的core记忆集合

        这个方法的作用:
        - 构建浓缩prompt(输入archive和core记忆)
        - 调用主LLM进行智能浓缩
        - 解析LLM返回的JSON
        - 构建新的Memory对象列表

        浓缩原理:
        - LLM分析archive和现有core记忆
        - 识别重要的、稳定的信息
        - 合并相似记忆,去除重复
        - 输出简洁、高质量的核心记忆

        Prompt设计要点:
        - 明确任务: 凝练核心长期记忆
        - 数量限制: <= core_limit条
        - 类型优先级: fact/preference/style优先
        - 内容要求: <= 40字,简短、具体、无废话
        - 输出格式: 严格JSON

        Args:
            qq_id: 用户QQ号
                - 类型: 字符串
                - 用途: 传递给LLM作为上下文
            archive: archive层记忆列表
                - 类型: List[Memory]
                - 来源: list_recent_archive()
            existing_core: 现有core记忆列表
                - 类型: List[Memory]
                - 来源: get_core_memories()
            core_limit: 核心记忆数量上限
                - 类型: 整数
                - 默认值: 5

        Returns:
            Optional[List[Memory]]: 新的核心记忆列表
                - 成功: 返回Memory对象列表
                - 失败: 返回None(LLM调用失败或JSON解析失败)

        JSON输出格式:
            {
              \"core_memories\": [
                {
                  \"type\": \"preference\",
                  \"content\": \"用户喜欢Python编程\",
                  \"confidence\": 0.9,
                  \"supporting_memory_ids\": [1, 2, 3]
                }
              ],
              \"merge_plan\": {
                \"dropped_memory_ids\": [4],
                \"updated_core_ids\": [10]
              }
            }

        Example:
            >>> archive = [Memory(content=\"用户喜欢Python\"), ...]
            >>> existing_core = [Memory(content=\"用户是程序员\"), ...]
            >>> new_core = await _call_llm_to_condense(\"123\", archive, existing_core, 5)
            >>> len(new_core)  # 3
        """

        # ==================== 步骤1: 构建输入条目列表 ====================

        # items: 所有输入记忆的文本表示
        # - 格式: \"C#记忆ID [类型] 内容前60字\" (C表示core)
        # - 格式: \"A#记忆ID [类型] 内容前60字\" (A表示archive)
        # - 目的: 让LLM知道每条记忆的来源和ID
        items: List[str] = []

        # 遍历现有core记忆
        for m in existing_core:
            # f\"C#{m.id} [{m.type}] {m.content[:60]}\": 格式化core记忆
            # - C#: 标识为core记忆
            # - m.id: 记忆ID
            # - [m.type]: 记忆类型(如\"fact\"、\"preference\")
            # - m.content[:60]: 内容前60字符
            items.append(f"C#{m.id} [{m.type}] {m.content[:60]}")

        # 遍历archive记忆
        for m in archive:
            # f\"A#{m.id} [{m.type}] {m.content[:60]}\": 格式化archive记忆
            # - A#: 标识为archive记忆
            items.append(f"A#{m.id} [{m.type}] {m.content[:60]}")

        # ==================== 步骤2: 构建浓缩prompt ====================

        # \"\\n\".join([...]): 用换行符连接prompt各部分
        prompt = "\n".join(
            [
                # 任务说明
                "你要为一个用户凝练\"核心长期记忆(core)\"。",
                # 目标和要求
                f"目标:生成 <= {core_limit} 条 core_memories,只保留稳定信息(事实/偏好/风格优先)。",
                # 输出格式说明
                "输出必须是严格 JSON,对象结构如下:",
                # JSON模板(单行,避免格式问题)
                '{"core_memories":[{"type":"preference","content":"<=40字","confidence":0.9,"supporting_memory_ids":[1,2,3]}],"merge_plan":{"dropped_memory_ids":[4],"updated_core_ids":[10]}}',
                "",  # 空行
                # 可选类型枚举
                "可选 type 枚举:fact/preference/constraint/style/relationship/goal",
                # 内容要求
                "注意:content 必须简短、具体、无废话。",
                "",  # 空行
                # 用户标识
                f"用户:{qq_id}",
                # 输入条目标题
                "输入条目:",
                # 输入条目列表(最多400条,避免prompt过长)
                # \"\\n\".join(items[:400]): 用换行符连接前400条记忆
                "\n".join(items[:400]),
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
                {"role": "system", "content": "你是记忆凝练器,只能输出 JSON。"},
                # 用户提示
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,  # 低温度
        )

        # ==================== 步骤4: 处理LLM返回值 ====================

        # not content: LLM调用失败或返回空
        if not content:
            return None  # 返回None

        # ==================== 步骤5: 解析JSON ====================

        # MemoryCondenser._extract_first_json_object(content): 从文本中提取JSON
        # - 参数: LLM返回的文本(可能包含非JSON内容)
        # - 返回: 解析后的dict/list或None
        data = MemoryCondenser._extract_first_json_object(content)

        # 类型检查: 确保是字典
        if not isinstance(data, dict):
            return None  # 格式错误

        # ==================== 步骤6: 提取core_memories数组 ====================

        # data.get(\"core_memories\"): 获取core_memories字段
        core_memories = data.get("core_memories")

        # 类型检查: 确保是列表
        if not isinstance(core_memories, list):
            return None  # 格式错误

        # ==================== 步骤7: 构建Memory对象列表 ====================

        # now_ts: 当前时间戳
        # int(time.time()): 获取Unix时间戳(秒级)
        now_ts = int(time.time())

        # results: 新的核心记忆列表
        results: List[Memory] = []

        # 遍历每条LLM返回的记忆数据
        for item in core_memories:
            # ==================== 步骤7.1: 类型检查 ====================

            if not isinstance(item, dict):  # 如果不是字典
                continue  # 跳过,处理下一条

            # ==================== 步骤7.2: 提取和规范化字段 ====================

            # str(item.get(\"type\") or \"fact\").strip(): 记忆类型
            # - 默认值: \"fact\"
            mem_type = str(item.get("type") or "fact").strip()

            # str(item.get(\"content\") or \"\").strip(): 记忆内容
            text = str(item.get("content") or "").strip()

            # 跳过空内容
            if not text:
                continue

            # 限制content长度为40字符
            # - 原因: 核心记忆应该高度凝练
            if len(text) > 40:
                text = text[:40]  # 截断

            # float(item.get(\"confidence\", 0.8)): 置信度
            # - 默认值: 0.8
            confidence = float(item.get("confidence", 0.8))

            # 限制confidence范围为[0.0, 1.0]
            # - max(0.0, ...): 不能小于0
            # - min(1.0, ...): 不能大于1
            confidence = max(0.0, min(1.0, confidence))

            # item.get(\"supporting_memory_ids\") or []: 支撑记忆ID列表
            # - 默认值: 空列表
            supporting = item.get("supporting_memory_ids") or []

            # 将supporting转为整数列表
            try:
                # [int(x) for x in supporting if str(x).isdigit()]: 列表推导式
                # - 过滤掉非数字的元素
                # - 转为整数
                supporting_ids = [int(x) for x in supporting if str(x).isdigit()]
            except Exception:
                # 转换失败,使用空列表
                supporting_ids = []

            # ==================== 步骤7.3: 创建Memory对象 ====================

            # Memory(): 创建记忆模型对象
            results.append(
                Memory(
                    qq_id=qq_id,  # 用户QQ号
                    tier="core",  # 层级: 核心层
                    type=mem_type,  # 类型: fact/preference等
                    content=text,  # 内容
                    confidence=confidence,  # 置信度
                    status="active",  # 状态: active
                    visibility="global",  # 可见性: global(核心记忆全局可见)
                    scope_scene_id=None,  # 作用域: None(全局)
                    ttl_days=None,  # 生存时间: None(永久保留)
                    # source_memory_ids: 来源记忆ID列表(JSON格式)
                    # json.dumps(supporting_ids, ensure_ascii=False): 转为JSON字符串
                    source_memory_ids=json.dumps(supporting_ids, ensure_ascii=False),
                    created_at=now_ts,  # 创建时间
                    updated_at=now_ts,  # 更新时间
                )
            )

        # ==================== 步骤8: 返回结果 ====================

        # not results: 如果没有生成任何有效记忆
        if not results:
            return None  # 返回None

        return results  # 返回新核心记忆列表

    @staticmethod
    async def _call_llm_to_merge_core(
        qq_id: str,
        core_items: List[Memory],
        core_limit: int,
    ) -> Optional[List[Memory]]:
        """当core超限时调用主模型进行合并压缩

        这个方法的作用:
        - 当LLM首次浓缩生成的core超过数量限制时调用
        - 要求LLM进行二次压缩,合并相似记忆
        - 确保最终core数量不超过上限

        为什么需要二次合并?
        - 首次浓缩可能生成过多核心记忆
        - 直接截断会丢失重要信息
        - 二次合并可以智能压缩,保留精华
        - 提高核心记忆的质量

        与_call_llm_to_condense的区别:
        - condense: 从archive+core → 新core (首次浓缩)
        - merge: 从超限core → 压缩core (二次合并)
        - merge的输入只有core,没有archive

        Args:
            qq_id: 用户QQ号
                - 类型: 字符串
            core_items: 超限的core记忆列表
                - 类型: List[Memory]
                - 来源: _call_llm_to_condense()的返回值
            core_limit: 核心记忆数量上限
                - 类型: 整数

        Returns:
            Optional[List[Memory]]: 压缩后的核心记忆列表
                - 成功: 返回Memory对象列表(数量<=core_limit)
                - 失败: 返回None

        Example:
            >>> core_items = [Memory(...), ...] # 10条core
            >>> merged = await _call_llm_to_merge_core(\"123\", core_items, 5)
            >>> len(merged)  # 5条或更少
        """

        # ==================== 步骤1: 构建输入条目列表 ====================

        # 列表推导式: 格式化每条core记忆
        # f\"- [{m.type}] {m.content[:60]}\": 格式化记忆
        # - [m.type]: 记忆类型
        # - m.content[:60]: 内容前60字符
        # [:200]: 最多200条记忆(避免prompt过长)
        lines = [f"- [{m.type}] {m.content[:60]}" for m in core_items[:200]]

        # ==================== 步骤2: 构建合并prompt ====================

        # \"\\n\".join([...]): 用换行符连接prompt各部分
        prompt = "\n".join(
            [
                # 任务说明
                f"将以下核心记忆合并提炼到 <= {core_limit} 条,只输出严格 JSON。",
                # JSON模板
                '{"core_memories":[{"type":"fact","content":"<=40字","confidence":0.9,"supporting_memory_ids":[]}]}',
                "",  # 空行
                # 用户标识
                f"用户:{qq_id}",
                # 当前core列表标题
                "当前 core:",
                # 当前core列表
                "\n".join(lines),
            ]
        )

        # ==================== 步骤3: 调用主LLM ====================

        # await main_llm.chat_completion(): 调用主LLM
        # temperature=0.2: 低温度,确定性输出
        content = await main_llm.chat_completion(
            [
                # 系统提示
                {"role": "system", "content": "你是记忆合并器,只能输出 JSON。"},
                # 用户提示
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        # ==================== 步骤4: 处理LLM返回值 ====================

        # not content: LLM调用失败
        if not content:
            return None

        # ==================== 步骤5: 解析JSON ====================

        # _extract_first_json_object(content): 提取JSON
        data = MemoryCondenser._extract_first_json_object(content)

        # 类型检查
        if not isinstance(data, dict):
            return None

        # ==================== 步骤6: 提取core_memories数组 ====================

        core_memories = data.get("core_memories")

        # 类型检查
        if not isinstance(core_memories, list):
            return None

        # ==================== 步骤7: 构建Memory对象列表 ====================

        now_ts = int(time.time())
        results: List[Memory] = []

        # 遍历每条LLM返回的记忆数据
        for item in core_memories:
            # 类型检查
            if not isinstance(item, dict):
                continue

            # 提取和规范化字段
            mem_type = str(item.get("type") or "fact").strip()
            text = str(item.get("content") or "").strip()

            # 跳过空内容
            if not text:
                continue

            # 限制content长度为40字符
            if len(text) > 40:
                text = text[:40]

            # 置信度
            confidence = float(item.get("confidence", 0.8))
            confidence = max(0.0, min(1.0, confidence))

            # 创建Memory对象
            results.append(
                Memory(
                    qq_id=qq_id,
                    tier="core",
                    type=mem_type,
                    content=text,
                    confidence=confidence,
                    status="active",
                    visibility="global",
                    scope_scene_id=None,
                    ttl_days=None,
                    source_memory_ids="[]",  # 二次合并后来源清空
                    created_at=now_ts,
                    updated_at=now_ts,
                )
            )

        # ==================== 步骤8: 返回结果 ====================

        # not results: 没有生成任何有效记忆
        if not results:
            return None

        # results[:core_limit]: 截断到上限
        # - 保险措施: 即使LLM返回超限,也强制截断
        return results[:core_limit]

    @staticmethod
    def _extract_first_json_object(text: str) -> Optional[object]:
        """从文本中提取第一个JSON对象或数组

        这个方法的作用:
        - 从LLM返回的文本中提取JSON部分
        - 支持对象{}和数组[]两种格式
        - 处理LLM输出中的多余文本

        为什么需要这个方法?
        - LLM可能输出额外内容: \"这是浓缩的记忆: {...}\"
        - 需要正则匹配提取纯JSON部分
        - 提高解析成功率

        Args:
            text: LLM返回的文本
                - 类型: 字符串
                - 可能包含: JSON + 非JSON内容

        Returns:
            Optional[object]: 解析后的Python对象
                - 成功: dict或list
                - 失败: None

        Example:
            >>> text = \"这是结果: {\\\"core_memories\\\":[]}\"
            >>> obj = MemoryCondenser._extract_first_json_object(text)
            >>> print(obj)  # {\"core_memories\": []}
        """

        # ==================== 步骤1: 去除首尾空格 ====================
        s = text.strip()

        # ==================== 步骤2: 正则匹配JSON部分 ====================

        # re.search(): 搜索匹配
        # r\"(\\{.*\\}|\\[.*\\])\": 正则表达式
        # - (\\{.*\\}): 匹配对象 {...}
        # - |: 或
        # - (\\[.*\\]): 匹配数组 [...]
        # - .*: 匹配任意字符(贪婪模式)
        # flags=re.DOTALL: DOTALL模式,让.匹配换行符
        # - 支持多行JSON
        m = re.search(r"(\{.*\}|\[.*\])", s, flags=re.DOTALL)

        # not m: 没有匹配到JSON
        if not m:
            return None

        # ==================== 步骤3: 解析JSON ====================

        try:
            # m.group(1): 获取第一个捕获组(JSON字符串)
            # json.loads(): 解析JSON字符串为Python对象
            return json.loads(m.group(1))
        except Exception:
            # 解析失败: JSON格式错误
            return None
