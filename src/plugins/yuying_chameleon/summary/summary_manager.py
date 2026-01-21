"""对话摘要管理模块 - 滚动窗口摘要生成和状态管理

这个模块的作用:
1. 实现滚动窗口摘要机制(按消息数或时间触发)
2. 调用LLM生成结构化的对话摘要
3. 管理每个场景(群/私聊)的摘要窗口状态
4. 提供完整的降级策略,确保LLM不可用时也能生成摘要

对话摘要原理(新手必读):
- 问题: 历史消息太多,无法全部传给LLM
- 解决: 定期将历史对话压缩为简短摘要
- 流程: 积累消息 → 达到阈值 → LLM生成摘要 → 存入数据库 → 清空窗口
- 好处: 节省LLM上下文,保留对话连贯性

滚动窗口机制:
- 窗口: 从window_start_ts到当前时间的消息集合
- 触发条件(满足任一即可):
  1. 消息数触发: 窗口内消息数 >= N条(默认20条)
  2. 时间触发: 窗口时长 >= N秒(默认900秒=15分钟)
- 触发后: 生成摘要,清空窗口,重新计数
- 好处: 平衡摘要的实时性和完整性

摘要结构(5段固定格式):
1. 话题: 对话的主要话题
2. 关键事实: 对话中的重要信息
3. 结论: 对话得出的结论
4. 未解问题: 还没解决的问题
5. 下一步: 后续计划或待办事项

使用场景:
- RAG检索: 摘要作为远期上下文补充
- 对话理解: 帮助LLM快速了解历史话题
- 数据压缩: 减少数据库和向量库的存储压力
- 用户回顾: 用户可以查看历史对话摘要

降级策略:
- LLM失败: 生成固定格式的降级摘要
- 格式不符: 检查关键字段,不符合则降级
- 消息为空: 重置窗口,不生成摘要
- 好处: 即使LLM故障,摘要系统仍可运行

使用方式:
```python
from .summary.summary_manager import SummaryManager

# 每次收到新消息时调用
summary = await SummaryManager.on_message(
    scene_type=\"group\",
    scene_id=\"123456\",
    _msg_id=101,
    msg_ts=1234567890
)

if summary:
    print(f\"生成了新摘要: {summary.summary_text}\")
else:
    print(\"窗口未触发,继续积累消息\")
```

配置项:
- yuying_summary_window_message_count: 消息数触发阈值(默认20条)
- yuying_summary_window_seconds: 时间触发阈值(默认900秒)
"""

from __future__ import annotations

import json  # Python标准库,用于JSON编解码

from nonebot import logger  # NoneBot日志记录器
from typing import Optional  # 类型提示

# 导入项目模块
from ..config import plugin_config  # 插件配置
from ..llm.client import get_task_llm  # 支持模型组回落
from ..storage.models import IndexJob, Summary  # 数据库模型
from ..storage.db_writer import db_writer  # 数据库写入队列
from ..storage.write_jobs import AddIndexJobJob, AsyncCallableJob  # 写入任务
from ..storage.repositories.raw_repo import RawRepository  # 原始消息仓库
from ..storage.repositories.summary_repo import SummaryRepository  # 摘要仓库
from .summary_state import summary_state_store  # 摘要状态存储


class SummaryManager:
    """对话摘要管理器 - 维护每个场景的滚动窗口并生成摘要

    这个类的作用:
    - 管理每个场景(群/私聊)的摘要窗口状态
    - 检测窗口触发条件(消息数或时间)
    - 调用LLM生成结构化摘要
    - 处理摘要的存储和索引

    设计模式:
    - 静态方法类: 所有方法都是静态的,无需实例化
    - 好处: 作为全局服务使用,避免状态管理

    核心流程:
    1. on_message()在每条新消息时调用
    2. 检查窗口是否达到触发条件
    3. 如果触发: 查询窗口内消息 → 调用LLM → 存储摘要 → 重置窗口
    4. 如果未触发: 增加计数,继续积累

    与summary_state的协作:
    - summary_state_store: 全局状态存储,记录每个场景的窗口状态
    - bump(): 增加消息计数,更新时间
    - reset(): 重置窗口,开始新的积累周期

    配置项:
    - yuying_summary_window_message_count: 消息数阈值
    - yuying_summary_window_seconds: 时间阈值
    """

    @staticmethod
    async def on_message(
        scene_type: str,
        scene_id: str,
        _msg_id: int,
        msg_ts: int,
    ) -> Optional[Summary]:
        """在新消息到达后推进摘要窗口,必要时生成摘要

        这个方法的作用:
        - 作为摘要系统的入口点,每条新消息都会调用
        - 更新场景的窗口状态(消息计数+1)
        - 检查是否达到触发条件(消息数或时间)
        - 如果触发: 生成摘要并重置窗口
        - 如果未触发: 继续积累

        触发条件(满足任一即可):
        1. 消息数触发: message_count >= window_message_count
        2. 时间触发: (当前时间 - window_start_ts) >= window_seconds

        生成流程:
        1. 查询窗口内的所有消息(最多300条)
        2. 调用generate_summary()生成摘要文本
        3. 创建Summary记录并存入数据库
        4. 创建向量化任务(IndexJob)
        5. 重置窗口状态,开始新周期

        Args:
            scene_type: 场景类型
                - 类型: 字符串
                - 取值: \"group\" 或 \"private\"
                - 用途: 标识是群聊还是私聊
            scene_id: 场景标识
                - 类型: 字符串
                - 内容: 群号或QQ号
                - 用途: 标识具体的场景
            _msg_id: 最新消息ID
                - 类型: 整数
                - 来源: RawMessage.id
                - 注意: 参数名前缀_表示未使用
            msg_ts: 最新消息时间戳
                - 类型: 整数(Unix时间戳,秒级)
                - 用途: 判断时间触发条件

        Returns:
            Optional[Summary]: 生成的摘要对象
                - 触发窗口: 返回新生成的Summary对象
                - 未触发: 返回None
                - 消息为空: 返回None并重置窗口

        Side Effects:
            - 更新summary_state_store的窗口状态
            - 写入Summary表
            - 创建IndexJob任务
            - 输出日志信息

        Example:
            >>> # 第1-19条消息: 未触发,返回None
            >>> summary = await SummaryManager.on_message(
            ...     \"group\", \"123\", 101, 1234567890
            ... )
            >>> print(summary)  # None

            >>> # 第20条消息: 触发,返回Summary对象
            >>> summary = await SummaryManager.on_message(
            ...     \"group\", \"123\", 120, 1234567900
            ... )
            >>> print(summary.summary_text)  # \"话题:讨论Python...\"
        """

        # ==================== 步骤1: 更新窗口状态并检查触发条件 ====================

        # summary_state_store.bump(): 推进窗口状态
        # - 参数: scene_type, scene_id, now_ts(当前时间)
        # - 功能: message_count += 1, 更新last_message_ts
        # - 返回: SummaryState对象(包含window_start_ts, message_count等)
        state = summary_state_store.bump(scene_type, scene_id, now_ts=msg_ts)

        # ==================== 步骤2: 判断消息数触发条件 ====================

        # state.message_count >= 阈值: 消息数是否达到触发条件
        # int(plugin_config.yuying_summary_window_message_count): 消息数阈值
        # - 默认值: 20条
        window_by_count = state.message_count >= int(plugin_config.yuying_summary_window_message_count)

        # ==================== 步骤3: 判断时间触发条件 ====================

        # (msg_ts - state.window_start_ts) >= 阈值: 窗口时长是否达到触发条件
        # - msg_ts: 当前消息时间
        # - state.window_start_ts: 窗口起始时间
        # - 差值: 窗口的时长(秒)
        # int(plugin_config.yuying_summary_window_seconds): 时间阈值
        # - 默认值: 900秒(15分钟)
        window_by_time = (msg_ts - state.window_start_ts) >= int(plugin_config.yuying_summary_window_seconds)

        # ==================== 步骤4: 检查是否触发 ====================

        # not (window_by_count or window_by_time): 两个条件都不满足
        # - 原因: 窗口还没积累足够的消息或时间
        # - 处理: 返回None,继续积累
        if not (window_by_count or window_by_time):
            return None  # 未触发,继续积累消息

        # ==================== 步骤5: 记录窗口范围 ====================

        # window_start: 窗口起始时间
        # state.window_start_ts: 从状态中读取
        window_start = state.window_start_ts

        # window_end: 窗口结束时间
        # msg_ts: 当前消息的时间戳
        window_end = msg_ts

        # ==================== 步骤6: 查询窗口内的所有消息 ====================

        # await RawRepository.get_messages_by_scene_time_range(): 按时间范围查询消息
        # - 参数:
        #   * scene_type: 场景类型
        #   * scene_id: 场景标识
        #   * start_ts: 起始时间
        #   * end_ts: 结束时间
        #   * limit: 最多查询300条(避免过多)
        # - SQL: WHERE scene_type=? AND scene_id=? AND timestamp BETWEEN start AND end
        # - 返回: RawMessage对象列表
        messages = await RawRepository.get_messages_by_scene_time_range(
            scene_type,
            scene_id,
            start_ts=window_start,
            end_ts=window_end,
            limit=300,  # 限制最多300条消息
        )

        # ==================== 步骤7: 处理消息为空的情况 ====================

        # not messages: 查询结果为空
        # - 可能原因: 窗口内的消息已被清理或数据库异常
        if not messages:
            # 重置窗口状态,开始新的积累周期
            # summary_state_store.reset(): 重置状态
            # - 功能: window_start_ts=now_ts, message_count=0
            summary_state_store.reset(scene_type, scene_id, now_ts=msg_ts)
            return None  # 无消息可摘要,返回None

        # ==================== 步骤8: 调用LLM生成摘要 ====================

        # await SummaryManager.generate_summary(): 生成摘要文本
        # - 参数: [m.content for m in messages] - 消息内容列表
        #   * 列表推导式: 提取所有消息的content字段
        # - 返回: 摘要文本(字符串)
        summary_text = await SummaryManager.generate_summary([m.content for m in messages])

        # ==================== 步骤9: 创建Summary记录 ====================

        # Summary(): 创建摘要模型对象
        record = Summary(
            scene_type=scene_type,  # 场景类型
            scene_id=scene_id,  # 场景标识
            window_start_ts=window_start,  # 窗口起始时间
            window_end_ts=window_end,  # 窗口结束时间
            summary_text=summary_text,  # 摘要文本
            topic_state_json=None,  # 话题状态(可选,暂未使用)
        )

        # ==================== 步骤10: 存储摘要到数据库 ====================

        # await db_writer.submit_and_wait(): 提交写入任务并等待完成
        # AsyncCallableJob: 异步可调用任务
        # SummaryRepository.add(record): 插入摘要记录
        # - 操作: INSERT INTO summaries VALUES (...)
        # - 返回: 插入后的Summary对象(带自动生成的id)
        created = await db_writer.submit_and_wait(
            AsyncCallableJob(SummaryRepository.add, args=(record,)),
            priority=5,  # 优先级5(中等)
        )
        if not isinstance(created, Summary):
            return None
        record = created

        # ==================== 步骤11: 创建向量化任务(索引双写) ====================

        # 为新摘要创建向量化任务
        # - 目的: 将摘要文本向量化,存入Qdrant
        # - 用途: 支持基于向量的摘要检索

        # payload: 任务载荷
        # {\"summary_id\": 摘要ID}
        payload = {"summary_id": record.id}

        # await db_writer.submit(): 提交写入任务(不等待)
        # AddIndexJobJob: 添加索引任务
        await db_writer.submit(
            AddIndexJobJob(
                IndexJob(
                    item_type="summary",  # 任务类型: 摘要向量化
                    ref_id=str(record.id),  # 引用ID: 摘要ID
                    # payload_json: 任务载荷(JSON格式)
                    # json.dumps(payload, ensure_ascii=False): 转为JSON字符串
                    payload_json=json.dumps(payload, ensure_ascii=False),
                    status="pending",  # 状态: 待处理
                )
            ),
            priority=5,  # 优先级5(中等)
        )

        # ==================== 步骤12: 重置窗口状态 ====================

        # summary_state_store.reset(): 重置窗口状态
        # - 功能: window_start_ts=now_ts, message_count=0
        # - 效果: 开始新的积累周期
        summary_state_store.reset(scene_type, scene_id, now_ts=msg_ts)

        # ==================== 步骤13: 输出完成日志 ====================

        # logger.info(): 记录信息级别日志
        logger.info(f"已生成摘要:{scene_type}:{scene_id} summary_id={record.id}")

        # ==================== 步骤14: 返回摘要对象 ====================

        return record  # 返回新生成的摘要

    @staticmethod
    async def generate_summary(messages: list[str]) -> str:
        """生成结构化摘要文本(5段固定格式)

        这个方法的作用:
        - 调用主LLM生成对话摘要
        - 限制摘要为5段固定格式
        - 提供降级策略(LLM失败时生成默认摘要)
        - 截断过长的摘要文本

        摘要格式(5段):
        1. 话题: 对话的主要话题
        2. 关键事实: 对话中的重要信息和数据
        3. 结论: 对话得出的结论或共识
        4. 未解问题: 还没有解决的问题或疑问
        5. 下一步: 后续计划、待办事项或行动

        为什么需要固定格式?
        - 结构化: 便于程序解析和提取信息
        - 一致性: 所有摘要格式统一,便于阅读
        - 完整性: 5段覆盖对话的主要维度
        - 可解释: 明确各部分的含义

        Args:
            messages: 窗口内消息文本列表
                - 类型: List[str]
                - 来源: [m.content for m in RawMessage列表]
                - 内容: 归一化后的消息文本(可能包含[image:xxx]占位符)

        Returns:
            str: 结构化摘要文本
                - 成功: LLM生成的5段摘要(最多2000字符)
                - 失败: 降级摘要(固定格式)

        降级策略:
            - LLM调用失败: 生成降级摘要
            - 格式不符: 检查是否包含\"话题\"和\"关键事实\",不包含则降级
            - 降级摘要内容: 固定5段,标注\"降级\"

        Example:
            >>> messages = [\"今天天气真好\", \"是啊,适合出去玩\", \"那我们去公园吧\"]
            >>> summary = await SummaryManager.generate_summary(messages)
            >>> print(summary)
            # 话题:讨论天气和出行计划
            # 关键事实:用户认为天气好适合出去玩
            # 结论:决定去公园
            # 未解问题:具体时间未定
            # 下一步:确定出发时间
        """

        # ==================== 步骤1: 格式化消息列表 ====================

        # \"\\n\".join([f\"- {m}\" for m in messages[-60:]]): 拼接消息
        # - messages[-60:]: 取最后60条消息(避免prompt过长)
        #   * [-60:]: Python切片,取列表的最后60个元素
        #   * 如果消息不足60条,则全部使用
        # - f\"- {m}\": 格式化每条消息,前面加\"- \"(markdown列表格式)
        # - \"\\n\".join(...): 用换行符连接所有消息
        # 示例输出:
        #   - 今天天气真好
        #   - 是啊,适合出去玩
        #   - 那我们去公园吧
        joined = "\n".join([f"- {m}" for m in messages[-60:]])

        # ==================== 步骤2: 构建摘要prompt ====================

        # \"\\n\".join([...]): 用换行符连接prompt各部分
        prompt = "\n".join(
            [
                # 任务说明
                "为以下聊天记录生成结构化摘要。",
                # 格式要求
                "输出格式固定为 5 段,每段一行:",
                "话题:...",
                "关键事实:...",
                "结论:...",
                "未解问题:...",
                "下一步:...",
                "",  # 空行
                # 聊天记录标题
                "聊天记录:",
                # 聊天记录内容
                joined,
            ]
        )

        # ==================== 步骤3: 调用主LLM生成摘要 ====================

        # await main_llm.chat_completion(): 调用主LLM
        # 参数:
        # - messages: 对话消息列表(OpenAI格式)
        # - temperature=0.2: 低温度,生成确定性输出
        #   * 0.2比默认0.7更低,适合摘要任务
        #   * 确保输出格式稳定
        llm = get_task_llm("summary_generation")
        content = await llm.chat_completion(
            [
                # 系统提示
                {"role": "system", "content": "你是一个摘要生成器，根据用户提供的格式和聊天记录生成对应格式的摘要，其余什么都不要输出。输出示例：话题:讨论天气和出行计划\n关键事实:用户认为天气好适合出去玩\n结论:决定去公园\n未解问题:具体时间未定\n下一步:确定出发时间"},
                # 用户提示
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,  # 低温度
        )

        # ==================== 步骤4: 处理LLM返回值 ====================

        if content:  # LLM调用成功且返回非空内容
            # content.strip(): 去除首尾空格
            text = content.strip()

            # ==================== 步骤4.1: 验证格式 ====================

            # \"话题\" in text and \"关键事实\" in text: 检查关键字段
            # - 目的: 验证LLM是否按照要求的格式输出
            # - 如果包含这两个字段,认为格式正确
            if "话题" in text and "关键事实" in text:
                # text[:2000]: 截断到2000字符
                # - 原因: 限制摘要长度,避免过长
                # - 2000字符约1000个中文字
                return text[:2000]  # 返回成功生成的摘要

        # ==================== 步骤5: 降级处理(LLM失败或格式不符) ====================

        # 当LLM不可用或输出格式不符时,生成固定格式的降级摘要
        # - 目的: 确保摘要系统始终可用
        # - 内容: 固定5段,标注\"降级\"

        # \"\\n\".join([...]): 用换行符连接5段内容
        compressed = "\n".join(
            [
                # 第1段: 话题(标注降级)
                "话题:聊天摘要(降级)",
                # 第2段: 关键事实(显示消息数)
                # min(len(messages), 60): 取消息数和60的最小值
                # - 原因: 最多只用了最后60条消息
                f"关键事实:窗口内消息 {min(len(messages), 60)} 条",
                # 第3段: 结论(暂未生成)
                "结论:暂未生成",
                # 第4段: 未解问题(暂无)
                "未解问题:暂无",
                # 第5段: 下一步(继续观察)
                "下一步:继续观察对话",
            ]
        )

        return compressed  # 返回降级摘要
