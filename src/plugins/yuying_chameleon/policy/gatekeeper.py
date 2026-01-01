"""消息门控策略模块 - 回复决策、冷却机制、刷屏检测

这个模块的作用:
1. 决定机器人是否回复某条消息(should_reply)
2. 实现冷却机制(cooldown),避免过度回复
3. 刷屏检测,消息密度高时降低回复概率
4. 动态调整冷却时间,根据场景活跃度自适应
5. 标记发送(mark_sent),更新冷却状态

门控策略原理(新手必读):
- 问题: 机器人不能对每条消息都回复,会刷屏
- 解决: 通过策略控制回复频率和时机
- 流程: 消息到达 → 策略判断 → 决定是否回复 → 发送后更新冷却
- 好处: 避免刷屏,自然融入对话,节省LLM调用

冷却机制(Cooldown):
- 概念: 发送一次后进入冷却期,期间不再回复
- 基础冷却: 全局冷却(默认10秒) + 场景冷却(群聊默认30秒)
- 动态冷却: 刷屏时冷却时间翻倍(base * 2)
- 实现: rate_limits表记录cooldown_until_ts
- 好处: 避免连续回复,保持对话节奏

刷屏检测(Spam Detection):
- 检测窗口: 默认60秒内
- 消息阈值: 默认20条
- 触发效果:   1. 回复概率降低75%(prob *= 0.25)
  2. 冷却时间翻倍(base * 2)
- 好处: 在高强度对话时降低机器人的参与度

概率回复(Probability-based Reply):
- 群聊: 默认概率0.3(30%),不必每条都回
- 私聊: 默认概率1.0(100%),通常都回
- 调整: 刷屏时降低到原概率的25%
- 实现: random.random() < prob
- 好处: 让机器人的参与更加自然

必回情况(Bypass):
- @机器人: mentioned_bot=True时必然回复
- 私聊: 默认概率1.0,通常都回
- 好处: 主动呼叫机器人时保证响应

与其他模块的协作:
- normalizer: 提供归一化后的消息(mentioned_bot标记)
- RateLimitRepository: 读写冷却状态(rate_limits表)
- RawRepository: 查询消息密度(刷屏检测)
- action_planner: should_reply()返回True后调用规划
- action_sender: 发送消息后调用mark_sent()更新冷却

使用方式:
```python
from .policy.gatekeeper import Gatekeeper

# 1. 判断是否应该回复
should = await Gatekeeper.should_reply(
    scene_type="group",
    scene_id="123456",
    _msg_content="今天天气真好",
    mentioned_bot=False
)

if should:
    # 2. 规划和发送回复
    actions = await ActionPlanner.plan(...)
    await ActionSender.execute(...)

    # 3. 发送后标记,更新冷却
    await Gatekeeper.mark_sent("group", "123456")
```

配置项:
- yuying_global_cooldown_seconds: 全局基础冷却(默认10秒)
- yuying_group_cooldown_seconds: 群聊基础冷却(默认30秒)
- yuying_group_reply_probability: 群聊回复概率(默认0.3)
- yuying_private_reply_probability: 私聊回复概率(默认1.0)
- yuying_spam_window_seconds: 刷屏检测窗口(默认60秒)
- yuying_spam_msg_threshold: 刷屏消息阈值(默认20条)
"""

from __future__ import annotations

import asyncio  # Python异步编程标准库，用于 Semaphore 并发限制
import random  # Python标准库,随机数生成
import time  # Python标准库,时间戳
import unicodedata  # Unicode类别判断（用于"纯表情"短路规则）

from nonebot import logger  # NoneBot日志记录器

# 导入项目模块
from ..config import plugin_config  # 插件配置
from ..llm.flow_decider import nano_should_reply_yes_no  # 心流模式：nano yes/no 判定
from ..storage.repositories.rate_limit_repo import RateLimitRepository  # 冷却状态仓库
from ..storage.repositories.raw_repo import RawRepository  # 原始消息仓库
from ..storage.db_writer import db_writer  # 数据库写入队列
from ..storage.write_jobs import AsyncCallableJob  # 异步任务


class Gatekeeper:
    """回复决策器 - 策略控制机器人的回复行为

    这个类的作用:
    - 决定机器人是否回复某条消息
    - 管理冷却状态,避免过度回复
    - 检测刷屏,动态调整回复策略
    - 提供概率回复机制
    - 支持心流模式（使用 nano 模型智能决策）

    设计模式:
    - 静态方法类: 所有方法都是静态的,无需实例化
    - 策略模式: 封装回复决策逻辑
    - 好处: 集中管理策略,易于调整和扩展

    核心决策流程(传统模式):
    1. should_reply()判断是否回复
       - @机器人: 必回
       - 冷却中: 不回
       - 概率判断: 群聊30%(可调),私聊100%
       - 刷屏降级: 概率降低75%
    2. mark_sent()发送后更新冷却
       - 计算动态冷却时间
       - 更新rate_limits表

    核心决策流程(心流模式):
    1. should_reply()判断是否回复
       - 冷却中: 不回
       - @机器人/私聊/明确指向: 必回（受冷却）
       - 短路规则: 纯表情/过短→不回，明显提问→回
       - 概率抽样: 决定是否调用 nano
       - nano 判定: 根据上下文智能决策
    2. mark_sent()同传统模式

    Example:
        >>> # 判断是否回复
        >>> should = await Gatekeeper.should_reply(
        ...     "group", "123456", "今天天气真好", mentioned_bot=False
        ... )
        >>> print(should)
        # True 或 False (基于概率和冷却状态)
    """

    # ==================== 心流模式：nano 并发限制 ====================
    # 目的: 限制同时发起的 nano LLM 请求数量，避免瞬时高峰导致延迟飙升/触发限流
    # 策略: 拿不到令牌则直接返回 False（保守，不刷屏）
    # 位置: 类变量（全进程共享）
    _FLOW_MODE_NANO_SEMAPHORE = asyncio.Semaphore(3)

    @staticmethod
    def _flow_mode_enabled() -> bool:
        """判断是否启用心流模式。

        Returns:
            bool: True 表示启用心流模式
        """
        return bool(plugin_config.yuying_enable_flow_mode)

    @staticmethod
    def _cooldown_seconds(scene_type: str) -> int:
        """计算当前场景应使用的基础冷却秒数

        这个方法的作用:
        - 根据场景类型(群聊/私聊)计算基础冷却时间
        - 取全局冷却和场景冷却的较大值
        - 用于后续的动态调整

        冷却时间计算规则:
        - 全局冷却: yuying_global_cooldown_seconds(默认10秒)
        - 群聊冷却: yuying_group_cooldown_seconds(默认30秒)
        - 私聊冷却: 仅使用全局冷却(默认10秒)
        - 最终取值: max(全局冷却, 场景冷却)

        为什么群聊冷却更长?
        - 群聊消息多: 更容易刷屏,需要更长冷却
        - 私聊专注: 一对一对话,可以更频繁回复
        - 自然性: 群聊中频繁回复显得不自然

        Args:
            scene_type: 场景类型
                - 类型: 字符串
                - 取值: "group"(群聊) 或 "private"(私聊)
                - 用途: 决定使用哪个冷却配置

        Returns:
            int: 基础冷却秒数
                - 群聊: max(全局冷却, 群聊冷却) (通常为30秒)
                - 私聊: 全局冷却 (通常为10秒)

        Example:
            >>> # 群聊冷却时间
            >>> cd = Gatekeeper._cooldown_seconds("group")
            >>> print(cd)
            # 30 (max(10, 30))

            >>> # 私聊冷却时间
            >>> cd = Gatekeeper._cooldown_seconds("private")
            >>> print(cd)
            # 10 (仅全局冷却)
        """

        # ==================== 步骤1: 读取全局冷却配置 ====================

        # int(plugin_config.yuying_global_cooldown_seconds): 全局基础冷却
        # - 配置项: yuying_global_cooldown_seconds
        # - 默认值: 10秒
        # - 适用于: 所有场景(群聊和私聊)
        global_cd = int(plugin_config.yuying_global_cooldown_seconds)

        # ==================== 步骤2: 根据场景类型返回冷却时间 ====================

        # scene_type == "group": 群聊场景
        if scene_type == "group":
            # ==================== 群聊: 取全局和群聊冷却的较大值 ====================

            # int(plugin_config.yuying_group_cooldown_seconds): 群聊基础冷却
            # - 配置项: yuying_group_cooldown_seconds
            # - 默认值: 30秒
            # - 适用于: 仅群聊场景
            group_cd = int(plugin_config.yuying_group_cooldown_seconds)

            # max(global_cd, group_cd): 取两者的较大值
            # - 原因: 确保群聊的冷却时间不低于全局冷却
            # - 示例: max(10, 30) = 30
            return max(global_cd, group_cd)

        # ==================== 私聊: 仅返回全局冷却 ====================

        # return global_cd: 私聊只使用全局冷却
        # - 原因: 私聊对话更专注,不需要额外的冷却
        # - 默认: 10秒
        return global_cd

    @staticmethod
    def _flow_mode_cooldown_seconds(scene_type: str) -> int:
        """心流模式下的基础冷却秒数（与传统模式分离）。

        Args:
            scene_type: 场景类型（group/private）

        Returns:
            int: 心流模式下的基础冷却秒数
                - 群聊: max(全局冷却, 群聊冷却)
                - 私聊: 全局冷却
        """
        flow_global_cd = int(plugin_config.yuying_flow_mode_global_cooldown_seconds)
        if scene_type == "group":
            flow_group_cd = int(plugin_config.yuying_flow_mode_group_cooldown_seconds)
            return max(flow_global_cd, flow_group_cd)
        return flow_global_cd

    @staticmethod
    def _base_cooldown_seconds(scene_type: str) -> int:
        """根据是否启用心流模式选择基础冷却配置。

        Args:
            scene_type: 场景类型（group/private）

        Returns:
            int: 当前模式下应使用的基础冷却秒数
        """
        if Gatekeeper._flow_mode_enabled():
            return Gatekeeper._flow_mode_cooldown_seconds(scene_type)
        return Gatekeeper._cooldown_seconds(scene_type)

    @staticmethod
    async def _dynamic_cooldown_seconds(scene_type: str, scene_id: str) -> int:
        """根据消息密度动态调整冷却秒数 - 刷屏时翻倍

        这个方法的作用:
        - 检测场景的消息密度(刷屏检测)
        - 如果消息密度高(刷屏),冷却时间翻倍
        - 返回动态调整后的冷却时间

        刷屏检测逻辑:
        1. 检测窗口: yuying_spam_window_seconds(默认60秒)
        2. 消息阈值: yuying_spam_msg_threshold(默认20条)
        3. 查询: 窗口内消息数量 >= 阈值?
        4. 是: 冷却时间 = 基础冷却 * 2
        5. 否: 冷却时间 = 基础冷却

        为什么刷屏时冷却翻倍?
        - 减少参与: 高强度对话时降低机器人的参与度
        - 避免打断: 让人类用户充分讨论
        - 节省资源: 减少LLM调用和数据库写入
        - 自然性: 避免在热烈讨论时频繁插话

        Args:
            scene_type: 场景类型
                - 类型: 字符串
                - 取值: "group" 或 "private"
            scene_id: 场景标识
                - 类型: 字符串
                - 内容: 群号或QQ号

        Returns:
            int: 动态调整后的冷却秒数
                - 正常: 基础冷却(10-30秒)
                - 刷屏: 基础冷却 * 2(20-60秒)

        异常处理:
            - 数据库查询失败: 返回基础冷却(不影响主流程)

        Example:
            >>> # 正常情况(窗口内15条消息,未达阈值20)
            >>> cd = await Gatekeeper._dynamic_cooldown_seconds("group", "123456")
            >>> print(cd)
            # 30 (基础冷却,未翻倍)

            >>> # 刷屏情况(窗口内25条消息,超过阈值20)
            >>> cd = await Gatekeeper._dynamic_cooldown_seconds("group", "123456")
            >>> print(cd)
            # 60 (基础冷却30 * 2)
        """

        # ==================== 步骤1: 获取基础冷却时间 ====================

        # Gatekeeper._base_cooldown_seconds(scene_type): 计算基础冷却（传统/心流 自动切换）
        # - 返回: 群聊30秒(默认)或私聊10秒(默认)，心流模式使用独立配置
        base = Gatekeeper._base_cooldown_seconds(scene_type)

        # ==================== 步骤2: 读取刷屏检测配置 ====================

        # int(plugin_config.yuying_spam_window_seconds): 检测窗口(秒)
        # - 配置项: yuying_spam_window_seconds
        # - 默认值: 60秒
        # - 用途: 统计过去N秒内的消息数量
        window = int(plugin_config.yuying_spam_window_seconds)

        # int(plugin_config.yuying_spam_msg_threshold): 消息阈值(条)
        # - 配置项: yuying_spam_msg_threshold
        # - 默认值: 20条
        # - 用途: 窗口内消息数 >= 阈值 → 判定为刷屏
        threshold = int(plugin_config.yuying_spam_msg_threshold)

        # ==================== 步骤3: 检查配置有效性 ====================

        # window <= 0 or threshold <= 0: 配置无效或禁用
        # - 原因: 窗口或阈值为0表示禁用刷屏检测
        if window <= 0 or threshold <= 0:
            return base  # 禁用刷屏检测,直接返回基础冷却

        # ==================== 步骤4: 计算检测窗口的起始时间 ====================

        # int(time.time()): 获取当前Unix时间戳(秒级)
        now_ts = int(time.time())

        # now_ts - window: 窗口起始时间
        # - 示例: 当前时间1000秒,窗口60秒 → 起始时间940秒
        since_ts = now_ts - window

        # ==================== 步骤5: 查询窗口内消息数量 ====================

        try:
            # await RawRepository.count_scene_messages_since(): 统计消息数
            # - 参数: scene_type, scene_id, since_ts(起始时间)
            # - 查询: SELECT COUNT(*) FROM raw_messages
            #         WHERE scene_type=? AND scene_id=? AND timestamp >= ?
            # - 返回: 消息数量(整数)
            count = await RawRepository.count_scene_messages_since(scene_type, scene_id, since_ts)

        except Exception:
            # ==================== 查询失败: 返回基础冷却 ====================

            # 捕获所有异常,不影响主流程
            # - 原因: 刷屏检测是辅助功能,不应阻塞回复决策
            return base  # 返回基础冷却

        # ==================== 步骤6: 判定是否刷屏并调整冷却 ====================

        # count >= threshold: 消息数量达到阈值
        if count >= threshold:
            # ==================== 刷屏: 冷却时间翻倍 ====================

            # int(base * 2): 基础冷却 * 2
            # - 示例: 30秒 * 2 = 60秒
            # - 效果: 在刷屏场景降低回复频率
            return int(base * 2)

        # ==================== 正常: 返回基础冷却 ====================

        return base

    @staticmethod
    def _looks_like_question(text: str) -> bool:
        """短路规则: 检测消息是否明显是提问

        检测规则:
        - 包含问号(中英文): ? ？
        - 包含常见疑问词开头: 什么/怎么/为什么/如何/能不能/可以/哪里等

        Args:
            text: 消息文本

        Returns:
            bool: True表示明显是提问
        """
        t = (text or "").strip()
        if not t:
            return False

        # 包含问号
        if "?" in t or "？" in t:
            return True

        # 常见疑问词开头
        question_words = [
            "什么", "怎么", "为什么", "如何", "能不能", "可以", "可不可以",
            "哪里", "哪个", "哪些", "谁", "几", "多少", "是不是", "对不对",
            "行不行", "好不好", "会不会", "有没有", "要不要"
        ]
        for word in question_words:
            if t.startswith(word):
                return True

        return False

    @staticmethod
    def _is_emoji_only(text: str) -> bool:
        """短路规则: 检测消息是否仅包含表情符号

        检测规则:
        - 去除所有Emoji/符号/标点后是否还有实质内容
        - 使用Unicode类别判断

        Args:
            text: 消息文本

        Returns:
            bool: True表示仅包含表情/符号
        """
        t = (text or "").strip()
        if not t:
            return True

        # 检查是否有实质字符(字母/数字/汉字等)
        has_content = False
        for char in t:
            cat = unicodedata.category(char)
            # L*: 字母, N*: 数字, M*: 标记
            if cat.startswith(('L', 'N', 'M')):
                has_content = True
                break

        return not has_content

    @staticmethod
    def _is_too_short_no_question_mark(text: str) -> bool:
        """短路规则: 检测消息是否过短且不是提问

        检测规则:
        - 长度 <= 3 且不包含问号
        - 过短的消息通常是口语化应答(嗯/好/哦等),不需要回复

        Args:
            text: 消息文本

        Returns:
            bool: True表示过短且不是提问
        """
        t = (text or "").strip()
        if len(t) <= 3:
            if "?" not in t and "？" not in t:
                return True
        return False

    @staticmethod
    async def _traditional_mode_should_reply(
        scene_type: str,
        scene_id: str,
        _msg_content: str,
        *,
        directed_to_bot: bool = False,
        mentioned_bot: bool = False,
    ) -> bool:
        """传统模式: 基于概率的回复决策

        决策流程:
        1. directed_to_bot/mentioned_bot → 必回
        2. 冷却中 → 不回
        3. 概率判断 + 刷屏降级

        Args:
            scene_type: 场景类型(group/private)
            scene_id: 场景标识
            _msg_content: 消息内容(当前未使用)
            directed_to_bot: 是否明确指向机器人
            mentioned_bot: 是否@机器人

        Returns:
            bool: 是否允许回复
        """
        # 步骤1: 明确指向机器人 → 必回
        if directed_to_bot or mentioned_bot:
            return True

        # 步骤2: 检查冷却
        now_ts = int(time.time())
        state = await RateLimitRepository.get_or_create(scene_type, scene_id)
        if state.cooldown_until_ts > now_ts:
            return False

        # 步骤3: 基础概率
        prob = float(
            plugin_config.yuying_group_reply_probability
            if scene_type == "group"
            else plugin_config.yuying_private_reply_probability
        )
        prob = max(0.0, min(1.0, prob))

        # 步骤4: 刷屏检测并降低概率
        window = int(plugin_config.yuying_spam_window_seconds)
        threshold = int(plugin_config.yuying_spam_msg_threshold)
        if window > 0 and threshold > 0:
            since_ts = now_ts - window
            try:
                count = await RawRepository.count_scene_messages_since(scene_type, scene_id, since_ts)
                if count >= threshold:
                    prob *= 0.25
            except Exception:
                pass

        # 步骤5: 概率判断
        return random.random() < prob

    @staticmethod
    async def _flow_mode_should_reply(
        scene_type: str,
        scene_id: str,
        _msg_content: str,
        *,
        directed_to_bot: bool = False,
        mentioned_bot: bool = False,
        image_inputs: list[dict[str, str]] | None = None,
        raw_msg_id: int | None = None,
    ) -> bool:
        """心流模式: 使用nano模型智能决策是否回复

        决策流程:
        1. 冷却中 → 不回
        2. directed_to_bot/mentioned_bot → 必回(受冷却)
        3. 短路规则: 纯表情→不回, 过短→不回, 明显提问→回
        4. 概率抽样: 决定是否调用nano
        5. nano判定: 根据上下文智能决策

        Args:
            scene_type: 场景类型(group/private)
            scene_id: 场景标识
            _msg_content: 消息内容
            directed_to_bot: 是否明确指向机器人
            mentioned_bot: 是否@机器人
            image_inputs: 当前消息的图片输入（包含url/media_key/caption）
            raw_msg_id: 当前消息的ID(用于排除上下文)

        Returns:
            bool: 是否允许回复
        """
        # 步骤1: 检查冷却(所有消息都受冷却限制)
        now_ts = int(time.time())
        state = await RateLimitRepository.get_or_create(scene_type, scene_id)
        if state.cooldown_until_ts > now_ts:
            return False

        # 步骤2: 明确指向机器人 → 必回(已过冷却检查)
        if directed_to_bot or mentioned_bot:
            return True

        # 步骤3: 短路规则
        # 3.1 纯表情/符号 → 不回
        if Gatekeeper._is_emoji_only(_msg_content):
            return False

        # 3.2 过短且不是提问 → 不回
        if Gatekeeper._is_too_short_no_question_mark(_msg_content):
            return False

        # 3.3 明显提问 → 回
        if Gatekeeper._looks_like_question(_msg_content):
            return True

        # 步骤4: 概率抽样(决定是否调用nano)
        check_prob = float(
            plugin_config.yuying_flow_mode_group_check_probability
            if scene_type == "group"
            else plugin_config.yuying_flow_mode_private_check_probability
        )
        check_prob = max(0.0, min(1.0, check_prob))
        if random.random() >= check_prob:
            return False  # 未抽中,不调用nano,保守返回False

        # 步骤5: 尝试获取Semaphore令牌(非阻塞)
        try:
            await asyncio.wait_for(
                Gatekeeper._FLOW_MODE_NANO_SEMAPHORE.acquire(),
                timeout=0.001  # 1ms超时,相当于非阻塞
            )
        except asyncio.TimeoutError:
            # 拿不到令牌,保守返回False
            return False

        try:
            # 步骤6: 获取最近消息作为上下文
            recent_lines: list[str] = []
            try:
                recent_msgs = await RawRepository.get_recent_by_scene(
                    scene_type, scene_id, limit=15  # 多取几条以备过滤
                )
                # 过滤并构建上下文(排除当前消息,按时间正序)
                for msg in reversed(recent_msgs):  # 倒序变正序
                    # 排除当前消息(通过ID精确匹配)
                    if raw_msg_id is not None and msg.id == raw_msg_id:
                        continue  # 跳过当前消息

                    # 获取消息内容
                    msg_content = (msg.content or "").strip()

                    # 跳过空内容
                    if not msg_content:
                        continue

                    # 截断过长内容
                    if len(msg_content) > 100:
                        msg_content = msg_content[:100] + "..."

                    # 构建上下文行
                    sender = msg.qq_id or "UNKNOWN"
                    msg_content_clean = msg_content.replace("\n", " ")
                    if msg.is_bot:
                        recent_lines.append(f"BOT: {msg_content_clean}")
                    else:
                        recent_lines.append(f"USER({sender}): {msg_content_clean}")

                # 只保留最近10条
                recent_lines = recent_lines[-10:]
            except Exception as e:
                # 记录异常以便调试
                logger.debug(f"心流模式获取上下文失败: {e}")
                pass  # 查询失败,使用空列表
            # 步骤7: 调用nano模型
            decided = await nano_should_reply_yes_no(
                scene_type=scene_type,
                directed_to_bot=directed_to_bot,
                mentioned_bot=mentioned_bot,
                recent_lines=recent_lines,
                current_message=_msg_content,
                image_inputs=image_inputs,  # 传入图片信息用于多模态判断
            )

            # 步骤8: 处理nano结果
            if decided is None:
                # nano调用失败或无法解析,保守返回False
                return False

            return decided

        finally:
            # 释放Semaphore令牌
            Gatekeeper._FLOW_MODE_NANO_SEMAPHORE.release()

    @staticmethod
    async def should_reply(
        scene_type: str,
        scene_id: str,
        _msg_content: str,
        *,
        mentioned_bot: bool = False,
        directed_to_bot: bool = False,
        image_inputs: list[dict[str, str]] | None = None,
        raw_msg_id: int | None = None,
    ) -> bool:
        """判断是否应该回复 - 核心决策方法

        这个方法的作用:
        - 作为门控策略的核心入口
        - 综合考虑多个因素决定是否回复
        - 返回True表示允许回复,False表示跳过

        决策流程(按优先级):
        1. 直接对机器人说话: directed_to_bot=True → 必回(return True)
           - 包括: 私聊消息、@机器人、回复机器人消息
        2. 冷却期: cooldown_until_ts > 当前时间 → 不回(return False)
        3. 概率判断:
           a. 基础概率: 群聊30%(默认),私聊100%(默认)
           b. 刷屏降级: 检测到刷屏时概率降低75%(prob *= 0.25)
           c. 随机判断: random.random() < prob

        为什么需要门控?
        - 避免刷屏: 不能对每条消息都回复
        - 自然性: 概率回复让机器人的参与更自然
        - 资源节省: 减少LLM调用和数据库写入
        - 灵活性: 可通过配置调整策略

        Args:
            scene_type: 场景类型
                - 类型: 字符串
                - 取值: "group" 或 "private"
            scene_id: 场景标识
                - 类型: 字符串
                - 内容: 群号或QQ号
            _msg_content: 归一化后的消息文本
                - 类型: 字符串
                - 注意: 参数名前缀_表示未使用(保留用于未来扩展)
                - 可能用途: 基于内容的策略(如关键词触发)
            mentioned_bot: 是否@了机器人
                - 类型: 布尔值
                - 默认值: False
                - 来源: normalizer的mentioned_bot标记
                - True: 群聊中@机器人
                - 注意: 保留此参数用于向后兼容,优先使用directed_to_bot
            directed_to_bot: 消息是否直接对机器人说的
                - 类型: 布尔值
                - 默认值: False
                - 计算逻辑: 私聊 OR @机器人 OR 回复机器人消息
                - 用途: 更准确地识别是否应该必然回复
            image_inputs: 当前消息的图片输入
                - 类型: 列表或None
                - 格式: [{"url": str, "media_key": str, "caption": str}]
                - 用途: 心流模式下用于多模态判断
            raw_msg_id: 当前消息的数据库ID
                - 类型: 整数(可选)
                - 默认值: None
                - 用途: 心流模式下精确排除当前消息,避免上下文重复

        Returns:
            bool: 是否允许回复
                - True: 允许回复,继续规划和发送
                - False: 跳过回复,不触发后续流程

        Side Effects:
            - 查询rate_limits表(读取冷却状态)
            - 查询raw_messages表(刷屏检测)

        Example:
            >>> # 示例1: 直接对机器人说话(必回)
            >>> should = await Gatekeeper.should_reply(
            ...     "group", "123456", "今天天气真好",
            ...     mentioned_bot=True, directed_to_bot=True
            ... )
            >>> print(should)
            # True (直接对机器人说话,必回)

            >>> # 示例2: 冷却期(不回)
            >>> # (假设刚发送过消息,还在冷却期)
            >>> should = await Gatekeeper.should_reply(
            ...     "group", "123456", "今天天气真好",
            ...     mentioned_bot=False, directed_to_bot=False
            ... )
            >>> print(should)
            # False (冷却中,不回)

            >>> # 示例3: 概率回复(群聊30%)
            >>> should = await Gatekeeper.should_reply(
            ...     "group", "123456", "今天天气真好",
            ...     mentioned_bot=False, directed_to_bot=False
            ... )
            >>> print(should)
            # True 或 False (基于30%概率)

            >>> # 示例4: 刷屏降级(概率降低到7.5%)
            >>> # (假设窗口内消息数超过阈值)
            >>> should = await Gatekeeper.should_reply(
            ...     "group", "123456", "今天天气真好",
            ...     mentioned_bot=False, directed_to_bot=False
            ... )
            >>> print(should)
            # True 或 False (基于7.5%概率, 30% * 0.25)
        """
        # ==================== 模式分发: 根据配置选择决策模式 ====================

        if Gatekeeper._flow_mode_enabled():
            # 心流模式: 使用nano模型智能决策
            return await Gatekeeper._flow_mode_should_reply(
                scene_type, scene_id, _msg_content,
                directed_to_bot=directed_to_bot,
                mentioned_bot=mentioned_bot,
                image_inputs=image_inputs,  # 传入图片信息
                raw_msg_id=raw_msg_id,
            )
        else:
            # 传统模式: 基于概率的决策
            return await Gatekeeper._traditional_mode_should_reply(
                scene_type, scene_id, _msg_content,
                directed_to_bot=directed_to_bot,
                mentioned_bot=mentioned_bot,
            )

    @staticmethod
    async def mark_sent(scene_type: str, scene_id: str) -> None:
        """在实际发送后调用,用于刷新冷却状态

        这个方法的作用:
        - 在机器人发送消息后更新冷却状态
        - 计算动态冷却时间(考虑刷屏)
        - 更新rate_limits表的cooldown_until_ts

        为什么需要标记发送?
        - 更新冷却: 发送后进入冷却期,避免连续回复
        - 动态调整: 根据刷屏情况动态调整冷却时间
        - 状态同步: 确保冷却状态与实际发送同步

        调用时机:
        - action_sender发送消息后立即调用
        - 在消息成功发送后调用,失败不调用
        - 每次发送都调用,无论是文本还是表情包

        Args:
            scene_type: 场景类型
                - 类型: 字符串
                - 取值: "group" 或 "private"
            scene_id: 场景标识
                - 类型: 字符串
                - 内容: 群号或QQ号

        Returns:
            None: 无返回值

        Side Effects:
            - 更新rate_limits表(cooldown_until_ts字段)
            - 输出调试日志

        Example:
            >>> # 发送消息后标记
            >>> await bot.send(event, "今天天气真好")
            >>> await Gatekeeper.mark_sent("group", "123456")
            # DEBUG: 已更新冷却: group:123456 cd=30s
        """

        # ==================== 步骤1: 计算动态冷却时间 ====================

        # await Gatekeeper._dynamic_cooldown_seconds(): 计算冷却时间
        # - 参数: scene_type, scene_id
        # - 返回: 基础冷却或翻倍冷却(刷屏时)
        # - 示例: 正常30秒,刷屏60秒
        cooldown_seconds = await Gatekeeper._dynamic_cooldown_seconds(scene_type, scene_id)

        # ==================== 步骤2: 更新冷却状态 ====================

        # await db_writer.submit_and_wait(): 提交写入任务并等待完成
        # AsyncCallableJob: 异步可调用任务
        await db_writer.submit_and_wait(
            AsyncCallableJob(
                # RateLimitRepository.mark_sent: 标记发送方法
                # - 参数: scene_type, scene_id
                # - kwargs: cooldown_seconds=冷却秒数
                # - 效果: UPDATE rate_limits
                #         SET cooldown_until_ts=当前时间+冷却秒数
                #         WHERE scene_type=? AND scene_id=?
                RateLimitRepository.mark_sent,
                args=(scene_type, scene_id),
                kwargs={"cooldown_seconds": cooldown_seconds},
            ),
            priority=5,  # 优先级5(中等)
        )

        # ==================== 步骤3: 输出调试日志 ====================

        # logger.debug(): 记录调试级别日志
        # - 内容: 场景标识和冷却时间
        # - 用途: 调试和监控冷却状态
        logger.debug(f"已更新冷却:{scene_type}:{scene_id} cd={cooldown_seconds}s")
