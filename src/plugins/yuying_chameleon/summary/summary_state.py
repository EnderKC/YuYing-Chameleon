"""摘要窗口状态管理模块 - 进程内状态存储与窗口计数管理

这个模块的作用:
1. 管理每个场景(群/私聊)的摘要窗口状态
2. 提供窗口计数推进(bump)和重置(reset)功能
3. 进程内内存存储,不持久化到数据库
4. 与SummaryManager协作,支持滚动窗口摘要机制

进程内状态管理原理(新手必读):
- 问题: 每个场景需要独立追踪窗口状态(起始时间、消息计数)
- 解决: 使用Python字典在内存中存储状态
- 好处: 读写速度快,无需数据库查询
- 代价: 进程重启后状态丢失,需重新积累

为什么不持久化到数据库?
- 状态是临时的: 生成摘要后就会重置,无需长期保存
- 高频写入: 每条消息都要更新计数,落库会影响性能
- 容错性高: 重启后重新积累,最多损失一个窗口周期
- 设计权衡: 用可接受的状态丢失换取简洁和性能

与SummaryManager的协作:
- SummaryManager.on_message()调用bump()增加计数
- SummaryManager检查触发条件(消息数或时间)
- 触发后生成摘要,调用reset()重置窗口
- 状态存储不参与摘要生成逻辑,仅提供状态管理

状态丢失的影响:
- 进程重启: 窗口状态清零,从0开始积累
- 最坏情况: 重启前积累了19条消息(阈值20),重启后需再积累20条
- 实际影响: 可接受,因为对话总会继续积累新消息

使用方式:
```python
from .summary.summary_state import summary_state_store

# 1. 推进窗口计数(每条新消息时调用)
state = summary_state_store.bump(
    scene_type="group",
    scene_id="123456",
    now_ts=1234567890
)
print(f"当前窗口: {state.message_count}条消息")
print(f"窗口起始: {state.window_start_ts}")

# 2. 检查是否达到触发条件(在SummaryManager中)
if state.message_count >= 20:  # 消息数触发
    # 生成摘要...
    pass

# 3. 生成摘要后重置窗口
summary_state_store.reset(
    scene_type="group",
    scene_id="123456",
    now_ts=1234567900
)
# 重置后窗口从0开始,起始时间更新为now_ts
```

配置项:
- 本模块不依赖配置,触发阈值由SummaryManager的配置控制
- yuying_summary_window_message_count: 消息数触发阈值(默认20)
- yuying_summary_window_seconds: 时间触发阈值(默认900秒)
"""

from __future__ import annotations

import time  # Python标准库,用于获取当前时间戳
from dataclasses import dataclass  # Python标准库,用于定义数据类
from typing import Dict, Optional, Tuple  # 类型提示


@dataclass
class SummaryWindowState:
    """单个场景的摘要窗口状态 - 追踪窗口起始时间和消息计数

    这个数据类的作用:
    - 封装单个场景(群/私聊)的窗口状态
    - 包含窗口起始时间和消息计数两个核心字段
    - 使用dataclass自动生成__init__、__repr__等方法

    为什么使用dataclass?
    - 简洁: 无需手动写__init__方法
    - 可读: 字段定义清晰,类型提示明确
    - 功能: 自动生成相等比较、字符串表示等方法

    字段说明:
        window_start_ts: 窗口起始时间戳
            - 类型: 整数(Unix时间戳,秒级)
            - 作用: 记录当前窗口何时开始积累消息
            - 用途: 计算窗口时长,判断时间触发条件
            - 示例: 1609459200 (2021-01-01 00:00:00 UTC)

        message_count: 窗口内消息计数
            - 类型: 整数
            - 作用: 记录当前窗口已积累的消息数量
            - 默认值: 0(新窗口从0开始)
            - 用途: 判断消息数触发条件
            - 示例: 15 (表示已有15条消息,还差5条到达阈值20)

    生命周期:
    1. 创建: 首次收到场景消息时,window_start_ts=当前时间,message_count=0
    2. 推进: 每条新消息,message_count += 1
    3. 检查: SummaryManager检查是否达到触发条件
    4. 重置: 生成摘要后,window_start_ts=当前时间,message_count=0
    5. 循环: 重复2-4步骤

    Example:
        >>> state = SummaryWindowState(window_start_ts=1609459200, message_count=15)
        >>> print(state.window_start_ts)  # 1609459200
        >>> print(state.message_count)    # 15
        >>> state.message_count += 1      # 推进计数
        >>> print(state.message_count)    # 16
    """

    # 窗口起始时间戳(Unix时间戳,秒级)
    window_start_ts: int

    # 窗口内消息计数(默认值0)
    message_count: int = 0


class SummaryStateStore:
    """摘要窗口状态存储 - 进程内内存存储,管理所有场景的窗口状态

    这个类的作用:
    - 管理多个场景(群/私聊)的窗口状态
    - 提供状态查询、推进、重置功能
    - 使用字典存储,场景(scene_type, scene_id)作为key
    - 进程内内存存储,不持久化到数据库

    设计模式:
    - 单例模式: 模块级创建唯一实例summary_state_store
    - 状态管理器: 封装状态的读写操作
    - 懒初始化: 场景的状态在首次访问时创建

    核心数据结构:
    - _states: Dict[Tuple[str, str], SummaryWindowState]
    - key: (scene_type, scene_id)元组,唯一标识场景
    - value: SummaryWindowState对象,存储窗口状态

    与数据库的区别:
    - 数据库: 持久化,重启后数据仍在,但读写慢
    - 内存存储: 临时的,重启后丢失,但读写快
    - 本模块选择内存: 状态是临时的,性能优先

    线程安全性:
    - NoneBot2是单进程单线程异步框架
    - 所有操作在同一事件循环中执行
    - 无需加锁,天然线程安全

    Example:
        >>> store = SummaryStateStore()
        >>> state1 = store.bump("group", "123", now_ts=1000)
        >>> print(state1.message_count)  # 1
        >>> state2 = store.bump("group", "123", now_ts=1001)
        >>> print(state2.message_count)  # 2
        >>> store.reset("group", "123", now_ts=1002)
        >>> state3 = store.bump("group", "123", now_ts=1003)
        >>> print(state3.message_count)  # 1 (重置后重新计数)
    """

    def __init__(self) -> None:
        """初始化状态存储

        这个方法的作用:
        - 创建空字典_states用于存储所有场景的窗口状态
        - 在模块加载时由全局实例summary_state_store调用一次

        初始状态:
        - _states为空字典{}
        - 场景状态在首次访问时懒初始化(延迟创建)

        Side Effects:
            - 创建_states字典

        Example:
            >>> store = SummaryStateStore()
            >>> print(len(store._states))  # 0 (初始为空)
        """

        # ==================== 创建状态字典 ====================

        # self._states: 场景状态字典
        # - key: (scene_type, scene_id)元组
        #   * scene_type: "group" 或 "private"
        #   * scene_id: 群号或QQ号(字符串)
        # - value: SummaryWindowState对象
        # - 初始为空字典,场景状态在首次bump时创建
        self._states: Dict[Tuple[str, str], SummaryWindowState] = {}

    def bump(self, scene_type: str, scene_id: str, now_ts: Optional[int] = None) -> SummaryWindowState:
        """推进某个场景的窗口计数(message_count += 1)

        这个方法的作用:
        - 在每条新消息到达时调用
        - 如果场景状态不存在,创建新状态(懒初始化)
        - 将message_count增加1
        - 返回更新后的状态对象

        懒初始化机制:
        - 首次访问场景时,创建SummaryWindowState对象
        - window_start_ts=当前时间,message_count=0
        - 存入_states字典
        - 好处: 节省内存,只为活跃场景分配状态

        Args:
            scene_type: 场景类型
                - 类型: 字符串
                - 取值: "group"(群聊) 或 "private"(私聊)
                - 用途: 与scene_id组合作为字典key
            scene_id: 场景标识
                - 类型: 字符串
                - 内容: 群号或QQ号
                - 用途: 与scene_type组合作为字典key
            now_ts: 当前时间戳(可选)
                - 类型: 整数或None
                - 默认值: None(自动获取当前时间)
                - 用途: 首次创建状态时作为window_start_ts
                - 示例: 1609459200

        Returns:
            SummaryWindowState: 更新后的窗口状态对象
                - 如果是新场景: window_start_ts=now_ts, message_count=1
                - 如果是已有场景: window_start_ts不变, message_count += 1

        Side Effects:
            - 如果场景状态不存在,创建并存入_states
            - 更新现有状态的message_count字段(+1)

        Example:
            >>> store = SummaryStateStore()
            >>> # 第1条消息
            >>> state1 = store.bump("group", "123", now_ts=1000)
            >>> print(state1.message_count)       # 1
            >>> print(state1.window_start_ts)     # 1000
            >>> # 第2条消息
            >>> state2 = store.bump("group", "123", now_ts=1001)
            >>> print(state2.message_count)       # 2
            >>> print(state2.window_start_ts)     # 1000 (不变)
            >>> # 第20条消息,触发摘要生成,然后重置
            >>> store.reset("group", "123", now_ts=1020)
            >>> state3 = store.bump("group", "123", now_ts=1021)
            >>> print(state3.message_count)       # 1 (重新计数)
            >>> print(state3.window_start_ts)     # 1020 (新窗口)
        """

        # ==================== 步骤1: 获取当前时间戳 ====================

        # now_ts is None: 如果未提供时间戳
        if now_ts is None:
            # int(time.time()): 获取当前Unix时间戳(秒级)
            # time.time()返回浮点数,int()转为整数
            now_ts = int(time.time())

        # ==================== 步骤2: 构造场景key ====================

        # key: (scene_type, scene_id)元组
        # - 作用: 唯一标识场景
        # - 示例: ("group", "123456")
        key = (scene_type, scene_id)

        # ==================== 步骤3: 获取或创建场景状态 ====================

        # self._states.get(key): 从字典获取场景状态
        # - 如果存在: 返回SummaryWindowState对象
        # - 如果不存在: 返回None
        state = self._states.get(key)

        # state is None: 场景状态不存在(首次访问)
        if state is None:
            # 创建新的窗口状态
            # SummaryWindowState(window_start_ts=now_ts, message_count=0):
            # - window_start_ts: 当前时间,作为窗口起点
            # - message_count: 0,新窗口从0开始
            state = SummaryWindowState(window_start_ts=now_ts, message_count=0)

            # 存入字典
            # self._states[key] = state: 保存状态,下次可直接获取
            self._states[key] = state

        # ==================== 步骤4: 推进消息计数 ====================

        # state.message_count += 1: 消息计数增加1
        # - 无论是新创建的状态(0→1)还是已有状态(N→N+1)
        # - 这是bump方法的核心操作
        state.message_count += 1

        # ==================== 步骤5: 返回更新后的状态 ====================

        # return state: 返回状态对象
        # - SummaryManager可以读取message_count和window_start_ts
        # - 用于判断是否达到触发条件
        return state

    def reset(self, scene_type: str, scene_id: str, now_ts: Optional[int] = None) -> None:
        """重置某个场景的摘要窗口(创建新窗口,计数归零)

        这个方法的作用:
        - 在生成摘要后调用,重置窗口状态
        - 创建新的SummaryWindowState对象
        - window_start_ts=当前时间,message_count=0
        - 覆盖旧状态,开始新的积累周期

        为什么需要重置?
        - 摘要生成: 已将窗口内消息总结为摘要
        - 清空窗口: 避免重复总结相同消息
        - 新周期: 从当前时间开始积累新消息

        与bump的区别:
        - bump: 增加计数,window_start_ts不变
        - reset: 重置计数,window_start_ts更新为当前时间

        Args:
            scene_type: 场景类型
                - 类型: 字符串
                - 取值: "group" 或 "private"
            scene_id: 场景标识
                - 类型: 字符串
                - 内容: 群号或QQ号
            now_ts: 当前时间戳(可选)
                - 类型: 整数或None
                - 默认值: None(自动获取当前时间)
                - 用途: 作为新窗口的window_start_ts

        Returns:
            None: 无返回值

        Side Effects:
            - 覆盖_states中的场景状态
            - 创建新的SummaryWindowState对象
            - 旧状态对象被丢弃,由Python垃圾回收器回收

        调用时机:
            - SummaryManager.on_message()生成摘要后
            - SummaryManager.on_message()查询消息为空时

        Example:
            >>> store = SummaryStateStore()
            >>> # 积累20条消息
            >>> for i in range(20):
            ...     store.bump("group", "123", now_ts=1000+i)
            >>> state = store._states[("group", "123")]
            >>> print(state.message_count)       # 20
            >>> print(state.window_start_ts)     # 1000
            >>> # 生成摘要后重置
            >>> store.reset("group", "123", now_ts=1020)
            >>> state = store._states[("group", "123")]
            >>> print(state.message_count)       # 0 (重置为0)
            >>> print(state.window_start_ts)     # 1020 (更新为当前时间)
        """

        # ==================== 步骤1: 获取当前时间戳 ====================

        # now_ts is None: 如果未提供时间戳
        if now_ts is None:
            # int(time.time()): 获取当前Unix时间戳(秒级)
            now_ts = int(time.time())

        # ==================== 步骤2: 创建并存储新的窗口状态 ====================

        # self._states[(scene_type, scene_id)] = ...: 覆盖旧状态
        # SummaryWindowState(window_start_ts=now_ts, message_count=0):
        # - window_start_ts: 当前时间,作为新窗口的起点
        # - message_count: 0,新窗口从0开始计数
        # 效果: 旧状态被丢弃,新状态生效
        self._states[(scene_type, scene_id)] = SummaryWindowState(window_start_ts=now_ts, message_count=0)


# ==================== 模块级全局实例 ====================

# summary_state_store: 全局状态存储实例
# - 作用: 整个项目共享同一个状态存储对象
# - 时机: 模块导入时立即创建
# - 用途: SummaryManager调用bump()和reset()管理窗口状态
# - 单例模式: 确保所有场景的状态集中管理
# - 示例:
#   from .summary.summary_state import summary_state_store
#   state = summary_state_store.bump("group", "123", now_ts=1000)
summary_state_store = SummaryStateStore()
