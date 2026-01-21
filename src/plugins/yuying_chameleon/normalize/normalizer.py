"""消息归一化模块 - 短标记化、文本截断、有效性判定

这个模块的作用:
1. 将入站消息(InboundMessage)归一化为统一格式(NormalizedMessage)
2. 处理图片标记,转换为短标识(12字符media_key)
3. 注入图片说明(caption),最多20字符
4. 文本截断,避免极长消息(最多300字符)
5. 判定消息有效性(is_effective),用于触发记忆抽取

消息归一化原理(新手必读):
- 问题: 入站消息格式多样(文本、图片、表情、回复等),难以统一处理
- 解决: 将所有消息转换为统一的NormalizedMessage格式
- 流程: InboundMessage → 文本截断 → 图片短标记 → 有效性判定 → NormalizedMessage
- 好处: 后续存储/检索/规划/LLM调用都使用统一格式

图片短标记机制:
- 问题: 图片URL很长(http://...?file=xxx),直接传给LLM浪费token
- 解决: 使用SHA256前12位作为短标识(media_key)
- 格式: [image:a1b2c3d4e5f6] 或 [image:a1b2c3d4e5f6:猫咪图片]
- 好处: 节省token,保留图片引用,可注入图片说明

图片说明注入:
- 来源: media_cache表(media_worker预处理的图片说明)
- 格式: [image:media_key:caption]
- 截断: caption最多20字符,超出用"…"标记
- 用途: 帮助LLM理解图片内容,无需重复调用视觉模型

文本截断:
- 阈值: 300字符
- 格式: 超出部分用"..."替代
- 原因: 避免极长消息(复制粘贴的长文)拖垮下游处理
- 效果: 既保留主要内容,又控制长度

有效性判定(is_effective):
- 用途: 决定是否触发记忆抽取、摘要生成等
- 无效消息: 空消息、命令(/help)、短消息(<5字符)
- 有效消息: 正常对话内容(>=5字符)
- 好处: 过滤噪音,只对真正的对话进行分析

与其他模块的协作:
- lagrange_parser: 提供InboundMessage原始消息
- MediaCacheRepository: 读取图片说明(caption)
- gatekeeper: 接收NormalizedMessage,进行策略判定
- RawRepository: 存储NormalizedMessage.content到raw_messages表
- MemoryManager: 根据is_effective决定是否触发记忆抽取

使用方式:
```python
from .normalize.normalizer import Normalizer

# 1. 从Lagrange解析入站消息
inbound = await LagrangeParser.parse(event)

# 2. 归一化消息
normalized = await Normalizer.normalize(inbound)

# 3. 检查归一化结果
print(f"内容: {normalized.content}")
print(f"有效: {normalized.is_effective}")
print(f"图片映射: {normalized.image_ref_map}")

# 4. 后续处理
if normalized.is_effective:
    # 存储消息、触发记忆抽取等
    await RawRepository.add(normalized)
```

消息流转:
```
OneBot事件 → LagrangeParser → InboundMessage
           ↓
      Normalizer.normalize()
           ↓
      NormalizedMessage
           ↓
      gatekeeper策略判定 → action_planner动作规划
```
"""

from __future__ import annotations

import hashlib  # Python标准库,哈希算法(SHA256)
import re  # Python标准库,正则表达式
from dataclasses import dataclass  # Python标准库,数据类装饰器
from dataclasses import field  # Python标准库,数据类字段工厂
from typing import Dict, Optional, Tuple  # 类型提示

# 导入项目模块
from ..adapters.lagrange_parser import InboundMessage  # 入站消息类型
from ..storage.repositories.media_cache_repo import MediaCacheRepository  # 媒体缓存仓库

@dataclass
class NormalizedMessage:
    """归一化后的消息结构 - 统一的内部消息格式

    这个数据类的作用:
    - 封装归一化后的消息数据
    - 提供统一的消息格式供后续处理使用
    - 包含消息的所有关键信息(用户、场景、内容、引用等)

    为什么需要归一化?
    - 统一格式: 不同来源的消息(文本/图片/表情)统一为相同格式
    - 简化处理: 后续存储/检索/LLM调用只需处理一种格式
    - 元数据保留: 保留消息的关键元数据(时间戳、回复关系、@机器人等)
    - 有效性标记: 标记消息是否有效,决定是否触发记忆抽取等

    字段说明:
        qq_id: 发送者QQ号
            - 类型: 字符串
            - 用途: 标识消息发送者
            - 示例: "123456789"

        scene_type: 场景类型
            - 类型: 字符串
            - 取值: "group"(群聊) 或 "private"(私聊)
            - 用途: 区分消息来源

        scene_id: 场景标识
            - 类型: 字符串
            - 内容: 群号或QQ号
            - 用途: 标识具体的场景

        timestamp: 消息时间戳
            - 类型: 整数(Unix时间戳,秒级)
            - 用途: 记录消息发送时间

        msg_type: 消息类型
            - 类型: 字符串
            - 取值: "text"(纯文本)、"image"(图片)、"mixed"(混合)等
            - 用途: 标识消息的主要类型

        content: 归一化后的消息内容
            - 类型: 字符串
            - 格式: 短标记化的文本
            - 示例: "今天天气真好 [image:a1b2c3d4e5f6:猫咪图片]"
            - 特点: 最多300字符,图片用短标记代替

        raw_ref: 原始引用
            - 类型: 字符串或None
            - 用途: 保留原始消息引用(用于调试或特殊处理)
            - 默认值: None

        image_ref_map: 图片引用映射
            - 类型: Dict[str, str]
            - 格式: {原始引用: media_key}
            - 示例: {"http://...file=xxx": "a1b2c3d4e5f6"}
            - 用途: 将短标识映射回原始图片URL
            - 默认值: 空字典(field(default_factory=dict))

        reply_to_msg_id: 回复的消息ID
            - 类型: 整数或None
            - 用途: 标识是否回复其他消息
            - 默认值: None

        mentioned_bot: 是否@了机器人
            - 类型: 布尔值
            - 用途: 标识消息是否主动@机器人
            - 默认值: False

        is_effective: 是否为有效发言
            - 类型: 布尔值
            - 用途: 决定是否触发记忆抽取、摘要生成等
            - 判定规则: 长度>=5字符且非命令
            - 默认值: False

    生命周期:
    1. Normalizer.normalize()创建NormalizedMessage对象
    2. gatekeeper根据is_effective决定是否处理
    3. RawRepository存储到raw_messages表
    4. MemoryManager根据is_effective触发记忆抽取

    Example:
        >>> msg = NormalizedMessage(
        ...     qq_id="123456789",
        ...     scene_type="group",
        ...     scene_id="987654321",
        ...     timestamp=1234567890,
        ...     msg_type="mixed",
        ...     content="今天天气真好 [image:a1b2c3d4e5f6:猫咪]",
        ...     image_ref_map={"http://...": "a1b2c3d4e5f6"},
        ...     is_effective=True
        ... )
        >>> print(msg.content)
        # "今天天气真好 [image:a1b2c3d4e5f6:猫咪]"
        >>> print(msg.is_effective)
        # True
    """

    # 发送者QQ号(字符串)
    qq_id: str

    # 场景类型("group"或"private")
    scene_type: str

    # 场景标识(群号或QQ号)
    scene_id: str

    # 消息时间戳(Unix时间戳,秒级)
    timestamp: int

    # 消息类型("text"/"image"/"mixed"等)
    msg_type: str

    # 归一化后的消息内容(短标记化,最多300字符)
    content: str

    # OneBot 平台消息 ID（可选）
    # - 用途: 用于 reply 引用等需要平台 message_id 的场景
    onebot_message_id: Optional[int] = None

    # 原始引用(可选,用于调试)
    raw_ref: Optional[str] = None

    # 图片引用映射({原始引用: media_key})
    # field(default_factory=dict): 使用工厂函数创建默认值(避免可变默认值陷阱)
    image_ref_map: Dict[str, str] = field(default_factory=dict)

    # 回复的消息ID(可选)
    reply_to_msg_id: Optional[int] = None

    # 是否@了机器人(默认False)
    mentioned_bot: bool = False

    # 是否为有效发言(默认False)
    is_effective: bool = False

class Normalizer:
    """消息归一化器 - 将入站消息转换为统一格式

    这个类的作用:
    - 提供消息归一化的核心功能
    - 处理图片短标记转换
    - 判定消息有效性
    - 生成图片的稳定短标识

    设计模式:
    - 静态方法类: 所有方法都是静态的,无需实例化
    - 工具类: 提供消息归一化功能
    - 好处: 简单直接,作为工具函数使用

    核心流程:
    1. normalize()接收InboundMessage
    2. 文本截断(最多300字符)
    3. _shorten_image_markers()转换图片标记
    4. is_effective()判定有效性
    5. 返回NormalizedMessage

    设计约束:
    - 图片/表情永远以短标记进入prompt
    - 图片说明的预处理结果从media_cache读取并注入
    - 文本最多300字符(避免极长消息)

    Example:
        >>> # 归一化入站消息
        >>> inbound = InboundMessage(
        ...     qq_id="123456789",
        ...     scene_type="group",
        ...     scene_id="987654321",
        ...     timestamp=1234567890,
        ...     msg_type="mixed",
        ...     content="今天天气真好 [image:http://...]"
        ... )
        >>> normalized = await Normalizer.normalize(inbound)
        >>> print(normalized.content)
        # "今天天气真好 [image:a1b2c3d4e5f6:猫咪]"
    """

    @staticmethod
    async def normalize(inbound: InboundMessage) -> NormalizedMessage:
        """将入站消息归一化为内部可控的短文本表示

        这个方法的作用:
        - 作为归一化的主入口
        - 处理文本截断(最多300字符)
        - 转换图片标记为短标识
        - 注入图片说明(如果可用)
        - 判定消息有效性
        - 返回统一格式的NormalizedMessage

        归一化流程:
        1. 提取消息内容(inbound.content)
        2. 文本截断(超过300字符用"..."替代)
        3. 转换图片标记为短标识并注入说明
        4. 判定消息有效性(is_effective)
        5. 创建NormalizedMessage对象
        6. 返回归一化结果

        设计约束:
        - 图片/表情永远以短标记进入prompt
        - 图片说明的预处理结果可从media_cache读取并以短形式注入
        - 文本最多300字符,避免极长消息拖垮下游

        为什么需要文本截断?
        - 避免极长消息: 用户可能复制粘贴大段文本
        - 节省存储: 减少数据库存储压力
        - 节省token: 减少LLM调用的token消耗
        - 保留主要内容: 300字符足够表达一条消息的核心内容

        Args:
            inbound: 入站消息
                - 类型: InboundMessage
                - 来源: LagrangeParser解析的OneBot事件
                - 示例: InboundMessage(
                    qq_id="123456789",
                    content="今天天气真好 [image:http://...]"
                  )

        Returns:
            NormalizedMessage: 归一化后的消息
                - 内容: 短标记化的文本(最多300字符)
                - 图片: 转换为短标识([image:a1b2c3d4e5f6])
                - 说明: 注入图片说明(如果可用)
                - 有效性: is_effective标记

        Side Effects:
            - 查询media_cache表(读取图片说明)

        Example:
            >>> # 示例1: 纯文本消息
            >>> inbound = InboundMessage(
            ...     qq_id="123",
            ...     content="今天天气真好"
            ... )
            >>> normalized = await Normalizer.normalize(inbound)
            >>> print(normalized.content)
            # "今天天气真好"
            >>> print(normalized.is_effective)
            # True (长度>=5字符)

            >>> # 示例2: 带图片的消息
            >>> inbound = InboundMessage(
            ...     qq_id="123",
            ...     content="看这只猫 [image:http://...file=xxx]"
            ... )
            >>> normalized = await Normalizer.normalize(inbound)
            >>> print(normalized.content)
            # "看这只猫 [image:a1b2c3d4e5f6:猫咪图片]"
            >>> print(normalized.image_ref_map)
            # {"http://...file=xxx": "a1b2c3d4e5f6"}

            >>> # 示例3: 极长消息
            >>> inbound = InboundMessage(
            ...     qq_id="123",
            ...     content="A" * 400  # 400个字符
            ... )
            >>> normalized = await Normalizer.normalize(inbound)
            >>> print(len(normalized.content))
            # 303 (300字符 + "...")
        """

        # ==================== 步骤1: 提取消息内容 ====================

        # inbound.content: 入站消息的原始内容
        # - 可能包含图片标记: [image:http://...]
        # - 可能很长: 需要截断
        content = inbound.content

        # ==================== 步骤2: 文本截断(避免极长消息) ====================

        # len(content) > 300: 消息长度超过300字符
        if len(content) > 300:
            # content[:300] + "...": 截取前300字符并添加省略号
            # - 原因: 避免极长消息(复制粘贴的长文)拖垮下游处理
            # - 效果: 既保留主要内容,又控制长度
            content = content[:300] + "..."

        # ==================== 步骤3: 转换图片标记为短标识并注入说明 ====================

        # await Normalizer._shorten_image_markers(content): 处理图片标记
        # - 功能: 将[image:http://...]转换为[image:a1b2c3d4e5f6]
        # - 额外: 注入图片说明(如果可用): [image:a1b2c3d4e5f6:猫咪图片]
        # - 返回: (处理后的文本, {原始引用: media_key}映射)
        content, image_ref_map = await Normalizer._shorten_image_markers(content)

        # ==================== 步骤4: 判定消息有效性 ====================

        # Normalizer.is_effective(content): 判断是否为有效发言
        # - 规则: 长度>=5字符且非命令
        # - 用途: 决定是否触发记忆抽取、摘要生成等
        is_effective = Normalizer.is_effective(content)

        # ==================== 步骤5: 创建并返回NormalizedMessage ====================

        # NormalizedMessage(): 创建归一化消息对象
        return NormalizedMessage(
            # 从inbound复制基本字段
            qq_id=inbound.qq_id,  # 发送者QQ号
            scene_type=inbound.scene_type,  # 场景类型(group/private)
            scene_id=inbound.scene_id,  # 场景标识(群号/QQ号)
            timestamp=inbound.timestamp,  # 消息时间戳
            onebot_message_id=getattr(inbound, "onebot_message_id", None),
            msg_type=inbound.msg_type,  # 消息类型(text/image/mixed)

            # 归一化后的内容
            content=content,  # 短标记化的文本(最多300字符)

            # 保留原始引用(可选)
            raw_ref=inbound.raw_ref,

            # 图片引用映射({原始引用: media_key})
            image_ref_map=image_ref_map,

            # 回复关系(可选)
            reply_to_msg_id=inbound.reply_to_msg_id,

            # 是否@了机器人
            mentioned_bot=inbound.mentioned_bot,

            # 是否为有效发言
            is_effective=is_effective
        )

    @staticmethod
    def is_effective(text: str) -> bool:
        """判断是否为"有效发言" - 用于累计计数与记忆抽取触发

        这个方法的作用:
        - 判断消息是否为有效对话内容
        - 过滤无效消息(空消息、命令、短消息)
        - 决定是否触发记忆抽取、摘要生成等

        有效性判定规则:
        1. 非空: 去除空格后不为空字符串
        2. 非命令: 不以"/"或"!"开头
        3. 长度足够: 去除短标记和空格后>=5字符

        为什么需要有效性判定?
        - 过滤噪音: 命令(/help)、表情(单个图片)不应触发记忆抽取
        - 节省资源: 只对真正的对话内容进行分析
        - 提高质量: 避免短消息("好"、"哦")污染记忆和摘要

        无效消息示例:
        - 空消息: ""、"   "
        - 命令: "/help"、"!status"
        - 纯表情: "[image:a1b2c3d4e5f6]"(去除短标记后为空)
        - 短消息: "好"、"哦"、"嗯嗯"(长度<5)

        有效消息示例:
        - 正常对话: "今天天气真好"(长度>=5)
        - 带图片: "看这只猫 [image:xxx]"(去除短标记后仍>=5)
        - 长句子: "我昨天去了公园,看到很多人在放风筝"

        Args:
            text: 归一化后的文本
                - 类型: 字符串
                - 可能包含短标记: [image:xxx]、[face:xxx]
                - 已经过截断(最多300字符)
                - 示例: "今天天气真好 [image:a1b2c3d4e5f6]"

        Returns:
            bool: 是否有效
                - True: 有效发言,触发记忆抽取等
                - False: 无效消息,不触发

        Example:
            >>> # 示例1: 有效消息
            >>> Normalizer.is_effective("今天天气真好")
            True

            >>> # 示例2: 短消息(无效)
            >>> Normalizer.is_effective("好")
            False  # 长度<5

            >>> # 示例3: 命令(无效)
            >>> Normalizer.is_effective("/help")
            False  # 以"/"开头

            >>> # 示例4: 纯表情(无效)
            >>> Normalizer.is_effective("[image:a1b2c3d4e5f6]")
            False  # 去除短标记后为空

            >>> # 示例5: 带图片的有效消息
            >>> Normalizer.is_effective("看这只猫 [image:xxx]")
            True  # 去除短标记后仍>=5字符
        """

        # ==================== 步骤1: 去除首尾空格 ====================

        # text.strip(): 去除首尾的空格、换行、制表符
        stripped = text.strip()

        # ==================== 步骤2: 检查是否为空 ====================

        # not stripped: 去除空格后为空字符串
        if not stripped:
            return False  # 空消息,无效

        # ==================== 步骤3: 检查是否为命令 ====================

        # stripped.startswith("/"): 以"/"开头(命令)
        # stripped.startswith("!"): 以"!"开头(命令)
        # - 示例: "/help"、"!status"
        if stripped.startswith("/") or stripped.startswith("!"):
            return False  # 命令,无效

        # ==================== 步骤4: 去除短标记和空白后计算长度 ====================

        # re.sub(r"\[[^\]]+\]", "", stripped): 去除短标记
        # - 正则: \[[^\]]+\] 匹配[...]形式的标记
        #   * \[: 匹配左方括号(转义)
        #   * [^\]]+: 匹配1个或多个非右方括号字符
        #   * \]: 匹配右方括号(转义)
        # - 效果: [image:xxx]、[face:yyy]等标记被删除
        # - 示例: "看这只猫 [image:xxx]" → "看这只猫 "
        plain = re.sub(r"\[[^\]]+\]", "", stripped)

        # re.sub(r"\s+", "", plain): 去除所有空白字符
        # - 正则: \s+ 匹配1个或多个空白字符(空格、换行、制表符)
        # - 效果: 所有空格被删除
        # - 示例: "看这只猫 " → "看这只猫"
        plain = re.sub(r"\s+", "", plain)

        # ==================== 步骤5: 判定长度 ====================

        # len(plain) >= 5: 去除短标记和空格后长度是否>=5字符
        # - 阈值: 5字符(中文约2-3个字,英文约5个单词)
        # - 理由: 过滤单字回复("好"、"哦")和纯表情
        return len(plain) >= 5

    @staticmethod
    def _media_key_from_ref(raw_ref: str) -> str:
        """从图片引用生成稳定短标识 - SHA256前12位

        这个方法的作用:
        - 将长图片URL转换为短标识(media_key)
        - 使用SHA256哈希确保唯一性
        - 取前12位作为短标识(足够区分,节省空间)

        为什么需要短标识?
        - 节省token: 图片URL很长,直接传给LLM浪费token
        - 统一格式: 所有图片都用相同长度的短标识
        - 唯一性: SHA256前12位碰撞概率极低(2^48种组合)
        - 可读性: 12字符比64字符更易读

        Args:
            raw_ref: 原始图片引用
                - 类型: 字符串
                - 内容: 图片URL或文件路径
                - 示例: "http://example.com/image.jpg?file=xxx"

        Returns:
            str: 稳定短标识(12个十六进制字符)
                - 格式: SHA256哈希的前12位
                - 示例: "a1b2c3d4e5f6"

        为什么是12位?
        - 唯一性: 2^48 ≈ 281万亿种组合,碰撞概率极低
        - 简洁性: 足够短,节省存储和显示空间
        - 可读性: 比64位更易读和记忆
        - 实用性: 一个bot的图片量通常不会超过百万级

        Example:
            >>> # 示例1: 标准URL
            >>> media_key = Normalizer._media_key_from_ref("http://example.com/image.jpg")
            >>> print(len(media_key))
            # 12
            >>> print(media_key)
            # "a1b2c3d4e5f6"

            >>> # 示例2: 相同URL生成相同短标识
            >>> key1 = Normalizer._media_key_from_ref("http://example.com/cat.jpg")
            >>> key2 = Normalizer._media_key_from_ref("http://example.com/cat.jpg")
            >>> print(key1 == key2)
            # True (SHA256确保稳定性)

            >>> # 示例3: 不同URL生成不同短标识
            >>> key1 = Normalizer._media_key_from_ref("http://example.com/cat.jpg")
            >>> key2 = Normalizer._media_key_from_ref("http://example.com/dog.jpg")
            >>> print(key1 == key2)
            # False (不同输入产生不同哈希)
        """

        # ==================== 步骤1: 计算SHA256哈希 ====================

        # hashlib.sha256(raw_ref.encode("utf-8")): 计算SHA256哈希
        # - raw_ref.encode("utf-8"): 将字符串转为UTF-8字节
        # - hashlib.sha256(...): 计算SHA256哈希
        # .hexdigest(): 转为16进制字符串(64个字符)
        # - 格式: "a1b2c3d4e5f6...7890"(64字符)
        digest = hashlib.sha256(raw_ref.encode("utf-8")).hexdigest()

        # ==================== 步骤2: 返回前12位 ====================

        # digest[:12]: 取哈希的前12个字符
        # - 原因: 12位足够区分(2^48种组合),且足够简洁
        # - 示例: "a1b2c3d4e5f6"
        return digest[:12]

    @staticmethod
    async def _shorten_image_markers(text: str) -> Tuple[str, Dict[str, str]]:
        """将文本中的图片标记转换为短标识,并返回映射

        这个方法的作用:
        - 扫描文本中的所有图片标记: [image:原始引用]
        - 将原始引用转换为短标识: [image:media_key]
        - 注入图片说明(如果可用): [image:media_key:caption]
        - 返回处理后的文本和原始引用→media_key的映射

        转换流程:
        1. 使用正则表达式扫描[image:...]标记
        2. 对每个标记:
           a. 提取原始引用(raw_ref)
           b. 计算短标识(media_key)
           c. 查询media_cache表(读取图片说明)
           d. 如果有说明: [image:media_key:caption]
           e. 如果无说明: [image:media_key]
        3. 返回处理后的文本和映射

        为什么需要图片说明?
        - 帮助LLM理解: LLM看不到图片,需要文字描述
        - 节省调用: 说明由media_worker预处理,无需重复调用视觉模型
        - 控制长度: caption最多20字符,超出用"…"标记

        Args:
            text: 原始文本
                - 类型: 字符串
                - 可能包含图片标记: [image:http://...]
                - 示例: "看这只猫 [image:http://example.com/cat.jpg]"

        Returns:
            Tuple[str, Dict[str, str]]: (处理后的文本, 映射)
                - 处理后的文本: 短标记化的文本
                - 映射: {原始引用: media_key}
                - 示例: (
                    "看这只猫 [image:a1b2c3d4e5f6:猫咪图片]",
                    {"http://example.com/cat.jpg": "a1b2c3d4e5f6"}
                  )

        Side Effects:
            - 查询media_cache表(读取图片说明)

        处理细节:
            - 正则不支持异步回调,采用手动迭代拼接
            - 对每个匹配调用异步函数repl()
            - 拼接处理前、处理后、剩余部分

        Example:
            >>> # 示例1: 无说明的图片
            >>> text = "看这只猫 [image:http://example.com/cat.jpg]"
            >>> result, mapping = await Normalizer._shorten_image_markers(text)
            >>> print(result)
            # "看这只猫 [image:a1b2c3d4e5f6]"
            >>> print(mapping)
            # {"http://example.com/cat.jpg": "a1b2c3d4e5f6"}

            >>> # 示例2: 有说明的图片
            >>> # (假设media_cache中已有caption="猫咪图片")
            >>> text = "看这只猫 [image:http://example.com/cat.jpg]"
            >>> result, mapping = await Normalizer._shorten_image_markers(text)
            >>> print(result)
            # "看这只猫 [image:a1b2c3d4e5f6:猫咪图片]"

            >>> # 示例3: 多个图片
            >>> text = "[image:http://a.jpg] 和 [image:http://b.jpg]"
            >>> result, mapping = await Normalizer._shorten_image_markers(text)
            >>> print(result)
            # "[image:abc123] 和 [image:def456]"
            >>> print(len(mapping))
            # 2
        """

        # ==================== 步骤1: 编译正则表达式 ====================

        # re.compile(r"\[image:(?P<ref>[^\]]+)\]"): 编译正则
        # - 正则: \[image:(?P<ref>[^\]]+)\]
        #   * \[image:: 匹配"[image:"(转义左括号)
        #   * (?P<ref>...): 命名捕获组"ref"
        #   * [^\]]+: 匹配1个或多个非右方括号字符(引用内容)
        #   * \]: 匹配右方括号(转义)
        # - 示例: [image:http://example.com/cat.jpg]
        #   * 匹配整体: "[image:http://example.com/cat.jpg]"
        #   * 捕获"ref": "http://example.com/cat.jpg"
        pattern = re.compile(r"\[image:(?P<ref>[^\]]+)\]")

        # mapping: 存储{原始引用: media_key}映射
        mapping: Dict[str, str] = {}

        # ==================== 步骤2: 定义异步替换函数 ====================

        async def repl(match: re.Match) -> str:
            """将单个图片标记替换为短标识形式

            这个内部函数的作用:
            - 处理单个图片标记的转换
            - 提取原始引用
            - 计算短标识
            - 查询图片说明
            - 返回新的标记格式

            Args:
                match: 正则匹配对象
                    - 包含原始引用: match.group("ref")

            Returns:
                str: 替换后的标记
                    - 格式: [image:media_key] 或 [image:media_key:caption]
            """

            # ==================== 步骤2.1: 提取原始引用 ====================

            # match.group("ref"): 获取命名捕获组"ref"的内容
            # - 示例: "http://example.com/cat.jpg"
            raw_ref = match.group("ref")

            # ==================== 步骤2.2: 检查引用有效性 ====================

            # not raw_ref or raw_ref == "None": 引用为空或字符串"None"
            # - 可能原因: 解析失败、临时文件被删除等
            if not raw_ref or raw_ref == "None":
                return "[image:unknown]"  # 返回占位符

            # ==================== 步骤2.3: 计算短标识 ====================

            # Normalizer._media_key_from_ref(raw_ref): 生成短标识
            # - 功能: SHA256前12位
            # - 示例: "a1b2c3d4e5f6"
            media_key = Normalizer._media_key_from_ref(raw_ref)

            # ==================== 步骤2.4: 存储映射 ====================

            # mapping[raw_ref] = media_key: 记录原始引用→短标识映射
            # - 用途: 后续可以通过短标识找回原始URL
            mapping[raw_ref] = media_key

            # ==================== 步骤2.5: 查询图片说明 ====================

            # await MediaCacheRepository.get(media_key): 查询media_cache表
            # - 参数: media_key(短标识)
            # - 返回: MediaCache对象或None
            # - 字段: media_key, caption, labels等
            cached = await MediaCacheRepository.get(media_key)

            # ==================== 步骤2.6: 注入图片说明(如果可用) ====================

            # cached and cached.caption: 缓存存在且有说明
            if cached and cached.caption:
                # caption.strip(): 去除首尾空格
                caption = cached.caption.strip()

                # len(caption) > 20: 说明过长
                if len(caption) > 20:
                    # caption[:20] + "…": 截断到20字符并添加省略号
                    # - 原因: 避免说明过长占用token
                    # - 示例: "这是一只非常可爱的小猫咪" → "这是一只非常可爱的小猫咪…"
                    caption = caption[:20] + "…"

                # 返回带说明的标记
                # f"[image:{media_key}:{caption}]": 拼接短标识和说明
                # - 格式: [image:a1b2c3d4e5f6:猫咪图片]
                return f"[image:{media_key}:{caption}]"

            # ==================== 步骤2.7: 返回不带说明的标记 ====================

            # f"[image:{media_key}]": 只有短标识,无说明
            # - 格式: [image:a1b2c3d4e5f6]
            return f"[image:{media_key}]"

        # ==================== 步骤3: 手动迭代拼接(因为正则替换不支持异步) ====================

        # 注释: 正则表达式的re.sub()不支持异步回调函数
        # 解决方案: 手动迭代所有匹配,依次调用异步函数,最后拼接结果

        # parts: 存储拼接的文本片段
        parts = []

        # last: 上次匹配结束的位置
        last = 0

        # pattern.finditer(text): 迭代所有匹配
        # - 返回: 迭代器,每次yield一个Match对象
        for m in pattern.finditer(text):
            # ==================== 步骤3.1: 添加匹配前的文本 ====================

            # text[last : m.start()]: 上次匹配结束到本次匹配开始的文本
            # - 示例: "看这只猫 " (匹配前的普通文本)
            parts.append(text[last : m.start()])

            # ==================== 步骤3.2: 添加替换后的标记 ====================

            # await repl(m): 调用异步函数处理匹配
            # - 返回: 替换后的标记(如[image:a1b2c3d4e5f6])
            parts.append(await repl(m))

            # ==================== 步骤3.3: 更新位置 ====================

            # m.end(): 本次匹配结束的位置
            last = m.end()

        # ==================== 步骤3.4: 添加最后剩余的文本 ====================

        # text[last:]: 最后一次匹配结束到文本结尾的部分
        # - 示例: " 真可爱" (最后的普通文本)
        parts.append(text[last:])

        # ==================== 步骤4: 返回拼接结果和映射 ====================

        # "".join(parts): 拼接所有片段
        # - 效果: 将parts列表中的所有字符串连接为一个字符串
        # - 示例: ["看这只猫 ", "[image:a1b2c3d4e5f6]", " 真可爱"] → "看这只猫 [image:a1b2c3d4e5f6] 真可爱"
        # mapping: {原始引用: media_key}映射
        return "".join(parts), mapping
