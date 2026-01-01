"""表情包选择模块 - 意图推断+冷却去重+候选过滤

这个模块的作用:
1. 基于消息内容推断表情包意图(情感分类)
2. 从数据库查询符合意图的候选表情包
3. 应用冷却机制避免短时间内重复发送
4. 提供回退策略确保总能选出表情包

表情包选择原理(新手必读):
- 问题: 随机发表情包容易"脱离语境",显得机械
- 解决: 根据对话上下文的情感选择合适的表情包
- 流程: 消息文本 → 意图推断 → 查询候选 → 冷却过滤 → 选择发送
- 好处: 表情包与对话语境匹配,更自然、更智能

意图分类(Intent):
- agree: 赞同、同意(行/可以/好/没错)
- thanks: 感谢(谢谢/多谢/thx)
- sorry: 道歉(对不起/抱歉/不好意思)
- shock: 震惊(卧槽/离谱/真的假的/!!!)
- tease: 调侃、逗趣(哈哈/笑死/你真逗)
- think: 思考(怎么办/想想/考虑下)
- urge: 催促(快点/赶紧/冲)
- awkward: 尴尬(呃/嗯/哈)
- neutral: 中性(默认,无明确情感)

意图推断策略:
- 规则匹配: 基于关键词和正则表达式
- 轻量可控: 不依赖LLM,响应快,可预测
- 优先级: 按规则顺序匹配,首次命中即返回
- 回退: 未匹配任何规则→neutral

冷却机制(Cooldown):
- 目的: 避免短时间内重复发送同一表情包
- 原理: 记录每个场景最后使用每个表情包的时间
- 策略: 距离上次使用<冷却时间→跳过
- 配置: yuying_sticker_cooldown_seconds(默认300秒=5分钟)
- 回退: 全部候选都在冷却期→随机选一个(防止没表情包可发)

选择流程:
1. 规范化意图(intent→normalized_intent)
2. 查询符合意图的候选表情包(limit=80)
3. 如果没有候选且意图非neutral→回退查询neutral
4. 随机打乱候选列表(避免总选第一个)
5. 遍历候选,跳过冷却期内的表情包
6. 返回第一个不在冷却期的表情包
7. 全部在冷却期→随机选一个(降级策略)

使用方式:
```python
from .stickers.selector import StickerSelector

# 1. 推断意图
intent = StickerSelector.infer_intent("谢谢你啊")
print(intent)  # "thanks"

intent = StickerSelector.infer_intent("今天天气真好")
print(intent)  # "neutral"

# 2. 选择表情包
sticker = await StickerSelector.select_sticker(
    intent="thanks",
    scene_type="group",
    scene_id="123456"
)

# 3. 发送表情包(在消息处理器中)
if sticker:
    msg = StickerSender.create_message(sticker)
    await bot.send(event, msg)
```

配置项:
- yuying_sticker_cooldown_seconds: 冷却时间(默认300秒)
"""

from __future__ import annotations

import random  # Python标准库,随机数生成
import re  # Python标准库,正则表达式
import time  # Python标准库,时间戳获取
from typing import Optional  # 类型提示

from nonebot import logger  # NoneBot日志

# 导入项目模块
from ..config import plugin_config  # 插件配置
from ..storage.models import Sticker  # 表情包模型
from ..storage.repositories.sticker_repo import StickerRepository  # 表情包仓库
from ..storage.repositories.sticker_usage_repo import StickerUsageRepository  # 使用记录仓库


class StickerSelector:
    """表情包选择器 - 意图推断+候选过滤+冷却去重

    这个类的作用:
    - 根据消息内容推断表情包意图
    - 查询符合意图的候选表情包
    - 应用冷却机制避免重复
    - 提供回退策略确保可用

    设计模式:
    - 静态方法类: 所有方法都是静态的,无需实例化
    - 工具类: 提供表情包选择功能
    - 好处: 简单直接,易于调用

    核心功能:
    1. infer_intent(): 意图推断(规则匹配)
    2. select_sticker(): 表情包选择(查询+过滤+冷却)

    意图类型:
    - _INTENTS: 支持的意图集合(9种)
    - 用于验证和规范化意图

    Example:
        >>> # 推断意图
        >>> intent = StickerSelector.infer_intent("谢谢你")
        >>> print(intent)  # "thanks"
        >>> # 选择表情包
        >>> sticker = await StickerSelector.select_sticker("thanks", "group", "123")
        >>> print(sticker.sticker_id)  # "a1b2c3d4..."
    """

    # ==================== 类属性: 支持的意图集合 ====================

    # _INTENTS: 所有支持的意图类型
    # - 类型: set(集合)
    # - 用途: 验证和规范化意图
    # - 元素: 意图字符串(9种)
    _INTENTS = {
        "agree",      # 赞同、同意
        "tease",      # 调侃、逗趣
        "shock",      # 震惊
        "sorry",      # 道歉
        "thanks",     # 感谢
        "awkward",    # 尴尬
        "think",      # 思考
        "urge",       # 催促
        "neutral",    # 中性(默认)
    }

    @staticmethod
    def infer_intent(text: str) -> str:
        """基于消息内容推断表情包意图(规则优先,轻量可控)

        这个方法的作用:
        - 分析消息文本的关键词和模式
        - 使用规则匹配推断情感意图
        - 返回意图字符串供select_sticker()使用

        为什么使用规则匹配?
        - 快速: 不需要调用LLM,毫秒级响应
        - 可控: 规则明确,行为可预测
        - 轻量: 无需额外依赖,降低复杂度
        - 够用: 9种意图覆盖大多数对话场景

        匹配优先级(按顺序):
        1. thanks: 感谢表达
        2. sorry: 道歉表达
        3. shock: 震惊表达
        4. tease: 调侃逗趣
        5. agree: 赞同同意
        6. think: 思考犹豫
        7. urge: 催促激励
        8. awkward: 尴尬无语
        9. neutral: 默认兜底

        Args:
            text: 消息文本
                - 类型: 字符串
                - 可能为None或空字符串
                - 示例: "谢谢你啊", "卧槽离谱", "行吧"

        Returns:
            str: 意图字符串
                - 取值范围: _INTENTS中的9种意图
                - 默认值: "neutral"(未匹配任何规则)
                - 示例: "thanks", "shock", "neutral"

        规则说明:
            thanks: ["谢谢", "thx", "thanks", "多谢"]
            sorry: ["对不起", "抱歉", "不好意思", "sorry"]
            shock: [!?？！]{2,}(2个以上感叹/疑问) 或 ["震惊", "卧槽", "离谱", "真的假的"]
            tease: ["哈哈", "笑死", "你真", "逗", "调侃"]
            agree: ["行", "可以", "好", "同意", "确实", "没错"]
            think: ["怎么办", "想想", "我觉得", "考虑下"]
            urge: ["快点", "赶紧", "冲", "上"]
            awkward: ["尴尬", "呃", "嗯", "哈"]

        Example:
            >>> StickerSelector.infer_intent("谢谢你帮忙")
            'thanks'

            >>> StickerSelector.infer_intent("卧槽这也太离谱了!!!")
            'shock'

            >>> StickerSelector.infer_intent("行吧可以")
            'agree'

            >>> StickerSelector.infer_intent("今天天气真好")
            'neutral'

            >>> StickerSelector.infer_intent(None)
            'neutral'
        """

        # ==================== 步骤1: 预处理文本 ====================

        # (text or "").strip().lower(): 归一化文本
        # - (text or ""): 如果text是None,转为空字符串
        # - .strip(): 去除首尾空格
        # - .lower(): 转小写(统一大小写,便于匹配)
        t = (text or "").strip().lower()

        # ==================== 步骤2: 空值检查 ====================

        # not t: 如果处理后是空字符串
        if not t:
            return "neutral"  # 返回中性意图

        # ==================== 步骤3: thanks(感谢)匹配 ====================

        # any(k in t for k in [...]): 检查是否包含任一关键词
        # - k in t: 子串检查,如"谢谢" in "谢谢你"
        # - for k in [...]: 遍历关键词列表
        # - any(...): 只要有一个True就返回True
        if any(k in t for k in ["谢谢", "thx", "thanks", "多谢"]):
            return "thanks"  # 返回感谢意图

        # ==================== 步骤4: sorry(道歉)匹配 ====================

        if any(k in t for k in ["对不起", "抱歉", "不好意思", "sorry"]):
            return "sorry"  # 返回道歉意图

        # ==================== 步骤5: shock(震惊)匹配 ====================

        # re.search(r"[!?？！]{2,}", t): 正则匹配连续感叹/疑问符
        # - r"[!?？！]{2,}": 2个或更多感叹/疑问符
        # - 例如: "!!!", "？？", "?!", "！？！"
        # - 作用: 识别强烈情绪表达
        # or any(k in t for k in [...]): 或包含震惊关键词
        if re.search(r"[!?？！]{2,}", t) or any(k in t for k in ["震惊", "卧槽", "离谱", "真的假的"]):
            return "shock"  # 返回震惊意图

        # ==================== 步骤6: tease(调侃)匹配 ====================

        if any(k in t for k in ["哈哈", "笑死", "你真", "逗", "调侃"]):
            return "tease"  # 返回调侃意图

        # ==================== 步骤7: agree(赞同)匹配 ====================

        if any(k in t for k in ["行", "可以", "好", "同意", "确实", "没错"]):
            return "agree"  # 返回赞同意图

        # ==================== 步骤8: think(思考)匹配 ====================

        if any(k in t for k in ["怎么办", "想想", "我觉得", "考虑下"]):
            return "think"  # 返回思考意图

        # ==================== 步骤9: urge(催促)匹配 ====================

        if any(k in t for k in ["快点", "赶紧", "冲", "上"]):
            return "urge"  # 返回催促意图

        # ==================== 步骤10: awkward(尴尬)匹配 ====================

        if any(k in t for k in ["尴尬", "呃", "嗯", "哈"]):
            return "awkward"  # 返回尴尬意图

        # ==================== 步骤11: neutral(默认兜底) ====================

        # 未匹配任何规则,返回中性意图
        return "neutral"

    @staticmethod
    async def select_sticker(intent: str, scene_type: str, scene_id: str) -> Optional[Sticker]:
        """按意图选择一个可用表情包,并应用冷却避免重复

        这个方法的作用:
        - 根据意图查询候选表情包
        - 过滤掉冷却期内的表情包
        - 返回第一个可用的表情包
        - 提供回退策略确保总能选出

        选择流程:
        1. 规范化意图(检查是否在_INTENTS中)
        2. 查询符合意图的候选表情包(limit=80)
        3. 如果没有候选→回退查询neutral意图
        4. 如果仍然没有→返回None
        5. 随机打乱候选列表(避免总选第一个)
        6. 遍历候选,检查冷却状态
        7. 返回第一个不在冷却期的表情包
        8. 全部在冷却期→随机选一个(降级策略)

        冷却机制:
        - 查询上次使用时间: StickerUsageRepository.get_last_used_ts()
        - 计算时间差: now_ts - last_used
        - 判断是否冷却: 时间差 < cooldown_seconds → 跳过
        - 配置: yuying_sticker_cooldown_seconds(默认300秒)

        回退策略:
        - 策略1: 意图无候选→回退neutral意图
        - 策略2: neutral也无候选→返回None
        - 策略3: 全部冷却→随机选一个(防止没表情包可发)

        Args:
            intent: 表情包意图
                - 类型: 字符串
                - 推荐值: _INTENTS中的9种意图
                - 自动规范化: 不在_INTENTS中→转为neutral
                - 示例: "thanks", "shock", "neutral"
            scene_type: 场景类型
                - 类型: 字符串
                - 取值: "group" 或 "private"
                - 用途: 冷却检查(不同场景独立冷却)
            scene_id: 场景标识
                - 类型: 字符串
                - 内容: 群号或QQ号
                - 用途: 冷却检查(不同场景独立冷却)

        Returns:
            Optional[Sticker]: 选中的表情包对象
                - 成功: 返回Sticker对象
                - 失败: 返回None(无可用表情包)
                - 示例: Sticker(sticker_id="a1b2c3d4", file_path="/path/to/sticker.jpg", ...)

        Side Effects:
            - 查询Stickers表
            - 查询StickerUsage表
            - 输出调试日志(全部冷却时)

        性能优化:
            - limit=80: 限制候选数量,避免查询过多
            - 随机打乱: 避免总选同一批表情包
            - 冷却检查: 逐个检查,找到即返回,避免全查

        Example:
            >>> # 场景1: 正常选择
            >>> sticker = await StickerSelector.select_sticker("thanks", "group", "123")
            >>> print(sticker.intents)  # "thanks"

            >>> # 场景2: 意图无候选,回退neutral
            >>> sticker = await StickerSelector.select_sticker("unknown", "group", "123")
            >>> print(sticker.intents)  # "neutral"

            >>> # 场景3: 全部在冷却期,随机选一个
            >>> sticker = await StickerSelector.select_sticker("agree", "group", "123")
            # DEBUG: 表情包候选全部处于冷却期,回退随机选择。
            >>> print(sticker.sticker_id)  # 随机选中的ID
        """

        # ==================== 步骤1: 规范化意图 ====================

        # (intent or "neutral").strip(): 规范化意图字符串
        # - (intent or "neutral"): 如果intent为None,使用默认值
        # - .strip(): 去除首尾空格
        normalized_intent = (intent or "neutral").strip()

        # normalized_intent not in StickerSelector._INTENTS: 检查是否为支持的意图
        if normalized_intent not in StickerSelector._INTENTS:
            # 不支持的意图→转为neutral
            normalized_intent = "neutral"

        # ==================== 步骤2: 查询符合意图的候选表情包 ====================

        # await StickerRepository.list_enabled_by_intent(): 查询表情包
        # - 参数: 意图字符串, limit=80(最多80个候选)
        # - SQL: WHERE intents LIKE '%intent%' AND is_enabled=True AND is_banned=False
        # - 返回: Sticker对象列表
        # - 排序: 按sticker_id(随机性)
        candidates = await StickerRepository.list_enabled_by_intent(normalized_intent, limit=80)

        # ==================== 步骤3: 回退策略 - 意图无候选 ====================

        # not candidates: 如果查询结果为空
        # normalized_intent != "neutral": 且不是neutral意图
        if not candidates and normalized_intent != "neutral":
            # 回退查询neutral意图的表情包
            # - 原因: 特定意图可能没有表情包,neutral作为兜底
            candidates = await StickerRepository.list_enabled_by_intent("neutral", limit=80)

        # ==================== 步骤4: 无候选可选 ====================

        # not candidates: 如果仍然没有候选(neutral也没有)
        if not candidates:
            return None  # 返回None,无法选择表情包

        # ==================== 步骤5: 读取冷却配置 ====================

        # int(plugin_config.yuying_sticker_cooldown_seconds): 冷却时间(秒)
        # - 默认值: 300秒(5分钟)
        # - 用途: 避免短时间内重复发送同一表情包
        cooldown = int(plugin_config.yuying_sticker_cooldown_seconds)

        # int(time.time()): 当前时间戳(秒级)
        now_ts = int(time.time())

        # ==================== 步骤6: 随机打乱候选列表 ====================

        # random.shuffle(candidates): 原地打乱列表
        # - 作用: 避免总是选择前几个表情包
        # - 效果: 每次选择的表情包具有随机性
        # - 注意: shuffle()是原地修改,无返回值
        random.shuffle(candidates)

        # ==================== 步骤7: 遍历候选,检查冷却状态 ====================

        # 遍历打乱后的候选列表
        for s in candidates:
            # ==================== 步骤7.1: 查询上次使用时间 ====================

            # await StickerUsageRepository.get_last_used_ts(): 查询上次使用时间
            # - 参数: scene_type, scene_id, sticker_id
            # - SQL: SELECT MAX(used_at) FROM sticker_usage WHERE scene_type=? AND scene_id=? AND sticker_id=?
            # - 返回: Unix时间戳(秒级)或None(从未使用)
            last_used = await StickerUsageRepository.get_last_used_ts(scene_type, scene_id, s.sticker_id)

            # ==================== 步骤7.2: 检查是否在冷却期 ====================

            # last_used: 如果查到上次使用时间(非None)
            # now_ts - last_used < cooldown: 距离上次使用时间<冷却时间
            if last_used and now_ts - last_used < cooldown:
                # 仍在冷却期,跳过这个表情包
                continue  # 继续检查下一个候选

            # ==================== 步骤7.3: 找到可用表情包 ====================

            # 不在冷却期(从未使用 或 已过冷却时间)
            return s  # 返回这个表情包

        # ==================== 步骤8: 降级策略 - 全部冷却 ====================

        # 循环结束,所有候选都在冷却期
        # 为了确保总能发送表情包,随机选一个

        # logger.debug(): 输出调试级别日志
        logger.debug("表情包候选全部处于冷却期，回退随机选择。")

        # random.choice(candidates): 随机选择一个元素
        # - 参数: 非空列表
        # - 返回: 随机选中的元素
        # - 作用: 即使全部冷却,也要发一个(降级策略)
        return random.choice(candidates)
