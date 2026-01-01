"""检索增强生成(RAG)模块 - 混合查询+向量检索+记忆上下文

这个模块的作用:
1. 实现RAG(Retrieval-Augmented Generation)检索逻辑
2. 构建混合查询(Hybrid Query)结合当前消息和历史上下文
3. 从Qdrant向量库检索相关消息片段
4. 选择用户记忆提供个性化上下文
5. 提供完整的降级策略,确保服务不可用时不影响主流程

RAG(检索增强生成)原理(新手必读):
- RAG: Retrieval-Augmented Generation的缩写
- 问题: LLM上下文窗口有限,无法记住所有历史对话
- 解决: 检索最相关的历史片段,动态补充到上下文
- 流程: 用户消息 → 向量化 → 检索相似消息 → 拼接到prompt → LLM生成回复
- 优势: 突破上下文限制,实现长期对话连贯性

Hybrid Query(混合查询)设计:
- 目标: 构建一个简洁但信息完整的查询文本
- 组成部分:
  1. 【当前用户】当前消息(必有)
  2. 【最近用户】用户自己最近3条消息(理解隐藏意图/代指)
  3. 【最近机器人】机器人最后一句话(对话连贯性)
  4. 【最近对方】其他人最后一句话(群聊理解)
  5. 【最近摘要】历史对话摘要(兜底补充)
- 原因: 单纯用当前消息检索可能遗漏关键上下文

降级策略(服务可用性保障):
- Embedder失败: 返回空RAG片段,不阻塞回复
- Qdrant失败: 返回空RAG片段,不阻塞回复
- MemoryManager失败: 返回空记忆,不阻塞回复
- 数据库查询失败: 降级为仅使用当前消息
- 好处: 即使所有检索服务都挂了,机器人仍可正常回复(只是缺少上下文增强)

使用方式:
```python
from .retrieval.retriever import Retriever

# 1. 构建混合查询
query = await Retriever.build_hybrid_query(
    qq_id="123456",
    scene_type="group",
    scene_id="789",
    current_msg="今天天气怎么样"
)
# query内容:
# 【当前用户】今天天气怎么样
# 【最近用户】我在北京 / 明天想出去玩
# 【最近机器人】好的,我帮你查一下
# 【最近对方】北京这两天降温了

# 2. 执行检索
context = await Retriever.retrieve(
    qq_id="123456",
    scene_type="group",
    scene_id="789",
    query=query
)
# context结构:
# {
#   "rag_snippets": ["去年冬天北京很冷", "记得带羽绒服", ...],
#   "memories": [Memory(content="用户住在北京"), ...]
# }

# 3. 拼接到LLM的prompt
prompt = f\"\"\"
历史相关对话:
{chr(10).join(context['rag_snippets'])}

用户背景:
{chr(10).join([m.content for m in context['memories']])}

当前对话:
{query}
\"\"\"
```
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional  # 类型提示

from nonebot import logger  # NoneBot日志记录器
from qdrant_client.http import models as qmodels  # Qdrant过滤器模型

# 导入项目模块
from ..config import plugin_config  # 插件配置
from ..memory.memory_manager import MemoryManager  # 记忆管理器
from ..storage.models import Memory  # 记忆模型
from ..storage.repositories.raw_repo import RawRepository  # 原始消息仓库
from ..storage.repositories.media_cache_repo import MediaCacheRepository  # 媒体缓存仓库
from ..storage.repositories.summary_repo import SummaryRepository  # 摘要仓库
from ..vector.embedder import embedder  # 向量化客户端
from ..vector.qdrant_client import qdrant_manager  # Qdrant客户端


class Retriever:
    """RAG检索器 - 负责上下文检索和记忆选择

    这个类的作用:
    - 构建智能的混合查询(Hybrid Query)
    - 从向量库检索相关历史消息片段
    - 选择合适的用户记忆作为上下文
    - 提供完整的降级处理,确保高可用性

    设计模式:
    - 静态方法类: 所有方法都是静态的,无需实例化
    - 好处: 作为工具类使用,避免状态管理

    核心方法:
    - build_hybrid_query(): 构建混合查询文本
    - retrieve(): 执行完整的RAG检索流程
    - _enrich_images(): 图片占位符增强(私有辅助方法)
    """

    @staticmethod
    async def _enrich_images(text: str) -> str:
        """将文本中的图片占位符尽力替换为带说明的形式

        这个方法的作用:
        - 识别消息中的图片占位符[image:xxx]
        - 查询MediaCache获取图片的caption(说明)
        - 将占位符替换为[image:xxx:说明文本]的形式
        - 帮助LLM理解图片内容

        为什么需要这个方法?
        - 消息中的图片在存储时被替换为[image:media_key]占位符
        - 占位符本身不包含图片内容信息,LLM无法理解
        - 通过添加caption,LLM可以知道图片大概内容
        - 例如: [image:a3f2b1c5d6e7] → [image:a3f2b1c5d6e7:一只橘猫]

        占位符格式:
        - 无说明: [image:a3f2b1c5d6e7]
        - 有说明: [image:a3f2b1c5d6e7:一只橘猫在沙发上]
        - media_key: 12位十六进制字符(SHA256哈希的前12位)

        Args:
            text: 包含图片占位符的文本
                - 类型: 字符串
                - 示例: "你看这个[image:a3f2b1c5d6e7]是不是很可爱"

        Returns:
            str: 替换后的文本
                - 成功: "[image:a3f2b1c5d6e7:一只橘猫]"
                - 失败/无caption: 保持原样"[image:a3f2b1c5d6e7]"

        性能优化:
        - 去重: 同一图片只查询一次
        - 限流: 最多处理前3张图片,避免过多数据库查询
        - 截断: caption最多20字符,避免过长

        Example:
            >>> text = "你看[image:abc123]和[image:def456]"
            >>> enriched = await Retriever._enrich_images(text)
            >>> print(enriched)
            # "你看[image:abc123:一只猫]和[image:def456:一只狗]"
        """

        # ==================== 步骤1: 导入正则表达式模块 ====================
        import re  # Python标准库,用于正则匹配

        # ==================== 步骤2: 空值检查 ====================
        if not text:  # 如果文本为空或None
            return text  # 直接返回,无需处理

        # ==================== 步骤3: 定义占位符的正则表达式 ====================

        # 占位符格式: [image:media_key] 或 [image:media_key:caption]
        # - (?P<key>[0-9a-f]{12}): 命名捕获组"key",匹配12位十六进制字符
        # - (?::(?P<cap>[^\]]+))?: 可选的caption部分
        #   * (?:...): 非捕获组
        #   * ::  匹配冒号
        #   * (?P<cap>[^\]]+): 命名捕获组"cap",匹配任意非]字符
        #   * ?: 整个caption部分是可选的
        pattern = re.compile(r"\[image:(?P<key>[0-9a-f]{12})(?::(?P<cap>[^\]]+))?\]")

        # ==================== 步骤4: 提取所有需要查询的media_key ====================

        keys: List[str] = []  # 需要查询caption的key列表

        # pattern.finditer(text): 遍历text中所有匹配的占位符
        for m in pattern.finditer(text):
            # m.group("cap"): 获取caption捕获组的内容
            if m.group("cap"):  # 如果已有caption
                continue  # 跳过,不需要查询

            # m.group("key"): 获取media_key捕获组的内容
            keys.append(m.group("key"))  # 添加到待查询列表

        # ==================== 步骤5: 如果没有需要处理的图片,直接返回 ====================
        if not keys:  # 如果列表为空
            return text  # 无需查询,直接返回原文本

        # ==================== 步骤6: 去重并限制数量 ====================

        # 去重: 保持原顺序的去重
        uniq = []  # 去重后的key列表
        for k in keys:
            if k not in uniq:  # 如果还没添加过
                uniq.append(k)  # 添加到去重列表

        # 限制数量: 最多处理前3张图片
        # [:3]: 切片,取前3个元素
        # 原因: 避免一次查询过多MediaCache记录,影响性能
        uniq = uniq[:3]

        # ==================== 步骤7: 批量查询MediaCache获取caption ====================

        captions: Dict[str, str] = {}  # key → caption的映射

        # 遍历去重后的key列表
        for k in uniq:
            try:
                # await MediaCacheRepository.get(k): 查询MediaCache表
                # 参数: media_key
                # 返回: MediaCache对象或None
                cached = await MediaCacheRepository.get(k)

                # cached.caption: 图片的说明文本
                if cached and cached.caption:  # 如果记录存在且有caption
                    # .strip(): 去除首尾空格
                    captions[k] = cached.caption.strip()

            except Exception:
                # 查询失败: 数据库错误、表不存在等
                # 静默忽略,继续处理下一个
                continue

        # ==================== 步骤8: 如果没有查到任何caption,直接返回 ====================
        if not captions:  # 如果字典为空
            return text  # 无法增强,返回原文本

        # ==================== 步骤9: 定义替换函数 ====================

        def repl(match: re.Match) -> str:
            """正则替换的回调函数

            Args:
                match: 正则匹配对象

            Returns:
                str: 替换后的字符串
            """

            # 提取匹配组
            key = match.group("key")  # media_key
            cap = match.group("cap")  # 已有的caption(可能是None)

            # 如果已有caption,保持不变
            if cap:
                return match.group(0)  # 返回原匹配字符串

            # 查询是否有新的caption
            caption = captions.get(key)
            if not caption:  # 如果没有查到
                return match.group(0)  # 保持原样

            # 截断caption为20字符
            # caption[:20]: 切片,取前20个字符
            # + ("…" if len(caption) > 20 else ""): 超过20字加省略号
            short = caption[:20] + ("…" if len(caption) > 20 else "")

            # 构建新的占位符: [image:key:caption]
            return f"[image:{key}:{short}]"

        # ==================== 步骤10: 执行正则替换 ====================

        # pattern.sub(repl, text): 用repl函数替换text中所有匹配
        # - 对每个匹配调用repl(match)
        # - 用返回值替换原匹配字符串
        return pattern.sub(repl, text)

    @staticmethod
    async def build_hybrid_query(
        qq_id: str,
        scene_type: str,
        scene_id: str,
        current_msg: str,
    ) -> str:
        """构建混合查询(Hybrid Query) - 当前消息+最近对话+摘要

        这个方法的作用:
        - 构建一个简洁但信息完整的查询文本
        - 结合当前消息和多种历史上下文
        - 提供给向量检索和LLM作为查询输入
        - 帮助理解用户的隐藏意图、代指、暗示

        为什么需要Hybrid Query?
        - 单纯用当前消息检索: 信息不足,可能遗漏关键上下文
        - 例如用户说"那个呢",如果没有上下文,无法理解"那个"指什么
        - Hybrid Query包含最近对话,可以理解代指和隐藏意图

        查询组成部分(按优先级):
        1. 【当前用户】当前消息(必有) - 用户刚发的消息
        2. 【最近用户】用户自己最近3条消息 - 理解连续提问/话题延续
        3. 【最近机器人】机器人最后一句话 - 对话连贯性
        4. 【最近对方】其他人最后一句话(群聊) - 理解群聊中的指代
        5. 【最近摘要】历史对话摘要(兜底) - 补充远期背景

        Args:
            qq_id: 当前用户的QQ号
                - 类型: 字符串
                - 用途: 区分用户自己的消息和其他人的消息
            scene_type: 场景类型
                - 类型: 字符串
                - 取值: "group"(群聊) 或 "private"(私聊)
            scene_id: 场景标识
                - 类型: 字符串
                - 内容: 群号或对方QQ号
            current_msg: 当前用户消息
                - 类型: 字符串
                - 示例: "今天天气怎么样"

        Returns:
            str: 混合查询文本,多行字符串,每行一个上下文部分
                格式:
                【当前用户】今天天气怎么样
                【最近用户】我在北京 / 明天想出去玩
                【最近机器人】好的,我帮你查一下
                【最近对方】北京这两天降温了
                【最近摘要】最近在讨论北京的天气和出行计划

        降级策略:
            - 数据库查询失败: 只返回【当前用户】部分
            - 摘要查询失败: 忽略摘要,返回其他部分
            - 图片caption查询失败: 图片占位符保持原样

        Example:
            >>> query = await Retriever.build_hybrid_query(
            ...     qq_id="123",
            ...     scene_type="group",
            ...     scene_id="456",
            ...     current_msg="那个怎么样"
            ... )
            >>> print(query)
            # 【当前用户】那个怎么样
            # 【最近用户】我想买个笔记本电脑 / 预算5000左右
            # 【最近机器人】联想小新Pro 14很适合你
            # → 有了上下文,可以理解"那个"指的是"联想小新Pro 14"
        """

        # ==================== 步骤1: 增强当前消息(替换图片占位符) ====================

        # await Retriever._enrich_images(current_msg): 替换图片占位符为带说明的形式
        current_msg = await Retriever._enrich_images(current_msg)

        # ==================== 步骤2: 初始化查询部分列表 ====================

        # parts: 查询文本的各个部分,最后用换行符拼接
        parts: List[str] = [f"【当前用户】{current_msg}"]
        # 第一部分: 当前消息(必有)

        # ==================== 步骤3: 初始化上下文变量 ====================

        last_peer_text: Optional[str] = None  # 其他人最后一句话
        last_bot_text: Optional[str] = None  # 机器人最后一句话
        recent_user_texts: List[str] = []  # 用户自己最近的消息列表

        # ==================== 步骤4: 查询最近消息 ====================

        try:
            # await RawRepository.get_recent_by_scene(): 查询场景的最近消息
            # 参数:
            # - scene_type: 场景类型
            # - scene_id: 场景标识
            # - limit=30: 查询最近30条消息
            # 返回: RawMessage对象列表,按时间倒序(最新在前)
            recent = await RawRepository.get_recent_by_scene(scene_type, scene_id, limit=30)

            # ==================== 步骤5: 遍历最近消息,提取有用上下文 ====================

            for m in recent:
                # ==================== 情况1: 机器人的消息 ====================

                # getattr(m, "is_bot", False): 安全获取is_bot属性,默认False
                if getattr(m, "is_bot", False):  # 如果是机器人发的消息
                    # 只记录最后一句机器人回复
                    if last_bot_text is None and m.content:
                        last_bot_text = m.content  # 保存机器人最后一句话
                    continue  # 继续处理下一条消息

                # ==================== 情况2: 当前用户自己的消息 ====================

                if m.qq_id == qq_id:  # 如果是当前用户发的
                    # 收集用户最近的3条消息(不包括当前消息)
                    # 用途: 理解用户的连续提问、话题延续、隐藏意图
                    # 条件:
                    # - m.content: 有内容
                    # - m.content != current_msg: 不是当前消息(避免重复)
                    # - len(recent_user_texts) < 3: 最多3条
                    if m.content and m.content != current_msg and len(recent_user_texts) < 3:
                        recent_user_texts.append(m.content)  # 添加到列表
                    continue  # 继续处理下一条消息

                # ==================== 情况3: 其他用户的消息(群聊场景) ====================

                # 只记录最后一句其他人说的话
                # 用途: 群聊中理解指代、暗示、话题
                if last_peer_text is None and m.content:
                    last_peer_text = m.content  # 保存其他人最后一句话

        except Exception as exc:
            # 查询失败: 数据库错误、表不存在等
            # 降级: 只使用当前消息,不阻塞主流程
            logger.warning(f"读取最近消息失败,将降级为仅当前消息:{exc}")

        # ==================== 步骤6: 添加【最近用户】部分 ====================

        if recent_user_texts:  # 如果有用户最近消息
            # 增强图片占位符
            # [await Retriever._enrich_images(t) for t in recent_user_texts[:3]]:
            # - 列表推导式
            # - 对每条消息调用_enrich_images
            # - [:3]: 最多取前3条
            enriched = [await Retriever._enrich_images(t) for t in recent_user_texts[:3]]

            # " / ".join(enriched): 用" / "连接多条消息
            # 例如: "消息1 / 消息2 / 消息3"
            parts.append("【最近用户】" + " / ".join(enriched))

        # ==================== 步骤7: 添加【最近机器人】部分 ====================

        if last_bot_text:  # 如果有机器人最后回复
            # 增强图片占位符后添加
            parts.append(f"【最近机器人】{await Retriever._enrich_images(last_bot_text)}")

        # ==================== 步骤8: 添加【最近对方】部分 ====================

        if last_peer_text:  # 如果有其他人最后一句话(群聊场景)
            # 增强图片占位符后添加
            parts.append(f"【最近对方】{await Retriever._enrich_images(last_peer_text)}")

        # ==================== 步骤9: 添加【最近摘要】部分(兜底) ====================

        try:
            # await SummaryRepository.get_latest(): 查询场景的最新摘要
            # 参数: scene_type, scene_id
            # 返回: Summary对象或None
            last_summary = await SummaryRepository.get_latest(scene_type, scene_id)

            # 仅在缺少"最近对方/最近机器人"时用摘要兜底
            # 原因: 避免query过长,最近对话优先级高于摘要
            if last_summary and (not last_peer_text) and (not last_bot_text):
                # last_summary.summary_text: 摘要的文本内容
                parts.append(f"【最近摘要】{last_summary.summary_text}")

        except Exception as exc:
            # 摘要查询失败: 数据库错误等
            # 静默忽略,不影响其他部分
            logger.warning(f"读取摘要失败,将忽略摘要:{exc}")

        # ==================== 步骤10: 输出调试日志 ====================

        # logger.debug(parts): 输出查询的各个部分,用于调试
        logger.debug(parts)

        # ==================== 步骤11: 拼接并返回查询文本 ====================

        # "\n".join(parts): 用换行符连接各个部分
        # 每个部分独占一行,便于阅读和解析
        return "\n".join(parts)

    @staticmethod
    async def retrieve(
        qq_id: str,
        scene_type: str,
        scene_id: str,
        query: str,
    ) -> Dict[str, Any]:
        """执行完整的RAG检索流程 - 向量检索+记忆选择

        这个方法的作用:
        - 协调整个RAG检索流程
        - 调用MemoryManager选择用户记忆
        - 调用embedder和qdrant_manager进行向量检索
        - 返回RAG片段和记忆列表供LLM使用

        RAG检索流程:
        1. 选择用户记忆(active/core层级,按相关度排序)
        2. 将query向量化(文本→2048维向量)
        3. 在Qdrant中搜索最相似的向量(top-k)
        4. 提取相似向量的payload文本
        5. 返回RAG片段和记忆列表

        过滤策略:
        - 场景过滤: 只检索当前场景(群/私聊)的消息
        - 排除机器人: 不检索机器人自己发的消息(避免复读)
        - 长度截断: 片段最多max_chars字符(默认200)

        Args:
            qq_id: 用户QQ号
                - 类型: 字符串
                - 用途: 选择该用户的记忆
            scene_type: 场景类型
                - 类型: 字符串
                - 取值: "group" 或 "private"
                - 用途: 过滤检索范围
            scene_id: 场景标识
                - 类型: 字符串
                - 内容: 群号或对方QQ号
                - 用途: 过滤检索范围
            query: 混合查询文本
                - 类型: 字符串
                - 来源: build_hybrid_query()的返回值
                - 用途: 向量化后作为检索query

        Returns:
            Dict[str, Any]: 检索结果字典
                格式: {
                    "rag_snippets": List[str],  # RAG片段列表
                    "memories": List[Memory],   # 记忆对象列表
                }

            rag_snippets示例:
                ["去年冬天北京很冷",
                 "记得带羽绒服",
                 "天气预报说明天下雪"]

            memories示例:
                [Memory(content="用户住在北京", confidence=0.9),
                 Memory(content="用户怕冷", confidence=0.8)]

        降级策略:
            - MemoryManager失败: 返回空记忆列表
            - embedder失败: 返回空RAG片段
            - qdrant_manager失败: 返回空RAG片段
            - 任何异常都不会抛出,确保主流程继续

        性能配置:
            - yuying_retrieval_topk: 检索top-k数量(默认5)
            - yuying_retrieval_snippet_max_chars: 片段最大字符数(默认200)

        Example:
            >>> query = await Retriever.build_hybrid_query(...)
            >>> context = await Retriever.retrieve(
            ...     qq_id="123",
            ...     scene_type="group",
            ...     scene_id="456",
            ...     query=query
            ... )
            >>> print(len(context["rag_snippets"]))  # 5
            >>> print(len(context["memories"]))      # 3
            >>> # 然后将context传给LLM作为上下文
        """

        # ==================== 步骤1: 初始化返回结果 ====================

        memories: List[Memory] = []  # 记忆对象列表
        rag_snippets: List[str] = []  # RAG片段文本列表

        # ==================== 步骤2: 选择用户记忆 ====================

        try:
            # await MemoryManager.select_for_context(): 选择相关记忆
            # 参数:
            # - qq_id: 用户QQ号
            # - scene_type: 场景类型
            # - scene_id: 场景标识
            # - query: 查询文本(用于计算相关度)
            # 返回: Memory对象列表,按相关度排序
            memories = await MemoryManager.select_for_context(qq_id, scene_type, scene_id, query)

        except Exception as exc:
            # 记忆选择失败: 数据库错误、MemoryManager内部异常等
            # 降级: 返回空记忆列表,不影响RAG检索
            logger.error(f"选择记忆上下文失败,将降级为空记忆:{exc}")

        # ==================== 步骤3: 向量检索(Embedder + Qdrant) ====================

        # 检索增强: 向量化与向量库检索
        # 任何一步失败都不阻塞主流程,只记录警告并返回空结果

        try:
            # ==================== 步骤3.1: 向量化query ====================

            # await embedder.get_embedding(query): 将query文本转为向量
            # - 输入: 混合查询文本(可能很长)
            # - 输出: 浮点数列表,如2048维向量
            # - 模型: yuying_embedder_model配置的embedding模型
            vector = await embedder.get_embedding(query)

            # ==================== 步骤3.2: 构建Qdrant过滤条件 ====================

            # qmodels.Filter: Qdrant的过滤器对象
            # must: 必须满足的条件列表(AND逻辑)
            filt = qmodels.Filter(
                must=[
                    # 条件1: scene_type字段必须等于当前场景类型
                    # qmodels.FieldCondition: 字段条件
                    # qmodels.MatchValue: 精确匹配值
                    qmodels.FieldCondition(key="scene_type", match=qmodels.MatchValue(value=scene_type)),

                    # 条件2: scene_id字段必须等于当前场景标识
                    qmodels.FieldCondition(key="scene_id", match=qmodels.MatchValue(value=scene_id)),
                ]
            )

            # ==================== 步骤3.3: 添加排除条件(避免复读) ====================

            # must_not: 必须不满足的条件列表(NOT逻辑)
            # 排除机器人自己发的消息
            # 原因: 避免检索到机器人之前的回复,导致"复读机"现象
            filt.must_not = [
                # is_bot字段必须不等于True
                qmodels.FieldCondition(key="is_bot", match=qmodels.MatchValue(value=True)),
            ]

            # ==================== 步骤3.4: 执行向量检索 ====================

            # await qdrant_manager.search(): 在Qdrant中搜索最相似向量
            # 参数:
            # - collection_name="rag_items": 搜索消息向量collection
            # - vector: 查询向量(2048维)
            # - limit: 返回top-k个最相似结果
            # - query_filter: 过滤条件(场景+排除机器人)
            # 返回: List[ScoredPoint],每个包含score和payload
            results = await qdrant_manager.search(
                collection_name="rag_items",
                vector=vector,
                limit=int(plugin_config.yuying_retrieval_topk),  # top-k配置
                query_filter=filt,
            )

            # ==================== 步骤3.5: 提取RAG片段文本 ====================

            # int(plugin_config.yuying_retrieval_snippet_max_chars): 片段最大字符数
            max_chars = int(plugin_config.yuying_retrieval_snippet_max_chars)

            # 遍历检索结果
            for r in results:
                # r.payload: ScoredPoint的payload字段,存储消息的原始数据
                # 类型: 字典或None
                payload = r.payload or {}

                # payload.get("text", ""): 获取text字段,默认空字符串
                # str(...): 确保是字符串类型
                # .strip(): 去除首尾空格
                text = str(payload.get("text", "")).strip()

                # 跳过空文本
                if not text:
                    continue

                # ==================== 步骤3.6: 截断过长文本 ====================

                if len(text) > max_chars:  # 如果超过最大字符数
                    # 截断并添加省略号
                    # text[:max_chars]: 取前max_chars个字符
                    # + "…": 添加省略号标记
                    text = text[:max_chars] + "…"

                # ==================== 步骤3.7: 添加到RAG片段列表 ====================

                rag_snippets.append(text)

        except Exception as exc:
            # RAG检索失败: embedder失败、Qdrant失败、网络错误等
            # 降级: 返回空RAG片段列表,不影响主流程
            # logger.warning: 记录警告级别日志
            logger.warning(f"RAG 检索失败,将降级为空 RAG:{exc}")

        # ==================== 步骤4: 返回检索结果 ====================

        # 返回字典,包含RAG片段和记忆列表
        # 即使检索失败,也会返回空列表,不会抛异常
        return {"rag_snippets": rag_snippets, "memories": memories}
