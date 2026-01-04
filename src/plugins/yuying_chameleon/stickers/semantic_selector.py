"""表情包语义检索选择器：基于向量相似度和混合rerank的智能表情包匹配

这个模块的作用:
1. 使用向量语义检索替代传统的 intent+SQL 匹配
2. 实现混合检索策略：向量召回 topK → tag/intent 精排 → cooldown 过滤
3. 提供详细的日志记录便于调优和问题排查

为什么需要语义检索?
- 传统 intent 匹配只有 9 个粗粒度分类，表达能力有限
- 向量检索能理解更复杂的语义，如"尴尬"、"无语"等细微情绪
- 结合 tag 和 intent 精排，既保证语义相关性，又避免丢失特殊梗图

技术架构:
- 向量召回：用户消息 embedding → Qdrant stickers collection 搜索 topK
- Tag 精排：子串匹配计算 tag 命中数，调整相似度分数
- Intent 精排：匹配当前意图，给予额外加分
- Cooldown 过滤：避免短时间内重复发送同一表情包

使用方式:
```python
from .stickers.semantic_selector import StickerSemanticSelector

# 语义检索选择表情包
sticker = await StickerSemanticSelector.select_sticker(
    intent="funny",          # LLM 输出的意图
    query_text="哈哈太搞笑了",  # 用于向量检索的文本
    scene_type="group",
    scene_id="123456"
)
```
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from nonebot import logger
from qdrant_client.http import models as qmodels

from ..config import plugin_config
from ..storage.models import Sticker
from ..storage.repositories.sticker_repo import StickerRepository
from ..storage.repositories.sticker_usage_repo import StickerUsageRepository
from ..vector.embedder import embedder
from ..vector.qdrant_client import qdrant_manager
from .selector import StickerSelector  # 降级用的旧版 selector


class StickerSemanticSelector:
    """表情包语义检索选择器（基于向量相似度 + 混合精排）

    这个类的作用:
    - 实现基于向量的语义检索
    - 融合 tag 和 intent 信息进行精排
    - 提供详细的日志记录
    - 失败时降级到旧版 SQL selector

    设计特点:
    - 所有方法都是静态方法，无状态设计
    - 完善的降级策略保证可用性
    - 详细的日志便于调优

    核心流程:
    1. Query embedding（文本向量化）
    2. Qdrant 向量检索（召回 topK）
    3. Rerank（根据 tag/intent 重新排序）
    4. Cooldown 过滤（跳过冷却期内的表情包）
    5. 返回得分最高的表情包
    """

    @staticmethod
    def _split_csv(s: Optional[str]) -> List[str]:
        """将逗号分隔的字符串拆分为列表

        这个方法的作用:
        - 复用 IndexWorker 的逻辑，保持一致性
        - 将 "tag1, tag2, tag3" 拆分为 ["tag1", "tag2", "tag3"]

        Args:
            s: 逗号分隔的字符串

        Returns:
            list[str]: 拆分后的列表
        """
        raw = (s or "").strip()
        if not raw:
            return []
        return [p.strip() for p in raw.split(",") if p.strip()]

    @staticmethod
    def _normalize_vector_score(score: float) -> float:
        """将向量相似度归一化到 [0, 1] 区间

        这个方法的作用:
        - Qdrant 的 cosine 相似度范围是 [-1, 1]
        - 归一化到 [0, 1] 便于与其他分数（tag, intent）组合
        - 避免负数分数干扰排序

        为什么需要归一化?
        - 不同度量（向量相似度、tag 匹配数）的量纲不同
        - 归一化后可以直接加权组合
        - 便于调整各部分权重

        Args:
            score: 原始向量相似度分数
                - 范围: [-1, 1]
                - 1 表示完全相同
                - -1 表示完全相反
                - 0 表示正交（无关）

        Returns:
            float: 归一化后的分数 [0, 1]
                - 1 表示完全相同
                - 0 表示完全相反
                - 0.5 表示正交

        Example:
            >>> StickerSemanticSelector._normalize_vector_score(1.0)
            1.0
            >>> StickerSemanticSelector._normalize_vector_score(0.0)
            0.5
            >>> StickerSemanticSelector._normalize_vector_score(-1.0)
            0.0
        """
        # 线性归一化: (x + 1) / 2
        # -1 → 0, 0 → 0.5, 1 → 1
        return (score + 1.0) / 2.0

    @staticmethod
    def _rerank(
        *,
        query_text: str,
        intent: str,
        vector_score: float,
        tags: List[str],
        intents: List[str],
    ) -> float:
        """对向量召回的结果进行精排（Rerank）

        这个方法的作用:
        - 结合向量相似度、tag 匹配、intent 匹配计算综合分数
        - 使用子串匹配而非完整匹配（LLM 不会完整打出 tag）
        - 归一化向量分数避免量纲不一致

        精排算法:
        1. 向量相似度（归一化）：基础分数，权重 70%
        2. Intent 匹配：如果匹配则加 0.2 分，权重 20%
        3. Tag 匹配：子串匹配计数，每个命中 +0.05，最多 +0.1，权重 10%

        为什么用子串匹配?
        - LLM 通常不会完整输出 tag（例如："太搞笑了" vs "搞笑"）
        - 子串匹配可以捕获部分匹配，提高召回率
        - 用户明确表示这是合理的策略

        Args:
            query_text: 查询文本
                - 用于 tag 子串匹配
                - 示例: "哈哈太搞笑了"
            intent: 当前意图
                - LLM 输出的 intent
                - 示例: "funny", "cute"
            vector_score: 原始向量相似度分数
                - 范围: [-1, 1]
                - 来自 Qdrant 搜索结果
            tags: 表情包的 tags 列表
                - 从 payload.tags_list 获取
                - 示例: ["搞笑", "猫咪"]
            intents: 表情包的 intents 列表
                - 从 payload.intents_list 获取
                - 示例: ["funny", "cute"]

        Returns:
            float: 综合排序分数 [0, ~1.3]
                - 向量相似度贡献: [0, 1]
                - Intent 匹配贡献: 0 或 0.2
                - Tag 匹配贡献: [0, 0.1]
                - 理论最大值: 1.0 + 0.2 + 0.1 = 1.3

        Example:
            >>> # 向量分数 0.8，intent 匹配，1 个 tag 命中
            >>> score = StickerSemanticSelector._rerank(
            ...     query_text="太搞笑了",
            ...     intent="funny",
            ...     vector_score=0.8,
            ...     tags=["搞笑", "猫咪"],
            ...     intents=["funny"]
            ... )
            >>> # 归一化: (0.8+1)/2=0.9, intent: +0.2, tag: +0.05
            >>> # 总分: 0.9*0.7 + 0.2 + 0.05 ≈ 0.88
        """

        # ==================== 步骤1: 归一化向量分数 ====================

        normalized_vector = StickerSemanticSelector._normalize_vector_score(
            float(vector_score)
        )

        # ==================== 步骤2: Intent 匹配加分 ====================

        # intent 归一化：去除首尾空格并转小写
        intent_normalized = (intent or "").strip().lower()

        # 检查 intent 是否在表情包的 intents 列表中
        # intents 也归一化处理
        intent_bonus = 0.0
        if intent_normalized and any(
            intent_normalized == i.strip().lower() for i in intents if i
        ):
            intent_bonus = 0.2  # Intent 匹配加 0.2 分

        # ==================== 步骤3: Tag 子串匹配加分 ====================

        # 统计有多少个 tag 在 query_text 中出现
        tag_hit_count = 0
        query_lower = query_text.lower()  # 查询文本转小写

        for tag in tags:
            if not tag:  # 跳过空 tag
                continue
            # 子串匹配：tag 是否在 query 中
            # 转小写进行不区分大小写的匹配
            if tag.strip().lower() in query_lower:
                tag_hit_count += 1

        # 每个命中的 tag 加 0.05 分，最多加 4 个（总共 +0.2）
        # 但为了避免 tag 权重过高，最多 +0.1
        tag_bonus = min(tag_hit_count * 0.05, 0.1)

        # ==================== 步骤4: 加权组合 ====================

        # 最终分数 = 向量分数 * 0.7 + intent 加分 + tag 加分
        # 权重分配：
        # - 向量相似度：70%（主要依据）
        # - Intent 匹配：20%（保证意图准确）
        # - Tag 匹配：10%（辅助提升精准度）
        final_score = normalized_vector * 0.7 + intent_bonus + tag_bonus

        return final_score

    @staticmethod
    async def select_sticker(
        intent: str, query_text: str, scene_type: str, scene_id: str
    ) -> Optional[Sticker]:
        """语义检索选择表情包（主入口）

        这个方法的作用:
        - 完整的语义检索流程：向量召回 → rerank → cooldown 过滤
        - 详细的日志记录便于调试和调优
        - 失败时自动降级到旧版 SQL selector

        流程:
        1. Query 文本向量化
        2. Qdrant 向量检索（召回 topK）
        3. Rerank（精排）
        4. Cooldown 过滤
        5. 返回得分最高的可用表情包

        Args:
            intent: LLM 输出的意图
                - 示例: "funny", "cute", "sad"
                - 用于 intent 匹配加分
            query_text: 用于向量检索的文本
                - 应该是机器人当前回复中的 text 内容
                - 示例: "哈哈太搞笑了"
            scene_type: 场景类型
                - 示例: "group", "private"
                - 用于 cooldown 检查
            scene_id: 场景 ID
                - 示例: 群号或用户 QQ 号
                - 用于 cooldown 检查

        Returns:
            Optional[Sticker]: 选中的表情包
                - 成功: 返回 Sticker 对象
                - 失败: 返回 None
                - 降级: 调用旧版 selector

        Side Effects:
            - 调用 embedder.get_embedding（耗时操作）
            - 调用 Qdrant 搜索（网络请求）
            - 查询数据库（sticker 详情、cooldown）
            - 输出详细日志

        Example:
            >>> sticker = await StickerSemanticSelector.select_sticker(
            ...     intent="funny",
            ...     query_text="哈哈太搞笑了",
            ...     scene_type="group",
            ...     scene_id="123456"
            ... )
        """

        # ==================== 步骤1: 读取配置 ====================

        # topK: 向量召回数量（默认 50）
        top_k = int(getattr(plugin_config, "yuying_sticker_vector_top_k", 50) or 50)

        # cooldown: 冷却时间（秒）
        cooldown = int(plugin_config.yuying_sticker_cooldown_seconds)

        logger.info(
            f"[语义检索] 开始选择表情包: intent={intent}, query_text={query_text}, "
            f"scene={scene_type}:{scene_id}, topK={top_k}, cooldown={cooldown}s"
        )

        try:
            # ==================== 步骤2: Query 文本向量化 ====================

            logger.debug(f"[语义检索] 开始向量化查询文本: {query_text}")
            qvec = await embedder.get_embedding(query_text)
            logger.debug(
                f"[语义检索] 查询文本向量化完成, 维度: {len(qvec)}, "
                f"前5维: {[round(x, 3) for x in qvec[:5]]}"
            )

            # ==================== 步骤3: Qdrant 向量检索（硬过滤 + 召回）====================

            # 构建过滤条件：只召回可用且未封禁的表情包
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="kind", match=qmodels.MatchValue(value="sticker")
                    ),
                    qmodels.FieldCondition(
                        key="is_enabled", match=qmodels.MatchValue(value=True)
                    ),
                    qmodels.FieldCondition(
                        key="is_banned", match=qmodels.MatchValue(value=False)
                    ),
                ]
            )

            logger.debug(
                f"[语义检索] 开始 Qdrant 搜索: collection=stickers, topK={top_k}"
            )
            points = await qdrant_manager.search(
                collection_name="stickers",
                vector=qvec,
                limit=top_k,
                query_filter=query_filter,
            )
            logger.info(
                f"[语义检索] Qdrant 召回完成: 返回 {len(points)} 个候选表情包"
            )

            # 如果没有召回任何结果，降级
            if not points:
                logger.warning("[语义检索] Qdrant 未召回任何结果，降级到 SQL selector")
                return await StickerSelector.select_sticker(intent, scene_type, scene_id)

            # ==================== 步骤4: Rerank（精排） ====================

            ranked: List[Tuple[float, str, float]] = []  # (final_score, sticker_id, vector_score)

            for p in points:
                # 获取 payload
                payload: Dict[str, Any] = dict(getattr(p, "payload", None) or {})
                sticker_id = str(payload.get("sticker_id") or "").strip()

                if not sticker_id:
                    # payload 缺少 sticker_id，跳过
                    continue

                # 获取 tags 和 intents（优先使用数组格式）
                tags_list = payload.get("tags_list")
                intents_list = payload.get("intents_list")

                # 如果数组格式不存在，从字符串格式解析
                if not isinstance(tags_list, list):
                    tags = StickerSemanticSelector._split_csv(
                        str(payload.get("tags") or "")
                    )
                else:
                    tags = list(tags_list)

                if not isinstance(intents_list, list):
                    intents_ = StickerSemanticSelector._split_csv(
                        str(payload.get("intents") or "")
                    )
                else:
                    intents_ = list(intents_list)

                # 获取原始向量分数
                vector_score = float(getattr(p, "score", 0.0) or 0.0)

                # 计算 rerank 分数
                final_score = StickerSemanticSelector._rerank(
                    query_text=query_text,
                    intent=intent,
                    vector_score=vector_score,
                    tags=tags,
                    intents=intents_,
                )

                ranked.append((final_score, sticker_id, vector_score))

            # 按 final_score 降序排序
            ranked.sort(key=lambda x: x[0], reverse=True)

            logger.info(
                f"[语义检索] Rerank 完成: 候选数={len(ranked)}, "
                f"Top3分数: {[round(x[0], 3) for x in ranked[:3]]}"
            )

            # ==================== 步骤5: Cooldown 过滤 + 选择 ====================

            current_ts = int(time.time())
            for final_score, sid, vector_score in ranked:
                # 从数据库获取完整的 sticker 对象
                s = await StickerRepository.get_by_id(sid)

                if not s:
                    # 数据库中已不存在（可能被删除）
                    logger.debug(
                        f"[语义检索] 跳过不存在的表情包: sticker_id={sid}"
                    )
                    continue

                if not s.is_enabled or s.is_banned:
                    # 再次检查状态（防止 Qdrant 数据陈旧）
                    logger.debug(
                        f"[语义检索] 跳过已禁用/封禁的表情包: sticker_id={sid}, "
                        f"is_enabled={s.is_enabled}, is_banned={s.is_banned}"
                    )
                    continue

                # 检查 cooldown
                last_used = await StickerUsageRepository.get_last_used_ts(
                    scene_type, scene_id, s.sticker_id
                )

                if last_used and (current_ts - int(last_used) < cooldown):
                    elapsed = current_ts - int(last_used)
                    logger.debug(
                        f"[语义检索] 跳过冷却期内的表情包: sticker_id={sid}, "
                        f"冷却期={cooldown}s, 已过={elapsed}s"
                    )
                    continue

                # 找到了！
                logger.info(
                    f"[语义检索] 选中表情包: sticker_id={sid}, name={s.name}, "
                    f"final_score={round(final_score, 3)}, vector_score={round(vector_score, 3)}, "
                    f"tags={s.tags}, intents={s.intents}"
                )
                return s

            # 所有候选都因为 cooldown 或状态问题被过滤掉了
            logger.warning(
                "[语义检索] 所有候选表情包均被过滤（cooldown/状态），返回 None"
            )
            return None

        except Exception as exc:
            # 任何异常都降级到旧版 selector
            logger.warning(
                f"[语义检索] 向量检索失败，降级到 SQL selector: {exc}", exc_info=True
            )
            return await StickerSelector.select_sticker(intent, scene_type, scene_id)
