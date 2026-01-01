"""Qdrant向量数据库客户端管理模块 - 封装向量存储和检索操作

这个模块的作用:
1. 提供Qdrant向量数据库的异步客户端封装
2. 管理向量collection的创建和初始化
3. 提供向量的存储(upsert)和检索(search)接口
4. 使用单例模式确保全局只有一个客户端实例

Qdrant是什么?(新手必读)
- Qdrant: 专门用于向量检索的数据库
- 向量: 文本转为的数字列表(如[0.1, -0.2, 0.3, ...])
- 作用: 根据向量相似度快速找到相关内容
- 优势: 比SQLite快几个数量级,专为向量优化

本项目的3个向量Collection:
1. rag_items: 存储消息向量,用于上下文检索(RAG)
2. memories: 存储用户记忆向量,用于记忆检索
3. stickers: 存储表情包向量,用于表情包匹配

向量检索原理:
- 查询文本 → Embedding模型 → 查询向量
- 在Qdrant中搜索最相似的向量(cosine距离)
- 返回top-k最相关的结果
- 结合payload(附加数据)返回完整信息

为什么使用单例模式?
- Qdrant客户端应该全局唯一
- 避免重复创建连接,节省资源
- 确保配置一致性

使用方式:
```python
from .vector.qdrant_client import qdrant_manager

# 初始化collections(启动时调用一次)
await qdrant_manager.init_collections()

# 存储向量
await qdrant_manager.upsert_text_point(
    collection_name="rag_items",
    point_id="msg_123",
    vector=[0.1, -0.2, ...],  # 2048维向量
    payload={"content": "今天天气真好", "timestamp": 1234567890}
)

# 检索向量
results = await qdrant_manager.search(
    collection_name="rag_items",
    vector=[0.15, -0.25, ...],  # 查询向量
    limit=5  # 返回top-5结果
)
for point in results:
    print(point.payload["content"], point.score)  # 内容和相似度分数
```
"""

from __future__ import annotations

from nonebot import logger  # NoneBot日志记录器
from qdrant_client import AsyncQdrantClient  # Qdrant异步客户端
from qdrant_client.http import models  # Qdrant数据模型
from typing import List, Optional, Dict, Any, Union  # 类型提示

from ..config import plugin_config  # 导入插件配置


class QdrantManager:
    """Qdrant客户端单例管理器

    这个类的作用:
    - 管理全局唯一的Qdrant客户端实例
    - 提供collection初始化和管理
    - 提供向量的存储和检索接口
    - 处理Qdrant不可用时的降级

    设计模式:
    - 单例模式(Singleton): 确保全局只有一个实例
    - 实现方式: __new__方法控制实例创建

    为什么需要单例?
    - Qdrant连接应该全局共享
    - 避免重复创建客户端,节省资源
    - 确保配置一致性

    类属性:
    - _instance: 单例实例(类级别,所有实例共享)
    - client: AsyncQdrantClient客户端实例

    实例方法:
    - init_collections(): 初始化向量collections
    - upsert_text_point(): 存储向量点
    - search(): 向量检索

    静态方法:
    - _extract_vector_size(): 从collection信息中提取维度
    """

    # ==================== 类属性(单例模式) ====================

    _instance = None  # 单例实例,初始为None
    client: AsyncQdrantClient  # Qdrant异步客户端(类型提示)

    def __new__(cls):
        """创建/获取单例实例,并初始化Qdrant客户端

        __new__方法说明(Python对象创建机制):
        - __new__: 在__init__之前调用,负责创建实例
        - __init__: 在__new__之后调用,负责初始化实例
        - 单例模式: 通过控制__new__确保只创建一个实例

        这个方法的作用:
        - 第一次调用: 创建实例并初始化Qdrant客户端
        - 后续调用: 直接返回已存在的实例
        - 读取配置: host、port、api_key、https
        - 创建客户端: AsyncQdrantClient实例

        Returns:
            QdrantManager: 单例实例

        Side Effects:
            - 创建AsyncQdrantClient连接
            - 读取plugin_config配置

        Example:
            >>> manager1 = QdrantManager()
            >>> manager2 = QdrantManager()
            >>> manager1 is manager2  # True,同一个实例
        """

        # ==================== 步骤1: 检查单例是否已存在 ====================

        if cls._instance is None:  # 如果还没有创建实例
            # ==================== 步骤2: 创建新实例 ====================

            # super(QdrantManager, cls).__new__(cls): 调用父类的__new__创建实例
            # cls: 当前类(QdrantManager)
            # super(): 获取父类(object)
            cls._instance = super(QdrantManager, cls).__new__(cls)

            # ==================== 步骤3: 读取Qdrant配置 ====================

            # API密钥(可选,本地部署通常不需要)
            # (plugin_config.yuying_qdrant_api_key or "").strip(): 去空格
            # or None: 空字符串转为None
            api_key = (plugin_config.yuying_qdrant_api_key or "").strip() or None

            # Qdrant服务器地址
            # (plugin_config.yuying_qdrant_host or "localhost").strip(): 去空格
            # or "localhost": 默认本地
            host = (plugin_config.yuying_qdrant_host or "localhost").strip() or "localhost"

            # 是否使用HTTPS
            # plugin_config.yuying_qdrant_https: 从配置读取
            # 可能是True、False或None
            https = plugin_config.yuying_qdrant_https

            # 本地部署的特殊处理
            if https is None and host in {"localhost", "127.0.0.1"}:
                # qdrant-client在api_key非None时会默认启用https
                # 但本地自建Qdrant通常是HTTP,所以这里强制关闭
                # 原因: 本地部署通常不配置SSL证书
                https = False

            # ==================== 步骤4: 创建Qdrant客户端 ====================

            # AsyncQdrantClient: Qdrant的异步客户端
            # host: 服务器地址
            # port: 服务器端口(默认6333)
            # api_key: API密钥(可选)
            # https: 是否使用HTTPS连接
            cls._instance.client = AsyncQdrantClient(
                host=host,
                port=plugin_config.yuying_qdrant_port,
                api_key=api_key,
                https=https,
            )

        # ==================== 步骤5: 返回单例实例 ====================

        return cls._instance  # 返回已创建的实例(第一次创建,后续直接返回)

    async def init_collections(self):
        """初始化Qdrant的向量collections(不存在则创建)

        这个方法的作用:
        - 检查3个collection是否存在
        - 不存在则创建新collection
        - 已存在则检查向量维度是否匹配
        - 维度不匹配时根据配置决定是否重建

        3个Collection说明:
        1. rag_items: 消息向量,用于上下文检索(RAG)
           - 存储: 每条消息转为向量
           - 检索: 根据查询找到相关消息
        2. memories: 用户记忆向量,用于记忆检索
           - 存储: 每条记忆转为向量
           - 检索: 根据对话找到相关记忆
        3. stickers: 表情包向量,用于表情包匹配
           - 存储: 表情包描述转为向量
           - 检索: 根据情绪找到合适表情

        降级策略:
        - Qdrant不可用时: 只记录警告,不阻塞启动
        - 创建失败时: 继续处理下一个collection
        - 好处: 即使向量库故障,机器人仍可运行(只是失去检索功能)

        Side Effects:
            - 创建Qdrant collections
            - 可能删除并重建collection(如果维度不匹配且配置了recreate)
            - 输出日志信息

        配置项:
            - vector_size: 向量维度(默认2048,需与embedding模型匹配)
            - qdrant_recreate_collections: 是否自动重建不匹配的collection

        Example:
            >>> await qdrant_manager.init_collections()
            # 输出: Creating Qdrant collection: rag_items
            # 输出: Creating Qdrant collection: memories
            # 输出: Creating Qdrant collection: stickers
        """

        # ==================== 步骤1: 读取配置 ====================

        # 向量维度大小
        # getattr(plugin_config, "yuying_vector_size", 2048): 获取配置,默认2048
        # int(...): 转为整数
        # or 2048: 如果是None或0,使用2048
        vector_size = int(getattr(plugin_config, "yuying_vector_size", 2048) or 2048)

        # 是否自动重建不匹配的collection
        # bool(...): 转为布尔值
        recreate = bool(getattr(plugin_config, "yuying_qdrant_recreate_collections", False))

        # collections字典: {collection名称: 向量维度}
        collections = {
            "rag_items": vector_size,  # 消息向量(RAG检索)
            "memories": vector_size,  # 记忆向量(记忆检索)
            "stickers": vector_size,  # 表情包向量(表情匹配)
        }

        # ==================== 步骤2: 获取已存在的collections ====================

        try:
            # await self.client.get_collections(): 获取所有collection列表
            # 异步调用,需要await
            existing = await self.client.get_collections()
        except Exception as exc:
            # Qdrant不可用(未启动、网络问题、认证失败等)
            # 降级策略: 只记录警告,不阻塞插件启动
            # 好处: 即使Qdrant故障,机器人仍可运行
            logger.warning(f"Qdrant 不可用，跳过 collection 初始化:{exc}")
            return  # 提前返回,不抛异常

        # 提取所有已存在collection的名称
        # c.name: 每个collection对象的name属性
        # [c.name for c in existing.collections]: 列表推导式
        existing_names = [c.name for c in existing.collections]

        # ==================== 步骤3: 遍历每个collection,创建或检查 ====================

        # items(): 返回(name, size)元组
        for name, size in collections.items():
            # ==================== 情况1: Collection不存在,创建新的 ====================

            if name not in existing_names:  # 如果不在已存在列表中
                logger.info(f"Creating Qdrant collection: {name}")

                try:
                    # await self.client.create_collection(): 创建collection
                    # collection_name: collection的名称
                    # vectors_config: 向量配置
                    await self.client.create_collection(
                        collection_name=name,
                        vectors_config=models.VectorParams(
                            size=size,  # 向量维度(如2048)
                            distance=models.Distance.COSINE,  # 距离度量: 余弦相似度
                            # COSINE: 取值[-1, 1],1表示完全相同,-1表示完全相反
                            # 适合文本向量检索
                        ),
                    )
                except Exception as exc:
                    # 创建失败(权限、配置错误等)
                    # 继续处理下一个collection,不中断
                    logger.warning(f"创建 Qdrant collection 失败({name}),将继续:{exc}")

                # continue: 跳过后续检查,处理下一个collection
                continue

            # ==================== 情况2: Collection已存在,检查维度是否匹配 ====================

            try:
                # await self.client.get_collection(): 获取collection详细信息
                info = await self.client.get_collection(name)

                # _extract_vector_size(): 从info中提取向量维度
                # 返回None或整数
                current_size = self._extract_vector_size(info)

                # 检查维度是否匹配
                # current_size is not None: 成功提取到维度
                # int(current_size) != int(size): 维度不匹配
                if current_size is not None and int(current_size) != int(size):
                    # 维度不匹配!

                    if recreate:  # 如果配置了自动重建
                        logger.warning(
                            f"Qdrant collection 维度不匹配，将重建:{name} expected={size} got={current_size}"
                        )
                        # 删除旧collection
                        await self.client.delete_collection(name)
                        # 创建新collection(维度正确)
                        await self.client.create_collection(
                            collection_name=name,
                            vectors_config=models.VectorParams(
                                size=size,
                                distance=models.Distance.COSINE,
                            ),
                        )
                    else:  # 未配置自动重建
                        # 只记录警告,提示用户手动处理
                        logger.warning(
                            f"Qdrant collection 维度不匹配:{name} expected={size} got={current_size};"
                            "请设置 `vector_size` 为 embedding 实际维度并手动清空/重建 collection,"
                            "或在配置中开启 `qdrant_recreate_collections=true` 自动重建。"
                        )
            except Exception as exc:
                # 检查失败(网络问题等)
                # 继续处理,不中断
                logger.warning(f"检查 Qdrant collection 维度失败({name}),将继续:{exc}")

    @staticmethod
    def _extract_vector_size(info: models.CollectionInfo) -> Optional[int]:
        """从collection信息中提取向量维度

        这个方法的作用:
        - 解析CollectionInfo对象
        - 提取向量配置中的size字段
        - 兼容多种配置格式(单向量/多向量)
        - 返回维度大小或None

        为什么需要这个方法?
        - Qdrant支持单向量和多向量配置
        - 不同版本的qdrant-client返回格式可能不同
        - 需要兼容多种情况

        配置类型:
        1. 单向量: VectorParams对象,直接包含size
        2. 多向量: dict,每个向量有名称和VectorParams
        3. Pydantic v2: 需要用model_dump()导出

        Args:
            info: Qdrant返回的collection信息对象
                - 类型: models.CollectionInfo
                - 包含: config, params, vectors等字段

        Returns:
            Optional[int]:
                - 成功: 返回向量维度(如2048)
                - 失败: 返回None

        Example:
            >>> info = await client.get_collection("rag_items")
            >>> size = QdrantManager._extract_vector_size(info)
            >>> print(size)  # 2048
        """

        # ==================== 步骤1: 获取vectors配置 ====================

        try:
            # 尝试从info中提取vectors配置
            # info.config.params.vectors: 访问嵌套属性
            # 如果任何一个是None,整个表达式返回None
            vectors = info.config.params.vectors if info and info.config and info.config.params else None
        except Exception:
            # 属性访问失败(版本不兼容、格式变化等)
            vectors = None

        if vectors is None:  # 如果无法获取vectors
            return None  # 返回None

        # ==================== 步骤2: 单向量配置(VectorParams对象) ====================

        # isinstance(vectors, models.VectorParams): 检查是否是VectorParams类型
        if isinstance(vectors, models.VectorParams):
            # 单向量配置,直接返回size
            # int(vectors.size): 转为整数
            return int(vectors.size)

        # ==================== 步骤3: 多向量配置(dict) ====================

        if isinstance(vectors, dict):  # 如果是字典
            if not vectors:  # 如果是空字典
                return None

            # next(iter(vectors.values())): 获取字典的第一个value
            # iter(vectors.values()): 创建values迭代器
            # next(...): 获取第一个元素
            first = next(iter(vectors.values()))

            # 情况3.1: value是VectorParams对象
            if isinstance(first, models.VectorParams):
                return int(first.size)

            # 情况3.2: value是字典且包含size字段
            if isinstance(first, dict) and "size" in first:
                return int(first["size"])

            # 其他情况: 无法识别
            return None

        # ==================== 步骤4: Pydantic v2 model(尝试model_dump) ====================

        try:
            # Pydantic v2模型需要用model_dump()导出为字典
            # vectors.model_dump(): 导出模型为字典
            # type: ignore[attr-defined]: 告诉类型检查器忽略属性不存在的警告
            dumped = vectors.model_dump()  # type: ignore[attr-defined]

            if isinstance(dumped, dict) and dumped:  # 如果导出成功且非空
                # 获取第一个value
                first = next(iter(dumped.values()))
                # 如果是字典且包含size
                if isinstance(first, dict) and "size" in first:
                    return int(first["size"])
        except Exception:
            # model_dump失败(不是Pydantic模型、版本不兼容等)
            pass  # 继续,返回None

        # ==================== 步骤5: 无法提取 ====================

        return None  # 无法识别格式,返回None

    async def upsert_text_point(
        self,
        *,
        collection_name: str,
        point_id: str,
        vector: List[float],
        payload: Dict[str, Any],
    ) -> None:
        """向指定collection写入一个向量点(upsert=insert or update)

        这个方法的作用:
        - 将一个向量及其payload存入Qdrant
        - 如果point_id已存在,则更新(update)
        - 如果point_id不存在,则插入(insert)
        - upsert: 合并insert和update的操作

        使用场景:
        - 消息向量化: 将新消息转为向量存入rag_items
        - 记忆向量化: 将用户记忆转为向量存入memories
        - 表情包索引: 将表情包描述转为向量存入stickers

        Args:
            collection_name: collection名称(关键字参数,必须显式指定)
                - 可选值: "rag_items", "memories", "stickers"
            point_id: 向量点的唯一标识(关键字参数)
                - 类型: 字符串
                - 示例: "msg_123", "memory_456", "sticker_789"
                - 要求: 在collection内唯一
            vector: 向量数据(关键字参数)
                - 类型: 浮点数列表
                - 长度: 必须等于collection的vector_size(如2048)
                - 示例: [0.123, -0.456, 0.789, ...]
            payload: 附加数据(关键字参数)
                - 类型: 字典
                - 内容: 任意JSON可序列化的数据
                - 示例: {"content": "今天天气真好", "timestamp": 1234567890}
                - 作用: 检索时返回payload,获取完整信息

        Returns:
            None: 无返回值(成功无声,失败抛异常)

        Raises:
            Exception: Qdrant API调用失败时抛出异常

        Example:
            >>> # 存储消息向量
            >>> await qdrant_manager.upsert_text_point(
            ...     collection_name="rag_items",
            ...     point_id="msg_123",
            ...     vector=[0.1, -0.2, 0.3, ...],  # 2048维
            ...     payload={"content": "今天天气真好", "timestamp": 1234567890}
            ... )

            >>> # 存储记忆向量
            >>> await qdrant_manager.upsert_text_point(
            ...     collection_name="memories",
            ...     point_id="memory_456",
            ...     vector=[0.15, -0.25, ...],
            ...     payload={"content": "用户喜欢吃辣", "type": "preference"}
            ... )
        """

        # ==================== 调用Qdrant API执行upsert ====================

        # await self.client.upsert(): 插入或更新向量点
        # collection_name: 目标collection
        # points: 要插入的点列表(可以一次插入多个,这里只插入1个)
        await self.client.upsert(
            collection_name=collection_name,
            points=[
                # models.PointStruct: Qdrant的点结构
                models.PointStruct(
                    id=point_id,  # 点的唯一标识
                    vector=vector,  # 向量数据
                    payload=payload,  # 附加数据
                )
            ],
        )

    async def search(
        self,
        *,
        collection_name: str,
        vector: List[float],
        limit: int,
        query_filter: Optional[models.Filter] = None,
    ) -> List[models.ScoredPoint]:
        """向量检索(根据查询向量找到最相似的top-k个点)

        这个方法的作用:
        - 在指定collection中搜索最相似的向量
        - 根据余弦相似度(cosine similarity)排序
        - 返回top-k个最相关的结果
        - 每个结果包含: 向量ID、相似度分数、payload

        向量检索原理:
        1. 查询向量 vs collection中所有向量
        2. 计算余弦相似度(cosine similarity)
        3. 按相似度降序排序
        4. 返回前k个最相似的

        相似度分数:
        - 余弦相似度范围: [-1, 1]
        - 1: 完全相同(向量方向一致)
        - 0: 无关(向量正交)
        - -1: 完全相反(向量方向相反)
        - 文本向量通常在[0, 1]之间

        Args:
            collection_name: collection名称(关键字参数)
                - 可选值: "rag_items", "memories", "stickers"
            vector: 查询向量(关键字参数)
                - 类型: 浮点数列表
                - 长度: 必须等于collection的vector_size
                - 示例: [0.123, -0.456, 0.789, ...]
            limit: 返回结果数量(关键字参数)
                - 类型: 整数
                - 示例: 5表示返回top-5最相关结果
            query_filter: 过滤条件(关键字参数,可选)
                - 类型: models.Filter或None
                - 作用: 限制搜索范围(如只搜索某个用户的消息)
                - 默认值: None(不过滤,搜索全部)

        Returns:
            List[models.ScoredPoint]: 检索结果列表,按相似度降序
                - 每个ScoredPoint包含:
                  * id: 向量点ID
                  * score: 相似度分数(0-1,越大越相似)
                  * payload: 附加数据(字典)
                - 示例: [
                    ScoredPoint(id="msg_123", score=0.87, payload={"content": "..."}),
                    ScoredPoint(id="msg_456", score=0.76, payload={"content": "..."}),
                  ]

        版本兼容性:
            - qdrant-client 1.16+: 使用query_points方法
            - 老版本: 使用search方法
            - 本方法自动检测并使用正确的API

        Example:
            >>> # 检索相关消息(RAG)
            >>> query_vector = await embedder.get_embedding("今天天气怎么样")
            >>> results = await qdrant_manager.search(
            ...     collection_name="rag_items",
            ...     vector=query_vector,
            ...     limit=5
            ... )
            >>> for point in results:
            ...     print(f"相似度: {point.score:.2f}, 内容: {point.payload['content']}")
            # 输出: 相似度: 0.87, 内容: 今天天气真好
            # 输出: 相似度: 0.76, 内容: 明天可能下雨
        """

        # ==================== 版本兼容处理 ====================

        # hasattr(self.client, "query_points"): 检查client是否有query_points方法
        # qdrant-client 1.16+: 有query_points方法(新API)
        if hasattr(self.client, "query_points"):
            # ==================== 使用新版API: query_points ====================

            # await self.client.query_points(): 查询向量点(新API)
            # collection_name: 目标collection
            # query: 查询向量
            # limit: 返回数量
            # query_filter: 过滤条件(可选)
            # with_payload=True: 返回结果包含payload
            resp = await self.client.query_points(
                collection_name=collection_name,
                query=vector,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
            )
            # resp.points: 提取points列表
            # list(...): 转为标准Python列表
            return list(resp.points)

        # ==================== 使用旧版API: search ====================

        # await self.client.search(): 搜索向量(旧API)
        # type: ignore[attr-defined]: 告诉类型检查器忽略方法可能不存在的警告
        # query_vector: 查询向量(旧API用这个参数名)
        resp = await self.client.search(  # type: ignore[attr-defined]
            collection_name=collection_name,
            query_vector=vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        # 旧API直接返回列表,不需要.points
        return list(resp)


# ==================== 模块级全局实例 ====================

# qdrant_manager: 全局QdrantManager单例实例
# 在模块导入时立即创建(单例模式保证全局唯一)
# 好处: 避免重复创建,配置集中管理
qdrant_manager = QdrantManager()
