"""文本向量化(Embedding)客户端封装模块 - OpenAI兼容协议的embedding调用

这个模块的作用:
1. 封装文本向量化API调用(将文本转为数字向量)
2. 兼容多种embedding服务提供商(OpenAI、火山方舟等)
3. 处理不同endpoint格式(标准/multimodal)
4. 智能重试多种请求格式,提高成功率
5. 从各种响应结构中提取embedding向量

为什么需要Embedding?
- 向量检索(RAG): 将文本转为向量,用于语义相似度搜索
- 消息索引: 每条消息转为向量存入Qdrant,用于上下文检索
- 记忆检索: 用户查询转为向量,检索最相关的记忆

向量化原理(新手必读):
- 文本 → Embedding模型 → 向量(一串数字,如2048维)
- 语义相似的文本 → 向量距离近
- 不相关的文本 → 向量距离远
- 向量维度: 通常512-4096维,由模型决定

兼容性挑战:
- 不同厂商的endpoint不同: /embeddings vs /embeddings/multimodal
- 请求格式不同: input可能是字符串、数组、对象
- 响应格式不同: embedding可能在data[0].embedding、data.embedding等位置
- 本模块通过智能重试和多格式解析解决这些问题

使用方式:
```python
from .vector.embedder import embedder

# 将文本转为向量
vector = await embedder.get_embedding("今天天气真好")
print(len(vector))  # 2048 (向量维度)
print(vector[:5])  # [0.123, -0.456, 0.789, ...]

# 存入Qdrant用于检索
await qdrant_client.upsert(collection_name="messages",
                           points=[{"id": msg_id, "vector": vector}])
```
"""

from __future__ import annotations

import asyncio  # Python异步编程标准库
import io  # 字节流操作
from pathlib import Path  # 文件路径处理
from typing import Any, List, Optional, Tuple, cast  # 类型提示

import httpx  # HTTP客户端库,支持异步请求
from nonebot import logger  # NoneBot日志记录器

from ..config import plugin_config  # 导入插件配置
from ..llm.vision import VisionHelper  # 导入 data URL 工具


def _split_base_url_and_endpoint(raw: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """从用户配置中智能拆分base_url与endpoint

    这个函数的作用:
    - 兼容两种常见的配置写法
    - 自动识别URL中的endpoint部分
    - 返回拆分后的base_url和endpoint

    为什么需要这个函数?
    - 用户配置习惯不同,有的人会把endpoint写进base_url
    - 例如: "https://host/api/v3/embeddings/multimodal"
    - 需要智能识别并拆分,避免重复拼接

    兼容两种写法:
    1. 标准写法: embedder_base_url="https://host/api/v3" + embedder_endpoint="/embeddings"
    2. 合并写法: embedder_base_url="https://host/api/v3/embeddings/multimodal"

    Args:
        raw: 用户配置的原始URL字符串
            - 可能是None或空字符串
            - 可能包含endpoint部分
            - 例如: "https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal"

    Returns:
        Tuple[Optional[str], Optional[str]]:
            - 第一个元素: base_url部分(如"https://host/api/v3")
            - 第二个元素: endpoint部分(如"/embeddings/multimodal")
            - 如果无法拆分,返回(原URL, None)

    识别优先级:
        - 优先识别 /embeddings/multimodal (更具体)
        - 其次识别 /embeddings (标准)

    Example:
        >>> _split_base_url_and_endpoint("https://host/api/v3/embeddings/multimodal")
        ("https://host/api/v3", "/embeddings/multimodal")

        >>> _split_base_url_and_endpoint("https://api.openai.com/v1")
        ("https://api.openai.com/v1", None)
    """

    # ==================== 步骤1: 处理空值情况 ====================

    if not raw:  # 如果raw是None或False
        return None, None

    # .strip(): 去除首尾空格
    s = raw.strip()
    if not s:  # 如果去除空格后是空字符串
        return None, None

    # ==================== 步骤2: 识别并拆分endpoint ====================

    # 优先识别包含embeddings路径的情况
    # token: endpoint的候选字符串
    # 顺序很重要: 先匹配更具体的,避免误匹配
    for token in ("/embeddings/multimodal", "/embeddings"):
        # s.find(token): 查找token在字符串中的位置
        # 返回索引位置,如果找不到返回-1
        idx = s.find(token)

        if idx != -1:  # 如果找到了
            # s[:idx]: 切片,取从开头到idx之前的部分(不包含idx)
            # .rstrip("/"): 移除末尾的斜杠
            # or None: 如果是空字符串,转为None
            base = s[:idx].rstrip("/") or None

            # s[idx:]: 切片,取从idx开始到末尾的部分(包含idx)
            # 即endpoint部分
            endpoint = s[idx:] or None

            return base, endpoint  # 返回拆分结果

    # ==================== 步骤3: 无法拆分,返回原URL ====================

    # 如果URL中没有识别到endpoint,认为整个是base_url
    # .rstrip("/"): 移除末尾斜杠
    return s.rstrip("/") or None, None


def _build_payload(endpoint: str, model: str, text: str) -> List[dict]:
    """根据endpoint类型构造多种可能的请求体候选列表

    这个函数的作用:
    - 生成多种格式的请求体(payload)候选
    - 按成功概率排序,优先尝试最可能成功的格式
    - 用于重试策略: 第一种失败就尝试第二种

    为什么需要多种格式?
    - 不同厂商的API要求不同
    - OpenAI标准: {"model": "...", "input": "text"}
    - 火山方舟multimodal: {"model": "...", "input": [{"type": "text", "text": "..."}]}
    - 其他变体: input可能是数组、对象、嵌套对象

    Args:
        endpoint: API的endpoint路径
            - 示例: "/embeddings", "/embeddings/multimodal"
        model: embedding模型名称
            - 示例: "text-embedding-3-small", "doubao-embedding"
        text: 要向量化的文本内容

    Returns:
        List[dict]: 请求体候选列表,按优先级排序
            - 第一个元素: 最可能成功的格式
            - 后续元素: 备用格式
            - 将依次尝试,直到成功或全部失败

    请求体格式说明:
        标准格式 (/embeddings):
            - {"model": "xxx", "input": "text"}
            - {"model": "xxx", "input": ["text"]}

        Multimodal格式 (/embeddings/multimodal):
            - {"model": "xxx", "input": [{"type": "text", "text": "..."}]}
            - {"model": "xxx", "input": [{"type": "text", "content": "..."}]}
            - {"model": "xxx", "input": [{"text": "..."}]}
            - {"model": "xxx", "input": {"text": "..."}}

    Example:
        >>> payloads = _build_payload("/embeddings", "model-x", "hello")
        >>> len(payloads)  # 2
        >>> payloads[0]    # {"model": "model-x", "input": "hello"}
    """

    # ==================== 步骤1: 判断endpoint类型 ====================

    # (endpoint or "").lower(): 转小写,便于比较
    endpoint_l = (endpoint or "").lower()

    # ==================== 步骤2: Multimodal endpoint的多种格式 ====================

    if "multimodal" in endpoint_l:  # 如果是multimodal类型
        # 返回多种multimodal格式候选
        # 按成功概率排序:
        # 1. 最标准的格式: input为数组,元素包含type和text字段
        # 2. 变体1: 用content替代text
        # 3. 变体2: 去掉type字段
        # 4. 变体3: input为对象而不是数组
        return [
            {"model": model, "input": [{"type": "text", "text": text}]},
            {"model": model, "input": [{"type": "text", "content": text}]},
            {"model": model, "input": [{"text": text}]},
            {"model": model, "input": {"text": text}},
        ]

    # ==================== 步骤3: 标准endpoint的格式 ====================

    # 标准OpenAI embedding API格式
    return [
        {"model": model, "input": text},  # 格式1: input为字符串
        {"model": model, "input": [text]},  # 格式2: input为字符串数组
    ]


def _build_mm_embedding_payloads(
    endpoint: str, model: str, *, text: str = "", image_url: str
) -> List[dict]:
    """构造多模态 embedding 的请求体候选列表（按兼容性优先级排序）

    这个函数的作用:
    - 生成多种格式的多模态 embedding 请求体
    - 针对不同厂商的 API 差异提供多种格式兜底
    - 支持 text + image 的混合输入

    为什么需要多种格式?
    - 不同 OpenAI 兼容网关对 multimodal input 的结构要求不同
    - 火山方舟: input 为 content 数组 [{"type": "text", ...}, {"type": "image_url", ...}]
    - 其他实现: 可能要求不同的字段顺序或嵌套结构
    - 通过多候选重试提高成功率

    Args:
        endpoint: API endpoint 路径
            - 示例: "/embeddings/multimodal"
            - 用于判断是否支持多模态
        model: embedding 模型名称
            - 示例: "doubao-embedding-vision-250615"
        text: 文本内容（可选）
            - 作为辅助信息提升 embedding 质量
            - 示例: "表情包, tags: 猫咪, intent: 可爱"
        image_url: 图片的 URL 或 data URL
            - 必需参数
            - 示例: "data:image/jpeg;base64,..."

    Returns:
        List[dict]: 请求体候选列表，按优先级排序
            - 第一个: 最标准的格式
            - 后续: 各种变体格式
            - 依次尝试直到成功

    请求体格式说明:
        标准格式（OpenAI multimodal API）:
            {
                "model": "xxx",
                "input": [
                    {"type": "text", "text": "..."},
                    {"type": "image_url", "image_url": {"url": "data:..."}}
                ]
            }

        变体格式:
            - 交换 text 和 image 的顺序
            - input 包裹在额外数组中
            - 只包含 image（text 为空时）

    Example:
        >>> payloads = _build_mm_embedding_payloads(
        ...     "/embeddings/multimodal",
        ...     "doubao-vision",
        ...     text="表情包",
        ...     image_url="data:image/jpeg;base64,..."
        ... )
        >>> len(payloads) >= 3  # 至少3种格式
        True
    """

    # ==================== 步骤1: 检查 endpoint 是否支持多模态 ====================

    endpoint_l = (endpoint or "").lower()

    # 如果 endpoint 不包含 "multimodal"，降级到文本 embedding
    if "multimodal" not in endpoint_l:
        # 非多模态 endpoint，只能用文本
        # 这会让上层自动回退到 text-only embedding
        logger.warning(
            f"endpoint '{endpoint}' 不支持多模态，将降级为文本 embedding"
        )
        return [{"model": model, "input": text or "表情包"}]

    # ==================== 步骤2: 构造多种 content 结构候选 ====================

    # text 部分: 如果没有文本，使用默认值"表情包"
    text_content = {"type": "text", "text": text or "表情包"}

    # image 部分: 标准 OpenAI 格式
    image_content = {"type": "image_url", "image_url": {"url": image_url}}

    # 候选1: text 在前, image 在后（常见顺序）
    content_text_first = [text_content, image_content]

    # 候选2: image 在前, text 在后（某些实现偏好这个顺序）
    content_image_first = [image_content, text_content]

    # 候选3: 只有 image（某些实现不接受 text）
    content_image_only = [image_content]

    # ==================== 步骤3: 构造多种 payload 候选 ====================

    return [
        # 优先级1: 标准格式（text 在前）
        {"model": model, "input": content_text_first},
        # 优先级2: image 在前
        {"model": model, "input": content_image_first},
        # 优先级3: input 包裹在额外数组中（某些实现要求 batch 格式）
        {"model": model, "input": [content_text_first]},
        # 优先级4: image only（如果 API 不支持 text+image）
        {"model": model, "input": content_image_only},
        # 优先级5: image only 的 batch 格式
        {"model": model, "input": [content_image_only]},
    ]


def _image_path_to_data_url(path: str) -> str:
    """将本地图片路径转换为 data URL（GIF 取首帧转 JPEG）

    这个函数的作用:
    - 读取本地图片文件并转为 data URL
    - 针对 GIF 动图特殊处理: 取首帧转 JPEG
    - 压缩图片以减小 base64 体积

    为什么 GIF 需要特殊处理?
    - 大多数 embedding API 不接受 GIF 格式
    - GIF 包含多帧，体积大，base64 编码后可能超过网关限制
    - 对于表情包语义理解，首帧通常已足够
    - JPEG 压缩后体积更小，传输更快

    Args:
        path: 本地图片文件路径
            - 类型: 字符串
            - 示例: "/data/stickers/cat.gif"
            - 必须是存在的文件

    Returns:
        str: data URL 格式的字符串
            - GIF: 取首帧转 JPEG，返回 "data:image/jpeg;base64,..."
            - 其他: 直接转 data URL，返回 "data:image/{type};base64,..."

    Raises:
        FileNotFoundError: 如果文件不存在
        PIL.UnidentifiedImageError: 如果文件不是有效图片
        其他 I/O 错误

    技术细节:
        - GIF 处理流程:
          1. 用 Pillow 打开 GIF
          2. seek(0) 定位到首帧
          3. convert("RGB") 转为 RGB 模式（JPEG 不支持透明通道）
          4. save 为 JPEG，quality=85（平衡质量和体积）
          5. 转为 data URL

        - 非 GIF 处理:
          1. 直接读取文件字节流
          2. 用 VisionHelper 转 data URL

    Example:
        >>> # GIF 表情包
        >>> url = _image_path_to_data_url("/stickers/funny.gif")
        >>> url.startswith("data:image/jpeg;base64,")
        True

        >>> # PNG 表情包
        >>> url = _image_path_to_data_url("/stickers/cat.png")
        >>> url.startswith("data:image/png;base64,")
        True
    """

    # ==================== 步骤1: 解析文件路径和后缀 ====================

    p = Path(path)
    suffix = (p.suffix or "").lower()  # 获取文件后缀，如 ".gif", ".png"

    # ==================== 步骤2: GIF 特殊处理（取首帧转 JPEG） ====================

    if suffix == ".gif":
        # 使用 Pillow 处理 GIF
        # 注意: Pillow 已经是项目依赖（stickers/registry.py 中使用）
        try:
            from PIL import Image  # 导入 Pillow 库
        except ImportError:
            # 如果没有 Pillow，降级为直接读取（可能失败）
            logger.warning("Pillow 未安装，无法处理 GIF，尝试直接读取")
            return VisionHelper.to_data_url(p.read_bytes(), suffix)

        # 打开 GIF 文件
        with Image.open(p) as im:
            try:
                # seek(0): 定位到第一帧
                # GIF 是多帧动画，seek(0) 确保读取首帧
                im.seek(0)
            except Exception:
                # 某些损坏的 GIF 可能 seek 失败，忽略错误
                pass

            # convert("RGB"): 转换颜色模式
            # - GIF 可能是 P 模式（调色板）或 RGBA 模式（带透明度）
            # - JPEG 只支持 RGB 模式
            # - 转换会丢失透明度，但对语义理解影响不大
            frame = im.convert("RGB")

            # 创建内存字节流，用于保存 JPEG
            buf = io.BytesIO()

            # 保存为 JPEG 格式
            frame.save(
                buf,
                format="JPEG",  # 输出格式
                quality=85,  # 压缩质量 (1-100)，85 是质量和体积的良好平衡
                optimize=True,  # 启用优化，进一步减小体积
            )

            # 获取字节流内容并转为 data URL
            return VisionHelper.to_data_url(buf.getvalue(), ".jpg")

    # ==================== 步骤3: 非 GIF 图片（直接读取） ====================

    # 直接读取文件字节流并转为 data URL
    return VisionHelper.to_data_url(p.read_bytes(), suffix or ".png")


def _coerce_embedding(value: Any) -> Optional[List[float]]:
    """将多种可能的embedding表达形式规范化为List[float]

    这个函数的作用:
    - 接受各种格式的embedding数据
    - 尝试提取出浮点数列表
    - 返回标准化的向量格式

    为什么需要规范化?
    - 不同厂商返回的embedding格式不同
    - 可能是直接的数组: [0.1, 0.2, ...]
    - 可能嵌套在对象中: {"embedding": [0.1, 0.2, ...]}
    - 可能用不同的key: "vector", "data", "values"

    Args:
        value: 任意类型的值,可能包含embedding向量
            - 可以是列表: [0.1, 0.2, 0.3, ...]
            - 可以是字典: {"embedding": [...]}
            - 可以是None

    Returns:
        Optional[List[float]]:
            - 成功: 返回浮点数列表 [0.1, 0.2, ...]
            - 失败: 返回None

    识别策略:
        1. 如果是列表且全是数字 → 直接返回
        2. 如果是字典 → 尝试从常见key中提取
            - 尝试key: "embedding", "vector", "data", "values"

    Example:
        >>> _coerce_embedding([0.1, 0.2, 0.3])
        [0.1, 0.2, 0.3]

        >>> _coerce_embedding({"embedding": [0.1, 0.2]})
        [0.1, 0.2]

        >>> _coerce_embedding({"vector": [0.1, 0.2]})
        [0.1, 0.2]

        >>> _coerce_embedding(None)
        None
    """

    # ==================== 情况1: value为None ====================
    if value is None:
        return None

    # ==================== 情况2: value是数字列表 ====================

    # isinstance(value, list): 检查是否是列表
    # value: 列表非空
    # all(isinstance(x, (int, float)) for x in value): 检查所有元素都是数字
    if isinstance(value, list) and value and all(isinstance(x, (int, float)) for x in value):
        # cast(List[float], value): 类型转换,告诉类型检查器这是List[float]
        # cast不会改变运行时行为,只是类型提示
        return cast(List[float], value)

    # ==================== 情况3: value是字典,尝试从常见key提取 ====================

    if isinstance(value, dict):
        # 尝试常见的embedding字段名
        # key: 候选的字段名
        for key in ("embedding", "vector", "data", "values"):
            # value.get(key): 获取字典中key对应的值
            inner = value.get(key)

            # 检查inner是否是数字列表
            if isinstance(inner, list) and inner and all(isinstance(x, (int, float)) for x in inner):
                return cast(List[float], inner)  # 找到了,返回

    # ==================== 情况4: 无法识别 ====================
    return None  # 无法提取,返回None


def _extract_embedding_from_response(data: Any) -> List[float]:
    """从不同提供商的响应结构中提取embedding向量

    这个函数的作用:
    - 处理各种API响应格式
    - 递归搜索embedding向量
    - 返回标准化的浮点数列表

    为什么需要这个函数?
    - 不同厂商的响应结构差异巨大
    - OpenAI: {"data": [{"embedding": [...]}]}
    - 火山方舟: {"data": {"embedding": [...]}}
    - 其他: {"data": {"0": {"embedding": [...]}}}
    - 需要智能识别所有可能的格式

    Args:
        data: API响应的JSON数据(已解析为Python对象)
            - 通常是字典类型
            - 可能有各种嵌套结构

    Returns:
        List[float]: 提取出的embedding向量
            - 成功: 返回浮点数列表
            - 失败: 抛出异常

    Raises:
        TypeError: 如果响应不是字典类型
        RuntimeError: 如果无法从响应中提取embedding

    提取策略(按优先级):
        1. OpenAI标准格式: data[0].embedding
        2. 嵌套格式: data.embedding
        3. 特殊格式: data["0"].embedding (某些实现用对象表示列表)
        4. 顶层直接给: .embedding / .vector / .result
        5. 递归嵌套: data.data.embedding

    Example:
        >>> # OpenAI标准格式
        >>> _extract_embedding_from_response({
        ...     "data": [{"embedding": [0.1, 0.2]}]
        ... })
        [0.1, 0.2]

        >>> # 嵌套格式
        >>> _extract_embedding_from_response({
        ...     "data": {"embedding": [0.1, 0.2]}
        ... })
        [0.1, 0.2]
    """

    # ==================== 步骤1: 类型检查 ====================

    if not isinstance(data, dict):
        # 响应必须是字典类型,否则无法处理
        # type(data).__name__: 获取类型名称(如"list", "str")
        raise TypeError(f"响应不是对象:{type(data).__name__}")

    # ==================== 步骤2: 尝试OpenAI标准格式 ====================

    # data.get("data"): 获取data字段的值
    # OpenAI标准: {"data": [{"embedding": [...]}]}
    d = data.get("data")

    if isinstance(d, list) and d:  # 如果data是非空列表
        # 取列表的第一个元素
        first = d[0]

        if isinstance(first, dict):  # 如果第一个元素是字典
            # 尝试从多个可能的字段中提取embedding
            # first.get("embedding"): 尝试embedding字段
            # first.get("vector"): 尝试vector字段
            # first: 如果都没有,尝试整个对象
            emb = _coerce_embedding(first.get("embedding") or first.get("vector") or first)

            if emb is not None:  # 如果成功提取
                return emb  # 返回向量

    # ==================== 步骤3: 尝试字典格式 ====================

    if isinstance(d, dict):  # 如果data是字典
        # 某些实现会返回 {"data": {"embedding": [...]}}

        # 特殊情况: 用对象表示列表 {"data": {"0": {...}}}
        # 兼容把列表用对象表达的情况
        if "0" in d and isinstance(d.get("0"), dict):
            first = d.get("0")  # 获取"0"索引的值
            emb0 = _coerce_embedding(first)
            if emb0 is not None:
                return emb0

        # 尝试从常见字段提取
        emb = _coerce_embedding(d.get("embedding") or d.get("vector") or d)
        if emb is not None:
            return emb

        # 递归处理嵌套的data字段
        # 某些实现: {"data": {"data": {"embedding": [...]}}}
        nested = d.get("data")
        if nested is not None:
            # 递归调用自己,处理嵌套结构
            return _extract_embedding_from_response({"data": nested})

    # ==================== 步骤4: 尝试顶层字段 ====================

    # 某些实现直接在顶层给出embedding
    # data.get("embedding"): 尝试embedding字段
    # data.get("vector"): 尝试vector字段
    # data.get("result"): 尝试result字段
    emb = _coerce_embedding(data.get("embedding") or data.get("vector") or data.get("result"))
    if emb is not None:
        return emb

    # ==================== 步骤5: 提取失败,抛出异常 ====================

    # list(data.keys()): 获取字典的所有key
    keys = list(data.keys())

    # type(d).__name__: 获取data字段的类型名称
    data_type = type(d).__name__

    # 抛出详细的错误信息,帮助调试
    raise RuntimeError(f"无法从响应中提取 embedding:keys={keys} data_type={data_type}")


class Embedder:
    """OpenAI兼容embedding API的异步封装客户端(支持可配置endpoint)

    这个类的作用:
    - 统一封装各种embedding服务的调用
    - 智能处理不同厂商的API差异
    - 提供重试机制,提高成功率
    - 从各种响应格式中提取向量

    设计特点:
    - 多格式重试: 自动尝试多种请求格式
    - 智能解析: 从各种响应结构中提取向量
    - 详细日志: 失败时提供可操作的错误提示
    - 异常安全: 失败时抛出异常,由上层决定降级策略

    支持的服务商:
    - OpenAI官方
    - 火山方舟(Ark)
    - 其他OpenAI兼容服务

    配置项:
    - embedder_base_url: API基础URL
    - embedder_endpoint: API endpoint路径
    - embedder_api_key: API密钥
    - embedder_model: 模型名称
    - embedder_timeout: 请求超时时间

    属性:
    - _base_url: API基础URL
    - _endpoint: API endpoint路径
    - _api_key: API密钥
    - _timeout: 请求超时时间(秒)
    - model: 模型名称
    """

    def __init__(self) -> None:
        """初始化Embedding客户端

        初始化流程:
        1. 拆分并解析base_url和endpoint
        2. 处理api_key的回退逻辑(配置 → openai_api_key → 占位符)
        3. 保存配置参数供后续调用使用

        Side Effects:
            - 读取plugin_config的多个配置项
            - 如果api_key未配置,输出警告日志
        """

        # ==================== 步骤1: 解析和标准化URL配置 ====================

        # _split_base_url_and_endpoint(): 智能拆分URL
        # plugin_config.yuying_embedder_base_url: 从配置读取base_url
        base_url, endpoint_from_url = _split_base_url_and_endpoint(
            plugin_config.yuying_embedder_base_url
        )

        # 确定最终的endpoint
        # 优先级: URL中的endpoint > 配置的endpoint > 默认"/embeddings"
        # (endpoint_from_url or plugin_config.yuying_embedder_endpoint or "").strip():
        #   1. 先用URL中解析出的endpoint
        #   2. 如果没有,用配置的endpoint
        #   3. 如果都没有,用空字符串
        #   .strip(): 去除空格
        # or "/embeddings": 如果为空,使用默认值
        endpoint = (endpoint_from_url or plugin_config.yuying_embedder_endpoint or "").strip() or "/embeddings"

        # 确保endpoint以斜杠开头
        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint

        # ==================== 步骤2: 处理API密钥的回退逻辑 ====================

        # 尝试使用embedder_api_key
        api_key = (plugin_config.yuying_embedder_api_key or "").strip() or None

        if not api_key:  # 如果embedder_api_key未配置
            # 回退: 尝试使用openai_api_key
            api_key = (plugin_config.yuying_openai_api_key or "").strip() or None

        if not api_key:  # 如果都没有
            # 最终回退: 使用占位符,避免启动失败
            api_key = "DUMMY"
            logger.warning("未配置 embedder_api_key/openai_api_key，将使用占位 key 以保证进程可启动。")

        # ==================== 步骤3: 保存配置参数 ====================

        self._base_url = base_url  # API基础URL
        self._endpoint = endpoint  # API endpoint路径
        self._api_key = api_key  # API密钥
        # getattr(plugin_config, "yuying_embedder_timeout", 30.0): 获取超时配置,默认30秒
        # float(...): 确保是浮点数
        self._timeout = float(getattr(plugin_config, "yuying_embedder_timeout", 30.0) or 30.0)
        self.model = plugin_config.yuying_embedder_model  # 模型名称

    async def get_embedding(self, text: str) -> List[float]:
        """将文本转换为embedding向量

        这个方法的作用:
        - 调用embedding API将文本向量化
        - 自动重试多种请求格式
        - 从响应中提取向量
        - 失败时抛出异常(由上层决定降级策略)

        Args:
            text: 要向量化的文本
                - 类型: 字符串
                - 示例: "今天天气真好"
                - 长度限制: 取决于具体模型(通常8192 tokens以内)

        Returns:
            List[float]: embedding向量(浮点数列表)
                - 维度: 取决于模型(如text-embedding-3-small是1536维)
                - 示例: [0.123, -0.456, 0.789, ...]

        Raises:
            RuntimeError: 如果base_url未配置
            httpx.HTTPStatusError: 如果API调用失败(认证、网络等)
            RuntimeError: 如果响应解析失败
            asyncio.CancelledError: 如果任务被取消(不捕获,直接抛出)

        重试策略:
            - 自动尝试多种payload格式
            - 第一种失败就尝试第二种
            - 全部失败才抛出异常

        错误提示:
            - 模型不支持API: 提示更换模型或endpoint
            - JSON解析失败: 提示检查input结构

        Example:
            >>> vector = await embedder.get_embedding("你好,世界")
            >>> print(len(vector))  # 1536 (如果是text-embedding-3-small)
            >>> print(vector[:3])   # [0.123, -0.456, 0.789]
        """

        # ==================== 步骤1: 检查必需配置 ====================

        if not self._base_url:
            # base_url是必需的,没有就无法发送请求
            raise RuntimeError("未配置 embedder_base_url")

        # ==================== 步骤2: 构建请求参数 ====================

        # 拼接完整的API URL
        # 例如: "https://api.openai.com/v1" + "/embeddings"
        url = f"{self._base_url}{self._endpoint}"

        # 构建HTTP请求头
        headers = {
            "Authorization": f"Bearer {self._api_key}",  # Bearer token认证
            "Content-Type": "application/json",  # JSON格式请求体
        }
        # 可选: 复用 OpenAI SDK 的默认headers配置(例如 User-Agent)
        # 说明:
        # - embedding 模块未使用 openai-python SDK,而是直接用 httpx 发请求
        # - 为了让"User-Agent 等自定义header"对 embedding 请求也生效,这里做一次合并
        # - 保护关键header: 不允许覆盖 Authorization / Content-Type
        extra_headers = getattr(plugin_config, "yuying_openai_default_headers", None)
        if isinstance(extra_headers, dict) and extra_headers:
            for k, v in extra_headers.items():
                ks = str(k).strip()
                if not ks:
                    continue
                if ks.lower() in {"authorization", "content-type"}:
                    continue
                vs = str(v).strip()
                if vs:
                    headers[ks] = vs

        # _build_payload(): 生成多种payload候选
        # 按成功概率排序,依次尝试
        payload_candidates = _build_payload(self._endpoint, self.model, text)

        # last_body: 保存最后一次失败的响应体,用于错误提示
        last_body: object = ""

        # ==================== 步骤3: 发送HTTP请求(带重试) ====================

        try:
            # httpx.AsyncClient: 创建异步HTTP客户端
            # timeout=self._timeout: 设置请求超时时间
            # async with: 自动管理客户端生命周期
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                last_error: Optional[httpx.HTTPStatusError] = None
                data: Optional[dict] = None

                # 遍历所有payload候选,依次尝试
                for payload in payload_candidates:
                    try:
                        # await client.post(): 发送POST请求
                        # url: API URL
                        # headers: 请求头(包含认证)
                        # json=payload: 自动将payload转为JSON并设置Content-Type
                        resp = await client.post(url, headers=headers, json=payload)

                        # resp.raise_for_status(): 检查HTTP状态码
                        # 如果是4xx或5xx,抛出HTTPStatusError异常
                        resp.raise_for_status()

                        # resp.json(): 解析响应的JSON数据
                        data = resp.json()

                        # 成功了,跳出循环
                        break

                    except httpx.HTTPStatusError as e:
                        # HTTP状态错误(401、404、500等)
                        last_error = e  # 保存错误,如果全部失败就抛出

                        try:
                            # 尝试解析错误响应的JSON
                            last_body = e.response.json()
                        except Exception:
                            # 如果JSON解析失败,直接用文本
                            last_body = e.response.text

                        # 继续尝试下一个payload
                        continue

                # 如果所有payload都失败了
                if data is None and last_error is not None:
                    raise last_error  # 抛出最后一个错误

                if data is None:
                    # 不应该到这里,但以防万一
                    raise RuntimeError("Embedding 请求失败:未获得有效响应")

        # ==================== 步骤4: 异常处理 ====================

        except asyncio.CancelledError:
            # 任务被取消,直接抛出,不捕获
            raise

        except httpx.HTTPStatusError as e:
            # HTTP状态错误,提供详细的错误信息和解决建议

            body = last_body
            if not body:  # 如果之前没保存响应体,现在获取
                try:
                    body = e.response.json()
                except Exception:
                    body = e.response.text

            # 构建基本错误消息
            msg = f"Embedding 失败:{e} - {body}"

            # 如果响应体是字典,尝试提取错误信息并给出建议
            if isinstance(body, dict):
                # body.get("error"): 获取error字段
                # isinstance(..., dict): 检查error是否是字典
                err = body.get("error") if isinstance(body.get("error"), dict) else None

                if err and isinstance(err.get("message"), str):
                    error_message = err["message"]

                    # 错误类型1: 模型不支持该API
                    if "does not support this api" in error_message:
                        msg += "(提示:该模型不支持当前 embedding 接口;请更换 `embedder_model`,或设置 `embedder_endpoint` 为该模型支持的接口,例如 `/embeddings/multimodal`。)"

                    # 错误类型2: JSON解析失败
                    if "could not parse the json body" in error_message.lower():
                        msg += '''(提示:当前 endpoint 可能要求结构化 input;已自动尝试多种 payload 仍失败。建议改用"文本 embedding 模型 + /embeddings",或按厂商文档调整 input 结构。)'''

            # logger.error(): 输出错误日志
            logger.error(msg)
            # raise: 重新抛出异常,让上层处理
            raise

        except Exception as e:
            # 其他异常(网络错误、超时等)
            logger.error(f"Embedding 失败:{e}")
            raise

        # ==================== 步骤5: 从响应中提取embedding向量 ====================

        try:
            # _extract_embedding_from_response(): 智能提取向量
            # data: API响应的JSON数据
            return _extract_embedding_from_response(data)

        except Exception as e:
            # 提取失败,记录错误日志

            # 只打印结构信息,避免日志塞入大量向量内容
            shape = None
            try:
                # 尝试获取响应的结构信息,用于调试
                if isinstance(data, dict):
                    d = data.get("data")
                    if isinstance(d, list):
                        # 如果data是列表,显示长度
                        shape = f"data[list,len={len(d)}]"
                    elif isinstance(d, dict):
                        # 如果data是字典,显示所有key
                        shape = f"data[dict,keys={list(d.keys())}]"
                    else:
                        # 其他类型,显示类型名
                        shape = f"data[{type(d).__name__}]"
            except Exception:
                shape = None

            # 输出错误日志,包含结构信息
            logger.error(f"Embedding 响应解析失败:{e}(shape={shape})")

            # 抛出RuntimeError,包含原始异常信息
            # from e: 保留异常链,方便追踪
            raise RuntimeError(f"Embedding 响应解析失败:{e}") from e

    async def get_embedding_multimodal(
        self,
        *,
        text: str = "",
        image_path: Optional[str] = None,
        image_url: Optional[str] = None,
    ) -> List[float]:
        """将（图片 + 文本）转换为多模态 embedding 向量

        这个方法的作用:
        - 调用多模态 embedding API 将图片（+文本）向量化
        - 用于表情包的语义向量化存储
        - 支持本地文件路径和在线 URL 两种输入方式
        - 自动处理 GIF（取首帧转 JPEG）
        - 失败时抛出异常（由上层决定降级策略）

        技术特性:
        - 复用 get_embedding 的核心逻辑（鉴权、重试、解析）
        - 自动构造多种 payload 格式并重试
        - GIF 自动转首帧 JPEG（减小体积）
        - endpoint 不支持多模态时自动降级为文本 embedding

        Args:
            text: 文本内容（可选）
                - 作为辅助信息提升 embedding 质量
                - 示例: "表情包, tags: 猫咪, intents: 可爱"
                - 如果为空，使用默认值 "表情包"
            image_path: 本地图片文件路径（可选）
                - 类型: 字符串或 None
                - 示例: "/data/stickers/funny.gif"
                - 与 image_url 二选一，image_url 优先
            image_url: 在线图片 URL 或 data URL（可选）
                - 类型: 字符串或 None
                - 示例: "https://example.com/cat.jpg"
                - 或 data URL: "data:image/jpeg;base64,..."
                - 优先级高于 image_path

        Returns:
            List[float]: embedding 向量（浮点数列表）
                - 维度: 取决于模型（doubao-embedding-vision 是 2048 维）
                - 示例: [0.123, -0.456, 0.789, ...]

        Raises:
            RuntimeError: 如果 base_url 未配置
            httpx.HTTPStatusError: 如果 API 调用失败
            RuntimeError: 如果响应解析失败
            FileNotFoundError: 如果 image_path 指定的文件不存在
            asyncio.CancelledError: 如果任务被取消

        降级策略:
            - 如果 endpoint 不支持多模态 → 自动降级为 text-only embedding
            - 如果没有提供 image → 自动降级为 text-only embedding
            - 由上层决定是否捕获异常并进一步降级

        使用场景:
            - 表情包入库时的向量化
            - 需要理解图片语义的场景
            - 图文混合内容的向量化

        Example:
            >>> # 使用本地文件
            >>> vec = await embedder.get_embedding_multimodal(
            ...     text="猫咪表情包",
            ...     image_path="/stickers/cat.gif"
            ... )
            >>> len(vec)  # 2048

            >>> # 使用 data URL
            >>> vec = await embedder.get_embedding_multimodal(
            ...     image_url="data:image/jpeg;base64,..."
            ... )
        """

        # ==================== 步骤1: 检查必需配置 ====================

        if not self._base_url:
            raise RuntimeError("未配置 embedder_base_url")

        # ==================== 步骤2: 准备图片 URL ====================

        # 优先使用 image_url，否则从 image_path 读取并转换
        final_image_url = (image_url or "").strip() or None

        if not final_image_url and image_path:
            # 读取本地文件并转为 data URL
            # GIF 会自动取首帧转 JPEG
            try:
                final_image_url = _image_path_to_data_url(image_path)
            except Exception as e:
                logger.error(f"图片转 data URL 失败: {image_path}, 错误: {e}")
                raise

        # 如果没有图片，降级为文本 embedding
        if not final_image_url:
            logger.debug("未提供图片，降级为文本 embedding")
            return await self.get_embedding(text or "表情包")

        # ==================== 步骤3: 构建请求参数 ====================

        # 拼接完整的 API URL
        url = f"{self._base_url}{self._endpoint}"

        # 构建 HTTP 请求头
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        # 可选: 复用 OpenAI SDK 的默认 headers 配置（例如 User-Agent）
        extra_headers = getattr(plugin_config, "yuying_openai_default_headers", None)
        if isinstance(extra_headers, dict) and extra_headers:
            for k, v in extra_headers.items():
                ks = str(k).strip()
                if not ks or ks.lower() in {"authorization", "content-type"}:
                    continue
                vs = str(v).strip()
                if vs:
                    headers[ks] = vs

        # 构建多种 payload 候选
        payload_candidates = _build_mm_embedding_payloads(
            self._endpoint, self.model, text=text, image_url=final_image_url
        )

        # 保存最后一次失败的响应体，用于错误提示
        last_body: object = ""

        # ==================== 步骤4: 发送 HTTP 请求（带重试） ====================

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                last_error: Optional[httpx.HTTPStatusError] = None
                data: Optional[dict] = None

                # 遍历所有 payload 候选，依次尝试
                for payload in payload_candidates:
                    try:
                        resp = await client.post(url, headers=headers, json=payload)
                        resp.raise_for_status()  # 检查 HTTP 状态码
                        data = resp.json()
                        break  # 成功了，跳出循环

                    except httpx.HTTPStatusError as e:
                        last_error = e
                        try:
                            last_body = e.response.json()
                        except Exception:
                            last_body = e.response.text
                        continue  # 尝试下一个 payload

                # 如果所有 payload 都失败了
                if data is None and last_error is not None:
                    raise last_error

                if data is None:
                    raise RuntimeError("Multimodal Embedding 请求失败: 未获得有效响应")

        # ==================== 步骤5: 异常处理 ====================

        except asyncio.CancelledError:
            # 任务被取消，直接抛出，不捕获
            raise

        except httpx.HTTPStatusError as e:
            # HTTP 状态错误，提供详细的错误信息
            body = last_body
            if not body:
                try:
                    body = e.response.json()
                except Exception:
                    body = e.response.text

            msg = f"Multimodal Embedding 失败: {e} - {body}"
            logger.error(msg)
            raise

        except Exception as e:
            # 其他异常（网络错误、超时等）
            logger.error(f"Multimodal Embedding 失败: {e}")
            raise

        # ==================== 步骤6: 从响应中提取 embedding 向量 ====================

        try:
            return _extract_embedding_from_response(data)

        except Exception as e:
            # 提取失败，记录错误日志
            shape = None
            try:
                if isinstance(data, dict):
                    d = data.get("data")
                    if isinstance(d, list):
                        shape = f"data[list,len={len(d)}]"
                    elif isinstance(d, dict):
                        shape = f"data[dict,keys={list(d.keys())}]"
                    else:
                        shape = f"data[{type(d).__name__}]"
            except Exception:
                shape = None

            logger.error(f"Multimodal Embedding 响应解析失败: {e} (shape={shape})")
            raise RuntimeError(f"Multimodal Embedding 响应解析失败: {e}") from e


# ==================== 模块级全局实例 ====================

# embedder: 全局Embedder实例
# 在模块导入时立即创建,供全项目使用
# 好处: 避免重复创建,配置集中管理
embedder = Embedder()
