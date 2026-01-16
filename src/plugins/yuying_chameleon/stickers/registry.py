"""表情包注册模块 - 扫描目录并将本地图片文件注册到数据库

这个模块的作用:
1. 扫描指定目录下的所有图片文件(.png、.jpg、.jpeg、.gif)
2. 计算文件哈希(SHA256)和感知哈希(pHash)
3. 使用OCR识别图片中的文字
4. 生成去重指纹(fingerprint = pHash + OCR文本)
5. 将表情包元数据存入Stickers表

表情包注册原理(新手必读):
- 问题: 本地有大量表情包图片,需要管理和检索
- 解决: 将图片元数据(路径、哈希、文字内容)存入数据库
- 流程: 扫描目录 → 读取文件 → 计算哈希 → OCR识别 → 入库
- 好处: 支持快速检索、去重、分类、意图标注

两种哈希算法:
1. SHA256(内容哈希):
   - 用途: 精确去重,检测完全相同的文件
   - 特点: 即使1个字节不同,哈希值完全不同
   - 示例: 同一图片不同格式(PNG vs JPG)→不同SHA256
   - 作用: sticker_id主键,防止重复注册同一文件

2. pHash(感知哈希):
   - 用途: 模糊去重,检测相似的图片
   - 特点: 图片压缩、缩放、轻微修改后哈希值相近
   - 示例: 同一表情包不同尺寸(500x500 vs 200x200)→相近pHash
   - 作用: 与OCR组合生成fingerprint,识别同一表情包的不同版本

Fingerprint去重机制:
- 组成: pHash + 归一化OCR文本
- 格式: "a1b2c3d4e5f6g7h8+hello world"
- 用途: 识别"实质相同"的表情包(相似图+相同文字)
- 例子:
  * 图A(高清): pHash=a1b2c3d4, OCR="今天天气真好"
  * 图B(压缩): pHash=a1b2c3d5(相近), OCR="今天天气真好"
  * Fingerprint相近→判定为重复,只保留一个

Pack分类说明:
- default: 默认表情包(手动收集,质量高)
- auto: 自动学习的表情包(从群聊中窃取)
- 用途: 区分来源,便于管理和选择策略

Intent意图标注:
- neutral: 中性表情(默认)
- happy/sad/angry/...: 情感分类(可由LLM打标)
- 用途: 根据对话上下文选择合适情感的表情包

使用方式:
```python
from .stickers.registry import StickerRegistry

# 1. 扫描整个目录(递归)
await StickerRegistry.scan_local_stickers("/home/user/stickers")
# 自动扫描所有子目录(除了tmp、__pycache__)
# default_pack目录 → pack="default"
# auto目录 → pack="auto"
# 其他目录 → pack="default"

# 2. 注册单个文件
await StickerRegistry.register_sticker(
    "/home/user/stickers/cat.jpg",
    pack="default"
)
```

目录结构建议:
```
/home/user/stickers/
├── default_pack/        # pack=default
│   ├── funny/
│   │   ├── cat1.jpg
│   │   └── cat2.png
│   └── cute/
│       └── dog.gif
├── auto/               # pack=auto (自动学习)
│   ├── learned1.jpg
│   └── learned2.png
└── tmp/               # 忽略
    └── temp.jpg
```
"""

from __future__ import annotations

import os  # Python标准库,文件系统操作
import hashlib  # Python标准库,哈希算法(SHA256)
import imagehash  # 第三方库,感知哈希算法
from PIL import Image  # Python图像处理库,读取图片

# 导入项目模块
from ..storage.repositories.sticker_repo import StickerRepository  # 表情包仓库
from ..storage.repositories.index_jobs_repo import IndexJobRepository  # 索引任务仓库
from ..storage.models import Sticker, IndexJob  # 表情包模型、索引任务模型
from nonebot import logger  # NoneBot日志
from ..llm.vision import VisionHelper  # 视觉模型客户端(OCR)
from .utils import normalize_ocr_text  # OCR文本归一化
from ..storage.db_writer import db_writer  # 数据库写入队列
from ..storage.write_jobs import AsyncCallableJob  # 异步任务
import json  # JSON序列化


class StickerRegistry:
    """表情包注册器 - 扫描目录并批量注册表情包到数据库

    这个类的作用:
    - 扫描指定目录下的所有图片文件
    - 计算文件哈希和感知哈希
    - 使用OCR识别图片文字
    - 生成去重指纹
    - 批量入库到Stickers表

    设计模式:
    - 静态方法类: 所有方法都是静态的,无需实例化
    - 工具类: 提供目录扫描和注册功能
    - 好处: 简单直接,作为启动时初始化工具使用

    核心流程:
    1. scan_local_stickers()扫描目录
    2. 遍历所有图片文件
    3. 对每个文件调用register_sticker()
    4. 计算哈希、OCR、生成fingerprint
    5. 检查是否已存在(SHA256去重)
    6. 写入数据库

    去重策略:
    - 一级去重: SHA256(文件内容完全相同)
    - 二级去重: fingerprint(pHash+OCR,实质相同)
    - 本模块只做一级去重,二级去重由stealer模块处理

    Example:
        >>> # 启动时注册所有表情包
        >>> await StickerRegistry.scan_local_stickers("/home/user/stickers")
        # INFO: Scanning stickers in /home/user/stickers...
        # INFO: Registered sticker a1b2c3d4e5f6...
        # INFO: Registered sticker 7h8i9j0k1l2m...
    """

    @staticmethod
    async def scan_local_stickers(base_path: str) -> None:
        """扫描目录并注册符合后缀的图片文件(递归扫描)

        这个方法的作用:
        - 递归遍历base_path下的所有子目录
        - 识别图片文件(.png、.jpg、.jpeg、.gif)
        - 根据目录名确定pack分类
        - 调用register_sticker()逐个注册

        目录名与pack的映射:
        - default_pack、default → pack="default"
        - auto → pack="auto"
        - 其他 → pack="default"(默认)

        特殊目录(跳过):
        - tmp: 临时文件目录
        - __pycache__: Python缓存目录

        Args:
            base_path: 扫描的根目录
                - 类型: 字符串
                - 必须存在且可访问
                - 支持相对路径和绝对路径
                - 示例: "/home/user/stickers"

        Returns:
            None: 无返回值

        Side Effects:
            - 递归遍历目录
            - 调用register_sticker()注册文件
            - 输出日志: "Scanning stickers in ..."
            - 写入Stickers表

        性能特征:
            - I/O密集: 读取大量文件
            - CPU密集: 计算哈希、OCR识别
            - 适合启动时批量处理,不适合频繁调用

        异常处理:
            - 目录不存在: os.walk()会抛出OSError
            - 文件读取失败: register_sticker()内部捕获并记录
            - 不会中断整体流程,失败的文件跳过

        Example:
            >>> # 扫描整个目录
            >>> await StickerRegistry.scan_local_stickers("/home/user/stickers")

            >>> # 目录结构:
            >>> # /home/user/stickers/
            >>> #   ├── default_pack/  → pack="default"
            >>> #   │   ├── cat.jpg
            >>> #   │   └── dog.png
            >>> #   ├── auto/         → pack="auto"
            >>> #   │   └── learned.gif
            >>> #   └── tmp/          → 跳过
            >>> #       └── temp.jpg

            >>> # 结果: 注册3个表情包(cat.jpg, dog.png, learned.gif)
        """

        # ==================== 步骤1: 输出扫描开始日志 ====================

        # logger.info(): 记录信息级别日志
        logger.info(f"Scanning stickers in {base_path}...")

        # 统计信息
        total_scanned = 0  # 扫描到的图片文件总数
        total_registered = 0  # 新注册的表情包数量
        total_skipped = 0  # 跳过的表情包数量（已存在）
        total_errors = 0  # 注册失败的数量

        # ==================== 步骤2: 递归遍历目录 ====================

        # os.walk(base_path): 递归遍历目录树
        # - 返回: 生成器,每次yield (root, dirs, files)
        #   * root: 当前目录的路径
        #   * dirs: 当前目录下的子目录列表(可修改来控制遍历)
        #   * files: 当前目录下的文件列表
        # - 递归: 自动进入所有子目录
        for root, _, files in os.walk(base_path):
            # ==================== 步骤3: 判断是否跳过当前目录 ====================

            # os.path.basename(root): 获取目录名(不包含父路径)
            # 例如: "/home/user/stickers/tmp" → "tmp"
            base = os.path.basename(root)

            # base in {"tmp", "__pycache__"}: 检查是否为特殊目录
            if base in {"tmp", "__pycache__"}:
                continue  # 跳过,不处理这些目录中的文件

            # ==================== 步骤4: 根据目录名确定pack分类 ====================

            # pack: 表情包分类标识
            # 默认值: "default"
            pack = "default"

            # 情况1: default相关目录
            if base in {"default_pack", "default"}:
                pack = "default"

            # 情况2: auto目录(自动学习)
            elif base in {"auto"}:
                pack = "auto"

            # ==================== 步骤5: 遍历当前目录下的所有文件 ====================

            for file in files:
                # ==================== 步骤5.1: 检查文件扩展名 ====================

                # file.lower(): 转小写,统一格式
                # .endswith(('.png', '.jpg', '.jpeg', '.gif')): 检查是否为图片
                # - 支持的格式: PNG、JPG、JPEG、GIF
                # - 元组形式: endswith()支持多个后缀
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    # ==================== 步骤5.2: 构造完整文件路径 ====================

                    # os.path.join(root, file): 拼接目录和文件名
                    # - root: 当前目录路径
                    # - file: 文件名
                    # - 返回: 完整的文件路径
                    # - 例如: "/home/user/stickers/default_pack/cat.jpg"
                    file_path = os.path.join(root, file)

                    # ==================== 步骤5.3: 注册表情包 ====================

                    total_scanned += 1  # 扫描计数

                    # await StickerRegistry.register_sticker(): 注册单个文件
                    # - file_path: 文件的完整路径
                    # - pack: 分类标识(default/auto)
                    # - 返回: "registered" 或 "skipped" 或 "error"
                    result = await StickerRegistry.register_sticker(file_path, pack=pack)

                    if result == "registered":
                        total_registered += 1
                    elif result == "skipped":
                        total_skipped += 1
                    elif result == "error":
                        total_errors += 1

        # ==================== 步骤6: 输出扫描完成统计 ====================

        logger.info(
            f"Sticker scanning completed: scanned={total_scanned}, "
            f"registered={total_registered}, skipped={total_skipped}, errors={total_errors}"
        )

    @staticmethod
    async def register_sticker(file_path: str, pack: str = "default") -> str:
        """将单个文件注册入stickers表(计算哈希、OCR、入库)

        这个方法的作用:
        - 读取图片文件内容
        - 计算SHA256哈希(文件唯一ID)
        - 检查是否已注册(SHA256去重)
        - 计算感知哈希pHash(相似检测)
        - OCR识别图片文字
        - 生成fingerprint去重指纹
        - 创建Sticker记录并入库

        注册流程:
        1. 读取文件 → SHA256哈希
        2. 查询数据库 → 已存在则跳过
        3. 打开图片 → pHash感知哈希
        4. 调用OCR → 识别文字
        5. 生成fingerprint → pHash + 归一化OCR
        6. 创建Sticker对象
        7. 写入数据库

        Args:
            file_path: 图片文件的完整路径
                - 类型: 字符串
                - 必须是绝对路径
                - 文件必须存在且可读
                - 示例: "/home/user/stickers/cat.jpg"
            pack: 表情包分类
                - 类型: 字符串
                - 默认值: "default"
                - 可选值: "default"、"auto"
                - 用途: 区分来源和管理策略

        Returns:
            str: 注册结果状态
                - "registered": 成功注册新表情包
                - "skipped": 跳过已存在的表情包
                - "error": 注册过程中发生错误

        Side Effects:
            - 读取文件内容(I/O)
            - 计算哈希(CPU密集)
            - 调用OCR API(网络请求)
            - 写入Stickers表
            - 输出日志

        异常处理:
            - 任何异常都被捕获并记录
            - 不会抛出异常,确保批量注册不中断
            - 失败的文件会记录错误日志

        去重逻辑:
            - 一级去重: 根据SHA256查询数据库
            - 如果已存在: 直接返回,不重复注册
            - 二级去重: stealer模块根据fingerprint去重

        默认字段值:
            - intents="neutral": 中性情感(未打标)
            - is_enabled=True: 启用状态
            - is_banned=False: 未封禁

        Example:
            >>> await StickerRegistry.register_sticker(
            ...     "/home/user/stickers/cat.jpg",
            ...     pack="default"
            ... )
            # INFO: Registered sticker a1b2c3d4e5f6...

            >>> # 重复注册同一文件
            >>> await StickerRegistry.register_sticker(
            ...     "/home/user/stickers/cat.jpg",
            ...     pack="default"
            ... )
            # 静默跳过,因为SHA256已存在
        """

        try:
            # ==================== 步骤0: 输出处理开始日志 ====================

            # os.path.basename(file_path): 获取文件名（不含路径）
            filename = os.path.basename(file_path)
            # logger.debug(f"Processing sticker: {filename}")

            # ==================== 步骤1: 读取文件内容 ====================

            # open(file_path, "rb"): 以二进制只读模式打开文件
            # - "rb": read binary,读取二进制数据
            # - with: 上下文管理器,自动关闭文件
            with open(file_path, "rb") as f:
                # f.read(): 读取所有内容为bytes
                content = f.read()

            # ==================== 步骤2: 计算SHA256哈希 ====================

            # hashlib.sha256(content): 计算SHA256哈希
            # - 输入: 文件的二进制内容
            # - 输出: hash对象
            # .hexdigest(): 转为16进制字符串
            # - 格式: 64个字符(256 bits / 4 bits per hex char = 64)
            # - 示例: "a1b2c3d4e5f6789012345678901234567890123456789012345678901234abcd"
            file_sha256 = hashlib.sha256(content).hexdigest()

            # ==================== 步骤3: 检查是否已注册(SHA256去重) ====================

            # await StickerRepository.get_by_id(file_sha256): 查询数据库
            # - sticker_id就是SHA256哈希
            # - 如果存在: 返回Sticker对象
            # - 如果不存在: 返回None
            existing = await StickerRepository.get_by_id(file_sha256)

            # existing: 如果查到记录
            if existing:
                # logger.debug(
                #     f"Skipping existing sticker: {filename} (SHA256: {file_sha256[:8]}...)"
                # )
                return "skipped"  # 已注册,直接返回

            # ==================== 步骤4: 计算感知哈希pHash ====================

            # Image.open(file_path): 使用PIL打开图片
            # - 返回: Image对象
            # - with: 自动释放图片资源
            with Image.open(file_path) as img:
                # imagehash.phash(img): 计算感知哈希
                # - 算法: Perceptual Hash(pHash)
                # - 特点: 图片缩放、压缩、轻微修改后哈希值相近
                # - 输出: ImageHash对象
                # str(...): 转为字符串
                # - 格式: 16个十六进制字符(64 bits)
                # - 示例: "a1b2c3d4e5f6g7h8"
                phash = str(imagehash.phash(img))

            # ==================== 步骤5: OCR 由后台任务完成 ====================

            # OCR 不再在注册时调用，而是延迟到 StickerWorker 与打标签合并完成
            # 这样可以节省 50% 的 token 消耗（从 2 次 LLM 调用降到 1 次）
            # StickerWorker 会同时完成：OCR + tags + intents + is_banned
            ocr_text = None  # 暂时为空，等待后台任务填充

            # ==================== 步骤6: 生成fingerprint去重指纹 ====================

            # fingerprint: 表情包的去重指纹
            # 格式: "{pHash}+{归一化OCR文本}"
            # - pHash: 感知哈希(16字符)
            # - "+": 分隔符
            # - normalize_ocr_text(ocr_text): 归一化后的OCR文本(最多200字符)
            # 例如: "a1b2c3d4e5f6g7h8+hello world"
            # 用途: 识别"实质相同"的表情包(相似图+相同文字)
            fingerprint = f"{phash}+{normalize_ocr_text(ocr_text)}"

            # ==================== 步骤7: 创建Sticker对象 ====================

            # Sticker(): 创建表情包模型对象
            sticker = Sticker(
                # sticker_id: 主键,使用SHA256哈希
                sticker_id=file_sha256,

                # pack: 表情包分类(default/auto)
                pack=pack,

                # file_path: 文件的完整路径
                file_path=file_path,

                # file_sha256: 文件内容的SHA256哈希
                file_sha256=file_sha256,

                # phash: 感知哈希(用于相似检测)
                phash=phash,

                # ocr_text: OCR识别的原始文本
                ocr_text=ocr_text,

                # fingerprint: 去重指纹(phash + 归一化OCR)
                fingerprint=fingerprint,

                # intents: 意图标注(默认neutral,中性)
                # 注释: 默认包不强制打标,先给neutral,保证selector能选到
                intents="neutral",

                # is_enabled: 是否启用(True=可用)
                is_enabled=True,

                # is_banned: 是否封禁(False=未封禁)
                is_banned=False,
            )

            # ==================== 步骤8: 写入数据库 ====================

            # await db_writer.submit_and_wait(): 提交写入任务并等待完成
            # - 参数1: AsyncCallableJob包装的异步函数调用
            # - 参数2: priority=5(中等优先级)
            # - 效果: 插入Stickers表
            await db_writer.submit_and_wait(
                # AsyncCallableJob: 异步可调用任务
                # - 第一个参数: 异步函数
                # - args: 位置参数元组
                # StickerRepository.add(sticker): 插入表情包记录
                AsyncCallableJob(StickerRepository.add, args=(sticker,)),
                priority=5,
            )

            # ==================== 步骤9: 创建索引任务 ====================

            # 自动为新注册的表情包创建两个索引任务：
            # 1. 向量化任务：将图片向量化并存入 Qdrant（用于语义检索）
            # 2. 标签生成任务：调用 LLM 分析图片生成 tags/intents（用于分类和过滤）
            try:
                # 任务1: 向量化任务（IndexWorker 处理）
                vector_job = IndexJob(
                    item_type="sticker",
                    ref_id=file_sha256,
                    payload_json=json.dumps(
                        {"sticker_id": file_sha256}, ensure_ascii=False
                    ),
                    status="pending",
                    retry_count=0,
                    next_retry_ts=0,
                )

                # 任务2: 标签生成任务（StickerWorker 处理）
                # 注意：OCR 不再预先完成，StickerWorker 会同时完成 OCR + 打标签
                tag_job = IndexJob(
                    item_type="sticker_tag",
                    ref_id=file_sha256,
                    payload_json=json.dumps(
                        {
                            "sticker_id": file_sha256,
                            "intent_hint": "",  # 手动导入的表情包没有 intent hint
                            # ocr_text 字段被移除：由 StickerWorker 的 LLM 调用同时完成
                        },
                        ensure_ascii=False,
                    ),
                    status="pending",
                    retry_count=0,
                    next_retry_ts=0,
                )

                # 提交到数据库写入队列
                await db_writer.submit_and_wait(
                    AsyncCallableJob(IndexJobRepository.add, args=(vector_job,)),
                    priority=5,
                )
                await db_writer.submit_and_wait(
                    AsyncCallableJob(IndexJobRepository.add, args=(tag_job,)),
                    priority=5,
                )
                logger.debug(
                    f"Created 2 index jobs for sticker {file_sha256[:8]}... "
                    f"(vectorization + tag generation)"
                )
            except Exception as e:
                # 索引任务创建失败不影响表情包注册
                # 可以后续通过 backfill 补齐
                logger.warning(
                    f"Failed to create index jobs for sticker {file_sha256[:8]}...: {e}"
                )

            # ==================== 步骤10: 输出成功日志 ====================

            # logger.info(): 记录信息级别日志
            # 输出SHA256前缀,便于追踪
            logger.info(f"Registered sticker {filename} (SHA256: {file_sha256[:8]}...)")

            # 返回成功状态
            return "registered"

        except Exception as e:
            # ==================== 异常处理: 记录错误并继续 ====================

            # 捕获所有异常,不中断批量注册流程
            # logger.error(): 记录错误级别日志
            # - file_path: 失败的文件路径
            # - e: 异常对象
            logger.error(f"注册表情包失败 {file_path}: {e}")

            # 返回错误状态
            return "error"
