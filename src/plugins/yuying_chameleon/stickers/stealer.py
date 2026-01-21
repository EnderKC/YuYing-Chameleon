"""表情包窃取(学习)模块 - 从群聊图片中学习新表情包

这个模块的作用:
1. 监听群聊中的图片消息,自动学习新表情包
2. 使用meme分数筛选表情包质量(大小、尺寸、比例)
3. 使用fingerprint聚合同类表情包,统计出现次数
4. 达到阈值后晋升为正式表情包(auto包)
5. 触发意图打标和违规判定

表情包学习原理(新手必读):
- 问题: 默认表情包库有限,需要持续扩充新表情包
- 解决: 自动学习群聊中高质量的表情包
- 流程: 图片消息 → meme分数筛选 → 候选池聚合 → 出现次数统计 → 晋升入库
- 好处: 无需人工收集,自动扩充表情包库

Meme分数判定:
- 目的: 过滤低质量图片,只学习真正的表情包
- 三个维度(每满足1个+1分,满分3分):
  1. 文件大小 ≤ 1MB (小文件,适合发送)
  2. 图片尺寸 max(宽,高) ≤ 700px (中小尺寸,表情包特征)
  3. 宽高比 0.75-1.35 (接近方形,表情包常见比例)
- 阈值: 默认2分,可配置(yuying_sticker_meme_score_threshold)
- 效果: 过滤风景照、截图、大图等非表情包内容

Fingerprint聚合机制:
- 组成: pHash + 归一化OCR文本
- 格式: "a1b2c3d4e5f6g7h8+hello world"
- 用途: 将相似图片(同一表情包的不同版本)聚合到一个候选
- 出现次数: seen_count,每次看到同类表情包+1
- 好处: 避免重复入库,统计热度

候选池与晋升:
- StickerCandidate表: 候选表情包池(status=pending)
- 首次出现: 创建候选记录,seen_count=1
- 再次出现: 增加seen_count,追加source_qq_ids
- 达到阈值: seen_count >= promote_threshold(默认3次)
- 晋升: 复制到auto包,写入Stickers表,status=promoted
- 好处: 只学习真正流行的表情包,避免噪音

晋升后流程:
1. 复制文件到assets/stickers/auto/目录
2. 创建Sticker记录(pack="auto")
3. 更新候选状态(status="promoted")
4. 创建打标任务(IndexJob: item_type="sticker_tag")
5. 后续: 意图标注(LLM) + 违规判定(LLM)

与其他模块的协作:
- gatekeeper: 拦截图片消息,调用StickerStealer.process_image()
- VisionHelper: OCR识别图片文字
- selector: 从auto包选择表情包发送
- sticker_worker: 处理打标任务(意图+违规判定)

使用方式:
```python
from .stickers.stealer import StickerStealer

# 在消息处理器中,拦截群聊图片
@bot.on_message()
async def handle_image(event):
    if event.message.has_image():
        file_path = await download_image(event.message.image_url)

        # 处理图片,自动学习
        await StickerStealer.process_image(
            scene_id=event.group_id,
            source_qq_id=event.user_id,
            file_path=file_path,
            intent_hint="funny"  # 可选,意图提示
        )
```

配置项:
- yuying_sticker_meme_score_threshold: meme分数阈值(默认2分)
- yuying_sticker_promote_threshold: 晋升阈值(默认3次)

目录结构:
```
/assets/stickers/
├── default_pack/    # 手动收集的表情包(pack=default)
│   └── cat.jpg
└── auto/            # 自动学习的表情包(pack=auto)
    ├── a1b2c3d4e5f6...7890.jpg  # SHA256文件名
    └── 1234567890ab...cdef.png
```
"""

from __future__ import annotations

import hashlib  # Python标准库,哈希算法(SHA256)
import shutil  # Python标准库,文件操作(复制)
import time  # Python标准库,时间戳
import json  # Python标准库,JSON编解码
from pathlib import Path  # Python标准库,路径操作
from typing import Optional, Tuple  # 类型提示

import imagehash  # 第三方库,感知哈希算法
from nonebot import logger  # NoneBot日志记录器
from PIL import Image  # Python图像处理库,读取图片

# 导入项目模块
from ..config import plugin_config  # 插件配置
from ..llm.vision import VisionHelper  # 视觉模型客户端(OCR)
from ..storage.models import Sticker, StickerCandidate  # 数据库模型
from ..storage.models import IndexJob  # 索引任务模型
from ..storage.db_writer import db_writer  # 数据库写入队列
from ..storage.write_jobs import AddIndexJobJob, AsyncCallableJob  # 写入任务
from ..storage.repositories.sticker_candidate_repo import StickerCandidateRepository  # 候选表情包仓库
from ..storage.repositories.sticker_repo import StickerRepository  # 表情包仓库
from .utils import normalize_ocr_text  # OCR文本归一化
from ..paths import assets_dir  # 获取assets目录


class StickerStealer:
    """表情包窃取器 - 从群聊图片中学习新表情包

    这个类的作用:
    - 监听群聊中的图片消息,自动学习表情包
    - 计算meme分数,过滤低质量图片
    - 使用fingerprint聚合同类表情包
    - 统计出现次数,达到阈值后晋升
    - 触发意图打标和违规判定

    设计模式:
    - 静态方法类: 所有方法都是静态的,无需实例化
    - 工具类: 提供表情包学习功能
    - 好处: 简单直接,作为事件处理器使用

    核心流程:
    1. gatekeeper拦截图片消息
    2. 调用process_image()处理图片
    3. 计算meme分数,低于阈值则跳过
    4. 计算fingerprint,查询候选池
    5. 如果已有候选: 增加seen_count
    6. 如果是新候选: 创建候选记录
    7. 检查seen_count是否达到晋升阈值
    8. 如果达到: 调用promote_candidate()晋升

    Example:
        >>> # 在消息处理器中使用
        >>> await StickerStealer.process_image(
        ...     scene_id="123456",
        ...     source_qq_id="789012",
        ...     file_path="/tmp/image.jpg",
        ...     intent_hint="funny"
        ... )
        # INFO: 表情包候选已晋升: candidate_id=1 sticker_id=a1b2c3d4...
    """

    @staticmethod
    async def process_image(
        *,
        scene_id: str,
        source_qq_id: str,
        file_path: str,
        intent_hint: Optional[str] = None,
    ) -> None:
        """处理一张图片文件,更新候选池并在满足阈值时触发晋升

        这个方法的作用:
        - 作为表情包学习的主入口
        - 计算图片特征(meme分数、pHash、SHA256)
        - 判断是否符合表情包标准(meme分数阈值)
        - OCR识别图片文字,生成fingerprint
        - 查询候选池,更新或创建候选记录
        - 检查晋升条件,触发晋升流程

        处理流程:
        1. 计算特征(score, phash, sha256)
        2. 判断meme分数,低于阈值则跳过
        3. OCR识别图片文字
        4. 生成fingerprint(phash + 归一化OCR)
        5. 检查是否已在正式库(Stickers表)
        6. 查询候选池(StickerCandidate表)
        7. 如果已有候选: 增加seen_count, 追加source_qq_id
        8. 如果是新候选: 创建候选记录
        9. 检查seen_count >= 阈值,触发晋升

        为什么需要候选池?
        - 避免噪音: 只学习真正流行的表情包
        - 统计热度: seen_count反映表情包的使用频率
        - 去重: 同类表情包(fingerprint相同)聚合到一个候选
        - 延迟决策: 积累足够证据再决定是否入库

        Args:
            scene_id: 来源群号(或其它场景标识)
                - 类型: 字符串
                - 内容: 群号
                - 用途: 记录表情包的来源场景
                - 示例: "123456789"
            source_qq_id: 来源用户QQ号
                - 类型: 字符串
                - 内容: QQ号
                - 用途: 追踪表情包的发送者
                - 示例: "987654321"
            file_path: 本地文件路径
                - 类型: 字符串
                - 必须是绝对路径
                - 文件必须存在且可读
                - 示例: "/tmp/image_123.jpg"
            intent_hint: 意图提示(可选)
                - 类型: 字符串或None
                - 默认值: None
                - 用途: 后续打标签时的意图参考
                - 示例: "funny", "happy", "sad"

        Returns:
            None: 无返回值

        Side Effects:
            - 读取文件内容(I/O)
            - 计算哈希(CPU密集)
            - 调用OCR API(网络请求)
            - 写入StickerCandidate表
            - 可能写入Stickers表(晋升时)
            - 可能创建IndexJob任务(晋升时)
            - 输出日志

        异常处理:
            - 任何异常都被捕获并记录
            - 不会抛出异常,确保不中断消息处理流程
            - 失败的图片会记录错误日志

        Example:
            >>> # 示例1: 首次出现的表情包
            >>> await StickerStealer.process_image(
            ...     scene_id="123456",
            ...     source_qq_id="789012",
            ...     file_path="/tmp/cat.jpg",
            ...     intent_hint="funny"
            ... )
            # 创建候选记录,seen_count=1,status=pending

            >>> # 示例2: 再次出现(第2次)
            >>> await StickerStealer.process_image(
            ...     scene_id="123456",
            ...     source_qq_id="345678",
            ...     file_path="/tmp/cat_v2.jpg",  # 同一表情包,不同版本
            ...     intent_hint="funny"
            ... )
            # 更新候选记录,seen_count=2,追加source_qq_id

            >>> # 示例3: 达到阈值(第3次)
            >>> await StickerStealer.process_image(
            ...     scene_id="123456",
            ...     source_qq_id="901234",
            ...     file_path="/tmp/cat_v3.jpg",
            ...     intent_hint="funny"
            ... )
            # 晋升为正式表情包,复制到auto包,写入Stickers表
        """

        try:
            # ==================== 步骤1: 计算图片特征 ====================

            # StickerStealer._compute_features(): 计算meme分数和哈希
            # - 参数: file_path(图片路径)
            # - 返回: (score, phash, sha256)元组
            #   * score: meme分数(0-3分,满足1个条件+1分)
            #   * phash: 感知哈希(16个十六进制字符)
            #   * sha256: 文件内容哈希(64个十六进制字符)
            score, phash, sha256 = StickerStealer._compute_features(file_path)

            # ==================== 步骤2: 判断meme分数阈值 ====================

            # score < 阈值: meme分数不达标
            # int(plugin_config.yuying_sticker_meme_score_threshold): 阈值
            # - 默认值: 2分
            # - 意义: 至少满足2个条件(大小/尺寸/比例)才是表情包
            if score < int(plugin_config.yuying_sticker_meme_score_threshold):
                return  # 不符合表情包标准,跳过

            # ==================== 步骤3: OCR识别图片文字 ====================

            # await VisionHelper.ocr_image(file_path): 调用OCR识别
            # - 轻量文字识别(失败则降级为空字符串)
            # - 成功: 返回识别的文本
            # - 失败: 返回None或空字符串(降级策略)
            ocr_text = await VisionHelper.ocr_image(file_path)

            # ==================== 步骤4: 生成fingerprint去重指纹 ====================

            # fingerprint: 表情包的去重指纹
            # f"{phash}+{normalize_ocr_text(ocr_text)}": 拼接pHash和归一化OCR
            # - phash: 感知哈希(相似图片的phash相近)
            # - normalize_ocr_text(ocr_text): 归一化后的OCR文本
            # - 格式: "a1b2c3d4e5f6g7h8+hello world"
            # - 用途: 聚合同类表情包(相似图+相同文字)
            fingerprint = f"{phash}+{normalize_ocr_text(ocr_text)}"

            # ==================== 步骤5: 检查是否已在正式库 ====================

            # await StickerRepository.get_by_id(sha256): 查询Stickers表
            # - sticker_id就是SHA256哈希
            # - 如果存在: 返回Sticker对象
            # - 如果不存在: 返回None
            existing = await StickerRepository.get_by_id(sha256)

            # existing: 如果查到记录
            if existing:
                return  # 已在正式库,无需再学习,直接返回

            # ==================== 步骤6: 查询候选池 ====================

            # await StickerCandidateRepository.get_by_fingerprint(): 按fingerprint查询
            # - 参数: fingerprint(去重指纹)
            # - 返回: StickerCandidate对象或None
            # - 作用: 查找是否已有同类表情包的候选记录
            candidate = await StickerCandidateRepository.get_by_fingerprint(fingerprint)

            # ==================== 步骤7: 更新或创建候选记录 ====================

            # candidate: 如果已有候选记录
            if candidate:
                # ==================== 情况1: 已有候选,更新计数 ====================

                # ==================== 步骤7.1: 增加seen_count(出现次数+1) ====================

                # await db_writer.submit_and_wait(): 提交写入任务并等待完成
                # AsyncCallableJob: 异步可调用任务
                # StickerCandidateRepository.increment_seen_count(): 增加计数
                # - 参数: candidate_id(候选ID)
                # - 效果: UPDATE sticker_candidates SET seen_count=seen_count+1
                # - 返回: 更新后的StickerCandidate对象
                candidate = await db_writer.submit_and_wait(
                    AsyncCallableJob(
                        StickerCandidateRepository.increment_seen_count,
                        args=(candidate.candidate_id,),
                    ),
                    priority=5,  # 优先级5(中等)
                )
                if not isinstance(candidate, StickerCandidate):
                    return

                # ==================== 步骤7.2: 追加source_qq_id(发送者列表) ====================

                # await db_writer.submit_and_wait(): 提交写入任务并等待完成
                # StickerCandidateRepository.append_source_qq_id(): 追加QQ号
                # - 参数: candidate_id(候选ID), source_qq_id(发送者QQ号)
                # - 效果: 将QQ号追加到source_qq_ids JSON数组
                # - 用途: 追踪谁在使用这个表情包
                await db_writer.submit_and_wait(
                    AsyncCallableJob(
                        StickerCandidateRepository.append_source_qq_id,
                        args=(candidate.candidate_id, source_qq_id),
                    ),
                    priority=5,
                )

                # ==================== 步骤7.3: 检查是否达到晋升条件 ====================

                # 晋升条件(同时满足):
                # 1. candidate.status == "pending": 状态是待定(未晋升)
                # 2. candidate.seen_count >= 阈值: 出现次数达到阈值
                if (
                    candidate.status == "pending"
                    and candidate.seen_count >= int(plugin_config.yuying_sticker_promote_threshold)
                ):
                    # ==================== 达到阈值,触发晋升 ====================

                    # await StickerStealer.promote_candidate(): 晋升为正式表情包
                    # 参数:
                    # - candidate: 候选记录对象
                    # - sha256: 文件SHA256哈希
                    # - phash: 感知哈希
                    # - ocr_text: OCR识别的文本
                    # - intent_hint: 意图提示(可选)
                    await StickerStealer.promote_candidate(
                        candidate,
                        sha256=sha256,
                        phash=phash,
                        ocr_text=ocr_text,
                        intent_hint=intent_hint,
                    )

            else:
                # ==================== 情况2: 新候选,创建记录 ====================

                # ==================== 步骤7.4: 创建StickerCandidate对象 ====================

                # StickerCandidate(): 创建候选表情包模型对象
                new_candidate = StickerCandidate(
                    # fingerprint: 去重指纹(pHash + 归一化OCR)
                    fingerprint=fingerprint,

                    # phash: 感知哈希(用于相似检测)
                    phash=phash,

                    # ocr_text: OCR识别的原始文本
                    ocr_text=ocr_text,

                    # sha256_sample: 样本文件的SHA256哈希
                    sha256_sample=sha256,

                    # sample_file_path: 样本文件的路径
                    sample_file_path=file_path,

                    # scene_id: 来源场景(群号)
                    scene_id=scene_id,

                    # first_seen_ts: 首次出现时间戳
                    # int(time.time()): 当前Unix时间戳(秒级)
                    first_seen_ts=int(time.time()),

                    # last_seen_ts: 最后出现时间戳(初始=首次)
                    last_seen_ts=int(time.time()),

                    # status: 状态(pending=待定,promoted=已晋升)
                    status="pending",

                    # source_qq_ids: 发送者QQ号列表(JSON数组格式)
                    # f'["{source_qq_id}"]': 初始只有一个QQ号
                    # 格式: '["123456789"]'
                    source_qq_ids=f'["{source_qq_id}"]',
                )

                # ==================== 步骤7.5: 写入数据库 ====================

                # await db_writer.submit_and_wait(): 提交写入任务并等待完成
                # AsyncCallableJob: 异步可调用任务
                # StickerCandidateRepository.add(new_candidate): 插入候选记录
                # - 操作: INSERT INTO sticker_candidates VALUES (...)
                await db_writer.submit_and_wait(
                    AsyncCallableJob(StickerCandidateRepository.add, args=(new_candidate,)),
                    priority=5,
                )

        except Exception as exc:
            # ==================== 异常处理: 记录错误并继续 ====================

            # 捕获所有异常,不中断消息处理流程
            # logger.error(): 记录错误级别日志
            # - exc: 异常对象
            logger.error(f"表情包偷取处理失败：{exc}")
            # 不抛出异常,让消息处理流程继续

    @staticmethod
    def _compute_features(file_path: str) -> Tuple[int, str, str]:
        """计算候选判定与指纹所需特征(meme分数、pHash、SHA256)

        这个方法的作用:
        - 读取图片文件
        - 计算SHA256哈希(精确去重)
        - 计算pHash感知哈希(相似检测)
        - 计算meme分数(表情包质量判定)

        Meme分数计算规则(满分3分):
        1. 文件大小 ≤ 1MB(1024 KB): +1分
           - 理由: 表情包通常是小文件,大文件可能是照片或截图
           - 效果: 过滤高清照片、长图等大文件

        2. 图片尺寸 max(宽,高) ≤ 700px: +1分
           - 理由: 表情包通常是中小尺寸,大尺寸图片可能不是表情包
           - 效果: 过滤高清图片、海报等大图

        3. 宽高比 0.75-1.35: +1分
           - 理由: 表情包通常接近方形(1:1),长条形图片可能不是表情包
           - 效果: 过滤横幅、长图、截图等非方形图片
           - 示例:
             * 1:1(1.0)正方形 ✓
             * 4:3(1.33)接近方形 ✓
             * 16:9(1.78)长条形 ✗
             * 1:2(0.5)竖长条 ✗

        为什么需要meme分数?
        - 质量控制: 过滤非表情包图片(照片、截图、海报等)
        - 自动化: 无需人工审核,自动判定
        - 灵活性: 阈值可配置(默认2分)
        - 效率: 轻量计算,无需LLM

        Args:
            file_path: 图片文件的完整路径
                - 类型: 字符串
                - 必须是绝对路径
                - 文件必须存在且可读
                - 示例: "/tmp/image_123.jpg"

        Returns:
            Tuple[int, str, str]: 特征元组
                - score: meme分数(0-3分,整数)
                - phash: 感知哈希(16个十六进制字符)
                - sha256: 文件SHA256哈希(64个十六进制字符)

        Side Effects:
            - 读取文件内容(I/O)
            - 打开图片文件(PIL)
            - 计算哈希(CPU密集)

        Example:
            >>> # 示例1: 标准表情包(满分3分)
            >>> score, phash, sha256 = StickerStealer._compute_features("/tmp/cat.jpg")
            >>> # 文件: 200KB(✓), 尺寸: 500x500(✓), 比例: 1.0(✓)
            >>> print(score)  # 3
            >>> print(phash)  # "a1b2c3d4e5f6g7h8"
            >>> print(sha256)  # "1234567890abcdef..."

            >>> # 示例2: 低质量图片(1分)
            >>> score, phash, sha256 = StickerStealer._compute_features("/tmp/photo.jpg")
            >>> # 文件: 5MB(✗), 尺寸: 4000x3000(✗), 比例: 1.33(✓)
            >>> print(score)  # 1 (只满足宽高比条件)

            >>> # 示例3: 长条图(0分)
            >>> score, phash, sha256 = StickerStealer._compute_features("/tmp/banner.jpg")
            >>> # 文件: 2MB(✗), 尺寸: 1920x300(✗), 比例: 6.4(✗)
            >>> print(score)  # 0 (不满足任何条件)
        """

        # ==================== 步骤1: 读取文件内容 ====================

        # Path(file_path): 创建Path对象
        p = Path(file_path)

        # p.read_bytes(): 读取文件的所有字节内容
        # - 返回: bytes对象
        content = p.read_bytes()

        # ==================== 步骤2: 计算SHA256哈希 ====================

        # hashlib.sha256(content): 计算SHA256哈希
        # .hexdigest(): 转为16进制字符串(64个字符)
        sha256 = hashlib.sha256(content).hexdigest()

        # ==================== 步骤3: 打开图片并计算pHash ====================

        # Image.open(p): 使用PIL打开图片
        # with: 自动释放图片资源
        with Image.open(p) as img:
            # img.convert("RGB"): 转换为RGB模式
            # - 原因: 确保图片是RGB格式(pHash需要RGB)
            # - 效果: RGBA/灰度/索引色等都转为RGB
            img = img.convert("RGB")

            # imagehash.phash(img): 计算感知哈希
            # str(...): 转为字符串(16个十六进制字符)
            phash = str(imagehash.phash(img))

            # img.size: 获取图片尺寸(宽, 高)元组
            w, h = img.size

        # ==================== 步骤4: 计算文件大小(KB) ====================

        # len(content): 文件字节数
        # int(len(content) / 1024): 转换为KB(整数)
        # max(1, ...): 最小值为1KB(避免0导致的除法错误)
        size_kb = max(1, int(len(content) / 1024))

        # ==================== 步骤5: 计算宽高比 ====================

        # w / h: 宽除以高
        # if h: 如果高度不为0
        #   * 返回: 宽高比(浮点数)
        #   * 示例: 800x600 → 1.33
        # else: 如果高度为0(异常情况)
        #   * 返回: 0.0(避免除零错误)
        ratio = w / h if h else 0.0

        # ==================== 步骤6: 计算meme分数(0-3分) ====================

        # score: 初始为0分
        score = 0

        # ==================== 条件1: 文件大小 ≤ 1MB ====================
        # size_kb <= 1024: 文件大小是否 ≤ 1MB(1024 KB)
        if size_kb <= 1024:
            score += 1  # 满足条件,+1分

        # ==================== 条件2: 图片尺寸 max(宽,高) ≤ 700px ====================
        # max(w, h) <= 700: 宽或高的最大值是否 ≤ 700像素
        if max(w, h) <= 700:
            score += 1  # 满足条件,+1分

        # ==================== 条件3: 宽高比 0.75-1.35 ====================
        # 0.75 <= ratio <= 1.35: 宽高比是否在合理范围内
        # - 0.75: 3:4竖版(0.75)
        # - 1.0: 正方形(1:1)
        # - 1.35: 4:3横版(1.33)
        if 0.75 <= ratio <= 1.35:
            score += 1  # 满足条件,+1分

        # ==================== 步骤7: 返回特征元组 ====================

        # return (score, phash, sha256): 返回3个特征
        # - score: meme分数(0-3分)
        # - phash: 感知哈希(16字符)
        # - sha256: 文件哈希(64字符)
        return score, phash, sha256

    @staticmethod
    async def promote_candidate(
        candidate: StickerCandidate,
        *,
        sha256: str,
        phash: str,
        ocr_text: str,
        intent_hint: Optional[str],
    ) -> None:
        """将候选晋升为正式表情包(复制到auto包并写入stickers表)

        这个方法的作用:
        - 将候选表情包(StickerCandidate)晋升为正式表情包(Sticker)
        - 复制文件到assets/stickers/auto/目录
        - 使用SHA256作为文件名(确保唯一性)
        - 写入Stickers表(pack="auto")
        - 更新候选状态(status="promoted")
        - 创建打标任务(IndexJob: item_type="sticker_tag")

        晋升流程:
        1. 获取auto包目录,创建目录(如果不存在)
        2. 推断文件扩展名(.png/.jpg/.gif/.webp)
        3. 复制文件到auto包,文件名为SHA256
        4. 生成fingerprint(pHash + 归一化OCR)
        5. 创建Sticker记录(pack="auto")
        6. 写入Stickers表
        7. 更新候选状态(status="promoted")
        8. 创建打标任务(意图标注+违规判定)

        为什么需要复制文件?
        - 持久化: 原始文件可能在临时目录,会被清理
        - 规范化: 统一存储在assets/stickers/auto/目录
        - 命名: 使用SHA256命名,避免冲突
        - 管理: 便于备份、迁移、清理

        为什么需要打标任务?
        - 意图标注: 使用LLM识别表情包的情感意图(funny/happy/sad等)
        - 违规判定: 使用LLM检测敏感内容(政治/色情/暴力等)
        - 异步处理: 不阻塞当前流程,由sticker_worker后台处理
        - 可追溯: IndexJob记录处理状态

        Args:
            candidate: 候选表情包对象
                - 类型: StickerCandidate
                - 必须字段: candidate_id, sample_file_path
                - 示例: StickerCandidate(candidate_id=1, sample_file_path="/tmp/img.jpg")
            sha256: 文件SHA256哈希
                - 类型: 字符串(64个十六进制字符)
                - 用途: 作为sticker_id和文件名
                - 示例: "a1b2c3d4e5f6...7890"
            phash: 感知哈希
                - 类型: 字符串(16个十六进制字符)
                - 用途: 生成fingerprint
                - 示例: "a1b2c3d4e5f6g7h8"
            ocr_text: OCR识别的文本
                - 类型: 字符串
                - 可能为空字符串(OCR失败时)
                - 用途: 生成fingerprint,作为打标参考
                - 示例: "今天天气真好"
            intent_hint: 意图提示(可选)
                - 类型: 字符串或None
                - 默认值: None
                - 用途: 打标任务的意图参考
                - 示例: "funny", "happy", "sad"

        Returns:
            None: 无返回值

        Side Effects:
            - 创建assets/stickers/auto/目录(如果不存在)
            - 复制文件到auto包目录
            - 写入Stickers表
            - 更新StickerCandidate表(status="promoted")
            - 创建IndexJob任务
            - 输出日志

        异常处理:
            - 文件复制失败: 使用read_bytes()回退
            - 扩展名识别失败: 默认使用.png

        Example:
            >>> candidate = StickerCandidate(
            ...     candidate_id=1,
            ...     sample_file_path="/tmp/cat.jpg",
            ...     fingerprint="a1b2c3d4e5f6g7h8+funny cat",
            ...     seen_count=3
            ... )
            >>> await StickerStealer.promote_candidate(
            ...     candidate,
            ...     sha256="1234567890abcdef...",
            ...     phash="a1b2c3d4e5f6g7h8",
            ...     ocr_text="funny cat",
            ...     intent_hint="funny"
            ... )
            # INFO: 表情包候选已晋升: candidate_id=1 sticker_id=1234567890abcdef...
            # 结果:
            # - 文件复制到: assets/stickers/auto/1234567890abcdef...jpg
            # - Stickers表新增记录(pack="auto", sticker_id="1234567890abcdef...")
            # - 候选状态更新(status="promoted")
            # - IndexJob任务创建(item_type="sticker_tag")
        """

        # ==================== 步骤1: 获取auto包目录 ====================

        # StickerStealer._auto_pack_dir(): 获取auto包目录
        # - 返回: Path对象(assets/stickers/auto/)
        auto_dir = StickerStealer._auto_pack_dir()

        # auto_dir.mkdir(parents=True, exist_ok=True): 创建目录
        # - parents=True: 递归创建父目录(如果不存在)
        # - exist_ok=True: 如果目录已存在,不抛出异常
        auto_dir.mkdir(parents=True, exist_ok=True)

        # ==================== 步骤2: 推断文件扩展名 ====================

        # Path(candidate.sample_file_path): 创建Path对象
        src = Path(candidate.sample_file_path)

        # src.suffix: 获取文件扩展名(包括".")
        # .lower(): 转小写
        # 示例: "/tmp/cat.JPG" → ".jpg"
        ext = (src.suffix or "").lower()

        # ext not in {...}: 扩展名不在支持的格式中
        # - 原因: src可能是临时.img文件,需要用真实格式兜底
        if ext not in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
            # ==================== 步骤2.1: 使用PIL检测真实格式 ====================

            try:
                # Image.open(src): 打开图片
                # img.format: 获取图片格式(PIL检测的真实格式)
                # 示例: "PNG", "JPEG", "GIF", "WEBP"
                with Image.open(src) as img:
                    fmt = (img.format or "").lower()

                # {...}.get(fmt, ".png"): 格式映射
                # - 将PIL格式名转换为文件扩展名
                # - 默认值: ".png"(如果格式不在映射中)
                ext = {
                    "png": ".png",
                    "jpeg": ".jpg",
                    "jpg": ".jpg",
                    "gif": ".gif",
                    "webp": ".webp",
                }.get(fmt, ".png")

            except Exception:
                # PIL检测失败,使用默认扩展名
                ext = ".png"

        # ==================== 步骤3: 构造目标文件路径 ====================

        # auto_dir / f"{sha256}{ext}": 拼接路径
        # - auto_dir: assets/stickers/auto/
        # - sha256: 文件SHA256哈希(64字符)
        # - ext: 文件扩展名(.png/.jpg/.gif/.webp)
        # 示例: assets/stickers/auto/1234567890abcdef...jpg
        dst = auto_dir / f"{sha256}{ext}"

        # ==================== 步骤4: 复制文件 ====================

        try:
            # shutil.copyfile(src, dst): 复制文件
            # - src: 源文件路径
            # - dst: 目标文件路径
            # - 效果: 复制文件内容(不复制元数据)
            shutil.copyfile(src, dst)

        except Exception:
            # ==================== 复制失败回退策略 ====================

            # 若跨盘失败(源和目标在不同磁盘),则回退为复制字节内容
            # dst.write_bytes(src.read_bytes()): 读取源文件字节并写入目标
            # - src.read_bytes(): 读取源文件的所有字节
            # - dst.write_bytes(...): 写入目标文件
            dst.write_bytes(src.read_bytes())

        # ==================== 步骤5: 生成fingerprint ====================

        # fingerprint: 表情包的去重指纹
        # f"{phash}+{normalize_ocr_text(ocr_text)}": 拼接pHash和归一化OCR
        fingerprint = f"{phash}+{normalize_ocr_text(ocr_text)}"

        # ==================== 步骤6: 创建Sticker对象 ====================

        # Sticker(): 创建表情包模型对象
        sticker = Sticker(
            # sticker_id: 主键,使用SHA256哈希
            sticker_id=sha256,

            # pack: 表情包分类(auto=自动学习)
            pack="auto",

            # file_path: 文件的完整路径(auto包目录下)
            # str(dst): Path对象转字符串
            file_path=str(dst),

            # file_sha256: 文件内容的SHA256哈希
            file_sha256=sha256,

            # phash: 感知哈希(用于相似检测)
            phash=phash,

            # ocr_text: OCR识别的原始文本
            ocr_text=ocr_text,

            # fingerprint: 去重指纹(phash + 归一化OCR)
            fingerprint=fingerprint,

            # is_enabled: 是否启用(True=可用)
            is_enabled=True,

            # is_banned: 是否封禁(False=未封禁)
            is_banned=False,
        )

        # ==================== 步骤7: 写入Stickers表 ====================

        # await db_writer.submit_and_wait(): 提交写入任务并等待完成
        # AsyncCallableJob: 异步可调用任务
        # StickerRepository.add(sticker): 插入表情包记录
        # - 操作: INSERT INTO stickers VALUES (...)
        await db_writer.submit_and_wait(
            AsyncCallableJob(StickerRepository.add, args=(sticker,)),
            priority=5,
        )

        # ==================== 步骤8: 更新候选状态 ====================

        # await db_writer.submit_and_wait(): 提交写入任务并等待完成
        # StickerCandidateRepository.update_status(): 更新状态
        # - 参数: candidate_id(候选ID), status="promoted"(已晋升)
        # - 操作: UPDATE sticker_candidates SET status='promoted' WHERE candidate_id=?
        await db_writer.submit_and_wait(
            AsyncCallableJob(StickerCandidateRepository.update_status, args=(candidate.candidate_id, "promoted")),
            priority=5,
        )

        # ==================== 步骤9: 输出晋升日志 ====================

        # logger.info(): 记录信息级别日志
        logger.info(f"表情包候选已晋升:candidate_id={candidate.candidate_id} sticker_id={sha256}")

        # ==================== 步骤10: 创建打标任务(异步) ====================

        # payload: 任务载荷(包含打标所需信息)
        payload = {
            # sticker_id: 表情包ID
            "sticker_id": sha256,

            # intent_hint: 意图提示(可选)
            # (intent_hint or "").strip() or None: 去除空格,空字符串转None
            "intent_hint": (intent_hint or "").strip() or None,

            # ocr_text: OCR识别的文本(可选)
            "ocr_text": (ocr_text or "").strip() or None,
        }

        # await db_writer.submit(): 提交写入任务(不等待)
        # AddIndexJobJob: 添加索引任务
        await db_writer.submit(
            AddIndexJobJob(
                IndexJob(
                    # item_type: 任务类型(sticker_tag=表情包打标)
                    item_type="sticker_tag",

                    # ref_id: 引用ID(表情包ID)
                    # str(sha256): SHA256哈希转字符串
                    ref_id=str(sha256),

                    # payload_json: 任务载荷(JSON格式)
                    # json.dumps(payload, ensure_ascii=False): 转为JSON字符串
                    # - ensure_ascii=False: 保留中文字符,不转义为\uXXXX
                    payload_json=json.dumps(payload, ensure_ascii=False),

                    # status: 状态(pending=待处理)
                    status="pending",
                )
            ),
            priority=5,
        )

        # 注释: 打标任务的后续处理
        # - sticker_worker会从index_jobs表读取待处理任务
        # - 调用LLM进行意图标注(agree/thanks/sorry/shock等9种)
        # - 调用LLM进行违规判定(is_banned=True/False)
        # - 更新Stickers表的intents和is_banned字段
        # - 更新IndexJob状态(status="done")

    @staticmethod
    def _auto_pack_dir() -> Path:
        """获取自动表情包目录(assets/stickers/auto)

        这个方法的作用:
        - 返回auto包的完整路径
        - 用于存储自动学习的表情包

        为什么需要auto目录?
        - 区分来源: 区分手动收集(default_pack)和自动学习(auto)
        - 便于管理: 可以单独清理或备份auto包
        - 质量控制: 可以定期审查auto包的质量
        - 配置策略: 可以单独配置auto包的使用策略(如权重、冷却时间)

        Returns:
            Path: auto包目录的Path对象
                - 完整路径: assets/stickers/auto/
                - 注意: 只返回路径,不创建目录(由调用方创建)

        Example:
            >>> auto_dir = StickerStealer._auto_pack_dir()
            >>> print(auto_dir)
            # /home/user/project/assets/stickers/auto

            >>> auto_dir.mkdir(parents=True, exist_ok=True)
            # 创建目录(如果不存在)
        """

        # ==================== 构造并返回auto包路径 ====================

        # assets_dir(): 获取assets目录的Path对象
        # / "stickers": 拼接stickers子目录
        # / "auto": 拼接auto子目录
        # 最终路径: assets/stickers/auto/
        return assets_dir() / "stickers" / "auto"
