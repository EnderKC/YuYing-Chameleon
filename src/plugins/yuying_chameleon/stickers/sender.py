"""表情包消息构造模块 - 将Sticker对象转换为可发送的OneBot消息

这个模块的作用:
1. 封装表情包消息的构造逻辑
2. 将数据库的Sticker记录转换为OneBot协议的MessageSegment
3. 使用base64编码发送图片，解决容器化部署中的文件系统隔离问题
4. 提供统一的表情包发送接口

OneBot协议与MessageSegment(新手必读):
- OneBot: QQ机器人的标准协议,定义消息格式和API
- MessageSegment: OneBot消息的基本单元(文本/图片/表情等)
- 消息类型: text(文本)、image(图片)、face(QQ表情)、at(@某人)等
- 本模块: 专注于image类型的MessageSegment构造

base64发送方式说明:
- 格式: base64:// + base64编码的图片数据
- 用途: 将图片数据直接嵌入消息，无需文件系统访问
- 优势: 解决容器化部署中的文件系统隔离问题
- 性能: 使用LRU缓存优化重复发送，体积增加约33%

与其他发送方式的对比:
1. base64编码: 无需文件系统，跨容器可用，适合容器化部署
2. file://协议: 本地文件，速度快，但要求文件系统可访问
3. http://协议: 网络图片，需下载，适合远程图片

使用方式:
```python
from .stickers.sender import StickerSender
from ..storage.repositories.sticker_repo import StickerRepository

# 1. 从数据库查询表情包
sticker = await StickerRepository.get(sticker_id=123)

# 2. 构造可发送的消息段（自动处理base64编码和错误）
msg_segment = StickerSender.create_message(sticker)

# 3. 发送表情包(在消息处理器中)
await bot.send(event, msg_segment)

# 4. 或者与文本组合发送
from nonebot.adapters.onebot.v11 import Message
msg = Message([
    MessageSegment.text("看这个表情包: "),
    msg_segment
])
await bot.send(event, msg)
```

设计模式:
- 静态方法类: 所有方法都是静态的,无需实例化
- 工具类: 提供单一职责的辅助功能
- 解耦: 将消息构造逻辑与业务逻辑分离
- 缓存优化: 使用LRU缓存避免重复编码
"""

from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

from nonebot import logger
from nonebot.adapters.onebot.v11 import MessageSegment  # OneBot v11协议的消息段类
from ..storage.models import Sticker  # 表情包数据库模型

# 缓存配置：只缓存小于此大小的文件（字节）
# 理由：大文件缓存会占用大量内存，且通常发送频率较低
MAX_CACHEABLE_BYTES = 1_500_000  # 约1.5MB

# LRU 缓存大小限制（缓存条目数）
# 注意：1.5MB 图片 base64 后约 2MB，64 张最坏约 128MB 内存
# 建议：根据容器内存限制调整此值（32-128 为合理范围）
LRU_CACHE_SIZE = 64


class StickerSender:
    """表情包消息发送器 - 构造OneBot协议的图片消息段

    这个类的作用:
    - 封装表情包消息的构造逻辑
    - 将Sticker对象转换为MessageSegment
    - 使用base64编码发送，解决文件系统隔离问题
    - 提供统一的消息构造接口

    设计模式:
    - 静态方法类: 所有方法都是静态的,无需实例化
    - 工具类: 单一职责,专注于消息构造
    - 缓存优化: LRU缓存避免重复编码
    - 容错设计: 错误时返回文本提示而非抛异常

    为什么使用base64编码?
    - 跨容器: 解决NoneBot和OneBot后端在不同文件系统的问题
    - 可靠性: 不依赖文件路径映射和挂载配置
    - 灵活性: 支持分布式部署
    - 性能: 通过缓存优化重复发送

    Example:
        >>> sticker = Sticker(file_path="/app/stickers/cat.jpg")
        >>> msg_segment = StickerSender.create_message(sticker)
        >>> print(msg_segment)
        # MessageSegment.image(file="base64://iVBORw0KGgoAAAA...")
    """

    @staticmethod
    @lru_cache(maxsize=LRU_CACHE_SIZE)
    def _encode_image_to_base64(file_path: str, mtime_ns: int, size: int) -> str:
        """将图片文件编码为base64字符串（带LRU缓存）

        缓存策略:
        - 使用文件路径+修改时间+大小作为缓存键，确保文件更新后缓存失效
        - 缓存大小由 LRU_CACHE_SIZE 控制，采用LRU淘汰策略
        - 只缓存≤1.5MB的文件，大文件不缓存避免内存占用

        重要：mtime_ns 和 size 参数仅用于构建缓存键，不参与函数逻辑

        Args:
            file_path: 图片文件的绝对路径
            mtime_ns: 文件修改时间（纳秒级时间戳，仅用于缓存键）
            size: 文件大小（字节，仅用于缓存键）

        Returns:
            str: base64编码的图片数据（纯字符串，不含前缀）

        Raises:
            FileNotFoundError: 文件不存在
            OSError: 文件读取失败（权限、IO错误等）
            ValueError: 文件路径非法（如包含空字节）
        """
        image_bytes = Path(file_path).read_bytes()
        return base64.b64encode(image_bytes).decode("ascii")

    @staticmethod
    def create_message(sticker: Sticker) -> MessageSegment:
        """根据Sticker记录构造可发送的OneBot MessageSegment

        这个方法的作用:
        - 读取Sticker对象的file_path字段
        - 读取图片文件并编码为base64
        - 构造OneBot的image类型MessageSegment
        - 返回可直接发送的消息段

        为什么使用base64而不是file://协议?
        - 容器化部署: NoneBot和OneBot后端可能在不同容器/文件系统
        - 无需挂载: 不需要配置复杂的volume映射
        - 跨平台: 路径格式统一，无需处理Windows/Linux差异
        - 可靠性: OneBot协议原生支持base64://

        性能优化:
        - LRU缓存: 相同文件只编码一次（基于路径+mtime+size）
        - 智能缓存: 只缓存≤1.5MB的小文件，大文件每次现算
        - 快速失效: 文件修改后自动使缓存失效

        Args:
            sticker: 表情包数据库记录
                - 类型: Sticker对象
                - 必须字段: file_path(绝对路径)
                - 示例: Sticker(
                    id=123,
                    file_path="/app/stickers/cat.jpg",
                    sticker_id="abc123...",
                    ...
                  )

        Returns:
            MessageSegment: OneBot协议的图片或文本消息段
                - 成功: MessageSegment.image(file="base64://...")
                - 失败: MessageSegment.text("（表情包发送失败：...）")
                - 可直接发送给bot.send()

        错误处理:
            - file_path为空: 返回文本提示，记录警告日志
            - 文件不存在: 返回文本提示，记录警告日志
            - 文件为空(0字节): 返回文本提示，记录警告日志
            - 文件路径非法: 返回文本提示，记录警告日志
            - 文件不可读: 返回文本提示，记录警告日志
            - 不抛异常: 确保不会中断上层发送流程

        性能注意事项:
            - 同步IO: 当前实现使用同步文件读取，在 async 上下文中会阻塞事件循环
            - 建议: 高频发送场景或大文件场景建议改为异步实现(asyncio.to_thread)
            - 缓存: 小文件使用 LRU 缓存优化，但每次仍需 stat() 系统调用

        OneBot协议说明:
            MessageSegment.image(file=...)支持多种格式:
            - base64编码: base64://iVBORw0KG... (本方法使用)
            - file://本地文件: file:///path/to/image.jpg
            - http://网络图片: http://example.com/image.jpg

        Example:
            >>> sticker = Sticker(file_path="/app/stickers/cat.jpg")
            >>> msg = StickerSender.create_message(sticker)
            >>> # 成功: MessageSegment.image(file="base64://iVBORw0KG...")
            >>> # 失败: MessageSegment.text("（表情包发送失败：文件不存在）")

            >>> # 在消息处理器中使用
            >>> @bot.on_message()
            >>> async def handle(bot, event):
            ...     sticker = await StickerRepository.random()
            ...     msg = StickerSender.create_message(sticker)
            ...     await bot.send(event, msg)  # 总是成功，失败时发送错误提示
        """

        # ==================== 1. 验证输入参数 ====================

        # 获取文件路径和sticker_id（用于日志）
        # - getattr: 安全获取属性，避免属性不存在时抛异常
        # - or "": 处理None值，确保后续strip()不出错
        file_path = str(getattr(sticker, "file_path", "") or "").strip()
        sticker_id = str(getattr(sticker, "sticker_id", "") or "").strip() or "unknown"

        # 检查file_path是否为空
        if not file_path:
            logger.warning(
                f"[表情包发送] file_path为空 | sticker_id={sticker_id}"
            )
            return MessageSegment.text("（表情包发送失败：无文件路径）")

        # ==================== 2. 读取文件并编码为base64 ====================

        try:
            # 获取文件元数据（用于缓存键和大小判断）
            path = Path(file_path)
            stat = path.stat()

            # 构建缓存键参数
            # - st_mtime_ns: 纳秒级修改时间（更精确）
            # - st_size: 文件大小（字节）
            # 这两个参数确保文件修改后缓存失效
            mtime_ns = getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))
            size = int(stat.st_size)

            # 检查文件大小是否为0
            if size == 0:
                logger.warning(
                    f"[表情包发送] 文件为空 | sticker_id={sticker_id} | "
                    f"file_path={file_path}"
                )
                return MessageSegment.text("（表情包发送失败：文件为空）")

            # 根据文件大小选择编码策略
            if size <= MAX_CACHEABLE_BYTES:
                # 小文件：使用缓存编码（LRU缓存会自动管理）
                base64_data = StickerSender._encode_image_to_base64(file_path, mtime_ns, size)
            else:
                # 大文件：每次现算不缓存（避免内存占用）
                image_bytes = path.read_bytes()
                base64_data = base64.b64encode(image_bytes).decode("ascii")
                logger.debug(
                    f"[表情包发送] 大文件不缓存 | sticker_id={sticker_id} | "
                    f"size={size/1_000_000:.2f}MB"
                )

        except FileNotFoundError:
            # 文件不存在：记录警告并返回文本提示
            logger.warning(
                f"[表情包发送] 文件不存在 | sticker_id={sticker_id} | "
                f"file_path={file_path}"
            )
            return MessageSegment.text("（表情包发送失败：文件不存在）")

        except ValueError as exc:
            # 路径非法（如包含空字节\0）：记录警告并返回文本提示
            logger.warning(
                f"[表情包发送] 文件路径非法 | sticker_id={sticker_id} | "
                f"file_path={file_path} | error={exc}"
            )
            return MessageSegment.text("（表情包发送失败：文件路径非法）")

        except OSError as exc:
            # 其他IO错误（权限、磁盘错误等）：记录警告并返回文本提示
            logger.warning(
                f"[表情包发送] 文件不可读 | sticker_id={sticker_id} | "
                f"file_path={file_path} | error={exc}"
            )
            return MessageSegment.text("（表情包发送失败：文件不可读）")

        # ==================== 3. 构造并返回图片消息段 ====================

        # MessageSegment.image(file="base64://..."): 创建base64图片消息段
        # - base64://前缀: OneBot v11协议要求的格式
        # - base64_data: 纯base64字符串（无MIME类型）
        # - 注意: 不要使用data:image/jpeg;base64,前缀（那是给浏览器用的）
        return MessageSegment.image(file=f"base64://{base64_data}")
