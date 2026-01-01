"""表情包消息构造模块 - 将Sticker对象转换为可发送的OneBot消息

这个模块的作用:
1. 封装表情包消息的构造逻辑
2. 将数据库的Sticker记录转换为OneBot协议的MessageSegment
3. 处理本地文件路径到file://协议的转换
4. 提供统一的表情包发送接口

OneBot协议与MessageSegment(新手必读):
- OneBot: QQ机器人的标准协议,定义消息格式和API
- MessageSegment: OneBot消息的基本单元(文本/图片/表情等)
- 消息类型: text(文本)、image(图片)、face(QQ表情)、at(@某人)等
- 本模块: 专注于image类型的MessageSegment构造

file://协议说明:
- 格式: file:// + 绝对路径
- 用途: 告诉OneBot从本地文件系统读取图片
- 示例: file:///home/user/stickers/cat.jpg
- 优势: 无需上传,直接读取本地文件,速度快

与其他发送方式的对比:
1. file://协议: 本地文件,速度快,适合已下载的表情包
2. http://协议: 网络图片,需下载,适合远程图片
3. base64编码: 图片数据直接嵌入,适合小图片

使用方式:
```python
from .stickers.sender import StickerSender
from ..storage.repositories.sticker_repo import StickerRepository

# 1. 从数据库查询表情包
sticker = await StickerRepository.get(sticker_id=123)

# 2. 构造可发送的消息段
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
"""

from __future__ import annotations

from nonebot.adapters.onebot.v11 import MessageSegment  # OneBot v11协议的消息段类
from ..storage.models import Sticker  # 表情包数据库模型


class StickerSender:
    """表情包消息发送器 - 构造OneBot协议的图片消息段

    这个类的作用:
    - 封装表情包消息的构造逻辑
    - 将Sticker对象转换为MessageSegment
    - 提供统一的消息构造接口

    设计模式:
    - 静态方法类: 所有方法都是静态的,无需实例化
    - 工具类: 单一职责,专注于消息构造
    - 好处: 简单、直接、易于使用

    为什么需要这个类?
    - 封装细节: 隐藏file://协议的拼接细节
    - 统一接口: 所有表情包发送都通过这个类
    - 易于维护: 如果发送方式改变,只需修改这个类
    - 类型安全: 明确输入输出类型,减少错误

    Example:
        >>> sticker = Sticker(file_path="/home/user/stickers/cat.jpg")
        >>> msg_segment = StickerSender.create_message(sticker)
        >>> print(msg_segment)
        # MessageSegment.image(file="file:///home/user/stickers/cat.jpg")
    """

    @staticmethod
    def create_message(sticker: Sticker) -> MessageSegment:
        """根据Sticker记录构造可发送的OneBot MessageSegment

        这个方法的作用:
        - 读取Sticker对象的file_path字段
        - 拼接file://协议前缀
        - 构造OneBot的image类型MessageSegment
        - 返回可直接发送的消息段

        为什么使用file://协议?
        - 本地文件: 表情包已下载到本地磁盘
        - 速度快: 无需网络传输,直接读取
        - 可靠性高: 不依赖外部服务
        - OneBot支持: OneBot协议原生支持file://

        Args:
            sticker: 表情包数据库记录
                - 类型: Sticker对象
                - 必须字段: file_path(绝对路径)
                - 示例: Sticker(
                    id=123,
                    file_path="/home/user/stickers/cat.jpg",
                    text_fingerprint="funny cat",
                    ...
                  )

        Returns:
            MessageSegment: OneBot协议的图片消息段
                - 类型: image类型的MessageSegment
                - file参数: file://绝对路径
                - 可直接发送给bot.send()
                - 示例: MessageSegment.image(
                    file="file:///home/user/stickers/cat.jpg"
                  )

        OneBot协议说明:
            MessageSegment.image(file=...)支持多种格式:
            - file://本地文件: file:///path/to/image.jpg
            - http://网络图片: http://example.com/image.jpg
            - base64编码: base64://iVBORw0KG...

        异常情况:
            - 如果file_path为None: 会抛出TypeError
            - 如果文件不存在: OneBot发送时会失败(本方法不检查)
            - 如果路径格式错误: OneBot会返回错误

        Example:
            >>> sticker = Sticker(file_path="/home/user/cat.jpg")
            >>> msg = StickerSender.create_message(sticker)
            >>> print(msg)
            # MessageSegment(type='image', data={'file': 'file:///home/user/cat.jpg'})

            >>> # 在消息处理器中使用
            >>> @bot.on_message()
            >>> async def handle(bot, event):
            ...     sticker = await StickerRepository.random()
            ...     msg = StickerSender.create_message(sticker)
            ...     await bot.send(event, msg)
        """

        # ==================== 构造并返回图片消息段 ====================

        # MessageSegment.image(file=...): 创建image类型的消息段
        # - MessageSegment: OneBot协议的消息段类
        # - .image(...): 静态方法,创建图片消息段
        # - file参数: 图片的来源(file://、http://、base64://)

        # f"file://{sticker.file_path}": 拼接file://协议
        # - sticker.file_path: 表情包的本地绝对路径
        #   * 示例: "/home/user/stickers/cat.jpg"
        # - f"file://...": 添加file://协议前缀
        #   * 结果: "file:///home/user/stickers/cat.jpg"
        # - 注意: file://后面是绝对路径,所以有3个斜杠(file:// + /)

        # return: 返回构造好的MessageSegment对象
        # - 可以直接传给bot.send()发送
        # - 或者与其他MessageSegment组合成Message
        return MessageSegment.image(file=f"file://{sticker.file_path}")
