"""视觉模型辅助模块 - 基于LLM的图片理解和文字识别

这个模块的作用:
1. 提供图片说明生成功能(caption)
2. 提供轻量级OCR文字识别功能
3. 支持本地文件和在线URL两种图片输入方式
4. 将图片转为base64 data URL供视觉模型处理

为什么需要视觉功能?
- 让机器人能够"看懂"用户发送的图片
- 将图片内容转为文本,供LLM理解和回复
- 识别表情包中的文字,用于表情包匹配
- 理解图文混合消息的完整含义

使用场景:
- 用户发送图片消息时,生成图片说明补充上下文
- 表情包偷取时,识别表情包上的文字
- 媒体缓存(MediaCache)预处理图片内容

关键概念(新手必读):
- Vision Model: 支持图像输入的多模态LLM(如GPT-4V)
- 本项目实现说明:
  * 图片说明/OCR 完全复用 cheap_llm_* 配置(模型/地址/key/超时)
  * 若"看不懂图片"，请将 cheap_llm_model 配成支持图片输入的模型
  * 若图片任务更易超时/更慢，请调大 cheap_llm_timeout
- Caption: 图片说明,用简短文字描述图片内容
- OCR: 光学字符识别(Optical Character Recognition),识别图中文字
- Data URL: 将图片编码为base64字符串,嵌入到URL中
- MIME类型: 标识文件类型的标准(如image/png、image/jpeg)

Data URL格式说明:
格式: data:[MIME类型];base64,[base64编码的数据]
示例: data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...
好处: 无需上传文件,直接在API请求中传递图片数据

使用方式:
```python
from .llm.vision import VisionHelper

# 方式1: 使用本地文件路径
caption = await VisionHelper.caption_image(file_path="/path/to/image.png")

# 方式2: 使用在线URL
caption = await VisionHelper.caption_image(url="https://example.com/image.jpg")

# OCR识别文字
text = await VisionHelper.ocr_image(file_path="/path/to/meme.png")
```
"""

from __future__ import annotations

import base64  # Python标准库,用于base64编码
import io
import json
import re
from pathlib import Path  # 文件路径处理
from typing import Optional, Tuple  # 类型提示

from nonebot import logger  # NoneBot日志记录器
from PIL import Image

from .client import get_task_llm  # 支持模型组回落


class VisionHelper:
    """图片说明与文字识别辅助类

    这个类的作用:
    - 提供静态方法封装视觉模型的常见调用场景
    - 统一处理图片的输入格式(文件路径 or URL)
    - 统一处理异常和返回值(失败返回空字符串)
    - 限制输出长度,避免冗长的描述

    设计模式:
    - 静态方法类: 所有方法都是静态的,无需实例化
    - 好处: 作为命名空间使用,避免全局函数污染
    - 用法: VisionHelper.caption_image(...) 直接调用

    两个核心方法:
    1. caption_image(): 生成简短的图片说明(<=20字)
    2. ocr_image(): 识别图片中的文字(<=200字)

    私有方法:
    - _to_data_url(): 将图片bytes转为base64 data URL
    """

    @staticmethod
    def _to_data_url(image_bytes: bytes, suffix: str) -> str:
        """将图片bytes转为base64 data URL

        Data URL格式:
        data:[MIME类型];base64,[base64编码的数据]

        这个方法的作用:
        - 将图片的二进制数据编码为base64字符串
        - 根据文件后缀判断MIME类型
        - 拼接成标准的data URL格式
        - 供视觉模型API使用

        为什么使用data URL?
        - 无需上传文件到服务器
        - 直接在API请求中传递图片
        - 简化文件处理流程
        - 支持本地文件的视觉分析

        Args:
            image_bytes: 图片的二进制数据(bytes类型)
            suffix: 文件后缀名,如 ".png", ".jpg", ".jpeg"

        Returns:
            str: data URL格式的字符串
                格式: "data:image/png;base64,iVBORw0KGgo..."

        MIME类型映射:
            - .png → image/png
            - .jpg, .jpeg → image/jpeg
            - .gif → image/gif
            - 其他 → image/png (默认)

        Example:
            >>> image_bytes = Path("cat.jpg").read_bytes()
            >>> data_url = VisionHelper._to_data_url(image_bytes, ".jpg")
            >>> print(data_url[:50])  # "data:image/jpeg;base64,/9j/4AAQSkZJRgABA..."
        """

        # ==================== 步骤1: 解析文件后缀,判断MIME类型 ====================

        # (suffix or ""): 如果suffix是None,转为空字符串
        # .lower(): 转为小写,统一格式
        # .lstrip("."): 移除开头的点号
        # 例如: ".PNG" → "png", "jpg" → "jpg"
        ext = (suffix or "").lower().lstrip(".")

        # 默认MIME类型为PNG
        mime = "image/png"

        # 根据文件扩展名设置正确的MIME类型
        if ext in {"jpg", "jpeg"}:  # JPEG格式
            mime = "image/jpeg"
        elif ext in {"gif"}:  # GIF格式
            # 多数多模态模型/网关不支持 image/gif；将动图取首帧转为 PNG
            try:
                with Image.open(io.BytesIO(image_bytes)) as img:
                    img.seek(0)
                    frame = img.convert("RGBA")
                    buf = io.BytesIO()
                    frame.save(buf, format="PNG")
                    image_bytes = buf.getvalue()
                mime = "image/png"
            except Exception:
                # 转换失败时仍回退为 PNG mime，避免宣称 image/gif
                mime = "image/png"
        # 其他格式(如png、webp等)使用默认的image/png

        # ==================== 步骤2: Base64编码 ====================

        # base64.b64encode(image_bytes): 将bytes编码为base64
        # - 输入: 图片的二进制数据(bytes)
        # - 输出: base64编码后的bytes对象
        # .decode("ascii"): 将base64 bytes转为ASCII字符串
        # - base64只包含ASCII字符,所以可以安全decode
        b64 = base64.b64encode(image_bytes).decode("ascii")

        # ==================== 步骤3: 拼接data URL ====================

        # f-string格式化拼接data URL
        # 格式: data:[MIME类型];base64,[base64数据]
        # 示例: "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAA..."
        return f"data:{mime};base64,{b64}"

    @staticmethod
    def _extract_first_json_object(text: str) -> Optional[dict[str, object]]:
        """从输出中尽力提取第一个 JSON object。"""

        raw = (text or "").strip()
        if not raw:
            return None

        # 去除常见代码块包裹
        raw = raw.strip("` \n\r\t")
        m = re.search(r"(\{.*\})", raw, flags=re.S)
        if not m:
            return None
        try:
            obj = json.loads(m.group(1))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    @staticmethod
    def to_data_url(image_bytes: bytes, suffix: str) -> str:
        """公开的 data URL 工具方法（供 embedding 等模块复用）

        这个方法的作用:
        - 为其他模块提供图片转 data URL 的能力
        - 复用内部 _to_data_url 的实现
        - 用于向量化、OCR 等需要将图片转为可传输格式的场景

        为什么需要公开接口?
        - 向量化模块需要将表情包转为 data URL 发送给 embedding API
        - 避免重复实现图片编码逻辑
        - 保持代码复用性和一致性

        Args:
            image_bytes: 图片的字节流数据
                - 类型: bytes
                - 示例: Path("sticker.jpg").read_bytes()
            suffix: 文件后缀名
                - 类型: str
                - 示例: ".jpg", ".png", ".gif"
                - 用于判断 MIME 类型

        Returns:
            str: data URL 格式的字符串
                - 格式: "data:image/jpeg;base64,..."
                - 可直接用于 API 调用

        Example:
            >>> from pathlib import Path
            >>> image_data = Path("sticker.jpg").read_bytes()
            >>> url = VisionHelper.to_data_url(image_data, ".jpg")
            >>> # 可用于 embedding API 的图片输入
        """
        return VisionHelper._to_data_url(image_bytes, suffix)

    @staticmethod
    async def caption_and_ocr_image(
        file_path: Optional[str] = None,
        *,
        url: Optional[str] = None,
    ) -> Tuple[str, str]:
        """一次调用同时生成图片说明与轻量 OCR（节省 token/请求次数）。"""

        image_url = (url or "").strip() or None

        if not image_url:
            try:
                p = Path(str(file_path))
                image_url = VisionHelper._to_data_url(p.read_bytes(), p.suffix)
            except Exception as exc:
                logger.warning(f"读取图片失败，无法生成说明/OCR：{exc}")
                return "", ""

        messages = [
            {"role": "system", "content": "你是图片分析器，只能输出严格 JSON。"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "\n".join(
                            [
                                "请分析这张图片，并只输出严格 JSON：",
                                '{"caption":"<=20字中文说明","ocr_text":"图片中的文字(没有则空字符串)"}',
                                "要求：",
                                "- caption：<=20 字，中文，客观简短；",
                                "- ocr_text：只输出图片中文字，不解释不翻译；没有文字则输出空字符串；",
                            ]
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ]

        llm = get_task_llm("vision_ocr")
        content = await llm.chat_completion(messages, temperature=0.2)
        data = VisionHelper._extract_first_json_object(str(content or ""))
        if not isinstance(data, dict):
            return "", ""

        caption = str(data.get("caption") or "").strip()
        ocr_text = str(data.get("ocr_text") or "").strip()
        return caption[:20], ocr_text[:200]

    @staticmethod
    async def caption_image(file_path: Optional[str] = None, *, url: Optional[str] = None) -> str:
        """生成简短的图片说明(<=20字)

        这个方法的作用:
        - 调用视觉模型生成图片的简短描述
        - 支持本地文件和在线URL两种输入方式
        - 限制说明长度为20字以内
        - 失败时返回空字符串而不抛异常

        使用场景:
        - 用户发送图片消息时,生成说明补充上下文
        - MediaCache预处理图片内容
        - 日志记录和调试时显示图片内容

        为什么限制为20字?
        - 保持简洁,避免占用过多LLM上下文
        - 提供足够信息让LLM理解图片
        - 20字约等于40个字符,适合简短描述

        Args:
            file_path: 本地图片文件路径(可选)
                - 类型: 字符串或None
                - 示例: "/path/to/image.png"
                - 与url参数二选一

            url: 在线图片URL(可选,关键字参数)
                - 类型: 字符串或None
                - 示例: "https://example.com/cat.jpg"
                - 优先级高于file_path

        Returns:
            str: 图片说明文本(<=20字)
                - 成功: 返回简短描述,如"一只橘色的猫坐在沙发上"
                - 失败: 返回空字符串""

        Side Effects:
            - 调用vision_llm.chat_completion(异步LLM调用)
            - 读取本地文件(如果使用file_path)
            - 失败时输出警告日志

        Example:
            >>> # 使用本地文件
            >>> caption = await VisionHelper.caption_image(file_path="/tmp/cat.jpg")
            >>> print(caption)  # "一只橘色的猫"

            >>> # 使用在线URL
            >>> caption = await VisionHelper.caption_image(url="https://example.com/dog.png")
            >>> print(caption)  # "一只棕色的狗在草地上跑"
        """

        # ==================== 步骤1: 准备图片URL ====================

        # 优先使用url参数(如果提供)
        # (url or "").strip(): 去除首尾空格
        # or None: 如果是空字符串,转为None
        image_url = (url or "").strip() or None

        if not image_url:  # 如果没有提供url,尝试读取本地文件
            try:
                # Path(str(file_path)): 将文件路径转为Path对象
                # str(file_path): 确保是字符串类型
                p = Path(str(file_path))

                # p.read_bytes(): 读取文件的二进制内容
                # p.suffix: 获取文件后缀名(如".png")
                # _to_data_url(): 转为base64 data URL
                data_url = VisionHelper._to_data_url(p.read_bytes(), p.suffix)
                image_url = data_url  # 使用data URL

            except Exception as exc:
                # 读取文件失败(文件不存在、权限不足、路径错误等)
                # logger.warning(): 输出警告日志
                logger.warning(f"读取图片失败，无法生成说明：{exc}")
                return ""  # 返回空字符串

        # ==================== 步骤2: 构建视觉模型的请求消息 ====================

        # OpenAI Vision API的消息格式
        messages = [
            {
                "role": "user",  # 角色是用户
                "content": [  # content是一个列表,包含多个部分
                    {
                        "type": "text",  # 第一部分: 文本提示
                        "text": "用中文生成一句<=20字的图片说明。只输出说明文本。"
                        # 提示词说明:
                        # - "用中文": 确保输出中文
                        # - "<=20字": 限制长度
                        # - "只输出说明文本": 避免输出多余解释
                    },
                    {
                        "type": "image_url",  # 第二部分: 图片
                        "image_url": {"url": image_url}
                        # image_url可以是:
                        # - 在线URL: "https://example.com/image.jpg"
                        # - data URL: "data:image/png;base64,iVBORw0..."
                    },
                ],
            }
        ]

        # ==================== 步骤3: 调用视觉模型生成说明 ====================

        # await vision_llm.chat_completion(): 异步调用视觉LLM
        # - messages: 包含文本提示和图片的消息列表
        # - temperature=0.2: 低温度,生成更确定性的描述
        #   * 0.2比默认0.7更低,适合描述性任务
        #   * 避免过于创意的描述,保持客观
        llm = get_task_llm("vision_caption")
        content = await llm.chat_completion(messages, temperature=0.2)

        # ==================== 步骤4: 处理返回结果 ====================

        # (content or "").strip(): 去除首尾空格,None转为空字符串
        # [:20]: 截取前20个字符
        # - Python的字符串切片,超出长度不会报错
        # - 例如: "一只橘色的猫坐在沙发上看着窗外"[:20] → "一只橘色的猫坐在沙发上看着窗外"
        return (content or "").strip()[:20]

    @staticmethod
    async def ocr_image(file_path: Optional[str] = None, *, url: Optional[str] = None) -> str:
        """进行轻量级文字识别,输出简短的纯文本(<=200字)

        这个方法的作用:
        - 调用视觉模型识别图片中的文字
        - 支持本地文件和在线URL两种输入方式
        - 限制输出长度为200字以内
        - 失败时返回空字符串而不抛异常

        使用场景:
        - 表情包偷取时识别表情包上的文字
        - 理解包含文字的图片消息
        - MediaCache缓存OCR结果

        OCR vs Caption的区别:
        - OCR: 识别图片中的文字,原样输出
        - Caption: 描述图片内容,生成新文本

        为什么限制为200字?
        - 覆盖大部分表情包和图片文字场景
        - 避免超长文本占用过多存储和上下文
        - 200字约等于400个字符,足够详细

        Args:
            file_path: 本地图片文件路径(可选)
                - 类型: 字符串或None
                - 示例: "/path/to/meme.png"
                - 与url参数二选一

            url: 在线图片URL(可选,关键字参数)
                - 类型: 字符串或None
                - 示例: "https://example.com/meme.jpg"
                - 优先级高于file_path

        Returns:
            str: 识别出的文字内容(<=200字)
                - 成功: 返回图片中的文字,如"我裂开了"
                - 失败: 返回空字符串""
                - 无文字: 返回空字符串""

        Side Effects:
            - 调用vision_llm.chat_completion(异步LLM调用)
            - 读取本地文件(如果使用file_path)
            - 失败时输出警告日志

        Example:
            >>> # 识别表情包文字
            >>> text = await VisionHelper.ocr_image(file_path="/tmp/meme.png")
            >>> print(text)  # "我裂开了"

            >>> # 识别在线图片文字
            >>> text = await VisionHelper.ocr_image(url="https://example.com/sign.jpg")
            >>> print(text)  # "营业时间: 9:00-18:00"
        """

        # ==================== 步骤1: 准备图片URL ====================
        # 逻辑与caption_image完全相同

        image_url = (url or "").strip() or None

        if not image_url:  # 如果没有提供url,尝试读取本地文件
            try:
                p = Path(str(file_path))
                data_url = VisionHelper._to_data_url(p.read_bytes(), p.suffix)
                image_url = data_url
            except Exception as exc:
                logger.warning(f"读取图片失败，无法进行文字识别：{exc}")
                return ""  # 返回空字符串

        # ==================== 步骤2: 构建视觉模型的请求消息 ====================

        messages = [
            {
                "role": "user",  # 角色是用户
                "content": [  # content是一个列表
                    {
                        "type": "text",  # 文本提示
                        "text": "识别图片中的文字，输出纯文本（不解释），尽量短。"
                        # 提示词说明:
                        # - "识别图片中的文字": 明确任务是OCR
                        # - "输出纯文本": 只要文字内容,不要格式和解释
                        # - "不解释": 避免模型添加多余说明
                        # - "尽量短": 简洁输出,去除冗余
                    },
                    {
                        "type": "image_url",  # 图片
                        "image_url": {"url": image_url}
                    },
                ],
            }
        ]

        # ==================== 步骤3: 调用视觉模型进行OCR ====================

        # temperature=0.2: 低温度,生成确定性的识别结果
        # - OCR任务需要准确识别,不需要创造性
        llm = get_task_llm("vision_ocr")
        content = await llm.chat_completion(messages, temperature=0.2)

        # ==================== 步骤4: 处理返回结果 ====================

        # (content or "").strip(): 去除首尾空格
        # [:200]: 截取前200个字符
        # - 限制长度,避免超长文本
        return (content or "").strip()[:200]
