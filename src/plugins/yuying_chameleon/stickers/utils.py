"""表情包处理工具模块 - OCR文本归一化与去重指纹生成

这个模块的作用:
1. 提供OCR文本的归一化处理函数
2. 用于表情包去重(fingerprint聚合)
3. 消除格式差异,识别相同内容的表情包

Fingerprint聚合原理(新手必读):
- 问题: 同一表情包可能有多个版本(不同压缩、尺寸、质量)
- 解决: 通过OCR文本生成指纹(fingerprint),相同文本=相同表情包
- 流程: 图片 → OCR识别文字 → 归一化处理 → 生成指纹 → 去重
- 好处: 避免存储重复表情包,节省存储空间

为什么需要归一化?
- OCR识别有差异: "Hello World" vs "hello  world" vs "HELLO WORLD"
- 引号符号不同: "你好" vs '你好' vs "你好" vs „你好"
- 空格差异: "今天  天气" vs "今天 天气"
- 归一化后: 都变成统一格式,便于比对去重

使用场景:
- 表情包学习(stealer): 识别已有表情包,避免重复学习
- 表情包注册(registry): 生成表情包的唯一标识
- 表情包检索: 基于文本内容搜索表情包

Example:
```python
from .stickers.utils import normalize_ocr_text

# 示例1: 大小写和空格归一化
text1 = "Hello  World"
text2 = "HELLO   WORLD"
print(normalize_ocr_text(text1))  # "hello world"
print(normalize_ocr_text(text2))  # "hello world"
# 结果相同,识别为同一表情包

# 示例2: 引号和特殊符号去除
text3 = '"今天天气真好"'
text4 = "'今天天气真好'"
print(normalize_ocr_text(text3))  # "今天天气真好"
print(normalize_ocr_text(text4))  # "今天天气真好"
# 结果相同,识别为同一表情包

# 示例3: 长度截断
text5 = "A" * 300  # 300个字符
result = normalize_ocr_text(text5)
print(len(result))  # 200 (截断到200字符)
```
"""

from __future__ import annotations

import re  # Python标准库,正则表达式模块


def normalize_ocr_text(text: str) -> str:
    """归一化OCR文本,用于表情包fingerprint聚合(去重)

    这个函数的作用:
    - 将OCR识别的文本标准化为统一格式
    - 消除大小写、空格、引号等格式差异
    - 生成表情包的去重指纹(fingerprint)
    - 用于判断两个表情包是否为同一个(基于文本内容)

    归一化步骤(按顺序):
    1. 去除首尾空格并转小写
    2. 合并多个连续空格为单个空格
    3. 删除所有引号类字符
    4. 截断到200字符以内

    为什么这样处理?
    - 转小写: "Hello" 和 "hello" 应该被认为是相同的
    - 合并空格: OCR可能识别出多余空格,需要规范化
    - 删除引号: 不同OCR引擎可能识别出不同引号样式
    - 截断长度: 限制指纹长度,提高比对效率

    Args:
        text: OCR识别的原始文本
            - 类型: 字符串
            - 可能为None或空字符串
            - 可能包含各种格式差异
            - 示例: "  Hello   World  ", '"你好世界"', 'HELLO WORLD'

    Returns:
        str: 归一化后的文本
            - 全部小写
            - 单个空格分隔
            - 无引号
            - 最多200字符
            - 示例: "hello world", "你好世界", "hello world"

    处理细节:
        支持的引号类型:
        - 双引号: " (ASCII 34)
        - 单引号: ' (ASCII 39)
        - 中文双引号: " " (U+201C, U+201D)
        - 中文单引号: ' ' (U+2018, U+2019)
        - 重音符: ` ´ (grave, acute)

    Example:
        >>> normalize_ocr_text("  Hello   World  ")
        'hello world'

        >>> normalize_ocr_text('"今天天气真好"')
        '今天天气真好'

        >>> normalize_ocr_text("HELLO    WORLD")
        'hello world'

        >>> normalize_ocr_text(None)
        ''

        >>> normalize_ocr_text("   ")
        ''

        >>> normalize_ocr_text("A" * 300)
        'aaa...aaa'  # 200个字符
    """

    # ==================== 步骤1: 去除首尾空格并转小写 ====================

    # (text or ""): 如果text是None,转为空字符串
    # .strip(): 去除首尾的空格、换行、制表符
    # .lower(): 转换为全小写
    s = (text or "").strip().lower()

    # ==================== 步骤2: 空值检查 ====================

    # not s: 如果处理后是空字符串
    # - 可能原因: 输入为None、空字符串、纯空格
    if not s:
        return ""  # 返回空字符串

    # ==================== 步骤3: 合并多个连续空格为单个空格 ====================

    # re.sub(pattern, replacement, string): 正则替换
    # r"\s+": 匹配一个或多个空白字符(空格、制表符、换行等)
    # " ": 替换为单个空格
    # 例如: "hello   world" → "hello world"
    s = re.sub(r"\s+", " ", s)

    # ==================== 步骤4: 删除所有引号类字符 ====================

    # re.sub(pattern, replacement, string): 正则替换
    # r"[\"'""''`´]": 字符类,匹配任意一种引号
    # - \": 双引号(需转义)
    # - ': 单引号
    # - "": 中文双引号(左右)
    # - '': 中文单引号(左右)
    # - `: 重音符(grave accent)
    # - ´: 尖音符(acute accent)
    # "": 替换为空字符串(删除)
    # 例如: '"hello"' → 'hello'
    s = re.sub(r"[\"'""''`´]", "", s)

    # ==================== 步骤5: 截断到200字符 ====================

    # s[:200]: 切片,取前200个字符
    # - 原因: 限制指纹长度,避免过长文本影响性能
    # - 200字符足够区分大多数表情包
    # - 如果原文本<200字符,切片无影响
    return s[:200]
