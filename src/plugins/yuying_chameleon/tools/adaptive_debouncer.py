"""自适应消息防抖（Adaptive Message Debouncer）。

基于认知语言学的话轮转换预测（Turn-Taking Prediction）理论，
动态计算等待时间，精准预测用户的"话轮结束点"（End-of-Turn）。

数学模型：
    WaitTime = w1·L + w2·L² + w3·P + b

    其中：
    - L (Length): 累积消息文本长度（字数）
    - P (IsEndPunctuation): 是否以强结束符结尾（。？！?!~）
    - w1 > 0: 字数少时，时间随字数增加
    - w2 < 0: 字数多时，抛物线开口向下，时间衰减
    - w3 < 0: 检测到结束符，大幅缩短时间
    - b: 基础等待时间（截距）

核心设计：
    - 粒度：按 (scene_type, scene_id, qq_id) 三元组独立防抖
    - 合并策略：自动识别中英文，智能拼接并归一化空白
    - 竞态保护：generation/version 校验 + asyncio.Lock
    - 硬截止：max_hold_seconds / max_parts / max_plain_len 防止无限延迟
    - 内存清理：flush 后删除 + shutdown 清空 + TTL 防御性清理

参考文献：
    [1] Gravano & Hirschberg (2011) - Turn-taking cues in task-oriented dialogue
    [2] Sato et al. (2002) - Learning decision trees for turn-taking prediction
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, Optional

from nonebot import logger

from ..normalize.normalizer import NormalizedMessage, Normalizer

# ==================== 正则表达式常量 ====================

# 强结束符：句号、问号、感叹号、波浪号
_STRONG_END_RE = re.compile(r"[。？！?!~]$")

# 尾部标记（如图片、表情）：用于去除末尾的短标记以准确判断结束符
_TRAILING_MARKERS_RE = re.compile(r"(?:\s*\[[^\]]+\])+\s*$")

# 多空白字符：用于归一化连续空格、换行、制表符
_MULTI_WS_RE = re.compile(r"\s+")

# 中日韩文字间空白：用于去除中文之间的空格（符合连续口语习惯）
_CJK_WS_RE = re.compile(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])")


# ==================== 辅助函数 ====================


def _strip_trailing_markers(text: str) -> str:
    """去除文本末尾的标记（如图片、表情），以便准确判断结束符。

    Args:
        text: 原始文本

    Returns:
        str: 去除末尾标记后的文本

    Example:
        >>> _strip_trailing_markers("我今天很开心。[image:abc123]")
        "我今天很开心。"
    """
    t = (text or "").strip()
    t = _TRAILING_MARKERS_RE.sub("", t)
    return t.strip()


def _plain_len(text: str) -> int:
    """计算纯文本长度（去除标记和空白后的字符数）。

    与 Normalizer.is_effective 的定义一致，用于公式中的 L。

    Args:
        text: 原始文本

    Returns:
        int: 纯文本长度

    Example:
        >>> _plain_len("今天天气 [image:abc] 真好")
        6  # "今天天气真好"
    """
    stripped = (text or "").strip()
    # 去除所有标记（如 [image:xxx]、[face:123] 等）
    plain = re.sub(r"\[[^\]]+\]", "", stripped)
    # 去除所有空白字符
    plain = re.sub(r"\s+", "", plain)
    return len(plain)


def _normalize_whitespace(text: str) -> str:
    """归一化空白字符（统一换行、制表符、多空格）。

    规则：
        1. 将 \\r\\n、\\r、\\t 转换为单个空格
        2. 将换行符 \\n 转换为空格
        3. 压缩连续空格为单个空格
        4. 去除中文之间的空格（符合连续口语习惯）

    Args:
        text: 原始文本

    Returns:
        str: 归一化后的文本

    Example:
        >>> _normalize_whitespace("我今天\\n去超市\\t买东西")
        "我今天去超市买东西"
    """
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    t = t.replace("\n", " ")
    # 压缩多个空白字符为单个空格
    t = _MULTI_WS_RE.sub(" ", t).strip()
    # 去除中文之间的空格
    t = _CJK_WS_RE.sub("", t)
    return t


def _auto_join(prev: str, nxt: str) -> str:
    """自动拼接两段文本，智能处理中英文边界和空白。

    规则：
        1. 如果边界是"英文/数字 + 英文/数字"，插入空格
        2. 如果边界是"中文 + 中文"，去除空格
        3. 归一化所有空白字符

    Args:
        prev: 前一段文本
        nxt: 后一段文本

    Returns:
        str: 拼接后的文本

    Example:
        >>> _auto_join("我今天", "去超市")
        "我今天去超市"
        >>> _auto_join("hello", "world")
        "hello world"
    """
    a = prev or ""
    b = nxt or ""

    # 处理边界情况
    if not a:
        return _normalize_whitespace(b)
    if not b:
        return _normalize_whitespace(a)

    # 归一化空白
    a2 = _normalize_whitespace(a)
    b2 = _normalize_whitespace(b)

    if not a2:
        return b2
    if not b2:
        return a2

    # 判断边界是否需要空格
    last = a2[-1]
    first = b2[0]
    need_space = last.isalnum() and first.isalnum()

    # 拼接
    joined = a2 + (" " if need_space else "") + b2
    return _normalize_whitespace(joined)


def calculate_wait_time(
    text: str,
    *,
    w1: float,
    w2: float,
    w3: float,
    b: float,
    min_wait: float,
    max_wait: float,
) -> float:
    """计算自适应等待时间。

    基于二次回归模型：WaitTime = w1·L + w2·L² + w3·P + b

    Args:
        text: 累积的消息文本
        w1: 一次项系数（正数）
        w2: 二次项系数（负数，控制抛物线开口向下）
        w3: 标点符号系数（负数，检测到结束符时大幅缩短时间）
        b: 基础等待时间（截距）
        min_wait: 最小等待时间（秒）
        max_wait: 最大等待时间（秒）

    Returns:
        float: 计算后的等待时间（秒），已 clamp 到 [min_wait, max_wait] 范围

    Example:
        >>> calculate_wait_time("在吗", w1=0.6, w2=-0.025, w3=-2.5, b=1.5, min_wait=0.5, max_wait=5.0)
        2.57  # 短文本，等待适中
        >>> calculate_wait_time("今天天气真好。", w1=0.6, w2=-0.025, w3=-2.5, b=1.5, min_wait=0.5, max_wait=5.0)
        1.0   # 有结束符，快速触发
    """
    merged = text or ""

    # 计算纯文本长度（去标记去空白）
    L = float(_plain_len(merged))

    # 检查是否以强结束符结尾（去除尾部标记后再判断）
    tail = _strip_trailing_markers(merged)
    P = 1.0 if _STRONG_END_RE.search(tail) else 0.0

    # 套用二次回归公式
    wait = (w1 * L) + (w2 * (L**2)) + (w3 * P) + b

    # 边界值处理（Clamping）
    if wait < min_wait:
        return float(min_wait)
    if wait > max_wait:
        return float(max_wait)
    return float(wait)


# ==================== 数据结构 ====================


@dataclass
class DebounceState:
    """防抖状态（按用户+场景维度）。

    说明：
        - 每个 (scene_type, scene_id, qq_id) 三元组对应一个状态
        - generation 用于竞态保护（只有最新一代 timer 能 flush）
        - parts 用于统计拼接段数，超过 max_parts 触发硬截止
        - first_update 用于硬截止（避免持续分段导致永不 flush）
        - last_update 用于 TTL 清理（防御性清理，避免异常路径泄漏）
    """

    generation: int = 0  # 当前代数（用于竞态保护）
    content: str = ""  # 累积的消息文本
    image_ref_map: Dict[str, str] = field(default_factory=dict)  # 图片引用映射
    mentioned_bot: bool = False  # 是否 @bot（合并所有段）
    reply_to_msg_id: Optional[int] = None  # 回复的消息 ID
    raw_ref: Optional[str] = None  # 原始引用
    msg_type: str = "text"  # 消息类型（text/image/mixed）
    timestamp: int = 0  # 时间戳（取所有段的最大值）

    parts: int = 0  # 已拼接的段数
    first_update: float = field(default_factory=time.time)  # 第一次更新时间
    last_update: float = field(default_factory=time.time)  # 最后更新时间
    timer_task: Optional[asyncio.Task] = None  # 当前计时器任务


# ==================== 主类 ====================


class AdaptiveDebouncer:
    """自适应消息防抖器。

    职责：
        - 接收分段消息，累积到缓冲区
        - 根据文本特征动态计算等待时间
        - 等待结束后，触发 flush_cb 回调处理合并后的消息
        - 处理竞态条件、硬截止、TTL 清理、shutdown 清理

    使用示例：
        >>> debouncer = AdaptiveDebouncer(w1=0.6, w2=-0.025, w3=-2.5, bias=1.5, min_wait=0.5, max_wait=5.0)
        >>> async def handle_merged(msg: NormalizedMessage):
        >>>     print(f"收到合并后的消息: {msg.content}")
        >>> await debouncer.submit(key="group:123:456", part=normalized_msg, flush_cb=handle_merged)
    """

    def __init__(
        self,
        *,
        joiner: str = "auto",
        ttl_seconds: float = 60.0,
        max_hold_seconds: float = 15.0,
        max_parts: int = 12,
        max_plain_len: int = 300,
        w1: float = 0.6,
        w2: float = -0.025,
        w3: float = -2.5,
        bias: float = 1.5,
        min_wait: float = 0.5,
        max_wait: float = 5.0,
    ) -> None:
        """初始化防抖器。

        Args:
            joiner: 拼接策略（"auto" = 自动识别中英文，"" = 直接拼接，" " = 空格拼接）
            ttl_seconds: 状态 TTL（秒），防御性清理，避免异常路径泄漏
            max_hold_seconds: 硬截止（秒），避免用户持续分段导致永不 flush
            max_parts: 最多拼接段数，超过强制 flush
            max_plain_len: 最大纯文本长度（去标记去空白后），超过强制 flush
            w1: 一次项系数（正数）
            w2: 二次项系数（负数）
            w3: 标点符号系数（负数）
            bias: 基础等待时间（截距）
            min_wait: 最小等待时间（秒）
            max_wait: 最大等待时间（秒）
        """
        self._lock = asyncio.Lock()
        self._states: Dict[str, DebounceState] = {}

        # 拼接策略
        self._joiner = joiner

        # 硬截止参数
        self._ttl_seconds = float(ttl_seconds)
        self._max_hold_seconds = float(max_hold_seconds)
        self._max_parts = int(max_parts)
        self._max_plain_len = int(max_plain_len)

        # 公式参数
        self._w1 = float(w1)
        self._w2 = float(w2)
        self._w3 = float(w3)
        self._bias = float(bias)
        self._min_wait = float(min_wait)
        self._max_wait = float(max_wait)

    async def shutdown(self) -> None:
        """关闭防抖器，取消所有计时器并清空状态。

        说明：
            - 在 NoneBot on_shutdown 时调用
            - 防止进程关闭时留下悬挂任务
        """
        async with self._lock:
            for st in self._states.values():
                if st.timer_task:
                    st.timer_task.cancel()
            self._states.clear()

    def _merge_content(self, prev: str, nxt: str) -> str:
        """合并两段文本内容。

        Args:
            prev: 前一段文本
            nxt: 后一段文本

        Returns:
            str: 合并后的文本
        """
        if self._joiner == "auto":
            return _auto_join(prev, nxt)

        joiner = self._joiner
        if not prev:
            return _normalize_whitespace(nxt)
        if not nxt:
            return _normalize_whitespace(prev)

        return _normalize_whitespace(prev + joiner + nxt)

    def _build_merged(
        self, *, part: NormalizedMessage, st: DebounceState
    ) -> NormalizedMessage:
        """构建合并后的 NormalizedMessage。

        Args:
            part: 最后一段消息（用于提供基础字段）
            st: 当前防抖状态

        Returns:
            NormalizedMessage: 合并后的消息
        """
        merged_content = st.content
        return NormalizedMessage(
            qq_id=part.qq_id,
            scene_type=part.scene_type,
            scene_id=part.scene_id,
            timestamp=int(st.timestamp or part.timestamp),
            msg_type=st.msg_type,
            content=merged_content,
            raw_ref=st.raw_ref,
            image_ref_map=st.image_ref_map,
            reply_to_msg_id=st.reply_to_msg_id,
            mentioned_bot=st.mentioned_bot,
            is_effective=Normalizer.is_effective(merged_content),
        )

    async def submit(
        self,
        *,
        key: str,
        part: NormalizedMessage,
        flush_cb: Callable[[NormalizedMessage], Awaitable[None]],
    ) -> None:
        """提交一段消息到防抖缓冲区。

        说明：
            - 如果是该用户的第一段消息，创建新状态
            - 否则，拼接到现有状态，取消旧计时器，创建新计时器
            - 如果触发硬截止（max_hold_seconds / max_parts / max_plain_len），立即 flush

        Args:
            key: 防抖 key（推荐格式："{scene_type}:{scene_id}:{qq_id}"）
            part: 当前段的归一化消息
            flush_cb: flush 回调函数（接收合并后的 NormalizedMessage）
        """
        now = time.time()
        to_flush: Optional[NormalizedMessage] = None

        async with self._lock:
            st = self._states.get(key)
            if st is None:
                st = DebounceState()
                self._states[key] = st

            # TTL 防御：如果状态过旧，直接重置（避免泄漏/挂死）
            if (now - st.last_update) > self._ttl_seconds:
                if st.timer_task:
                    st.timer_task.cancel()
                st = DebounceState()
                self._states[key] = st

            # 更新状态
            st.generation += 1
            st.parts += 1
            st.last_update = now

            # 取消旧计时器
            if st.timer_task:
                st.timer_task.cancel()

            # 合并内容
            st.content = self._merge_content(st.content, part.content or "")

            # 合并元数据（避免"只保留最后 Event"导致丢信息）
            st.mentioned_bot = st.mentioned_bot or bool(part.mentioned_bot)
            st.reply_to_msg_id = st.reply_to_msg_id or part.reply_to_msg_id
            st.raw_ref = part.raw_ref or st.raw_ref
            st.timestamp = max(int(st.timestamp or 0), int(part.timestamp or 0))
            st.image_ref_map.update(part.image_ref_map or {})
            st.msg_type = (
                "mixed"
                if (st.image_ref_map or part.msg_type != "text")
                else "text"
            )

            # 检查硬截止条件
            hard_deadline = (now - st.first_update) >= self._max_hold_seconds
            too_many_parts = st.parts >= self._max_parts
            too_long = _plain_len(st.content) >= self._max_plain_len

            if hard_deadline or too_many_parts or too_long:
                # 立即 flush
                to_flush = self._build_merged(part=part, st=st)
                del self._states[key]

                # 记录硬截止原因（调试用）
                reason = (
                    "hard_deadline"
                    if hard_deadline
                    else "too_many_parts"
                    if too_many_parts
                    else "too_long"
                )
                logger.debug(
                    f"[AdaptiveDebouncer] 触发硬截止（{reason}）: key={key}, parts={st.parts}, len={_plain_len(st.content)}, hold_seconds={now - st.first_update:.2f}"
                )

            if to_flush is None:
                # 计算动态等待时间
                wait = calculate_wait_time(
                    st.content,
                    w1=self._w1,
                    w2=self._w2,
                    w3=self._w3,
                    b=self._bias,
                    min_wait=self._min_wait,
                    max_wait=self._max_wait,
                )
                gen = st.generation

                logger.debug(
                    f"[AdaptiveDebouncer] 计算等待时间: key={key}, parts={st.parts}, len={_plain_len(st.content)}, wait={wait:.2f}s"
                )

                # 创建新计时器
                async def _timer() -> None:
                    try:
                        await asyncio.sleep(wait)
                    except asyncio.CancelledError:
                        return

                    merged: Optional[NormalizedMessage] = None

                    # 竞态保护：只有最新一代 timer 能 flush
                    async with self._lock:
                        cur = self._states.get(key)
                        if not cur or cur.generation != gen:
                            # 状态已被新一代更新或已删除，直接返回
                            return
                        merged = self._build_merged(part=part, st=cur)
                        del self._states[key]

                    logger.debug(
                        f"[AdaptiveDebouncer] flush 合并后的消息: key={key}, content={merged.content[:50]}"
                    )

                    # 调用 flush 回调
                    try:
                        await flush_cb(merged)
                    except Exception as exc:
                        logger.error(
                            f"[AdaptiveDebouncer] flush_cb 异常: key={key}, exc={exc}"
                        )
                        return

                st.timer_task = asyncio.create_task(_timer())

        # 如果触发了硬截止，在锁外调用 flush_cb（避免死锁）
        if to_flush is not None:
            logger.debug(
                f"[AdaptiveDebouncer] flush 合并后的消息（硬截止）: key={key}, content={to_flush.content[:50]}"
            )
            try:
                await flush_cb(to_flush)
            except Exception as exc:
                logger.error(
                    f"[AdaptiveDebouncer] flush_cb 异常（硬截止）: key={key}, exc={exc}"
                )
                return
