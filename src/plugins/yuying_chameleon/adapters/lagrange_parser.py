"""Lagrange/OneBot 事件解析器。

将 OneBot v11 的事件解析为内部统一的 `InboundMessage` 结构，便于后续归一化与存储。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from nonebot.adapters.onebot.v11 import Event, GroupMessageEvent, PrivateMessageEvent


@dataclass
class InboundMessage:
    """内部统一的入站消息表示。"""

    qq_id: str
    scene_type: str
    scene_id: str
    timestamp: int
    onebot_message_id: Optional[int]
    msg_type: str
    content: str  # 原始内容或拼接后的段表示
    raw_ref: Optional[str] = None
    reply_to_msg_id: Optional[int] = None
    mentioned_bot: bool = False
    original_event: Optional[Event] = None

class LagrangeParser:
    @staticmethod
    def parse_event(event: Event) -> Optional[InboundMessage]:
        """将 OneBot 事件解析为 InboundMessage。

        参数：
            event: OneBot v11 事件。

        返回：
            Optional[InboundMessage]: 非群聊/私聊消息事件返回 None。
        """

        if not isinstance(event, (GroupMessageEvent, PrivateMessageEvent)):
            return None

        qq_id = str(event.user_id)
        timestamp = int(event.time)

        if isinstance(event, GroupMessageEvent):
            scene_type = "group"
            scene_id = str(event.group_id)
        else:
            scene_type = "private"
            scene_id = str(event.user_id)

        onebot_message_id: Optional[int] = None
        try:
            onebot_message_id = int(getattr(event, "message_id", None) or 0) or None
        except Exception:
            onebot_message_id = None

        # 基础段解析：将图片/表情等段转换为短标记
        msg_type = "text"
        content_parts = []
        raw_ref = None
        mentioned_bot = event.is_tome()
        reply_to_msg_id = None

        if event.reply:
            reply_to_msg_id = event.reply.message_id

        for seg in event.message:
            if seg.type == "text":
                content_parts.append(seg.data["text"])
            elif seg.type == "image":
                msg_type = "image" if msg_type == "text" else "mixed"
                # 优先使用文件标识，否则回退到链接
                seg_ref = seg.data.get("file") or seg.data.get("file_id") or seg.data.get("url")
                if raw_ref is None and seg_ref:
                    raw_ref = seg_ref
                content_parts.append(f"[image:{seg_ref or 'unknown'}]")
            elif seg.type == "face":
                content_parts.append(f"[face:{seg.data.get('id')}]")
            elif seg.type == "at":
                content_parts.append(f"@{seg.data.get('qq')}")
            # 可按需扩展更多段类型

        content = "".join(content_parts)

        return InboundMessage(
            qq_id=qq_id,
            scene_type=scene_type,
            scene_id=scene_id,
            timestamp=timestamp,
            onebot_message_id=onebot_message_id,
            msg_type=msg_type,
            content=content,
            raw_ref=raw_ref,
            reply_to_msg_id=reply_to_msg_id,
            mentioned_bot=mentioned_bot,
            original_event=event
        )
