from email import message
from nonebot.adapters import Bot
from typing import Optional, Dict, Any
from nonebot.log import logger
from nonebot.adapters.onebot.v11 import MessageEvent
from hunchibot.plugins.HunChi.db import MessageModel
from hunchibot.plugins.HunChi.db import MessageContentType




@Bot.on_called_api
async def handle_api_result(
    bot: Bot, exception: Optional[Exception], api: str, data: Dict[str, Any], result: Any
):
    logger.info(f"data: {data}")
    if not exception and api == "send_msg":
        await MessageModel.create(
            message_id=result['message_id'],
            message_type=data['message_type'],
            message=data['message'].extract_plain_text(),
            message_content_type=MessageContentType.TEXT if (data['message']).get('type') == "text" else MessageContentType.IMAGE,
            message_sender_id='ME',
            message_sender_name='ME',
            group_id=data['group_id'] if data['message_type'] == "group" else 0,
            friend_id=data['user_id'] if data['message_type'] == "private" else 0
        )
