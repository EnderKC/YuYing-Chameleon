from email import message
from nonebot.adapters import Bot
from typing import Optional, Dict, Any, Self
from nonebot.log import logger
from nonebot.adapters.onebot.v11 import MessageEvent, MessageSegment
from hunchibot.plugins.HunChi.db import MessageModel
from hunchibot.plugins.HunChi.db import MessageContentType
import httpx
import base64
import re




@Bot.on_called_api
async def handle_api_result(
    bot: Bot, exception: Optional[Exception], api: str, data: Dict[str, Any], result: Any
):
    message_content_type = MessageContentType.IMAGE if (str(data['message'])).startswith("[CQ:image") else MessageContentType.TEXT
    if message_content_type == MessageContentType.IMAGE:
        url_pattern = r"'file':\s*'(https?://[^']+)'"
        urls = re.findall(url_pattern, str(data['message']))
        async with httpx.AsyncClient() as client:
            response = await client.get(urls[0])
            logger.info(f"response: {response}")
            image_data = response.content
            image_base64 = f'data:image/jpeg;base64,{base64.b64encode(image_data).decode("utf-8")}'
    if not exception and api == "send_msg":
        await MessageModel.create(
            message_id=result['message_id'],
            message_type=data['message_type'],
            message=image_base64 if message_content_type == MessageContentType.IMAGE else data['message'].extract_plain_text(),
            message_content_type=message_content_type,
            message_sender_id='ME',
            message_sender_name='ME',
            group_id=data['group_id'] if data['message_type'] == "group" else 0,
            friend_id=data['user_id'] if data['message_type'] == "private" else 0
        )
