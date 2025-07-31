from httpx import request
from nonebot.plugin import on_message
from nonebot.adapters.onebot.v11 import Bot, MessageEvent, MessageSegment, GroupMessageEvent, PrivateMessageEvent
from nonebot.log import logger
from nonebot.rule import Rule,to_me
from hunchibot.plugins.HunChi.service import save_message,message_response,get_history
from nonebot.exception import FinishedException
import random
import httpx
import base64

msg = on_message(priority=10)
msg_toMe = on_message(rule=to_me(), priority=9)


async def process_message(bot: Bot, event: MessageEvent, always_respond: bool = False):
    images = [seg for seg in event.message if seg.type == "image"]
    img_message = None
    
    if images:
        img_url = images[0].data["url"]
        async with httpx.AsyncClient() as client:
            response = await client.get(img_url)
            img_message = response.content
        # 图片的base64
        img_base64 = base64.b64encode(img_message).decode("utf-8")
    logger.info(f"保存消息...")
    await save_message(bot, event, img_base64 if images else None)
    if always_respond or random.random() < 0.18:
        await message_response(bot, event, img_base64 if images else None, toMe=always_respond)
        
@msg_toMe.handle()
async def handle_at(bot: Bot, event: MessageEvent):
    logger.info("检测到被@或私聊，正在处理...")
    await process_message(bot, event, always_respond=True)

@msg.handle()
async def handle_message(bot: Bot, event: MessageEvent):
    if isinstance(event, GroupMessageEvent):
        await process_message(bot, event)
    else:
        logger.info("检测到私聊消息，正在处理...")
        await process_message(bot, event)
    