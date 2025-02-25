from nonebot.plugin import on_message
from nonebot.adapters.onebot.v11 import Bot, MessageEvent, MessageSegment, GroupMessageEvent
from nonebot.log import logger
from nonebot.rule import Rule,to_me
from hunchibot.plugins.HunChi.service import save_message,message_response,get_history,analyze_img
import random

message = on_message(priority=10)

msg_handler = on_message(rule=to_me(), priority=9,block=True)

@msg_handler.handle()
async def handle_at(bot: Bot, event: GroupMessageEvent):
    logger.info("检测到被@，正在处理...")
    images = [seg for seg in event.message if seg.type == "image"]
    
    if images:
        # 获取第一张图片的URL并分析
        img_url = images[0].data["url"]
        img_message = await analyze_img(img_url)
        await save_message(bot, event, img_message)
    else:
        await save_message(bot, event)
    await message_response(bot, event,toMe=True)

@message.handle()
async def handle_message(bot: Bot, event: MessageEvent):
    # 提取所有图片消息段
    images = [seg for seg in event.message if seg.type == "image"]
    
    if images:
        # 获取第一张图片的URL并分析
        img_url = images[0].data["url"]
        img_message = await analyze_img(img_url)
        await save_message(bot, event, img_message)
    else:
        await save_message(bot, event)

    # 随机决定是否响应
    if random.random() < 0.2:
        await message_response(bot, event, img_message if images else None)
    