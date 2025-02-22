from nonebot.plugin import on_message
from nonebot.adapters.onebot.v11 import Bot, MessageEvent
from nonebot.log import logger
from hunchibot.plugins.HunChi.service import save_message,message_response,get_history
import random

message = on_message(priority=10)


@message.handle()
async def handle_message(bot: Bot, event: MessageEvent):
    await save_message(bot,event)
    if random.random() < 0.3:
        await message_response(bot,event)