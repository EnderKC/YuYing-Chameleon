import asyncio
import random
from sys import argv
from nonebot import require
from datetime import datetime, timedelta
from nonebot.log import logger
from hunchibot.plugins.HunChi.db import MessageModel
import nonebot
from hunchibot.plugins.HunChi.db import GroupModel,MessageModel
from hunchibot.plugins.HunChi.service.messageResponse import active_group
require("nonebot_plugin_apscheduler")

from nonebot_plugin_apscheduler import scheduler

config = nonebot.get_driver().config


# @Description: 每天0点清除1天前的消息
@scheduler.scheduled_job("cron", hour="0", minute="0", id="job_0")
async def run_every_24_hour_to_clear_db():
    await MessageModel.filter(message_time__lt=datetime.now() - timedelta(days=1)).delete()
    logger.info("已清除1天前的消息")


# @Description: 自动活跃群聊
@scheduler.scheduled_job("cron", hour="8-22", minute="0/25", id="job_1")
async def run_every_24_hour_to_active_group():
    logger.info("开始活跃群聊")
    bot = nonebot.get_bot()
    groups = await GroupModel.all()
    for group in groups:
        if random.random() < 0.7:
            logger.info(f"群{group.group_id}不活跃")
            continue
        last_message = await MessageModel.filter(group_id=group.group_id).order_by('-message_time').first()
        # 使用 datetime.now(tz=last_message.message_time.tzinfo) 来获取相同时区的当前时间
        current_time = datetime.now(tz=last_message.message_time.tzinfo) if last_message else datetime.now()
        # 如果群聊中没有消息，或者消息时间超过30分钟，则活跃群聊
        if not last_message or last_message.message_time < current_time - timedelta(minutes=30):
            await active_group(bot, group.group_id, last_message.message_time)
        await asyncio.sleep(3)


