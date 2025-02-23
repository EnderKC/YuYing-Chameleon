from sys import argv
from nonebot import require
from datetime import datetime, timedelta
from nonebot.log import logger
from hunchibot.plugins.HunChi.db import MessageModel

require("nonebot_plugin_apscheduler")

from nonebot_plugin_apscheduler import scheduler

# 基于装饰器的方式
@scheduler.scheduled_job("cron", hour="*/24", id="job_0", args=[1], kwargs={arg2: 2})
async def run_every_24_hour_to_clear_db(arg1: int, arg2: int):
    await MessageModel.filter(message_time__lt=datetime.now() - timedelta(days=1)).delete()
    logger.info("已清除1天前的消息")
