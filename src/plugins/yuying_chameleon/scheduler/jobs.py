"""定时任务注册（apscheduler）。"""

from __future__ import annotations

from nonebot import require
require("nonebot_plugin_apscheduler")
from nonebot_plugin_apscheduler import scheduler
from ..memory.condenser import MemoryCondenser
from ..config import plugin_config

_inited = False

def init_scheduler():
    """初始化定时任务。"""

    global _inited
    if _inited:
        return
    _inited = True

    # 每日 03:00 记忆凝练
    scheduler.add_job(
        MemoryCondenser.run_daily_condenser,
        "cron",
        hour=plugin_config.yuying_memory_condense_hour,
        minute=0,
        id="daily_memory_condenser",
        replace_existing=True,
        coalesce=True,
        misfire_grace_time=3600,
    )
