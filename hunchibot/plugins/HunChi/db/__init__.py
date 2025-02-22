from nonebot import get_driver

from tortoise import fields, models
from tortoise import Tortoise

class MessageModel(models.Model):
    message_id = fields.IntField(pk=True)
    group_id = fields.IntField()
    friend_id = fields.IntField()
    message = fields.TextField()
    message_time = fields.DatetimeField(auto_now_add=True)
    message_type = fields.TextField()
    message_sender_id = fields.TextField()
    message_sender_name = fields.TextField()

class GroupModel(models.Model):
    group_id = fields.IntField(pk=True)
    group_name = fields.TextField()

driver = get_driver()

# 初始化数据库
async def init():

    await Tortoise.init(
        db_url='sqlite://./HunChiData/db.sqlite3',
        modules={'models': ['hunchibot.plugins.HunChi.db']}
    )
    await Tortoise.generate_schemas()  # 自动迁移

@driver.on_startup
async def do_something():
    await init()