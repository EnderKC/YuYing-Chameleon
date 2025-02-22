from tortoise import Tortoise
from nonebot.adapters.onebot.v11 import Message, MessageEvent, Bot
from hunchibot.plugins.HunChi.db import MessageModel, GroupModel
from nonebot.log import logger

# @Description: 保存消息
# @Param: message: Message
# @Return: None
async def save_message(bot: Bot,event: MessageEvent) -> None:
    logger.info(event.message_type)
    if event.message_type == "group":
        await check_group_exist(bot,event)
        await save_message_to_db(event,group_id= event.group_id)
    elif event.message_type == "private":
        await save_message_to_db(event,friend_id=event.user_id)

# @Description: 检查群组是否存在，若不存在创建
# @Param: event: MessageEvent
# @Return: bool
async def check_group_exist(bot: Bot,event: MessageEvent) -> bool:
    group = await GroupModel.filter(group_id=event.group_id).first()
    if group:
        logger.info(f"群组 {event.group_id} 已存在")
    else:
        group_info = await bot.call_api("get_group_info", group_id=event.group_id)
        group = GroupModel(
            group_id=event.group_id,
            group_name=group_info["group_name"]
        )
        await group.save()
        logger.info(f"群组 {event.group_id} 不存在，已创建")

# @Description: 保存消息到数据库
# @Param: message: Message
# @Return: None
async def save_message_to_db(event: MessageEvent,group_id: int = 0,friend_id: int = 0) -> None:
    message_id = event.message_id
    message = event.message
    message_type = event.message_type
    message_sender_id = event.user_id
    message_sender_name = event.sender.nickname
    message_saver = MessageModel(
        message_id=message_id,
        group_id=group_id,
        friend_id=friend_id,
        message=message,
        message_type=message_type,
        message_sender_id=message_sender_id,
        message_sender_name=message_sender_name
    )
    logger.info(message_saver)
    await message_saver.save()

