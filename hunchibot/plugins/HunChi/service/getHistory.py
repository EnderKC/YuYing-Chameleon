from nonebot.adapters.onebot.v11 import Bot, MessageEvent
from nonebot.log import logger
from hunchibot.plugins.HunChi.db import MessageModel


async def get_history(bot: Bot, event: MessageEvent) -> list:
    if event.message_type == "group":
        return await get_group_history(bot, event)
    elif event.message_type == "private":
        return await get_friend_history(bot, event)
    
# 获取群聊历史消息,通过group_id获取
async def get_history_by_group_id(group_id:str) -> list:
    messages = await MessageModel.filter(group_id=group_id).order_by('-message_time').limit(20).all()
    formatted_messages = [
        {
            "用户id": message.message_sender_id,
            "用户": message.message_sender_name,
            "消息": message.message
        }
        for message in reversed(messages)
    ]
    return formatted_messages


# 获取群聊历史消息
async def get_group_history(bot: Bot, event: MessageEvent) -> list:
    messages = await MessageModel.filter(group_id=event.group_id).order_by('-message_time').limit(20).all()
    # 构建包含发送者信息的消息列表
    formatted_messages = [
        {
            "用户id": message.message_sender_id,
            "用户": message.message_sender_name,
            "消息": message.message
        }
        for message in reversed(messages)
    ]
    
    return formatted_messages

# 获取好友历史消息
async def get_friend_history(bot: Bot, event: MessageEvent) -> list:
    messages = await MessageModel.filter(friend_id=event.user_id).order_by('-message_time').limit(20).all()
    # 构建包含发送者信息的消息列表
    formatted_messages = [
        {
            "sender_id": message.message_sender_id,
            "sender_name": message.message_sender_name,
            "message": message.message
        }
        for message in reversed(messages)
    ]
    
    return formatted_messages