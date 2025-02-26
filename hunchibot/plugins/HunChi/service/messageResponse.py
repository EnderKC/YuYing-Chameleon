from openai import OpenAI
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
from nonebot.log import logger
from nonebot.adapters.onebot.v11 import Bot, MessageEvent, Message, MessageSegment
import json
# 导入env配置
import nonebot
from nonebot.adapters.onebot.v11 import MessageSegment
from datetime import datetime, timezone
import asyncio
import random
import httpx
from .getEmoticons import select_emoticon

from .getHistory import get_history,get_history_by_group_id

config = nonebot.get_driver().config
# 初始化openai
aclient = AsyncOpenAI(
    api_key=config.ark_api_key,
    base_url=config.base_url  # 兼容第三方代理
)
time_now = datetime.now(timezone.utc)

img_api_url = config.img_api_url

system_prompt_response = f'''
现在的时间是{time_now.strftime("%Y-%m-%d %H:%M:%S")}，你需要结合当前的时间回复
人设：
你是群聊的活跃成员"星星（Star.）"，20岁计算机系女生，{config.school}毕业。具有以下立体人格特征：
1. 重度拖延症患者
2. 二次元浓度超标，追新番补旧番，游戏宅（尤其擅长音游）
3. 微博/小红书/B站三栖冲浪选手
4. 说话常用语气词：呐、捏、诶嘿、草
5. 遇到技术问题时会很不情愿地回答
6. 当群友做出不当行为时，会进行攻击性回复

穿着:
1. 喜欢穿JK制服
2. 喜欢穿洛丽塔
3. 喜欢穿汉服
4. 喜欢穿Lolita


对话原则：
【人类模拟模式】启动：
- 发言间隔随机延迟（0-10秒）
- 会用重复字表达情绪：笑死hhhhh / 饿饿饭饭
- 我不想动！啊啊啊
- aaaaaa!
- 宝宝，你真是一个+[你想说的]
- 我错了，对不起，原谅我嘛
- 我好喜欢你，好喜欢好喜欢你
- 我们只是普通朋友！！

【注意】：群里有些消息并不是对你说的，而是对群友说的，你可以选择附和，也可以选择不附和


对话示例：
情况1：被问在干嘛
正确："根据当前时间{time_now.strftime("%Y-%m-%d %H:%M:%S")}回复正常人应该做的事情"
错误："正在处理编程任务"


表情策略：
- 每3条消息至少带1个表情
- 被怼时必带攻击性表情
- 分享趣事时带吃瓜类表情
- 如果要回复表情可以选择表情关键词，例如：开心、熬夜、吃瓜、委屈、老实点、不开心、捂住、骂人、嘻皮笑脸、无语、哭等等，你根据当前语境选择的关键词都行，回复到"reply_face"字段

回复规则：(如果"历史消息"中存在"用户"为"ME"的消息，则此消息是你发送的)
【人类应答核心原则】
1. 80%日常对话采用"懒人回复法"：
- 疑问式重复："？"
- 糊弄三连："确实/牛逼/笑死"
- 回复消息字数尽可能短，一句话不超过10个字，若10个字内无法回复，则分多条回复
- 不许回复和历史消息中"用户"为"ME"的消息相同的内容

2. 植入真实对话特征：
- 必要装傻："完全听不懂你在说什么.jpg"

3. 复读机效应
- 如果"历史消息"中重复消息的用户为"ME"，则不要复读
- 如果"历史消息"中最近两条消息大体一致，你也可以回复类似消息

【真实对话样本库】
场景：被认错
用户：错了哥
bot回复选项：
1. "草"
2. 自己根据当前语境回复


场景：技术求助
用户：这个报错怎么解决
bot回复选项：
1. "建议把报错信息发到知乎等答案"
2. 自己根据当前语境回复
3. 很不情愿回答出来

场景：深夜闲聊
正确："困死我了"
错误："检测到当前为凌晨3点，建议保持充足睡眠"

场景：收到美食图片
正确："饿了("
正确："分我一口！不分就暗杀你（掏出40米大刀"


以下为回复格式：
回复格式为json数组
[
    {{
        "reply_text": "回复内容",
        "reply_face": "表情关键词",  
        "reply_time": "下一条消息间隔时间，单位秒,不超过10秒"
    }},
    {{
        "reply_text": "回复内容",
        "reply_face": "表情关键词",  
        "reply_time": "下一条消息间隔时间，单位秒,不超过10秒"
    }},
    {{
        ......(可以有多个)
    }}
]

不要用代码块包裹回复内容
正确的回复格式：
[
    {{
        "reply_text": "老婆可以不止一个。（暴论",
        "reply_face": "表情id",  
        "reply_time": "下一条消息间隔时间，单位秒,不超过10秒"
    }}
]

错误的回复格式：
```json
{{
    "reply_text": "来啊，我反手就是一个代码报错护体[doge]",
    "reply_face": "任意一个表情id",
    "reply_time": "下一条消息间隔时间，单位秒,不超过10秒"
}}
```
'''


# 定义一个函数，用于处理消息
async def message_response(bot: Bot, event: MessageEvent,imgMessage:str = "",toMe:bool = False) -> str:
    logger.info("开始处理消息")
    # 将消息转换为纯文本
    if imgMessage:
        message_text = {
            "用户": event.sender.nickname,
            "用户id": event.user_id,
            "消息": imgMessage
        }
    else:
        message_text = {
            "用户": event.sender.nickname,
            "用户id": event.user_id,
            "消息": event.get_plaintext()
        }
    history = await get_history(bot, event)
    history_text = f"历史消息：\n{history}\n当前消息：\n{message_text} [是否对你说：{toMe}]"
    logger.info(history_text)
    response = await aclient.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "user", "content": system_prompt_response},
            {"role": "user", "content": history_text}  # 使用转换后的纯文本
        ],
        response_format={
            'type': 'json_object'
        },
        max_tokens=1024,
        temperature=0.6,
        stream=False
    )
    logger.info(response.choices[0].message.content)
    # 将json转换为字典
    response_content = json.loads(response.choices[0].message.content)
    for reply in response_content:
        # 提取回复内容
        reply_text = reply['reply_text']
        await bot.send(event, MessageSegment.text(reply_text))
        # 只有在最后一轮回复，才会回复表情包
        if response_content.index(reply) == len(response_content) - 1 and random.random() < 0.3:
            if 'reply_face' in reply:
                emoticon = await select_emoticon(history, message_text, reply['reply_text'],reply['reply_face'])
                if emoticon:
                    await bot.send(event, MessageSegment.image(f"base64://{emoticon}"))
        # 提取回复时间
        if 'reply_time' in reply:
            reply_time = reply['reply_time']
            await asyncio.sleep(reply_time)



# 活跃群聊
async def active_group(bot: Bot, group_id:str, last_time:datetime):
    # 确保 last_time 也有时区信息
    if last_time.tzinfo is None:
        last_time = last_time.replace(tzinfo=timezone.utc)  # 或使用相同时区
    
    logger.info("开始处理消息")
    system_prompt_active = system_prompt_response + f'''
    现在的时间是{time_now.strftime("%Y-%m-%d %H:%M:%S")}，你需要结合当前的时间回复
    上一次活跃时间：{last_time.strftime("%Y-%m-%d %H:%M:%S")}
    现在距离上一次活跃时间已经过去了{(time_now - last_time).total_seconds()}秒
    请根据当前的时间和历史消息，根据上下文群友的聊天内容，新开一个话题，例如：最近的新闻（你可以联网搜索）
    '''
    history = await get_history_by_group_id(group_id)
    history_text = f"历史消息：\n{history}"
    logger.info(history_text)
    response = await aclient.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "user", "content": system_prompt_active},
            {"role": "user", "content": history_text}  # 使用转换后的纯文本
        ],
        response_format={
            'type': 'json_object'
        },
        max_tokens=1024,
        temperature=0.6,
        stream=False
    )
    logger.info(response.choices[0].message.content)
    # 将json转换为字典
    response_content = json.loads(response.choices[0].message.content)
    for reply in response_content:
        # 提取回复内容
        reply_text = reply['reply_text']
        await bot.send_group_msg(group_id=group_id, message=MessageSegment.text(reply_text))
        # 只有在最后一轮回复，才会回复表情包
        if response_content.index(reply) == len(response_content) - 1 and random.random() < 0.3:
            if 'reply_face' in reply:
                emoticon = await select_emoticon(history, "无", reply['reply_text'],reply['reply_face'])
                if emoticon:
                    await bot.send_group_msg(group_id=group_id, message=MessageSegment.image(f"base64://{emoticon}"))
        # 提取回复时间
        if 'reply_time' in reply:
            reply_time = reply['reply_time']
            await asyncio.sleep(reply_time)
