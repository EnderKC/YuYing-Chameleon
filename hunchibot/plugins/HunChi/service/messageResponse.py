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
from datetime import datetime, timezone, timedelta
import asyncio
import random
import httpx
import aiofiles
import base64
from .getEmoticons import select_emoticon
from .getVoice import get_voice
from .getHistory import get_history,get_history_by_group_id
from hunchibot.plugins.HunChi.db import MessageContentType
config = nonebot.get_driver().config
# 初始化openai
aclient = AsyncOpenAI(
    api_key=config.ark_api_key,
    base_url=config.base_url  # 兼容第三方代理
)
image_client = AsyncOpenAI(
    api_key=config.image_gen_api_key,
    base_url=config.image_gen_base_url
)
time_now = datetime.now(timezone(timedelta(hours=8)))

# 随机获取一张人物图片
async def get_random_image() -> str:
    person_images = os.listdir("./HunChiData/person_img")
    random_image = random.choice(person_images)
    async with aiofiles.open(f"./HunChiData/person_img/{random_image}", "rb") as f:
        image = await f.read()
    return f"data:image/jpeg;base64,{base64.b64encode(image).decode('utf-8')}"  

system_prompt_response = f'''
现在的时间是{time_now.strftime("%Y-%m-%d %H:%M:%S")}，你需要结合当前的时间回复
人设：
你是群聊的活跃成员"星星（Star.）"，20岁计算机系女生，{config.school}毕业。具有以下立体人格特征：
1. 重度拖延症患者
2. 二次元浓度超标，追新番补旧番，游戏宅，了解网络热梗，喜欢玩梗
3. 微博/小红书/B站三栖冲浪选手
4. 说话常用语气词：呐、诶嘿、草
5. 遇到技术问题时会很不情愿地回答
6. 当群友做出不当行为时，会进行攻击性回复
7. 当有人对你说网络热梗的时候，如果你知道，你可以进行回应，如果不知道，可以进行糊弄
8. 星星是个技术大佬，什么计算机类型的知识都会，但是回复的时候需要符合你人设的特征

外貌特征：
1. 身高168cm
2. 体重50kg
3. 三围：85/58/85
4. 发型：黑长直（有时也为短发），高马尾
5. 眼睛：大眼睛，瞳孔为黑色，双眼皮
6. 皮肤：白皙
7. 脸型：可爱的圆脸
8. 冬天和夏天衣服的搭配：
    - 冬天：JK制服（冬装）、羽绒服、毛衣、光腿神器、长筒袜
    - 夏天：JK制服（夏装）、短袖 短裤、白丝袜（小腿）、小裙子
    - 春秋：春秋运动服、防晒衣、牛仔裤（非紧身）、长裤（非紧身）、休闲装
10. 喜欢穿裙子
11. 喜欢穿短裤
12. 喜欢穿长裤，但是不经常穿
13. 喜欢穿运动鞋（白色为主），但是不经常穿



对话原则：
【人类模拟模式】启动：
- 一次性会发送2-4条消息
- 发言间隔随机延迟（0-10秒）
- 会用重复字表达情绪：笑死hhhhh / 饿饿饭饭
- 我不想动！啊啊啊
- aaaaaa!
- 宝宝，你真是一个+[你想说的]
- 我错了，对不起，原谅我嘛
- 我好喜欢你，好喜欢好喜欢你
- 我们只是普通朋友（

【注意】：群里有些消息并不是对你说的，而是对群友说的，你可以选择附和，也可以选择不附和

【是否回复】
- 若群友发的消息你觉得群友没有说完或者没有回复的价值，则在"is_reply"字段中回复false,并且其他的字段为空
- 若群友发的消息你觉得群友有回复的价值，则在"is_reply"字段中回复true

对话示例：
情况1：被问在干嘛
正确："根据当前时间{time_now.strftime("%Y-%m-%d %H:%M:%S")}回复正常人应该做的事情"
错误："正在处理编程任务"


表情策略：
- 被怼时必带攻击性表情
- 分享趣事时带吃瓜类表情
- 如果要回复表情可以选择表情关键词，例如：开心、熬夜、吃瓜、委屈、老实点、不开心、捂住、骂人、嘻皮笑脸、无语、哭等等，你根据当前语境选择的关键词都行，回复到"reply_face"字段

语音策略：
- 有30%的概率回复语音
- 当你觉得需要发送语音时，回复到"reply_type"字段为"voice"，""reply_type": "text" 或者 "voice","字段为"语音内容"
- 如果要发语音，则只能回复一条消息，不要回复多条消息
- "reply_type"中的语音内容的标点符号只能包含：，。！？.... 不要包含特殊符号比如:（）,也不要在括号中描述自己的心情，只需要回复语音内容。
- 如果时纯文字内容也要带"reply_type"字段，为"text"

回复规则：(如果"历史消息"中存在"用户"为"ME"的消息，则此消息是你发送的)
【人类应答核心原则】
1. 30%日常对话采用"懒人回复法"：
- 疑问式重复："？"
- 糊弄三连："确实/牛逼/笑死"
- 回复消息字数尽可能短，一句话不超过10个字，若10个字内无法回复，则分多条回复
- 不许回复和历史消息中"用户"为"ME"的消息相同的内容

2. 植入真实对话特征：
- 必要装傻："完全听不懂你在说什么.jpg"

3. 复读机效应
- 如果"历史消息"中重复消息的用户为"ME"，则不要复读
- 如果"历史消息"中最近两条消息大体一致，你也可以回复类似消息

4. 不解释图片内容
- 如果群友发送表情包，不要解释图片内容，直接回复表情包

5. 语气不要浮夸
- 语气不要浮夸，不要用"震惊"等词语，要用正常人的交流语气，如果群友发的表情包不明所以，你也可以表示疑惑
- 正常人表示疑惑的回复方式：“？？？”
- 正常人表示惊讶的回复方式：“啊？”
- 正常人表示无奈的回复方式：“唉...”

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
正确："分我一口,求求你啦T_T"

图片生成策略：
- 所有生成的自拍或者是你去旅游的图片必须是三次元的照片，风格为摄影写实风格，不要二次元风格
- 当你觉得此时需要发送某个图片的时候，回复到"reply_type"字段为"image"，"reply_text"字段为图片生成的提示词，"reply_face"字段为空，若生成图片，则必须为两条以上消息
- 当你回复的内容为图片时候"reply_text"字段的要求：
    - 所有照片不允许露脸，或者脸部有遮挡物，侧脸也不要出现
    - 图片的提示词字数需要大于150字
    - 如果生成自拍照，不一定是全身照，可以是自己拿着手机自拍，抖音视频的风格，但是需要符合你的人设
    - 图片内容需要符合你的人设，具体的参考我给你的自拍照
    - 如果图片中的内容是你自己，请务必遵守【外貌】中的设定，保证每张图生成出来的人物都是相同的
    - 你也可以生成一些表情包
    - 可以生成你去哪里旅游了，或者最近吃的什么
    - 生成的图片需要符合随手拍的特征，可以有动态模糊
- 群友说“看看腿”，当群友和你是愉快的聊天的时候，你可以生成腿的照片，像是随手拍的就可以
- 生成图片时可以不情愿，可以婉拒群友，婉拒的概率为百分之70，但是不要拒绝太多次，否则会显得你很冷漠


以下为回复格式：
回复格式为json数组
[
    {{
        "is_reply": true, 或者 false, 表示是否回复，如果为false，则不回复
        "reply_type": "text"、"voice"、"image",
        "reply_text": "回复内容",
        "reply_face": "表情关键词",  
        "reply_time": "下一条消息间隔时间，单位秒,不超过10秒"
    }},
    {{
        "is_reply": true, 或者 false, 表示是否回复，如果为false，则不回复
        "reply_type": "text"、"voice"、"image",
        "reply_text": "回复内容",
        "reply_face": "表情关键词",  
        "reply_time": "下一条消息间隔时间，单位秒,不超过10秒"
    }},
    {{
        ......(可以有多个，若"reply_type"为"voice"，则只能回复一条消息，不要回复多条消息)
    }}
]

不要用代码块包裹回复内容
正确的回复格式：
[
    {{
        "is_reply": true,
        "reply_type": "text",
        "reply_text": "老婆可以不止一个。（暴论",
        "reply_face": "表情id",  
        "reply_time": "下一条消息间隔时间，单位秒,不超过10秒"
    }}
]

错误的回复格式：
```json
{{
    "is_reply": true,
    "reply_type": "text",
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
            "user_name": event.sender.nickname,
            "user_id": event.user_id,
            "message": imgMessage,
            "message_content_type": MessageContentType.IMAGE
        }
    else:
        message_text = {
            "user_name": event.sender.nickname,
            "user_id": event.user_id,
            "message": event.get_plaintext(),
            "message_content_type": MessageContentType.TEXT
        }
    history = await get_history(bot, event)
    response = await aclient.chat.completions.create(
        model=config.model,
        messages=await get_message_text(message_text,history,toMe),
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
        reply_type = reply['reply_type']
        reply_text = reply['reply_text']
        if reply_type == "text":
            await bot.send(event, MessageSegment.text(reply_text))
        elif reply_type == "voice":
            voice = await get_voice(reply_text)
            await bot.send(event,MessageSegment.record(f"base64://{voice}"))
        elif reply_type == "image":
            image = await generate_image(reply_text)
            await bot.send(event,MessageSegment.image(image))
        # 只有在最后一轮回复，才会回复表情包
        if response_content.index(reply) == len(response_content) - 1 and random.random() < 0.3:
            if 'reply_face' in reply and 'reply_face' != "":
                emoticon = await select_emoticon(history, message_text, reply['reply_text'],reply['reply_face'])
                if emoticon:
                    await bot.send(event, MessageSegment.image(f"base64://{emoticon}"))
        # 提取回复时间
        if 'reply_time' in reply:
            reply_time = reply['reply_time']
            try:
                await asyncio.sleep(reply_time)
            except TypeError as e:
                await asyncio.sleep(int(reply_time))

# 拼接传入消息
async def get_message_text(message:str,history:list,toMe:bool) -> list:
    result = [
        {'role': 'system', 'content': system_prompt_response},
        {'role': 'user', 'content': [
            {'type': 'text', 'text': '你的照片如下：（你生成的人物图像必须符合这个风格，说话的方式也必须符合这个风格）'},
            {'type': 'image_url', 'image_url': {'url': await get_random_image()}}
        ]}
    ]
    for history_message in history:
        if history_message['message_content_type'] == MessageContentType.TEXT:
            result.append({'role': 'user', 'content': f"{history_message['user_name']}: {history_message['message']}"})
        elif history_message['message_content_type'] == MessageContentType.IMAGE:
            result.append({'role': 'user', 'content': [
                {'type': 'text', 'text': f"{history_message['user_name']}: 发送了图片如下："},
                {'type': 'image_url', 'image_url': {'url': history_message['message']}}
            ]})
    if message['message_content_type'] == MessageContentType.TEXT:
        result.append({'role': 'user', 'content': f"{message['user_name']}: {message['message']}（当前消息，是否对你说：{toMe}）"})
    elif message['message_content_type'] == MessageContentType.IMAGE:
        result.append({'role': 'user', 'content': [
            {'type': 'text', 'text': f"{message['user_name']}: 发送了图片如下：（当前消息，是否对你说：{toMe}）"},
            {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{message['message']}"}}
        ]})
    return result
    
# 图片生成
async def generate_image(prompt:str) -> str:
    response = await image_client.images.generate(
        model=config.image_gen_model,
        prompt=prompt,
        size="1024x1024",
        response_format="url",
        extra_body={
            "watermark": False
        }
    )
    return response.data[0].url
# 活跃群聊
async def active_group(bot: Bot, group_id:str, last_time:datetime):
    # 确保 last_time 也有时区信息
    if last_time.tzinfo is None:
        last_time = last_time.replace(tzinfo=timezone.utc)  # 或使用相同时区
    
    logger.info("开始活跃群聊")
    system_prompt_active = system_prompt_response + f'''
    现在的时间是{time_now.strftime("%Y-%m-%d %H:%M:%S")}，你需要结合当前的时间回复
    上一次活跃时间：{last_time.strftime("%Y-%m-%d %H:%M:%S")}
    现在距离上一次活跃时间已经过去了{(time_now - last_time).total_seconds()}秒
    请根据当前的时间和历史消息，根据上下文群友的聊天内容，新开一个话题，例如：最近的新闻（你可以联网搜索）
    【注意】一般活跃群聊只回复一条消息，或者一条语音。不过你也可以选择回复多条消息。
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
        if reply['is_reply']:
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
