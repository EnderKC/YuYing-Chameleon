import os
import base64
from openai import AsyncOpenAI
import nonebot
import json
from nonebot.log import logger

config = nonebot.get_driver().config

client = AsyncOpenAI(
    api_key=config.ark_api_key,
    base_url=config.v3_base_url  # 兼容第三方代理
)

async def select_emoticon(original_text: str, realy_text: str, reply_text: str,reply_face:str):
    # 第一轮：根据关键词粗选
    emoticons = []
    with open("./HunChiData/emoji.json", "r", encoding="utf-8") as file:
        emojis = json.load(file)
    for emoji in emojis:
        if len(emoticons) < 20 and (any(reply_face in tag for tag in emoji["tags"]) or reply_face in emoji["describe"]):
            emoticons.append(emoji)
    # 若没有匹配到表情包，则返回空
    if len(emoticons) == 0:
        return None
    
    # 第二轮：选择具体表情包
    emoticon_prompt = f"""
    历史消息: "{original_text}"
    当前消息: "{realy_text}"
    回复: "{reply_text}"
    从以下表情包标题中选择最合适的一个：
    {', '.join([emoji["describe"] for emoji in emoticons])}
    
    只需返回表情包的"describe"字段，不需要其他解释。
    """
    
    emoticon_response = await get_ai_response(emoticon_prompt)
    selected_emoticon = emoticon_response.strip()
    logger.info(f"选择表情包: {selected_emoticon}")
    emoticon_id = [emoji["id"] for emoji in emoticons if emoji["describe"] == selected_emoticon][0]
    logger.info(f"id: {emoticon_id}")
    # 获得base64
    with open(f"./HunChiData/emoji/{emoticon_id}.jpg", "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    return base64_image

async def get_ai_response(prompt: str) -> str:
    response = await client.chat.completions.create(
        model=config.v3_model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=50
    )
    return response.choices[0].message.content
