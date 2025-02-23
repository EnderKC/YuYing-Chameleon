import os
import base64
from openai import AsyncOpenAI
import nonebot
from nonebot.log import logger

config = nonebot.get_driver().config

client = AsyncOpenAI(
    api_key=config.ark_api_key,
    base_url=config.v3_base_url  # 兼容第三方代理
)

async def select_emoticon(original_text: str, realy_text: str, reply_text: str):
    # 第一轮：选择合适的文件夹
    folders = os.listdir("./EmojiPackage")
    folder_prompt = f"""
    历史消息: "{original_text}"
    当前消息: "{realy_text}"
    回复: "{reply_text}"
    
    根据以上对话的语境和情感，从以下文件夹中选择最合适的一个：
    {', '.join(folders)}
    
    只需返回文件夹名称，不需要其他解释。
    """
    
    folder_response = await get_ai_response(folder_prompt)
    selected_folder = folder_response.strip()
    
    # 第二轮：选择具体表情包
    folder_path = os.path.join("EmojiPackage", selected_folder)
    emoticons = os.listdir(folder_path)
    
    emoticon_prompt = f"""
    历史消息: "{original_text}"
    当前消息: "{realy_text}"
    回复: "{reply_text}"
    从以下表情包标题中选择最合适的一个：
    {', '.join(emoticons)}
    
    只需返回表情包文件名，不需要其他解释。
    """
    
    emoticon_response = await get_ai_response(emoticon_prompt)
    selected_emoticon = emoticon_response.strip()
    logger.info(f"选择表情包: {selected_emoticon}")
    # 完整路径
    emoticon_path = os.path.join(folder_path, selected_emoticon)
    # 获取base64
    with open(emoticon_path, "rb") as image_file:
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
