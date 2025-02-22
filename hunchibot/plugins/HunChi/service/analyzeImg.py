from openai import AsyncOpenAI
from nonebot.log import logger
import nonebot

config = nonebot.get_driver().config
# 配置豆包专用参数
client = AsyncOpenAI(
    api_key=config.ark_api_key,  # 火山引擎AK/SK获取方式见下文
    base_url=config.img_base_url  # 豆包专用API端点
)


async def analyze_img(img_url: str | None = None, img_base64: str | None = None) -> str:
    if img_url:
        response = await client.chat.completions.create(
            model=config.img_model,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": "请描述这张图片的内容"}, {
                    "type": "image_url", "image_url": {"url": img_url}}]}
            ]
        )
    else:
        response = await client.chat.completions.create(
            model=config.img_model,
            messages=[{"role": "user", "content": [{"type": "text", "text": "请描述这张图片的内容"}, {
                "type": "image_url", "image_url": {"url": img_base64}}]}]
        )
    logger.info(response.choices[0].message.content)
    return response.choices[0].message.content
