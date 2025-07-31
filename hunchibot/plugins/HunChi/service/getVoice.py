import httpx
import nonebot
from nonebot.log import logger
import uuid
config = nonebot.get_driver().config

appid = config.voice_clone_appid
token = config.voice_clone_token
cluster = config.voice_clone_cluster
voice_type = config.voice_type

host = "openspeech.bytedance.com"
api_url = f"https://{host}/api/v1/tts"

header = {"Authorization": f"Bearer;{token}"}

async def get_voice(text: str):
    request_json = {
    "app": {
        "appid": appid,
        "token": "access_token",
        "cluster": cluster
    },
    "user": {
        "uid": "388808087185088"
    },
    "audio": {
        "voice_type": voice_type,
        "encoding": "mp3",
        "speed_ratio": 1.0,
        "volume_ratio": 1.0,
        "pitch_ratio": 1.0,
    },
    "request": {
        "reqid": str(uuid.uuid4()),
        "text": text,
        "text_type": "plain",
        "operation": "query",
        "with_frontend": 1,
        "frontend_type": "unitTson"
        }
    }
    response = httpx.post(api_url, json=request_json, headers=header)
    data = response.json()
    if data['code'] == 3000:
        return data['data']
    else:
        logger.error(f"获取语音失败: {data['message']}")
        return None
