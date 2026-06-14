from fastapi import APIRouter
from google.cloud import translate_v2 as translate

router = APIRouter()

@router.post("/api/translate")
async def translate_text(payload: dict):
    text = payload.get("text")
    target = payload.get("targetLang")
    client = translate.Client()
    result = client.translate(text, target_language=target)
    return {"translatedText": result["translatedText"]}
