import json
from starlette.requests import Request

async def ensure_body_available(request: Request):
    """
    Read and cache the request body, then re-inject it so downstream handlers
    can still consume it.
    """
    body = await request.body()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request._receive = receive
    return body
