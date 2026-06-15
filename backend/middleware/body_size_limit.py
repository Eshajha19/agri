from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

MAX_BODY_SIZE = 5 * 1024 * 1024  # 5 MB default

class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        body = await request.body()
        if len(body) > MAX_BODY_SIZE:
            return JSONResponse(
                {"error": f"Request body too large (>{MAX_BODY_SIZE} bytes)"},
                status_code=413,
            )
        return await call_next(request)
