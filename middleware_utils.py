from starlette.requests import Request


async def ensure_body_available(request: Request) -> bytes:
    """
    Safely read and cache the request body.

    Starlette caches request.body() internally, so downstream
    middleware and endpoint handlers can continue accessing the
    body without mutating request._receive or other private APIs.
    """

    if hasattr(request.state, "_body_cache"):
        return request.state._body_cache

    body = await request.body()
    request.state._body_cache = body

    return body


async def get_cached_body(request: Request) -> bytes:
    """
    Return cached body if available, otherwise load it.
    """

    return await ensure_body_available(request)


def has_cached_body(request: Request) -> bool:
    """
    Check whether request body has already been cached.
    """

    return hasattr(request.state, "_body_cache")