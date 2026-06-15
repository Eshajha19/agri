from starlette.requests import Request


async def ensure_body_available(request: Request) -> bytes:
    """
    Safely read and cache the request body.

    Starlette caches request.body() internally, so downstream middleware
    and endpoint handlers can continue accessing the payload without
    mutating request._receive or other private APIs.
    """

    if hasattr(request.state, "_body_cache"):
        return request.state._body_cache

    body = await request.body()
    request.state._body_cache = body

    return body


async def validate_body_size(
    request: Request,
    max_size: int,
) -> bool:
    """
    Validate request size using Content-Length when available.

    Returns True when the request is within the allowed limit.
    """

    content_length = request.headers.get("content-length")

    if content_length is None:
        return True

    try:
        return int(content_length) <= max_size
    except ValueError:
        return True


async def get_cached_body(request: Request) -> bytes:
    """
    Return cached request body if available,
    otherwise load and cache it.
    """

    return await ensure_body_available(request)


def has_cached_body(request: Request) -> bool:
    """
    Check whether the request body has already been cached.
    """

    return hasattr(request.state, "_body_cache")