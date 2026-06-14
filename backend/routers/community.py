"""Community-facing backend endpoints."""
import asyncio
import os
import time
import logging

import httpx
from fastapi import APIRouter, HTTPException, Query

router = APIRouter()
logger = logging.getLogger(__name__)

GITHUB_OWNER = os.getenv("GITHUB_CONTRIBUTORS_OWNER", "Eshajha19")
GITHUB_REPO = os.getenv("GITHUB_CONTRIBUTORS_REPO", "agri")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_API_TOKEN")
CACHE_TTL_SECONDS = int(os.getenv("GITHUB_CONTRIBUTORS_CACHE_TTL", "300"))

# Limit contributor records returned to community feeds to reduce
# rendering overhead and improve scalability in high-engagement views.
MAX_VISIBLE_CONTRIBUTORS = 50

_contributors_cache = {"expires_at": 0.0, "data": []}

# asyncio.Lock serialises concurrent cache-miss fetches within a single
# worker process.  Without this, two requests that both find the cache
# expired will both fire a GitHub API request simultaneously (cache
# stampede), consuming rate-limit quota twice for no benefit.
# asyncio.Lock is the correct primitive here because get_contributors is
# an async handler running in the event loop — threading.Lock would block
# the loop, while asyncio.Lock yields control while waiting.
#
# The lock is initialized at module load time by init_community(), which
# must be called during the application's lifespan setup before any async
# handlers execute. This ensures all coroutines share a single lock instance,
# preventing lazy initialization race conditions where multiple coroutines
# might create separate lock instances.
_cache_lock: asyncio.Lock | None = None


def init_community():
    """Initialize the community module with proper asyncio context.
    
    Must be called during application startup (in lifespan context) to ensure
    the cache lock is created inside a running event loop. This is called from
    main.py lifespan.
    """
    global _cache_lock
    if _cache_lock is None:
        _cache_lock = asyncio.Lock()


def _get_cache_lock() -> asyncio.Lock:
    """Return the module-level asyncio.Lock.
    
    The lock must have been initialized by init_community() during startup.
    Raises RuntimeError if init_community() was not called.
    """
    if _cache_lock is None:
        raise RuntimeError(
            "Community module not initialized. init_community() must be "
            "called during application startup."
        )
    return _cache_lock


def _get_cached_contributors():
    if time.time() < _contributors_cache["expires_at"]:
        return _contributors_cache["data"]
    return None


def _set_cached_contributors(data):
    # Store a shallow copy so the cached list is independent of the caller's
    # reference. If the caller (or any future code path) mutates the original
    # list after calling this function, the cached data remains unchanged.
    # Equally, callers that receive the cached list via _get_cached_contributors
    # cannot corrupt the cache by mutating the returned reference.
    _contributors_cache["data"] = list(data)
    _contributors_cache["expires_at"] = time.time() + CACHE_TTL_SECONDS


def _calculate_feed_ranking_score(contributor):
    """Calculate a lightweight ranking score for community feed ordering."""
    contributions = contributor.get("contributions", 0)

    return max(contributions, 0) * 5


def _rank_contributors(contributors):
    """Apply deterministic feed ranking while preserving existing behavior."""
    ranked = list(contributors)

    for contributor in ranked:
        contributor["ranking_score"] = _calculate_feed_ranking_score(
            contributor
        )

    ranked.sort(
        key=lambda contributor: (
            contributor.get("ranking_score", 0),
            contributor.get("contributions", 0),
        ),
        reverse=True,
    )

    for rank, contributor in enumerate(
        ranked,
        start=1,
    ):
        contributor["feed_rank"] = rank

    return ranked


@router.get("/contributors")
async def get_contributors(
    per_page: int = Query(default=100, ge=1, le=100),
    limit: int = Query(default=20, ge=1, le=50),
):
    # Fast path: return cached data without acquiring the lock.
    # The check is a single dict read — safe to do outside the lock because
    # Python's GIL makes individual dict reads atomic, and a stale-but-valid
    # cache hit is always acceptable.
    cached = _get_cached_contributors()
    if cached is not None:
        return {
            "success": True,
            "source": "cache",
            "contributors": cached[:limit],
            "total": len(cached),
            "ranking": {
                "enabled": True,
                "strategy": "contribution_based_feed_ranking",
            },
        }

    # Slow path: cache is expired.  Acquire the lock so only one coroutine
    # fetches from GitHub while others wait.  After the lock is released the
    # waiting coroutines will find a warm cache and return immediately.
    async with _get_cache_lock():
        # Re-check inside the lock: a previous waiter may have already
        # populated the cache while we were waiting to acquire it.
        cached = _get_cached_contributors()
        if cached is not None:
            return {
                "success": True,
                "source": "cache",
                "contributors": cached[:limit],
                "total": len(cached),
                "ranking": {
                    "enabled": True,
                    "strategy": "contribution_based_feed_ranking",
                },
            }

        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "FasalSaathi-Backend",
        }

        if GITHUB_TOKEN:
            headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

        url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contributors"

        try:
            async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:
                response = await client.get(url, params={"per_page": per_page})

            if response.status_code == 403:
                raise HTTPException(status_code=503, detail="GitHub rate limit reached. Try again later.")

            if response.status_code >= 400:
                raise HTTPException(status_code=502, detail="Unable to load contributors right now.")

            payload = response.json()
            contributors = [
                {
                    "id": item.get("id"),
                    "login": item.get("login"),
                    "avatar_url": item.get("avatar_url"),
                    "html_url": item.get("html_url"),
                    "contributions": item.get("contributions", 0),
                }
                for item in payload
                if isinstance(item, dict) and item.get("login")
            ]

            contributors = _rank_contributors(
                contributors
            )

            contributors = contributors[:MAX_VISIBLE_CONTRIBUTORS]

            logger.info(
                "[COMMUNITY_FEED] contributors=%s source=github",
                len(contributors),
            )

            _set_cached_contributors(contributors)
            return {
                "success": True,
                "source": "github",
                "contributors": contributors[:limit],
                "total": len(contributors),
                "ranking": {
                    "enabled": True,
                    "strategy": "contribution_based_feed_ranking",
                    "signals": [
                        "contributions",
                        "ranking_score",
                    ],
                },
            }
        except httpx.TimeoutException as exc:
            logger.warning("Contributor fetch timed out: %s", exc)
            raise HTTPException(status_code=504, detail="Contributor data request timed out.")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Contributor fetch failed: %s", exc)
            raise HTTPException(status_code=502, detail="Unable to load contributors right now.")
