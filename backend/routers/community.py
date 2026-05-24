"""Community-facing backend endpoints."""
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

_contributors_cache = {"expires_at": 0.0, "data": []}


def _get_cached_contributors():
    if time.time() < _contributors_cache["expires_at"]:
        return _contributors_cache["data"]
    return None


def _set_cached_contributors(data):
    _contributors_cache["data"] = data
    _contributors_cache["expires_at"] = time.time() + CACHE_TTL_SECONDS


@router.get("/contributors")
async def get_contributors(per_page: int = Query(default=100, ge=1, le=100)):
    cached = _get_cached_contributors()
    if cached is not None:
        return {"success": True, "source": "cache", "contributors": cached}

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

        _set_cached_contributors(contributors)
        return {"success": True, "source": "github", "contributors": contributors}
    except httpx.TimeoutException as exc:
        logger.warning("Contributor fetch timed out: %s", exc)
        raise HTTPException(status_code=504, detail="Contributor data request timed out.")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Contributor fetch failed: %s", exc)
        raise HTTPException(status_code=502, detail="Unable to load contributors right now.")
