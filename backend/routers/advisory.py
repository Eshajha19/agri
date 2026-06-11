"""Rule-based farmer advisory API."""
import html
import threading
import uuid
from collections import OrderedDict, deque
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from advisory_rules import generate_advisories
from backend.core.logging_config import setup_logging


router = APIRouter()
logger = setup_logging(__name__)
_MAX_STORED_ALERTS = 50
_MAX_STORED_GRAPH_HISTORY = 25

# Maximum number of distinct Firebase UIDs tracked in each in-process store.
# When the limit is reached the oldest UID entry is evicted (LRU order via
# OrderedDict) before the new one is inserted, keeping memory consumption
# proportional to this constant regardless of how many users the process
# has served over its lifetime.
_MAX_TRACKED_UIDS = 500


class _BoundedUidStore:
    """
    Thread-safe, UID-keyed store backed by an OrderedDict with a hard cap on
    the number of distinct UIDs it will hold.

    Each UID maps to a deque whose maximum length is fixed at construction
    time.  When a new UID would exceed ``max_uids``, the least-recently-used
    UID is evicted before the new entry is created.  Accessing an existing
    UID moves it to the most-recently-used position so active users are
    never evicted while idle ones are.

    This replaces the previous ``defaultdict(lambda: deque(maxlen=N))``
    pattern, which created one dict entry per UID and never removed any of
    them, causing the outer dict to grow without bound for the lifetime of
    the process.
    """

    def __init__(self, deque_maxlen: int, max_uids: int = _MAX_TRACKED_UIDS) -> None:
        self._deque_maxlen = deque_maxlen
        self._max_uids = max_uids
        self._data: OrderedDict[str, deque] = OrderedDict()
        self._lock = threading.Lock()

    def extend(self, uid: str, items) -> None:
        """Append *items* to the deque for *uid*, creating it if necessary."""
        with self._lock:
            self._touch(uid)
            self._data[uid].extend(items)

    def appendleft(self, uid: str, item: Any) -> None:
        """Prepend *item* to the deque for *uid*, creating it if necessary."""
        with self._lock:
            self._touch(uid)
            self._data[uid].appendleft(item)

    def get(self, uid: str) -> list:
        """Return a snapshot list for *uid* (empty list if not present)."""
        with self._lock:
            if uid not in self._data:
                return []
            self._data.move_to_end(uid)
            return list(self._data[uid])

    def _touch(self, uid: str) -> None:
        """Ensure *uid* has a deque entry, evicting the LRU entry if needed.

        Must be called with ``self._lock`` already held.

        Uses a retry loop to guarantee the size limit is respected even
        under high concurrency where multiple threads may observe the
        same pre-eviction size.  Each iteration evicts one LRU entry and
        rechecks the size, ensuring that when we finally add the new entry
        the store is guaranteed to be within capacity.
        """
        with self._lock:
            while True:
                if uid in self._data:
                    self._data.move_to_end(uid)
                    return
                if len(self._data) < self._max_uids:
                    self._data[uid] = deque(maxlen=self._deque_maxlen)
                    return
                # Capacity reached — evict one entry and retry to ensure we
                # never exceed _max_uids even under concurrent contention.
                self._data.popitem(last=False)


_stored_alerts = _BoundedUidStore(deque_maxlen=_MAX_STORED_ALERTS)
_graph_history = _BoundedUidStore(deque_maxlen=_MAX_STORED_GRAPH_HISTORY)

# Single module-level lock kept for any remaining code that needs it, but
# _BoundedUidStore manages its own internal lock for all store operations.
_store_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Dependency injection — wired in main.py lifespan
# ---------------------------------------------------------------------------
_verify_role_fn = None
_db = None


def init_advisory(verify_role_fn, db_client=None) -> None:
    global _verify_role_fn, _db
    _verify_role_fn = verify_role_fn
    _db = db_client


# ---------------------------------------------------------------------------
# Helper — UID extraction & validation
# ---------------------------------------------------------------------------

# Helpers
# ---------------------------------------------------------------------------

def _sanitise_alert_text(text: Any) -> str:
    """Strip HTML/script from alert text to prevent stored XSS in advisories."""
    return html.escape(str(text or ""), quote=True)


def _sanitise_alert_value(value: Any) -> Any:
    """Recursively convert alert values into render-safe primitives."""
    if isinstance(value, dict):
        return {str(key): _sanitise_alert_value(nested_value) for key, nested_value in value.items()}
    if isinstance(value, list):
        return [_sanitise_alert_value(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitise_alert_value(item) for item in value]
    if isinstance(value, set):
        return [_sanitise_alert_value(item) for item in value]
    return _sanitise_alert_text(value)


def _sanitise_alert(alert: Any) -> dict:
    """Sanitise a single alert dict so stored content is safe to render."""
    if not isinstance(alert, dict):
        return {"message": _sanitise_alert_text(alert), "sanitised": "true"}
    safe = {}
    for k, v in alert.items():
        safe[str(k)] = _sanitise_alert_value(v)
    safe["sanitised"] = "true"
    return safe


def _sanitise_alerts(alerts: list) -> list:
    """Return a sanitised copy of the alerts list — originals are untouched."""
    return [_sanitise_alert(a) for a in alerts]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _level_score(value: Any) -> int:
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric <= 1:
            return int(max(0, min(100, numeric * 100)))
        return int(max(0, min(100, numeric)))

    text = str(value or "").strip().lower()
    mapping = {
        "low": 20,
        "moderate": 45,
        "medium": 55,
        "elevated": 65,
        "high": 80,
        "critical": 95,
        "severe": 95,
        "dry": 75,
        "wet": 40,
    }
    return mapping.get(text, 0)


def _extract_metric(payload: dict[str, Any], keys: tuple[str, ...]) -> Optional[float]:
    for key in keys:
        value = _safe_float(payload.get(key))
        if value is not None:
            return value
    return None


def _pest_risk_score(weather: dict[str, Any], pest: dict[str, Any]) -> int:
    score = 20
    temperature = _extract_metric(weather, ("temperature", "temp", "temperature_2m"))
    humidity = _extract_metric(weather, ("humidity", "relative_humidity", "relative_humidity_2m"))
    rainfall = _extract_metric(weather, ("rainfall_next_24h", "rainfall_mm", "precipitation", "rain"))
    pest_pressure = max(
        _level_score(pest.get("pressure")),
        _level_score(pest.get("risk")),
        _level_score(pest.get("severity")),
        _level_score(pest.get("outbreak")),
    )

    if temperature is not None:
        if temperature >= 34:
            score += 18
        elif temperature >= 30:
            score += 12
        elif temperature <= 18:
            score += 6

    if humidity is not None:
        if humidity >= 80:
            score += 18
        elif humidity >= 70:
            score += 12

    if rainfall is not None and rainfall > 8:
        score += 8

    score += pest_pressure // 2
    return min(100, score)


def _irrigation_score(weather: dict[str, Any], soil: dict[str, Any]) -> int:
    score = 20
    temperature = _extract_metric(weather, ("temperature", "temp", "temperature_2m"))
    rainfall = _extract_metric(weather, ("rainfall_next_24h", "rainfall_mm", "precipitation", "rain"))
    soil_moisture = _extract_metric(soil, ("moisture", "soil_moisture", "water_content"))

    if temperature is not None and temperature >= 32:
        score += 14
    if rainfall is not None:
        if rainfall <= 2:
            score += 18
        elif rainfall <= 5:
            score += 10
        elif rainfall >= 10:
            score -= 10
    if soil_moisture is not None:
        if soil_moisture <= 25:
            score += 22
        elif soil_moisture <= 40:
            score += 12
        elif soil_moisture >= 65:
            score -= 10

    return max(0, min(100, score))


def _market_score(market: dict[str, Any]) -> int:
    score = 35
    trend = str(market.get("trend") or market.get("direction") or "stable").strip().lower()
    price = _extract_metric(market, ("price", "current_price", "modal_price", "market_price"))

    if trend in {"up", "rising", "bullish"}:
        score += 20
    elif trend in {"down", "falling", "bearish"}:
        score += 8

    if price is not None:
        if price >= 7000:
            score += 15
        elif price >= 3500:
            score += 8
        elif price <= 1500:
            score -= 8

    return max(0, min(100, score))


def _build_farm_graph(payload: "FarmIntelligenceRequest") -> dict[str, Any]:
    weather = payload.weather or {}
    soil = payload.soil or {}
    pest = payload.pest or {}
    market = payload.market or {}

    pest_risk = _pest_risk_score(weather, pest)
    irrigation_score = _irrigation_score(weather, soil)
    market_score = _market_score(market)

    weather_summary = {
        "temperature": _extract_metric(weather, ("temperature", "temp", "temperature_2m")),
        "humidity": _extract_metric(weather, ("humidity", "relative_humidity", "relative_humidity_2m")),
        "rainfall": _extract_metric(weather, ("rainfall_next_24h", "rainfall_mm", "precipitation", "rain")),
    }
    soil_summary = {
        "ph": _extract_metric(soil, ("ph", "soil_ph", "pH")),
        "moisture": _extract_metric(soil, ("moisture", "soil_moisture", "water_content")),
        "nitrogen": soil.get("nitrogen"),
        "phosphorus": soil.get("phosphorus"),
        "potassium": soil.get("potassium"),
    }
    pest_summary = {
        "pressure": pest.get("pressure") or pest.get("risk") or pest.get("severity") or pest.get("outbreak"),
        "observed": pest.get("observed") or pest.get("name") or pest.get("pest_name"),
    }
    market_summary = {
        "commodity": market.get("commodity") or payload.crop_type,
        "price": _extract_metric(market, ("price", "current_price", "modal_price", "market_price")),
        "trend": market.get("trend") or market.get("direction") or "stable",
    }

    nodes = [
        {"id": "weather", "label": "Weather", "type": "dataset", "summary": weather_summary},
        {"id": "soil", "label": "Soil", "type": "dataset", "summary": soil_summary},
        {"id": "crop", "label": "Crop", "type": "dataset", "summary": {"crop_type": payload.crop_type, "location": payload.location}},
        {"id": "pest", "label": "Pest Signals", "type": "dataset", "summary": pest_summary},
        {"id": "market", "label": "Market", "type": "dataset", "summary": market_summary},
        {"id": "pest-risk", "label": "Pest Risk", "type": "derived", "score": pest_risk},
        {"id": "irrigation", "label": "Irrigation", "type": "action", "score": irrigation_score},
        {"id": "harvest-timing", "label": "Harvest Timing", "type": "action", "score": market_score},
    ]

    edges = [
        {"from": "weather", "to": "pest-risk", "reason": "Warm, humid, and wet conditions increase pest pressure"},
        {"from": "pest", "to": "pest-risk", "reason": "Observed pests and field symptoms lift the risk score"},
        {"from": "weather", "to": "irrigation", "reason": "Heat and low rainfall increase water demand"},
        {"from": "soil", "to": "irrigation", "reason": "Moisture and soil condition determine how soon the field needs watering"},
        {"from": "pest-risk", "to": "irrigation", "reason": "Higher pest pressure can change spray and irrigation timing"},
        {"from": "crop", "to": "harvest-timing", "reason": "Crop type anchors the market timing advice"},
        {"from": "market", "to": "harvest-timing", "reason": "Price trend and commodity value affect sell/hold decisions"},
    ]

    reasoning = []
    if weather_summary["temperature"] is not None or weather_summary["humidity"] is not None:
        reasoning.append(
            f"Weather conditions raise pest risk to {pest_risk}% and can change the irrigation window."
        )
    if irrigation_score >= 55:
        reasoning.append("Low moisture or low rainfall suggests scheduling irrigation earlier in the day.")
    else:
        reasoning.append("Rainfall and soil moisture are sufficient to delay irrigation for now.")
    if market_score >= 60:
        reasoning.append("Market conditions favor holding or staging the harvest for a better sale window.")
    else:
        reasoning.append("Market pressure is moderate, so harvest timing should stay aligned with crop readiness.")

    recommendations = []
    if pest_risk >= 60:
        recommendations.append(
            {
                "title": "Escalate pest scouting",
                "priority": "high",
                "action": "Inspect the crop canopy within 24 hours and prepare a targeted pest control plan.",
                "why": "Weather and pest signals are aligning to raise outbreak risk.",
            }
        )
    else:
        recommendations.append(
            {
                "title": "Keep routine scouting",
                "priority": "medium",
                "action": "Continue field inspection on the normal schedule and watch for local pest spikes.",
                "why": "Current conditions do not indicate a sharp pest surge.",
            }
        )

    if irrigation_score >= 55:
        recommendations.append(
            {
                "title": "Adjust irrigation",
                "priority": "high",
                "action": "Use short early-morning irrigation and avoid overwatering until the next weather update.",
                "why": "Weather and soil moisture together point to higher water stress.",
            }
        )
    else:
        recommendations.append(
            {
                "title": "Delay irrigation",
                "priority": "low",
                "action": "Hold irrigation unless the field dries out faster than expected.",
                "why": "Stored moisture and forecast rainfall cover the near-term demand.",
            }
        )

    if market_score >= 60:
        recommendations.append(
            {
                "title": "Plan the sell window",
                "priority": "medium",
                "action": "Track market movement over the next few days before committing to a sale.",
                "why": "The current market trend supports a better selling window.",
            }
        )

    return {
        "graph": {
            "nodes": nodes,
            "edges": edges,
        },
        "reasoning": reasoning,
        "recommendations": recommendations,
        "summary": " | ".join(reasoning[:3]),
        "scores": {
            "pest_risk": pest_risk,
            "irrigation": irrigation_score,
            "market": market_score,
        },
    }


def _store_graph_history(uid: str, entry: dict[str, Any]) -> str:
    history_id = entry.get("history_id") or uuid.uuid4().hex[:12]
    record = {**entry, "history_id": history_id}

    # Write to Firestore first so it is the authoritative source of truth.
    #
    # The previous order was: in-memory write → Firestore write.  That
    # created a race: if the Firestore write was slow and another concurrent
    # request triggered LRU eviction of this UID's entry from _graph_history
    # between the two writes, the in-memory record would be gone before
    # Firestore confirmed the write.  _load_graph_history would then fall
    # back to the in-memory store and return an empty list, silently losing
    # the history entry for the lifetime of the process even though Firestore
    # eventually persisted it correctly.
    #
    # Correct order: Firestore first (durable), in-memory second (cache).
    # If Firestore fails, we still populate the in-memory cache so the
    # current request's response is correct and the user sees their history
    # within the same process session.  If the in-memory entry is later
    # evicted, _load_graph_history will correctly re-read from Firestore.
    if _db is not None:
        try:
            _db.collection("users").document(uid).collection("farm_intelligence_history").document(history_id).set(record, merge=True)
        except Exception as e:
            import logging
            logging.error(f"Advisory error: {e}")
            # Firestore unavailable — the in-memory write below still
            # preserves the entry for the current process session.
            pass

    _graph_history.appendleft(uid, record)

    return history_id


def _load_graph_history(uid: str) -> list[dict[str, Any]]:
    if _db is not None:
        try:
            docs = _db.collection("users").document(uid).collection("farm_intelligence_history").get()
            items = []
            for doc_snapshot in docs:
                data = doc_snapshot.to_dict() or {}
                items.append({"history_id": doc_snapshot.id, **data})
            if items:
                items.sort(key=lambda item: item.get("createdAt", ""), reverse=True)
                return items
        except Exception as exc:
            logger.warning(
                "Firestore history lookup failed for uid=%s, falling back to in-memory store: %s",
                uid,
                exc,
            )

    return _graph_history.get(uid)


async def _get_authenticated_uid(request: Request) -> str:
    """Extract and validate the caller's Firebase UID from the verified token.

    Returns the uid string on success.

    Raises
        HTTPException 401 — token missing, invalid, or uid empty.
        HTTPException 500 — advisory service not initialised.
    """
    if _verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Advisory service not initialized")

    token_data = await _verify_role_fn(request)
    uid = (token_data or {}).get("uid")

    if not uid or not isinstance(uid, str) or not uid.strip():
        raise HTTPException(
            status_code=401,
            detail="Valid authentication required — uid missing or invalid in token",
        )
    return uid.strip()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

# Maximum number of top-level keys accepted in any advisory dict field.
# Prevents memory exhaustion from oversized payloads submitted by authenticated
# users (FarmIntelligenceRequest) or anonymous callers (AdvisoryRequest).
_MAX_DICT_KEYS = 50
# Maximum character length for any string value inside a dict field.
_MAX_DICT_VALUE_LEN = 500


def _validate_advisory_dict(value: dict[str, Any], field_name: str) -> dict[str, Any]:
    """Enforce key count and value length limits on advisory dict fields."""
    if len(value) > _MAX_DICT_KEYS:
        raise ValueError(
            f"{field_name} must not exceed {_MAX_DICT_KEYS} keys (got {len(value)})"
        )
    for k, v in value.items():
        if isinstance(v, dict):
            # Nested dicts are flattened for advisory scoring; block deep nesting
            # to prevent combinatorial traversal cost.
            if len(v) > _MAX_DICT_KEYS:
                raise ValueError(
                    f"{field_name}.{k} sub-dict must not exceed {_MAX_DICT_KEYS} keys"
                )
        elif isinstance(v, str) and len(v) > _MAX_DICT_VALUE_LEN:
            raise ValueError(
                f"{field_name}.{k} value length must not exceed {_MAX_DICT_VALUE_LEN} characters"
            )
    return value


class AdvisoryRequest(BaseModel):
    model_config = {"extra": "forbid"}  # reject unknown fields (e.g. user_id)

    weather: dict[str, Any] = Field(default_factory=dict)
    soil: dict[str, Any] = Field(default_factory=dict)
    crop_type: Optional[str] = Field(default=None, max_length=50)
    store_alerts: bool = False

    @field_validator("crop_type", mode="before")
    @classmethod
    def _strip_and_reject_whitespace_only(cls, v: Any) -> Any:
        if isinstance(v, str):
            stripped = v.strip()
            if stripped == "":
                raise ValueError("crop_type must not be blank or whitespace-only")
            return stripped
        return v

    @field_validator("weather", "soil", mode="before")
    @classmethod
    def _limit_dict_size(cls, v: Any, info) -> Any:
        if isinstance(v, dict):
            return _validate_advisory_dict(v, info.field_name)
        return v


class FarmIntelligenceRequest(BaseModel):
    model_config = {"extra": "forbid"}

    crop_type: str = Field(..., min_length=1, max_length=50)
    weather: dict[str, Any] = Field(default_factory=dict)
    soil: dict[str, Any] = Field(default_factory=dict)
    pest: dict[str, Any] = Field(default_factory=dict)
    market: dict[str, Any] = Field(default_factory=dict)
    location: Optional[str] = Field(default=None, max_length=120)
    store_history: bool = True

    @field_validator("crop_type", "location", mode="before")
    @classmethod
    def _strip_and_reject_whitespace_only(cls, v: Any, info) -> Any:
        if isinstance(v, str):
            stripped = v.strip()
            if stripped == "" and info.field_name == "crop_type":
                raise ValueError("crop_type must not be blank or whitespace-only")
            return stripped
        return v

    @field_validator("weather", "soil", "pest", "market", mode="before")
    @classmethod
    def _limit_dict_size(cls, v: Any, info) -> Any:
        if isinstance(v, dict):
            return _validate_advisory_dict(v, info.field_name)
        return v


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/advisory")
async def create_advisory(payload: AdvisoryRequest, request: Request):
    """
    Generate rule-based farm advisories.

    If store_alerts is True the generated alerts are persisted server-side
    under the caller's verified Firebase UID so they can be retrieved later
    via GET /advisory/me.

    Authentication is required when store_alerts is True:
    1. Alerts are always bound to a verified identity.
    2. An unauthenticated caller cannot pollute another user's alert store.

    Unauthenticated callers may generate transient advisories
    (store_alerts=False) for the climate simulator and public widgets but are
    subject to a rate limit to prevent unbounded advisory engine load.
    """
    # Rate-limit all callers (authenticated and anonymous alike) to prevent
    # unbounded rule-evaluation passes triggered by anonymous clients sending
    # large payloads at high frequency.
    from compute_rate_limit import enforce_compute_rate_limit
    rate_response = enforce_compute_rate_limit(
        request,
        scope="advisory",
        uid=None,
        limit=30,
        window_seconds=60,
    )
    if rate_response is not None:
        return rate_response

    alerts = generate_advisories(
        weather=payload.weather,
        soil=payload.soil,
        crop_type=payload.crop_type,
    )
    safe_alerts = _sanitise_alerts(alerts)

    stored = False
    if payload.store_alerts:
        # Derive uid from the verified token — never from the request body.
        uid = await _get_authenticated_uid(request)

        _stored_alerts.extend(uid, safe_alerts)
        stored = True

    return {
        "success": True,
        "data": safe_alerts,
        "count": len(alerts),
        "stored": stored,
    }


@router.post("/farm-intelligence/recommend")
async def create_farm_intelligence(payload: "FarmIntelligenceRequest", request: Request):
    uid = await _get_authenticated_uid(request)
    result = _build_farm_graph(payload)
    # Cap each user-supplied dict to at most _HISTORY_DICT_KEY_CAP entries before
    # writing to Firestore. The advisory engine only inspects a small set of known
    # keys, so truncating any excess is safe and prevents permanently storing
    # multi-kilobyte documents from over-sized request payloads.
    _HISTORY_DICT_KEY_CAP = 30
    # Fields that must be preserved even if they exceed the cap.
    # These are visibility-critical fields that users depend on.
    _PRESERVED_FIELDS = {"region_id", "farm_id", "user_id", "visibility"}

    def _cap_dict(d: dict) -> dict:
        if not isinstance(d, dict):
            return {}
        # Preserve critical fields, then fill remaining slots with other entries
        preserved = {k: v for k, v in d.items() if k in _PRESERVED_FIELDS}
        other_items = [
            (k, v) for k, v in d.items()
            if k not in _PRESERVED_FIELDS
        ][: _HISTORY_DICT_KEY_CAP - len(preserved)]
        result_dict = {**preserved, **dict(other_items)}
        return result_dict

    history_entry = {
        "uid": uid,
        "crop_type": payload.crop_type,
        "location": payload.location,
        "weather": _cap_dict(payload.weather),
        "soil": _cap_dict(payload.soil),
        "pest": _cap_dict(payload.pest),
        "market": _cap_dict(payload.market),
        "graph": result["graph"],
        "reasoning": result["reasoning"],
        "recommendations": result["recommendations"],
        "scores": result["scores"],
        "summary": result["summary"],
        "createdAt": _now_iso(),
    }

    history_id = _store_graph_history(uid, history_entry) if payload.store_history else uuid.uuid4().hex[:12]

    return {
        "success": True,
        "history_id": history_id,
        **result,
    }


@router.get("/advisory/me")
async def get_my_advisories(request: Request):
    """
    Return stored advisories for the authenticated caller.

    Authentication is required — a caller can only read their own stored
    alerts. The previous GET /advisory/{user_id} endpoint accepted any
    user_id as a path parameter with no token check, allowing any caller
    to read another user's alerts (IDOR).

    The endpoint is now /advisory/me so the caller's identity is always
    derived from the verified Firebase token, not from a URL parameter.
    """
    uid = await _get_authenticated_uid(request)

    data = _stored_alerts.get(uid)

    return {"success": True, "data": data}


@router.get("/farm-intelligence/me")
async def get_my_farm_intelligence(request: Request):
    uid = await _get_authenticated_uid(request)
    return {"success": True, "data": _load_graph_history(uid)}
