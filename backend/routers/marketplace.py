"""P2P Agri-Equipment Marketplace Router.

All equipment listings and booking requests are persisted server-side so
they survive page refreshes, are visible across devices, and cannot be
fabricated by editing client-side state.

Storage strategy
----------------
The module uses two in-process dicts (_listings, _bookings) as the
persistence layer.  This is intentional for the current deployment
(single-process FastAPI on a single server) and matches the pattern used
by the rest of the codebase (SupplyChainBlockchain, NotificationStore,
etc.).  A Firestore or SQL backend can be swapped in later by replacing
the dict operations in the helper functions below.

Authentication
--------------
- GET  /marketplace/listings  — public (consumers can browse)
- POST /marketplace/listings  — requires auth (farmer must be logged in)
- POST /marketplace/bookings  — requires auth (farmer must be logged in)
- GET  /marketplace/bookings  — requires auth (returns caller's bookings)
"""
import hashlib
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field, validator, ValidationError

from backend.core.logging_config import setup_logging

router = APIRouter()
logger = setup_logging(__name__)

# ---------------------------------------------------------------------------
# Marketplace Filter Validation
# ---------------------------------------------------------------------------

class MarketplaceFilters(BaseModel):
    """Centralized validation for marketplace listing query filters."""
    search: str = Field(default="", max_length=100)
    location: str = Field(default="", max_length=100)
    type: Optional[str] = Field(default=None, max_length=50)
    available_only: bool = Field(default=False)
    page: int = Field(default=1, ge=1)
    limit: int = Field(default=20, ge=1, le=100)

    @validator("search", "location", "type", pre=True)
    def strip_and_validate_filters(cls, v):
        """Strip whitespace and reject injection patterns."""
        if v is None or v == "":
            return ""
        
        if not isinstance(v, str):
            raise ValueError("Filter values must be strings")
        
        v = v.strip()
        
        # Reject common injection patterns
        injection_patterns = ["<script", "javascript:", "onerror=", "onclick=", "{%", "${"]
        v_lower = v.lower()
        for pattern in injection_patterns:
            if pattern in v_lower:
                raise ValueError(f"Invalid characters detected in filter: {pattern}")
        
        return v

    @validator("type")
    def validate_equipment_type(cls, v):
        """Validate equipment type against allowed values."""
        if v is None or v == "":
            return None
        
        valid_types = {"Tractor", "Harvester", "Drone", "Tillage", "Sowing", "Other"}
        if v not in valid_types:
            raise ValueError(f"Equipment type must be one of: {', '.join(valid_types)}")
        
        return v


def validate_listing_filters(
    search: str = Query(default="", max_length=100),
    location: str = Query(default="", max_length=100),
    type: Optional[str] = Query(default=None, max_length=50),
    available_only: bool = Query(default=False),
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
) -> MarketplaceFilters:
    """Dependency function to validate marketplace filters centrally.
    
    Validates all query parameters through a Pydantic model to ensure:
    - Type coercion and bounds checking (page >= 1, limit <= 100)
    - String length limits (search/location <= 100 chars)
    - Equipment type against allowed values
    - Injection pattern detection and rejection
    - Consistent error responses
    """
    try:
        filters = MarketplaceFilters(
            search=search,
            location=location,
            type=type,
            available_only=available_only,
            page=page,
            limit=limit,
        )
        return filters
    except ValidationError as exc:
        logger.warning("Filter validation failed: %s", exc.errors())
        # Build user-friendly error message
        errors = exc.errors()
        first_error = errors[0] if errors else {}
        field = first_error.get("loc", ("unknown",))[0]
        msg = first_error.get("msg", "Invalid filter value")
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_filter",
                "message": f"Invalid {field}: {msg}",
                "errors": errors,
            },
        )

# ---------------------------------------------------------------------------
# In-process stores (thread-safe via a single RLock)
# ---------------------------------------------------------------------------
_lock = threading.RLock()
_listings: Dict[str, Dict[str, Any]] = {}
_bookings: Dict[str, Dict[str, Any]] = {}

# Caps that prevent a single authenticated user from exhausting the
# in-process store and causing denial of service for other users.
_MAX_TOTAL_LISTINGS = 1000
_MAX_LISTINGS_PER_USER = 10

# Seed with the 16 canonical listings so the marketplace is not empty on
# first boot.  These are stored server-side — the frontend no longer
# carries INITIAL_EQUIPMENT.
_SEED_LISTINGS = [
    {"name": "John Deere Tractor 5050D",    "type": "Tractor",   "price": 800,  "priceUnit": "hr",  "location": "Karnal, Haryana",       "owner": "Suresh Kumar",        "available": True},
    {"name": "Mahindra Rice Harvester",      "type": "Harvester", "price": 2500, "priceUnit": "day", "location": "Ludhiana, Punjab",       "owner": "Hardeep Singh",       "available": True},
    {"name": "DJI Agras T40 Drone",          "type": "Drone",     "price": 1500, "priceUnit": "hr",  "location": "Bhopal, MP",            "owner": "TechAgri Solutions",  "available": False},
    {"name": "Sonalika Rotavator 200",        "type": "Tillage",   "price": 400,  "priceUnit": "hr",  "location": "Nagpur, Maharashtra",    "owner": "Ramesh Patil",        "available": True},
    {"name": "Kubota MU5501 Tractor",         "type": "Tractor",   "price": 900,  "priceUnit": "hr",  "location": "Pune, Maharashtra",      "owner": "Santosh Shinde",      "available": True},
    {"name": "Mahindra JIVO 245 DI",          "type": "Tractor",   "price": 750,  "priceUnit": "hr",  "location": "Nashik, Maharashtra",    "owner": "Ganesh Deshmukh",     "available": True},
    {"name": "Preet 987 Combine Harvester",   "type": "Harvester", "price": 3200, "priceUnit": "day", "location": "Amravati, Maharashtra",  "owner": "Vijay Bhalerao",      "available": True},
    {"name": "AgriDrone Sprayer X8",          "type": "Drone",     "price": 1200, "priceUnit": "hr",  "location": "Pune, Maharashtra",      "owner": "DroneAgri Pvt Ltd",   "available": True},
    {"name": "New Holland Excel 4710",        "type": "Tractor",   "price": 1100, "priceUnit": "hr",  "location": "Mumbai, Maharashtra",    "owner": "Pramod Jadhav",       "available": False},
    {"name": "Swaraj 744 FE Tractor",         "type": "Tractor",   "price": 700,  "priceUnit": "hr",  "location": "Solapur, Maharashtra",   "owner": "Arjun Kulkarni",      "available": True},
    {"name": "Fieldking Offset Disc Harrow",  "type": "Tillage",   "price": 350,  "priceUnit": "hr",  "location": "Aurangabad, Maharashtra", "owner": "Sunil Mane",          "available": True},
    {"name": "VST Shakti Tractor 270 DI",     "type": "Tractor",   "price": 600,  "priceUnit": "hr",  "location": "Bengaluru, Karnataka",   "owner": "Ravi Naik",           "available": True},
    {"name": "CLAAS Crop Tiger 30 Terra",      "type": "Harvester", "price": 4000, "priceUnit": "day", "location": "Hyderabad, Telangana",   "owner": "Krishnamurthy Agro",  "available": True},
    {"name": "Ecorobotix ARA Sprayer",        "type": "Drone",     "price": 1800, "priceUnit": "hr",  "location": "Jaipur, Rajasthan",      "owner": "SmartFarm Solutions", "available": True},
    {"name": "Landforce Potato Planter",      "type": "Sowing",    "price": 500,  "priceUnit": "hr",  "location": "Agra, UP",               "owner": "Dinesh Agarwal",      "available": True},
    {"name": "Massey Ferguson 9500",          "type": "Tractor",   "price": 950,  "priceUnit": "hr",  "location": "Patna, Bihar",           "owner": "Manoj Singh",         "available": False},
]

def _stable_seed_id(item: dict) -> str:
    """Derive a deterministic listing ID from the seed item's content.

    Using uuid.uuid4() produced a different ID on every process start,
    so each Uvicorn worker seeded the same 16 logical listings with
    different UUIDs.  A client that bookmarked a listing ID from worker A
    would get a 404 from worker B.

    The ID is derived from the item name (unique across seed data) via
    SHA-256 so every worker always produces the same stable ID for the
    same logical listing, regardless of restart order or worker count.
    """
    digest = hashlib.sha256(item["name"].encode("utf-8")).hexdigest()
    # Format as a UUID-shaped string so it is indistinguishable from
    # user-created listing IDs in API responses.
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def _seed_listings() -> None:
    with _lock:
        if _listings:
            return
        for item in _SEED_LISTINGS:
            lid = _stable_seed_id(item)
            _listings[lid] = {
                "id": lid,
                "name": item["name"],
                "type": item["type"],
                "price": item["price"],
                "priceUnit": item["priceUnit"],
                "location": item["location"],
                "owner": item["owner"],
                "ownerUid": None,   # seed listings have no registered owner
                "available": item["available"],
                "rating": 4.5,
                "createdAt": datetime.now(timezone.utc).isoformat(),
            }

_seed_listings()

# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------
verify_role_fn = None
notify_booking_fn = None

def init_marketplace(vr_fn, notify_fn=None) -> None:
    global verify_role_fn, notify_booking_fn
    verify_role_fn = vr_fn
    notify_booking_fn = notify_fn

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ListEquipmentRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    type: str = Field(..., pattern=r"^(Tractor|Harvester|Drone|Tillage|Sowing|Other)$")
    price: float = Field(..., gt=0, le=1_000_000)
    priceUnit: str = Field(..., pattern=r"^(hr|day)$")
    location: str = Field(..., min_length=2, max_length=200)

    @validator("name", "location", pre=True)
    def strip_whitespace(cls, v):
        return v.strip() if isinstance(v, str) else v


class BookEquipmentRequest(BaseModel):
    equipmentId: str = Field(..., min_length=1, max_length=100)
    date: str = Field(..., min_length=8, max_length=20)
    time: str = Field(..., min_length=4, max_length=10)
    duration: int = Field(..., ge=1, le=365)

    @validator("date")
    def validate_date(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("date must be in YYYY-MM-DD format")
        return v


class UpdateBookingRequest(BaseModel):
    status: str = Field(..., pattern=r"^(confirmed|completed|cancelled)$")

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/listings")
async def get_listings(filters: MarketplaceFilters = Query(...) if False else None, 
                       search: str = Query(default="", max_length=100),
                       location: str = Query(default="", max_length=100),
                       type: Optional[str] = Query(default=None, max_length=50),
                       available_only: bool = Query(default=False),
                       page: int = Query(default=1, ge=1, description="1-based page number"),
                       limit: int = Query(default=20, ge=1, le=100, description="Results per page (max 100)")):
    """Return equipment listings with pagination.  Public -- no auth required.

    Pagination prevents large in-memory copies and unbounded JSON payloads
    when the listing store is large or has been flooded.
    
    All query filters are validated centrally through MarketplaceFilters model
    to ensure type safety, bounds checking, and injection attack prevention.
    """
    # Validate filters through centralized model
    try:
        validated_filters = MarketplaceFilters(
            search=search,
            location=location,
            type=type,
            available_only=available_only,
            page=page,
            limit=limit,
        )
    except ValidationError as exc:
        logger.warning("Filter validation failed: %s", exc.errors())
        errors = exc.errors()
        first_error = errors[0] if errors else {}
        field = first_error.get("loc", ("unknown",))[0]
        msg = first_error.get("msg", "Invalid filter value")
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_filter",
                "message": f"Invalid {field}: {msg}",
                "errors": errors,
            },
        )
    
    with _lock:
        items = list(_listings.values())

    search_lower = validated_filters.search.lower()
    location_lower = validated_filters.location.lower()

    results = []
    for item in items:
        if search_lower and search_lower not in item["name"].lower():
            continue
        if location_lower and location_lower not in item["location"].lower():
            continue
        if validated_filters.type and item["type"] != validated_filters.type:
            continue
        if validated_filters.available_only and not item["available"]:
            continue
        results.append(item)

    results.sort(key=lambda x: x["createdAt"], reverse=True)

    total = len(results)
    start = (validated_filters.page - 1) * validated_filters.limit
    page_data = results[start : start + validated_filters.limit]

    return {
        "success": True,
        "data": page_data,
        "pagination": {
            "page": validated_filters.page,
            "limit": validated_filters.limit,
            "total": total,
            "pages": max(1, -(-total // validated_filters.limit)),  # ceiling division
        },
    }


@router.post("/listings")
async def list_equipment(request: Request, data: ListEquipmentRequest):
    """Register a new equipment listing.  Requires authentication."""
    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Not initialized")

    token_data = await verify_role_fn(request)
    uid = token_data.get("uid")

    lid = str(uuid.uuid4())
    listing = {
        "id": lid,
        "name": data.name,
        "type": data.type,
        "price": data.price,
        "priceUnit": data.priceUnit,
        "location": data.location,
        "owner": "Farmer",   # display name; could be enriched from Firestore
        "ownerUid": uid,
        "available": True,
        "rating": 5.0,
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }

    with _lock:
        # Global cap: prevent unbounded heap growth from listing floods.
        if len(_listings) >= _MAX_TOTAL_LISTINGS:
            raise HTTPException(
                status_code=503,
                detail="Marketplace capacity reached. Please try again later.",
            )
        # Per-user cap: prevent a single user from monopolising the store.
        user_listing_count = sum(
            1 for lst in _listings.values() if lst.get("ownerUid") == uid
        )
        if user_listing_count >= _MAX_LISTINGS_PER_USER:
            raise HTTPException(
                status_code=429,
                detail=f"You have reached the maximum of {_MAX_LISTINGS_PER_USER} active listings.",
            )
        _listings[lid] = listing

    logger.info("New equipment listing %s by uid=%s", lid, uid)
    return {"success": True, "listing": listing}


@router.post("/bookings")
async def book_equipment(request: Request, data: BookEquipmentRequest):
    """Submit a booking request for a piece of equipment.  Requires auth.

    The booking is persisted server-side so:
    - The equipment owner can see it (GET /marketplace/bookings?owner=true).
    - The farmer who booked can see their own bookings.
    - The booking survives page refreshes and device changes.

    The availability check and the booking write are performed inside a
    single lock acquisition so no concurrent request can observe
    available=True and create a second booking for the same equipment
    between the check and the write (double-booking race condition).
    """
    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Not initialized")

    token_data = await verify_role_fn(request)
    booker_uid = token_data.get("uid")

    bid = str(uuid.uuid4())
    booking = None

    # Hold the lock for the entire check-then-act sequence so no concurrent
    # request can slip in between the availability read and the booking write.
    with _lock:
        listing = _listings.get(data.equipmentId)

        if listing is None:
            raise HTTPException(status_code=404, detail="Equipment listing not found")
        if not listing["available"]:
            raise HTTPException(status_code=409, detail="Equipment is not available for booking")
        if listing.get("ownerUid") is not None and listing["ownerUid"] == booker_uid:
            raise HTTPException(status_code=400, detail="You cannot book your own equipment listing")

        booking = {
            "id": bid,
            "equipmentId": data.equipmentId,
            "equipmentName": listing["name"],
            "equipmentType": listing["type"],
            "ownerUid": listing["ownerUid"],
            "ownerName": listing["owner"],
            "bookerUid": booker_uid,
            "date": data.date,
            "time": data.time,
            "duration": data.duration,
            "priceUnit": listing["priceUnit"],
            "totalCost": listing["price"] * data.duration,
            "status": "pending",   # pending → confirmed → completed / cancelled
            "createdAt": datetime.now(timezone.utc).isoformat(),
        }

        # Mark the listing unavailable BEFORE inserting the booking so that
        # if the process crashes between the two writes the listing is already
        # locked and a concurrent request cannot observe available=True while
        # a booking record for the same equipment already exists.
        _listings[data.equipmentId] = {**listing, "available": False}
        _bookings[bid] = booking

    logger.info(
        "Booking %s created: equipment=%s booker=%s date=%s",
        bid, data.equipmentId, booker_uid, data.date,
    )

    # Notify the equipment owner asynchronously.
    if notify_booking_fn is not None:
        try:
            await notify_booking_fn(booking)
        except Exception as exc:
            logger.warning("Booking notification failed for %s: %s", bid, exc)

    return {"success": True, "booking": booking}


@router.get("/bookings")
async def get_bookings(request: Request):
    """Return bookings for the authenticated user (as booker or owner)."""
    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Not initialized")

    token_data = await verify_role_fn(request)
    uid = token_data.get("uid")

    with _lock:
        all_bookings = list(_bookings.values())

    user_bookings = [
        b for b in all_bookings
        if b["bookerUid"] == uid or b["ownerUid"] == uid
    ]
    user_bookings.sort(key=lambda x: x["createdAt"], reverse=True)
    return {"success": True, "data": user_bookings}


# ---------------------------------------------------------------------------
# Allowed status transitions
# ---------------------------------------------------------------------------
_VALID_TRANSITIONS: dict[str, set[str]] = {
    # Equipment owner confirms a pending booking.
    "pending":   {"confirmed", "cancelled"},
    # Owner marks job done; either party may cancel a confirmed booking.
    "confirmed": {"completed", "cancelled"},
    # Terminal states — no further transitions.
    "completed": set(),
    "cancelled":  set(),
}

# The new status values that release the equipment back to available.
_RELEASES_EQUIPMENT: set[str] = {"completed", "cancelled"}


@router.patch("/bookings/{booking_id}")
async def update_booking_status(
    booking_id: str,
    data: UpdateBookingRequest,
    request: Request,
):
    """Transition a booking to a new status.

    Permission rules:
    - ``confirmed``: only the equipment owner may confirm a pending booking.
    - ``completed``: only the equipment owner may mark a booking completed.
    - ``cancelled``: the booker or the equipment owner may cancel at any point
      before completion.

    When a booking moves to ``completed`` or ``cancelled`` the equipment
    listing is restored to ``available=True`` so it can be rebooked.
    """
    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Not initialized")

    token_data = await verify_role_fn(request)
    uid = token_data.get("uid")
    new_status = data.status

    with _lock:
        booking = _bookings.get(booking_id)
        if booking is None:
            raise HTTPException(status_code=404, detail="Booking not found")

        # Authorisation check — only parties to the booking may update it.
        is_owner = booking.get("ownerUid") == uid
        is_booker = booking.get("bookerUid") == uid
        if not is_owner and not is_booker:
            raise HTTPException(status_code=403, detail="Not authorised to update this booking")

        current_status = booking["status"]
        allowed = _VALID_TRANSITIONS.get(current_status, set())
        if new_status not in allowed:
            raise HTTPException(
                status_code=409,
                detail=f"Cannot transition from '{current_status}' to '{new_status}'",
            )

        # Only the equipment owner may confirm or complete.
        if new_status in {"confirmed", "completed"} and not is_owner:
            raise HTTPException(
                status_code=403,
                detail="Only the equipment owner may confirm or complete a booking",
            )

        booking = {**booking, "status": new_status, "updatedAt": datetime.now(timezone.utc).isoformat()}
        _bookings[booking_id] = booking

        # Release the equipment back to available when the booking ends.
        if new_status in _RELEASES_EQUIPMENT:
            equipment_id = booking.get("equipmentId")
            if equipment_id and equipment_id in _listings:
                _listings[equipment_id] = {**_listings[equipment_id], "available": True}

    logger.info(
        "Booking %s transitioned %s -> %s by uid=%s",
        booking_id, current_status, new_status, uid,
    )
    return {"success": True, "booking": booking}
