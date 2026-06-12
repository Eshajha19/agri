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
import logging
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field, validator

router = APIRouter()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-process stores (thread-safe via a single RLock)
# ---------------------------------------------------------------------------
_lock = threading.RLock()
_listings: Dict[str, Dict[str, Any]] = {}
_bookings: Dict[str, Dict[str, Any]] = {}

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

def _seed_listings() -> None:
    with _lock:
        if _listings:
            return
        for item in _SEED_LISTINGS:
            lid = str(uuid.uuid4())
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
                "createdAt": datetime.utcnow().isoformat() + "Z",
            }

_seed_listings()

BOOKING_TTL = timedelta(hours=24)


def _release_expired_bookings() -> None:
    """Release listings whose pending bookings have exceeded the TTL."""
    now = datetime.now(timezone.utc)
    cutoff = now - BOOKING_TTL
    expired_bids = []
    for bid, b in list(_bookings.items()):
        if b["status"] != "pending":
            continue
        created = b.get("createdAt", "")
        if not created:
            continue
        try:
            created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            if created_dt < cutoff:
                expired_bids.append(bid)
        except Exception:
            continue
    for bid in expired_bids:
        booking = _bookings.pop(bid, None)
        if booking:
            eq_id = booking["equipmentId"]
            listing = _listings.get(eq_id)
            if listing and listing.get("ownerUid") is None:
                # Seed listings with no owner revert to available.
                listing["available"] = True
                logger.info(
                    "Released expired booking %s for equipment %s", bid, eq_id
                )

# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------
verify_role_fn = None

def init_marketplace(vr_fn) -> None:
    global verify_role_fn
    verify_role_fn = vr_fn

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

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/listings")
async def get_listings(
    search: str = Query(default="", max_length=100),
    location: str = Query(default="", max_length=100),
    type: Optional[str] = Query(default=None, max_length=50),
    available_only: bool = Query(default=False),
):
    """Return all equipment listings.  Public — no auth required."""
    with _lock:
        _release_expired_bookings()
        items = list(_listings.values())

    search_lower = search.lower()
    location_lower = location.lower()

    results = []
    for item in items:
        if search_lower and search_lower not in item["name"].lower():
            continue
        if location_lower and location_lower not in item["location"].lower():
            continue
        if type and item["type"] != type:
            continue
        if available_only and not item["available"]:
            continue
        results.append(item)

    results.sort(key=lambda x: x["createdAt"], reverse=True)
    return {"success": True, "data": results}


@router.post("/listings")
async def list_equipment(request: Request, data: ListEquipmentRequest):
    """Register a new equipment listing.  Requires authentication."""
    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Not initialized")

    token_data = await verify_role_fn(request)
    uid = token_data["uid"]

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
        "createdAt": datetime.utcnow().isoformat() + "Z",
    }

    with _lock:
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
    booker_uid = token_data["uid"]

    bid = str(uuid.uuid4())
    booking = None

    # Hold the lock for the entire check-then-act sequence so no concurrent
    # request can slip in between the availability read and the booking write.
    with _lock:
        _release_expired_bookings()
        listing = _listings.get(data.equipmentId)

        if listing is None:
            raise HTTPException(status_code=404, detail="Equipment listing not found")
        if not listing["available"]:
            raise HTTPException(status_code=409, detail="Equipment is not available for booking")

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
            "createdAt": datetime.utcnow().isoformat() + "Z",
        }

        _bookings[bid] = booking
        # Mark the listing as unavailable while a booking is pending.
        _listings[data.equipmentId] = {**listing, "available": False}

    logger.info(
        "Booking %s created: equipment=%s booker=%s date=%s",
        bid, data.equipmentId, booker_uid, data.date,
    )
    return {"success": True, "booking": booking}


@router.get("/bookings")
async def get_bookings(request: Request):
    """Return bookings for the authenticated user (as booker or owner)."""
    if verify_role_fn is None:
        raise HTTPException(status_code=500, detail="Not initialized")

    token_data = await verify_role_fn(request)
    uid = token_data["uid"]

    with _lock:
        all_bookings = list(_bookings.values())

    user_bookings = [
        b for b in all_bookings
        if b["bookerUid"] == uid or b["ownerUid"] == uid
    ]
    user_bookings.sort(key=lambda x: x["createdAt"], reverse=True)
    return {"success": True, "data": user_bookings}
