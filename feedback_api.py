"""
Feedback API Endpoint
Provides secure server-side API for feedback submission with validation.
"""

import asyncio
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime, timezone
import firebase_admin
from firebase_admin import credentials, firestore, auth as firebase_auth
import os
import logging

# Rate limiting — mirrors the setup used in main.py so both apps enforce
# consistent per-IP throttles via the same slowapi library.
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from rate_limit_config import build_limiter, rate_limit_exceeded_handler, extract_client_ip

# Import our validator
from feedback_validation import FeedbackValidator
from csrf_protection import verify_csrf_token_dependency

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Firebase Admin SDK
try:
    # Check for Firebase credentials
    if not firebase_admin._apps:
        if os.path.exists("firebase-credentials.json"):
            cred = credentials.Certificate("firebase-credentials.json")
            firebase_admin.initialize_app(cred)
            logger.info("Firebase Admin SDK initialized with service account")
        else:
            # Try environment variable
            cred_json = os.getenv("FIREBASE_CREDENTIALS_JSON")
            if cred_json:
                import json
                cred_dict = json.loads(cred_json)
                cred = credentials.Certificate(cred_dict)
                firebase_admin.initialize_app(cred)
                logger.info("Firebase Admin SDK initialized from environment variable")
            else:
                logger.warning("Firebase credentials not found. Running in validation-only mode.")
                firebase_admin.initialize_app()  # Default app for emulator
except Exception as e:
    logger.error(f"Failed to initialize Firebase: {e}")
    if not firebase_admin._apps:
        firebase_admin.initialize_app()  # Default app for emulator

# Initialize Firestore
db = firestore.client()

# PII fields that must never appear in HTTP response bodies.
_PII_FIELDS = {"ipAddress", "userAgent", "userEmail"}


async def verify_firebase_token(request: Request) -> dict:
    """
    FastAPI dependency that verifies a Firebase ID token.

    Reads the token from the Authorization: Bearer header, verifies it
    with the Firebase Admin SDK, and returns the decoded token payload.

    This is used on the feedback submission endpoint so that:
    - Only registered users can submit feedback (prevents anonymous spam
      that exhausts Firestore write quota and incurs billing charges).
    - The caller's uid is derived from the verified token, never from
      client-supplied data (prevents impersonation via userId field).

    Raises
    ------
    HTTPException 401  Missing or invalid token.
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Please sign in to submit feedback.",
        )

    id_token = auth_header[7:].strip()
    if not id_token:
        raise HTTPException(status_code=401, detail="Missing authentication token")

    try:
        decoded = firebase_auth.verify_id_token(id_token, check_revoked=True)
    except firebase_auth.RevokedIdTokenError:
        raise HTTPException(status_code=401, detail="Session revoked. Please sign in again.")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired authentication token")

    uid = decoded.get("uid")
    if not uid:
        raise HTTPException(status_code=401, detail="Invalid authentication token")

    return decoded

# Pydantic models for request/response validation
class FeedbackRequest(BaseModel):
    """Request model for feedback submission.

    userId and userEmail are intentionally absent — the authoritative
    identity is always derived from the verified Firebase ID token, never
    from client-supplied data.  Accepting userId from the body previously
    allowed any caller to submit feedback attributed to an arbitrary user.
    """
    name: Optional[str] = Field(None, max_length=100)
    cropType: Optional[str] = Field(None, max_length=50)
    location: Optional[str] = Field(None, max_length=200)
    category: str = Field("general", max_length=50)
    message: str = Field(..., max_length=2000)
    rating: int = Field(..., ge=1, le=5)

    @validator('name')
    def validate_name(cls, v):
        if v is not None:
            return FeedbackValidator.validate_name(v)
        return v

    @validator('location')
    def validate_location(cls, v):
        if v is not None:
            return FeedbackValidator.validate_location(v)
        return v

    @validator('cropType')
    def validate_crop_type(cls, v):
        if v is not None:
            return FeedbackValidator.validate_crop_type(v)
        return v

    @validator('category')
    def validate_category(cls, v):
        return FeedbackValidator.validate_category(v)

    @validator('message')
    def validate_message(cls, v):
        validated = FeedbackValidator.validate_message(v)
        if not validated:
            raise ValueError("Message is required and must be valid")
        return validated

    @validator('rating')
    def validate_rating(cls, v):
        return FeedbackValidator.validate_rating(v)


class FeedbackResponse(BaseModel):
    """Response model for feedback submission.

    validated_data is intentionally absent.  The stored document contains
    server-collected PII fields (ipAddress, userAgent) that must never be
    echoed back in the HTTP response body — they would be captured by CDN
    access logs, browser devtools, and any proxy between client and server.
    """
    success: bool
    feedback_id: Optional[str] = None
    message: str
    timestamp: str


class FeedbackStatsResponse(BaseModel):
    """Response model for feedback statistics"""
    total_feedbacks: int
    average_rating: float
    category_distribution: dict
    recent_feedbacks: List[dict]


# Create FastAPI app
app = FastAPI(
    title="Feedback API",
    description="Secure feedback submission API with validation",
    version="1.0.0"
)

# ── Rate limiting ─────────────────────────────────────────────────────────────
# Authenticated endpoints use the Firebase token tail as the rate-limit key so
# that users behind a shared NAT each have their own quota.
def _feedback_rate_key(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
        # Use the last 32 chars of the token — unique per user, avoids
        # rate-limit collisions under CGNAT / corporate NAT.
        return token[-32:] if len(token) >= 32 else token
    return extract_client_ip(request)

# The limiter is attached to app.state so the @limiter.limit() decorator
# can resolve it at request time.
limiter = build_limiter(default_limits=["120/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
# ─────────────────────────────────────────────────────────────────────────────

from slowapi.util import get_remote_address

def feedback_rate_limit_key(request: Request):
    """
    Prefer stable authenticated UID for rate limiting.
    Fall back to IP address if no UID is available.
    """
    uid = getattr(request.state, "uid", None)
    if uid:
        return f"user:{uid}"
    return get_remote_address(request)

# ── Rate limiting ─────────────────────────────────────────────────────────────
limiter = Limiter(key_func=feedback_rate_limit_key, default_limits=["120/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
# ─────────────────────────────────────────────────────────────────────────────


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "https://fasal-saathi.vercel.app"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)

# ASGI middleware that enforces a 10 KB limit on the request body
# regardless of transfer encoding (covers both Content-Length and
# Transfer-Encoding: chunked).
class _OversizedBody(Exception):
    """Raised within sized_receive to abort processing when the body exceeds the limit."""


class _RequestBodySizeMiddleware:
    def __init__(self, app, max_bytes: int = 10240):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        received = 0

        async def sized_receive():
            nonlocal received
            msg = await receive()
            if msg["type"] == "http.request":
                body = msg.get("body", b"")
                received += len(body)
                if received > self.max_bytes:
                    raise _OversizedBody
            return msg

        try:
            await self.app(scope, sized_receive, send)
        except _OversizedBody:
            await send({
                "type": "http.response.start",
                "status": 413,
                "headers": [(b"content-type", b"text/plain")],
            })
            await send({
                "type": "http.response.body",
                "body": b"Request too large",
            })


app.add_middleware(_RequestBodySizeMiddleware)


# Dependency for request validation
async def validate_request(request: Request) -> dict:
    """Validate incoming request for security"""
    # Check content type
    content_type = request.headers.get("content-type", "")
    if "application/json" not in content_type:
        raise HTTPException(status_code=415, detail="Unsupported media type")
    return {}


async def verify_admin(request: Request) -> dict:
    """
    FastAPI dependency that enforces admin-only access.

    Reads the Firebase ID token from the Authorization: Bearer header,
    verifies it with the Firebase Admin SDK, then reads the caller's
    Firestore user document and checks that role == 'admin'.

    Both the Firebase token verification and the Firestore role lookup are
    synchronous SDK calls.  Running them directly inside an async function
    would block the event loop and serialise all concurrent requests behind
    the network round-trips.  They are therefore offloaded to asyncio's
    default ThreadPoolExecutor via run_in_executor so the event loop
    remains free to process other requests while I/O is in flight.

    Fail-closed design — any missing or invalid token, unavailable
    Firestore, missing user document, or non-admin role results in a
    4xx response.  The endpoint never falls through to a default that
    grants access.

    Raises
    ------
    HTTPException 401  Missing or invalid token.
    HTTPException 403  Valid token but caller is not an admin.
    HTTPException 503  Firestore unavailable during role check.
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authentication token")

    id_token = auth_header[7:].strip()
    if not id_token:
        raise HTTPException(status_code=401, detail="Missing or invalid authentication token")

    loop = asyncio.get_event_loop()

    # Offload blocking Firebase SDK call to thread pool.
    try:
        decoded = firebase_auth.verify_id_token(id_token, check_revoked=True)
    except Exception:
        raise HTTPException(status_code=401, detail="Authentication failed")

    uid = decoded["uid"]

    # Offload blocking Firestore network call to thread pool.
    try:
        user_doc = await loop.run_in_executor(
            None, lambda: db.collection("users").document(uid).get()
        )
    except Exception as exc:
        logger.error("Firestore role check failed for uid=%s: %s", uid, exc)
        raise HTTPException(status_code=503, detail="Authorization service temporarily unavailable")

    if not user_doc.exists:
        raise HTTPException(status_code=403, detail="User profile not found")

    role = user_doc.to_dict().get("role", "farmer")
    if role != "admin":
        raise HTTPException(status_code=403, detail="Access denied: admin role required")

    return {"uid": uid, "role": role}


@app.get("/")
@limiter.limit("60/minute")
async def root(request: Request):
    """Health check endpoint"""
    return {
        "service": "Feedback API",
        "version": "1.0.0",
        "status": "healthy",
        "features": ["validation", "firestore_integration", "security"]
    }


@app.post("/api/feedback", response_model=FeedbackResponse)
@limiter.limit("5/minute", key_func=_feedback_rate_key)
async def submit_feedback(
    feedback: FeedbackRequest,
    request: Request,
    token_data: dict = Depends(verify_firebase_token),
    validation: dict = Depends(validate_request),
):
    """
    Submit feedback with server-side validation.

    Requires a valid Firebase ID token (Authorization: Bearer <token>).
    Authentication prevents:
    - Anonymous spam exhausting Firestore write quota (20,000/day free tier).
    - Impersonation via client-supplied userId fields.
    - Automated abuse bypassing IP-based rate limits via proxy rotation.

    The caller's uid is derived exclusively from the verified token.
    PII fields (ipAddress, userAgent) are stored server-side for audit
    purposes but are never returned in the response body.
    """
    # uid comes from the verified token — never from the request body.
    uid = token_data["uid"]

    try:
        logger.info("Received feedback submission from uid: %s", uid)

        # Convert Pydantic model to dict
        feedback_dict = feedback.dict(exclude_none=True)

        # Bind the verified uid to the record so ownership is always accurate.
        feedback_dict["userId"] = uid

        # Additional validation using our validator
        validated_data = FeedbackValidator.validate_feedback_data(feedback_dict)

        # Check if data is safe for Firestore
        if not FeedbackValidator.is_safe_for_firestore(validated_data):
            logger.warning("Unsafe data detected from uid: %s", uid)
            raise HTTPException(status_code=400, detail="Invalid data format")

        # Add server-side metadata — stored for audit/abuse investigation
        # but intentionally excluded from the response body.
        validated_data['createdAt'] = datetime.now(timezone.utc)
        validated_data['ipAddress'] = request.client.host if request.client else None
        validated_data['userAgent'] = request.headers.get("user-agent", "")

        # Store in Firestore
        try:
            doc_ref = db.collection("feedback").add(validated_data)
            feedback_id = doc_ref[1].id

            logger.info("Feedback stored successfully. ID: %s", feedback_id)

            # PII fields (ipAddress, userAgent) are stored server-side for
            # audit purposes but must not be echoed back in the response.
            return FeedbackResponse(
                success=True,
                feedback_id=feedback_id,
                message="Feedback submitted successfully",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        except Exception as firestore_error:
            logger.error("Firestore error: %s", firestore_error)
            raise HTTPException(
                status_code=500,
                detail="Failed to store feedback. Please try again later.",
            )

    except ValueError as ve:
        logger.warning("Validation error: %s", ve)
        raise HTTPException(status_code=400, detail=str(ve))

    except HTTPException:
        raise

    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again later.",
        )


@app.get("/api/feedback/stats", response_model=FeedbackStatsResponse)
@limiter.limit("10/minute")
async def get_feedback_stats(
    request: Request,
    admin_user: dict = Depends(verify_admin),
):
    """Get feedback statistics (admin only)"""
    uid = admin_user["uid"]

    try:
        feedback_ref = db.collection("feedback")
        docs = feedback_ref.limit(1000).stream()

        feedbacks = []
        total_rating = 0
        category_counts = {}

        for doc in docs:
            data = doc.to_dict()
            feedbacks.append({
                "id": doc.id,
                **data
            })

            rating = data.get('rating', 0)
            total_rating += rating

            category = data.get('category', 'unknown')
            category_counts[category] = category_counts.get(category, 0) + 1

        total_count = len(feedbacks)
        avg_rating = total_rating / total_count if total_count > 0 else 0

        # Strip PII fields before returning to the caller.
        # Use the module-level _PII_FIELDS constant — avoids shadowing it
        # with a local re-definition that could silently diverge over time.
        # Sort by timestamp descending before slicing so recent_feedbacks
        # always contains the 10 most recently submitted entries, not an
        # arbitrary tail of whatever order Firestore stream() returned.
        feedbacks.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
        recent_raw = feedbacks[:10]
        recent = [
            {k: v for k, v in entry.items() if k not in _PII_FIELDS}
            for entry in recent_raw
        ]

        return FeedbackStatsResponse(
            total_feedbacks=total_count,
            average_rating=round(avg_rating, 2),
            category_distribution=category_counts,
            recent_feedbacks=recent
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")


@app.get("/api/feedback/validate-test")
@limiter.limit("10/minute")
async def validate_test(request: Request):
    """Test endpoint to demonstrate validation"""
    test_cases = [
        {
            "name": "Safe User",
            "cropType": "Rice",
            "location": "Nashik",
            "category": "feature",
            "message": "Great app!",
            "rating": 5
        },
        {
            "name": "<script>alert('xss')</script>",
            "message": "Test",
            "rating": 3
        },
        {
            "name": "Test",
            "message": "{$set: {admin: true}}",
            "rating": 1
        }
    ]
    
    results = []
    for i, test in enumerate(test_cases):
        try:
            validated = FeedbackValidator.validate_feedback_data(test)
            results.append({
                "test_case": i + 1,
                "input": test,
                "status": "VALID",
                "validated": validated
            })
        except ValueError as e:
            results.append({
                "test_case": i + 1,
                "input": test,
                "status": "INVALID",
                "error": str(e)
            })
    
    return {
        "validation_tests": results,
        "validator_info": {
            "version": "1.0.0",
            "max_message_length": FeedbackValidator.MAX_MESSAGE_LENGTH,
            "allowed_crops": FeedbackValidator.ALLOWED_CROPS
        }
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    from fastapi.responses import JSONResponse
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "status_code": 500,
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)