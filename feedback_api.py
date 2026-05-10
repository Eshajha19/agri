"""
Feedback API Endpoint
Provides secure server-side API for feedback submission with validation.
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
import os
import logging

# Import our validator
from feedback_validation import FeedbackValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Firebase Admin SDK
try:
    # Check for Firebase credentials
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
    firebase_admin.initialize_app()  # Default app for emulator

# Initialize Firestore
db = firestore.client()

# Pydantic models for request/response validation
class FeedbackRequest(BaseModel):
    """Request model for feedback submission"""
    name: Optional[str] = Field(None, max_length=100)
    cropType: Optional[str] = Field(None, max_length=50)
    location: Optional[str] = Field(None, max_length=200)
    category: str = Field("general", max_length=50)
    message: str = Field(..., max_length=2000)
    rating: int = Field(..., ge=1, le=5)
    userId: Optional[str] = Field(None, max_length=128)
    userEmail: Optional[str] = Field(None, max_length=254)
    
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
    
    @validator('userEmail')
    def validate_email(cls, v):
        if v and ('@' not in v or '.' not in v or len(v) > 254):
            raise ValueError("Invalid email format")
        return v


class FeedbackResponse(BaseModel):
    """Response model for feedback submission"""
    success: bool
    feedback_id: Optional[str] = None
    message: str
    validated_data: Optional[dict] = None
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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "https://fasal-saathi.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency for request validation
async def validate_request(request: Request) -> dict:
    """Validate incoming request for security"""
    # Check content type
    content_type = request.headers.get("content-type", "")
    if "application/json" not in content_type:
        raise HTTPException(status_code=415, detail="Unsupported media type")
    
    # Check request size
    content_length = request.headers.get("content-length", 0)
    if int(content_length) > 10240:  # 10KB max
        raise HTTPException(status_code=413, detail="Request too large")
    
    return {}


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Feedback API",
        "version": "1.0.0",
        "status": "healthy",
        "features": ["validation", "firestore_integration", "security"]
    }


@app.post("/api/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    feedback: FeedbackRequest,
    request: Request,
    validation: dict = Depends(validate_request)
):
    """
    Submit feedback with server-side validation.
    
    This endpoint validates all input data, sanitizes it, and stores it securely
    in Firestore. It prevents NoSQL injection and ensures data integrity.
    """
    try:
        logger.info(f"Received feedback submission from user: {feedback.userId}")
        
        # Convert Pydantic model to dict
        feedback_dict = feedback.dict(exclude_none=True)
        
        # Additional validation using our validator
        validated_data = FeedbackValidator.validate_feedback_data(feedback_dict)
        
        # Check if data is safe for Firestore
        if not FeedbackValidator.is_safe_for_firestore(validated_data):
            logger.warning(f"Unsafe data detected from user: {feedback.userId}")
            raise HTTPException(status_code=400, detail="Invalid data format")
        
        # Add timestamp
        validated_data['createdAt'] = datetime.utcnow()
        validated_data['ipAddress'] = request.client.host if request.client else None
        validated_data['userAgent'] = request.headers.get("user-agent", "")
        
        # Store in Firestore
        try:
            doc_ref = db.collection("feedback").add(validated_data)
            feedback_id = doc_ref[1].id
            
            logger.info(f"Feedback stored successfully. ID: {feedback_id}")
            
            return FeedbackResponse(
                success=True,
                feedback_id=feedback_id,
                message="Feedback submitted successfully",
                validated_data=validated_data,
                timestamp=datetime.utcnow().isoformat()
            )
            
        except Exception as firestore_error:
            logger.error(f"Firestore error: {firestore_error}")
            raise HTTPException(
                status_code=500,
                detail="Failed to store feedback. Please try again later."
            )
            
    except ValueError as ve:
        logger.warning(f"Validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again later."
        )


@app.get("/api/feedback/stats", response_model=FeedbackStatsResponse)
async def get_feedback_stats():
    """Get feedback statistics (admin only)"""
    try:
        # In production, add authentication/authorization here
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
            
            # Calculate statistics
            rating = data.get('rating', 0)
            total_rating += rating
            
            category = data.get('category', 'unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        total_count = len(feedbacks)
        avg_rating = total_rating / total_count if total_count > 0 else 0
        
        # Get recent feedbacks (last 10)
        recent = feedbacks[-10:] if len(feedbacks) > 10 else feedbacks
        
        return FeedbackStatsResponse(
            total_feedbacks=total_count,
            average_rating=round(avg_rating, 2),
            category_distribution=category_counts,
            recent_feedbacks=recent
        )
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")


@app.get("/api/feedback/validate-test")
async def validate_test():
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
    return {
        "success": False,
        "error": exc.detail,
        "status_code": exc.status_code
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return {
        "success": False,
        "error": "Internal server error",
        "status_code": 500
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)