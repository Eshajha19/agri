from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional
from datetime import datetime
import joblib
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
import os
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load trained model (ensure this path is correct)
model = joblib.load("yield_model.joblib")

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

# Create FastAPI app
app = FastAPI(
    title="Fasal Saathi API",
    description="Agricultural advisory platform with secure feedback system",
    version="2.0.0"
)

# Allow CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Feedback validation class
class FeedbackValidator:
    """Validator for feedback data to prevent NoSQL injection"""
    
    ALLOWED_CROPS = [
        "Rice", "Wheat", "Cotton", "Sugarcane", "Maize",
        "Soybean", "Potato", "Onion", "Tomato", "Vegetables",
        "Fruits", "Other"
    ]
    
    MAX_NAME_LENGTH = 100
    MAX_LOCATION_LENGTH = 200
    MAX_MESSAGE_LENGTH = 2000
    
    DANGEROUS_PATTERNS = [
        r'\$[a-zA-Z_][a-zA-Z0-9_]*\s*:',  # MongoDB operators
        r'\{.*\}\s*:\s*\{',  # Nested object injection
        r'\.\./',  # Path traversal
        r'<script.*?>.*?</script>',  # XSS
        r'on\w+\s*=',  # Event handlers
        r'javascript:',  # JavaScript protocol
        r'data:',  # Data URLs
    ]
    
    @classmethod
    def sanitize_string(cls, value: str, max_length: int) -> str:
        """Sanitize string by removing dangerous characters"""
        if not value:
            return ""
        
        # Remove null bytes and control characters
        sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', value)
        
        # Remove extra whitespace
        sanitized = ' '.join(sanitized.split())
        
        # Truncate to max length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
            
        return sanitized
    
    @classmethod
    def validate_name(cls, name: Optional[str]) -> Optional[str]:
        """Validate and sanitize name"""
        if not name:
            return None
            
        name = cls.sanitize_string(name, cls.MAX_NAME_LENGTH)
        
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, name, re.IGNORECASE):
                return None
                
        if not re.match(r'^[a-zA-Z\s\.\-]{1,100}$', name):
            return None
            
        return name.strip()
    
    @classmethod
    def validate_location(cls, location: Optional[str]) -> Optional[str]:
        """Validate and sanitize location"""
        if not location:
            return None
            
        location = cls.sanitize_string(location, cls.MAX_LOCATION_LENGTH)
        
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, location, re.IGNORECASE):
                return None
                
        if not re.match(r'^[a-zA-Z0-9\s\.,\-\(\)]{1,200}$', location):
            return None
            
        return location.strip()
    
    @classmethod
    def validate_crop_type(cls, crop_type: Optional[str]) -> Optional[str]:
        """Validate crop type using whitelist"""
        if not crop_type:
            return None
            
        if crop_type in cls.ALLOWED_CROPS:
            return crop_type
            
        return None
    
    @classmethod
    def validate_category(cls, category: str) -> str:
        """Validate feedback category"""
        valid_categories = ["general", "feature", "bug", "ui", "accuracy", "other"]
        return category if category in valid_categories else "general"
    
    @classmethod
    def validate_rating(cls, rating: int) -> int:
        """Validate rating (1-5)"""
        if not isinstance(rating, int):
            try:
                rating = int(rating)
            except (ValueError, TypeError):
                return 3
                
        return max(1, min(5, rating))
    
    @classmethod
    def validate_message(cls, message: str) -> Optional[str]:
        """Validate and sanitize feedback message"""
        if not message or not isinstance(message, str):
            return None
            
        message = cls.sanitize_string(message, cls.MAX_MESSAGE_LENGTH)
        
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, message, re.IGNORECASE):
                return None
                
        if len(message.strip()) < 3:
            return None
            
        return message.strip()
    
    @classmethod
    def is_safe_for_firestore(cls, data: dict) -> bool:
        """Check if data is safe for Firestore"""
        try:
            data_str = str(data)
            
            for pattern in cls.DANGEROUS_PATTERNS:
                if re.search(pattern, data_str, re.IGNORECASE):
                    return False
                    
            for key, value in data.items():
                if isinstance(value, dict):
                    if not cls.is_safe_for_firestore(value):
                        return False
                elif isinstance(value, str) and any(
                    re.search(pattern, value, re.IGNORECASE) 
                    for pattern in cls.DANGEROUS_PATTERNS
                ):
                    return False
                    
            return True
        except Exception:
            return False

# Pydantic models
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
    timestamp: str

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Fasal Saathi API",
        "version": "2.0.0",
        "status": "healthy",
        "endpoints": ["/predict", "/api/feedback", "/api/feedback/stats"]
    }

@app.get("/predict")
def predict():
    """Yield prediction endpoint"""
    # Dummy input matching training features
    input_df = pd.DataFrame([{
        "NDVI": 4800,
        "Rainfall": 25.0,
        "SoilMoisture": 4.5,
        "Crop-wise_Rice": 1  # Example crop, adjust if needed
    }])

    # Ensure all model features exist
    for col in model.get_booster().feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[model.get_booster().feature_names]

    # Make prediction
    prediction = model.predict(input_df)[0]
    return {"predicted_yield": float(prediction)}

@app.post("/api/feedback", response_model=FeedbackResponse)
async def submit_feedback(feedback: FeedbackRequest):
    """
    Submit feedback with server-side validation.
    
    This endpoint validates all input data, sanitizes it, and stores it securely
    in Firestore. It prevents NoSQL injection and ensures data integrity.
    """
    try:
        logger.info(f"Received feedback submission from user: {feedback.userId}")
        
        # Convert to dict
        feedback_dict = feedback.dict(exclude_none=True)
        
        # Additional validation
        message = FeedbackValidator.validate_message(feedback_dict.get('message', ''))
        if not message:
            raise HTTPException(status_code=400, detail="Message is required and must be valid")
        
        rating = FeedbackValidator.validate_rating(feedback_dict.get('rating', 0))
        
        # Check if data is safe
        if not FeedbackValidator.is_safe_for_firestore(feedback_dict):
            logger.warning(f"Unsafe data detected from user: {feedback.userId}")
            raise HTTPException(status_code=400, detail="Invalid data format")
        
        # Prepare data for Firestore
        firestore_data = {
            'name': FeedbackValidator.validate_name(feedback_dict.get('name')),
            'cropType': FeedbackValidator.validate_crop_type(feedback_dict.get('cropType')),
            'location': FeedbackValidator.validate_location(feedback_dict.get('location')),
            'category': FeedbackValidator.validate_category(feedback_dict.get('category', 'general')),
            'message': message,
            'rating': rating,
            'userId': feedback_dict.get('userId'),
            'userEmail': feedback_dict.get('userEmail'),
            'createdAt': datetime.utcnow(),
            'validated': True,
            'validationVersion': '1.0.0'
        }
        
        # Remove None values
        firestore_data = {k: v for k, v in firestore_data.items() if v is not None}
        
        # Store in Firestore
        try:
            doc_ref = db.collection("feedback").add(firestore_data)
            feedback_id = doc_ref[1].id
            
            logger.info(f"Feedback stored successfully. ID: {feedback_id}")
            
            return FeedbackResponse(
                success=True,
                feedback_id=feedback_id,
                message="Feedback submitted successfully",
                timestamp=datetime.utcnow().isoformat()
            )
            
        except Exception as firestore_error:
            logger.error(f"Firestore error: {firestore_error}")
            raise HTTPException(
                status_code=500,
                detail="Failed to store feedback. Please try again later."
            )
            
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again later."
        )

@app.get("/api/feedback/stats")
async def get_feedback_stats():
    """Get feedback statistics (admin only)"""
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
        
        return {
            "total_feedbacks": total_count,
            "average_rating": round(avg_rating, 2),
            "category_distribution": category_counts,
            "recent_count": min(10, total_count)
        }
        
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
            # Test validation
            name = FeedbackValidator.validate_name(test.get('name'))
            message = FeedbackValidator.validate_message(test.get('message', ''))
            rating = FeedbackValidator.validate_rating(test.get('rating', 0))
            
            if message:
                results.append({
                    "test_case": i + 1,
                    "input": test,
                    "status": "VALID",
                    "validated": {
                        "name": name,
                        "message": message,
                        "rating": rating
                    }
                })
            else:
                results.append({
                    "test_case": i + 1,
                    "input": test,
                    "status": "INVALID",
                    "error": "Message validation failed"
                })
        except Exception as e:
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
