"""
Feedback Validation Module
Provides server-side validation for feedback data to prevent NoSQL injection and ensure data integrity.
"""

import re
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class FeedbackCategory(str, Enum):
    """Valid feedback categories"""
    GENERAL = "general"
    FEATURE = "feature"
    BUG = "bug"
    UI = "ui"
    ACCURACY = "accuracy"
    OTHER = "other"


class FeedbackValidator:
    """
    Validator for feedback data to prevent NoSQL injection and ensure data integrity.
    Implements strict input validation, sanitization, and schema enforcement.
    """
    
    # Allowed crop types (whitelist approach)
    ALLOWED_CROPS = [
        "Rice", "Wheat", "Cotton", "Sugarcane", "Maize",
        "Soybean", "Potato", "Onion", "Tomato", "Vegetables",
        "Fruits", "Other"
    ]
    
    # Maximum lengths for fields
    MAX_NAME_LENGTH = 100
    MAX_LOCATION_LENGTH = 200
    MAX_MESSAGE_LENGTH = 2000
    
    # Regex patterns for validation
    NAME_PATTERN = r'^[a-zA-Z\s\.\-]{1,100}$'
    LOCATION_PATTERN = r'^[a-zA-Z0-9\s\.,\-\(\)]{1,200}$'
    
    # Dangerous patterns to reject (NoSQL injection prevention)
    DANGEROUS_PATTERNS = [
        r'\$[a-zA-Z_][a-zA-Z0-9_]*\s*:',  # MongoDB operators like $set, $where
        r'\{.*\}\s*:\s*\{',  # Nested object injection
        r'\.\./',  # Path traversal
        r'<script.*?>.*?</script>',  # Script tags
        r'on\w+\s*=',  # Event handlers
        r'javascript:',  # JavaScript protocol
        r'data:',  # Data URLs
        r'vbscript:',  # VBScript
    ]
    
    @classmethod
    def sanitize_string(cls, value: str, max_length: int) -> str:
        """
        Sanitize a string by removing dangerous characters and truncating to max length.
        
        Args:
            value: Input string to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized string
        """
        if not value:
            return ""
        
        # Remove null bytes and control characters (except newline and tab)
        sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', value)
        
        # Remove extra whitespace
        sanitized = ' '.join(sanitized.split())
        
        # Truncate to max length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
            
        return sanitized
    
    @classmethod
    def validate_name(cls, name: Optional[str]) -> Optional[str]:
        """
        Validate and sanitize name field.
        
        Args:
            name: User's name (optional)
            
        Returns:
            Sanitized name or None if invalid
        """
        if not name:
            return None
            
        name = cls.sanitize_string(name, cls.MAX_NAME_LENGTH)
        
        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, name, re.IGNORECASE):
                return None
                
        # Basic name validation (allow letters, spaces, dots, hyphens)
        if not re.match(cls.NAME_PATTERN, name):
            return None
            
        return name.strip()
    
    @classmethod
    def validate_location(cls, location: Optional[str]) -> Optional[str]:
        """
        Validate and sanitize location field.
        
        Args:
            location: User's location (optional)
            
        Returns:
            Sanitized location or None if invalid
        """
        if not location:
            return None
            
        location = cls.sanitize_string(location, cls.MAX_LOCATION_LENGTH)
        
        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, location, re.IGNORECASE):
                return None
                
        # Basic location validation
        if not re.match(cls.LOCATION_PATTERN, location):
            return None
            
        return location.strip()
    
    @classmethod
    def validate_crop_type(cls, crop_type: Optional[str]) -> Optional[str]:
        """
        Validate crop type using whitelist approach.
        
        Args:
            crop_type: Crop type (optional)
            
        Returns:
            Validated crop type or None if invalid
        """
        if not crop_type:
            return None
            
        # Whitelist validation - only allow predefined crop types
        if crop_type in cls.ALLOWED_CROPS:
            return crop_type
            
        return None
    
    @classmethod
    def validate_category(cls, category: str) -> str:
        """
        Validate feedback category.
        
        Args:
            category: Feedback category
            
        Returns:
            Validated category or 'general' as default
        """
        try:
            # Use enum validation
            return FeedbackCategory(category).value
        except ValueError:
            # Default to general if invalid
            return FeedbackCategory.GENERAL.value
    
    @classmethod
    def validate_rating(cls, rating: int) -> int:
        """
        Validate rating (1-5).
        
        Args:
            rating: User rating (1-5)
            
        Returns:
            Validated rating (clamped to 1-5 range)
        """
        if not isinstance(rating, int):
            try:
                rating = int(rating)
            except (ValueError, TypeError):
                return 3  # Default rating
                
        # Clamp to valid range
        return max(1, min(5, rating))
    
    @classmethod
    def validate_message(cls, message: str) -> Optional[str]:
        """
        Validate and sanitize feedback message.
        
        Args:
            message: Feedback message (required)
            
        Returns:
            Sanitized message or None if invalid
        """
        if not message or not isinstance(message, str):
            return None
            
        message = cls.sanitize_string(message, cls.MAX_MESSAGE_LENGTH)
        
        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, message, re.IGNORECASE):
                return None
                
        # Message must have some content after sanitization
        if len(message.strip()) < 3:
            return None
            
        return message.strip()
    
    @classmethod
    def validate_feedback_data(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive validation of all feedback data.
        
        Args:
            data: Raw feedback data from client
            
        Returns:
            Validated and sanitized feedback data
            
        Raises:
            ValueError: If validation fails
        """
        validated = {}
        
        # Validate required fields
        message = cls.validate_message(data.get('message', ''))
        if not message:
            raise ValueError("Message is required and must be valid")
        validated['message'] = message
        
        # Validate rating
        rating = cls.validate_rating(data.get('rating', 0))
        validated['rating'] = rating
        
        # Validate category
        category = cls.validate_category(data.get('category', 'general'))
        validated['category'] = category
        
        # Validate optional fields
        name = cls.validate_name(data.get('name'))
        if name:
            validated['name'] = name
            
        location = cls.validate_location(data.get('location'))
        if location:
            validated['location'] = location
            
        crop_type = cls.validate_crop_type(data.get('cropType'))
        if crop_type:
            validated['cropType'] = crop_type
            
        # Add metadata
        validated['validatedAt'] = datetime.utcnow().isoformat()
        validated['validationVersion'] = '1.0.0'
        
        # Add user info if available
        if 'userId' in data:
            validated['userId'] = str(data['userId'])
        if 'userEmail' in data:
            # Basic email validation
            email = str(data['userEmail'])
            if '@' in email and '.' in email and len(email) <= 254:
                validated['userEmail'] = email
                
        return validated
    
    @classmethod
    def is_safe_for_firestore(cls, data: Dict[str, Any]) -> bool:
        """
        Check if data is safe for Firestore storage.
        
        Args:
            data: Data to check
            
        Returns:
            True if data is safe, False otherwise
        """
        try:
            # Convert to string for pattern checking
            data_str = str(data)
            
            # Check for dangerous patterns
            for pattern in cls.DANGEROUS_PATTERNS:
                if re.search(pattern, data_str, re.IGNORECASE):
                    return False
                    
            # Check for nested dangerous structures
            for key, value in data.items():
                if isinstance(value, dict):
                    # Recursively check nested dictionaries
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


# Example usage
if __name__ == "__main__":
    # Test the validator
    test_data = {
        "name": "John Doe",
        "cropType": "Rice",
        "location": "Nashik, Maharashtra",
        "category": "feature",
        "message": "Great app! Please add more crop varieties.",
        "rating": 5,
        "userId": "user123",
        "userEmail": "john@example.com"
    }
    
    try:
        validated = FeedbackValidator.validate_feedback_data(test_data)
        print("✅ Validation successful:")
        print(validated)
        print(f"✅ Safe for Firestore: {FeedbackValidator.is_safe_for_firestore(validated)}")
    except ValueError as e:
        print(f"❌ Validation failed: {e}")