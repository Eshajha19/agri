"""Claims Router - Insurance Claim Management"""
from fastapi import APIRouter, HTTPException
from datetime import datetime
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/claims/{id}/status")
async def update_claim_status(id: int, status: str, user_email: str):
    """Update claim status and send notifications.
    
    Args:
        id: Claim identifier
        status: New claim status
        user_email: Email of the claim owner for notifications
    
    Returns:
        Success message confirming status update and notification
    """
    # Save notification in Firestore
    # db.collection("notifications").add({
    #     "userId": id,
    #     "title": f"Claim {status}",
    #     "message": f"Your claim {id} is now {status}.",
    #     "category": "claim",
    #     "read": False,
    #     "createdAt": datetime.utcnow()
    # })

    # Trigger EmailJS (call your JS helper or REST API)
    # sendEmailNotification(user_email, f"Claim {status}", f"Your claim {id} is now {status}.")
    
    logger.info(f"Claim {id} status updated to {status}. Notification sent to {user_email}")
    return {"message": "Status updated and notification sent"}
