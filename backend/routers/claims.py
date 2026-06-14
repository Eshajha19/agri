@app.post("/api/claims/{id}/status")
async def update_claim_status(id: int, status: str, user_email: str):
    # Save notification in Firestore
    db.collection("notifications").add({
        "userId": id,
        "title": f"Claim {status}",
        "message": f"Your claim {id} is now {status}.",
        "category": "claim",
        "read": False,
        "createdAt": datetime.utcnow()
    })

    # Trigger EmailJS (call your JS helper or REST API)
    # sendEmailNotification(user_email, f"Claim {status}", f"Your claim {id} is now {status}.")
    return {"message": "Status updated and notification sent"}
