"""Alerts & Notifications Router"""
from fastapi import APIRouter, Request, HTTPException, Query, Form
from pydantic import BaseModel, Field
from datetime import datetime

router = APIRouter()

class AlertTriggerRequest(BaseModel):
    alert_type: str = Field(..., pattern=r'^(weather|pest|advisory)$')
    message: str = Field(..., min_length=1, max_length=500)

notification_store = None
subscriber_store = None
generate_alerts_fn = None
send_whatsapp_fn = None
format_alert_fn = None
verify_role_fn = None

def init_alerts(ns, ss, ga_fn, sw_fn, fa_fn, vr_fn):
    global notification_store, subscriber_store, generate_alerts_fn, send_whatsapp_fn, format_alert_fn, verify_role_fn
    notification_store = ns
    subscriber_store = ss
    generate_alerts_fn = ga_fn
    send_whatsapp_fn = sw_fn
    format_alert_fn = fa_fn
    verify_role_fn = vr_fn

@router.get("/notifications")
async def get_notifications(request: Request, crop: str = Query(None), irrigation_count: int = Query(None, ge=0), water_coverage: int = Query(None, ge=0, le=100), season: str = Query(None)):
    if notification_store is None or generate_alerts_fn is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    dynamic_alerts = generate_alerts_fn(crop=crop, irrigation_count=irrigation_count, water_coverage=water_coverage, season=season)
    return {"success": True, "data": notification_store.get_recent() + dynamic_alerts}

@router.post("/whatsapp/subscribe")
async def subscribe_whatsapp(request: Request, phone_number: str = Form(...), name: str = Form(...)):
    if not all([subscriber_store, send_whatsapp_fn, verify_role_fn]):
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        token_data = await verify_role_fn(request)
        uid = token_data["uid"]
        subscriber = {"phone_number": phone_number, "name": name, "subscribed_at": datetime.now().isoformat()}
        subscriber_store.upsert(uid, subscriber)
        welcome_msg = f"Namaste {name}! 🙏\nWelcome to *Fasal Saathi WhatsApp Alerts*."
        send_whatsapp_fn(phone_number, welcome_msg)
        return {"success": True, "message": "Successfully subscribed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/whatsapp/trigger-alert")
async def trigger_whatsapp_alert(request: Request, data: AlertTriggerRequest):
    if not all([subscriber_store, send_whatsapp_fn, format_alert_fn, notification_store, verify_role_fn]):
        raise HTTPException(status_code=500, detail="Not initialized")
    try:
        await verify_role_fn(request, required_roles=["admin", "expert"])
        subscribers = subscriber_store.get_all()
        results = []
        formatted_msg = format_alert_fn(data.alert_type, data.message)
        for user_id, info in subscribers.items():
            res = send_whatsapp_fn(info["phone_number"], formatted_msg)
            results.append({"user_id": user_id, "success": res.get("success", False), "status": res.get("status", "error")})
        notification_store.append(alert_type=data.alert_type, message=data.message)
        delivered = sum(1 for r in results if r["success"])
        return {"success": True, "results": results, "delivered": delivered, "total": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/whatsapp/webhook")
async def whatsapp_webhook(Body: str = Form(...), From: str = Form(...)):
    if send_whatsapp_fn is None:
        raise HTTPException(status_code=500, detail="Not initialized")
    sender_number = From.replace("whatsapp:", "")
    
    from celery_worker import process_whatsapp_webhook_task
    process_whatsapp_webhook_task.delay(Body, sender_number)
    
    return {"status": "success"}
