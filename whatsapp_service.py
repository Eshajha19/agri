import os
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()

# Twilio configuration - these should be in your .env file
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "your_account_sid")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "your_auth_token")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER", "+14155238886") # Twilio Sandbox number

def get_twilio_client():
    try:
        return Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    except Exception as e:
        print(f"Failed to initialize Twilio client: {e}")
        return None

def send_whatsapp_message(to_number, message_body):
    """
    Sends a WhatsApp message using Twilio API.
    to_number: The recipient's phone number (with country code, e.g., +911234567890)
    message_body: The text content of the message
    """
    client = get_twilio_client()
    if not client:
        return {"success": False, "error": "Twilio client not initialized"}

    try:
        # Ensure number format for Twilio WhatsApp
        if not to_number.startswith("whatsapp:"):
            to_number = f"whatsapp:{to_number}"
        
        message = client.messages.create(
            from_=f"whatsapp:{TWILIO_WHATSAPP_NUMBER}",
            body=message_body,
            to=to_number
        )
        return {"success": True, "sid": message.sid}
    except Exception as e:
        print(f"Error sending WhatsApp message to {to_number}: {e}")
        return {"success": False, "error": str(e)}

def format_alert_message(alert_type, content):
    """
    Formats the alert message for WhatsApp.
    """
    header = "🌾 *Fasal Saathi Alert* 🌾\n\n"
    
    if alert_type == "weather":
        icon = "⛈️"
        title = "*Weather Warning*"
    elif alert_type == "pest":
        icon = "🐛"
        title = "*Pest Outbreak Alert*"
    elif alert_type == "advisory":
        icon = "📝"
        title = "*Farming Advisory*"
    else:
        icon = "📢"
        title = "*Notification*"
        
    return f"{header}{icon} {title}\n\n{content}\n\n_Stay safe and stay informed with Fasal Saathi._"
