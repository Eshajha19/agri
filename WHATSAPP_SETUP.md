# WhatsApp Integration Setup 🌾📲

This feature enables Fasal Saathi to send real-time alerts (Weather, Pests, Irrigation) directly to farmers via WhatsApp using the Twilio API.

## 🚀 How to Activate

### 1. Twilio Configuration
1.  Create a free account at [Twilio](https://www.twilio.com/).
2.  Go to the **Twilio Console** and copy your **Account SID** and **Auth Token**.
3.  Set up the **WhatsApp Sandbox** in the Twilio Console (Messaging > Try it Out > Send a WhatsApp Message).
4.  Note down the sandbox keyword (e.g., `join thirty-captured`).

### 2. Environment Variables
Create a `.env` file in the `agri/` directory based on `.env.example`:
```env
TWILIO_ACCOUNT_SID=your_sid_here
TWILIO_AUTH_TOKEN=your_token_here
TWILIO_WHATSAPP_NUMBER=+14155238886
```

### 3. Setting up the Webhook (For Bot Responses)
To allow the bot to respond to user messages (e.g., "Hi", "Weather"):
1.  Start your backend: `uvicorn main:app --reload`
2.  Expose your port (8000) using **ngrok** or **localtunnel**:
    *   `npx localtunnel --port 8000`
3.  Copy the public URL provided.
4.  In Twilio Sandbox Settings, paste the URL in the **"WHEN A MESSAGE COMES IN"** field:
    *   `https://your-url.loca.lt/api/whatsapp/webhook`

## 🧪 How to Test
1.  **Join the Sandbox**: On your phone, send the join command (e.g., `join thirty-captured`) to the Twilio number.
2.  **Subscribe in App**: Go to the Dashboard, enter your number, and toggle "Enable WhatsApp Alerts".
3.  **Trigger Alert**: Use the Pest Management tool. If a severe pest is detected, click "Send Alert to WhatsApp".
4.  **Chat with Bot**: Send "Hi" or "Weather" to the bot on WhatsApp!

## 📂 New/Modified Files
- `main.py`: Added WhatsApp alert and webhook endpoints.
- `whatsapp_service.py`: core logic for Twilio integration.
- `Dashboard.jsx/css`: Added WhatsApp management UI.
- `App.css`: Global floating WhatsApp button styles.
