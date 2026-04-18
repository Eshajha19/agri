from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
@app.route('/')
def home():
    return "Backend is running ✅"

@app.route('/api/crop/recommend', methods=['POST'])
def recommend_crop():
    data = request.json
    print("Received Data:", data)   # 👈 ADD THIS

    ph = float(data.get("ph", 0))
    nitrogen = float(data.get("nitrogen", 0))
    phosphorus = float(data.get("phosphorus", 0))
    potassium = float(data.get("potassium", 0))
    location = data.get("location", "").lower()

    # 🔥 Simple intelligent logic
    if ph < 5.5:
        crop = "Rice 🌾"
    elif ph < 7:
        crop = "Wheat 🌿"
    elif ph < 8:
        crop = "Maize 🌽"
    else:
        crop = "Barley 🌱"

    # Adjust based on nutrients
    if nitrogen > 80:
        crop = "Sugarcane 🍬"
    elif phosphorus < 30:
        crop = "Pulses 🌼"

    # Location-based tweak
    if "maharashtra" in location:
        crop = "Cotton 🌿"

    return jsonify({
        "status": "success",
        "crop": crop
    })
if __name__ == '__main__':
    app.run(debug=True)   # remove port=5001