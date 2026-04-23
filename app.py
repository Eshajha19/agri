from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

# Load trained model (ensure this path is correct)
try:
    model = joblib.load("yield_model.joblib")
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    model = None

# Create FastAPI app
app = FastAPI()
class YieldInput(BaseModel):
    data: List[float]

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("yield_model.joblib")  # existing model
model_lag = joblib.load("sklearn_yield_model.pkl")  # your new model

@app.get("/predict")
def predict():
    if model is None:
        return {"error": "Model not available", "predicted_yield": None}
    
    # Dummy input matching training features
    input_df = pd.DataFrame([{
        "NDVI": 4800,
        "Rainfall": 25.0,
        "SoilMoisture": 4.5,
        "Crop-wise_Rice": 1
    }])

    for col in model.get_booster().feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[model.get_booster().feature_names]

    prediction = model.predict(input_df)[0]
    return {"predicted_yield": float(prediction)}

@app.post("/predict-yield-lag")
async def predict_yield_lag(payload: YieldInput):
    try:
        data = payload.data

        if len(data) != 5:
            raise ValueError("Exactly 5 values are required")

        data = np.array(data).reshape(1, -1)

        prediction = model_lag.predict(data)

        return {
            "prediction": round(float(prediction[0]), 2),
            "model": "RandomForest Time Series (Lag Features)"
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error") from e


@app.post("/predict-yield-trend")
async def predict_yield_trend(payload: YieldInput):
    try:
        data = payload.data

        if len(data) != 5:
            raise ValueError("Exactly 5 values are required")

        temp = data[::-1]  # reverse once
        trend = []

        for _ in range(5):
            features = temp[:5]  # latest 5 values (correct order)

            pred = model_lag.predict([features])[0]
            pred_value = round(float(pred), 2)

            trend.append(pred_value)
            

            temp = [pred_value] + temp  # maintain correct lag order

        # ✅ return AFTER loop (inside try)
        return {
            "trend": trend,
            "prediction": trend[-1],
            "model": "RandomForest Trend Forecast (Lag Features)"
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error") from e