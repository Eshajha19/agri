import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np
import os
import hmac
import hashlib

df = pd.read_csv("Train.csv")
# Convert SDate to datetime
df['SDate'] = pd.to_datetime(df['SDate'], errors='coerce')
df = df.dropna(subset=['SDate'])
df = df.sort_values('SDate')
print(df[['SDate', 'ExpYield']].head())

X = df.drop(columns=["FarmID", "category", "State", "District", "Sub-District", "SDate", "HDate", "ExpYield", "geometry"])
y = df["ExpYield"]

categorical_cols = ['Crop', 'CNext', 'CLast', 'CTransp', 'IrriType', 'IrriSource', 'Season']
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = xgb.XGBRegressor(n_estimators=200, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("✅ Model trained successfully")
print("📊 RMSE:", rmse)

# Save model
joblib.dump(model, "yield_model.joblib")

# Optionally sign the model if a signing key is available in environment
signing_key = os.getenv("MODEL_SIGNING_KEY")
if signing_key:
    with open("yield_model.joblib", "rb") as f:
        data = f.read()
    sig = hmac.new(signing_key.encode("utf-8"), data, hashlib.sha256).hexdigest()
    with open("yield_model.joblib.sig", "w", encoding="utf-8") as sf:
        sf.write(sig)
    print("Wrote signature to yield_model.joblib.sig")
else:
    print("MODEL_SIGNING_KEY not set; no signature file written for yield_model.joblib")
