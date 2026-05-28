import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np
import os
import hmac
import hashlib
import json
from datetime import datetime
from pathlib import Path

df = pd.read_csv("Train.csv")
# Convert SDate to datetime
df['SDate'] = pd.to_datetime(df['SDate'], errors='coerce')
df = df.dropna(subset=['SDate'])
df = df.sort_values('SDate')
print(df[['SDate', 'ExpYield']].head())

X = df.drop(columns=["FarmID", "category", "State", "District", "Sub-District", "SDate", "HDate", "ExpYield", "geometry"], errors='ignore')
y = df["ExpYield"]

categorical_cols = ['Crop', 'CNext', 'CLast', 'CTransp', 'IrriType', 'IrriSource', 'Season']
numeric_cols = [c for c in X.columns if c not in categorical_cols]

# ------------------------------------------------------------------
# Save feature baseline BEFORE get_dummies so we capture raw values
# ------------------------------------------------------------------
def save_feature_baseline(df_raw, cat_cols, num_cols, output_path="feature_baseline.json"):
    """
    Saves training feature statistics to feature_baseline.json.
    Called once after training so the drift detector has a reference
    distribution to compare against at inference time.

    Numeric features  : mean, std, min, max, up to 500 sample values
    Categorical features : list of known categories + frequency fractions
    """
    numeric_features = {}
    categorical_features = {}

    for col in num_cols:
        if col not in df_raw.columns:
            continue
        series = df_raw[col].dropna()
        try:
            series = series.astype(float)
        except (TypeError, ValueError):
            continue
        sample = series.sample(min(500, len(series)), random_state=42).tolist()
        numeric_features[col] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "sample_values": sample,
        }

    for col in cat_cols:
        if col not in df_raw.columns:
            continue
        vc = df_raw[col].dropna().value_counts(normalize=True)
        categorical_features[col] = {
            "categories": vc.index.tolist(),
            "value_counts": vc.to_dict(),
        }

    baseline = {
        "generated_at": datetime.utcnow().isoformat(),
        "csv_path": "Train.csv",
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }

    tmp_path = Path(output_path).with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)
    os.replace(tmp_path, output_path)

    num_count = len(numeric_features)
    cat_count = len(categorical_features)
    print(f"✅ Feature baseline saved to {output_path}")
    print(f"   📊 {num_count} numeric features + {cat_count} categorical features = {num_count + cat_count} total")
    return baseline

# Save baseline from the raw X (before get_dummies) so we have
# original category strings, not one-hot column names.
save_feature_baseline(X, categorical_cols, numeric_cols)

# ------------------------------------------------------------------
# Model training (unchanged from original)
# ------------------------------------------------------------------
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