import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load data
df = pd.read_csv("Train.csv")

# Convert date
df['SDate'] = pd.to_datetime(df['SDate'], errors='coerce', dayfirst=True)
df = df.dropna(subset=['SDate'])
df = df.sort_values('SDate')

# Group by date
df = df.groupby('SDate')['ExpYield'].mean().reset_index()

# 🔥 CREATE LAG FEATURES (THIS REPLACES LSTM)
for i in range(1, 6):
    df[f'lag_{i}'] = df['ExpYield'].shift(i)

# Drop missing rows
df = df.dropna()

# Features and target
X = df[[f'lag_{i}' for i in range(1, 6)]]
y = df['ExpYield']

# Train model
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

# Save model
joblib.dump(model, "sklearn_yield_model.pkl")

print("✅ Sklearn time-series model trained and saved")