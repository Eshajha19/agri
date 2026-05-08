import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np

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
