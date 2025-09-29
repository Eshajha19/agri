# train_model.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import joblib

# === 1. Load Data directly from CSV files ===
train_path = 'train.csv'  # Your train.csv path
test_path = 'test.csv'    # Your test.csv path

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# === 2. Preprocess ===
target = 'ExpYield'
y_train = train_df[target].values
y_test = test_df[target].values

drop_cols = ['FarmID', 'SDate', 'HDate', 'geometry', 'category']
train = train_df.drop(columns=drop_cols, errors='ignore')
test = test_df.drop(columns=drop_cols, errors='ignore')

cat_cols = ['Crop', 'State', 'District', 'Sub-District', 'IrriType', 'IrriSource', 'Season', 'CNext', 'CLast', 'CTransp']
encoders = {}

for col in cat_cols:
    if col in train.columns:
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        combined_data = pd.concat([train[col], test[col]], axis=0).astype(str).unique().reshape(-1, 1)
        oe.fit(combined_data)
        train[col] = oe.transform(train[col].astype(str).values.reshape(-1, 1))
        test[col] = oe.transform(test[col].astype(str).values.reshape(-1, 1))
        encoders[col] = oe

features = [col for col in train.columns if col != target]

X_train = train[features].values
X_test = test[features].values

# === 3. Scale data ===
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === 4. Reshape for LSTM ===
X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# === 5. Build Model ===
n_timesteps, n_features = X_train_reshaped.shape[1], X_train_reshaped.shape[2]

model = Sequential([
    LSTM(128, activation='tanh', input_shape=(n_timesteps, n_features)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.1),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])
model.summary()

# === 6. Callbacks ===
callbacks = [
    EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-6, verbose=1),
    ModelCheckpoint('best_lstm_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
]

# === 7. Train ===
history = model.fit(
    X_train_reshaped.astype('float32'), y_train.astype('float32'),
    validation_data=(X_test_reshaped.astype('float32'), y_test.astype('float32')),
    epochs=60,
    batch_size=32,
    callbacks=callbacks,
    verbose=2
)

# === 8. Evaluate ===
y_pred = model.predict(X_test_reshaped).flatten()

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"Test RMSE : {rmse:.4f}")
print(f"Test MAE  : {mae:.4f}")
print(f"Test R2   : {r2:.4f}")
print(f"Test MAPE : {100*mape:.2f}%")

# === 9. Save artifacts ===
model.save('final_lstm_model.h5')
joblib.dump(scaler, 'x_scaler.save')
joblib.dump(encoders, 'label_encoders.save')

print("Saved model and preprocessing artifacts.")
