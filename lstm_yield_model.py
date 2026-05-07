import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.read_csv("Train.csv")

df['SDate'] = pd.to_datetime(df['SDate'], errors='coerce', dayfirst=True)
df = df.dropna(subset=['SDate'])
df = df.sort_values('SDate')

df = df.groupby('SDate')['ExpYield'].mean().reset_index()
df.set_index('SDate', inplace=True)

print("After grouping:")
print(df.head())

# Scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

def create_sequences(data, seq_length=5):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data)

print("X shape:", X.shape)
print("y shape:", y.shape)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(64, activation='relu', input_shape=(X.shape[1], 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=20, batch_size=16)
model.save("lstm_yield_model.keras")
print("✅ LSTM model trained and saved")