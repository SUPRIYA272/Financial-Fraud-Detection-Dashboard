# Financial Fraud Detection Dashboard with LSTM

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import seaborn as sns

# ---------------------- ETL & Feature Engineering ----------------------
def fetch_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    df.dropna(inplace=True)
    df['return'] = df['Close'].pct_change()
    df['volatility'] = df['Close'].rolling(window=5).std()
    df['volume_change'] = df['Volume'].pct_change()
    df.dropna(inplace=True)
    return df[['Close', 'Volume', 'return', 'volatility', 'volume_change']]

# ---------------------- ML Model: Isolation Forest ----------------------
def apply_isolation_forest(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    model = IsolationForest(contamination=0.05)
    preds = model.fit_predict(scaled_data)
    return np.where(preds == -1, 1, 0)  # 1 = anomaly

# ---------------------- DL Model: Autoencoder ----------------------
def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(8, activation='relu')(input_layer)
    bottleneck = Dense(4, activation='relu')(encoded)
    decoded = Dense(8, activation='relu')(bottleneck)
    output_layer = Dense(input_dim, activation='linear')(decoded)
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer=Adam(), loss='mse')
    return autoencoder

def apply_autoencoder(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    ae = build_autoencoder(scaled_data.shape[1])
    ae.fit(scaled_data, scaled_data, epochs=50, batch_size=16, verbose=0)
    reconstructions = ae.predict(scaled_data)
    mse = np.mean(np.power(scaled_data - reconstructions, 2), axis=1)
    threshold = np.percentile(mse, 95)
    return np.where(mse > threshold, 1, 0)  # 1 = anomaly

# ---------------------- DL Model: LSTM ----------------------
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(input_shape[1]))
    model.compile(optimizer='adam', loss='mse')
    return model

def apply_lstm_anomaly_detection(data, window_size=10):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    X, y = create_sequences(scaled_data, window_size)
    model = build_lstm_model((window_size, data.shape[1]))
    model.fit(X, y, epochs=30, batch_size=16, verbose=0)
    y_pred = model.predict(X)
    mse = np.mean(np.power(y - y_pred, 2), axis=1)
    threshold = np.percentile(mse, 95)
    anomalies = np.zeros(len(data))
    anomalies[window_size:] = np.where(mse > threshold, 1, 0)
    return anomalies

# ---------------------- Streamlit Dashboard ----------------------
st.set_page_config(page_title="Financial Fraud Detection", layout="wide")
st.title("Financial Fraud Detection Dashboard")

# Sidebar
st.sidebar.header("Select Inputs")
ticker = st.sidebar.selectbox("Select Stock Ticker:", options=["AAPL", "MSFT", "TSLA", "GOOGL", "META", "JPM", "PYPL", "WFC", "SQ", "GS"], index=0)
days = st.sidebar.slider("Number of past days to analyze:", min_value=30, max_value=180, value=90)
model_choice = st.sidebar.selectbox("Select Algorithm:", ["Isolation Forest (ML)", "Autoencoder (DL)", "LSTM (DL)"])

# Date range
from datetime import datetime, timedelta
end_date = datetime.today()
start_date = end_date - timedelta(days=days)

# Fetch Data
df = fetch_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

# ---------------------- Dataset Summary ----------------------
st.subheader("Dataset Summary")
st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# Show Raw Data
st.subheader("Sample of Processed Dataset")
st.dataframe(df.tail())

# EDA Visuals
st.subheader("Exploratory Data Analysis")
st.line_chart(df['Close'], use_container_width=True)
st.bar_chart(df['Volume'], use_container_width=True)
st.write("Correlation Heatmap:")
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
st.pyplot(plt.gcf())

# Run Model
st.subheader("Fraud Detection Results")
if model_choice == "Isolation Forest (ML)":
    df['anomaly'] = apply_isolation_forest(df[['return', 'volatility', 'volume_change']])
elif model_choice == "Autoencoder (DL)":
    df['anomaly'] = apply_autoencoder(df[['return', 'volatility', 'volume_change']])
else:
    df['anomaly'] = apply_lstm_anomaly_detection(df[['return', 'volatility', 'volume_change']])

anomalies = df[df['anomaly'] == 1]
st.write("Total Anomalies Detected:", anomalies.shape[0])
st.dataframe(anomalies)
st.line_chart(df['anomaly'], use_container_width=True)
