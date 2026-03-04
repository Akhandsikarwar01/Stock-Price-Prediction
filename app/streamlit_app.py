import streamlit as st
import pandas as pd
import numpy as np
import os
from src.data_fetcher import download_stock_data
from src.data_preprocessing import preprocess_data
from src.feature_engineering import create_sequences
from tensorflow.keras.models import load_model
import pickle

MODEL_PATH = os.path.join('..', 'models', 'lstm_model.h5')
SCALER_PATH = os.path.join('..', 'models', 'scaler.pkl')
SEQ_LEN = 60

def load_scaler(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def main():
    st.title('📈 Stock Price Prediction System')
    ticker = st.selectbox('Select Stock Ticker', ['AAPL', 'TSLA', 'GOOG', 'MSFT', 'AMZN'])
    period = st.selectbox('Select Period', ['1y', '2y', '5y'])
    if st.button('Predict'):
        df = download_stock_data(ticker, period)
        data, scaler = preprocess_data(df)
        X, y = create_sequences(data, SEQ_LEN)
        model = load_model(MODEL_PATH)
        y_pred = model.predict(X)
        y_pred_rescaled = scaler.inverse_transform(np.concatenate([np.zeros((y_pred.shape[0], 4)), y_pred], axis=1))[:, -1]
        y_true_rescaled = scaler.inverse_transform(np.concatenate([np.zeros((y.shape[0], 4)), y.reshape(-1,1)], axis=1))[:, -1]
        st.line_chart(pd.DataFrame({'Actual': y_true_rescaled, 'Predicted': y_pred_rescaled}))
        st.success('Prediction complete!')

if __name__ == '__main__':
    main()
